"""Verify and conservatively repair Surya table blocks against native tokens.

Surya (image route) and the PDF text layer (native route) fail differently:
Surya gets structure and signs right on upright pages but can truncate or
invent cells; the text layer has exact digits but mis-encodes minus glyphs.
This module compares the two numerically, in both directions, and assigns a
block status:

- ``verified``    every Surya number is in the native pool and vice versa.
- ``repaired``    only authoritative sign disagreements, fixed deterministically.
- ``unresolved``  unmatched numbers, truncation, sign conflicts that cannot be
                  settled, or duplicated rows; numeric cells must be withheld.
- ``single-route`` no usable native numbers in the region (scan or math).

Structural findings (empty body columns, merged estimate/SE cells, repeated
rows the text layer also repeats) are reported separately; they do not change
numeric status. Numbers that differ only in spelling (``.5``/``0.50``,
``9,081``/``9,081.000``) match; different values never do (``5.037`` is not
``5,037``). Text-layer numbers that belong to labels, column-number rows,
notes inside the table region or another sidecar block are reported but do
not withhold a table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .sidecar_native import AUTHORITATIVE_SIGN_SOURCES, NativeToken

_MINUS_CHARS = "\u2212\u2afa\u2013\u2014\u2012\u2010\u2011\ufe63\uff0d-"
# A comma separates digit groups only before exactly three digits and before
# any decimal point, as in the native tokenizer: "0.33,0.67" and "1,2" are two.
_NUM_RE = re.compile(
    rf"(?<![\d.])[{_MINUS_CHARS}]?(?:\d{{1,3}}(?:,\d{{3}})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
)
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<(t[dh])\b([^>]*)>(.*?)</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SPAN_RE = re.compile(r'colspan\s*=\s*"?(\d+)', re.I)
_EST_SE_RE = re.compile(r"(\d\**)\s*(?=[\[(][-\u2212]?\.?\d)")


def normalize_number(raw: str) -> str:
    text = raw.strip().rstrip(",.")
    if text and text[0] in _MINUS_CHARS:
        text = "-" + text[1:]
    return text


def _fragment_text(fragment: str) -> str:
    return _TAG_RE.sub(" ", fragment).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def _plain_numbers(plain: str) -> list[str]:
    out = []
    for m in _NUM_RE.finditer(plain):
        n = normalize_number(m.group(0))
        if any(c.isdigit() for c in n):
            out.append(n)
    return out


def text_numbers(fragment: str) -> list[str]:
    return _plain_numbers(_fragment_text(fragment))


_LEAD_MINUS_RE = re.compile(rf"\s*[{_MINUS_CHARS}]\s{{1,2}}\.?\d")
_EXP_MINUS_RE = re.compile(rf"(?<=\d)(\s*[Ee]\s*)[{_MINUS_CHARS}]\s{{0,2}}(?=\d)")


def cell_numbers(fragment: str) -> list[str]:
    """Numbers in one table cell.

    A cell that opens with a detached minus (``– 0.15``, ``- 1.78``) holds a
    negative first number, as the text layer reads it (``codepoint_spaced``).
    Only at the start of a cell: elsewhere a spaced dash is a range or
    punctuation. An exponent keeps its minus: ``9.38 E – 06`` gives ``-06``.
    """
    plain = _EXP_MINUS_RE.sub(r"\1-", _fragment_text(fragment))
    nums = _plain_numbers(plain)
    if nums and not nums[0].startswith("-") and _LEAD_MINUS_RE.match(plain):
        nums[0] = "-" + nums[0]
    return nums


def _flip(n: str) -> str:
    return n[1:] if n.startswith("-") else "-" + n


@dataclass
class Cell:
    tag: str
    attrs: str
    inner: str

    @property
    def colspan(self) -> int:
        m = _SPAN_RE.search(self.attrs)
        return int(m.group(1)) if m else 1

    @property
    def numbers(self) -> list[str]:
        return cell_numbers(self.inner)


def parse_rows(html: str) -> list[list[Cell]]:
    return [
        [Cell(t.lower(), a, inner) for t, a, inner in _CELL_RE.findall(row)]
        for row in _ROW_RE.findall(html)
    ]


@dataclass
class Finding:
    kind: str
    detail: str
    values: list[str] = field(default_factory=list)


@dataclass
class TableCheck:
    status: str
    html: str  # possibly repaired and normalized
    findings: list[Finding] = field(default_factory=list)
    repairs: list[dict] = field(default_factory=list)
    surya_numbers: int = 0
    native_numbers: int = 0

    @property
    def structure_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.kind in STRUCTURAL]


STRUCTURAL = {"empty_body_column", "merged_estimate_se", "label_or_header_mismatch", "duplicate_rows"}
#: Findings that never change status and are left out of the status line.
INFORMATIONAL = {"merged_estimate_se", "label_or_header_mismatch", "format_variant", "split_token",
                 "ocr_detached_dash", "native_label_unmatched", "native_outside_table"}

_THEAD_RE = re.compile(r"<thead\b[^>]*>(.*?)</thead>", re.S | re.I)
_SCRIPT_RE = re.compile(r"<(sup|sub)\b[^>]*>(.*?)</\1>", re.S | re.I)
_COLNUM_RE = re.compile(r"^\(\d{1,2}\)$")
_MATH_RE = re.compile(r"<math\b[^>]*>.*?</math>", re.S | re.I)
_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d*)?")


def _plain(fragment: str) -> str:
    return _TAG_RE.sub(" ", _MATH_RE.sub(" ", fragment)).strip()


def _label_text(fragment: str) -> str:
    """Cell text for the label test: math kept, LaTeX command names dropped.

    ``\\Delta\\text{Ln Rents}_{t-1}`` is a row label, not a data cell.
    """
    kept = _MATH_RE.sub(lambda m: " " + _LATEX_CMD_RE.sub(" ", _TAG_RE.sub(" ", m.group(0))) + " ", fragment)
    return _TAG_RE.sub(" ", kept)


def numeric_key(n: str) -> str:
    """Spelling-independent value: ``.5``, ``0.50`` -> ``0.5``; ``9,081.000`` -> ``9081``.

    Commas are dropped only as thousands separators and the sign is kept, so
    ``4.312`` != ``4,312``, ``12,34`` stays itself and ``-0`` != ``0``.
    """
    neg = n.startswith("-")
    m = n[1:] if neg else n
    if _THOUSANDS_RE.fullmatch(m):
        m = m.replace(",", "")
    whole, _, frac = m.partition(".")
    frac = frac.rstrip("0")
    return ("-" if neg else "") + (whole or "0") + ("." + frac if frac else "")


def _digits(n: str) -> str:
    return re.sub(r"\D", "", n)


def _gap_ok(a, b) -> bool:
    """Two text-layer pieces close enough to be one printed number (gap under 0.6 em).

    Works on horizontal and sideways lines; a column gap between two cells is
    wider than this.
    """
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    if y_overlap > 0 and y_overlap >= x_overlap:  # side by side on a horizontal line
        gap, height = max(b[0] - a[2], a[0] - b[2]), min(a[3] - a[1], b[3] - b[1])
    elif x_overlap > 0:  # stacked along a sideways line
        gap, height = max(b[1] - a[3], a[1] - b[3]), min(a[2] - a[0], b[2] - b[0])
    else:
        return False
    return gap <= 0.6 * height


#: Native roles to try, in order, for a Surya number of each role.
_ROLE_PREFERENCE = {
    "data": ("data", "label", "colnum", "note"),
    "label": ("label", "data", "colnum", "note"),
    "header": ("colnum", "label", "data", "note"),
}


def _grid_cell(t: NativeToken, agreed: list[NativeToken]) -> bool:
    """``t`` shares a row with one agreed number and a column with another."""

    def overlap(a, b, axis: int) -> bool:
        return min(a[axis + 2], b[axis + 2]) - max(a[axis], b[axis]) > 0

    row = any(overlap(t.bbox, o.bbox, 1) and not overlap(t.bbox, o.bbox, 0) for o in agreed)
    col = any(overlap(t.bbox, o.bbox, 0) and not overlap(t.bbox, o.bbox, 1) for o in agreed)
    return row and col


def numbers_with_roles(html: str) -> list[tuple[str, str]]:
    """Return ``(number, role)`` for every number in the table.

    Roles: ``header`` (thead rows, th cells, rows of column numbers such as
    ``(1) (2) (3)``), ``label`` (cells whose text contains words, e.g.
    ``Small (0-50)`` or ``1978 earnings``, superscripts or subscripts such
    as ``R<sup>2</sup>``, and text outside the rows: the notes and titles
    Surya writes as paragraphs around ``<table>``), otherwise ``data``.
    """
    head_spans = [m.span(1) for m in _THEAD_RE.finditer(html)]
    out: list[tuple[str, str]] = []
    for rm in _ROW_RE.finditer(html):
        in_head = any(a <= rm.start(1) < b for a, b in head_spans)
        cells = [Cell(t.lower(), a, inner) for t, a, inner in _CELL_RE.findall(rm.group(1))]
        numeric = [c for c in cells if c.numbers]
        colnum_row = bool(numeric) and all(_COLNUM_RE.match(_plain(c.inner)) for c in numeric)
        for c in cells:
            if in_head or c.tag == "th" or colnum_row:
                role = "header"
            elif sum(ch.isalpha() for ch in _label_text(c.inner)) >= 2:
                role = "label"
            else:
                scripts = [n for m in _SCRIPT_RE.finditer(c.inner) for n in text_numbers(m.group(2))]
                out.extend((n, "data") for n in cell_numbers(_SCRIPT_RE.sub(" ", c.inner)))
                out.extend((n, "label") for n in scripts)
                continue
            out.extend((n, role) for n in c.numbers)
    out.extend((n, "label") for n in text_numbers(_ROW_RE.sub(" ", html)))
    return out


def _structure(rows: list[list[Cell]]) -> list[Finding]:
    findings: list[Finding] = []
    body = [r for r in rows if r and all(c.tag == "td" for c in r)]
    if body:
        width = max(sum(c.colspan for c in r) for r in body)
        for col in range(width):
            vals, pos_ok = [], True
            for r in body:
                pos = 0
                for c in r:
                    if pos == col and c.colspan == 1:
                        vals.append(_TAG_RE.sub("", c.inner).strip())
                        break
                    if pos <= col < pos + c.colspan:
                        pos_ok = False
                        break
                    pos += c.colspan
            if pos_ok and vals and len(vals) == len(body) and not any(vals):
                findings.append(Finding("empty_body_column", f"column {col + 1} is empty in every body row"))
    return findings


def _duplicate_rows(rows: list[list[Cell]]) -> list[Finding]:
    seen: dict[tuple, int] = {}
    dups = []
    for i, r in enumerate(rows):
        nums = tuple(n for c in r for n in c.numbers)
        if len(nums) >= 3:
            if nums in seen:
                dups.append(f"row {i + 1} repeats row {seen[nums] + 1}")
            else:
                seen[nums] = i
    return [Finding("duplicate_rows", "; ".join(dups))] if dups else []


def split_estimate_se(html: str) -> tuple[str, int]:
    count = 0

    def fix_cell(m: re.Match) -> str:
        nonlocal count
        inner, n = _EST_SE_RE.subn(r"\1 ", m.group(3))
        count += n
        return f"<{m.group(1)}{m.group(2)}>{inner}</{m.group(1)}>"

    return _CELL_RE.sub(fix_cell, html), count


def verify_table(
    html: str,
    native_tokens: Iterable[NativeToken],
    context_numbers: Iterable[str] = (),
    native_is_ocr: bool = False,
) -> TableCheck:
    """Compare one Surya table with native tokens from the same region.

    ``native_is_ocr`` marks a region whose text layer is itself OCR (a scan, or
    invisible text over an image). Such a layer is a weak witness: agreement
    caps the table at ``single-route`` (recorded as ``ocr_layer_agreement``),
    it never drives a repair, and only a few isolated digit disagreements are
    tolerated. An OCR-layer number in the table grid (sharing a row and a
    column with agreed numbers) that no unmatched Surya value can pair with
    as a misread is a cell Surya dropped; that is never tolerated.

    An OCR layer that holds under half as many numbers as the table and
    agrees on under half of its own is no witness at all (``ocr_layer_unusable``):
    the table is ``single-route``, as on a scan without a text layer.

    ``native_tokens`` should be clipped to the table's region in displayed-page
    coordinates. ``context_numbers`` are numbers that legitimately appear in
    the region but outside the table body (captions, notes); they are removed
    from the native pool before the truncation check. Tokens flagged ``owned``
    or with a ``label``/``colnum`` role never count as truncation.
    """
    native = list(native_tokens)
    html, merged = split_estimate_se(html)
    rows = parse_rows(html)
    roled = numbers_with_roles(html)
    surya = [n for n, _role in roled]
    check = TableCheck("verified", html, surya_numbers=len(surya), native_numbers=len(native))
    if merged:
        check.findings.append(Finding("merged_estimate_se", f"split {merged} estimate/uncertainty cells"))
    check.findings.extend(_structure(rows))

    if not native:
        check.status = "single-route"
        return check

    # Formula tables: digits are subscripts/exponents inside LaTeX, which the
    # text layer cannot represent. Numeric comparison is meaningless; treat
    # the table like display math.
    all_cells = [c for r in rows for c in r]
    math_cells = [c for c in all_cells if _MATH_RE.search(c.inner)]
    if all_cells and len(math_cells) / len(all_cells) >= 0.25:
        check.status = "single-route"
        check.findings.append(Finding(
            "math_table", f"{len(math_cells)} of {len(all_cells)} cells are LaTeX; no text-layer check"))
        return check

    by_text: dict[str, list[NativeToken]] = {}
    for t in native:
        by_text.setdefault(t.text, []).append(t)
    # Unconsumed native instances; roles matter, so match instances, not counts.
    avail: dict[str, list[NativeToken]] = {k: list(v) for k, v in by_text.items()}
    for n in context_numbers:
        n = normalize_number(n)
        if avail.get(n) and n not in surya:
            avail[n].pop()

    def consume(t: NativeToken) -> None:
        ts = avail[t.text]
        del ts[next(i for i, x in enumerate(ts) if x is t)]

    def pick(cands: list[NativeToken], role: str) -> NativeToken | None:
        """Consume one candidate: this table's own tokens before copies held by a
        caption or note, then the native role that fits the Surya role."""
        if not cands:
            return None
        order = _ROLE_PREFERENCE[role]
        t = min(cands, key=lambda c: (c.owned, order.index(c.role)))
        consume(t)
        return t

    def take(match, role: str, ok=lambda t: True) -> NativeToken | None:
        return pick([t for text, ts in avail.items() if match(text) for t in ts if ok(t)], role)

    unmatched: list[str] = []
    non_data: list[str] = []
    # Match data cells first so a shared value is credited to the data cell.
    for n, role in sorted(roled, key=lambda x: x[1] != "data"):
        if pick(list(avail.get(n, ())), role) is None:
            (unmatched if role == "data" else non_data).append(n)

    # Same value, different spelling: ".5" vs "0.50", "9,081" vs a clipped "9,081.000".
    variants: list[str] = []
    for pending, role in ((unmatched, "data"), (non_data, "label")):
        for n in list(pending):
            t = take(lambda text, k=numeric_key(n): numeric_key(text) == k, role)
            if t is not None:
                pending.remove(n)
                variants.append(f"{n}={t.text}")

    # OCR text layer: "- 16.9" with synthetic character boxes cannot say whether
    # the dash is a sign or a range. It does not contradict a Surya minus.
    dashed: list[str] = []
    if native_is_ocr:
        for pending, role in ((unmatched, "data"), (non_data, "label")):
            for n in [x for x in pending if x.startswith("-")]:
                t = take(lambda text, k=numeric_key(n[1:]): numeric_key(text) == k, role,
                         lambda t: t.dash_before)
                if t is not None:
                    pending.remove(n)
                    dashed.append(n)

    conflicts: list[str] = []
    repairable: list[str] = []
    for n in list(unmatched):
        k = numeric_key(_flip(n))
        t = take(lambda text, k=k: numeric_key(text) == k, "data")
        if t is None:
            continue
        unmatched.remove(n)
        # Deterministic repair only for a Surya positive that native shows
        # negative via an authoritative glyph, on every native copy of the
        # value. A missing native sign is never proof of positivity (broken
        # fonts drop minus glyphs).
        same = [x for x in native if numeric_key(x.text) == k]
        if t.negative and all(x.sign_source in ("codepoint", "font_glyph") for x in same):
            repairable.append(n)
        else:
            conflicts.append(n)

    # Split tokens: a printed 9.33 the text layer holds as "9" and "33" (a lost
    # decimal glyph) or, in an OCR layer, "0.25 18" for 0.2518. The digits must
    # agree exactly and the pieces be neighbours on one line, one space apart.
    splits: list[str] = []
    pairs = [(a, b) for a, b in zip(native, native[1:])
             if b.sep_before is not None and len(b.sep_before) == 1 and b.sep_before.isspace()
             and _gap_ok(a.bbox, b.bbox)]
    for n in list(unmatched):
        k = numeric_key(n)
        for a, b in pairs:
            if not (any(x is a for x in avail.get(a.text, ())) and any(x is b for x in avail.get(b.text, ()))):
                continue
            joined = []
            if a.text.lstrip("-").isdigit() and b.text.isdigit():
                joined.append(a.text + "." + b.text)
            if native_is_ocr and not b.negative:
                joined.append(a.text + b.text)
            if any(numeric_key(x) == k for x in joined):
                consume(a)
                consume(b)
                unmatched.remove(n)
                splits.append(f"{n}={a.text}+{b.text}")
                break

    # Labels and headers: an inferred native sign (a control glyph that may be
    # sigma or "x", a spaced dash) settles nothing, and "R 0125d" vs "R 0.125v"
    # is the same label text read two ways. Neither is a data disagreement.
    for n in list(non_data):
        mag = n.lstrip("-")
        t = take(lambda text, mag=mag: text.lstrip("-") == mag, "label",
                 lambda t: t.sign_source not in AUTHORITATIVE_SIGN_SOURCES)
        if t is None:
            t = take(lambda text, d=_digits(n): _digits(text) == d, "label",
                     lambda t: t.role != "data" or t.glued)
        if t is not None:
            non_data.remove(n)

    left = [t for ts in avail.values() for t in ts]
    missing = [t.text for t in left if not t.owned and t.role == "data"]
    native_label = [t.text for t in left if not t.owned and t.role != "data"]
    outside = [t.text for t in left if t.owned]

    for n in repairable:
        if surya.count(n) != 1:
            conflicts.append(n)
            continue
        target = "-" + n

        def swap(m: re.Match, n=n, target=target) -> str:
            inner = re.sub(rf"(?<![\d.\-\u2212]){re.escape(n)}(?![\d])", target, m.group(3), count=1)
            return f"<{m.group(1)}{m.group(2)}>{inner}</{m.group(1)}>"

        new_html = _CELL_RE.sub(swap, check.html)
        if new_html != check.html:
            check.html = new_html
            check.repairs.append({"kind": "sign", "before": n, "after": target, "route": "native_font_validated"})
        else:
            conflicts.append(n)

    dup = _duplicate_rows(rows)
    check.findings.extend(dup)
    if variants:
        check.findings.append(Finding(
            "format_variant", "same value spelled differently in Surya and the text layer", variants))
    if splits:
        check.findings.append(Finding(
            "split_token", "one number held by the text layer as two neighbouring pieces", splits))
    if dashed:
        check.findings.append(Finding(
            "ocr_detached_dash", "Surya minus backed by a detached dash in the OCR text layer", dashed))
    if non_data:
        check.findings.append(Finding(
            "label_or_header_mismatch", "numbers in labels or header rows absent from the text layer", non_data))
    if native_label:
        check.findings.append(Finding(
            "native_label_unmatched", "text-layer numbers in labels, notes or column-number rows absent from Surya",
            native_label))
    if outside:
        check.findings.append(Finding(
            "native_outside_table", "text-layer numbers held by another sidecar block (caption, note)", outside))
    if unmatched:
        check.findings.append(Finding("surya_unmatched", "data numbers absent from the text layer", unmatched))
    if missing:
        check.findings.append(Finding("native_missing", "text-layer numbers absent from Surya (truncation)", missing))
    if conflicts:
        check.findings.append(Finding("sign_conflict", "sign disagreement not settled by an authoritative glyph", conflicts))

    if native_is_ocr:
        # Undo any repair: an OCR text layer is not authoritative for signs.
        if check.repairs:
            check.html = html
            conflicts.extend(r["before"] for r in check.repairs)
            check.repairs = []
        disagreements = len(unmatched) + len(missing) + len(conflicts)
        agreed = len(native) - len(left)
        agreed_tokens = [t for t in native if not any(t is x for x in left)]
        in_grid = [t for t in left if not t.owned and t.role == "data" and _grid_cell(t, agreed_tokens)]
        if len(native) < 0.5 * len(surya) and agreed < 0.5 * len(native):
            check.status = "single-route"
            check.findings.append(Finding(
                "ocr_layer_unusable",
                f"scan: the OCR text layer holds {len(native)} numbers for {len(surya)} in the table "
                f"and agrees on {agreed}; no independent check"))
        elif (len(in_grid) > len(unmatched) or (dup and disagreements)
              or disagreements > max(2, len(surya) // 50)):
            check.status = "unresolved"
        else:
            check.status = "single-route"
            check.findings.append(Finding(
                "ocr_layer_agreement",
                f"scan: Surya agrees with the OCR text layer except {disagreements} value(s)"))
        return check

    # A repeated row is fine when the text layer repeats it too; an unbacked
    # copy already shows up as unmatched numbers.
    if unmatched or missing or conflicts:
        check.status = "unresolved"
    elif check.repairs:
        check.status = "repaired"
    return check


WITHHELD = "⟦withheld: unverified number, see PDF p. {page}⟧"


def withhold_numbers(html: str, page: int) -> str:
    """Replace numeric cells in data rows with a visible omission marker."""
    marker = WITHHELD.format(page=page)

    def fix_cell(m: re.Match) -> str:
        if m.group(1).lower() == "td" and text_numbers(m.group(3)):
            return f"<{m.group(1)}{m.group(2)}>{marker}</{m.group(1)}>"
        return m.group(0)

    return _CELL_RE.sub(fix_cell, html)
