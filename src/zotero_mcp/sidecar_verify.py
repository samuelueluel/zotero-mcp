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

Structural findings (empty body columns, merged estimate/SE cells) are
reported separately; they do not change numeric status.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .sidecar_native import NativeToken

_MINUS_CHARS = "\u2212\u2afa\u2013\u2014\u2012-"
_NUM_RE = re.compile(
    rf"(?<![\d.])[{_MINUS_CHARS}]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
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


def text_numbers(fragment: str) -> list[str]:
    plain = _TAG_RE.sub(" ", fragment).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    out = []
    for m in _NUM_RE.finditer(plain):
        n = normalize_number(m.group(0))
        if any(c.isdigit() for c in n):
            out.append(n)
    return out


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
        return text_numbers(self.inner)


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


STRUCTURAL = {"empty_body_column", "merged_estimate_se", "label_or_header_mismatch"}

_THEAD_RE = re.compile(r"<thead\b[^>]*>(.*?)</thead>", re.S | re.I)
_COLNUM_RE = re.compile(r"^\(\d{1,2}\)$")
_MATH_RE = re.compile(r"<math\b[^>]*>.*?</math>", re.S | re.I)


def _plain(fragment: str) -> str:
    return _TAG_RE.sub(" ", _MATH_RE.sub(" ", fragment)).strip()


def numbers_with_roles(html: str) -> list[tuple[str, str]]:
    """Return ``(number, role)`` for every number in the table.

    Roles: ``header`` (thead rows, th cells, rows of column numbers such as
    ``(1) (2) (3)``), ``label`` (cells whose text contains words, e.g.
    ``Small (0-50)`` or ``1978 earnings``), otherwise ``data``.
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
            elif sum(ch.isalpha() for ch in _plain(c.inner)) >= 2:
                role = "label"
            else:
                role = "data"
            out.extend((n, role) for n in c.numbers)
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

    ``native_is_ocr`` marks a scanned page whose text layer is itself OCR.
    Such a layer is a weak witness: agreement caps the table at
    ``single-route`` (recorded as ``ocr_layer_agreement``), it never drives a
    repair, and only a few isolated digit disagreements are tolerated.

    ``native_tokens`` should be clipped to the table's region in PDF
    coordinates. ``context_numbers`` are numbers that legitimately appear in
    the region but outside the table body (captions, notes); they are removed
    from the native pool before the truncation check.
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

    pool = Counter(t.text for t in native)
    for n in context_numbers:
        n = normalize_number(n)
        if pool[n] > 0 and n not in surya:
            pool[n] -= 1
    by_text: dict[str, list[NativeToken]] = {}
    for t in native:
        by_text.setdefault(t.text, []).append(t)

    unmatched: list[str] = []
    non_data: list[str] = []
    # Match data cells first so a shared value is credited to the data cell.
    for n, role in sorted(roled, key=lambda x: x[1] != "data"):
        if pool[n] > 0:
            pool[n] -= 1
        elif role == "data":
            unmatched.append(n)
        else:
            non_data.append(n)

    conflicts: list[str] = []
    repairable: list[str] = []
    for n in list(unmatched):
        flipped = _flip(n)
        if pool[flipped] > 0:
            toks = by_text.get(flipped, [])
            # Deterministic repair only for a Surya positive that native shows
            # negative via an authoritative glyph. A missing native sign is
            # never proof of positivity (broken fonts drop minus glyphs).
            if flipped.startswith("-") and toks and all(t.sign_source in ("codepoint", "font_glyph") for t in toks):
                repairable.append(n)
            else:
                conflicts.append(n)
            pool[flipped] -= 1
            unmatched.remove(n)

    missing = [n for n, k in pool.items() for _ in range(k) if k > 0]

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
    if non_data:
        check.findings.append(Finding(
            "label_or_header_mismatch", "numbers in labels or header rows absent from the text layer", non_data))
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
        if dup or disagreements > max(2, len(surya) // 50):
            check.status = "unresolved"
        else:
            check.status = "single-route"
            check.findings.append(Finding(
                "ocr_layer_agreement",
                f"scan: Surya agrees with the OCR text layer except {disagreements} value(s)"))
        return check

    if unmatched or missing or conflicts or dup:
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
