"""Verify and conservatively repair Surya table blocks against native tokens.

Surya (image route) and the PDF text layer (native route) fail differently:
Surya gets structure and signs right on upright pages but can truncate or
invent cells; the text layer has exact digits but mis-encodes minus glyphs.
This module compares the two numerically, in both directions, and assigns a
block status:

- ``verified``    every Surya number is in the native pool and vice versa.
- ``repaired``    fixed deterministically from the text layer: authoritative
                  sign disagreements, or cells put where the page prints them
                  (:func:`fill_from_text_layer`), then re-checked in full.
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

import html as _html
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Iterable

from .sidecar_native import AUTHORITATIVE_SIGN_SOURCES, NativeToken

_MINUS_CHARS = "\u2212\u2afa\u2013\u2014\u2012\u2010\u2011\ufe63\uff0d-"
# A comma separates digit groups only before exactly three digits and before
# any decimal point, as in the native tokenizer: "0.33,0.67" and "1,2" are two.
_NUM_RE = re.compile(
    rf"(?<![\d.])[{_MINUS_CHARS}]?(?:\d+(?:\.\d+){{2,}}|\d{{1,3}}(?:,\d{{3}})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
)
# A run of two or more dots ("02.13.2007", "11.4.3") is one token, a date or
# a section number, as the native tokenizer keeps it.
# In table cells one space may follow the comma: TeX math sets "1,000" as
# "1, 000", and the text layer (and Surya copying it) keeps that space. Not in
# prose, where "22, 301" is a volume and a page.
_CELL_NUM_RE = re.compile(
    rf"(?<![\d.])[{_MINUS_CHARS}]?(?:\d+(?:\.\d+){{2,}}|\d{{1,3}}(?:, ?\d{{3}})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
)
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<(t[dh])\b([^>]*)>(.*?)</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SPAN_RE = re.compile(r'colspan\s*=\s*"?(\d+)', re.I)
_ROWSPAN_RE = re.compile(r'rowspan\s*=\s*"?(\d+)', re.I)
_EST_SE_RE = re.compile(r"(\d\**)\s*(?=[\[(][-\u2212]?\.?\d)")


def normalize_number(raw: str) -> str:
    text = raw.strip().rstrip(",.").replace(", ", ",")
    if text and text[0] in _MINUS_CHARS:
        text = "-" + text[1:]
    return text


def _fragment_text(fragment: str) -> str:
    return _TAG_RE.sub(" ", fragment).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def _plain_numbers(plain: str, pattern: re.Pattern = _NUM_RE) -> list[str]:
    out = []
    for m in pattern.finditer(plain):
        n = normalize_number(m.group(0))
        if any(c.isdigit() for c in n):
            out.append(n)
    return out


def text_numbers(fragment: str) -> list[str]:
    return _plain_numbers(_DF_RE.sub(r"\1 ;", _fragment_text(fragment)))


_ENUM_RE = re.compile(r"\d{1,7}\.")
# Degrees of freedom, "F( 1, 156)" or "chi2(2, 744)": two numbers, not 1,156.
_DF_RE = re.compile(r"((?<![A-Za-z])(?:F|chi2|t)\(\s*\d+)\s*,(?=\s*\d+\s*\))")
_LEAD_MINUS_RE = re.compile(rf"\s*[{_MINUS_CHARS}]\s{{1,2}}\.?\d")
_EXP_MINUS_RE = re.compile(rf"(?<=\d)(\s*[Ee]\s*)[{_MINUS_CHARS}]\s{{0,2}}(?=\d)")


def cell_numbers(fragment: str) -> list[str]:
    """Numbers in one table cell.

    A cell that opens with a detached minus (``– 0.15``, ``- 1.78``) holds a
    negative first number, as the text layer reads it (``codepoint_spaced``).
    Only at the start of a cell: elsewhere a spaced dash is a range or
    punctuation. An exponent keeps its minus: ``9.38 E – 06`` gives ``-06``.
    """
    plain = _DF_RE.sub(r"\1 ;", _EXP_MINUS_RE.sub(r"\1-", _fragment_text(fragment)))
    nums = _plain_numbers(plain, _CELL_NUM_RE)
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
    def rowspan(self) -> int:
        m = _ROWSPAN_RE.search(self.attrs)
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
INFORMATIONAL = {"merged_estimate_se", "header_rowspan_clamped", "label_or_header_mismatch", "format_variant", "split_token",
                 "ocr_detached_dash", "native_label_unmatched", "native_outside_table",
                 "text_layer_fill_failed", "text_table"}
#: A table needs at least this many anchors (values unique on both sides) for
#: the position check.
MIN_ANCHORS = 4
#: Printed words missing from the table and its neighbours: at least this many,
#: and this share of the region's distinct words, mean lost labels or headers.
LOST_WORDS_MIN = 2
LOST_WORDS_SHARE = 0.08

_THEAD_RE = re.compile(r"<thead\b[^>]*>(.*?)</thead>", re.S | re.I)
_SCRIPT_RE = re.compile(r"<(sup|sub)\b[^>]*>(.*?)</\1>", re.S | re.I)
_COLNUM_RE = re.compile(r"^\(\d{1,2}\)$")
_MATH_RE = re.compile(r"<math\b[^>]*>.*?</math>", re.S | re.I)
#: Share of LaTeX cells from which a table counts as a formula table.
MATH_TABLE_SHARE = 0.25
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
    Surya writes as paragraphs around ``<table>``; a row's first cell that is
    only an observation number such as ``1023.``, as in a Stata listing),
    otherwise ``data``.
    """
    head_spans = [m.span(1) for m in _THEAD_RE.finditer(html)]
    out: list[tuple[str, str]] = []
    for rm in _ROW_RE.finditer(html):
        in_head = any(a <= rm.start(1) < b for a, b in head_spans)
        cells = [Cell(t.lower(), a, inner) for t, a, inner in _CELL_RE.findall(rm.group(1))]
        numeric = [c for c in cells if c.numbers]
        colnum_row = bool(numeric) and all(_COLNUM_RE.match(_plain(c.inner)) for c in numeric)
        for i, c in enumerate(cells):
            if in_head or c.tag == "th" or colnum_row:
                role = "header"
            elif sum(ch.isalpha() for ch in _label_text(c.inner)) >= 2 or (
                    i == 0 and _ENUM_RE.fullmatch(_plain(c.inner))):
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


@dataclass
class _Row:
    index: int  # position among the table's <tr> rows
    segment: int
    header: bool
    placed: list[tuple[int, int, Cell]]  # (cell index in the row, logical column, cell)
    width: int  # logical columns occupied, rowspans from above included


def _layout(html: str) -> list[_Row]:
    """Rows with rowspans and colspans expanded to logical columns.

    A header row (thead, all ``th``, or column numbers) starts a new
    ``segment``: stacked panels often print their columns at different
    positions.
    """
    head_spans = [m.span(1) for m in _THEAD_RE.finditer(html)]
    carry: dict[int, int] = {}  # column -> further rows a rowspan occupies
    out = []
    segment, in_header = 0, False
    for r, rm in enumerate(_ROW_RE.finditer(html)):
        in_head = any(a <= rm.start(1) < b for a, b in head_spans)
        cells = [Cell(t.lower(), a, inner) for t, a, inner in _CELL_RE.findall(rm.group(1))]
        numeric = [c for c in cells if c.numbers]
        colnum_row = bool(numeric) and all(_COLNUM_RE.match(_plain(c.inner)) for c in numeric)
        placed, new_carry, col = [], {}, 0
        for ci, c in enumerate(cells):
            while carry.get(col, 0) > 0:
                col += 1
            placed.append((ci, col, c))
            if c.rowspan > 1:
                new_carry.update({col + k: c.rowspan - 1 for k in range(c.colspan)})
            col += c.colspan
        width = max([col] + [k + 1 for k, v in carry.items() if v > 0])
        carry = {k: v - 1 for k, v in carry.items() if v > 1}
        carry.update(new_carry)
        header = bool(in_head or colnum_row or (cells and all(c.tag == "th" for c in cells)))
        if header and not in_header:
            segment += 1
        in_header = header
        out.append(_Row(r, segment, header, placed, width))
    return out


def clamp_header_rowspans(html: str) -> tuple[str, int]:
    """Stop header cells' rowspans at the end of the header block.

    Surya sometimes gives the stub header (``Method``) one row too many, so
    it covers the first body row's stub cell and pushes that whole row one
    column right. Only when the spill makes that body row wider than the
    header: then the body row has its own cell where the rowspan lands.
    """
    rows = _layout(html)
    fix: dict[tuple[int, int], int] = {}
    i = 0
    while i < len(rows):
        if not rows[i].header:
            i += 1
            continue
        start = i
        while i < len(rows) and rows[i].header:
            i += 1
        # rows[start:i] is a header block; rows[i] the first body row after it
        if i == len(rows) or rows[i].width <= max(r.width for r in rows[start:i]):
            continue
        fix.update({(r.index, ci): i - r.index for r in rows[start:i] for ci, _col, c in r.placed
                    if c.rowspan > 1 and r.index + c.rowspan > i})
    if not fix:
        return html, 0

    def row_sub(rm: re.Match, r=[-1]) -> str:
        r[0] += 1
        k = [-1]

        def cell_sub(cm: re.Match) -> str:
            k[0] += 1
            span = fix.get((r[0], k[0]))
            if span is None:
                return cm.group(0)
            attrs = _ROWSPAN_ATTR_RE.sub("" if span == 1 else f' rowspan="{span}"', cm.group(2))
            return f"<{cm.group(1)}{attrs}>{cm.group(3)}</{cm.group(1)}>"

        return rm.group(0).replace(rm.group(1), _CELL_RE.sub(cell_sub, rm.group(1)), 1)

    return _ROW_RE.sub(row_sub, html), len(fix)


_ROWSPAN_ATTR_RE = re.compile(r'\s*rowspan\s*=\s*"?\d+"?', re.I)


def _data_cell(col: int, c: Cell) -> bool:
    if col == 0:
        return False  # the stub column: row labels such as R^2/1000
    if c.tag == "th" or sum(ch.isalpha() for ch in _label_text(c.inner)) >= 2:
        return False
    # "114 598.30": one value that both sides split in two
    return not _SPACED_THOUSANDS_RE.fullmatch(_plain(c.inner))


def _data_grid(html: str) -> list[tuple[int, int, int, int, int, str]]:
    """``(segment, row, line, col_from_left, col_from_right, number)`` per data number.

    ``line`` is the number's position inside its cell (an estimate over its
    standard error). ``col_from_right`` counts from the row's last occupied
    column, for ragged rows that drop leading empty cells.
    """
    out = []
    for row in _layout(html):
        if row.header:
            continue
        for _ci, col, c in row.placed:
            if _data_cell(col, c):
                for line, n in enumerate(cell_numbers(_SCRIPT_RE.sub(" ", c.inner))):
                    out.append((row.segment, row.index, line, col, row.width - col - c.colspan, n))
    return out


def _anchor_list(grid, native_tokens) -> list[tuple[tuple, NativeToken]]:
    """Grid entries whose value occurs once in the grid and once among the text
    layer's data tokens, paired with that token. Years are left out: they head
    columns in rows Surya does not mark as a header."""
    by_key: dict[str, list[NativeToken]] = {}
    for t in native_tokens:
        if t.role == "data" and not t.owned:
            by_key.setdefault(numeric_key(t.text), []).append(t)
    counts: dict[str, int] = {}
    for *_pos, n in grid:
        counts[numeric_key(n)] = counts.get(numeric_key(n), 0) + 1
    return [(g, by_key[k][0]) for g in grid
            if counts[k := numeric_key(g[5])] == 1 and len(by_key.get(k, ())) == 1 and not _YEAR_RE.fullmatch(g[5])]


def _clusters(spans: list[tuple[float, float]]) -> list[int]:
    """Cluster id per interval: intervals that overlap, directly or through a chain, share one."""
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    ids, cid, end = [0] * len(spans), -1, float("-inf")
    for i in order:
        a, b = spans[i]
        if a >= end:
            cid += 1
        ids[i] = cid
        end = max(end, b) if a < end else b
    return ids


def _line_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The middle half of each vertical extent, for grouping tokens into printed lines.

    Glyph boxes include the font's full ascent and descent, which in tightly
    set monospace output (Stata logs) overlap the lines above and below and
    would chain a whole table into one line.
    """
    return [((a + b) / 2 - (b - a) / 4, (a + b) / 2 + (b - a) / 4) for a, b in spans]


def misplaced_cells(html: str, native_tokens: Iterable[NativeToken]) -> list[str]:
    """Data values the HTML puts in a different row or column than the page prints them.

    Anchors are values that occur once among the table's data cells and once
    among the text layer's data tokens, so their printed position is known.
    Two tests:

    - row: numbers in one HTML row (and line of a cell) share a printed line;
    - column: each HTML column (and line of a cell: an estimate and its
      standard error may print side by side) maps to one printed column
      band. A row passes when its cells line up counting columns from the
      left or from the right (ragged rows that drop empty cells are common
      and harmless when every value keeps its column).

    Values are reported, not repaired. Needs ``MIN_ANCHORS`` anchors.
    """
    anchors = [(r, line, (seg, cl, line), (seg, cr, line), n, t)
               for (seg, r, line, cl, cr, n), t in _anchor_list(_data_grid(html), native_tokens)]
    if len(anchors) < MIN_ANCHORS:
        return []
    boxes = [a[5].bbox for a in anchors]
    if sum(a[5].vertical for a in anchors) * 2 > len(anchors):
        boxes = [(b[1], b[0], b[3], b[2]) for b in boxes]  # sideways table
    line_of = _clusters(_line_spans([(b[1], b[3]) for b in boxes]))
    band_of = _clusters([(b[0], b[2]) for b in boxes])
    bad: set[int] = set()

    rows: dict[tuple[int, int], list[int]] = {}
    for i, a in enumerate(anchors):
        rows.setdefault((a[0], a[1]), []).append(i)
    # A one-line cell may be printed centred beside two-line cells (an N or a
    # mean next to estimates over standard errors): on the second line or
    # between the two. It is on its row when inside the span of that row's
    # two-line cells.
    two_line = {(r, cl) for _seg, r, line, cl, _cr, _n in _data_grid(html) if line >= 1}
    span: dict[int, tuple[float, float]] = {}
    for i, a in enumerate(anchors):
        if a[1] == 0 and (a[0], a[2][1]) in two_line:
            b = boxes[i]
            lo, hi = span.get(a[0], (b[1], b[3] + 1.6 * (b[3] - b[1])))
            span[a[0]] = (min(lo, b[1]), max(hi, b[3] + 1.6 * (b[3] - b[1])))

    def centred(i: int) -> bool:
        a = anchors[i]
        if a[1] != 0 or (a[0], a[2][1]) in two_line or a[0] not in span:
            return False
        lo, hi = span[a[0]]
        return lo - 1 <= boxes[i][1] and boxes[i][3] <= hi + 1

    for idx in rows.values():
        if any(anchors[i][1] == 0 for i in idx):
            multi = [i for i in idx if not centred(i)]
            idx = multi if multi else idx
        tally: dict[int, int] = {}
        for i in idx:
            tally[line_of[i]] = tally.get(line_of[i], 0) + 1
        if len(tally) < 2:
            continue
        top = max(tally, key=tally.get)
        bad |= {i for i in idx if line_of[i] != top} if tally[top] * 2 > len(idx) else set(idx)

    def dominant(side: int) -> dict[int, int]:
        seen: dict[int, dict[int, int]] = {}
        for i, a in enumerate(anchors):
            col = seen.setdefault(a[side], {})
            col[band_of[i]] = col.get(band_of[i], 0) + 1
        return {c: max(t, key=t.get) for c, t in seen.items()
                if max(t.values()) >= 2 and max(t.values()) * 2 > sum(t.values())}

    left, right = dominant(2), dominant(3)
    # Only a value squarely in another column's band is out of place; one
    # centred across two columns (an N under a mean/SD pair) forms a band of
    # its own or merges them.
    columns = set(left.values()) | set(right.values())
    by_row: dict[int, list[int]] = {}
    for i, a in enumerate(anchors):
        by_row.setdefault(a[0], []).append(i)
    for idx in by_row.values():
        off_l = {i for i in idx if anchors[i][2] in left and band_of[i] != left[anchors[i][2]]
                 and band_of[i] in columns}
        off_r = {i for i in idx if anchors[i][3] in right and band_of[i] != right[anchors[i][3]]
                 and band_of[i] in columns}
        if off_l and off_r:
            bad |= off_l if len(off_l) <= len(off_r) else off_r
    return [anchors[i][4] for i in sorted(bad)]


_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_YEAR_RE = re.compile(r"(19|20)\d\d")
_SPACED_THOUSANDS_RE = re.compile(r"-?\d{1,3}(?:[ \u00a0\u2009\u202f]\d{3})+(?:\.\d+)?")


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def lost_label_words(html: str, printed_words: Iterable[str], context_html: Iterable[str] = ()) -> list[str]:
    """Words printed in the table region that neither the table nor a neighbouring
    block (caption, note) holds: dropped row labels, panel or column headers.

    A printed word counts as held when its letters occur anywhere in the
    letters of the HTML, which absorbs the text layer's broken words: a lost
    ligature (``signi cant``, ``xed``), a hyphenated break, a footnote letter
    glued on (``incomeb``) or a subscript run together (``yit``). A letter
    glued to either end (``drentt`` for Δ Rent_t) is trimmed, and words under
    five letters are ignored. Words are returned only past ``LOST_WORDS_MIN``
    and ``LOST_WORDS_SHARE``.
    """
    printed = _words(" ".join(printed_words))
    if not printed:
        return []
    have = " ".join(_letters(_html.unescape(_TAG_RE.sub(" ", h))) for h in (html, *context_html))
    lost = sorted(w for w in printed if len(w) >= 5 and not any(
        w[a:len(w) - b] in have for a in (0, 1) for b in (0, 1)))
    if len(lost) >= LOST_WORDS_MIN and len(lost) >= LOST_WORDS_SHARE * len(printed):
        return lost
    return []


_LIGATURES = ("ffi", "ffl", "ff", "fi", "fl")


def check_text_table(html: str, printed_words: Iterable[str], context_html: Iterable[str] = (),
                     printed_lines: list[tuple[str, tuple]] | None = None) -> Finding | None:
    """Check a table that holds no numbers (option lists, key terms) by its words.

    Returns None when the table passes: every printed word of the region
    (three letters or more) is in the table or a neighbouring block, every
    table word (four letters or more) is printed there, and rows whose first
    cell can be found at the start of a printed line come in printed order.
    Otherwise the finding that caps the table at ``single-route``. Cells of
    symbols or single letters (``▲``, ``N``, ``X``) and LaTeX cannot be
    checked this way at all; neither can the pairing of cells within a row.
    """
    printed_text = " ".join(printed_words)
    printed = _words(printed_text)
    if not printed:
        return Finding("no_text_layer", "the table has no PDF text layer (an image)")
    rows = parse_rows(html)
    cells = [c for r in rows for c in r]
    if any(_MATH_RE.search(c.inner) for c in cells):
        return Finding("math_table", "a table without numbers that holds LaTeX; no text-layer check")
    marks = [p for c in cells if (p := _html.unescape(_plain(c.inner))) and sum(ch.isalnum() for ch in p) < 2]
    if marks:
        return Finding("mark_cells", "cells of symbols or single letters that the text layer cannot place", marks[:20])
    have = " ".join(_letters(_html.unescape(_TAG_RE.sub(" ", h))) for h in (html, *context_html))
    lost = sorted(w for w in printed if not any(w[a:len(w) - b] in have for a in (0, 1) for b in (0, 1)))
    if lost:
        return Finding("label_text_lost", "printed labels or headers missing from the table", lost)
    page_letters = _letters(printed_text)
    stripped = page_letters
    for lig in _LIGATURES:  # "signi cant": the text layer lost the ligature glyph
        stripped = stripped.replace(lig, "")
    extra = []
    for w in _words(_html.unescape(_TAG_RE.sub(" ", html))):
        if len(w) < 4 or w in page_letters:
            continue
        bare = w
        for lig in _LIGATURES:
            bare = bare.replace(lig, "")
        if bare not in stripped and not any(bare.replace(lig, "", 1) in page_letters for lig in _LIGATURES):
            extra.append(w)
    if extra:
        return Finding("text_unmatched", "table words absent from the text layer", sorted(extra))
    if printed_lines:
        starts = [(_letters(text), (b[1] + b[3]) / 2) for text, b in printed_lines]
        ys = []
        for r in rows:
            key = _letters(_html.unescape(_plain(r[0].inner)))[:16] if r else ""
            if len(key) < 5:
                continue
            hits = {round(y, 1) for letters, y in starts if letters.startswith(key)}
            if len(hits) == 1:
                ys.append(hits.pop())
        if any(b <= a for a, b in zip(ys, ys[1:])):
            return Finding("text_rows_out_of_order", "rows in a different order than the page prints them")
    return None


_TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")


def _swap_magnitude(inner: str, old: str, new: str) -> str | None:
    """Replace the one occurrence of number ``old`` in a cell's text (outside
    tags) with ``new``; None when it is not there exactly once."""
    pat = re.compile(rf"(?<![\d.,]){re.escape(old)}(?![\d]|[.,]\d)")
    parts = _TAG_SPLIT_RE.split(inner)
    hits = [(i, m) for i, p in enumerate(parts) if not p.startswith("<") for m in pat.finditer(p)]
    if len(hits) != 1:
        return None
    i, m = hits[0]
    parts[i] = parts[i][:m.start()] + new + parts[i][m.end():]
    return "".join(parts)


def _set_cells(html: str, edits: dict[tuple[int, int], str]) -> str:
    """Replace the inner HTML of cells keyed ``(row index, cell index)``."""

    def row_sub(r: int, rm: re.Match) -> str:
        k = -1

        def cell_sub(m: re.Match) -> str:
            nonlocal k
            k += 1
            if (r, k) not in edits:
                return m.group(0)
            return f"<{m.group(1)}{m.group(2)}>{edits[(r, k)]}</{m.group(1)}>"

        return rm.group(0)[:rm.start(1) - rm.start(0)] + _CELL_RE.sub(cell_sub, rm.group(1)) \
            + rm.group(0)[rm.end(1) - rm.start(0):]

    rows = iter(range(10 ** 9))
    return _ROW_RE.sub(lambda rm: row_sub(next(rows), rm), html)


def fill_from_text_layer(
    html: str,
    native: list[NativeToken],
    missing: list[NativeToken],
    unmatched: list[str],
    printed_lines: list[tuple[str, tuple]],
) -> tuple[str, list[dict]] | None:
    """Put text-layer values into the cells where the page prints them.

    Born-digital pages only. Anchors (values unique in both the table and the
    text layer) tie each HTML row to a printed line and each HTML column to a
    printed column band. Rows without an anchor are placed by their label,
    found at the start of a text-layer line between the neighbouring rows.
    A printed line of data numbers belongs to the placed row above it; its
    rank below that row's first line is its line in the cell (an estimate
    over its standard error). Unplaced rows in between must hold no numbers
    in the HTML, and the lines must not outnumber the placed row's own.

    A text-layer value Surya lacks (``missing``, any printed copy of it) that
    falls in exactly one row line and one column band then either replaces a
    Surya number there that the text layer lacks (``unmatched``: a misread
    digit), keeping Surya's stars and brackets, or fills an empty cell with
    the brackets and stars printed around it. Nothing else in the table
    changes: labels, headers and structure stay Surya's.

    Refused, cell by cell: an inferred text-layer sign (control glyph,
    spaced dash) for a value Surya did not read; a Surya minus the text
    layer lacks (broken fonts drop minus glyphs); ragged rows whose own
    anchors do not settle how their cells line up; fills when the table has
    stars but the text layer prints none (the fonts draw them as glyphs the
    layer omits). The whole fill is refused when the anchors contradict
    the line model: rows out of printed order, or one row on two lines (a
    misplaced value, not a missing one). The caller re-verifies the result.
    """
    data = [t for t in native if t.role == "data" and not t.owned]
    if not missing or not data:
        return None
    grid = _data_grid(html)
    anchors = _anchor_list(grid, native)
    if len(anchors) < MIN_ANCHORS:
        return None

    # Work in reading coordinates: rows run down, columns across.
    sideways = sum(t.vertical for t in data) * 2 > len(data)

    def turn(b):
        return (b[1], b[0], b[3], b[2]) if sideways else tuple(b)

    box = {id(t): turn(t.bbox) for t in data}
    lines_in = [(text, turn(b)) for text, b in printed_lines]
    ys = [box[id(t)][1] for _g, t in sorted(anchors, key=lambda a: (a[0][1], a[0][2]))]
    if sideways and ys and ys[0] > ys[-1]:  # rows run right to left or bottom to top
        box = {k: (b[0], -b[3], b[2], -b[1]) for k, b in box.items()}
        lines_in = [(text, (b[0], -b[3], b[2], -b[1])) for text, b in lines_in]

    line_ids = _clusters(_line_spans([(box[id(t)][1], box[id(t)][3]) for t in data]))
    line_of = {id(t): i for t, i in zip(data, line_ids)}
    extent: dict[int, list[float]] = {}
    for t, i in zip(data, line_ids):
        e = extent.setdefault(i, [box[id(t)][1], box[id(t)][3]])
        e[0], e[1] = min(e[0], box[id(t)][1]), max(e[1], box[id(t)][3])
    center = {i: (a + b) / 2 for i, (a, b) in extent.items()}
    heights = sorted(b - a for a, b in extent.values())
    tol = 0.5 * heights[len(heights) // 2]

    rows = _layout(html)
    order = [row.index for row in rows]
    by_index = {row.index: row for row in rows}
    own_lines = {row.index: max([len(cell_numbers(_SCRIPT_RE.sub(" ", c.inner)))
                                 for _ci, col, c in row.placed if _data_cell(col, c)] + [0])
                 for row in rows if not row.header}
    ypos: dict[int, float] = {}
    for (_seg, r, li, *_rest), t in anchors:
        if li == 0:
            if r in ypos and abs(ypos[r] - center[line_of[id(t)]]) > tol:
                return None
            ypos[r] = center[line_of[id(t)]]
    placed_y = [ypos[r] for r in order if r in ypos]
    if any(b <= a for a, b in zip(placed_y, placed_y[1:])):
        return None
    def alnum(text: str) -> str:  # labels may differ only in digits: "5-10 miles", "10-15 miles"
        return re.sub(r"[^a-z0-9]", "", text.lower())

    starts = [(alnum(text), (b[1] + b[3]) / 2) for text, b in lines_in]
    for k, r in enumerate(order):
        stub = next((c for _ci, col, c in by_index[r].placed if col == 0), None)
        label = alnum(_html.unescape(_label_text(stub.inner))) if stub else ""
        if r in ypos or sum(ch.isalpha() for ch in label) < 4:
            continue
        lo = max((ypos[x] for x in order[:k] if x in ypos), default=float("-inf"))
        hi = min((ypos[x] for x in order[k + 1:] if x in ypos), default=float("inf"))
        # The printed line may hold only the first line of a wrapped label.
        hits = {round(y, 1) for letters, y in starts
                if sum(ch.isalpha() for ch in letters) >= 4 and (letters.startswith(label) or label.startswith(letters))
                and lo + tol < y < hi - tol}
        if len(hits) == 1:
            ypos[r] = hits.pop()

    depth: dict[int, int] = {}
    for (seg, _r, li, *_rest), _t in anchors:
        depth[seg] = max(depth.get(seg, 0), li + 1)
    gaps: dict[int, list[int]] = {}  # position in order of the placed row above -> printed lines
    for lid in sorted(center, key=center.get):
        k = max((j for j, r in enumerate(order) if r in ypos and ypos[r] <= center[lid] + tol), default=None)
        if k is not None:
            gaps.setdefault(k, []).append(lid)
    rank: dict[int, tuple[int, int]] = {}
    for k, lids in gaps.items():
        r = order[k]
        nxt = next((j for j in range(k + 1, len(order)) if order[j] in ypos), len(order))
        between = order[k + 1:nxt]
        if between:
            row = by_index[r]
            cap = own_lines.get(r) or depth.get(row.segment, 1)
            if len(lids) > cap or any(by_index[x].header or own_lines.get(x) for x in between):
                continue
        rank.update({lid: (r, i) for i, lid in enumerate(lids)})
    for (_seg, r, li, *_rest), t in anchors:
        got = rank.get(line_of[id(t)])
        if got is not None and got != (r, li):
            return None

    def bands(side: int) -> dict[tuple[int, int], list[float]]:
        out: dict[tuple[int, int], list[float]] = {}
        for g, t in anchors:
            b = out.setdefault((g[0], g[side]), [box[id(t)][0], box[id(t)][2]])
            b[0], b[1] = min(b[0], box[id(t)][0]), max(b[1], box[id(t)][2])
        return out

    left, right = bands(3), bands(4)

    def inside(band, b) -> bool:
        return band is not None and min(band[1], b[2]) - max(band[0], b[0]) > 0

    full: dict[int, int] = {}
    for row in rows:
        if not row.header:
            full[row.segment] = max(full.get(row.segment, 0), row.width)
    sides_of: dict[int, list[str]] = {}
    for row in rows:
        if row.header:
            continue
        if row.width == full[row.segment]:
            sides_of[row.index] = ["left"]
            continue
        mine = [(g, t) for g, t in anchors if g[1] == row.index]
        ok_l = bool(mine) and all(inside(left.get((g[0], g[3])), box[id(t)]) for g, t in mine)
        ok_r = bool(mine) and all(inside(right.get((g[0], g[4])), box[id(t)]) for g, t in mine)
        # Both: the row's anchors cannot tell; a target must then be the same cell either way.
        sides_of[row.index] = ["left"] * ok_l + ["right"] * ok_r

    def target(row: _Row, side: str, b) -> tuple | None:
        table = left if side == "left" else right
        keys = [k for (seg, k), band in table.items() if seg == row.segment and inside(band, b)]
        if len(keys) != 1:
            return None
        return next(((ci, col, c) for ci, col, c in row.placed if c.colspan == 1 and (
            col if side == "left" else row.width - col - 1) == keys[0]), None)

    need = Counter(numeric_key(t.text) for t in missing)
    candidates = [t for t in data if need[numeric_key(t.text)]]
    minus = "\u2212" if "\u2212" in html else "-"
    stars_ok = "*" not in html or any("*" in text or "\u2217" in text for text, _b in printed_lines)
    todo = Counter(unmatched)
    targets: dict[tuple[int, int], list[tuple[int, NativeToken]]] = {}
    for t in candidates:
        got = rank.get(line_of[id(t)])
        if got is None:
            continue
        r, li = got
        row = by_index[r]
        if row.header or not sides_of.get(r) or li >= depth.get(row.segment, 1):
            continue
        found = [target(row, side, box[id(t)]) for side in sides_of[r]]
        cells = {c[0]: c for c in found if c is not None}
        if None in found:
            continue
        if len(cells) != 1:
            continue
        cell = next(iter(cells.values()))
        if _data_cell(cell[1], cell[2]):
            targets.setdefault((r, cell[0]), []).append((li, t))

    edits: dict[tuple[int, int], str] = {}
    repairs: list[dict] = []
    for (r, ci), items in sorted(targets.items()):
        _ci, col, c = by_index[r].placed[ci]
        nums = cell_numbers(_SCRIPT_RE.sub(" ", c.inner))
        items.sort(key=lambda x: x[0])
        if len({li for li, _t in items}) != len(items):
            continue
        added = [(li, t) for li, t in items if li >= len(nums)]
        if nums and added:  # lines Surya left out below its own (a standard error)
            if not stars_ok or [li for li, _t in added] != list(range(len(nums), len(nums) + len(added))) or any(
                    not need[numeric_key(t.text)] or (t.negative and t.sign_source not in AUTHORITATIVE_SIGN_SOURCES)
                    for _li, t in added):
                added = []
            for _li, t in added:
                need[numeric_key(t.text)] -= 1
                repairs.append({"kind": "cell", "before": "", "after": t.text, "route": "native_cell",
                                "row": r, "col": col})
        if nums:  # misread digits: same line, Surya's value unbacked
            inner = c.inner + "".join("<br/>" + t.pre + t.text.replace("-", minus, 1) + t.post for _li, t in added)
            for li, t in items:
                if li >= len(nums) or not todo[nums[li]] or not need[numeric_key(t.text)]:
                    continue
                old = nums[li]
                if old.startswith("-") != t.negative and (
                        old.startswith("-") or t.sign_source not in AUTHORITATIVE_SIGN_SOURCES):
                    continue
                new = t.magnitude if old.startswith("-") == t.negative else minus + t.magnitude
                swapped = _swap_magnitude(inner, old.lstrip("-"), new)
                if swapped is None:
                    continue
                inner = swapped
                todo[old] -= 1
                need[numeric_key(t.text)] -= 1
                repairs.append({"kind": "cell", "before": old, "after": t.text, "route": "native_cell",
                                "row": r, "col": col})
            if inner != c.inner:
                edits[(r, ci)] = inner
        elif stars_ok and [li for li, _t in items] == list(range(len(items))):  # an empty cell
            if any(not need[numeric_key(t.text)] or (t.negative and t.sign_source not in AUTHORITATIVE_SIGN_SOURCES)
                   for _li, t in items):
                continue
            for _li, t in items:
                need[numeric_key(t.text)] -= 1
            edits[(r, ci)] = "<br/>".join(t.pre + t.text.replace("-", minus, 1) + t.post for _li, t in items)
            repairs.extend({"kind": "cell", "before": "", "after": t.text, "route": "native_cell",
                            "row": r, "col": col} for _li, t in items)
    if not edits:
        return None
    return _set_cells(html, edits), repairs


_STARS_RE = re.compile(r"[*\u2217]+")


def fix_stars(html: str, native_tokens: Iterable[NativeToken]) -> tuple[str, list[dict]]:
    """Set significance stars from the text layer where it prints them.

    Only anchored estimates (values unique in both the table and the text
    layer, first line of their cell) whose text-layer token carries stars
    (after the value or after a space; or after the standard error below
    it, which only corrects a star run already in the cell): there the text layer is exact, while
    Surya often reads ``**`` as ``***``. A token without stars settles
    nothing (stars drawn as glyphs the text layer omits), so its cell is
    left alone.
    """
    stars_of: dict[tuple[int, int], tuple[str, str, bool]] = {}
    for (_seg, r, line, col, _cr, n), t in _anchor_list(_data_grid(html), native_tokens):
        want = t.stars or t.se_stars
        # Stars after a standard error may sit on its line or the estimate's:
        # there only correct a star run Surya already put in this cell.
        replace_only = t.pre[-1:] in ("(", "[") or not t.stars
        if line == 0 and want:
            stars_of[(r, col)] = (n, want, replace_only)
    if not stars_of:
        return html, []
    edits: dict[tuple[int, int], str] = {}
    repairs: list[dict] = []
    for row in _layout(html):
        for ci, col, c in row.placed:
            if (row.index, col) not in stars_of:
                continue
            n, want, replace_only = stars_of[(row.index, col)]
            runs = list(_STARS_RE.finditer(c.inner))
            have = "".join(m.group(0) for m in runs)
            if have == want or len(runs) > 1 or (replace_only and not runs):
                continue
            if runs:
                m = runs[0]
                inner = c.inner[:m.start()] + want + c.inner[m.end():]
            else:
                mag = n.lstrip("-")
                inner = _swap_magnitude(c.inner, mag, mag + want)
                if inner is None:
                    continue
            edits[(row.index, ci)] = inner
            repairs.append({"kind": "stars", "value": n, "before": have, "after": want,
                            "route": "native_stars", "row": row.index, "col": col})
    return (_set_cells(html, edits), repairs) if edits else (html, [])


#: A table's text layer must attach stars to at least this many anchored
#: values before a value without stars counts as printed without them.
STARS_BACKED_MIN = 3


def unbacked_stars(html: str, native_tokens: Iterable[NativeToken]) -> list[str]:
    """Anchored values where Surya has stars the text layer does not print.

    Only in tables whose text layer carries stars elsewhere, so they are not
    drawn as glyphs it omits. Seen on the page: stars Surya added, and a
    ``+`` marker read as ``*``.
    """
    tokens = list(native_tokens)
    anchors = [(k, t) for k, t in _anchor_list(_data_grid(html), tokens) if k[2] == 0]
    if sum(bool(t.stars or t.se_stars) for _k, t in anchors) < STARS_BACKED_MIN:
        return []
    bare = {(k[1], k[3]): k[5] for k, t in anchors
            if not (t.stars or t.se_stars or t.vertical or t.pre[-1:] in ("(", "[") or _spaced_group(t, tokens))}
    out = []
    for row in _layout(html):
        for _ci, col, c in row.placed:
            if (row.index, col) in bare and _STARS_RE.search(c.inner):
                out.append(bare[(row.index, col)])
    return out


def _spaced_group(t: NativeToken, tokens: list[NativeToken]) -> bool:
    """``t`` is the head of a number printed with spaced thousands (``1 469.215``)."""
    height = t.bbox[3] - t.bbox[1]
    return any(o is not t and abs(o.bbox[1] - t.bbox[1]) < height / 2
               and 0 <= o.bbox[0] - t.bbox[2] <= height / 2 and o.magnitude[:3].isdigit()
               for o in tokens)


_HEADER_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_BR_RE = re.compile(r"<br\s*/?>", re.I)


def misplaced_headers(html: str, native_tokens: Iterable[NativeToken],
                      word_boxes: list[tuple[str, tuple]]) -> list[str]:
    """Column headers the HTML puts over a different column than the page prints them.

    Per panel, each data column's printed band comes from its anchored values
    (rows of the panel's usual width, so one shifted row cannot move it). A
    header cell is located by its least repeated word among the printed words
    above the data; a word used by several header cells is paired with its
    printed occurrences left to right, only when both counts agree. A header
    printed inside another column's band and clear of every column it spans
    in the HTML is misplaced. So is a labelled header cell in a column no
    body row reaches. A sideways table (most anchors on vertical lines) is
    turned into its own frame first. ``word_boxes`` are the region's
    text-layer words: ``(word, displayed-page box[, vertical])``.
    """
    rows = _layout(html)
    if not rows or not word_boxes:
        return []
    anchors = _anchor_list(_data_grid(html), list(native_tokens))
    page_words = [(w[0], w[1], len(w) > 2 and w[2]) for w in word_boxes]
    flip = None  # box transform into the table's own frame for a sideways table
    if anchors and sum(t.vertical for _k, t in anchors) * 2 > len(anchors):
        # Columns run along the page's y axis; which way depends on the rotation.
        pts = [(k[3], (t.bbox[1] + t.bbox[3]) / 2) for k, t in anchors]
        mc = sum(c for c, _y in pts) / len(pts)
        my = sum(y for _c, y in pts) / len(pts)
        if sum((c - mc) * (y - my) for c, y in pts) < 0:  # read bottom to top
            flip = lambda b: (-b[3], b[0], -b[1], b[2])  # noqa: E731
        else:  # read top to bottom
            flip = lambda b: (b[1], -b[2], b[3], -b[0])  # noqa: E731
        anchors = [(k, replace(t, bbox=flip(t.bbox), vertical=False)) for k, t in anchors if t.vertical]
        page_words = [(w, flip(b), False) for w, b, v in page_words if v]
    page_words = [(w, b) for w, b, v in page_words if not v]
    out: list[str] = []
    body_width = max((r.width for r in rows if not r.header), default=0)
    for row in rows:
        if row.header:
            for _ci, col, c in row.placed:
                if col >= body_width and _HEADER_WORD_RE.search(_label_text(_BR_RE.sub(" ", c.inner))):
                    out.append(f"header in column {col + 1}, beyond the data")
    segments: dict[int, list[_Row]] = {}
    for r in rows:
        segments.setdefault(r.segment, []).append(r)
    for seg, srows in segments.items():
        body = [r for r in srows if not r.header]
        heads = [r for r in srows if r.header]
        if not body or not heads:
            continue
        usual = Counter(r.width for r in body).most_common(1)[0][0]
        rows_ok = {r.index for r in body if r.width == usual}
        boxes: dict[int, list[tuple]] = {}
        for (sg, r, _line, col, _cr, _n), t in anchors:
            if sg == seg and r in rows_ok and not t.vertical:
                boxes.setdefault(col, []).append(t.bbox)
        if len(boxes) < 2:
            continue
        bands = {c: (min(b[0] for b in v), max(b[2] for b in v)) for c, v in boxes.items()}
        top = min(b[1] for v in boxes.values() for b in v)
        cells = []
        for row in heads:
            for _ci, col, c in row.placed:
                if col == 0 or not any(k in bands for k in range(col, col + c.colspan)):
                    continue
                words = [w.lower() for w in _HEADER_WORD_RE.findall(_label_text(_BR_RE.sub(" ", c.inner)))]
                if words:
                    cells.append(((col, col + c.colspan - 1), words))
        freq = Counter(w for _span, words in cells for w in set(words))
        by_key: dict[str, list[tuple[int, int]]] = {}
        for span, words in cells:
            by_key.setdefault(min(words, key=lambda w: (freq[w], words.index(w))), []).append(span)
        for key, spans in by_key.items():
            printed = sorted((b for w, b in page_words if b[3] <= top + 2 and key in w.lower()),
                             key=lambda b: b[0])
            if not (len(printed) == len(spans) == freq[key]):
                continue
            for (c0, c1), b in zip(sorted(spans), printed):
                cx = (b[0] + b[2]) / 2

                def gap(band: tuple[float, float]) -> float:
                    return 0.0 if band[0] <= cx <= band[1] else min(abs(cx - band[0]), abs(cx - band[1]))

                own = [bands[k] for k in range(c0, c1 + 1) if k in bands]
                own_band = (min(x for x, _ in own), max(x for _, x in own))
                best = min(bands, key=lambda k: gap(bands[k]))
                if not c0 <= best <= c1 and gap(own_band) > 0 and gap(bands[best]) == 0:
                    out.append(f"'{key}' printed over column {best + 1}, not {c0 + 1}"
                               + (f"-{c1 + 1}" if c1 > c0 else ""))
    return out


def column_streams(html: str) -> list[tuple[str, ...]]:
    """Numbers of each logical column read top to bottom, stub column left out.

    Independent of row layout: an estimate and its standard error in one
    cell or in two rows give the same stream. Empty cells do not count, but a
    value shifted into a neighbouring column changes two streams. Only data
    cells count, and LaTeX is left out: one reader writes ``\gamma_1`` where
    the other writes a Unicode subscript, and those digits are labels.
    """
    cols: dict[int, list[str]] = {}
    for row in _layout(html):
        if row.header:
            continue
        for _ci, col, c in row.placed:
            if not _data_cell(col, c):
                continue
            for n in cell_numbers(_SCRIPT_RE.sub(" ", _MATH_RE.sub(" ", c.inner))):
                cols.setdefault(col, []).append(numeric_key(n))
    return [tuple(cols[k]) for k in sorted(cols)]


def readings_agree(html_a: str, html_b: str) -> bool:
    """Two readings of one table hold the same numbers, column by column, in order."""
    a = column_streams(html_a)
    return bool(a) and any(a) and a == column_streams(html_b)


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
    printed_words: Iterable[str] | None = None,
    context_html: Iterable[str] = (),
    printed_lines: list[tuple[str, tuple]] | None = None,
    fill: bool = True,
    printed_word_boxes: list[tuple[str, tuple]] | None = None,
    _formula_attempt: bool = False,
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

    On a born-digital page, values that agree but sit in the wrong row or
    column (:func:`misplaced_cells`) make the table ``unresolved``, and
    ``printed_words`` (the region's text-layer words) missing from the table
    and ``context_html`` (:func:`lost_label_words`) cap it at ``single-route``:
    the numbers are right but what they belong to is not.

    ``printed_lines`` (the region's text-layer lines with their boxes) enable
    :func:`fill_from_text_layer` on a born-digital table that fails on
    unmatched or missing numbers: the result is re-checked in full and kept
    only if it passes, as ``repaired`` (or ``single-route`` when labels are
    lost); otherwise the original check stands. ``fill=False`` skips it.

    A formula table (at least :data:`MATH_TABLE_SHARE` of its cells LaTeX)
    whose LaTeX sits in labels (under that share of its body cells outside the
    stub column) gets the full check too, and keeps its result only when that
    passes as ``verified`` or ``repaired``: the numbers then match the text
    layer one to one like any other table. Tables of formulas never do. Otherwise digits inside LaTeX may be what
    disagrees, so the table is ``single-route`` with a ``math_table`` finding,
    as before, never ``unresolved``.

    ``printed_word_boxes`` (the region's text-layer words with boxes) enable
    :func:`misplaced_headers`: headers printed over other columns than the
    HTML gives them cap the table at ``single-route`` (``header_structure``).

    ``native_tokens`` should be clipped to the table's region in displayed-page
    coordinates. ``context_numbers`` are numbers that legitimately appear in
    the region but outside the table body (captions, notes); they are removed
    from the native pool before the truncation check. Tokens flagged ``owned``
    or with a ``label``/``colnum`` role never count as truncation.
    """
    native = list(native_tokens)
    source_html = html
    html, clamped = clamp_header_rowspans(html)
    html, merged = split_estimate_se(html)
    rows = parse_rows(html)
    roled = numbers_with_roles(html)
    surya = [n for n, _role in roled]
    check = TableCheck("verified", html, surya_numbers=len(surya), native_numbers=len(native))
    if merged:
        check.findings.append(Finding("merged_estimate_se", f"split {merged} estimate/uncertainty cells"))
    if clamped:
        check.findings.append(Finding(
            "header_rowspan_clamped", f"{clamped} header rowspan(s) cut back to the header rows"))
    check.findings.extend(_structure(rows))

    if not native:
        check.status = "single-route"
        if surya:
            # A picture of a table on a born-digital page (pasted software
            # output): nothing to check against, like a scanned page.
            check.findings.append(Finding("no_text_layer", "the table has no PDF text layer (an image)"))
        elif native_is_ocr or printed_words is None:
            check.findings.append(Finding("no_numbers", "the table holds no numbers; nothing to compare"))
        elif (failed := check_text_table(check.html, printed_words, context_html, printed_lines)) is not None:
            check.findings.append(failed)
        else:
            check.findings.append(Finding("text_table", "no numbers; every word matches the text layer"))
            empty = [f.detail for f in check.findings if f.kind == "empty_body_column"]
            if empty:
                check.findings.append(Finding(
                    "header_structure", "column headers or column numbers do not match the page", empty))
            else:
                check.status = "verified"
        return check

    # Formula tables: digits are subscripts/exponents inside LaTeX, which the
    # text layer cannot represent. Numeric comparison is meaningless; treat
    # the table like display math.
    all_cells = [c for r in rows for c in r]
    math_cells = [c for c in all_cells if _MATH_RE.search(c.inner)]
    if all_cells and len(math_cells) / len(all_cells) >= MATH_TABLE_SHARE and not _formula_attempt:
        # Only when the LaTeX is in the labels: data cells that are formulas
        # hold far more than the digits the text layer can confirm.
        body = [c for r in _layout(html) if not r.header for _ci, col, c in r.placed if col > 0]
        if body and sum(bool(_MATH_RE.search(c.inner)) for c in body) / len(body) < MATH_TABLE_SHARE:
            full = verify_table(source_html, native, context_numbers, native_is_ocr, printed_words,
                                context_html, printed_lines, fill, printed_word_boxes, _formula_attempt=True)
            if full.status in ("verified", "repaired"):
                return full
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
    lost_colnums = [t.text for t in left if not t.owned and t.role == "colnum"]
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
        if fill and printed_lines is not None and (unmatched or missing):
            filled = fill_from_text_layer(
                check.html, native, [t for t in left if not t.owned and t.role == "data"], unmatched, printed_lines)
            if filled is not None:
                again = verify_table(filled[0], native, context_numbers, False, printed_words, context_html,
                                     printed_lines, fill=False)
                if again.status in ("verified", "repaired", "single-route"):
                    again.repairs = check.repairs + filled[1] + again.repairs
                    if again.status != "single-route":
                        again.status = "repaired"
                    again.surya_numbers = check.surya_numbers
                    return again
                check.findings.append(Finding(
                    "text_layer_fill_failed", "cells filled from the text layer did not pass the check",
                    sorted({f.kind for f in again.findings} - INFORMATIONAL)))
    elif check.repairs:
        check.status = "repaired"
    if check.status in ("verified", "repaired"):
        check.html, star_fixes = fix_stars(check.html, native)
        if star_fixes:
            check.repairs.extend(star_fixes)
            check.status = "repaired"
        if extra := unbacked_stars(check.html, native):
            check.status = "single-route"
            check.findings.append(Finding(
                "stars_unconfirmed", "significance stars the PDF text layer does not print", extra[:40]))
        if moved := misplaced_cells(check.html, native):
            check.status = "unresolved"
            check.findings.append(Finding(
                "misplaced_cells", "values in a different row or column than the page prints them", moved))
        elif printed_words is not None and (lost := lost_label_words(check.html, printed_words, context_html)):
            check.status = "single-route"
            check.findings.append(Finding(
                "label_text_lost", "printed labels or headers missing from the table", lost))
    if check.status in ("verified", "repaired"):
        # Headers that do not sit over the data: a body column no header or
        # value fills, or printed column numbers the table dropped. In a
        # 90-paper audit every such table had shifted, merged or missing
        # column headers while its numbers were right.
        empty = [f.detail for f in check.findings if f.kind == "empty_body_column"]
        if not native_is_ocr and printed_word_boxes:
            empty += misplaced_headers(check.html, native, printed_word_boxes)
        if empty or lost_colnums:
            check.status = "single-route"
            check.findings.append(Finding(
                "header_structure", "column headers or column numbers do not match the page",
                [f"({n})" for n in lost_colnums] + empty))
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
