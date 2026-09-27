"""Font-aware numeric tokens from a PDF's native text layer.

The native text layer is the independent check against the Surya VLM route,
but journal fonts often encode the minus sign badly:

- a known minus code point (U+2212, U+2AFA, hyphens, en/em dash);
- a control character (``\\x01``) from a symbol font with no ToUnicode entry;
- a plain ``2`` drawn from a separate symbol font (so ``-.0016`` extracts as
  ``2.0016``), or, in some Advent fonts, a ``)``.

The last kinds are recognised by font, never by value: a sign candidate must
come from a font different from the digits it precedes and be directly
followed by a digit or decimal point on the same line. Every token records
how its sign was established (``sign_source``) so the verifier can refuse
deterministic substitution when the sign was only inferred.

Trust levels:

- ``codepoint`` and ``font_glyph`` (a ``2`` or ``)`` from a validated symbol
  font lacking a full digit set) are authoritative signs.
- ``control_glyph`` is inferred only. Symbol fonts reuse control codes for
  several operators (Diamond and McQuade's ``AdvP4C4E46`` draws ``\\x03``
  before an exponent in equations), so a control-glyph sign may confirm an
  agreeing Surya value but must never drive a substitution.
- ``codepoint_spaced`` (a minus code point, a narrow space, then digits, as
  in ``- 1.78``) is also inferred only: it confirms but never substitutes.

Control and private-use glyphs next to digits are classified by their drawn
shape (:func:`classify_glyph`): a thin bar can be a sign, a baseline dot is a
decimal point, a dot with a descending tail is a comma, and anything taller
(``×``, ``≈``, ``σ``) is an operator or letter, never a sign. Other text-layer
quirks handled here: digits of a validated symbol font are operators (``5``
for ``=``), and a zero-width decimal point followed by a synthetic space
(``37. 2``) is one number.

Roles keep text numbers from counting as table cells: a number glued to or
set at word spacing after a word is a ``label`` (``R3``, ``Table 6``,
``n = 495``), a ``(1) (2) (3)`` row is ``colnum``, and a number on a line of
running text (six or more words, no column gap: a note or title inside the
table region) is a ``note``. An exponent is always data.

Coordinates: PyMuPDF reports text in *unrotated* page space, while page
renders, pixmap clips and Surya's boxes live on the *displayed* page (with
``/Rotate`` applied). Every bbox here is converted to displayed-page
coordinates, so clips from Surya boxes select the right text on rotated pages.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

AUTHORITATIVE_SIGN_SOURCES = frozenset({"none", "codepoint", "font_glyph"})

# U+2010/U+2011 hyphens are what Word and Calibri tables often emit for minus.
KNOWN_MINUS = {"\u2212", "\u2afa", "-", "\u2013", "\u2014", "\u2012", "\u2010", "\u2011", "\ufe63", "\uff0d"}
_NUM_CHARS = set("0123456789.,")
_DIGITS = frozenset("0123456789")
_ORDINALS = frozenset({"st", "nd", "rd", "th"})
#: Parenthesised small integers in one row, e.g. ``(1) (2) (3)``, are column numbers.
COLNUM_MAX = 30
_PRE_MARKS = "(["
_POST_MARKS = ")]*\u2217\u2020\u2021%"
COLNUM_MIN_RUN = 3
#: Instances of one odd glyph rendered to classify its shape.
SHAPE_SAMPLES = 5
#: A line of running text has this many words and no gap wider than
#: PROSE_MAX_GAP text heights; a table row always has a column gap.
PROSE_MIN_WORDS = 6
PROSE_MAX_GAP = 0.6


@dataclass
class NativeToken:
    text: str  # normalized: optional leading "-", digits, "." and ","
    negative: bool
    sign_source: str  # "none" | "codepoint" | "codepoint_spaced" | "control_glyph" | "font_glyph"
    bbox: tuple[float, float, float, float]
    page: int
    fonts: tuple[str, ...] = field(default_factory=tuple)
    #: "data"; "label" when glued to a preceding letter (``y2000``, ``t-1``,
    #: ``R3``), after a word at word spacing (``Table 6``) or with an ordinal
    #: suffix (``5th``); "colnum" for a ``(1) (2) (3)`` row; "note" on a line
    #: of running text (:func:`is_prose`).
    role: str = "data"
    #: Set by the caller when another sidecar block over this spot contains
    #: the number (a caption or note Surya kept outside the table).
    owned: bool = False
    #: Exact text between the previous number on the same line and this one,
    #: or None when this is the first number of its line inside the clip.
    sep_before: str | None = None
    #: A letter directly before or after the number (``R3``, ``0125d``).
    glued: bool = False
    #: An unsigned number with a detached minus code point before it (``- 16.9``)
    #: that geometry did not confirm as a sign. In an OCR text layer, whose
    #: character boxes are synthetic, that dash may be the number's sign.
    dash_before: bool = False
    #: On a line that runs top to bottom on the displayed page (a sideways table).
    vertical: bool = False
    #: Brackets printed directly before the number, and brackets, stars,
    #: daggers or a percent sign directly after it: ``(`` and ``)`` of a
    #: standard error, ``**`` of a coefficient.
    pre: str = ""
    post: str = ""
    #: Significance stars printed directly after the number or after one or
    #: two spaces on its line.
    stars: str = ""
    #: For an estimate without stars of its own: the stars printed after the
    #: bracketed standard error directly below it (:func:`_attach_se_stars`).
    se_stars: str = ""

    @property
    def magnitude(self) -> str:
        return self.text.lstrip("-")


@dataclass
class GlyphProfile:
    """Per-document glyph facts from :func:`glyph_profile`."""

    #: Symbol fonts whose ``2`` before a number is drawn as a minus bar.
    minus_fonts: set[str] = field(default_factory=set)
    #: Symbol fonts whose ``)`` before a number is drawn as a minus bar.
    paren_minus_fonts: set[str] = field(default_factory=set)
    #: ``(font, char) -> "bar" | "period" | "comma" | "other" | "unknown"`` for
    #: control and private-use glyphs drawn next to digits (and candidate
    #: ``2``/``)`` minus glyphs).
    glyph_shapes: dict[tuple[str, str], str] = field(default_factory=dict)
    #: Fonts with a few digits that sit before numbers (math symbol and
    #: extension fonts): their digit codes are operators or delimiter pieces,
    #: never number characters. Defaults to ``minus_fonts``.
    symbol_fonts: set[str] | None = None

    def __post_init__(self):
        if self.symbol_fonts is None:
            self.symbol_fonts = set(self.minus_fonts)


def _odd_glyph(ch: str) -> bool:
    """A control or private-use code: a font's own glyph with no Unicode meaning."""
    return len(ch) == 1 and (ord(ch) < 0x20 or 0xE000 <= ord(ch) <= 0xF8FF)


def _display_matrix(page):
    return page.rotation_matrix if getattr(page, "rotation", 0) else None


def _point_to_display(pt, m) -> tuple[float, float]:
    if m is None:
        return (pt[0], pt[1])
    return (m.a * pt[0] + m.c * pt[1] + m.e, m.b * pt[0] + m.d * pt[1] + m.f)


def _to_display(bbox, m) -> tuple[float, float, float, float]:
    """Map an unrotated-page bbox onto the displayed page."""
    if m is None:
        return tuple(bbox)
    corners = [_point_to_display(p, m) for p in
               ((bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]))]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _raw_lines(page) -> list[tuple[tuple[float, float], list[tuple]]]:
    """Unrotated ``(dir, chars)`` per line; chars are ``(char, font, bbox, origin, size)``."""
    out = []
    for block in page.get_text("rawdict").get("blocks", []):
        for line in block.get("lines", []):
            chars = [
                (c["c"], span.get("font", ""), tuple(c["bbox"]), tuple(c.get("origin", c["bbox"][:2])),
                 float(span.get("size", 0.0)))
                for span in line.get("spans", []) for c in span.get("chars", [])
            ]
            if chars:
                out.append((tuple(line.get("dir", (1.0, 0.0))), chars))
    return out


def page_lines(page) -> list[tuple[bool, list[tuple[str, str, tuple[float, ...]]]]]:
    """Return ``(vertical, chars)`` per line; chars are ``(char, font, bbox)``.

    Bboxes are on the displayed page; ``vertical`` is the displayed writing
    direction (a sideways table on an unrotated page). Compute once per page
    and pass to :func:`native_number_tokens` when checking several blocks.
    """
    m = _display_matrix(page)
    lines = []
    for (dx, dy), chars in _raw_lines(page):
        if m is not None:
            dx, dy = m.a * dx + m.c * dy, m.b * dx + m.d * dy
        lines.append((abs(dy) > abs(dx), [(c, f, _to_display(bb, m)) for c, f, bb, _o, _s in chars]))
    return lines


_page_lines = page_lines


def _page_chars(page) -> list[list[tuple[str, str, tuple[float, ...]]]]:
    """Return lines as lists of ``(char, font, bbox)`` (displayed-page bboxes)."""
    return [chars for _vertical, chars in page_lines(page)]


def upright_rotation(page) -> int:
    """Clockwise degrees that turn the displayed page's dominant text upright.

    Line directions come in unrotated page space, so the page's own
    ``/Rotate`` (which renders already apply) is subtracted.
    """
    dirs: Counter = Counter()
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            dirs[tuple(round(x) for x in line["dir"])] += len(line["spans"])
    if not dirs:
        return 0
    top = dirs.most_common(1)[0][0]
    unrotated = {(0, -1): 90, (0, 1): 270, (-1, 0): 180}.get(top, 0)
    return (unrotated - getattr(page, "rotation", 0)) % 360


def invisible_text_fraction(page, clip, trace=None) -> float:
    """Share of text-layer characters in ``clip`` drawn invisibly (render mode 3).

    Invisible text over an image is an OCR layer, even on a page that is not a
    full-page scan: a weak witness, like a scanned page's text layer.
    ``trace`` is ``page.get_texttrace()``, when already computed.
    """
    m = _display_matrix(page)
    total = hidden = 0
    for span in (trace if trace is not None else page.get_texttrace()):
        for ch in span.get("chars", ()):
            x0, y0, x1, y1 = _to_display(ch[3], m)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if clip[0] <= cx <= clip[2] and clip[1] <= cy <= clip[3]:
                total += 1
                hidden += span.get("type") == 3
    return hidden / total if total else 0.0


def _is_digit_start(ch: str) -> bool:
    return ch in _DIGITS or ch == "."


def _count_symbol_glyphs(line, stats: dict[str, Counter], digits: dict[str, set[str]]) -> None:
    for i, (ch, font, *_rest) in enumerate(line):
        if ch in _DIGITS:
            digits.setdefault(font, set()).add(ch)
        nxt = line[i + 1] if i + 1 < len(line) else None
        if ch == ")":
            # Some symbol fonts (Advent's AdvPSSPS) draw the minus at the ')' code.
            before = nxt is not None and _is_digit_start(nxt[0]) and nxt[1] != font
            stats.setdefault(font, Counter())["paren_before_number" if before else "paren_elsewhere"] += 1
            continue
        if not (ch == "2" or (len(ch) == 1 and ord(ch) < 0x20)):
            continue
        is_candidate = (
            nxt is not None
            and _is_digit_start(nxt[0])
            and nxt[1] != font
        )
        if ch == "2":
            # Only profile '2's from fonts that differ from their
            # neighbours; body-font digits are ordinary digits.
            prev = line[i - 1] if i > 0 else None
            neighbour_fonts = {x[1] for x in (prev, nxt) if x is not None}
            if font in neighbour_fonts and not is_candidate:
                continue
        bucket = stats.setdefault(font, Counter())
        bucket["before_number" if is_candidate else "elsewhere"] += 1


def _font_profile(stats, digits) -> dict[str, dict[str, int]]:
    return {
        font: {**dict(c), "distinct_digits": len(digits.get(font, set()))}
        for font, c in stats.items()
    }


def symbol_minus_fonts(doc_or_pages: Iterable[Any]) -> dict[str, dict[str, int]]:
    """Profile fonts whose ``2``/control glyphs sit directly before numbers.

    Returns ``{font: {"before_number": n, "elsewhere": m}}`` for every font
    that draws a ``2`` or control glyph, across the given pages. A font is a
    validated minus font when ``before_number`` > 0 and ``elsewhere`` == 0.
    """
    stats: dict[str, Counter] = {}
    digits: dict[str, set[str]] = {}
    for page in doc_or_pages:
        for line in _page_chars(page):
            _count_symbol_glyphs(line, stats, digits)
    return _font_profile(stats, digits)


#: A text font draws (nearly) all ten digits somewhere in a paper. Symbol
#: fonts that repurpose digit code points for operators (``1``=+, ``2``=−,
#: ``5``== in AdvTT-style math fonts) draw only a few.
SYMBOL_FONT_MAX_DIGITS = 5


def validated_minus_fonts(profile: dict[str, dict[str, int]]) -> set[str]:
    """Fonts whose ``2`` glyph is a minus sign.

    Requires the font to precede a number at least once and to lack a full
    digit set (a symbol font), so a body-font ``2`` is never read as a minus.
    """
    return {
        font
        for font, c in profile.items()
        if c.get("before_number", 0) > 0
        and c.get("distinct_digits", 10) <= SYMBOL_FONT_MAX_DIGITS
    }


def validated_paren_minus_fonts(profile: dict[str, dict[str, int]]) -> set[str]:
    """Symbol fonts whose ``)`` glyph is a minus sign.

    The font's ``)`` must precede a number in another font and never appear
    anywhere else (a real closing parenthesis follows its content).
    """
    return {
        font
        for font, c in profile.items()
        if c.get("paren_before_number", 0) > 0
        and c.get("paren_elsewhere", 0) == 0
        and c.get("distinct_digits", 10) <= SYMBOL_FONT_MAX_DIGITS
    }


def classify_glyph(page, bbox, origin, size: float, direction=(1.0, 0.0)) -> str:
    """Classify one drawn glyph by its ink.

    Returns ``bar`` (minus, dash), ``period``, ``comma``, ``other`` (anything
    with real height: letters, ``×``, ``≈``, raised dots) or ``unknown``.
    ``bbox``, ``origin`` and ``direction`` are PyMuPDF's unrotated values for
    the character. The glyph box is rendered from the displayed page and its
    ink measured in em relative to the baseline; ink touching the box edge (a
    neighbouring glyph, a table rule) makes the answer ``unknown``.
    """
    import pymupdf

    m = _display_matrix(page)
    rect = pymupdf.Rect(_to_display(bbox, m))
    if size <= 0 or rect.width <= 0 or rect.height <= 0:
        return "unknown"
    ox, oy = _point_to_display(origin, m)
    dx, dy = direction
    if m is not None:
        dx, dy = m.a * dx + m.c * dy, m.b * dx + m.d * dy
    ux, uy = dy, -dx  # "up": perpendicular to the writing direction
    zoom = min(20.0, 48.0 / size)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect, colorspace=pymupdf.csGRAY, alpha=False)
    w, h, stride, data = pix.width, pix.height, pix.stride, pix.samples
    if w < 3 or h < 3:
        return "unknown"
    along: list[float] = []
    up: list[float] = []
    for y in range(h):
        row = data[y * stride:y * stride + w]
        for x, v in enumerate(row):
            if v >= 128:
                continue
            if x in (0, w - 1) or y in (0, h - 1):
                return "unknown"
            px, py = (pix.x + x + 0.5) / zoom, (pix.y + y + 0.5) / zoom
            along.append(((px - ox) * dx + (py - oy) * dy) / size)
            up.append(((px - ox) * ux + (py - oy) * uy) / size)
    if not along:
        return "unknown"
    width, low, high = max(along) - min(along), min(up), max(up)
    height = high - low
    if height <= 0.13 and width >= 0.25 and low >= 0.08:
        return "bar"
    if height <= 0.2 and width <= 0.22 and -0.04 <= low <= 0.08 and high <= 0.22:
        return "period"
    if low < -0.05 and high <= 0.3 and width <= 0.25:
        return "comma"
    if height >= 0.2 or (width <= 0.22 and low > 0.08):
        return "other"
    return "unknown"


def _vote(shapes: list[str]) -> str:
    known = {s for s in shapes if s != "unknown"}
    return known.pop() if len(known) == 1 else "unknown"


def glyph_profile(doc_or_pages: Iterable[Any]) -> GlyphProfile:
    """Profile a document once; pass the result to :func:`native_number_tokens`.

    One pass over the text layer counts symbol fonts and samples every
    control or private-use glyph drawn next to a digit, plus every ``2`` or
    ``)`` drawn directly before a number in another font. Sampled glyphs are
    classified by shape; a glyph keeps a class only when all its classifiable
    samples agree. A symbol font's ``2`` or ``)`` counts as a minus only when
    it is drawn as a bar: TeX extension fonts (CMEX10, MTEX) put bracket
    pieces at ``2`` and ordinary fonts draw a real ``)``.
    """
    stats: dict[str, Counter] = {}
    digits: dict[str, set[str]] = {}
    samples: dict[tuple[str, str], list] = {}
    for page in doc_or_pages:
        for direction, chars in _raw_lines(page):
            _count_symbol_glyphs(chars, stats, digits)
            seen: set[tuple[str, str]] = set()
            for i, (ch, font, bb, origin, size) in enumerate(chars):
                if (font, ch) in seen:
                    continue
                if _odd_glyph(ch):
                    near = [chars[k][0] for k in (i - 1, i + 1) if 0 <= k < len(chars)]
                    if not any(_is_digit_start(c) for c in near):
                        continue
                elif ch in "2)":
                    nxt = chars[i + 1] if i + 1 < len(chars) else None
                    if not (nxt is not None and _is_digit_start(nxt[0]) and nxt[1] != font):
                        continue
                else:
                    continue
                bucket = samples.setdefault((font, ch), [])
                if len(bucket) < SHAPE_SAMPLES:
                    bucket.append((page, bb, origin, size, direction))
                    seen.add((font, ch))
    fonts = _font_profile(stats, digits)
    symbol = validated_minus_fonts(fonts)
    paren = validated_paren_minus_fonts(fonts)
    wanted = [key for key in samples
              if _odd_glyph(key[1]) or (key[1] == "2" and key[0] in symbol) or (key[1] == ")" and key[0] in paren)]
    shapes = {key: _vote([classify_glyph(*s) for s in samples[key]]) for key in wanted}
    return GlyphProfile(
        minus_fonts={f for f in symbol if shapes.get((f, "2")) == "bar"},
        paren_minus_fonts={f for f in paren if shapes.get((f, ")")) == "bar"},
        glyph_shapes=shapes,
        symbol_fonts=symbol,
    )


def image_bbox_to_pdf(
    bbox: Iterable[float],
    image_size: tuple[float, float],
    page_size: tuple[float, float],
    rotated_cw: bool = False,
    pad: float = 2.0,
    rotation: int | None = None,
) -> tuple[float, float, float, float]:
    """Map a Surya pixel bbox back to PDF points.

    ``image_size`` is the size of the image Surya saw. When ``rotated_cw`` is
    true that image is the page render rotated 90 degrees clockwise (sideways
    tables turned upright); the bbox is un-rotated before scaling.
    ``page_size`` is the displayed page (``page.rect``: width, height) in
    points, so the result is in displayed-page coordinates.
    """
    x0, y0, x1, y1 = bbox
    img_w, img_h = image_size
    page_w, page_h = page_size
    rotation = (90 if rotated_cw else 0) if rotation is None else rotation % 360
    if rotation == 270:
        # Counter-clockwise render: original (x, y) -> (y, W - x).
        scale = img_h / page_w
        ox0, ox1 = (img_h - y1) / scale, (img_h - y0) / scale
        oy0, oy1 = x0 / scale, x1 / scale
    elif rotation == 90:
        # Clockwise rotation maps original (x, y) -> (H - y, x), where H is the
        # original render height == rotated image width.
        scale = img_h / page_w
        ox0, ox1 = y0 / scale, y1 / scale
        oy0, oy1 = (img_w - x1) / scale, (img_w - x0) / scale
    else:
        scale = img_w / page_w
        ox0, oy0, ox1, oy1 = x0 / scale, y0 / scale, x1 / scale, y1 / scale
    return (
        max(0.0, ox0 - pad),
        max(0.0, oy0 - pad),
        min(page_w, ox1 + pad),
        min(page_h, oy1 + pad),
    )


def _gap(a, b) -> float:
    """Distance between two bboxes along whichever axis separates them."""
    return max(0.0, b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])


def _near(a, b, factor: float) -> bool:
    """Gap of at most ``factor`` text heights (taken from ``b``)."""
    height = max(b[2] - b[0], b[3] - b[1])
    return _gap(a, b) <= factor * height


def _zero_advance(bb) -> bool:
    return min(bb[2] - bb[0], bb[3] - bb[1]) < 0.05


def _operand_before(line, i: int) -> bool:
    """A nearby digit, bracket or % before a spaced dash makes it binary (``1990 - 2000``).

    A distant one is the previous table cell: ``0.74      - 1.78`` is signed.
    """
    k = i - 1
    while k >= 0 and line[k][0] == " ":
        k -= 1
    return k >= 0 and (line[k][0] in _DIGITS or line[k][0] in ".)]%") and _near(line[k][2], line[i][2], 1.0)


def is_prose(chars) -> bool:
    """A line of running text: a note, title or long label, never a table row.

    Six or more words and no gap wider than ``PROSE_MAX_GAP`` text heights
    between neighbouring characters; a row of cells always has a column gap.
    ``chars`` is one line from :func:`page_lines`.
    """
    ink = [c for c in chars if c[0].strip()]
    if len(ink) < 2 * PROSE_MIN_WORDS:
        return False
    words = [w for w in "".join(c[0] for c in chars).split() if sum(ch.isalpha() for ch in w) >= 2]
    if len(words) < PROSE_MIN_WORDS:
        return False
    height = statistics.median(max(c[2][2] - c[2][0], c[2][3] - c[2][1]) for c in ink)
    return all(_gap(a[2], b[2]) <= PROSE_MAX_GAP * height for a, b in zip(ink, ink[1:]))


def _after_word(line, start: int) -> bool:
    """The number follows a word at word spacing: ``Table 6``, ``September 1988``,
    ``n = 495``, ``q < 1``, a subscript ``LMaint 89``. That is text, not a table
    cell; a cell sits a column gap away from its row label.
    """
    k, cur = start - 1, line[start][2]
    if k >= 0 and line[k][0] == " ":
        k -= 1
    if k >= 0 and line[k][0] in "=<>≤≥":
        if not _near(line[k][2], cur, 0.5):
            return False
        k, cur = k - 1, line[k][2]
        if k >= 0 and line[k][0] == " ":
            k -= 1
    return k >= 0 and line[k][0].isalpha() and _near(line[k][2], cur, 0.5)


def _range_end(line, start: int, body: str) -> bool:
    """An integer dashed to a preceding integer with no space: the end of a
    period or range label (``1970-1980``, ``1988-II`` read by OCR as ``1988-11``).
    Decimal ranges such as a ``0.12–0.35`` interval stay data.
    """
    if not body.isdigit() or start < 2 or line[start - 1][0] not in KNOWN_MINUS:
        return False
    k = start - 2
    while k >= 0 and line[k][0] in _DIGITS:
        k -= 1
    return k < start - 2 and (k < 0 or line[k][0] not in ".,")


def _exponent(line, start: int) -> bool:
    """The number is the exponent of ``1.160E+13`` or ``9.38E-06`` (a ``+`` may be
    encoded as ``þ``): part of a data value, whatever letter precedes it."""
    k = start - 1
    while k >= 0 and line[k][0] == " ":
        k -= 1
    if k >= 0 and line[k][0] in "+þ":
        k -= 1
        while k >= 0 and line[k][0] == " ":
            k -= 1
    if not (k >= 1 and line[k][0] in "Ee"):
        return False
    k -= 1
    while k >= 0 and line[k][0] == " ":
        k -= 1
    return k >= 0 and line[k][0] in _DIGITS


def _group_sep(line, j: int, out: list[str], digit) -> bool:
    """A comma at ``line[j]`` separates digit groups: three digits follow, no more,
    and no decimal point came before (``0.33,0.67`` is two numbers)."""
    if "." in out:
        return False
    ahead = line[j + 1:j + 5]
    return (len(ahead) >= 3 and all(digit(c) for c in ahead[:3])
            and not (len(ahead) == 4 and digit(ahead[3])))


def _mark_colnums(found: list[tuple[NativeToken, bool, bool]]) -> None:
    """Tag ``(1) (2) (3)`` rows: parenthesised integers sharing a row, >= 3 in sequence."""
    for vertical in (False, True):
        cands = sorted(
            (((t.bbox[0] + t.bbox[2]) / 2 if vertical else (t.bbox[1] + t.bbox[3]) / 2, t)
             for t, v, paren in found if paren and v == vertical and t.role != "note"),
            key=lambda pt: pt[0],
        )
        groups: list[list[NativeToken]] = []
        anchor = None
        for pos, t in cands:
            if anchor is None or pos - anchor > 2.5:
                groups.append([])
                anchor = pos
            groups[-1].append(t)
        for group in groups:
            values = sorted({int(t.text) for t in group})
            run: list[int] = []
            keep: set[int] = set()
            for v in values:
                run = run + [v] if run and v == run[-1] + 1 else [v]
                if len(run) >= COLNUM_MIN_RUN:
                    keep.update(run)
            for t in group:
                if int(t.text) in keep:
                    t.role = "colnum"


def native_number_tokens(
    page,
    page_number: int,
    clip: tuple[float, float, float, float] | None = None,
    minus_fonts: set[str] | None = None,
    profile: GlyphProfile | None = None,
    lines=None,
) -> list[NativeToken]:
    """Extract numeric tokens with font-aware sign detection.

    ``profile`` (from :func:`glyph_profile`) carries the document's validated
    symbol fonts and glyph shapes; ``minus_fonts`` alone is accepted for
    older callers (control glyphs then count as inferred signs, as before
    shapes existed). ``lines`` is :func:`page_lines` output, to avoid
    re-extracting a page per block. ``clip`` and token bboxes are in
    displayed-page coordinates.
    """
    if profile is None:
        profile = GlyphProfile(minus_fonts=set(minus_fonts or ()))
    symbol = profile.symbol_fonts
    shapes = profile.glyph_shapes
    found: list[tuple[NativeToken, bool, bool]] = []  # token, vertical line, (n) candidate

    def inside(bb) -> bool:
        if clip is None:
            return True
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        return clip[0] <= cx <= clip[2] and clip[1] <= cy <= clip[3]

    def digit(c) -> bool:
        return c[0] in _DIGITS and c[1] not in symbol

    def sep(c) -> str | None:
        """``.`` or ``,`` when ``c`` draws a decimal point or digit-group separator."""
        if c[0] in ".,":
            return c[0]
        if _odd_glyph(c[0]):
            shape = shapes.get((c[1], c[0]))
            return "." if shape == "period" else "," if shape == "comma" else None
        return None

    def digit_start(c) -> bool:
        return digit(c) or sep(c) == "."

    for vertical, line in (lines if lines is not None else page_lines(page)):
        i, n = 0, len(line)
        last_end: int | None = None
        prose: bool | None = None  # computed on the line's first token in the clip
        while i < n:
            ch, font, bb = line[i]
            sign_source = "none"
            start = i
            nxt = line[i + 1] if i + 1 < n else None
            prev = line[i - 1] if i > 0 else None
            # A dash between digits is a range ("1993–98"), not a sign.
            after_digit = prev is not None and _is_digit_start(prev[0])
            if nxt is not None and digit_start(nxt) and not after_digit and not digit_start(line[i]):
                if ch in KNOWN_MINUS:
                    sign_source = "codepoint"
                elif _odd_glyph(ch) and nxt[1] != font:
                    shape = shapes.get((font, ch), "unknown")
                    # An unclassified control glyph stays an inferred sign, as
                    # before shapes; a private-use glyph must look like a bar.
                    if shape == "bar" or (shape == "unknown" and ord(ch) < 0x20):
                        sign_source = "control_glyph"
                elif ch == "2" and font in profile.minus_fonts and nxt[1] != font:
                    sign_source = "font_glyph"
                elif ch == ")" and font in profile.paren_minus_fonts and nxt[1] != font:
                    sign_source = "font_glyph"
            elif (ch in KNOWN_MINUS and nxt is not None and nxt[0] == " " and i + 2 < n
                  and digit_start(line[i + 2]) and _near(bb, line[i + 2][2], 0.5)
                  and not _operand_before(line, i)):
                sign_source = "codepoint_spaced"
            body_at = i + {"none": 0, "codepoint_spaced": 2}.get(sign_source, 1)
            if body_at < n and digit_start(line[body_at]):
                idx: list[int] = []
                out: list[str] = []
                j = body_at
                while j < n:
                    c = line[j]
                    if digit(c):
                        idx.append(j)
                        out.append(c[0])
                        j += 1
                        continue
                    s = sep(c)
                    if s is None or (s == "," and not _group_sep(line, j, out, digit)):
                        break
                    if j + 1 < n and digit(line[j + 1]):
                        idx.append(j)
                        out.append(s)
                        j += 1
                        continue
                    # A zero-width '.' then a synthetic space: the text
                    # layer's "37. 2" is a printed 37.2.
                    if (s == "." and _zero_advance(c[2]) and j + 2 < n and line[j + 1][0] == " "
                            and digit(line[j + 2]) and _near(c[2], line[j + 2][2], 0.5)):
                        idx.append(j)
                        out.append(s)
                        j += 2
                        continue
                    break
                body = "".join(out)
                if any(c in _DIGITS for c in body):
                    span = line[start:idx[-1] + 1]
                    x0 = min(b[0] for _c, _f, b in span)
                    y0 = min(b[1] for _c, _f, b in span)
                    x1 = max(b[2] for _c, _f, b in span)
                    y1 = max(b[3] for _c, _f, b in span)
                    if inside((x0, y0, x1, y1)):
                        if prose is None:
                            prose = is_prose(line)
                        negative = sign_source != "none"
                        before = line[start - 1][0] if start > 0 else ""
                        after = line[idx[-1] + 1][0] if idx[-1] + 1 < n else ""
                        suffix = "".join(c[0] for c in line[idx[-1] + 1:idx[-1] + 3]).lower()
                        glued = before.isalpha() or after.isalpha()
                        k = start - 1
                        while k >= 0 and start - k <= 2 and line[k][0] == " ":
                            k -= 1
                        dash_before = not negative and 0 <= k < start - 1 and line[k][0] in KNOWN_MINUS
                        paren = (before == "(" and after == ")" and not negative
                                 and body.isdigit() and 0 < int(body) <= COLNUM_MAX)
                        k = start
                        while k > 0 and start - k < 2 and line[k - 1][0] in _PRE_MARKS:
                            k -= 1
                        m = idx[-1] + 1
                        while m < n and m - idx[-1] <= 5 and line[m][0] in _POST_MARKS:
                            m += 1
                        post = "".join(c[0] for c in line[idx[-1] + 1:m]).replace("\u2217", "*")
                        stars = "".join(ch for ch in post if ch == "*")
                        if not stars and m == idx[-1] + 1:
                            q = m
                            while q < n and q - m < 2 and line[q][0] == " ":
                                q += 1
                            r = q
                            while q > m and r < n and r - q < 4 and line[r][0] in "*\u2217":
                                r += 1
                            if r > q and (r == n or not line[r][0].isdigit()):
                                stars = "*" * (r - q)
                        found.append((NativeToken(
                            text=("-" if negative else "") + body,
                            negative=negative,
                            sign_source=sign_source,
                            bbox=(x0, y0, x1, y1),
                            page=page_number,
                            fonts=tuple(sorted({f for _c, f, _b in span})),
                            role="note" if prose else "data" if _exponent(line, start) else
                            "label" if (before.isalpha() or suffix in _ORDINALS
                                        or _after_word(line, start)
                                        or _range_end(line, start, body)) else "data",
                            sep_before=None if last_end is None
                            else "".join(c[0] for c in line[last_end + 1:start]),
                            glued=glued,
                            dash_before=dash_before,
                            vertical=bool(vertical),
                            pre="".join(c[0] for c in line[k:start]),
                            post=post,
                            stars=stars,
                        ), vertical, paren))
                        last_end = idx[-1]
                    else:
                        last_end = None
                    i = j
                    continue
                i = max(body_at, start + 1)
                continue
            i = start + 1
    _mark_colnums(found)
    tokens = [t for t, _v, _p in found]
    _attach_se_stars(tokens)
    return tokens


def _attach_se_stars(tokens: list[NativeToken]) -> None:
    """Note on an estimate the stars printed after its standard error.

    Some layouts print ``0.0252`` over ``(0.0141)**``. Readers may put those
    stars on either line, so the standard error keeps them and the estimate
    gets a copy in ``se_stars``, only when it has no stars of its own and
    the bracketed number sits directly below it, overlapping it across.
    """
    def bracketed(t: NativeToken) -> bool:
        return t.pre[-1:] in ("(", "[") and t.post[:1] in (")", "]")

    ses = [t for t in tokens if t.stars and bracketed(t) and not t.vertical]
    for se in ses:
        height = se.bbox[3] - se.bbox[1]
        above = [t for t in tokens
                 if t is not se and not t.vertical and not t.stars and not bracketed(t)
                 and t.bbox[0] < se.bbox[2] and se.bbox[0] < t.bbox[2]
                 and 0 <= se.bbox[1] - t.bbox[1] <= 2.2 * height]
        if len(above) == 1:
            above[0].se_stars = se.stars
