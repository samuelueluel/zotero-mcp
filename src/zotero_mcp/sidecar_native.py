"""Font-aware numeric tokens from a PDF's native text layer.

The native text layer is the independent check against the Surya VLM route,
but journal fonts often encode the minus sign badly:

- a known minus code point (U+2212, U+2AFA, hyphen, en/em dash);
- a control character (``\\x01``) from a symbol font with no ToUnicode entry;
- a plain ``2`` drawn from a separate symbol font (so ``-.0016`` extracts as
  ``2.0016``).

The last two are recognised by font, never by value: a sign candidate must
come from a font different from the digits it precedes and be directly
followed by a digit or decimal point on the same line. Every token records
how its sign was established (``sign_source``) so the verifier can refuse
deterministic substitution when the sign was only inferred.

Trust levels:

- ``codepoint`` and ``font_glyph`` (a ``2`` from a validated symbol font
  lacking a full digit set) are authoritative signs.
- ``control_glyph`` is inferred only. Symbol fonts reuse control codes for
  several operators (Diamond and McQuade's ``AdvP4C4E46`` draws ``\\x03``
  before an exponent in equations), so a control-glyph sign may confirm an
  agreeing Surya value but must never drive a substitution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

AUTHORITATIVE_SIGN_SOURCES = frozenset({"none", "codepoint", "font_glyph"})

KNOWN_MINUS = {"\u2212", "\u2afa", "-", "\u2013", "\u2014", "\u2012"}
_NUM_CHARS = set("0123456789.,")


@dataclass
class NativeToken:
    text: str  # normalized: optional leading "-", digits, "." and ","
    negative: bool
    sign_source: str  # "none" | "codepoint" | "control_glyph" | "font_glyph"
    bbox: tuple[float, float, float, float]
    page: int
    fonts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def magnitude(self) -> str:
        return self.text.lstrip("-")


def _page_chars(page) -> list[list[tuple[str, str, tuple[float, ...]]]]:
    """Return lines as lists of ``(char, font, bbox)``."""
    lines: list[list[tuple[str, str, tuple[float, ...]]]] = []
    for block in page.get_text("rawdict").get("blocks", []):
        for line in block.get("lines", []):
            chars = []
            for span in line.get("spans", []):
                font = span.get("font", "")
                for c in span.get("chars", []):
                    chars.append((c["c"], font, tuple(c["bbox"])))
            if chars:
                lines.append(chars)
    return lines


def _is_digit_start(ch: str) -> bool:
    return ch.isdigit() or ch == "."


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
            for i, (ch, font, _bb) in enumerate(line):
                if ch.isdigit():
                    digits.setdefault(font, set()).add(ch)
                if not (ch == "2" or (len(ch) == 1 and ord(ch) < 0x20)):
                    continue
                nxt = line[i + 1] if i + 1 < len(line) else None
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
    return {
        font: {**dict(c), "distinct_digits": len(digits.get(font, set()))}
        for font, c in stats.items()
    }


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


def image_bbox_to_pdf(
    bbox: Iterable[float],
    image_size: tuple[float, float],
    page_size: tuple[float, float],
    rotated_cw: bool = False,
    pad: float = 2.0,
) -> tuple[float, float, float, float]:
    """Map a Surya pixel bbox back to PDF points.

    ``image_size`` is the size of the image Surya saw. When ``rotated_cw`` is
    true that image is the page render rotated 90 degrees clockwise (sideways
    tables turned upright); the bbox is un-rotated before scaling.
    ``page_size`` is the unrotated PDF page (width, height) in points.
    """
    x0, y0, x1, y1 = bbox
    img_w, img_h = image_size
    page_w, page_h = page_size
    if rotated_cw:
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


def native_number_tokens(
    page,
    page_number: int,
    clip: tuple[float, float, float, float] | None = None,
    minus_fonts: set[str] | None = None,
) -> list[NativeToken]:
    """Extract numeric tokens with font-aware sign detection.

    ``minus_fonts`` is the document's validated symbol-minus font set (from
    :func:`validated_minus_fonts`). A ``2`` glyph is read as a minus only when
    its font is in that set; control glyphs are read as a minus when directly
    followed by a number in another font.
    """
    minus_fonts = minus_fonts or set()
    tokens: list[NativeToken] = []

    def inside(bb) -> bool:
        if clip is None:
            return True
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        return clip[0] <= cx <= clip[2] and clip[1] <= cy <= clip[3]

    for line in _page_chars(page):
        i, n = 0, len(line)
        while i < n:
            ch, font, bb = line[i]
            sign_source = "none"
            start = i
            nxt = line[i + 1] if i + 1 < n else None
            prev = line[i - 1] if i > 0 else None
            # A dash between digits is a range ("1993–98"), not a sign.
            after_digit = prev is not None and (prev[0].isdigit() or prev[0] == ".")
            if nxt is not None and _is_digit_start(nxt[0]) and not after_digit:
                if ch in KNOWN_MINUS:
                    sign_source = "codepoint"
                elif len(ch) == 1 and ord(ch) < 0x20 and nxt[1] != font:
                    sign_source = "control_glyph"
                elif ch == "2" and font in minus_fonts and nxt[1] != font:
                    sign_source = "font_glyph"
            if sign_source != "none":
                i += 1
            if i < n and _is_digit_start(line[i][0]):
                j = i
                while j < n and line[j][0] in _NUM_CHARS:
                    j += 1
                body = "".join(c for c, _f, _b in line[i:j]).rstrip(".,")
                if any(c.isdigit() for c in body):
                    span = line[start:i + len(body)]
                    x0 = min(b[0] for _c, _f, b in span)
                    y0 = min(b[1] for _c, _f, b in span)
                    x1 = max(b[2] for _c, _f, b in span)
                    y1 = max(b[3] for _c, _f, b in span)
                    if inside((x0, y0, x1, y1)):
                        negative = sign_source != "none"
                        tokens.append(NativeToken(
                            text=("-" if negative else "") + body,
                            negative=negative,
                            sign_source=sign_source,
                            bbox=(x0, y0, x1, y1),
                            page=page_number,
                            fonts=tuple(sorted({f for _c, f, _b in span})),
                        ))
                    i = j
                    continue
                i = max(i, start + 1)
                continue
            i = start + 1
    return tokens
