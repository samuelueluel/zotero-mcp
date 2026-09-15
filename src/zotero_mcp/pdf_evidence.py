"""Bounded, source-preserving PDF evidence primitives.

This module deliberately has no Zotero or MCP dependencies.  It owns the
small pieces of work needed by the PDF evidence tools:

* literal, case-insensitive matching against already extracted page text;
* page-aware, bounded excerpts whose offsets still refer to the raw text;
* honest classification of text-layer coverage; and
* normalized page geometry plus in-memory PNG rendering.

PDF text is never normalized for output.  Whitespace tolerance is used only
by the search pattern, so excerpts remain byte-for-byte equivalent to the
text returned by :func:`zotero_mcp.extract.extract_pdf` after decoding.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

from zotero_mcp.utils import install_hint

if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from zotero_mcp.extract import ExtractedDoc

# Search bounds are intentionally aligned with the existing bounded source
# lookup tool where the units overlap.  A PDF call may inspect at most this
# many pages; it must reject a larger requested range rather than silently
# searching only a prefix.
MAX_PDF_SEARCH_PAGES = 50
DEFAULT_PDF_SEARCH_MAX_MATCHES = 5
MIN_PDF_SEARCH_MAX_MATCHES = 1
MAX_PDF_SEARCH_MAX_MATCHES = 10
MIN_PDF_SEARCH_MAX_CHARS = 256
DEFAULT_PDF_SEARCH_MAX_CHARS = 8000
MAX_PDF_SEARCH_MAX_CHARS = 16000
DEFAULT_PDF_MATCH_CONTEXT_CHARS = 600
MAX_PDF_MATCH_CONTEXT_CHARS = 4000
MAX_PDF_QUERY_CHARS = 500

# Rendering bounds.  DPI is a requested quality setting, not a knob that the
# implementation may lower to make a request fit.  The pixel and encoded-byte
# checks therefore happen before/after rendering and raise on overflow.
DEFAULT_PDF_RENDER_DPI = 144
MIN_PDF_RENDER_DPI = 36
MAX_PDF_RENDER_DPI = 300
MAX_PDF_RENDER_PIXELS = 16_000_000
MAX_PDF_RENDER_PNG_BYTES = 8_000_000
MAX_PDF_REGION_PADDING = 0.05

# Short aliases make the limits convenient for callers and tests without
# making the names used in the public tools part of the module's API contract.
MAX_SEARCH_PAGES = MAX_PDF_SEARCH_PAGES
MAX_MATCHES = MAX_PDF_SEARCH_MAX_MATCHES
MAX_OUTPUT_CHARS = MAX_PDF_SEARCH_MAX_CHARS
MAX_RENDER_PIXELS = MAX_PDF_RENDER_PIXELS
MAX_PNG_BYTES = MAX_PDF_RENDER_PNG_BYTES

_COVERAGE_STATES = frozenset(
    {"complete", "partial_text_coverage", "no_usable_text"}
)


class PdfEvidenceLimitError(ValueError):
    """Raised when a PDF evidence request exceeds an explicit hard limit."""


class PdfEvidenceInputError(ValueError):
    """Raised when a PDF evidence locator or option is invalid."""


def _is_number(value: Any) -> bool:
    """Return whether *value* is a finite real number, excluding booleans."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_bounded_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PdfEvidenceInputError(f"{name} must be an integer between {minimum} and {maximum}.")
    if not minimum <= value <= maximum:
        raise PdfEvidenceLimitError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _validate_page_number(page: Any, *, name: str = "page") -> int:
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise PdfEvidenceInputError(f"{name} must be a positive 1-indexed PDF page number.")
    return page


def _literal_whitespace_pattern(query: str) -> str:
    """Build an escaped regex that treats each query whitespace run as ``\\s+``."""
    # Stripping only the query used for matching prevents an accidental
    # leading/trailing space from becoming a surprising required source
    # character.  No source text is changed.
    query = query.strip()
    parts = re.split(r"(\s+)", query)
    return "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts if part)


def compile_literal_pattern(query: str) -> re.Pattern[str]:
    """Compile a Unicode case-insensitive, whitespace-tolerant literal query.

    Special characters are escaped before compilation.  This is deliberately
    not a fuzzy matcher: non-whitespace characters must occur in order and
    adjacent words still require at least one source whitespace character.
    """
    if not isinstance(query, str) or not query.strip():
        raise PdfEvidenceInputError("query must contain non-whitespace text.")
    if len(query) > MAX_PDF_QUERY_CHARS:
        raise PdfEvidenceLimitError(
            f"query must be at most {MAX_PDF_QUERY_CHARS} characters."
        )
    return re.compile(_literal_whitespace_pattern(query), flags=re.IGNORECASE)


def _page_inputs(
    pages_or_doc: Sequence[str] | ExtractedDoc,
    page_numbers: Sequence[int] | None,
    needs_ocr: Sequence[int] | None,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """Unpack either an ExtractedDoc or explicit page sequences."""
    # Import lazily so the module remains a small independent seam and does
    # not make extraction's optional parser import happen during registration.
    from zotero_mcp.extract import ExtractedDoc

    if isinstance(pages_or_doc, ExtractedDoc):
        pages = tuple(pages_or_doc.pages)
        source_numbers = tuple(pages_or_doc.page_numbers)
        source_needs_ocr = tuple(pages_or_doc.needs_ocr)
        if page_numbers is not None:
            source_numbers = tuple(page_numbers)
        if needs_ocr is not None:
            source_needs_ocr = tuple(needs_ocr)
    else:
        pages = tuple(pages_or_doc)
        source_numbers = tuple(range(len(pages))) if page_numbers is None else tuple(page_numbers)
        source_needs_ocr = () if needs_ocr is None else tuple(needs_ocr)

    # Synthetic callers sometimes construct an ExtractedDoc without the PDF
    # page-number tuple.  Preserve the ordinary zero-based order in that case;
    # a real extract_pdf result always supplies absolute page numbers.
    if not source_numbers and pages:
        source_numbers = tuple(range(len(pages)))

    if len(pages) != len(source_numbers):
        raise PdfEvidenceInputError("pages and page_numbers must have the same length.")
    if any(isinstance(number, bool) or not isinstance(number, int) or number < 0 for number in source_numbers):
        raise PdfEvidenceInputError("page_numbers must be nonnegative 0-indexed PDF page numbers.")
    if any(isinstance(page, bool) or not isinstance(page, int) or page < 0 for page in source_needs_ocr):
        raise PdfEvidenceInputError("needs_ocr must contain nonnegative 0-indexed PDF page numbers.")
    return pages, source_numbers, source_needs_ocr


def classify_text_coverage(
    pages_or_doc: Sequence[str] | ExtractedDoc,
    *,
    page_numbers: Sequence[int] | None = None,
    needs_ocr: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Classify usable text coverage for the supplied PDF pages.

    ``needs_ocr`` is authoritative when present: a page flagged there is not
    counted as usable even if a parser emitted incidental text.  An unflagged
    empty/whitespace-only page is also not usable.  Page numbers in the
    returned lists are human-facing one-based PDF page numbers; they are not
    printed labels, sidecar lines, or indexed offsets.
    """
    pages, source_numbers, source_needs_ocr = _page_inputs(
        pages_or_doc, page_numbers, needs_ocr
    )
    ocr_set = set(source_needs_ocr)
    usable: list[int] = []
    no_text: list[int] = []
    ocr_pages: list[int] = []

    for text, source_page in zip(pages, source_numbers):
        human_page = source_page + 1
        if source_page in ocr_set:
            ocr_pages.append(human_page)
            no_text.append(human_page)
        elif isinstance(text, str) and text.strip():
            usable.append(human_page)
        else:
            no_text.append(human_page)

    if not pages or not usable:
        state = "no_usable_text"
    elif len(usable) == len(pages):
        state = "complete"
    else:
        state = "partial_text_coverage"

    return {
        "state": state,
        "requested_pages": [number + 1 for number in source_numbers],
        "requested_page_count": len(pages),
        "usable_text_pages": usable,
        "usable_text_page_count": len(usable),
        "pages_without_usable_text": no_text,
        "pages_without_usable_text_count": len(no_text),
        "pages_needing_ocr": ocr_pages,
        "pages_needing_ocr_count": len(ocr_pages),
    }


def coverage_state(
    pages_or_doc: Sequence[str] | ExtractedDoc,
    *,
    page_numbers: Sequence[int] | None = None,
    needs_ocr: Sequence[int] | None = None,
) -> str:
    """Return only the bounded coverage state for a page collection."""
    return classify_text_coverage(
        pages_or_doc, page_numbers=page_numbers, needs_ocr=needs_ocr
    )["state"]


# Backward/reading-friendly aliases for the pure coverage seam.
classify_coverage = classify_text_coverage
classify_page_coverage = classify_text_coverage


def _window_for_match(
    text: str,
    match_start: int,
    match_end: int,
    *,
    context_chars: int,
    budget: int,
) -> tuple[int, int] | None:
    """Choose a raw-text window that contains a match and fits the budget."""
    match_length = match_end - match_start
    if match_length > budget:
        # Returning a partial match would violate the source-evidence
        # contract.  The caller reports this as an output-cap omission.
        return None

    context = min(context_chars, max(0, (budget - match_length) // 2))
    start = max(0, match_start - context)
    end = min(len(text), match_end + context)

    # Use remaining budget near a document edge where the first symmetric
    # choice could leave available room unused.  Never cross the match or
    # rewrite its characters.
    if end - start < budget:
        extra_left = min(start, budget - (end - start))
        start -= extra_left
    if end - start < budget:
        extra_right = min(len(text) - end, budget - (end - start))
        end += extra_right
    return start, end


def find_literal_matches(
    pages_or_doc: Sequence[str] | ExtractedDoc,
    query: str,
    *,
    page_numbers: Sequence[int] | None = None,
    needs_ocr: Sequence[int] | None = None,
    max_matches: int = DEFAULT_PDF_SEARCH_MAX_MATCHES,
    max_chars: int = DEFAULT_PDF_SEARCH_MAX_CHARS,
    context_chars: int = DEFAULT_PDF_MATCH_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Find a literal query across page text with exact accounting.

    The function scans every supplied page and counts every non-overlapping
    match.  It materializes at most ``max_matches`` raw-text windows and keeps
    their combined text within ``max_chars``.  A character cap can therefore
    reduce the returned count without changing ``total_matches``.

    Returned page locators are one-based PDF pages.  Character offsets are
    zero-based offsets within that page's original extracted text, with an
    exclusive end.  No normalized or fuzzy text is returned.
    """
    pages, source_numbers, source_needs_ocr = _page_inputs(
        pages_or_doc, page_numbers, needs_ocr
    )
    pattern = compile_literal_pattern(query)
    max_matches = _validate_bounded_int(
        max_matches,
        name="max_matches",
        minimum=MIN_PDF_SEARCH_MAX_MATCHES,
        maximum=MAX_PDF_SEARCH_MAX_MATCHES,
    )
    max_chars = _validate_bounded_int(
        max_chars,
        name="max_chars",
        minimum=MIN_PDF_SEARCH_MAX_CHARS,
        maximum=MAX_PDF_SEARCH_MAX_CHARS,
    )
    if isinstance(context_chars, bool) or not isinstance(context_chars, int):
        raise PdfEvidenceInputError("context_chars must be an integer.")
    if not 0 <= context_chars <= MAX_PDF_MATCH_CONTEXT_CHARS:
        raise PdfEvidenceLimitError(
            f"context_chars must be between 0 and {MAX_PDF_MATCH_CONTEXT_CHARS}."
        )

    coverage = classify_text_coverage(
        pages,
        page_numbers=source_numbers,
        needs_ocr=source_needs_ocr,
    )
    matches: list[dict[str, Any]] = []
    total_matches = 0
    used_chars = 0

    # Deliberately iterate all pages even after output caps are reached.  This
    # is what makes total_matches and has_more_matches truthful.
    ocr_pages = set(source_needs_ocr)
    for text, source_page in zip(pages, source_numbers):
        # A page classified as needing OCR is not a usable text-layer source.
        # Do not search incidental parser output from it: this route never
        # performs OCR or substitutes another source for that page.
        if source_page in ocr_pages or not isinstance(text, str) or not text:
            continue
        for found in pattern.finditer(text):
            total_matches += 1
            if len(matches) >= max_matches:
                continue

            window = _window_for_match(
                text,
                found.start(),
                found.end(),
                context_chars=context_chars,
                budget=max_chars - used_chars,
            )
            if window is None:
                # The matched source itself cannot fit in the remaining
                # declared budget.  Do not truncate it or manufacture a
                # partial evidence window; later matches cannot improve the
                # fact that this output cap was reached.
                continue
            excerpt_start, excerpt_end = window
            excerpt = text[excerpt_start:excerpt_end]
            matched_text = text[found.start():found.end()]
            record = {
                "page": source_page + 1,
                "page_index": source_page,
                "text": excerpt,
                "excerpt": excerpt,
                "match_text": matched_text,
                "match_char_start": found.start(),
                "match_char_end": found.end(),
                "excerpt_char_start": excerpt_start,
                "excerpt_char_end": excerpt_end,
                # These aliases keep the locator shape familiar to callers
                # of find_in_item while the explicit match_* fields remove
                # any ambiguity about what the offset refers to.
                "char_start": found.start(),
                "char_end": found.end(),
            }
            matches.append(record)
            used_chars += len(excerpt)

    return {
        "matches": matches,
        "total_matches": total_matches,
        "returned_matches": len(matches),
        "has_more_matches": total_matches > len(matches),
        "coverage": coverage,
        "source_chars_returned": used_chars,
    }


# Search-oriented aliases.  They intentionally point at the same function so
# there is one implementation of the source-preserving matching contract.
search_pdf_pages = find_literal_matches
find_in_pdf_text = find_literal_matches


def validate_normalized_region(
    region: Sequence[Real] | None,
    *,
    name: str = "region",
) -> tuple[float, float, float, float] | None:
    """Validate and return a normalized ``[x, y, width, height]`` region.

    Coordinates use the visible page's top-left coordinate system, exactly as
    ``detect_pdf_regions`` does.  Invalid, non-finite, empty, or overflowing
    boxes are rejected; this function never clamps or repairs them.
    """
    if region is None:
        return None
    if isinstance(region, (str, bytes)) or not isinstance(region, Sequence):
        raise PdfEvidenceInputError(
            f"{name} must be null or [x, y, width, height] normalized to [0, 1]."
        )
    if len(region) != 4:
        raise PdfEvidenceInputError(
            f"{name} must contain exactly four values: [x, y, width, height]."
        )
    values = tuple(float(value) if _is_number(value) else math.nan for value in region)
    if not all(math.isfinite(value) for value in values):
        raise PdfEvidenceInputError(f"{name} values must be finite numbers.")
    x, y, width, height = values
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise PdfEvidenceInputError(
            f"{name} must be non-empty with x/y >= 0 and width/height > 0."
        )
    if x > 1 or y > 1 or x + width > 1 or y + height > 1:
        raise PdfEvidenceInputError(f"{name} must fit within the normalized page.")
    return values


def validate_region_padding(padding: Real) -> float:
    """Validate normalized page-relative crop padding."""
    if not _is_number(padding) or float(padding) < 0:
        raise PdfEvidenceInputError("padding must be a finite nonnegative number.")
    value = float(padding)
    if value > MAX_PDF_REGION_PADDING:
        raise PdfEvidenceLimitError(
            f"padding must be between 0 and {MAX_PDF_REGION_PADDING}."
        )
    return value


def validate_render_dpi(dpi: Real) -> float:
    """Validate the requested raster resolution without reducing it."""
    if not _is_number(dpi):
        raise PdfEvidenceInputError(
            f"dpi must be a finite number between {MIN_PDF_RENDER_DPI} and {MAX_PDF_RENDER_DPI}."
        )
    value = float(dpi)
    if value < MIN_PDF_RENDER_DPI or value > MAX_PDF_RENDER_DPI:
        raise PdfEvidenceLimitError(
            f"dpi must be between {MIN_PDF_RENDER_DPI} and {MAX_PDF_RENDER_DPI}."
        )
    return value


def normalized_region_to_rect(
    page_rect: Any,
    region: Sequence[Real] | None = None,
    *,
    padding: Real = 0,
) -> tuple[Any, list[float]]:
    """Map a normalized region to a visible PyMuPDF page rectangle.

    ``page_rect`` is expected to expose ``x0``, ``y0``, ``width`` and
    ``height`` (a ``fitz.Rect`` does).  Padding is normalized to the visible
    page and is intentionally clipped at the visible page edge; the returned
    normalized box records that rendered crop, so the expansion is never
    hidden from the caller.  The input region itself is validated without
    clamping.
    """
    normalized = validate_normalized_region(region)
    pad = validate_region_padding(padding)
    if normalized is None:
        if pad:
            raise PdfEvidenceInputError("padding requires an explicit region.")
        normalized = (0.0, 0.0, 1.0, 1.0)

    x, y, width, height = normalized
    rendered_x = max(0.0, x - pad)
    rendered_y = max(0.0, y - pad)
    rendered_right = min(1.0, x + width + pad)
    rendered_bottom = min(1.0, y + height + pad)
    rendered = [
        rendered_x,
        rendered_y,
        rendered_right - rendered_x,
        rendered_bottom - rendered_y,
    ]

    page_width = float(page_rect.width)
    page_height = float(page_rect.height)
    if not math.isfinite(page_width) or not math.isfinite(page_height) or page_width <= 0 or page_height <= 0:
        raise PdfEvidenceInputError("PDF page has invalid visible dimensions.")

    # Importing PyMuPDF here keeps validation and matching usable without the
    # optional pdf extra.  A caller can also pass a compatible Rect factory in
    # tests, but the production path uses the actual module's Rect.
    fitz = _pymupdf()
    rect = fitz.Rect(
        float(page_rect.x0) + rendered[0] * page_width,
        float(page_rect.y0) + rendered[1] * page_height,
        float(page_rect.x0) + (rendered[0] + rendered[2]) * page_width,
        float(page_rect.y0) + (rendered[1] + rendered[3]) * page_height,
    )
    return rect, rendered


def _pymupdf():
    """Import PyMuPDF without triggering the deprecated ``fitz`` warning."""
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        try:
            import fitz

            return fitz
        except ImportError as exc:
            raise ImportError(
                f"PDF image rendering requires PyMuPDF. {install_hint('pdf')}"
            ) from exc


@dataclass(frozen=True)
class RenderedPdfPage:
    """In-memory PNG plus the geometry needed to describe its provenance."""

    png: bytes
    page: int
    dpi: float
    width: int
    height: int
    requested_region: list[float] | None
    rendered_region: list[float]
    page_width: float
    page_height: float

    @property
    def pixels(self) -> int:
        return self.width * self.height


def render_page_to_png(
    pdf_path: str | os.PathLike[str],
    page: int,
    *,
    region: Sequence[Real] | None = None,
    padding: Real = 0,
    dpi: Real = DEFAULT_PDF_RENDER_DPI,
) -> RenderedPdfPage:
    """Render one PDF page or normalized region to PNG bytes in memory.

    The page locator is one-based.  ``region`` and the reported geometry use
    the visible, top-left page coordinate system.  Requests over the pixel or
    encoded PNG caps raise ``PdfEvidenceLimitError`` instead of downscaling or
    truncating the result.
    """
    try:
        pdf_path = os.fspath(pdf_path)
    except TypeError as exc:
        raise PdfEvidenceInputError("pdf_path must be a non-empty path.") from exc
    if not pdf_path:
        raise PdfEvidenceInputError("pdf_path must be a non-empty path.")
    page = _validate_page_number(page)
    dpi_value = validate_render_dpi(dpi)
    requested = validate_normalized_region(region)
    pad = validate_region_padding(padding)

    fitz = _pymupdf()
    document = None
    try:
        document = fitz.open(pdf_path)
        if not getattr(document, "is_pdf", True):
            raise PdfEvidenceInputError("file is not a valid PDF.")
        if page > len(document):
            raise PdfEvidenceInputError(
                f"page {page} is out of range; PDF has {len(document)} pages."
            )
        source_page = document[page - 1]
        visible = source_page.rect
        clip, rendered_region = normalized_region_to_rect(
            visible, requested, padding=pad
        )
        render_width = rendered_region[2] * float(visible.width)
        render_height = rendered_region[3] * float(visible.height)
        projected_width = max(1, math.ceil(render_width * dpi_value / 72.0))
        projected_height = max(1, math.ceil(render_height * dpi_value / 72.0))
        projected_pixels = projected_width * projected_height
        if projected_pixels > MAX_PDF_RENDER_PIXELS:
            raise PdfEvidenceLimitError(
                f"rendered image would contain {projected_pixels:,} pixels; "
                f"the limit is {MAX_PDF_RENDER_PIXELS:,}."
            )

        matrix = fitz.Matrix(dpi_value / 72.0, dpi_value / 72.0)
        pixmap = source_page.get_pixmap(
            matrix=matrix,
            clip=clip if requested is not None else None,
            alpha=False,
        )
        actual_pixels = int(pixmap.width) * int(pixmap.height)
        if actual_pixels > MAX_PDF_RENDER_PIXELS:
            raise PdfEvidenceLimitError(
                f"rendered image contains {actual_pixels:,} pixels; "
                f"the limit is {MAX_PDF_RENDER_PIXELS:,}."
            )
        png = pixmap.tobytes("png")
        if len(png) > MAX_PDF_RENDER_PNG_BYTES:
            raise PdfEvidenceLimitError(
                f"encoded PNG is {len(png):,} bytes; the limit is "
                f"{MAX_PDF_RENDER_PNG_BYTES:,}."
            )
        return RenderedPdfPage(
            png=png,
            page=page,
            dpi=dpi_value,
            width=int(pixmap.width),
            height=int(pixmap.height),
            requested_region=list(requested) if requested is not None else None,
            rendered_region=list(rendered_region),
            page_width=float(visible.width),
            page_height=float(visible.height),
        )
    finally:
        if document is not None:
            document.close()


# Common names for callers that describe the operation as rendering rather
# than converting a page to PNG.
render_pdf_page_png = render_page_to_png
render_pdf_png = render_page_to_png


__all__ = [
    "DEFAULT_PDF_MATCH_CONTEXT_CHARS",
    "DEFAULT_PDF_RENDER_DPI",
    "DEFAULT_PDF_SEARCH_MAX_CHARS",
    "DEFAULT_PDF_SEARCH_MAX_MATCHES",
    "MAX_PDF_MATCH_CONTEXT_CHARS",
    "MAX_PDF_QUERY_CHARS",
    "MAX_PDF_REGION_PADDING",
    "MAX_PDF_RENDER_DPI",
    "MAX_PDF_RENDER_PNG_BYTES",
    "MAX_PDF_RENDER_PIXELS",
    "MAX_PDF_SEARCH_MAX_CHARS",
    "MAX_PDF_SEARCH_MAX_MATCHES",
    "MAX_PDF_SEARCH_PAGES",
    "MIN_PDF_RENDER_DPI",
    "MIN_PDF_SEARCH_MAX_CHARS",
    "PdfEvidenceInputError",
    "PdfEvidenceLimitError",
    "RenderedPdfPage",
    "classify_coverage",
    "classify_page_coverage",
    "classify_text_coverage",
    "compile_literal_pattern",
    "coverage_state",
    "find_in_pdf_text",
    "find_literal_matches",
    "normalized_region_to_rect",
    "render_page_to_png",
    "render_pdf_page_png",
    "render_pdf_png",
    "search_pdf_pages",
    "validate_normalized_region",
    "validate_region_padding",
    "validate_render_dpi",
]
