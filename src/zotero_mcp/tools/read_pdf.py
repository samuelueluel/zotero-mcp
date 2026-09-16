"""Tools for bounded PDF text evidence and page-image rendering."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from typing import Any

from fastmcp import Context
from fastmcp.tools.base import ToolResult
from mcp.types import ImageContent, TextContent

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp.client import ZoteroApiBusyError, zotero_api_lock
from zotero_mcp.config import load_config
from zotero_mcp.extract import extract_pdf, pdf_page_count
from zotero_mcp.pdf_evidence import (
    DEFAULT_PDF_MATCH_CONTEXT_CHARS,
    DEFAULT_PDF_RENDER_DPI,
    DEFAULT_PDF_SEARCH_MAX_CHARS,
    DEFAULT_PDF_SEARCH_MAX_MATCHES,
    MAX_PDF_SEARCH_MATCH_OFFSET,
    MAX_PDF_SEARCH_MAX_CHARS,
    MAX_PDF_SEARCH_MAX_MATCHES,
    MAX_PDF_SEARCH_PAGES,
    MIN_PDF_SEARCH_MAX_CHARS,
    PdfEvidenceInputError,
    PdfEvidenceLimitError,
    classify_text_coverage,
    compile_literal_pattern,
    find_literal_matches,
    render_page_to_png,
    validate_normalized_region,
    validate_region_padding,
    validate_render_dpi,
)
from zotero_mcp.tools import _helpers

_TMPDIR_PREFIX = "zotero_pdf_"


def _cleanup_path(file_path: str) -> None:
    """Remove a PDF this module downloaded, along with the directory it made.

    Deletes the file's *parent directory*, so it must only ever be handed a
    path inside a directory this module created with ``mkdtemp``. Two things
    are checked before removing anything, both of which have bitten:

    - The directory's name must carry our ``zotero_pdf_`` prefix. A bare
      "is it under the temp dir" test is not enough: on Linux
      ``gettempdir()`` is ``/tmp``, so a path like ``/tmp/paper.pdf`` has
      ``/tmp`` as its parent and passes that test, and the call then wipes
      the entire system temp directory. (This is not hypothetical; a test
      stub returning ``/tmp/test.pdf`` did exactly that on CI, which
      presented as unrelated tests failing with FileNotFoundError on
      pytest's own temp root.) macOS hides the bug, because there
      ``gettempdir()`` is under ``/var/folders`` and the prefix never
      matches ``/tmp``.
    - The directory must still be a strict subdirectory of the temp root, so
      the root itself can never be the target.

    A file resolved out of the user's Zotero storage must never be passed
    here: deleting its parent takes the user's own copy of the PDF with it.
    """
    try:
        parent = os.path.dirname(os.path.abspath(file_path))
        temp_root = os.path.abspath(tempfile.gettempdir())
        if not os.path.isdir(parent):
            return
        if os.path.samefile(parent, temp_root):
            return
        if os.path.commonpath([parent, temp_root]) != temp_root:
            return
        if not os.path.basename(parent).startswith(_TMPDIR_PREFIX):
            return
        import shutil

        shutil.rmtree(parent, ignore_errors=True)
    except Exception:
        pass


def _get_pdf_path(item_key: str, ctx: Context) -> tuple[str, str, bool] | None:
    """Resolve a PDF attachment and return ``(file_path, title, is_temp)``.

    Tries local storage first (via LocalZoteroReader), then downloads via API.
    Returns None if no PDF attachment is found.

    ``is_temp`` says whether the caller owns the file. It is True only for a
    file downloaded into a directory this function created, which the caller
    must clean up. It is False for a file resolved out of the user's Zotero
    storage, which must be left alone: those paths point into the real
    library, and deleting one takes the user's copy of the PDF with it.
    """
    zot = _client.get_zotero_client()
    item = zot.item(item_key)

    # Try local storage first (persists on disk — no cleanup needed)
    try:
        from zotero_mcp.local_db import LocalZoteroReader

        if _utils.is_local_mode():
            with LocalZoteroReader(db_path=load_config().resolve_zotero_db_path()) as reader:
                # The key may name the PDF attachment itself. Attachments have
                # no children, so the parent scan below comes up empty and we
                # would wrongly report "No PDF attachment found" (#372).
                attachment = reader.get_attachment_by_key(item_key)
                if attachment and "pdf" in (attachment["content_type"] or "").lower():
                    resolved = reader._resolve_attachment_path(
                        item_key, attachment["zotero_path"] or ""
                    )
                    if not (resolved and resolved.exists()):
                        # Recorded filename drifted on disk — scan the folder (#291)
                        resolved = reader._scan_storage_for_attachment(
                            item_key, attachment["content_type"]
                        )
                    if resolved and resolved.exists():
                        return str(resolved), attachment["title"] or item_key, False

                local_item = reader.get_item_by_key(item_key)
                if local_item:
                    for att_key, path, ctype in reader._iter_parent_attachments(local_item.item_id):
                        if ctype == "application/pdf":
                            resolved = reader._resolve_attachment_path(att_key, path or "")
                            if resolved and resolved.exists():
                                return str(resolved), local_item.title or item_key, False
    except Exception:
        pass

    # Fallback: resolve via the multi-source downloader (local -> WebDAV ->
    # Zotero cloud) so WebDAV-backed attachments work, not just cloud storage.
    # PDF only: this tool renders page ranges, so a markdown-first
    # attachment_priority must not hand it a file it cannot paginate.
    attachment = _client.get_attachment_details(zot, item, priority=("pdf",))
    if not attachment:
        return None

    pdf_extensions = {".pdf", ".PDF"}
    filename = attachment.filename or f"{attachment.key}.pdf"
    if not any(filename.endswith(ext) for ext in pdf_extensions):
        content_type = attachment.content_type or ""
        if "pdf" not in content_type.lower():
            return None

    tmpdir = tempfile.mkdtemp(prefix="zotero_pdf_")
    probe = os.path.join(tmpdir, os.path.basename(filename))
    try:
        download = _client.download_attachment_file(
            attachment.key,
            tmpdir,
            os.path.basename(filename),
            local_client=_client.get_local_zotero_client(),
            web_client=None if _utils.is_local_mode() else zot,
        )
    except Exception:
        _cleanup_path(probe)
        raise

    if download.path and download.path.exists() and download.path.stat().st_size > 0:
        return str(download.path), attachment.title, True

    _cleanup_path(probe)
    return None


def _pdf_json_error(code: str, message: str) -> str:
    """Return the bounded error envelope used by the source lookup tools."""
    return json.dumps(
        {"ok": False, "error": {"code": code, "message": message}},
        ensure_ascii=False,
    )


def _validate_search_page_range(
    start_page: Any,
    end_page: Any,
    total_pages: int,
) -> tuple[int, int]:
    """Validate a one-based, contiguous PDF page range without truncation."""
    if isinstance(start_page, bool) or not isinstance(start_page, int):
        raise PdfEvidenceInputError(
            "start_page must be a positive 1-indexed PDF page number."
        )
    if start_page < 1 or start_page > total_pages:
        raise PdfEvidenceInputError(
            f"start_page {start_page} is out of range; PDF has {total_pages} pages."
        )
    if end_page is None:
        actual_end = total_pages
    else:
        if isinstance(end_page, bool) or not isinstance(end_page, int):
            raise PdfEvidenceInputError(
                "end_page must be a positive 1-indexed PDF page number."
            )
        if end_page < start_page:
            raise PdfEvidenceInputError(
                "end_page must be greater than or equal to start_page."
            )
        if end_page > total_pages:
            raise PdfEvidenceInputError(
                f"end_page {end_page} is out of range; PDF has {total_pages} pages."
            )
        actual_end = end_page

    span = actual_end - start_page + 1
    if span > MAX_PDF_SEARCH_PAGES:
        raise PdfEvidenceLimitError(
            f"requested page range contains {span} pages; the limit is "
            f"{MAX_PDF_SEARCH_PAGES}. Narrow the page range."
        )
    return start_page, actual_end


def _pdf_source_route(is_temp: bool) -> str:
    """Describe how the resolved PDF reached the read-only processing path."""
    return "downloaded_pdf" if is_temp else "local_storage_pdf"


def _parse_region_argument(region: list[float] | str | None) -> list[float] | None:
    """Accept arrays and JSON-stringified arrays from MCP clients."""
    if isinstance(region, str):
        try:
            region = json.loads(region)
        except json.JSONDecodeError as exc:
            raise PdfEvidenceInputError(
                "region must be a JSON array [x, y, width, height]."
            ) from exc
    if region is not None and not isinstance(region, list):
        raise PdfEvidenceInputError(
            "region must be null or [x, y, width, height]."
        )
    return region


@mcp.tool(
    name="find_in_pdf",
    description=(
        "Find a literal, case-insensitive phrase in the PDF text layer for a Zotero item or PDF "
        "attachment and return bounded verbatim windows. Parent and attachment keys both work. "
        "Whitespace in the query spans source whitespace; special characters are literal — no regex, "
        "fuzzy, semantic, OCR, sidecar, or neighboring-page search. Pages are one-based PDF pages, "
        "not printed labels or indexed offsets. Searches the complete requested range and reports exact "
        "match accounting plus complete/partial/no-usable-text coverage. offset paginates matching "
        "windows; match_pages and omitted_match_pages expose later matching pages even when excerpts "
        "are capped. max_matches is 1–10; max_chars is 256–16000; a range over 50 pages is rejected."
    ),
)
def find_in_pdf(
    item_key: str,
    query: str,
    start_page: int = 1,
    end_page: int | None = None,
    max_matches: int = DEFAULT_PDF_SEARCH_MAX_MATCHES,
    offset: int = 0,
    max_chars: int = DEFAULT_PDF_SEARCH_MAX_CHARS,
    context_chars: int = DEFAULT_PDF_MATCH_CONTEXT_CHARS,
    *,
    ctx: Context,
) -> str:
    """Search a bounded PDF page range using the authoritative text extractor."""
    pdf_path: str | None = None
    is_temp = False
    try:
        if not isinstance(item_key, str) or not item_key.strip():
            return _pdf_json_error("INVALID_ARGUMENT", "item_key cannot be empty.")

        # Validate limits before touching Zotero so bad requests do not cause
        # an unnecessary API/download operation.
        if isinstance(max_matches, bool) or not isinstance(max_matches, int):
            raise PdfEvidenceInputError("max_matches must be an integer between 1 and 10.")
        if not 1 <= max_matches <= MAX_PDF_SEARCH_MAX_MATCHES:
            raise PdfEvidenceLimitError(
                f"max_matches must be between 1 and {MAX_PDF_SEARCH_MAX_MATCHES}."
            )
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise PdfEvidenceInputError(
                f"offset must be an integer between 0 and {MAX_PDF_SEARCH_MATCH_OFFSET}."
            )
        if not 0 <= offset <= MAX_PDF_SEARCH_MATCH_OFFSET:
            raise PdfEvidenceLimitError(
                f"offset must be between 0 and {MAX_PDF_SEARCH_MATCH_OFFSET}."
            )
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise PdfEvidenceInputError(
                f"max_chars must be an integer between {MIN_PDF_SEARCH_MAX_CHARS} and "
                f"{MAX_PDF_SEARCH_MAX_CHARS}."
            )
        if not MIN_PDF_SEARCH_MAX_CHARS <= max_chars <= MAX_PDF_SEARCH_MAX_CHARS:
            raise PdfEvidenceLimitError(
                f"max_chars must be between {MIN_PDF_SEARCH_MAX_CHARS} and "
                f"{MAX_PDF_SEARCH_MAX_CHARS}."
            )
        if isinstance(context_chars, bool) or not isinstance(context_chars, int):
            raise PdfEvidenceInputError("context_chars must be an integer.")

        compile_literal_pattern(query)
        ctx.info(f"Searching PDF text for item {item_key}")

        # Only source resolution/download uses the API lock. Extraction and
        # matching happen after it is released.
        with zotero_api_lock():
            result = _get_pdf_path(item_key, ctx)
        if result is None:
            return _pdf_json_error(
                "PDF_NOT_FOUND", f"No PDF attachment found for item: {item_key}"
            )
        pdf_path, title, is_temp = result
        source_is_temp = is_temp

        try:
            total_pages = pdf_page_count(pdf_path)
            start, end = _validate_search_page_range(
                start_page, end_page, total_pages
            )
            doc = extract_pdf(pdf_path, pages=list(range(start - 1, end)))
            evidence = find_literal_matches(
                doc,
                query,
                max_matches=max_matches,
                match_offset=offset,
                max_chars=max_chars,
                context_chars=context_chars,
            )
            coverage = classify_text_coverage(doc)
        finally:
            if is_temp:
                _cleanup_path(pdf_path)
                is_temp = False

        total_matches = evidence["total_matches"]
        payload = {
            "ok": True,
            "item_key": item_key,
            "title": str(title or ""),
            "query": query,
            "route": "pdf_extraction",
            "source_route": _pdf_source_route(source_is_temp),
            "extraction_route": "direct_pdf_text",
            "extraction_engine": "pdf-inspector",
            "page_basis": (
                "one-based PDF pages; distinct from printed labels, MinerU sidecar lines, "
                "and indexed offsets"
            ),
            "page_range": {"start": start, "end": end},
            "searched_page_range": [start, end],
            "total_pages": total_pages,
            "coverage": coverage["state"],
            "text_layer_coverage": coverage,
            "total_matches": total_matches,
            "offset": evidence["match_offset"],
            "next_offset": evidence["next_offset"],
            "returned_matches": evidence["returned_matches"],
            "has_more_matches": evidence["has_more_matches"],
            "match_pages": evidence["match_pages"],
            "returned_pages": evidence["returned_pages"],
            "omitted_match_pages": evidence["omitted_match_pages"],
            "matches": evidence["matches"],
            "offset_basis": "zero-based characters within each extracted PDF page; end exclusive",
            "returned_excerpt_chars": evidence["source_chars_returned"],
        }
        if total_matches == 0:
            if coverage["state"] == "complete":
                payload["message"] = (
                    "No literal matches found in the extracted text for the requested pages."
                )
            else:
                payload["message"] = (
                    "No literal matches found in usable extracted text; the requested pages have "
                    f"{coverage['state']} and absence cannot be established for pages without usable text."
                )
        return json.dumps(payload, ensure_ascii=False)

    except ZoteroApiBusyError:
        raise
    except (PdfEvidenceInputError, PdfEvidenceLimitError) as exc:
        if is_temp and pdf_path:
            _cleanup_path(pdf_path)
        return _pdf_json_error("INVALID_ARGUMENT", str(exc))
    except Exception as exc:
        if is_temp and pdf_path:
            _cleanup_path(pdf_path)
        ctx.error(f"Bounded PDF lookup failed: {exc}")
        return _pdf_json_error("SOURCE_UNAVAILABLE", f"Could not search the PDF text: {exc}")


@mcp.tool(
    name="render_pdf_page",
    description=(
        "Render exactly one PDF page or one normalized region as actual PNG image content for "
        "visual evidence inspection. Parent and PDF attachment keys both work. page is a one-based "
        "PDF page, distinct from printed labels and indexed offsets; region is [x, y, width, height] "
        "in [0, 1] using the visible page and detect_pdf_regions coordinate system. Optional padding "
        "is normalized page-relative and bounded. dpi, pixel, and PNG-size limits are explicit: "
        "oversized requests fail rather than being silently downscaled. Requires PyMuPDF."
    ),
)
def render_pdf_page(
    item_key: str,
    page: int,
    region: list[float] | str | None = None,
    padding: float = 0,
    dpi: int = DEFAULT_PDF_RENDER_DPI,
    *,
    ctx: Context,
) -> Any:
    """Render one page or normalized region and return a FastMCP image result."""
    pdf_path: str | None = None
    is_temp = False
    try:
        if not isinstance(item_key, str) or not item_key.strip():
            return "Error: item_key cannot be empty."
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            return "Error: page must be a positive 1-indexed PDF page number."
        region = _parse_region_argument(region)
        validate_normalized_region(region)
        validate_region_padding(padding)
        validate_render_dpi(dpi)
        ctx.info(f"Rendering PDF page {page} for item {item_key}")

        # Resolve/download while the API lock is held, then release it before
        # PyMuPDF opens or rasterizes the file.
        with zotero_api_lock():
            result = _get_pdf_path(item_key, ctx)
        if result is None:
            return f"Error: No PDF attachment found for item: {item_key}"
        pdf_path, title, is_temp = result
        source_is_temp = is_temp

        try:
            rendered = render_page_to_png(
                pdf_path,
                page,
                region=region,
                padding=padding,
                dpi=dpi,
            )
        finally:
            if is_temp:
                _cleanup_path(pdf_path)
                is_temp = False

        provenance = {
            "ok": True,
            "item_key": item_key,
            "title": str(title or ""),
            "route": "pdf_rendering",
            "source_route": _pdf_source_route(source_is_temp),
            "page_basis": (
                "one-based PDF pages; distinct from printed labels, MinerU sidecar lines, "
                "and indexed offsets"
            ),
            "page": rendered.page,
            "dpi": rendered.dpi,
            "requested_region": rendered.requested_region,
            "rendered_region": rendered.rendered_region,
            "padding": float(padding),
            "width": rendered.width,
            "height": rendered.height,
            "image_width": rendered.width,
            "image_height": rendered.height,
            "pixels": rendered.pixels,
            "mime_type": "image/png",
            "encoded_bytes": len(rendered.png),
        }
        text = "PDF image provenance:\n" + json.dumps(
            provenance, ensure_ascii=False, sort_keys=True
        )
        return ToolResult(
            content=[
                TextContent(type="text", text=text),
                ImageContent(
                    type="image",
                    data=base64.b64encode(rendered.png).decode("ascii"),
                    mimeType="image/png",
                ),
            ],
            structured_content=provenance,
        )
    except ZoteroApiBusyError:
        raise
    except (PdfEvidenceInputError, PdfEvidenceLimitError) as exc:
        if is_temp and pdf_path:
            _cleanup_path(pdf_path)
        return f"Error: {exc}"
    except Exception as exc:
        if is_temp and pdf_path:
            _cleanup_path(pdf_path)
        ctx.error(f"PDF image rendering failed: {exc}")
        return f"Error rendering PDF page: {exc}"


@mcp.tool(
    name="read_pdf_pages",
    description="Read specific page range(s) from a PDF attachment of a Zotero item. "
    "Use this when you know which pages to read — for example after getting the PDF "
    "outline via get_pdf_outline. Pages are 1-indexed. "
    "Returns Markdown with the page's heading structure preserved.",
)
def read_pdf_pages(
    item_key: str,
    start_page: int,
    end_page: int | None = None,
    *,
    ctx: Context,
) -> str:
    """Extract and return text from a specific page range of a PDF.

    Args:
        item_key: Zotero item key/ID of the paper or its PDF attachment.
        start_page: First page to read (1-indexed).
        end_page: Last page to read (1-indexed). If omitted, reads only start_page.
        ctx: MCP context.

    Returns:
        Markdown-formatted page content with metadata header.
    """
    try:
        if not item_key or not item_key.strip():
            return "Error: item_key cannot be empty."

        if end_page is not None and end_page < start_page:
            return "Error: end_page must be greater than or equal to start_page."

        ctx.info(f"Reading PDF pages {start_page}-{end_page or start_page} for item {item_key}")

        result = _get_pdf_path(item_key, ctx)
        if result is None:
            return f"No PDF attachment found for item: {item_key}"

        pdf_path, title, is_temp = result

        def _release() -> None:
            """Drop the working copy, but never a file in the user's library."""
            if is_temp:
                _cleanup_path(pdf_path)

        try:
            total_pages = pdf_page_count(pdf_path)
        except Exception as exc:
            _release()
            return f"Could not read PDF for item {item_key}: {exc}"

        actual_end = end_page if end_page is not None else start_page

        if start_page < 1 or start_page > total_pages:
            _release()
            return f"Start page {start_page} is out of range. PDF has {total_pages} pages (1-{total_pages})."
        if end_page is not None and end_page > total_pages:
            _release()
            return f"End page {end_page} is out of range. PDF has {total_pages} pages (1-{total_pages})."

        requested = actual_end - start_page + 1
        if requested > 50:
            _release()
            return f"Requested {requested} pages (max 50). Please narrow your page range."

        try:
            # extract_pdf takes 0-indexed pages; the tool's API is 1-indexed.
            doc = extract_pdf(pdf_path, pages=list(range(start_page - 1, actual_end)))
        except Exception as exc:
            return f"Could not read PDF for item {item_key}: {exc}"
        finally:
            _release()

        output = [
            f"# PDF Pages {start_page}-{actual_end} of {title}",
            f"**Item Key:** {item_key}",
            f"**Total pages in PDF:** {total_pages}",
            "",
        ]

        for page_index, markdown in zip(doc.page_numbers, doc.pages):
            output.append(f"## Page {page_index + 1}")
            output.append("")
            if markdown.strip():
                output.append(markdown.strip())
            elif page_index in doc.needs_ocr:
                output.append("*[No text layer on this page — it is a scanned image]*")
            else:
                output.append("*[No extractable text on this page]*")
            output.append("")
        return _helpers._prepend_size_warning(
            "\n".join(output),
            "Consider using semantic_search to find specific content instead of reading full pages.",
        )

    except Exception as e:
        ctx.error(f"Error reading PDF pages: {str(e)}")
        return f"Error reading PDF pages: {str(e)}"
