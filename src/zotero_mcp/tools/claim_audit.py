"""MCP adapter for bounded, claim-level Zotero RAG auditing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from zotero_mcp import client as _client
from zotero_mcp import mineru as _mineru
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.claim_audit import (
    MAX_CLAIM_CHARS,
    MAX_EVIDENCE_WINDOW_CHARS,
    MAX_QUOTE_CHARS,
    AuditDependencies,
    AuditService,
    ClaimInput,
    MineruSidecarEvidenceRef,
    parse_claims,
)

_logger = logging.getLogger("zotero_mcp.claim_audit")


def _config_path() -> str:
    return os.environ.get(
        "ZOTERO_MCP_CONFIG",
        str(Path.home() / ".config" / "zotero-mcp" / "config.json"),
    )


def _resolve_item_metadata(item_key: str) -> dict[str, Any] | None:
    """Resolve one exact item for provenance and fail closed on key drift."""

    item = _client.get_zotero_client().item(item_key)
    if not isinstance(item, dict) or not item:
        return None
    returned_key = str(
        item.get("key") or item.get("data", {}).get("key") or ""
    ).strip().upper()
    if returned_key and returned_key != item_key.upper():
        return {"key": returned_key, "data": {"key": returned_key}}
    return item


def _attachment_belongs_to_item(item_key: str, attachment_key: str) -> bool:
    """Prove that a requested attachment is a child of the requested parent."""

    if not attachment_key:
        return True
    item_key = item_key.upper()
    attachment_key = attachment_key.upper()

    # Local SQLite has the parent relationship even when the local Zotero API
    # does not expose child metadata.
    if _utils.is_local_mode():
        try:
            from zotero_mcp.config import load_config
            from zotero_mcp.local_db import LocalZoteroReader

            with LocalZoteroReader(db_path=load_config().resolve_zotero_db_path()) as reader:
                attachment = reader.get_attachment_by_key(attachment_key)
            return bool(
                attachment
                and str(attachment.get("parent_key") or "").upper() == item_key
            )
        except Exception:
            pass

    try:
        zot = _client.get_zotero_client()
        attachment = zot.item(attachment_key)
        data = attachment.get("data", {}) if isinstance(attachment, dict) else {}
        parent = str(data.get("parentItem") or "").strip().upper()
        if parent:
            return parent == item_key
        children = zot.children(item_key)
        return any(
            str(child.get("key") or child.get("data", {}).get("key") or "")
            .strip()
            .upper()
            == attachment_key
            and child.get("data", {}).get("itemType") == "attachment"
            for child in (children or [])
            if isinstance(child, dict)
        )
    except Exception:
        return False


def _read_pdf_window(
    item_key: str,
    start_page: int,
    end_page: int | None,
    attachment_key: str | None,
    ctx: Context,
) -> dict[str, Any]:
    """Read only the requested PDF pages through the existing local/download route."""

    if attachment_key and not _attachment_belongs_to_item(item_key, attachment_key):
        return {"error_code": "ATTACHMENT_MISMATCH"}

    from zotero_mcp.extract import extract_pdf, pdf_page_count
    from zotero_mcp.tools.read_pdf import _cleanup_path, _get_pdf_path

    target_key = attachment_key or item_key
    resolved = _get_pdf_path(target_key, ctx)
    if resolved is None:
        return {"error_code": "PDF_NOT_FOUND"}
    pdf_path, title, is_temp = resolved
    actual_end = end_page or start_page
    try:
        total_pages = pdf_page_count(pdf_path)
        if start_page < 1 or actual_end > total_pages:
            return {"error_code": "PDF_PAGE_OUT_OF_RANGE"}
        doc = extract_pdf(
            pdf_path,
            pages=list(range(start_page - 1, actual_end)),
        )
        pages: list[str] = []
        for page_number, text in zip(doc.page_numbers, doc.pages):
            if text and text.strip():
                pages.append(text.strip())
            else:
                pages.append("")
        text = "\n\n".join(page for page in pages if page)
        return {
            "text": text,
            "locator": f"pages {start_page}-{actual_end}",
            "title": title,
            "needs_ocr": bool(doc.needs_ocr),
            "source": "pdf",
        }
    except Exception as exc:
        _logger.debug("claim audit PDF read failed for %s: %s", item_key, exc)
        return {"error_code": "PDF_READ_FAILED"}
    finally:
        if is_temp:
            _cleanup_path(pdf_path)


def _line_window(text: str, start_line: int, end_line: int | None) -> tuple[str, str]:
    lines = text.splitlines()
    if start_line > len(lines):
        return "", f"lines {start_line}-{end_line or start_line}"
    actual_end = min(end_line or start_line, start_line + 399, len(lines))
    return "\n".join(lines[start_line - 1 : actual_end]), f"lines {start_line}-{actual_end}"


def _char_window(text: str, start: int, end: int | None) -> tuple[str, str]:
    if start < 0 or start >= len(text):
        return "", f"chars {start}-{end or start}"
    actual_end = min(end or start + MAX_EVIDENCE_WINDOW_CHARS, start + MAX_EVIDENCE_WINDOW_CHARS, len(text))
    return text[start:actual_end], f"chars {start}-{actual_end}"


def _read_sidecar_window(item_key: str, ref: MineruSidecarEvidenceRef) -> dict[str, Any]:
    """Resolve a bounded sidecar window without exposing a filesystem path."""

    cfg = _mineru.load_mineru_config(_config_path())
    full_text = _mineru.read_sidecar(cfg, item_key)
    if full_text is None:
        return {"error_code": "SIDECAR_NOT_FOUND"}

    if ref.start_line is not None:
        text, locator = _line_window(full_text, ref.start_line, ref.end_line)
    elif ref.locator:
        line_match = re.fullmatch(
            r"\s*lines?\s+(\d+)(?:\s*[-:]\s*(\d+))?\s*",
            ref.locator,
            flags=re.IGNORECASE,
        )
        char_match = re.fullmatch(
            r"\s*chars?\s+(\d+)(?:\s*[-:]\s*(\d+))?\s*",
            ref.locator,
            flags=re.IGNORECASE,
        )
        if line_match:
            text, locator = _line_window(
                full_text,
                int(line_match.group(1)),
                int(line_match.group(2)) if line_match.group(2) else None,
            )
        elif char_match:
            text, locator = _char_window(
                full_text,
                int(char_match.group(1)),
                int(char_match.group(2)) if char_match.group(2) else None,
            )
        else:
            position = full_text.casefold().find(ref.locator.casefold())
            if position < 0:
                return {"error_code": "SIDECAR_LOCATOR_NOT_FOUND"}
            start = max(0, position - 800)
            text, locator = _char_window(
                full_text,
                start,
                start + MAX_EVIDENCE_WINDOW_CHARS,
            )
    else:  # guarded by MineruSidecarEvidenceRef, retained for defensive callers
        return {"error_code": "SIDECAR_LOCATOR_REQUIRED"}

    if len(text) > MAX_EVIDENCE_WINDOW_CHARS:
        text = text[:MAX_EVIDENCE_WINDOW_CHARS]
    return {
        "text": text,
        "locator": locator,
        "source": "mineru-sidecar",
        # The full-sidecar hash is useful for detecting a changed sidecar while
        # the returned text remains bounded.
        "content_hash": hashlib.sha256(full_text.encode("utf-8", errors="replace")).hexdigest(),
    }


def _search_sidecar(item_key: str, query: str) -> dict[str, Any] | None:
    """Find one bounded sidecar window for exact-item escalation."""

    cfg = _mineru.load_mineru_config(_config_path())
    full_text = _mineru.read_sidecar(cfg, item_key)
    if not full_text:
        return None
    words = [word for word in re.findall(r"[A-Za-z0-9]{4,}", query or "")]
    positions = [full_text.casefold().find(word.casefold()) for word in words]
    positions = [position for position in positions if position >= 0]
    position = min(positions) if positions else 0
    start = max(0, position - 600)
    text = full_text[start : start + MAX_EVIDENCE_WINDOW_CHARS]
    if not text:
        return None
    return {
        "text": text,
        "quote": text[:MAX_QUOTE_CHARS],
        "locator": f"chars {start}-{start + len(text)}",
        "content_hash": hashlib.sha256(full_text.encode("utf-8", errors="replace")).hexdigest(),
    }


def _retrieve_exact(query: str, item_key: str) -> list[dict[str, Any]]:
    """Run one fresh raw semantic retrieval restricted to one exact parent key."""

    from zotero_mcp.semantic_search import create_semantic_search

    search = create_semantic_search(_config_path())
    return search.search_evidence_hits(
        query=query,
        item_key=item_key,
        limit=12,
        group_id=_client.get_active_group_id(),
    )


def _build_dependencies(ctx: Context) -> AuditDependencies:
    # Context is deliberately not passed into the pure service; the callback
    # interface keeps unit tests independent of FastMCP and prevents accidental
    # calls to a decorated public tool from inside the audit.
    return AuditDependencies(
        retriever=_retrieve_exact,
        page_reader=lambda item_key, start, end, attachment: _read_pdf_window(
            item_key, start, end, attachment, ctx
        ),
        sidecar_reader=_read_sidecar_window,
        sidecar_search=_search_sidecar,
        metadata_resolver=_resolve_item_metadata,
    )


@mcp.tool(
    name="audit_claims",
    description=(
        "Audit up to 8 atomic draft claims against caller-identified Zotero evidence. "
        "Evidence routes are semantic, pdf_page, or mineru_sidecar and require an exact "
        "8-character item_key; titles, DOIs, collections, filesystem paths, source_text, "
        "and caller-supplied rerank scores are rejected. Semantic evidence is re-retrieved "
        "for that exact item and requires fresh raw Rerank > 0 plus quote containment. "
        "Numeric claims require direct PDF-page confirmation, or a clearly weaker MinerU "
        "sidecar fallback after a failed page route. The audit performs deterministic "
        "evidence validation only; `supported` means the evidence contract passed and "
        "does not replace agent review of claim wording. escalation='bounded' permits "
        "at most three exact-item follow-ups. Returns compact JSON with evidence-gate "
        "statuses and does not synthesize answers or adjudicate zotero-extract packets."
    ),
)
def audit_claims(
    claims: list[ClaimInput] | str,
    escalation: str = "none",
    *,
    ctx: Context,
) -> str:
    """Audit claims while keeping all source access bounded and provenance-preserving."""

    try:
        if escalation not in {"none", "bounded"}:
            raise ValueError("escalation must be 'none' or 'bounded'")
        # Parse here so direct function calls and clients that send a JSON
        # string receive the same strict validation as the MCP schema.
        parsed = parse_claims(claims)
        response = AuditService(_build_dependencies(ctx)).audit(
            parsed,
            escalation=escalation,  # type: ignore[arg-type]
        )
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        # Validation failures are data, not process-level failures.
        # Keep this error bounded and do not echo arbitrary source payloads.
        if isinstance(exc, ValidationError):
            message = "claim input failed schema validation"
            log_message = message
        else:
            message = str(exc)[:MAX_CLAIM_CHARS]
            log_message = message
        try:
            ctx.error(f"Claim audit rejected request: {log_message}")
        except Exception:
            pass
        return json.dumps(
            {
                "schema_version": 1,
                "error": {
                    "code": "INVALID_INPUT",
                    "message": message,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )


__all__ = ["audit_claims"]
