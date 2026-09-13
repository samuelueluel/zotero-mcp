"""Explicit, bounded source expansion without search, inference, or downloads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from zotero_mcp import client as _client
from zotero_mcp import mineru as _mineru
from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.passage_context import (
    DEFAULT_CONTEXT_CHARS,
    KEY_PATTERN,
    MAX_CONTEXT_CHARS,
    chunk_fingerprint,
    find_windows,
    line_starts,
    parse_evidence_id,
    source_window,
    text_hash,
)
from zotero_mcp.passage_context import (
    evidence_id as make_evidence_id,
)

MAX_SIDECAR_BYTES = 16 * 1024 * 1024


def _config_path() -> str:
    return os.environ.get("ZOTERO_MCP_CONFIG", str(Path.home() / ".config/zotero-mcp/config.json"))


def _error(code: str, message: str) -> str:
    return json.dumps({"ok": False, "error": {"code": code, "message": message}})


def _parent(item_key: str) -> dict[str, Any]:
    if not KEY_PATTERN.fullmatch(item_key):
        raise ValueError("item_key must be an exact eight-character parent key.")
    item = _client.get_zotero_client().item(item_key)
    if not isinstance(item, dict):
        raise ValueError("The requested item could not be verified in the active library.")
    data = item.get("data") or {}
    if (item.get("key") or data.get("key")) != item_key or data.get("itemType") in {"attachment", "note", "annotation"}:
        raise ValueError("The returned record is not the requested parent item.")
    library = item.get("library") or {}
    if library:
        group = 0 if library.get("type") == "user" else library.get("id")
        if group != _client.get_active_group_id():
            raise ValueError("The record belongs to a different library.")
    return {
        "item_key": item_key,
        "title": str(data.get("title") or "")[:300],
        "library_id": _client.get_active_group_id(),
    }


def _index_documents(ids: list[str]) -> dict[str, Any]:
    # Lazy import: find_in_item does not need the semantic dependencies at all.
    from zotero_mcp.chroma_client import read_index_documents

    return read_index_documents(ids, _config_path())


@mcp.tool(
    name="read_passage",
    description=(
        "Expand an evidence_id returned by semantic_search into its full indexed chunk. "
        "No embedding/reranking, full-paper read, or download. Defaults to the matched chunk; "
        "neighbors=1 or 2 adds adjacent chunks from the same item/library within a TOTAL text budget "
        "(max_chars 256–16000, default 8000). Requires the token's library to be active; never switches "
        "libraries. Rejects changed/missing evidence. Indexed text and offsets are NOT verified PDF pages "
        "or sidecar lines. Copy next_char_start into start_char to continue a truncated anchor."
    ),
)
@with_zotero_api_lock
def read_passage(
    evidence_id: str,
    neighbors: int = 0,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
    start_char: int = 0,
    *,
    ctx: Context,
) -> str:
    try:
        group, anchor_id, fingerprint = parse_evidence_id(evidence_id)
        if not 0 <= neighbors <= 2 or not 256 <= max_chars <= MAX_CONTEXT_CHARS or start_char < 0:
            return _error("INVALID_ARGUMENT", "neighbors must be 0–2, max_chars 256–16000, start_char nonnegative.")
        if group != _client.get_active_group_id():
            return _error(
                "LIBRARY_MISMATCH", "The evidence library is not active; select it explicitly before reading."
            )
        key = anchor_id.split("#", 1)[0]
        parent = _parent(key)
        ids = [anchor_id]
        if "#" in anchor_id:
            index = int(anchor_id.split("#")[1])
            ids += [f"{key}#{n}" for n in range(max(0, index - neighbors), index + neighbors + 1) if n != index]
        raw = _index_documents(ids)
        records = {}
        for cid, document, metadata in zip(
            raw.get("ids") or [], raw.get("documents") or [], raw.get("metadatas") or []
        ):
            if cid not in ids:
                return _error("SCOPE_MISMATCH", "The index returned an unrequested chunk.")
            meta = metadata if isinstance(metadata, dict) else {}
            if meta.get("group_id") != group or any(meta.get(k, key) != key for k in ("item_key", "parent_item_key")):
                return _error("SCOPE_MISMATCH", "Indexed library or item provenance does not match the request.")
            if not isinstance(document, str):
                return _error("SOURCE_UNAVAILABLE", "Indexed text is unavailable.")
            records[cid] = (document, meta)
        if anchor_id not in records:
            return _error("STALE_EVIDENCE", "The indexed passage no longer exists; search again.")
        anchor, anchor_meta = records[anchor_id]
        if chunk_fingerprint(anchor, anchor_meta) != fingerprint:
            return _error("STALE_EVIDENCE", "The passage text or metadata changed; search again.")
        if start_char > len(anchor):
            return _error("INVALID_ARGUMENT", "start_char is outside the anchor chunk.")
        chunks = []
        remaining = max_chars
        truncated = False
        next_char = None
        omitted = []
        # Anchor first: neighbors must never crowd out the requested evidence.
        for cid in ids:
            if cid not in records:
                continue
            document, meta = records[cid]
            start = start_char if cid == anchor_id else 0
            if remaining <= 0:
                omitted.append(cid)
                truncated = True
                continue
            end = min(len(document), start + remaining)
            partial = end < len(document)
            if cid == anchor_id and partial:
                next_char = end
            chunks.append(
                {
                    "chunk_id": cid,
                    "evidence_id": make_evidence_id(cid, document, meta),
                    "content_hash": text_hash(document),
                    "text": document[start:end],
                    "char_start": start,
                    "char_end": end,
                    "chunk_chars": len(document),
                    "truncated": partial,
                    "chunk_index": meta.get("chunk_index"),
                    "source": meta.get("fulltext_source", "indexed-text"),
                }
            )
            remaining -= end - start
            truncated = truncated or partial
        return json.dumps(
            {
                "ok": True,
                **parent,
                "route": "indexed_passage",
                "anchor_chunk_id": anchor_id,
                "offset_basis": "zero-based characters within each stored chunk; end exclusive",
                "chunks": chunks,
                "truncated": truncated,
                "next_char_start": next_char,
                "omitted_chunk_ids": omitted,
                "note": "Indexed context, not a new semantic score or verified PDF-page read. Neighbors may overlap and are current index context.",
            },
            ensure_ascii=False,
        )
    except ValueError as exc:
        return _error("INVALID_ARGUMENT", str(exc))
    except Exception:
        ctx.error("Bounded passage read failed; no alternate source was substituted.")
        return _error(
            "SOURCE_UNAVAILABLE",
            "Could not verify the item or read the existing index. No search or rebuild was attempted.",
        )


@mcp.tool(
    name="find_in_item",
    description=(
        "Find a literal phrase in an exact parent item's existing MinerU sidecar and return bounded "
        "source windows with line/character locators. Case-insensitive; no regex, semantic search, OCR, "
        "download, or index needed. Personal library only (legacy sidecars are not library-namespaced). "
        "Use query=null to read lines (default 40 lines), or start_char to continue a long truncated line. "
        "max_chars is a TOTAL text budget, 256–16000; max_matches 1–10; context_lines 0–20. "
        "Pass the returned source_hash as expected_hash on follow-ups to reject changed text."
    ),
)
@with_zotero_api_lock
def find_in_item(
    item_key: str,
    query: str | None = None,
    start_line: int = 1,
    end_line: int | None = None,
    context_lines: int = 3,
    max_matches: int = 5,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
    expected_hash: str | None = None,
    start_char: int | None = None,
    *,
    ctx: Context,
) -> str:
    try:
        if not 256 <= max_chars <= MAX_CONTEXT_CHARS:
            return _error("INVALID_ARGUMENT", "max_chars must be 256–16000.")
        if _client.get_active_group_id() != 0:
            return _error(
                "SIDECAR_LIBRARY_UNSUPPORTED",
                "Legacy sidecars are not library-namespaced; this route supports the personal library only.",
            )
        parent = _parent(item_key)
        cfg = _mineru.load_mineru_config(_config_path())
        path = _mineru.sidecar_path(cfg, item_key)
        # Bound source I/O as well as output; do not process arbitrary paths.
        if not path.is_file():
            return _error("SIDECAR_NOT_FOUND", "No existing MinerU sidecar for this item; nothing was created.")
        with path.open("rb") as stream:
            raw = stream.read(MAX_SIDECAR_BYTES + 1)
        if len(raw) > MAX_SIDECAR_BYTES:
            return _error("SOURCE_TOO_LARGE", "Sidecar exceeds the 16 MiB source-read limit.")
        text = raw.decode("utf-8", errors="replace")
        digest = text_hash(text)
        if expected_hash is not None and expected_hash != digest:
            return _error("STALE_EVIDENCE", "The sidecar changed; repeat the initial lookup before continuing.")
        if start_char is not None:
            if query is not None or end_line is not None or start_line != 1 or not 0 <= start_char <= len(text):
                return _error(
                    "INVALID_ARGUMENT",
                    "start_char requires query=null, default start_line, and no end_line, and must be in range.",
                )
            end = min(len(text), start_char + max_chars)
            window = source_window(text, start_char, end, line_starts(text))
            window["truncated"] = end < len(text)
            result = {
                "windows": [window],
                "truncated": end < len(text),
                "next_char_start": end if end < len(text) else None,
            }
        else:
            result = find_windows(
                text,
                query,
                start_line=start_line,
                end_line=end_line,
                context_lines=context_lines,
                max_matches=max_matches,
                max_chars=max_chars,
            )
            # The exact char offset supports continuation even inside huge HTML table lines.
            windows = result["windows"]
            result["next_char_start"] = windows[-1]["char_end"] if windows and windows[-1]["truncated"] else None
        return json.dumps(
            {
                "ok": True,
                **parent,
                "route": "mineru_sidecar",
                "source_hash": digest,
                "offset_basis": "one-based lines; zero-based source characters, end exclusive",
                **result,
            },
            ensure_ascii=False,
        )
    except ValueError as exc:
        return _error("INVALID_ARGUMENT", str(exc))
    except Exception:
        ctx.error("Bounded sidecar lookup failed; no alternate source was substituted.")
        return _error("SOURCE_UNAVAILABLE", "Could not verify the item or read its existing sidecar.")
