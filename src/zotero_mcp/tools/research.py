"""Composite, bounded research workflow tools."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from zotero_mcp import client as _client
from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.client import with_zotero_api_lock, zotero_api_lock
from zotero_mcp.research_workflows import (
    CandidateScopeDependencies,
    CandidateScopeRequest,
    CandidateScopeService,
    DraftEvidenceClaim,
    EvidenceBundleRecord,
    EvidenceBundleValidationRequest,
    EvidenceBundleValidator,
    ResultEvidenceDependencies,
    ResultEvidenceItemRequest,
    ResultEvidenceRequest,
    ResultEvidenceService,
)
from zotero_mcp.tools import _helpers
from zotero_mcp.tools.retrieval import _is_top_level_item


def _config_path() -> str:
    return str(Path.home() / ".config" / "zotero-mcp" / "config.json")


def _json_error(code: str, message: str) -> str:
    return json.dumps(
        {"schema_version": 1, "ok": False, "error": {"code": code, "message": message[:1000]}},
        ensure_ascii=False,
        sort_keys=True,
    )


def _parse_json_list(value: list[str] | str, field: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be a JSON array: {exc.msg}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list or JSON-stringified list")
    return value


def _parse_filters(value: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"filters must be a JSON object: {exc.msg}") from exc
    if value is not None and not isinstance(value, dict):
        raise ValueError("filters must be an object or JSON-stringified object")
    return copy.deepcopy(value)


def _collection_inventory(collection_key: str, include_subcollections: bool) -> dict[str, Any]:
    """Read and deduplicate the complete collection scope through Zotero's API."""

    zot = _client.get_zotero_client()
    try:
        collection = zot.collection(collection_key)
    except Exception:
        return {
            "error": {
                "code": "COLLECTION_NOT_FOUND",
                "message": f"Collection {collection_key} could not be verified in the active library.",
            }
        }
    data = collection.get("data", {}) if isinstance(collection, dict) else {}
    returned_key = str(collection.get("key") or data.get("key") or "").strip().upper()
    if returned_key and returned_key != collection_key.upper():
        return {
            "error": {
                "code": "COLLECTION_MISMATCH",
                "message": "The collection lookup returned a different key.",
            }
        }

    scope_keys = _helpers.expand_collection_scope(
        zot, collection_key, include_subcollections
    )
    all_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scope_key in scope_keys:
        for item in _helpers._paginate(zot.collection_items, scope_key):
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or item.get("data", {}).get("key") or "").strip().upper()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            all_items.append(item)

    attachment_info: dict[str, dict[str, bool]] = {}
    for item in all_items:
        item_data = item.get("data", {})
        parent = str(item_data.get("parentItem") or "").strip().upper()
        if not parent:
            continue
        info = attachment_info.setdefault(parent, {"has_pdf": False, "has_notes": False})
        if item_data.get("itemType") == "attachment" and item_data.get("contentType") == "application/pdf":
            info["has_pdf"] = True
        elif item_data.get("itemType") == "note":
            info["has_notes"] = True

    parents: list[dict[str, Any]] = []
    for item in all_items:
        if not _is_top_level_item(item):
            continue
        key = str(item.get("key") or item.get("data", {}).get("key") or "").strip().upper()
        info = attachment_info.get(key, {})
        item_data = item.get("data", {})
        standalone_pdf = (
            item_data.get("itemType") == "attachment"
            and not item_data.get("parentItem")
            and item_data.get("contentType") == "application/pdf"
        )
        parents.append(
            {
                **item,
                "has_pdf": bool(info.get("has_pdf") or standalone_pdf),
                "has_notes": bool(info.get("has_notes")),
            }
        )

    return {
        "collection_name": str(data.get("name") or "")[:500],
        "collection_keys": scope_keys,
        "items": parents,
    }


def _merge_item_scope(
    filters: dict[str, Any] | None,
    allowed_keys: set[str],
) -> dict[str, Any]:
    scoped = copy.deepcopy(filters) if filters else {}
    requested = scoped.pop("item_key", None)
    requested_many = scoped.pop("item_keys", None)
    raw = requested_many if requested_many is not None else requested
    if raw is None:
        keys = allowed_keys
    else:
        values = raw if isinstance(raw, list) else [raw]
        keys = {
            str(value).strip().upper()
            for value in values
            if str(value).strip().upper() in allowed_keys
        }
    scoped["item_keys"] = sorted(keys)
    return scoped


def _make_semantic_searcher():
    """Create one semantic service for every facet in a candidate-scope call."""

    from zotero_mcp.semantic_search import create_semantic_search

    search = None

    def run(
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
        collection_key: str,
        include_subcollections: bool,
        member_keys: set[str],
    ) -> dict[str, Any]:
        nonlocal search
        if search is None:
            search = create_semantic_search(_config_path())
        kwargs: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "filters": copy.deepcopy(filters),
            "group_id": _client.get_active_group_id(),
        }
        if include_subcollections:
            kwargs["collection_key"] = collection_key
        else:
            kwargs["filters"] = _merge_item_scope(filters, member_keys)
        return search.search(**kwargs)

    return run


@mcp.tool(
    name="build_candidate_scope",
    description=(
        "Build one bounded collection-scoped paper-discovery artifact from 1–4 agent-supplied semantic "
        "query facets. Verifies and inventories the collection, includes subcollections by default, "
        "runs only the supplied searches, deduplicates parent items, preserves evidence IDs and raw "
        "rerank scores, and rejects out-of-scope hits. Returns compact JSON; candidates are not substantive "
        "inclusion decisions. Does not update the index or synthesize findings."
    ),
)
@with_zotero_api_lock
def build_candidate_scope(
    collection_key: str,
    query_facets: list[str] | str,
    limit_per_facet: int = 8,
    include_subcollections: bool = True,
    filters: dict[str, Any] | str | None = None,
    inventory_limit: int = 250,
    *,
    ctx: Context,
) -> str:
    """Return a frozen candidate manifest with scope and evidence provenance."""

    try:
        parsed = CandidateScopeRequest(
            collection_key=collection_key,
            query_facets=_parse_json_list(query_facets, "query_facets"),
            limit_per_facet=limit_per_facet,
            include_subcollections=include_subcollections,
            filters=_parse_filters(filters),
            inventory_limit=inventory_limit,
        )
        response = CandidateScopeService(
            CandidateScopeDependencies(
                inventory_reader=_collection_inventory,
                semantic_searcher=_make_semantic_searcher(),
            )
        ).build(parsed)
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except ValidationError:
        message = "candidate-scope input failed schema validation"
    except Exception as exc:
        message = str(exc)[:1000]
    try:
        ctx.error(f"Candidate-scope request rejected: {message}")
    except Exception:
        pass
    return _json_error("INVALID_INPUT", message)


def _resolve_parent(item_key: str) -> dict[str, Any]:
    from zotero_mcp.tools.context import _parent

    try:
        with zotero_api_lock():
            return _parent(item_key)
    except ValueError as exc:
        return {"error": {"code": "INVALID_ITEM", "message": str(exc)}}
    except Exception:
        return {
            "error": {
                "code": "SOURCE_UNAVAILABLE",
                "message": "Could not verify the exact parent item in the active library.",
            }
        }


def _read_passage_record(evidence_id: str, neighbors: int, max_chars: int, ctx: Context) -> dict[str, Any]:
    from zotero_mcp.tools.context import read_passage

    raw = read_passage(
        evidence_id=evidence_id,
        neighbors=neighbors,
        max_chars=max_chars,
        ctx=ctx,
    )
    return json.loads(raw)


def _read_sidecar_record(
    item_key: str,
    *,
    query: str | None,
    max_chars: int,
    expected_hash: str | None,
    ctx: Context,
    start_char: int | None = None,
) -> dict[str, Any]:
    from zotero_mcp.tools.context import find_in_item

    context_lines = 10 if query and "table" in query.casefold() else 3
    raw = find_in_item(
        item_key=item_key,
        query=query,
        context_lines=context_lines,
        max_matches=3,
        max_chars=max_chars,
        expected_hash=expected_hash,
        start_char=start_char,
        ctx=ctx,
    )
    return json.loads(raw)


def _aggregate_coverage(states: list[str]) -> str:
    if states and all(state == "complete" for state in states):
        return "complete"
    if states and all(state == "no_usable_text" for state in states):
        return "no_usable_text"
    return "partial_text_coverage"


def _read_pdf_queries(
    item_key: str,
    queries: list[str],
    start_page: int | None,
    end_page: int | None,
    max_windows: int,
    max_chars: int,
    ctx: Context,
) -> dict[str, Any]:
    """Resolve one PDF once and search supplied phrases over bounded page windows."""

    from zotero_mcp.extract import extract_pdf, pdf_page_count
    from zotero_mcp.pdf_evidence import (
        PdfEvidenceInputError,
        classify_text_coverage,
        compile_literal_pattern,
        find_literal_matches,
    )
    from zotero_mcp.tools.read_pdf import _cleanup_path, _get_pdf_path, _pdf_source_route

    pdf_path: str | None = None
    is_temp = False
    try:
        for query in queries:
            compile_literal_pattern(query)
        with zotero_api_lock():
            resolved = _get_pdf_path(item_key, ctx)
        if resolved is None:
            return {
                "ok": False,
                "error": {"code": "PDF_NOT_FOUND", "message": "No PDF attachment was found."},
            }
        pdf_path, title, is_temp = resolved
        source_is_temp = is_temp
        total_pages = pdf_page_count(pdf_path)
        start = start_page or 1
        if start < 1 or start > total_pages:
            raise PdfEvidenceInputError(
                f"pdf_start_page {start} is out of range; PDF has {total_pages} pages."
            )
        requested_end = end_page or total_pages
        if requested_end < start or requested_end > total_pages:
            raise PdfEvidenceInputError(
                f"pdf_end_page {requested_end} is outside {start}-{total_pages}."
            )
        maximum_end = min(requested_end, start + max_windows * 50 - 1)
        windows = [
            (window_start, min(maximum_end, window_start + 49))
            for window_start in range(start, maximum_end + 1, 50)
        ]
        extracted: list[tuple[int, int, Any, dict[str, Any]]] = []
        for window_start, window_end in windows:
            document = extract_pdf(
                pdf_path,
                pages=list(range(window_start - 1, window_end)),
            )
            extracted.append(
                (
                    window_start,
                    window_end,
                    document,
                    classify_text_coverage(document),
                )
            )

        query_rows: list[dict[str, Any]] = []
        for query in queries:
            matches: list[dict[str, Any]] = []
            total_matches = 0
            returned_chars = 0
            coverage_states: list[str] = []
            for _, _, document, coverage in extracted:
                coverage_states.append(str(coverage["state"]))
                remaining_chars = max_chars - returned_chars
                if remaining_chars <= 0:
                    evidence = find_literal_matches(
                        document,
                        query,
                        max_matches=1,
                        max_chars=256,
                        context_chars=0,
                    )
                    total_matches += int(evidence["total_matches"])
                    continue
                evidence = find_literal_matches(
                    document,
                    query,
                    max_matches=max(1, 10 - len(matches)),
                    max_chars=max(256, remaining_chars),
                    context_chars=min(600, max(0, remaining_chars // 2)),
                )
                total_matches += int(evidence["total_matches"])
                for match in evidence["matches"]:
                    if len(matches) >= 10:
                        break
                    text = str(match.get("text") or "")
                    if returned_chars + len(text) > max_chars:
                        break
                    matches.append(match)
                    returned_chars += len(text)
            coverage_state = _aggregate_coverage(coverage_states)
            query_rows.append(
                {
                    "ok": True,
                    "query": query,
                    "coverage": coverage_state,
                    "total_matches": total_matches,
                    "returned_matches": len(matches),
                    "has_more_matches": total_matches > len(matches),
                    "matches": matches,
                    "returned_excerpt_chars": returned_chars,
                }
            )

        return {
            "ok": True,
            "item_key": item_key,
            "title": str(title or "")[:500],
            "route": "pdf_extraction",
            "source_route": _pdf_source_route(source_is_temp),
            "extraction_route": "direct_pdf_text",
            "extraction_engine": "pdf-inspector",
            "page_basis": "one-based PDF pages",
            "total_pages": total_pages,
            "requested_page_range": [start, requested_end],
            "searched_page_range": [start, maximum_end],
            "search_complete": maximum_end == requested_end,
            "window_count": len(windows),
            "text_layer_coverage_windows": [
                {
                    "page_range": [window_start, window_end],
                    "coverage": coverage,
                }
                for window_start, window_end, _, coverage in extracted
            ],
            "queries": query_rows,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": {"code": "PDF_LOOKUP_FAILED", "message": str(exc)[:1000]},
        }
    finally:
        if is_temp and pdf_path:
            _cleanup_path(pdf_path)


def _parse_request_list(
    value: list[ResultEvidenceItemRequest] | str,
) -> list[ResultEvidenceItemRequest] | list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"requests must be a JSON array: {exc.msg}") from exc
    if not isinstance(value, list):
        raise ValueError("requests must be a list or JSON-stringified list")
    return value


@mcp.tool(
    name="collect_result_evidence",
    description=(
        "Collect bounded indexed, MinerU-sidecar, and PDF-text evidence for 1–4 exact parent items in "
        "one call. Expands supplied evidence IDs first, chains sidecar source hashes across lookups and "
        "one continuation, searches at most two literal phrases per text route, preserves one-based PDF "
        "page provenance and text-layer coverage, and keeps routes separate. Numeric or star disagreements "
        "are flagged for visual review, never repaired. Does not render images or decide that a substantive "
        "field was verified."
    ),
)
def collect_result_evidence(
    requests: list[ResultEvidenceItemRequest] | str,
    max_chars_per_route: int = 8000,
    max_pdf_windows: int = 2,
    *,
    ctx: Context,
) -> str:
    """Return route-separated candidate result evidence for exact items."""

    try:
        parsed = ResultEvidenceRequest(
            requests=_parse_request_list(requests),
            max_chars_per_route=max_chars_per_route,
            max_pdf_windows=max_pdf_windows,
        )
        response = ResultEvidenceService(
            ResultEvidenceDependencies(
                parent_resolver=_resolve_parent,
                passage_reader=lambda token, neighbors, max_chars: _read_passage_record(
                    token, neighbors, max_chars, ctx
                ),
                sidecar_reader=lambda key, **kwargs: _read_sidecar_record(
                    key, ctx=ctx, **kwargs
                ),
                pdf_reader=lambda key, queries, start, end, windows, max_chars: _read_pdf_queries(
                    key,
                    queries,
                    start,
                    end,
                    windows,
                    max_chars,
                    ctx,
                ),
            )
        ).collect(parsed)
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except ValidationError:
        message = "result-evidence input failed schema validation"
    except Exception as exc:
        message = str(exc)[:1000]
    try:
        ctx.error(f"Result-evidence request rejected: {message}")
    except Exception:
        pass
    return _json_error("INVALID_INPUT", message)


def _parse_json_array(value: list[Any] | str, field: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be a JSON array: {exc.msg}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list or JSON-stringified list")
    return value


@mcp.tool(
    name="validate_evidence_bundle",
    description=(
        "Deterministically lint 1–20 draft claims against 1–40 caller-supplied, route-specific evidence "
        "records. Checks evidence linkage, distinct comparator items, numeric context, quote-contained "
        "numbers, calculation labels, one-based PDF-page provenance, and unresolved ambiguity flags. "
        "Does not read Zotero, repair source text, judge causality or comparability, or prove substantive "
        "support. A passing result must never be cited as evidence."
    ),
)
def validate_evidence_bundle(
    claims: list[DraftEvidenceClaim] | str,
    evidence: list[EvidenceBundleRecord] | str,
    *,
    ctx: Context,
) -> str:
    """Return bounded structural claim-to-evidence validation results."""

    try:
        parsed = EvidenceBundleValidationRequest(
            claims=_parse_json_array(claims, "claims"),
            evidence=_parse_json_array(evidence, "evidence"),
        )
        response = EvidenceBundleValidator().validate(parsed)
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except ValidationError:
        message = "evidence-bundle input failed schema validation"
    except Exception as exc:
        message = str(exc)[:1000]
    try:
        ctx.error(f"Evidence-bundle request rejected: {message}")
    except Exception:
        pass
    return _json_error("INVALID_INPUT", message)


__all__ = [
    "build_candidate_scope",
    "collect_result_evidence",
    "validate_evidence_bundle",
]
