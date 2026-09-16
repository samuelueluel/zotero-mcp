"""Deterministic services for bounded Zotero research workflows.

The services in this module assemble and lint evidence artifacts. They do not
classify papers, interpret findings, repair extracted text, or decide whether
estimates are comparable. MCP adapters provide the source readers so the pure
logic remains hermetic and testable.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from zotero_mcp.claim_audit import ExpectedNumericValue

SCHEMA_VERSION = 1
MAX_QUERY_FACETS = 4
MAX_QUERY_CHARS = 500
MAX_RESULTS_PER_FACET = 12
MAX_INVENTORY_ROWS = 500
MAX_PREVIEW_CHARS = 600

_ITEM_KEY_PATTERN = r"^[A-Za-z0-9]{8}$"


class _StrictModel(BaseModel):
    """Fail closed on misspelled or unsupported public input fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


_QueryFacet = Annotated[str, Field(min_length=1, max_length=MAX_QUERY_CHARS)]


class CandidateScopeRequest(_StrictModel):
    """Validated input for one bounded collection-scoped discovery run."""

    collection_key: str = Field(pattern=_ITEM_KEY_PATTERN)
    query_facets: list[_QueryFacet] = Field(min_length=1, max_length=MAX_QUERY_FACETS)
    limit_per_facet: int = Field(default=8, ge=1, le=MAX_RESULTS_PER_FACET)
    include_subcollections: bool = True
    filters: dict[str, Any] | None = None
    inventory_limit: int = Field(default=250, ge=1, le=MAX_INVENTORY_ROWS)

    @field_validator("query_facets")
    @classmethod
    def _facets_are_distinct(cls, value: list[str]) -> list[str]:
        normalized = [facet.casefold() for facet in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("query_facets must be distinct")
        return value


@dataclass(frozen=True)
class CandidateScopeDependencies:
    """Bounded readers supplied by the MCP adapter or hermetic tests."""

    inventory_reader: Callable[[str, bool], Mapping[str, Any]]
    semantic_searcher: Callable[
        [str, int, Mapping[str, Any] | None, str, bool, set[str]],
        Mapping[str, Any],
    ]


def _item_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _creator_name(creator: Mapping[str, Any]) -> str:
    return str(
        creator.get("lastName")
        or creator.get("name")
        or creator.get("firstName")
        or ""
    ).strip()


def compact_inventory_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded identity metadata for one top-level inventory item."""

    data = item.get("data") if isinstance(item.get("data"), Mapping) else item
    data = data if isinstance(data, Mapping) else {}
    key = _item_key(item.get("key") or data.get("key"))
    creators = data.get("creators") if isinstance(data.get("creators"), list) else []
    names = [
        _creator_name(creator)
        for creator in creators[:6]
        if isinstance(creator, Mapping) and _creator_name(creator)
    ]
    return {
        "item_key": key,
        "title": str(data.get("title") or data.get("filename") or "Untitled")[:500],
        "date": str(data.get("date") or "")[:100],
        "item_type": str(data.get("itemType") or "")[:100],
        "authors": names,
        "has_pdf": bool(item.get("has_pdf")),
        "has_notes": bool(item.get("has_notes")),
    }


def _candidate_metadata(
    result: Mapping[str, Any],
    inventory_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    key = _item_key(result.get("item_key"))
    inventory = dict(inventory_by_key.get(key) or {"item_key": key})
    zotero_item = result.get("zotero_item")
    if isinstance(zotero_item, Mapping):
        hydrated = compact_inventory_item(zotero_item)
        compact = {**inventory, **hydrated}
        # Attachment/note flags come from the complete collection inventory;
        # semantic result hydration returns only the parent metadata record.
        compact["has_pdf"] = bool(inventory.get("has_pdf"))
        compact["has_notes"] = bool(inventory.get("has_notes"))
    else:
        compact = inventory
    compact["item_key"] = key
    source_group = result.get("source_group")
    if source_group:
        compact["source_group"] = str(source_group)[:100]
    return compact


def _hit_record(
    result: Mapping[str, Any],
    *,
    facet_index: int,
    query: str,
    rank: int,
) -> dict[str, Any]:
    rerank = result.get("rerank_score")
    rerank_value = float(rerank) if isinstance(rerank, (int, float)) else None
    relevance = result.get("similarity_score")
    relevance_value = (
        float(relevance) if isinstance(relevance, (int, float)) else None
    )
    preview = str(result.get("matched_passage") or "")[:MAX_PREVIEW_CHARS]
    is_reference = bool(result.get("is_reference"))
    evidence_id = result.get("evidence_id")
    location: dict[str, Any] = {}
    for name in (
        "page",
        "chunk_index",
        "n_chunks",
        "char_start",
        "char_end",
        "passage_offset",
    ):
        if result.get(name) is not None:
            location[name] = result[name]
    return {
        "facet_index": facet_index,
        "query": query,
        "rank": rank,
        "relevance": relevance_value,
        "rerank": rerank_value,
        "is_reference": is_reference,
        "direct_support_eligible": bool(
            evidence_id and rerank_value is not None and rerank_value > 0 and not is_reference
        ),
        "evidence_id": evidence_id,
        "chunk_id": result.get("chunk_id"),
        "content_hash": result.get("content_hash"),
        "location": location,
        "preview": preview,
        "preview_truncated": bool(result.get("preview_truncated")),
    }


class CandidateScopeService:
    """Build a frozen, provenance-preserving candidate-discovery artifact."""

    def __init__(self, dependencies: CandidateScopeDependencies):
        self.dependencies = dependencies

    def build(self, request: CandidateScopeRequest) -> dict[str, Any]:
        inventory_raw = self.dependencies.inventory_reader(
            request.collection_key,
            request.include_subcollections,
        )
        if inventory_raw.get("error"):
            return {
                "schema_version": SCHEMA_VERSION,
                "ok": False,
                "error": dict(inventory_raw["error"]),
            }

        raw_items = inventory_raw.get("items") or []
        inventory: list[dict[str, Any]] = []
        seen_inventory: set[str] = set()
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            compact = compact_inventory_item(raw_item)
            key = compact["item_key"]
            if not key or key in seen_inventory:
                continue
            seen_inventory.add(key)
            inventory.append(compact)
        inventory.sort(key=lambda row: (row.get("title", "").casefold(), row["item_key"]))
        inventory_by_key = {row["item_key"]: row for row in inventory}
        member_keys = set(inventory_by_key)

        candidates: dict[str, dict[str, Any]] = {}
        scope_leaks: list[dict[str, Any]] = []
        facet_summaries: list[dict[str, Any]] = []
        for facet_index, query in enumerate(request.query_facets):
            result = self.dependencies.semantic_searcher(
                query,
                request.limit_per_facet,
                request.filters,
                request.collection_key,
                request.include_subcollections,
                member_keys,
            )
            if result.get("error"):
                return {
                    "schema_version": SCHEMA_VERSION,
                    "ok": False,
                    "error": {
                        "code": "SEARCH_FAILED",
                        "message": str(result.get("error"))[:1000],
                        "facet_index": facet_index,
                    },
                }
            rows = result.get("results") or []
            accepted = 0
            for rank, row in enumerate(rows, 1):
                if not isinstance(row, Mapping):
                    continue
                key = _item_key(row.get("item_key"))
                if not key or key not in member_keys:
                    scope_leaks.append(
                        {"facet_index": facet_index, "rank": rank, "item_key": key}
                    )
                    continue
                accepted += 1
                candidate = candidates.setdefault(
                    key,
                    {
                        **_candidate_metadata(row, inventory_by_key),
                        "hits": [],
                    },
                )
                candidate["hits"].append(
                    _hit_record(
                        row,
                        facet_index=facet_index,
                        query=query,
                        rank=rank,
                    )
                )
            facet_summaries.append(
                {
                    "facet_index": facet_index,
                    "query": query,
                    "returned": len(rows),
                    "accepted_in_scope": accepted,
                }
            )

        candidate_rows: list[dict[str, Any]] = []
        for candidate in candidates.values():
            hits = candidate["hits"]
            positive = [hit["rerank"] for hit in hits if hit["rerank"] is not None and hit["rerank"] > 0]
            candidate["facet_count"] = len({hit["facet_index"] for hit in hits})
            candidate["max_positive_rerank"] = max(positive) if positive else None
            candidate["has_direct_support_eligible_hit"] = any(
                hit["direct_support_eligible"] for hit in hits
            )
            candidate_rows.append(candidate)

        candidate_rows.sort(
            key=lambda row: (
                -(row["max_positive_rerank"] if row["max_positive_rerank"] is not None else float("-inf")),
                -row["facet_count"],
                str(row.get("title") or "").casefold(),
                row["item_key"],
            )
        )

        inventory_limit = request.inventory_limit
        returned_inventory = inventory[:inventory_limit]
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "scope": {
                "collection_key": request.collection_key,
                "collection_name": str(inventory_raw.get("collection_name") or "")[:500],
                "include_subcollections": request.include_subcollections,
                "collection_keys": list(inventory_raw.get("collection_keys") or []),
                "member_count": len(inventory),
                "inventory_returned": len(returned_inventory),
                "inventory_complete": len(returned_inventory) == len(inventory),
            },
            "query_facets": facet_summaries,
            "filters": request.filters,
            "inventory": returned_inventory,
            "candidate_count": len(candidate_rows),
            "candidates": candidate_rows,
            "scope_leak_count": len(scope_leaks),
            "scope_leaks": scope_leaks[:20],
            "note": (
                "Candidates are discovery results, not substantive inclusion decisions. "
                "Positive rerank eligibility does not establish support or causality."
            ),
        }


_LiteralQuery = Annotated[str, Field(min_length=1, max_length=MAX_QUERY_CHARS)]


class ResultEvidenceItemRequest(_StrictModel):
    """One exact-item request for bounded evidence representations."""

    item_key: str = Field(pattern=_ITEM_KEY_PATTERN)
    evidence_id: str | None = Field(default=None, max_length=500)
    sidecar_queries: list[_LiteralQuery] = Field(default_factory=list, max_length=2)
    pdf_queries: list[_LiteralQuery] = Field(default_factory=list, max_length=2)
    pdf_start_page: int | None = Field(default=None, ge=1)
    pdf_end_page: int | None = Field(default=None, ge=1)
    neighbors: int = Field(default=1, ge=0, le=2)

    @field_validator("sidecar_queries", "pdf_queries")
    @classmethod
    def _queries_are_distinct(cls, value: list[str]) -> list[str]:
        normalized = [query.casefold() for query in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("literal queries must be distinct within each route")
        return value

    @field_validator("pdf_end_page")
    @classmethod
    def _page_range_is_forward(cls, value: int | None, info: Any) -> int | None:
        start = info.data.get("pdf_start_page")
        if value is not None and start is None:
            raise ValueError("pdf_end_page requires pdf_start_page")
        if value is not None and start is not None and value < start:
            raise ValueError("pdf_end_page must be greater than or equal to pdf_start_page")
        return value

    def model_post_init(self, __context: Any) -> None:
        if not self.evidence_id and not self.sidecar_queries and not self.pdf_queries:
            raise ValueError("each item requires evidence_id, sidecar_queries, or pdf_queries")


class ResultEvidenceRequest(_StrictModel):
    """Validated input for one bounded multi-item evidence collection run."""

    requests: list[ResultEvidenceItemRequest] = Field(min_length=1, max_length=4)
    max_chars_per_route: int = Field(default=8000, ge=512, le=16000)
    max_pdf_windows: int = Field(default=2, ge=1, le=3)

    @field_validator("requests")
    @classmethod
    def _items_are_distinct(cls, value: list[ResultEvidenceItemRequest]) -> list[ResultEvidenceItemRequest]:
        keys = [request.item_key.upper() for request in value]
        if len(set(keys)) != len(keys):
            raise ValueError("requests must use distinct item_key values")
        return value


@dataclass(frozen=True)
class ResultEvidenceDependencies:
    """Exact-item readers supplied by the adapter or hermetic tests."""

    parent_resolver: Callable[[str], Mapping[str, Any]]
    passage_reader: Callable[[str, int, int], Mapping[str, Any]]
    sidecar_reader: Callable[..., Mapping[str, Any]]
    pdf_reader: Callable[[str, list[str], int | None, int | None, int, int], Mapping[str, Any]]


def _route_texts(record: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    if record.get("route") == "indexed_passage":
        for chunk in record.get("chunks") or []:
            if isinstance(chunk, Mapping) and chunk.get("text"):
                texts.append(str(chunk["text"]))
    for window in record.get("windows") or []:
        if isinstance(window, Mapping) and window.get("text"):
            texts.append(str(window["text"]))
    for match in record.get("matches") or []:
        if isinstance(match, Mapping) and match.get("text"):
            texts.append(str(match["text"]))
    continuation = record.get("continuation")
    if isinstance(continuation, Mapping):
        texts.extend(_route_texts(continuation))
    return texts


def _signature_summary(texts: list[str]) -> dict[str, dict[str, list[str]]]:
    """Index signs and adjacent significance stars by numeric magnitude."""

    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?P<sign>[-+−]?)\s*(?P<number>\d[\d,]*(?:\.\d+)?)"
        r"\s*(?P<unit>%|percent|percentage points?|pp|bps)?\s*(?P<stars>\*{1,3})?",
        flags=re.IGNORECASE,
    )
    signs: dict[str, set[str]] = {}
    stars: dict[str, set[str]] = {}
    for text in texts:
        for match in pattern.finditer(text):
            unit = str(match.group("unit") or "").casefold()
            key = match.group("number").replace(",", "") + unit
            sign = match.group("sign").replace("−", "-")
            signs.setdefault(key, set()).add(sign)
            if marker := match.group("stars"):
                stars.setdefault(key, set()).add(marker)
    return {
        "signs": {key: sorted(value) for key, value in signs.items()},
        "stars": {key: sorted(value) for key, value in stars.items()},
    }


_TABLE_REFERENCE_PATTERN = re.compile(r"\btable\s+(\d{1,3})\b", re.IGNORECASE)


def _referenced_table_numbers(texts: list[str]) -> set[int]:
    return {
        int(match.group(1))
        for text in texts
        for match in _TABLE_REFERENCE_PATTERN.finditer(text)
    }


def _conflict_flags(
    sidecar_records: list[Mapping[str, Any]],
    pdf_record: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not pdf_record or not pdf_record.get("ok"):
        return []
    pdf_by_query = {
        str(row.get("query")): row
        for row in pdf_record.get("queries") or []
        if isinstance(row, Mapping)
    }
    flags: list[dict[str, Any]] = []
    for sidecar in sidecar_records:
        query = str(sidecar.get("query") or "")
        pdf = pdf_by_query.get(query)
        if not query or not pdf or not sidecar.get("ok") or not pdf.get("ok"):
            continue
        side_sig = _signature_summary(_route_texts(sidecar))
        pdf_sig = _signature_summary(_route_texts(pdf))
        shared = set(side_sig["signs"]).intersection(pdf_sig["signs"])
        sign_conflicts = {
            magnitude: {
                "sidecar": side_sig["signs"][magnitude],
                "pdf": pdf_sig["signs"][magnitude],
            }
            for magnitude in shared
            if side_sig["signs"][magnitude] != pdf_sig["signs"][magnitude]
            and "-" in set(side_sig["signs"][magnitude] + pdf_sig["signs"][magnitude])
        }
        if sign_conflicts:
            flags.append(
                {
                    "code": "NUMERIC_SIGNATURE_MISMATCH",
                    "query": query,
                    "conflicts": sign_conflicts,
                }
            )
        # Significance markers are meaningful only when both route windows
        # contain the same estimate-like numeric magnitude. Restrict this to
        # decimals or explicit statistical units so Markdown bold around
        # headings such as **Table 5** or page numbers cannot trigger review.
        star_keys = {
            magnitude
            for magnitude in shared
            if "." in magnitude
            or any(unit in magnitude for unit in ("%", "percent", "pp", "bps"))
        }
        star_conflicts = {
            magnitude: {
                "sidecar": side_sig["stars"].get(magnitude, []),
                "pdf": pdf_sig["stars"].get(magnitude, []),
            }
            for magnitude in star_keys
            if side_sig["stars"].get(magnitude, []) != pdf_sig["stars"].get(magnitude, [])
        }
        if star_conflicts:
            flags.append(
                {
                    "code": "SIGNIFICANCE_MARKER_MISMATCH",
                    "query": query,
                    "conflicts": star_conflicts,
                }
            )
    return flags


class ResultEvidenceService:
    """Collect exact-item evidence while preserving route boundaries."""

    def __init__(self, dependencies: ResultEvidenceDependencies):
        self.dependencies = dependencies

    def collect(self, request: ResultEvidenceRequest) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for item_request in request.requests:
            key = item_request.item_key.upper()
            parent = self.dependencies.parent_resolver(key)
            if parent.get("error"):
                items.append(
                    {
                        "item_key": key,
                        "ok": False,
                        "error": dict(parent["error"]),
                    }
                )
                continue

            record: dict[str, Any] = {
                "item_key": key,
                "title": str(parent.get("title") or "")[:500],
                "library_id": parent.get("library_id"),
                "ok": True,
                "requested": {
                    "evidence_id": item_request.evidence_id,
                    "sidecar_queries": item_request.sidecar_queries,
                    "pdf_queries": item_request.pdf_queries,
                    "pdf_start_page": item_request.pdf_start_page,
                    "pdf_end_page": item_request.pdf_end_page,
                },
            }

            if item_request.evidence_id:
                record["indexed_passage"] = dict(
                    self.dependencies.passage_reader(
                        item_request.evidence_id,
                        item_request.neighbors,
                        request.max_chars_per_route,
                    )
                )

            sidecar_records: list[dict[str, Any]] = []
            expected_hash: str | None = None
            for query in item_request.sidecar_queries:
                sidecar = dict(
                    self.dependencies.sidecar_reader(
                        key,
                        query=query,
                        max_chars=request.max_chars_per_route,
                        expected_hash=expected_hash,
                    )
                )
                sidecar["query"] = query
                if sidecar.get("ok") and sidecar.get("source_hash"):
                    expected_hash = str(sidecar["source_hash"])
                    if sidecar.get("truncated") and sidecar.get("next_char_start") is not None:
                        continuation = dict(
                            self.dependencies.sidecar_reader(
                                key,
                                query=None,
                                max_chars=request.max_chars_per_route,
                                expected_hash=expected_hash,
                                start_char=int(sidecar["next_char_start"]),
                            )
                        )
                        sidecar["continuation"] = continuation
                sidecar_records.append(sidecar)
            if sidecar_records:
                record["sidecar"] = sidecar_records

            pdf_record: dict[str, Any] | None = None
            if item_request.pdf_queries:
                pdf_record = dict(
                    self.dependencies.pdf_reader(
                        key,
                        item_request.pdf_queries,
                        item_request.pdf_start_page,
                        item_request.pdf_end_page,
                        request.max_pdf_windows,
                        request.max_chars_per_route,
                    )
                )
                record["pdf"] = pdf_record

            flags = _conflict_flags(sidecar_records, pdf_record)
            non_pdf_texts: list[str] = []
            indexed = record.get("indexed_passage")
            if isinstance(indexed, Mapping):
                non_pdf_texts.extend(_route_texts(indexed))
            for sidecar in sidecar_records:
                non_pdf_texts.extend(_route_texts(sidecar))
            referenced_tables = _referenced_table_numbers(non_pdf_texts)
            read_tables: set[int] = set()
            if pdf_record and pdf_record.get("ok"):
                for row in pdf_record.get("queries") or []:
                    if not isinstance(row, Mapping) or not row.get("matches"):
                        continue
                    read_tables.update(
                        _referenced_table_numbers(
                            [str(row.get("query") or ""), *_route_texts(row)]
                        )
                    )
            missing_tables = sorted(referenced_tables - read_tables)
            if referenced_tables:
                record["referenced_tables"] = sorted(referenced_tables)
                record["read_referenced_tables"] = sorted(read_tables & referenced_tables)
            if missing_tables:
                record["referenced_tables_not_read"] = missing_tables
                flags.append(
                    {
                        "code": "REFERENCED_TABLE_NOT_READ",
                        "tables": missing_tables,
                    }
                )

            route_errors: list[dict[str, Any]] = []
            indexed = record.get("indexed_passage")
            if isinstance(indexed, Mapping) and not indexed.get("ok"):
                route_errors.append(
                    {"route": "indexed_passage", "error": indexed.get("error")}
                )
            for sidecar in sidecar_records:
                if not sidecar.get("ok"):
                    route_errors.append(
                        {
                            "route": "mineru_sidecar",
                            "query": sidecar.get("query"),
                            "error": sidecar.get("error"),
                        }
                    )
                continuation = sidecar.get("continuation")
                if isinstance(continuation, Mapping) and not continuation.get("ok"):
                    route_errors.append(
                        {
                            "route": "mineru_sidecar_continuation",
                            "query": sidecar.get("query"),
                            "error": continuation.get("error"),
                        }
                    )
            if pdf_record and not pdf_record.get("ok"):
                route_errors.append(
                    {"route": "pdf_extraction", "error": pdf_record.get("error")}
                )
            if pdf_record:
                if pdf_record.get("ok") and pdf_record.get("search_complete") is False:
                    flags.append(
                        {
                            "code": "PDF_SEARCH_INCOMPLETE",
                            "searched_page_range": pdf_record.get("searched_page_range"),
                            "requested_page_range": pdf_record.get("requested_page_range"),
                        }
                    )
                for row in pdf_record.get("queries") or []:
                    if not isinstance(row, Mapping):
                        continue
                    if row.get("coverage") != "complete" and not row.get("matches"):
                        flags.append(
                            {
                                "code": "INCOMPLETE_PDF_TEXT_NO_MATCH",
                                "query": row.get("query"),
                                "coverage": row.get("coverage"),
                            }
                        )
            record["route_errors"] = route_errors
            record["conflict_flags"] = flags
            visual_codes = {
                "NUMERIC_SIGNATURE_MISMATCH",
                "SIGNIFICANCE_MARKER_MISMATCH",
                "INCOMPLETE_PDF_TEXT_NO_MATCH",
            }
            record["requires_visual_review"] = any(
                flag.get("code") in visual_codes for flag in flags
            )
            record["requires_follow_up"] = bool(missing_tables)
            record["ok"] = not route_errors
            items.append(record)

        return {
            "schema_version": SCHEMA_VERSION,
            "ok": all(item.get("ok") for item in items),
            "items": items,
            "note": (
                "Route success means candidate evidence was retrieved, not that a requested substantive field "
                "was verified. Text-route agreement is not image inspection."
            ),
        }


EvidenceRoute = Literal[
    "indexed_passage",
    "mineru_sidecar",
    "pdf_extraction",
    "pdf_rendering",
]
EvidenceRiskTag = Literal[
    "numeric",
    "comparison",
    "calculated",
    "causal",
    "attribution",
]
ResultClass = Literal[
    "main",
    "subgroup",
    "dosage",
    "dynamic",
    "supplemental",
    "robustness",
    "model_based",
]
EligibleResultPolicy = Literal["primary_only", "substantive_all", "custom"]
WinnerStatus = Literal["clear", "not_clear"]
UncertaintyStatus = Literal[
    "reported",
    "threshold_only",
    "not_reported_in_checked_result",
    "not_retrieved",
    "ambiguous",
]


class EvidenceBundleRecord(_StrictModel):
    """One bounded route-specific evidence record selected for a draft claim."""

    evidence_id: str = Field(min_length=1, max_length=200)
    item_key: str = Field(pattern=_ITEM_KEY_PATTERN)
    route: EvidenceRoute
    locator: str = Field(min_length=1, max_length=500)
    page: int | None = Field(default=None, ge=1)
    quote: str = Field(default="", max_length=2000)
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    source_hash: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    ambiguity_flags: list[str] = Field(default_factory=list, max_length=10)


class ClaimEvidenceContext(_StrictModel):
    """Statistical and design context the agent attached to one draft claim."""

    outcome: str | None = Field(default=None, max_length=500)
    estimate: str | None = Field(default=None, max_length=200)
    scale: str | None = Field(default=None, max_length=300)
    treatment: str | None = Field(default=None, max_length=500)
    dose: str | None = Field(default=None, max_length=500)
    denominator: str | None = Field(default=None, max_length=500)
    sample: str | None = Field(default=None, max_length=500)
    geography: str | None = Field(default=None, max_length=300)
    time_horizon: str | None = Field(default=None, max_length=500)
    uncertainty_status: UncertaintyStatus | None = None
    calculation_method: str | None = Field(default=None, max_length=1000)


class DraftEvidenceClaim(_StrictModel):
    """One bounded draft claim linked to selected evidence records."""

    claim_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=1500)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    risk_tags: list[EvidenceRiskTag] = Field(default_factory=list, max_length=5)
    expected_values: list[ExpectedNumericValue] = Field(default_factory=list, max_length=16)
    unverified: bool = False
    context: ClaimEvidenceContext = Field(default_factory=ClaimEvidenceContext)

    @field_validator("evidence_ids", "risk_tags")
    @classmethod
    def _values_are_distinct(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("list values must be distinct")
        return value


class EvidenceBundleValidationRequest(_StrictModel):
    """Validated input for deterministic claim-to-evidence linting."""

    claims: list[DraftEvidenceClaim] = Field(min_length=1, max_length=20)
    evidence: list[EvidenceBundleRecord] = Field(min_length=1, max_length=40)
    allowed_item_keys: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("claims")
    @classmethod
    def _claim_ids_are_distinct(cls, value: list[DraftEvidenceClaim]) -> list[DraftEvidenceClaim]:
        ids = [claim.claim_id for claim in value]
        if len(set(ids)) != len(ids):
            raise ValueError("claim_id values must be distinct")
        return value

    @field_validator("evidence")
    @classmethod
    def _evidence_ids_are_distinct(cls, value: list[EvidenceBundleRecord]) -> list[EvidenceBundleRecord]:
        ids = [record.evidence_id for record in value]
        if len(set(ids)) != len(ids):
            raise ValueError("evidence_id values must be distinct")
        return value

    @field_validator("allowed_item_keys")
    @classmethod
    def _allowed_item_keys_are_valid(cls, value: list[str]) -> list[str]:
        normalized = [key.upper() for key in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_item_keys must be distinct")
        if any(not re.fullmatch(_ITEM_KEY_PATTERN, key) for key in value):
            raise ValueError("allowed_item_keys must contain exact parent item keys")
        return value


def _reason(code: str, message: str, *, blocking: bool = True) -> dict[str, Any]:
    return {"code": code, "message": message, "blocking": blocking}


class ComparisonResultRecord(_StrictModel):
    """One result eligible for a task-specific cross-paper comparison."""

    result_id: str = Field(min_length=1, max_length=100)
    result_class: ResultClass
    outcome: str = Field(min_length=1, max_length=500)
    point_estimate: str = Field(min_length=1, max_length=300)
    scale: str = Field(min_length=1, max_length=300)
    uncertainty: str = Field(min_length=1, max_length=300)
    treatment: str = Field(min_length=1, max_length=500)
    dose: str = Field(min_length=1, max_length=500)
    denominator: str = Field(min_length=1, max_length=500)
    population: str = Field(min_length=1, max_length=500)
    geography: str = Field(min_length=1, max_length=300)
    time_horizon: str = Field(min_length=1, max_length=500)
    specification: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)

    @field_validator("evidence_ids")
    @classmethod
    def _evidence_ids_are_distinct(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("evidence_ids must be distinct")
        return value


ComparisonStatus = Literal["eligible", "no_eligible_result", "unresolved"]


class ComparisonResultCard(_StrictModel):
    """Terminal result inventory for one frozen parent item."""

    item_key: str = Field(pattern=_ITEM_KEY_PATTERN)
    status: ComparisonStatus
    results: list[ComparisonResultRecord] = Field(default_factory=list, max_length=100)
    primary_result_id: str | None = Field(default=None, max_length=100)
    maximum_substantive_result_id: str | None = Field(default=None, max_length=100)
    selected_result_id: str | None = Field(default=None, max_length=100)
    inventory_locators: list[str] = Field(default_factory=list, max_length=20)
    reason: str | None = Field(default=None, max_length=1000)

    @field_validator("results")
    @classmethod
    def _result_ids_are_distinct(cls, value: list[ComparisonResultRecord]) -> list[ComparisonResultRecord]:
        ids = [result.result_id for result in value]
        if len(set(ids)) != len(ids):
            raise ValueError("result_id values must be distinct within an item")
        return value

    @field_validator("inventory_locators")
    @classmethod
    def _inventory_locators_are_bounded(cls, value: list[str]) -> list[str]:
        if any(not locator.strip() or len(locator) > 500 for locator in value):
            raise ValueError("inventory_locators must be nonempty strings of at most 500 characters")
        if len(set(value)) != len(value):
            raise ValueError("inventory_locators must be distinct")
        return value


class ComparisonManifestRequest(_StrictModel):
    """Frozen-set result manifest submitted before a comparison is reported."""

    frozen_item_keys: list[str] = Field(min_length=1, max_length=500)
    cards: list[ComparisonResultCard] = Field(min_length=1, max_length=500)
    ranking_rule: str = Field(min_length=1, max_length=1500)
    eligible_result_policy: EligibleResultPolicy
    conclusion_scope: Literal["complete", "verified_only"] = "complete"
    winner_type: Literal["clear_winner", "top_k"] = "clear_winner"
    numerical_winner_status: WinnerStatus
    substantive_winner_status: WinnerStatus
    alternative_policy_changes_top_k: bool
    max_reported_items: int = Field(default=1, ge=1, le=3)
    selected_item_keys: list[str] = Field(min_length=1, max_length=3)
    reported_item_keys: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("frozen_item_keys", "selected_item_keys", "reported_item_keys")
    @classmethod
    def _keys_are_distinct(cls, value: list[str]) -> list[str]:
        normalized = [key.upper() for key in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("item keys must be distinct")
        return value

    @field_validator("cards")
    @classmethod
    def _card_keys_are_distinct(cls, value: list[ComparisonResultCard]) -> list[ComparisonResultCard]:
        keys = [card.item_key.upper() for card in value]
        if len(set(keys)) != len(keys):
            raise ValueError("cards must contain one entry per item_key")
        return value

    @field_validator("frozen_item_keys")
    @classmethod
    def _frozen_keys_are_distinct(cls, value: list[str]) -> list[str]:
        normalized = [key.upper() for key in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("frozen_item_keys must be distinct")
        return value


class ComparisonManifestValidator:
    """Check frozen-set coverage and final selection without judging estimates."""

    def validate(self, request: ComparisonManifestRequest) -> dict[str, Any]:
        frozen = {key.upper() for key in request.frozen_item_keys}
        card_map = {card.item_key.upper(): card for card in request.cards}
        reasons: list[dict[str, Any]] = []

        missing = sorted(frozen - set(card_map))
        extra = sorted(set(card_map) - frozen)
        if missing or extra:
            details = []
            if missing:
                details.append("missing cards: " + ", ".join(missing))
            if extra:
                details.append("out-of-scope cards: " + ", ".join(extra))
            reasons.append({
                "code": "CARD_SET_MISMATCH",
                "message": "; ".join(details),
                "blocking": True,
            })

        unresolved = [
            key for key, card in card_map.items() if card.status == "unresolved"
        ]
        if unresolved and request.conclusion_scope == "complete":
            reasons.append({
                "code": "UNRESOLVED_ITEMS",
                "message": "unresolved items block a complete-scope comparison: " + ", ".join(sorted(unresolved)),
                "blocking": True,
            })

        for key, card in card_map.items():
            result_ids = {result.result_id for result in card.results}
            if card.status == "eligible":
                if not card.results:
                    reasons.append({
                        "code": "ELIGIBLE_CARD_EMPTY",
                        "message": f"{key} is eligible but has no result records",
                        "blocking": True,
                    })
                required_ids = {
                    "primary_result_id": card.primary_result_id,
                    "maximum_substantive_result_id": card.maximum_substantive_result_id,
                    "selected_result_id": card.selected_result_id,
                }
                for field_name, result_id in required_ids.items():
                    if result_id is None:
                        reasons.append({
                            "code": "RESULT_ROLE_MISSING",
                            "message": f"{key} has no {field_name}",
                            "blocking": True,
                        })
                    elif result_id not in result_ids:
                        reasons.append({
                            "code": "RESULT_ROLE_NOT_FOUND",
                            "message": f"{key} {field_name} is absent from its result inventory",
                            "blocking": True,
                        })
                if not card.inventory_locators:
                    reasons.append({
                        "code": "INVENTORY_LOCATOR_MISSING",
                        "message": f"{key} has no exact result-table or passage locator",
                        "blocking": True,
                    })
                if (
                    request.eligible_result_policy == "primary_only"
                    and card.selected_result_id != card.primary_result_id
                ):
                    reasons.append({
                        "code": "SELECTED_RESULT_POLICY_MISMATCH",
                        "message": f"{key} does not select its primary result under primary_only",
                        "blocking": True,
                    })
                if (
                    request.eligible_result_policy == "substantive_all"
                    and card.selected_result_id != card.maximum_substantive_result_id
                ):
                    reasons.append({
                        "code": "SELECTED_RESULT_POLICY_MISMATCH",
                        "message": f"{key} does not select its maximum substantive result under substantive_all",
                        "blocking": True,
                    })
            elif card.status == "no_eligible_result":
                if card.results:
                    reasons.append({
                        "code": "INELIGIBLE_CARD_HAS_RESULTS",
                        "message": f"{key} is marked no_eligible_result but contains result records",
                        "blocking": True,
                    })
                if any(
                    result_id is not None
                    for result_id in (
                        card.primary_result_id,
                        card.maximum_substantive_result_id,
                        card.selected_result_id,
                    )
                ):
                    reasons.append({
                        "code": "INELIGIBLE_CARD_SELECTED",
                        "message": f"{key} is marked no_eligible_result but selects a result",
                        "blocking": True,
                    })
                if not card.reason:
                    reasons.append({
                        "code": "EXCLUSION_REASON_MISSING",
                        "message": f"{key} needs a reason for having no eligible result",
                        "blocking": True,
                    })
            elif card.status == "unresolved" and any(
                result_id is not None
                for result_id in (
                    card.primary_result_id,
                    card.maximum_substantive_result_id,
                    card.selected_result_id,
                )
            ):
                reasons.append({
                    "code": "UNRESOLVED_CARD_SELECTED",
                    "message": f"{key} cannot select a paper-level result while unresolved",
                    "blocking": True,
                })

        selected = [key.upper() for key in request.selected_item_keys]
        if len(selected) > request.max_reported_items:
            reasons.append({
                "code": "REPORT_LIMIT_EXCEEDED",
                "message": f"selected_item_keys exceeds max_reported_items={request.max_reported_items}",
                "blocking": True,
            })
        if request.winner_type == "clear_winner":
            if request.max_reported_items != 1 or len(selected) != 1:
                reasons.append({
                    "code": "CLEAR_WINNER_SELECTION_INVALID",
                    "message": "clear_winner requires max_reported_items=1 and exactly one selected item",
                    "blocking": True,
                })
            if request.substantive_winner_status != "clear":
                reasons.append({
                    "code": "SUBSTANTIVE_WINNER_NOT_CLEAR",
                    "message": "clear_winner requires a clear substantive winner; use top_k for a numerical-only leader",
                    "blocking": True,
                })
        reported = [key.upper() for key in request.reported_item_keys]
        for field_name, keys in (("selected_item_keys", selected), ("reported_item_keys", reported)):
            outside = sorted(set(keys) - frozen)
            if outside:
                reasons.append({
                    "code": "OUT_OF_SCOPE_SELECTION",
                    "message": f"{field_name} contains out-of-scope items: " + ", ".join(outside),
                    "blocking": True,
                })
        selected_set = set(selected)
        for key in selected:
            card = card_map.get(key)
            if card is None:
                continue
            if card.status != "eligible":
                reasons.append({
                    "code": "NON_ELIGIBLE_SELECTION",
                    "message": f"selected item {key} does not have eligible status",
                    "blocking": True,
                })
        reported_outside_selection = sorted(set(reported) - selected_set)
        if reported_outside_selection:
            reasons.append({
                "code": "REPORTED_ITEM_NOT_SELECTED",
                "message": "final analysis includes items outside the permitted selection: " + ", ".join(reported_outside_selection),
                "blocking": True,
            })

        blocking = [reason for reason in reasons if reason["blocking"]]
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": not blocking,
            "ready": not blocking,
            "complete": not unresolved,
            "qualified_scope": "complete" if not unresolved else "verified_only",
            "summary": {
                "frozen_items": len(frozen),
                "cards": len(card_map),
                "eligible": sum(card.status == "eligible" for card in card_map.values()),
                "no_eligible_result": sum(card.status == "no_eligible_result" for card in card_map.values()),
                "unresolved": len(unresolved),
                "selected": len(selected),
                "eligible_result_policy": request.eligible_result_policy,
                "numerical_winner_status": request.numerical_winner_status,
                "substantive_winner_status": request.substantive_winner_status,
                "alternative_policy_changes_top_k": request.alternative_policy_changes_top_k,
            },
            "reason_codes": [reason["code"] for reason in reasons],
            "reasons": reasons[:20],
        }


class EvidenceBundleValidator:
    """Lint evidence structure without re-reading or adjudicating sources."""

    _NUMERIC_REQUIRED = ("outcome", "estimate", "scale", "treatment", "uncertainty_status")
    _NUMERIC_RECOMMENDED = ("dose", "denominator", "time_horizon")

    def validate(self, request: EvidenceBundleValidationRequest) -> dict[str, Any]:
        from zotero_mcp.claim_audit import _expected_numeric_signatures, _numeric_signatures

        evidence_by_id = {record.evidence_id: record for record in request.evidence}
        allowed_item_keys = {key.upper() for key in request.allowed_item_keys}
        results: list[dict[str, Any]] = []
        for claim in request.claims:
            reasons: list[dict[str, Any]] = []
            linked: list[EvidenceBundleRecord] = []
            for evidence_id in claim.evidence_ids:
                record = evidence_by_id.get(evidence_id)
                if record is None:
                    reasons.append(
                        _reason(
                            "EVIDENCE_NOT_FOUND",
                            f"evidence_id {evidence_id!r} was not supplied",
                        )
                    )
                else:
                    linked.append(record)

            tags = set(claim.risk_tags)
            item_keys = sorted({record.item_key.upper() for record in linked})
            outside_allowed = sorted(set(item_keys) - allowed_item_keys) if allowed_item_keys else []
            if outside_allowed:
                reasons.append(
                    _reason(
                        "ITEM_OUTSIDE_ALLOWED_SCOPE",
                        "claim uses evidence from unselected items: " + ", ".join(outside_allowed),
                    )
                )
            if claim.unverified:
                reasons.append(
                    _reason(
                        "CLAIM_MARKED_UNVERIFIED",
                        "claim is explicitly marked unverified",
                        blocking=False,
                    )
                )
            if "comparison" in tags and len(item_keys) < 2:
                reasons.append(
                    _reason(
                        "COMPARISON_ITEM_MISSING",
                        "comparison claims require linked evidence from at least two distinct items",
                    )
                )

            for record in linked:
                if record.route in {"pdf_extraction", "pdf_rendering"} and record.page is None:
                    reasons.append(
                        _reason(
                            "PDF_PAGE_MISSING",
                            f"{record.evidence_id} uses a PDF route without a one-based page",
                        )
                    )
                if record.ambiguity_flags and not claim.unverified:
                    reasons.append(
                        _reason(
                            "AMBIGUOUS_EVIDENCE",
                            f"{record.evidence_id} has unresolved ambiguity flags",
                        )
                    )

            if "numeric" in tags:
                missing = [
                    field
                    for field in self._NUMERIC_REQUIRED
                    if getattr(claim.context, field) in {None, ""}
                ]
                if missing:
                    reasons.append(
                        _reason(
                            "NUMERIC_CONTEXT_MISSING",
                            "numeric claim is missing: " + ", ".join(missing),
                        )
                    )
                for field in self._NUMERIC_RECOMMENDED:
                    if getattr(claim.context, field) in {None, ""}:
                        reasons.append(
                            _reason(
                                "NUMERIC_CONTEXT_INCOMPLETE",
                                f"numeric claim does not record {field}; use 'not applicable' when appropriate",
                                blocking=False,
                            )
                        )
                if claim.context.uncertainty_status == "ambiguous" and not claim.unverified:
                    reasons.append(
                        _reason(
                            "UNCERTAINTY_AMBIGUOUS",
                            "uncertainty is ambiguous but the claim is not marked unverified",
                        )
                    )

                if "calculated" not in tags:
                    claim_numbers = (
                        _expected_numeric_signatures(claim.expected_values)
                        if claim.expected_values
                        else _numeric_signatures(claim.text)
                    )
                    quote_numbers = set()
                    for record in linked:
                        quote_numbers.update(_numeric_signatures(record.quote))
                    missing_numbers = sorted(claim_numbers - quote_numbers)
                    if missing_numbers:
                        display = [
                            f"{number}{unit}" for number, unit in missing_numbers
                        ]
                        reasons.append(
                            _reason(
                                "NUMERIC_VALUE_NOT_IN_QUOTE",
                                "claim numbers absent from linked quotes: " + ", ".join(display),
                            )
                        )

            if "calculated" in tags and not claim.context.calculation_method:
                reasons.append(
                    _reason(
                        "CALCULATION_METHOD_MISSING",
                        "calculated claims require a calculation_method",
                    )
                )

            blocking = any(reason["blocking"] for reason in reasons)
            results.append(
                {
                    "claim_id": claim.claim_id,
                    "status": "blocked" if blocking else ("warning" if reasons else "ready"),
                    "reason_codes": [reason["code"] for reason in reasons],
                    "reasons": reasons,
                    "item_keys": item_keys,
                    "evidence_ids": claim.evidence_ids,
                }
            )

        ready = all(result["status"] != "blocked" for result in results)
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "ready": ready,
            "results": results,
            "allowed_item_keys": sorted(allowed_item_keys),
            "note": (
                "This is structural validation only. Passing does not establish substantive support, "
                "causality, comparability, or source accuracy and must not be cited as evidence."
            ),
        }


__all__ = [
    "CandidateScopeDependencies",
    "CandidateScopeRequest",
    "CandidateScopeService",
    "ClaimEvidenceContext",
    "ComparisonManifestRequest",
    "ComparisonManifestValidator",
    "ComparisonResultCard",
    "ComparisonResultRecord",
    "DraftEvidenceClaim",
    "EvidenceBundleRecord",
    "EvidenceBundleValidationRequest",
    "EvidenceBundleValidator",
    "ResultEvidenceDependencies",
    "ResultEvidenceItemRequest",
    "ResultEvidenceRequest",
    "ResultEvidenceService",
    "compact_inventory_item",
]
