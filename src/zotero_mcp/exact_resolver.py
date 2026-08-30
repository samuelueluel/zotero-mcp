"""Strict metadata identity resolution for zotero-mcp.

This module is copied into the installed zotero-mcp package by
``zotero-mcp-exact-resolver-patch.py``.  It deliberately performs metadata
lookups only: no semantic search, full-text retrieval, or fallback cascade is
used to establish source identity.  The public wrapper in ``tools/search.py``
adds the MCP tool and API lock.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Literal

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._context import Context
from zotero_mcp.local_db import get_local_zotero_reader
from zotero_mcp.tools import _helpers


# [exact resolver patch] These are intentionally conservative.  The resolver
# accepts punctuation/whitespace/case variation in titles, but never accepts a
# substring or semantic-neighbour match as an exact title.
_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")
_DOI_SCAN_RE = re.compile(r"10\.\d{4,9}/[^\s<>\"]+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_IDENTIFIER_TYPES = {"auto", "title", "doi", "citation_key", "item_key"}
_SOURCE_ITEM_TYPES = {"attachment", "note", "annotation"}


def _fold(value: Any) -> str:
    """Case/Unicode/punctuation-fold a metadata value for exact comparison."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    # Punctuation is a formatting difference for title identity, not a word.
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _extract_quoted_title(source: str) -> str | None:
    """Extract a title from common prose such as ``paper titled 'X'``."""
    match = re.search(
        r"(?:titled|called|title)\s*(?:is\s*)?[\"'“‘](.+?)[\"'”’]",
        source,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _extract_doi(source: str) -> str | None:
    for token in _DOI_SCAN_RE.findall(source):
        if normalized := _helpers._normalize_doi(token):
            return normalized
    return None


def _parse_request(
    source: str,
    identifier_type: str,
    title: str | None,
    author: str | None,
    year: str | None,
    doi: str | None,
    citation_key: str | None,
    item_key: str | None,
) -> tuple[dict[str, str | None], list[str]]:
    """Return explicit identity fields without treating prose as evidence."""
    errors: list[str] = []
    raw = str(source or "").strip()
    kind = str(identifier_type or "auto").strip().lower()
    if kind not in _IDENTIFIER_TYPES:
        errors.append(
            "identifier_type must be one of auto, title, doi, citation_key, or item_key"
        )
        kind = "auto"

    parsed_title = _clean_optional(title)
    parsed_author = _clean_optional(author)
    parsed_year = _clean_optional(year)
    parsed_doi = _clean_optional(doi)
    parsed_citekey = _clean_optional(citation_key)
    parsed_item_key = _clean_optional(item_key)

    if kind == "title" and not parsed_title:
        parsed_title = raw
    elif kind == "doi" and not parsed_doi:
        parsed_doi = _helpers._normalize_doi(raw)
    elif kind == "citation_key" and not parsed_citekey:
        parsed_citekey = raw
    elif kind == "item_key" and not parsed_item_key:
        parsed_item_key = raw

    # Auto mode accepts an explicit DOI or Zotero item key, and extracts a
    # quoted title from a natural-language request.  A bare string otherwise
    # means a title; this keeps the tool useful without making a one-word title
    # silently become a citation key.
    if kind == "auto":
        if not parsed_doi:
            parsed_doi = _extract_doi(raw)
        if not parsed_item_key and _KEY_RE.fullmatch(raw.upper()):
            parsed_item_key = raw.upper()
        if not parsed_title:
            parsed_title = _extract_quoted_title(raw)
        if not parsed_title and not parsed_doi and not parsed_item_key and not parsed_citekey:
            parsed_title = raw or None

    if parsed_doi:
        parsed_doi = _helpers._normalize_doi(parsed_doi)
        if not parsed_doi:
            errors.append(f"Invalid DOI: {doi!r}")
    if parsed_item_key:
        parsed_item_key = parsed_item_key.strip().upper()
        if not parsed_item_key:
            parsed_item_key = None
    if parsed_year:
        year_match = _YEAR_RE.search(parsed_year)
        if year_match:
            parsed_year = year_match.group(0)

    if not any(
        [parsed_title, parsed_author, parsed_year, parsed_doi, parsed_citekey, parsed_item_key]
    ):
        errors.append(
            "Provide a source title, author/year, DOI, citation key, or Zotero item key"
        )

    return {
        "title": parsed_title,
        "author": parsed_author,
        "year": parsed_year,
        "doi": parsed_doi,
        "citation_key": parsed_citekey,
        "item_key": parsed_item_key,
    }, errors


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _error(message: str, fields: dict[str, str | None] | None = None) -> str:
    return _json(
        {
            "identity_status": "error",
            "error": message,
            "query": fields or {},
            "exact_matches": [],
            "ambiguous_matches": [],
            "related_matches": [],
            "match_basis": [],
        }
    )


def _creator_names(data: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for creator in data.get("creators", []) or []:
        if not isinstance(creator, dict):
            continue
        if creator.get("name"):
            names.append(str(creator["name"]).strip())
            continue
        first = str(creator.get("firstName") or "").strip()
        last = str(creator.get("lastName") or "").strip()
        name = " ".join(part for part in (first, last) if part).strip()
        if name:
            names.append(name)
    return names


def _item_title(data: dict[str, Any]) -> str:
    title = data.get("title")
    if title:
        return str(title)
    # The ordinary formatter knows type-specific title fields.  Avoid relying
    # on a private display helper if a future package drops it.
    for field in ("nameOfAct", "caseName", "subject", "filename"):
        if data.get(field):
            return str(data[field])
    return ""


def _item_summary(
    item: dict[str, Any],
    reason: str | None = None,
    *,
    in_requested_scope: bool | None = None,
    scope_basis: str = "no_collection_scope_requested",
) -> dict[str, Any]:
    data = item.get("data", {}) if isinstance(item, dict) else {}
    summary: dict[str, Any] = {
        "item_key": item.get("key") or data.get("key"),
        "title": _item_title(data),
        "item_type": data.get("itemType"),
        "date": data.get("date") or "",
        "year": next(iter(_YEAR_RE.findall(str(data.get("date") or ""))), None),
        "authors": _creator_names(data),
        "doi": data.get("DOI") or "",
        "collections": list(data.get("collections") or []),
        # `collections` is often omitted from local/API summaries. These
        # fields report the resolver's authoritative requested-collection
        # calculation instead of asking callers to infer scope from it.
        "in_requested_scope": in_requested_scope,
        "scope_basis": scope_basis,
    }
    if isinstance(item.get("library"), dict):
        summary["library"] = item["library"]
    if reason:
        summary["match_basis"] = reason
    return summary


def _local_collection_scope(
    reader,
    collection_key: str,
    include_subcollections: bool,
    expected_group_id: int | None,
) -> set[str] | None:
    """Return live collection membership, or None when the collection is absent.

    Collection and item keys are only unique within a Zotero library. Refuse
    an active-library mismatch rather than allowing a same-key item from a
    different library to satisfy the scope accidentally.
    """
    conn = reader._get_connection()
    row = conn.execute(
        "SELECT c.collectionID, c.libraryID, l.type AS library_type, g.groupID "
        "FROM collections c JOIN libraries l ON l.libraryID = c.libraryID "
        "LEFT JOIN groups g ON g.libraryID = c.libraryID "
        "WHERE c.key = ? AND ((? = 0 AND l.type = 'user') OR g.groupID = ?)",
        (collection_key, expected_group_id, expected_group_id),
    ).fetchone()
    if row is None:
        return None
    collection_group_id = int(row["groupID"]) if row["groupID"] is not None else 0
    if expected_group_id is not None and collection_group_id != int(expected_group_id):
        raise ValueError(
            f"Collection '{collection_key}' belongs to library groupID "
            f"{collection_group_id}, but the active library is groupID "
            f"{expected_group_id}; switch libraries before resolving it"
        )
    if include_subcollections:
        collection_keys = reader.resolve_collection_keys(collection_key)
    else:
        collection_keys = [collection_key]
    if not collection_keys:
        return set()
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "('collectionItems', 'itemCollections') ORDER BY name DESC LIMIT 1"
    ).fetchone()
    join_table = table_row[0] if table_row else "collectionItems"
    placeholders = ",".join("?" for _ in collection_keys)
    rows = conn.execute(
        f"SELECT DISTINCT i.key FROM {join_table} ic "
        "JOIN items i ON i.itemID = ic.itemID "
        "JOIN collections c ON c.collectionID = ic.collectionID "
        f"WHERE c.libraryID = ? AND c.key IN ({placeholders})",
        [int(row["libraryID"]), *collection_keys],
    ).fetchall()
    return {str(row[0]) for row in rows if row[0]}


def _api_collection_scope(zot, collection_key: str, include_subcollections: bool) -> set[str] | None:
    try:
        collection = zot.collection(collection_key)
    except Exception:
        return None
    if not collection or collection.get("key") != collection_key:
        return None
    scope_keys = _helpers.expand_collection_scope(
        zot, collection_key, include_subcollections
    )
    item_keys: set[str] = set()
    for scope_key in scope_keys:
        for item in _utils._paginate(zot.collection_items, scope_key):
            if item.get("key"):
                item_keys.add(str(item["key"]))
    return item_keys


def _local_citation_key_search(reader, citekey: str, group_id: int | None) -> list[dict]:
    """Find candidate records whose Extra/native citation key contains citekey."""
    try:
        conn = reader._get_connection()
        library_ids = reader._resolve_scope_library_ids(group_id)
    except Exception:
        return []
    if not library_ids:
        return []
    placeholders = ",".join("?" for _ in library_ids)
    rows = conn.execute(
        "SELECT i.key, LOWER(f.fieldName) AS field_name, v.value FROM items i "
        "JOIN itemData d ON d.itemID = i.itemID "
        "JOIN itemDataValues v ON v.valueID = d.valueID "
        "JOIN fields f ON f.fieldID = d.fieldID "
        f"WHERE i.libraryID IN ({placeholders}) "
        "AND i.itemID NOT IN (SELECT itemID FROM deletedItems) "
        "AND (LOWER(f.fieldName) = 'extra' OR LOWER(f.fieldName) = 'citationkey') "
        "AND (LOWER(v.value) LIKE '%citation key:%' OR LOWER(f.fieldName) = 'citationkey')",
        list(library_ids),
    ).fetchall()
    if not rows:
        return []
    keys = [str(row[0]) for row in rows]
    items = reader.get_items_by_keys(keys)
    # row_to_api_item intentionally omits Extra/citationKey because ordinary
    # search formatting does not need them. Add the identity fields back for
    # this resolver before applying the package's exact line parser.
    identity_values: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        identity_values.setdefault(str(row[0]), {}).setdefault(
            str(row[1]), []
        ).append(str(row[2] or ""))
    out: list[dict] = []
    for key, item in items.items():
        data = item.get("data", {})
        values = identity_values.get(key, {})
        extra = "\n".join(values.get("extra", []))
        native_values = values.get("citationkey", [])
        if extra:
            data["extra"] = extra
        if native_values:
            data["citationKey"] = native_values[0]
        native = str(data.get("citationKey") or "")
        if native == citekey or _helpers._extra_has_citekey(extra, citekey):
            out.append(item)
    return out


def _local_metadata_search(
    reader,
    query: str,
    kind: str,
    group_id: int | None,
    limit: int,
) -> list[dict] | None:
    try:
        if kind == "doi":
            return reader.advanced_search_sql(
                [{"field": "DOI", "operation": "is", "value": query}],
                join_mode="all",
                group_id=group_id,
            )
        if kind == "citation_key":
            return _local_citation_key_search(reader, query, group_id)
        exact = reader.advanced_search_sql(
            [{"field": "title", "operation": "is", "value": query}],
            join_mode="all",
            group_id=group_id,
        )
        fuzzy = reader.search_items_sql(
            query,
            qmode="titleCreatorYear",
            item_type="-attachment",
            limit=max(100, limit),
            group_id=group_id,
        )
        merged: dict[tuple[str, str], dict] = {}
        for item in [*exact, *fuzzy]:
            key = str(item.get("key") or "")
            library_id = str((item.get("library") or {}).get("id", ""))
            if key:
                merged.setdefault((key, library_id), item)
        return list(merged.values())
    except Exception:
        return None


def _api_metadata_search(
    zot,
    query: str,
    kind: str,
    limit: int,
    collection_keys: list[str] | None = None,
) -> list[dict]:
    """Metadata-only query; importantly, never invokes semantic fallback."""
    qmode = "everything" if kind in {"doi", "citation_key"} else "titleCreatorYear"
    variants = _utils._generate_search_variants(query) or [query]
    methods = []
    if collection_keys:
        methods = [(zot.collection_items, collection_key) for collection_key in collection_keys]
    else:
        methods = [(zot.items, None)]
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for variant in variants:
        for method, collection_key in methods:
            params = {
                "q": variant,
                "qmode": qmode,
                "itemType": "-attachment",
            }
            # Identity resolution must fail closed. Paginate the metadata
            # query rather than turning an API error or a full first page into
            # a false `absent` result; only the returned summaries are bounded.
            batch = (
                _utils._paginate(method, collection_key, **params)
                if collection_key
                else _utils._paginate(method, **params)
            )
            for item in batch:
                key = item.get("key")
                library_id = str((item.get("library") or {}).get("id", ""))
                identity = (str(key or ""), library_id)
                if key and identity not in seen:
                    seen.add(identity)
                    results.append(item)
    return results


def _item_key_candidates(reader, zot, item_key: str, group_id: int | None) -> list[dict]:
    if reader is not None:
        items = reader.get_items_by_keys([item_key])
        if group_id is None:
            return list(items.values())
        return [
            item
            for item in items.values()
            if int((item.get("library") or {}).get("id", -1)) == int(group_id)
        ]
    try:
        item = zot.item(item_key)
    except Exception:
        return []
    return [item] if item and item.get("key") == item_key else []


def _creator_matches(data: dict[str, Any], requested: str) -> bool:
    target = _fold(requested)
    if not target:
        return True
    target_reversed = " ".join(reversed(target.split()))
    for creator in data.get("creators", []) or []:
        if not isinstance(creator, dict):
            continue
        full = _fold(
            creator.get("name")
            or " ".join(
                part
                for part in (creator.get("firstName"), creator.get("lastName"))
                if part
            )
        )
        last = _fold(creator.get("lastName"))
        if target in {full, last} or target_reversed == full:
            return True
    return False


def _year_matches(data: dict[str, Any], requested: str) -> bool:
    return str(requested) in _YEAR_RE.findall(str(data.get("date") or ""))


def _citekey_matches(data: dict[str, Any], requested: str) -> bool:
    return str(data.get("citationKey") or "") == requested or _helpers._extra_has_citekey(
        str(data.get("extra") or ""), requested
    )


def _identity_checks(
    item: dict[str, Any], fields: dict[str, str | None]
) -> tuple[list[str], list[str]]:
    """Return (satisfied fields, mismatched fields) for one metadata record."""
    data = item.get("data", {})
    satisfied: list[str] = []
    mismatched: list[str] = []
    if fields.get("item_key"):
        (satisfied if str(item.get("key")) == fields["item_key"] else mismatched).append("item_key")
    if fields.get("title"):
        (satisfied if _fold(_item_title(data)) == _fold(fields["title"]) else mismatched).append("title")
    if fields.get("author"):
        (satisfied if _creator_matches(data, fields["author"]) else mismatched).append("author")
    if fields.get("year"):
        (satisfied if _year_matches(data, fields["year"]) else mismatched).append("year")
    if fields.get("doi"):
        record_doi = _helpers._normalize_doi(data.get("DOI") or "")
        (satisfied if record_doi and record_doi.casefold() == str(fields["doi"]).casefold() else mismatched).append("doi")
    if fields.get("citation_key"):
        (satisfied if _citekey_matches(data, fields["citation_key"]) else mismatched).append("citation_key")
    return satisfied, mismatched


def _relaxed_title_queries(title: str) -> list[str]:
    """Bounded metadata-only related lookup for common subtitle mutations."""
    queries: list[str] = []
    # A related result may be useful after an impossible suffix was appended,
    # but it is never allowed into exact_matches.
    for part in re.split(r"\s+[—–-]\s+", title):
        part = part.strip()
        if part != title.strip() and len(_fold(part)) >= 12:
            queries.append(part)
    parenthetical = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    if parenthetical != title.strip() and len(_fold(parenthetical)) >= 12:
        queries.append(parenthetical)
    return list(dict.fromkeys(queries))[:2]


def _route_name(local_used: bool, api_used: bool) -> str:
    if local_used and api_used:
        return "local_sqlite_metadata+zotero_api_metadata"
    if local_used:
        return "local_sqlite_metadata"
    return "zotero_api_metadata"


def resolve_exact_source(
    source: str,
    identifier_type: Literal["auto", "title", "doi", "citation_key", "item_key"] = "auto",
    title: str | None = None,
    author: str | None = None,
    year: str | None = None,
    doi: str | None = None,
    citation_key: str | None = None,
    item_key: str | None = None,
    collection_key: str | None = None,
    include_subcollections: bool = True,
    search_all_libraries: bool = False,
    limit: int | str | None = 20,
    *,
    ctx: Context,
) -> str:
    """Resolve source identity using exact metadata and return JSON."""
    fields, errors = _parse_request(
        source, identifier_type, title, author, year, doi, citation_key, item_key
    )
    if errors:
        return _error("; ".join(errors), fields)
    if search_all_libraries and collection_key:
        return _error(
            "collection_key cannot be combined with search_all_libraries; collection keys are library-scoped",
            fields,
        )
    if search_all_libraries and fields.get("item_key"):
        return _error(
            "search_all_libraries with item_key is refused because Zotero item keys are only unique within a library",
            fields,
        )

    try:
        max_results = _helpers._normalize_limit(limit, default=20, max_val=100)
    except Exception as exc:
        return _error(f"Invalid limit: {exc}", fields)

    group_id = None if search_all_libraries else _client.get_active_group_id()
    reader = get_local_zotero_reader()
    local_used = reader is not None
    api_used = False
    try:
        if search_all_libraries and reader is None:
            return _error(
                "search_all_libraries requires local SQLite mode and a readable zotero.sqlite database",
                fields,
            )

        zot = _client.get_zotero_client()
        scope_item_keys: set[str] | None = None
        scope_source = "library"
        if collection_key:
            collection_key = str(collection_key).strip()
            if not collection_key:
                return _error("collection_key cannot be empty", fields)
            if reader is not None:
                scope_item_keys = _local_collection_scope(
                    reader,
                    collection_key,
                    include_subcollections,
                    group_id,
                )
                scope_source = "local_sqlite_collection_membership"
            else:
                scope_item_keys = _api_collection_scope(
                    zot, collection_key, include_subcollections
                )
                api_used = True
                scope_source = "zotero_api_collection_membership"
            if scope_item_keys is None:
                return _error(f"Collection not found: '{collection_key}'", fields)

        ctx.info(
            "Resolving Zotero source identity with exact metadata only"
            + (f" in collection {collection_key}" if collection_key else "")
        )

        candidates: dict[tuple[str, str], dict] = {}
        query_specs: list[tuple[str, str]] = []
        if fields.get("item_key"):
            query_specs.append(("item_key", fields["item_key"]))
        if fields.get("doi"):
            query_specs.append(("doi", fields["doi"]))
        if fields.get("citation_key"):
            query_specs.append(("citation_key", fields["citation_key"]))
        if fields.get("title"):
            query_specs.append(("title", fields["title"]))
        elif fields.get("author") or fields.get("year"):
            query_specs.append(
                (
                    "title",
                    " ".join(
                        value
                        for value in (fields.get("author"), fields.get("year"))
                        if value
                    ),
                )
            )

        def add_items(items: list[dict], route: str) -> None:
            nonlocal local_used, api_used
            if route == "local":
                local_used = True
            else:
                api_used = True
            for item in items:
                key = str(item.get("key") or item.get("data", {}).get("key") or "")
                if key:
                    # A key is library-scoped; preserve library identity when a
                    # global local-DB query returns the same key twice.
                    library_id = str((item.get("library") or {}).get("id", ""))
                    candidates.setdefault((key, library_id), item)

        for kind, query in query_specs:
            if kind == "item_key":
                add_items(_item_key_candidates(reader, zot, query, group_id), "local" if reader else "api")
                continue
            if reader is not None:
                local_items = _local_metadata_search(
                    reader, query, kind, group_id, max_results
                )
                if local_items:
                    add_items(local_items, "local")
                    continue
                # A local DB can be stale or omit a field (notably BBT keys),
                # so one bounded API metadata retry is allowed when not global.
                if search_all_libraries:
                    continue
            collection_scopes = None
            if collection_key and scope_item_keys is not None and not reader:
                collection_scopes = _helpers.expand_collection_scope(
                    zot, collection_key, include_subcollections
                )
            api_items = _api_metadata_search(
                zot, query, kind, max_results, collection_scopes
            )
            add_items(api_items, "api")

        # A title-prefix/subtitle lookup is related-only and runs only when the
        # exact metadata queries did not already produce a full identity match.
        related_query_specs: list[tuple[str, str]] = []
        if fields.get("title"):
            related_query_specs = [
                ("title", query)
                for query in _relaxed_title_queries(fields["title"])
            ]
        for kind, query in related_query_specs:
            if reader is not None:
                related_items = _local_metadata_search(
                    reader, query, kind, group_id, max_results
                )
                if related_items:
                    add_items(related_items, "local")
                    continue
            if not search_all_libraries:
                collection_scopes = None
                if collection_key and scope_item_keys is not None and not reader:
                    collection_scopes = _helpers.expand_collection_scope(
                        zot, collection_key, include_subcollections
                    )
                add_items(
                    _api_metadata_search(
                        zot, query, kind, max_results, collection_scopes
                    ),
                    "api",
                )

        records: list[dict[str, Any]] = []
        for item in candidates.values():
            satisfied, mismatched = _identity_checks(item, fields)
            key = str(item.get("key") or item.get("data", {}).get("key") or "")
            in_scope = scope_item_keys is None or key in scope_item_keys
            records.append(
                {
                    "item": item,
                    "satisfied": satisfied,
                    "mismatched": mismatched,
                    "all_metadata": bool(satisfied) and not mismatched and len(satisfied) == sum(bool(v) for v in fields.values()),
                    "in_scope": in_scope,
                }
            )

        exact_records = [r for r in records if r["all_metadata"] and r["in_scope"]]
        partial_records = [
            r for r in records
            if r["in_scope"] and r["satisfied"] and r["mismatched"]
        ]
        out_of_scope_exact = [r for r in records if r["all_metadata"] and not r["in_scope"]]
        related_records = [
            r for r in records
            if r["in_scope"]
            and r not in exact_records
            and r not in partial_records
            and r not in out_of_scope_exact
        ]

        conflicts: list[dict[str, Any]] = []
        for record in partial_records:
            if fields.get("doi") and "doi" in record["mismatched"]:
                data = record["item"].get("data", {})
                conflicts.append(
                    {
                        "item_key": record["item"].get("key"),
                        "title": _item_title(data),
                        "requested_doi": fields["doi"],
                        "record_doi": data.get("DOI") or "",
                        "mismatched_fields": record["mismatched"],
                    }
                )

        def summarize_record(record: dict[str, Any], reason: str) -> dict[str, Any]:
            if scope_item_keys is None:
                in_requested_scope = None
                scope_basis = "no_collection_scope_requested"
            elif record["in_scope"]:
                in_requested_scope = True
                scope_basis = scope_source
            else:
                in_requested_scope = False
                scope_basis = (
                    f"{scope_source}:not_member_of_requested_collection"
                )
            return _item_summary(
                record["item"],
                reason,
                in_requested_scope=in_requested_scope,
                scope_basis=scope_basis,
            )

        exact_summaries = [
            summarize_record(r, "all supplied metadata matched")
            for r in exact_records
        ]
        ambiguous_summaries = [
            summarize_record(
                r,
                "matched " + ", ".join(r["satisfied"])
                + "; mismatch in " + ", ".join(r["mismatched"]),
            )
            for r in partial_records
        ]
        ambiguous_summaries.extend(
            summarize_record(r, "metadata match is outside requested collection scope")
            for r in out_of_scope_exact
        )
        related_summaries = [
            summarize_record(r, "related metadata result; not an exact identity match")
            for r in related_records
        ]
        # Keep the response bounded even if an API returns many partials.
        ambiguous_summaries = ambiguous_summaries[:max_results]
        related_summaries = related_summaries[:max_results]

        if len(exact_records) == 1:
            status = "exact"
            match_basis = [
                f"{field} exact"
                for field, value in fields.items()
                if value
            ]
            if scope_item_keys is not None:
                match_basis.append("collection membership verified")
        elif len(exact_records) > 1:
            status = "ambiguous"
            # Do not expose multiple full matches as an unqualified exact
            # answer; place them in the explicit ambiguity field.
            ambiguous_summaries = [
                summarize_record(r, "multiple records satisfy all supplied metadata")
                for r in exact_records
            ] + ambiguous_summaries
            exact_summaries = []
            match_basis = ["multiple records satisfy the supplied metadata"]
        elif partial_records or out_of_scope_exact:
            status = "ambiguous" if partial_records else "absent"
            match_basis = (
                ["one or more metadata fields matched, but the supplied identity did not"]
                if partial_records
                else ["a metadata match exists outside the requested collection scope"]
            )
        else:
            status = "absent"
            match_basis = ["no in-scope record satisfied all supplied identity metadata"]

        if status == "absent" and related_summaries:
            match_basis.append("related metadata results are reported separately and are not substitutes")
        if conflicts:
            match_basis.append("identifier conflict requires clarification")

        next_action = {
            "exact": "Use the returned item_key for substantive retrieval and citation.",
            "ambiguous": "Stop source-specific retrieval; disclose the conflict or request clarification.",
            "absent": "Stop source-specific retrieval; report absence. Related records are metadata-only and are not substitutes.",
        }[status]
        return _json(
            {
                "identity_status": status,
                "query": {
                    "source": str(source or ""),
                    "identifier_type": identifier_type,
                    **fields,
                },
                "collection_scope": {
                    "collection_key": collection_key,
                    "include_subcollections": bool(include_subcollections),
                    "applied": collection_key is not None,
                    "item_count": len(scope_item_keys) if scope_item_keys is not None else None,
                    "membership_route": scope_source,
                    "search_all_libraries": bool(search_all_libraries),
                },
                "search_route": _route_name(local_used, api_used),
                "match_basis": match_basis,
                "next_action": next_action,
                "prohibited_after_status": (
                    [
                        "semantic_search",
                        "get_item_fulltext",
                        "get_pdf_outline",
                        "get_citation_neighbors",
                        "find_bibliographically_coupled_papers",
                    ]
                    if status in {"ambiguous", "absent"}
                    else []
                ),
                "conflicts": conflicts,
                "exact_matches": exact_summaries,
                "ambiguous_matches": ambiguous_summaries,
                "related_matches": related_summaries,
                "warnings": [
                    "Exact identity is metadata-only; related matches must not be treated as the requested source."
                ]
                + [
                    "The exact record is a non-source Zotero item type; substantive paper evidence may not exist."
                ]
                * any(
                    summary.get("item_type") in _SOURCE_ITEM_TYPES
                    for summary in exact_summaries
                ),
            }
        )
    except Exception as exc:
        return _error(f"Exact-source resolution failed: {exc}", fields)
    finally:
        if reader is not None:
            reader.close()
