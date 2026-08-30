"""Source-group and semantic-filter helpers for zotero-mcp.

This module is copied into the installed ``zotero_mcp`` package by
``zotero-mcp-source-filters-patch.py``.  Zotero's native ``itemType`` remains
the source of truth; ``SOURCE_GROUP_ITEM_TYPES`` is a query-time convenience
mapping and is never written back as a tag or field.
"""
from __future__ import annotations

import json
import re
from typing import Any


SOURCE_GROUP_ITEM_TYPES: dict[str, frozenset[str]] = {
    "reference": frozenset(
        {"book", "bookSection", "dictionaryEntry", "encyclopediaArticle"}
    ),
    "article": frozenset({"journalArticle", "conferencePaper"}),
    "unpublished": frozenset({"preprint", "manuscript", "presentation"}),
    "institutional": frozenset({"report", "dataset", "standard"}),
    "web-media": frozenset(
        {
            "webpage",
            "blogPost",
            "forumPost",
            "magazineArticle",
            "newspaperArticle",
            "podcast",
            "radioBroadcast",
            "film",
            "videoRecording",
            "tvBroadcast",
            "audioRecording",
            "interview",
            "letter",
            "email",
            "instantMessage",
        }
    ),
    "other": frozenset({"artwork", "map", "document", "computerProgram"}),
}

# These are deliberately not source groups in the paper RAG.  The first three
# are internal/non-paper items; the rest were explicitly excluded from the
# default paper corpus in the design discussion.
DEFAULT_EXCLUDED_ITEM_TYPES = frozenset(
    {
        "note",
        "thesis",
        "case",
        "bill",
        "hearing",
        "statute",
        "patent",
        "attachment",
        "annotation",
    }
)

KNOWN_ITEM_TYPES = frozenset().union(
    *SOURCE_GROUP_ITEM_TYPES.values(), DEFAULT_EXCLUDED_ITEM_TYPES
)

_GROUP_ALIASES = {
    "web_media": "web-media",
    "web/media": "web-media",
    "unpublished-research": "unpublished",
    "institutional-document": "institutional",
}


def _as_list(value: Any, field_name: str) -> list[str]:
    """Normalize a string/list/JSON-string input without splitting tag names."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(entry).strip() for entry in value if str(entry).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        if isinstance(parsed, list):
            return _as_list(parsed, field_name)
        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []
        raise ValueError(f"{field_name} must be a string or list of strings")
    raise ValueError(f"{field_name} must be a string or list of strings")


def _pop_aliases(raw: dict[str, Any], aliases: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for alias in aliases:
        if alias in raw:
            values.extend(_as_list(raw.pop(alias), alias))
    return values


def canonical_item_type(value: str) -> str:
    """Return the exact Zotero itemType spelling or raise on an unknown type."""
    raw = str(value).strip()
    for known in KNOWN_ITEM_TYPES:
        if raw == known or raw.lower() == known.lower():
            return known
    raise ValueError(
        f"Unknown Zotero itemType {value!r}. Use a native itemType such as "
        "book, journalArticle, preprint, report, or manuscript."
    )


def canonical_source_group(value: str) -> str:
    raw = str(value).strip().lower()
    raw = _GROUP_ALIASES.get(raw, raw)
    if raw not in SOURCE_GROUP_ITEM_TYPES:
        raise ValueError(
            f"Unknown source_group {value!r}. Valid groups: "
            f"{', '.join(SOURCE_GROUP_ITEM_TYPES)}"
        )
    return raw


def source_group_for_item_type(item_type: str | None) -> str | None:
    """Return the derived group for one native itemType, if it has one."""
    if not item_type:
        return None
    for group, item_types in SOURCE_GROUP_ITEM_TYPES.items():
        if item_type in item_types:
            return group
    return None


def expand_source_groups(values: list[str]) -> tuple[list[str], list[str]]:
    """Return canonical groups and their union of native itemTypes."""
    groups: list[str] = []
    item_types: set[str] = set()
    for value in values:
        group = canonical_source_group(value)
        if group not in groups:
            groups.append(group)
        item_types.update(SOURCE_GROUP_ITEM_TYPES[group])
    return groups, sorted(item_types)


_ITEM_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")


def parse_item_keys(values: list[str]) -> list[str]:
    """Normalize an item_key/item_keys filter to canonical uppercase keys.

    Zotero item keys are 8 alphanumeric characters and case-insensitive in
    common usage. The result is the resolved scope itself: unlike tags or
    itemTypes, no database lookup is needed to apply it.
    """
    keys: list[str] = []
    for entry in values:
        # Accept JSON-encoded lists or single strings per entry, mirroring
        # how _pop_aliases normalizes other filter values.
        for item in _as_list(entry, "item_keys"):
            for piece in str(item).replace(";", ",").split(","):
                key = piece.strip().upper()
                if not key:
                    continue
                if not _ITEM_KEY_RE.fullmatch(key):
                    raise ValueError(
                        f"Invalid Zotero item key {piece!r}; expected 8 "
                        "alphanumeric characters, e.g. UJ99YV3A"
                    )
                if key not in keys:
                    keys.append(key)
    if len(keys) > 100:
        raise ValueError("item_keys filter accepts at most 100 keys")
    return keys


def parse_semantic_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Split semantic filters into Chroma ``where`` and live Zotero filters.

    Native metadata and tags remain separate in Zotero.  This function only
    removes the convenience keys from the caller's filter object, expands
    source groups to native types, and leaves all other Chroma predicates
    untouched.
    """
    raw = dict(filters or {})
    item_type_values = _pop_aliases(
        raw, ("item_type", "itemType", "item_types", "itemTypes")
    )
    source_group_values = _pop_aliases(
        raw, ("source_group", "source_groups")
    )
    tag_values = _pop_aliases(
        raw, ("tag", "tags", "required_tags")
    )
    excluded_tag_values = _pop_aliases(raw, ("exclude_tags",))
    item_key_filter_present = any(
        alias in raw for alias in ("item_key", "item_keys")
    )
    item_key_values = _pop_aliases(raw, ("item_key", "item_keys"))
    tag_values.extend(
        value if value.startswith("-") else f"-{value}"
        for value in excluded_tag_values
        if value
    )

    item_types: list[str] = []
    for value in item_type_values:
        canonical = canonical_item_type(value)
        if canonical not in item_types:
            item_types.append(canonical)

    source_groups, group_item_types = expand_source_groups(source_group_values)
    if group_item_types:
        if item_types:
            intersection = sorted(set(item_types).intersection(group_item_types))
            if not intersection:
                raise ValueError(
                    "item_type and source_group filters have no overlapping "
                    "native Zotero itemTypes"
                )
            item_types = intersection
        else:
            item_types = group_item_types

    # Textbooks are a controlled subtype of native books.  A tag-only query
    # therefore carries the native book restriction automatically.  Lecture
    # notes intentionally have no single native itemType.
    textbook_tag = any(
        value.strip().lower() == "type:textbook"
        for value in tag_values
        if not value.startswith("-")
    )
    if textbook_tag:
        if item_types and "book" not in item_types:
            raise ValueError(
                "type:textbook is only compatible with native itemType=book"
            )
        item_types = ["book"]

    # Chroma requires each where object to contain exactly one field/operator.
    # Compile independent metadata predicates as explicit AND clauses instead
    # of returning a multi-key dict such as {"year": "2022", "item_type":
    # "book"}, which Chroma rejects at query time.
    clauses: list[dict[str, Any]] = [
        {key: value} for key, value in raw.items()
    ]
    if item_types:
        clauses.append({
            "item_type": (
                item_types[0] if len(item_types) == 1 else {"$in": item_types}
            )
        })
    where: dict[str, Any] = (
        {} if not clauses
        else clauses[0] if len(clauses) == 1
        else {"$and": clauses}
    )

    item_keys = parse_item_keys(item_key_values)
    if item_key_filter_present and not item_keys:
        raise ValueError(
            "item_key/item_keys was supplied but contained no valid keys; "
            "refusing to widen semantic search scope"
        )

    return {
        "where": where,
        "item_types": item_types,
        "source_groups": source_groups,
        "tags": tag_values,
        "item_keys": item_keys,
    }
