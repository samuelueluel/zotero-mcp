"""Public MCP tool-name contract for the Samuel fork."""

from __future__ import annotations

import asyncio

from zotero_mcp.server import mcp
from zotero_mcp.toolsets import apply_toolsets

EXPECTED_TOOL_NAMES = {
    "add_item",
    "add_item_relation",
    "attach_file",
    "audit_pdf_coverage",
    "batch_edit_tags_and_extra",
    "compile_annotation_digest",
    "create_annotation",
    "create_collection",
    "delete_annotation",
    "delete_collection",
    "delete_item",
    "detect_pdf_regions",
    "discover_citing_and_referenced_works",
    "export_bibliography",
    "fetch",
    "find_bibliographically_coupled_papers",
    "find_duplicate_items",
    "find_item_by_citation_key",
    "get_annotations",
    "get_attachment_paths",
    "get_citation_neighbors",
    "get_item_fulltext",
    "get_item_metadata",
    "get_notes",
    "get_pdf_outline",
    "get_reference_index_status",
    "get_semantic_index_status",
    "list_collection_items",
    "list_collections",
    "list_feed_items",
    "list_feeds",
    "list_item_children",
    "list_libraries",
    "list_recent_items",
    "list_related_items",
    "list_tags",
    "manage_note",
    "merge_duplicate_items",
    "rank_works_by_inbound_citations",
    "read_pdf_pages",
    "rebuild_citation_graph",
    "rebuild_reference_index",
    "remove_item_relation",
    "resolve_exact_source",
    "scite_check_retractions",
    "scite_enrich_item",
    "scite_enrich_search",
    "search",
    "search_bibliography_entries",
    "search_collections",
    "search_items",
    "search_items_advanced",
    "search_items_by_tag",
    "semantic_search",
    "set_item_collections",
    "set_item_parent",
    "switch_library",
    "update_annotation",
    "update_item",
    "update_semantic_index",
}


def test_complete_public_tool_name_contract():
    """All active and inactive toolsets use the approved raw names."""
    try:
        apply_toolsets(mcp, raw="all", transport="streamable-http")
        names = {tool.name for tool in asyncio.run(mcp.list_tools())}
        assert names == EXPECTED_TOOL_NAMES
        assert len(names) == 60
        assert not any(name.startswith("zotero_") for name in names)
    finally:
        apply_toolsets(mcp, raw="all", transport="streamable-http")
