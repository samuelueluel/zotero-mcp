# Tool-name migration

This fork intentionally makes a breaking MCP API change. Ordinary raw tool names no longer repeat the server name: Pi (or another gateway) supplies the single `zotero_` namespace prefix. Schemas and behavior are unchanged unless a row records a behavior-clarifying name.

- **Raw name** is registered by the MCP server.
- **Pi-visible name** is the name used by Samuel’s agent skills.
- `search` and `fetch` remain fixed because the ChatGPT connector contract requires those exact raw names.
- `scite_*` names were already clear and remain unchanged.
- Toolset `core` is always registered; other rows retain their existing optional/default-on toolset membership.

| Toolset | Old raw name | New raw name | New Pi-visible name |
|---|---|---|---|
| `core` | `zotero_add_item` | `add_item` | `zotero_add_item` |
| `core` | `zotero_attach_file` | `attach_file` | `zotero_attach_file` |
| `core` | `zotero_batch_update` | `batch_edit_tags_and_extra` | `zotero_batch_edit_tags_and_extra` |
| `core` | `zotero_synthesize_annotations` | `compile_annotation_digest` | `zotero_compile_annotation_digest` |
| `core` | `zotero_create_annotation` | `create_annotation` | `zotero_create_annotation` |
| `core` | `zotero_create_collection` | `create_collection` | `zotero_create_collection` |
| `core` | `zotero_delete_annotation` | `delete_annotation` | `zotero_delete_annotation` |
| `core` | `zotero_delete_collection` | `delete_collection` | `zotero_delete_collection` |
| `core` | `zotero_delete_item` | `delete_item` | `zotero_delete_item` |
| `core` | `zotero_export_bibliography` | `export_bibliography` | `zotero_export_bibliography` |
| `core` | `zotero_find_connected_papers` | `find_bibliographically_coupled_papers` | `zotero_find_bibliographically_coupled_papers` |
| `core` | `zotero_search_by_citation_key` | `find_item_by_citation_key` | `zotero_find_item_by_citation_key` |
| `core` | `zotero_get_annotations` | `get_annotations` | `zotero_get_annotations` |
| `core` | `zotero_get_attachment_path` | `get_attachment_paths` | `zotero_get_attachment_paths` |
| `core` | `zotero_get_paper_lineage` | `get_citation_neighbors` | `zotero_get_citation_neighbors` |
| `core` | `zotero_get_item_fulltext` | `get_item_fulltext` | `zotero_get_item_fulltext` |
| `core` | `zotero_get_item_metadata` | `get_item_metadata` | `zotero_get_item_metadata` |
| `core` | `zotero_get_notes` | `get_notes` | `zotero_get_notes` |
| `core` | `zotero_audit_references` | `get_reference_index_status` | `zotero_get_reference_index_status` |
| `core` | `zotero_get_collection_items` | `list_collection_items` | `zotero_list_collection_items` |
| `core` | `zotero_get_collections` | `list_collections` | `zotero_list_collections` |
| `core` | `zotero_get_item_children` | `list_item_children` | `zotero_list_item_children` |
| `core` | `zotero_get_recent` | `list_recent_items` | `zotero_list_recent_items` |
| `core` | `zotero_get_tags` | `list_tags` | `zotero_list_tags` |
| `core` | `zotero_manage_note` | `manage_note` | `zotero_manage_note` |
| `core` | `zotero_get_collection_hubs` | `rank_works_by_inbound_citations` | `zotero_rank_works_by_inbound_citations` |
| `core` | `zotero_read_pdf_pages` | `read_pdf_pages` | `zotero_read_pdf_pages` |
| `core` | `zotero_rebuild_citation_graph` | `rebuild_citation_graph` | `zotero_rebuild_citation_graph` |
| `core` | `zotero_rebuild_reference_index` | `rebuild_reference_index` | `zotero_rebuild_reference_index` |
| `core` | `zotero_resolve_exact_source` | `resolve_exact_source` | `zotero_resolve_exact_source` |
| `core` | `zotero_search_references` | `search_bibliography_entries` | `zotero_search_bibliography_entries` |
| `core` | `zotero_search_collections` | `search_collections` | `zotero_search_collections` |
| `core` | `zotero_search_items` | `search_items` | `zotero_search_items` |
| `core` | `zotero_advanced_search` | `search_items_advanced` | `zotero_search_items_advanced` |
| `core` | `zotero_search_by_tag` | `search_items_by_tag` | `zotero_search_items_by_tag` |
| `core` | `zotero_semantic_search` | `semantic_search` | `zotero_semantic_search` |
| `core` | `zotero_set_item_collections` | `set_item_collections` | `zotero_set_item_collections` |
| `core` | `zotero_set_item_parent` | `set_item_parent` | `zotero_set_item_parent` |
| `core` | `zotero_update_annotation` | `update_annotation` | `zotero_update_annotation` |
| `core` | `zotero_update_item` | `update_item` | `zotero_update_item` |
| `libraries` | `zotero_list_libraries` | `list_libraries` | `zotero_list_libraries` |
| `libraries` | `zotero_switch_library` | `switch_library` | `zotero_switch_library` |
| `search-admin` | `zotero_get_search_database_status` | `get_semantic_index_status` | `zotero_get_semantic_index_status` |
| `search-admin` | `zotero_update_search_database` | `update_semantic_index` | `zotero_update_semantic_index` |
| `pdf-geometry` | `zotero_get_page_layout` | `detect_pdf_regions` | `zotero_detect_pdf_regions` |
| `pdf-geometry` | `zotero_get_pdf_outline` | `get_pdf_outline` | `zotero_get_pdf_outline` |
| `discovery` | `zotero_library_coverage` | `audit_pdf_coverage` | `zotero_audit_pdf_coverage` |
| `discovery` | `zotero_find_related_papers` | `discover_citing_and_referenced_works` | `zotero_discover_citing_and_referenced_works` |
| `duplicates` | `zotero_find_duplicates` | `find_duplicate_items` | `zotero_find_duplicate_items` |
| `duplicates` | `zotero_merge_duplicates` | `merge_duplicate_items` | `zotero_merge_duplicate_items` |
| `relations` | `zotero_add_item_relation` | `add_item_relation` | `zotero_add_item_relation` |
| `relations` | `zotero_get_item_related` | `list_related_items` | `zotero_list_related_items` |
| `relations` | `zotero_remove_item_relation` | `remove_item_relation` | `zotero_remove_item_relation` |
| `feeds` | `zotero_get_feed_items` | `list_feed_items` | `zotero_list_feed_items` |
| `feeds` | `zotero_list_feeds` | `list_feeds` | `zotero_list_feeds` |
| `scite` | `scite_check_retractions` | `scite_check_retractions` | `zotero_scite_check_retractions` |
| `scite` | `scite_enrich_item` | `scite_enrich_item` | `zotero_scite_enrich_item` |
| `scite` | `scite_enrich_search` | `scite_enrich_search` | `zotero_scite_enrich_search` |
| `chatgpt-connector` | `fetch` | `fetch` | `zotero_fetch` |
| `chatgpt-connector` | `search` | `search` | `zotero_search` |

Total registered tools: **60**.
