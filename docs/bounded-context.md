# Bounded source context

The Samuel fork separates **discovery** from **reading**. Search stays compact and
keeps one best hit per distinct paper. Agents can expand the selected hit without
repeating semantic search or requesting an entire document.

Both new tools are core tools (no optional toolset is required). Pi exposes them
as `zotero_read_passage` and `zotero_find_in_item`.

## Discover → expand

```python
semantic_search(query="demolition crime estimates", collection="TRGBCDX5", limit=5)
# Copy the Evidence ID from the relevant hit, unchanged:
read_passage(evidence_id="<returned ID>")
# If the result needs adjacent table notes or continuation:
read_passage(evidence_id="<returned ID>", neighbors=1, max_chars=12000)
```

`semantic_search` returns an explicitly labeled **Preview**, `Preview truncated`,
`Chunk ID`, `Content hash`, and (when indexed library provenance exists) an
`Evidence ID`. The preview is at most 600 characters and prefers sentence/paragraph
boundaries. Injected document breadcrumbs are excluded from the preview's starting
position where possible. Selection remains lexical, not an entailment judgment.
The reranker evaluates the stored chunk, not just its displayed preview.

Search `limit` still counts **distinct items**, not passages. Increasing it for an
exact-item query does not reveal additional hits from that paper. This preserves
candidate diversity during collection discovery. Use expansion or literal lookup
rather than repeating increasingly similar queries to see more text.

### `read_passage` contract

- Returns JSON in the MCP text content, with `ok`, parent identity, `route`,
  `anchor_chunk_id`, `chunks`, `truncated`, and continuation information.
- Defaults to the complete matched indexed chunk. `neighbors=1` or `2` requests up
  to that many preceding and following chunks from the **same item and library**.
- `max_chars` is the total source-text budget across all chunks, default 8,000;
  allowed range 256–16,000. JSON metadata is additional, bounded overhead.
- The anchor receives the budget first. Neighbor chunks may overlap, be partial,
  or be omitted; each returned chunk names its own index, hash, and evidence ID.
- For a truncated anchor, copy `next_char_start` to `start_char` with the same
  evidence ID. For a partial neighbor, use its own ID and `char_end` as the next
  `start_char`. Omitted neighbors are named in `omitted_chunk_ids`.
- Offsets are **zero-based, end-exclusive characters in the stored chunk**. They
  are not PDF pages or sidecar line numbers. Indexed text may include a synthetic
  breadcrumb and metadata prefix. Do not directly reuse these offsets on a sidecar.
- The reader checks the active library and exact parent identity. It never switches
  libraries, substitutes another attachment/source, searches, reranks, embeds,
  downloads, creates a collection, or rebuilds an index.
- Evidence IDs bind the chunk ID, library, complete stored text and metadata.
  Changed text/metadata or a missing anchor returns `STALE_EVIDENCE`; search again.
  An identical reindex remains valid. Neighbors are current index context, not
  a frozen historical snapshot of the original search.
- IDs are locators, **not authorization credentials**. Library and item checks are
  independent of the token. Group-library tokens require that group to be active.
- Legacy chunks missing `group_id` cannot mint a scoped ID; search explicitly
  recommends `find_in_item` instead. No automatic backfill or re-embedding occurs.

The indexed reader deliberately opens an **existing** Chroma collection with an
embedding-disabled function. It does not use `ChromaClient`'s model-initializing,
collection-creating constructor. An unavailable index returns an error rather
than attempting recovery.

## Literal lookup → source-window reading

```python
find_in_item(item_key="ITEM0001", query="Table 6", context_lines=4)
# Read the relevant source lines using the returned source_hash:
find_in_item(item_key="ITEM0001", start_line=120, end_line=145,
             expected_hash="<returned source_hash>")
# Continue inside a long HTML table line:
find_in_item(item_key="ITEM0001", start_char=2400,
             expected_hash="<returned source_hash>")
```

`find_in_item` searches an **existing local MinerU sidecar**. It does not require
Chroma, the embedder, or the reranker, and never runs OCR or downloads a source.
This is the MCP equivalent of bounded known-item `grep`/`sed`, not corpus search.

- Exact eight-character **parent** item key; no arbitrary path, title, or DOI.
- `query` is a case-insensitive **literal phrase**, at most 500 characters; regex
  syntax is escaped. An empty query is invalid; omit it or use null for reading.
- `start_line` is one-based. Optional `end_line` limits the search/read range.
  Without a query or end line, read up to 40 lines.
- `context_lines` is 0–20 (default 3). `max_matches` is 1–10 (default 5); this caps
  windows, not individual occurrences. Matches already covered by a returned
  window are not emitted again.
- `max_chars` is the **total** text budget, default 8,000, allowed 256–16,000.
  Windows may be partial; text is not repaired or replaced with summaries.
- Windows return actual source `start_line`, `end_line`, `char_start`, `char_end`,
  and flags for partial lines and truncation. A long row is centered around the
  literal match so preceding cells cannot consume the budget and hide the match.
- `next_start_line` indicates further search results/range continuation when
  available. `next_char_start` supports exact reading inside a partial line.
  Character continuation requires no query, default start_line, and no end_line.
- Pass `source_hash` back as `expected_hash` to reject a changed sidecar on a
  follow-up. The hash covers the complete decoded source text, not just a window.
- Input sidecar reads are capped at **16 MiB**. Oversized sources return
  `SOURCE_TOO_LARGE`, not a partial source passed off as complete.
- **Personal library only:** existing sidecars use `<item_key>.md`, not a
  library-namespaced path. Group-library requests fail explicitly rather than
  guessing which library owns an on-disk sidecar.

A successful literal lookup is direct extracted-source evidence, not a new
positive reranker score. Cite its line range, not a fabricated PDF page. For exact
text verification, use the PDF-page reader with a known locator; for visual/table
verification, use `render_pdf_page` with that locator.

## Locate exact PDF text and inspect visual evidence

Use the narrowest route that answers the evidence question:

1. Discover a candidate passage with `semantic_search` or another bounded lookup.
2. For a known item's PDF, call `find_in_pdf` when the exact phrase, table label, or heading needs a verified PDF-page locator.
3. Call `read_pdf_pages` for the targeted one-based PDF page when the extracted page text is the evidence needed.
4. Call `render_pdf_page` only when unresolved visual ambiguity remains—for example column alignment, a missing sign, a table layout, or a figure. It returns one actual PNG image block plus provenance; coordinates, extracted text, and generated descriptions are not image inspection.

`find_in_pdf` accepts a parent item key or PDF attachment key. It uses only the authoritative `extract_pdf` text layer: matching is literal and case-insensitive, query whitespace spans source whitespace, and no fuzzy, semantic, OCR, sidecar, or neighboring-page substitution occurs. It searches the complete requested one-based PDF range even when output is capped. Results report the extraction engine, page basis, exact total and returned counts, `has_more_matches`, and verbatim page-text windows. The coverage state is `complete`, `partial_text_coverage`, or `no_usable_text`; a no-match on an incomplete text layer is not evidence of absence.

`render_pdf_page` accepts the same key forms, a one-based PDF page, and an optional normalized `[x, y, width, height]` region compatible with `detect_pdf_regions`. It returns a FastMCP `ToolResult` containing one provenance text block, one `image/png` `ImageContent` block, and matching structured provenance. It renders in memory, enforces page/DPI/pixel/PNG bounds, rejects invalid coordinates rather than repairing them, and never changes a local Zotero attachment.

All three PDF routes are read-only. They do not create sidecars, alter chunking or indexes, run OCR, or infer printed page labels from PDF page indices. Keep the actual PDF-page locator attached to any source claim; sidecar lines, indexed offsets, and printed labels remain distinct locations.

## Errors and compatibility

Errors return `{"ok": false, "error": {"code": "...", "message": "..."}}`.
Important codes include `INVALID_ARGUMENT`, `LIBRARY_MISMATCH`, `SCOPE_MISMATCH`,
`STALE_EVIDENCE`, `SIDECAR_NOT_FOUND`, `SIDECAR_LIBRARY_UNSUPPORTED`,
`SOURCE_TOO_LARGE`, and `SOURCE_UNAVAILABLE`. No error initiates maintenance.

This change does **not** alter embeddings, chunking, the index schema, PDF parsing,
collection filters, or reranking. Existing indexes work when their provenance is
sufficient. Search display labels change from `Matched Passage` to `Preview`;
clients parsing that Markdown label must update. The search signature is unchanged.
`get_item_fulltext` and connector `fetch` retain their existing behavior.

Tests use synthetic source text and isolated/mocked stores. They cover the public
search → evidence-ID → expansion path, scope rejection, stale evidence, budgets,
continuations, literal lookup, and the no-embedding/no-creation storage route.
They do not establish improved scientific answer quality; evaluate that separately
with representative local-model research questions before changing retrieval
ranking or OCR settings.
