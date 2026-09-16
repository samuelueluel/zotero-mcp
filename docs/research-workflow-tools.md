# Bounded Research Workflow Tools

This document specifies four read-only composite tools for agentic Zotero research. They reduce repeated MCP calls while preserving the existing source-identity, scope, hash, PDF-page, and visual-verification contracts. `validate_comparison_manifest` adds comparison-specific coverage and output-scope checks; it does not decide substantive inclusion, ranking, causality, or comparability.

The tools do not decide whether a paper satisfies a substantive inclusion rule, whether an estimate is causal, or whether unlike estimates are comparable. Agents retain those decisions. The tools never repair extracted text or treat two text representations as visual verification.

## `build_candidate_scope`

### Purpose

Build one collection-scoped discovery artifact from a bounded set of agent-supplied search facets.

### Inputs

- `collection_key`: Exact eight-character collection key.
- `query_facets`: One to four nonempty semantic queries, supplied as an array or JSON-stringified array.
- `limit_per_facet`: One to twelve distinct parent items per query; default 8.
- `include_subcollections`: Include descendant collections; default true.
- `filters`: Optional semantic metadata filters, supplied as an object or JSON string.
- `inventory_limit`: Maximum inventory rows returned; 1–500, default 250. The tool still scans the complete collection scope internally.

### Deterministic behavior

1. Verify the collection before reading its items.
2. Expand the collection subtree when requested and paginate every scoped collection.
3. Deduplicate top-level parent items across collections.
4. Run only the supplied query facets. Do not start an index update or broaden scope.
5. Apply the same exact item-key scope to dense and sparse search legs.
6. Reject any search hit whose parent key is outside the verified inventory.
7. Deduplicate candidates by parent item key while retaining every facet-specific hit and evidence ID.
8. Sort candidates by maximum positive reranker score, then facet coverage, then title.
9. Return compact metadata and bounded previews. Do not classify candidates as included or excluded.

### Output

The JSON artifact contains:

- `schema_version`
- verified collection scope and member count
- inventory completeness and bounded inventory rows
- exact query facets and limits
- deduplicated candidates
- per-candidate metadata and per-facet evidence handles
- scope-leak count and warnings

A deterministic `direct_support_eligible` field may report whether a hit has positive raw rerank and is not a bibliography chunk. It is not a substantive inclusion decision.

## `collect_result_evidence`

### Purpose

Collect bounded indexed, sidecar, and PDF-text representations for exact parent items in one call. The output keeps every route separate and flags unresolved conflicts for agent review.

### Inputs

- `requests`: One to four item requests, supplied as an array or JSON-stringified array.
- Each request contains:
  - exact `item_key`
  - optional `evidence_id` from `semantic_search`
  - optional `sidecar_queries`, maximum two distinctive literal phrases or table labels
  - optional `pdf_queries`, maximum two distinctive literal phrases or table labels
  - optional one-based `pdf_start_page` and `pdf_end_page`
  - `neighbors`: 0–2 for passage expansion; default 1
- `max_chars_per_route`: 512–16000; default 8000.
- `max_pdf_windows`: One to three 50-page windows per PDF query; default 2.

At least one evidence ID or literal query is required for each item.

### Deterministic behavior

1. Verify each exact parent item in the active library.
2. Expand a supplied evidence ID before any new source lookup.
3. Run sidecar literal lookups with narrow context and a bounded output budget.
4. If a sidecar window is truncated, perform at most one continuation using the returned `source_hash` as `expected_hash`.
5. Search the authoritative PDF text using the supplied phrase and bounded 50-page windows.
6. Preserve PDF source route, extraction route, coverage, exact match accounting, and one-based page provenance.
7. Keep indexed, sidecar, and PDF representations in separate fields. Agreement is not independent corroboration.
8. Detect literal numeric-signature or direction-marker disagreements as review flags. Never choose or repair a value.
9. Set `requires_visual_review` when a decisive table may have missing signs, detached stars, conflicting numeric signatures, or ambiguous column alignment.
10. Do not render images automatically. The agent must call `render_pdf_page` for actual visual inspection.

### Output

Each item record contains:

- verified identity
- requested lookups
- route-specific success/error status
- indexed passage bundle when requested
- sidecar windows and source hash
- PDF matches, coverage, and page indices
- bounded conflict flags
- `requires_visual_review`
- referenced result tables and `REFERENCED_TABLE_NOT_READ` follow-up flags

Route success means only that candidate evidence was retrieved. The agent must decide whether it supplies the requested estimate, uncertainty, dose, denominator, or other substantive field.

## `validate_comparison_manifest`

### Purpose

Validate the result-level coverage and reporting scope of a comparison over a frozen parent-item set. A global comparison requires a terminal result card for every frozen item before ranking.

### Inputs

- `manifest`: An object or JSON-stringified object containing:
  - `frozen_item_keys`: Exact parent keys for the frozen comparison set.
  - `cards`: One result card per frozen item.
  - `ranking_rule`: Task-specific rule selected before extraction.
  - `eligible_result_policy`: `primary_only`, `substantive_all`, or `custom`.
  - `conclusion_scope`: `complete` or `verified_only`.
  - `winner_type`: `clear_winner` or `top_k`.
  - `numerical_winner_status` and `substantive_winner_status`: `clear` or `not_clear`.
  - `alternative_policy_changes_top_k`: Whether primary-only and substantive-all policies select different top sets.
  - `max_reported_items`: Maximum number of papers the final answer may analyze, from 1 to 3.
  - `selected_item_keys`: Paper-level winners or requested top-k set.
  - optional `reported_item_keys`: Items actually analyzed in a draft answer.

Each result card has `status` `eligible`, `no_eligible_result`, or `unresolved`. Eligible cards must contain result records with a `result_class` plus outcome, estimate, scale, uncertainty, treatment, dose, denominator, population, geography, horizon, specification, and evidence IDs. They also require `primary_result_id`, `maximum_substantive_result_id`, `selected_result_id`, and exact `inventory_locators`. A no-result card requires a reason and no result records.

### Deterministic checks

- Every frozen item has exactly one card, with no out-of-scope cards.
- Every eligible card has at least one result, primary and maximum-substantive IDs, a policy-consistent selected result, and inventory locators.
- Every no-result card has an exclusion reason.
- Unresolved cards block a `complete` conclusion.
- Selected items are eligible, in scope, and within the declared reporting limit.
- `clear_winner` selects exactly one item, permits only one analyzed item, and requires a clear substantive winner rather than a numerical-only leader.
- `reported_item_keys`, when supplied, are a subset of the selected items.

The tool checks manifest structure only. It does not read sources, determine whether the result inventory is substantively exhaustive, rank estimates, or decide whether unlike estimates are comparable.

### Output

Return `ready`, `complete`, a compact item-count summary, and blocking reason codes. A `verified_only` result is a qualified comparison over the items whose result cards were resolved; it is not an unqualified collection-wide maximum.

## `validate_evidence_bundle`

### Purpose

Lint draft claim-to-evidence structure before synthesis. This tool does not re-read Zotero, audit semantic support, or make causal judgments.

### Inputs

- `claims`: One to twenty bounded claim records.
- `evidence`: One to forty bounded evidence records.
- optional `allowed_item_keys`: Exact selected parent keys permitted in the final claim set.

A claim record contains:

- `claim_id`
- `text`
- `evidence_ids`
- optional `risk_tags`: `numeric`, `comparison`, `calculated`, `causal`, `attribution`
- optional structured `expected_values` for estimate, SE, CI, p-value, threshold, and sample-size fields
- optional `unverified`
- structured context fields: outcome, estimate, scale, treatment, dose, denominator, sample, geography, time horizon, uncertainty status, and calculation method

An evidence record contains:

- unique `evidence_id`
- exact `item_key`
- route: `indexed_passage`, `mineru_sidecar`, `pdf_extraction`, or `pdf_rendering`
- actual locator and optional one-based PDF page
- bounded quote
- optional content/source hash
- ambiguity flags

### Deterministic checks

- Every evidence reference resolves to a supplied evidence record.
- Every claim has at least one source item; comparisons have at least two distinct items.
- When `allowed_item_keys` is supplied, no claim links evidence from an unselected item.
- Numeric claims supply outcome, estimate, scale, treatment, and uncertainty status.
- Numeric signatures in the claim occur in at least one linked quote when quotes are supplied.
- Calculated claims include a calculation method.
- PDF routes do not derive page provenance from sidecar or passage locators.
- Claims do not present evidence marked ambiguous as resolved unless the claim is explicitly `unverified`.
- Evidence IDs and item keys are unique and correctly formatted.

### Output

Return one status per claim, stable reason codes, aggregate `ready`, and bounded warnings. Passing means only that the structural contract is complete. It does not establish substantive support and must never be cited as evidence.

## Compatibility and safety

- Raw tool names omit the `zotero_` prefix; Pi adds the server namespace.
- Existing primitive tools remain public and unchanged.
- The new tools belong to a `research-workflows` toolset enabled in the default profile.
- All inputs and outputs are bounded; invalid bounds fail rather than clamp silently.
- Tests must cover scope leaks, pagination, deduplication, stale hashes, incomplete PDF text, temporary-file cleanup, route separation, ambiguity flags, and tool-name registration.
