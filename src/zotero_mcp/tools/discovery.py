"""Discovery tools: find related papers via OpenAlex and assess library coverage."""

import re
from typing import Literal

import requests

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils  # noqa: F401  (kept for module-level conventions)
from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.tools import _helpers

_OPENALEX_BASE = "https://api.openalex.org"
_MAILTO = "zotero-mcp@users.noreply.github.com"
_ITEM_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")
_HTTP_TIMEOUT = 30


def _doi_in_library(zot, doi: str) -> bool:
    """Best-effort membership check: is a paper with this DOI already in Zotero?

    Tolerates any pyzotero error by returning False (treat as "not in library").
    """
    if not doi:
        return False
    try:
        zot.add_parameters(q=doi, qmode="everything", itemType="-attachment", limit=5)
        results = zot.items()
    except Exception:
        try:
            results = zot.items(q=doi, qmode="everything", itemType="-attachment", limit=5)
        except Exception:
            return False
    norm = doi.strip().lower()
    for item in results or []:
        item_doi = str(item.get("data", {}).get("DOI", "")).strip().lower()
        if item_doi and item_doi == norm:
            return True
    return False


def _short_id(openalex_id: str) -> str:
    """Reduce a full OpenAlex URL/ID to its bare 'Wxxxx' form."""
    if not openalex_id:
        return ""
    return openalex_id.rstrip("/").rsplit("/", 1)[-1]


def _work_summary(work: dict) -> dict:
    """Extract a compact summary from an OpenAlex work object."""
    title = work.get("title") or work.get("display_name") or "Untitled"
    year = work.get("publication_year")
    doi = _helpers._normalize_doi(work.get("doi")) or ""
    cited_by = work.get("cited_by_count", 0) or 0

    authors = []
    for authorship in (work.get("authorships") or [])[:3]:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)
    if len(work.get("authorships") or []) > 3:
        authors.append("et al.")

    return {
        "title": title,
        "year": year,
        "doi": doi,
        "cited_by": cited_by,
        "authors": authors,
    }


def _openalex_get(url: str, params: dict | None = None) -> dict | None:
    """GET an OpenAlex endpoint, returning parsed JSON or None on any failure."""
    p = {"mailto": _MAILTO}
    if params:
        p.update(params)
    try:
        resp = requests.get(url, params=p, timeout=_HTTP_TIMEOUT)
    except Exception:
        return None
    if getattr(resp, "status_code", None) != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def _resolve_doi(identifier: str, zot) -> str | None:
    """Resolve an identifier (Zotero item key or DOI/URL) to a normalized DOI."""
    ident = (identifier or "").strip()
    if not ident:
        return None
    if _ITEM_KEY_RE.match(ident):
        try:
            item = zot.item(ident)
        except Exception:
            return None
        raw_doi = (item or {}).get("data", {}).get("DOI")
        return _helpers._normalize_doi(raw_doi)
    return _helpers._normalize_doi(ident)


def _render_related(papers: list[dict], heading: str) -> list[str]:
    """Render a list of related-paper summaries as markdown lines."""
    lines = [f"## {heading} ({len(papers)})", ""]
    if not papers:
        lines.append("_None found._")
        lines.append("")
        return lines
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"]) if p["authors"] else "Unknown authors"
        year = p["year"] if p["year"] else "n.d."
        marker = "in library ✓" if p.get("in_library") else "not in library"
        lines.append(f"{i}. **{p['title']}** ({year})")
        lines.append(f"   - Authors: {authors}")
        if p["doi"]:
            lines.append(f"   - DOI: {p['doi']}")
        lines.append(f"   - Cited by: {p['cited_by']}")
        lines.append(f"   - {marker}")
        lines.append("")
    return lines


@mcp.tool(
    name="discover_citing_and_referenced_works",
    description=(
        "Discover papers related to a known work by following its citation "
        "graph via OpenAlex (a free scholarly index). Use this to expand a "
        "literature review: find what a paper CITES (its references) and what "
        "CITES it (newer follow-up work). Each related paper is flagged as "
        "already in your Zotero library or not, so you can quickly spot gaps "
        "to fetch (e.g. via add_item). "
        "identifier: either an 8-char Zotero item key (its DOI is looked up) "
        "or a DOI / DOI-URL directly (e.g. '10.1038/nature12373' or "
        "'https://doi.org/10.1038/nature12373'). "
        "direction: 'references' (works this paper cites), 'citations' (works "
        "citing this paper), or 'both' (default). "
        "limit: max related papers per direction (default 20, max 50). "
        "Citations are sorted by citation count (most-cited first); references "
        "keep their original order. Requires the work to have a resolvable "
        "DOI present in OpenAlex. "
        "Example: discover_citing_and_referenced_works(identifier='10.1038/nature12373', "
        "direction='citations', limit=10)."
    ),
)
@with_zotero_api_lock
def discover_citing_and_referenced_works(
    identifier: str,
    direction: Literal["references", "citations", "both"] = "both",
    limit: int | str | None = 20,
    *,
    ctx: Context,
) -> str:
    """Find references and/or citing works for a paper via OpenAlex."""
    try:
        if direction not in {"references", "citations", "both"}:
            return "Error: direction must be 'references', 'citations', or 'both'."

        limit = _helpers._normalize_limit(limit, default=20, max_val=50)
        zot = _client.get_zotero_client()

        ctx.info(f"Resolving identifier to DOI: {identifier}")
        doi = _resolve_doi(identifier, zot)
        if not doi:
            return (
                f"Could not resolve a DOI for '{identifier}'. Provide a valid "
                "DOI / DOI-URL, or an 8-char Zotero item key whose item has a "
                "DOI in its metadata."
            )

        ctx.info(f"Querying OpenAlex for DOI {doi}")
        work = _openalex_get(f"{_OPENALEX_BASE}/works/https://doi.org/{doi}")
        if not work:
            return f"OpenAlex has no record for DOI '{doi}', or the lookup failed."

        want_refs = direction in {"references", "both"}
        want_cites = direction in {"citations", "both"}

        references: list[dict] = []
        citations: list[dict] = []

        if want_refs:
            ref_ids = [_short_id(r) for r in (work.get("referenced_works") or [])]
            ref_ids = [r for r in ref_ids if r][:limit]
            if ref_ids:
                filter_val = "openalex_id:" + "|".join(ref_ids)
                data = _openalex_get(
                    f"{_OPENALEX_BASE}/works",
                    {"filter": filter_val, "per-page": min(len(ref_ids), 50)},
                )
                results = (data or {}).get("results", []) or []
                # Preserve referenced_works order.
                by_id = {_short_id(w.get("id")): w for w in results}
                for rid in ref_ids:
                    if rid in by_id:
                        references.append(_work_summary(by_id[rid]))

        if want_cites:
            cited_by_url = work.get("cited_by_api_url")
            if cited_by_url:
                data = _openalex_get(cited_by_url, {"per-page": min(limit, 50)})
                results = (data or {}).get("results", []) or []
                citations = [_work_summary(w) for w in results]
                citations.sort(key=lambda p: p["cited_by"], reverse=True)
                citations = citations[:limit]

        # Flag library membership for every related paper.
        for p in references + citations:
            p["in_library"] = _doi_in_library(zot, p["doi"]) if p["doi"] else False

        src_title = work.get("title") or work.get("display_name") or doi
        output = [
            f"# Related Papers for: {src_title}",
            "",
            f"Source DOI: {doi}",
        ]
        summary_bits = []
        if want_refs:
            summary_bits.append(f"{len(references)} references")
        if want_cites:
            summary_bits.append(f"{len(citations)} citations")
        in_lib = sum(1 for p in references + citations if p.get("in_library"))
        summary_bits.append(f"{in_lib} already in library")
        output.append("Found " + ", ".join(summary_bits) + ".")
        output.append("")

        if want_refs:
            output.extend(_render_related(references, "References (works this paper cites)"))
        if want_cites:
            output.extend(_render_related(citations, "Citations (works citing this paper)"))

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error finding related papers: {e}")
        return f"Error finding related papers: {e}"


def _item_has_pdf(zot, item: dict) -> bool:
    """Return True if the item is/has a PDF attachment.

    A standalone PDF attachment counts directly; otherwise we inspect the
    item's children for any PDF attachment. Tolerant of children() errors.
    """
    data = item.get("data", {})
    if data.get("itemType") == "attachment" and data.get("contentType") == "application/pdf":
        return True
    key = item.get("key") or data.get("key")
    if not key:
        return False
    try:
        children = _helpers._paginate(zot.children, key)
    except Exception:
        return False
    for child in children or []:
        cdata = child.get("data", {})
        if cdata.get("itemType") == "attachment" and cdata.get("contentType") == "application/pdf":
            return True
    return False


@mcp.tool(
    name="audit_pdf_coverage",
    description=(
        "Audit PDF coverage across your Zotero library (or one collection): "
        "which items have a downloaded PDF attachment and which are missing "
        "one. Use this to find papers you can still fetch full text for — the "
        "missing list includes each item's DOI so you can pass it to "
        "add_item's open-access download cascade. "
        "collection_key: optional 8-char key to scope the audit to one "
        "collection; omit to scan the whole library. "
        "limit: max top-level items to scan (default 200). "
        "Attachments, notes, and annotations are skipped as scan targets; an "
        "item counts as covered if it (or any child) is a PDF attachment. "
        "Reports total scanned, covered count, missing count, coverage "
        "percentage, and a capped list (first 50) of missing items with "
        "title, year, key, and DOI. "
        "Example: audit_pdf_coverage(collection_key='ABCD1234', "
        "limit=100)."
    ),
)
@with_zotero_api_lock
def audit_pdf_coverage(
    collection_key: str | None = None,
    limit: int | str | None = 200,
    *,
    ctx: Context,
) -> str:
    """Report PDF-attachment coverage across the library or a collection."""
    try:
        limit = _helpers._normalize_limit(limit, default=200, max_val=2000)
        zot = _client.get_zotero_client()

        skip_types = {"attachment", "note", "annotation"}

        ctx.info("Scanning library for PDF coverage...")
        if collection_key:
            items = _helpers._paginate(
                zot.collection_items,
                collection_key,
                max_items=limit,
                itemType="-attachment",
            )
        else:
            items = _helpers._paginate(
                zot.items,
                max_items=limit,
                itemType="-attachment",
            )

        scanned = 0
        with_pdf = 0
        missing: list[dict] = []

        for item in items or []:
            data = item.get("data", {})
            item_type = data.get("itemType")
            # A standalone PDF attachment is a valid scan target even though
            # we excluded attachments at the API level (defensive).
            is_standalone_pdf = item_type == "attachment" and data.get("contentType") == "application/pdf"
            if item_type in skip_types and not is_standalone_pdf:
                continue

            scanned += 1
            if _item_has_pdf(zot, item):
                with_pdf += 1
            else:
                title = data.get("title") or data.get("filename") or "Untitled"
                year = str(data.get("date", ""))[:4]
                missing.append(
                    {
                        "title": title,
                        "year": year,
                        "key": item.get("key", ""),
                        "doi": data.get("DOI", ""),
                    }
                )

        missing_count = len(missing)
        pct = (with_pdf / scanned * 100) if scanned else 0.0

        scope = f"collection {collection_key}" if collection_key else "entire library"
        output = [
            f"# PDF Coverage Report ({scope})",
            "",
            f"- Items scanned: {scanned}",
            f"- With PDF: {with_pdf}",
            f"- Missing PDF: {missing_count}",
            f"- Coverage: {pct:.1f}%",
            "",
        ]

        if missing:
            shown = missing[:50]
            output.append(f"## Items Missing a PDF (showing {len(shown)} of {missing_count})")
            output.append("")
            for i, m in enumerate(shown, 1):
                year = m["year"] if m["year"] else "n.d."
                line = f"{i}. **{m['title']}** ({year}) — key: {m['key']}"
                if m["doi"]:
                    line += f" — DOI: {m['doi']}"
                output.append(line)
            output.append("")
        else:
            output.append("All scanned items have a PDF attachment. ✓")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error computing library coverage: {e}")
        return f"Error computing library coverage: {e}"


# [graph patch] Deterministic Citation Graph tools
from zotero_mcp.citation_graph import CitationGraph

_GRAPH_INSTANCE = None


def _get_graph() -> CitationGraph:
    global _GRAPH_INSTANCE
    if _GRAPH_INSTANCE is None:
        _GRAPH_INSTANCE = CitationGraph()
        _GRAPH_INSTANCE.load()
    return _GRAPH_INSTANCE


def _node_marker(node: dict) -> str:
    if node.get("node_type") != "external_reference":
        return ""
    ext_id = node.get("external_id", "")
    if ext_id.startswith("meta:"):
        return " [external ref: metadata]"
    return " [external reference]"


def _scope_label(scope: str, collection_key: str = "") -> str:
    if collection_key:
        return f"{scope}; collection={collection_key}"
    return scope


@mcp.tool()
def rebuild_citation_graph(ctx: Context = None) -> str:
    """Rebuild the local citation graph from Zotero metadata and MinerU sidecars.

    This is graph-only and does not re-embed ChromaDB or rebuild the semantic
    search database. It also refreshes the process-local graph used by the
    other citation tools.
    """
    global _GRAPH_INSTANCE
    try:
        _GRAPH_INSTANCE = CitationGraph()
        stats = _GRAPH_INSTANCE.build()
        try:
            from zotero_mcp.reference_index import invalidate_reference_index_cache
            invalidate_reference_index_cache()
        except Exception:
            pass
        return (
            "# Citation Graph Rebuilt\n\n"
            f"- Library nodes: **{stats.get('library_nodes', 0)}**\n"
            f"- External-reference nodes: **{stats.get('external_nodes', 0)}**\n"
            f"- Citation edges: **{stats.get('directed_citations', 0)}**\n"
            f"- Resolved citation edges: **{stats.get('resolved_citations', 0)}**\n"
            f"- External citation edges: **{stats.get('external_citations', 0)}**\n"
            f"- Reference evidence records: **{stats.get('reference_evidence', 0)}**\n"
            f"- Parsed reference sections: **{stats.get('reference_sections', 0)}**\n"
            f"- Parsed reference entries: **{stats.get('reference_entries', 0)}**\n"
            f"- DOI-bearing entries: **{stats.get('reference_entries_with_doi', 0)}**\n"
            f"- Resolved entries: **{stats.get('resolved_reference_entries', 0)}**\n"
            f"- External DOI entries: **{stats.get('external_reference_entries', 0)}**\n"
            f"- Metadata-derived external entries: **{stats.get('metadata_external_reference_entries', 0)}**\n"
            f"- Ambiguous entries: **{stats.get('ambiguous_reference_entries', 0)}**\n"
            f"- Unresolved entries: **{stats.get('unresolved_reference_entries', 0)}**\n"
            f"- Orphan sidecars: **{stats.get('orphan_reference_sidecars', 0)}** "
            f"({stats.get('orphan_reference_entries', 0)} entries)\n"
            f"- Database: `{stats.get('db_path', '')}`"
        )
    except Exception as e:
        return f"Error rebuilding citation graph: {e}"


@mcp.tool()
def rank_works_by_inbound_citations(
    collection_key: str = "",
    top_n: int = 5,
    scope: str = "library",
    ctx: Context = None,
) -> str:
    """Rank works by resolved inbound citation edges within an explicit scope.

    This is a simple citation-subgraph in-degree ranking, not a network hub,
    HITS, or centrality measure. Scopes are ``collection``, ``library``
    (legacy default), ``collection-expanded``, and ``library-expanded``.
    Expanded scopes may return external-reference nodes recovered from
    sidecar bibliographies.

    ``inward_citations`` counts RESOLVED inbound edges only: bibliography
    entries that never resolved to a graph node (``unresolved`` — often the
    majority in economics sidecars) are invisible to it, so it is a lower
    bound. The output carries a per-scope resolution-coverage line; use
    ``search_bibliography_entries`` to count true citation occurrences.

    Args:
        collection_key: Collection key; required for collection scopes.
        top_n: Number of hub nodes to return (default: 5).
        scope: Graph scope (default: library).
    """
    try:
        g = _get_graph()
        ranked = g.rank_works_by_inbound_citations(collection_key, top_n=top_n, scope=scope)
        if not ranked:
            return f"No cited works found for {_scope_label(scope, collection_key)}."

        lines = [f"# Works Ranked by Inbound Citations ({_scope_label(scope, collection_key)})\n"]
        cov = ranked[0].get("resolution_coverage") if ranked else None
        if cov:
            entries = sum(cov.values())
            lines.append(
                f"*Resolution coverage (citing sidecars in scope): {entries} entries — "
                + ", ".join(f"{k}: {v}" for k, v in sorted(cov.items()))
            )
            lines.append(
                "*Counts are graph inbound edges only; unresolved/ambiguous entries are not "
                "counted. `ext:meta` counts are approximate — use `search_bibliography_entries` "
                "for exact totals.\n"
            )
        for i, work in enumerate(ranked, 1):
            yr = f" ({work['year']})" if work['year'] else ""
            au = f" — *{work['creators']}*" if work['creators'] else ""
            marker = _node_marker(work)
            lines.append(f"{i}. **{work['title']}**{yr}{au}{marker}")
            lines.append(
                f"   - Key: `{work['item_key']}` | Inward citations: **{work['inward_citations']}** (graph edges only)"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error ranking works by inbound citations: {e}"


@mcp.tool()
def get_citation_neighbors(
    item_key: str,
    depth: int = 1,
    scope: str = "library",
    collection_key: str = "",
    ctx: Context = None,
) -> str:
    """Return a work's direct cited and citing neighbors under a graph scope.

    This currently returns one-hop neighbors only. Expanded scopes can expose
    external-reference nodes. Such nodes represent bibliography evidence only;
    their own outgoing references require their full text.

    Args:
        item_key: Zotero item key or external-reference graph key.
        depth: Traversal depth (direct neighbors are currently returned).
        scope: Graph scope (default: library).
        collection_key: Required for collection scopes.
    """
    try:
        g = _get_graph()
        data = g.get_citation_neighbors(
            item_key,
            depth=depth,
            scope=scope,
            collection_key=collection_key,
        )
        if "error" in data:
            return f"Error: {data['error']}"

        target = data["target_paper"]
        t_yr = f" ({target['year']})" if target['year'] else ""
        t_au = f" — *{target['creators']}*" if target['creators'] else ""
        marker = _node_marker(target)
        lines = [
            f"# Direct Citation Neighbors for **{target['title']}**{t_yr}{t_au}{marker} (`{item_key}`)",
            f"Scope: `{_scope_label(data.get('scope', scope), collection_key)}`\n",
        ]

        lines.append(f"## Papers Cited ({len(data['cites'])})")
        if data["cites"]:
            for i, cited in enumerate(data["cites"], 1):
                yr = f" ({cited['year']})" if cited['year'] else ""
                au = f" — *{cited['creators']}*" if cited['creators'] else ""
                lines.append(
                    f"{i}. **{cited['title']}**{yr}{au}{_node_marker(cited)} (`{cited['item_key']}`)"
                )
        else:
            lines.append("  (No cited nodes found in this scope)")

        lines.append(f"\n## Papers Citing This ({len(data['cited_by'])})")
        if data["cited_by"]:
            for i, citer in enumerate(data["cited_by"], 1):
                yr = f" ({citer['year']})" if citer['year'] else ""
                au = f" — *{citer['creators']}*" if citer['creators'] else ""
                lines.append(
                    f"{i}. **{citer['title']}**{yr}{au}{_node_marker(citer)} (`{citer['item_key']}`)"
                )
        else:
            lines.append("  (No citing nodes found in this scope)")

        return "\n".join(lines)
    except Exception as e:
        return f"Error retrieving citation neighbors: {e}"


@mcp.tool()
def find_bibliographically_coupled_papers(
    item_key: str,
    top_n: int = 5,
    scope: str = "library",
    collection_key: str = "",
    ctx: Context = None,
) -> str:
    """Find resolved papers connected by shared citations under a graph scope.

    In expanded scopes, external-reference nodes may participate as shared
    citation targets, while result papers remain resolved Zotero items.

    Args:
        item_key: Zotero item key of the target paper.
        top_n: Number of connected papers to return (default: 5).
        scope: Graph scope (default: library).
        collection_key: Required for collection scopes.
    """
    try:
        g = _get_graph()
        connected = g.find_bibliographically_coupled_papers(
            item_key,
            top_n=top_n,
            scope=scope,
            collection_key=collection_key,
        )
        if not connected:
            return f"No connected papers found for `{item_key}` in {_scope_label(scope, collection_key)}."

        lines = [
            f"# Structurally Connected Papers for `{item_key}` (Bibliographic Coupling)",
            f"Scope: `{_scope_label(scope, collection_key)}`\n",
        ]
        for i, paper in enumerate(connected, 1):
            yr = f" ({paper['year']})" if paper['year'] else ""
            au = f" — *{paper['creators']}*" if paper['creators'] else ""
            lines.append(f"{i}. **{paper['title']}**{yr}{au} (`{paper['item_key']}`)")
            lines.append(
                f"   - Coupling Jaccard Score: **{paper['coupling_score']}** "
                f"({paper['shared_citations_count']} shared citations)"
            )
            if paper["shared_citations"]:
                lines.append(f"   - Key Shared Citations: {'; '.join(paper['shared_citations'])}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error finding connected papers: {e}"


# [reference patch] bibliography/reference-only BM25 retrieval
from zotero_mcp.reference_index import (
    audit_reference_index,
    build_reference_index,
    search_reference_index,
)


def _reference_scope_label(collection_key: str, item_key: str) -> str:
    if collection_key:
        return f"collection={collection_key}"
    if item_key:
        return f"item={item_key}"
    return "library"


def _reference_marker(result: dict) -> str:
    return "[reference entry]"


@mcp.tool()
def rebuild_reference_index(ctx: Context = None) -> str:
    """Build the separate BM25 index over individual local bibliography entries.

    This parses MinerU sidecars and joins the graph's per-entry audit data. It
    performs no embedding and does not modify ChromaDB content.
    """
    try:
        stats = build_reference_index()
        status = stats.get("status_counts", {})
        return (
            "# Reference Index Rebuilt\n\n"
            f"- Reference sections: **{stats.get('reference_sections', 0)}**\n"
            f"- Reference entries: **{stats.get('entries', 0)}**\n"
            f"- Source items: **{stats.get('source_items', 0)}** "
            f"(library: **{stats.get('library_source_items', 0)}**; "
            f"orphan sidecar: **{stats.get('orphan_source_items', 0)}**)\n"
            f"- Source sidecars: **{stats.get('source_sidecars', 0)}** "
            f"(orphan: **{stats.get('orphan_source_sidecars', 0)}**)\n"
            f"- Entries with DOI: **{stats.get('doi_entries', 0)}**\n"
            f"- Resolved to Zotero: **{stats.get('resolved_entries', 0)}**\n"
            f"- External DOI entries: **{stats.get('external_doi_entries', 0)}**\n"
            f"- Ambiguous: **{stats.get('ambiguous_entries', 0)}**\n"
            f"- Unresolved: **{stats.get('unresolved_entries', 0)}**\n"
            f"- Orphan-source entries: **{stats.get('orphan_source_entries', 0)}**\n"
            f"- Status counts: `{status}`\n"
            f"- BM25 documents: **{stats.get('docs', 0)}**\n"
            f"- Terms: **{stats.get('terms', 0)}**\n"
            f"- Index: `{stats.get('path', '')}`\n"
            f"- Metadata: `{stats.get('metadata_path', '')}`\n"
            f"- Built: `{stats.get('built_at', '')}`"
        )
    except Exception as e:
        return f"Error rebuilding reference index: {e}"


@mcp.tool()
def get_reference_index_status(ctx: Context = None) -> str:
    """Report parsed bibliography coverage and resolution status."""
    try:
        stats = audit_reference_index()
        return (
            "# Bibliography Reference Audit\n\n"
            f"- Reference sections: **{stats.get('reference_sections', 0)}**\n"
            f"- Reference entries: **{stats.get('entries', 0)}**\n"
            f"- Source items: **{stats.get('source_items', 0)}** "
            f"(library: **{stats.get('library_source_items', 0)}**; "
            f"orphan sidecar: **{stats.get('orphan_source_items', 0)}**)\n"
            f"- Source sidecars: **{stats.get('source_sidecars', 0)}** "
            f"(orphan: **{stats.get('orphan_source_sidecars', 0)}**)\n"
            f"- Entries with DOI: **{stats.get('doi_entries', 0)}**\n"
            f"- Resolved to Zotero: **{stats.get('resolved_entries', 0)}**\n"
            f"- External DOI entries: **{stats.get('external_doi_entries', 0)}**\n"
            f"- Mixed entries: **{stats.get('mixed_entries', 0)}**\n"
            f"- Ambiguous: **{stats.get('ambiguous_entries', 0)}**\n"
            f"- Unresolved: **{stats.get('unresolved_entries', 0)}**\n"
            f"- Orphan-source entries: **{stats.get('orphan_source_entries', 0)}**\n"
            f"- Source item types: `{stats.get('source_item_types', {})}`\n"
            f"- Split methods: `{stats.get('split_methods', {})}`\n"
            f"- Status counts: `{stats.get('status_counts', {})}`\n"
            f"- Built: `{stats.get('built_at', '')}`"
        )
    except Exception as e:
        return f"Error auditing bibliography references: {e}"


@mcp.tool()
def search_bibliography_entries(
    query: str,
    limit: int = 10,
    collection_key: str = "",
    item_key: str = "",
    ctx: Context = None,
) -> str:
    """Search individual bibliography entries separately from content RAG.

    Uses BM25 over parsed local MinerU sidecar entries. Results include
    bibliographic metadata and graph-resolution status, not evidence of the
    cited paper's substantive findings.

    Args:
        query: Author, title, year, DOI, or other reference text to search.
        limit: Maximum number of reference entries to return (default: 10).
        collection_key: Optional source collection key to restrict the search.
        item_key: Optional citing Zotero item key to restrict the search.
    """
    try:
        limit = max(1, min(int(limit), 100))
        hits = search_reference_index(
            query,
            top_n=limit,
            collection_key=collection_key or None,
            item_key=item_key or None,
        )
        if not hits:
            return (
                f"No reference entries matched `{query}` in "
                f"{_reference_scope_label(collection_key, item_key)}."
            )

        lines = [
            f"# Reference Search: `{query}`",
            f"Scope: `{_reference_scope_label(collection_key, item_key)}`\n",
        ]
        for number, hit in enumerate(hits, 1):
            source_title = hit.get("source_title") or hit.get("source_key") or "Untitled source"
            lines.append(f"{number}. **{source_title}** {_reference_marker(hit)}")
            lines.append(
                f"   - Citing item: `{hit.get('source_key') or hit.get('citing_item_key', '')}`"
                f" | Entry: `{hit.get('entry_index', '')}` | BM25: **{hit.get('score', 0)}**"
            )
            lines.append(
                f"   - Source status: `{hit.get('source_status', 'unknown')}`"
                f" | Item type: `{hit.get('source_item_type', 'unknown')}`"
            )
            if hit.get("source_sidecar"):
                lines.append(f"   - Source sidecar: `{hit['source_sidecar']}`")
            if hit.get("section_heading"):
                lines.append(
                    f"   - Section: `{hit['section_heading']}`"
                    f" (#{hit.get('section_index', '')}); split: `{hit.get('split_method', '')}`"
                )
            if hit.get("dois"):
                lines.append(f"   - DOI(s): `{', '.join(hit['dois'])}`")
            lines.append(
                f"   - Resolution: `{hit.get('target_status', 'unresolved')}`"
                f" via `{hit.get('match_method', 'unresolved')}`"
                f" (confidence: **{hit.get('confidence', 0.0)}**;"
                f" parse: **{hit.get('parse_confidence', 0.0)}**)")
            if hit.get("target_keys"):
                lines.append(f"   - Target key(s): `{', '.join(hit['target_keys'])}`")
            if hit.get("target_types"):
                lines.append(f"   - Target type(s): `{', '.join(hit['target_types'])}`")
            if hit.get("collections"):
                lines.append(f"   - Collections: `{', '.join(hit['collections'])}`")
            lines.append("   - Raw reference:")
            lines.append(f"     > {hit.get('raw_reference', '')}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error searching bibliography references: {e}"
