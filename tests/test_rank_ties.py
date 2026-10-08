"""rank_works_by_inbound_citations reports works tied with the last one shown."""

from __future__ import annotations

import networkx as nx

from zotero_mcp.citation_graph import CitationGraph


def _graph(in_degrees: dict[str, int]) -> CitationGraph:
    g = CitationGraph.__new__(CitationGraph)
    graph = nx.DiGraph()
    for target, n in in_degrees.items():
        graph.add_node(target, collections=["COLL"])
        for i in range(n):
            src = f"S{target}{i}"
            graph.add_node(src, collections=["COLL"])
            graph.add_edge(src, target)
    g.graph = graph
    g._loaded = True
    g._audit_coverage = lambda keys: {}
    g._citation_subgraph = lambda scope, key="": graph
    return g


def test_ties_past_cutoff_are_counted():
    ranked = _graph({"A": 4, "B": 2, "C": 2, "D": 2}).rank_works_by_inbound_citations(
        "COLL", top_n=2, scope="collection"
    )
    assert [r["item_key"] for r in ranked] == ["A", "B"]
    assert ranked[-1]["tied_beyond_top_n"] == 2


def test_no_tie_reported_at_clean_cutoff():
    ranked = _graph({"A": 4, "B": 3, "C": 1}).rank_works_by_inbound_citations(
        "COLL", top_n=2, scope="collection"
    )
    assert ranked[-1]["tied_beyond_top_n"] == 0
