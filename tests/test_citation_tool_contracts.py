"""Focused public-contract tests for citation graph tools and resources."""

from __future__ import annotations

import asyncio

import pytest

from zotero_mcp.server import mcp
from zotero_mcp.tools import discovery


@pytest.mark.parametrize("depth", [0, 2, -1])
def test_citation_neighbors_rejects_unsupported_depth_before_loading_graph(
    monkeypatch, depth
):
    """The retained compatibility parameter must never silently ignore other depths."""

    def fail_if_called():
        raise AssertionError("graph must not load for an unsupported depth")

    monkeypatch.setattr(discovery, "_get_graph", fail_if_called)

    result = discovery.get_citation_neighbors(item_key="ABCDEFGH", depth=depth)

    assert result.startswith("Error: get_citation_neighbors supports only depth=1")
    assert "multi-hop traversal is not implemented" in result


def test_citation_neighbors_accepts_omitted_depth_and_forwards_one(monkeypatch):
    """The default call remains a direct-neighbor query with depth=1."""

    class FakeGraph:
        def get_citation_neighbors(self, item_key, *, depth, scope, collection_key):
            assert item_key == "ABCDEFGH"
            assert depth == 1
            assert scope == "library"
            assert collection_key == ""
            return {
                "target_paper": {
                    "title": "Seed",
                    "year": "2020",
                    "creators": "Author",
                    "node_type": "zotero_item",
                },
                "cites": [],
                "cited_by": [],
                "scope": scope,
                "depth": depth,
            }

    monkeypatch.setattr(discovery, "_get_graph", lambda: FakeGraph())

    result = discovery.get_citation_neighbors(item_key="ABCDEFGH")

    assert result.startswith("# Direct Citation Neighbors for **Seed**")


def test_static_collections_resource_is_not_registered():
    """Collection discovery has one canonical surface: list_collections."""
    resources = asyncio.run(mcp.list_resources())
    assert all(str(resource.uri) != "zotero://collections" for resource in resources)


def test_parameterized_read_resources_remain_registered():
    """Removing the duplicate listing must not remove item/context resources."""
    templates = asyncio.run(mcp.list_resource_templates())
    uris = {template.uri_template for template in templates}
    assert uris == {
        "zotero://items/{item_key}",
        "zotero://collections/{collection_key}/items",
    }
