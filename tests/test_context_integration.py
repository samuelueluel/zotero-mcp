"""Exercise the public discovery/expansion boundary against hermetic stores."""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import DummyContext

from zotero_mcp.passage_context import evidence_id, parse_evidence_id, text_hash
from zotero_mcp.tools import context as context_tool
from zotero_mcp.tools import search as search_tool

KEY = "ITEM0001"


def test_discovery_id_expands_exact_stored_chunk(monkeypatch, tmp_path):
    from zotero_mcp import semantic_search as semantic
    from zotero_mcp.semantic_search import ZoteroSemanticSearch

    text = (
        "[Paper: A paper | Section: Results]\n"
        + "Background sentence. " * 50
        + "\nThe treatment reduced crime by 11%. Notes: follow-up was 14 months.\n"
    )
    meta = {"group_id": 0, "item_key": KEY, "parent_item_key": KEY, "chunk_index": 1, "n_chunks": 3}
    raw = {
        "ids": [[f"{KEY}#1", f"{KEY}#2"]],
        "documents": [[text, "Another relevant passage"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[meta, {**meta, "chunk_index": 2}]],
        "rerank_scores": [[4.2, 3.8]],
    }
    item = {"key": KEY, "data": {"title": "A paper", "itemType": "journalArticle", "creators": []}}

    class Client:
        def item(self, key):
            assert key == KEY
            return item

    sem = ZoteroSemanticSearch.__new__(ZoteroSemanticSearch)
    sem._attach_zotero_items = lambda results: [r.update(zotero_item=item) for r in results]
    enriched = sem._enrich_search_results(raw, "crime treatment", limit=5)
    assert len(enriched) == 1  # retain document diversity in discovery
    hit = enriched[0]
    assert hit["preview_truncated"]
    assert "11%" in hit["matched_passage"]
    assert hit["matched_text"] == text
    token = hit["evidence_id"]
    assert parse_evidence_id(token)[1] == f"{KEY}#1"

    class Search:
        def search(self, **kwargs):
            assert kwargs["collection_key"] == "COLL0001"
            assert kwargs["filters"] == {"item_keys": [KEY]}
            return {"results": enriched}

    monkeypatch.setattr(semantic, "create_semantic_search", lambda _: Search())
    monkeypatch.setattr(search_tool, "_maybe_fire_presearch_sync", lambda _: None)
    monkeypatch.setattr(search_tool._client, "get_active_group_id", lambda: 0)
    monkeypatch.setattr(search_tool._client, "get_zotero_client", lambda: Client())
    displayed = search_tool.semantic_search(
        query="crime treatment", collection="COLL0001", filters={"item_keys": [KEY]}, ctx=DummyContext()
    )
    assert "**Preview:**" in displayed and "**Matched Passage:**" not in displayed
    assert "**Preview truncated:** yes" in displayed
    assert token in displayed and text_hash(text) in displayed
    assert displayed.count(token) == 1
    monkeypatch.setattr(
        context_tool, "_index_documents", lambda ids: {"ids": [ids[0]], "documents": [text], "metadatas": [meta]}
    )
    expanded = json.loads(context_tool.read_passage(token, ctx=DummyContext()))
    assert expanded["ok"] and expanded["chunks"][0]["text"] == text


def test_legacy_preview_has_explicit_fallback_without_unscoped_id():
    assert evidence_id(f"{KEY}#0", "text", {"item_key": KEY}) is None


def test_index_read_does_not_create_missing_database(monkeypatch, tmp_path):
    from zotero_mcp import chroma_client

    monkeypatch.setattr(chroma_client.Path, "home", lambda: tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("No client/model initialization for a missing index")

    monkeypatch.setattr(chroma_client.chromadb, "PersistentClient", forbidden)
    with pytest.raises(FileNotFoundError):
        chroma_client.read_index_documents([f"{KEY}#0"])
    assert not (tmp_path / ".config").exists()


def test_existing_index_read_never_embeds_or_creates_collection(monkeypatch, tmp_path):
    from zotero_mcp import chroma_client

    monkeypatch.setattr(chroma_client.Path, "home", lambda: tmp_path)
    db = tmp_path / ".config/zotero-mcp/chroma_db/chroma.sqlite3"
    db.parent.mkdir(parents=True)
    db.touch()
    calls = []

    class Collection:
        def get(self, **kwargs):
            calls.append(kwargs)
            return {"ids": [f"{KEY}#0"], "documents": ["stored text"], "metadatas": [{"group_id": 0}]}

    class Client:
        def get_collection(self, *, name, embedding_function):
            assert name == "zotero_library"
            with pytest.raises(RuntimeError):
                embedding_function(["must never embed"])
            return Collection()

    monkeypatch.setattr(chroma_client.chromadb, "PersistentClient", lambda **kwargs: Client())
    result = chroma_client.read_index_documents([f"{KEY}#0"])
    assert result["documents"] == ["stored text"]
    assert calls == [{"ids": [f"{KEY}#0"], "include": ["documents", "metadatas"]}]
    with pytest.raises(ValueError):
        chroma_client.read_index_documents([str(i) for i in range(6)])


def test_context_tools_are_core_with_bounded_contracts():
    from zotero_mcp.server import mcp
    from zotero_mcp.toolsets import apply_toolsets

    try:
        apply_toolsets(mcp, raw="none", transport="stdio")
        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
        assert {"read_passage", "find_in_item"} <= tools.keys()
        assert "evidence_id" in tools["read_passage"].parameters["properties"]
        assert "expected_hash" in tools["find_in_item"].parameters["properties"]
    finally:
        apply_toolsets(mcp, raw="all", transport="streamable-http")
