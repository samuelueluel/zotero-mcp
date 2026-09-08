"""Regression coverage for explicit exact-item semantic-index updates."""

import json

from zotero_mcp import semantic_search
from zotero_mcp.local_db import ZoteroItem


class _FakeChroma:
    embedding_max_tokens = 8000
    embedding_model = "openai"

    def __init__(self, docs=None, texts=None):
        self.docs = {key: dict(value) for key, value in (docs or {}).items()}
        self.texts = dict(texts or {})
        self.metadata_lookups = []

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_all_ids(self, where=None):
        return set(self.docs)

    def iter_metadatas(self, batch_size=500):
        if self.docs:
            yield list(self.docs), list(self.docs.values())

    def update_metadatas(self, ids, metadatas):
        for key, metadata in zip(ids, metadatas):
            self.docs.setdefault(key, {}).update(metadata)

    def get_existing_ids(self, ids):
        return {key for key in ids if key in self.docs}

    def get_document_metadata(self, key):
        self.metadata_lookups.append(key)
        return self.docs.get(key)

    def upsert_documents(self, documents, metadatas, ids):
        for document, key, metadata in zip(documents, ids, metadatas):
            self.docs[key] = dict(metadata)
            self.texts[key] = document

    def delete_item_chunks(self, item_key, group_id=None):
        stale = [
            key for key, metadata in self.docs.items()
            if metadata.get("parent_item_key") == item_key
            or key == item_key
            or key.startswith(f"{item_key}#")
        ]
        for key in stale:
            self.docs.pop(key, None)
            self.texts.pop(key, None)

    def iter_documents(self):
        keys = list(self.docs)
        yield keys, [self.texts.get(key, "") for key in keys], [self.docs[key] for key in keys]


class _FakeReader:
    attachment_priority = ["pdf", "html"]
    items = [
        ZoteroItem(
            item_id=1,
            key="6WTDX4R3",
            item_type_id=1,
            item_type="preprint",
            doi="10.1016/j.jeconom.2020.12.001",
            title="Same title",
        ),
        ZoteroItem(
            item_id=2,
            key="F7EGNIBT",
            item_type_id=1,
            item_type="journalArticle",
            doi="10.1016/j.jeconom.2020.12.001",
            title="Same title",
        ),
    ]

    def __init__(self, **kwargs):
        self.key_filters = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get_all_item_keys(self):
        return {item.key for item in self.items}

    def get_key_group_map(self):
        return ({item.key: 0 for item in self.items}, set())

    def get_items_with_text(self, **kwargs):
        self.key_filters.append(kwargs.get("key_filter"))
        key_filter = kwargs.get("key_filter")
        if key_filter is None:
            return list(self.items)
        return [item for item in self.items if item.key in key_filter]


def _make_search(monkeypatch, chroma, *, config_path=None):
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", _FakeReader)
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: _FakeZotero())
    return semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=config_path)


class _FakeZotero:
    library_type = "user"
    library_id = "0"

    def __init__(self):
        self.item_calls = []

    def item(self, key):
        self.item_calls.append(key)
        return {
            "key": key,
            "data": {
                "key": key,
                "itemType": "journalArticle",
                "title": f"Title {key}",
                "abstractNote": "abstract",
            },
        }

    def last_modified_version(self):
        raise AssertionError("exact-item updates must not read or promote a library watermark")


def test_local_exact_scope_bypasses_dedup_and_existing_skip(monkeypatch):
    chroma = _FakeChroma()
    search = _make_search(monkeypatch, chroma)

    exact = search._get_items_from_local_db(
        extract_fulltext=False,
        chroma_client=chroma,
        item_keys=["6WTDX4R3", "F7EGNIBT"],
    )
    assert [item["key"] for item in exact] == ["6WTDX4R3", "F7EGNIBT"]

    # An explicit key scope is refreshed even when a record already exists;
    # the normal DOI/title and skip logic is not consulted.
    assert chroma.metadata_lookups == []

    normal = search._get_items_from_local_db(extract_fulltext=False)
    assert [item["key"] for item in normal] == ["F7EGNIBT"]


def test_api_exact_scope_fetches_only_requested_keys(monkeypatch):
    chroma = _FakeChroma()
    search = _make_search(monkeypatch, chroma)
    zotero = search.zotero_client

    result = search._get_items_from_api(item_keys=["6WTDX4R3", "F7EGNIBT"])

    assert [item["key"] for item in result] == ["6WTDX4R3", "F7EGNIBT"]
    assert zotero.item_calls == ["6WTDX4R3", "F7EGNIBT"]


def test_update_exact_scope_does_not_advance_watermark(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "last_sync_versions": {"0": 12345},
                    "update_config": {},
                }
            }
        )
    )
    chroma = _FakeChroma()
    search = _make_search(monkeypatch, chroma, config_path=str(config_path))
    captured = {}

    def exact_source(**kwargs):
        captured.update(kwargs)
        return [
            {
                "key": "6WTDX4R3",
                "data": {
                    "key": "6WTDX4R3",
                    "itemType": "preprint",
                    "title": "Same title",
                    "abstractNote": "abstract",
                    "DOI": "10.1016/j.jeconom.2020.12.001",
                },
            },
            {
                "key": "F7EGNIBT",
                "data": {
                    "key": "F7EGNIBT",
                    "itemType": "journalArticle",
                    "title": "Same title",
                    "abstractNote": "abstract",
                    "DOI": "10.1016/j.jeconom.2020.12.001",
                },
            },
        ]

    monkeypatch.setattr(search, "_get_items_from_source", exact_source)
    monkeypatch.setattr(search, "_load_index_schema_version", lambda: 99)
    monkeypatch.setattr(search, "_resolve_batch_mode", lambda **kwargs: (False, "openai"))

    stats = search.update_database(
        item_keys=["6wtdx4r3", "f7egnibt"],
        use_batch=False,
    )

    assert "error" not in stats
    assert stats["total_items"] == 2
    assert stats["processed_items"] == 2
    assert captured["item_keys"] == ["6WTDX4R3", "F7EGNIBT"]
    assert set(chroma.docs) == {"6WTDX4R3", "F7EGNIBT"}

    saved = json.loads(config_path.read_text())
    assert saved["semantic_search"]["last_sync_versions"] == {"0": 12345}


def test_exact_update_replaces_only_requested_chunks_and_rebuilds_bm25(monkeypatch, tmp_path):
    requested = "6WTDX4R3"
    duplicate = "F7EGNIBT"
    chroma = _FakeChroma(
        docs={
            f"{requested}#0": {"parent_item_key": requested},
            f"{requested}#1": {"parent_item_key": requested},
            f"{requested}#2": {"parent_item_key": requested},
            f"{duplicate}#0": {"parent_item_key": duplicate},
        },
        texts={
            f"{requested}#0": "obsoletefjord",
            f"{requested}#1": "obsoletefjord older",
            f"{requested}#2": "obsoletefjord oldest",
            f"{duplicate}#0": "duplicatefjord",
        },
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "last_sync_versions": {"0": 555},
                    "update_config": {},
                }
            }
        )
    )
    search = _make_search(monkeypatch, chroma, config_path=str(config_path))
    search._chunking_config = {
        "enabled": True,
        "chunk_size": 100,
        "overlap": 0,
        "max_chunks_per_item": 10,
    }
    bm25_path = tmp_path / "bm25.json"
    search._hybrid_config = {"enabled": True, "index_path": str(bm25_path)}
    monkeypatch.setattr(
        semantic_search,
        "split_into_passages",
        lambda text, chunk_size, overlap, max_chunks: [
            ("replacementfjord one", 0, 20),
            ("replacementfjord two", 20, 40),
        ],
    )
    monkeypatch.setattr(search, "_load_index_schema_version", lambda: 99)
    monkeypatch.setattr(search, "_resolve_batch_mode", lambda **kwargs: (False, "openai"))
    monkeypatch.setattr(
        search,
        "_get_items_from_source",
        lambda **kwargs: [
            {
                "key": requested,
                "data": {
                    "key": requested,
                    "itemType": "preprint",
                    "title": "Updated title",
                    "abstractNote": "Updated abstract",
                    "fulltext": "updated source",
                },
            }
        ],
    )

    stats = search.update_database(item_keys=[requested], use_batch=False)

    assert "error" not in stats
    assert stats["updated_items"] == 1
    assert set(chroma.docs) == {f"{requested}#0", f"{requested}#1", f"{duplicate}#0"}
    assert chroma.texts[f"{duplicate}#0"] == "duplicatefjord"

    from zotero_mcp.sparse_index import BM25Index

    index = BM25Index(bm25_path)
    assert index.load()
    assert index.search("obsoletefjord", top_n=10) == []
    assert {key for key, _score in index.search("replacementfjord", top_n=10)} == {
        f"{requested}#0",
        f"{requested}#1",
    }
    saved = json.loads(config_path.read_text())
    assert saved["semantic_search"]["last_sync_versions"] == {"0": 555}
