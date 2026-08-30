"""Regression tests for reranker lifecycle and the local-only policy.

The production path uses a local HTTP reranker and fails closed when that
endpoint is enabled but missing. The old process-wide in-process cache helper
is retained and tested directly for compatibility, while startup/search tests
ensure the disabled Hugging Face fallback is never loaded.
"""

import pytest

from zotero_mcp import semantic_search


class _FakeReranker:
    """Counts how many times a reranker is actually constructed (= model load)."""

    load_count = 0

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        type(self).load_count += 1
        self.model_name = model_name


class _FakeChromaClient:
    embedding_max_tokens = 8000

    def search(self, *a, **k):
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch):
    """Each test gets a clean cache and a fresh, fake, non-loading reranker."""
    semantic_search._RERANKER_CACHE.clear()
    _FakeReranker.load_count = 0
    monkeypatch.setattr(semantic_search, "CrossEncoderReranker", _FakeReranker)
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    yield
    semantic_search._RERANKER_CACHE.clear()


def test_get_cached_reranker_loads_once_per_model():
    r1 = semantic_search.get_cached_reranker("model-a")
    r2 = semantic_search.get_cached_reranker("model-a")
    assert r1 is r2
    assert _FakeReranker.load_count == 1  # second call hits the cache

    r3 = semantic_search.get_cached_reranker("model-b")
    assert r3 is not r1
    assert _FakeReranker.load_count == 2  # distinct model loads separately


def test_reranker_requires_local_endpoint():
    """Enabled reranking must fail closed without the configured local service."""
    s = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChromaClient())
    s._reranker_config = {"enabled": True, "model": "shared-model"}

    with pytest.raises(RuntimeError, match="Local reranker is enabled"):
        s._get_reranker()

    assert _FakeReranker.load_count == 0


def test_get_reranker_returns_none_when_disabled():
    s = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChromaClient())
    s._reranker_config = {"enabled": False, "model": "shared-model"}
    assert s._get_reranker() is None
    assert _FakeReranker.load_count == 0  # disabled never loads


def test_warmup_reranker_requires_local_endpoint(monkeypatch):
    monkeypatch.setattr(semantic_search.os.path, "exists", lambda p: False)
    # Disabled config -> no warmup, no load.
    assert semantic_search.warmup_reranker(config_path=None) is False
    assert _FakeReranker.load_count == 0

    # Enabled config without a local endpoint fails closed and never loads the
    # disabled in-process CrossEncoder.
    monkeypatch.setattr(
        semantic_search,
        "load_reranker_config",
        lambda cp: {"enabled": True, "model": "warm-model"},
    )
    assert semantic_search.warmup_reranker(config_path=None) is False
    assert _FakeReranker.load_count == 0

    # A configured local endpoint passes the startup gate without making a
    # network request; the HTTP client is created lazily on first search.
    monkeypatch.setattr(
        semantic_search,
        "load_reranker_config",
        lambda cp: {"enabled": True, "url": "http://127.0.0.1:8083/v1/rerank"},
    )
    assert semantic_search.warmup_reranker(config_path=None) is True
    assert _FakeReranker.load_count == 0
