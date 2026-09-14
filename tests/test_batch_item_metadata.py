"""Contract tests for batch_get_item_metadata.

Covers the aggregation contract agents rely on for shortlist screening:
request-order output, first-occurrence deduplication, per-key failure
isolation (a missing or erroring item never aborts the batch), malformed-key
reporting, the 50-key fail-fast cap, and include_abstract passthrough.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from conftest import DummyContext

from zotero_mcp.tools import retrieval


@pytest.fixture()
def fake_zot(monkeypatch):
    zot = MagicMock()
    monkeypatch.setattr(retrieval._client, "get_zotero_client", lambda: zot)
    return zot


def _item(key: str, title: str) -> dict:
    return {
        "key": key,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": title,
            "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "J."}],
        },
    }


def test_batch_renders_items_in_request_order(fake_zot):
    fake_zot.item.side_effect = lambda k: _item(k, f"Title {k}")

    out = retrieval.batch_get_item_metadata(
        item_keys=["BBBB2222", "AAAA1111"], ctx=DummyContext()
    )

    assert out.startswith("# Batch item metadata: 2 fetched, 0 failed")
    assert out.index("Title BBBB2222") < out.index("Title AAAA1111")
    assert "\n\n---\n\n" in out  # horizontal-rule separation between items


def test_batch_deduplicates_preserving_first_occurrence(fake_zot):
    fake_zot.item.side_effect = lambda k: _item(k, f"Title {k}")

    out = retrieval.batch_get_item_metadata(
        item_keys=["AAAA1111", "AAAA1111", "BBBB2222"], ctx=DummyContext()
    )

    assert fake_zot.item.call_count == 2
    assert out.startswith("# Batch item metadata: 2 fetched, 0 failed")


def test_batch_isolates_missing_and_erroring_keys(fake_zot):
    def _fetch(key):
        if key == "ERRK0003":
            raise ValueError("boom")
        if key == "MISS0004":
            return None
        return _item(key, f"Title {key}")

    fake_zot.item.side_effect = _fetch

    out = retrieval.batch_get_item_metadata(
        item_keys=["OKAY0001", "ERRK0003", "MISS0004"], ctx=DummyContext()
    )

    assert out.startswith("# Batch item metadata: 1 fetched, 2 failed")
    assert "Title OKAY0001" in out
    assert "## Failed keys" in out
    assert "- MISS0004: no item found with this key" in out
    assert "- ERRK0003: boom" in out


def test_batch_rejects_oversized_requests_without_calling_api(fake_zot):
    keys = [f"K{i:07d}" for i in range(retrieval.BATCH_METADATA_MAX_ITEMS + 1)]

    out = retrieval.batch_get_item_metadata(item_keys=keys, ctx=DummyContext())

    assert "capped at 50" in out
    assert "Split the request" in out
    fake_zot.item.assert_not_called()


def test_batch_reports_malformed_keys_and_keeps_going(fake_zot):
    fake_zot.item.side_effect = lambda k: _item(k, f"Title {k}")

    out = retrieval.batch_get_item_metadata(
        item_keys=["short", "OKAY0001"], ctx=DummyContext()
    )

    assert "Malformed keys ignored" in out
    assert "`short`" in out
    assert "Title OKAY0001" in out


def test_batch_with_no_usable_keys_fails_fast(fake_zot):
    out = retrieval.batch_get_item_metadata(item_keys=[], ctx=DummyContext())
    assert "requires a non-empty item_keys list" in out
    fake_zot.item.assert_not_called()


def test_batch_passes_include_abstract_through(monkeypatch, fake_zot):
    recorded = {}
    real_format = retrieval._client.format_item_metadata

    def _spy(item, include_abstract):
        recorded["include_abstract"] = include_abstract
        return real_format(item, include_abstract)

    monkeypatch.setattr(retrieval._client, "format_item_metadata", _spy)
    fake_zot.item.side_effect = lambda k: _item(k, "Title")

    retrieval.batch_get_item_metadata(
        item_keys=["OKAY0001"], include_abstract=False, ctx=DummyContext()
    )

    assert recorded["include_abstract"] is False
