"""Hermetic contracts for discovery → expansion → literal source lookup."""

from __future__ import annotations

import json

import pytest
from conftest import DummyContext

from zotero_mcp.passage_context import (
    MAX_CONTEXT_CHARS,
    chunk_fingerprint,
    evidence_id,
    find_windows,
    parse_evidence_id,
    source_preview,
    text_hash,
)
from zotero_mcp.tools import context as tools

KEY = "ITEM0001"
OTHER = "ITEM0002"


def record(text="The treatment reduced crime by 11%. Notes: matched neighborhoods.", index=1, **extra):
    metadata = {
        "group_id": 0,
        "item_key": KEY,
        "parent_item_key": KEY,
        "chunk_index": index,
        "n_chunks": 3,
        "fulltext_source": "mineru-sidecar",
        **extra,
    }
    return f"{KEY}#{index}", text, metadata


def setup_tool(monkeypatch, tmp_path, records=None):
    monkeypatch.setattr(tools._client, "get_active_group_id", lambda: 0)

    class Client:
        def item(self, key):
            return {"key": key, "data": {"title": "A paper", "itemType": "journalArticle"}}

    monkeypatch.setattr(tools._client, "get_zotero_client", lambda: Client())
    calls = []
    records = records if records is not None else [record()]

    def fetch(ids):
        calls.append(ids)
        return {
            "ids": [r[0] for r in records],
            "documents": [r[1] for r in records],
            "metadatas": [r[2] for r in records],
        }

    monkeypatch.setattr(tools, "_index_documents", fetch)
    monkeypatch.setattr(tools._mineru, "load_mineru_config", lambda _: {"sidecar_dir": str(tmp_path)})
    return calls


def call_read(token=None, **kwargs):
    token = token or evidence_id(*record())
    return json.loads(tools.read_passage(evidence_id=token, ctx=DummyContext(), **kwargs))


def call_find(**kwargs):
    return json.loads(tools.find_in_item(item_key=KEY, ctx=DummyContext(), **kwargs))


def test_token_binds_text_location_and_library():
    cid, text, meta = record()
    token = evidence_id(cid, text, meta)
    assert parse_evidence_id(token) == (0, cid, chunk_fingerprint(text, meta))
    assert evidence_id(cid, text + " Changed.", meta) != token
    assert evidence_id(cid, text, {**meta, "char_start": 150}) != token
    assert evidence_id(cid, text, {**meta, "group_id": 42}) != token


@pytest.mark.parametrize(
    "value", ["", "../../secret", "ITEM0001#2", "zr1:0:../../file:" + "a" * 64, "zr1:0:ITEM0001#0:" + "a" * 65]
)
def test_invalid_evidence_ids(value):
    with pytest.raises(ValueError):
        parse_evidence_id(value)


@pytest.mark.parametrize("group", [None, "0", -1, True])
def test_missing_or_invalid_scope_cannot_mint_token(group):
    cid, text, meta = record(group_id=group)
    assert evidence_id(cid, text, meta) is None


def test_preview_is_bounded_verbatim_and_avoids_breadcrumb():
    text = (
        "[Paper: Firearm violence | Section: Firearm violence]\n# Firearm violence\n"
        + "Unrelated background. " * 50
        + "\nWe estimate firearm violence fell by 11%. The confidence interval is 7–15%.\n"
    )
    preview, offset = source_preview("firearm violence estimate", text, width=180)
    assert "11%" in preview
    assert not preview.startswith("[Paper:")
    assert text[offset : offset + len(preview)] == preview
    assert len(preview) <= 180


def test_preview_long_row_keeps_query_and_does_not_invent_ellipsis():
    text = "x " * 1000 + "Treatment estimate -0.072" + " x" * 1000
    preview, offset = source_preview("Treatment estimate", text, width=200)
    assert "-0.072" in preview
    assert preview == text[offset : offset + len(preview)]


def test_expansion_returns_full_anchor_no_extra_search(monkeypatch, tmp_path):
    calls = setup_tool(monkeypatch, tmp_path)
    result = call_read()
    assert result["ok"]
    assert result["route"] == "indexed_passage"
    assert result["chunks"][0]["text"] == record()[1]
    assert result["chunks"][0]["content_hash"] == text_hash(record()[1])
    assert result["truncated"] is False
    assert calls == [[f"{KEY}#1"]]
    assert "page" not in result["chunks"][0]


def test_expansion_neighbors_are_bounded_and_anchor_first(monkeypatch, tmp_path):
    records = [record("B" * 400, 2), record("A" * 400, 1), record("C" * 400, 0)]
    calls = setup_tool(monkeypatch, tmp_path, records)
    result = call_read(evidence_id(*records[1]), neighbors=1, max_chars=700)
    assert calls == [[f"{KEY}#1", f"{KEY}#0", f"{KEY}#2"]]
    assert result["chunks"][0]["chunk_id"] == f"{KEY}#1"
    assert sum(len(c["text"]) for c in result["chunks"]) == 700
    assert result["truncated"] and result["omitted_chunk_ids"] == [f"{KEY}#2"]


def test_truncated_anchor_has_exact_continuation(monkeypatch, tmp_path):
    rec = record("0123456789" * 100)
    setup_tool(monkeypatch, tmp_path, [rec])
    token = evidence_id(*rec)
    first = call_read(token, max_chars=256)
    second = call_read(token, start_char=first["next_char_start"], max_chars=256)
    assert first["chunks"][0]["text"] + second["chunks"][0]["text"] == rec[1][:512]


@pytest.mark.parametrize("records", [[], [record("changed")], [record(char_start=9)]])
def test_stale_evidence_fails_closed(monkeypatch, tmp_path, records):
    setup_tool(monkeypatch, tmp_path, records)
    assert call_read()["error"]["code"] == "STALE_EVIDENCE"


def test_switching_library_does_not_reinterpret_token(monkeypatch, tmp_path):
    calls = setup_tool(monkeypatch, tmp_path)
    monkeypatch.setattr(tools._client, "get_active_group_id", lambda: 8)
    assert call_read()["error"]["code"] == "LIBRARY_MISMATCH"
    assert calls == []


@pytest.mark.parametrize(
    "records", [[record(group_id=2)], [record(parent_item_key=OTHER)], [(f"{OTHER}#1", "foreign", {"group_id": 0})]]
)
def test_backend_scope_leak_is_rejected(monkeypatch, tmp_path, records):
    setup_tool(monkeypatch, tmp_path, records)
    assert call_read()["error"]["code"] == "SCOPE_MISMATCH"


@pytest.mark.parametrize("kwargs", [{"neighbors": 3}, {"max_chars": MAX_CONTEXT_CHARS + 1}, {"start_char": -1}])
def test_read_limits_reject_not_silently_clamp(monkeypatch, tmp_path, kwargs):
    calls = setup_tool(monkeypatch, tmp_path)
    assert call_read(**kwargs)["error"]["code"] == "INVALID_ARGUMENT"
    assert calls == []


def test_literal_find_has_exact_locators(monkeypatch, tmp_path):
    setup_tool(monkeypatch, tmp_path)
    text = "# Results\n\nEstimate: -0.072 (SE 0.02).\nTable notes: quarterly count.\n"
    (tmp_path / f"{KEY}.md").write_text(text)
    result = call_find(query="ESTIMATE", context_lines=1)
    assert result["ok"] and result["route"] == "mineru_sidecar"
    assert result["source_hash"] == text_hash(text)
    window = result["windows"][0]
    assert window["start_line"] == 2 and window["end_line"] == 4
    assert window["text"] == text[window["char_start"] : window["char_end"]]
    assert "quarterly count" in window["text"]


def test_literal_lookup_never_executes_regex():
    result = find_windows("literal .* value\nother words", ".*", context_lines=0)
    assert len(result["windows"]) == 1
    assert result["windows"][0]["match_char_start"] == 8


def test_match_windows_coalesce_and_budget_is_total():
    text = "\n".join(f"match {i}" for i in range(1000))
    result = find_windows(text, "match", context_lines=1, max_matches=2, max_chars=256)
    windows = result["windows"]
    assert len(windows) == 2
    assert windows[0]["char_end"] <= windows[1]["char_start"]
    assert sum(len(w["text"]) for w in windows) <= 256
    assert result["truncated"] and result["next_start_line"] is not None


def test_long_table_line_retains_match_and_allows_char_continuation(monkeypatch, tmp_path):
    setup_tool(monkeypatch, tmp_path)
    text = "<table>" + "a " * 2000 + "estimate -0.072" + " b" * 2000 + "</table>"
    (tmp_path / f"{KEY}.md").write_text(text)
    result = call_find(query="estimate", max_chars=256)
    assert "estimate -0.072" in result["windows"][0]["text"]
    assert result["windows"][0]["starts_mid_line"]
    assert result["truncated"]
    continuation = call_find(start_char=result["next_char_start"], expected_hash=result["source_hash"], max_chars=256)
    assert continuation["windows"][0]["text"] == text[result["next_char_start"] : result["next_char_start"] + 256]


def test_sidecar_line_read_and_changed_source(monkeypatch, tmp_path):
    setup_tool(monkeypatch, tmp_path)
    path = tmp_path / f"{KEY}.md"
    path.write_text("one\ntwo\nthree\nfour\n")
    result = call_find(start_line=2, end_line=3)
    assert result["windows"][0]["text"] == "two\nthree\n"
    path.write_text("changed\n")
    assert call_find(expected_hash=result["source_hash"])["error"]["code"] == "STALE_EVIDENCE"


def test_missing_sidecar_does_not_create_or_parse(monkeypatch, tmp_path):
    setup_tool(monkeypatch, tmp_path)
    assert call_find(query="anything")["error"]["code"] == "SIDECAR_NOT_FOUND"
    assert list(tmp_path.iterdir()) == []


def test_wrong_key_and_attachment_rejected(monkeypatch, tmp_path):
    setup_tool(monkeypatch, tmp_path)

    class Wrong:
        def item(self, key):
            return {"key": OTHER, "data": {"itemType": "attachment"}}

    monkeypatch.setattr(tools._client, "get_zotero_client", lambda: Wrong())
    assert call_find(query="test")["error"]["code"] == "INVALID_ARGUMENT"


def test_sidecar_scope_is_not_guessed(monkeypatch, tmp_path):
    setup_tool(monkeypatch, tmp_path)
    monkeypatch.setattr(tools._client, "get_active_group_id", lambda: 12)
    assert call_find(query="anything")["error"]["code"] == "SIDECAR_LIBRARY_UNSUPPORTED"


@pytest.mark.parametrize("key", ["../../secret", "", "the title", "item0001"])
def test_no_arbitrary_paths(monkeypatch, tmp_path, key):
    setup_tool(monkeypatch, tmp_path)
    result = json.loads(tools.find_in_item(item_key=key, query="test", ctx=DummyContext()))
    assert result["error"]["code"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chars": 20000},
        {"max_matches": 11},
        {"context_lines": 21},
        {"start_line": 0},
        {"end_line": 0},
        {"query": ""},
        {"query": "a" * 501},
    ],
)
def test_find_argument_limits(kwargs):
    params = {"query": "match", **kwargs}
    with pytest.raises(ValueError):
        find_windows("match\ntext", **params)


def test_sidecar_source_io_is_bounded(monkeypatch, tmp_path):
    setup_tool(monkeypatch, tmp_path)
    monkeypatch.setattr(tools, "MAX_SIDECAR_BYTES", 20)
    (tmp_path / f"{KEY}.md").write_text("x" * 21)
    assert call_find()["error"]["code"] == "SOURCE_TOO_LARGE"


def test_unicode_match_offsets_are_source_offsets():
    text = "Straße\nİstanbul estimate 5%\n"
    result = find_windows(text, "estimate", context_lines=0)
    w = result["windows"][0]
    assert text[w["match_char_start"] : w["match_char_end"]] == "estimate"
