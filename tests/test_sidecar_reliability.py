"""Tests for sidecar page anchors and reliability signals."""

from __future__ import annotations

import json

from zotero_mcp import sidecar_reliability as rel
from zotero_mcp.semantic_search import _page_for_offset

DOC = (
    "<!-- pdf-page: 1 -->\n\n## Intro\n\nText.\n\n"
    "<!-- pdf-page: 2 -->\n\n[Table status: UNRESOLVED (duplicate_rows); unverified numbers withheld. "
    "Verify on PDF p. 2.]\n\n<table><tr><td>⟦withheld: unverified number, see PDF p. 2⟧</td></tr></table>\n\n"
    "<!-- pdf-page: 3 -->\n\nMore text.\n"
)


def test_page_for_offset_uses_anchors():
    assert _page_for_offset(DOC, 0) == 1
    assert _page_for_offset(DOC, DOC.index("[Table status")) == 2
    assert _page_for_offset(DOC, DOC.index("More text")) == 3
    assert _page_for_offset("plain text, no pages", 5) is None


def test_pages_in_span():
    start = DOC.index("Text.")
    assert rel.pages_in_span(DOC, start, len(DOC)) == [1, 2, 3]


def test_chunk_reliability_statuses():
    assert rel.chunk_reliability("plain prose", rel.SURYA_SOURCE) == {"block_status": "verified", "check_pages": []}
    r = rel.chunk_reliability(DOC, rel.SURYA_SOURCE)
    assert r == {"block_status": "unresolved", "check_pages": [2]}
    single = "[Table status: SINGLE-ROUTE (math_table); numbers not independently confirmed. Check PDF p. 41 before quoting.]"
    assert rel.chunk_reliability(single, rel.SURYA_SOURCE)["block_status"] == "single-route"
    assert rel.chunk_reliability(DOC, rel.LEGACY_SOURCE)["block_status"] == "legacy-unverified"


def test_result_reliability_and_source(tmp_path):
    (tmp_path / "KEY1.md").write_text("x")
    assert rel.sidecar_source(tmp_path, "KEY1") == rel.LEGACY_SOURCE
    (tmp_path / "KEY1.reliability.json").write_text(json.dumps({"level": "warn", "warning": "1 of 3 tables unresolved"}))
    assert rel.sidecar_source(tmp_path, "KEY1") == rel.SURYA_SOURCE
    out = rel.result_reliability({"block_status": "unresolved", "check_pages": "2"},
                                 rel.load_item_reliability(tmp_path, "KEY1"))
    assert out["requires_pdf_check"] is True and out["item_level"] == "warn"
    legacy = rel.result_reliability({}, None)
    assert legacy["block_status"] == "legacy-unverified" and legacy["requires_pdf_check"] is True
    assert rel.result_reliability({"block_status": "verified"}, {"level": "ok"})["requires_pdf_check"] is False
