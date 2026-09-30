"""Sidecar writes: atomic replacement, with blocks.json written last."""

from __future__ import annotations

import pytest

pytest.importorskip("pymupdf")

from zotero_mcp import sidecar_assemble  # noqa: E402

META = {"pages": {"1": {"page": 1, "scan_like": False, "rotation": 0}}}
BLOCKS = [{"id": "p1", "page": 1, "label": "Text", "html": "<p>Intro</p>", "status": "verified", "findings": []}]


def test_write_outputs_writes_blocks_json_last(tmp_path, monkeypatch):
    order = []
    real = sidecar_assemble.atomic_write_text
    monkeypatch.setattr(sidecar_assemble, "atomic_write_text", lambda p, t: (order.append(p.name), real(p, t)))
    rel = sidecar_assemble.write_outputs("K", META, BLOCKS, {"batch_dir": "b"}, tmp_path)
    assert order == ["K.md", "K.reliability.json", "K.blocks.json"]
    assert rel["level"] == "ok" and rel["batch_dir"] == "b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["K.blocks.json", "K.md", "K.reliability.json"]


def test_interrupted_write_keeps_the_old_file(tmp_path, monkeypatch):
    p = tmp_path / "K.md"
    p.write_text("old sidecar")

    def stop(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(sidecar_assemble.os, "replace", stop)
    with pytest.raises(KeyboardInterrupt):
        sidecar_assemble.atomic_write_text(p, "new sidecar")
    assert p.read_text() == "old sidecar"


def test_blank_scanned_page_withholds_what_ocr_invented():
    import pymupdf

    from zotero_mcp.sidecar_assemble import BLANK_INK, guard_blank_pages, page_ink, render_markdown
    from zotero_mcp.sidecar_reliability import SURYA_SOURCE, chunk_reliability
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    blank, text = doc[0], doc[1]
    for y in range(80, 700, 14):
        text.insert_text((72, y), "The last lines of a chapter, set in the usual type size.", fontsize=11)
    assert page_ink(blank) < BLANK_INK < page_ink(text)
    blocks = [{"id": "K:p1:b0", "page": 1, "label": "Table", "status": "single-route", "findings": [],
               "repairs": [], "html": "<table><tr><td>0.0000</td></tr></table>"},
              {"id": "K:p1:b1", "page": 1, "label": "Picture", "status": "not-text", "findings": [],
               "repairs": [], "html": ""}]
    guard_blank_pages(blocks, {1})
    assert blocks[0]["status"] == "unresolved" and blocks[0]["html"] == "" and "0.0000" in blocks[0]["surya_html"]
    assert blocks[1]["status"] == "not-text"
    md = render_markdown(blocks)
    assert "0.0000" not in md and "[Page status: UNRESOLVED" in md
    assert chunk_reliability(md, SURYA_SOURCE)["block_status"] == "unresolved"
