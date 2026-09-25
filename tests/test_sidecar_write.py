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
