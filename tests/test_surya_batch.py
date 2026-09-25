"""Batch driver: chunking, state resume, and the live-config guard."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

from zotero_mcp import surya_batch  # noqa: E402
from zotero_mcp.surya_runner import ItemJob  # noqa: E402


def _jobs(*keys):
    return [ItemJob(k, Path(f"/x/{k}.pdf")) for k in keys]


def test_chunks_respect_page_budget_and_isolate_oversized():
    jobs = _jobs("A", "B", "C", "D")
    pages = {"A": 60, "B": 60, "C": 300, "D": 30}
    chunks = surya_batch.chunk_jobs(jobs, pages, 160)
    assert [[j.item_key for j in c] for c in chunks] == [["A", "B"], ["C"], ["D"]]


def test_state_persists_and_resumes(tmp_path):
    s = surya_batch.State(tmp_path / "state.json")
    s.mark("K", "ocr", level="ok")
    s.error("K", "repair", "boom")
    s2 = surya_batch.State(tmp_path / "state.json")
    assert s2.done("K", "ocr") and not s2.done("K", "repair")
    assert s2.item("K")["level"] == "ok" and s2.item("K")["errors"][0]["stage"] == "repair"


def test_refuses_live_config(monkeypatch, tmp_path):
    live = tmp_path / "config.json"
    live.write_text("{}")
    monkeypatch.setattr(surya_batch, "LIVE_CONFIG", live)
    with pytest.raises(SystemExit, match="refusing the live config"):
        surya_batch.main(["--config", str(live), "--item", "K", "--run", "t"])


def test_index_waits_for_enrich(tmp_path, monkeypatch):
    s = surya_batch.State(tmp_path / "state.json")
    s.mark("A", "ocr"); s.mark("A", "repair")
    calls = []
    monkeypatch.setattr(surya_batch.subprocess, "run", lambda *a, **k: calls.append(a))
    surya_batch.stage_index(_jobs("A"), s, tmp_path, tmp_path / "c.json")
    assert calls == []
