"""Batch driver: chunking, state resume, the live-config guard, and recovery after a stop."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

from zotero_mcp import surya_batch  # noqa: E402
from zotero_mcp.sidecar_assemble import write_outputs  # noqa: E402
from zotero_mcp.surya_runner import ItemJob  # noqa: E402

META = {"pages": {"1": {"page": 1, "scan_like": False, "rotation": 0}}}


def _jobs(*keys):
    return [ItemJob(k, Path(f"/x/{k}.pdf")) for k in keys]


def _blocks(table_status="unresolved"):
    return [
        {"id": "p1", "page": 1, "label": "Text", "html": "<p>Intro text</p>", "status": "verified", "findings": []},
        {"id": "t1", "page": 1, "label": "Table", "html": "<table><tr><td>1.25</td></tr></table>",
         "status": table_status, "findings": []},
    ]


def _completed(cmd, rc=0, err=""):
    return subprocess.CompletedProcess(cmd, rc, "", err)


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
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


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


def test_start_vlm_reuses_a_healthy_server(monkeypatch):
    calls = []
    monkeypatch.setattr(surya_batch, "_healthy", lambda url: True)
    monkeypatch.setattr(surya_batch.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    surya_batch.start_vlm()
    assert calls == []


def test_start_vlm_clears_a_leftover_container_first(monkeypatch):
    calls = []
    monkeypatch.setattr(surya_batch, "_healthy", lambda url: False)
    monkeypatch.setattr(surya_batch.subprocess, "run", lambda cmd, **kw: calls.append(cmd[-1]) or _completed(cmd))
    surya_batch.start_vlm()
    assert calls == ["stop", "start"]


def test_start_vlm_failure_reports_the_script_message(monkeypatch):
    monkeypatch.setattr(surya_batch, "_healthy", lambda url: False)

    def fake_run(cmd, **kw):
        if cmd[-1] == "start":
            return _completed(cmd, 1, "Container zotero-vlm-rocm already exists")
        return _completed(cmd)

    monkeypatch.setattr(surya_batch.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="already exists"):
        surya_batch.start_vlm()


def test_sidecar_consistency_check(tmp_path):
    write_outputs("K", META, _blocks(), {"batch_dir": "b"}, tmp_path)
    assert surya_batch.sidecar_consistent("K", tmp_path)
    md = tmp_path / "K.md"
    good = md.read_text()
    md.write_text(good[: len(good) // 2])  # cut off mid-write
    assert not surya_batch.sidecar_consistent("K", tmp_path)
    md.write_text(good + "\n[Figure Schema]\n- Type: map\n")  # enrichment has started
    assert surya_batch.sidecar_consistent("K", tmp_path)
    md.write_text(good)
    record = json.loads((tmp_path / "K.blocks.json").read_text())
    record["blocks"][1]["status"] = "repaired"  # blocks.json ahead of the Markdown
    (tmp_path / "K.blocks.json").write_text(json.dumps(record))
    assert not surya_batch.sidecar_consistent("K", tmp_path)
    (tmp_path / "K.blocks.json").write_text('{"item_key": ')  # truncated JSON
    assert not surya_batch.sidecar_consistent("K", tmp_path)


def test_heal_rebuilds_an_inconsistent_sidecar_and_redoes_repair(tmp_path, monkeypatch):
    s = surya_batch.State(tmp_path / "state.json")
    for k in ("A", "B"):
        s.mark(k, "ocr", batch=str(tmp_path / "chunk"))
        s.mark(k, "repair")
    write_outputs("B", META, _blocks(), {}, tmp_path)  # B is consistent; A has no sidecar
    calls = []
    monkeypatch.setattr(surya_batch, "reassemble",
                        lambda k, batch, d: calls.append(k) or {"level": "warn", "tables": {"unresolved": 1}})
    surya_batch.heal_sidecars(_jobs("A", "B"), s, tmp_path, tmp_path)
    assert calls == ["A"]
    assert s.item("A")["done"] == ["ocr"] and s.item("B")["done"] == ["ocr", "repair"]
    assert "rebuilt from its OCR chunk" in (tmp_path / "run.log").read_text()


def test_heal_falls_back_to_ocr_when_the_chunk_is_gone(tmp_path):
    s = surya_batch.State(tmp_path / "state.json")
    s.mark("A", "ocr", batch=str(tmp_path / "missing-chunk"))
    surya_batch.heal_sidecars(_jobs("A"), s, tmp_path, tmp_path)
    assert s.item("A")["done"] == [] and s.item("A")["errors"][0]["stage"] == "heal"


def test_abort_reason_reaches_run_log(tmp_path, monkeypatch):
    cfg = tmp_path / "shadow.json"
    cfg.write_text(json.dumps({"semantic_search": {"mineru": {"sidecar_dir": str(tmp_path / "side")}}}))
    monkeypatch.setattr(surya_batch, "load_surya_config", lambda p: {"work_dir": str(tmp_path / "work")})
    monkeypatch.setattr(surya_batch, "resolve_items", lambda *a: (_jobs("A"), {}))
    monkeypatch.setattr(surya_batch, "reset_if_pdf_changed", lambda jobs, state: None)
    monkeypatch.setattr(surya_batch.signal, "signal", lambda *a: None)

    def boom(*a, **k):
        raise RuntimeError("podman run zotero-surya failed: no space left")

    monkeypatch.setattr(surya_batch, "stage_ocr", boom)
    with pytest.raises(RuntimeError):
        surya_batch.main(["--config", str(cfg), "--item", "A", "--run", "t", "--stages", "ocr"])
    log = (tmp_path / "work" / "runs" / "t" / "run.log").read_text()
    assert "ABORTED: RuntimeError: podman run zotero-surya failed: no space left" in log


def test_seed_reassembles_from_the_seed_run_and_never_runs_ocr(tmp_path, monkeypatch):
    (tmp_path / "seed").mkdir()
    seed = surya_batch.State(tmp_path / "seed" / "state.json")
    seed.mark("A", "ocr", batch="/chunks/1", pdf_sha256="h")
    seed.mark("B", "ocr", batch="/chunks/1", pdf_sha256="old")  # PDF changed since
    s = surya_batch.State(tmp_path / "state.json")
    for k, h in (("A", "h"), ("B", "new"), ("C", "c")):  # C is not in the seed run
        s.item(k)["pdf_sha256"] = h
    calls = []
    monkeypatch.setattr(surya_batch, "reassemble",
                        lambda k, batch, d: calls.append((k, str(batch))) or {"level": "ok", "tables": {"verified": 2}})
    monkeypatch.setattr(surya_batch, "run_batch", lambda *a, **k: pytest.fail("OCR must not run"))
    surya_batch.stage_seed(_jobs("A", "B", "C"), s, tmp_path, tmp_path, tmp_path / "seed")
    assert calls == [("A", "/chunks/1")]
    assert s.done("A", "ocr") and s.item("A")["seeded_from"] == "seed"
    assert not s.done("B", "ocr") and not s.done("C", "ocr")
    assert "PDF changed" in s.item("B")["errors"][0]["msg"]
    assert "no OCR" in s.item("C")["errors"][0]["msg"]


def test_appendix_pdfs_are_recognised(tmp_path):
    import pymupdf

    def pdf(name, first_line):
        doc = pymupdf.open()
        doc.new_page().insert_text((50, 50), first_line)
        path = tmp_path / name
        doc.save(path)
        return path

    assert surya_batch._is_appendix(pdf("Notowidigdo_APPENDIX.pdf", "Online Appendix for"))
    assert surya_batch._is_appendix(pdf("paper2.pdf", "Online Appendix A"))
    assert not surya_batch._is_appendix(pdf("Bruhn-2017.pdf", "CRIME AND PUBLIC HOUSING"))
