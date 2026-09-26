"""Repair pass accepts a VLM re-read only when it passes the table check."""

from __future__ import annotations

import json

import pytest

pymupdf = pytest.importorskip("pymupdf")

from zotero_mcp import sidecar_repair  # noqa: E402


def _setup(tmp_path):
    pdf = tmp_path / "t.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Age   .512   .634")
    doc.save(pdf)
    block = {
        "id": "K:p1:b0", "page": 1, "order": 0, "label": "Table",
        "pdf_bbox": [0, 0, 612, 792], "rotation": 0, "scan_like": False,
        "html": "<table><tr><td>Age</td><td>.512</td></tr></table>",
        "status": "unresolved", "findings": [{"kind": "native_missing", "detail": "", "values": [".634"]}],
        "repairs": [],
    }
    meta = {"pdf_path": str(pdf), "pdf_sha256": "x", "pages": {"K__p001": {
        "page": 1, "rotation": 0, "scan_like": False, "image_size": [1, 1], "page_size": [612, 792]}}}
    (tmp_path / "KEY00001.blocks.json").write_text(json.dumps(
        {"item_key": "KEY00001", "pdf_path": str(pdf), "pdf_sha256": "x", "meta": meta, "blocks": [block]}))


def test_accepts_reread_that_passes_check(monkeypatch, tmp_path):
    _setup(tmp_path)
    monkeypatch.setattr(sidecar_repair, "ask_vlm",
                        lambda png, url: "<table><tr><td>Age</td><td>.512</td><td>.634</td></tr></table>")
    out = sidecar_repair.repair_item("KEY00001", tmp_path, "http://x")
    assert out["accepted"] == 1
    block = json.loads((tmp_path / "KEY00001.blocks.json").read_text())["blocks"][0]
    # A whole-table rewrite can pin right numbers on wrong labels: never "repaired".
    assert block["status"] == "single-route" and block["repairs"][0]["route"] == "vlm_table_reread"
    assert any(f["kind"] == "vlm_rewrite" for f in block["findings"])
    assert "labels are unchecked" in (tmp_path / "KEY00001.md").read_text()


def test_rejects_reread_that_fails_check(monkeypatch, tmp_path):
    _setup(tmp_path)
    monkeypatch.setattr(sidecar_repair, "ask_vlm",
                        lambda png, url: "<table><tr><td>Age</td><td>.999</td><td>.634</td></tr></table>")
    out = sidecar_repair.repair_item("KEY00001", tmp_path, "http://x")
    assert out["rejected"] == 1
    block = json.loads((tmp_path / "KEY00001.blocks.json").read_text())["blocks"][0]
    assert block["status"] == "unresolved"
    assert any(f["kind"] == "vlm_repair_failed" for f in block["findings"])
    assert "withheld" in (tmp_path / "KEY00001.md").read_text()
    assert block["vlm_rejected"]["html"].count(".999") == 1


def test_star_disagreement():
    surya = "<table><tr><td>a</td><td>.5</td><td>***</td></tr></table>"
    dropped = "<table><tr><td>a</td><td>.5</td></tr></table>"
    assert sidecar_repair.star_disagreement(surya, dropped, "a .5") is not None  # no stars in the text layer
    assert sidecar_repair.star_disagreement(surya, surya, "") is None
    assert sidecar_repair.star_disagreement(dropped, surya, "a .5***") is None  # the text layer backs the re-read


def _scan_setup(tmp_path, html):
    _setup(tmp_path)
    p = tmp_path / "KEY00001.blocks.json"
    rec = json.loads(p.read_text())
    rec["blocks"][0].update(status="single-route", scan_like=True, findings=[], html=html)
    p.write_text(json.dumps(rec))


def test_second_reading_adds_agreement_on_scans(monkeypatch, tmp_path):
    html = "<table><tr><td>Age</td><td>.512</td><td>.634</td></tr></table>"
    _scan_setup(tmp_path, html)
    monkeypatch.setattr(sidecar_repair, "ask_vlm", lambda png, url: html.replace(".512", "0.512"))
    out = sidecar_repair.repair_item("KEY00001", tmp_path, "http://x")
    assert out["second_read"] == 1 and out["agreed"] == 1
    block = json.loads((tmp_path / "KEY00001.blocks.json").read_text())["blocks"][0]
    assert block["status"] == "single-route" and block["html"] == html
    assert any(f["kind"] == "vlm_agreement" for f in block["findings"])
    assert "second model agree on every number" in (tmp_path / "KEY00001.md").read_text()


def test_second_reading_disagreement_changes_nothing_visible(monkeypatch, tmp_path):
    html = "<table><tr><td>Age</td><td>.512</td><td>.634</td></tr></table>"
    _scan_setup(tmp_path, html)
    monkeypatch.setattr(sidecar_repair, "ask_vlm", lambda png, url: html.replace(".634", ".684"))
    out = sidecar_repair.repair_item("KEY00001", tmp_path, "http://x")
    assert out["agreed"] == 0
    block = json.loads((tmp_path / "KEY00001.blocks.json").read_text())["blocks"][0]
    assert block["status"] == "single-route" and block["html"] == html and not block["findings"]
    assert block["vlm_second_read"]["agree"] is False
