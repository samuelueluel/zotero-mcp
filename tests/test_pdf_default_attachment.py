"""PDF tools default to the attachment the item's sidecar was built from."""

from __future__ import annotations

import json

from zotero_mcp import sidecar_reliability
from zotero_mcp.pdf_evidence import PdfEvidenceInputError
from zotero_mcp.tools import read_pdf


def _stub(monkeypatch, recorded, resolvable):
    calls = []

    def resolve(item_key, ctx, attachment_key=None):
        calls.append(attachment_key)
        if attachment_key and attachment_key not in resolvable:
            raise PdfEvidenceInputError("not a PDF child")
        return ("/x.pdf", "t", False, attachment_key or "FIRST")

    monkeypatch.setattr(read_pdf, "_sidecar_attachment", lambda key: recorded)
    monkeypatch.setattr(read_pdf, "_resolve_pdf_path", resolve)
    return calls


def test_default_uses_sidecar_attachment(monkeypatch):
    calls = _stub(monkeypatch, "MAINPAPR", {"MAINPAPR"})
    assert read_pdf._get_pdf_path("ITEMKEY1", None)[3] == "MAINPAPR"
    assert calls == ["MAINPAPR"]


def test_explicit_attachment_wins(monkeypatch):
    calls = _stub(monkeypatch, "MAINPAPR", {"MAINPAPR", "APPENDIX"})
    assert read_pdf._get_pdf_path("ITEMKEY1", None, "APPENDIX")[3] == "APPENDIX"
    assert calls == ["APPENDIX"]


def test_missing_recorded_attachment_falls_back(monkeypatch):
    calls = _stub(monkeypatch, "GONE0000", set())
    assert read_pdf._get_pdf_path("ITEMKEY1", None)[3] == "FIRST"
    assert calls == ["GONE0000", None]


def test_reliability_reports_attachment_key(tmp_path):
    (tmp_path / "ITEMKEY1.reliability.json").write_text(
        json.dumps({"level": "warn", "attachment_key": "mainpapr"}), encoding="utf-8"
    )
    assert sidecar_reliability.sidecar_attachment_key(tmp_path, "ITEMKEY1") == "MAINPAPR"
    item_rel = sidecar_reliability.load_item_reliability(tmp_path, "ITEMKEY1")
    assert sidecar_reliability.result_reliability({}, item_rel)["attachment_key"] == "mainpapr"
