"""Hermetic tests for bounded PDF evidence primitives and tool wrappers."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest
from conftest import DummyContext
from fastmcp import Client

from zotero_mcp import server
from zotero_mcp.extract import ExtractedDoc
from zotero_mcp.pdf_evidence import (
    MAX_PDF_RENDER_PIXELS,
    MAX_PDF_SEARCH_PAGES,
    PdfEvidenceInputError,
    classify_text_coverage,
    find_literal_matches,
    normalized_region_to_rect,
    render_page_to_png,
    validate_normalized_region,
)
from zotero_mcp.tools import read_pdf as read_pdf_tools


def _doc(pages, *, needs_ocr=()):
    pages = tuple(pages)
    return ExtractedDoc(
        text="\f".join(pages),
        pages=pages,
        page_numbers=tuple(range(len(pages))),
        page_count=len(pages),
        source="pdf",
        needs_ocr=tuple(needs_ocr),
    )


def _write_pdf(path: Path, pages: list[str], *, rotation: int = 0) -> Path:
    fitz = pytest.importorskip("pymupdf")
    document = fitz.open()
    for body in pages:
        page = document.new_page(width=200, height=100)
        page.insert_text((10, 30), body, fontsize=12)
        if rotation:
            page.set_rotation(rotation)
    document.save(str(path))
    document.close()
    return path


class TestCoverage:
    def test_complete_partial_and_no_usable_states(self):
        assert classify_text_coverage(_doc(["one", "two"]))["state"] == "complete"
        assert classify_text_coverage(_doc(["one", "", "three"], needs_ocr=(1,)))["state"] == "partial_text_coverage"
        assert classify_text_coverage(_doc(["", ""], needs_ocr=(0, 1)))["state"] == "no_usable_text"

    def test_coverage_uses_one_based_pdf_pages(self):
        result = classify_text_coverage(_doc(["text", ""], needs_ocr=(1,)))
        assert result["requested_pages"] == [1, 2]
        assert result["usable_text_pages"] == [1]
        assert result["pages_needing_ocr"] == [2]


class TestLiteralSearch:
    def test_case_insensitive_whitespace_tolerant_and_literal(self):
        result = find_literal_matches(
            _doc(["prefix A+B\nvalue suffix", "other"]),
            "a+b value",
            context_chars=0,
        )
        assert result["total_matches"] == 1
        match = result["matches"][0]
        assert match["page"] == 1
        assert match["match_text"] == "A+B\nvalue"
        assert match["match_text"] in match["text"]

    def test_total_count_is_not_capped_by_returned_matches(self):
        result = find_literal_matches(
            _doc(["needle needle", "needle", "needle"]),
            "needle",
            max_matches=2,
            context_chars=0,
        )
        assert result["total_matches"] == 4
        assert result["returned_matches"] == 2
        assert result["has_more_matches"] is True
        assert result["match_pages"] == [1, 2, 3]
        assert result["returned_pages"] == [1]
        assert result["omitted_match_pages"] == [2, 3]
        assert result["next_offset"] == 2

    def test_match_offset_paginates_without_losing_complete_page_summary(self):
        result = find_literal_matches(
            _doc(["needle needle", "needle", "needle"]),
            "needle",
            max_matches=2,
            match_offset=2,
            context_chars=0,
        )
        assert result["total_matches"] == 4
        assert result["match_offset"] == 2
        assert [match["match_index"] for match in result["matches"]] == [2, 3]
        assert [match["page"] for match in result["matches"]] == [2, 3]
        assert result["match_pages"] == [1, 2, 3]
        assert result["omitted_match_pages"] == [1]
        assert result["has_more_matches"] is False
        assert result["next_offset"] is None

    def test_ocr_pages_are_not_searched(self):
        result = find_literal_matches(
            _doc(["needle", "needle"], needs_ocr=(1,)),
            "needle",
            context_chars=0,
        )
        assert result["total_matches"] == 1
        assert result["matches"][0]["page"] == 1

    def test_excerpt_is_verbatim_and_within_total_budget(self):
        raw = "before\n" + ("x" * 300) + " needle " + ("y" * 300) + "\nafter"
        result = find_literal_matches(
            _doc([raw]), "needle", max_chars=256, context_chars=4000
        )
        excerpt = result["matches"][0]["text"]
        assert len(excerpt) <= 256
        assert " needle " in excerpt
        assert raw[raw.index("needle") : raw.index("needle") + 6] == "needle"

    def test_page_numbers_are_absolute_when_subset_is_supplied(self):
        result = find_literal_matches(
            ["match"], "match", page_numbers=[7], context_chars=0
        )
        assert result["matches"][0]["page"] == 8
        assert result["matches"][0]["page_index"] == 7

    def test_rejects_empty_query_and_bounds(self):
        with pytest.raises(PdfEvidenceInputError):
            find_literal_matches(_doc(["text"]), "   ")
        with pytest.raises(ValueError):
            find_literal_matches(_doc(["text"]), "x", max_matches=11)
        with pytest.raises(ValueError):
            find_literal_matches(_doc(["text"]), "x", max_chars=255)


class TestRegionsAndRendering:
    def test_region_validation_never_repairs_input(self):
        assert validate_normalized_region([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)
        for region in ([], [0, 0, 0, 1], [0, 0, 1.1, 1], [float("nan"), 0, 1, 1]):
            with pytest.raises(ValueError):
                validate_normalized_region(region)

    def test_normalized_region_matches_visible_page_geometry(self):
        fitz = pytest.importorskip("pymupdf")
        rect, normalized = normalized_region_to_rect(
            fitz.Rect(10, 20, 210, 120), [0.1, 0.2, 0.3, 0.4]
        )
        assert normalized == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert rect == fitz.Rect(30, 40, 90, 80)

    def test_render_returns_png_and_region_dimensions(self, tmp_path):
        path = _write_pdf(tmp_path / "paper.pdf", ["page one"])
        full = render_page_to_png(str(path), 1, dpi=72)
        crop = render_page_to_png(str(path), 1, region=[0.25, 0.1, 0.5, 0.5], dpi=72)
        assert full.png.startswith(b"\x89PNG\r\n\x1a\n")
        assert (full.width, full.height) == (200, 100)
        assert (crop.width, crop.height) == (100, 50)
        assert crop.rendered_region == [0.25, 0.1, 0.5, 0.5]

    def test_render_handles_rotation(self, tmp_path):
        path = _write_pdf(tmp_path / "rotated.pdf", ["rotated"], rotation=90)
        rendered = render_page_to_png(str(path), 1, dpi=72)
        assert (rendered.width, rendered.height) == (100, 200)
        assert rendered.png.startswith(b"\x89PNG")

    def test_render_closes_document_on_limit_failure(self, tmp_path, monkeypatch):
        path = _write_pdf(tmp_path / "paper.pdf", ["page one"])
        monkeypatch.setattr(
            "zotero_mcp.pdf_evidence.MAX_PDF_RENDER_PIXELS", 1
        )
        with pytest.raises(ValueError, match="pixels"):
            render_page_to_png(str(path), 1, dpi=72)
        assert path.exists()
        # Restore the module constant for other tests in this process.
        monkeypatch.setattr(
            "zotero_mcp.pdf_evidence.MAX_PDF_RENDER_PIXELS", MAX_PDF_RENDER_PIXELS
        )

    def test_render_rejects_dpi_and_encoded_size_overflows(self, tmp_path, monkeypatch):
        path = _write_pdf(tmp_path / "paper.pdf", ["page one"])
        with pytest.raises(ValueError, match="dpi"):
            render_page_to_png(str(path), 1, dpi=301)
        monkeypatch.setattr("zotero_mcp.pdf_evidence.MAX_PDF_RENDER_PNG_BYTES", 1)
        with pytest.raises(ValueError, match="PNG"):
            render_page_to_png(str(path), 1, dpi=72)
        assert path.exists()


class TestFindInPdfWrapper:
    def test_wrapper_uses_extraction_seam_and_reports_coverage(self, monkeypatch):
        monkeypatch.setattr(
            read_pdf_tools,
            "_get_pdf_path",
            lambda _key, _ctx, _a=None: ("/tmp/paper.pdf", "Paper", False, "ATTACH01"),
        )
        monkeypatch.setattr(read_pdf_tools, "pdf_page_count", lambda _path: 2)
        monkeypatch.setattr(
            read_pdf_tools,
            "extract_pdf",
            lambda _path, **_kwargs: _doc(["needle", ""], needs_ocr=(1,)),
        )
        result = json.loads(
            read_pdf_tools.find_in_pdf("ITEM", "needle", ctx=DummyContext())
        )
        assert result["page_range"] == {"start": 1, "end": 2}
        assert result["coverage"] == "partial_text_coverage"
        assert result["matches"][0]["page"] == 1
        assert result["match_pages"] == [1]
        assert result["returned_pages"] == [1]
        assert result["omitted_match_pages"] == []
        assert result["offset"] == 0
        assert result["extraction_engine"] == "pdf-inspector"

    def test_wrapper_exposes_late_match_pages_and_offset_pagination(self, monkeypatch):
        monkeypatch.setattr(
            read_pdf_tools,
            "_get_pdf_path",
            lambda _key, _ctx, _a=None: ("/tmp/paper.pdf", "Paper", False, "ATTACH01"),
        )
        monkeypatch.setattr(read_pdf_tools, "pdf_page_count", lambda _path: 3)
        monkeypatch.setattr(
            read_pdf_tools,
            "extract_pdf",
            lambda _path, **_kwargs: _doc(["needle needle", "needle", "needle"]),
        )
        first = json.loads(
            read_pdf_tools.find_in_pdf(
                "ITEM", "needle", max_matches=2, ctx=DummyContext()
            )
        )
        assert first["match_pages"] == [1, 2, 3]
        assert first["omitted_match_pages"] == [2, 3]
        assert first["next_offset"] == 2

        second = json.loads(
            read_pdf_tools.find_in_pdf(
                "ITEM", "needle", max_matches=2, offset=2, ctx=DummyContext()
            )
        )
        assert [match["page"] for match in second["matches"]] == [2, 3]
        assert second["has_more_matches"] is False

    def test_wrapper_rejects_page_ranges_over_limit(self, monkeypatch):
        monkeypatch.setattr(
            read_pdf_tools,
            "_get_pdf_path",
            lambda _key, _ctx, _a=None: ("/tmp/paper.pdf", "Paper", False, "ATTACH01"),
        )
        monkeypatch.setattr(read_pdf_tools, "pdf_page_count", lambda _path: MAX_PDF_SEARCH_PAGES + 1)
        result = json.loads(
            read_pdf_tools.find_in_pdf(
                "ITEM", "needle", start_page=1, end_page=MAX_PDF_SEARCH_PAGES + 1, ctx=DummyContext()
            )
        )
        assert result["ok"] is False
        assert "limit" in result["error"]["message"]

    def test_wrapper_does_not_claim_absence_on_unusable_pages(self, monkeypatch):
        monkeypatch.setattr(
            read_pdf_tools,
            "_get_pdf_path",
            lambda _key, _ctx, _a=None: ("/tmp/paper.pdf", "Paper", False, "ATTACH01"),
        )
        monkeypatch.setattr(read_pdf_tools, "pdf_page_count", lambda _path: 2)
        monkeypatch.setattr(
            read_pdf_tools,
            "extract_pdf",
            lambda _path, **_kwargs: _doc(["text layer", ""], needs_ocr=(1,)),
        )
        result = json.loads(
            read_pdf_tools.find_in_pdf("ITEM", "missing", ctx=DummyContext())
        )
        assert result["total_matches"] == 0
        assert result["coverage"] == "partial_text_coverage"
        assert "absence cannot be established" in result["message"]

    def test_temp_source_is_cleaned_when_extraction_fails(self, monkeypatch, tmp_path):
        pdf = tmp_path / "downloaded.pdf"
        pdf.write_bytes(b"not a usable pdf")
        removed = []
        monkeypatch.setattr(
            read_pdf_tools,
            "_get_pdf_path",
            lambda _key, _ctx, _a=None: (str(pdf), "Paper", True, "ATTACH01"),
        )
        monkeypatch.setattr(read_pdf_tools, "_cleanup_path", removed.append)
        monkeypatch.setattr(
            read_pdf_tools, "pdf_page_count", lambda _path: (_ for _ in ()).throw(ValueError("corrupt"))
        )
        result = json.loads(
            read_pdf_tools.find_in_pdf("ITEM", "needle", ctx=DummyContext())
        )
        assert result["ok"] is False
        assert removed == [str(pdf)]


class TestRenderWrapper:
    def test_success_contains_one_text_and_one_image_block(self, monkeypatch):
        rendered = type(
            "Rendered",
            (),
            {
                "page": 2,
                "dpi": 72.0,
                "requested_region": [0.1, 0.2, 0.3, 0.4],
                "rendered_region": [0.1, 0.2, 0.3, 0.4],
                "width": 60,
                "height": 40,
                "pixels": 2400,
                "png": b"\x89PNG fake",
            },
        )()
        monkeypatch.setattr(
            read_pdf_tools,
            "_get_pdf_path",
            lambda _key, _ctx, _a=None: ("/tmp/paper.pdf", "Paper", False, "ATTACH01"),
        )
        monkeypatch.setattr(read_pdf_tools, "render_page_to_png", lambda *_a, **_k: rendered)
        result = read_pdf_tools.render_pdf_page(
            "ITEM", 2, region=[0.1, 0.2, 0.3, 0.4], dpi=72, ctx=DummyContext()
        )
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[1].type == "image"
        assert result.content[1].mimeType == "image/png"
        assert base64.b64decode(result.content[1].data) == b"\x89PNG fake"
        assert result.structured_content["page"] == 2
        assert result.structured_content["width"] == 60

    def test_local_pdf_is_not_cleaned(self, monkeypatch):
        removed = []
        monkeypatch.setattr(read_pdf_tools, "_cleanup_path", removed.append)
        rendered = type(
            "Rendered", (), {
                "page": 1, "dpi": 72.0, "requested_region": None,
                "rendered_region": [0.0, 0.0, 1.0, 1.0],
                "width": 10, "height": 10, "pixels": 100, "png": b"png",
            }
        )()
        monkeypatch.setattr(read_pdf_tools, "_get_pdf_path", lambda *_a: ("/library/paper.pdf", "P", False, "ATTACH01"))
        monkeypatch.setattr(read_pdf_tools, "render_page_to_png", lambda *_a, **_k: rendered)
        read_pdf_tools.render_pdf_page("ITEM", 1, ctx=DummyContext())
        assert removed == []

    def test_temp_pdf_is_cleaned_after_render_failure(self, monkeypatch, tmp_path):
        pdf = tmp_path / "downloaded.pdf"
        pdf.write_bytes(b"not a usable pdf")
        removed = []
        monkeypatch.setattr(
            read_pdf_tools,
            "_get_pdf_path",
            lambda _key, _ctx, _a=None: (str(pdf), "Paper", True, "ATTACH01"),
        )
        monkeypatch.setattr(read_pdf_tools, "_cleanup_path", removed.append)
        monkeypatch.setattr(
            read_pdf_tools,
            "render_page_to_png",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad PDF")),
        )
        result = read_pdf_tools.render_pdf_page("ITEM", 1, ctx=DummyContext())
        assert "bad PDF" in result
        assert removed == [str(pdf)]

    def test_registered_tool_delivers_actual_image_content(self, monkeypatch, tmp_path):
        path = _write_pdf(tmp_path / "paper.pdf", ["visible page"])
        monkeypatch.setattr(
            read_pdf_tools,
            "_get_pdf_path",
            lambda _key, _ctx, _a=None: (str(path), "Paper", False, "ATTACH01"),
        )

        async def call():
            async with Client(server.mcp) as client:
                return await client.call_tool_mcp(
                    "render_pdf_page", {"item_key": "ITEM", "page": 1, "dpi": 72}
                )

        result = asyncio.run(call())
        assert [block.type for block in result.content] == ["text", "image"]
        image = result.content[1]
        assert image.mimeType == "image/png"
        assert base64.b64decode(image.data).startswith(b"\x89PNG\r\n\x1a\n")
        assert result.structuredContent["page"] == 1
        assert result.structuredContent["width"] == 200


def test_find_literal_matches_deduplicates_identical_clamped_windows():
    """Regression (2026-09-15 Detroit run): several matches clamped to the same
    short-page window each returned the full excerpt. Later matches in an
    already-returned window now carry an empty excerpt plus a marker while
    per-match accounting stays complete."""
    page = "alpha NEEDLE beta NEEDLE gamma NEEDLE delta"
    result = find_literal_matches(
        [page],
        "needle",
        page_numbers=[0],
        needs_ocr=[],
        max_matches=10,
        max_chars=4096,
        context_chars=0,
    )
    assert result["total_matches"] == 3
    assert result["returned_matches"] == 3
    first = result["matches"][0]
    assert first["text"] and not first.get("duplicate_window")
    windows = {(m["excerpt_char_start"], m["excerpt_char_end"]) for m in result["matches"]}
    assert len(windows) == 1
    for duplicate in result["matches"][1:]:
        assert duplicate["text"] == ""
        assert duplicate["excerpt"] == ""
        assert duplicate["duplicate_window"] is True
    assert result["source_chars_returned"] == len(first["text"])
