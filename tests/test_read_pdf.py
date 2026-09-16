"""Tests for bounded PDF tools."""

import json
import tempfile

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp import server
from zotero_mcp.extract import PAGE_SEPARATOR, ExtractedDoc
from zotero_mcp.tools import read_pdf as read_pdf_tools

# ---------------------------------------------------------------------------
# Helpers: fake the extraction seam
# ---------------------------------------------------------------------------


def _patch_extract(monkeypatch, page_texts, total=None, needs_ocr=()):
    """Stand in for ``extract_pdf``/``pdf_page_count`` with known page text.

    Mirrors the real contract the tool depends on: out-of-range indices are
    dropped, and ``page_numbers`` reports the absolute source page for each
    returned page.
    """
    total_pages = total if total is not None else len(page_texts)

    def _fake_page_count(_path):
        return total_pages

    def _fake_extract_pdf(_path, *, pages=None, max_pages=None):
        wanted = [
            p for p in (range(total_pages) if pages is None else pages)
            if 0 <= p < total_pages
        ]
        texts = [page_texts[p % len(page_texts)] for p in wanted]
        return ExtractedDoc(
            text=PAGE_SEPARATOR.join(texts),
            pages=tuple(texts),
            page_numbers=tuple(wanted),
            page_count=total_pages,
            source="pdf",
            needs_ocr=tuple(needs_ocr),
        )

    monkeypatch.setattr("zotero_mcp.tools.read_pdf.pdf_page_count", _fake_page_count)
    monkeypatch.setattr("zotero_mcp.tools.read_pdf.extract_pdf", _fake_extract_pdf)


def _patch_extract_failure(monkeypatch, exc):
    """Make the seam raise, as it does for a corrupt or non-PDF file."""
    def _boom(_path):
        raise exc

    monkeypatch.setattr("zotero_mcp.tools.read_pdf.pdf_page_count", _boom)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_ctx():
    return DummyContext()


@pytest.fixture
def fake_zot():
    return FakeZotero()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Single page and page range reads."""

    def test_single_page(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["Page 1 content."] * 10, total=10)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Test Paper", True, "ATTACH01"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=3, ctx=dummy_ctx)

        assert "## Page 3" in result
        assert "Page 1 content." in result

    def test_page_range(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, [
            "Content of page 1.",
            "Content of page 2.",
            "Content of page 3.",
            "Content of page 4.",
            "Content of page 5.",
        ])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Test Paper", True, "ATTACH01"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=2, end_page=4, ctx=dummy_ctx)

        assert "## Page 2" in result
        assert "Content of page 2." in result
        assert "## Page 3" in result
        assert "Content of page 3." in result
        assert "## Page 4" in result
        assert "Content of page 4." in result
        assert "## Page 1" not in result
        assert "## Page 5" not in result

    def test_header_contains_metadata(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["hello"])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "My Paper Title", True, "ATTACH01"),
        )

        result = server.read_pdf_pages(item_key="KEY123", start_page=1, ctx=dummy_ctx)

        assert "# PDF Pages 1-1 of My Paper Title" in result
        assert "**Item Key:** KEY123" in result
        assert "**Total pages in PDF:** 1" in result


class TestErrors:
    """Input validation and error cases."""

    def test_empty_item_key(self, dummy_ctx):
        result = server.read_pdf_pages(item_key="", start_page=1, ctx=dummy_ctx)
        assert "item_key cannot be empty" in result

    def test_whitespace_item_key(self, dummy_ctx):
        result = server.read_pdf_pages(item_key="   ", start_page=1, ctx=dummy_ctx)
        assert "item_key cannot be empty" in result

    def test_end_page_less_than_start_page(self, dummy_ctx):
        result = server.read_pdf_pages(item_key="ITEM01", start_page=5, end_page=3, ctx=dummy_ctx)
        assert "end_page must be greater than or equal to start_page" in result

    def test_no_pdf_attachment(self, monkeypatch, dummy_ctx, fake_zot):
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: None,
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, ctx=dummy_ctx)

        assert "No PDF attachment found" in result

    def test_start_page_out_of_range(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["p1"], total=1)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Paper", True, "ATTACH01"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=5, ctx=dummy_ctx)

        assert "out of range" in result
        assert "1-1" in result

    def test_end_page_out_of_range(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["p1"] * 3, total=3)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Paper", True, "ATTACH01"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, end_page=10, ctx=dummy_ctx)

        assert "out of range" in result
        assert "1-3" in result

    def test_too_many_pages(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["p"] * 100, total=100)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Paper", True, "ATTACH01"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, end_page=55, ctx=dummy_ctx)

        assert "max 50" in result

    def test_unreadable_pdf_reports_the_reason(self, monkeypatch, dummy_ctx, fake_zot):
        """A corrupt or non-PDF file surfaces the parser's message rather
        than an empty page range."""
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Paper", True, "ATTACH01"),
        )
        _patch_extract_failure(monkeypatch, ValueError("Not a PDF: file is empty"))

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, ctx=dummy_ctx)

        assert "Could not read PDF" in result
        assert "Not a PDF" in result


class TestEdgeCases:
    """Edge case behaviors."""

    def test_end_page_equals_start_page(self, monkeypatch, dummy_ctx, fake_zot):
        """When end_page == start_page, should behave like single page."""
        _patch_extract(monkeypatch, ["p1", "p2", "p3"])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Test Paper", True, "ATTACH01"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=2, end_page=2, ctx=dummy_ctx)

        assert "## Page 2" in result
        assert "## Page 1" not in result
        assert "## Page 3" not in result

    def test_reads_last_page(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["first", "last"])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Paper", True, "ATTACH01"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=2, ctx=dummy_ctx)

        assert "## Page 2" in result
        assert "last" in result

    def test_empty_page_text(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["", "has text", ""])
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Paper", True, "ATTACH01"),
        )

        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, end_page=3, ctx=dummy_ctx)

        assert "[No extractable text on this page]" in result
        assert "has text" in result


class TestFindInPdf:
    """The literal PDF lookup stays on the direct extraction seam."""

    def test_search_returns_one_based_pages_and_exact_counts(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["before Needle after", "needle again"], total=2)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Test Paper", False, "ATTACH01"),
        )

        result = json.loads(
            server.find_in_pdf(
                item_key="ITEM01",
                query="needle",
                start_page=1,
                end_page=2,
                max_matches=1,
                context_chars=0,
                ctx=dummy_ctx,
            )
        )

        assert result["route"] == "pdf_extraction"
        assert result["page_range"] == {"start": 1, "end": 2}
        assert result["total_matches"] == 2
        assert result["returned_matches"] == 1
        assert result["has_more_matches"] is True
        assert result["matches"][0]["page"] == 1

    def test_invalid_query_is_rejected_before_source_lookup(self, monkeypatch, dummy_ctx):
        called = []
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda *_args: called.append(True),
        )
        result = json.loads(
            server.find_in_pdf(item_key="ITEM01", query="  ", ctx=dummy_ctx)
        )
        assert result["error"]["code"] == "INVALID_ARGUMENT"
        assert called == []


class TestCleanupPathSafety:
    """`_cleanup_path` deletes a directory, so its guards have to be tight.

    The original guard was `parent.startswith(tempfile.gettempdir())`. On
    Linux that is `/tmp`, so `_cleanup_path("/tmp/test.pdf")` resolved its
    parent to `/tmp` and called `shutil.rmtree("/tmp")` — wiping the system
    temp directory, including pytest's own temp root, which surfaced as
    unrelated tests erroring with FileNotFoundError. macOS never showed it
    because `gettempdir()` there lives under `/var/folders`.
    """

    def test_never_removes_the_temp_root_itself(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        canary = tmp_path / "canary.txt"
        canary.write_text("do not delete me")

        read_pdf_tools._cleanup_path(str(tmp_path / "test.pdf"))

        assert tmp_path.exists()
        assert canary.exists()

    def test_ignores_directories_we_did_not_create(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        storage = tmp_path / "storage" / "ABCD1234"
        storage.mkdir(parents=True)
        pdf = storage / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        read_pdf_tools._cleanup_path(str(pdf))

        assert pdf.exists(), "a file in the user's Zotero storage must survive"

    def test_removes_our_own_download_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        owned = tmp_path / "zotero_pdf_abc123"
        owned.mkdir()
        pdf = owned / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        read_pdf_tools._cleanup_path(str(pdf))

        assert not owned.exists()

    def test_library_file_is_not_released_by_the_tool(self, monkeypatch, dummy_ctx, fake_zot):
        """A local-storage hit reports is_temp=False and must survive the read."""
        _patch_extract(monkeypatch, ["Body text."], total=1)
        removed = []
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._cleanup_path", lambda p: removed.append(p)
        )
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/home/me/Zotero/storage/ABCD/paper.pdf", "Paper", False, "ATTACH01"),
        )

        server.read_pdf_pages(item_key="ITEM01", start_page=1, ctx=dummy_ctx)

        assert removed == []


class TestAttachmentProvenance:
    """Multi-PDF items must name which attachment served the read."""

    def test_find_in_pdf_echoes_default_attachment_selection(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["before Needle after"], total=1)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Test Paper", False, "ATTACH01"),
        )
        result = json.loads(
            server.find_in_pdf(item_key="ITEM01", query="needle", ctx=dummy_ctx)
        )
        assert result["attachment_key"] == "ATTACH01"
        assert result["attachment_selection"] == "default"

    def test_find_in_pdf_echoes_explicit_attachment_selection(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["before Needle after"], total=1)
        captured = {}

        def _fake_resolver(item_key, ctx, attachment_key=None):
            captured["key"] = attachment_key
            return ("/tmp/test.pdf", "Test Paper", False, "ATTACH02")

        monkeypatch.setattr("zotero_mcp.tools.read_pdf._get_pdf_path", _fake_resolver)
        result = json.loads(
            server.find_in_pdf(
                item_key="ITEM01", query="needle", attachment_key="ATTACH02", ctx=dummy_ctx
            )
        )
        assert captured["key"] == "ATTACH02"
        assert result["attachment_key"] == "ATTACH02"
        assert result["attachment_selection"] == "explicit"

    def test_read_pdf_pages_marks_resolved_attachment(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["Page 1 content."], total=1)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Test Paper", False, "ATTACH01"),
        )
        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, ctx=dummy_ctx)
        assert "**Attachment:** ATTACH01" in result

    def test_explicit_attachment_mismatch_is_a_bounded_input_error(self, monkeypatch, dummy_ctx, fake_zot):
        def _raise(_key, _ctx, _attachment=None):
            raise read_pdf_tools.PdfEvidenceInputError(
                "attachment ATTACH09 is not a PDF child of item ITEM01; "
                "pass the key of one of the item's PDF attachments or omit "
                "attachment_key to use the default PDF"
            )

        monkeypatch.setattr("zotero_mcp.tools.read_pdf._get_pdf_path", _raise)
        result = json.loads(
            server.find_in_pdf(item_key="ITEM01", query="needle", ctx=dummy_ctx)
        )
        assert result["error"]["code"] == "INVALID_ARGUMENT"
        assert "ATTACH09" in result["error"]["message"]

    def test_find_in_pdf_emits_clickable_locators_and_uris(self, monkeypatch, dummy_ctx, fake_zot):
        _patch_extract(monkeypatch, ["before Needle after"], total=1)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Test Paper", False, "ATTACH01"),
        )
        result = json.loads(
            server.find_in_pdf(item_key="ITEM01", query="needle", ctx=dummy_ctx)
        )
        assert result["zotero_select_uri"] == "zotero://select/library/items/ITEM01"
        assert result["zotero_open_pdf_uri"] == "zotero://open-pdf/library/items/ATTACH01?page=1"
        assert result["page_locators"]["1"] == "[PDF p. 1](zotero://open-pdf/library/items/ATTACH01?page=1)"
        assert (
            result["matches"][0]["locator"]
            == "[PDF p. 1](zotero://open-pdf/library/items/ATTACH01?page=1)"
        )

    def test_read_pdf_pages_emits_clickable_heading_and_attachment_links(
        self, monkeypatch, dummy_ctx, fake_zot
    ):
        _patch_extract(monkeypatch, ["Page 1 content."], total=1)
        monkeypatch.setattr(
            "zotero_mcp.tools.read_pdf._get_pdf_path",
            lambda _k, _c, _a=None: ("/tmp/test.pdf", "Test Paper", False, "ATTACH01"),
        )
        result = server.read_pdf_pages(item_key="ITEM01", start_page=1, ctx=dummy_ctx)
        assert (
            "[Open in Zotero Reader](zotero://open-pdf/library/items/ATTACH01?page=1)"
            in result
        )
        assert (
            "## Page 1 ([PDF p. 1](zotero://open-pdf/library/items/ATTACH01?page=1))"
            in result
        )

    def test_get_attachment_paths_emits_deep_links(self, monkeypatch, dummy_ctx):
        from pathlib import Path

        class FakeReader:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def get_attachment_paths(self, item_key):
                return [
                    {
                        "key": "ATT01",
                        "content_type": "application/pdf",
                        "zotero_path": "storage:paper.pdf",
                        "resolved_path": Path("/storage/ATT01/paper.pdf"),
                        "exists": True,
                    }
                ]

        monkeypatch.setattr("zotero_mcp.tools.retrieval._utils.is_local_mode", lambda: True)
        monkeypatch.setattr("zotero_mcp.local_db.LocalZoteroReader", FakeReader)
        from zotero_mcp.tools import retrieval

        result = retrieval.get_attachment_paths(item_key="ITEM01", ctx=dummy_ctx)
        assert "[View in Library](zotero://select/library/items/ITEM01)" in result
        assert "[Open PDF](zotero://open-pdf/library/items/ATT01?page=1)" in result
