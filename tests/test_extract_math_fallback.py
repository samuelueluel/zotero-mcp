"""PyMuPDF replaces pdf-inspector text on pages where TeX math glyphs were lost."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pymupdf = pytest.importorskip("pymupdf")

from zotero_mcp import extract  # noqa: E402
from zotero_mcp.pdf_evidence import compile_literal_pattern  # noqa: E402


@pytest.mark.parametrize(
    ("inspector", "reference"),
    [
        ("significant (*p <* 0*:* 001)", "significant (p < 0.001)"),
        ("these estimates are 0.152 and 3.757", "these estimates are −0.152 and 3.757"),
        ("The estimate of is precise and of is large", "The estimate of β is precise and δ is large"),
        ("strongly signicant", "strongly signiﬁcant"),
    ],
)
def test_damage_detected(inspector, reference):
    assert extract._math_font_damage(inspector, reference)


@pytest.mark.parametrize(
    ("inspector", "reference"),
    [
        ("a clean page, p < 0.001, -0.152", "a clean page, p < 0.001, −0.152"),
        ("strongly significant", "strongly signiﬁcant"),
        ("time 10:30 and ratio 3:1", "time 10:30 and ratio 3:1"),
    ],
)
def test_clean_page_not_flagged(inspector, reference):
    assert not extract._math_font_damage(inspector, reference)


def test_extract_pdf_swaps_damaged_page_only(tmp_path, monkeypatch):
    path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for text in ("estimates are 0.152 and 3.757", "plain second page"):
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    fake = SimpleNamespace(
        extract_pages_markdown=lambda p, pages=None: SimpleNamespace(pages=[
            SimpleNamespace(page=0, markdown="estimates are 0*:* 152 and 3*:* 757", needs_ocr=False),
            SimpleNamespace(page=1, markdown="**plain second page**", needs_ocr=False),
        ]),
        classify_pdf=lambda p: SimpleNamespace(page_count=2),
    )
    monkeypatch.setattr(extract, "_pdf_inspector", lambda: fake)
    out = extract.extract_pdf(path)
    assert out.fallback_pages == (0,)
    assert "0.152 and 3.757" in out.pages[0]
    assert out.pages[1] == "**plain second page**"


def test_search_folds_minus_variants_only():
    pattern = compile_literal_pattern("-0.152")
    assert pattern.search("are −0.152 and")
    assert pattern.search("are –0.152 and")
    assert not pattern.search("are 0.152 and")
