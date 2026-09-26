"""Fixture tests for font-aware native numeric tokens.

Uses two real library PDFs whose fonts mis-encode the minus sign. Skipped
when the local Zotero storage is unavailable (CI, other machines).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from zotero_mcp.sidecar_native import (  # noqa: E402
    AUTHORITATIVE_SIGN_SOURCES,
    native_number_tokens,
    symbol_minus_fonts,
    validated_minus_fonts,
)

# conftest points HOME at a scratch dir; the fixture PDFs live in the real one.
REAL_HOME = Path(os.environ.get("ZOTERO_MCP_TEST_REAL_HOME") or Path.home())
HOUSING = REAL_HOME / "Zotero/storage/9SG7BZKC/Rossi-Hansberg et al. 2010. JPE. Housing externalities..pdf"
DIAMOND = REAL_HOME / "Zotero/storage/2XQYTHXM/Diamond-McQuade-2019.pdf"


def _tokens(path: Path, page: int):
    if not path.is_file():
        pytest.skip(f"fixture PDF not present: {path.name}")
    doc = pymupdf.open(path)
    fonts = validated_minus_fonts(symbol_minus_fonts(doc))
    return fonts, native_number_tokens(doc[page - 1], page, minus_fonts=fonts)


def test_control_glyph_minus_housing_table4():
    _fonts, toks = _tokens(HOUSING, 25)
    by_text = {t.text: t for t in toks}
    for value in ("-.059", "-3.453", "-.815", "-37.652"):
        assert value in by_text
        assert by_text[value].sign_source == "control_glyph"
    assert by_text[".094"].negative is False


def test_symbol_font_two_is_minus_diamond_table2():
    fonts, toks = _tokens(DIAMOND, 23)
    assert "AdvTT61dabe7c" in fonts
    assert "AdvPSA33E" not in fonts  # body font draws all ten digits
    texts = {t.text for t in toks}
    assert {"-.0016", "-.0250", ".0640"} <= texts
    assert "2.0016" not in texts


def test_body_font_leading_two_stays_positive():
    _fonts, toks = _tokens(DIAMOND, 49)
    by_text = {t.text: t for t in toks}
    assert by_text["-3,636"].sign_source == "font_glyph"
    assert by_text["-3,636"].sign_source in AUTHORITATIVE_SIGN_SOURCES
    assert "23,403" in by_text and by_text["23,403"].negative is False
