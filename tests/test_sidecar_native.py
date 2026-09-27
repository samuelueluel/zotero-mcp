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


# Synthetic text-layer lines: (char, font, bbox) with 5 pt wide, 10 pt tall
# glyphs; "|" is a 40 pt column gap.
def _line(text: str, font: str = "Body"):
    out, x = [], 0.0
    for ch in text:
        if ch == "|":
            x += 40
            continue
        out.append((ch, font, (x, 0.0, x + 5, 10.0)))
        x += 5
    return out


def _start(line, text: str) -> int:
    return "".join(c[0] for c in line).rindex(text)


def test_running_text_line_is_prose():
    from zotero_mcp.sidecar_native import is_prose
    assert is_prose(_line("Note: The sample of apartments in Seattle covers the period September 1988"))
    assert not is_prose(_line("Model VI Vacant Tax delinquent Foreclosed|0.252*|0.121"))
    assert not is_prose(_line("Adjusted R2"))


def test_integer_range_end_is_a_label():
    from zotero_mcp.sidecar_native import _range_end
    line = _line("1988-11")
    assert _range_end(line, _start(line, "11"), "11")
    line = _line("0.12-0.35")
    assert not _range_end(line, _start(line, "35"), "35")  # decimal interval stays data
    line = _line("Age|-5")
    assert not _range_end(line, _start(line, "5"), "5")


def test_number_after_word_and_operator_is_a_label():
    from zotero_mcp.sidecar_native import _after_word
    for text, number in (("n = 495", "495"), ("Corporate q < 1", "1"), ("Table 6", "6")):
        line = _line(text)
        assert _after_word(line, _start(line, number)), text
    line = _line("Model I|0.006")
    assert not _after_word(line, _start(line, "0.006"))


BOEHM = REAL_HOME / "Zotero/storage/9I4FUAEH/Boehm-Pandalai-Nayar-2020.pdf"
RIDDEL = REAL_HOME / "Zotero/storage/52PHHZ4D/Riddel-2004.pdf"


def _doc(path: Path):
    if not path.is_file():
        pytest.skip(f"fixture PDF not present: {path.name}")
    return pymupdf.open(path)


def test_minus_fonts_need_a_bar_shaped_glyph():
    from zotero_mcp.sidecar_native import glyph_profile
    diamond = glyph_profile(_doc(DIAMOND))
    assert "AdvTT61dabe7c" in diamond.minus_fonts
    boehm = glyph_profile(_doc(BOEHM))
    # CMEX10 digits are bracket pieces: operators, never a sign.
    assert "CMEX10" in boehm.symbol_fonts and "CMEX10" not in boehm.minus_fonts


def test_paren_minus_font_riddel():
    from zotero_mcp.sidecar_native import glyph_profile
    doc = _doc(RIDDEL)
    profile = glyph_profile(doc)
    assert profile.paren_minus_fonts == {"AdvPSSPS-AS"}
    texts = {t.text for t in native_number_tokens(doc[7], 8, profile=profile) if t.sign_source == "font_glyph"}
    assert {"-0.265", "-1.868"} <= texts


def test_rotated_page_clip_uses_displayed_coordinates():
    from zotero_mcp.sidecar_native import glyph_profile
    doc = _doc(BOEHM)
    page = doc[84]
    assert page.rotation == 90
    toks = native_number_tokens(page, 85, clip=(108, 78, 710, 284), profile=glyph_profile(doc))
    texts = [t.text for t in toks]
    assert len(texts) > 60 and {"5411", "5415", "55"} <= set(texts)


def test_tokens_carry_printed_brackets_and_stars():
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Age   0.512**   (0.034)   [1.5]   7")
    toks = {t.text: t for t in native_number_tokens(page, 1)}
    assert (toks["0.512"].pre, toks["0.512"].post) == ("", "**")
    assert (toks["0.034"].pre, toks["0.034"].post) == ("(", ")")
    assert (toks["1.5"].pre, toks["1.5"].post) == ("[", "]")
    assert (toks["7"].pre, toks["7"].post) == ("", "")


def test_stars_after_a_space_and_after_the_standard_error():
    import pymupdf
    from zotero_mcp.sidecar_native import native_number_tokens

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 50), "26.32 ***", fontsize=10)
    page.insert_text((150, 50), "0.0252", fontsize=10)
    page.insert_text((148, 62), "(0.0141)**", fontsize=10)
    toks = {t.text: t for t in native_number_tokens(page, 1)}
    assert toks["26.32"].stars == "***"
    assert toks["0.0252"].stars == "" and toks["0.0252"].se_stars == "**"
    assert toks["0.0141"].stars == "**"


def test_tex_thin_space_thousands_join_but_word_spaces_do_not():
    from zotero_mcp.sidecar_native import _math_group_sep

    def line(text, space_width):
        chars, x = [], 0.0
        for ch in text:
            w = space_width if ch == " " else 5.0
            chars.append((ch, "cmr10", (x, 0.0, x + w, 10.0)))
            x += w
        return chars

    digit = lambda c: c[0].isdigit()  # noqa: E731
    assert _math_group_sep(line("1, 000", 1.6), 1, ["1"], digit)  # TeX thin space
    assert not _math_group_sep(line("22, 301", 3.3), 2, ["2", "2"], digit)  # word space
    assert not _math_group_sep(line("1, 0000", 1.6), 1, ["1"], digit)  # four digits
    assert not _math_group_sep(line("0.5, 100", 1.6), 3, ["0", ".", "5"], digit)  # decimal first
