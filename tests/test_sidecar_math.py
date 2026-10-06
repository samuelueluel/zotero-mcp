"""Surya LaTeX against the PDF text layer (sidecar_math)."""

from collections import Counter

import pymupdf

from zotero_mcp import sidecar_math as sm
from zotero_mcp.sidecar_assemble import _math_status_line
from zotero_mcp.sidecar_reliability import SURYA_SOURCE, chunk_reliability


def _page(*lines):
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    for y, text in lines:
        page.insert_text((40, y), text, fontsize=11)
    return doc, page


def _chars(runs):
    """(font, text) runs -> the (font, char-dict) pairs _line_symbols reads."""
    return [(font.lower(), {"c": ch}) for font, text in runs for ch in text]


def test_latex_symbols_map_commands_and_ignore_markup():
    got = sm.latex_symbols(r"\sum_k \frac{\partial \det(\mathbf{X})}{\partial X_{ik}} = \nu^\top")
    assert got == Counter({"X": 2, "k": 2, "∂": 2, "d": 1, "e": 1, "t": 1, "i": 1, "∑": 1, "ν": 1, "T": 1})
    assert sm.latex_symbols(r"\begin{array}{rcl} a & = & b \end{array}") == Counter("ab")


def test_equation_numbers_are_split_off():
    assert sm.split_labels(r"a &= b & (9) \\ c &= d \quad (10)")[1] == ["9", "10"]
    assert sm.split_labels(r"4.14 \quad |x| < 1")[1] == ["4.14"]
    assert sm.split_labels(r"y = f(x) \tag{3.2}")[1] == ["3.2"]
    assert sm.split_labels(r"f(x) = (1)")[1] == []  # no spacing command: part of the equation


def test_text_layer_font_quirks():
    assert sm._line_symbols(_chars([("CMEX10", "PZ"), ("CMMI10", "k")])) == Counter({"∑": 1, "∫": 1, "k": 1})
    assert sm._line_symbols(_chars([("CMSY10", "7→"), ("CMMI10", "x")])) == Counter({"x": 1})  # \mapsto piece
    assert sm._line_symbols(_chars([("CMR10", "IR")])) == Counter({"R": 1})  # blackboard R
    assert sm._line_symbols(_chars([("CMR10", "AIC")])) == Counter("AIC")
    assert sm._line_symbols(_chars([("CMMI10", "ǫ∆")])) == Counter({"ε": 1, "Δ": 1})


def test_equation_symbols_against_the_page():
    _doc, page = _page((60, "x = 1 / k + y"))
    box = (30, 40, 370, 70)
    assert sm.check_equation(r'<math display="block">x = \frac{1}{k} + y</math>', page, box).status == "agree"
    wrong = sm.check_equation(r'<math display="block">x = \frac{3}{k} + y</math>', page, box)
    assert wrong.status == "differ"
    assert wrong.findings[0]["values"] == ["sidecar has: 3", "PDF has: 1"]


def test_equation_without_text_layer_is_not_checked():
    _doc, page = _page()
    assert sm.check_equation(r'<math display="block">x = y + z</math>', page, (30, 40, 370, 70)).status == "no-layer"


class _FakePage:
    """A text layer of (text, y0, y1) lines, each 5 pt per glyph."""
    rotation = 0

    def __init__(self, lines, font="CMMI10"):
        self.lines = lines
        self.font = font

    def get_text(self, kind, clip=None):
        out = []
        for glyphs in self.lines:
            chars = [{"c": c, "bbox": (40 + 5 * i, y0, 45 + 5 * i, y1)} for i, (c, y0, y1) in enumerate(glyphs)]
            box = (40, min(c["bbox"][1] for c in chars), 40 + 5 * len(chars), max(c["bbox"][3] for c in chars))
            out.append({"lines": [{"bbox": box, "spans": [{"font": self.font, "chars": chars}]}]})
        return {"blocks": out}


def test_neighbouring_line_does_not_leak_into_the_box():
    above = [(c, 30, 40) for c in "fQy"] + [("n", 42, 50)]  # "f_Q y_n": only the subscript n dips into the box
    eq = [(c, 50, 60) for c in "x=1/k"]
    page = _FakePage([above, eq])
    html = r'<math display="block">x = \frac{1}{k}</math>'
    assert sm.check_equation(html, page, (30, 44, 370, 62)).status == "agree"
    # A line mostly inside the box counts, but only its glyphs inside the box.
    tall = [(c, 50, 60) for c in "x=1/k"] + [("n", 60, 70)]
    assert sm.check_equation(r'<math display="block">x = \frac{1}{k_n}</math>', _FakePage([tall]),
                             (30, 44, 370, 62)).status == "differ"


def test_inline_math_ignores_case_accents_and_list_markers():
    _doc, page = _page((60, "(a) Schrodinger: let v be FINITE."))
    html = '<ol type="a"><li>Schrödinger: let <math>v</math> be finite.</li></ol>'
    assert sm.check_inline_math(html, page, (30, 40, 370, 70)).status == "agree"
    nu = sm.check_inline_math('<ol type="a"><li>Schrödinger: let <math>\\nu</math> be finite.</li></ol>',
                              page, (30, 40, 370, 70))
    assert nu.status == "differ" and nu.findings[0]["kind"] == "inline_math_symbols_differ"


def test_status_lines_for_math():
    eq = {"page": 7, "findings": [{"kind": "math_symbols_differ", "detail": "",
                                   "values": ["sidecar has: 3", "PDF has: 1"]}]}
    line = _math_status_line(eq)
    assert line == ("[Equation status: UNRESOLVED (symbols differ from the PDF text layer: "
                    "sidecar has: 3; PDF has: 1). Check PDF p. 7 before quoting.]")
    assert chunk_reliability(line, SURYA_SOURCE) == {"block_status": "unresolved", "check_pages": [7]}
    assert _math_status_line({"page": 7, "findings": [{"kind": "math_symbols_agree", "values": []}]}) is None


def test_only_status_lines_lower_a_math_chunk():
    chunk = "<!-- pdf-page: 12 -->\n\nThen\n\n$$\nx = 1\n$$\n"
    assert chunk_reliability(chunk, SURYA_SOURCE) == {"block_status": "verified", "check_pages": []}
    flagged = chunk + "[Math status: SINGLE-ROUTE (inline math or text differs). Check PDF p. 12 before quoting.]"
    assert chunk_reliability(flagged, SURYA_SOURCE) == {"block_status": "single-route", "check_pages": [12]}


def test_glyphs_without_unicode_meaning_disable_the_check():
    assert not sm._trusted("\x12")  # an unmapped glyph
    assert not sm._trusted("ð") and not sm._trusted("ǁ")  # fake delimiters
    assert not sm._trusted("൬")  # Word math bracket in Malayalam
    assert sm._trusted("ﬁ") and sm._trusted("µ") and sm._trusted("ˆ")
    assert sm._trusted("é") and sm._trusted("î")
    page = _FakePage([[(c, 50, 60) for c in "x=\x12+y"]], font="Unknown-Math")
    assert sm.check_equation(r'<math display="block">x = \epsilon + y</math>', page, (30, 44, 370, 62)).status \
        == "no-layer"
    # Knuth's math fonts without a Unicode map: read through TeX's OML and OMS encodings.
    page = _FakePage([[(c, 50, 60) for c in "x=\x12+y"]])
    assert sm.check_equation(r'<math display="block">x = \theta + y</math>', page, (30, 44, 370, 62)).status == "agree"
    assert [sm._decode("cmsy10", c) for c in "\x00jfgp1s"] == list("−|{}√∞∫")
    assert sm._decode("cmsy10", "{") == "{" and sm._decode("mtmi", "\x05") == "\x05"  # mapped / not Knuth's
    assert sm._line_symbols(_chars([("TTdccsc10", "yd")])) == Counter({"Y": 1, "D": 1})  # small capitals
    assert sm._line_symbols(_chars([("NimbusRomNo9L-ReguItal", "IBa")])) == Counter({"B": 1, "a": 1})  # unit ball
    assert sm._line_symbols(_chars([("CMR10", "IBM")])) == Counter("IBM")
    assert sm._line_symbols(_chars([("LMMathExtension10-Regular", "X")])) == Counter({"∑": 1})


def _eq(i, kind):
    status = "unresolved" if kind == "math_symbols_differ" else "single-route"
    return {"id": f"K:p1:b{i}", "label": "Equation", "status": status, "surya_error": False,
            "findings": [{"kind": kind, "detail": "", "values": []}]}


def test_document_gate_drops_disagreements_when_the_layer_is_unreliable():
    from zotero_mcp.sidecar_assemble import gate_math
    tex = [_eq(i, "math_symbols_differ") for i in range(2)] + [_eq(i, "math_symbols_agree") for i in range(2, 10)]
    gate_math(tex)
    assert [b["status"] for b in tex[:2]] == ["unresolved", "unresolved"]
    broken = [_eq(i, "math_symbols_differ") for i in range(3)] + [_eq(i, "math_symbols_agree") for i in range(3, 10)]
    gate_math(broken)
    assert {b["status"] for b in broken} == {"single-route"}
    assert broken[0]["findings"][0]["kind"] == "math_single_route"
    prose = [{"label": "Text", "status": "single-route", "_math_checked": "verified", "surya_error": False,
              "findings": [{"kind": "inline_math_symbols_differ", "detail": "", "values": []}]} for _ in range(10)]
    gate_math(prose)
    assert {b["status"] for b in prose} == {"verified"} and not any(b["findings"] for b in prose)
    from zotero_mcp.sidecar_assemble import repair_math
    repair_math(prose, None)  # nothing left to repair, so no page is opened
    assert not any("_math_checked" in b for b in prose)


def test_unmapped_extension_glyph_may_be_a_big_operator():
    class Page(_FakePage):
        """Glyph "\t" in CMEX10 (a display sum without a Unicode map), the rest in CMMI10."""
        def get_text(self, kind, clip=None):
            out = super().get_text(kind, clip)
            for sp in (s for b in out["blocks"] for ln in b["lines"] for s in ln["spans"]):
                sp["font"] = "CMEX10" if sp["chars"][0]["c"] == "\t" else "CMMI10"
            return out
    box = (30, 44, 370, 62)
    with_op = Page([[("\t", 50, 60)], [(c, 50, 60) for c in "x=ab+i"]])
    layer = sm.layer_math(with_op, box)
    assert layer.unmapped == 1 and layer.untrusted == 0
    assert sm.check_equation(r'<math display="block">x = ab + \sum i</math>', with_op, box).status == "agree"
    without = Page([[(c, 50, 60) for c in "x=ab+i"]])
    assert sm.check_equation(r'<math display="block">x = ab + \sum i</math>', without, box).status == "differ"


def test_relations_are_not_mistaken_for_tags():
    page = _FakePage([[(c, 50, 60) for c in "Lx<a"], [(c, 62, 72) for c in "Ux>b"]])
    html = r'<math display="block">L &amp;= x &lt; a \\ U &amp;= x &gt; b</math>'
    assert sm.check_equation(html, page, (30, 44, 370, 74)).status == "agree"


def test_composed_accents_match_latex_accents():
    page = _FakePage([[(c, 50, 60) for c in "σî=ab"]])
    assert sm.check_equation(r'<math display="block">\sigma_{\hat{i}} = ab</math>', page,
                             (30, 44, 370, 62)).status == "agree"


def test_margin_labels_are_not_matrix_entries_and_script_l_is_skipped():
    assert sm.split_labels(r"a & b \\ 0.0000 & 1.0000")[1] == []
    assert sm.split_labels(r"\\ 11.42 & x")[1] == ["11.42"]
    assert sm.latex_symbols(r"\ell_z") == sm.prose_symbols("ℓz") == Counter("z")


def _differ_eq(latex, layer):
    return {"id": "K:p1:b0", "label": "Equation", "page": 1, "pdf_bbox": (30, 44, 370, 62), "status": "unresolved",
            "surya_error": False, "repairs": [], "html": f'<math display="block">{latex}</math>',
            "findings": [{"kind": "math_symbols_differ", "detail": "", "values": []}]}, [[(c, 50, 60) for c in layer]]


def test_lookalike_symbols_are_restored_from_the_text_layer():
    from zotero_mcp.sidecar_assemble import repair_math
    b, layer = _differ_eq(r"v(A) = \int_A f \, dv", "ν(A)=∫Afdν")
    repair_math([b], [_FakePage(layer)])
    assert b["status"] == "repaired" and b["html"] == r'<math display="block">\nu (A) = \int_A f \, d\nu </math>'
    assert b["surya_html"].endswith("dv</math>") and b["repairs"][0]["route"] == "native_symbol"
    assert _math_status_line(b) == ("[Equation status: REPAIRED (ν where the sidecar read v, from the PDF text "
                                    "layer). Check PDF p. 1 before quoting.]")
    assert chunk_reliability(_math_status_line(b), SURYA_SOURCE)["block_status"] == "repaired"


def test_lookalike_repair_refuses_unsafe_swaps():
    from zotero_mcp.sidecar_assemble import repair_math
    # One v is right and one is wrong: which one is unknown.
    b, layer = _differ_eq(r"v + v", "v+ν")
    repair_math([b], [_FakePage(layer)])
    assert b["status"] == "unresolved"
    # A Latin letter in the layer never overrides Surya's Greek (Symbol fonts store eta at "n").
    b, layer = _differ_eq(r"\eta + x", "n+x")
    repair_math([b], [_FakePage(layer)])
    assert b["status"] == "unresolved"
    # Not a look-alike: a digit error stays flagged.
    b, layer = _differ_eq(r"\frac{3}{k}", "1/k")
    repair_math([b], [_FakePage(layer)])
    assert b["status"] == "unresolved"


def test_empty_set_printed_as_a_letter_is_not_compared():
    assert sm.latex_symbols(r"\mathbb{P}(\emptyset) = 0") == Counter("P0")
    assert sm._line_symbols(_chars([("CMR10", "P(Ø)=0")])) == Counter("P0")
