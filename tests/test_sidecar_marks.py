"""Primes, tildes and accents against the text layer, Stata row-shape fixes, and re-read words."""

import re
from collections import Counter

import pymupdf

from zotero_mcp import sidecar_math as sm
from zotero_mcp.sidecar_assemble import _math_status_line, restore_marks
from zotero_mcp.sidecar_repair import reconcile_rewrite
from zotero_mcp.sidecar_verify import fix_table_structure


def _page(*lines):
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    for y, text in lines:
        page.insert_text((40, y), text, fontsize=11)
    return doc, page


def _chars(runs):
    return [(font.lower(), {"c": ch}) for font, text in runs for ch in text]


def _rows(html):
    return [[re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
            for r in re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, re.S)]


def test_marks_are_counted_on_both_sides():
    assert sm.latex_marks(r"X'_t \beta + \tilde{x} + y^{\prime\prime} \sim N") == Counter(
        prime=3, tilde=2, **{"tilde:x": 1})
    assert sm.latex_marks(r"a~b \text{it's}") == Counter()  # ~ is a space in LaTeX
    assert sm.latex_marks(r"A \cong B \simeq C") == Counter(tilde=2)  # the layer may print "∼="
    assert sm.prose_marks("Muñoz, ln_populat~n, Jensen's") == Counter(tilde=2, **{"tilde:n": 1})
    assert sm._line_marks(_chars([("CMSY10", "0"), ("CMMI10", "X"), ("CMR10", "˜x")])) == Counter(prime=1, tilde=1)
    assert sm._line_marks(_chars([("Helvetica", "Mun˜oz")])) == Counter(tilde=1)
    assert sm._line_marks(_chars([("Helvetica", "Muñoz")])) == Counter({"tilde:n": 1})
    assert sm.html_marks("<p>Let <math>A(1)^{-1}'</math> be ~</p>") == Counter(prime=1, tilde=1)


def test_only_marks_the_page_has_and_the_sidecar_lacks_are_flagged():
    assert sm.marks_findings(Counter(prime=1), Counter(prime=2, tilde=1))[0]["values"] == [
        "PDF has 1 more ′", "PDF has 1 more ~"]
    assert sm.marks_findings(Counter(prime=2), Counter(prime=1)) == []
    # A name the layer spells "n˜" and Surya writes composed is no flag.
    assert sm.marks_findings(sm.prose_marks("Muñoz"), Counter(tilde=1)) == []


def test_accents_are_paired_with_the_letter_under_them():
    def glyph(ch, x0, x1, y0, y1):
        return ("cmr10", {"c": ch, "bbox": (x0, y0, x1, y1)})
    # Layers write the hat before or after its letter, even across a space: position decides.
    line = [glyph("\u0302", 10, 14, 0, 4), glyph(" ", 14, 16, 4, 12), glyph("σ", 9, 15, 4, 12),
            glyph("x", 20, 26, 4, 12), glyph("¯", 30, 35, 0, 4), glyph("y", 29, 35, 4, 12)]
    assert sm._line_marks(line) == Counter({"hat:σ": 1, "bar:y": 1})
    # A zero-width combining hat before a space belongs to the next letter, not the one it touches.
    stata = [glyph("d", 88, 93.2, 446, 455), glyph("\u0302", 93.2, 93.2, 445, 455), glyph(" ", 93.2, 95.8, 445, 455),
             glyph("V", 95.8, 101, 445, 455), glyph("D", 103, 109, 444, 451)]
    assert sm._line_marks(stata) == Counter({"hat:v": 1})
    # A spacing hat over a letter with a superscript beside it: the taller letter is the base.
    sup = [glyph("ˆ", 96, 101, 440, 444), glyph("V", 95.8, 101, 445, 455), glyph("D", 100, 106, 444, 450)]
    assert sm._line_marks(sup) == Counter({"hat:v": 1})
    assert sm.latex_marks(r"\widehat{\boldsymbol{\sigma}} \overline{\mathbf{Y}}_n \hat{\text{ATE}}") == Counter(
        {"hat:σ": 1, "bar:y": 1})
    assert sm.latex_marks(r"\hat{\bar{p}} \overline{\hat{f}(x)}") == Counter(
        {"hat:p": 1, "bar:p": 1, "hat:f": 1, "bar:f": 1})


def test_a_swapped_accent_is_flagged_but_a_missing_one_is_not():
    page = Counter({"hat:γ": 1, "tilde:x": 1, "tilde": 1})
    ours = sm.latex_marks(r"\tilde{\mathbf{x}} \tilde{\gamma}'")
    assert sm.marks_findings(ours, page)[0]["values"] == ["PDF prints γ\u0302, the sidecar γ\u0303"]
    assert sm.marks_findings(sm.latex_marks(r"\tilde{x} \gamma"), page) == []
    tilde_to_bar = sm.marks_findings(sm.latex_marks(r"g(\bar{X}_n)"), Counter({"tilde:x": 1, "tilde": 1}))
    assert tilde_to_bar[0]["values"] == ["PDF has 1 more ~", "PDF prints x\u0303, the sidecar x\u0304"]


def test_relations_are_compared():
    assert sm.latex_marks(r"p \leq 0.05, a < b \pm c") == Counter({"rel:≤": 1, "rel:<": 1, "rel:±": 1})
    assert sm.prose_marks("p < 0.05 and x ≥ 2") == Counter({"rel:<": 1, "rel:≥": 1})
    # The page's "<" read as "≤": flagged; the sidecar's extra relation alone is not.
    assert sm.marks_findings(sm.latex_marks(r"p \le 0.05"), Counter({"rel:<": 1}))[0]["values"] == ["PDF has 1 more <"]
    assert sm.marks_findings(sm.latex_marks(r"p < 0.05"), Counter()) == []
    assert sm.html_marks("<p>* p < .10, ** p < .05</p>") == Counter({"rel:<": 2})  # a bare "<" is no tag
    assert sm.marks_findings(sm.latex_marks(r"\nu \ll \mu"), Counter({"rel:<": 2})) == []
    # Pearson's math font prints S/n as "S>n": the sidecar's slash or fraction excuses the ">".
    assert sm.marks_findings(sm.prose_marks("p = S/n"), Counter({"rel:>": 1})) == []
    assert sm.marks_findings(sm.latex_marks(r"\frac{S}{n}"), Counter({"rel:>": 1})) == []


def test_relation_glyphs_that_are_something_else():
    def g(font, ch, x, h=8.0):
        return (font, {"c": ch, "bbox": (x, 100, x + 4, 100 + h)})
    # AMS msam prints ⩾ at the code of ">"; a bracket piece far taller than the text is no relation.
    assert sm._line_marks([g("msam10", ">", 10), g("cmmi10", "x", 20)]) == Counter({"rel:≥": 1})
    assert sm._line_marks([g("pearson", "≥", 10, h=40), g("pearson", "b", 20)]) == Counter()
    # A font that prints the en dash of page ranges as "±".
    assert sm._line_marks([g("advp", "2", 0), g("advp", "±", 5), g("advp", "3", 10)]) == Counter()
    assert sm._line_marks([g("advp", "6", 0), g("advp", "±", 5)]) == Counter()
    assert sm._line_marks([g("cmr", "a", 0), g("cmsy", "±", 5), g("cmr", "b", 10)]) == Counter({"rel:±": 1})
    # List notation "< a, b >" printed with < and >: the sidecar's \langle \rangle excuse them.
    assert sm.marks_findings(sm.latex_marks(r"\langle a, b \rangle"), Counter({"rel:<": 1, "rel:>": 1})) == []


def test_a_dropped_tilde_makes_an_equation_differ():
    _doc, page = _page((60, "x ~ N(0, 1)"))
    box = (30, 40, 370, 70)
    assert sm.check_equation(r'<math display="block">x \sim N(0, 1)</math>', page, box).status == "agree"
    lost = sm.check_equation(r'<math display="block">x \in N(0, 1)</math>', page, box)
    assert lost.status == "differ" and [f["kind"] for f in lost.findings] == ["math_marks_missing"]
    line = _math_status_line({"label": "Equation", "page": 3, "findings": lost.findings, "repairs": []})
    assert line.startswith("[Equation status: UNRESOLVED (primes, tildes, accents or relations") and "PDF p. 3" in line
    prose = sm.check_inline_math("<p><math>x \\in N(0, 1)</math></p>", page, box)
    assert [f["kind"] for f in prose.findings] == ["math_marks_missing"]


def test_stata_abbreviations_and_fences_get_their_tildes_back():
    _doc, page = _page((60, "ln_populat~n -.0475582"), (80, "~~~~"))
    html, repairs = restore_marks("<td>ln_populat-n</td><p>-----</p><math>a-b</math>", page, (30, 40, 370, 90))
    assert html == "<td>ln_populat~n</td><p>~~~~</p><math>a-b</math>"
    assert [(r["before"], r["after"]) for r in repairs] == [("ln_populat-n", "ln_populat~n"), ("-----", "~~~~")]
    # A dashed word the page prints with its dash stays.
    _doc, page = _page((60, "two-way fixed effects"))
    assert restore_marks("<p>two-way</p>", page, (30, 40, 370, 90)) == ("<p>two-way</p>", [])


_STATA = ("<table><tr><th>hrate</th><th>Coefficient</th><th>Std. err.</th><th>z</th><th>P&gt;|z|</th>"
          "<th>[95% conf. interval]</th></tr>"
          "<tr><td>gini</td><td>77.1</td><td>5.33</td><td>14.46</td><td>0.000</td><td>66.6</td><td>87.5</td></tr>"
          "<tr><td>_cons</td><td>-28.8</td><td>2.95</td><td>-9.78</td><td>0.000</td><td>-34.6</td><td>-23.0</td></tr>"
          "{extra}</table>")


def test_equation_name_merged_into_a_row_is_split_off():
    merged = ("<tr><th>W</th><th>hrate</th><th>.227</th><th>.0607</th><th>3.74</th><th>0.000</th>"
              "<th>.108</th><th>.346</th></tr>")
    html, notes = fix_table_structure(_STATA.format(extra=merged))
    assert _rows(html)[-2:] == [["W", "", "", "", "", "", ""], ["hrate", ".227", ".0607", "3.74", "0.000", ".108", ".346"]]
    assert notes == ["split equation label 'W' from row 'hrate'"]
    html, notes = fix_table_structure(_STATA.format(extra=merged.replace(">W<", ">exp2<")))
    assert notes == ["split equation label 'exp2' from row 'hrate'"]  # a digit does not make a label a number
    spanned = _STATA.format(extra=(
        '<tr><td rowspan="2">W</td><td>dry</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>'
        "<tr><td>Yes</td><td>.287</td><td>.221</td><td>1.30</td><td>0.194</td><td>-.146</td><td>.720</td></tr>"))
    html, notes = fix_table_structure(spanned)
    assert "rowspan" not in html and _rows(html)[-3:-1] == [["W", "", "", "", "", "", ""], ["dry", "", "", "", "", "", ""]]
    assert fix_table_structure(_STATA.format(extra="")) == (_STATA.format(extra=""), [])
    # A label followed by a value in a form the number test reads ("0.65(0.06)") is a data row.
    data = merged.replace(">hrate<", ">0.65(0.06)<")
    assert fix_table_structure(_STATA.format(extra=data))[1] == []


def test_headers_are_not_reshaped():
    shifted = _STATA.format(extra="").replace("<tr><th>hrate</th>", "<tr><th></th><th>hrate</th>", 1)
    assert fix_table_structure(shifted) == (shifted, [])


def test_rereads_keep_the_page_words_and_number_spellings():
    printed = ["_ID long %12.0g Spatial-unit ID", "gini .3050009 -.3489221"]
    html, repairs, unknown = reconcile_rewrite(
        "<td>Spacial-unit ID</td><td>0.3050009</td><td>-0.3489221</td><td>gini</td>",
        "<td>Spatial-unit ID</td>", printed)
    assert html == "<td>Spatial-unit ID</td><td>.3050009</td><td>-.3489221</td><td>gini</td>"
    assert unknown == [] and len(repairs) == 3
    _html, _r, unknown = reconcile_rewrite("<td>Population</td>", "<td>gini</td>", printed)
    assert unknown == ["Population"]
