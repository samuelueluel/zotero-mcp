"""Primes and tildes against the text layer, Stata row-shape fixes, and re-read words."""

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
    assert sm.latex_marks(r"X'_t \beta + \tilde{x} + y^{\prime\prime} \sim N") == Counter(prime=3, tilde=2)
    assert sm.latex_marks(r"a~b \text{it's}") == Counter()  # ~ is a space in LaTeX
    assert sm.latex_marks(r"A \cong B \simeq C") == Counter(tilde=2)  # the layer may print "∼="
    assert sm.prose_marks("Muñoz, ln_populat~n, Jensen's") == Counter(tilde=2)
    assert sm._line_marks(_chars([("CMSY10", "0"), ("CMMI10", "X"), ("CMR10", "˜x")])) == Counter(prime=1, tilde=1)
    assert sm._line_marks(_chars([("Helvetica", "Mun˜oz")])) == Counter(tilde=1)
    assert sm.html_marks("<p>Let <math>A(1)^{-1}'</math> be ~</p>") == Counter(prime=1, tilde=1)


def test_only_marks_the_page_has_and_the_sidecar_lacks_are_flagged():
    assert sm.marks_findings(Counter(prime=1), Counter(prime=2, tilde=1))[0]["values"] == [
        "PDF has 1 more ′", "PDF has 1 more ~"]
    assert sm.marks_findings(Counter(prime=2), Counter(prime=1)) == []
    # A name the layer spells "n˜" and Surya writes composed is no flag.
    assert sm.marks_findings(sm.prose_marks("Muñoz"), Counter(tilde=1)) == []


def test_a_dropped_tilde_makes_an_equation_differ():
    _doc, page = _page((60, "x ~ N(0, 1)"))
    box = (30, 40, 370, 70)
    assert sm.check_equation(r'<math display="block">x \sim N(0, 1)</math>', page, box).status == "agree"
    lost = sm.check_equation(r'<math display="block">x \in N(0, 1)</math>', page, box)
    assert lost.status == "differ" and [f["kind"] for f in lost.findings] == ["math_marks_missing"]
    line = _math_status_line({"label": "Equation", "page": 3, "findings": lost.findings, "repairs": []})
    assert line.startswith("[Equation status: UNRESOLVED (primes or tildes") and "PDF p. 3" in line
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
