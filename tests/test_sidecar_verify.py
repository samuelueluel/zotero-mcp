"""Unit tests for Surya table verification against native tokens."""

from __future__ import annotations

from zotero_mcp.sidecar_native import NativeToken
from zotero_mcp.sidecar_verify import (
    cell_numbers,
    split_estimate_se,
    text_numbers,
    verify_table,
    withhold_numbers,
)


def tok(text: str, sign_source: str | None = None) -> NativeToken:
    neg = text.startswith("-")
    return NativeToken(
        text=text,
        negative=neg,
        sign_source=sign_source or ("codepoint" if neg else "none"),
        bbox=(0, 0, 1, 1),
        page=1,
    )


TABLE = (
    "<table><tr><th>Var</th><th>Coef</th><th>t</th></tr>"
    "<tr><td>Age</td><td>{a}</td><td>{b}</td></tr>"
    "<tr><td>Acreage</td><td>-.815</td><td>-37.652</td></tr></table>"
)


def test_verified_when_both_directions_match():
    html = TABLE.format(a="-.007", b="-.218")
    check = verify_table(html, [tok(x) for x in ("-.007", "-.218", "-.815", "-37.652")])
    assert check.status == "verified"
    assert check.surya_numbers == check.native_numbers == 4


def test_authoritative_sign_is_repaired():
    html = TABLE.format(a=".007", b="-.218")
    native = [tok("-.007", "font_glyph"), tok("-.218"), tok("-.815"), tok("-37.652")]
    check = verify_table(html, native)
    assert check.status == "repaired"
    assert check.repairs == [{"kind": "sign", "before": ".007", "after": "-.007", "route": "native_font_validated"}]
    assert "-.007" in check.html


def test_inferred_sign_is_never_substituted():
    html = TABLE.format(a=".007", b="-.218")
    native = [tok("-.007", "control_glyph"), tok("-.218"), tok("-.815"), tok("-37.652")]
    check = verify_table(html, native)
    assert check.status == "unresolved"
    assert any(f.kind == "sign_conflict" for f in check.findings)
    assert not check.repairs


def test_surya_negative_native_positive_is_conflict_not_repair():
    # A missing native minus can be a broken glyph; never strip Surya's sign.
    html = TABLE.format(a="-.007", b="-.218")
    native = [tok(".007"), tok("-.218"), tok("-.815"), tok("-37.652")]
    check = verify_table(html, native)
    assert check.status == "unresolved"
    assert "-.007" in check.html


def test_truncation_is_detected():
    html = TABLE.format(a="-.007", b="-.218")
    native = [tok(x) for x in ("-.007", "-.218", "-.815", "-37.652", ".5", ".6")]
    check = verify_table(html, native)
    assert check.status == "unresolved"
    missing = next(f for f in check.findings if f.kind == "native_missing")
    assert sorted(missing.values) == [".5", ".6"]


DUP_ROW = "<tr><td>Log price</td><td>-.0019</td><td>-.0037</td><td>.0011</td></tr>"


def test_unbacked_duplicate_panel_is_unresolved():
    check = verify_table(f"<table>{DUP_ROW}{DUP_ROW}</table>", [tok(x) for x in ("-.0019", "-.0037", ".0011")])
    assert check.status == "unresolved"
    assert any(f.kind == "duplicate_rows" for f in check.findings)
    assert any(f.kind == "surya_unmatched" for f in check.findings)


def test_duplicate_rows_backed_by_text_layer_verify():
    native = [tok(x) for x in ("-.0019", "-.0037", ".0011") * 2]
    check = verify_table(f"<table>{DUP_ROW}{DUP_ROW}</table>", native)
    assert check.status == "verified"
    assert any(f.kind == "duplicate_rows" for f in check.findings)


def test_single_route_without_native_numbers():
    check = verify_table(TABLE.format(a="-.007", b="-.218"), [])
    assert check.status == "single-route"


def test_merged_estimate_se_is_split():
    html, n = split_estimate_se("<table><tr><td>.0640***[.0166]</td><td>.0053(.0098)</td></tr></table>")
    assert n == 2
    assert ".0640*** [.0166]" in html and ".0053 (.0098)" in html


def test_header_column_numbers_are_structural():
    html = ("<table><tr><td></td><td>(1)</td><td>(2)</td></tr>"
            "<tr><td>Age</td><td>.5</td><td>.6</td></tr></table>")
    check = verify_table(html, [tok(".5"), tok(".6")])
    assert check.status == "verified"
    assert any(f.kind == "label_or_header_mismatch" for f in check.findings)


def test_scan_agreement_caps_at_single_route():
    html = TABLE.format(a="-.007", b="-.218")
    check = verify_table(html, [tok(x) for x in ("-.007", "-.218", "-.815", "-37.652")], native_is_ocr=True)
    assert check.status == "single-route"
    assert any(f.kind == "ocr_layer_agreement" for f in check.findings)


def test_scan_never_repairs():
    html = TABLE.format(a=".007", b="-.218")
    native = [tok("-.007", "codepoint"), tok("-.218"), tok("-.815"), tok("-37.652")]
    check = verify_table(html, native, native_is_ocr=True)
    assert not check.repairs and "-.007" not in check.html


def test_formula_table_is_single_route_not_withheld():
    cell = "<td><math>R_{0,Y} - R_{1,i}^*</math></td>"
    html = f"<table><tr><td>Homeowners</td>{cell}{cell}</tr><tr><td>Renters</td>{cell}<td>0</td></tr></table>"
    check = verify_table(html, [tok("1"), tok("2"), tok("0")])
    assert check.status == "single-route"
    assert any(f.kind == "math_table" for f in check.findings)


def test_ranges_and_unicode_minus():
    assert text_numbers("1993–98 period") == ["1993", "98"]
    assert text_numbers("<td>\u2212.059</td>") == ["-.059"]


def test_withhold_replaces_only_numeric_data_cells():
    html = withhold_numbers(TABLE.format(a="-.007", b="-.218"), page=25)
    assert "Age" in html and "Coef" in html
    assert "-.007" not in html and "withheld" in html and "p. 25" in html


def test_detached_minus_opening_a_cell_is_a_sign():
    assert cell_numbers("– 0.15") == ["-0.15"]
    assert cell_numbers("0.24 – 0.15") == ["0.24", "0.15"]  # a range, not a sign
    check = verify_table(TABLE.format(a="– .007", b="-.218"), [tok(x) for x in ("-.007", "-.218", "-.815", "-37.652")])
    assert check.status == "verified"


def test_comma_groups_only_three_digits():
    assert text_numbers("(0.48,0.51)") == ["0.48", "0.51"]
    assert text_numbers("5,037 and 12,34") == ["5,037", "12", "34"]


def test_spaced_exponent_minus():
    assert cell_numbers("9.38 E – 06") == ["9.38", "-06"]


def test_note_paragraph_outside_rows_is_not_data():
    html = TABLE.format(a="-.007", b="-.218") + "<p>Notes: n = 495; significant at the 1% level.</p>"
    native = [tok(x) for x in ("-.007", "-.218", "-.815", "-37.652")]
    check = verify_table(html, native)
    assert check.status == "verified"
    assert any(f.kind == "label_or_header_mismatch" for f in check.findings)


def test_ocr_layer_one_to_one_disagreement_is_single_route():
    # Surya -9.000 against an OCR layer's -9.008: the weak witness cannot settle it.
    native = [tok(x) for x in ("-.007", "-.218", "-.815", "-37.658")]
    check = verify_table(TABLE.format(a="-.007", b="-.218"), native, native_is_ocr=True)
    assert check.status == "single-route"


def test_born_digital_digit_misread_is_unresolved():
    native = [tok(x) for x in ("-.007", "-.218", "-.815", "-37.658")]
    check = verify_table(TABLE.format(a="-.007", b="-.218"), native)
    assert check.status == "unresolved"


def ptok(text: str, x: float, y: float) -> NativeToken:
    t = tok(text)
    t.bbox = (x, y, x + 20, y + 8)
    return t


# A 3x3 grid printed at x = 100/200/300, y = 10/30/50.
GRID = [["1.11", "2.22", "3.33"], ["4.44", "5.55", "6.66"], ["7.77", "8.88", "9.99"]]
PRINTED = [ptok(v, 100 + 100 * c, 10 + 20 * r) for r, row in enumerate(GRID) for c, v in enumerate(row)]


def _html(rows):
    return "<table>" + "".join(
        "<tr><td>row</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows) + "</table>"


def test_positions_agree():
    from zotero_mcp.sidecar_verify import misplaced_cells
    assert misplaced_cells(_html(GRID), PRINTED) == []
    assert verify_table(_html(GRID), PRINTED).status == "verified"


def test_value_moved_to_another_row_is_misplaced():
    from zotero_mcp.sidecar_verify import misplaced_cells
    rows = [["1.11", "2.22", "6.66"], ["4.44", "5.55", "3.33"], GRID[2]]
    assert set(misplaced_cells(_html(rows), PRINTED)) >= {"6.66", "3.33"}
    check = verify_table(_html(rows), PRINTED)
    assert check.status == "unresolved" and any(f.kind == "misplaced_cells" for f in check.findings)


def test_row_shifted_into_an_empty_column_is_misplaced():
    from zotero_mcp.sidecar_verify import misplaced_cells
    printed = [t for t in PRINTED if t.text != "4.44"]  # row 2 prints nothing in column 1
    rows = [GRID[0], ["", "5.55", "6.66"], GRID[2]]
    assert misplaced_cells(_html(rows), printed) == []
    shifted = "<tr><td>row</td><td>5.55</td><td>6.66</td><td></td></tr>"
    html = _html([GRID[0]]) .replace("</table>", shifted) + _html([GRID[2]]).replace("<table>", "")
    assert misplaced_cells(html, printed)


def test_ragged_row_that_keeps_its_columns_passes():
    from zotero_mcp.sidecar_verify import misplaced_cells
    printed = [t for t in PRINTED if t.text != "4.44"]
    ragged = "<tr><td>row</td><td>5.55</td><td>6.66</td></tr>"  # leading empty cell dropped
    html = _html([GRID[0]]).replace("</table>", ragged) + _html([GRID[2]]).replace("<table>", "")
    assert misplaced_cells(html, printed) == []


def test_row_widened_by_a_merged_label_is_misplaced():
    from zotero_mcp.sidecar_verify import misplaced_cells
    # Row 2 carries an extra leading cell: from the right its values line up, from the left
    # each sits one column too far, under the wrong header.
    merged = "<tr><td>eq</td><td>row</td><td>4.44</td><td>5.55</td><td>6.66</td></tr>"
    html = _html([GRID[0]]).replace("</table>", merged) + _html([GRID[2]]).replace("<table>", "")
    assert set(misplaced_cells(html, PRINTED)) == {"4.44", "5.55"}  # 6.66 has no column to compare


def test_row_that_lost_its_label_is_misplaced():
    from zotero_mcp.sidecar_verify import misplaced_cells
    # Row 2 lost its label: its values start in the label column, one column left.
    lost = "<tr><td>4.44</td><td>5.55</td><td>6.66</td></tr>"
    html = _html([GRID[0]]).replace("</table>", lost) + _html([GRID[2]]).replace("<table>", "")
    assert set(misplaced_cells(html, PRINTED)) >= {"5.55", "6.66"}


def test_lost_labels():
    from zotero_mcp.sidecar_verify import lost_label_words
    html = "<table><tr><td>Whole city</td><td>1.2</td></tr><tr><td>Signi\ufb01cant</td></tr></table>"
    assert lost_label_words(html, ["Whole city 1.2", "Alderman Districts 2.3", "signi cant"]) == [
        "alderman", "districts"]
    # Broken text-layer words: a lost ligature, a glued footnote letter.
    assert lost_label_words("<table><tr><td>Significant income<sup>b</sup></td></tr></table>",
                            ["signi cant incomeb"]) == []


# Filling cells from the text layer ------------------------------------------


def _labelled(rows, labels=("Alpha", "Bravo", "Charlie")):
    return "<table>" + "".join(
        f"<tr><td>{label}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for label, row in zip(labels, rows)) + "</table>"


def _cells(check):
    return [(r["before"], r["after"]) for r in check.repairs if r["route"] == "native_cell"]


def test_fill_replaces_a_misread_digit_in_place():
    rows = [GRID[0], ["4.44", "5.85", "6.66"], GRID[2]]
    check = verify_table(_labelled(rows), PRINTED, printed_lines=[])
    assert check.status == "repaired" and _cells(check) == [("5.85", "5.55")]
    assert "5.55" in check.html and "5.85" not in check.html
    assert "cell" in check.repairs[0]["kind"]


def test_fill_needs_printed_lines_and_can_be_off():
    rows = [GRID[0], ["4.44", "5.85", "6.66"], GRID[2]]
    assert verify_table(_labelled(rows), PRINTED).status == "unresolved"
    assert verify_table(_labelled(rows), PRINTED, printed_lines=[], fill=False).status == "unresolved"


def test_fill_puts_a_lost_value_into_its_empty_cell():
    rows = [GRID[0], ["4.44", "", "6.66"], GRID[2]]
    check = verify_table(_labelled(rows), PRINTED, printed_lines=[])
    assert check.status == "repaired" and _cells(check) == [("", "5.55")]


def test_fill_places_an_empty_row_by_its_label():
    rows = [GRID[0], ["", "", ""], GRID[2]]
    lines = [("Bravo 4.44 5.55 6.66", (20, 30, 320, 38))]
    check = verify_table(_labelled(rows), PRINTED, printed_lines=lines)
    assert check.status == "repaired" and sorted(a for _b, a in _cells(check)) == ["4.44", "5.55", "6.66"]
    # Without the label line the row's position is unknown: nothing is filled.
    assert verify_table(_labelled(rows), PRINTED, printed_lines=[]).status == "unresolved"


def test_fill_appends_a_lost_standard_error_with_its_brackets():
    printed = []
    for r, row in enumerate([["1.11", "2.22", "3.33"], ["4.44", "5.55", "6.66"]]):
        for c, v in enumerate(row):
            printed.append(ptok(v, 100 + 100 * c, 10 + 40 * r))
            se = ptok(f"0.{v[0] * 2}", 100 + 100 * c, 22 + 40 * r)
            se.pre, se.post = "(", ")"
            printed.append(se)
    cells = [[f"{t.text}<br/>({s.text})" for t, s in zip(printed[i:i + 6:2], printed[i + 1:i + 6:2])]
             for i in (0, 6)]
    cells[1][1] = "5.55"  # its standard error (0.55) lost
    check = verify_table(_labelled(cells), printed, printed_lines=[])
    assert check.status == "repaired" and _cells(check) == [("", "0.55")]
    assert "5.55<br/>(0.55)" in check.html


def test_fill_refuses_an_inferred_sign():
    printed = [t for t in PRINTED if t.text != "5.55"] + [ptok("-5.55", 200, 30)]
    printed[-1].sign_source = "control_glyph"
    rows = [GRID[0], ["4.44", "", "6.66"], GRID[2]]
    check = verify_table(_labelled(rows), printed, printed_lines=[])
    assert check.status == "unresolved" and not _cells(check)


def test_fill_refuses_empty_cells_when_stars_are_not_in_the_text_layer():
    rows = [["1.11**", "2.22", "3.33"], ["4.44", "", "6.66"], GRID[2]]
    assert verify_table(_labelled(rows), PRINTED, printed_lines=[]).status == "unresolved"
    starred = [("1.11** 2.22 3.33", (100, 10, 320, 18))]
    assert verify_table(_labelled(rows), PRINTED, printed_lines=starred).status == "repaired"


def test_fill_leaves_a_misplaced_value_alone():
    # 6.66 sits in row 1's line of print but Surya put it in row 0; 3.33 is lost.
    rows = [["1.11", "2.22", "6.66"], ["4.44", "5.55", ""], GRID[2]]
    check = verify_table(_labelled(rows), PRINTED, printed_lines=[])
    assert check.status == "unresolved" and not _cells(check)


def _colnum(text: str) -> NativeToken:
    t = tok(text)
    t.role = "colnum"
    return t


def test_dropped_column_numbers_demote_a_verified_table():
    # Printed (1) (2) (3); Surya kept (1) and (3) and lost (2).
    html = ("<table><tr><th></th><th>(1)</th><th colspan=\"2\">(3)</th></tr>"
            "<tr><td>Age</td><td>-.007</td><td>-.218</td><td>.5</td></tr>"
            "<tr><td>Acreage</td><td>-.815</td><td>-37.652</td><td>.6</td></tr></table>")
    native = [tok(x) for x in ("-.007", "-.218", ".5", "-.815", "-37.652", ".6")]
    native += [_colnum(x) for x in ("1", "2", "3")]
    check = verify_table(html, native)
    assert check.status == "single-route"
    finding = next(f for f in check.findings if f.kind == "header_structure")
    assert finding.values == ["(2)"]


def test_kept_column_numbers_stay_verified():
    html = ("<table><tr><th></th><th>(1)</th><th>(2)</th></tr>"
            "<tr><td>Age</td><td>-.007</td><td>-.218</td></tr>"
            "<tr><td>Acreage</td><td>-.815</td><td>-37.652</td></tr></table>")
    native = [tok(x) for x in ("-.007", "-.218", "-.815", "-37.652")] + [_colnum("1"), _colnum("2")]
    assert verify_table(html, native).status == "verified"


def test_empty_body_column_demotes_a_verified_table():
    # A stray header cell pushed the headers one column right of the data.
    html = ("<table><tr><th>Var</th><th>Coef</th><th>t</th><th>Var</th></tr>"
            "<tr><td>Age</td><td>-.007</td><td>-.218</td><td></td></tr>"
            "<tr><td>Acreage</td><td>-.815</td><td>-37.652</td><td></td></tr></table>")
    check = verify_table(html, [tok(x) for x in ("-.007", "-.218", "-.815", "-37.652")])
    assert check.status == "single-route"
    assert any(f.kind == "header_structure" for f in check.findings)


def test_header_structure_status_line():
    from zotero_mcp.sidecar_assemble import _status_line

    block = {"status": "single-route", "label": "Table", "page": 39, "repairs": [],
             "findings": [{"kind": "header_structure", "detail": "", "values": ["(2)"]}]}
    line = _status_line(block)
    assert "SINGLE-ROUTE (column headers do not line up with the page)" in line
    assert "numbers match the PDF text layer" in line


def test_readings_agree_ignores_row_layout_but_not_column_shifts():
    from zotero_mcp.sidecar_verify import readings_agree
    one_cell = "<table><tr><td>a</td><td>0.12<br/>(0.03)</td><td>5</td></tr></table>"
    two_rows = "<table><tr><td>a</td><td>0.12</td><td>5</td></tr><tr><td></td><td>(0.03)</td><td></td></tr></table>"
    shifted = "<table><tr><td>a</td><td></td><td>0.12<br/>(0.03)</td><td>5</td></tr></table>"
    empty_dropped = "<table><tr><td>a</td><td>0.12</td><td></td></tr><tr><td>b</td><td></td><td>7</td></tr></table>"
    left_shift = "<table><tr><td>a</td><td>0.12</td></tr><tr><td>b</td><td>7</td></tr></table>"
    assert readings_agree(one_cell, two_rows)
    # An extra empty column moves every value alike: the numbers still agree.
    assert readings_agree(one_cell, shifted)
    # One row shifting against the others does not.
    assert not readings_agree(empty_dropped, left_shift)
    assert not readings_agree("<table></table>", "<table></table>")


def _starred(post: dict[str, str]):
    printed = []
    for t in PRINTED:
        t2 = ptok(t.text, t.bbox[0], t.bbox[1])
        t2.post = t2.stars = post.get(t.text, "")
        printed.append(t2)
    return printed


def test_stars_are_set_from_the_text_layer_where_it_prints_them():
    printed = _starred({"1.11": "**", "5.55": "*"})
    rows = [["1.11***", "2.22", "3.33"], ["4.44", "5.55", "6.66"], GRID[2]]
    check = verify_table(_labelled(rows), printed)
    assert check.status == "repaired"
    assert "1.11**<" in check.html and "5.55*<" in check.html and "***" not in check.html
    assert sorted((r["value"], r["before"], r["after"]) for r in check.repairs) == [
        ("1.11", "***", "**"), ("5.55", "", "*")]


def test_stars_without_adjacent_text_layer_stars_are_left_alone():
    rows = [["1.11***", "2.22", "3.33"], GRID[1], GRID[2]]
    check = verify_table(_labelled(rows), PRINTED)
    assert check.status == "verified" and "1.11***" in check.html


def test_stars_after_the_standard_error_only_correct_an_existing_run():
    # Text layer: 1.11 with no stars of its own, (0.5)** below it.
    printed = _starred({})
    printed[0].se_stars = "**"
    fixed = verify_table(_labelled([["1.11***", "2.22", "3.33"], GRID[1], GRID[2]]), printed)
    assert "1.11**<" in fixed.html and fixed.status == "repaired"
    bare = verify_table(_labelled(GRID), printed)
    assert bare.status == "verified" and "*" not in bare.html


def test_stars_the_text_layer_does_not_print_demote_the_table():
    printed = _starred({"1.11": "**", "2.22": "*", "3.33": "***"})
    rows = [["1.11**", "2.22*", "3.33***"], ["4.44**", "5.55", "6.66"], GRID[2]]
    check = verify_table(_labelled(rows), printed)
    assert check.status == "single-route"
    assert next(f for f in check.findings if f.kind == "stars_unconfirmed").values == ["4.44"]
    # Too few printed stars to trust their absence: left alone.
    sparse = _starred({"1.11": "**"})
    assert verify_table(_labelled([["1.11**", "2.22", "3.33"], ["4.44**", "5.55", "6.66"], GRID[2]]),
                        sparse).status == "verified"


def test_table_without_a_text_layer_is_marked():
    check = verify_table(TABLE.format(a="-.007", b="-.218"), [])
    assert check.status == "single-route"
    assert any(f.kind == "no_text_layer" for f in check.findings)


def test_cells_read_tex_thousands_but_prose_does_not():
    assert cell_numbers("Presidential election, 3, 107 Observations") == ["3,107"]
    assert text_numbers("Journal of Urban Affairs, 22, 301-315") == ["22", "301", "315"]


def test_formula_table_is_verified_when_the_full_check_passes():
    html = ("<table><tr><th></th><th>Coef</th></tr>"
            "<tr><td><math>\\rho</math></td><td>0.512</td></tr>"
            "<tr><td><math>\\beta_1</math></td><td>0.634</td></tr></table>")
    subscript = tok("1")
    subscript.role = "label"  # the text layer's subscript of beta_1
    assert verify_table(html, [tok("0.512"), tok("0.634"), subscript]).status == "verified"
    # A disagreement keeps the old formula-table result, never unresolved.
    check = verify_table(html, [tok("0.512"), tok("0.684"), subscript])
    assert check.status == "single-route"
    assert [f.kind for f in check.findings if f.kind == "math_table"] == ["math_table"]


def test_a_table_of_formulas_stays_single_route():
    html = ("<table><tr><th></th><th>Value</th></tr>"
            "<tr><td>a</td><td><math>\\frac{1}{2}\\beta</math></td></tr>"
            "<tr><td>b</td><td><math>1 - \\alpha</math></td></tr></table>")
    native = [tok("1"), tok("2"), tok("1")]
    assert verify_table(html, native).status == "single-route"


def test_header_rowspan_spilling_into_the_body_is_cut_back():
    from zotero_mcp.sidecar_verify import clamp_header_rowspans
    html = ('<table><tr><th rowspan="3">Method</th><th colspan="2">P</th></tr>'
            "<tr><th>a</th><th>b</th></tr>"
            "<tr><td>OLS</td><td>1</td><td>2</td></tr><tr><td>X</td><td>3</td><td>4</td></tr></table>")
    fixed, n = clamp_header_rowspans(html)
    assert n == 1 and 'rowspan="2">Method' in fixed
    # A rowspan that covers a body row with no cell of its own is left alone.
    ok = '<table><tr><th rowspan="2">M</th><th>a</th></tr><tr><td>1</td></tr><tr><td>X</td><td>3</td></tr></table>'
    assert clamp_header_rowspans(ok) == (ok, 0)


def test_headers_printed_over_other_columns_demote_the_table():
    from zotero_mcp.sidecar_verify import misplaced_headers
    # Page: stub | (param) | mean | std ; the HTML header row lacks the param column.
    html = ("<table><tr><th>Variables</th><th>mean</th><th>std</th></tr>"
            "<tr><td>Constant</td><td>(a)</td><td>0.0123</td><td>0.0222</td></tr>"
            "<tr><td>Slope</td><td>(b)</td><td>1.9708</td><td>0.0419</td></tr></table>")
    native = [ptok("0.0123", 200, 30), ptok("0.0222", 300, 30), ptok("1.9708", 200, 50), ptok("0.0419", 300, 50)]
    words = [("Variables", (10, 10, 60, 18)), ("mean", (200, 10, 220, 18)), ("std", (300, 10, 315, 18))]
    assert misplaced_headers(html, native, words)
    check = verify_table(html, native, printed_word_boxes=words)
    assert check.status == "single-route" and any(f.kind == "header_structure" for f in check.findings)
    right = html.replace("<th>Variables</th>", "<th>Variables</th><th></th>")
    assert not misplaced_headers(right, native, words)
    assert verify_table(right, native, printed_word_boxes=words).status == "verified"


def test_a_header_that_runs_over_the_next_column_is_widened():
    from zotero_mcp.sidecar_verify import misplaced_headers, widen_headers
    # Page: "[95% conf. interval]" printed across both bound columns; Surya gives it one.
    html = ("<table><tr><th></th><th>Coef.</th><th>[95% conf. interval]</th></tr>"
            "<tr><td>x</td><td>0.512</td><td>0.301</td><td>0.723</td></tr>"
            "<tr><td>z</td><td>1.204</td><td>0.988</td><td>1.420</td></tr></table>")
    native = [ptok(v, x, y) for y, row in ((30, ("0.512", "0.301", "0.723")), (50, ("1.204", "0.988", "1.420")))
              for x, v in zip((100, 200, 300), row)]
    words = [("Coef.", (100, 10, 120, 18)), ("[95%", (190, 10, 210, 18)), ("conf.", (212, 10, 232, 18)),
             ("interval]", (300, 10, 325, 18))]
    assert misplaced_headers(html, native, words) == ["column 4 has a printed header the table lacks"]
    wide, notes = widen_headers(html, native, words)
    assert 'colspan="2">[95% conf. interval]' in wide and notes and not misplaced_headers(wide, native, words)
    assert verify_table(html, native, printed_word_boxes=words).status == "verified"


def test_one_line_cells_centred_beside_estimates_are_in_their_row():
    from zotero_mcp.sidecar_verify import misplaced_cells
    # Each row: N printed on the standard-error line, beside estimate/SE pairs.
    rows, native = [], []
    for r, (n, e1, e2) in enumerate([("6502", "0.074", "0.172"), ("8678", "0.112", "0.310"),
                                     ("9739", "0.092", "0.405"), ("1234", "0.321", "0.518")]):
        y = 10 + 40 * r
        rows.append(f"<tr><td>R{r}</td><td>{n}</td><td>{e1}<br/>(0.00{r + 1})</td><td>{e2}<br/>(0.01{r})</td></tr>")
        native += [ptok(n, 100, y + 10), ptok(e1, 200, y), ptok(f"(0.00{r + 1})", 200, y + 10),
                   ptok(e2, 300, y), ptok(f"(0.01{r})", 300, y + 10)]
    assert misplaced_cells("<table>" + "".join(rows) + "</table>", native) == []


def test_headers_of_a_sideways_table_are_checked_too():
    from zotero_mcp.sidecar_verify import misplaced_headers
    html = ("<table><tr><th>Variables</th><th>mean</th><th>std</th></tr>"
            "<tr><td>Constant</td><td>(a)</td><td>0.0123</td><td>0.0222</td></tr>"
            "<tr><td>Slope</td><td>(b)</td><td>1.9708</td><td>0.0419</td></tr></table>")

    def turn(b):  # the upright layout rotated to read bottom to top
        return (b[1], 400 - b[2], b[3], 400 - b[0])

    native = []
    for text, x, y in (("0.0123", 200, 30), ("0.0222", 300, 30), ("1.9708", 200, 50), ("0.0419", 300, 50)):
        t = ptok(text, x, y)
        t.bbox, t.vertical = turn(t.bbox), True
        native.append(t)
    words = [(w, turn(b), True) for w, b in
             (("Variables", (10, 10, 60, 18)), ("mean", (200, 10, 220, 18)), ("std", (300, 10, 315, 18)))]
    assert misplaced_headers(html, native, words)


def test_fill_works_when_glyph_boxes_of_neighbouring_lines_overlap():
    # Monospace log output: lines 9 pt apart, glyph boxes 10 pt tall.
    printed = [ptok(v, 100 + 100 * c, 10 + 9 * r) for r, row in enumerate(GRID) for c, v in enumerate(row)]
    for t in printed:
        t.bbox = (t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[1] + 10)
    rows = [GRID[0], ["4.44", "5.85", "6.66"], GRID[2]]
    check = verify_table(_labelled(rows), printed, printed_lines=[])
    assert check.status == "repaired" and _cells(check) == [("5.85", "5.55")]


def test_dotted_dates_and_section_numbers_are_one_number():
    from zotero_mcp.sidecar_verify import cell_numbers, text_numbers
    assert cell_numbers("1 02.13.2007 08:00") == ["1", "02.13.2007", "08", "00"]
    assert text_numbers("see 11.4.3 and 1.5.") == ["11.4.3", "1.5"]


def test_degrees_of_freedom_are_two_numbers():
    from zotero_mcp.sidecar_verify import cell_numbers
    assert cell_numbers("F( 1, 156) =") == ["1", "156"]
    assert cell_numbers("F(2,744)= 46.95") == ["2", "744", "46.95"]
    assert cell_numbers("at (1,234)") == ["1,234"]
    from zotero_mcp.sidecar_verify import text_numbers
    assert text_numbers("F(2,178) = 3.1 and 1,234 cases") == ["2", "178", "3.1", "1,234"]


def test_an_observation_number_in_the_stub_is_a_label():
    from zotero_mcp.sidecar_verify import numbers_with_roles
    html = "<table><tr><td>1023.</td><td>140</td></tr><tr><td>7</td><td>3.5</td></tr></table>"
    assert numbers_with_roles(html) == [("1023", "label"), ("140", "data"), ("7", "data"), ("3.5", "data")]


OPTIONS = ("<table><tr><th>Command</th><th>Description</th></tr>"
           "<tr><td>contrast</td><td>contrasts and joint tests</td></tr>"
           "<tr><td>lincom</td><td>linear combinations of parameters</td></tr></table>")
OPTION_LINES = [("Command Description", (10, 10, 300, 18)), ("contrast contrasts and joint tests", (10, 30, 300, 38)),
                ("lincom linear combinations of parameters", (10, 50, 300, 58))]


def test_a_table_without_numbers_verifies_by_its_words():
    words = [w for text, _b in OPTION_LINES for w in text.split()]
    check = verify_table(OPTIONS, [], printed_words=words, printed_lines=OPTION_LINES)
    assert check.status == "verified" and [f.kind for f in check.findings] == ["text_table"]


def test_a_table_without_numbers_with_a_lost_or_added_word_is_single_route():
    words = [w for text, _b in OPTION_LINES for w in text.split()]
    lost = verify_table(OPTIONS, [], printed_words=words + ["hausman"], printed_lines=OPTION_LINES)
    assert lost.status == "single-route" and "label_text_lost" in [f.kind for f in lost.findings]
    added = verify_table(OPTIONS.replace("joint", "jointly nested"), [], printed_words=words,
                         printed_lines=OPTION_LINES)
    assert added.status == "single-route" and "text_unmatched" in [f.kind for f in added.findings]


def test_a_table_without_numbers_in_the_wrong_row_order_is_single_route():
    lines = [OPTION_LINES[0], (OPTION_LINES[1][0], (10, 50, 300, 58)), (OPTION_LINES[2][0], (10, 30, 300, 38))]
    words = [w for text, _b in lines for w in text.split()]
    check = verify_table(OPTIONS, [], printed_words=words, printed_lines=lines)
    assert check.status == "single-route" and "text_rows_out_of_order" in [f.kind for f in check.findings]


def test_marks_and_scans_without_numbers_are_single_route_with_a_reason():
    marks = "<table><tr><th>Method</th><th>Trees</th></tr><tr><td>Handles missing values</td><td>▲</td></tr></table>"
    words = ["Method", "Trees", "Handles", "missing", "values"]
    check = verify_table(marks, [], printed_words=words)
    assert check.status == "single-route" and "mark_cells" in [f.kind for f in check.findings]
    scan = verify_table(OPTIONS, [], native_is_ocr=True)
    assert scan.status == "single-route" and [f.kind for f in scan.findings] == ["no_numbers"]


def test_readings_agree_ignores_label_subscripts():
    from zotero_mcp.sidecar_verify import readings_agree
    a = ("<table><tr><td></td><td><math>\\gamma_1</math></td><td><math>\\gamma_2</math></td></tr>"
         "<tr><td>2</td><td>0.6959</td><td>0.03044</td></tr></table>")
    b = ("<table><tr><td></td><td>γ₁</td><td>γ₂</td></tr>"
         "<tr><td>2</td><td>0.6959</td><td>0.03044</td></tr></table>")
    assert readings_agree(a, b)
    assert not readings_agree(a, b.replace("0.03044", "0.3044"))


def test_prose_dotted_run_broken_across_lines_matches_its_pieces():
    from types import SimpleNamespace

    from zotero_mcp.sidecar_assemble import _check_text_numbers

    def t(text):
        return SimpleNamespace(text=text, spaced_group=False)

    doi = "https://doi.org/10.1093/0199245282.001.0001 and section 11.4.3"
    assert _check_text_numbers(doi, [t("10.1093"), t("0199245282.001"), t("0001"), t("11.4.3")])[0] == "verified"
    assert _check_text_numbers("see 11.4.3", [t("11.4")])[0] == "single-route"
