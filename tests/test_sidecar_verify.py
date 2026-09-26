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
        t2.post = post.get(t.text, "")
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
