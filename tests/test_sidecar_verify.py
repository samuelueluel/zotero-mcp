"""Unit tests for Surya table verification against native tokens."""

from __future__ import annotations

from zotero_mcp.sidecar_native import NativeToken
from zotero_mcp.sidecar_verify import (
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


def test_duplicate_panel_is_detected():
    row = "<tr><td>Log price</td><td>-.0019</td><td>-.0037</td><td>.0011</td></tr>"
    html = f"<table>{row}{row}</table>"
    native = [tok(x) for x in ("-.0019", "-.0037", ".0011") * 2]
    check = verify_table(html, native)
    assert check.status == "unresolved"
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
