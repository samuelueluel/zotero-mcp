"""html_to_markdown keeps text after inline math that contains ``<`` or ``>``."""

from __future__ import annotations

import pytest

pytest.importorskip("pymupdf")

from zotero_mcp.sidecar_assemble import html_to_markdown  # noqa: E402


def test_less_than_in_inline_math_keeps_rest_of_paragraph():
    html = (
        "<p>The estimate of <math>\\beta</math> is significant (<math>p &lt; 0.001</math>), "
        "which verifies the measure. The estimate of <math>\\delta</math> is "
        "(<math>\\hat{\\delta} = 28.010</math>, s.e. 7.905); the estimates are "
        "<math>-0.152</math> and <math>3.757</math>.<sup>22</sup></p>"
    )
    md = html_to_markdown(html)
    assert "($p < 0.001$), which verifies the measure." in md
    assert "$\\hat{\\delta} = 28.010$, s.e. 7.905" in md
    assert "$-0.152$ and $3.757$.^22" in md


def test_greater_than_and_both_relations_in_math():
    md = html_to_markdown("<p>Fair if <math>p &gt; q</math> and <math>a &lt; b &lt; c</math> hold.</p>")
    assert md == "Fair if $p > q$ and $a < b < c$ hold."


def test_block_math_with_relation_and_following_text():
    html = '<p>Then <math display="block">x &lt; y</math> follows, so <b>z</b> holds.</p>'
    md = html_to_markdown(html)
    assert "$$\nx < y\n$$" in md
    assert "follows, so **z** holds." in md


def test_escaped_less_than_in_plain_text_survives():
    assert html_to_markdown("<p>Rates (p &lt; 0.05) fell.</p>") == "Rates (p < 0.05) fell."
