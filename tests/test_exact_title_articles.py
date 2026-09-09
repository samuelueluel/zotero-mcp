"""Leading-article tolerance in exact title identity.

A query and a record whose folded titles differ by nothing but a leading
the/a/an resolve exact (flagged in `match_basis`); genuinely different
titles still mismatch. Pure-function coverage: no database or API needed.
"""

import pytest

from zotero_mcp.exact_resolver import _fold_title, _identity_checks


def _item(title, **extra):
    data = {"title": title}
    data.update(extra)
    return {"key": "ABCD1234", "data": data}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The American Community Survey", "american community survey"),
        ("A Brief History of Time", "brief history of time"),
        ("An Inquiry into Understanding", "inquiry into understanding"),
        ("THE ECONOMIST STYLE GUIDE", "economist style guide"),
        ("American Community Survey", "american community survey"),
        ("Theory of Value", "theory of value"),
        ("An", "an"),
        ("The", "the"),
        ("", ""),
    ],
)
def test_fold_title_strips_single_leading_article(raw, expected):
    assert _fold_title(raw) == expected


def test_identity_exact_without_article_difference_is_unflagged():
    satisfied, mismatched, relaxed = _identity_checks(
        _item("Taxed Out"), {"title": "Taxed Out"}
    )
    assert satisfied == ["title"]
    assert mismatched == []
    assert relaxed is False


def test_identity_article_only_difference_resolves_with_flag():
    satisfied, mismatched, relaxed = _identity_checks(
        _item("The American Community Survey"),
        {"title": "American Community Survey"},
    )
    assert satisfied == ["title"]
    assert mismatched == []
    assert relaxed is True


def test_identity_article_difference_either_direction():
    satisfied, mismatched, relaxed = _identity_checks(
        _item("American Community Survey"),
        {"title": "The American Community Survey"},
    )
    assert satisfied == ["title"]
    assert mismatched == []
    assert relaxed is True


def test_identity_genuinely_different_titles_still_mismatch():
    satisfied, mismatched, relaxed = _identity_checks(
        _item("The American Community Survey: Summary of a Workshop"),
        {"title": "American Community Survey"},
    )
    assert satisfied == []
    assert mismatched == ["title"]
    assert relaxed is False


def test_identity_other_fields_untouched_by_article_logic():
    satisfied, mismatched, relaxed = _identity_checks(
        _item("Some Paper", creators=[{"lastName": "Doe"}], date="2020"),
        {"title": "Some Paper", "author": "Smith", "year": "2020"},
    )
    assert "title" in satisfied
    assert "author" in mismatched
    assert "year" in satisfied
    assert relaxed is False
