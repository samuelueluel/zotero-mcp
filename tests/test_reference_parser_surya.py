"""Reference splitting on Surya sidecar markup."""

from zotero_mcp.reference_parser import extract_reference_sections, parse_reference_entries


def test_list_items_with_bullets_bold_and_page_anchors():
    body = (
        '- ► Acemoglu, Daron. 2002. "Directed Technical Change." *Review of Economic Studies* 69 (4): 781–809.\n'
        "- ► Albouy, David. 2008. \"Are Big Cities Bad Places to Live?\" NBER Working Paper 14472.\n"
        "\n<!-- pdf-page: 45 -->\n\n"
        "- ▶ **Kennan, John, and James R. Walker.** 2011. \"The Effect of Expected Income.\" *Econometrica* 79 (1): 211–51.\n"
    )
    entries = parse_reference_entries(body)
    assert [e.split_method for e in entries] == ["list-item"] * 3
    assert entries[2].raw_text.startswith("Kennan, John, and James R. Walker. 2011.")
    assert not any("pdf-page" in e.raw_text or "►" in e.raw_text for e in entries)


def test_list_items_win_over_stray_number_markers():
    body = "\n\n".join(
        f"- Author{i}, A. 19{60 + i}. \"Title {i}.\" *Journal*, 1{i}(1): 1–2{i}." for i in range(6)
    ) + "\n\n1. stray\n\n2. stray\n"
    entries = parse_reference_entries(body)
    assert len(entries) == 6
    assert {e.split_method for e in entries} == {"list-item"}


def test_glued_entries_split_at_author_year_boundary():
    body = (
        "Actipedia (2014) Heidelberg Project. Available online.Akers J (2015) Emerging market city. "
        "*Environment and Planning A* 47: 1–17.Bennett L (2010) The Third City. Chicago.\n"
        "Hackworth J (2015) Right-sizing as spatial austerity. *Environment and Planning A* 47: 766–782.\n"
    )
    raws = [e.raw_text for e in parse_reference_entries(body)]
    assert any(r.startswith("Akers J (2015)") for r in raws)
    assert any(r.startswith("Bennett L (2010)") for r in raws)
    assert len(raws) == 4


def test_abbreviations_are_not_glued_boundaries():
    body = (
        "Smith, J. 2001. Cities in the U.S. Economy. *Journal of Urban Economics* 1: 1–2.\n"
        "Jones, K. 2002. Housing. *Review* 2: 3–4.\n"
    )
    assert len(parse_reference_entries(body)) == 2


def test_figure_schema_blocks_are_dropped():
    body = (
        "- Alonso, William. 1964. *Location and Land Use*. Cambridge: Harvard University Press.\n"
        "- Barro, Robert J. 1991. \"Economic Growth.\" *QJE* 106 (2): 407–43.\n\n"
        "![Figure, PDF p. 47](images/p047_b0.png)\n"
        "[Figure Schema]\n- Type: Line chart\n- X-Axis: Year\n  - Other Gang\n\n"
    )
    entries = parse_reference_entries(body)
    assert len(entries) == 2
    assert not any("Line chart" in e.raw_text or "Other Gang" in e.raw_text for e in entries)


def test_cited_by_list_ends_the_reference_section():
    text = (
        "## REFERENCES\n\n"
        "- Acemoglu, Daron. 2002. \"Directed Technical Change.\" *REStud* 69 (4): 781–809.\n"
        "- Albouy, David. 2008. \"Big Cities.\" NBER Working Paper 14472.\n\n"
        "This article has been cited by:\n\n"
        "- 1. Mona Ahmadiani, Susana Ferreira. 2019. Environmental amenities. *JUE*.\n"
    )
    (section,) = extract_reference_sections(text)
    assert "cited by" not in section.body
    assert len(parse_reference_entries(section.body)) == 2
