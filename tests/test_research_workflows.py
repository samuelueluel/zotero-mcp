"""Hermetic contracts for bounded composite research workflows."""

from __future__ import annotations

import json

import pytest
from conftest import DummyContext
from pydantic import ValidationError

from zotero_mcp.research_workflows import (
    CandidateScopeDependencies,
    CandidateScopeRequest,
    CandidateScopeService,
    ComparisonManifestRequest,
    ComparisonManifestValidator,
    EvidenceBundleValidationRequest,
    EvidenceBundleValidator,
    ResultEvidenceDependencies,
    ResultEvidenceRequest,
    ResultEvidenceService,
)
from zotero_mcp.tools import research as research_tool

COLLECTION = "COLL0001"
ITEM = "ITEM0001"
OTHER = "ITEM0002"
OUTSIDE = "ITEM9999"


def _item(key: str, title: str, *, parent: str | None = None, item_type: str = "journalArticle"):
    data = {
        "key": key,
        "title": title,
        "date": "2024",
        "itemType": item_type,
        "creators": [{"lastName": "Author"}],
    }
    if parent:
        data["parentItem"] = parent
    return {"key": key, "data": data}


def _hit(key: str, title: str, *, rerank: float | None, chunk: int, reference: bool = False):
    return {
        "item_key": key,
        "similarity_score": 0.75,
        "rerank_score": rerank,
        "matched_passage": f"{title} reports the requested result.",
        "preview_truncated": True,
        "chunk_id": f"{key}#{chunk}",
        "content_hash": "a" * 64,
        "evidence_id": f"zr1:0:{key}#{chunk}:" + "b" * 64,
        "chunk_index": chunk,
        "n_chunks": 5,
        "is_reference": reference,
        "source_group": "article",
        "zotero_item": _item(key, title),
    }


def _inventory():
    first = _item(ITEM, "First")
    first["has_pdf"] = True
    return {
        "collection_name": "Project",
        "collection_keys": [COLLECTION, "SUBC0001"],
        "items": [first, _item(OTHER, "Second")],
    }


def test_candidate_scope_deduplicates_retains_facets_and_sorts_by_positive_rerank():
    calls = []

    def searcher(query, limit, filters, collection_key, include_subcollections, member_keys):
        calls.append((query, limit, filters, collection_key, include_subcollections, member_keys))
        if query == "facet one":
            return {"results": [_hit(ITEM, "First", rerank=2.0, chunk=1), _hit(OTHER, "Second", rerank=-1.0, chunk=2)]}
        return {"results": [_hit(ITEM, "First", rerank=3.0, chunk=3), _hit(OTHER, "Second", rerank=1.0, chunk=4)]}

    request = CandidateScopeRequest(
        collection_key=COLLECTION,
        query_facets=["facet one", "facet two"],
        filters={"source_group": "article"},
    )
    result = CandidateScopeService(
        CandidateScopeDependencies(
            inventory_reader=lambda key, subcollections: _inventory(),
            semantic_searcher=searcher,
        )
    ).build(request)

    assert result["ok"] is True
    assert result["scope"]["member_count"] == 2
    assert result["candidate_count"] == 2
    assert [row["item_key"] for row in result["candidates"]] == [ITEM, OTHER]
    leader = result["candidates"][0]
    assert leader["facet_count"] == 2
    assert leader["has_pdf"] is True
    assert leader["max_positive_rerank"] == pytest.approx(3.0)
    assert len(leader["hits"]) == 2
    assert all(hit["evidence_id"].startswith("zr1:") for hit in leader["hits"])
    assert calls[0][5] == {ITEM, OTHER}


def _comparison_result(result_id="r1", result_class="main"):
    return {
        "result_id": result_id,
        "result_class": result_class,
        "outcome": "crime count",
        "point_estimate": "-0.10",
        "scale": "percentage change",
        "uncertainty": "SE 0.03",
        "treatment": "demolition",
        "dose": "one demolition",
        "denominator": "baseline crime count",
        "population": "neighborhoods",
        "geography": "city",
        "time_horizon": "one year",
        "specification": "fixed effects",
        "evidence_ids": ["evidence-" + result_id],
    }


def _comparison_card(key, status="eligible", result_id="r1"):
    card = {"item_key": key, "status": status}
    if status == "eligible":
        card.update(
            {
                "results": [_comparison_result(result_id)],
                "primary_result_id": result_id,
                "maximum_substantive_result_id": result_id,
                "selected_result_id": result_id,
                "inventory_locators": ["Table 2, PDF p. 6"],
            }
        )
    elif status == "no_eligible_result":
        card["reason"] = "No crime outcome was estimated."
    return card


def _comparison_manifest_defaults():
    return {
        "eligible_result_policy": "substantive_all",
        "numerical_winner_status": "clear",
        "substantive_winner_status": "clear",
        "alternative_policy_changes_top_k": False,
    }


def test_comparison_manifest_requires_complete_frozen_item_coverage():
    request = ComparisonManifestRequest(
        frozen_item_keys=[ITEM, OTHER],
        cards=[_comparison_card(ITEM), _comparison_card(OTHER, "no_eligible_result")],
        ranking_rule="Largest absolute percentage change in a primary crime outcome.",
        selected_item_keys=[ITEM],
        reported_item_keys=[ITEM],
        **_comparison_manifest_defaults(),
    )
    result = ComparisonManifestValidator().validate(request)
    assert result["ready"] is True
    assert result["complete"] is True
    assert result["summary"]["eligible"] == 1


def test_comparison_manifest_blocks_missing_cards_and_unresolved_complete_scope():
    request = ComparisonManifestRequest(
        frozen_item_keys=[ITEM, OTHER],
        cards=[_comparison_card(ITEM), _comparison_card(OTHER, "unresolved")],
        ranking_rule="Largest point estimate.",
        selected_item_keys=[ITEM],
        **_comparison_manifest_defaults(),
    )
    result = ComparisonManifestValidator().validate(request)
    assert result["ready"] is False
    assert "UNRESOLVED_ITEMS" in result["reason_codes"]

    missing = ComparisonManifestRequest(
        frozen_item_keys=[ITEM, OTHER],
        cards=[_comparison_card(ITEM)],
        ranking_rule="Largest point estimate.",
        selected_item_keys=[ITEM],
        **_comparison_manifest_defaults(),
    )
    result = ComparisonManifestValidator().validate(missing)
    assert "CARD_SET_MISMATCH" in result["reason_codes"]


def test_comparison_manifest_rejects_inconsistent_card_selection():
    card = _comparison_card(ITEM, "no_eligible_result")
    card["selected_result_id"] = "r1"
    request = ComparisonManifestRequest(
        frozen_item_keys=[ITEM],
        cards=[card],
        ranking_rule="Largest point estimate.",
        selected_item_keys=[ITEM],
        **_comparison_manifest_defaults(),
    )
    result = ComparisonManifestValidator().validate(request)
    assert result["ready"] is False
    assert "INELIGIBLE_CARD_SELECTED" in result["reason_codes"]


def test_comparison_manifest_enforces_reported_top_set():
    request = ComparisonManifestRequest(
        frozen_item_keys=[ITEM, OTHER],
        cards=[_comparison_card(ITEM), _comparison_card(OTHER)],
        ranking_rule="Largest absolute percentage change.",
        selected_item_keys=[ITEM],
        reported_item_keys=[OTHER],
        **_comparison_manifest_defaults(),
    )
    result = ComparisonManifestValidator().validate(request)
    assert result["ready"] is False
    assert "REPORTED_ITEM_NOT_SELECTED" in result["reason_codes"]


def test_comparison_manifest_enforces_primary_and_maximum_substantive_policy():
    card = _comparison_card(ITEM)
    card["results"].append(_comparison_result("dynamic", "dynamic"))
    card["maximum_substantive_result_id"] = "dynamic"
    request = ComparisonManifestRequest(
        frozen_item_keys=[ITEM],
        cards=[card],
        ranking_rule="Largest significant substantive effect.",
        eligible_result_policy="substantive_all",
        numerical_winner_status="clear",
        substantive_winner_status="clear",
        alternative_policy_changes_top_k=False,
        selected_item_keys=[ITEM],
    )
    result = ComparisonManifestValidator().validate(request)
    assert result["ready"] is False
    assert "SELECTED_RESULT_POLICY_MISMATCH" in result["reason_codes"]

    card["selected_result_id"] = "dynamic"
    request = ComparisonManifestRequest(
        frozen_item_keys=[ITEM],
        cards=[card],
        ranking_rule="Largest significant substantive effect.",
        eligible_result_policy="substantive_all",
        numerical_winner_status="clear",
        substantive_winner_status="clear",
        alternative_policy_changes_top_k=False,
        selected_item_keys=[ITEM],
    )
    result = ComparisonManifestValidator().validate(request)
    assert result["ready"] is True


def test_comparison_manifest_requires_top_k_when_only_numerical_winner_is_clear():
    request = ComparisonManifestRequest(
        frozen_item_keys=[ITEM],
        cards=[_comparison_card(ITEM)],
        ranking_rule="Largest heterogeneous effect.",
        eligible_result_policy="substantive_all",
        numerical_winner_status="clear",
        substantive_winner_status="not_clear",
        alternative_policy_changes_top_k=True,
        winner_type="clear_winner",
        selected_item_keys=[ITEM],
    )
    result = ComparisonManifestValidator().validate(request)
    assert result["ready"] is False
    assert "SUBSTANTIVE_WINNER_NOT_CLEAR" in result["reason_codes"]


def test_candidate_scope_rejects_backend_scope_leaks():
    result = CandidateScopeService(
        CandidateScopeDependencies(
            inventory_reader=lambda key, subcollections: _inventory(),
            semantic_searcher=lambda *args: {
                "results": [
                    _hit(ITEM, "First", rerank=1.0, chunk=1),
                    _hit(OUTSIDE, "Outside", rerank=9.0, chunk=1),
                ]
            },
        )
    ).build(CandidateScopeRequest(collection_key=COLLECTION, query_facets=["facet"]))

    assert result["candidate_count"] == 1
    assert result["scope_leak_count"] == 1
    assert result["scope_leaks"][0]["item_key"] == OUTSIDE


def test_candidate_scope_inventory_output_is_bounded_without_changing_member_scope():
    inventory = _inventory()
    result = CandidateScopeService(
        CandidateScopeDependencies(
            inventory_reader=lambda key, subcollections: inventory,
            semantic_searcher=lambda *args: {"results": []},
        )
    ).build(
        CandidateScopeRequest(
            collection_key=COLLECTION,
            query_facets=["facet"],
            inventory_limit=1,
        )
    )

    assert result["scope"]["member_count"] == 2
    assert result["scope"]["inventory_returned"] == 1
    assert result["scope"]["inventory_complete"] is False


def test_candidate_scope_fails_closed_when_one_facet_errors():
    result = CandidateScopeService(
        CandidateScopeDependencies(
            inventory_reader=lambda key, subcollections: _inventory(),
            semantic_searcher=lambda *args: {"error": "reranker unavailable"},
        )
    ).build(CandidateScopeRequest(collection_key=COLLECTION, query_facets=["facet"]))

    assert result["ok"] is False
    assert result["error"]["code"] == "SEARCH_FAILED"


def test_candidate_scope_input_bounds_and_duplicate_facets():
    with pytest.raises(ValidationError):
        CandidateScopeRequest(collection_key="bad", query_facets=["facet"])
    with pytest.raises(ValidationError):
        CandidateScopeRequest(collection_key=COLLECTION, query_facets=["same", "SAME"])
    with pytest.raises(ValidationError):
        CandidateScopeRequest(collection_key=COLLECTION, query_facets=[str(i) for i in range(5)])


def test_build_candidate_scope_adapter_parses_json_and_preserves_request(monkeypatch):
    captured = {}

    class Service:
        def __init__(self, dependencies):
            captured["dependencies"] = dependencies

        def build(self, request):
            captured["request"] = request
            return {"schema_version": 1, "ok": True, "candidate_count": 0}

    monkeypatch.setattr(research_tool, "CandidateScopeService", Service)
    raw = research_tool.build_candidate_scope(
        collection_key=COLLECTION,
        query_facets='["one", "two"]',
        filters='{"source_group": "article"}',
        ctx=DummyContext(),
    )
    result = json.loads(raw)
    assert result["ok"] is True
    assert captured["request"].query_facets == ["one", "two"]
    assert captured["request"].filters == {"source_group": "article"}


def test_direct_collection_scope_intersects_existing_item_filter():
    result = research_tool._merge_item_scope(
        {"item_keys": [ITEM, OUTSIDE], "source_group": "article"},
        {ITEM, OTHER},
    )
    assert result == {"item_keys": [ITEM], "source_group": "article"}


def _result_dependencies(
    *,
    passage_text="Indexed estimate discussion.",
    sidecar_text="Estimate -0.072*** (0.020)",
    pdf_text="Estimate -0.072*** (0.020)",
    pdf_coverage="complete",
):
    calls = {"passage": [], "sidecar": [], "pdf": []}

    def passage_reader(token, neighbors, max_chars):
        calls["passage"].append((token, neighbors, max_chars))
        return {
            "ok": True,
            "route": "indexed_passage",
            "chunks": [{"text": passage_text}],
        }

    def sidecar_reader(key, **kwargs):
        calls["sidecar"].append((key, kwargs))
        if kwargs.get("start_char") is not None:
            return {
                "ok": True,
                "route": "mineru_sidecar",
                "source_hash": "a" * 64,
                "windows": [{"text": "continued table notes"}],
                "truncated": False,
            }
        return {
            "ok": True,
            "route": "mineru_sidecar",
            "source_hash": "a" * 64,
            "windows": [{"text": sidecar_text}],
            "truncated": True,
            "next_char_start": 500,
        }

    def pdf_reader(key, queries, start, end, windows, max_chars):
        calls["pdf"].append((key, queries, start, end, windows, max_chars))
        return {
            "ok": True,
            "route": "pdf_extraction",
            "queries": [
                {
                    "ok": True,
                    "query": query,
                    "coverage": pdf_coverage,
                    "matches": ([{"page": 6, "text": pdf_text}] if pdf_text else []),
                }
                for query in queries
            ],
        }

    return (
        ResultEvidenceDependencies(
            parent_resolver=lambda key: {"item_key": key, "title": "Paper", "library_id": 0},
            passage_reader=passage_reader,
            sidecar_reader=sidecar_reader,
            pdf_reader=pdf_reader,
        ),
        calls,
    )


def test_result_evidence_preserves_routes_and_hash_chains_continuation():
    dependencies, calls = _result_dependencies()
    request = ResultEvidenceRequest(
        requests=[
            {
                "item_key": ITEM,
                "evidence_id": "zr1:0:ITEM0001#1:" + "b" * 64,
                "sidecar_queries": ["Table 5", "Table notes"],
                "pdf_queries": ["Table 5"],
            }
        ]
    )
    result = ResultEvidenceService(dependencies).collect(request)
    row = result["items"][0]

    assert row["indexed_passage"]["route"] == "indexed_passage"
    assert row["sidecar"][0]["route"] == "mineru_sidecar"
    assert row["pdf"]["route"] == "pdf_extraction"
    assert calls["passage"][0][1] == 1
    assert calls["sidecar"][1][1]["expected_hash"] == "a" * 64
    assert calls["sidecar"][1][1]["start_char"] == 500
    assert calls["sidecar"][2][1]["expected_hash"] == "a" * 64
    assert row["requires_visual_review"] is False


def test_result_evidence_flags_numeric_and_star_conflicts_without_repair():
    dependencies, _ = _result_dependencies(
        sidecar_text="Estimate -0.072*** (0.020)",
        pdf_text="Estimate 0.072 (0.020)",
    )
    result = ResultEvidenceService(dependencies).collect(
        ResultEvidenceRequest(
            requests=[
                {
                    "item_key": ITEM,
                    "sidecar_queries": ["Table 5"],
                    "pdf_queries": ["Table 5"],
                }
            ]
        )
    )
    row = result["items"][0]
    codes = {flag["code"] for flag in row["conflict_flags"]}
    assert codes == {"NUMERIC_SIGNATURE_MISMATCH", "SIGNIFICANCE_MARKER_MISMATCH"}
    assert row["requires_visual_review"] is True


def test_result_evidence_ignores_markdown_bold_around_table_numbers():
    dependencies, _ = _result_dependencies(
        sidecar_text="Results refer to Table 5.",
        pdf_text="**TABLE 5** Results.",
    )
    row = ResultEvidenceService(dependencies).collect(
        ResultEvidenceRequest(
            requests=[
                {
                    "item_key": ITEM,
                    "sidecar_queries": ["Table 5"],
                    "pdf_queries": ["Table 5"],
                }
            ]
        )
    )["items"][0]
    assert row["conflict_flags"] == []
    assert row["requires_visual_review"] is False


def test_result_evidence_flags_referenced_table_not_read():
    dependencies, _ = _result_dependencies(
        passage_text="The decisive estimates appear in Table 4.",
        pdf_text="Results prose without the table caption.",
    )
    row = ResultEvidenceService(dependencies).collect(
        ResultEvidenceRequest(
            requests=[
                {
                    "item_key": ITEM,
                    "evidence_id": "zr1:0:ITEM0001#1:" + "b" * 64,
                    "pdf_queries": ["decisive estimates"],
                }
            ]
        )
    )["items"][0]
    assert row["referenced_tables"] == [4]
    assert row["referenced_tables_not_read"] == [4]
    assert row["requires_follow_up"] is True
    assert "REFERENCED_TABLE_NOT_READ" in {
        flag["code"] for flag in row["conflict_flags"]
    }


def test_result_evidence_clears_table_follow_up_when_pdf_table_is_read():
    dependencies, _ = _result_dependencies(
        passage_text="The decisive estimates appear in Table 4.",
        pdf_text="Table 4: Decisive estimates.",
    )
    row = ResultEvidenceService(dependencies).collect(
        ResultEvidenceRequest(
            requests=[
                {
                    "item_key": ITEM,
                    "evidence_id": "zr1:0:ITEM0001#1:" + "b" * 64,
                    "pdf_queries": ["Table 4"],
                }
            ]
        )
    )["items"][0]
    assert row["read_referenced_tables"] == [4]
    assert "referenced_tables_not_read" not in row
    assert row["requires_follow_up"] is False


def test_result_evidence_flags_no_match_on_incomplete_pdf_text():
    dependencies, _ = _result_dependencies(pdf_text="", pdf_coverage="partial_text_coverage")
    result = ResultEvidenceService(dependencies).collect(
        ResultEvidenceRequest(
            requests=[{"item_key": ITEM, "pdf_queries": ["Table 5"]}]
        )
    )
    row = result["items"][0]
    assert row["conflict_flags"] == [
        {
            "code": "INCOMPLETE_PDF_TEXT_NO_MATCH",
            "query": "Table 5",
            "coverage": "partial_text_coverage",
        }
    ]
    assert row["requires_visual_review"] is True


def test_result_evidence_marks_requested_route_failure_without_discarding_other_routes():
    dependencies, _ = _result_dependencies()
    dependencies = ResultEvidenceDependencies(
        parent_resolver=dependencies.parent_resolver,
        passage_reader=dependencies.passage_reader,
        sidecar_reader=lambda *args, **kwargs: {
            "ok": False,
            "error": {"code": "SIDECAR_NOT_FOUND", "message": "missing"},
        },
        pdf_reader=dependencies.pdf_reader,
    )
    result = ResultEvidenceService(dependencies).collect(
        ResultEvidenceRequest(
            requests=[
                {
                    "item_key": ITEM,
                    "sidecar_queries": ["Table 5"],
                    "pdf_queries": ["Table 5"],
                }
            ]
        )
    )
    row = result["items"][0]
    assert result["ok"] is False
    assert row["ok"] is False
    assert row["pdf"]["ok"] is True
    assert row["route_errors"][0]["route"] == "mineru_sidecar"


def test_result_evidence_distinguishes_incomplete_pdf_search_from_visual_conflict():
    dependencies, _ = _result_dependencies()

    def pdf_reader(*args):
        return {
            "ok": True,
            "route": "pdf_extraction",
            "search_complete": False,
            "searched_page_range": [1, 100],
            "requested_page_range": [1, 150],
            "queries": [
                {
                    "ok": True,
                    "query": "Table 5",
                    "coverage": "complete",
                    "matches": [],
                }
            ],
        }

    dependencies = ResultEvidenceDependencies(
        parent_resolver=dependencies.parent_resolver,
        passage_reader=dependencies.passage_reader,
        sidecar_reader=dependencies.sidecar_reader,
        pdf_reader=pdf_reader,
    )
    row = ResultEvidenceService(dependencies).collect(
        ResultEvidenceRequest(
            requests=[{"item_key": ITEM, "pdf_queries": ["Table 5"]}]
        )
    )["items"][0]
    assert row["conflict_flags"][0]["code"] == "PDF_SEARCH_INCOMPLETE"
    assert row["requires_visual_review"] is False


def test_result_evidence_parent_failure_stops_item_reads():
    called = []
    dependencies = ResultEvidenceDependencies(
        parent_resolver=lambda key: {"error": {"code": "INVALID_ITEM", "message": "bad"}},
        passage_reader=lambda *args: called.append("passage"),
        sidecar_reader=lambda *args, **kwargs: called.append("sidecar"),
        pdf_reader=lambda *args: called.append("pdf"),
    )
    result = ResultEvidenceService(dependencies).collect(
        ResultEvidenceRequest(requests=[{"item_key": ITEM, "pdf_queries": ["Table 5"]}])
    )
    assert result["ok"] is False
    assert result["items"][0]["error"]["code"] == "INVALID_ITEM"
    assert called == []


def test_result_evidence_input_requires_distinct_items_and_a_route():
    with pytest.raises(ValidationError):
        ResultEvidenceRequest(requests=[{"item_key": ITEM}])
    with pytest.raises(ValidationError):
        ResultEvidenceRequest(
            requests=[
                {"item_key": ITEM, "pdf_queries": ["Table 5"]},
                {"item_key": ITEM, "pdf_queries": ["Table 6"]},
            ]
        )


def test_collect_result_evidence_adapter_parses_json(monkeypatch):
    captured = {}

    class Service:
        def __init__(self, dependencies):
            captured["dependencies"] = dependencies

        def collect(self, request):
            captured["request"] = request
            return {"schema_version": 1, "ok": True, "items": []}

    monkeypatch.setattr(research_tool, "ResultEvidenceService", Service)
    raw = research_tool.collect_result_evidence(
        requests='[{"item_key":"ITEM0001","pdf_queries":["Table 5"]}]',
        ctx=DummyContext(),
    )
    assert json.loads(raw)["ok"] is True
    assert captured["request"].requests[0].pdf_queries == ["Table 5"]


def _evidence_record(evidence_id="e1", item_key=ITEM, **extra):
    return {
        "evidence_id": evidence_id,
        "item_key": item_key,
        "route": "pdf_extraction",
        "locator": "Table 5, PDF p. 6",
        "page": 6,
        "quote": "The treatment reduced crime by 11% (95% CI 7–15%).",
        **extra,
    }


def _numeric_claim(**extra):
    claim = {
        "claim_id": "c1",
        "text": "The treatment reduced crime by 11% (95% CI 7–15%).",
        "evidence_ids": ["e1"],
        "risk_tags": ["numeric"],
        "context": {
            "outcome": "reported crime",
            "estimate": "11% reduction",
            "scale": "percentage change",
            "treatment": "program treatment",
            "dose": "one program assignment",
            "denominator": "control mean",
            "time_horizon": "post-treatment period",
            "uncertainty_status": "reported",
        },
    }
    claim.update(extra)
    return claim


def test_evidence_bundle_validator_accepts_complete_numeric_claim():
    result = EvidenceBundleValidator().validate(
        EvidenceBundleValidationRequest(
            claims=[_numeric_claim()],
            evidence=[_evidence_record()],
        )
    )
    assert result["ready"] is True
    assert result["results"][0]["status"] == "ready"


def test_evidence_bundle_validator_uses_structured_expected_values():
    claim = _numeric_claim(
        text="Table 6 reports an effect of -0.164 with SE 0.052.",
        expected_values=[
            {"role": "estimate", "value": "-0.164"},
            {"role": "se", "value": "0.052"},
        ],
    )
    evidence = _evidence_record(quote="Table row | -.164** | .052")
    result = EvidenceBundleValidator().validate(
        EvidenceBundleValidationRequest(claims=[claim], evidence=[evidence])
    )
    assert result["ready"] is True


def test_evidence_bundle_validator_blocks_claims_outside_allowed_items():
    result = EvidenceBundleValidator().validate(
        EvidenceBundleValidationRequest(
            claims=[_numeric_claim()],
            evidence=[_evidence_record()],
            allowed_item_keys=[OTHER],
        )
    )
    row = result["results"][0]
    assert row["status"] == "blocked"
    assert "ITEM_OUTSIDE_ALLOWED_SCOPE" in row["reason_codes"]


def test_evidence_bundle_validator_blocks_missing_context_and_quote_numbers():
    claim = _numeric_claim(
        text="The treatment reduced crime by 17%.",
        context={"outcome": "crime", "estimate": "17%"},
    )
    result = EvidenceBundleValidator().validate(
        EvidenceBundleValidationRequest(claims=[claim], evidence=[_evidence_record()])
    )
    codes = set(result["results"][0]["reason_codes"])
    assert "NUMERIC_CONTEXT_MISSING" in codes
    assert "NUMERIC_VALUE_NOT_IN_QUOTE" in codes
    assert result["ready"] is False


def test_evidence_bundle_validator_requires_two_comparison_items():
    claim = _numeric_claim(risk_tags=["numeric", "comparison"])
    result = EvidenceBundleValidator().validate(
        EvidenceBundleValidationRequest(claims=[claim], evidence=[_evidence_record()])
    )
    assert "COMPARISON_ITEM_MISSING" in result["results"][0]["reason_codes"]


def test_evidence_bundle_validator_requires_calculation_method():
    claim = _numeric_claim(risk_tags=["numeric", "calculated"])
    result = EvidenceBundleValidator().validate(
        EvidenceBundleValidationRequest(claims=[claim], evidence=[_evidence_record()])
    )
    assert "CALCULATION_METHOD_MISSING" in result["results"][0]["reason_codes"]


def test_evidence_bundle_validator_blocks_ambiguity_unless_claim_is_unverified():
    evidence = _evidence_record(ambiguity_flags=["SIGN_MISMATCH"])
    blocked = EvidenceBundleValidator().validate(
        EvidenceBundleValidationRequest(claims=[_numeric_claim()], evidence=[evidence])
    )
    assert "AMBIGUOUS_EVIDENCE" in blocked["results"][0]["reason_codes"]

    allowed = EvidenceBundleValidator().validate(
        EvidenceBundleValidationRequest(
            claims=[_numeric_claim(unverified=True)],
            evidence=[evidence],
        )
    )
    assert allowed["ready"] is True


def test_evidence_bundle_validator_requires_page_for_pdf_route():
    evidence = _evidence_record()
    evidence.pop("page")
    result = EvidenceBundleValidator().validate(
        EvidenceBundleValidationRequest(claims=[_numeric_claim()], evidence=[evidence])
    )
    assert "PDF_PAGE_MISSING" in result["results"][0]["reason_codes"]


def test_validate_evidence_bundle_adapter_parses_json(monkeypatch):
    captured = {}

    class Validator:
        def validate(self, request):
            captured["request"] = request
            return {"schema_version": 1, "ok": True, "ready": True, "results": []}

    monkeypatch.setattr(research_tool, "EvidenceBundleValidator", Validator)
    raw = research_tool.validate_evidence_bundle(
        claims=json.dumps([_numeric_claim()]),
        evidence=json.dumps([_evidence_record()]),
        allowed_item_keys=json.dumps([ITEM]),
        ctx=DummyContext(),
    )
    assert json.loads(raw)["ready"] is True
    assert captured["request"].claims[0].claim_id == "c1"
    assert captured["request"].allowed_item_keys == [ITEM]


def test_validate_comparison_manifest_adapter_parses_json(monkeypatch):
    captured = {}

    class Validator:
        def validate(self, request):
            captured["request"] = request
            return {"schema_version": 1, "ok": True, "ready": True}

    monkeypatch.setattr(research_tool, "ComparisonManifestValidator", Validator)
    manifest = {
        "frozen_item_keys": [ITEM],
        "cards": [_comparison_card(ITEM)],
        "ranking_rule": "Largest point estimate.",
        "selected_item_keys": [ITEM],
        **_comparison_manifest_defaults(),
    }
    raw = research_tool.validate_comparison_manifest(
        manifest=json.dumps(manifest),
        ctx=DummyContext(),
    )
    result = json.loads(raw)
    assert result["ready"] is True
    assert captured["request"].frozen_item_keys == [ITEM]


def test_candidate_searcher_reuses_one_semantic_service(monkeypatch):
    from zotero_mcp import semantic_search as semantic_module

    created = []

    class Search:
        def search(self, **kwargs):
            return {"results": [], "kwargs": kwargs}

    def create(path):
        created.append(path)
        return Search()

    monkeypatch.setattr(semantic_module, "create_semantic_search", create)
    monkeypatch.setattr(research_tool._client, "get_active_group_id", lambda: 0)
    searcher = research_tool._make_semantic_searcher()
    first = searcher("one", 8, None, COLLECTION, True, {ITEM})
    second = searcher("two", 8, None, COLLECTION, True, {ITEM})
    assert len(created) == 1
    assert first["kwargs"]["collection_key"] == COLLECTION
    assert second["kwargs"]["query"] == "two"


def test_pdf_query_reader_uses_bounded_windows_and_cleans_temp_file(monkeypatch):
    from zotero_mcp import extract as extract_module
    from zotero_mcp import pdf_evidence
    from zotero_mcp.tools import read_pdf as read_pdf_tool

    extracted_pages = []
    cleaned = []

    class Document:
        def __init__(self, pages):
            self.page_numbers = [page + 1 for page in pages]
            self.pages = [f"page {page + 1}" for page in pages]

    monkeypatch.setattr(
        read_pdf_tool,
        "_get_pdf_path",
        lambda key, ctx: ("/tmp/zotero_pdf_test/paper.pdf", "Paper", True),
    )
    monkeypatch.setattr(read_pdf_tool, "_cleanup_path", lambda path: cleaned.append(path))
    monkeypatch.setattr(extract_module, "pdf_page_count", lambda path: 120)

    def extract(path, pages):
        extracted_pages.append(pages)
        return Document(pages)

    monkeypatch.setattr(extract_module, "extract_pdf", extract)
    monkeypatch.setattr(pdf_evidence, "compile_literal_pattern", lambda query: query)
    monkeypatch.setattr(
        pdf_evidence,
        "classify_text_coverage",
        lambda document: {"state": "complete"},
    )
    monkeypatch.setattr(
        pdf_evidence,
        "find_literal_matches",
        lambda document, query, **kwargs: {
            "total_matches": 1,
            "matches": [{"page": document.page_numbers[0], "text": f"{query} match"}],
        },
    )

    result = research_tool._read_pdf_queries(
        ITEM,
        ["Table 5"],
        None,
        None,
        2,
        8000,
        DummyContext(),
    )
    assert result["ok"] is True
    assert result["source_route"] == "downloaded_pdf"
    assert result["searched_page_range"] == [1, 100]
    assert result["search_complete"] is False
    assert [len(pages) for pages in extracted_pages] == [50, 50]
    assert result["queries"][0]["total_matches"] == 2
    assert cleaned == ["/tmp/zotero_pdf_test/paper.pdf"]
