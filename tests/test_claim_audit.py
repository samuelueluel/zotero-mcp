"""Hermetic tests for the bounded Zotero RAG claim audit."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from conftest import DummyContext
from pydantic import ValidationError

from zotero_mcp.claim_audit import (
    AuditDependencies,
    AuditService,
    CheckerResult,
    EvidenceRecord,
    LocalOpenAIClaimChecker,
    parse_claims,
    quote_contained,
)
from zotero_mcp.tools import claim_audit as claim_audit_tool

ITEM = "ITEM0001"
OTHER_ITEM = "ITEM0002"


class FakeChecker:
    def __init__(self, result="supported"):
        self.result = result
        self.calls = []

    def metadata(self):
        return {"configured": True, "model_id": "fake-checker", "prompt_version": "test"}

    def check(self, claim, evidence):
        self.calls.append((claim, list(evidence)))
        return CheckerResult(self.result, "fake rationale", "add the limitation" if self.result == "revise" else None)


def _metadata(key=ITEM):
    return {"key": key, "data": {"key": key, "itemType": "journalArticle"}}


def _semantic_ref(quote="The treatment reduced emissions by 5%.", **extra):
    return {
        "route": "semantic",
        "item_key": ITEM,
        "query": "treatment emissions",
        "quote": quote,
        **extra,
    }


def _pdf_ref(quote="The treatment reduced emissions by 5%.", **extra):
    return {
        "route": "pdf_page",
        "item_key": ITEM,
        "page": 4,
        "quote": quote,
        **extra,
    }


def _sidecar_ref(quote="The treatment reduced emissions by 5%.", **extra):
    return {
        "route": "mineru_sidecar",
        "item_key": ITEM,
        "locator": "chars 0-100",
        "quote": quote,
        **extra,
    }


def _claim(evidence, text="The treatment reduced emissions by 5%.", tags=None, claim_id="c1"):
    return {
        "claim_id": claim_id,
        "text": text,
        "risk_tags": tags or [],
        "evidence": evidence,
    }


def _deps(*, checker=None, retriever=None, page_reader=None, sidecar_reader=None, sidecar_search=None):
    return AuditDependencies(
        retriever=retriever,
        page_reader=page_reader,
        sidecar_reader=sidecar_reader,
        sidecar_search=sidecar_search,
        metadata_resolver=lambda key: _metadata(key),
        checker=checker,
    )


def _positive_hit(key=ITEM, quote="The treatment reduced emissions by 5%.", **extra):
    return {
        "item_key": key,
        "chunk_id": f"{key}#0",
        "document": f"Intro. {quote} Conclusion.",
        "matched_passage": quote,
        "rerank_score": 1.25,
        **extra,
    }


def test_quote_matching_tolerates_line_hyphenation_but_not_fuzzy_text():
    assert quote_contained("cost-effective policy", "The cost-\neffective policy worked.")
    assert not quote_contained("cost effective policy", "The policy worked.")


def test_strict_input_rejects_paths_content_and_caller_scores():
    for forbidden in ("path", "source_text", "content", "rerank_score"):
        with pytest.raises(ValidationError):
            parse_claims([_claim([{**_semantic_ref(), forbidden: "bad"}])])


def test_input_bounds_are_enforced():
    with pytest.raises(ValueError, match="between 1 and 8"):
        parse_claims([_claim([_semantic_ref()], claim_id=str(i)) for i in range(9)])
    with pytest.raises(ValidationError):
        parse_claims([_claim([_semantic_ref(quote="x" * 1601)])])
    with pytest.raises(ValidationError):
        parse_claims([_claim([_pdf_ref(end_page=8)])])


def test_positive_semantic_evidence_can_be_verified():
    checker = FakeChecker()
    quote = "The treatment reduced emissions."
    service = AuditService(
        _deps(
            checker=checker,
            retriever=lambda query, key: [_positive_hit(key, quote=quote)],
        )
    )

    result = service.audit(
        [_claim([_semantic_ref(quote=quote)], text="The treatment reduced emissions.")]
    )

    row = result["results"][0]
    assert row["verified"] is True
    assert row["status"] == "supported"
    assert row["checker_status"] == "supported"
    assert row["evidence"][0]["item_key"] == ITEM
    assert row["evidence"][0]["route"] == "zotero_semantic_search"
    assert row["evidence"][0]["raw_rerank"] == pytest.approx(1.25)
    assert len(checker.calls) == 1


@pytest.mark.parametrize(
    "score,code",
    [(None, "RERANK_MISSING"), (0.0, "NONPOSITIVE_RERANK"), (-1.0, "NONPOSITIVE_RERANK")],
)
def test_semantic_evidence_requires_a_positive_raw_rerank(score, code):
    checker = FakeChecker()
    hit = _positive_hit()
    hit["rerank_score"] = score
    result = AuditService(_deps(checker=checker, retriever=lambda q, k: [hit])).audit(
        [_claim([_semantic_ref()], text="The policy worked.")]
    )
    row = result["results"][0]
    assert row["verified"] is False
    assert row["status"] == "insufficient"
    assert code in row["reason_codes"]
    assert not checker.calls


def test_reference_hit_is_not_substantive_evidence():
    hit = _positive_hit(is_reference=True)
    result = AuditService(_deps(retriever=lambda q, k: [hit])).audit(
        [_claim([_semantic_ref()], text="The policy worked.")]
    )
    row = result["results"][0]
    assert row["verified"] is False
    assert "BIBLIOGRAPHY_ONLY" in row["reason_codes"]


def test_foreign_hit_is_rejected_even_when_quote_and_score_are_good():
    result = AuditService(
        _deps(retriever=lambda q, k: [_positive_hit(OTHER_ITEM)])
    ).audit([_claim([_semantic_ref()], text="The policy worked.")])
    row = result["results"][0]
    assert row["verified"] is False
    assert row["status"] == "insufficient"
    assert "ITEM_MISMATCH" in row["reason_codes"]


def test_quote_mismatch_is_a_deterministic_failure_and_checker_cannot_override_it():
    checker = FakeChecker()
    result = AuditService(
        _deps(checker=checker, retriever=lambda q, k: [_positive_hit(quote="Different text.")])
    ).audit([_claim([_semantic_ref(quote="Candidate text.")], text="The policy worked.")])
    row = result["results"][0]
    assert row["verified"] is False
    assert "QUOTE_NOT_FOUND" in row["reason_codes"]
    assert not checker.calls


def test_numeric_claim_requires_direct_evidence():
    checker = FakeChecker()
    result = AuditService(
        _deps(checker=checker, retriever=lambda q, k: [_positive_hit()])
    ).audit([_claim([_semantic_ref()])])
    row = result["results"][0]
    assert row["verified"] is False
    assert "NEEDS_DIRECT_EVIDENCE" in row["reason_codes"]
    assert not checker.calls


def test_numeric_claim_can_use_pdf_page_evidence():
    checker = FakeChecker()
    page_text = "Results. The treatment reduced emissions by 5%."
    result = AuditService(
        _deps(
            checker=checker,
            page_reader=lambda *args: {"text": page_text, "needs_ocr": False},
        )
    ).audit([_claim([_pdf_ref()])])
    row = result["results"][0]
    assert row["verified"] is True
    assert row["evidence"][0]["route"] == "zotero_read_pdf_pages"
    assert "raw_rerank" not in row["evidence"][0]


def test_sidecar_without_failed_page_is_not_a_numeric_verification():
    checker = FakeChecker()
    sidecar_text = "Results. The treatment reduced emissions by 5%."
    result = AuditService(
        _deps(
            checker=checker,
            sidecar_reader=lambda *args: {"text": sidecar_text},
        )
    ).audit([_claim([_sidecar_ref()])])
    row = result["results"][0]
    assert row["verified"] is False
    assert "SIDECAR_FALLBACK_REQUIRES_PAGE_FAILURE" in row["reason_codes"]
    assert not checker.calls


def test_sidecar_can_rescue_a_scanned_pdf_page_as_weaker_evidence():
    checker = FakeChecker()
    sidecar_text = "Results. The treatment reduced emissions by 5%."
    result = AuditService(
        _deps(
            checker=checker,
            page_reader=lambda *args: {"text": "", "needs_ocr": True},
            sidecar_reader=lambda *args: {"text": sidecar_text},
        )
    ).audit([_claim([_pdf_ref(quote="No text."), _sidecar_ref()])])
    row = result["results"][0]
    assert row["verified"] is True
    assert row["evidence"][0]["route"] == "mineru_sidecar"
    assert row["evidence"][0]["weaker_evidence"] is True
    assert "SCANNED_PAGE" not in row["reason_codes"]


def test_stale_content_hash_is_rejected():
    checker = FakeChecker()
    result = AuditService(
        _deps(
            checker=checker,
            page_reader=lambda *args: {"text": "The policy worked.", "needs_ocr": False},
        )
    ).audit([_claim([_pdf_ref(quote="The policy worked.", content_hash="0" * 64)], text="The policy worked.")])
    assert "STALE_EVIDENCE" in result["results"][0]["reason_codes"]
    assert not checker.calls


def test_rules_only_never_claims_semantic_support():
    checker = FakeChecker()
    result = AuditService(
        _deps(checker=checker, retriever=lambda q, k: [_positive_hit()])
    ).audit(
        [_claim([_semantic_ref()], text="The policy worked.")],
        check_mode="rules_only",
    )
    row = result["results"][0]
    assert row["verified"] is False
    assert row["checker_status"] == "skipped"
    assert "CHECKER_SKIPPED" in row["reason_codes"]
    assert not checker.calls


def test_comparison_claim_requires_two_distinct_validated_item_keys():
    checker = FakeChecker()
    one_key = AuditService(
        _deps(
            checker=checker,
            retriever=lambda q, k: [_positive_hit(k, quote="The study found an effect.")],
        )
    ).audit(
        [
            _claim(
                [_semantic_ref(quote="The study found an effect.")],
                text="Item A performed better than Item B.",
                tags=["comparison"],
            )
        ]
    )
    assert one_key["results"][0]["status"] == "insufficient"
    assert "COMPARATOR_EVIDENCE_MISSING" in one_key["results"][0]["reason_codes"]
    assert not checker.calls

    second_ref = {
        "route": "semantic",
        "item_key": OTHER_ITEM,
        "query": "study effect",
        "quote": "The study found an effect.",
    }
    two_keys = AuditService(
        _deps(
            checker=checker,
            retriever=lambda q, k: [_positive_hit(k, quote="The study found an effect.")],
        )
    ).audit(
        [
            _claim(
                [_semantic_ref(quote="The study found an effect."), second_ref],
                text="Item A performed better than Item B.",
                tags=["comparison"],
                claim_id="c2",
            )
        ]
    )
    assert two_keys["results"][0]["verified"] is True

    invalid_second = {
        "route": "semantic",
        "item_key": OTHER_ITEM,
        "query": "study effect",
        "quote": "missing from returned evidence",
    }
    rejected = AuditService(
        _deps(
            checker=checker,
            retriever=lambda q, k: [_positive_hit(k, quote="The study found an effect.")]
            if k == ITEM
            else [_positive_hit(ITEM, quote="The study found an effect.")],
        )
    ).audit(
        [
            _claim(
                [_semantic_ref(quote="The study found an effect."), invalid_second],
                text="Item A performed better than Item B.",
                tags=["comparison"],
                claim_id="c3",
            )
        ]
    )
    assert rejected["results"][0]["verified"] is False
    assert "ITEM_MISMATCH" in rejected["results"][0]["reason_codes"]
    assert "COMPARATOR_EVIDENCE_MISSING" in rejected["results"][0]["reason_codes"]


def test_checker_contradicted_is_normalized_to_public_unsupported():
    checker = FakeChecker("contradicted")
    result = AuditService(
        _deps(
            checker=checker,
            retriever=lambda q, k: [_positive_hit(quote="The study found an effect.")],
        )
    ).audit(
        [_claim([_semantic_ref(quote="The study found an effect.")], text="The study found an effect.")]
    )
    row = result["results"][0]
    assert row["status"] == "unsupported"
    assert row["verdict"] == "unsupported"
    assert row["checker_status"] == "unsupported"


def test_empty_metadata_is_not_treated_as_a_resolved_item():
    result = AuditService(
        AuditDependencies(
            page_reader=lambda *args: {"text": "The policy worked.", "needs_ocr": False},
            metadata_resolver=lambda key: {},
        )
    ).audit([_claim([_pdf_ref(quote="The policy worked.")], text="The policy worked.")])
    assert "ITEM_NOT_FOUND" in result["results"][0]["reason_codes"]


def test_checker_revise_status_is_returned_without_verified_true():
    checker = FakeChecker("revise")
    result = AuditService(
        _deps(checker=checker, retriever=lambda q, k: [_positive_hit()])
    ).audit([_claim([_semantic_ref()], text="The policy worked.")])
    row = result["results"][0]
    assert row["verified"] is False
    assert row["status"] == "revise"
    assert row["missing_qualification"] == "add the limitation"


def test_bounded_escalation_uses_same_item_and_reads_a_page_for_numeric_claim():
    checker = FakeChecker()
    calls = []
    page_calls = []

    def retrieve(query, key):
        calls.append((query, key))
        return [_positive_hit(key, page=7)]

    def read_page(*args):
        page_calls.append(args)
        return {"text": "The treatment reduced emissions by 5%.", "needs_ocr": False}

    result = AuditService(
        _deps(
            checker=checker,
            retriever=retrieve,
            page_reader=read_page,
        )
    ).audit([_claim([_semantic_ref()])], escalation="bounded")
    row = result["results"][0]
    assert row["verified"] is True
    assert row["escalation"]["performed"] is True
    assert calls == [("treatment emissions", ITEM), ("The treatment reduced emissions by 5%.", ITEM)]
    assert all(key == ITEM for _query, key in calls)
    assert page_calls[0][1:3] == (7, 7)


def test_bounded_escalation_attempts_targeted_sidecar_without_a_hit():
    sidecar_calls = []

    def sidecar_search(key, query):
        sidecar_calls.append((key, query))
        return {
            "text": "The policy worked.",
            "quote": "The policy worked.",
            "locator": "chars 0-18",
        }

    result = AuditService(
        _deps(
            retriever=lambda q, k: [],
            sidecar_search=sidecar_search,
        )
    ).audit(
        [_claim([_semantic_ref(quote="The policy worked.")], text="The policy worked.")],
        escalation="bounded",
    )
    assert result["results"][0]["verified"] is False  # no checker configured
    assert sidecar_calls == [(ITEM, "The policy worked.")]
    assert result["results"][0]["evidence"][0]["route"] == "mineru_sidecar"


def test_bounded_escalation_is_capped_at_three_claims():
    calls = []

    def retrieve(query, key):
        calls.append((query, key))
        return []

    claims = [
        _claim([_semantic_ref()], claim_id=f"c{i}")
        for i in range(4)
    ]
    result = AuditService(_deps(retriever=retrieve)).audit(claims, escalation="bounded")
    assert [row["escalation"]["performed"] for row in result["results"]] == [True, True, True, False]
    assert len(calls) == 7  # four initial calls plus one bounded retry for three claims


def test_checker_requires_loopback_configuration():
    assert LocalOpenAIClaimChecker.from_environment() is None
    assert not LocalOpenAIClaimChecker.is_loopback_url("https://example.com/v1")
    assert LocalOpenAIClaimChecker.is_loopback_url("http://127.0.0.1:1234/v1")


def test_local_checker_rejects_redirects_without_parsing_the_body():
    calls = {}

    class Response:
        status_code = 302

    def post(*args, **kwargs):
        calls.update(kwargs)
        return Response()

    checker = LocalOpenAIClaimChecker(
        "http://127.0.0.1:1234/v1",
        "checker-model",
        post=post,
    )
    claim = parse_claims([_claim([_semantic_ref()], text="The policy worked.")])[0]
    record = EvidenceRecord(
        evidence_id="e1",
        item_key=ITEM,
        route="zotero_semantic_search",
        locator="ITEM0001#0",
        quote="The policy worked.",
        excerpt="The policy worked.",
        source_text="The policy worked.",
        raw_rerank=1.0,
    )
    checked = checker.check(claim, [record])
    assert checked.status == "insufficient"
    assert calls["allow_redirects"] is False


def test_local_checker_disables_environment_proxy(monkeypatch):
    import requests

    sessions = []

    class Response:
        status_code = 302

    class Session:
        def __init__(self):
            self.trust_env = True
            sessions.append(self)

        def post(self, *args, **kwargs):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(requests, "Session", Session)
    checker = LocalOpenAIClaimChecker("http://127.0.0.1:1234/v1", "checker-model")
    claim = parse_claims([_claim([_semantic_ref()], text="The policy worked.")])[0]
    record = EvidenceRecord(
        evidence_id="e1",
        item_key=ITEM,
        route="zotero_semantic_search",
        locator="ITEM0001#0",
        quote="The policy worked.",
        excerpt="The policy worked.",
        source_text="The policy worked.",
        raw_rerank=1.0,
    )
    assert checker.check(claim, [record]).status == "insufficient"
    assert sessions and sessions[0].trust_env is False


def test_local_checker_parses_bounded_openai_json():
    class Response:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict": "supported",
                                    "rationale": "The passage entails the claim.",
                                }
                            )
                        }
                    }
                ]
            }

    checker = LocalOpenAIClaimChecker(
        "http://127.0.0.1:1234/v1",
        "checker-model",
        post=lambda *args, **kwargs: Response(),
    )
    claim = parse_claims([_claim([_semantic_ref()], text="The policy worked.")])[0]
    service = AuditService(
        _deps(retriever=lambda q, k: [_positive_hit()])
    )
    audit_result = service._materialize_claim(claim)[0]
    checked = checker.check(claim, audit_result)
    assert checked.status == "supported"


def test_tool_returns_bounded_invalid_input_json_without_calling_sources():
    raw = claim_audit_tool.audit_claims(
        claims=[_claim([{**_semantic_ref(), "path": "/tmp/secret.pdf"}])],
        ctx=DummyContext(),
    )
    payload = json.loads(raw)
    assert payload["error"]["code"] == "INVALID_INPUT"
    assert "/tmp/secret.pdf" not in payload["error"]["message"]


def test_pdf_adapter_uses_shared_pdf_resolver_and_bounds_pages(monkeypatch):
    from zotero_mcp import extract
    from zotero_mcp.tools import read_pdf

    doc = SimpleNamespace(
        page_numbers=(3, 4),
        pages=("First page.", "Second page."),
        needs_ocr=(),
    )
    monkeypatch.setattr(
        read_pdf,
        "_get_pdf_path",
        lambda item_key, ctx: ("/tmp/shared-resolver.pdf", "Paper", False),
    )
    monkeypatch.setattr(extract, "pdf_page_count", lambda path: 10)
    monkeypatch.setattr(extract, "extract_pdf", lambda path, pages: doc)

    payload = claim_audit_tool._read_pdf_window(
        ITEM, 4, 5, None, DummyContext()
    )
    assert payload["text"] == "First page.\n\nSecond page."
    assert payload["locator"] == "pages 4-5"


def test_attachment_mismatch_is_a_deterministic_failure():
    result = AuditService(
        _deps(
            page_reader=lambda *args: {"error_code": "ATTACHMENT_MISMATCH"},
        )
    ).audit([_claim([_pdf_ref(attachment_key="ATTACH01")])])
    row = result["results"][0]
    assert row["verified"] is False
    assert "ATTACHMENT_MISMATCH" in row["reason_codes"]


def test_raw_evidence_hits_preserve_chunks_scores_and_exact_scope(monkeypatch):
    from zotero_mcp import semantic_search

    class Chroma:
        def __init__(self):
            self.where = None

        def search(self, **kwargs):
            self.where = kwargs["where"]
            return {
                "ids": [[f"{ITEM}#0", f"{OTHER_ITEM}#0"]],
                "documents": [["The policy worked.", "Foreign text."]],
                "metadatas": [[{"page": 3}, {"page": 8}]],
                "distances": [[0.1, 0.2]],
            }

    class Reranker:
        def rerank_with_scores(self, query, documents, top_k):
            return [(0, 2.0), (1, -1.0)]

    chroma = Chroma()
    search = semantic_search.ZoteroSemanticSearch.__new__(semantic_search.ZoteroSemanticSearch)
    search.chroma_client = chroma
    search._chunking_config = {"enabled": False}
    search._reranker_config = {"candidate_multiplier": 3}
    monkeypatch.setattr(search, "_get_reranker", lambda: Reranker())
    monkeypatch.setattr(search, "_get_sparse_index", lambda: None)

    hits = search.search_evidence_hits("policy", ITEM, limit=2, group_id=0)

    assert chroma.where == {"$and": [{"item_key": ITEM}, {"group_id": 0}]}
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == f"{ITEM}#0"
    assert hits[0]["page"] == 3
    assert hits[0]["rerank_score"] == pytest.approx(2.0)


def test_content_hash_fixture_is_sha256_shaped():
    # Keep this small assertion close to the tests that use hash gates; it
    # documents that the accepted hash shape is the source-text digest.
    assert len(hashlib.sha256(b"source").hexdigest()) == 64
