"""Bounded, claim-level evidence auditing for Zotero RAG.

This module is deliberately independent of MCP registration and document
retrieval.  The tool adapter supplies narrowly scoped readers/retrievers;
this module validates their results and applies deterministic evidence gates. It
never synthesizes an answer
or adjudicates exhaustive extraction packets.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_VERSION = 1

MAX_CLAIMS = 8
MAX_EVIDENCE_REFS = 4
MAX_CLAIM_CHARS = 1_000
MAX_CLAIM_ID_CHARS = 80
MAX_QUERY_CHARS = 500
MAX_QUOTE_CHARS = 1_600
MAX_LOCATOR_CHARS = 500
MAX_EVIDENCE_WINDOW_CHARS = 1_600
MAX_OUTPUT_EXCERPT_CHARS = 320
MAX_REASON_CODES = 6
MAX_ESCALATED_CLAIMS = 3
MAX_PDF_PAGE_SPAN = 3

_ITEM_KEY_PATTERN = r"^[A-Za-z0-9]{8}$"
_SHA256_PATTERN = r"^[0-9a-fA-F]{64}$"

RiskTag = Literal[
    "numeric",
    "quotation",
    "causal",
    "comparison",
    "within_item_comparison",
    "attribution",
    "other",
]

# Stable machine-readable gate codes. The alias is intentionally not used to
# validate arbitrary internal failures: adding a code is a deliberate API
# change, while the public result remains forward-compatible as a JSON object.
ReasonCode = Literal[
    "ITEM_MISMATCH",
    "ITEM_NOT_FOUND",
    "ATTACHMENT_MISMATCH",
    "STALE_EVIDENCE",
    "QUOTE_NOT_FOUND",
    "BIBLIOGRAPHY_ONLY",
    "RERANK_MISSING",
    "NONPOSITIVE_RERANK",
    "RERANK_INVALID",
    "NEEDS_DIRECT_EVIDENCE",
    "SIDECAR_FALLBACK_REQUIRES_PAGE_FAILURE",
    "NUMBER_MISMATCH",
    "UNIT_MISMATCH",
    "COMPARATOR_EVIDENCE_MISSING",
    "ROUTE_MISLABELED",
]


class _StrictModel(BaseModel):
    """Pydantic base that makes the public evidence contract fail closed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SemanticEvidenceRef(_StrictModel):
    """A candidate quote from a fresh exact-item semantic retrieval."""

    route: Literal["semantic"] = "semantic"
    item_key: str = Field(pattern=_ITEM_KEY_PATTERN)
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    quote: str = Field(min_length=1, max_length=MAX_QUOTE_CHARS)
    chunk_id: str | None = Field(default=None, max_length=200)
    content_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    index_generation: str | None = Field(default=None, max_length=200)


class PdfPageEvidenceRef(_StrictModel):
    """A candidate quote from one to three 1-indexed PDF pages."""

    route: Literal["pdf_page"] = "pdf_page"
    item_key: str = Field(pattern=_ITEM_KEY_PATTERN)
    page: int = Field(ge=1)
    end_page: int | None = Field(default=None, ge=1)
    attachment_key: str | None = Field(default=None, pattern=_ITEM_KEY_PATTERN)
    quote: str = Field(min_length=1, max_length=MAX_QUOTE_CHARS)
    content_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _valid_page_span(self) -> PdfPageEvidenceRef:
        end = self.end_page or self.page
        if end < self.page:
            raise ValueError("end_page must be greater than or equal to page")
        if end - self.page + 1 > MAX_PDF_PAGE_SPAN:
            raise ValueError(f"PDF evidence may span at most {MAX_PDF_PAGE_SPAN} pages")
        return self


class MineruSidecarEvidenceRef(_StrictModel):
    """A bounded window in a local MinerU sidecar."""

    route: Literal["mineru_sidecar"] = "mineru_sidecar"
    item_key: str = Field(pattern=_ITEM_KEY_PATTERN)
    locator: str | None = Field(default=None, max_length=MAX_LOCATOR_CHARS)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    quote: str = Field(min_length=1, max_length=MAX_QUOTE_CHARS)
    content_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    index_generation: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _bounded_locator(self) -> MineruSidecarEvidenceRef:
        if self.locator is None and self.start_line is None:
            raise ValueError("sidecar evidence requires locator or start_line")
        if self.end_line is not None and self.start_line is None:
            raise ValueError("end_line requires start_line")
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line - self.start_line + 1 > 400
        ):
            raise ValueError("sidecar evidence may span at most 400 lines")
        return self


EvidenceRef = Annotated[
    SemanticEvidenceRef | PdfPageEvidenceRef | MineruSidecarEvidenceRef,
    Field(discriminator="route"),
]


class ClaimInput(_StrictModel):
    """One atomic claim and up to four caller-provided evidence candidates."""

    claim_id: str = Field(min_length=1, max_length=MAX_CLAIM_ID_CHARS)
    text: str = Field(min_length=1, max_length=MAX_CLAIM_CHARS)
    risk_tags: list[RiskTag] = Field(default_factory=list, max_length=6)
    evidence: list[EvidenceRef] = Field(
        min_length=1,
        max_length=MAX_EVIDENCE_REFS,
    )

    @model_validator(mode="after")
    def _valid_comparison_scope(self) -> ClaimInput:
        tags = set(self.risk_tags)
        if {"comparison", "within_item_comparison"} <= tags:
            raise ValueError(
                "comparison and within_item_comparison are mutually exclusive"
            )
        if "within_item_comparison" in tags:
            item_keys = {ref.item_key.upper() for ref in self.evidence}
            if len(item_keys) != 1:
                raise ValueError(
                    "within_item_comparison evidence must use exactly one item_key"
                )
        return self


def parse_claims(value: Sequence[ClaimInput | Mapping[str, Any]] | str) -> list[ClaimInput]:
    """Parse the public claim input, including clients that JSON-stringify arrays."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"claims must be a JSON array: {exc.msg}") from exc
    if not isinstance(value, list):
        raise ValueError("claims must be a list or a JSON-stringified list")
    if not 1 <= len(value) <= MAX_CLAIMS:
        raise ValueError(f"claims must contain between 1 and {MAX_CLAIMS} entries")
    try:
        return [ClaimInput.model_validate(claim) for claim in value]
    except ValidationError:
        raise


@dataclass(frozen=True)
class EvidenceRecord:
    """Validated evidence retained internally and rendered compactly outside."""

    evidence_id: str
    item_key: str
    route: Literal["zotero_semantic_search", "zotero_read_pdf_pages", "mineru_sidecar"]
    locator: str
    quote: str
    excerpt: str
    source_text: str = field(repr=False, default="")
    raw_rerank: float | None = None
    content_hash: str | None = None
    index_generation: str | None = None
    source_classification: str | None = None
    weaker_evidence: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def public(self) -> dict[str, Any]:
        """Return a bounded provenance record; never expose the full source window."""

        result: dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "item_key": self.item_key,
            "route": self.route,
            "locator": self.locator[:MAX_LOCATOR_CHARS],
            "quote": self.quote[:MAX_OUTPUT_EXCERPT_CHARS],
            "excerpt": _excerpt_around(
                self.excerpt,
                self.quote,
                limit=MAX_OUTPUT_EXCERPT_CHARS,
            ),
            "content_hash": self.content_hash,
            "source_classification": self.source_classification,
            "weaker_evidence": self.weaker_evidence,
        }
        if self.raw_rerank is not None:
            result["raw_rerank"] = self.raw_rerank
        if self.index_generation is not None:
            result["index_generation"] = self.index_generation
        return result


@dataclass(frozen=True)
class GateFailure:
    code: str
    message: str
    blocking: bool = True


@dataclass
class AuditDependencies:
    """Narrow readers supplied by the MCP adapter or by hermetic tests."""

    retriever: Callable[[str, str], Sequence[Mapping[str, Any]]] | None = None
    page_reader: Callable[
        [str, int, int | None, str | None], Mapping[str, Any] | str
    ] | None = None
    sidecar_reader: Callable[
        [str, MineruSidecarEvidenceRef], Mapping[str, Any] | str
    ] | None = None
    sidecar_search: Callable[[str, str], Mapping[str, Any] | None] | None = None
    metadata_resolver: Callable[[str], Mapping[str, Any] | None] | None = None


# Quote matching is intentionally conservative. It normalizes Unicode and
# whitespace and permits two interpretations of a hyphen at a PDF line break
# (including extractors that flatten it to spaces): lexical (``cost-effective``)
# or discretionary (``property`` emitted as ``prop-\nerty`` or ``prop- erty``).
# It does not do fuzzy or semantic matching.
def _normalized_text_with_offsets(
    value: str,
    *,
    drop_linebreak_hyphens: bool = False,
) -> tuple[str, list[int]]:
    """Normalize text while retaining offsets into the original string."""

    chars: list[tuple[str, int]] = []
    for original_index, character in enumerate(value or ""):
        for normalized in unicodedata.normalize("NFKC", character):
            if normalized != "\u00ad":
                chars.append((normalized, original_index))

    collapsed: list[tuple[str, int]] = []
    index = 0
    hyphens = "-‐‑‒–—"
    while index < len(chars):
        character, origin = chars[index]
        if character in hyphens and index > 0:
            next_index = index + 1
            saw_separator_whitespace = False
            while next_index < len(chars) and chars[next_index][0].isspace():
                saw_separator_whitespace = True
                next_index += 1
            previous_character = chars[index - 1][0]
            next_character = chars[next_index][0] if next_index < len(chars) else ""
            if (
                saw_separator_whitespace
                and previous_character.isalpha()
                and next_character.isalpha()
            ):
                if not drop_linebreak_hyphens:
                    collapsed.append(("-", origin))
                index = next_index
                continue
        collapsed.append((character, origin))
        index += 1

    whitespace_normalized: list[tuple[str, int]] = []
    for character, origin in collapsed:
        if character.isspace():
            if not whitespace_normalized or whitespace_normalized[-1][0] != " ":
                whitespace_normalized.append((" ", origin))
        else:
            whitespace_normalized.append((character, origin))

    folded: list[tuple[str, int]] = []
    for character, origin in whitespace_normalized:
        folded.extend(
            (folded_character, origin) for folded_character in character.casefold()
        )

    start = 0
    end = len(folded)
    while start < end and folded[start][0].isspace():
        start += 1
    while end > start and folded[end - 1][0].isspace():
        end -= 1
    trimmed = folded[start:end]
    return "".join(character for character, _origin in trimmed), [
        origin for _character, origin in trimmed
    ]


def _normalized_text_variants(value: str) -> tuple[str, ...]:
    variants: list[str] = []
    for drop_linebreak_hyphens in (False, True):
        normalized, _offsets = _normalized_text_with_offsets(
            value,
            drop_linebreak_hyphens=drop_linebreak_hyphens,
        )
        if normalized and normalized not in variants:
            variants.append(normalized)
    return tuple(variants)


def normalize_quote_text(value: str) -> str:
    """Return the primary normalized form retained for API compatibility."""

    variants = _normalized_text_variants(value)
    return variants[0] if variants else ""


def quote_contained(quote: str, source_text: str) -> bool:
    """Return true only when a bounded normalized quote occurs in source text."""

    quote_variants = _normalized_text_variants(quote)
    source_variants = _normalized_text_variants(source_text)
    return any(
        normalized_quote in normalized_source
        for normalized_quote in quote_variants
        for normalized_source in source_variants
    )


def _excerpt_around(
    source_text: str,
    quote: str,
    limit: int = MAX_EVIDENCE_WINDOW_CHARS,
) -> str:
    """Return a bounded original-text window that contains the quote."""

    if not source_text:
        return ""

    for normalized_quote in _normalized_text_variants(quote):
        for drop_linebreak_hyphens in (False, True):
            normalized_source, offsets = _normalized_text_with_offsets(
                source_text,
                drop_linebreak_hyphens=drop_linebreak_hyphens,
            )
            position = normalized_source.find(normalized_quote)
            if position < 0 or not offsets:
                continue

            quote_end = min(position + len(normalized_quote) - 1, len(offsets) - 1)
            quote_start_original = offsets[position]
            quote_end_original = offsets[quote_end] + 1
            quote_length = max(0, quote_end_original - quote_start_original)
            if quote_length >= limit:
                return source_text[quote_start_original : quote_start_original + limit]

            half = max(80, (limit - quote_length) // 2)
            start = max(0, quote_start_original - half)
            end = min(len(source_text), start + limit)
            start = max(0, end - limit)
            return source_text[start:end]

    return source_text[:limit]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _unique_codes(failures: Sequence[GateFailure]) -> list[str]:
    result: list[str] = []
    for failure in failures:
        if failure.code not in result:
            result.append(failure.code)
        if len(result) >= MAX_REASON_CODES:
            break
    return result


def _mapping_value(value: Any, *keys: str, default: Any = None) -> Any:
    if not isinstance(value, Mapping):
        return default
    for key in keys:
        if key in value:
            return value[key]
    return default


def _hit_item_key(hit: Mapping[str, Any]) -> str:
    raw_id = str(_mapping_value(hit, "chunk_id", "id", default=""))
    item_key = _mapping_value(hit, "item_key", default="")
    if not item_key and raw_id:
        item_key = raw_id.split("#", 1)[0]
    return str(item_key or "").strip().upper()


def _hit_chunk_id(hit: Mapping[str, Any]) -> str:
    raw_id = _mapping_value(hit, "chunk_id", "id", default="")
    return str(raw_id or "")


def _hit_metadata(hit: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _mapping_value(hit, "metadata", "meta", default={})
    return metadata if isinstance(metadata, Mapping) else {}


def _hit_document(hit: Mapping[str, Any]) -> str:
    value = _mapping_value(hit, "document", "matched_text", "text", default="")
    return str(value or "")


def _strip_context_prefix(text: str) -> str:
    """Remove the optional DCR line before using a passage as a page quote."""

    return re.sub(r"^\[Paper:[^\n]*\]\n", "", text or "", count=1)


def _hit_quote(hit: Mapping[str, Any]) -> str:
    candidate = str(_mapping_value(hit, "matched_passage", "quote", default="") or "")
    candidate = _strip_context_prefix(candidate).strip()
    if candidate:
        return candidate[:MAX_QUOTE_CHARS]
    document = _strip_context_prefix(_hit_document(hit)).strip()
    return document[:MAX_QUOTE_CHARS]


def _is_reference_hit(hit: Mapping[str, Any]) -> bool:
    metadata = _hit_metadata(hit)
    return bool(
        _mapping_value(hit, "is_reference", default=False)
        or _mapping_value(metadata, "is_reference", "is_bibliography", default=False)
        or _mapping_value(metadata, "reference_kind", default="")
        in {"bibliography_chunk", "bibliography", "reference"}
    )


_NUMBER_TOKEN = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNIT_TOKEN = (
    r"(?:percentage\s+points?|basis\s+points?|per\s+cent|"
    r"percent(?:age)?|%|pp|bps?|million|billion|thousand)"
)
_RANGE_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_.,])"
    rf"(?P<left>[+-]?{_NUMBER_TOKEN})(?![\d,])"
    rf"(?:\s*(?P<left_unit>{_UNIT_TOKEN}))?\s*"
    rf"(?:to|[-‐‑‒–—])\s*"
    rf"(?P<right>[+-]?{_NUMBER_TOKEN})(?![\d,])"
    rf"(?:\s*(?P<right_unit>{_UNIT_TOKEN}))?"
    rf"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SCALAR_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_.,])"
    rf"(?P<number>[+-]?{_NUMBER_TOKEN})(?![\d,])"
    rf"(?:\s*(?P<unit>{_UNIT_TOKEN}))?"
    rf"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _canonical_numeric_unit(value: str | None) -> str:
    unit = re.sub(r"\s+", " ", (value or "").casefold()).strip()
    if unit in {"%", "percent", "percentage", "per cent"}:
        return "%"
    if unit in {"pp", "percentage point", "percentage points"}:
        return "pp"
    if unit in {"bp", "bps", "basis point", "basis points"}:
        return "bps"
    return unit


def _decimal_number(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", "")).normalize()
    except InvalidOperation:
        return None


def _numeric_signatures(text: str) -> set[tuple[Decimal, str]]:
    """Extract conservative scalar and range signatures with canonical units."""

    normalized = unicodedata.normalize("NFKC", text or "").replace("−", "-")
    signatures: set[tuple[Decimal, str]] = set()
    remaining = list(normalized)

    # Parse ranges first so a separator is never mistaken for the sign of the
    # right endpoint. A unit written on only one endpoint applies to both.
    for match in _RANGE_PATTERN.finditer(normalized):
        left = _decimal_number(match.group("left"))
        right = _decimal_number(match.group("right"))
        left_unit = _canonical_numeric_unit(match.group("left_unit"))
        right_unit = _canonical_numeric_unit(match.group("right_unit"))
        left_unit = left_unit or right_unit
        right_unit = right_unit or left_unit
        if left is not None:
            signatures.add((left, left_unit))
        if right is not None:
            signatures.add((right, right_unit))
        remaining[match.start() : match.end()] = " " * (match.end() - match.start())

    # Remove consumed range spans before parsing ordinary signed scalars.
    for match in _SCALAR_PATTERN.finditer("".join(remaining)):
        number = _decimal_number(match.group("number"))
        if number is not None:
            signatures.add(
                (number, _canonical_numeric_unit(match.group("unit")))
            )
    return signatures


def _numeric_claim(claim: ClaimInput) -> bool:
    return "numeric" in claim.risk_tags or bool(_numeric_signatures(claim.text))


def _source_classification(metadata: Mapping[str, Any]) -> str | None:
    for key in ("source_group", "item_type", "itemType", "classification"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _coerce_reader_payload(value: Mapping[str, Any] | str) -> tuple[str, Mapping[str, Any]]:
    if isinstance(value, str):
        return value, {}
    if not isinstance(value, Mapping):
        return "", {}
    text = _mapping_value(value, "text", "content", "source_text", default="")
    return str(text or ""), value


class AuditService:
    """Run bounded deterministic evidence gates for a claim batch."""

    def __init__(self, dependencies: AuditDependencies | None = None):
        self.dependencies = dependencies or AuditDependencies()

    def audit(
        self,
        claims: Sequence[ClaimInput | Mapping[str, Any]] | str,
        *,
        escalation: Literal["none", "bounded"] = "none",
    ) -> dict[str, Any]:
        parsed_claims = parse_claims(claims)
        results: list[dict[str, Any]] = []
        escalation_budget = MAX_ESCALATED_CLAIMS
        for claim in parsed_claims:
            records, failures, page_failure = self._materialize_claim(claim)
            needs_escalation = self._needs_escalation(claim, records, failures)
            escalation_performed = False
            if escalation == "bounded" and needs_escalation and escalation_budget > 0:
                escalation_budget -= 1
                escalation_performed = True
                had_records = bool(records)
                extra_records, extra_failures, extra_page_failure = self._escalate(claim)
                if extra_records and not had_records:
                    recovered_codes = {
                        "NO_RETRIEVAL_HIT",
                        "RERANK_MISSING",
                        "NONPOSITIVE_RERANK",
                        "RERANK_INVALID",
                        "QUOTE_NOT_FOUND",
                        "PDF_READER_UNAVAILABLE",
                        "PDF_TEXT_UNAVAILABLE",
                        "PDF_READ_FAILED",
                        "SCANNED_PAGE",
                    }
                    failures = [failure for failure in failures if failure.code not in recovered_codes]
                records.extend(extra_records)
                failures.extend(extra_failures)
                page_failure = page_failure or extra_page_failure
            results.append(
                self._evaluate(
                    claim,
                    records,
                    failures,
                    page_failure=page_failure,
                    escalation_performed=escalation_performed,
                )
            )

        counts = {status: 0 for status in ("supported", "unsupported", "insufficient")}
        for result in results:
            counts[result["status"]] += 1
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": "deterministic",
            "summary": {
                "total": len(results),
                **counts,
            },
            "results": results,
        }

    def _resolve_metadata(self, item_key: str) -> tuple[Mapping[str, Any], list[GateFailure]]:
        resolver = self.dependencies.metadata_resolver
        if resolver is None:
            return {}, []
        try:
            metadata = resolver(item_key)
        except Exception:
            return {}, [GateFailure("ITEM_NOT_FOUND", "item metadata could not be resolved")]
        if metadata is None or not isinstance(metadata, Mapping) or not metadata:
            return {}, [GateFailure("ITEM_NOT_FOUND", "item metadata could not be resolved")]
        returned_key = str(
            _mapping_value(metadata, "key", default="")
            or _mapping_value(metadata.get("data", {}), "key", default="")
        ).strip().upper()
        if returned_key and returned_key != item_key:
            return {}, [GateFailure("ITEM_MISMATCH", "metadata returned a different item key")]
        return metadata, []

    def _materialize_claim(
        self, claim: ClaimInput
    ) -> tuple[list[EvidenceRecord], list[GateFailure], bool]:
        records: list[EvidenceRecord] = []
        failures: list[GateFailure] = []
        page_failure = False
        # Metadata is resolved once per claim and is never caller-supplied.
        metadata_cache: dict[str, tuple[Mapping[str, Any], list[GateFailure]]] = {}
        for index, ref in enumerate(claim.evidence):
            item_key = ref.item_key.upper()
            if item_key not in metadata_cache:
                metadata_cache[item_key] = self._resolve_metadata(item_key)
            metadata, metadata_failures = metadata_cache[item_key]
            if metadata_failures:
                failures.extend(metadata_failures)
                continue
            record, ref_failures, ref_page_failure = self._materialize_ref(
                ref,
                evidence_id=f"{claim.claim_id}:{index + 1}",
                metadata=metadata,
            )
            failures.extend(ref_failures)
            page_failure = page_failure or ref_page_failure
            if record is not None:
                records.append(record)
        if page_failure and records:
            records = [
                replace(record, weaker_evidence=True)
                if record.route == "mineru_sidecar"
                else record
                for record in records
            ]
        return records, failures, page_failure

    def _materialize_ref(
        self,
        ref: EvidenceRef,
        *,
        evidence_id: str,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceRecord | None, list[GateFailure], bool]:
        if isinstance(ref, SemanticEvidenceRef):
            return self._materialize_semantic(ref, evidence_id=evidence_id, metadata=metadata)
        if isinstance(ref, PdfPageEvidenceRef):
            return self._materialize_pdf(ref, evidence_id=evidence_id, metadata=metadata)
        return self._materialize_sidecar(ref, evidence_id=evidence_id, metadata=metadata)

    def _materialize_semantic(
        self,
        ref: SemanticEvidenceRef,
        *,
        evidence_id: str,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceRecord | None, list[GateFailure], bool]:
        retriever = self.dependencies.retriever
        if retriever is None:
            return None, [GateFailure("RETRIEVER_UNAVAILABLE", "semantic retriever is not configured")], False
        item_key = ref.item_key.upper()
        try:
            hits = list(retriever(ref.query, item_key))
        except Exception:
            return None, [GateFailure("RETRIEVER_UNAVAILABLE", "semantic retrieval failed")], False
        if not hits:
            return None, [GateFailure("NO_RETRIEVAL_HIT", "exact-item retrieval returned no hit")], False

        failures: list[GateFailure] = []
        matched_chunk = False
        for hit in hits:
            hit_key = _hit_item_key(hit)
            if hit_key != item_key:
                failures.append(GateFailure("ITEM_MISMATCH", "retrieval returned a foreign item key"))
                continue
            chunk_id = _hit_chunk_id(hit)
            if ref.chunk_id is not None and chunk_id != ref.chunk_id:
                continue
            if ref.chunk_id is not None:
                matched_chunk = True
            if _is_reference_hit(hit):
                failures.append(
                    GateFailure(
                        "BIBLIOGRAPHY_ONLY",
                        "bibliography/reference chunks are not substantive evidence",
                        blocking=False,
                    )
                )
                continue
            document = _hit_document(hit)
            if not document:
                failures.append(GateFailure("EMPTY_EVIDENCE", "retrieval hit had no document text"))
                continue
            current_hash = _sha256_text(document)
            if ref.content_hash and ref.content_hash.casefold() != current_hash.casefold():
                failures.append(GateFailure("STALE_EVIDENCE", "retrieval content hash does not match"))
                continue
            hit_generation = str(
                _mapping_value(hit, "index_generation", default="")
                or _mapping_value(_hit_metadata(hit), "index_generation", "generation", default="")
            )
            if ref.index_generation and hit_generation != ref.index_generation:
                failures.append(GateFailure("STALE_EVIDENCE", "retrieval index generation does not match"))
                continue
            raw_score = _mapping_value(hit, "rerank_score", "raw_rerank", default=None)
            if raw_score is None:
                failures.append(
                    GateFailure("RERANK_MISSING", "retrieval evidence has no raw reranker score", blocking=False)
                )
                continue
            try:
                raw_score = float(raw_score)
            except (TypeError, ValueError):
                failures.append(GateFailure("RERANK_INVALID", "retrieval reranker score is not numeric", blocking=False))
                continue
            if raw_score <= 0.0:
                failures.append(
                    GateFailure(
                        "NONPOSITIVE_RERANK",
                        "retrieval evidence requires fresh raw Rerank > 0",
                        blocking=False,
                    )
                )
                continue
            if not quote_contained(ref.quote, document):
                failures.append(GateFailure("QUOTE_NOT_FOUND", "candidate quote is not contained in the retrieved text"))
                continue
            locator_parts = [chunk_id or ref.chunk_id or "item"]
            hit_page = _mapping_value(hit, "page", default=None) or _mapping_value(
                _hit_metadata(hit), "page", default=None
            )
            if hit_page is not None:
                locator_parts.append(f"page {hit_page}")
            char_start = _mapping_value(hit, "char_start", default=None) or _mapping_value(
                _hit_metadata(hit), "char_start", default=None
            )
            if char_start is not None:
                locator_parts.append(f"char {char_start}")
            return (
                EvidenceRecord(
                    evidence_id=evidence_id,
                    item_key=item_key,
                    route="zotero_semantic_search",
                    locator=", ".join(locator_parts),
                    quote=ref.quote,
                    excerpt=_excerpt_around(document, ref.quote),
                    source_text=document,
                    raw_rerank=raw_score,
                    content_hash=current_hash,
                    index_generation=hit_generation or None,
                    source_classification=_source_classification(
                        {**dict(_hit_metadata(hit)), **dict(metadata)}
                    ),
                    metadata=dict(metadata),
                ),
                [],
                False,
            )
        if ref.chunk_id is not None and not matched_chunk:
            failures.append(GateFailure("CHUNK_NOT_FOUND", "requested chunk was not returned"))
        return None, failures, False

    def _materialize_pdf(
        self,
        ref: PdfPageEvidenceRef,
        *,
        evidence_id: str,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceRecord | None, list[GateFailure], bool]:
        reader = self.dependencies.page_reader
        if reader is None:
            return None, [GateFailure("PDF_READER_UNAVAILABLE", "PDF page reader is not configured")], True
        item_key = ref.item_key.upper()
        end_page = ref.end_page or ref.page
        try:
            payload = reader(item_key, ref.page, end_page, ref.attachment_key)
        except Exception:
            return None, [GateFailure("PDF_READ_FAILED", "PDF page read failed")], True
        text, details = _coerce_reader_payload(payload)
        if (error_code := _mapping_value(details, "error_code", default=None)):
            messages = {
                "ATTACHMENT_MISMATCH": "requested attachment is not a child of the item",
                "PDF_NOT_FOUND": "no PDF attachment was found",
                "PDF_PAGE_OUT_OF_RANGE": "requested PDF page is out of range",
                "PDF_READ_FAILED": "PDF page read failed",
            }
            return None, [GateFailure(str(error_code), messages.get(str(error_code), "PDF evidence route failed"))], True
        if bool(_mapping_value(details, "needs_ocr", "scanned", default=False)):
            return None, [GateFailure("SCANNED_PAGE", "PDF page has no reliable text layer")], True
        if not text.strip():
            return None, [GateFailure("PDF_TEXT_UNAVAILABLE", "PDF page returned no extractable text")], True
        current_hash = _sha256_text(text)
        if ref.content_hash and ref.content_hash.casefold() != current_hash.casefold():
            return None, [GateFailure("STALE_EVIDENCE", "PDF page content hash does not match")], True
        if not quote_contained(ref.quote, text):
            return None, [GateFailure("QUOTE_NOT_FOUND", "candidate quote is not contained in the PDF page text")], True
        locator = f"pages {ref.page}-{end_page}"
        if ref.attachment_key:
            locator += f", attachment {ref.attachment_key}"
        return (
            EvidenceRecord(
                evidence_id=evidence_id,
                item_key=item_key,
                route="zotero_read_pdf_pages",
                locator=locator,
                quote=ref.quote,
                excerpt=_excerpt_around(text, ref.quote),
                source_text=text,
                content_hash=current_hash,
                source_classification=_source_classification(metadata),
                metadata=dict(metadata),
            ),
            [],
            False,
        )

    def _materialize_sidecar(
        self,
        ref: MineruSidecarEvidenceRef,
        *,
        evidence_id: str,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceRecord | None, list[GateFailure], bool]:
        reader = self.dependencies.sidecar_reader
        if reader is None:
            return None, [GateFailure("SIDECAR_READER_UNAVAILABLE", "MinerU sidecar reader is not configured")], False
        item_key = ref.item_key.upper()
        try:
            payload = reader(item_key, ref)
        except Exception:
            return None, [GateFailure("SIDECAR_READ_FAILED", "MinerU sidecar read failed")], False
        text, details = _coerce_reader_payload(payload)
        if (error_code := _mapping_value(details, "error_code", default=None)):
            messages = {
                "SIDECAR_NOT_FOUND": "MinerU sidecar was not found",
                "SIDECAR_LOCATOR_NOT_FOUND": "sidecar locator was not found",
                "SIDECAR_LOCATOR_REQUIRED": "sidecar locator is required",
            }
            return None, [GateFailure(str(error_code), messages.get(str(error_code), "sidecar evidence route failed"))], False
        if not text.strip():
            return None, [GateFailure("SIDECAR_TEXT_UNAVAILABLE", "MinerU sidecar window was empty")], False
        current_hash = str(_mapping_value(details, "content_hash", default="") or _sha256_text(text))
        if ref.content_hash and ref.content_hash.casefold() != current_hash.casefold():
            return None, [GateFailure("STALE_EVIDENCE", "sidecar content hash does not match")], False
        generation = str(_mapping_value(details, "index_generation", "generation", default="") or "")
        if ref.index_generation and generation != ref.index_generation:
            return None, [GateFailure("STALE_EVIDENCE", "sidecar index generation does not match")], False
        if not quote_contained(ref.quote, text):
            return None, [GateFailure("QUOTE_NOT_FOUND", "candidate quote is not contained in the sidecar window")], False
        locator = str(_mapping_value(details, "locator", default="") or ref.locator or "sidecar window")
        if ref.start_line is not None:
            locator = f"lines {ref.start_line}-{ref.end_line or ref.start_line}"
        return (
            EvidenceRecord(
                evidence_id=evidence_id,
                item_key=item_key,
                route="mineru_sidecar",
                locator=locator,
                quote=ref.quote,
                excerpt=_excerpt_around(text, ref.quote),
                source_text=text,
                content_hash=current_hash,
                index_generation=generation or None,
                source_classification=_source_classification(metadata),
                weaker_evidence=True,
                metadata=dict(metadata),
            ),
            [],
            False,
        )

    def _needs_escalation(
        self,
        claim: ClaimInput,
        records: Sequence[EvidenceRecord],
        failures: Sequence[GateFailure],
    ) -> bool:
        if not records:
            return True
        if _numeric_claim(claim) and not any(
            record.route in {"zotero_read_pdf_pages", "mineru_sidecar"} for record in records
        ):
            return True
        return any(
            failure.code
            in {
                "QUOTE_NOT_FOUND",
                "NO_RETRIEVAL_HIT",
                "RERANK_MISSING",
                "NONPOSITIVE_RERANK",
                "PDF_READER_UNAVAILABLE",
                "PDF_TEXT_UNAVAILABLE",
                "SCANNED_PAGE",
            }
            for failure in failures
        ) and not records

    def _materialize_escalated_sidecar(
        self,
        claim: ClaimInput,
        item_key: str,
        metadata: Mapping[str, Any],
    ) -> EvidenceRecord | None:
        """Turn one internally discovered sidecar window into evidence."""

        search = self.dependencies.sidecar_search
        if search is None:
            return None
        try:
            payload = search(item_key, claim.text)
        except Exception:
            return None
        if not payload:
            return None
        text, details = _coerce_reader_payload(payload)
        quote = str(
            _mapping_value(payload, "quote", "matched_quote", default="")
            or text[:MAX_QUOTE_CHARS]
        )
        if not text or not quote_contained(quote, text):
            return None
        return EvidenceRecord(
            evidence_id=f"{claim.claim_id}:escalated-sidecar",
            item_key=item_key,
            route="mineru_sidecar",
            locator=str(
                _mapping_value(details, "locator", default="sidecar search")
                or "sidecar search"
            ),
            quote=quote,
            excerpt=_excerpt_around(text, quote),
            source_text=text,
            content_hash=str(
                _mapping_value(details, "content_hash", default="")
                or _sha256_text(text)
            ),
            source_classification=_source_classification(metadata),
            weaker_evidence=True,
            metadata=dict(metadata),
        )

    def _escalate(
        self, claim: ClaimInput
    ) -> tuple[list[EvidenceRecord], list[GateFailure], bool]:
        """Perform one exact-item retrieval and at most one narrow follow-up route."""

        records: list[EvidenceRecord] = []
        failures: list[GateFailure] = []
        page_failure = False
        item_keys = {ref.item_key.upper() for ref in claim.evidence}
        # Each claim's explicit refs must identify one or more exact items. A
        # bounded escalation never discovers a new item universe.
        for item_key in sorted(item_keys):
            sidecar_attempted = False

            def _try_sidecar(metadata: Mapping[str, Any]) -> EvidenceRecord | None:
                nonlocal sidecar_attempted
                if sidecar_attempted or self.dependencies.sidecar_search is None:
                    return None
                sidecar_attempted = True
                return self._materialize_escalated_sidecar(
                    claim, item_key, metadata
                )

            if self.dependencies.retriever is None:
                failures.append(GateFailure("RETRIEVER_UNAVAILABLE", "semantic retriever is not configured"))
                hits: list[Mapping[str, Any]] = []
            else:
                try:
                    hits = list(self.dependencies.retriever(claim.text, item_key))
                except Exception:
                    failures.append(GateFailure("RETRIEVER_UNAVAILABLE", "bounded retrieval failed"))
                    hits = []
            for hit in hits:
                if _hit_item_key(hit) != item_key:
                    failures.append(GateFailure("ITEM_MISMATCH", "bounded retrieval returned a foreign item key"))
                    continue
                if _is_reference_hit(hit):
                    continue
                raw_score = _mapping_value(hit, "rerank_score", "raw_rerank", default=None)
                try:
                    if raw_score is None or float(raw_score) <= 0.0:
                        continue
                    raw_score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                document = _hit_document(hit)
                if not document:
                    continue
                metadata = _hit_metadata(hit)
                item_metadata, metadata_failures = self._resolve_metadata(item_key)
                failures.extend(metadata_failures)
                if metadata_failures:
                    continue
                page = _mapping_value(hit, "page", default=None) or _mapping_value(metadata, "page", default=None)
                candidate_quote = _hit_quote(hit)
                if page is not None and self.dependencies.page_reader is not None:
                    try:
                        page_number = int(page)
                    except (TypeError, ValueError):
                        page_number = 0
                    if page_number > 0:
                        page_ref = PdfPageEvidenceRef(
                            item_key=item_key,
                            page=page_number,
                            end_page=page_number,
                            quote=candidate_quote,
                        )
                        record, ref_failures, ref_page_failure = self._materialize_pdf(
                            page_ref,
                            evidence_id=f"{claim.claim_id}:escalated-pdf",
                            metadata={**dict(item_metadata), **dict(metadata)},
                        )
                        failures.extend(ref_failures)
                        page_failure = page_failure or ref_page_failure
                        if record is not None:
                            records.append(record)
                            return records, [], page_failure
                record = _try_sidecar({**dict(item_metadata), **dict(metadata)})
                if record is not None:
                    records.append(record)
                    return records, [], page_failure
            # If retrieval found no usable hit, the targeted sidecar search is
            # still allowed once; it does not broaden the item universe.
            item_metadata, metadata_failures = self._resolve_metadata(item_key)
            failures.extend(metadata_failures)
            if not metadata_failures:
                record = _try_sidecar(item_metadata)
                if record is not None:
                    records.append(record)
                    return records, [], page_failure
            # One exact-item retrieval and one sidecar/page follow-up per key.
        return records, failures, page_failure

    def _evaluate(
        self,
        claim: ClaimInput,
        records: list[EvidenceRecord],
        failures: list[GateFailure],
        *,
        page_failure: bool,
        escalation_performed: bool,
    ) -> dict[str, Any]:
        failures = list(failures)
        direct_records = [
            record
            for record in records
            if record.route in {"zotero_read_pdf_pages", "mineru_sidecar"}
        ]
        if "comparison" in claim.risk_tags and len({record.item_key for record in records}) < 2:
            failures.append(
                GateFailure(
                    "COMPARATOR_EVIDENCE_MISSING",
                    "comparison claims require validated evidence for at least two distinct item keys",
                )
            )
        if _numeric_claim(claim):
            if not direct_records:
                failures.append(
                    GateFailure(
                        "NEEDS_DIRECT_EVIDENCE",
                        "numeric claims require PDF-page evidence or an explicitly labeled sidecar fallback",
                    )
                )
            elif all(record.route == "mineru_sidecar" for record in direct_records) and not page_failure:
                failures.append(
                    GateFailure(
                        "SIDECAR_FALLBACK_REQUIRES_PAGE_FAILURE",
                        "sidecar evidence is a fallback and requires a failed or unreadable page route",
                    )
                )
            else:
                claim_numbers = _numeric_signatures(claim.text)
                # A number elsewhere on the page is not evidence for the claim.
                # Only signatures inside caller-supplied quotes that passed
                # containment may satisfy the deterministic numeric gate.
                source_numbers = _numeric_signatures(
                    "\n".join(record.quote for record in direct_records)
                )
                missing = claim_numbers - source_numbers
                source_values = {number for number, _unit in source_numbers}
                if any(number not in source_values for number, _unit in missing):
                    failures.append(
                        GateFailure(
                            "NUMBER_MISMATCH",
                            "numeric values in the claim were not confirmed by direct evidence quotes",
                        )
                    )
                if any(number in source_values for number, _unit in missing):
                    failures.append(
                        GateFailure(
                            "UNIT_MISMATCH",
                            "numeric units in the claim were not confirmed by direct evidence quotes",
                        )
                    )

        # A failed PDF route may be rescued by a truthful, weaker sidecar
        # fallback; the failure remains visible only when no such record exists.
        if any(record.route == "mineru_sidecar" for record in direct_records) and page_failure:
            failures = [
                failure
                for failure in failures
                if failure.code not in {"SCANNED_PAGE", "PDF_TEXT_UNAVAILABLE", "PDF_READ_FAILED"}
            ]

        blocking_failures = [failure for failure in failures if failure.blocking]
        status: Literal["supported", "unsupported", "insufficient"] = "insufficient"
        verified = False
        eligible_records = list(records[:MAX_EVIDENCE_REFS])
        if blocking_failures:
            if any(failure.code in {"NUMBER_MISMATCH", "UNIT_MISMATCH"} for failure in blocking_failures):
                status = "unsupported"
        elif not eligible_records:
            failures.append(GateFailure("NO_ELIGIBLE_EVIDENCE", "no evidence passed deterministic gates"))
        else:
            # This status means that the evidence contract passed. It does not
            # replace the agent's review of claim wording or entailment.
            status = "supported"
            verified = True

        codes = _unique_codes(failures)
        return {
            "claim_id": claim.claim_id,
            "claim": claim.text,
            "risk_tags": list(claim.risk_tags),
            "verified": verified,
            "evidence_verified": verified,
            "status": status,
            "verdict": status,
            "reason_codes": codes,
            "gate_failure_codes": codes,
            "escalation": {"performed": escalation_performed},
            "evidence": [record.public() for record in eligible_records],
        }


__all__ = [
    "AuditDependencies",
    "AuditService",
    "ClaimInput",
    "EvidenceRecord",
    "MAX_CLAIMS",
    "MAX_EVIDENCE_REFS",
    "MineruSidecarEvidenceRef",
    "PdfPageEvidenceRef",
    "SemanticEvidenceRef",
    "parse_claims",
    "normalize_quote_text",
    "quote_contained",
]
