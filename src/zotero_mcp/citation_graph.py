"""Deterministic Academic Citation & Co-Citation Graph for zotero-mcp.

Extracts ground-truth citation, authorship, and collection relationships
directly from local Zotero SQLite metadata and MinerU bibliography sidecars ([REF]).
Maintains an in-memory NetworkX directed graph for fast topological queries:
- Collection Hubs (In-Degree / PageRank)
- Methodological Lineage (Ancestor / Descendant chains)
- Connected Papers (Bibliographic Coupling / Co-Citation overlap)

Unmatched DOI references are retained as explicitly labeled external-reference
nodes for expanded/open-world graph views. Graph rebuilds do not touch semantic
embeddings.
"""

import hashlib
import json
import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from .reference_parser import iter_all_reference_entries, reference_dois

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_DB_PATH = Path.home() / ".config" / "zotero-mcp" / "citation_graph.sqlite"
DEFAULT_SIDECAR_DIR = Path.home() / ".config" / "zotero-mcp" / "mineru-sidecars"
DEFAULT_ZOTERO_DB = Path.home() / "Zotero" / "zotero.sqlite"

_STOPWORDS = {
    "with", "from", "that", "this", "what", "where", "when", "using",
    "evidence", "effect", "effects", "impact", "impacts", "journal",
    "economics", "review", "economic", "paper", "working", "series",
    "study", "analysis", "empirical", "model", "models", "approach",
}

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b", re.IGNORECASE)
_EXTERNAL_NODE_TYPE = "external_reference"
_MAX_METADATA_MATCH_LENGTH = 1200
_MAX_METADATA_MATCH_YEARS = 3
_LIBRARY_NODE_TYPE = "zotero_item"
_VALID_SCOPES = {
    "collection",
    "library",
    "collection-expanded",
    "library-expanded",
}


def _normalize_doi(value: str) -> str:
    """Return a conservative canonical DOI suitable for exact matching."""
    doi = (value or "").strip().lower()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.rstrip(".,;:)]}>")


def _normalize_reference(value: str) -> str:
    """Collapse OCR whitespace and list markers for stable reference IDs."""
    ref = re.sub(r"^\s*(?:\[\d+\]|\d+\.)\s*", "", value or "")
    return re.sub(r"\s+", " ", ref).strip(" \t\r\n")


def _external_key(kind: str, value: str) -> str:
    """Create a stable, non-Zotero graph key for an external reference."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"ext:{kind}:{digest}"


def _reference_context(text: str, start: int, end: int, limit: int = 800) -> str:
    """Return a bounded local bibliography context around a matched DOI."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    context = _normalize_reference(text[line_start:line_end])
    if 20 <= len(context) <= limit:
        return context

    window_start = max(0, start - limit // 2)
    window_end = min(len(text), end + limit // 2)
    return _normalize_reference(text[window_start:window_end])[:limit]


def _reference_contains_citekey(
    reference_text: str,
    citekey: str,
) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(citekey)}(?![A-Za-z0-9])",
            reference_text,
            re.IGNORECASE,
        )
    )


def _metadata_match_allowed(reference_text: str, split_method: str = "") -> bool:
    """Reject broad/collapsed entries before title-based resolution."""
    if split_method == "whole-section":
        return False
    if len(reference_text) > _MAX_METADATA_MATCH_LENGTH:
        return False
    years = {match.group(0)[:4] for match in _YEAR_RE.finditer(reference_text)}
    return len(years) <= _MAX_METADATA_MATCH_YEARS


def _match_library_targets(
    reference_text: str,
    src_key: str,
    by_doi: dict[str, str],
    by_citekey: dict[str, str],
    by_title_words: list[tuple[set[str], str, str, list[str], str]],
) -> set[str]:
    """Match one conservative reference entry against library metadata."""
    ref_lower = reference_text.lower()
    ref_tokens = set(re.findall(r"[a-z0-9]+", ref_lower))
    ref_years = {match.group(0)[:4] for match in _YEAR_RE.finditer(reference_text)}
    matches: set[str] = set()

    for doi, target_key in by_doi.items():
        if target_key != src_key and doi in ref_lower:
            matches.add(target_key)

    for citekey, target_key in by_citekey.items():
        if target_key != src_key and _reference_contains_citekey(reference_text, citekey):
            matches.add(target_key)

    for words, target_key, _title, target_creators, target_year in by_title_words:
        if target_key == src_key or not target_year or target_year not in ref_years:
            continue
        matching_words = words & ref_tokens
        required_words = min(3, len(words))
        if (
            len(matching_words) >= required_words
            and len(matching_words) >= len(words) * 0.7
        ):
            author_match = (
                any(creator in ref_tokens for creator in target_creators)
                if target_creators
                else False
            )
            if author_match:
                matches.add(target_key)

    return matches


def _external_title(raw_reference: str, fallback: str) -> str:
    """Return a bounded display label while retaining the raw reference separately."""
    label = _normalize_reference(raw_reference)
    if len(label) > 240:
        label = label[:237].rstrip() + "..."
    return label or fallback


def _title_fingerprint(title: str) -> str:
    """Stable alphanumeric fingerprint of a title for external-node identity.

    Lowercases, drops a leading article, keeps alnum only, and truncates so
    that citation variants of the same paper (e.g. "A.E.R." vs "American
    Economic Review") produce the same identity key.
    """
    t = (title or "").lower()
    t = re.sub(r"^(?:the|a|an)\s+", "", t)
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t[:64]


def _extract_meta_identity(reference_text: str) -> Optional[tuple[str, str, str]]:
    """Extract (first-author surname, year, title) from a DOI-less reference.

    Used to create metadata-based external graph nodes (``ext:meta:*``) for
    entries that resolve to no library item and carry no DOI. Returns None
    when the reference is too ambiguous, too short, or unparseable. The title
    is cut at the first quoted string or the first sentence boundary so
    journal/volume/page noise stays out of the identity fingerprint.
    """
    text = _normalize_reference(reference_text)
    if not text or len(text) > _MAX_METADATA_MATCH_LENGTH:
        return None
    year_match = _YEAR_RE.search(text)
    if not year_match:
        return None
    year = year_match.group(0)[:4]
    pre = text[: year_match.start()].strip()
    post = text[year_match.end():].strip()
    # Strip leading separators left by the author-list year (e.g. ". " after
    # "2011." or "(1997)") so they don't become the title cut point.
    post = re.sub(r"^[^A-Za-z0-9À-ÿ]+", "", post)
    if not pre or not post:
        return None
    author_tokens = re.findall(r"[A-Za-zÀ-ÿ]+", pre)
    if not author_tokens:
        return None
    surname = author_tokens[0].lower()
    # Title: prefer the first quoted string (try BEFORE stripping leading
    # separators so a leading opening quote is still present; tolerate
    # mismatched OCR quote characters). Otherwise strip leading separators
    # left by the author-list year and cut at the first sentence boundary.
    quoted = re.search(r"[\"\u201c]([^\"\u201d\u201c]{4,}?)[\"\u201d\u201c]", post)
    if quoted:
        title = quoted.group(1)
    else:
        post = re.sub(r"^[^A-Za-z0-9À-ÿ]+", "", post)
        cut = re.search(r"\.\s*[\"\u201d]?\s*(?=[A-ZÀ-ÿ])", post)
        if cut and cut.start() > 0:
            title = post[: cut.start()].strip()
        else:
            # No usable sentence boundary: keep a bounded head of the post-year
            # text so journal/volume noise stays out of the fingerprint.
            head = post.split(",")[0]
            title = head[:120] if len(head) > 4 else post[:120]
    title = re.sub(r"\s+", " ", title).strip(" \t\r\n.,;:")
    if len(title) < 4 or len(title) > 400:
        return None
    if len(re.findall(r"[A-Za-z0-9]+", title)) < 2:
        return None
    return surname, year, title


@dataclass
class GraphNode:
    item_key: str
    title: str
    creators: str
    year: str
    citekey: str
    doi: str
    collections: list[str]
    node_type: str = _LIBRARY_NODE_TYPE
    external_id: str = ""
    raw_reference: str = ""
    confidence: float = 1.0


class CitationGraph:
    """Deterministic Citation Graph over a local Zotero library."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_GRAPH_DB_PATH
        # A source/target pair can legitimately have both a citation and a
        # coauthor relation. Keep both in memory; citation subgraphs below
        # collapse only the relation-specific view.
        self.graph = nx.MultiDiGraph()
        self._loaded = False

    # -- Building -------------------------------------------------------------
    def build(
        self,
        zotero_db_path: Optional[Path | str] = None,
        sidecar_dir: Optional[Path | str] = None,
    ) -> dict[str, Any]:
        """Build the citation graph from Zotero SQLite + MinerU sidecars."""
        z_path = Path(zotero_db_path) if zotero_db_path else DEFAULT_ZOTERO_DB
        sc_dir = Path(sidecar_dir) if sidecar_dir else DEFAULT_SIDECAR_DIR

        if not z_path.exists():
            raise FileNotFoundError(f"Zotero database not found at {z_path}")

        # 1. Read library items from Zotero SQLite (immutable mode bypasses locks)
        uri = f"file:{z_path.resolve()}?immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Query items
        query_items = """
        SELECT
            i.itemID,
            i.key,
            (SELECT value FROM itemDataValues JOIN itemData ON itemData.valueID = itemDataValues.valueID JOIN fields ON fields.fieldID = itemData.fieldID WHERE itemData.itemID = i.itemID AND fields.fieldName = 'title') as title,
            (SELECT value FROM itemDataValues JOIN itemData ON itemData.valueID = itemDataValues.valueID JOIN fields ON fields.fieldID = itemData.fieldID WHERE itemData.itemID = i.itemID AND fields.fieldName = 'date') as date,
            (SELECT value FROM itemDataValues JOIN itemData ON itemData.valueID = itemDataValues.valueID JOIN fields ON fields.fieldID = itemData.fieldID WHERE itemData.itemID = i.itemID AND fields.fieldName = 'extra') as extra,
            (SELECT value FROM itemDataValues JOIN itemData ON itemData.valueID = itemDataValues.valueID JOIN fields ON fields.fieldID = itemData.fieldID WHERE itemData.itemID = i.itemID AND fields.fieldName = 'DOI') as doi,
            (SELECT typeName FROM itemTypes WHERE itemTypeID = i.itemTypeID) as item_type,
            (SELECT GROUP_CONCAT(CASE WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL THEN c.lastName || ', ' || c.firstName WHEN c.lastName IS NOT NULL THEN c.lastName ELSE c.firstName END, '; ')
             FROM itemCreators ic
             JOIN creators c ON ic.creatorID = c.creatorID
             WHERE ic.itemID = i.itemID
             ORDER BY ic.orderIndex) as creators
        FROM items i
        JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
        WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
        -- Filter by type NAME, not hardcoded itemTypeID: IDs are not stable
        -- across Zotero versions (e.g. Zotero 10 renumbered them), which
        -- previously leaked attachment nodes in and dropped documents out.
        """
        cur.execute(query_items)
        rows = cur.fetchall()

        # Query collections for items
        query_cols = """
        SELECT i.key as item_key, c.key as collection_key
        FROM collectionItems ci
        JOIN items i ON ci.itemID = i.itemID
        JOIN collections c ON ci.collectionID = c.collectionID
        """
        cur.execute(query_cols)
        item_cols = defaultdict(list)
        for r in cur.fetchall():
            item_cols[r["item_key"]].append(r["collection_key"])

        conn.close()

        nodes: dict[str, GraphNode] = {}
        source_item_types: dict[str, str] = {}
        by_title_words: list[tuple[set[str], str, str, list[str], str]] = []
        by_doi: dict[str, str] = {}
        by_citekey: dict[str, str] = {}

        for r in rows:
            key = r["key"]
            source_item_types[key] = (r["item_type"] or "unknown").strip() or "unknown"
            title = (r["title"] or "").strip()
            creators = (r["creators"] or "").strip()
            extra = (r["extra"] or "").strip()
            doi = _normalize_doi(r["doi"] or "")
            date_str = (r["date"] or "").strip()

            # Year resolution
            year = ""
            m_year = re.search(r"\b(19\d\d|20\d\d)\b", date_str or extra)
            if m_year:
                year = m_year.group(1)

            # Citekey resolution from Extra
            citekey = ""
            m_ck = re.search(r"Citation Key:\s*([^\s\n]+)", extra, re.IGNORECASE)
            if m_ck:
                candidate_citekey = m_ck.group(1).strip().strip(".,;")
                # Zotero Extra sometimes contains labels such as ``Report``
                # or the next ``DOI:`` field after a malformed citation-key
                # line. Only retain key-like values with a year/digit token.
                if (
                    candidate_citekey
                    and not candidate_citekey.endswith(":")
                    and re.search(r"\d", candidate_citekey)
                ):
                    citekey = candidate_citekey
                    by_citekey[citekey.lower()] = key

            if doi:
                by_doi[doi] = key

            # Clean creator last names
            creators_clean = []
            if creators:
                for c in creators.split(";"):
                    last = c.strip().split(",")[0].strip().lower()
                    if len(last) > 2:
                        creators_clean.append(last)

            # Title keywords
            if title and len(title) > 8:
                words = {
                    w for w in re.findall(r"[a-z0-9]+", title.lower())
                    if len(w) > 3 and w not in _STOPWORDS
                }
                if len(words) >= 2:
                    by_title_words.append((words, key, title, creators_clean, year))

            cols = item_cols.get(key, [])
            nodes[key] = GraphNode(
                item_key=key,
                title=title,
                creators=creators,
                year=year,
                citekey=citekey,
                doi=doi,
                collections=cols,
            )

        # 2. Parse sidecar bibliographies entry by entry. Exact DOI matches
        # remain deterministic; citekey/title matches are retained with an
        # explicit method and confidence; unresolved and ambiguous entries are
        # persisted as audit evidence but do not become graph edges.
        edges: set[tuple[str, str, str, float]] = set()
        reference_evidence: list[tuple[str, str, str, int, str, str, float]] = []
        reference_audit: list[tuple[str, str, int, str, str, str, float, str, str, str, str, float, str]] = []
        reference_sidecars = 0
        reference_sections = 0
        reference_entries = 0
        orphan_reference_sidecars = 0
        orphan_reference_entries = 0
        reference_entries_with_doi = 0
        resolved_reference_entries = 0
        external_reference_entries = 0
        ambiguous_reference_entries = 0
        unresolved_reference_entries = 0
        metadata_external_reference_entries = 0
        reference_split_methods: dict[str, int] = defaultdict(int)
        reference_entries_by_item_type: dict[str, int] = defaultdict(int)

        if sc_dir.exists():
            for sc_path in sorted(sc_dir.glob("*.md")):
                src_key = sc_path.stem

                try:
                    text = sc_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                entry_records = list(iter_all_reference_entries(text))
                if not entry_records:
                    continue
                section_count = len({section.section_index for section, _entry in entry_records})
                if src_key not in nodes:
                    orphan_reference_sidecars += 1
                    orphan_reference_entries += len(entry_records)
                    continue
                reference_sidecars += 1
                reference_sections += section_count
                source_item_type = source_item_types.get(src_key, "unknown")

                for reference_section, entry in entry_records:
                    raw_reference = entry.raw_text
                    entry_index = entry.index
                    reference_entries += 1
                    reference_entries_by_item_type[source_item_type] += 1
                    reference_split_methods[entry.split_method] += 1
                    dois = reference_dois(raw_reference)
                    if dois:
                        reference_entries_with_doi += 1

                    target_keys: set[str] = set()
                    target_types: set[str] = set()
                    match_methods: set[str] = set()
                    match_confidences: list[float] = []
                    ambiguous = False
                    self_reference = False
                    external = False
                    metadata_match_allowed = _metadata_match_allowed(
                        raw_reference,
                        entry.split_method,
                    )

                    # Resolve every DOI in the entry independently. The normal
                    # case is one DOI; multiple DOI strings are retained as
                    # multiple evidence rows while sharing the parsed ordinal.
                    seen_entry_dois: set[str] = set()
                    for doi_match in _DOI_RE.finditer(raw_reference):
                        doi = _normalize_doi(doi_match.group(0))
                        if not doi or doi in seen_entry_dois:
                            continue
                        seen_entry_dois.add(doi)
                        target_key = by_doi.get(doi)
                        if target_key and target_key != src_key:
                            target_keys.add(target_key)
                            target_types.add(_LIBRARY_NODE_TYPE)
                            match_methods.add("doi")
                            match_confidences.append(0.99)
                            edges.add((src_key, target_key, "cites", 1.0))
                            reference_evidence.append(
                                (
                                    src_key,
                                    target_key,
                                    str(sc_path),
                                    entry_index,
                                    raw_reference,
                                    "doi",
                                    0.99,
                                )
                            )
                            continue
                        if doi in by_doi:
                            # The reference repeats the DOI of its own source
                            # item (common in AEA ``Dataset`` records and
                            # textbook sidecars). Do not manufacture an
                            # external node merely because self-edges are
                            # suppressed. Preserve the audit evidence without
                            # adding a graph edge.
                            self_reference = True
                            match_methods.add("self_reference")
                            reference_evidence.append(
                                (
                                    src_key,
                                    "",
                                    str(sc_path),
                                    entry_index,
                                    raw_reference,
                                    "self_reference",
                                    0.0,
                                )
                            )
                            continue

                        # A DOI absent from the library can still belong to a
                        # library item whose DOI field is missing. Only accept
                        # this fallback when the entry has one unambiguous
                        # library match; ambiguous candidates remain audit-only.
                        candidates = (
                            _match_library_targets(
                                raw_reference,
                                src_key,
                                by_doi,
                                by_citekey,
                                by_title_words,
                            )
                            if metadata_match_allowed
                            else set()
                        )
                        if len(candidates) == 1:
                            resolved_key = next(iter(candidates))
                            target_keys.add(resolved_key)
                            target_types.add(_LIBRARY_NODE_TYPE)
                            match_methods.add("doi+title_author_year")
                            match_confidences.append(min(0.85, entry.confidence))
                            edges.add((src_key, resolved_key, "cites", 1.0))
                            reference_evidence.append(
                                (
                                    src_key,
                                    resolved_key,
                                    str(sc_path),
                                    entry_index,
                                    raw_reference,
                                    "doi+title_author_year",
                                    min(0.85, entry.confidence),
                                )
                            )
                        elif len(candidates) > 1:
                            ambiguous = True
                        else:
                            doi_context = _reference_context(
                                raw_reference, doi_match.start(), doi_match.end()
                            )
                            external_key = _external_key("doi", doi)
                            year_match = re.search(r"\b(?:19|20)\d{2}\b", doi_context)
                            nodes.setdefault(
                                external_key,
                                GraphNode(
                                    item_key=external_key,
                                    title=_external_title(doi_context, f"DOI {doi}"),
                                    creators="",
                                    year=year_match.group(0) if year_match else "",
                                    citekey="",
                                    doi=doi,
                                    collections=[],
                                    node_type=_EXTERNAL_NODE_TYPE,
                                    external_id=f"doi:{doi}",
                                    raw_reference=doi_context,
                                    confidence=0.95,
                                ),
                            )
                            target_keys.add(external_key)
                            target_types.add(_EXTERNAL_NODE_TYPE)
                            match_methods.add("doi")
                            match_confidences.append(0.95)
                            external = True
                            edges.add((src_key, external_key, "cites", 1.0))
                            reference_evidence.append(
                                (
                                    src_key,
                                    external_key,
                                    str(sc_path),
                                    entry_index,
                                    raw_reference,
                                    "doi",
                                    0.95,
                                )
                            )

                    if not dois:
                        candidates = set()
                        # Broad/collapsed entries are retained for audit and
                        # retrieval, but are never title-matched into graph
                        # edges. DOI identity above remains safe.
                        if metadata_match_allowed:
                            candidates = _match_library_targets(
                                raw_reference,
                                src_key,
                                by_doi,
                                by_citekey,
                                by_title_words,
                            )
                        if len(candidates) == 1:
                            resolved_key = next(iter(candidates))
                            method = (
                                "citekey"
                                if any(
                                    target_key == resolved_key
                                    and _reference_contains_citekey(raw_reference, citekey)
                                    for citekey, target_key in by_citekey.items()
                                )
                                else "title_author_year"
                            )
                            confidence = min(0.90, entry.confidence)
                            target_keys.add(resolved_key)
                            target_types.add(_LIBRARY_NODE_TYPE)
                            match_methods.add(method)
                            match_confidences.append(confidence)
                            edges.add((src_key, resolved_key, "cites", 1.0))
                            reference_evidence.append(
                                (
                                    src_key,
                                    resolved_key,
                                    str(sc_path),
                                    entry_index,
                                    raw_reference,
                                    method,
                                    confidence,
                                )
                            )
                        elif len(candidates) > 1:
                            ambiguous = True

                        # No library match and no DOI: create a metadata-based
                        # external node from a conservative identity extraction
                        # (first-author surname + year + title fingerprint).
                        # [graph meta-external nodes]
                        if (
                            not target_keys
                            and not ambiguous
                            and metadata_match_allowed
                        ):
                            meta = _extract_meta_identity(raw_reference)
                            if meta:
                                surname, meta_year, meta_title = meta
                                src_meta = nodes.get(src_key)
                                if (
                                    src_meta
                                    and src_meta.year
                                    and src_meta.year[:4] == meta_year
                                    and src_meta.creators
                                    and surname in src_meta.creators.lower()
                                ):
                                    # Same first author + year as the source
                                    # item: almost certainly a self-citation
                                    # (e.g. own NBER WP). Do not self-edge.
                                    meta = None
                            if meta:
                                surname, meta_year, meta_title = meta
                                meta_key = _external_key(
                                    "meta",
                                    f"{surname}|{meta_year}|{_title_fingerprint(meta_title)}",
                                )
                                meta_conf = min(0.72, entry.confidence)
                                nodes.setdefault(
                                    meta_key,
                                    GraphNode(
                                        item_key=meta_key,
                                        title=_external_title(meta_title, f"External {surname} {meta_year}"),
                                        creators="",
                                        year=meta_year,
                                        citekey="",
                                        doi="",
                                        collections=[],
                                        node_type=_EXTERNAL_NODE_TYPE,
                                        external_id=f"meta:{surname}:{meta_year}",
                                        raw_reference=raw_reference,
                                        confidence=meta_conf,
                                    ),
                                )
                                target_keys.add(meta_key)
                                target_types.add(_EXTERNAL_NODE_TYPE)
                                match_methods.add("meta_external")
                                match_confidences.append(meta_conf)
                                external = True
                                metadata_external_reference_entries += 1
                                edges.add((src_key, meta_key, "cites", 1.0))
                                reference_evidence.append(
                                    (
                                        src_key,
                                        meta_key,
                                        str(sc_path),
                                        entry_index,
                                        raw_reference,
                                        "meta_external",
                                        meta_conf,
                                    )
                                )

                    if ambiguous and not target_keys:
                        match_methods.add("ambiguous_title_author_year")
                        match_confidences.append(0.40)
                        reference_evidence.append(
                            (
                                src_key,
                                "",
                                str(sc_path),
                                entry_index,
                                raw_reference,
                                "ambiguous_title_author_year",
                                0.40,
                            )
                        )
                    elif not target_keys:
                        if not self_reference:
                            match_methods.add(
                                "unresolved_complex_entry"
                                if not metadata_match_allowed
                                else "unresolved"
                            )
                            reference_evidence.append(
                                (
                                    src_key,
                                    "",
                                    str(sc_path),
                                    entry_index,
                                    raw_reference,
                                    "unresolved",
                                    0.0,
                                )
                            )

                    if target_keys and external and any(
                        target_type == _LIBRARY_NODE_TYPE for target_type in target_types
                    ):
                        target_status = "mixed"
                    elif target_keys and external:
                        target_status = "external_reference"
                    elif target_keys:
                        target_status = "resolved"
                    elif ambiguous:
                        target_status = "ambiguous"
                    else:
                        target_status = "unresolved"

                    if target_status == "resolved":
                        resolved_reference_entries += 1
                    elif target_status == "external_reference":
                        external_reference_entries += 1
                    elif target_status == "mixed":
                        resolved_reference_entries += 1
                        external_reference_entries += 1
                    elif target_status == "ambiguous":
                        ambiguous_reference_entries += 1
                    else:
                        unresolved_reference_entries += 1

                    match_method = "+".join(sorted(match_methods))
                    resolution_confidence = max(match_confidences, default=0.0)
                    reference_audit.append(
                        (
                            src_key,
                            str(sc_path),
                            entry_index,
                            raw_reference,
                            reference_section.heading,
                            entry.split_method,
                            entry.confidence,
                            ",".join(dois),
                            json.dumps(sorted(target_keys)),
                            target_status,
                            match_method,
                            resolution_confidence,
                            source_item_type,
                        )
                    )

        # 3. Add co-authorship and shared-collection edges
        author_papers = defaultdict(list)
        for key, node in nodes.items():
            if node.creators:
                for c in node.creators.split(";"):
                    c_clean = c.strip().split(",")[0].strip().lower()
                    if len(c_clean) > 3:
                        author_papers[c_clean].append(key)

        for author, p_keys in author_papers.items():
            if len(p_keys) > 1:
                for i in range(len(p_keys)):
                    for j in range(i + 1, len(p_keys)):
                        k1, k2 = p_keys[i], p_keys[j]
                        edges.add((k1, k2, "coauthor", 0.5))
                        edges.add((k2, k1, "coauthor", 0.5))

        # 4. Save to local SQLite database
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db_conn = sqlite3.connect(self.db_path)
        db_cur = db_conn.cursor()

        db_cur.execute("DROP TABLE IF EXISTS reference_audit")
        db_cur.execute("DROP TABLE IF EXISTS reference_evidence")
        db_cur.execute("DROP TABLE IF EXISTS nodes")
        db_cur.execute("DROP TABLE IF EXISTS edges")

        db_cur.execute("""
        CREATE TABLE nodes (
            item_key TEXT PRIMARY KEY,
            title TEXT,
            creators TEXT,
            year TEXT,
            citekey TEXT,
            doi TEXT,
            collections TEXT,
            node_type TEXT NOT NULL DEFAULT 'zotero_item',
            external_id TEXT,
            raw_reference TEXT,
            confidence REAL NOT NULL DEFAULT 1.0
        )
        """)

        db_cur.execute("""
        CREATE TABLE edges (
            source_key TEXT,
            target_key TEXT,
            relation TEXT,
            weight REAL,
            PRIMARY KEY (source_key, target_key, relation)
        )
        """)

        db_cur.execute("""
        CREATE TABLE reference_evidence (
            source_key TEXT,
            target_key TEXT,
            source_path TEXT,
            reference_index INTEGER,
            raw_reference TEXT,
            match_method TEXT,
            confidence REAL,
            PRIMARY KEY (source_key, target_key, source_path, reference_index)
        )
        """)

        db_cur.execute("""
        CREATE TABLE reference_audit (
            source_key TEXT,
            source_path TEXT,
            reference_index INTEGER,
            raw_reference TEXT,
            section_heading TEXT,
            split_method TEXT,
            parse_confidence REAL,
            doi TEXT,
            target_keys TEXT,
            target_status TEXT,
            match_method TEXT,
            confidence REAL,
            source_item_type TEXT,
            PRIMARY KEY (source_key, source_path, reference_index)
        )
        """)

        for node in nodes.values():
            db_cur.execute(
                """
                INSERT INTO nodes (
                    item_key, title, creators, year, citekey, doi, collections,
                    node_type, external_id, raw_reference, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.item_key,
                    node.title,
                    node.creators,
                    node.year,
                    node.citekey,
                    node.doi,
                    json.dumps(node.collections),
                    node.node_type,
                    node.external_id,
                    node.raw_reference,
                    node.confidence,
                ),
            )

        for src, tgt, rel, w in edges:
            db_cur.execute("INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)", (src, tgt, rel, w))

        for evidence in dict.fromkeys(reference_evidence):
            db_cur.execute(
                "INSERT OR IGNORE INTO reference_evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
                evidence,
            )

        for audit_record in dict.fromkeys(reference_audit):
            db_cur.execute(
                "INSERT OR IGNORE INTO reference_audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                audit_record,
            )

        db_cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(source_key)")
        db_cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_tgt ON edges(target_key)")
        db_conn.commit()
        db_cur.execute("SELECT COUNT(*) FROM reference_evidence")
        persisted_reference_evidence = int(db_cur.fetchone()[0])
        db_cur.execute("SELECT COUNT(*) FROM reference_audit")
        persisted_reference_audit = int(db_cur.fetchone()[0])
        db_conn.close()

        # Refresh in-memory graph
        self.load()

        stats = {
            "nodes": len(nodes),
            "library_nodes": len([n for n in nodes.values() if n.node_type == _LIBRARY_NODE_TYPE]),
            "external_nodes": len([n for n in nodes.values() if n.node_type == _EXTERNAL_NODE_TYPE]),
            "directed_citations": len([e for e in edges if e[2] == "cites"]),
            "resolved_citations": len([
                e for e in edges
                if e[2] == "cites"
                and nodes.get(e[1]) is not None
                and nodes[e[1]].node_type == _LIBRARY_NODE_TYPE
            ]),
            "external_citations": len([
                e for e in edges
                if e[2] == "cites"
                and nodes.get(e[1]) is not None
                and nodes[e[1]].node_type == _EXTERNAL_NODE_TYPE
            ]),
            "reference_evidence": persisted_reference_evidence,
            "reference_audit": persisted_reference_audit,
            "reference_sidecars": reference_sidecars,
            "reference_sections": reference_sections,
            "orphan_reference_sidecars": orphan_reference_sidecars,
            "orphan_reference_entries": orphan_reference_entries,
            "reference_entries": reference_entries,
            "reference_entries_with_doi": reference_entries_with_doi,
            "resolved_reference_entries": resolved_reference_entries,
            "external_reference_entries": external_reference_entries,
            "metadata_external_reference_entries": metadata_external_reference_entries,
            "ambiguous_reference_entries": ambiguous_reference_entries,
            "unresolved_reference_entries": unresolved_reference_entries,
            "reference_split_methods": dict(reference_split_methods),
            "reference_entries_by_item_type": dict(reference_entries_by_item_type),
            "total_edges": len(edges),
            "db_path": str(self.db_path),
        }
        return stats

    # -- Loading & Querying ---------------------------------------------------
    def load(self) -> bool:
        """Load SQLite database into in-memory NetworkX DiGraph."""
        if not self.db_path.exists():
            return False

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        self.graph.clear()

        cur.execute("SELECT * FROM nodes")
        for r in cur.fetchall():
            cols = json.loads(r["collections"]) if r["collections"] else []
            keys = set(r.keys())
            self.graph.add_node(
                r["item_key"],
                title=r["title"] or "",
                creators=r["creators"] or "",
                year=r["year"] or "",
                citekey=r["citekey"] or "",
                doi=r["doi"] or "",
                collections=cols,
                node_type=r["node_type"] if "node_type" in keys else _LIBRARY_NODE_TYPE,
                external_id=r["external_id"] if "external_id" in keys else "",
                raw_reference=r["raw_reference"] if "raw_reference" in keys else "",
                confidence=float(r["confidence"] or 1.0) if "confidence" in keys else 1.0,
            )

        cur.execute("SELECT * FROM edges")
        for r in cur.fetchall():
            self.graph.add_edge(
                r["source_key"],
                r["target_key"],
                relation=r["relation"],
                weight=float(r["weight"] or 1.0),
            )

        conn.close()
        self._loaded = True
        return True

    @staticmethod
    def _is_library_node(data: dict[str, Any]) -> bool:
        """Return whether a node represents a resolved Zotero item."""
        return data.get("node_type", _LIBRARY_NODE_TYPE) == _LIBRARY_NODE_TYPE

    @staticmethod
    def _normalize_scope(scope: str = "library", collection_key: str = "") -> str:
        """Normalize public scope names while preserving legacy defaults."""
        value = (scope or "library").strip().lower()
        aliases = {
            "within-library": "library",
            "library-only": "library",
            "strict": "collection",
        }
        value = aliases.get(value, value)
        if value == "expanded":
            value = "collection-expanded" if collection_key else "library-expanded"
        if value not in _VALID_SCOPES:
            allowed = ", ".join(sorted(_VALID_SCOPES))
            raise ValueError(f"Unknown graph scope {scope!r}; choose one of: {allowed}")
        if value in {"collection", "collection-expanded"} and not collection_key:
            raise ValueError(f"Graph scope {value!r} requires collection_key")
        return value

    def _scope_edge_allowed(
        self,
        source_key: str,
        target_key: str,
        scope: str,
        collection_key: str = "",
    ) -> bool:
        """Apply source/target filters for a citation-graph view."""
        source = self.graph.nodes.get(source_key, {})
        target = self.graph.nodes.get(target_key, {})
        if not self._is_library_node(source):
            return False
        if scope == "library":
            return self._is_library_node(target)
        if scope == "collection":
            return (
                self._is_library_node(target)
                and collection_key in source.get("collections", [])
                and collection_key in target.get("collections", [])
            )
        if scope == "collection-expanded":
            return collection_key in source.get("collections", [])
        if scope == "library-expanded":
            return True
        return False

    def _citation_subgraph(self, scope: str, collection_key: str = "") -> nx.DiGraph:
        """Return the citation-only graph restricted to a public scope."""
        citation_graph = nx.DiGraph()
        citation_graph.add_nodes_from(self.graph.nodes(data=True))
        citation_graph.add_edges_from(
            [
                (u, v)
                for u, v, data in self.graph.edges(data=True)
                if data.get("relation") == "cites"
                and self._scope_edge_allowed(u, v, scope, collection_key)
            ]
        )
        return citation_graph

    def _format_node(self, key: str) -> dict[str, Any]:
        data = self.graph.nodes.get(key, {})
        return {
            "item_key": key,
            "title": data.get("title", ""),
            "creators": data.get("creators", ""),
            "year": data.get("year", ""),
            "citekey": data.get("citekey", ""),
            "node_type": data.get("node_type", _LIBRARY_NODE_TYPE),
            "external_id": data.get("external_id", ""),
            "raw_reference": data.get("raw_reference", ""),
            "confidence": data.get("confidence", 1.0),
        }

    def _audit_coverage(self, source_keys: set[str]) -> Optional[dict[str, int]]:
        """Count reference entries by resolution status for a source scope.

        Queries the ``reference_audit`` table for bibliography entries whose
        source item is in ``source_keys`` and groups by ``target_status``.
        This is the honest complement to ``inward_citations``: hub counts
        reflect *resolved* edges only, and every entry counted here as
        ``unresolved``/``ambiguous`` is invisible to those counts. Returns
        None when the audit table is unavailable (graph never built, or the
        source scope is empty).
        """
        if not source_keys or not self.db_path.exists():
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            ph = ",".join(["?"] * len(source_keys))
            cur.execute(
                f"SELECT target_status, COUNT(*) AS n FROM reference_audit "
                f"WHERE source_key IN ({ph}) GROUP BY target_status",
                sorted(source_keys),
            )
            rows = {r["target_status"]: int(r["n"]) for r in cur.fetchall()}
            conn.close()
            return rows
        except sqlite3.Error:
            return None

    def get_collection_hubs(
        self,
        collection_key: str = "",
        top_n: int = 5,
        scope: str = "library",
    ) -> list[dict[str, Any]]:
        """Find hub nodes under an explicit collection/library graph scope.

        Legacy behavior is preserved for the default ``library`` scope: when a
        collection key is supplied, candidates are collection items but their
        inward citations may originate anywhere among resolved library items.
        ``collection`` restricts both ends to the collection, while
        ``collection-expanded`` allows external-reference targets cited by
        collection items.
        """
        if not self._loaded and not self.load():
            return []

        scope = self._normalize_scope(scope, collection_key)
        resolved = {
            key for key, data in self.graph.nodes(data=True)
            if self._is_library_node(data)
        }
        collection_nodes = {
            key for key, data in self.graph.nodes(data=True)
            if self._is_library_node(data)
            and collection_key in data.get("collections", [])
        }

        if scope == "collection":
            source_keys = collection_nodes
            target_keys = collection_nodes
        elif scope == "library":
            source_keys = resolved
            target_keys = collection_nodes if collection_key else resolved
        elif scope == "collection-expanded":
            source_keys = collection_nodes
            target_keys = set(self.graph.nodes())
        else:  # library-expanded
            source_keys = resolved
            target_keys = set(self.graph.nodes())

        if not target_keys:
            return []

        coverage = self._audit_coverage(source_keys)

        citation_graph = self._citation_subgraph(scope, collection_key)
        # Explicitly restrict candidates to the requested target universe. This
        # keeps external nodes out of all legacy/library-only responses.
        in_degrees = [
            (key, citation_graph.in_degree(key)) for key in target_keys
        ]
        if scope.endswith("-expanded"):
            # Expanded hub discovery should return works actually cited by the
            # selected source set, not unrelated zero-degree library nodes.
            in_degrees = [pair for pair in in_degrees if pair[1] > 0]
        sorted_hubs = sorted(in_degrees, key=lambda pair: (-pair[1], pair[0]))[:top_n]

        results = []
        for key, degree in sorted_hubs:
            data = self.graph.nodes[key]
            results.append({
                **self._format_node(key),
                "inward_citations": degree,
                "scope": scope,
                "source_node_count": len(source_keys),
                "resolution_coverage": coverage,
            })
        return results

    def get_paper_lineage(
        self,
        item_key: str,
        depth: int = 1,
        scope: str = "library",
        collection_key: str = "",
    ) -> dict[str, Any]:
        """Return direct citation ancestors and descendants under a scope."""
        if not self._loaded and not self.load():
            return {"error": "Graph not loaded"}

        if item_key not in self.graph:
            return {"error": f"Item {item_key} not found in graph"}

        scope = self._normalize_scope(scope, collection_key)
        cit_sub = self._citation_subgraph(scope, collection_key)

        # ``depth`` remains part of the public API; this first expanded view
        # preserves the established direct-neighbor behavior.
        ancestors = list(cit_sub.successors(item_key)) if item_key in cit_sub else []
        descendants = list(cit_sub.predecessors(item_key)) if item_key in cit_sub else []

        return {
            "target_paper": self._format_node(item_key),
            "cites": [self._format_node(key) for key in ancestors],
            "cited_by": [self._format_node(key) for key in descendants],
            "scope": scope,
            "depth": depth,
        }

    def find_connected_papers(
        self,
        item_key: str,
        top_n: int = 5,
        scope: str = "library",
        collection_key: str = "",
    ) -> list[dict[str, Any]]:
        """Find bibliographically coupled resolved papers under a scope."""
        if not self._loaded and not self.load():
            return []

        if item_key not in self.graph:
            return []

        scope = self._normalize_scope(scope, collection_key)
        resolved = {
            key for key, data in self.graph.nodes(data=True)
            if self._is_library_node(data)
        }
        collection_nodes = {
            key for key, data in self.graph.nodes(data=True)
            if self._is_library_node(data)
            and collection_key in data.get("collections", [])
        }
        source_keys = collection_nodes if scope in {"collection", "collection-expanded"} else resolved
        if item_key not in source_keys:
            return []

        target_keys = (
            collection_nodes
            if scope == "collection"
            else resolved
            if scope == "library"
            else set(self.graph.nodes())
        )
        cit_sub = self._citation_subgraph(scope, collection_key)
        target_cites = (
            set(cit_sub.successors(item_key)).intersection(target_keys)
            if item_key in cit_sub
            else set()
        )
        if not target_cites:
            return []

        scores = []
        for other_key in sorted(source_keys):
            if other_key == item_key:
                continue
            other_cites = (
                set(cit_sub.successors(other_key)).intersection(target_keys)
                if other_key in cit_sub
                else set()
            )
            if not other_cites:
                continue
            shared = target_cites.intersection(other_cites)
            if shared:
                union = target_cites.union(other_cites)
                scores.append((other_key, len(shared) / len(union), shared))

        scores.sort(key=lambda row: (-row[1], row[0]))
        results = []
        for key, score, shared in scores[:top_n]:
            results.append({
                **self._format_node(key),
                "coupling_score": round(score, 3),
                "shared_citations_count": len(shared),
                "shared_citations": [
                    self.graph.nodes.get(shared_key, {}).get("title", shared_key)
                    for shared_key in sorted(shared)[:3]
                ],
                "scope": scope,
            })
        return results
