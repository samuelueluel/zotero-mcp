"""Reference-only retrieval over parsed bibliography entries.

Phase 2B keeps a separate BM25 index whose records come from local MinerU
sidecars. It reuses the graph's resolution/audit output when available and does
not create or update dense embeddings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .reference_parser import iter_all_reference_entries, reference_dois
from .sparse_index import BM25Index

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"
DEFAULT_SIDECAR_DIR = Path.home() / ".config" / "zotero-mcp" / "mineru-sidecars"
DEFAULT_GRAPH_DB_PATH = Path.home() / ".config" / "zotero-mcp" / "citation_graph.sqlite"
DEFAULT_ZOTERO_DB = Path.home() / "Zotero" / "zotero.sqlite"
DEFAULT_REFERENCE_INDEX_PATH = (
    Path.home() / ".config" / "zotero-mcp" / "bm25_reference_index.json"
)
_REFERENCE_BREADCRUMB_RE = re.compile(
    r"\b(?:references|reference|bibliography|bibliographic|works cited|literature cited)\b",
    re.IGNORECASE,
)
_REFERENCE_CACHE: dict[str, "ReferenceIndex"] = {}
_REFERENCE_INDEX_VERSION = 2


def is_bibliography_chunk(text: str) -> bool:
    """Return whether a stored chunk is explicitly marked as bibliography text."""
    first_line = (text or "").split("\n", 1)[0]
    return bool(_REFERENCE_BREADCRUMB_RE.search(first_line))


def _reference_metadata_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}.meta.json")


def _input_fingerprint(sidecar_dir: Path, graph_db_path: Path) -> str:
    """Fingerprint sidecar inputs so a process cannot serve stale metadata."""
    digest = hashlib.sha256()
    try:
        graph_stat = graph_db_path.stat()
        digest.update(f"graph:{graph_stat.st_mtime_ns}:{graph_stat.st_size}".encode())
    except OSError:
        digest.update(b"graph:missing")
    try:
        paths = sorted(sidecar_dir.glob("*.md"))
    except OSError:
        paths = []
    for path in paths:
        try:
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}".encode())
        except OSError:
            continue
    return digest.hexdigest()


def _resolve_index_path(config_path: str | Path | None = None) -> Path:
    """Resolve the optional configured reference-index path."""
    path = DEFAULT_REFERENCE_INDEX_PATH
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        configured = (
            config.get("semantic_search", {})
            .get("hybrid", {})
            .get("reference_index_path", "")
        )
        if configured:
            path = Path(str(configured)).expanduser()
            if not path.is_absolute():
                path = cfg_path.parent / path
    except Exception:
        pass
    return path


def _normalise_collections(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, tuple):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except Exception:
            pass
        return [value] if value else []
    return []


def _load_source_metadata(zotero_db_path: Path) -> dict[str, dict[str, Any]]:
    """Read source-item metadata without taking a Zotero write lock."""
    if not zotero_db_path.exists():
        return {}
    uri = f"file:{zotero_db_path.resolve()}?immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                i.key,
                (SELECT value FROM itemDataValues JOIN itemData ON itemData.valueID = itemDataValues.valueID JOIN fields ON fields.fieldID = itemData.fieldID WHERE itemData.itemID = i.itemID AND fields.fieldName = 'title') AS title,
                (SELECT GROUP_CONCAT(CASE WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL THEN c.lastName || ', ' || c.firstName WHEN c.lastName IS NOT NULL THEN c.lastName ELSE c.firstName END, '; ')
                 FROM itemCreators ic JOIN creators c ON ic.creatorID = c.creatorID
                 WHERE ic.itemID = i.itemID ORDER BY ic.orderIndex) AS creators,
                (SELECT typeName FROM itemTypes WHERE itemTypeID = i.itemTypeID) AS item_type
            FROM items i
            JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
            WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
            -- Filter by type NAME, not hardcoded itemTypeID (Zotero 10
            -- renumbered the IDs; the old (1, 14) leak in attachments and
            -- dropped documents).
            """
        )
        records: dict[str, dict[str, Any]] = {}
        for row in cur.fetchall():
            records[row["key"]] = {
                "source_key": row["key"],
                "title": (row["title"] or "").strip(),
                "creators": (row["creators"] or "").strip(),
                "item_type": (row["item_type"] or "unknown").strip() or "unknown",
                "collections": [],
            }
        cur.execute(
            """
            SELECT i.key AS item_key, c.key AS collection_key
            FROM collectionItems ci
            JOIN items i ON ci.itemID = i.itemID
            JOIN collections c ON ci.collectionID = c.collectionID
            """
        )
        for row in cur.fetchall():
            if row["item_key"] in records:
                records[row["item_key"]]["collections"].append(row["collection_key"])
        conn.close()
        return records
    except Exception as exc:
        logger.warning("Could not load Zotero source metadata: %s", exc)
        return {}


def _load_graph_snapshot(
    graph_db_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    """Load node metadata and Phase 2B per-entry audit rows."""
    if not graph_db_path.exists():
        return {}, {}
    nodes: dict[str, dict[str, Any]] = {}
    audit: dict[tuple[str, str, int], dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(graph_db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM nodes")
        for row in cur.fetchall():
            collections = _normalise_collections(row["collections"])
            nodes[row["item_key"]] = {
                "source_key": row["item_key"],
                "title": row["title"] or "",
                "creators": row["creators"] or "",
                "collections": collections,
                "node_type": row["node_type"] if "node_type" in row.keys() else "zotero_item",
            }
        table = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reference_audit'"
        ).fetchone()
        if table:
            cur.execute("SELECT * FROM reference_audit")
            for row in cur.fetchall():
                try:
                    target_keys = json.loads(row["target_keys"] or "[]")
                    if not isinstance(target_keys, list):
                        target_keys = []
                except Exception:
                    target_keys = []
                key = (row["source_key"], row["source_path"], int(row["reference_index"]))
                audit[key] = {
                    "target_keys": [str(item) for item in target_keys],
                    "target_status": row["target_status"] or "unresolved",
                    "match_method": row["match_method"] or "unresolved",
                    "confidence": float(row["confidence"] or 0.0),
                    "parse_confidence": float(row["parse_confidence"] or 0.0),
                    "doi": row["doi"] or "",
                }
        conn.close()
    except Exception as exc:
        logger.warning("Could not load citation-graph audit metadata: %s", exc)
    return nodes, audit


def _target_types(target_keys: list[str], nodes: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({
        nodes[key].get("node_type", "zotero_item")
        for key in target_keys
        if key in nodes
    })


def _entry_record(
    doc_id: str,
    source_key: str,
    source_path: Path,
    section: Any,
    entry: Any,
    source: dict[str, Any],
    audit: dict[str, Any] | None,
    graph_nodes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dois = reference_dois(entry.raw_text)
    audit = audit or {}
    target_keys = [str(key) for key in audit.get("target_keys", [])]
    target_status = audit.get("target_status", "unresolved")
    match_method = audit.get("match_method", "unresolved")
    confidence = float(audit.get("confidence", 0.0))
    return {
        "doc_id": doc_id,
        "reference_kind": "reference_entry",
        "is_bibliography": True,
        "source_key": source_key,
        "citing_item_key": source_key,
        "source_sidecar": str(source_path),
        "source_title": source.get("title", ""),
        "source_creators": source.get("creators", ""),
        "source_item_type": source.get("item_type", "unknown"),
        "source_status": source.get("source_status", "library_item"),
        "collections": _normalise_collections(source.get("collections", [])),
        "section_heading": section.heading,
        "section_index": section.section_index,
        "section_level": section.level,
        "section_confidence": section.confidence,
        "section_method": section.method,
        "entry_index": entry.index,
        "marker": entry.marker,
        "raw_reference": entry.raw_text,
        "split_method": entry.split_method,
        "parse_confidence": entry.confidence,
        "dois": dois,
        "doi": ",".join(dois),
        "target_keys": target_keys,
        "target_types": _target_types(target_keys, graph_nodes),
        "target_status": target_status,
        "match_method": match_method,
        "confidence": confidence,
    }


class ReferenceIndex:
    """BM25 index plus metadata for individual bibliography entries."""

    def __init__(
        self,
        index_path: str | Path | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.index_path = Path(index_path) if index_path else DEFAULT_REFERENCE_INDEX_PATH
        self.metadata_path = _reference_metadata_path(self.index_path)
        self.bm25 = BM25Index(self.index_path, k1=k1, b=b)
        self.metadata: dict[str, dict[str, Any]] = {}
        self.summary: dict[str, Any] = {}
        self.built_at: str | None = None
        self.input_fingerprint: str | None = None
        self.mode = "reference-entry"

    def build_entries(
        self,
        records: Iterable[tuple[str, str, dict[str, Any]]],
    ) -> dict[str, Any]:
        documents: list[tuple[str, str]] = []
        metadata: dict[str, dict[str, Any]] = {}
        for doc_id, text, raw_metadata in records:
            if not text:
                continue
            documents.append((doc_id, text))
            metadata[doc_id] = dict(raw_metadata)

        self.bm25.build(documents)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25.save()
        self.metadata = metadata
        self.built_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        statuses = Counter(record.get("target_status", "unresolved") for record in metadata.values())
        item_types = Counter(record.get("source_item_type", "unknown") for record in metadata.values())
        split_methods = Counter(record.get("split_method", "unknown") for record in metadata.values())
        source_keys = {
            record.get("source_key") for record in metadata.values() if record.get("source_key")
        }
        orphan_source_keys = {
            record.get("source_key")
            for record in metadata.values()
            if record.get("source_status") == "orphan_sidecar" and record.get("source_key")
        }
        self.summary = {
            "entries": len(metadata),
            "reference_sections": len({
                (record.get("source_sidecar"), record.get("section_index"))
                for record in metadata.values()
                if record.get("source_sidecar")
            }),
            "source_items": len(source_keys),
            "library_source_items": len(source_keys - orphan_source_keys),
            "orphan_source_items": len(orphan_source_keys),
            "source_sidecars": len({record.get("source_sidecar") for record in metadata.values()}),
            "orphan_source_sidecars": len({
                record.get("source_sidecar")
                for record in metadata.values()
                if record.get("source_status") == "orphan_sidecar"
            }),
            "orphan_source_entries": sum(
                record.get("source_status") == "orphan_sidecar" for record in metadata.values()
            ),
            "doi_entries": sum(bool(record.get("dois")) for record in metadata.values()),
            "resolved_entries": statuses.get("resolved", 0),
            "external_doi_entries": statuses.get("external_reference", 0),
            "mixed_entries": statuses.get("mixed", 0),
            "ambiguous_entries": statuses.get("ambiguous", 0),
            "unresolved_entries": statuses.get("unresolved", 0),
            "status_counts": dict(statuses),
            "source_item_types": dict(item_types),
            "split_methods": dict(split_methods),
        }
        payload = {
            "version": _REFERENCE_INDEX_VERSION,
            "mode": self.mode,
            "built_at": self.built_at,
            "input_fingerprint": self.input_fingerprint,
            "summary": self.summary,
            "documents": self.metadata,
        }
        tmp = self.metadata_path.with_name(self.metadata_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.metadata_path)
        return self.stats()

    def build(self, records: Iterable[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
        """Compatibility entry point for generic reference records."""
        return self.build_entries(records)

    def build_from_sidecars(
        self,
        sidecar_dir: str | Path | None = None,
        graph_db_path: str | Path | None = None,
        zotero_db_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Parse local sidecars and build one BM25 record per reference entry."""
        sc_dir = Path(sidecar_dir) if sidecar_dir else DEFAULT_SIDECAR_DIR
        graph_path = Path(graph_db_path) if graph_db_path else DEFAULT_GRAPH_DB_PATH
        zotero_path = Path(zotero_db_path) if zotero_db_path else DEFAULT_ZOTERO_DB
        source_metadata = _load_source_metadata(zotero_path)
        graph_nodes, graph_audit = _load_graph_snapshot(graph_path)
        self.mode = "reference-entry"
        self.input_fingerprint = _input_fingerprint(sc_dir, graph_path)

        def records():
            if not sc_dir.exists():
                return
            for source_path in sorted(sc_dir.glob("*.md")):
                source_key = source_path.stem
                try:
                    text = source_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                entry_records = list(iter_all_reference_entries(text))
                if not entry_records:
                    continue
                source_exists = source_key in graph_nodes or source_key in source_metadata
                source = dict(graph_nodes.get(source_key, {}))
                source.update(source_metadata.get(source_key, {}))
                if not source:
                    source = {
                        "source_key": source_key,
                        "title": "",
                        "creators": "",
                        "item_type": "unknown",
                        "collections": [],
                    }
                source["source_status"] = "library_item" if source_exists else "orphan_sidecar"
                for section, entry in entry_records:
                    doc_id = f"ref:{source_key}:{entry.index:04d}"
                    audit = graph_audit.get((source_key, str(source_path), entry.index))
                    yield (
                        doc_id,
                        entry.raw_text,
                        _entry_record(
                            doc_id,
                            source_key,
                            source_path,
                            section,
                            entry,
                            source,
                            audit,
                            graph_nodes,
                        ),
                    )

        return self.build_entries(records())

    def build_from_chroma(self, chroma_client: Any) -> dict[str, Any]:
        """Compatibility fallback for the Phase 2A chunk index."""
        def records():
            iterator = getattr(chroma_client, "iter_documents", None)
            if callable(iterator):
                for ids, documents, metadatas in iterator():
                    for doc_id, text, metadata in zip(ids, documents, metadatas):
                        if text and is_bibliography_chunk(text):
                            metadata = dict(metadata or {})
                            metadata.update({
                                "reference_kind": "bibliography_chunk",
                                "raw_reference": text,
                                "source_key": metadata.get("parent_item_key") or metadata.get("item_key"),
                                "target_status": "unresolved",
                                "match_method": "chunk",
                                "confidence": 0.0,
                            })
                            yield doc_id, text, metadata
                return
            result = chroma_client.collection.get(include=["documents", "metadatas"])
            for doc_id, text, metadata in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            ):
                if text and is_bibliography_chunk(text):
                    metadata = dict(metadata or {})
                    metadata.update({"reference_kind": "bibliography_chunk", "raw_reference": text})
                    yield doc_id, text, metadata

        self.mode = "bibliography-chunk"
        self.input_fingerprint = None
        return self.build_entries(records())

    def load(self, expected_fingerprint: str | None = None) -> bool:
        if not self.bm25.load() or not self.metadata_path.exists():
            return False
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if int(payload.get("version", 0)) < _REFERENCE_INDEX_VERSION:
                return False
            self.input_fingerprint = payload.get("input_fingerprint")
            if expected_fingerprint and self.input_fingerprint != expected_fingerprint:
                return False
            self.mode = payload.get("mode", "reference-entry")
            self.metadata = payload.get("documents", {})
            self.summary = payload.get("summary", {})
            self.built_at = payload.get("built_at")
        except Exception:
            self.metadata = {}
            self.summary = {}
            self.built_at = None
            self.input_fingerprint = None
            return False
        return True

    def stats(self) -> dict[str, Any]:
        sparse_stats = self.bm25.stats()
        stats = dict(self.summary)
        stats.update({
            "docs": sparse_stats["docs"],
            "terms": sparse_stats["terms"],
            "path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "built_at": self.built_at,
            "mode": self.mode,
        })
        return stats

    def search(
        self,
        query: str,
        top_n: int = 10,
        collection_key: str | None = None,
        item_key: str | None = None,
    ) -> list[dict[str, Any]]:
        allowed_ids: set[str] | None = None
        if collection_key or item_key:
            allowed_ids = set()
            for doc_id, record in self.metadata.items():
                if collection_key and collection_key not in record.get("collections", []):
                    continue
                if item_key and item_key not in {
                    record.get("source_key"),
                    record.get("citing_item_key"),
                    record.get("parent_item_key"),
                }:
                    continue
                allowed_ids.add(doc_id)

        # DOI queries should use deterministic exact matching rather than
        # BM25's tokenization of numeric DOI fragments ("10", "1111", etc.).
        query_dois = set(reference_dois(query))
        if query_dois:
            exact_doi_ids = {
                doc_id
                for doc_id, record in self.metadata.items()
                if query_dois.intersection(record.get("dois", []))
            }
            if exact_doi_ids:
                allowed_ids = (
                    exact_doi_ids
                    if allowed_ids is None
                    else allowed_ids.intersection(exact_doi_ids)
                )

        hits = self.bm25.search(query, top_n=max(1, int(top_n)), allowed_ids=allowed_ids)
        results: list[dict[str, Any]] = []
        for doc_id, score in hits:
            result = dict(self.metadata.get(doc_id, {}))
            result.update({"doc_id": doc_id, "score": round(score, 6)})
            results.append(result)
        return results


def get_cached_reference_index(
    config_path: str | Path | None = None,
) -> ReferenceIndex | None:
    path = _resolve_index_path(config_path)
    cache_key = str(path)
    expected_fingerprint = _input_fingerprint(DEFAULT_SIDECAR_DIR, DEFAULT_GRAPH_DB_PATH)
    cached = _REFERENCE_CACHE.get(cache_key)
    if cached is not None and cached.input_fingerprint == expected_fingerprint:
        return cached
    index = ReferenceIndex(path)
    if not index.load(expected_fingerprint=expected_fingerprint):
        _REFERENCE_CACHE.pop(cache_key, None)
        return None
    _REFERENCE_CACHE[cache_key] = index
    return index


def build_reference_index(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the reference index from local sidecars and graph audit data."""
    path = _resolve_index_path(config_path)
    index = ReferenceIndex(path)
    stats = index.build_from_sidecars()
    _REFERENCE_CACHE[str(path)] = index
    return stats


def search_reference_index(
    query: str,
    top_n: int = 10,
    collection_key: str | None = None,
    item_key: str | None = None,
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    index = get_cached_reference_index(config_path)
    if index is None:
        build_reference_index(config_path)
        index = get_cached_reference_index(config_path)
    if index is None:
        return []
    return index.search(
        query,
        top_n=top_n,
        collection_key=collection_key,
        item_key=item_key,
    )


def audit_reference_index(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return current reference-entry coverage statistics, building if needed."""
    index = get_cached_reference_index(config_path)
    if index is None:
        build_reference_index(config_path)
        index = get_cached_reference_index(config_path)
    return index.stats() if index is not None else {}


def invalidate_reference_index_cache(config_path: str | Path | None = None) -> None:
    _REFERENCE_CACHE.pop(str(_resolve_index_path(config_path)), None)
