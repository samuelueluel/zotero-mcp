"""Reliability signals for Surya sidecars, shared by indexing and retrieval.

Surya sidecars carry ``<!-- pdf-page: N -->`` anchors and visible status
lines (``[Table status: UNRESOLVED ...]``, ``[Equation status: ...]``,
``[Math status: ...]``) written by
:mod:`sidecar_assemble`. A ``<KEY>.reliability.json`` beside the sidecar
holds the paper-level summary. Legacy MinerU sidecars have neither and are
reported as ``legacy-unverified``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PAGE_MARKER_RE = re.compile(r"<!-- pdf-page: (\d+) -->")
STATUS_LINE_RE = re.compile(r"\[(?:Table|Equation|Math) status: (UNRESOLVED|SINGLE-ROUTE|REPAIRED)\b[^\]]*?PDF p\. (\d+)")
WITHHELD_RE = re.compile(r"⟦withheld: unverified number, see PDF p\. (\d+)⟧")

#: Severity order; the worst status in a chunk wins.
STATUS_RANK = {"verified": 0, "repaired": 1, "single-route": 2, "legacy-unverified": 3, "unresolved": 4}

SURYA_SOURCE = "surya-sidecar"
LEGACY_SOURCE = "mineru-sidecar"


def has_page_markers(text: str) -> bool:
    return bool(text) and "<!-- pdf-page: " in text


def page_for_offset(text: str, offset: int) -> int | None:
    """PDF page of the last anchor at or before ``offset``; None if no anchors."""
    page = None
    for m in PAGE_MARKER_RE.finditer(text):
        if m.start() > offset:
            break
        page = int(m.group(1))
    return page


def pages_in_span(text: str, start: int, end: int) -> list[int]:
    first = page_for_offset(text, start)
    pages = [first] if first is not None else []
    for m in PAGE_MARKER_RE.finditer(text, start, end):
        p = int(m.group(1))
        if p not in pages:
            pages.append(p)
    return pages


def chunk_reliability(chunk_text: str, source: str | None) -> dict:
    """Block-level status for one chunk.

    Returns ``{"block_status": ..., "check_pages": [...]}``. Surya chunks
    without a status line or withheld marker are ``verified`` as far as the
    checks go; legacy sidecars are ``legacy-unverified``. Display equations
    that agree with the text layer carry no status line and do not lower the
    chunk: their structure is never checked, which the skills handle by
    requiring the rendered page before any equation is quoted.
    """
    if source != SURYA_SOURCE:
        return {"block_status": "legacy-unverified", "check_pages": []}
    worst = "verified"
    pages: list[int] = []
    for m in STATUS_LINE_RE.finditer(chunk_text):
        status = m.group(1).lower()
        if STATUS_RANK[status] > STATUS_RANK[worst]:
            worst = status
        pages.append(int(m.group(2)))
    for m in WITHHELD_RE.finditer(chunk_text):
        worst = "unresolved"
        pages.append(int(m.group(1)))
    return {"block_status": worst, "check_pages": sorted(set(pages))}


def reliability_path(sidecar_dir: str | Path, item_key: str) -> Path:
    return Path(sidecar_dir) / f"{item_key}.reliability.json"


def load_item_reliability(sidecar_dir: str | Path, item_key: str) -> dict | None:
    try:
        return json.loads(reliability_path(sidecar_dir, item_key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sidecar_source(sidecar_dir: str | Path, item_key: str) -> str:
    return SURYA_SOURCE if reliability_path(sidecar_dir, item_key).is_file() else LEGACY_SOURCE


def requires_pdf_check(block_status: str) -> bool:
    return STATUS_RANK.get(block_status, 3) >= STATUS_RANK["single-route"]


def result_reliability(meta: dict, item_rel: dict | None) -> dict:
    """Reliability fields attached to a retrieval hit."""
    status = meta.get("block_status") or "legacy-unverified"
    out = {
        "block_status": status,
        "requires_pdf_check": requires_pdf_check(status),
        "pdf_pages": meta.get("pdf_pages") or "",
        "check_pages": meta.get("check_pages") or "",
    }
    if item_rel:
        out["item_level"] = item_rel.get("level")
        if item_rel.get("warning"):
            out["item_warning"] = item_rel["warning"]
    else:
        out["item_level"] = "legacy-unverified"
        out["item_warning"] = (
            "Legacy MinerU sidecar with no verification record; confirm any number on the PDF page."
        )
    return out
