"""Assemble Surya batch output into blocks.json, reliability.json and a sidecar.

Input: a batch directory written by :func:`surya_runner.run_batch`.
Per item output (in ``sidecar_dir``):

- ``<KEY>.blocks.json``      canonical per-block record (page, PDF bbox, label,
                             html, status, findings, repairs)
- ``<KEY>.reliability.json`` paper-level summary for retrieval warnings
- ``<KEY>.md``               Markdown sidecar in the existing contract
                             (headings, HTML tables, ``$$`` math) plus
                             ``<!-- pdf-page: N -->`` anchors and visible
                             status lines above every non-verified table.
"""

from __future__ import annotations

import html as _html
import json
import re
import time
from collections import Counter
from pathlib import Path

import pymupdf

from .sidecar_native import (
    image_bbox_to_pdf,
    native_number_tokens,
    symbol_minus_fonts,
    validated_minus_fonts,
)
from .sidecar_verify import text_numbers, verify_table, withhold_numbers

ASSEMBLER_VERSION = "sidecar-assemble/1"
SKIP_LABELS = {"PageHeader", "PageFooter"}
FIGURE_LABELS = {"Picture", "Figure", "Diagram", "ChemicalBlock"}
NUMERIC_STATUSES = ("verified", "repaired", "single-route", "unresolved")

_MATH_BLOCK = re.compile(r'<math\b[^>]*display="block"[^>]*>(.*?)</math>', re.S | re.I)
_MATH_INLINE = re.compile(r"<math\b[^>]*>(.*?)</math>", re.S | re.I)
_BR = re.compile(r"<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")


def html_to_markdown(fragment: str) -> str:
    text = _MATH_BLOCK.sub(lambda m: "\n$$\n" + _html.unescape(m.group(1)).strip() + "\n$$\n", fragment)
    text = _MATH_INLINE.sub(lambda m: "$" + _html.unescape(m.group(1)).strip() + "$", text)
    text = re.sub(r"<sup>(.*?)</sup>", r"^\1", text, flags=re.S | re.I)
    text = re.sub(r"<(b|strong)>(.*?)</\1>", r"**\2**", text, flags=re.S | re.I)
    text = re.sub(r"<(i|em)>(.*?)</\1>", r"*\2*", text, flags=re.S | re.I)
    text = re.sub(r"<li\b[^>]*>", "\n- ", text, flags=re.I)
    text = _BR.sub("\n", text)
    text = _html.unescape(_TAG.sub("", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _status_line(block: dict) -> str | None:
    status = block["status"]
    if status == "verified" or block["label"] != "Table":
        return None
    page = block["page"]
    reasons = ", ".join(sorted({f["kind"] for f in block["findings"]
                                if f["kind"] not in ("merged_estimate_se", "label_or_header_mismatch")})) or "no independent check"
    if status == "unresolved":
        return f"[Table status: UNRESOLVED ({reasons}); unverified numbers withheld. Verify on PDF p. {page}.]"
    if status == "repaired":
        return f"[Table status: REPAIRED ({len(block['repairs'])} sign fix(es) from the PDF text layer). Check PDF p. {page} before quoting.]"
    return f"[Table status: SINGLE-ROUTE ({reasons}); numbers not independently confirmed. Check PDF p. {page} before quoting.]"


def _check_text_numbers(fragment: str, tokens) -> tuple[str, list[dict]]:
    nums = text_numbers(fragment)
    if not nums:
        return "no-numbers", []
    pool = Counter(t.text for t in tokens)
    missing = []
    for n in nums:
        if pool[n] > 0:
            pool[n] -= 1
        else:
            missing.append(n)
    if not missing:
        return "verified", []
    return "single-route", [{"kind": "text_numbers_unmatched", "detail": "", "values": missing[:20]}]


def assemble_item(item_key: str, meta: dict, results: dict, out_dir: Path, batch_dir: Path) -> dict:
    doc = pymupdf.open(meta["pdf_path"])
    minus_fonts = validated_minus_fonts(symbol_minus_fonts(doc))
    blocks: list[dict] = []
    for stem in sorted(meta["pages"], key=lambda s: meta["pages"][s]["page"]):
        pm = meta["pages"][stem]
        page_no = pm["page"]
        page = doc[page_no - 1]
        pres = (results.get(stem) or [{}])[0]
        for order, b in enumerate(pres.get("blocks", [])):
            label = b.get("label", "Text")
            pdf_bbox = image_bbox_to_pdf(b["bbox"], pm["image_size"], pm["page_size"], rotation=pm["rotation"])
            rec = {
                "id": f"{item_key}:p{page_no}:b{order}",
                "page": page_no,
                "order": order,
                "label": label,
                "pdf_bbox": [round(x, 2) for x in pdf_bbox],
                "rotation": pm["rotation"],
                "scan_like": pm["scan_like"],
                "html": b.get("html", ""),
                "status": "unchecked",
                "findings": [],
                "repairs": [],
                "surya_error": bool(b.get("error")),
            }
            if label in SKIP_LABELS or label in FIGURE_LABELS or b.get("skipped"):
                rec["status"] = "not-text"
            else:
                tokens = native_number_tokens(page, page_no, clip=pdf_bbox, minus_fonts=minus_fonts)
                if label == "Table":
                    check = verify_table(rec["html"], tokens, native_is_ocr=pm["scan_like"])
                    rec["status"] = check.status
                    rec["html"] = check.html
                    rec["findings"] = [{"kind": f.kind, "detail": f.detail, "values": f.values[:40]} for f in check.findings]
                    rec["repairs"] = check.repairs
                    rec["counts"] = {"surya": check.surya_numbers, "native": check.native_numbers}
                elif label == "Equation":
                    rec["status"] = "single-route"
                    rec["findings"] = [{"kind": "math_single_route", "detail": "no text-layer check for LaTeX", "values": []}]
                else:
                    status, findings = _check_text_numbers(rec["html"], tokens)
                    if pm["scan_like"] and status == "verified":
                        status = "single-route"
                    rec["status"], rec["findings"] = status, findings
            if rec["surya_error"]:
                rec["status"] = "unresolved"
                rec["findings"].append({"kind": "surya_block_error", "detail": "", "values": []})
            blocks.append(rec)

    reliability = summarize(item_key, meta, blocks)
    markdown = render_markdown(blocks)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "assembler": ASSEMBLER_VERSION,
        "batch_dir": str(batch_dir),
        "pdf_sha256": meta["pdf_sha256"],
        "attachment_key": meta.get("attachment_key"),
        "assembled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / f"{item_key}.blocks.json").write_text(
        json.dumps({"item_key": item_key, **provenance, "blocks": blocks}, indent=1), encoding="utf-8")
    md_path = out_dir / f"{item_key}.md"
    md_path.write_text(markdown, encoding="utf-8")
    import hashlib
    reliability.update(provenance)
    reliability["sidecar_sha256"] = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    (out_dir / f"{item_key}.reliability.json").write_text(json.dumps(reliability, indent=1), encoding="utf-8")
    return reliability


def summarize(item_key: str, meta: dict, blocks: list[dict]) -> dict:
    tables = [b for b in blocks if b["label"] == "Table"]
    table_counts = Counter(b["status"] for b in tables)
    problems = [
        {"page": b["page"], "status": b["status"], "block_id": b["id"],
         "reasons": sorted({f["kind"] for f in b["findings"]})}
        for b in tables if b["status"] != "verified"
    ]
    pages = meta["pages"].values()
    scan_pages = sorted(p["page"] for p in pages if p["scan_like"])
    rotated = sorted(p["page"] for p in pages if p["rotation"])
    n_pages = len(meta["pages"])
    if table_counts.get("unresolved"):
        level = "warn"
    elif scan_pages or table_counts.get("single-route") or table_counts.get("repaired"):
        level = "caution"
    else:
        level = "ok"
    warning = None
    if level != "ok":
        parts = []
        if table_counts.get("unresolved"):
            parts.append(f"{table_counts['unresolved']} of {len(tables)} tables unresolved (numbers withheld)")
        if table_counts.get("single-route"):
            parts.append(f"{table_counts['single-route']} tables single-route")
        if table_counts.get("repaired"):
            parts.append(f"{table_counts['repaired']} tables sign-repaired")
        if scan_pages:
            parts.append(f"{len(scan_pages)} of {n_pages} pages scanned")
        warning = "; ".join(parts) + ". Verify table numbers on the PDF pages listed in problem_tables."
    return {
        "item_key": item_key,
        "level": level,
        "warning": warning,
        "pages": n_pages,
        "scan_pages": scan_pages,
        "rotated_pages": rotated,
        "tables": dict(table_counts),
        "equations": sum(1 for b in blocks if b["label"] == "Equation"),
        "problem_tables": problems,
        "missing_pages": meta.get("missing_pages", []),
    }


def render_markdown(blocks: list[dict]) -> str:
    out: list[str] = []
    current_page = None
    for b in blocks:
        if b["page"] != current_page:
            current_page = b["page"]
            out.append(f"<!-- pdf-page: {current_page} -->")
        label = b["label"]
        if label in SKIP_LABELS:
            continue
        if label in FIGURE_LABELS:
            out.append(f"[Figure on PDF p. {b['page']}; see caption]")
            continue
        if label == "Table":
            line = _status_line(b)
            if line:
                out.append(line)
            html = withhold_numbers(b["html"], b["page"]) if b["status"] == "unresolved" else b["html"]
            out.append(html.strip())
            continue
        text = html_to_markdown(b["html"])
        if not text:
            continue
        if label == "SectionHeader":
            out.append("## " + " ".join(text.split()))
        elif label == "Equation" and "$$" not in text:
            out.append("$$\n" + text + "\n$$")
        else:
            out.append(text)
    return "\n\n".join(out).strip() + "\n"


def assemble_batch(batch_dir: Path, out_dir: Path) -> dict[str, dict]:
    batch_dir = Path(batch_dir)
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads(next(iter(sorted((batch_dir / "ocr").glob("*/results.json")))).read_text(encoding="utf-8"))
    return {
        key: assemble_item(key, meta, results, Path(out_dir), batch_dir)
        for key, meta in manifest["items"].items()
    }
