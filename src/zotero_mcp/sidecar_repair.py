"""Second-model repair pass for unresolved Surya tables.

Runs after :mod:`sidecar_assemble` and before figure enrichment (it rewrites
the sidecar). Each ``unresolved`` table is cropped upright from the PDF and
re-read once by a general VLM (Qwen3-VL on an OpenAI-compatible endpoint).
The re-read is accepted only if it passes the same :func:`verify_table`
check against the native text layer, prepared exactly as in assembly
(:func:`sidecar_assemble.table_native`):

- born-digital text: must come back ``verified`` or ``repaired``;
- OCR text layer (scanned page or invisible text over an image):
  ``single-route`` via OCR-layer agreement, never via an unusable layer.

Anything else leaves the block ``unresolved`` with its numbers withheld and a
``vlm_repair_failed`` finding. No second attempt, no free-form correction.
This module never starts or stops inference services.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import pymupdf
import requests

from .sidecar_native import glyph_profile, upright_rotation
from .sidecar_assemble import PageText, owner_keys, table_native, write_outputs
from .sidecar_verify import verify_table

REPAIR_VERSION = "sidecar-repair/2"
CROP_ZOOM = 200 / 72
PROMPT = (
    "Transcribe this table exactly as one HTML <table>. One <td> per printed cell; keep the "
    "row label as the first cell of each row. Preserve every digit, minus sign, star, "
    "parenthesis and bracket exactly. Do not add, merge, repeat or omit rows or cells. "
    "Output only the HTML."
)
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)


def crop_png(page, pdf_bbox, rotation: int | None = None) -> bytes:
    """Render the table region upright.

    ``rotation`` defaults to the page's upright rotation relative to the
    displayed page. The block's stored rotation describes the image Surya saw,
    which for runner/1 was wrong on ``/Rotate 90`` pages, so it is not used.
    """
    rect = pymupdf.Rect(pdf_bbox) & page.rect
    if rotation is None:
        rotation = upright_rotation(page)
    matrix = pymupdf.Matrix(CROP_ZOOM, CROP_ZOOM).prerotate(rotation)
    return page.get_pixmap(matrix=matrix, clip=rect).tobytes("png")


def ask_vlm(png: bytes, url: str, timeout: float = 900) -> str | None:
    b64 = base64.b64encode(png).decode()
    resp = requests.post(url, json={
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT}]}],
        "temperature": 0, "max_tokens": 8000}, timeout=timeout)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"] or ""
    m = _TABLE_RE.search(text)
    return m.group(0) if m else None


def repair_item(item_key: str, sidecar_dir: Path, vlm_url: str) -> dict:
    sidecar_dir = Path(sidecar_dir)
    record = json.loads((sidecar_dir / f"{item_key}.blocks.json").read_text(encoding="utf-8"))
    meta, blocks = record["meta"], record["blocks"]
    targets = [b for b in blocks if b["label"] == "Table" and b["status"] == "unresolved"]
    summary = {"item_key": item_key, "attempted": len(targets), "accepted": 0, "rejected": 0}
    if not targets:
        return summary
    doc = pymupdf.open(record["pdf_path"])
    profile = glyph_profile(doc)
    texts: dict[int, PageText] = {}
    for b in targets:
        page = doc[b["page"] - 1]
        try:
            html = ask_vlm(crop_png(page, b["pdf_bbox"]), vlm_url)
        except Exception as exc:  # noqa: BLE001 - recorded, block stays unresolved
            html, error = None, str(exc)[:200]
        else:
            error = None if html else "no <table> in model output"
        if html:
            text = texts.setdefault(b["page"], PageText(page, b["page"], profile))
            owners = owner_keys([x for x in blocks if x["page"] == b["page"]])
            tokens, ocr = table_native(text, b["pdf_bbox"], owners, b.get("scan_like", False))
            check = verify_table(html, tokens, native_is_ocr=ocr)
            if ocr:
                ok = check.status == "single-route" and any(f.kind == "ocr_layer_agreement" for f in check.findings)
            else:
                ok = check.status in ("verified", "repaired")
            if ok:
                b.setdefault("surya_html", b["html"])
                b["html"] = check.html
                b["status"] = "single-route" if ocr else "repaired"
                b["findings"] = [{"kind": f.kind, "detail": f.detail, "values": f.values[:40]}
                                 for f in check.findings]
                b["repairs"] = [{"kind": "table", "route": "vlm_table_reread", "model_url": vlm_url,
                                 "check_status": check.status}, *check.repairs]
                b["counts"] = {"surya": check.surya_numbers, "native": check.native_numbers}
                summary["accepted"] += 1
                continue
            error = f"re-read failed the check ({check.status}: " + ", ".join(
                sorted({f.kind for f in check.findings})) + ")"
        b["findings"].append({"kind": "vlm_repair_failed", "detail": error or "", "values": []})
        summary["rejected"] += 1
    provenance = {k: v for k, v in record.items() if k not in ("item_key", "meta", "blocks")}
    provenance["repair"] = {"version": REPAIR_VERSION, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary}
    write_outputs(item_key, meta, blocks, provenance, sidecar_dir)
    return summary
