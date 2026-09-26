"""Second-model repair pass for unresolved Surya tables.

Runs after :mod:`sidecar_assemble` and before figure enrichment (it rewrites
the sidecar). Each ``unresolved`` table is cropped upright from the PDF and
re-read once by a general VLM (Qwen3-VL on an OpenAI-compatible endpoint).
The re-read is accepted only if it passes the same :func:`verify_table`
check against the native text layer, prepared exactly as in assembly
(:func:`sidecar_assemble.table_native`):

- born-digital text: must come back ``verified`` or ``repaired``, which
  includes the position and label checks;
- OCR text layer (scanned page or invisible text over an image):
  ``single-route`` via OCR-layer agreement, never via an unusable layer;
- significance stars must match Surya's, unless the text layer's star count
  backs the re-read (:func:`star_disagreement`).

An accepted re-read is ``single-route`` with a ``vlm_rewrite`` finding, never
``repaired``: its numbers passed the check, but a model that rewrote the
whole table can attach them to the wrong labels (a rowspan one row off) in
ways the text layer cannot catch, so the page must be looked at before use.

Anything else leaves the block ``unresolved`` with its numbers withheld and a
``vlm_repair_failed`` finding. No second attempt, no free-form correction.

Second reading: each ``single-route`` table on a scanned page or OCR text
layer is also read by the VLM. If both readings hold the same numbers column
by column (:func:`sidecar_verify.readings_agree`), the table keeps its status
and gains a ``vlm_agreement`` finding; otherwise nothing visible changes.
The model's reading is never used as the table text.
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
from .sidecar_assemble import PageText, neighbour_html, owner_keys, table_native, write_outputs
from .sidecar_verify import readings_agree, verify_table

REPAIR_VERSION = "sidecar-repair/4"
_STAR_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.S | re.I)
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


def _stars(html: str) -> int:
    """Significance stars in the table's cells."""
    return sum(re.sub(r"<[^>]+>", "", c).count("*") for c in _STAR_RE.findall(html))


def star_disagreement(surya_html: str, reread_html: str, printed_text: str) -> str | None:
    """Why the re-read's stars cannot be trusted, or None.

    Stars are not numbers, so the numeric check never sees them. A re-read
    that drops a star column or turns ``***`` into ``*`` still passes it.
    A different star count is accepted only when the text layer prints the
    re-read's count; many fonts draw stars as glyphs the layer omits (a
    count of zero), and then nothing settles it.
    """
    before, after = _stars(surya_html), _stars(reread_html)
    printed = printed_text.count("*")
    if before == after or (printed and printed == after):
        return None
    return f"stars: Surya {before}, re-read {after}, text layer {printed_text.count('*')}"


def repair_item(item_key: str, sidecar_dir: Path, vlm_url: str) -> dict:
    sidecar_dir = Path(sidecar_dir)
    record = json.loads((sidecar_dir / f"{item_key}.blocks.json").read_text(encoding="utf-8"))
    meta, blocks = record["meta"], record["blocks"]
    targets = [b for b in blocks if b["label"] == "Table" and b["status"] == "unresolved"]
    second = [b for b in blocks if b["label"] == "Table" and b["status"] == "single-route"
              and (b.get("scan_like") or b.get("ocr_layer"))
              and not any(f["kind"] in ("vlm_agreement", "vlm_rewrite") for f in b["findings"])
              and "vlm_second_read" not in b]
    summary = {"item_key": item_key, "attempted": len(targets), "accepted": 0, "rejected": 0,
               "second_read": len(second), "agreed": 0}
    if not targets and not second:
        return summary
    doc = pymupdf.open(record["pdf_path"])
    profile = glyph_profile(doc)
    texts: dict[int, PageText] = {}
    for b in second:
        try:
            html = ask_vlm(crop_png(doc[b["page"] - 1], b["pdf_bbox"]), vlm_url)
        except Exception as exc:  # noqa: BLE001 - recorded, block keeps its status
            html = None
            b["vlm_second_read"] = {"error": str(exc)[:200]}
        if html is None:
            b.setdefault("vlm_second_read", {"error": "no <table> in model output"})
        elif readings_agree(b["html"], html):
            b["findings"].append({
                "kind": "vlm_agreement",
                "detail": "a second model read the page image independently and agrees on every number, "
                          "column by column; no PDF text layer confirms them",
                "values": []})
            b["vlm_second_read"] = {"agree": True, "model_url": vlm_url}
            summary["agreed"] += 1
        else:
            b["vlm_second_read"] = {"agree": False, "model_url": vlm_url, "html": html}  # audits only
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
            page_blocks = [x for x in blocks if x["page"] == b["page"]]
            owners = owner_keys(page_blocks)
            tokens, ocr = table_native(text, b["pdf_bbox"], owners, b.get("scan_like", False))
            printed = None if ocr else text.words(b["pdf_bbox"])
            check = verify_table(html, tokens, native_is_ocr=ocr, printed_words=printed, fill=False,
                                 context_html=neighbour_html(
                                     b["pdf_bbox"], [(x["html"], x["pdf_bbox"], x["label"]) for x in page_blocks if x is not b]))
            if ocr:
                ok = check.status == "single-route" and any(f.kind == "ocr_layer_agreement" for f in check.findings)
            else:
                ok = check.status in ("verified", "repaired")
            stars = star_disagreement(b.get("surya_html", b["html"]), check.html, " ".join(printed or ()))
            if ok and not stars:
                b.setdefault("surya_html", b["html"])
                b["html"] = check.html
                b["status"] = "single-route"
                b["findings"] = [{"kind": f.kind, "detail": f.detail, "values": f.values[:40]}
                                 for f in check.findings]
                b["findings"].append({
                    "kind": "vlm_rewrite",
                    "detail": "table re-read by a second model; numbers match the text layer, "
                              "row and column labels unchecked",
                    "values": []})
                b["repairs"] = [{"kind": "table", "route": "vlm_table_reread", "model_url": vlm_url,
                                 "check_status": check.status}, *check.repairs]
                b["counts"] = {"surya": check.surya_numbers, "native": check.native_numbers}
                summary["accepted"] += 1
                continue
            error = "re-read failed the check (" + (stars if ok else f"{check.status}: " + ", ".join(
                sorted({f.kind for f in check.findings}))) + ")"
            # Kept for audits; never rendered.
            b["vlm_rejected"] = {"html": check.html, "findings": [
                {"kind": f.kind, "detail": f.detail, "values": f.values[:40]} for f in check.findings]}
        b["findings"].append({"kind": "vlm_repair_failed", "detail": error or "", "values": []})
        summary["rejected"] += 1
    provenance = {k: v for k, v in record.items() if k not in ("item_key", "meta", "blocks")}
    provenance["repair"] = {"version": REPAIR_VERSION, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary}
    write_outputs(item_key, meta, blocks, provenance, sidecar_dir)
    return summary
