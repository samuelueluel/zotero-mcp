"""Surya OCR 2 runner: upright rendering, batched OCR, raw-output retention.

Runs inside the zotero-mcp environment (PyMuPDF only). Surya itself lives in
its own venv and is invoked as a subprocess against an already-running
OpenAI-compatible llama-server (``inference_url``); this module never starts
or stops inference services.

Config (``~/.config/zotero-mcp/config.json`` -> ``semantic_search.surya``):

- surya_ocr: path to the ``surya_ocr`` CLI in the Surya venv
- inference_url: e.g. ``http://127.0.0.1:18090/v1``
- parallel: client concurrency (server ``--parallel`` should match), default 8
- dpi: render resolution, default 192
- work_dir: raw artifacts, default ``~/.cache/zotero-mcp/surya-work``
- sidecar_dir: sidecar output, default ``~/.config/zotero-mcp/surya-sidecars``
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pymupdf

from .sidecar_native import upright_rotation

RUNNER_VERSION = "surya-runner/2"

DEFAULTS = {
    "surya_ocr": str(Path.home() / "surya-spike-venv/bin/surya_ocr"),
    "inference_url": "http://127.0.0.1:18090/v1",
    "parallel": 8,
    "dpi": 192,
    "work_dir": str(Path.home() / ".cache/zotero-mcp/surya-work"),
    "sidecar_dir": str(Path.home() / ".config/zotero-mcp/surya-sidecars"),
}


def load_surya_config(config_path: str | None = None) -> dict:
    path = Path(config_path or Path.home() / ".config/zotero-mcp/config.json")
    cfg = dict(DEFAULTS)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cfg.update((raw.get("semantic_search", {}) or {}).get("surya", {}) or {})
    except (OSError, ValueError):
        pass
    return cfg


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def page_rotation(page) -> int:
    """Clockwise degrees needed to make the page's dominant text upright.

    Relative to the displayed page, which the renderer already rotates by
    ``/Rotate``; runner/1 ignored that and fed Surya sideways images of
    upright ``/Rotate 90`` pages.
    """
    return upright_rotation(page)


def scan_like(page) -> bool:
    """A page image covering most of the page: the text layer, if any, is OCR."""
    area = page.rect.width * page.rect.height
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        if (x1 - x0) * (y1 - y0) > 0.7 * area:
            return True
    return False


@dataclass
class ItemJob:
    item_key: str
    pdf_path: Path
    attachment_key: str | None = None
    #: Why this PDF was chosen among several (logged by the batch driver).
    note: str = ""
    pages: dict[str, dict] = field(default_factory=dict)  # image stem -> page meta
    pdf_sha256: str = ""


#: Longest rendered side. Some scans declare one point per scanned pixel
#: (Billingsley: 30 by 42 inch pages); at 192 dpi a page became 46 megapixels,
#: and Surya's client, which holds a chunk's images in memory, ran out of RAM.
MAX_SIDE_PX = 3000


def render_item(job: ItemJob, img_dir: Path, dpi: int) -> None:
    """Render every page upright into ``img_dir`` as ``<KEY>__pNNN.png``."""
    img_dir.mkdir(parents=True, exist_ok=True)
    job.pdf_sha256 = sha256_file(job.pdf_path)
    doc = pymupdf.open(job.pdf_path)
    for i, page in enumerate(doc):
        rot = page_rotation(page)
        zoom = min(dpi / 72, MAX_SIDE_PX / max(page.rect.width, page.rect.height))
        # prerotate(90) turns bottom-to-top text upright (clockwise render).
        matrix = pymupdf.Matrix(zoom, zoom).prerotate(rot)
        pix = page.get_pixmap(matrix=matrix)
        stem = f"{job.item_key}__p{i + 1:03d}"
        pix.save(img_dir / f"{stem}.png")
        job.pages[stem] = {
            "page": i + 1,
            "rotation": rot,
            "scan_like": scan_like(page),
            "image_size": [pix.width, pix.height],
            "page_size": [page.rect.width, page.rect.height],
        }


def run_surya(img_dir: Path, out_dir: Path, cfg: dict, log_path: Path) -> dict:
    """OCR every image in ``img_dir`` in one batched call; return results."""
    env = dict(os.environ)
    env.update({
        "SURYA_INFERENCE_BACKEND": "llamacpp",
        "SURYA_INFERENCE_URL": str(cfg["inference_url"]),
        "SURYA_INFERENCE_PARALLEL": str(cfg["parallel"]),
        "SURYA_INFERENCE_AUTOSTART": "false",
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("ab") as log:
        proc = subprocess.run(
            [str(cfg["surya_ocr"]), str(img_dir), "--output_dir", str(out_dir)],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"surya_ocr failed (rc={proc.returncode}); see {log_path}")
    results = sorted(out_dir.glob("*/results.json"))
    if not results:
        raise RuntimeError(f"surya_ocr wrote no results.json; see {log_path}")
    data = json.loads(results[-1].read_text(encoding="utf-8"))
    return {"results": data, "seconds": round(time.time() - start, 1)}


def surya_version(cfg: dict) -> str | None:
    python = Path(cfg["surya_ocr"]).parent / "python"
    try:
        out = subprocess.run(
            [str(python), "-c", "import importlib.metadata as m; print(m.version('surya-ocr'))"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance only
        return None


def run_batch(jobs: Iterable[ItemJob], cfg: dict | None = None, batch_name: str | None = None) -> Path:
    """Render and OCR several items in one Surya call; keep raw artifacts.

    Writes ``<work_dir>/<batch>/`` with the images, Surya results, and a
    ``manifest.json`` binding each item to its PDF hash and page metadata.
    Returns the batch directory. Assembly and verification happen afterwards
    from these retained artifacts.
    """
    cfg = cfg or load_surya_config()
    jobs = list(jobs)
    batch = Path(cfg["work_dir"]) / (batch_name or time.strftime("batch-%Y%m%d-%H%M%S"))
    img_dir = batch / "img"
    for job in jobs:
        render_item(job, img_dir, int(cfg["dpi"]))
    ocr = run_surya(img_dir, batch / "ocr", cfg, batch / "surya.log")
    results = ocr["results"]
    manifest = {
        "runner": RUNNER_VERSION,
        "surya_ocr_version": surya_version(cfg),
        "inference_url": cfg["inference_url"],
        "dpi": int(cfg["dpi"]),
        "ocr_seconds": ocr["seconds"],
        "items": {},
    }
    for job in jobs:
        missing = [s for s in job.pages if s not in results]
        manifest["items"][job.item_key] = {
            "pdf_path": str(job.pdf_path),
            "pdf_sha256": job.pdf_sha256,
            "attachment_key": job.attachment_key,
            "pages": job.pages,
            "missing_pages": missing,
        }
    (batch / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return batch
