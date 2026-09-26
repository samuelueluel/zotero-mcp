"""Resumable Surya sidecar batch driver for a Zotero collection or item list.

Stages, in order, with per-item state so an interrupted run resumes:

1. ``ocr``      Surya server up; render + OCR items in page-bounded chunks;
                assemble each chunk's sidecars (verify, withhold); Surya down.
2. ``repair``   Qwen VLM up; one re-read per unresolved table (sidecar_repair).
3. ``enrich``   Figure schemas + captions via ``zotero-vlm-enrich.py``; Qwen down.
                Must follow repair: repair rewrites the sidecar.
4. ``index``    ``update-db --fulltext --item-key ...`` against the given config.

``--seed-from RUN`` replaces the ``ocr`` stage: each paper's sidecar is
reassembled from that run's recorded OCR chunk (same PDF hash required), so a
verifier or assembler change can be rolled out without re-running Surya. An
item the seed run cannot supply is an error, never a fresh OCR.

Only one GPU server runs at a time, and every server this driver starts is
stopped on exit (including SIGTERM). The driver refuses the live config
(``~/.config/zotero-mcp/config.json``) unless ``--allow-live`` is given.

Usage::

    python -m zotero_mcp.surya_batch --config ~/.config/zotero-mcp-shadow/config.json \\
        --collection TRGBCDX5 --run detroit-1 [--limit 3] [--stages ocr,repair,enrich,index]
    python -m zotero_mcp.surya_batch --config ... --collection TRGBCDX5 --run detroit-2 --seed-from detroit-1
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pymupdf
import requests

from .sidecar_assemble import assemble_batch, assemble_item, atomic_write_text
from .surya_runner import ItemJob, load_surya_config, run_batch, sha256_file

STAGES = ("ocr", "repair", "enrich", "index")
LIVE_CONFIG = Path.home() / ".config/zotero-mcp/config.json"
VLM_SCRIPT = Path.home() / ".local/bin/zotero-vlm-rocm.sh"
ENRICH_SCRIPT = Path.home() / ".local/bin/zotero-vlm-enrich.py"
VLM_URL = "http://127.0.0.1:18084/v1/chat/completions"
SURYA_IMAGE = "localhost/llama-rocm-10.0-strix-llama:latest"
SURYA_CONTAINER = "zotero-surya"
SURYA_GGUF_REPO = Path.home() / ".cache/huggingface/hub/models--datalab-to--surya-ocr-2-gguf"


def log(run_dir: Path, msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with (run_dir / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- state

class State:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"items": {}}

    def item(self, key: str) -> dict:
        return self.data["items"].setdefault(key, {"done": [], "errors": []})

    def done(self, key: str, stage: str) -> bool:
        return stage in self.item(key)["done"]

    def mark(self, key: str, stage: str, **extra) -> None:
        it = self.item(key)
        if stage not in it["done"]:
            it["done"].append(stage)
        it.update(extra)
        self.save()

    def error(self, key: str, stage: str, msg: str) -> None:
        self.item(key)["errors"].append({"stage": stage, "msg": msg[:500], "at": time.strftime("%F %T")})
        self.save()

    def save(self) -> None:
        atomic_write_text(self.path, json.dumps(self.data, indent=1))


# ---------------------------------------------------------------- servers

def _healthy(url: str) -> bool:
    try:
        return requests.get(url, timeout=3).ok
    except requests.RequestException:
        return False


def start_surya(cfg: dict) -> None:
    models = cfg["inference_url"].rstrip("/") + "/models"
    if _healthy(models):
        return
    snaps = sorted((SURYA_GGUF_REPO / "snapshots").glob("*/surya-2.gguf"))
    if not snaps:
        raise RuntimeError(f"Surya GGUF not found under {SURYA_GGUF_REPO}")
    snap = snaps[-1].parent
    port = cfg["inference_url"].rsplit(":", 1)[1].split("/")[0]
    parallel = int(cfg["parallel"])
    subprocess.run(["podman", "rm", "-f", SURYA_CONTAINER], capture_output=True)
    proc = subprocess.run([
        "podman", "run", "-d", "--pull=never", "--name", SURYA_CONTAINER,
        "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "keep-groups",
        "--security-opt", "label=disable", "--log-driver=journald", "--log-opt", f"tag={SURYA_CONTAINER}",
        "-p", f"127.0.0.1:{port}:{port}",
        "-v", f"{(snap / 'surya-2.gguf').resolve()}:/mnt/model.gguf:ro",
        "-v", f"{(snap / 'surya-2-mmproj.gguf').resolve()}:/mnt/mmproj.gguf:ro",
        SURYA_IMAGE, "/usr/local/bin/llama-server", "--host", "0.0.0.0", "--port", port,
        "-m", "/mnt/model.gguf", "--mmproj", "/mnt/mmproj.gguf", "-ngl", "99",
        "--parallel", str(parallel), "--ctx-size", str(12288 * parallel),
        "--alias", "datalab-to/surya-ocr-2", "--jinja",
    ], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"podman run {SURYA_CONTAINER} failed: {proc.stderr.strip()[-400:]}")
    for _ in range(90):
        if _healthy(models):
            return
        time.sleep(2)
    raise RuntimeError(f"Surya server did not become ready; see `podman logs {SURYA_CONTAINER}`")


def stop_surya() -> None:
    subprocess.run(["podman", "stop", "-t", "10", SURYA_CONTAINER], capture_output=True)
    subprocess.run(["podman", "rm", "-f", SURYA_CONTAINER], capture_output=True)


def start_vlm() -> None:
    if _healthy(VLM_URL.removesuffix("chat/completions") + "models"):
        return
    # A hard stop (power loss, kill -9) leaves the dedicated container behind,
    # and the start script refuses to replace an existing one.
    stop_vlm()
    proc = subprocess.run([str(VLM_SCRIPT), "start"], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{VLM_SCRIPT.name} start failed: {(proc.stderr or proc.stdout).strip()[-400:]}")


def stop_vlm() -> None:
    subprocess.run([str(VLM_SCRIPT), "stop"], capture_output=True)


# ---------------------------------------------------------------- items

def resolve_items(config_path: Path, collection: str | None, keys: list[str]) -> tuple[list[ItemJob], dict]:
    """Return one ItemJob per parent with exactly one resolvable PDF, plus skip reasons."""
    from .local_db import LocalZoteroReader

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    db_path = raw.get("semantic_search", {}).get("zotero_db_path")
    jobs, skipped = [], {}
    with LocalZoteroReader(db_path=db_path) as reader:
        if collection:
            keys = sorted(reader.resolve_collection_item_keys(collection))
            if not keys:
                raise SystemExit(f"collection {collection}: no parent items")
        for key in keys:
            pdfs = []
            for att in reader.get_attachment_paths(key):
                rp = att.get("resolved_path")
                is_pdf = str(att.get("content_type") or "").lower() == "application/pdf" \
                    or str(rp or "").lower().endswith(".pdf")
                if is_pdf and rp and Path(rp).is_file():
                    pdfs.append((str(att.get("key") or ""), Path(rp)))
            if len(pdfs) != 1:
                skipped[key] = f"{len(pdfs)} resolvable PDFs"
                continue
            jobs.append(ItemJob(key, pdfs[0][1], attachment_key=pdfs[0][0]))
    return jobs, skipped


def chunk_jobs(jobs: list[ItemJob], pages: dict[str, int], max_pages: int) -> list[list[ItemJob]]:
    """Group items so each Surya call covers at most ``max_pages`` (an oversized item runs alone)."""
    chunks, cur, n = [], [], 0
    for job in jobs:
        p = pages[job.item_key]
        if cur and n + p > max_pages:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(job)
        n += p
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- integrity

def sidecar_consistent(key: str, sidecar_dir: Path) -> bool:
    """True if a paper's sidecar files agree with each other.

    Catches a repair write cut off by a stop: blocks.json must parse, its table
    statuses must match reliability.json, and until enrichment starts the
    Markdown must hash to the recorded sidecar_sha256. Enrichment edits the
    Markdown in place (atomically), so a sidecar with schema blocks skips the hash.
    """
    try:
        blocks = json.loads((sidecar_dir / f"{key}.blocks.json").read_text(encoding="utf-8"))["blocks"]
        rel = json.loads((sidecar_dir / f"{key}.reliability.json").read_text(encoding="utf-8"))
        md = (sidecar_dir / f"{key}.md").read_text(encoding="utf-8")
        tables = Counter(b["status"] for b in blocks if b["label"] == "Table")
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if dict(tables) != rel.get("tables"):
        return False
    if "[Figure Schema]" in md:
        return True
    return hashlib.sha256(md.encode("utf-8")).hexdigest() == rel.get("sidecar_sha256")


def reassemble(key: str, batch: Path, sidecar_dir: Path) -> dict:
    """Rebuild one paper's sidecar from its recorded OCR chunk (the pre-repair state)."""
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads(next(iter(sorted((batch / "ocr").glob("*/results.json")))).read_text(encoding="utf-8"))
    return assemble_item(key, manifest["items"][key], results, sidecar_dir, batch)


def heal_sidecars(jobs, state, run_dir, sidecar_dir) -> None:
    """Before repair and enrichment, rebuild any sidecar left inconsistent by a stop."""
    for j in jobs:
        k = j.item_key
        if not state.done(k, "ocr") or state.done(k, "enrich") or sidecar_consistent(k, sidecar_dir):
            continue
        it = state.item(k)
        try:
            rel = reassemble(k, Path(it["batch"]), sidecar_dir)
        except Exception as exc:  # noqa: BLE001 - fall back to a fresh OCR of this paper
            it["done"] = []
            state.error(k, "heal", f"{type(exc).__name__}: {exc}")
            log(run_dir, f"sidecar {k} inconsistent and not rebuildable ({exc}); OCR will be redone")
            continue
        it["done"] = ["ocr"]
        state.mark(k, "ocr", level=rel["level"], tables=rel["tables"])
        log(run_dir, f"sidecar {k} inconsistent (interrupted write); rebuilt from its OCR chunk, repair will be redone")


# ---------------------------------------------------------------- stages

def stage_seed(jobs, state, run_dir, sidecar_dir, seed_dir: Path) -> None:
    """Reassemble sidecars from another run's OCR instead of running Surya."""
    seed = State(seed_dir / "state.json")
    todo = [j for j in jobs if not state.done(j.item_key, "ocr")]
    if not todo:
        return
    log(run_dir, f"seed: reassembling {len(todo)} items from {seed_dir.name}")
    for j in todo:
        k = j.item_key
        src = seed.data["items"].get(k, {})
        batch = src.get("batch")
        if not batch or "ocr" not in src.get("done", []):
            state.error(k, "ocr", f"seed run {seed_dir.name} has no OCR for this item")
            log(run_dir, f"seed {k}: no OCR in {seed_dir.name}; skipped")
            continue
        if src.get("pdf_sha256") != state.item(k).get("pdf_sha256"):
            state.error(k, "ocr", f"PDF changed since seed run {seed_dir.name}")
            log(run_dir, f"seed {k}: PDF changed since {seed_dir.name}; skipped")
            continue
        try:
            rel = reassemble(k, Path(batch), sidecar_dir)
        except Exception as exc:  # noqa: BLE001 - record and continue
            state.error(k, "ocr", f"{type(exc).__name__}: {exc}")
            log(run_dir, f"seed {k} FAILED: {exc}")
            continue
        state.mark(k, "ocr", batch=batch, seeded_from=seed_dir.name, level=rel["level"], tables=rel["tables"])
    tables = Counter()
    for j in todo:
        tables.update(state.item(j.item_key).get("tables") or {})
    log(run_dir, f"seed done: tables={dict(tables)}")


def stage_ocr(jobs, state, cfg, run_dir, sidecar_dir, max_pages) -> None:
    todo = [j for j in jobs if not state.done(j.item_key, "ocr")]
    if not todo:
        return
    pages = {}
    for j in todo:
        with pymupdf.open(j.pdf_path) as doc:
            pages[j.item_key] = doc.page_count
    chunks = chunk_jobs(todo, pages, max_pages)
    log(run_dir, f"ocr: {len(todo)} items, {sum(pages.values())} pages, {len(chunks)} chunks")
    start_surya(cfg)
    try:
        for i, chunk in enumerate(chunks, 1):
            name = f"chunk-{time.strftime('%Y%m%d-%H%M%S')}-{i:03d}"
            keys = [j.item_key for j in chunk]
            t = time.time()
            try:
                batch = run_batch(chunk, cfg, name)
                rel = assemble_batch(batch, sidecar_dir)
            except Exception as exc:  # noqa: BLE001 - record and continue with the next chunk
                for k in keys:
                    state.error(k, "ocr", f"{type(exc).__name__}: {exc}")
                log(run_dir, f"ocr chunk {i}/{len(chunks)} FAILED {keys}: {exc}")
                if not _healthy(cfg["inference_url"].rstrip("/") + "/models"):
                    log(run_dir, "Surya server unhealthy; restarting")
                    stop_surya()
                    start_surya(cfg)
                continue
            for k in keys:
                state.mark(k, "ocr", batch=str(batch), level=rel[k]["level"], tables=rel[k]["tables"])
            log(run_dir, f"ocr chunk {i}/{len(chunks)} ok: {sum(pages[k] for k in keys)} pages "
                         f"in {time.time() - t:.0f}s; " +
                ", ".join(f"{k}={rel[k]['level']}" for k in keys))
    finally:
        stop_surya()
        log(run_dir, "Surya server stopped")


def stage_vlm(jobs, state, run_dir, sidecar_dir, stages) -> None:
    from .sidecar_repair import repair_item

    ready = [j.item_key for j in jobs if state.done(j.item_key, "ocr")]
    need = [k for k in ready if any(s in stages and not state.done(k, s) for s in ("repair", "enrich"))]
    if not need:
        return
    log(run_dir, f"vlm: {len(need)} items")
    start_vlm()
    env = dict(os.environ, ZOTERO_SIDECAR_DIR=str(sidecar_dir), ZOTERO_VLM_URL=VLM_URL)
    try:
        for k in need:
            if "repair" in stages and not state.done(k, "repair"):
                try:
                    out = repair_item(k, sidecar_dir, VLM_URL)
                    rel = json.loads((sidecar_dir / f"{k}.reliability.json").read_text(encoding="utf-8"))
                    state.mark(k, "repair", repair=out, level=rel["level"], tables=rel["tables"])
                    if out["attempted"]:
                        log(run_dir, f"repair {k}: {out['accepted']}/{out['attempted']} accepted -> {rel['level']}")
                    if out.get("second_read"):
                        log(run_dir, f"second read {k}: {out['agreed']}/{out['second_read']} scan tables agree")
                except Exception as exc:  # noqa: BLE001
                    state.error(k, "repair", f"{type(exc).__name__}: {exc}")
                    log(run_dir, f"repair {k} FAILED: {exc}")
                    continue
            if "enrich" in stages and not state.done(k, "enrich"):
                if not state.done(k, "repair"):
                    continue  # enrichment would be overwritten by a later repair
                proc = subprocess.run([sys.executable, str(ENRICH_SCRIPT), "--key", k],
                                      env=env, capture_output=True, text=True)
                line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(f"[{k}]")), "")
                if proc.returncode == 0:
                    state.mark(k, "enrich", enrich=line)
                else:
                    state.error(k, "enrich", (line or proc.stderr)[-500:])
                    log(run_dir, f"enrich {k} FAILED: {line or proc.stderr[-200:]}")
    finally:
        stop_vlm()
        log(run_dir, "VLM server stopped")


def stage_index(jobs, state, run_dir, config_path, group: int = 10) -> None:
    ready = [j.item_key for j in jobs
             if not state.done(j.item_key, "index")
             and all(state.done(j.item_key, s) for s in ("ocr", "repair", "enrich"))]
    if not ready:
        return
    exe = Path(sys.executable).parent / "zotero-mcp-server"
    env = dict(os.environ, ZOTERO_LOCAL="true", ZOTERO_MCP_CONFIG=str(config_path))
    for i in range(0, len(ready), group):
        keys = ready[i:i + group]
        cmd = [str(exe), "update-db", "--fulltext", "--no-batch", "--config-path", str(config_path)]
        for k in keys:
            cmd += ["--item-key", k]
        t = time.time()
        with (run_dir / "index.log").open("a", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            for k in keys:
                state.mark(k, "index")
            log(run_dir, f"indexed {len(keys)} items in {time.time() - t:.0f}s")
        else:
            for k in keys:
                state.error(k, "index", f"update-db rc={proc.returncode}; see index.log")
            log(run_dir, f"index group FAILED rc={proc.returncode}: {keys}")


# ---------------------------------------------------------------- main

def reset_if_pdf_changed(jobs, state) -> None:
    for j in jobs:
        it = state.item(j.item_key)
        sha = sha256_file(j.pdf_path)
        if it.get("pdf_sha256") and it["pdf_sha256"] != sha:
            it["done"] = []
        it.update(pdf_sha256=sha, pdf_path=str(j.pdf_path), attachment_key=j.attachment_key)
    state.save()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--collection")
    g.add_argument("--item", action="append", dest="items")
    ap.add_argument("--run", required=True, help="run name; reuse it to resume")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--sidecar-dir", type=Path, help="default: config semantic_search.mineru.sidecar_dir")
    ap.add_argument("--chunk-pages", type=int, default=160)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--allow-live", action="store_true")
    ap.add_argument("--seed-from", metavar="RUN", help="reassemble from this run's OCR instead of running Surya")
    args = ap.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    if config_path == LIVE_CONFIG.resolve() and not args.allow_live:
        raise SystemExit("refusing the live config; pass a shadow config (or --allow-live after cutover approval)")
    stages = tuple(s for s in args.stages.split(",") if s)
    if bad := set(stages) - set(STAGES):
        raise SystemExit(f"unknown stages: {sorted(bad)}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    sidecar_dir = (args.sidecar_dir or Path(raw["semantic_search"]["mineru"]["sidecar_dir"])).expanduser()
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_surya_config(str(config_path))
    run_dir = Path(cfg["work_dir"]).expanduser() / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["work_dir"] = str(run_dir / "ocr")
    seed_dir = None
    if args.seed_from:
        seed_dir = run_dir.parent / args.seed_from
        if seed_dir == run_dir or not (seed_dir / "state.json").is_file():
            raise SystemExit(f"--seed-from {args.seed_from}: no such other run under {run_dir.parent}")

    lock = (run_dir / "lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"run {args.run} is already in progress")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run finally blocks: stop servers

    jobs, skipped = resolve_items(config_path, args.collection, args.items or [])
    if args.limit:
        jobs = jobs[:args.limit]
    state = State(run_dir / "state.json")
    state.data.update(config=str(config_path), sidecar_dir=str(sidecar_dir),
                      scope=args.collection or args.items, skipped=skipped)
    reset_if_pdf_changed(jobs, state)
    log(run_dir, f"run {args.run}: {len(jobs)} items, {len(skipped)} skipped, stages={','.join(stages)}")
    for k, why in skipped.items():
        log(run_dir, f"skip {k}: {why}")

    t = time.time()
    try:
        if {"repair", "enrich"} & set(stages):
            heal_sidecars(jobs, state, run_dir, sidecar_dir)
        if "ocr" in stages and seed_dir is not None:
            stage_seed(jobs, state, run_dir, sidecar_dir, seed_dir)
        elif "ocr" in stages:
            stage_ocr(jobs, state, cfg, run_dir, sidecar_dir, args.chunk_pages)
        if {"repair", "enrich"} & set(stages):
            stage_vlm(jobs, state, run_dir, sidecar_dir, stages)
        if "index" in stages:
            stage_index(jobs, state, run_dir, config_path)
    except (SystemExit, KeyboardInterrupt):
        log(run_dir, f"stopped before completion; rerun with --run {args.run} to resume")
        raise
    except Exception as exc:
        log(run_dir, f"ABORTED: {type(exc).__name__}: {exc}; rerun with --run {args.run} after fixing")
        raise

    levels: dict[str, int] = {}
    for j in jobs:
        lv = state.item(j.item_key).get("level", "not-ocr")
        levels[lv] = levels.get(lv, 0) + 1
    complete = sum(all(state.done(j.item_key, s) for s in stages) for j in jobs)
    log(run_dir, f"done in {(time.time() - t) / 60:.1f} min: {complete}/{len(jobs)} complete; levels={levels}")
    return 0 if complete == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
