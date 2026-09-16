"""Auto-MinerU integration for zotero-mcp.  # [mineru patch]

Parses item PDFs with MinerU (magic-pdf) BEFORE embedding so equations come
out as LaTeX and tables as HTML instead of garbled Unicode. Sidecars are
cached per item at ``<sidecar_dir>/<item_key>.md`` and reused; the parse
runs once per item (policy: always for PDFs lacking a sidecar).

Hook points:
- ``semantic_search.py`` extraction block: ``try_auto_parse()`` runs before
  text-layer extraction for items being (re)embedded.
- ``tools/retrieval.py`` ``get_item_fulltext``: ``read_sidecar()`` is
  preferred over text-layer extraction so answer-time reads show the clean
  MinerU text.

Config (``~/.config/zotero-mcp/config.json`` -> ``semantic_search.mineru``):
- enabled: bool (default false; flip true once verified)
- bin: MinerU CLI binary (default ~/mineru-upgrade-venv/bin/mineru)
- config_json: optional CLI config path exported as MINERU_TOOLS_CONFIG_JSON
- sidecar_dir: sidecar cache dir (default ~/.config/zotero-mcp/mineru-sidecars)
- work_dir: per-item magic-pdf work dir (default ~/.cache/zotero-mcp/mineru-work)
- timeout_seconds: per-parse cap (default 3600)

Re-applied idempotently by zotero-mcp-mineru-patch.py via ``sjust update``;
see General-Tooling and MinerU-Setup.md. Requires the mineru-rocm-venv
(mineru setup memory note) and the ch-family models_config.yml rewire.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("zotero_mcp.mineru")

MARKER = "[mineru patch]"


def load_mineru_config(config_path: str | None = None) -> dict:
    """Read the ``semantic_search.mineru`` section from config.json (raw JSON).

    Reads the file directly rather than ``load_config()`` so an unknown
    ``mineru`` key never trips pydantic validation, and so defaults resolve
    even if the server's typed config is loaded with extra-ignore.
    """
    cfg_path = Path(
        config_path
        or os.environ.get(
            "ZOTERO_MCP_CONFIG",
            str(Path.home() / ".config" / "zotero-mcp" / "config.json"),
        )
    )
    cfg: dict = {}
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = (raw or {}).get("semantic_search", {}).get("mineru", {}) or {}
    except Exception as e:  # missing file / parse error -> defaults
        logger.debug("mineru: config read failed: %s", e)
    defaults = {
        "enabled": False,
        # MinerU 3.4.5 (upgrade 2026-08-19): CLI renamed magic-pdf -> mineru.
        # Explicit configs may still select the compatible 1.x magic-pdf fallback;
        # run_mineru detects the CLI capability before adding version-specific flags.
        "bin": str(Path.home() / "mineru-upgrade-venv/bin/mineru"),
        "config_json": None,
        "sidecar_dir": str(Path.home() / ".config" / "zotero-mcp" / "mineru-sidecars"),
        "work_dir": str(Path.home() / ".cache" / "zotero-mcp" / "mineru-work"),
        "timeout_seconds": 3600,
        # GTT balloon guard (MinerU 3.x env, renamed from 1.x VIRTUAL_VRAM_SIZE).
        # Unset -> MinerU reads real GPU mem (124 GB) -> batch_ratio 16 (fastest;
        # GTT stayed ~7 GB across dense manual windows; sidecar-watch backstops
        # genuine balloons). Set 4 for batch_ratio 1 (conservative).
        "virtual_vram_size": None,
        "backend": "pipeline",
    }
    merged = dict(defaults)
    merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def sidecar_path(cfg: dict, item_key: str) -> Path:
    return Path(cfg["sidecar_dir"]) / f"{item_key}.md"


def read_sidecar(cfg: dict, item_key: str) -> str | None:
    """Return the cached MinerU markdown for an item, or None."""
    if not item_key:
        return None
    p = sidecar_path(cfg, item_key)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("mineru: sidecar read failed for %s: %s", item_key, e)
        return None


def _find_output_md(out_dir: Path) -> Path | None:
    """mineru writes <out>/<stem>/txt/<stem>.md — locate it (same layout as 1.x)."""
    try:
        for md in sorted(out_dir.glob("*/txt/*.md")):
            return md
    except OSError:
        pass
    return None


_CLI_BACKEND_CAPABILITY_CACHE: dict[str, bool] = {}


def _supports_backend_flag(bin_: Path, env: dict[str, str]) -> bool:
    """Detect whether the selected MinerU CLI accepts ``-b/--backend``.

    MinerU 3.x (``mineru``) exposes the backend option; the retained 1.x
    ``magic-pdf`` CLI does not.  The configured binary is authoritative, not
    its filename or the package version installed in another virtualenv.
    """
    cache_key = str(bin_.resolve())
    cached = _CLI_BACKEND_CAPABILITY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    supported = False
    try:
        probe = subprocess.run(
            [str(bin_), "--help"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
        supported = "--backend" in (probe.stdout or "")
    except Exception as e:
        logger.warning("mineru: CLI capability probe failed for %s: %s", bin_, e)
    _CLI_BACKEND_CAPABILITY_CACHE[cache_key] = supported
    return supported


def _build_mineru_invocation(
    cfg: dict, pdf_path: Path, out_dir: Path
) -> tuple[list[str], dict[str, str], bool]:
    """Build a version-compatible command and environment for MinerU."""
    bin_ = Path(cfg["bin"])
    env = dict(os.environ)

    # ``config_json`` was part of the original magic-pdf integration.  Keep it
    # authoritative for both generations instead of silently relying on a
    # cwd-dependent ``~/magic-pdf.json`` default.
    config_json = cfg.get("config_json")
    if config_json:
        env["MINERU_TOOLS_CONFIG_JSON"] = str(Path(str(config_json)).expanduser())

    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    supports_backend = _supports_backend_flag(bin_, env)

    # MinerU 3.x renamed the guard variable.  Preserve the conservative 1.x
    # fallback that the pre-upgrade integration used when no new setting was
    # explicitly supplied.
    vvs = cfg.get("virtual_vram_size")
    if vvs is not None:
        env["MINERU_VIRTUAL_VRAM_SIZE" if supports_backend else "VIRTUAL_VRAM_SIZE"] = str(vvs)
    elif not supports_backend:
        env.setdefault("VIRTUAL_VRAM_SIZE", "4")

    cmd = [
        str(bin_),
        "-p", str(pdf_path),
        "-o", str(out_dir),
        "-m", "txt",
    ]
    if supports_backend and cfg.get("backend"):
        cmd.extend(["-b", str(cfg["backend"])])
    return cmd, env, supports_backend


def run_mineru(cfg: dict, pdf_path: Path, item_key: str) -> bool:
    """Run the configured MinerU CLI and copy its Markdown to the sidecar.

    The runner supports both MinerU 3.x (``mineru -b pipeline``) and the
    retained 1.x ``magic-pdf`` fallback (no ``-b``).  MinerU CLIs may exit 0
    without producing output, so success requires both a zero exit code and a
    generated Markdown file.
    """
    bin_ = Path(cfg["bin"])
    if not bin_.exists():
        logger.warning("mineru: binary not found: %s", bin_)
        return False
    work = Path(cfg["work_dir"]) / item_key
    work.mkdir(parents=True, exist_ok=True)
    out_dir = work / "out"
    log_path = work / "run.log"
    cmd, env, supports_backend = _build_mineru_invocation(cfg, pdf_path, out_dir)
    logger.info(
        "mineru: using %s CLI (backend_flag=%s, config=%s)",
        bin_, supports_backend, env.get("MINERU_TOOLS_CONFIG_JSON", "default"),
    )
    try:
        start = time.time()
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=int(cfg.get("timeout_seconds", 3600)),
            )
        elapsed = time.time() - start
        md = _find_output_md(out_dir)
        if proc.returncode == 0 and md is not None:
            side = sidecar_path(cfg, item_key)
            side.parent.mkdir(parents=True, exist_ok=True)
            side.write_text(md.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            logger.info(
                "mineru: parsed %s in %.0fs -> %s (%.1f KB)",
                item_key, elapsed, side, side.stat().st_size / 1024,
            )
            return True
        logger.warning(
            "mineru: parse failed for %s (rc=%s, %.0fs); log: %s",
            item_key, proc.returncode, elapsed, log_path,
        )
        return False
    except Exception as e:
        logger.warning("mineru: parse raised for %s: %s", item_key, e)
        return False


def try_auto_parse(item_key: str, reader, config_path: str | None = None) -> tuple[str, str] | None:
    """Return ``(fulltext, source)`` for an item, parsing with MinerU if needed.

    Fast path: sidecar already exists (covers manual parses and prior runs).
    Otherwise, if ``mineru.enabled`` and the item has a resolvable PDF, run
    magic-pdf, write the sidecar, and return its text. Returns None when no
    sidecar exists, MinerU is disabled, or the parse fails — the caller then
    falls back to text-layer extraction. Never raises.
    """
    if not item_key:
        return None
    cfg = load_mineru_config(config_path)

    text = read_sidecar(cfg, item_key)
    if text is not None:
        return text, "mineru-sidecar"

    if not cfg.get("enabled"):
        return None

    try:
        attachments = reader.get_attachment_paths(item_key)
    except Exception as e:
        logger.warning("mineru: attachment resolution failed for %s: %s", item_key, e)
        return None

    pdf: Path | None = None
    for att in attachments:
        rp = att.get("resolved_path")
        if rp and str(rp).lower().endswith(".pdf") and Path(rp).exists():
            pdf = Path(rp)
            break
    if pdf is None:
        return None

    if run_mineru(cfg, pdf, item_key):
        text = read_sidecar(cfg, item_key)
        if text is not None:
            return text, "mineru"
    return None


def _has_resolvable_pdf(item_key: str, reader) -> bool:
    """True when the item has a resolvable PDF attachment on disk."""
    try:
        for att in reader.get_attachment_paths(item_key):
            rp = att.get("resolved_path")
            if rp and str(rp).lower().endswith(".pdf") and Path(rp).exists():
                return True
    except Exception as e:
        logger.debug("mineru: attachment lookup failed for %s: %s", item_key, e)
    return False


def is_parseable(item_key: str, reader, config_path: str | None = None) -> bool:
    """True when MinerU is enabled and could serve this item: a cached
    sidecar exists, or the item has a resolvable PDF. Used by the update
    loop to decide whether a previously-failed item deserves a retry."""
    if not item_key:
        return False
    cfg = load_mineru_config(config_path)
    if not cfg.get("enabled"):
        return False
    if read_sidecar(cfg, item_key) is not None:
        return True
    return _has_resolvable_pdf(item_key, reader)


def is_backfill_target(item_key: str, reader, config_path: str | None = None) -> bool:
    """True when the item should be re-extracted as part of the MinerU
    backfill: mineru.enabled AND mineru.backfill AND (sidecar cached OR a
    resolvable PDF exists). This turns the first post-enable update into a
    one-time library parse; items without PDFs are never targets."""
    if not item_key:
        return False
    cfg = load_mineru_config(config_path)
    if not (cfg.get("enabled") and cfg.get("backfill")):
        return False
    if read_sidecar(cfg, item_key) is not None:
        return True
    return _has_resolvable_pdf(item_key, reader)
