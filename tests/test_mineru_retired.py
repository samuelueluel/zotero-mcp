"""MinerU is retired: config can no longer enable it and the runner refuses."""

import json
from pathlib import Path

from zotero_mcp import mineru


def test_config_cannot_enable_mineru(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"semantic_search": {"mineru": {"enabled": True, "sidecar_dir": str(tmp_path)}}}))
    loaded = mineru.load_mineru_config(str(cfg))
    assert loaded["enabled"] is False
    assert loaded["sidecar_dir"] == str(tmp_path)


def test_run_mineru_refuses_without_running_anything(tmp_path):
    cfg = {"bin": str(tmp_path / "missing-bin"), "sidecar_dir": str(tmp_path)}
    assert mineru.run_mineru(cfg, Path(tmp_path / "x.pdf"), "ABCD1234") is False
    assert not list(tmp_path.glob("*.md"))
