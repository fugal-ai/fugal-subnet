"""scripts/set_v5_activation_block.py on a copy of the files it writes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "set_v5_activation_block", ROOT / "scripts" / "set_v5_activation_block.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "fugal_subnet").mkdir()
    (tmp_path / "deploy" / "dstack").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    for rel in ("fugal_subnet/config.py", "deploy/dstack/docker-compose.yaml"):
        (tmp_path / rel).write_text((ROOT / rel).read_text(encoding="utf-8"), encoding="utf-8")
    docs = []
    for rel in ("CHANGELOG.md", "docs/MINER_GUIDE.md", "docs/VALIDATOR_GUIDE.md"):
        p = tmp_path / rel
        p.write_text("v5 at block TBD-block (epoch TBD-epoch).\n", encoding="utf-8")
        docs.append(p)
    mod.ROOT = tmp_path
    mod.CONFIG = tmp_path / "fugal_subnet" / "config.py"
    mod.COMPOSE = tmp_path / "deploy" / "dstack" / "docker-compose.yaml"
    mod.DOCS = tuple(docs)
    return mod


def test_set_writes_config_compose_and_docs_and_check_agrees(tmp_path, capsys):
    mod = _mod(tmp_path)
    v4 = mod._compiled("FINNEY_V4_ACTIVATION_BLOCK", mod.CONFIG.read_text(encoding="utf-8"))
    block = (v4 // 360 + 100) * 360
    assert mod.main([str(block)]) == 0
    assert f"FINNEY_V5_ACTIVATION_BLOCK: int | None = {block}" in mod.CONFIG.read_text()
    assert f"- FUGAL_V5_ACTIVATION_BLOCK={block}" in mod.COMPOSE.read_text()
    for p in mod.DOCS:
        assert p.read_text() == f"v5 at block {block:,} (epoch e{block // 360:08d}).\n"
    # moving it moves every form, and the compose line is not added twice
    new = block + 360
    assert mod.main([str(new)]) == 0
    assert mod.COMPOSE.read_text().count("FUGAL_V5_ACTIVATION_BLOCK=") == 1
    for p in mod.DOCS:
        assert p.read_text() == f"v5 at block {new:,} (epoch e{new // 360:08d}).\n"
    assert mod.check() == 0


@pytest.mark.parametrize("bad", ["not_boundary", "before_v4"])
def test_set_refuses_a_block_that_is_not_a_later_boundary(tmp_path, bad):
    mod = _mod(tmp_path)
    v4 = mod._compiled("FINNEY_V4_ACTIVATION_BLOCK", mod.CONFIG.read_text(encoding="utf-8"))
    block = v4 + 361 if bad == "not_boundary" else v4
    with pytest.raises(SystemExit):
        mod.set_block(block)
    assert "FINNEY_V5_ACTIVATION_BLOCK: int | None = None" in mod.CONFIG.read_text()


def test_the_shipped_tree_check_runs():
    spec = importlib.util.spec_from_file_location(
        "set_v5_activation_block", ROOT / "scripts" / "set_v5_activation_block.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check() == 0                   # unset in config and in the compose: they agree
