"""The protocol v5 release gate (scripts/check_safety_invariants.py,
`v5_release_blockers`, `--release-v5`).

Each test builds a tree where every v5 condition holds (the v4 gate's own
conditions are covered by test_release_gate_v4 and passed in as none) and
breaks exactly one; the gate on today's tree lists what is still open, and
the plain check does not fail on it."""
import importlib.util
import json
import pathlib
import shutil
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BPE = 360


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_safety_invariants", ROOT / "scripts" / "check_safety_invariants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(**kw):
    from fugal_subnet import config

    v4 = config.FINNEY_V4_ACTIVATION_BLOCK
    base = dict(
        FINNEY_V2_ACTIVATION_BLOCK=config.FINNEY_V2_ACTIVATION_BLOCK,
        FINNEY_V3_ACTIVATION_BLOCK=config.FINNEY_V3_ACTIVATION_BLOCK,
        FINNEY_ODDS_SEAT_BLOCK=config.FINNEY_ODDS_SEAT_BLOCK,
        FINNEY_MENU_B_BLOCK=config.FINNEY_MENU_B_BLOCK,
        FINNEY_V4_ACTIVATION_BLOCK=v4,
        FINNEY_V5_ACTIVATION_BLOCK=(v4 // BPE + 100) * BPE,
        EMBEDDINGS_V5_SHA256="ab" * 32,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _compose(env: dict[str, str]) -> str:
    lines = "\n".join(f"      - {k}={v}" for k, v in env.items())
    return ("services:\n  miner:\n    image: ghcr.io/x/y@sha256:" + "a" * 64 + "\n"
            "    command:\n      - --network\n      - finney\n"
            "    environment:\n" + lines + "\n")


@pytest.fixture
def tree(tmp_path):
    from fugal_subnet import consensus_v5

    cfg = _cfg()
    pkg = tmp_path / "fugal_subnet"
    pkg.mkdir()
    shutil.copy(ROOT / "fugal_subnet" / "config.py", pkg / "config.py")
    for _role, (rel, _req) in consensus_v5.PINNED_FILES.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text("pinned\n", encoding="utf-8")
    dd = tmp_path / "deploy" / "dstack"
    dd.mkdir(parents=True)
    env = {"FUGAL_V5_ACTIVATION_BLOCK": str(cfg.FINNEY_V5_ACTIVATION_BLOCK)}
    (dd / "app-compose.frozen-2026-11-25-finney.json").write_text(
        json.dumps({"docker_compose_file": _compose(env)}), encoding="utf-8")
    return types.SimpleNamespace(root=tmp_path, cfg=cfg, dstack=dd)


def _gate(tree, cfg=None, probe=None):
    mod = _checker()
    c = cfg or tree.cfg
    probe = probe or (lambda root, env: {"v5_block": int(env["FUGAL_V5_ACTIVATION_BLOCK"])})
    return mod.v5_release_blockers(tree.root, c, probe=probe, v4_blockers=())


def test_a_release_with_every_condition_passes(tree):
    assert _gate(tree) == []


@pytest.mark.parametrize("pin", ["TBD", None, ""])
def test_an_unpinned_embeddings_file_is_a_blocker(tree, pin):
    got = _gate(tree, cfg=_cfg(EMBEDDINGS_V5_SHA256=pin))
    assert any(g.startswith("pinned: config.EMBEDDINGS_V5_SHA256 is not pinned") for g in got)


def test_v4_conditions_are_v5_conditions(tree):
    mod = _checker()
    got = mod.v5_release_blockers(tree.root, tree.cfg, v4_blockers=("list: no signed list",),
                                  probe=lambda r, e: {"v5_block": tree.cfg.FINNEY_V5_ACTIVATION_BLOCK})
    assert got == ["v4: list: no signed list"]


@pytest.mark.parametrize("block, why", [
    (None, "is not set"),
    ("9600120", "is not an integer"),
    (9_600_121, "is not an epoch boundary"),
])
def test_the_block_must_be_set_aligned_and_an_integer(tree, block, why):
    got = _gate(tree, cfg=_cfg(FINNEY_V5_ACTIVATION_BLOCK=block))
    assert any(b.startswith("block:") and why in b for b in got)


def test_the_block_must_be_after_v4(tree):
    v4 = tree.cfg.FINNEY_V4_ACTIVATION_BLOCK
    got = _gate(tree, cfg=_cfg(FINNEY_V5_ACTIVATION_BLOCK=v4))
    assert any("is not after FINNEY_V4_ACTIVATION_BLOCK" in b for b in got)


def test_a_missing_pinned_file_is_a_blocker(tree):
    (tree.root / "fugal_subnet" / "graders_arena.py").unlink()
    got = _gate(tree)
    assert any(b.startswith("pinned:") and "graders_arena.py" in b for b in got)


def test_a_frozen_compose_with_another_v5_block_is_a_blocker(tree):
    f = next(tree.dstack.glob("app-compose.frozen-*-finney.json"))
    f.write_text(json.dumps({"docker_compose_file": _compose(
        {"FUGAL_V5_ACTIVATION_BLOCK": "1"})}), encoding="utf-8")
    got = _gate(tree, probe=lambda r, e: {"v5_block": 1})
    assert any(b.startswith("compose:") and "FUGAL_V5_ACTIVATION_BLOCK=1" in b for b in got)
    assert any("the miner reads v5 block 1" in b for b in got)


def test_no_frozen_compose_is_a_blocker(tree):
    for f in tree.dstack.glob("*.json"):
        f.unlink()
    assert any(b.startswith("compose: no readable") for b in _gate(tree))


def test_a_miner_that_refuses_to_start_is_a_blocker(tree):
    got = _gate(tree, probe=lambda r, e: {"error": "ValueError: no"})
    assert any("refuses to start" in b for b in got)


def test_a_pinned_file_excluded_from_the_image_is_a_blocker(tree):
    (tree.root / ".dockerignore").write_text("fugal_subnet/benchmarks/\n", encoding="utf-8")
    got = _gate(tree)
    assert any(b.startswith("image:") and "pool_manifest_v5.json" in b for b in got)


def test_the_gate_today_lists_the_open_conditions_and_the_plain_check_passes(capsys):
    mod = _checker()
    got = mod.v5_release_blockers(v4_blockers=())
    kinds = {b.split(":", 1)[0] for b in got}
    assert kinds <= {"v4", "block", "pinned", "compose", "image"}
    from fugal_subnet import config

    assert any(b.startswith("block: FINNEY_V5_ACTIVATION_BLOCK is not set") for b in got) \
        is (config.FINNEY_V5_ACTIVATION_BLOCK is None)
    mod.main([])
    assert "Protocol v5 release gate:" in capsys.readouterr().out
