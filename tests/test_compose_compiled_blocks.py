"""The finney compose must carry every block the miner reads only from its
environment, at the value the release compiles for the validator.

v0.5.0: the first v3 composes set FUGAL_V2_ACTIVATION_BLOCK but not
FUGAL_V3_ACTIVATION_BLOCK. The miner picks its protocol from that variable
(neurons/miner.py), so every miner on them would have kept serving v2 after the
v3 block while every validator scored v3.
"""
import importlib.util
import pathlib
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_safety_invariants", ROOT / "scripts" / "check_safety_invariants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compose(network: str, env: dict[str, str]) -> str:
    lines = "\n".join(f"      - {k}={v}" for k, v in env.items())
    return ("services:\n  miner:\n    image: ghcr.io/x/y@sha256:" + "a" * 64 + "\n"
            "    environment:\n" + lines + "\n"
            "    entrypoint:\n      - python\n      - --network\n      - " + network + "\n")


def _run(tmp_path, files: dict[str, str], v2=9485640, v3=9527760):
    d = tmp_path / "deploy" / "dstack"
    d.mkdir(parents=True)
    for name, text in files.items():
        (d / name).write_text(text)
    cfg = types.SimpleNamespace(FINNEY_V2_ACTIVATION_BLOCK=v2, FINNEY_V3_ACTIVATION_BLOCK=v3)
    errors: list[str] = []
    _checker().check_finney_compose_carries_compiled_blocks(errors, root=tmp_path, cfg=cfg)
    return errors


def test_a_finney_compose_without_the_v3_block_is_refused(tmp_path):
    errors = _run(tmp_path, {"docker-compose.yaml": _compose(
        "finney", {"FUGAL_V2_ACTIVATION_BLOCK": "9485640"})})
    assert len(errors) == 1 and "FUGAL_V3_ACTIVATION_BLOCK=(unset)" in errors[0]


def test_a_finney_compose_with_a_different_block_is_refused(tmp_path):
    errors = _run(tmp_path, {"docker-compose.yaml": _compose(
        "finney", {"FUGAL_V2_ACTIVATION_BLOCK": "9485640", "FUGAL_V3_ACTIVATION_BLOCK": "9527400"})})
    assert len(errors) == 1 and "9527400" in errors[0]


def test_a_finney_compose_with_both_compiled_blocks_passes(tmp_path):
    assert _run(tmp_path, {"docker-compose.yaml": _compose(
        "finney", {"FUGAL_V2_ACTIVATION_BLOCK": "9485640",
                   "FUGAL_V3_ACTIVATION_BLOCK": "9527760"})}) == []


def test_the_testnet_twin_is_not_held_to_finney_blocks(tmp_path):
    assert _run(tmp_path, {"docker-compose.test.yaml": _compose(
        "test", {"FUGAL_V2_ACTIVATION_BLOCK": "8414640"})}) == []


def test_a_block_the_release_does_not_compile_is_not_required(tmp_path):
    assert _run(tmp_path, {"docker-compose.yaml": _compose(
        "finney", {"FUGAL_V2_ACTIVATION_BLOCK": "9485640"})}, v3=None) == []


def test_the_shipped_finney_compose_passes():
    errors: list[str] = []
    _checker().check_finney_compose_carries_compiled_blocks(errors)
    assert errors == []


def test_the_v4_block_setter_finds_one_value_in_config_and_the_finney_compose():
    """scripts/set_v4_activation_block.py --check: the compiled block and the
    finney compose agree, and the block is an epoch boundary."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "set_v4_activation_block.py"
    spec = importlib.util.spec_from_file_location("set_v4_activation_block", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check() == 0
    from fugal_subnet import config

    if config.FINNEY_V4_ACTIVATION_BLOCK is not None:
        assert config.FINNEY_V4_ACTIVATION_BLOCK % mod.FINNEY_BLOCKS_PER_EPOCH == 0
        assert mod.forms(9572760) == ["9,572,760", "9572760", "e00026591", "26,591", "26591"]
