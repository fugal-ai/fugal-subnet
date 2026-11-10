"""The release gate (`check_safety_invariants.py --release`) checks what the
live composes are frozen AS, not only that they are frozen: a v2 release
compose pins its image by digest and carries a valid activation block (Fable
review 2026-11-09, item 8: the gate passed a frozen v1 compose)."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("csi", ROOT / "scripts" / "check_safety_invariants.py")
csi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csi)

DIGEST = "ghcr.io/fugal-ai/fugal-miner@sha256:" + "ab" * 32


def _tree(tmp_path, compose: str) -> pathlib.Path:
    d = tmp_path / "deploy" / "dstack"
    d.mkdir(parents=True)
    (d / "docker-compose.yaml").write_text(compose)
    return tmp_path


def _errors(tmp_path, compose):
    errors: list[str] = []
    csi.check_release_composes_target_v2(errors, _tree(tmp_path, compose))
    return errors


def _compose(image=DIGEST, env=("- FUGAL_EPOCH_INTERVAL=4320", "- FUGAL_V2_ACTIVATION_BLOCK=7200000")):
    body = "\n".join("      " + e for e in env)
    return f"services:\n  miner:\n    image: {image}\n    environment:\n{body}\n"


def test_a_v2_release_compose_passes(tmp_path):
    assert _errors(tmp_path, _compose()) == []


def test_the_mapping_form_is_read_too(tmp_path):
    assert _errors(tmp_path, _compose(env=('FUGAL_EPOCH_INTERVAL: "4320"',
                                           'FUGAL_V2_ACTIVATION_BLOCK: "7200000"'))) == []


@pytest.mark.parametrize("env,why", [
    (("- FUGAL_EPOCH_INTERVAL=4320",), "no FUGAL_V2_ACTIVATION_BLOCK"),
    (("- FUGAL_EPOCH_INTERVAL=4320", "- FUGAL_V2_ACTIVATION_BLOCK=7200001"), "not an epoch boundary"),
    (("- FUGAL_EPOCH_INTERVAL=4320", "- FUGAL_V2_ACTIVATION_BLOCK=0"), "above 0"),
    (("- FUGAL_EPOCH_INTERVAL=4320", "- FUGAL_V2_ACTIVATION_BLOCK=soon"), "not valid"),
])
def test_a_compose_without_a_valid_activation_block_is_refused(tmp_path, env, why):
    errors = _errors(tmp_path, _compose(env=env))
    assert len(errors) == 1 and why in errors[0], errors


def test_an_image_by_tag_is_refused(tmp_path):
    errors = _errors(tmp_path, _compose(image="ghcr.io/fugal-ai/fugal-miner:latest"))
    assert any("not pinned by digest" in e for e in errors)


def test_the_check_is_wired_into_the_release_gate():
    src = (ROOT / "scripts" / "check_safety_invariants.py").read_text()
    assert "check_release_composes_target_v2(errors if args.release else launch_gate)" in src
