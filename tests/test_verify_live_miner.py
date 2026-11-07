"""scripts/verify_live_miner.py asks for the epoch miners serve and checks the head as a validator does.

Measured 2026-11-07: without FUGAL_EPOCH_INTERVAL it used config's 3600 s and
asked finney/70 miners (4320 s epochs) for e00031524 while they served
e00026270. And it verified the proof without the head commitment or the head
bytes, so a miner every validator rejects could pass it.
"""
import importlib.util
import pathlib
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "verify_live_miner.py"
spec = importlib.util.spec_from_file_location("verify_live_miner_under_test", SRC)
vlm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlm)


class _Chain:
    def get_subnet_hyperparameters(self, netuid):
        return SimpleNamespace(tempo=360)


def test_the_epoch_comes_from_the_flag_the_env_or_the_chain(monkeypatch):
    monkeypatch.delenv("FUGAL_EPOCH_INTERVAL", raising=False)
    assert vlm.resolve_epoch_interval(None, _Chain(), 70) == 4320       # tempo 360 x 12 s
    monkeypatch.setenv("FUGAL_EPOCH_INTERVAL", "3600")
    assert vlm.resolve_epoch_interval(None, _Chain(), 70) == 3600
    assert vlm.resolve_epoch_interval(4320, _Chain(), 70) == 4320


def test_the_commitment_rule_is_the_validators():
    c = {"hk": ("h" * 64, 900)}
    assert vlm.check_commitment(c, "hk", "h" * 64, 1000)[0]
    assert not vlm.check_commitment(c, "hk", "h" * 64, 899)[0]      # after the boundary
    assert not vlm.check_commitment(c, "hk", "x" * 64, 1000)[0]     # a different head
    assert not vlm.check_commitment({}, "hk", "h" * 64, 1000)[0]    # never committed


def test_it_verifies_the_head_through_the_validators_own_parser():
    src = SRC.read_text(encoding="utf-8")
    assert "from neurons.validator import _get_bundle_for_uid" in src
    assert "head_bytes=head_bytes" in src and "expected_weights_hash=weights_hash" in src
    assert "config import (\n        EPOCH_INTERVAL" not in src
    # Saved before anything that can fail on the network or the bundle.
    assert src.index("proof saved to") < src.index("get_commitments_with_blocks(subtensor")
