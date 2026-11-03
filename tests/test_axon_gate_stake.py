"""The miner axon gate after D2 (2026-11-03): stake is required of every
caller, a permit alone is not enough, the floor defaults to 50 in the chain's
total_stake units, and a validator below the floor is told so at startup."""

import os
import subprocess
import sys

from neurons.miner import caller_allowed
from neurons.validator import axon_gate_notice


class MG:
    hotkeys = ["5big", "5permit_only", "5exact", "5below", "5unstaked"]
    validator_permit = [True, True, False, False, False]
    S = [363_354.5, 0.0042, 50.0, 49.99, 0.0]


def test_default_floor_is_50_in_total_stake_units():
    env = {**os.environ}
    env.pop("FUGAL_MIN_VALIDATOR_STAKE", None)
    out = subprocess.run(
        [sys.executable, "-c", "from fugal_subnet.config import MIN_VALIDATOR_STAKE as m; print(m)"],
        env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert float(out) == 50.0


def test_permit_without_stake_is_refused_and_told_why():
    ok, reason = caller_allowed(MG(), "5permit_only", live=True, min_stake=50)
    assert not ok
    assert "permit alone" in reason


def test_stake_at_the_floor_passes_without_a_permit():
    ok, reason = caller_allowed(MG(), "5exact", live=True, min_stake=50)
    assert ok and "stake 50" in reason
    ok, _ = caller_allowed(MG(), "5below", live=True, min_stake=50)
    assert not ok


def test_large_root_validator_passes():
    ok, reason = caller_allowed(MG(), "5big", live=True, min_stake=50)
    assert ok and "permit" in reason


def test_mock_still_admits_any_registered_hotkey():
    assert caller_allowed(MG(), "5unstaked", live=False)[0]
    assert not caller_allowed(MG(), "5stranger", live=False)[0]


def test_validator_startup_notice_names_the_floor_and_the_permit_trap():
    assert axon_gate_notice(50.0, False, 50.0) is None
    assert axon_gate_notice(11_900.0, True, 50.0) is None
    msg = axon_gate_notice(0.0042, True, 50.0)
    assert msg and "50" in msg and "permit does not substitute" in msg
    msg = axon_gate_notice(0.0, False, 50.0)
    assert msg and "No valid proofs received" in msg
    assert "permit does not substitute" not in msg
