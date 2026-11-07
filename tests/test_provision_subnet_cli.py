"""scripts/provision_subnet.py must work on a host that holds no owner coldkey.

On mainnet the subnet slot can be held by a different party than the one
running the validators (and, before the wallet transfer, by someone who is
not on this machine at all). The validator host holds a hotkey. It must still
be able to ask the chain whether the subnet is active and tempo-aligned
(MAINNET_LAUNCH §1-2), and to register and stake its own wallets. Until
2026-11-02 --owner-wallet was required for everything, including --status,
and the script loaded the owner wallet before doing anything, so a host
without it could not even run the read-only report.

The chain is stubbed; nothing here talks to a network or spends.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "provision_subnet.py"


class _HP:
    tempo = 360
    commit_reveal_weights_enabled = True
    commit_reveal_period = 1
    weights_rate_limit = 100
    serving_rate_limit = 50
    activity_cutoff = 5000
    registration_allowed = True


class _MG:
    n = 0
    axons: list = []
    S: list = []
    validator_permit: list = []
    hotkeys: list = []


class _Subtensor:
    """The subset of bt.Subtensor the script touches, recording what it saw."""

    def __init__(self, network="test", active=True):
        self.network = network
        self._active = active

    def get_current_block(self):
        return 1_000_000

    def is_subnet_active(self, netuid):
        return self._active

    def get_subnet_hyperparameters(self, netuid):
        return _HP()

    def metagraph(self, netuid):
        return _MG()

    def recycle(self, netuid):
        return "τ0.5"


def _load(monkeypatch, subtensor):
    """Import the script with `bt` replaced by a stub that refuses to make wallets."""
    def no_wallet(*a, **k):
        raise AssertionError("a wallet was loaded; --status and owner-less runs must not")

    bt_stub = types.SimpleNamespace(
        Subtensor=lambda network="test": subtensor,
        Wallet=no_wallet,
        Balance=types.SimpleNamespace(from_rao=lambda r: r, from_tao=lambda t: t),
    )
    monkeypatch.setitem(sys.modules, "bittensor", bt_stub)
    spec = importlib.util.spec_from_file_location("provision_subnet_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_status_needs_no_owner_wallet(monkeypatch, capsys):
    mod = _load(monkeypatch, _Subtensor())
    monkeypatch.setattr(sys, "argv", ["provision_subnet.py", "--network", "finney",
                                      "--netuid", "7", "--status",
                                      "--epoch-interval", "4320"])
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "SUBNET 7" in out and "ALIGNED" in out and "MISALIGNED" not in out


def test_without_owner_an_inactive_subnet_is_refused(monkeypatch):
    mod = _load(monkeypatch, _Subtensor(active=False))
    monkeypatch.setattr(sys, "argv", ["provision_subnet.py", "--netuid", "7",
                                      "--epoch-interval", "4320"])
    assert mod.main() == 1


def test_without_owner_a_misaligned_epoch_is_refused(monkeypatch):
    mod = _load(monkeypatch, _Subtensor(active=True))
    monkeypatch.setattr(sys, "argv", ["provision_subnet.py", "--netuid", "7",
                                      "--epoch-interval", "3600"])   # 300 blocks vs tempo 360
    assert mod.main() == 1


def test_without_owner_an_active_aligned_subnet_proceeds_to_the_report(monkeypatch, capsys):
    mod = _load(monkeypatch, _Subtensor(active=True))
    monkeypatch.setattr(sys, "argv", ["provision_subnet.py", "--netuid", "7",
                                      "--epoch-interval", "4320"])
    assert mod.main() == 0
    assert "SUBNET 7" in capsys.readouterr().out


def test_create_still_requires_the_owner(monkeypatch):
    mod = _load(monkeypatch, _Subtensor())
    monkeypatch.setattr(sys, "argv", ["provision_subnet.py", "--create"])
    assert mod.main() == 2


@pytest.mark.parametrize("flag", ["--owner-wallet"])
def test_the_flag_is_documented_as_optional(flag):
    text = SRC.read_text(encoding="utf-8")
    assert f'"{flag}", default=None' in text


def test_without_an_epoch_interval_the_chains_tempo_decides(monkeypatch, capsys):
    """A hard-coded 3600 default reported an aligned tempo-360 subnet as
    MISALIGNED whenever the flag was forgotten (review F-077)."""
    mod = _load(monkeypatch, _Subtensor())
    monkeypatch.delenv("FUGAL_EPOCH_INTERVAL", raising=False)
    monkeypatch.setattr(sys, "argv", ["provision_subnet.py", "--network", "finney",
                                      "--netuid", "7", "--status"])
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "ALIGNED" in out and "MISALIGNED" not in out


def test_the_operator_scripts_follow_the_epochs_protocol_version():
    """verify_live_miner must select and verify with the boundary-keyed version
    (F-076); tdx_measurement's probe must name a hotkey, or verify_proof refuses
    to run outside mock before the measurement is ever checked (F-075)."""
    root = SRC.parents[1]
    vl = (root / "scripts" / "verify_live_miner.py").read_text(encoding="utf-8")
    assert "protocol_version_for_boundary(" in vl and "protocol_version=version" in vl
    assert "scored_slice(version" in vl and "exploration_plan(version" in vl
    assert "select_slice(" not in vl and "expected_exploration(" not in vl
    tm = (root / "scripts" / "tdx_measurement.py").read_text(encoding="utf-8")
    assert 'expected_hotkey="5TdxMeasurementProbe"' in tm and 'hotkey="5TdxMeasurementProbe"' in tm
    assert "export FUGAL_TEE_MEASUREMENTS=" not in tm      # a bare base is refused by --live


# --- The owner path writes nothing it was not told to -----------------------

class _OwnerSubtensor(_Subtensor):
    def subnet(self, netuid):
        return types.SimpleNamespace(owner_coldkey="OWNER")

    def get_balance(self, address):
        return 1


def _owner_run(monkeypatch, *flags):
    """Run the owner path on a stubbed finney/70 with a stale 3600 interval in
    the environment (tempo is 360 blocks, so 3600 s is a 300-block mismatch),
    recording every chain write."""
    mod = _load(monkeypatch, _OwnerSubtensor(network="finney"))
    wallet = types.SimpleNamespace(coldkeypub=types.SimpleNamespace(ss58_address="OWNER"))
    monkeypatch.setattr(mod.bt, "Wallet", lambda **k: wallet)
    writes = []
    monkeypatch.setattr(mod, "_sudo_call", lambda s, w, call, params: writes.append((call, params)) or (True, ""))
    monkeypatch.setenv("FUGAL_EPOCH_INTERVAL", "3600")
    monkeypatch.setattr(sys, "argv", ["x", "--network", "finney", "--netuid", "70",
                                      "--owner-wallet", "owner", *flags])
    assert mod.main() == 0
    return writes


def test_without_yes_the_owner_path_writes_nothing(monkeypatch):
    """Measured 2026-11-07 on this stub: a run without --yes retuned tempo
    360 -> 300 and rewrote the identity."""
    assert _owner_run(monkeypatch) == []
    assert _owner_run(monkeypatch, "--set-tempo") == []


def test_tempo_changes_only_when_asked_for(monkeypatch):
    assert _owner_run(monkeypatch, "--yes") == []
    assert _owner_run(monkeypatch, "--yes", "--set-tempo") == [
        ("sudo_set_tempo", {"netuid": 70, "tempo": 300})]


@pytest.mark.parametrize("flag", ["--set-tempo", "--set-identity"])
def test_the_opt_in_flags_are_documented(flag):
    assert flag in SRC.read_text(encoding="utf-8").split('"""')[1]
