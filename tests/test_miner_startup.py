"""neurons/miner.py startup: refuse early, refuse loudly.

Three things the miner did in the wrong order or the wrong way (review of
neurons/, 2026-11-08): it checked the chain (mock on finney, consensus
overrides) only after receiving the operator's secrets and embedding the pool,
which takes hours; an unregistered hotkey logged an error and exited 0; and a
provisioning push with no head had an untested error.
"""
import ast
import pathlib
import types

import click
import pytest

import neurons.miner as miner

SRC = pathlib.Path(miner.__file__).read_text(encoding="utf-8")


def _main_call_lines():
    fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    lines = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            lines.setdefault(node.func.id, node.lineno)
    return lines


def test_the_chain_is_checked_before_secrets_arrive_or_the_pool_is_embedded():
    calls = _main_call_lines()
    assert calls["_assert_chain_allows"] < calls["_await_provisioning"]
    assert calls["_assert_chain_allows"] < calls["_compute_hidden_states"]


def test_mock_on_finney_is_refused_by_the_early_check(monkeypatch):
    import bittensor as bt

    from fugal_subnet import fingerprint
    from fugal_subnet.config import FINNEY_GENESIS_HASH

    closed = []
    monkeypatch.setattr(bt, "Subtensor", lambda network: types.SimpleNamespace(
        close=lambda: closed.append(True)))
    monkeypatch.setattr(fingerprint, "chain_genesis_hash", lambda st: FINNEY_GENESIS_HASH)
    with pytest.raises(click.ClickException, match="--mock refused on finney"):
        miner._assert_chain_allows("finney", mock=True)
    assert closed == [True]                       # the check connection is closed


def test_an_unregistered_hotkey_is_a_startup_error():
    mg = types.SimpleNamespace(hotkeys=["5a", "5b"])
    assert miner._my_uid(mg, "5b", 70) == 1
    with pytest.raises(click.ClickException, match="not registered on netuid 70"):
        miner._my_uid(mg, "5zz", 70)


def test_a_push_with_no_head_stops_the_miner_with_a_named_error(monkeypatch):
    from fugal_subnet.tee import provision

    class Store:
        ready = True

        def get(self, name):
            return None

    monkeypatch.setattr(provision, "ProvisionStore", Store)
    monkeypatch.setattr(provision, "serve", lambda store, port: types.SimpleNamespace(
        serve_forever=lambda: None))
    with pytest.raises(click.ClickException, match="provisioned without a head"):
        miner._await_provisioning(coldkey="c", hotkey="h")


def test_static_checks_run_before_secrets_arrive_and_before_the_embedding():
    """The unroutable-IP refusal and the registration check ran only after
    the pool was embedded (hours); for a v2 image the IP check ran only at the
    flip, after the v2 head was committed (independent review, 2026-11-08)."""
    calls = _main_call_lines()
    assert calls["_assert_axon_ip_routable"] < calls["_await_provisioning"]
    assert calls["_await_provisioning"] < calls["_assert_registered"]
    assert calls["_assert_registered"] < calls["_compute_hidden_states"]
    # The UID is still read again after the embedding, where it is used.
    assert calls["_my_uid"] > calls["_compute_hidden_states"]
    # And the IP checked is the one the axon will serve.
    fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    (ip_check,) = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", "") == "_assert_axon_ip_routable"]
    assert ast.unparse(ip_check.args[0]) == "os.getenv('FUGAL_AXON_IP')"
    assert "external_ip = os.getenv('FUGAL_AXON_IP')" in ast.unparse(fn)


@pytest.mark.parametrize("ip", ["127.0.0.1", "0.0.0.0", "::1"])
def test_an_unroutable_axon_ip_is_refused(ip):
    with pytest.raises(click.ClickException, match="not routable"):
        miner._assert_axon_ip_routable(ip)


@pytest.mark.parametrize("ip", [None, "", "34.120.5.6"])
def test_a_routable_or_unset_axon_ip_passes(ip):
    miner._assert_axon_ip_routable(ip)


def test_the_registration_check_refuses_and_closes_its_connection(monkeypatch):
    import bittensor as bt

    closed = []
    monkeypatch.setattr(bt, "Wallet", lambda name, hotkey, path: types.SimpleNamespace(
        hotkey=types.SimpleNamespace(ss58_address="5mine")))
    monkeypatch.setattr(bt, "Subtensor", lambda network: types.SimpleNamespace(
        metagraph=lambda netuid: types.SimpleNamespace(hotkeys=["5other"]),
        close=lambda: closed.append(True)))
    with pytest.raises(click.ClickException, match="not registered"):
        miner._assert_registered("test", 552, "c", "h", None)
    assert closed == [True]
    monkeypatch.setattr(bt, "Subtensor", lambda network: types.SimpleNamespace(
        metagraph=lambda netuid: types.SimpleNamespace(hotkeys=["5other", "5mine"]),
        close=lambda: closed.append(True)))
    miner._assert_registered("test", 552, "c", "h", None)
    assert closed == [True, True]
