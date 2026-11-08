"""neurons/miner.py serves its axon and keeps it served.

From the neurons/ review (2026-11-08): when axon.serve() failed five times at
startup the miner logged one error and ran at 0.0.0.0 for good, alive and
earning nothing, because nothing retried it; an unroutable FUGAL_AXON_IP only
logged; and the port and routability checks had no tests.
"""
import ast
import pathlib
import socket
import types

import click
import pytest

import neurons.miner as miner

SRC = pathlib.Path(miner.__file__).read_text(encoding="utf-8")


def _mg(ip):
    return types.SimpleNamespace(axons=[types.SimpleNamespace(ip=ip)])


@pytest.mark.parametrize("ip, ok", [("1.2.3.4", True), ("0.0.0.0", False), ("", False), (None, False)])
def test_registered_means_a_routable_address_on_chain(ip, ok):
    assert miner._axon_registered(_mg(ip), 0) is ok
    assert miner._axon_registered(_mg("1.2.3.4"), 5) is False      # no such UID


def test_a_failed_serve_is_reported_and_returns_the_chains_view():
    class Axon:
        def serve(self, **kw):
            raise RuntimeError("rate limited")

    sub = types.SimpleNamespace(metagraph=lambda netuid: _mg("0.0.0.0"))
    assert not miner._axon_registered(miner._serve_axon_once(Axon(), sub, 70), 0)


def test_the_refresh_loop_serves_the_axon_again_while_it_is_unregistered():
    """Structural: inside main's loop, the metagraph refresh re-serves an
    unregistered axon. The old loop retried only the commitment."""
    fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.While))
    called = {n.func.id for n in ast.walk(loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {"_axon_registered", "_serve_axon_once"} <= called


@pytest.mark.parametrize("ip, bad", [("127.0.0.1", True), ("0.0.0.0", True), ("::1", True),
                                     ("10.0.0.5", False), ("203.0.113.9", False)])
def test_loopback_and_unspecified_are_unroutable(ip, bad):
    assert miner._is_unroutable(ip) is bad


def test_an_unroutable_axon_ip_stops_startup():
    assert 'raise click.ClickException(\n            f"FUGAL_AXON_IP={external_ip} is not routable' in SRC


def test_a_port_already_in_use_is_refused():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        with pytest.raises(click.ClickException, match="already in use"):
            miner._assert_port_free(s.getsockname()[1])
