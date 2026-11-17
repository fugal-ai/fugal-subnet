"""The validator asks the DCAP collateral endpoint once at startup.

v3 testnet rehearsal (rc1, 2026-11-17): a follower pointed FUGAL_PCCS_URL at a
local caching PCCS that was not running. Nothing said so until the collection
point, and that epoch's proofs all went unverified.
"""
import ast
import io
import logging
import pathlib
import urllib.error

import click
import pytest

from fugal_subnet.tee.attestation import probe_collateral_endpoint


class _Resp(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(body=b"-----BEGIN X509 CRL-----", status=200, exc=None):
    seen = []

    def open_url(url, timeout):
        seen.append((url, timeout))
        if exc is not None:
            raise exc
        return _Resp(body, status)

    open_url.seen = seen
    return open_url


def test_a_serving_endpoint_passes_and_is_asked_for_the_root_crl():
    op = _opener()
    ok, why = probe_collateral_endpoint("https://pccs.example/", 5.0, opener=op)
    assert ok, why
    assert op.seen == [("https://pccs.example/sgx/certification/v4/rootcacrl", 5.0)]


@pytest.mark.parametrize("exc", [
    urllib.error.URLError("connection refused"),
    OSError("timed out"),
    urllib.error.HTTPError("u", 404, "not found", {}, None),
])
def test_an_unreachable_or_refusing_endpoint_fails(exc):
    ok, why = probe_collateral_endpoint("http://127.0.0.1:8081", 1.0, opener=_opener(exc=exc))
    assert not ok and "rootcacrl" in why


def test_an_empty_answer_or_no_url_fails():
    assert not probe_collateral_endpoint("https://pccs.example", 1.0, opener=_opener(body=b""))[0]
    assert not probe_collateral_endpoint("", 1.0, opener=_opener())[0]


def test_a_live_finney_validator_refuses_to_start_on_a_dead_endpoint():
    from neurons.validator import _check_collateral_endpoint

    with pytest.raises(click.ClickException, match="cannot serve collateral"):
        _check_collateral_endpoint(True, True, probe=lambda url, t: (False, "dead"))


def test_elsewhere_a_dead_endpoint_only_warns(caplog):
    from neurons.validator import _check_collateral_endpoint

    with caplog.at_level(logging.WARNING):
        _check_collateral_endpoint(True, False, probe=lambda url, t: (False, "dead"))
    assert "cannot serve collateral" in caplog.text


def test_mock_does_not_probe():
    from neurons.validator import _check_collateral_endpoint

    def boom(url, t):
        raise AssertionError("mock must not probe")

    _check_collateral_endpoint(False, True, probe=boom)


def test_the_validator_checks_the_endpoint_at_startup():
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    main = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = {c.func.id for c in ast.walk(main)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "_check_collateral_endpoint" in calls
