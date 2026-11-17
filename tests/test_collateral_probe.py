"""The validator asks the DCAP collateral endpoint once at startup, as advice.

v3 testnet rehearsal (rc1, 2026-11-17): a follower pointed FUGAL_PCCS_URL at a
local caching PCCS that was not running, and nothing said so until collection.
v0.5.0: the first probe refused to start on finney, sent urllib's default
User-Agent, got 403 from pccs.phala.network (which serves the verifier's real
requests), and took the live root down. So: the probe names itself, and the
validator only warns, on every chain.
"""
import ast
import io
import logging
import pathlib
import urllib.error

import pytest

from fugal_subnet.tee.attestation import COLLATERAL_PROBE_TIMEOUT_S, probe_collateral_endpoint


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

    def open_url(request, timeout):
        seen.append((request, timeout))
        if exc is not None:
            raise exc
        return _Resp(body, status)

    open_url.seen = seen
    return open_url


def test_a_serving_endpoint_passes_and_the_probe_names_itself():
    op = _opener()
    ok, why = probe_collateral_endpoint("https://pccs.example/", opener=op)
    assert ok, why
    (request, timeout), = op.seen
    assert request.full_url == "https://pccs.example/sgx/certification/v4/rootcacrl"
    assert timeout == COLLATERAL_PROBE_TIMEOUT_S <= 10
    ua = request.get_header("User-agent")
    assert ua and ua.startswith("fugal-validator/") and "Python-urllib" not in ua


@pytest.mark.parametrize("exc", [
    urllib.error.URLError("connection refused"),
    OSError("timed out"),
    urllib.error.HTTPError("u", 403, "Forbidden", {}, None),
])
def test_an_unreachable_or_refusing_endpoint_fails_the_probe(exc):
    ok, why = probe_collateral_endpoint("http://127.0.0.1:8081", 1.0, opener=_opener(exc=exc))
    assert not ok and "rootcacrl" in why


def test_an_empty_answer_or_no_url_fails_the_probe():
    assert not probe_collateral_endpoint("https://pccs.example", 1.0, opener=_opener(body=b""))[0]
    assert not probe_collateral_endpoint("", 1.0, opener=_opener())[0]


@pytest.mark.parametrize("on_finney", [True, False])
@pytest.mark.parametrize("probe", [
    lambda url: (False, "https://pccs.phala.network/... answered HTTP 403"),
    lambda url: (False, "is unreachable: timed out"),
])
def test_a_failed_probe_only_warns_and_startup_continues(caplog, on_finney, probe):
    from neurons.validator import _check_collateral_endpoint

    with caplog.at_level(logging.WARNING):
        _check_collateral_endpoint(True, on_finney, probe=probe)   # must not raise
    assert "did not answer the startup probe" in caplog.text and "advisory" in caplog.text


def test_a_probe_that_raises_only_warns(caplog):
    from neurons.validator import _check_collateral_endpoint

    def boom(url):
        raise RuntimeError("socket exploded")

    with caplog.at_level(logging.WARNING):
        _check_collateral_endpoint(True, True, probe=boom)
    assert "socket exploded" in caplog.text


def test_mock_does_not_probe():
    from neurons.validator import _check_collateral_endpoint

    def boom(url):
        raise AssertionError("mock must not probe")

    _check_collateral_endpoint(False, True, probe=boom)


def test_the_check_never_refuses_to_start():
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_check_collateral_endpoint")
    assert not any(isinstance(n, ast.Raise) for n in ast.walk(fn))
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = {c.func.id for c in ast.walk(main)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "_check_collateral_endpoint" in calls
