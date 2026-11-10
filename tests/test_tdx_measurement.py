"""scripts/tdx_measurement.py's verdicts, with the hardware faked.

Real runs need a TDX guest; these pin what the script concludes from what the
hardware and verifier return. A collateral outage judged nothing, so neither
check may call it a failure of this machine.
"""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def tdx(monkeypatch):
    spec = importlib.util.spec_from_file_location("tdx_measurement_under_test",
                                                  ROOT / "scripts" / "tdx_measurement.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from fugal_subnet.tee import attestation, runtime, verify

    monkeypatch.setattr(mod, "_require_tdx", lambda: None)
    monkeypatch.setattr(runtime, "TEERuntime",
                        lambda mock=False: SimpleNamespace(generate_attestation=lambda d: b"q"))
    quote = SimpleNamespace(mrtd="m", rtmr0="r0", rtmr1="r1", rtmr2="r2")
    monkeypatch.setattr(attestation, "parse_quote", lambda b: quote)
    monkeypatch.setattr(attestation, "measurement_id", lambda q: "ab" * 32)
    monkeypatch.setattr(attestation, "verify_dcap", lambda b: True)
    monkeypatch.setattr(sys, "argv", ["tdx_measurement.py", "--verify"])
    return mod, verify, monkeypatch


def test_an_unapproved_image_rejected_is_a_pass(tdx, capsys):
    mod, verify, mp = tdx
    mp.setattr(verify, "verify_proof", lambda *a, **k: verify.VerifyResult(
        False, "Unapproved runtime image: measurement abab... not among 1"))
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "[PASS] an unapproved image is rejected" in out
    assert "THIS VM's measurement" in out and "provision_td.py --inspect" in out


def test_a_collateral_outage_in_the_probe_is_a_skip_not_an_i8_failure(tdx, capsys):
    mod, verify, mp = tdx
    mp.setattr(verify, "verify_proof", lambda *a, **k: verify.VerifyResult(
        False, "DCAP collateral unavailable, so this proof was never judged: timeout",
        unverifiable=True))
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "[SKIP] the unapproved-image probe could not be judged" in out
    assert "I8 is not enforced" not in out


def test_an_unapproved_image_accepted_is_a_failure(tdx, capsys):
    mod, verify, mp = tdx
    mp.setattr(verify, "verify_proof", lambda *a, **k: verify.VerifyResult(True, "All checks passed"))
    assert mod.main() == 1
    assert "I8 is not enforced" in capsys.readouterr().out
