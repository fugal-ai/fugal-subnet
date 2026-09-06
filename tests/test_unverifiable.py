"""'I could not check this' must not be spelled the same way as 'this is bad'.

Collapsing them is what makes a transient network failure indistinguishable
from a forged quote. It is also why a bare timeout on the collateral fetch
would have converted a PCCS hang into a simultaneous false accusation against
every honest miner in the field.

The dangerous direction is the other one: a miner who can REACH the unverifiable
branch becomes unscoreable at will — neither rewarded nor punished, absent from
the invalid count, invisible to the attack suite. Most of this file exists to
hold that line rather than to prove the happy path.
"""
import pytest

from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.attestation import CollateralUnavailable
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from fugal_subnet.tee.runtime import TEERuntime
from fugal_subnet.tee.verify import VerifyResult, verify_proof

HOTKEY = "5TestMinerHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def _proof():
    p = BenchmarkProof(
        hotkey=HOTKEY, epoch_id="e1", nonce="ab" * 32,
        questions_hash="c" * 64, weights_hash="d" * 64, source_hash="e" * 64,
        results=[QuestionResult(
            question_id="q1", routed_model="m", correct=True, cost_usd=0.01,
            response_hash="f" * 64, prompt_tokens=1, completion_tokens=1)],
        total_cost_usd=0.01, per_model_costs={"m": 0.01},
        attestation_quote=b"", timestamp=1.0,
    )
    p.attestation_quote = TEERuntime(mock=True).generate_attestation(
        bytes.fromhex(p.content_hash()))
    return p


def _verify(monkeypatch, dcap):
    """Run verify_proof outside mock with verify_dcap stubbed.

    Stubs take `**_kw` so they keep matching verify_dcap's real signature as it
    grows (it gained `budget` for the epoch-level collateral ceiling). Worth
    knowing WHY that matters: verify_proof wraps the call in `except Exception:
    -> invalid`, so a stub that does not accept a new argument raises TypeError
    and is reported as "DCAP attestation verification failed" -- our own
    signature error, recorded as the miner's fault, with a green-looking test
    name. The failure is silent in exactly the direction the unverifiable work
    exists to prevent.
    """
    monkeypatch.setattr(verify_mod, "verify_dcap", dcap)
    p = _proof()
    return verify_proof(
        p, approved_measurements=set(), expected_questions_hash=p.questions_hash,
        expected_nonce=p.nonce, gold_answers={}, expected_hotkey=HOTKEY, mock=False,
    )


def test_the_default_is_that_a_failure_is_the_miners_fault():
    """Every existing construction must keep the meaning it had, or this change
    silently reclassifies failures nobody revisited."""
    assert VerifyResult(False, "anything").unverifiable is False
    assert VerifyResult(True).unverifiable is False


def test_unreachable_collateral_is_unverifiable_not_invalid(monkeypatch):
    def boom(_q, **_kw):
        raise CollateralUnavailable("could not fetch DCAP collateral from https://x")
    r = _verify(monkeypatch, boom)
    assert r.unverifiable is True
    assert "never judged" in r.reason


def test_unverifiable_still_does_not_accept_the_proof(monkeypatch):
    """The whole risk of this change. Nothing was checked, so nothing is
    accepted — it only changes what the failure is CALLED."""
    def boom(_q, **_kw):
        raise CollateralUnavailable("network down")
    r = _verify(monkeypatch, boom)
    assert r.valid is False


def test_a_genuinely_bad_quote_is_invalid_and_never_unverifiable(monkeypatch):
    """THE I4 CASE. If a miner can reach the unverifiable branch they become
    unscoreable at will: no reward, no punishment, absent from the invalid
    count. Anything the miner's bytes could have caused must land here."""
    r = _verify(monkeypatch, lambda _q, **_kw: False)
    assert r.valid is False
    assert r.unverifiable is False


def test_an_exception_from_miner_bytes_is_invalid_and_never_unverifiable(monkeypatch):
    """A raise that is NOT CollateralUnavailable came from parsing what the
    miner sent. It must not be laundered into an infrastructure excuse."""
    def boom(_q, **_kw):
        raise ValueError("Failed to parse quote")
    r = _verify(monkeypatch, boom)
    assert r.valid is False
    assert r.unverifiable is False


def test_a_missing_verifier_still_raises_rather_than_being_unverifiable(monkeypatch):
    """ImportError is the same category but must stay fatal: a --live validator
    with no verifier installed should stop, not quietly mark the whole field
    unverifiable and carry on looking healthy."""
    def boom(_q, **_kw):
        raise ImportError("dcap_qvl not installed")
    with pytest.raises(ImportError):
        _verify(monkeypatch, boom)


def test_failures_after_the_dcap_step_are_never_unverifiable(monkeypatch):
    """Everything past the collateral fetch is judged locally from bytes the
    miner supplied, so no later failure may claim infrastructure."""
    p = _proof()
    p.results[0].question_id = "tampered-after-attestation"
    monkeypatch.setattr(verify_mod, "verify_dcap", lambda _q, **_kw: True)
    r = verify_proof(
        p, approved_measurements=set(), expected_questions_hash=p.questions_hash,
        expected_nonce=p.nonce, gold_answers={}, expected_hotkey=HOTKEY, mock=False,
    )
    assert r.valid is False
    assert r.unverifiable is False
