"""No collateral is fetched for a proof that fails a free local check.

The DCAP fetch is network, budgeted per epoch and shared by every miner
(I4, I6), and its target FMSPC is unsigned at parse time. Before 2026-11-02
it ran FIRST, so a replayed or stale proof — wrong nonce, wrong report_data,
wrong advertised hash — cost every validator a fetch that ten lines of
comparison would have refused. Now every local check precedes it, and this
file counts the fetches.
"""
from __future__ import annotations

from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.verify import verify_proof
from tests.test_unverifiable import GOLD, HOTKEY, _proof


def _run(monkeypatch, proof, **over):
    calls = []

    def counting(_q, **_kw):
        calls.append(1)
        return True

    monkeypatch.setattr(verify_mod, "verify_dcap", counting)
    kwargs = dict(approved_measurements=set(), expected_questions_hash=proof.questions_hash,
                  expected_nonce=proof.nonce, gold_answers=GOLD, expected_hotkey=HOTKEY,
                  head_bytes=proof.head_bytes_for_test, mock=False)
    kwargs.update(over)
    return verify_proof(proof, **kwargs), len(calls)


def test_a_locally_sound_proof_reaches_the_fetch_exactly_once(monkeypatch):
    r, n = _run(monkeypatch, _proof())
    assert n == 1
    # The empty approved set then refuses it: ordering weakened nothing.
    assert r.valid is False and "Unapproved" in r.reason


def test_wrong_nonce_costs_no_fetch(monkeypatch):
    r, n = _run(monkeypatch, _proof(), expected_nonce="cd" * 32)
    assert n == 0 and not r.valid and "Nonce" in r.reason


def test_wrong_advertised_hash_costs_no_fetch(monkeypatch):
    r, n = _run(monkeypatch, _proof(), expected_proof_hash="0" * 64)
    assert n == 0 and not r.valid and "Bundle mismatch" in r.reason


def test_tampered_body_costs_no_fetch(monkeypatch):
    p = _proof()
    p.results[0].correct = False            # report_data no longer matches
    r, n = _run(monkeypatch, p)
    assert n == 0 and not r.valid and "report_data" in r.reason


def test_relayed_proof_costs_no_fetch(monkeypatch):
    r, n = _run(monkeypatch, _proof(), expected_hotkey="5SomebodyElse")
    assert n == 0 and not r.valid and "bound to hotkey" in r.reason


def test_wrong_slice_costs_no_fetch(monkeypatch):
    r, n = _run(monkeypatch, _proof(), expected_question_ids={"q1", "q2"})
    assert n == 0 and not r.valid and "assigned slice" in r.reason
