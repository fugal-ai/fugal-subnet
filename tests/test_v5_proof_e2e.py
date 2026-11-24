"""A v5 proof from the real TD path verifies under the validator's real v5 checks.

The TD (tee/harness, v5 path) hashes its own exam cells' questions into the
proof (`sorted(set(exam_qids))`, [] on a coverage-only epoch); the validator
must expect exactly that (tee/verify.questions_hash_v5 via
neurons/validator._v5_uid_checks), never the epoch's v4 scored slice. Found by
the real-chain rehearsal: every v5 proof failed "Questions hash mismatch".
No network, no spend: fake model calls, a mock attestation.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from fugal_subnet import exam_v5
from fugal_subnet.exam_v5 import ExamSpec
from fugal_subnet.tee import harness, td_v5
from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.proof import compute_questions_hash
from fugal_subnet.tee.runtime import TEERuntime
from neurons import validator
from tests import menu_fixtures as mf
from tests.test_harness_v4 import FIXTURE_IDS, HOTKEY, NONCE, Provider, _inputs, _Proxy
from tests.test_tee_e2e import _head, _pool

EPOCH = 102                      # the fixtures' menu is epoch 102's
OTHER = "5OtherHotkeyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
COVERAGE = {"q10": FIXTURE_IDS[2], "q11": FIXTURE_IDS[0]}
MINE = (("q1", FIXTURE_IDS[0]), ("q2", FIXTURE_IDS[1]), ("q2", FIXTURE_IDS[2]))
THEIRS = (("q3", FIXTURE_IDS[0]),)
SLICE_HASH = compute_questions_hash([f"q{i}" for i in range(12)])   # a v4 slice's, never a v5 proof's


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    mf.allow_small_menus(monkeypatch)


def _spec(mine=MINE, theirs=THEIRS) -> ExamSpec:
    assignment = {c: (HOTKEY, OTHER) for c in mine}
    assignment.update({c: (OTHER,) for c in theirs})
    cells = tuple(sorted(assignment))
    return ExamSpec(epoch=EPOCH, questions=tuple(sorted({c[0] for c in cells})), picks={},
                    baseline=FIXTURE_IDS[0], baseline_questions=(), cells=cells,
                    ring=tuple(sorted([(HOTKEY, "ck-mine"), (OTHER, "ck-other")])),
                    assignment=assignment)


def _td_proof(monkeypatch, spec: ExamSpec | None):
    """The frozen TD's own path: harness.run_benchmark(protocol_version=5)."""
    provider = Provider()
    pool = _pool()
    monkeypatch.setattr(harness, "_call_model", provider.call)
    # The same coverage draw on both sides (the TD's and expected_v5's).
    monkeypatch.setattr(td_v5, "coverage_v5",
                        lambda nonce, p, excluded, state, hotkey, index=None: dict(COVERAGE))
    task = (td_v5.ExamTask(epoch=EPOCH, spec=spec, spec_hash=spec.spec_hash(),
                           cells=tuple(sorted(exam_v5.assigned_cells(spec, HOTKEY))))
            if spec is not None else None)
    v4_inputs = _inputs(pool, epoch=EPOCH, challenge=None, fetch_listing=provider.fetch)
    inputs = td_v5.V5EpochInputs(epoch=EPOCH, v4=v4_inputs, exam=task)
    head = _head(FIXTURE_IDS)
    proof = harness.run_benchmark(
        nonce=NONCE, head_bytes=head, benchmark_pool=pool, proxy=_Proxy(port=0),
        hidden_states=np.zeros((len(pool), 1024), np.float32), slice_size=12,
        epoch_id=f"e{EPOCH:08d}", source_hash="img", hotkey=HOTKEY, protocol_version=5, v5=inputs)
    proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
        bytes.fromhex(proof.content_hash()))
    return proof, head, pool, inputs.menu()


def _validator_check(monkeypatch, proof, head, pool, state, spec):
    """The validator's real path: _v5_expected -> _v5_uid_checks -> _check_response."""
    exp = validator._v5_expected(state, epoch_index=EPOCH, nonce=bytes.fromhex(NONCE),
                                 pool=pool, hotkey=HOTKEY, spec=spec, plan=None)
    # The epoch-wide checks as the loop builds them: the v4 slice's hash and ids.
    checks = validator._EpochChecks(
        boundary_block=1_000, require_commitment=False, approved_measurements=set(),
        questions_hash=SLICE_HASH, nonce_hex=NONCE, slice_gold={},
        question_ids={f"q{i}" for i in range(12)}, explore_map={}, mock=True,
        collateral_budget=None, protocol_version=5, menu=None)
    by_id = {q["question_id"]: q for q in pool}
    uid_checks = validator._v5_uid_checks(checks, exp, {}, by_id)
    monkeypatch.setattr(validator, "_get_bundle_for_uid", lambda uid, resp: (proof, head))
    resp = SimpleNamespace(proof_hash=proof.content_hash(),
                           weights_hash=hashlib.sha256(head).hexdigest())
    return validator._check_response(0, resp, HOTKEY, ("", 0), 0, uid_checks), exp


def test_an_exam_proof_verifies_and_carries_its_own_cells_questions(monkeypatch):
    spec = _spec()
    proof, head, pool, state = _td_proof(monkeypatch, spec)
    assert proof.questions_hash == compute_questions_hash(["q1", "q2"]) != SLICE_HASH
    checked, exp = _validator_check(monkeypatch, proof, head, pool, state, spec)
    assert checked.status == "valid"
    assert verify_mod.questions_hash_v5(exp) == proof.questions_hash


def test_a_coverage_only_proof_verifies_with_the_empty_hash(monkeypatch):
    proof, head, pool, state = _td_proof(monkeypatch, None)
    assert proof.questions_hash == compute_questions_hash([])
    checked, _exp = _validator_check(monkeypatch, proof, head, pool, state, None)
    assert checked.status == "valid"


def test_a_proof_of_other_cells_than_this_hotkeys_is_refused(monkeypatch):
    # The TD answered the spec that gave it MINE; the validator's spec gives
    # this hotkey another hotkey's cell instead.
    proof, head, pool, state = _td_proof(monkeypatch, _spec())
    other = _spec(mine=THEIRS, theirs=MINE)
    checked, _exp = _validator_check(monkeypatch, proof, head, pool, state, other)
    assert checked.status == "invalid"


def test_verify_proof_never_holds_a_v5_proof_to_the_callers_slice_hash(monkeypatch):
    spec = _spec()
    proof, head, pool, state = _td_proof(monkeypatch, spec)
    exp = validator._v5_expected(state, epoch_index=EPOCH, nonce=bytes.fromhex(NONCE),
                                 pool=pool, hotkey=HOTKEY, spec=spec, plan=None)
    gold = validator._v5_gold({}, {q["question_id"]: q for q in pool}, exp)
    result = verify_mod.verify_proof(
        proof, approved_measurements=set(), expected_questions_hash=SLICE_HASH,
        expected_nonce=NONCE, gold_answers=gold, expected_exploration=exp.v4.exploration,
        expected_weights_hash=hashlib.sha256(head).hexdigest(), expected_hotkey=HOTKEY,
        expected_proof_hash=proof.content_hash(), head_bytes=head, mock=True,
        protocol_version=5, v5=exp)
    assert result.valid, result.reason
