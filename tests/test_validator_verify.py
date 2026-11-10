"""One miner's response, checked by the validator: commitment, bundle, proof.

This lived inline in neurons/validator.py's epoch loop, where 17 of 20 planted
bugs survived the whole suite (2026-11-08), among them: live verifying as
mock, the proof's own hotkey or weights hash trusted, an oversized head
accepted, a verifier crash ending the epoch. It is _check_response now, and
these tests check what it hands verify_proof, not only what it returns.
"""
import ast
import base64
import json
import math
import pathlib
from types import SimpleNamespace

import pytest

from fugal_subnet.commitments import seniority_key
from fugal_subnet.config import HEAD_MAX_BYTES
from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from neurons import validator
from neurons.validator import _check_response, _EpochChecks, _get_bundle_for_uid

HOTKEY = "5QueriedMinerHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
WHASH = "d" * 64
BOUNDARY = 3600


def _proof(hotkey="5SomebodyElseBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB", weights_hash="9" * 64):
    return BenchmarkProof(
        hotkey=hotkey, epoch_id="e00000010", nonce="ab" * 32, questions_hash="c" * 64,
        weights_hash=weights_hash, source_hash="e" * 64,
        results=[QuestionResult(question_id="q1", routed_model="m", correct=True,
                                cost_usd=0.01, response_hash="f" * 64,
                                prompt_tokens=10, completion_tokens=5)],
        total_cost_usd=0.01, per_model_costs={"m": 0.01}, attestation_quote=b"", timestamp=1.0,
    )


def _resp(proof=None, head=b"head-bytes", **kw):
    proof = proof or _proof()
    base = dict(proof_hash="ph" * 32, weights_hash=WHASH,
                proof_json=json.dumps(proof.to_dict()),
                head_npz_b64=base64.b64encode(head).decode())
    base.update(kw)
    return SimpleNamespace(**base)


def _checks(**kw):
    base = dict(boundary_block=BOUNDARY, require_commitment=True,
                approved_measurements={"approved"}, questions_hash="qh", nonce_hex="ab" * 32,
                slice_gold={"q1": {"question_id": "q1"}}, question_ids={"q1"},
                explore_map={"q2": "m"}, mock=False, collateral_budget="budget",
                protocol_version=2)
    base.update(kw)
    return _EpochChecks(**base)


class _Verifier:
    """Stands in for verify_proof and records what it was asked to check."""

    def __init__(self, valid=True, unverifiable=False, raises=None):
        self.calls, self.valid, self.unverifiable, self.raises = [], valid, unverifiable, raises

    def __call__(self, proof, **kw):
        self.calls.append((proof, kw))
        if self.raises:
            raise self.raises
        return SimpleNamespace(valid=self.valid, unverifiable=self.unverifiable,
                               reason="r", warnings=[])


@pytest.fixture
def verifier(monkeypatch):
    v = _Verifier()
    monkeypatch.setattr(verify_mod, "verify_proof", v)
    return v


def check(resp, committed=(WHASH, BOUNDARY - 10), checks=None, hotkey=HOTKEY, reg=7):
    return _check_response(5, resp, hotkey, committed, reg, checks or _checks())


# ------------------------------------------------- what verify_proof is asked --

def test_every_expectation_reaches_the_verifier(verifier):
    resp = _resp()
    out = check(resp)
    assert out.status == "valid"
    (proof, kw), = verifier.calls
    assert kw == dict(
        approved_measurements={"approved"},
        expected_questions_hash="qh",
        expected_nonce="ab" * 32,
        gold_answers={"q1": {"question_id": "q1"}},
        expected_question_ids={"q1"},
        expected_exploration={"q2": "m"},
        expected_weights_hash=WHASH,          # the hash committed on chain, not the proof's
        expected_hotkey=HOTKEY,               # the hotkey queried, not the proof's
        expected_proof_hash="ph" * 32,
        head_bytes=b"head-bytes",
        mock=False,
        collateral_budget="budget",
        protocol_version=2,
    )
    assert out.proof is proof and out.head_bytes == b"head-bytes"


def test_live_never_verifies_as_mock(verifier):
    check(_resp(), checks=_checks(mock=False))
    check(_resp(), checks=_checks(mock=True))
    assert [kw["mock"] for _, kw in verifier.calls] == [False, True]


# --------------------------------------------------------------- commitment --

def test_no_proof_offered_is_absent_not_invalid(verifier):
    assert check(None).status == "absent"
    assert check(_resp(proof_hash="")).status == "absent"
    assert not verifier.calls


def test_the_seniority_is_registration_then_commitment(verifier):
    assert check(_resp(), committed=(WHASH, 3000), reg=7).seniority == seniority_key(7, 3000)


@pytest.mark.parametrize("committed", [
    ("e" * 64, BOUNDARY - 10),          # a different head committed
    (WHASH, BOUNDARY + 1),              # committed after the boundary
    (WHASH, 0),                         # no commitment at all
])
def test_a_head_not_committed_by_the_boundary_is_rejected(verifier, committed):
    out = check(_resp(), committed=committed)
    assert out.status == "invalid" and out.seniority is None and not verifier.calls


def test_a_commitment_at_the_boundary_block_counts(verifier):
    assert check(_resp(), committed=(WHASH, BOUNDARY)).status == "valid"


def test_without_required_commitments_an_uncommitted_head_is_junior_to_all(verifier):
    out = check(_resp(), committed=("", 0), checks=_checks(require_commitment=False))
    assert out.status == "valid" and out.seniority == math.inf


def test_a_uid_with_no_hotkey_is_not_checked(verifier):
    assert check(_resp(), hotkey="").status == "invalid" and not verifier.calls


# ------------------------------------------------------------------ outcome --

def test_an_unverifiable_proof_is_not_counted_invalid(monkeypatch):
    monkeypatch.setattr(verify_mod, "verify_proof", _Verifier(valid=False, unverifiable=True))
    out = check(_resp())
    assert out.status == "unverifiable" and out.seniority is not None


def test_a_rejected_proof_is_invalid(monkeypatch):
    monkeypatch.setattr(verify_mod, "verify_proof", _Verifier(valid=False))
    assert check(_resp()).status == "invalid"


def test_a_verifier_crash_costs_the_miner_not_the_epoch(monkeypatch):
    monkeypatch.setattr(verify_mod, "verify_proof", _Verifier(raises=KeyError("boom")))
    assert check(_resp()).status == "invalid"


def test_a_missing_dependency_is_the_validators_problem_and_raises(monkeypatch):
    monkeypatch.setattr(verify_mod, "verify_proof", _Verifier(raises=ImportError("no dcap")))
    with pytest.raises(ImportError):
        check(_resp())


def test_a_bundle_parser_crash_costs_the_miner(verifier, monkeypatch):
    def boom(uid, resp):
        raise RuntimeError("parser bug")
    monkeypatch.setattr(validator, "_get_bundle_for_uid", boom)
    assert check(_resp()).status == "invalid" and not verifier.calls


# ------------------------------------------------------------------- bundle --

def test_a_well_formed_bundle_parses():
    proof, head = _get_bundle_for_uid(1, _resp(head=b"\x00\x01"))
    assert proof.weights_hash == "9" * 64 and head == b"\x00\x01"


@pytest.mark.parametrize("resp", [
    _resp(proof_json=""),                                   # incomplete
    _resp(head_npz_b64=""),                                 # incomplete
    _resp(proof_json="{not json"),                          # unparseable
    _resp(proof_json="[1, 2]"),                             # not a proof
    _resp(head_npz_b64="aGVhZA==!!"),                       # not strict base64
    _resp(head=b"x" * (HEAD_MAX_BYTES + 1)),                # over the size cap
])
def test_a_malformed_bundle_is_refused(resp):
    assert _get_bundle_for_uid(1, resp) is None


def test_a_head_at_the_size_cap_is_accepted():
    assert _get_bundle_for_uid(1, _resp(head=b"x" * HEAD_MAX_BYTES)) is not None


# ------------------------------------------ inline wiring, on the syntax tree --

def _main():
    tree = ast.parse(pathlib.Path(validator.__file__).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def test_the_loop_verifies_in_the_nonce_order_and_counts_each_outcome():
    main = _main()
    (loop,) = [n for n in ast.walk(main) if isinstance(n, ast.For)
               and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_check_response"
                       for c in ast.walk(n))]
    assert ast.unparse(loop.iter) == "order"
    body = ast.unparse(loop)
    assert "n_invalid += 1" in body and "n_unverifiable += 1" in body
    assert "head_hashes[uid] = checked.proof.weights_hash" in body
    # Dedup orders copies by seniority; without it every miner is equally
    # junior and a copier can outrank the author.
    assert "if checked.seniority is not None:\n        commit_blocks[uid] = checked.seniority" in body
    (checks,) = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "_EpochChecks"]
    kw = {k.arg: ast.unparse(k.value) for k in checks.keywords}
    assert kw["mock"] == "mock" and kw["require_commitment"] == "REQUIRE_COMMITMENT"
    assert kw["slice_gold"] == "slice_gold" and kw["protocol_version"] == "protocol_version"


def test_an_epoch_with_no_proofs_is_still_marked_done():
    """Not marking it would re-run the epoch every minute until it ended, and a
    proof that arrived late would then be scored by this validator alone."""
    main = _main()
    (branch,) = [n for n in ast.walk(main) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "not verified_proofs"]
    src = ast.unparse(branch)
    assert "last_epoch_index = epoch_index" in src
    assert src.index("last_epoch_index = epoch_index") < src.index("save_state(")


def test_a_v2_epoch_with_no_proofs_burns_instead_of_keeping_the_last_weights():
    """FABLE REVIEW, item 6 (2026-11-09): with no valid proof the validator
    skipped set_weights, so the chain kept its last vector. At the flip that
    paid the v1 winners until the first v2 proof verified. In v2 a seat pays
    only while its head is live, and with no proof no head is: the share
    burns. v1 epochs keep main's behaviour (agreement through the window)."""
    main = _main()
    (branch,) = [n for n in ast.walk(main) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "not verified_proofs"]
    src = ast.unparse(branch)
    assert "_burn_epoch_weights(" in src
    burn = src[src.index("if protocol_version >= 2:"):]
    # the vector it set is the next epoch's cap base, rebound at the commit point
    assert "burned = _burn_epoch_weights(" in burn[:burn.index("save_state(")]
    rebind = "prev_uids, prev_weights = burned or (prev_uids, prev_weights)"
    assert rebind in burn
    assert burn[burn.index(rebind) + len(rebind):].lstrip().startswith("save_state(")
    assert src.index("_burn_epoch_weights(") < src.index("save_state(")


def test_burn_epoch_weights_sets_the_burn_uid_and_never_raises():
    from types import SimpleNamespace

    import neurons.validator as v

    calls = []

    class Sub:
        def get_subnet_owner_hotkey(self, netuid, block=None):
            return "owner"

    mg = SimpleNamespace(hotkeys=["a", "owner", "b"])
    orig = v.set_weights_with_retry
    try:
        v.set_weights_with_retry = lambda *a, **k: (calls.append((a[4], a[5])) or (True, ""))
        out = v._burn_epoch_weights(Sub(), None, 1, 0, 100, mg, True, [], [], 5, 360)
        assert calls[-1] == ([1], [1.0]) and out == ([1], [1.0])
        # capped against the previous vector, like every epoch
        out = v._burn_epoch_weights(Sub(), None, 1, 0, 100, mg, True, [2], [1.0], 5, 360)
        assert calls[-1][0] == [1, 2] and abs(calls[-1][1][0] - 0.3) < 1e-9
        assert out == (calls[-1][0], calls[-1][1])
        # a failed submit keeps the previous vector as the cap's base
        v.set_weights_with_retry = lambda *a, **k: (False, "rate limited")
        assert v._burn_epoch_weights(Sub(), None, 1, 0, 100, mg, True, [2], [1.0], 5, 360) == ([2], [1.0])

        def boom(*a, **k):
            raise RuntimeError("chain down")
        v.set_weights_with_retry = boom
        assert v._burn_epoch_weights(Sub(), None, 1, 0, 100, mg, True, [2], [1.0], 5, 360) == ([2], [1.0])
        # an unreadable owner (live refuses to guess) never raises either
        bad = SimpleNamespace(hotkeys=["a"])
        assert v._burn_epoch_weights(Sub(), None, 1, 0, 100, bad, True, [2], [1.0], 5, 360) == ([2], [1.0])
    finally:
        v.set_weights_with_retry = orig
