"""Protocol v4: the memorisation package (S5) and the picks package (S3)
together (docs/decisions/SEAT_MINIMAL_V4.md 3.4, 3.7, 6.3).

S5's epoch plan (`epoch_plan.plan_v4(pool, published=, epoch=)`) draws the
scored slice from published questions only and puts the epoch's fresh
questions in every private set; S3 made the scored questions optional (the
proof's `scored` field) and bounds a proof's exploration failures
absolutely. Here, through the TD's own harness on such a plan:

- a proof with the scored questions on and one with them off both verify
  against one expectation, and the exploration each TD drew is the
  verifier's expectation, the epoch's fresh questions included;
- the absolute failure bound counts a failed fresh question like any other
  exploration call;
- a v4 window carrying both packages' state (labels, odds of heads with no
  answers, real-picks records) writes and reads back with every input the
  memorisation check reads, so the check gives the same evidence on the
  loaded window; a v1-v3 window is written as before;
- one v4 epoch given both inputs publishes both blocks and both digests,
  and each input adds only its own fields: without `memo` the reveal is the
  picks-only one byte for byte, and the other way round.
"""
from __future__ import annotations

import functools
import hashlib

import numpy as np
import pytest

from fugal_subnet import config, epoch_plan, labels_v4, memo_v4, menu_inputs, picks_v4, seat_v4
from fugal_subnet.memo_v4 import Published
from fugal_subnet.tee import harness
from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.proof import compute_questions_hash
from fugal_subnet.tee.runtime import TEERuntime
from tests import test_harness_v4 as h

EPOCH = 102                                   # the harness fixture's epoch
PER_SET = 900                                 # every category keeps 463+ published questions
NONCE = bytes.fromhex(h.NONCE)
HOTKEYS = ("5Fmemo-miner", "5Fmemo-other-a", "5Fmemo-other-b")


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    from tests import menu_fixtures as mf

    mf.allow_small_menus(monkeypatch)


@functools.lru_cache(maxsize=1)
def _world():
    """(pool, published set, the epoch's plan with it, hidden states): every
    set of the pinned v4 mix at PER_SET questions, two thirds published at
    epochs 1..98 by a keyed draw, folded through EPOCH - 1."""
    pool = [{"question_id": f"{b}_{i:04d}", "prompt": f"Question {i}?", "gold": str(i),
             "grader_id": "numeric_final", "benchmark": b, "metadata": {}}
            for b in sorted(epoch_plan.mix_v4().bench_category) for i in range(PER_SET)]
    first = {}
    for q in pool:
        k = int.from_bytes(hashlib.sha256(q["question_id"].encode()).digest()[:8], "big")
        if k % 3:
            first[q["question_id"]] = 1 + k % 98
    published = Published(first, EPOCH - 1)
    plan = epoch_plan.plan_v4(pool, published=published, epoch=EPOCH)
    hidden = np.random.default_rng(0).standard_normal((len(pool), 1024), dtype=np.float32)
    return pool, published, plan, hidden


def _fresh() -> set[str]:
    _pool, _p, plan, _h = _world()
    return {q["question_id"] for q in memo_v4.fresh_questions(
        NONCE, plan.pool_index, epoch_plan.mix_v4().targets)}


def _run(monkeypatch, hotkey: str, *, scored: bool, fail_ids=()):
    """The TD's harness on the epoch's published plan, for `hotkey`."""
    pool, _published, plan, hidden = _world()
    provider = h.Provider(fail_ids=fail_ids)
    monkeypatch.setattr(harness, "_call_model", provider.call)
    from fugal_subnet.tee import td_v4

    # "Scored": a 500-question challenge with no holder (the open form).
    inputs = h._inputs(pool, plan=plan, fetch_listing=provider.fetch,
                       challenge=td_v4.ChallengeSpec(500, None) if scored else None)
    proof = harness.run_benchmark(
        nonce=h.NONCE, head_bytes=h._head(h.FIXTURE_IDS), benchmark_pool=pool,
        proxy=h._Proxy(port=0), hidden_states=hidden, slice_size=config.SLICE_SIZE_V4,
        epoch_id=f"e{EPOCH:08d}", source_hash="img", hotkey=hotkey, protocol_version=4,
        v4=inputs)
    proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
        bytes.fromhex(proof.content_hash()))
    return proof, inputs.menu()


def _expected(state, hotkey: str):
    pool, _published, plan, _h = _world()
    slice_ids = [q["question_id"] for q in plan.scored_slice(NONCE, pool, config.SLICE_SIZE_V4)]
    return slice_ids, verify_mod.expected_v4(state, epoch=EPOCH, nonce=NONCE, pool=pool,
                                             scored_ids=slice_ids, hotkey=hotkey, plan=plan)


def _verify(proof, state):
    from neurons import validator

    pool = _world()[0]
    by_id = {q["question_id"]: q for q in pool}
    slice_ids, x = _expected(state, proof.hotkey)
    head = h._head(h.FIXTURE_IDS)
    gold = validator._v4_gold({q: by_id[q] for q in slice_ids}, by_id, x)
    return verify_mod.verify_proof(
        proof, approved_measurements=set(),
        expected_questions_hash=compute_questions_hash(slice_ids), expected_nonce=h.NONCE,
        gold_answers=gold, expected_question_ids=set(slice_ids),
        expected_weights_hash=hashlib.sha256(head).hexdigest(), expected_hotkey=proof.hotkey,
        expected_proof_hash=proof.content_hash(), head_bytes=head, mock=True,
        protocol_version=4, v4=x)


# --- (a) the published plan, with the scored questions on and off -------------------------------


def test_a_td_on_the_published_plan_verifies_with_the_scored_questions_on_and_off(monkeypatch):
    pool, published, plan, _h = _world()
    done = published.view(EPOCH - 2).first
    fresh = _fresh()
    assert 0 < len(fresh) == plan.pool_index.fresh_count
    on, state = _run(monkeypatch, HOTKEYS[0], scored=True)
    off, state_off = _run(monkeypatch, HOTKEYS[0], scored=False)
    assert state == state_off
    slice_ids, x = _expected(state, HOTKEYS[0])
    # The slice is drawn from published questions only, and drawn either way.
    assert len(slice_ids) == config.SLICE_SIZE_V4 and all(q in done for q in slice_ids)
    assert on.questions_hash == off.questions_hash == compute_questions_hash(slice_ids)
    assert on.scored is True and on.challenge.size == 500 and len(on.scored_results) == 500
    assert [r.question_id for r in on.scored_results][:len(slice_ids)] == slice_ids
    assert off.scored is False and off.scored_results == []
    # One expectation for both: the exploration each TD drew is the
    # verifier's, the fresh questions in it and nothing else unpublished.
    for proof in (on, off):
        drawn = {r.question_id: r.routed_model for r in proof.exploration_results}
        assert drawn == x.exploration
        assert fresh <= set(drawn) and set(drawn) - fresh <= set(done)
        result = _verify(proof, state)
        assert result.valid, result.reason


@pytest.mark.parametrize("scored", [True, False])
def test_the_failure_bound_counts_a_failed_fresh_question_like_any_other(monkeypatch, scored):
    """34% of 112 exploration calls is 38.08: 38 failed calls are within the
    bound, 39 are over it. The miner's failed calls are all of the fresh
    questions and enough published ones; two other operators' proofs are
    healthy, so the epoch is no outage. Counted without the fresh ones, the
    39 would be within the bound: the refusal is the fresh questions'."""
    _pool, _published, _plan, _h = _world()
    fresh = sorted(_fresh())
    n = config.EXPLORE_PRIVATE_SIZE_V4
    assert not picks_v4.over_bound(38, n) and picks_v4.over_bound(39, n)
    others = [_run(monkeypatch, hk, scored=False)[0] for hk in HOTKEYS[1:]]
    for k in (38, 39):
        state = _run(monkeypatch, HOTKEYS[0], scored=scored)[1]
        _slice, x = _expected(state, HOTKEYS[0])
        rest = sorted(set(x.exploration) - set(fresh))
        fail = set(fresh) | set(rest[:k - len(fresh)])
        miner, state = _run(monkeypatch, HOTKEYS[0], scored=scored, fail_ids=fail)
        ex = miner.exploration_results
        assert len(ex) == n and {r.question_id for r in ex if r.failed} == fail
        assert all(r.failed for r in ex if r.question_id in fresh)
        verdict = picks_v4.failure_refusals(
            [menu_inputs.VerifiedProof(p, f"ck-{p.hotkey}") for p in [miner, *others]], state)
        assert verdict.outage is False
        if k == 38:
            assert verdict.refused == {}
        else:
            assert verdict.refused == {HOTKEYS[0]: (39, n)}
            assert not picks_v4.over_bound(39 - len(fresh), n)
        for p in others:
            assert _verify(p, state).valid


# --- (b) the window file carries both packages' state ------------------------------------------


def test_a_v4_window_with_picks_and_labels_reads_back_with_the_same_memorisation_evidence(
        tmp_path):
    from tests import test_memo_v4_attacks as atk

    w, published = atk._window({"honest": atk.HONEST, "memoriser": atk.MEMORISER})
    cats = ("hard", "code", "math")
    w.picks = {e: picks_v4.PicksEpoch(e, cats, {"memoriser": (1, -2, 3)}, outage=(e == atk.LAST))
               for e in range(atk.FIRST, atk.LAST + 1, 7)}
    from fugal_subnet.v2_epoch import load_window, save_window

    path = str(tmp_path / "w.npz")
    save_window(path, w)
    back = load_window(path, 48)
    assert back.picks == w.picks
    assert sorted(back.odds) == sorted(w.odds) and back.heads == {} == w.heads
    now = atk.LAST
    for head in ("honest", "memoriser"):
        side = labels_v4.Side(head, "ckH", atk.SINCE)
        got = memo_v4.memo_evidence(back, labels_v4.reference(back, now=now), side, published,
                                    now=now)
        want = memo_v4.memo_evidence(w, labels_v4.reference(w, now=now), side, published, now=now)
        assert memo_v4.evidence_for_reveal(got) == memo_v4.evidence_for_reveal(want)
        assert (got.gap, got.bound, got.refused) == (want.gap, want.bound, want.refused)
    assert memo_v4.memo_evidence(back, labels_v4.reference(back, now=now),
                                 labels_v4.Side("memoriser", "ckH", atk.SINCE), published,
                                 now=now).refused


def test_a_v1_to_v3_window_file_holds_nothing_of_v4(tmp_path):
    """No picks record, no v4 labels, every head answered: the file's
    metadata and arrays are the keys a v3 window always had."""
    import json

    from fugal_subnet.v2_epoch import save_window
    from tests.test_v3_epoch import live

    results, _ = live(3)
    window = results[-1].window
    assert window.pairs and window.odds and not window.picks
    path = str(tmp_path / "w.npz")
    save_window(path, window)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        names = set(z.files)
    assert set(meta) == {"size", "slices", "heads", "pairs"}
    assert all(n == "meta" or n.split("_")[0] in {"g", "s", "r", "f", "p", "x"} for n in names)


# --- (c) run_v2_epoch with both inputs ---------------------------------------------------------


def _epoch(monkeypatch, *, picks: bool, memo: bool):
    """tests/test_v4_labels.run's epoch 30 (S5's attack case), with S3's
    picks input and S5's memo input each passed or not. Every served proof
    says it answered the scored questions (it carries no challenge, so none
    is counted: the record publishes every serving head's evidence)."""
    from types import SimpleNamespace

    from fugal_subnet import v2_epoch
    from tests import test_memo_v4_attacks as atk
    from tests import test_v4_labels as t4

    real = v2_epoch.run_v2_epoch

    def with_inputs(**kw):
        kw["verified_proofs"] = {u: SimpleNamespace(scored_results=p.scored_results, scored=True,
                                                    hotkey=f"hk{u}")
                                 for u, p in kw["verified_proofs"].items()}
        extra = {}
        if picks:
            extra["picks"] = v2_epoch.PicksInput(state=kw["odds_input"].menu_state)
        if memo:
            extra["memo"] = atk._memo(30, atk._state())
        return real(**kw, **extra)

    monkeypatch.setattr(t4, "run_v2_epoch", with_inputs)
    return t4.run(30, {5: "sharp-n1", 6: "flat-base", 7: "sharp-base"}, {},
                  committed={5: ("sharp-n1", 3), 6: ("flat-base", 3), 7: ("sharp-base", 3)})


def _without(reveal: dict, fields: set[str], digests: set[str]) -> dict:
    return {**{k: v for k, v in reveal.items() if k not in fields},
            "v2_state": {k: v for k, v in reveal["v2_state"].items() if k not in digests}}


def test_picks_and_memo_in_one_epoch_each_add_only_their_own_fields(monkeypatch):
    from fugal_subnet.v2_epoch import _digest_of

    plain = _epoch(monkeypatch, picks=False, memo=False)
    only_picks = _epoch(monkeypatch, picks=True, memo=False)
    only_memo = _epoch(monkeypatch, picks=False, memo=True)
    both = _epoch(monkeypatch, picks=True, memo=True)
    rv = both.reveal
    assert {"picks_v4", "memo_v4"} <= set(rv)
    assert rv["v2_state"][picks_v4.DIGEST_KEY] == picks_v4.digest(rv["picks_v4"])
    assert rv["v2_state"]["memo_digest"] == _digest_of(rv["memo_v4"])
    assert set(rv["picks_v4"]["evidence"]) == set(rv["memo_v4"]["evidence"]) == set(
        rv["v2_state"]["serving"])
    # The picks input is also what the v4 seat reads (S4: who answered, rule
    # B), so beside its own fields it moves the seat's evidence; the memo
    # input, with no head refused, moves nothing but its own fields.
    seat_fields, seat_digest = {seat_v4.REVEAL_KEY}, {seat_v4.DIGEST_KEY}
    picks_fields, picks_digest = {"picks_v4"}, {picks_v4.DIGEST_KEY}
    memo_fields, memo_digest = {"memo_v4"}, {"memo_digest"}
    assert _without(rv, memo_fields, memo_digest) == only_picks.reveal
    assert (_without(rv, picks_fields | seat_fields, picks_digest | seat_digest)
            == _without(only_memo.reveal, seat_fields, seat_digest))
    assert (_without(only_picks.reveal, picks_fields | seat_fields, picks_digest | seat_digest)
            == _without(plain.reveal, seat_fields, seat_digest))
    assert _without(only_memo.reveal, memo_fields, memo_digest) == plain.reveal
    assert rv[seat_v4.REVEAL_KEY] == only_picks.reveal[seat_v4.REVEAL_KEY]
    assert only_memo.reveal[seat_v4.REVEAL_KEY] == plain.reveal[seat_v4.REVEAL_KEY]
    # The memo input changes nothing the window keeps; the picks input adds its record.
    assert both.window.picks == only_picks.window.picks and 30 in both.window.picks
    assert only_memo.window.picks == plain.window.picks == {}
    assert sorted(both.window.odds) == sorted(plain.window.odds)
    assert both.weights == plain.weights
    assert both.decision == only_picks.decision and only_memo.decision == plain.decision
