"""Protocol v4's memorisation check under attack (SEAT_MINIMAL_V4.md 6,
10; fugal_subnet/memo_v4.py). Each case is refused, and each is shown to get
through once the one rule that refuses it is taken out:

- a head given an advantage on questions whose outcomes were published is
  refused once the evidence is in, while an honest head of the same overall
  skill is not (the seen/unseen split); an honest head whose skill differs
  by benchmark is not refused because the comparison is within benchmark;
- a miner cannot choose which of its questions are fresh: the fresh set is
  the epoch's, the same for every hotkey, and a proof that answers another
  set is refused by the verifier's exploration check;
- a validator holding a different published set is visible in the
  consensus view (the digest in `v2_state`), and the reveal of a v4 epoch
  carries the published set's digest, the fresh questions and every
  serving head's evidence.
"""
from __future__ import annotations

import functools
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from fugal_subnet import config, epoch_plan, labels_v4, memo_v4
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.labels_v4 import Side
from fugal_subnet.memo_v4 import Published
from fugal_subnet.tee import verify as verify_mod
from tests.test_memo_v4 import EPOCH, STATE, nonce, plan, published_state

# --- a field of labels: published questions and the epoch's fresh ones ------------------------

ENTRIES = tuple(f"m/{i}" for i in range(8))
OPERATORS = tuple(f"ck{i}" for i in range(8))            # the head's own is "ckH": never a voter
FIRST, LAST, SINCE = 100, 147, 90
SEEN_PER_EPOCH, FRESH_PER_EPOCH = 120, 6                  # 6 fresh questions x 8 operators


def _u(*parts) -> float:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8],
                          "big") / 2**64


@functools.lru_cache(maxsize=2)
def field(skew: bool = False):
    """(window pairs, published set): each epoch, SEEN_PER_EPOCH labels on
    questions published back in epoch 50 and FRESH_PER_EPOCH fresh questions
    answered by every operator on its own entry. Two benchmarks: `easy`
    (outcome right 80% of the time) and `hard` (30%), half and half; with
    `skew`, three in four published labels are `easy` and one in six fresh
    questions."""
    pairs, first = {}, {}
    for e in range(FIRST, LAST + 1):
        q, m, b, y, v = [], [], [], [], []
        for i in range(SEEN_PER_EPOCH):
            qid = f"s{e}_{i}"
            first[qid] = 50
            q.append(qid)
            m.append(ENTRIES[int(_u("m", qid) * 8)])
            b.append("easy" if (i % 4 if skew else i % 2) else "hard")
            v.append((OPERATORS[i % 8],))
        for j in range(FRESH_PER_EPOCH):
            qid = f"f{e}_{j}"
            first[qid] = e
            for k, ck in enumerate(OPERATORS):
                q.append(qid)
                m.append(ENTRIES[k])
                b.append("easy" if (j == 0 if skew else j % 2) else "hard")
                v.append((ck,))
        for qid, bench in zip(q, b):
            y.append(10000 if _u("y", e, qid, len(y)) < (0.8 if bench == "easy" else 0.3) else 0)
        n = len(q)
        base = EpochPairs(question_ids=tuple(q), models=tuple(m), benchmarks=tuple(b),
                          categories=("knowledge",) * n, y=np.array(y, np.int32),
                          prompt_tokens=np.zeros(n, np.int32), completion_tokens=np.zeros(n, np.int32))
        pairs[e] = labels_v4.label_pairs(base, v, np.full(n, -1, np.int16))
    return pairs, Published(first, LAST)


def _odds(pairs, knows):
    """A head's odds: on each label it 'knows' (knows(qid, bench) is the
    chance it does), the outcome at 9000 / 1000; otherwise the benchmark's
    base rate."""
    out = []
    for i, (qid, bench, y) in enumerate(zip(pairs.question_ids, pairs.benchmarks, pairs.y.tolist())):
        if _u("k", qid, i) < knows(qid, bench):
            out.append(9000 if y else 1000)
        else:
            out.append(8000 if bench == "easy" else 3000)
    return np.array(out, np.int16)


HONEST = lambda qid, bench: 0.35                                         # noqa: E731
MEMORISER = lambda qid, bench: 0.49 if qid.startswith("s") else 0.0      # noqa: E731
BY_BENCHMARK = lambda qid, bench: 0.9 if bench == "easy" else 0.0        # noqa: E731


def _window(heads, *, skew=False, merge_benchmarks=False):
    import dataclasses

    pairs, published = field(skew)
    w = Window(size=48, slices={}, heads={}, pairs=dict(pairs), odds={})
    for h, knows in heads.items():
        w.odds[h] = {e: _odds(pairs[e], knows) for e in pairs}
    if merge_benchmarks:
        w.pairs = {e: dataclasses.replace(p, benchmarks=("all",) * len(p.question_ids))
                   for e, p in pairs.items()}
    return w, published


def _upto(w, last):
    return Window(size=48, slices={}, heads={},
                  pairs={e: p for e, p in w.pairs.items() if e <= last},
                  odds={h: {e: o for e, o in per.items() if e <= last} for h, per in w.odds.items()})


def _evidence(h, w, published, now):
    ref = labels_v4.reference(w, now=now)
    return memo_v4.memo_evidence(w, ref, Side(h, "ckH", SINCE), published, now=now)


def _skill(h, w, now):
    ref = labels_v4.reference(w, now=now)
    s = labels_v4.skill_strata(w, ref, Side(h, "ckH", SINCE), now=now)
    return sum(x.total for x in s.values()) / sum(x.n for x in s.values()) / 1e6


def test_a_memoriser_is_refused_once_the_evidence_is_in_and_an_honest_head_is_not():
    w, published = _window({"honest": HONEST, "memoriser": MEMORISER})
    # the same overall skill over R, within a Brier point
    assert abs(_skill("honest", w, LAST) - _skill("memoriser", w, LAST)) < 1.0
    refused_at = None
    for now in range(FIRST, LAST + 1):
        wn = _upto(w, now)
        assert not _evidence("honest", wn, published, now).refused
        mem = _evidence("memoriser", wn, published, now)
        if mem.refused and refused_at is None:
            refused_at = now
            assert mem.unseen_labels >= memo_v4.MEMO_MIN_LABELS
    # 48 unseen labels an epoch: the minimum is in after 11 epochs, the refusal then
    assert refused_at is not None and refused_at - FIRST + 1 <= 12


def test_without_the_split_the_memoriser_is_not_refused():
    """Rule removed: every label counted unseen (no published set), so
    there is nothing to compare and the memoriser gets through."""
    w, published = _window({"memoriser": MEMORISER})
    assert _evidence("memoriser", w, published, LAST).refused
    assert not _evidence("memoriser", w, Published({}, LAST), LAST).refused


def test_an_honest_head_whose_skill_differs_by_benchmark_is_not_refused():
    """Skill on `easy` only, and the published labels lean to `easy` more
    than the fresh ones: within benchmark the gap is nothing. Rule removed
    (one benchmark for all), the mix is read as a gap and the honest head
    is refused."""
    w, published = _window({"byb": BY_BENCHMARK}, skew=True)
    assert not _evidence("byb", w, published, LAST).refused
    merged, _ = _window({"byb": BY_BENCHMARK}, skew=True, merge_benchmarks=True)
    assert _evidence("byb", merged, published, LAST).refused


# --- a miner cannot choose its fresh questions ------------------------------------------------


def _expected(p, pool, hotkey, the_plan=None):
    scored = [q["question_id"] for q in p.scored_slice(nonce(), pool, config.SLICE_SIZE_V4)]
    return scored, verify_mod.expected_v4(STATE, epoch=EPOCH, nonce=nonce(), pool=pool,
                                          scored_ids=scored, hotkey=hotkey,
                                          plan=the_plan or p)


def _proof(exploration: dict[str, str]):
    return SimpleNamespace(exploration_results=[
        SimpleNamespace(question_id=q, routed_model=m) for q, m in sorted(exploration.items())])


def test_the_fresh_questions_are_the_epochs_whoever_draws_them():
    pool, p = plan()
    fresh = {q["question_id"] for q in memo_v4.fresh_questions(nonce(), p.pool_index,
                                                                epoch_plan.mix_v4().targets)}
    for hk in ("5Fa", "5Fb", "5Fc", "5Fd", "5Fe"):
        scored, x = _expected(p, pool, hk)
        # the verifier's expectation is the TD's own draw (td_v4.exploration_v4)
        assert x.exploration == p.exploration(nonce(), pool, set(scored), STATE, hk)
        assert fresh <= set(x.exploration)
        assert fresh == set(x.exploration) - set(published_state().view(EPOCH - 2).first)


def test_a_proof_that_answers_other_questions_than_the_fresh_ones_is_refused():
    """A miner swapping a fresh question for one it prefers, or running the
    epoch without the published set (the old draw, no fresh questions), is
    refused by the exploration check. Rule removed (the verifier's
    expectation drawn without the published set), the no-fresh proof is
    accepted and the honest one refused."""
    pool, p = plan()
    scored, x = _expected(p, pool, "5Fminer")
    assert verify_mod._v4_exploration_check(_proof(x.exploration), x) is None
    done = published_state().view(EPOCH - 2).first
    fresh = sorted(q for q in x.exploration if q not in done)
    chosen = next(q["question_id"] for q in pool
                  if q["question_id"] in done and q["question_id"] not in x.exploration
                  and q["question_id"] not in scored)
    swapped = dict(x.exploration)
    swapped[chosen] = swapped.pop(fresh[0])
    assert "Exploration set mismatch" in verify_mod._v4_exploration_check(_proof(swapped), x)
    old_plan = epoch_plan.plan_v4(pool)
    old = old_plan.exploration(nonce(), pool, set(scored), STATE, "5Fminer")
    assert verify_mod._v4_exploration_check(_proof(old), x) is not None
    # rule removed: the verifier draws without the published set
    x_removed = verify_mod.expected_v4(STATE, epoch=EPOCH, nonce=nonce(), pool=pool,
                                       scored_ids=scored, hotkey="5Fminer", plan=old_plan)
    assert verify_mod._v4_exploration_check(_proof(old), x_removed) is None
    assert verify_mod._v4_exploration_check(_proof(x.exploration), x_removed) is not None


# --- the reveal and the consensus view ----------------------------------------------------------


def _small_pool():
    sets = sorted(epoch_plan.mix_v4().bench_category)
    return [{"question_id": f"{s}_{i:03d}", "benchmark": s, "prompt": "x"}
            for s in sets for i in range(40)]


def _memo(epoch, published):
    pool = _small_pool()
    index = memo_v4.reserve_index(published, pool, epoch)
    return memo_v4.epoch_memo(published, index, hashlib.sha256(f"n{epoch}".encode()).digest(),
                              epoch_plan.mix_v4().targets)


def _run(memo):
    from tests import test_v4_labels as t4

    return t4.run(30, {5: "sharp-n1", 6: "flat-base", 7: "sharp-base"}, {},
                  committed={5: ("sharp-n1", 3), 6: ("flat-base", 3), 7: ("sharp-base", 3)},
                  memo=memo)


@pytest.fixture
def run_with_memo(monkeypatch):
    """test_v4_labels.run, with the `memo` keyword passed to run_v2_epoch."""
    from fugal_subnet import v2_epoch
    from tests import test_v4_labels as t4

    real = v2_epoch.run_v2_epoch
    original_run = t4.run

    def run(*a, memo=None, **kw):
        def with_memo(**kwargs):
            return real(**kwargs, memo=memo)
        monkeypatch.setattr(t4, "run_v2_epoch", with_memo)
        try:
            return original_run(*a, **kw)
        finally:
            monkeypatch.setattr(t4, "run_v2_epoch", real)

    monkeypatch.setattr(t4, "run", run)
    return run


def _state(first_of_a=3):
    pool = _small_pool()
    first = {q["question_id"]: 2 + (i % 20) for i, q in enumerate(pool) if i % 3}
    first[pool[1]["question_id"]] = first_of_a
    return Published(first, 29)


def test_a_v4_reveal_carries_the_published_digest_the_fresh_questions_and_the_evidence(
        run_with_memo):
    memo = _memo(30, _state())
    rv = _run(memo).reveal
    doc = rv["memo_v4"]
    assert doc["through"] == 28 and doc["published_digest"] == _state().view(28).digest
    assert doc["fresh"] == list(memo.fresh) and doc["fresh_count"] == len(memo.fresh) > 0
    assert doc["reserve"] == memo.index.reserve_size
    assert set(doc["evidence"]) == set(rv["v2_state"]["serving"])
    for ev in doc["evidence"].values():
        assert set(ev) == {"seen", "unseen", "benchmarks", "gap_units", "lower_units", "refused"}
        assert ev["refused"] is False
    from fugal_subnet.v2_epoch import _digest_of
    assert rv["v2_state"]["memo_digest"] == _digest_of(doc)
    # without the published set the reveal is S2's, byte for byte
    plain = _run(None).reveal
    assert "memo_v4" not in plain and "memo_digest" not in plain["v2_state"]
    assert {**{k: v for k, v in rv.items() if k != "memo_v4"},
            "v2_state": {k: v for k, v in rv["v2_state"].items() if k != "memo_digest"}} == plain


def test_a_validator_with_a_different_published_set_shows_in_the_consensus_view(
        run_with_memo, monkeypatch):
    """Two validators whose published sets differ only in when one question
    was first published draw the same reserve and fresh questions, and
    their v2_state differs by the digest. Rule removed (the digest left out
    of the reveal), nothing shows."""
    a, b = _memo(30, _state(3)), _memo(30, _state(4))
    assert a.fresh == b.fresh and a.index.reserve_size == b.index.reserve_size
    ra, rb = _run(a).reveal, _run(b).reveal
    assert ra["memo_v4"]["published_digest"] != rb["memo_v4"]["published_digest"]
    assert ra["v2_state"] != rb["v2_state"]
    real = memo_v4.memo_for_reveal

    def without_digest(*args, **kw):
        doc = real(*args, **kw)
        doc.pop("published_digest")
        return doc

    monkeypatch.setattr(memo_v4, "memo_for_reveal", without_digest)
    assert _run(a).reveal["v2_state"] == _run(b).reveal["v2_state"]


def test_memorisation_fields_need_a_v4_epoch():
    from fugal_subnet.v2_epoch import run_v2_epoch

    with pytest.raises(ValueError, match="v4 epoch"):
        run_v2_epoch(**_v2_kwargs(), memo=_memo(30, _state()))


def _v2_kwargs():
    from fugal_subnet.champion import SeatState
    from fugal_subnet.copy_rule import CopyState
    from fugal_subnet.v2_epoch import seat_params_from_config
    from tests.test_v2_epoch import BPE, CATEGORY, MODELS, PRICES, questions

    return dict(epoch_index=30, epoch_id="e00000030", boundary_block=30 * BPE,
                blocks_per_epoch=BPE, questions=questions(30), pool_models=list(MODELS),
                prices=PRICES, verified_proofs={}, head_hashes={}, coldkey_of={}, hotkey_of={},
                records={}, commitments=[], window=Window.empty(48), seat=SeatState(),
                copy_state=CopyState(), burn_uid=0,
                params=seat_params_from_config(protocol_version=3),
                shares=(0.85, 0.10, 0.05), burn_in_questions=0, bench_category=CATEGORY)
