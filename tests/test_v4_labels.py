"""Protocol v4: the labels a head is scored on, and its odds on every epoch
(docs/decisions/SEAT_MINIMAL_V4.md 3.2 and 4; fugal_subnet/labels_v4.py).

- A comparison of X and Y reads a label only if it is in the last 48 epochs,
  was drawn after both rows it reads were committed, and was voted by
  neither head's owner.
- A head's odds are computed for every epoch from its commit, served or not:
  an epoch it missed is computed when it is next verified, from its bytes
  and the stored pairs, and published then.
- An entry's labels drawn while it is provisional score for no head
  (FAST_CROWN_V4.md 18); from the epoch it is full a head without the row is
  scored as odds 0; the label's own epoch decides.

The epochs below run the v4 path of `run_v2_epoch` (an `OddsInput` with the
epoch's `MenuState`) on a menu that changes from epoch to epoch.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from fugal_subnet import head_odds, labels_v4, seat_v4
from fugal_subnet import menu_state as ms
from fugal_subnet import routing_protocol as rp
from fugal_subnet.champion import EpochPairs, OddsParams, SeatState, Window
from fugal_subnet.copy_rule import CopyState
from fugal_subnet.explore_pool import ExploreAnswer
from fugal_subnet.labels_v4 import REFERENCE, LabelPairs, Side
from fugal_subnet.v2_epoch import (
    OddsInput,
    load_window,
    odds_from_b64,
    run_v2_epoch,
    save_window,
    seat_params_from_config,
)
from tests.test_v2_epoch import (
    ACC,
    BPE,
    CATEGORY,
    MODELS,
    PRICES,
    C,
    H,
    burned,
    oracle,
    proof,
    questions,
)
from tests.test_v3_epoch import BENCHES, DIM, difficulty_of, hidden_of

BASE = list(MODELS)                      # m/cheap, m/mid, m/strong: seed entries, no allowance
N1 = "m/new1"                            # sorts between m/mid and m/strong
FUTURE = "m/zz-future"                   # never on the menu
ACC_ALL = {**ACC, N1: 0.75, FUTURE: 0.5}
PRICES_ALL = {**PRICES, N1: (3e-6, 3e-6)}
# N1 is provisional for this many epochs from its join (its join epoch and
# the next: the step that scores the join epoch makes it full).
FULL_AFTER = ms.MENU_LAG


# --- menus ----------------------------------------------------------------------


def entry(eid: str, joined: int = 0, scored_from: int | None = None, rate: int = 5000) -> ms.Entry:
    return ms.Entry(id=eid, rev=1, model=eid, reasoning=None, max_tokens=4096,
                    provider_order=("host-a/fp8",), quantization="fp8", price_in=1000,
                    price_out=1000, mean_in=1000, mean_out=1000, token_pairs=0,
                    joined_epoch=joined, scored_from=joined if scored_from is None else scored_from,
                    measured_rate=rate, proposer="")


def menu_at(epoch: int, plan) -> ms.MenuState:
    """The menu of `epoch`: the base entries and every planned entry with
    `joined <= epoch < leaves`, provisional for FULL_AFTER epochs from its
    join and full from then."""
    entries = [entry(m) for m in BASE]
    records = []
    for eid, (joined, leaves) in plan.items():
        if joined <= epoch and (leaves is None or epoch < leaves):
            full = joined + FULL_AFTER
            e = entry(eid, joined, full if epoch >= full else ms.PROVISIONAL)
            entries.append(e)
            if e.provisional:
                records.append(ms.Provisional(eid, "ab" * 32, "", "ck-proposer"))
    return ms.MenuState(epoch=epoch, serial=1, entries=tuple(sorted(entries, key=lambda e: e.id)),
                        retired=(), labs=ms.AllowlistRef(0, ""), devices=ms.AllowlistRef(0, ""),
                        provisional=tuple(sorted(records, key=lambda r: r.id)))


# --- heads ----------------------------------------------------------------------


def _logit(p):
    return float(np.log(p / (1 - p)))


def sharp(models):
    W = np.zeros((len(models), DIM), dtype=np.float32)
    W[:, 0] = -60.0
    b = np.array([15.0 * _logit(ACC_ALL[m]) for m in models], dtype=np.float32)
    return W, b, list(models)


def flat(models):
    W = np.zeros((len(models), DIM), dtype=np.float32)
    b = np.array([_logit(ACC_ALL[m]) for m in models], dtype=np.float32)
    return W, b, list(models)


HEADS = {
    "flat-base": flat(BASE),
    "sharp-base": sharp(BASE),                     # no row for N1
    "sharp-n1": sharp(BASE + [N1]),
    "sharp-n1-future": sharp(BASE + [N1, FUTURE]),
}
UIDS = (5, 6, 7, 8)


def answers_for(epoch, menu):
    """Exploration from four operators ck5..ck8, every pair private and voted
    by one of them."""
    out, bench_of = [], {}
    for u in UIDS:
        for i in range(40):
            qid = f"p{epoch}_{u}_{i}"
            bench_of[qid] = BENCHES[(i + u) % 3]
            m = menu[(i + u) % len(menu)]
            out.append(ExploreAnswer(qid, m, f"ck{u}", difficulty_of(qid) < ACC_ALL[m],
                                     failed=False, private=True, prompt_tokens=20,
                                     completion_tokens=200))
    return out, bench_of


def params():
    p = dataclasses.replace(seat_params_from_config(protocol_version=3), min_paired_questions=60)
    return dataclasses.replace(p, odds=OddsParams(margin_points=0.1, min_pairs=200,
                                                  nv_noninferiority=0.5, ref_prior=20,
                                                  confirm_pairs=200))


def run(epoch, served, plan, *, window=None, seat=None, copy=None, committed,
        answering=None, bpe=BPE):
    """One v4 epoch on the menu `plan` gives it. `served` maps uid -> head
    name: the proofs verified this epoch. `committed` maps uid -> (head name,
    commit block): the chain's commitments, whether or not that uid served.
    `answering` (default: every served uid) answered the scored questions."""
    state = menu_at(epoch, plan)
    pool = list(state.ids())
    answers, bench_of = answers_for(epoch, pool)
    uids = sorted(served)
    answering = set(uids if answering is None else answering)
    oi = OddsInput(nonce=hashlib.sha256(f"n{epoch}".encode()).digest(), answers=answers,
                   benchmark_of=bench_of, heads={u: HEADS[served[u]] for u in uids},
                   hidden_of=hidden_of, single_model_min_pairs=30,
                   category_targets={"knowledge": 0.4, "math": 0.3, "hard": 0.3},
                   menu_state=state)
    hashes = {u: H(served[u]) for u in uids}

    def prf(u):
        p = proof(epoch, oracle)
        return p if u in answering else type(p)(scored_results=[])

    return run_v2_epoch(
        epoch_index=epoch, epoch_id=f"e{epoch:08d}", boundary_block=epoch * bpe,
        blocks_per_epoch=bpe, questions=questions(epoch), pool_models=pool,
        prices={m: PRICES_ALL[m] for m in pool},
        verified_proofs={u: prf(u) for u in uids}, head_hashes=hashes,
        coldkey_of={u: f"ck{u}" for u in uids}, hotkey_of={u: f"hk{u}" for u in uids},
        records={u: burned(hashes[u]) for u in uids},
        commitments=[C(u, H(name), block) for u, (name, block) in sorted(committed.items())],
        window=window or Window.empty(48), seat=seat or SeatState(),
        copy_state=copy or CopyState(), burn_uid=0, params=params(),
        shares=(0.85, 0.10, 0.05), burn_in_questions=0, bench_category=CATEGORY,
        odds_input=oi)


def live(first, last, plan, schedule, *, answering=lambda e: None):
    """Epochs first..last. `schedule(e)` maps uid -> (head name, served?):
    every uid listed is committed; the served ones are verified. A head is
    committed in the epoch before the first one it is listed in."""
    window = seat = copy = None
    blocks: dict = {}
    results = []
    for e in range(first, last + 1):
        sched = schedule(e)
        for u, (name, _) in sched.items():
            blocks.setdefault((u, name), (e - 1) * BPE + 7)
        committed = {u: (name, blocks[(u, name)]) for u, (name, _) in sched.items()}
        served = {u: name for u, (name, on) in sched.items() if on}
        r = run(e, served, plan, window=window, seat=seat, copy=copy, committed=committed,
                answering=answering(e))
        window, seat, copy = r.window, r.decision.state, r.copy_state
        results.append(r)
    return results


def on(*names):
    return {u: (n, True) for u, n in zip(UIDS, names)}


def entry_mask(pairs, eid):
    return np.array([m == eid for m in pairs.models])


# --- the labels: each condition ------------------------------------------------------


def _labels(epoch_y=((1, 7000), (2, 3000), (3, 9000), (4, 5000))):
    """One epoch of four labels on two entries, each voted by one operator."""
    ys = [y for _, y in epoch_y]
    n = len(ys)
    pairs = EpochPairs(question_ids=tuple(f"q{i}" for i in range(n)),
                       models=("m/a", "m/b", "m/a", "m/b"), benchmarks=("b",) * n,
                       categories=("knowledge", "math", "knowledge", "math"),
                       y=np.array(ys, dtype=np.int32), prompt_tokens=np.zeros(n, np.int32),
                       completion_tokens=np.zeros(n, np.int32))
    return labels_v4.label_pairs(pairs, [("ckX",), ("ckY",), ("ckZ",), ("ckY", "ckZ")],
                                 [-1, -1, -1, -1])


def test_a_label_drawn_before_either_rows_commit_is_not_read():
    lp = _labels()
    x, y = Side("hx", "ckA", 10), Side("hy", "ckB", 12)
    assert not labels_v4.eligible(lp, 11, x, y, now=20).any()          # before y's commit
    assert not labels_v4.eligible(lp, 9, x, y, now=20).any()           # before both
    assert labels_v4.eligible(lp, 12, x, y, now=20).all()              # the commit epoch on
    # A row a head inherited counts from its own epoch (the seat package's succession).
    x2 = Side("hx", "ckA", 15, rows_since={"m/a": 8})
    assert labels_v4.eligible(lp, 12, x2, y, now=20).tolist() == [True, False, True, False]


def test_a_label_its_own_operator_voted_on_is_never_read_for_its_head():
    lp = _labels()
    x, y = Side("hx", "ckX", 0), Side("hy", "ckB", 0)
    assert labels_v4.eligible(lp, 5, x, y, now=5).tolist() == [False, True, True, True]


def test_a_label_the_other_heads_operator_voted_on_is_not_read_either():
    lp = _labels()
    x, y = Side("hx", "ckA", 0), Side("hy", "ckZ", 0)
    # ckZ voted on the third label and, with ckY, on the fourth.
    assert labels_v4.eligible(lp, 5, x, y, now=5).tolist() == [True, True, False, False]
    # Against R only X's operator is left out, and R has no commit.
    assert labels_v4.eligible(lp, 5, Side("hx", "ckY", 0), REFERENCE, now=5).tolist() == [
        True, False, True, False]


def test_a_label_outside_the_last_48_epochs_is_not_read():
    lp = _labels()
    x, y = Side("hx", "ckA", 0), Side("hy", "ckB", 0)
    assert labels_v4.SEAT_WINDOW_EPOCHS == 48
    assert labels_v4.eligible(lp, 100 - 47, x, y, now=100).all()
    assert not labels_v4.eligible(lp, 100 - 48, x, y, now=100).any()
    assert not labels_v4.eligible(lp, 101, x, y, now=100).any()        # not yet drawn


def test_a_side_is_a_head_with_an_owner_and_a_commit_or_the_reference():
    for bad in (lambda: Side("h"), lambda: Side("h", "ck"), lambda: Side(None, "ck"),
                lambda: Side(None, since=3)):
        with pytest.raises(ValueError):
            bad()


def test_a_label_epoch_without_voters_is_refused():
    plain = _labels()
    v3 = EpochPairs(**{f.name: getattr(plain, f.name) for f in dataclasses.fields(EpochPairs)})
    with pytest.raises(ValueError, match="v4"):
        labels_v4.eligible(v3, 5, Side("h", "ck", 0), REFERENCE, now=5)
    with pytest.raises(ValueError, match="voters"):
        labels_v4.label_pairs(v3, [("b", "a")] * 4, [-1] * 4)          # unsorted voters
    with pytest.raises(ValueError, match="mask"):
        labels_v4.label_pairs(dataclasses.replace(v3, scored=np.ones(4, bool)), [("a",)] * 4,
                              [-1] * 4)


def test_the_items_are_the_weighted_brier_gain_in_exact_integers_and_their_sums_by_category():
    lp = dataclasses.replace(_labels(), weight=np.array([1_000_000, 2_000_000, 500_000, 1_000_000],
                                                        dtype=np.int64))
    window = Window(size=48, slices={}, heads={}, pairs={5: lp},
                    odds={"hx": {5: np.array([6000, 2000, 9900, -1], np.int16)},
                          "hy": {5: np.array([5000, 5000, 5000, 5000], np.int16)}})
    ref = labels_v4.reference(window, now=5)
    x, y = Side("hx", "ckA", 0), Side("hy", "ckB", 0)
    items = labels_v4.gain_items(window, ref, x, y, now=5)
    # loss_y - loss_x, weighted half up. The fourth: x has no row there
    # (missing -1), so it is scored as odds 0.
    assert items == {"knowledge": [(5000 - 7000) ** 2 - (6000 - 7000) ** 2,
                                   ((5000 - 9000) ** 2 - (9900 - 9000) ** 2) // 2],
                     "math": [2 * ((5000 - 3000) ** 2 - (2000 - 3000) ** 2),
                              (5000 - 5000) ** 2 - (0 - 5000) ** 2]}
    s = labels_v4.gain_strata(window, ref, x, y, now=5)
    assert s["knowledge"].n == 2 and s["knowledge"].total == sum(items["knowledge"])
    assert s["math"].squares == sum(v * v for v in items["math"])
    assert all(type(v) is int for vs in items.values() for v in vs)
    # A head the window holds no odds of for an epoch reads nothing there yet.
    assert labels_v4.gain_items(window, ref, Side("hz", "ckC", 0), y, now=5) == {}


# --- odds for every epoch, served or not ------------------------------------------------


def _miss_34(e):
    return {5: ("flat-base", True), 6: ("sharp-n1", e not in (3, 4)), 7: ("sharp-base", True)}


def _serve_all(e):
    return {5: ("flat-base", True), 6: ("sharp-n1", True), 7: ("sharp-base", True)}


def test_odds_for_an_epoch_a_head_missed_equal_those_it_would_have_had_served():
    plan = {N1: (2, None)}
    missed = live(1, 6, plan, _miss_34)
    served = live(1, 6, plan, _serve_all)
    h = H("sharp-n1")
    # While it is away the window holds nothing of it for those epochs ...
    assert 3 not in missed[2].window.odds[h] and 4 not in missed[3].window.odds[h]
    # ... and when it is next verified (epoch 5) both are computed and published.
    rv = missed[4].reveal
    assert [e for e, _ in rv["late_odds"][h]] == [3, 4]
    for e in (3, 4):
        assert np.array_equal(missed[-1].window.odds[h][e], served[-1].window.odds[h][e])
        n = len(missed[-1].window.pairs[e].question_ids)
        assert np.array_equal(odds_from_b64(dict(rv["late_odds"][h])[e], n),
                              served[-1].window.odds[h][e])
    # Nothing is computed twice, and a head that never missed has no late odds.
    assert "late_odds" in rv and set(rv["late_odds"]) == {h}
    assert missed[5].reveal["late_odds"] == {}
    # Its evidence over R is what it would have been.
    ref_m = labels_v4.reference(missed[-1].window, now=6)
    ref_s = labels_v4.reference(served[-1].window, now=6)
    side = Side(h, "ck6", 1)
    assert (labels_v4.skill_strata(missed[-1].window, ref_m, side, now=6)
            == labels_v4.skill_strata(served[-1].window, ref_s, side, now=6))


def test_a_head_is_given_odds_whether_or_not_it_answered_the_scored_questions():
    plan = {N1: (2, None)}
    res = live(1, 3, plan, _serve_all, answering=lambda e: (5, 7))
    h = H("sharp-n1")
    assert sorted(res[-1].window.odds[h]) == [1, 2, 3]
    assert h not in res[-1].window.heads                       # no scored answers
    assert h in res[-1].reveal["head_odds"]


def test_late_odds_start_at_the_heads_commit():
    plan = {N1: (2, None)}

    def sched(e):
        out = {5: ("flat-base", True)}
        if e >= 3:                         # committed in epoch 2, verified from epoch 5
            out[6] = ("sharp-n1", e >= 5)
        return out

    res = live(1, 6, plan, sched)
    h = H("sharp-n1")
    assert sorted(res[-1].window.odds[h]) == [3, 4, 5, 6]      # never before its commit epoch
    assert [e for e, _ in res[4].reveal["late_odds"][h]] == [3, 4]


# --- new entries: provisional labels score for no head; then odds 0 ------------------------


def test_provisional_labels_score_for_no_head_and_a_missing_row_is_odds_zero_after():
    """N1 joined at epoch 2, provisional through 3 and full from 4
    (FULL_AFTER): its labels of 2 and 3 are read by no comparison; from 4 a
    head without the row is scored as odds 0 ("never answers correctly"),
    and a head with the row on its own odds throughout."""
    plan = {N1: (2, None)}
    res = live(1, 6, plan, lambda e: on("flat-base", "sharp-base", "sharp-n1"))
    win = res[-1].window
    no_row, row = H("sharp-base"), H("sharp-n1")
    for e, want in ((2, labels_v4.PROVISIONAL_LABEL), (3, labels_v4.PROVISIONAL_LABEL),
                    (4, -1), (5, -1), (6, -1)):
        mask = entry_mask(win.pairs[e], N1)
        assert mask.any()
        assert (win.pairs[e].missing[mask] == want).all(), e
        assert (win.odds[no_row][e][mask] == -1).all(), e           # stored: no row
        assert (win.odds[row][e][mask] >= 0).all()
        assert (win.pairs[e].missing[~mask] == -1).all()           # the full entries
    assert all(win.pairs[e].scored is None for e in win.pairs)
    ref = labels_v4.reference(win, now=6)
    a, b = Side(row, "ck7", 1), Side(no_row, "ck6", 1)
    for e in (2, 3):
        read = labels_v4.eligible(win.pairs[e], e, a, b, now=6)
        assert not read[entry_mask(win.pairs[e], N1)].any() and read.any()
    # Odds 0 costs the head without the row against the one with it.
    late = {e: win.pairs[e] for e in (4, 5, 6)}
    gain = labels_v4.gain_items(dataclasses.replace(win, pairs=late), ref, a, b, now=6)
    assert sum(map(sum, gain.values())) > 0


def test_an_omitted_row_scores_exactly_as_a_row_that_says_zero():
    """Odds 0 is reachable by a row with zero weights and a bias below the
    first threshold: the head that omits the row and the head that commits
    the zero row are scored alike on every label a comparison reads."""
    table = head_odds.thresholds()
    bias = np.float32(table[0] - 1.0)
    assert head_odds.odds(np.zeros(DIM, np.float32), bias, hidden_of("x")) == 0
    W, b, m = flat(BASE)
    HEADS["zero-n1"] = (np.vstack([W, np.zeros((1, DIM), np.float32)]),
                        np.append(b, bias).astype(np.float32), m + [N1])
    HEADS["omits-n1"] = flat(BASE)
    try:
        res = live(1, 6, {N1: (2, None)},
                   lambda e: {5: ("zero-n1", True), 6: ("omits-n1", True)})
    finally:
        del HEADS["zero-n1"], HEADS["omits-n1"]
    win = res[-1].window
    ref = labels_v4.reference(win, now=6)
    zero, omits = H("zero-n1"), H("omits-n1")
    for e in (4, 5, 6):
        assert np.array_equal(labels_v4._scored(win, ref, Side(zero, "ck5", 1), e),
                              labels_v4._scored(win, ref, Side(omits, "ck6", 1), e))


def test_a_head_that_adds_the_row_by_succession_is_scored_on_its_own_row_from_its_commit():
    plan = {N1: (2, None)}

    def sched(e):
        # uid 6 serves a head without N1's row, and from epoch 5 a new head with it.
        return {5: ("flat-base", True), 6: ("sharp-base" if e <= 4 else "sharp-n1", True)}

    res = live(1, 7, plan, sched)
    win = res[-1].window
    old, new = H("sharp-base"), H("sharp-n1")
    # The new head is the old one's successor (the same hotkey and coldkey,
    # the base rows byte for byte, one row added): its inherited rows count
    # from its predecessor's; its new row counts from its own commit (epoch
    # 5), and N1's labels before it are read for it by nobody.
    hs = res[-1].decision.state.head(new)
    assert hs.since == 5 and set(hs.rows_since) == set(BASE) and hs.predecessors == (old,)
    # Before its commit its odds are its predecessor's stored ones, read, not
    # computed again (labels_v4.predecessor_odds): no late odds for them, and
    # on every label it is scored on they equal its own bytes' odds.
    assert sorted(win.odds[new]) == [5, 6, 7] and min(win.odds[old]) < 5
    chain = seat_v4.side_of(hs)
    for e in sorted(set(win.odds[old]) - {5, 6, 7}):
        assert labels_v4.predecessor_odds(win, chain, e) is win.odds[old][e]
        own = labels_v4.odds_on(HEADS["sharp-n1"], win.pairs[e], hidden_of)
        read = labels_v4.eligible(win.pairs[e], e, chain, REFERENCE, now=7,
                                  raw_x=labels_v4._raw(win, chain, e))
        # The row it added (N1) is read from the line's start, at odds 0 before
        # its own commit (the line had no row); every other label it reads is
        # its predecessor's odds, equal to its own bytes'.
        added = np.array([m in chain.added for m in win.pairs[e].models])
        assert chain.added == frozenset({N1})
        assert read.any() and np.array_equal(own[read & ~added], win.odds[old][e][read & ~added])
        assert (labels_v4._scored(win, None, chain, e)[read & added] == 0).all()
    for e in (5, 6, 7):
        mask = entry_mask(win.pairs[e], N1)
        assert (win.odds[new][e][mask] >= 0).all()
    ref = labels_v4.reference(win, now=7)
    side = Side(new, "ck6", hs.since, dict(hs.rows_since))
    read = labels_v4.eligible(win.pairs[4], 4, side, REFERENCE, now=7)
    assert not read[entry_mask(win.pairs[4], N1)].any() and read.any()
    assert labels_v4.skill_strata(win, ref, side, now=7)


def test_an_entry_retired_while_provisional_leaves_labels_no_one_is_scored_on():
    plan = {N1: (2, 4)}                                  # retired by its gates at 4
    res = live(1, 6, plan, _miss_34)
    win = res[-1].window
    assert all(N1 not in win.pairs[e].models for e in (4, 5, 6))
    for e in (2, 3):
        mask = entry_mask(win.pairs[e], N1)
        assert (win.pairs[e].missing[mask] == labels_v4.PROVISIONAL_LABEL).all()
    # The head that missed epoch 3 gets its late odds there, though the entry
    # is no longer on the menu it is verified on.
    h = H("sharp-n1")
    assert np.array_equal(win.odds[h][3], labels_v4.odds_on(HEADS["sharp-n1"], win.pairs[3],
                                                            hidden_of))


def test_a_row_for_an_entry_not_on_the_menu_is_ignored():
    res = live(1, 4, {N1: (2, None)}, lambda e: on("flat-base", "sharp-n1", "sharp-n1-future"))
    win = res[-1].window
    for e in win.pairs:
        assert np.array_equal(win.odds[H("sharp-n1")][e], win.odds[H("sharp-n1-future")][e])


def test_v4_has_no_grace_and_v3_menu_b_keeps_its_own():
    s = menu_at(5, {N1: (4, None)})
    for call in (rp.grace_excluded, rp.scored_menu):
        with pytest.raises(ValueError, match="no grace"):
            call(4, 5 * BPE, None, BPE, 0, menu_state=s)
        with pytest.raises(ValueError, match="no grace"):
            call(3, 5 * BPE, None, BPE, 0, menu_state=s)
    blk, bpe = 9_531_000, 360
    assert rp.grace_excluded(3, blk, blk, bpe, 20) == frozenset(rp.added_entries())
    assert rp.grace_excluded(3, blk + 20 * bpe, blk, bpe, 20) == frozenset()
    assert len(rp.scored_menu(3, blk, blk, bpe, 20)) == 53
    # Nor does a v4 odds input take v3's full-menu rule or grace set.
    from fugal_subnet.v2_epoch import _pool_and_odds
    for over in (dict(menu=s.ids()), dict(unscored=frozenset({N1}))):
        oi = OddsInput(nonce=b"\x01" * 32, answers=[], benchmark_of={}, heads={},
                       hidden_of=hidden_of, menu_state=s, **over)
        with pytest.raises(ValueError, match="no grace"):
            _pool_and_odds(Window.empty(8), oi, {}, {}, None)


# --- what is stored, published and rebuilt ------------------------------------------------


def test_the_saved_window_round_trips_labels_and_odds_of_heads_with_no_answers(tmp_path):
    res = live(1, 5, {N1: (2, None)}, _miss_34, answering=lambda e: (5,))
    win = res[-1].window
    assert H("sharp-n1") not in win.heads                     # only odds, no answers
    path = str(tmp_path / "w.npz")
    save_window(path, win)
    got = load_window(path, 48)
    assert sorted(got.pairs) == sorted(win.pairs)
    for e, p in win.pairs.items():
        q = got.pairs[e]
        assert isinstance(q, LabelPairs) and q.voters == p.voters
        assert np.array_equal(q.missing, p.missing) and np.array_equal(q.y, p.y)
    assert {h: sorted(x) for h, x in got.odds.items()} == {h: sorted(x) for h, x in win.odds.items()}
    for h, per in win.odds.items():
        for e, o in per.items():
            assert np.array_equal(got.odds[h][e], o)


def _reveal(epoch, result, served, plan):
    pool = list(menu_at(epoch, plan).ids())
    index = {m: i for i, m in enumerate(pool)}
    decisions = {}
    for u in served:
        rs = sorted(proof(epoch, oracle).scored_results, key=lambda r: r.question_id)
        decisions[str(u)] = [index[r.routed_model] for r in rs]
    return json.loads(json.dumps({"epoch_id": f"e{epoch:08d}", "protocol_version": 4,
                                  "questions": questions(epoch), "models": pool,
                                  "routing_decisions": decisions, **result.reveal}))


def test_the_rebuild_from_reveals_equals_the_live_window(tmp_path):
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals

    plan = {N1: (2, None)}
    res = live(1, 6, plan, _miss_34)
    for e, r in zip(range(1, 7), res):
        served = [u for u, (_, s) in _miss_34(e).items() if s]
        d = tmp_path / f"e{e:08d}"
        d.mkdir()
        (d / "reveal.json").write_text(json.dumps(_reveal(e, r, served, plan)))
    # Each epoch's own menu prices its slice: give the rebuild the union's table.
    prices = {m: PRICES_ALL[m] for m in BASE + [N1]}
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=prices,
                                 bench_category=CATEGORY)
    live_w = res[-1].window
    assert sorted(rb.window.pairs) == sorted(live_w.pairs)
    for e, p in live_w.pairs.items():
        q = rb.window.pairs[e]
        assert isinstance(q, LabelPairs)
        assert (q.question_ids, q.models, q.voters) == (p.question_ids, p.models, p.voters)
        assert np.array_equal(q.missing, p.missing) and np.array_equal(q.y, p.y)
    assert {h: sorted(x) for h, x in rb.window.odds.items()} == {
        h: sorted(x) for h, x in live_w.odds.items()}
    for h, per in live_w.odds.items():
        for e, o in per.items():
            assert np.array_equal(rb.window.odds[h][e], o)
    # The labels' digest is in the published state.
    assert len(res[-1].reveal["v2_state"]["labels_digest"]) == 64


def test_a_reveal_whose_late_odds_repeat_an_epoch_is_refused(tmp_path):
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals

    plan = {N1: (2, None)}
    res = live(1, 5, plan, _miss_34)
    reveals = []
    for e, r in zip(range(1, 6), res):
        served = [u for u, (_, s) in _miss_34(e).items() if s]
        reveals.append(_reveal(e, r, served, plan))
    h = H("sharp-n1")
    reveals[-1]["late_odds"][h].append([2, reveals[1]["head_odds"][h]])   # epoch 2 was served
    for rv in reveals:
        d = tmp_path / rv["epoch_id"]
        d.mkdir()
        (d / "reveal.json").write_text(json.dumps(rv))
    with pytest.raises(ValueError, match="already holds"):
        rebuild_v2_from_reveals(tmp_path, window_size=48,
                                prices={m: PRICES_ALL[m] for m in BASE + [N1]},
                                bench_category=CATEGORY)


# --- v3 is untouched ----------------------------------------------------------------

# sha256 of the reveals and weights of tests/test_v3_menu_b.py's run across the
# menu B block and its grace, computed on the tree before per-entry grace
# existed. v3 epochs score byte for byte as they did.
V3_MENU_B_DIGEST = "7dcfcacdc1c164fa0d5fa722eec2546a0a2a5a9918bf5e7558f0a2ab559e095d"


def test_v3_epochs_score_byte_for_byte_as_before():
    from tests.test_v3_menu_b import B_EPOCH
    from tests.test_v3_menu_b import GRACE as B_GRACE
    from tests.test_v3_menu_b import live as live_v3

    results, reveals = live_v3(B_EPOCH + B_GRACE + 2)
    doc = {"reveals": reveals, "weights": [[r.uids, r.weights] for r in results]}
    got = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert got == V3_MENU_B_DIGEST
