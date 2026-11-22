"""The fast crown of protocol v4 (docs/decisions/FAST_CROWN_V4.md, packages
F1-F7): the bound's tuning, the one-epoch challenge, its items, the holder
standing on exploration, and the reveal. Single process, no network."""
from __future__ import annotations

import hashlib
import itertools
from fractions import Fraction

import pytest

from fugal_subnet import config, epoch_plan, menu_state, picks_v4, seat_v4
from fugal_subnet.benchmarks.slicer import PoolIndex
from fugal_subnet.seat_bound import lower_bound
from fugal_subnet.tee import td_v4, v4_options
from fugal_subnet.tee.proof import BenchmarkProof, Challenge, QuestionResult

SEED = menu_state.load_seed()
A, B, C = SEED.ids()[:3]
CAT = "math"
H1, H2, H3 = (hashlib.sha256(x).hexdigest() for x in (b"one", b"two", b"three"))


def _v(entry, grade):
    return picks_v4.call_value(Fraction(grade), menu_state.pick_cost(SEED, entry, CAT))


# --- F1: constants and the bound's tuning ----------------------------------------------


def test_the_decided_constants():
    assert config.BOUND_N0_V4 == 100
    assert config.ODDS_MIN_PAIRS_V4 == 300
    assert config.PICKS_MIN_QUESTIONS_V4 == 500
    assert config.CHALLENGE_SIZES_V4 == (500, 1000)
    assert config.CHALLENGE_LIST_SIZE_V4 == 1000
    assert max(config.CHALLENGE_SIZES_V4) <= config.CHALLENGE_LIST_SIZE_V4
    # The holder a challenge pairs with is named by the newest root reveal at
    # the boundary, which is held as long as the menu is.
    assert config.CHALLENGE_HOLDER_LAG_V4 == menu_state.MENU_LAG
    from fugal_subnet.tee import proof
    assert proof.CHALLENGE_SIZES == config.CHALLENGE_SIZES_V4
    assert td_v4.CHALLENGE_CHOICES == (0, 500, 1000)


def test_rules_a_and_b_tune_the_bound_to_n0_100():
    assert seat_v4.BOUND_N0 == picks_v4.BOUND_N0 == 100
    items = {"a": [3_000_000 + (i % 7) * 1_000_000 for i in range(600)]}
    rec = picks_v4.PicksEpoch(1, ("a",) * 600, {"h": tuple(items["a"])}, holder="b")
    got = picks_v4.picks_check({1: rec}, "h", "b", through=1)
    want = lower_bound(items, margin_units=picks_v4.PICKS_MARGIN_UNITS, n_min=500, n0=100)
    assert (got.bound.lower, got.passes) == (want.lower, want.passes)
    tuned_to_minimum = lower_bound(items, margin_units=picks_v4.PICKS_MARGIN_UNITS, n_min=500,
                                   n0=500)
    assert got.bound.lower != tuned_to_minimum.lower


# --- F5: the items ------------------------------------------------------------------------


def _ans(entry, correct=True, failed=False):
    return picks_v4.Answer(entry, correct and not failed, failed)


def test_the_items_follow_the_note():
    own = [_ans(A), _ans(A, False), _ans(B), _ans(A)]
    hps = [A, B, C, None]
    held = {1: _ans(B), 2: _ans(C, False)}
    got = picks_v4.challenge_items(SEED, [CAT] * 4, own, hps, held)
    assert got == (0,                              # the same entry: one answer serves both
                   _v(A, 0) - _v(B, 1),
                   _v(B, 1) - _v(C, 0),
                   _v(A, 1))                       # no holder pick: its side is 0


def test_failed_calls_fail_closed():
    own_failed = picks_v4.challenge_items(SEED, [CAT], [_ans(A, failed=True)], [B],
                                          {0: _ans(B, False)})
    assert own_failed == (_v(A, 0) - _v(B, 0),)
    held_failed = picks_v4.challenge_items(SEED, [CAT], [_ans(A)], [B], {0: _ans(B, failed=True)})
    assert held_failed == (_v(A, 1) - _v(B, 1),)


def test_a_failed_call_on_either_side_never_raises_the_items():
    """Attack case 1: any subset of either side's calls failing."""
    base_own = [_ans(A, c) for c in (True, False, True, False)]
    base_held = {i: _ans(B, c) for i, c in enumerate((True, True, False, False))}
    clean = picks_v4.challenge_items(SEED, [CAT] * 4, base_own, [B] * 4, base_held)
    for k in range(1, 9):
        for fail in itertools.combinations(range(8), k):
            own = [_ans(A, a.correct, failed=i in fail) for i, a in enumerate(base_own)]
            held = {i: _ans(B, a.correct, failed=4 + i in fail) for i, a in base_held.items()}
            got = picks_v4.challenge_items(SEED, [CAT] * 4, own, [B] * 4, held)
            assert all(g <= c for g, c in zip(got, clean))


def test_a_missing_holder_side_call_is_an_error_and_items_stay_in_int32():
    with pytest.raises(ValueError, match="no holder-side call"):
        picks_v4.challenge_items(SEED, [CAT], [_ans(A)], [B], {})
    assert picks_v4._clamp32(2**40) == 2**31 - 1 and picks_v4._clamp32(-(2**40)) == -(2**31) + 1 != picks_v4.NO_ITEM


def test_item_means_are_by_the_challengers_entry_over_differing_picks():
    got = picks_v4.item_means((0, 5, 7, -3), [A, A, B, A], [A, B, A, None])
    assert got == {A: (2, 2), B: (1, 7)}


# --- rule B on challenge records ------------------------------------------------------------


def _rec(epoch, values, holder=H1, n=500):
    return picks_v4.PicksEpoch(epoch, ("a",) * 1000, {h: (x,) * n for h, x in values.items()},
                               holder=holder)


def test_rule_b_reads_the_answered_run_and_a_gap_restarts_it():
    good = 2_000_000
    recs = {1: _rec(1, {"c": good}), 2: _rec(2, {}), 3: _rec(3, {"c": good})}
    ev = picks_v4.picks_check(recs, "c", H1, through=3)
    assert ev.run == (3,) and ev.paired == 500
    recs[2] = _rec(2, {"c": good})
    assert picks_v4.picks_check(recs, "c", H1, through=3).paired == 1500


def test_items_count_only_against_the_holder_they_paired_with():
    recs = {1: _rec(1, {"c": 2_000_000}, holder=H2)}
    ev = picks_v4.picks_check(recs, "c", H1, through=1)
    assert ev.paired == 0 and not ev.passes
    assert picks_v4.picks_check(recs, "c", H2, through=1).paired == 500


def test_withholding_bad_attempts_does_not_pass_a_worse_router():
    """Attack case 2: a challenger 2 points worse serves only the attempts
    that look good; every gap restarts its run, so no kept attempt is judged
    with the ones it hid."""
    import random

    rnd = random.Random(7)
    for trial in range(20):
        recs, passed = {}, False
        for e in range(1, 301):
            items = tuple(int((rnd.gauss(-2.0, 20.0)) * 1_000_000) for _ in range(500))
            served = sum(items) / 500 > -2_000_000     # serve only the better half
            recs[e] = picks_v4.PicksEpoch(e, ("a",) * 500, {"c": items} if served else {},
                                          holder=H1)
            recs = {k: v for k, v in recs.items() if k > e - 48}
            passed |= picks_v4.picks_check(recs, "c", H1, through=e).passes
        assert not passed, trial


def test_a_prefix_of_one_sequence_is_allowed():
    """Attack case 3: reporting the first 500 of 1,000 is one look at the
    same sequence (a 1,000-question run's first half)."""
    items = tuple(range(1000))
    rec5 = picks_v4.PicksEpoch(1, ("a",) * 1000, {"c": items[:500]}, holder=H1)
    assert picks_v4.picks_check({1: rec5}, "c", H1, through=1).paired == 500


def test_the_reveal_round_trips_and_is_strict():
    rec = picks_v4.PicksEpoch(4, ("math", "code") * 500, {H2: tuple(range(-250, 250))}, False, H1)
    qs = [f"q{i}" for i in range(1000)]
    block = picks_v4.reveal_block(rec, {}, questions=qs)
    back = picks_v4.epoch_from_reveal(block, 4)
    assert back == rec
    for bad in ({**block, "holder": "x"}, {**block, "items": {H2: picks_v4.values_b64(range(7))}},
                {**block, "list": [["q", "math"], ["q", "code"]]},
                {**block, "items": {H3: block["items"][H2], H2: block["items"][H2]}}):
        with pytest.raises(ValueError):
            picks_v4.epoch_from_reveal(bad, 4)


# --- F2: the challenge list -----------------------------------------------------------------


def _pool(n=3000):
    benches = ["gsm8k", "math", "mmlu"]
    return [{"question_id": f"q{i:05d}", "benchmark": benches[i % 3]} for i in range(n)]


def test_the_challenge_list_starts_with_the_slice_and_is_a_function_of_the_nonce():
    pool = _pool()
    index = PoolIndex(pool, {"gsm8k": "easy_math", "math": "math", "mmlu": "knowledge"})
    targets = {"easy_math": 0.3, "math": 0.3, "knowledge": 0.4}
    sl = pool[:200]
    got = epoch_plan.challenge_list_v4(b"n" * 32, sl, index, targets)
    assert len(got) == 1000 and got[:200] == sl
    assert len({q["question_id"] for q in got}) == 1000
    assert got == epoch_plan.challenge_list_v4(b"n" * 32, sl, index, targets)
    assert got != epoch_plan.challenge_list_v4(b"m" * 32, sl, index, targets)


def test_the_epoch_plan_draws_the_list_on_published_questions_only():
    from fugal_subnet import memo_v4

    pool = _pool(6000)
    mix = epoch_plan.mix_v4()
    names = sorted(mix.bench_category)[:3]
    pool = [{**q, "benchmark": names[i % 3]} for i, q in enumerate(pool)]
    first = {q["question_id"]: 1 for q in pool[:4500]}
    published = memo_v4.Published(through=8, first=first)
    plan = epoch_plan.plan_v4(pool, published=published, epoch=10)
    lst = plan.challenge_list(b"k" * 32, pool, config.SLICE_SIZE_V4)
    assert [q["question_id"] for q in lst[:200]] == [
        q["question_id"] for q in plan.scored_slice(b"k" * 32, pool, 200)]
    assert all(q["question_id"] in first for q in lst)


# --- F3: the proof ------------------------------------------------------------------------


def test_the_challenge_field_is_strict():
    assert Challenge.from_json({"size": 500, "holder": None}) == Challenge(500, None)
    assert Challenge.from_json({"size": 1000, "holder": H1}).holder == H1
    for bad in ({"size": 200, "holder": None}, {"size": 750, "holder": None},
                {"size": True, "holder": None}, {"size": 500.0, "holder": None},
                {"size": 500}, {"size": 500, "holder": "ab"}, {"size": 500, "holder": H1.upper()},
                {"size": 500, "holder": None, "x": 1}):
        with pytest.raises(ValueError):
            Challenge.from_json(bad)


def _result(qid, entry, *, holder=None, explore=False):
    return QuestionResult(question_id=qid, routed_model=entry, correct=True, cost_usd=0.001,
                          response_hash="0" * 64, prompt_tokens=1, completion_tokens=1,
                          is_exploration=explore, latency_ms=1, provider_cost=None,
                          served_by="", holder_pick=holder)


def test_a_holder_side_call_is_not_the_miners_answer_and_round_trips():
    p = BenchmarkProof(epoch_id="e", nonce="0" * 64, questions_hash="0", weights_hash="0",
                       source_hash="0", results=[_result("q1", A), _result("q1", B, holder=True)],
                       total_cost_usd=0.002, per_model_costs={}, attestation_quote=b"",
                       timestamp=1.0, hotkey="hk", routing_protocol="fugal-v4:x",
                       menu_hash="1" * 64, endpoint_reads=(), scored=True,
                       challenge=Challenge(500, H1))
    assert [r.routed_model for r in p.scored_results] == [A]
    assert [r.routed_model for r in p.holder_results] == [B]
    back = BenchmarkProof.from_dict(p.to_dict())
    assert back.content_hash() == p.content_hash() and back.challenge == p.challenge
    d = p.to_dict()
    d["results"][1]["holder_pick"] = False
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(d)
    d["results"][1]["holder_pick"] = True
    d["results"][1]["is_exploration"] = True
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(d)


# --- F4: verification's structural check ------------------------------------------------


def _expected(lst):
    from fugal_subnet.tee.verify import V4Expected

    return V4Expected(state=SEED, menu_hash="1" * 64, hotkey="hk", shared={}, cost_aware={},
                      coverage={}, challenge=tuple(lst))


def _proof(results, size=500, holder=H1, scored=True):
    return BenchmarkProof(epoch_id="e", nonce="0" * 64, questions_hash="0", weights_hash="0",
                          source_hash="0", results=results, total_cost_usd=0.0,
                          per_model_costs={}, attestation_quote=b"", timestamp=1.0,
                          scored=scored,
                          challenge=Challenge(size, holder) if scored else None)


def test_verification_refuses_what_is_not_the_challenges_prefix():
    from fugal_subnet.tee.verify import _v4_scored_check

    lst = [f"q{i}" for i in range(1000)]
    exp = _expected(lst)
    ok = [_result(q, A) for q in lst[:500]]
    sl = set(lst[:200])
    assert _v4_scored_check(_proof(ok), exp, sl) is None
    assert _v4_scored_check(_proof(ok + [_result("q3", B, holder=True)]), exp, sl) is None
    cases = {
        "not a prefix": [_result(q, A) for q in lst[1:501]],
        "a question twice": ok[:-1] + [ok[0]],
        "one missing": ok[:-1],
        "holder call to its own pick": ok + [_result("q3", A, holder=True)],
        "holder call twice": ok + [_result("q3", B, holder=True)] * 2,
        "holder call outside the prefix": ok + [_result("q700", B, holder=True)],
    }
    for why, results in cases.items():
        assert _v4_scored_check(_proof(results), exp, sl), why
    assert _v4_scored_check(_proof(ok + [_result("q3", B, holder=True)], holder=None), exp, sl)
    assert _v4_scored_check(_proof(ok, size=1000), exp, sl)
    assert _v4_scored_check(_proof(ok, scored=False), exp, sl)
    assert _v4_scored_check(_proof([], scored=False), exp, sl) is None
    assert _v4_scored_check(_proof([], scored=None), exp, sl)


# --- F6: the seat's holder --------------------------------------------------------------------


def test_the_state_records_the_holders_and_round_trips():
    s = seat_v4.SeatStateV4(holders=((7, H1), (8, None), (6, H2)))
    assert s.holders == ((8, None), (7, H1))
    assert seat_v4.challenge_holder(s, 9) == H1
    assert seat_v4.challenge_holder(s, 10) is None
    assert seat_v4.challenge_holder(s, 11) is None
    back = seat_v4.from_dict(seat_v4.to_dict(s))
    assert back.holders == s.holders
    d = seat_v4.to_dict(s)
    for bad in ({"07": H1}, {"7": "zz"}, {"1": H1, "2": H1, "3": H1}):
        with pytest.raises(ValueError):
            seat_v4.from_dict({**d, "holders": bad})


def test_a_challenge_paired_with_a_predecessor_counts_for_its_successor():
    hs = seat_v4.HeadState(head=H2, owner="o", hotkey="k", block=5, since=1, seen=3, origin=H1,
                           origin_block=1, origin_owner="o", predecessors=(H1,),
                           predecessor_since=(0,))
    state = seat_v4.SeatStateV4(champion=H2, champion_since=1, heads=(hs,))
    recs = seat_v4.lineage_records({1: _rec(1, {"c": 1}, holder=H1)}, state)
    assert recs[1].holder == H2


# --- the operator's option ------------------------------------------------------------------


def test_the_challenge_option_is_exactly_0_500_or_1000():
    assert [v4_options.parse_challenge(x) for x in ("0", "500", "1000")] == [0, 500, 1000]
    for bad in ("", "750", "0500", " 500", "+500", "1e3", "on", 500, "10000"):
        with pytest.raises(v4_options.OptionRefused):
            v4_options.parse_challenge(bad)
    assert v4_options.FIELDS["FUGAL_V4_CHALLENGE"] == "v4_challenge"


# --- the TD's choice of holder -----------------------------------------------------------------


def _head_bytes():
    from tests.test_head_v4 import COST_FIELDS, _arrays, _bytes, _without

    return _bytes(_without(_arrays([A, B]), *COST_FIELDS))


def test_the_td_challenges_only_a_holder_it_can_verify():
    raw = _head_bytes()
    h = hashlib.sha256(raw).hexdigest()

    def policy(read, offer):
        return td_v4.ChallengePolicy(1000, first_epoch=10, read_holder=read, holder_bytes=offer)

    assert policy(lambda e: None, lambda s: raw)(11) == td_v4.ChallengeSpec(1000, None, None)
    got = policy(lambda e: (e - 2, h), lambda s: raw)(20)
    assert (got.size, got.holder) == (1000, h) and got.holder_head is not None
    assert policy(lambda e: (e - 2, None), lambda s: raw)(20).holder is None
    assert policy(lambda e: (e - 3, h), lambda s: raw)(20) is None       # an older reveal
    assert policy(lambda e: (e - 2, h), lambda s: raw + b"x")(20) is None  # not its bytes
    assert policy(lambda e: (e - 2, h), lambda s: None)(20) is None

    def boom(e):
        raise OSError("chain")
    assert policy(boom, lambda s: raw)(20) is None
    assert td_v4.ChallengePolicy(0, first_epoch=10)(20) is None
    with pytest.raises(ValueError):
        td_v4.ChallengePolicy(750, first_epoch=10, read_holder=boom, holder_bytes=boom)


# --- attack case 4: every pick recomputed; another holder refused -----------------------


def _epoch_case(mutate):
    from types import SimpleNamespace

    import numpy as np

    from fugal_subnet.champion import Window
    from fugal_subnet.head_odds import picks_and_odds_v4
    from fugal_subnet.v2_epoch import PicksInput, _artifact, _challenge_epoch

    rng = np.random.default_rng(3)
    n = 500
    qids = [f"c{i:04d}" for i in range(n)]
    hid = {q: rng.standard_normal(1024).astype(np.float32) for q in qids}
    mk = [(rng.standard_normal((3, 1024)) * 0.05, rng.standard_normal(3), [A, B, C])
          for _ in range(2)]
    holder_head, chal_head = mk
    hh, ch = H1, H2
    state = seat_v4.SeatStateV4(champion=hh, champion_since=1, holders=((8, hh),))
    menu = __import__("dataclasses").replace(SEED, epoch=10)
    H = np.stack([hid[q] for q in qids])
    own = picks_and_odds_v4(_artifact(chal_head), menu, H).picks
    theirs = picks_and_odds_v4(_artifact(holder_head), menu, H).picks

    def res(q, e):
        return SimpleNamespace(question_id=q, routed_model=e, correct=True, failed=False)
    proof = SimpleNamespace(
        scored=True, challenge=SimpleNamespace(size=n, holder=hh),
        scored_results=[res(q, e) for q, e in zip(qids, own)],
        holder_results=[res(q, t) for q, e, t in zip(qids, own, theirs) if t != e])
    mutate(proof, theirs)
    oi = SimpleNamespace(heads={1: holder_head, 2: chal_head}, hidden_of=lambda q: hid[q])
    sl = SimpleNamespace(question_ids=tuple(qids[:200]))
    rec, block = _challenge_epoch(
        PicksInput(state=menu, challenge=tuple((q, CAT) for q in qids)), 10, sl,
        {hh: 1, ch: 2}, {1: SimpleNamespace(scored=False), 2: proof}, state, Window.empty(48),
        oi)[:2]
    return rec, block, ch


def test_a_challenge_with_the_heads_own_picks_counts():
    rec, block, ch = _epoch_case(lambda p, t: None)
    assert ch in rec.values and not block["refused"] and len(rec.values[ch]) == 500


def test_a_challenge_naming_another_holder_or_stating_another_pick_is_refused():
    def other_holder(p, t):
        p.challenge.holder = H3

    def wrong_own_pick(p, t):
        r = p.scored_results[7]
        r.routed_model = next(e for e in (A, B, C) if e != r.routed_model)

    def wrong_holder_pick(p, t):
        if p.holder_results:
            r = p.holder_results[0]
            r.routed_model = next(e for e in (A, B, C) if e not in (r.routed_model,))

    def dropped_holder_call(p, t):
        p.holder_results = p.holder_results[1:]

    for mutate in (other_holder, wrong_own_pick, wrong_holder_pick, dropped_holder_call):
        rec, block, ch = _epoch_case(mutate)
        assert ch not in rec.values and ch in block["refused"], mutate.__name__


def test_a_challenge_with_most_holder_side_calls_failed_or_unmade_is_void_and_published():
    def fail_all(p, t):
        for i, r in enumerate(p.holder_results):
            r.failed, r.correct = True, False
            r.not_started = True if i % 2 else None

    rec, block, ch = _epoch_case(fail_all)
    assert picks_v4.is_void(rec.values[ch]) and ch in block["void"] and ch not in block["refused"]
    assert "holder-side calls failed or were not made" in block["void"][ch]
    assert not block["item_means"] and not block["cells"]
    assert picks_v4.epoch_from_reveal(block, 10).values == rec.values


# --- the simulation against the code ------------------------------------------------------


def test_the_built_rule_b_gives_the_notes_one_attempt_pass_rates_in_kind():
    """FAST_CROWN_V4.md 2's table, one attempt of 1,000 judged alone, items
    drawn from the simulation's model of the measured items (SD 20,
    scripts/experiments/experiment_fast_crown_v4.PicksModel) and run through
    the built record and rule B (picks_v4.picks_check: n0 100, margin -1.5,
    stratified by category): the note's 95% / 56% / 7.8% / 0 at +2 / +1 / 0 /
    -2 points, in kind, at 200 attempts each."""
    import importlib.util
    from pathlib import Path

    import numpy as np

    path = (Path(__file__).resolve().parents[1] / "scripts" / "experiments"
            / "experiment_fast_crown_v4.py")
    spec = importlib.util.spec_from_file_location("experiment_fast_crown_v4", path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    model = sim.PicksModel(sim.load_inputs())
    rng = np.random.default_rng(21)
    runs = 200
    rates = {}
    for mu in (2.0, 1.0, 0.0, -2.0):
        batch = model.batch(rng, 1000, mu, runs)
        passed = 0
        for r in range(runs):
            cats, items = [], []
            for c, arr in sorted(batch.items()):
                for x in arr[r]:
                    cats.append(c)
                    items.append(int(round(float(x) * picks_v4.UNITS_PER_POINT / 100)))
            rec = picks_v4.PicksEpoch(1, tuple(cats), {"c": tuple(items)}, holder="h")
            passed += picks_v4.picks_check({1: rec}, "c", "h", through=1).passes
        rates[mu] = passed / runs
    print("built rule B, one attempt of 1,000:", rates)
    assert rates[2.0] >= 0.85, rates
    assert 0.35 <= rates[1.0] <= 0.75, rates
    assert rates[0.0] <= 0.16, rates
    assert rates[-2.0] == 0, rates


# --- the holder's picks, published for the product's parity check ------------------------


def _publish_case(champion_serves=True, bytes_held=False):
    from types import SimpleNamespace

    import numpy as np

    from fugal_subnet.v2_epoch import PicksInput, _publish_holder_picks
    from tests.test_head_v4 import COST_FIELDS, _arrays, _bytes, _without

    raw = _bytes(_without(_arrays([A, B, C], seed=5), *COST_FIELDS))
    from fugal_subnet.head_eval import load_head_v4
    h = load_head_v4(raw)
    head = (h.W, h.b, list(h.models))
    hh = hashlib.sha256(raw).hexdigest()
    rng = np.random.default_rng(9)
    qids = [f"l{i:04d}" for i in range(1000)]
    hid = {q: rng.standard_normal(1024).astype(np.float32) for q in qids}
    oi = SimpleNamespace(heads={7: head} if champion_serves else {}, hidden_of=lambda q: hid[q])
    pi = PicksInput(state=SEED, challenge=tuple((q, CAT) for q in qids),
                    heads={hh: raw} if bytes_held else {})
    models = list(SEED.ids())
    return hh, head, hid, qids, oi, pi, models, _publish_holder_picks


def test_the_reveal_publishes_the_holders_pick_on_every_question_of_the_list():
    import numpy as np

    from fugal_subnet.head_odds import picks_and_odds_v4
    from fugal_subnet.v2_epoch import _artifact

    hh, head, hid, qids, oi, pi, models, publish = _publish_case()
    block = publish({}, hh, {}, pi, oi, {hh: 7}, models)
    want = picks_and_odds_v4(_artifact(head), SEED, np.stack([hid[q] for q in qids])).picks
    doc = block["holder_picks"]
    assert doc["head"] == hh and len(doc["picks"]) <= 1400          # 1,000 int8, base64
    assert picks_v4.picks_from_doc(doc, models) == (hh, tuple(want))
    # The same from the holder's bytes when it did not serve.
    hh2, _h, _x, _q, oi2, pi2, _m, _p = _publish_case(champion_serves=False, bytes_held=True)
    assert publish({}, hh2, {}, pi2, oi2, {}, models)["holder_picks"] == doc


def test_no_holder_publishes_no_picks_and_a_missing_head_is_not_guessed():
    import pytest as _pytest

    from fugal_subnet.v2_epoch import HolderHeadMissing

    hh, _head, _hid, _qids, oi, pi, models, publish = _publish_case(champion_serves=False)
    assert publish({}, None, {}, pi, oi, {}, models)["holder_picks"] == {}
    with _pytest.raises(HolderHeadMissing):
        publish({}, hh, {}, pi, oi, {}, models)


def test_the_holders_head_is_sought_in_one_fixed_order():
    """neurons.validator.holder_head_sources: what this node holds (verified
    now, then its replay store), then the miner (bytes miners served this
    epoch, verified or not), then the root's head publication; a source that
    fails or gives other bytes is passed over; with none, nothing (the epoch
    is then not scored here, HolderHeadMissing)."""
    import base64
    import types

    from neurons import validator as v

    data = b"the holder's head bytes"
    h = hashlib.sha256(data).hexdigest()
    calls = []

    def src(name, value=None, boom=False):
        def get(*_a):
            calls.append(name)
            if boom:
                raise OSError("unreachable")
            return value
        return get

    def find(held=None, **kw):
        calls.clear()
        args = dict(replay_store=src("store"), served=src("miner", []),
                    publication=src("root"))
        args.update(kw)
        return v.holder_head_sources(h, held=held or {}, **args)

    assert find({h: data}) == (data, "verified") and calls == []
    assert find(replay_store=src("store", data)) == (data, "replay store") and calls == ["store"]
    assert find(replay_store=src("store", boom=True), served=src("miner", [b"x", data])) == (
        data, "miner") and calls == ["store", "miner"]
    assert find(replay_store=src("store", b"other"), publication=src("root", data)) == (
        data, "root publication") and calls == ["store", "miner", "root"]
    assert find(publication=src("root", boom=True)) == (None, None)
    # the miner source reads the bytes a miner served, whatever its proof
    resp = types.SimpleNamespace(head_npz_b64=base64.b64encode(data).decode())
    junk = types.SimpleNamespace(head_npz_b64="not base64!")
    assert v._served_head_bytes({3: junk, 7: resp}) == [data]
