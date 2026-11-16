"""A protocol v3 epoch end to end: exploration answers in, the pool and every
head's odds out, in the window, the reveal, the saved state and the rebuild.

docs/INVARIANTS.md, "I1 — odds are computed exactly by the validator" and
"I1 — v2 state is rebuilt from reveals".
"""
import dataclasses
import hashlib
import json

import numpy as np
import pytest

from fugal_subnet import config, odds_score
from fugal_subnet.champion import OddsParams, Window
from fugal_subnet.explore_pool import ExploreAnswer
from fugal_subnet.head_odds import ODDS_SCALE
from fugal_subnet.v2_epoch import (
    OddsInput,
    load_window,
    odds_b64,
    odds_from_b64,
    save_window,
    seat_params_from_config,
)
from fugal_subnet.v2_rebuild import pairs_from_reveal, rebuild_v2_from_reveals
from fugal_subnet.v2_sync import consensus_view
from tests.test_v2_epoch import ACC, CATEGORY, MODELS, PRICES, H, always, oracle, run
from tests.test_v2_rebuild import reveal_of, same_window, write

DIM = 16
BENCHES = ("mmlu", "math", "aime")
UIDS = (5, 6, 7)
PRIVATE, SHARED = 40, 8


def hidden_of(qid):
    seed = int.from_bytes(hashlib.sha256(qid.encode()).digest()[:8], "big")
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def difficulty_of(qid):
    """A question's difficulty is a function of its embedding, so a head can learn it."""
    return float(1 / (1 + np.exp(-4.0 * hidden_of(qid)[0])))


def head(kind):
    """(W, b, models): a head that reads the difficulty direction, or one that does not."""
    W = np.zeros((3, DIM), dtype=np.float32)
    if kind == "sharp":
        # Model m is right when 4 * h0 < logit(ACC[m]): a steep logistic at that line.
        W[:, 0] = -60.0
        b = np.array([15.0 * np.log(ACC[m] / (1 - ACC[m])) for m in MODELS], dtype=np.float32)
    else:                                     # a constant per model: its base rate
        b = np.array([np.log(ACC[m] / (1 - ACC[m])) for m in MODELS], dtype=np.float32)
    return W, b, MODELS


HEADS = {5: head("flat"), 6: head("sharp"), 7: head("flat")}
POLICIES = {5: always("m/strong"), 6: oracle, 7: always("m/cheap")}


def answers_for(epoch, uids):
    out, bench_of = [], {}
    for i in range(SHARED):
        qid = f"s{epoch}_{i}"
        bench_of[qid] = BENCHES[i % 3]
        for u in uids:
            m = MODELS[i % 3]
            out.append(ExploreAnswer(qid, m, f"ck{u}", difficulty_of(qid) < ACC[m], False, False,
                                     20, 100 + i))
    for u in uids:
        for i in range(PRIVATE):
            qid = f"p{epoch}_{u}_{i}"
            bench_of[qid] = BENCHES[(i + u) % 3]
            m = MODELS[(i + u) % 3]
            out.append(ExploreAnswer(qid, m, f"ck{u}", difficulty_of(qid) < ACC[m],
                                     failed=(i == 0), private=True, prompt_tokens=20,
                                     completion_tokens=200))
    return out, bench_of


def odds_input(epoch, uids=UIDS, **kw):
    answers, bench_of = answers_for(epoch, uids)
    return OddsInput(nonce=hashlib.sha256(f"n{epoch}".encode()).digest(), answers=answers,
                     benchmark_of=bench_of, heads={u: HEADS[u] for u in uids},
                     hidden_of=hidden_of, single_model_min_pairs=30,
                     category_targets={"knowledge": 0.4, "math": 0.3, "hard": 0.3}, **kw)


def params(odds=True):
    p = dataclasses.replace(seat_params_from_config(protocol_version=3), min_paired_questions=60)
    if odds:
        p = dataclasses.replace(p, odds=OddsParams(margin_points=0.1, min_pairs=200,
                                                   nv_noninferiority=0.5, ref_prior=20,
                                                   confirm_pairs=200))
    return p


def live(epochs, odds=True, schedule=lambda e: UIDS):
    window = seat = copy = None
    results, reveals = [], []
    for e in range(epochs):
        uids = schedule(e)
        pols = {u: POLICIES[u] for u in uids}
        r = run(e, pols, window=window, seat=seat, copy=copy, params=params(odds),
                odds_input=odds_input(e, uids))
        window, seat, copy = r.window, r.decision.state, r.copy_state
        results.append(r)
        reveals.append(json.loads(json.dumps(reveal_of(e, r, pols))))
    return results, reveals


def test_a_v3_epoch_pools_the_exploration_and_scores_every_head_on_it():
    results, reveals = live(1)
    r, rv = results[0], reveals[0]
    pairs = r.window.pairs[1]
    # 8 shared pairs and, per operator, its private ones (40, one failed call)
    # cut to the cap of 36.
    assert len(pairs.question_ids) == SHARED + 3 * 36
    assert set(r.window.odds) == {H(5), H(6), H(7)}
    assert all(o.shape == (len(pairs.question_ids),) and o.dtype == np.int16
               for per in r.window.odds.values() for o in per.values())
    assert [row["question_id"] for row in rv["exploration_pool"]] == list(pairs.question_ids)
    assert rv["exploration_pool"][0].keys() == {
        "question_id", "model", "benchmark", "y", "voters", "prompt_tokens", "completion_tokens",
        "weight", "latency_ms"}
    assert set(rv["head_odds"]) == {H(5), H(6), H(7)}
    assert len(rv["v2_state"]["odds_digest"]) == 64
    assert set(rv["odds_shadow"][H(6)]) >= {"uid", "pairs", "skill_points"}
    assert rv["exploration_excluded_coldkeys"] == {}


def test_the_sharp_head_takes_the_seat_on_odds_and_would_not_on_net_value():
    """The same proofs, the two objectives. On net value the seat goes to the
    most accurate router; on odds it goes to the head that predicts."""
    on_odds, reveals = live(6, odds=True)
    on_nv, _ = live(6, odds=False)
    last = on_odds[-1]
    assert last.decision.champion == H(6)
    shadow = reveals[-1]["odds_shadow"]
    assert shadow[H(6)]["skill_points"] > 2.0
    assert abs(shadow[H(5)]["skill_points"]) < 1.0
    # In shadow the pool and the odds are published just the same.
    shadow_run, shadow_reveals = live(2, odds=False)
    assert shadow_reveals[1]["head_odds"] == reveals[1]["head_odds"]
    assert shadow_reveals[1]["exploration_pool"] == reveals[1]["exploration_pool"]
    assert on_nv[-1].decision.champion is not None


def test_the_rebuild_from_reveals_restores_pairs_and_odds_exactly(tmp_path):
    results, reveals = live(6)
    write(tmp_path, reveals)
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES, bench_category=CATEGORY)
    last = results[-1]
    same_window(rb.window, last.window)
    assert sorted(rb.window.pairs) == sorted(last.window.pairs)
    for e, ep in last.window.pairs.items():
        got = rb.window.pairs[e]
        assert (got.question_ids, got.models, got.benchmarks, got.categories) == (
            ep.question_ids, ep.models, ep.benchmarks, ep.categories)
        for f in ("y", "prompt_tokens", "completion_tokens"):
            assert np.array_equal(getattr(got, f), getattr(ep, f))
    assert {h: sorted(per) for h, per in rb.window.odds.items()} == {
        h: sorted(per) for h, per in last.window.odds.items()}
    for h, per in last.window.odds.items():
        for e, o in per.items():
            assert np.array_equal(rb.window.odds[h][e], o)
    assert rb.seat == last.decision.state
    # And a validator that continues from the rebuild decides the next epoch identically.
    from fugal_subnet.copy_rule import from_rebuild
    pols = {u: POLICIES[u] for u in UIDS}
    a = run(6, pols, window=last.window, seat=last.decision.state, copy=last.copy_state,
            params=params(), odds_input=odds_input(6))
    b = run(6, pols, window=rb.window, seat=rb.seat, copy=from_rebuild(rb),
            params=params(), odds_input=odds_input(6))
    assert a.reveal == b.reveal and a.weights == b.weights


def test_the_saved_window_round_trips_pairs_and_odds(tmp_path):
    results, _ = live(4)
    path = str(tmp_path / "state.v2window.npz")
    save_window(path, results[-1].window)
    back = load_window(path, 48)
    same_window(back, results[-1].window)
    for h, per in results[-1].window.odds.items():
        for e, o in per.items():
            assert np.array_equal(back.odds[h][e], o) and back.odds[h][e].dtype == np.int16
    assert back.pairs[2].categories == results[-1].window.pairs[2].categories
    # A window saved before v3 (no pairs) loads as one with none.
    plain = Window(48, results[-1].window.slices, results[-1].window.heads)
    save_window(path, plain)
    assert load_window(path, 48).pairs == {} and load_window(path, 48).odds == {}


def test_validators_with_different_odds_differ_in_the_consensus_view():
    _, reveals = live(2)
    rv = reveals[1]
    other = json.loads(json.dumps(rv))
    assert consensus_view(rv) == consensus_view(other)
    # One head scored one unit differently: a different digest, a different view.
    from fugal_subnet.v2_epoch import odds_digest
    h = sorted(rv["head_odds"])[0]
    o = odds_from_b64(rv["head_odds"][h], len(rv["exploration_pool"])).copy()
    o[0] = (int(o[0]) + 1) % ODDS_SCALE
    changed = {**rv["head_odds"], h: odds_b64(o)}
    assert odds_digest(rv["exploration_pool"], rv["head_odds"]) == rv["v2_state"]["odds_digest"]
    assert odds_digest(rv["exploration_pool"], changed) != rv["v2_state"]["odds_digest"]


def test_the_result_does_not_depend_on_the_order_answers_arrive_in():
    a = run(0, {u: POLICIES[u] for u in UIDS}, params=params(), odds_input=odds_input(0))
    oi = odds_input(0)
    oi = dataclasses.replace(oi, answers=list(reversed(oi.answers)))
    b = run(0, {u: POLICIES[u] for u in reversed(UIDS)}, params=params(), odds_input=oi)
    assert a.reveal == b.reveal


def test_an_operator_with_implausible_private_answers_is_left_out_of_the_pool():
    results, _ = live(4)
    last = results[-1]
    oi = odds_input(4)
    poisoned = [dataclasses.replace(a, correct=False) if a.coldkey == "ck7" and a.private else a
                for a in oi.answers]
    r = run(4, {u: POLICIES[u] for u in UIDS}, window=last.window, seat=last.decision.state,
            copy=last.copy_state, params=params(),
            odds_input=dataclasses.replace(oi, answers=poisoned))
    assert set(r.reveal["exploration_excluded_coldkeys"]) == {"ck7"}
    assert not any(q.startswith("p4_7_") for q in r.window.pairs[5].question_ids)
    # Its head is still scored on everyone else's pairs: a flag is never a score.
    assert H(7) in r.window.odds and 5 in r.window.odds[H(7)]
    # An operator the validator's other detectors flagged is left out the same way.
    r2 = run(4, {u: POLICIES[u] for u in UIDS}, window=last.window, seat=last.decision.state,
             copy=last.copy_state, params=params(),
             odds_input=dataclasses.replace(oi, flagged=frozenset({"ck5"})))
    assert not any(q.startswith("p4_5_") for q in r2.window.pairs[5].question_ids)


def test_the_single_model_table_lists_only_measured_models():
    _, reveals = live(3)
    table = reveals[-1]["single_model_table"]
    assert [row["model"] for row in table] == sorted(MODELS)
    for row in table:
        assert row["pairs"] >= 30 and 0.0 <= row["accuracy"] <= 1.0
        assert row["spend_per_question"] > 0 and row["mean_completion_tokens"] > 0
        assert set(row["by_category"]) <= {"knowledge", "math", "hard"}
    strong = next(r for r in table if r["model"] == "m/strong")
    cheap = next(r for r in table if r["model"] == "m/cheap")
    assert strong["accuracy"] > cheap["accuracy"]
    from fugal_subnet.single_model import best_at_or_below, single_model_table
    assert best_at_or_below(table, cheap["spend_per_question"])["model"] == "m/cheap"
    assert best_at_or_below(table, 0.0) is None
    # Below the floor a model is not listed, never assumed.
    results, _ = live(1)
    assert single_model_table(results[0].window.pairs, PRICES, {}, min_pairs=10_000) == []


@pytest.mark.parametrize("damage, why", [
    (lambda rv: rv["exploration_pool"].reverse(), "sorted"),
    (lambda rv: rv["exploration_pool"][0].__setitem__("y", ODDS_SCALE + 1), "above the scale"),
    (lambda rv: rv["exploration_pool"][0].__setitem__("y", 0.5), "non-negative integers"),
    (lambda rv: rv["exploration_pool"][0].pop("model"), "malformed"),
    (lambda rv: rv["head_odds"].__setitem__(sorted(rv["head_odds"])[0], "AAAA"), "head_odds"),
    (lambda rv: rv["head_odds"].__setitem__(sorted(rv["head_odds"])[0], "not base64!"), "head_odds"),
])
def test_a_malformed_published_pool_is_refused_by_name(damage, why):
    _, reveals = live(1)
    rv = reveals[0]
    damage(rv)
    answered = dict.fromkeys(rv["head_odds"])
    with pytest.raises(ValueError, match=why):
        pairs_from_reveal(rv, answered, CATEGORY)


def test_v3_prices_a_correct_answer_in_pinned_dollars():
    v2, v3 = seat_params_from_config(), seat_params_from_config(protocol_version=3)
    assert v3.accuracy_value * v3.anchor_spend == config.CHAMPION_VALUE_PER_CORRECT_USD == 0.49
    # Today's rule gives the same number to within a cent's fraction: nothing moves.
    assert abs(v2.accuracy_value * v2.anchor_spend - 0.49) < 0.001
    assert v3.odds is None
    on = seat_params_from_config(1000, None, protocol_version=3, odds_seat_block=1000)
    assert on.odds == OddsParams(config.BRIER_MARGIN, config.ODDS_MIN_PAIRS,
                                 config.NV_NONINFERIORITY, config.ODDS_REF_PRIOR,
                                 config.ODDS_CONFIRM_PAIRS)
    assert seat_params_from_config(999, None, protocol_version=3, odds_seat_block=1000).odds is None
    # The odds seat is a v3 rule: a v2 epoch never gets it.
    assert seat_params_from_config(1000, None, protocol_version=2, odds_seat_block=1000).odds is None


def test_the_v3_blocks_are_compiled_for_finney_and_ordered(monkeypatch):
    finney, other = config.FINNEY_GENESIS_HASH, "0x" + "ab" * 32
    assert config.v3_activation_block(finney) is None and config.odds_seat_block(finney) is None
    monkeypatch.setattr(config, "FINNEY_V3_ACTIVATION_BLOCK", 9_600_120)
    monkeypatch.setattr(config, "FINNEY_ODDS_SEAT_BLOCK", 9_603_000)
    assert (config.v3_activation_block(finney), config.odds_seat_block(finney)) == (9_600_120, 9_603_000)
    assert config.v3_activation_block(other) is None
    monkeypatch.setattr(config, "V3_ACTIVATION_BLOCK_ENV", 500)
    assert config.v3_activation_block(other) == 500
    with pytest.raises(ValueError, match="fixed by the release"):
        config.v3_activation_block(finney)
    monkeypatch.setattr(config, "ODDS_SEAT_BLOCK_ENV", 400)
    with pytest.raises(ValueError, match="must not be before"):
        config.odds_seat_block(other)
    monkeypatch.setattr(config, "ODDS_SEAT_BLOCK_ENV", 800)
    assert config.odds_seat_block(other) == 800


def test_at_a_cold_start_nobody_is_flagged_for_being_right():
    """With no history the reference is one half for every model. Honest
    operators on models that are right 85% of the time must not be thrown out
    of the first epochs' pools for that."""
    _, reveals = live(3)
    assert all(rv["exploration_excluded_coldkeys"] == {} for rv in reveals)
    w = live(1)[0][0].window
    cells = {("q", "m/strong"): "aime"}
    assert odds_score.expected_odds(w.pairs, 20, cells, min_model_pairs=50) == {}
    assert odds_score.expected_odds(w.pairs, 20, cells, min_model_pairs=10) != {}


def test_reference_from_the_window_is_what_new_answers_are_judged_against():
    results, _ = live(3)
    w = results[-1].window
    cells = {("new-q", "m/strong"): "aime", ("new-q2", "m/unseen"): "mmlu"}
    exp = odds_score.expected_odds(w.pairs, 20, cells)
    assert 0 < exp[("new-q", "m/strong")] < ODDS_SCALE
    assert exp[("new-q2", "m/unseen")] == ODDS_SCALE // 2      # never measured: the prior
