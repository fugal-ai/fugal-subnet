"""Protocol v2 epoch wiring: proofs -> cells -> window -> seat -> weights -> reveal.

docs/INVARIANTS.md, the v2 I3/I4 entries ("one champion", "seats, not
shares", "shared grades", "spend from shared cell tokens").
"""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from fugal_subnet.champion import SeatState, Window
from fugal_subnet.evidence import Evidence
from fugal_subnet.tee.proof import QuestionResult
from fugal_subnet.v2_epoch import (
    build_epoch_slice,
    load_window,
    run_v2_epoch,
    save_window,
    seat_from_dict,
    seat_params_from_config,
    seat_to_dict,
)

MODELS = ["m/cheap", "m/mid", "m/strong"]
PRICES = {"m/cheap": (1e-7, 1e-7), "m/mid": (1e-6, 1e-6), "m/strong": (1e-5, 1e-5)}
ACC = {"m/cheap": 0.55, "m/mid": 0.70, "m/strong": 0.85}
Q = 60
BENCH = ["mmlu"] * 20 + ["math"] * 20 + ["aime"] * 20
CATEGORY = {"mmlu": "knowledge", "math": "math", "aime": "hard"}


def questions(epoch):
    return [{"question_id": f"e{epoch}q{i}", "prompt": "p" * (10 + i), "benchmark": BENCH[i]}
            for i in range(Q)]


def difficulty(epoch):
    return np.random.default_rng(epoch).random(Q)


def oracle(d):
    out = []
    for x in d:
        out.append(next((m for m in MODELS if x < ACC[m]), "m/strong"))
    return out


def always(model):
    return lambda d: [model] * Q


def proof(epoch, policy, failed=()):
    d = difficulty(epoch)
    routed = policy(d)
    return SimpleNamespace(scored_results=[
        QuestionResult(
            question_id=f"e{epoch}q{i}", routed_model=m,
            correct=(i not in failed) and bool(d[i] < ACC[m]), cost_usd=0.0,
            response_hash="x", completion_tokens=0 if i in failed else 100 + 10 * MODELS.index(m),
            failed=i in failed,
        )
        for i, m in enumerate(routed)
    ])


def burned(h):
    return SimpleNamespace(evidence=Evidence(weights_hash=h, n_total=10_000.0, n_correct=8000.0))


def run(epoch, policies, *, window=None, seat=None, owners=None, commitments=None,
        exact=(), burn_in=0, seniority=None, records=None):
    uids = sorted(policies)
    hashes = {u: f"h{u}" for u in uids}
    seat = seat or SeatState()
    return run_v2_epoch(
        epoch_index=epoch, epoch_id=f"e{epoch:08d}", questions=questions(epoch),
        pool_models=MODELS, prices=PRICES,
        verified_proofs={u: proof(epoch, policies[u]) for u in uids},
        head_hashes=hashes, coldkey_of={u: f"ck{u}" for u in uids},
        hotkey_of={u: f"hk{u}" for u in uids},
        commit_block=seniority or {u: u for u in uids},
        commit_epoch={u: 0 for u in uids},
        records=records if records is not None else {u: burned(hashes[u]) for u in uids},
        exact_copies=set(exact),
        commitments=(commitments if commitments is not None
                     else {f"hk{u}": (hashes[u], u) for u in uids}),
        window=window or Window.empty(48), seat=seat, owners=owners or {}, burn_uid=0,
        params=seat_params_from_config(), shares=(0.85, 0.10, 0.05),
        burn_in_questions=burn_in, bench_category=CATEGORY,
    )


def test_spend_comes_from_cell_tokens_not_the_frame():
    cells = {("q1", "m/mid"): 1.0}
    sl = build_epoch_slice(0, [{"question_id": "q1", "prompt": "abcd", "benchmark": "math"}],
                           MODELS, cells, {("q1", "m/mid"): 250.0}, PRICES, CATEGORY)
    from fugal_subnet.pricing import question_input_tokens
    expected = question_input_tokens("abcd") * 1e-6 + 250.0 * 1e-6
    assert sl.spends[0, 1] == expected
    assert np.isnan(sl.spends[0, 0]) and np.isnan(sl.grades[0, 2])
    assert sl.categories == ("math",) and sl.hard == (False,)


def test_the_hard_flag_is_the_hard_category():
    sl = build_epoch_slice(0, questions(0), MODELS, {}, {}, PRICES, CATEGORY)
    assert sum(sl.hard) == 20 and all(h == (b == "aime") for h, b in zip(sl.hard, BENCH))


def test_first_epoch_seats_the_most_senior_eligible_head_and_pays_it_in_full():
    r = run(0, {5: oracle, 6: always("m/strong")}, seniority={5: 2, 6: 1})
    assert r.decision.event == "first" and r.decision.champion == "h6"
    assert dict(zip(r.uids, r.weights)) == {6: 1.0}     # unfilled seats pay the champion


def test_nothing_burns_for_frame_confidence_in_v2():
    # paid_fraction is 1 in v2: with a champion seated, no weight goes to burn.
    r = run(0, {5: oracle})
    assert 0 not in r.uids and sum(r.weights) == pytest.approx(1.0)


def test_no_burned_in_head_means_burn():
    r = run(0, {5: oracle}, burn_in=10**9)
    assert r.decision.event == "empty" and dict(zip(r.uids, r.weights)) == {0: 1.0}


def test_a_better_router_takes_the_seat_and_the_old_champion_keeps_a_seat():
    window, seat, owners = None, None, None
    # 30 epochs x 60 questions clears CHAMPION_MIN_PAIRED_QUESTIONS (1,500).
    for e in range(30):
        r = run(e, {5: always("m/strong"), 6: oracle}, window=window, seat=seat,
                owners=owners, seniority={5: 1, 6: 2})
        window, seat, owners = r.window, r.decision.state, r.owners
    assert seat.champion == "h6" and seat.history[0] == "h5"
    w = dict(zip(r.uids, r.weights))
    assert w[6] == pytest.approx(0.90) and w[5] == pytest.approx(0.10)


def test_one_uid_per_head_hash_the_first_committer():
    r = run_v2_epoch(
        epoch_index=0, epoch_id="e0", questions=questions(0), pool_models=MODELS,
        prices=PRICES,
        verified_proofs={7: proof(0, oracle), 8: proof(0, oracle)},
        head_hashes={7: "same", 8: "same"}, coldkey_of={7: "a", 8: "b"},
        hotkey_of={7: "hk7", 8: "hk8"}, commit_block={7: 5, 8: 3}, commit_epoch={7: 0, 8: 0},
        records={7: burned("same"), 8: burned("same")}, exact_copies=set(),
        commitments={"hk7": ("same", 5), "hk8": ("same", 3)}, window=Window.empty(48),
        seat=SeatState(), owners={}, burn_uid=0, params=seat_params_from_config(),
        shares=(0.85, 0.10, 0.05), burn_in_questions=0, bench_category=CATEGORY,
    )
    assert dict(zip(r.uids, r.weights)) == {8: 1.0}


def test_a_champion_whose_owner_committed_another_head_is_retired():
    first = run(0, {5: oracle})
    r = run(1, {}, window=first.window, seat=first.decision.state, owners=first.owners,
            commitments={"hk5": ("a-new-head", 50)})
    (g,) = r.decision.state.ghosts
    assert g.reason == "retired"
    missed = run(1, {}, window=first.window, seat=first.decision.state, owners=first.owners,
                 commitments={"hk5": ("h5", 5)})
    assert missed.decision.state.ghosts[0].reason == "missed"


def test_the_reveal_carries_cells_tokens_failures_and_the_seat():
    r = run(0, {5: oracle})
    rv = r.reveal
    assert {"shared_grades", "failed", "champion", "v2_state"} <= set(rv)
    st = rv["v2_state"]
    assert set(st) == {"seat", "owners", "serving", "burn_in"}
    assert st["serving"] == {"h5": 5} and st["owners"] == {"h5": "hk5"}
    assert st["burn_in"] == {"h5": 10_000.0} and st["seat"]["champion"] == "h5"
    assert all({"question_id", "model", "grade", "completion_tokens"} <= set(c)
               for c in rv["shared_grades"])
    assert rv["champion"]["champion"] == "h5" and rv["failed"] == {"5": []}
    json.dumps(rv)   # publishable as is


def test_the_epoch_does_not_depend_on_proof_order():
    pols = {5: always("m/strong"), 6: oracle, 7: always("m/cheap")}
    base = run(3, pols)
    keys = list(pols)
    random.Random(1).shuffle(keys)
    again = run(3, {k: pols[k] for k in keys})
    assert again.uids == base.uids and again.weights == base.weights
    assert again.reveal == base.reveal


def test_window_and_seat_round_trip_through_disk(tmp_path):
    window, seat, owners = None, None, None
    for e in range(3):
        r = run(e, {5: oracle, 6: always("m/mid")}, window=window, seat=seat, owners=owners)
        window, seat, owners = r.window, r.decision.state, r.owners
    path = str(tmp_path / "w.npz")
    save_window(path, window)
    back = load_window(path, 48)
    assert sorted(back.slices) == sorted(window.slices)
    for e, sl in window.slices.items():
        b = back.slices[e]
        assert b.question_ids == sl.question_ids and b.hard == sl.hard
        assert np.array_equal(b.grades, sl.grades, equal_nan=True)
        assert np.array_equal(b.spends, sl.spends, equal_nan=True)
    for h, per in window.heads.items():
        for e, he in per.items():
            assert np.array_equal(back.heads[h][e].routed, he.routed)
            assert np.array_equal(back.heads[h][e].failed, he.failed)
    assert seat_from_dict(json.loads(json.dumps(seat_to_dict(seat)))) == seat
    assert load_window(str(tmp_path / "missing.npz"), 48).slices == {}


def test_a_resumed_champion_with_restarted_burn_in_is_paid_at_once():
    # Pins the payout contract (review fix 2): payout eligibility is the hold
    # rule, so a champion back inside its ghost window is paid even though its
    # E'' burn-in counter restarted at the miss.
    first = run(0, {5: oracle})
    lost = run(1, {}, window=first.window, seat=first.decision.state, owners=first.owners)
    assert lost.decision.event == "ghost" and dict(zip(lost.uids, lost.weights)) == {0: 1.0}
    restarted = {5: SimpleNamespace(evidence=Evidence(weights_hash="h5", n_total=10_000.0,
                                                      n_correct=8000.0, burn_in_n=0.0))}
    back = run(2, {5: oracle}, window=lost.window, seat=lost.decision.state,
               owners=lost.owners, burn_in=3000, records=restarted)
    assert back.decision.event == "hold" and back.decision.champion == "h5"
    assert dict(zip(back.uids, back.weights)) == {5: 1.0}


def test_a_missed_ghost_is_upgraded_when_its_owner_commits_a_new_head():
    first = run(0, {5: oracle})
    lost = run(1, {}, window=first.window, seat=first.decision.state, owners=first.owners,
               commitments={"hk5": ("h5", 5)})
    assert lost.decision.state.ghosts[0].reason == "missed"
    moved = run(2, {}, window=lost.window, seat=lost.decision.state, owners=lost.owners,
                commitments={"hk5": ("a-new-head", 50)})
    assert moved.decision.state.ghosts[0].reason == "retired"


def test_v2_state_takes_extra_fields_but_not_reserved_ones():
    from fugal_subnet.v2_epoch import v2_state_for_reveal

    block = v2_state_for_reveal(seat=SeatState(), owners={}, serving={}, burn_in={},
                                extra={"first_commit": {"h": 7}})
    assert block["first_commit"] == {"h": 7}
    with pytest.raises(ValueError):
        v2_state_for_reveal(seat=SeatState(), owners={}, serving={}, burn_in={},
                            extra={"seat": {}})


def _copier_epoch(epoch, proofs, *, window, seat, owners, commitments):
    """Author on hk5/uid 5 owns h5; a copier on hk3/uid 3 (a lower uid, so
    it would win any uid-order tie) serves the same bytes."""
    return run_v2_epoch(
        epoch_index=epoch, epoch_id=f"e{epoch:08d}", questions=questions(epoch),
        pool_models=MODELS, prices=PRICES, verified_proofs=proofs,
        head_hashes={u: "h5" for u in proofs}, coldkey_of={u: f"ck{u}" for u in proofs},
        hotkey_of={u: f"hk{u}" for u in proofs}, commit_block={5: 100, 3: 200},
        commit_epoch={u: 0 for u in proofs}, records={u: burned("h5") for u in proofs},
        exact_copies=set(), commitments=commitments, window=window, seat=seat,
        owners=owners, burn_uid=0, params=seat_params_from_config(),
        shares=(0.85, 0.10, 0.05), burn_in_questions=0, bench_category=CATEGORY,
    )


def test_a_copier_of_the_champions_bytes_is_never_paid_never_owner_never_retires_it():
    commits = {"hk5": ("h5", 100), "hk3": ("h5", 200)}
    first = _copier_epoch(0, {5: proof(0, oracle), 3: proof(0, oracle)}, window=Window.empty(48),
                          seat=SeatState(), owners={}, commitments=commits)
    assert first.owners["h5"] == "hk5"                       # first committer, not lowest uid
    assert dict(zip(first.uids, first.weights)) == {5: 1.0}  # the copier earns nothing
    # The author's proof fails; only the copier serves h5. h5 is not live.
    gone = _copier_epoch(1, {3: proof(1, oracle)}, window=first.window,
                         seat=first.decision.state, owners=first.owners, commitments=commits)
    assert gone.decision.event == "ghost" and gone.owners["h5"] == "hk5"
    assert 3 not in gone.uids
    # The copier commits something else: that does NOT retire the author's head.
    moved = _copier_epoch(2, {}, window=gone.window, seat=gone.decision.state,
                          owners=gone.owners,
                          commitments={"hk5": ("h5", 100), "hk3": ("other", 300)})
    assert moved.decision.state.ghosts[0].reason == "missed"
    # The author returns inside the ghost window and resumes.
    back = _copier_epoch(3, {5: proof(3, oracle)}, window=moved.window,
                         seat=moved.decision.state, owners=moved.owners, commitments=commits)
    assert back.decision.event == "hold" and dict(zip(back.uids, back.weights)) == {5: 1.0}


def test_a_new_hash_is_owned_by_its_first_committer_even_if_its_proof_failed():
    commits = {"hk5": ("h5", 100), "hk3": ("h5", 200)}
    r = _copier_epoch(0, {3: proof(0, oracle)}, window=Window.empty(48), seat=SeatState(),
                      owners={}, commitments=commits)
    assert r.owners["h5"] == "hk5" and r.decision.champion is None


def test_a_retried_epoch_reaches_the_same_result():
    # Crash between writing the window and the JSON state: the retry runs the
    # SAME epoch against the old seat but a window that already holds it.
    first = run(0, {5: oracle, 6: always("m/mid")})
    retry = run(0, {5: oracle, 6: always("m/mid")}, window=first.window)
    assert (retry.uids, retry.weights) == (first.uids, first.weights)
    assert retry.reveal == first.reveal
    assert sorted(retry.window.slices) == sorted(first.window.slices)


def test_a_once_rerun_of_a_processed_epoch_does_not_fail():
    first = run(0, {5: oracle})
    rerun = run(0, {5: oracle}, window=first.window, seat=first.decision.state,
                owners=first.owners)
    assert rerun.decision.champion == "h5" and dict(zip(rerun.uids, rerun.weights)) == {5: 1.0}


def test_hard_questions_get_the_hard_weight():
    from fugal_subnet.champion import pair_units

    w = run(0, {5: oracle, 6: always("m/strong")}).window
    on = pair_units(w, "h5", "h6", accuracy_value=1e300, hard_weight=2.0)
    off = pair_units(w, "h5", "h6", accuracy_value=1e300, hard_weight=1.0)
    assert np.array_equal(on.by_category["hard"], 2 * off.by_category["hard"])
    assert np.array_equal(on.by_category["math"], off.by_category["math"])


def test_the_validator_reads_the_v2_category_map_and_not_v1_dedup():
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    call = src[src.index("v2_result = run_v2_epoch("):]
    call = call[:call.index("uids, weights = v2_result.uids")]
    assert 'getattr(config_module, "BENCHMARK_CATEGORY_V2", None)' in call
    assert "exact_copies=set()," in call
    assert "allow_take=not v2_window_degraded" in call


def test_the_window_survives_a_crash_between_its_write_and_the_json(tmp_path):
    first = run(0, {5: oracle})
    save_window(str(tmp_path / "w.npz"), first.window)       # JSON never written
    retry = run(0, {5: oracle}, window=load_window(str(tmp_path / "w.npz"), 48))
    assert (retry.uids, retry.weights) == (first.uids, first.weights)


def test_a_ghost_keeps_seat_one_and_the_other_seats_do_not_shift():
    # Three champions in turn: h5, then h6, then h7. Then h7 goes missing.
    # During its ghost window seat 1 (85%) burns and seats 2-3 still pay h6
    # 10% and h5 5%, not h6 5% and nothing for h5.
    from fugal_subnet.champion import Ghost

    seat = SeatState(champion="h7", champion_since=10, history=("h6", "h5"))
    w = None
    for e in range(3):
        w = run(e, {5: oracle, 6: oracle, 7: oracle}, window=w).window
    ghost = run(3, {5: oracle, 6: oracle}, window=w, seat=seat,
                owners={"h5": "hk5", "h6": "hk6", "h7": "hk7"})
    assert ghost.decision.event == "ghost"
    assert ghost.decision.state.ghosts == (Ghost("h7", 3 + 3 - 1, "missed"),)
    weights = dict(zip(ghost.uids, ghost.weights))
    assert weights == pytest.approx({6: 0.10, 5: 0.05, 0: 0.85})
