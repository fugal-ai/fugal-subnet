"""Protocol v2 epoch wiring: proofs -> cells -> window -> copy rule -> seat -> weights.

docs/INVARIANTS.md, the v2 I3/I4 entries ("one champion", "seats, not
shares", "shared grades", "v2 pays from the paired rule alone") and the copy
rule entry.
"""
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from fugal_subnet.champion import SeatState, Window
from fugal_subnet.copy_rule import Commitment, CopyState
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
BPE = 300


def H(name) -> str:
    """A well-formed weights hash for a test head."""
    return hashlib.sha256(f"head-{name}".encode()).hexdigest()


def questions(epoch):
    return [{"question_id": f"e{epoch}q{i}", "prompt": "p" * (10 + i), "benchmark": BENCH[i]}
            for i in range(Q)]


def difficulty(epoch):
    return np.random.default_rng(epoch).random(Q)


def oracle(d):
    return [next((m for m in MODELS if x < ACC[m]), "m/strong") for x in d]


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


def C(uid, head, block, coldkey=None):
    return Commitment(f"hk{uid}", coldkey or f"ck{uid}", head, block)


def run(epoch, policies, *, window=None, seat=None, copy=None, commits=None, burn_in=0,
        blocks=None, records=None, heads=None, coldkeys=None):
    """One v2 epoch. Epoch e is chain epoch e + 1, so every head's first-commit
    epoch (>= 1) is at or before it. By default UID u serves head H(u) from
    hotkey hk{u} under coldkey ck{u}, committed at block `blocks[u]` (= u)."""
    uids = sorted(policies)
    heads = heads or {u: H(u) for u in uids}
    blocks = blocks or {u: u for u in uids}
    coldkeys = coldkeys or {u: f"ck{u}" for u in uids}
    if commits is None:
        commits = [C(u, heads[u], blocks[u], coldkeys[u]) for u in uids]
    e = epoch + 1
    return run_v2_epoch(
        epoch_index=e, epoch_id=f"e{e:08d}", boundary_block=e * BPE, blocks_per_epoch=BPE,
        questions=questions(epoch), pool_models=MODELS, prices=PRICES,
        verified_proofs={u: proof(epoch, policies[u]) for u in uids},
        head_hashes=heads, coldkey_of=coldkeys, hotkey_of={u: f"hk{u}" for u in uids},
        records=records if records is not None else {u: burned(heads[u]) for u in uids},
        commitments=commits, window=window or Window.empty(48), seat=seat or SeatState(),
        copy_state=copy or CopyState(), burn_uid=0, params=seat_params_from_config(),
        shares=(0.85, 0.10, 0.05), burn_in_questions=burn_in, bench_category=CATEGORY,
    )


def chain(epochs, policies, **kw):
    """Run consecutive epochs, threading window, seat and copy state."""
    r = None
    for e in epochs:
        kwargs = dict(kw)
        if r is not None:
            kwargs.update(window=r.window, seat=r.decision.state, copy=r.copy_state)
        pols = policies(e) if callable(policies) else policies
        r = run(e, pols, **kwargs)
    return r


def w(r):
    return dict(zip(r.uids, r.weights))


# --- cells, spend, hard ------------------------------------------------------------


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


def test_hard_questions_get_the_hard_weight():
    from fugal_subnet.champion import pair_units

    win = run(0, {5: oracle, 6: always("m/strong")}).window
    on = pair_units(win, H(5), H(6), accuracy_value=1e300, hard_weight=2.0)
    off = pair_units(win, H(5), H(6), accuracy_value=1e300, hard_weight=1.0)
    assert np.array_equal(on.by_category["hard"], 2 * off.by_category["hard"])
    assert np.array_equal(on.by_category["math"], off.by_category["math"])


# --- the seat and the payout ----------------------------------------------------------


def test_first_epoch_seats_the_most_senior_head_by_first_commit():
    r = run(0, {5: oracle, 6: always("m/strong")}, blocks={5: 2, 6: 1})
    assert r.decision.event == "first" and r.decision.champion == H(6)
    assert w(r) == {6: 1.0}     # unfilled seats pay the champion


def test_nothing_burns_for_frame_confidence_in_v2():
    r = run(0, {5: oracle})
    assert 0 not in r.uids and sum(r.weights) == pytest.approx(1.0)


def test_no_burned_in_head_means_burn():
    r = run(0, {5: oracle}, burn_in=10**9)
    assert r.decision.event == "empty" and w(r) == {0: 1.0}


def test_a_better_router_takes_the_seat_and_the_old_champion_keeps_a_seat():
    r = chain(range(30), {5: always("m/strong"), 6: oracle}, blocks={5: 1, 6: 2})
    assert r.decision.state.champion == H(6) and r.decision.state.history[0] == H(5)
    assert w(r)[6] == pytest.approx(0.90) and w(r)[5] == pytest.approx(0.10)


def test_a_ghost_keeps_seat_one_and_the_other_seats_do_not_shift():
    # Three champions in turn: H5, then H6, then H7. Then H7 goes missing.
    # During its ghost window seat 1 (85%) burns and seats 2-3 still pay H6
    # 10% and H5 5%.
    from fugal_subnet.champion import Ghost

    pols = {5: oracle, 6: oracle, 7: oracle}
    r = chain(range(3), pols)
    seat = SeatState(champion=H(7), champion_since=1, history=(H(6), H(5)))
    ghost = run(3, {5: oracle, 6: oracle}, window=r.window, seat=seat, copy=r.copy_state,
                commits=[C(u, H(u), u) for u in (5, 6, 7)])
    assert ghost.decision.event == "ghost"
    assert ghost.decision.state.ghosts == (Ghost(H(7), 4 + 3 - 1, "missed"),)
    assert w(ghost) == pytest.approx({6: 0.10, 5: 0.05, 0: 0.85})


def test_a_resumed_champion_with_restarted_burn_in_is_paid_at_once():
    # Payout eligibility is the hold rule: a champion back inside its ghost
    # window is paid although its E'' burn-in counter restarted at the miss.
    first = run(0, {5: oracle})
    lost = run(1, {}, window=first.window, seat=first.decision.state, copy=first.copy_state,
               commits=[C(5, H(5), 5)])
    assert lost.decision.event == "ghost" and w(lost) == {0: 1.0}
    restarted = {5: SimpleNamespace(evidence=Evidence(weights_hash=H(5), n_total=10_000.0,
                                                      n_correct=8000.0, burn_in_n=0.0))}
    back = run(2, {5: oracle}, window=lost.window, seat=lost.decision.state,
               copy=lost.copy_state, burn_in=3000, records=restarted)
    assert back.decision.event == "hold" and back.decision.champion == H(5)
    assert w(back) == {5: 1.0}


# --- the copy rule, through the wiring ----------------------------------------------


def test_one_uid_per_head_hash_its_first_committing_coldkey():
    same = H("same")
    r = run(0, {7: oracle, 8: oracle}, heads={7: same, 8: same},
            commits=[C(7, same, 5), C(8, same, 3)])
    assert w(r) == {8: 1.0}
    assert r.reveal["copy_verdicts"] == {"7": "copy", "8": "owner"}


def test_a_champion_whose_owner_committed_another_head_is_retired():
    first = run(0, {5: oracle})
    r = run(1, {}, window=first.window, seat=first.decision.state, copy=first.copy_state,
            commits=[C(5, H("new"), 400)])
    (g,) = r.decision.state.ghosts
    assert g.reason == "retired"
    missed = run(1, {}, window=first.window, seat=first.decision.state, copy=first.copy_state,
                 commits=[C(5, H(5), 5)])
    assert missed.decision.state.ghosts[0].reason == "missed"


def test_a_missed_ghost_is_upgraded_when_its_owner_commits_a_new_head():
    first = run(0, {5: oracle})
    lost = run(1, {}, window=first.window, seat=first.decision.state, copy=first.copy_state,
               commits=[C(5, H(5), 5)])
    assert lost.decision.state.ghosts[0].reason == "missed"
    moved = run(2, {}, window=lost.window, seat=lost.decision.state, copy=lost.copy_state,
                commits=[C(5, H("new"), 700)])
    assert moved.decision.state.ghosts[0].reason == "retired"


def _copier(epoch, serving, *, prev=None, author_commit=True, copier_head=None):
    """Author UID 5 (hk5/ck5) committed H(5) at block 100; copier UID 3
    (hk3/ck3, a lower UID) serves the same bytes, committed at block 200."""
    pols = {u: oracle for u in serving}
    commits = []
    if author_commit:
        commits.append(C(5, H(5), 100))
    commits.append(C(3, copier_head or H(5), 200))
    kw = {} if prev is None else dict(window=prev.window, seat=prev.decision.state,
                                      copy=prev.copy_state)
    return run(epoch, pols, heads={u: H(5) for u in serving}, commits=commits, **kw)


def test_a_copier_of_the_champions_bytes_is_never_paid_never_owner_never_retires_it():
    first = _copier(0, (5, 3))
    assert w(first) == {5: 1.0}                                   # the copier earns nothing
    assert first.reveal["copy_verdicts"] == {"3": "copy", "5": "owner"}
    assert first.owners[H(5)] == "hk5"
    # The author's proof fails; only the copier serves H(5). H(5) is not live.
    gone = _copier(1, (3,), prev=first)
    assert gone.decision.event == "ghost" and 3 not in gone.uids
    assert gone.owners[H(5)] == "hk5"
    # The copier commits something else: that does NOT retire the author's head.
    moved = _copier(2, (), prev=gone, copier_head=H("other"))
    assert moved.decision.state.ghosts[0].reason == "missed"
    # The author returns inside the ghost window and resumes.
    back = _copier(3, (5, 3), prev=moved)
    assert back.decision.event == "hold" and w(back) == {5: 1.0}


def test_a_new_hash_is_owned_by_its_first_committer_even_if_its_proof_failed():
    r = _copier(0, (3,))
    assert r.owners[H(5)] == "hk5" and r.decision.champion is None


def test_a_mimic_on_another_coldkey_cannot_unseat_the_champion():
    # v1 routing-similarity dedup does not decide v2 eligibility: a junior
    # head that routes exactly like the champion (same answers, different
    # bytes) is a different head that has to beat the champion by the margin,
    # which an identical router cannot. The champion keeps the seat and pay.
    r = chain(range(12), {5: oracle, 9: oracle}, blocks={5: 1, 9: 2})
    assert r.decision.state.champion == H(5) and w(r) == {5: 1.0}


# --- the reveal, restarts, order -----------------------------------------------------


def test_the_reveal_carries_cells_tokens_failures_verdicts_and_the_seat():
    r = run(0, {5: oracle})
    rv = r.reveal
    assert {"shared_grades", "failed", "champion", "copy_verdicts", "v2_state"} <= set(rv)
    assert all({"question_id", "model", "grade", "completion_tokens"} <= set(c)
               for c in rv["shared_grades"])
    st = rv["v2_state"]
    assert set(st) == {"seat", "owners", "serving", "burn_in", "copy_rule"}
    assert st["serving"] == {H(5): 5} and st["owners"] == {H(5): "hk5"}
    assert st["burn_in"] == {H(5): 10_000.0} and st["seat"]["champion"] == H(5)
    assert CopyState.from_reveal(st["copy_rule"]) == r.copy_state
    assert rv["champion"]["champion"] == H(5) and rv["failed"] == {"5": []}
    json.dumps(rv)   # publishable as is


def test_v2_state_takes_extra_fields_but_not_reserved_ones():
    from fugal_subnet.v2_epoch import v2_state_for_reveal

    block = v2_state_for_reveal(seat=SeatState(), owners={}, serving={}, burn_in={},
                                extra={"first_commit": {"h": 7}})
    assert block["first_commit"] == {"h": 7}
    with pytest.raises(ValueError):
        v2_state_for_reveal(seat=SeatState(), owners={}, serving={}, burn_in={},
                            extra={"seat": {}})


def test_the_epoch_does_not_depend_on_proof_order():
    pols = {5: always("m/strong"), 6: oracle, 7: always("m/cheap")}
    base = run(3, pols)
    keys = list(pols)
    random.Random(1).shuffle(keys)
    again = run(3, {k: pols[k] for k in keys})
    assert again.uids == base.uids and again.weights == base.weights
    assert again.reveal == base.reveal


def test_a_retried_epoch_reaches_the_same_result():
    # Crash between writing the window and the JSON state: the retry runs the
    # SAME epoch against the old seat and copy state, with a window that
    # already holds the epoch.
    first = run(0, {5: oracle, 6: always("m/mid")})
    retry = run(0, {5: oracle, 6: always("m/mid")}, window=first.window)
    assert (retry.uids, retry.weights) == (first.uids, first.weights)
    assert retry.reveal == first.reveal


def test_a_once_rerun_of_a_processed_epoch_does_not_fail():
    first = run(0, {5: oracle})
    rerun = run(0, {5: oracle}, window=first.window, seat=first.decision.state,
                copy=first.copy_state)
    assert rerun.decision.champion == H(5) and w(rerun) == {5: 1.0}
    assert rerun.copy_state == first.copy_state


def test_the_window_survives_a_crash_between_its_write_and_the_json(tmp_path):
    first = run(0, {5: oracle})
    save_window(str(tmp_path / "w.npz"), first.window)       # JSON never written
    retry = run(0, {5: oracle}, window=load_window(str(tmp_path / "w.npz"), 48))
    assert (retry.uids, retry.weights) == (first.uids, first.weights)


def test_window_and_seat_round_trip_through_disk(tmp_path):
    r = chain(range(3), {5: oracle, 6: always("m/mid")})
    path = str(tmp_path / "w.npz")
    save_window(path, r.window)
    back = load_window(path, 48)
    assert sorted(back.slices) == sorted(r.window.slices)
    for e, sl in r.window.slices.items():
        b = back.slices[e]
        assert b.question_ids == sl.question_ids and b.hard == sl.hard
        assert np.array_equal(b.grades, sl.grades, equal_nan=True)
        assert np.array_equal(b.spends, sl.spends, equal_nan=True)
    for h, per in r.window.heads.items():
        for e, he in per.items():
            assert np.array_equal(back.heads[h][e].routed, he.routed)
            assert np.array_equal(back.heads[h][e].failed, he.failed)
    seat = r.decision.state
    assert seat_from_dict(json.loads(json.dumps(seat_to_dict(seat)))) == seat
    assert load_window(str(tmp_path / "missing.npz"), 48).slices == {}


def test_the_validator_reads_the_v2_category_map_and_the_copy_rule():
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    call = src[src.index("v2_result = run_v2_epoch("):]
    call = call[:call.index("uids, weights = v2_result.uids")]
    assert 'getattr(config_module, "BENCHMARK_CATEGORY_V2", None)' in call
    assert "commitments=snapshot" in call and "copy_state=v2_copy" in call
    assert "exact_copies" not in call
    assert "allow_take=not v2_window_degraded" in call
    assert "burn_in_questions=V2_BURN_IN_QUESTIONS" in call
