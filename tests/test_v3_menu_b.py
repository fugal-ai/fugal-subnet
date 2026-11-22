"""Protocol v3 menu B: entries added at a block, with a grace period.

docs/INVARIANTS.md, "I1 — v3 menu B applies from one named boundary" and
docs/decisions/V3_MENU_B.md. Menu A before the block, menu B (A unchanged plus
the added entries, appended) from it; heads built for menu A stay valid and
keep their seat, burn-in and window; for MENU_GRACE_EPOCHS the added entries
are explored and pooled but scored for no head; after it a head must give odds
for every menu B entry to take or hold the odds seat.
"""
import dataclasses
import hashlib
import io
import json

import numpy as np
import pytest

from fugal_subnet import config, odds_score
from fugal_subnet import routing_protocol as rp
from fugal_subnet.champion import OddsParams, SeatState, Window
from fugal_subnet.copy_rule import CopyState
from fugal_subnet.explore_pool import ExploreAnswer
from fugal_subnet.v2_epoch import (
    OddsInput,
    load_window,
    run_v2_epoch,
    save_window,
    seat_params_from_config,
)
from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals
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
from tests.test_v2_rebuild import reveal_of, same_window, write
from tests.test_v3_epoch import BENCHES, DIM, difficulty_of, hidden_of

NEW = "m/new"
ACC_ALL = {**ACC, NEW: 0.75}
MENU_A = list(MODELS)
MENU_B = MENU_A + [NEW]
PRICES_B = {**PRICES, NEW: (3e-6, 3e-6)}
B_EPOCH = 4          # chain epoch index of the menu B block
GRACE = 3
UIDS = (5, 6, 8)


def _logit(p):
    return float(np.log(p / (1 - p)))


def sharp(models):
    """A head that predicts every model it has a row for (test_v3_epoch's 'sharp')."""
    W = np.zeros((len(models), DIM), dtype=np.float32)
    W[:, 0] = -60.0
    b = np.array([15.0 * _logit(ACC_ALL[m]) for m in models], dtype=np.float32)
    return W, b, list(models)


def flat(models):
    W = np.zeros((len(models), DIM), dtype=np.float32)
    b = np.array([_logit(ACC_ALL[m]) for m in models], dtype=np.float32)
    return W, b, list(models)


# uid 5: a flat menu A head; uid 6: a sharp menu A head (the champion);
# uid 8: the same sharp head with a row for the added entry (a menu B head).
HEADS = {5: flat(MENU_A), 6: sharp(MENU_A), 8: sharp(MENU_B)}


def answers_for(epoch, menu):
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


def menu_of_run(e, b_block):
    return rp.menu_for_epoch(3, e * BPE, b_block)


def run(epoch, uids, *, window, seat, copy, b_block=B_EPOCH * BPE, heads=HEADS):
    """One v3 epoch at chain epoch `epoch` on the menu its boundary picks."""
    boundary = epoch * BPE
    menu_id = rp.menu_for_epoch(3, boundary, b_block)
    pool_models = MENU_B if menu_id == rp.MENU_B else MENU_A
    unscored = frozenset([NEW]) if (menu_id == rp.MENU_B and b_block is not None
                                    and epoch < b_block // BPE + GRACE) else frozenset()
    answers, bench_of = answers_for(epoch, pool_models)
    oi = OddsInput(nonce=hashlib.sha256(f"n{epoch}".encode()).digest(), answers=answers,
                   benchmark_of=bench_of, heads={u: heads[u] for u in uids},
                   hidden_of=hidden_of, single_model_min_pairs=30,
                   category_targets={"knowledge": 0.4, "math": 0.3, "hard": 0.3},
                   menu=tuple(m for m in pool_models if m not in unscored), unscored=unscored)
    pols = {5: lambda d: ["m/strong"] * len(d), 6: oracle, 8: oracle}
    return run_v2_epoch(
        epoch_index=epoch, epoch_id=f"e{epoch:08d}", boundary_block=boundary, blocks_per_epoch=BPE,
        questions=questions(epoch), pool_models=pool_models,
        prices=PRICES_B if menu_id == rp.MENU_B else PRICES,
        verified_proofs={u: proof(epoch, pols[u]) for u in uids},
        head_hashes={u: H(u) for u in uids}, coldkey_of={u: f"ck{u}" for u in uids},
        hotkey_of={u: f"hk{u}" for u in uids}, records={u: burned(H(u)) for u in uids},
        commitments=[C(u, H(u), u) for u in uids], window=window or Window.empty(48),
        seat=seat or SeatState(), copy_state=copy or CopyState(), burn_uid=0,
        params=params(), shares=(0.85, 0.10, 0.05), burn_in_questions=0,
        bench_category=CATEGORY, odds_input=oi,
    ), {u: pols[u] for u in uids}


def live(epochs, schedule=lambda e: UIDS, b_block=B_EPOCH * BPE):
    window = seat = copy = None
    results, reveals = [], []
    for e in range(1, epochs + 1):
        r, pols = run(e, schedule(e), window=window, seat=seat, copy=copy, b_block=b_block)
        window, seat, copy = r.window, r.decision.state, r.copy_state
        results.append(r)
        rv = reveal_of(e, r, pols)
        rv["epoch_id"] = f"e{e:08d}"                  # this runner's epoch e is chain epoch e
        rv["models"] = MENU_B if menu_of_run(e, b_block) == rp.MENU_B else MENU_A
        reveals.append(json.loads(json.dumps(rv)))
    return results, reveals


# --- the menu files and the epoch helpers -----------------------------------------


def test_menu_b_is_menu_a_with_the_added_entries_appended():
    a, b = rp.models_for_protocol(3), rp.models_for_protocol(3, rp.MENU_B)
    assert len(a) == 53 and len(b) == 60 and b[:53] == a
    assert rp.added_entries() == sorted([
        "openai/gpt-6.2-astra@low", "openai/gpt-6.2-astra@high", "openai/gpt-6.2-astra@max",
        "openai/gpt-6.2-sol@low", "openai/gpt-6.2-sol@medium", "openai/gpt-6.2-sol@high",
        "openai/gpt-6.2-sol@max"])
    pa, pb = rp.price_table(3), rp.price_table(3, rp.MENU_B)
    assert all(pb[k] == v for k, v in pa.items())
    assert pb["openai/gpt-6.2-astra@max"] == (8.0 / 1e6, 40.0 / 1e6)
    assert pb["openai/gpt-6.2-sol@low"] == (2.0 / 1e6, 10.0 / 1e6)
    req = rp.request_for(3, "openai/gpt-6.2-sol@medium", 16384, menu=rp.MENU_B)
    assert req == {"model": "openai/gpt-6.2-sol", "max_tokens": 128000,
                   "reasoning": {"effort": "medium"},
                   "provider": {"order": ["openai", "azure"], "allow_fallbacks": False}}
    with pytest.raises(KeyError):
        rp.request_for(3, "openai/gpt-6.2-sol@medium", 16384)      # not on menu A
    with pytest.raises(ValueError):
        rp.price_table(2, rp.MENU_B)
    # Measured token rows, not provisional.
    man = json.loads(rp.manifest_path(3, rp.MENU_B).read_text())
    rows = {r["id"]: r for r in man["models"]}
    assert man["status"] == "reviewed"
    assert all(rows[e]["samples"] == 10 and rows[e]["basis"].startswith("measured")
               for e in rp.added_entries())


def test_the_epoch_menu_and_grace_are_keyed_by_the_boundary():
    blk, bpe = 9_531_000, 360
    assert rp.menu_for_epoch(3, blk - bpe, blk) == rp.MENU_A
    assert rp.menu_for_epoch(3, blk, blk) == rp.MENU_B
    assert rp.menu_for_epoch(3, blk, None) == rp.MENU_A
    assert rp.menu_for_epoch(2, blk, blk) == rp.MENU_A
    added = frozenset(rp.added_entries())
    assert rp.grace_excluded(3, blk - bpe, blk, bpe, 20) == frozenset()
    assert rp.grace_excluded(3, blk, blk, bpe, 20) == added
    assert rp.grace_excluded(3, blk + 19 * bpe, blk, bpe, 20) == added
    assert rp.grace_excluded(3, blk + 20 * bpe, blk, bpe, 20) == frozenset()
    assert rp.grace_excluded(3, blk, blk, bpe, 0) == frozenset()
    assert len(rp.scored_menu(3, blk, blk, bpe, 20)) == 53
    assert len(rp.scored_menu(3, blk + 20 * bpe, blk, bpe, 20)) == 60
    assert rp.scored_menu(3, blk - bpe, blk, bpe, 20) == tuple(rp.models_for_protocol(3))


def test_the_menu_b_block_is_compiled_for_finney_one_epoch_after_the_odds_seat(monkeypatch):
    finney, other = config.FINNEY_GENESIS_HASH, "0x" + "ab" * 32
    odds, menu_b = config.odds_seat_block(finney), config.menu_b_block(finney)
    assert menu_b == 9_531_000 and menu_b % 360 == 0
    assert menu_b - odds == 360                       # they do not coincide
    assert config.menu_b_block(other) is None
    monkeypatch.setattr(config, "MENU_B_BLOCK_ENV", 9_000_000)
    with pytest.raises(ValueError, match="fixed by the release"):
        config.menu_b_block(finney)
    with pytest.raises(ValueError, match="must not be before"):
        monkeypatch.setattr(config, "V3_ACTIVATION_BLOCK_ENV", 9_100_000)
        config.menu_b_block(other)
    monkeypatch.setattr(config, "MENU_B_BLOCK_ENV", 9_100_000)
    assert config.menu_b_block(other) == 9_100_000


def test_the_odds_seat_then_menu_b_apply_in_order():
    """Finney's blocks: the seat goes on odds at 9,530,640, menu B arrives one
    epoch later, and its grace ends 20 epochs after that. Each rule turns on at
    its own boundary, whatever the others do."""
    v3, odds, mb = 9_527_760, 9_530_640, 9_531_000
    seat = [seat_params_from_config(b, None, protocol_version=3, odds_seat_block=odds).odds
            is not None for b in (odds - 360, odds, mb, mb + 20 * 360)]
    menu = [rp.menu_for_epoch(3, b, mb) for b in (odds - 360, odds, mb, mb + 20 * 360)]
    grace = [bool(rp.grace_excluded(3, b, mb, 360, config.MENU_GRACE_EPOCHS))
             for b in (odds - 360, odds, mb, mb + 19 * 360, mb + 20 * 360)]
    assert seat == [False, True, True, True]
    assert menu == ["a", "a", "b", "b"]
    assert grace == [False, False, True, True, False]
    assert v3 < odds < mb


def test_the_miner_takes_its_protocol_and_menu_from_the_compiled_blocks(monkeypatch):
    finney, other = config.FINNEY_GENESIS_HASH, "0x" + "ab" * 32
    monkeypatch.setattr(config, "FINNEY_V4_ACTIVATION_BLOCK", None)   # v3 and menu B only
    monkeypatch.setattr(config, "V3_ACTIVATION_BLOCK_ENV", None)
    monkeypatch.setattr(config, "MENU_B_BLOCK_ENV", None)
    assert config.protocol_for_chain(finney) == 3      # a compose with no variable is still v3
    assert config.protocol_for_chain(None) == 3        # unknown chain: cannot be shown not finney
    assert config.protocol_for_chain(other) in (1, 2)
    monkeypatch.setattr(config, "V3_ACTIVATION_BLOCK_ENV", 9_000)
    assert config.protocol_for_chain(other) == 3
    with pytest.raises(ValueError, match="fixed by the release"):
        config.protocol_for_chain(finney)


def test_exploration_draws_the_added_entries_from_the_block():
    from fugal_subnet.epoch_plan import explore_shares
    from fugal_subnet.explore_weights import assign_by_share
    a = explore_shares(3, rp.models_for_protocol(3))
    b = explore_shares(3, rp.models_for_protocol(3, rp.MENU_B))
    assert set(b) - set(a) == set(rp.added_entries())
    assert all(b[e] > 0 for e in rp.added_entries())
    chosen = [{"question_id": f"q{i:05d}"} for i in range(2000)]
    drawn_b = set(assign_by_share(b"\x01" * 32, chosen, b).values())
    drawn_a = set(assign_by_share(b"\x01" * 32, chosen, a).values())
    assert drawn_b >= set(rp.added_entries())
    assert not drawn_a & set(rp.added_entries())


def _head_bytes(models, menu):
    from fugal_subnet.vendor import success_contract as sc
    z = sc.make_head(np.zeros((len(models), 1024)), np.zeros(len(models)), models,
                     json.loads(rp.manifest_path(3, menu).read_text()), "menu-b-test")
    buf = io.BytesIO()
    np.savez(buf, **z)
    return buf.getvalue()


def test_a_menu_a_head_is_valid_on_both_menus_and_a_menu_b_head_on_b_only():
    from fugal_subnet.head_eval import load_head_from_npz
    a_head = load_head_from_npz(_head_bytes(rp.models_for_protocol(3)[:5], rp.MENU_A))
    b_head = load_head_from_npz(_head_bytes(
        rp.models_for_protocol(3)[:2] + rp.added_entries()[:2], rp.MENU_B))
    rp.benchmark_costs_for(a_head, 3, require_reviewed=True)
    rp.benchmark_costs_for(a_head, 3, require_reviewed=True, menu=rp.MENU_B)
    rp.benchmark_costs_for(b_head, 3, require_reviewed=True, menu=rp.MENU_B)
    with pytest.raises(ValueError):
        rp.benchmark_costs_for(b_head, 3)
    assert rp.head_menu(a_head) == rp.MENU_A and rp.head_menu(b_head) == rp.MENU_B


# --- the window across the block ------------------------------------------------


def test_a_menu_a_champion_keeps_its_seat_and_window_across_the_block():
    results, _ = live(B_EPOCH + GRACE - 1, schedule=lambda e: (5, 6))
    before = results[B_EPOCH - 2]                  # the last menu A epoch
    after = results[-1]                            # inside the grace period
    assert before.decision.champion == H(6)
    assert after.decision.champion == H(6)
    assert after.decision.state.champion_since == before.decision.state.champion_since
    # The window spans the block: menu A slices (3 models) and menu B (4).
    shapes = {e: sl.grades.shape[1] for e, sl in after.window.slices.items()}
    assert shapes[B_EPOCH - 1] == 3 and shapes[B_EPOCH] == 4
    assert sorted(after.window.heads[H(6)]) == list(range(1, B_EPOCH + GRACE))


def test_the_added_entries_are_pooled_but_unscored_during_grace():
    results, reveals = live(B_EPOCH + 1)
    r, rv = results[-1], reveals[-1]
    ep = r.window.pairs[B_EPOCH]
    assert NEW in ep.models                               # explored and pooled
    assert ep.scored is not None and not ep.scored[[m == NEW for m in ep.models]].any()
    assert rv["odds_unscored_entries"] == [NEW]
    assert any(p["model"] == NEW for p in rv["exploration_pool"])
    # A menu A head misses no row on the pairs it is scored on.
    assert odds_score.missing_rows(r.window, H(6), None, run_only=False) == 0
    # Before the block nothing is masked and the reveal has no new field.
    assert results[B_EPOCH - 2].window.pairs[B_EPOCH - 1].scored is None
    assert "odds_unscored_entries" not in reveals[B_EPOCH - 2]


def test_leaving_the_added_entries_out_during_grace_gains_nothing():
    """ATTACK: during grace, a head without the new rows vs the same head with
    them. Scored identically: the new pairs count for nobody."""
    results, _ = live(B_EPOCH + GRACE - 1)
    win = results[-1].window
    ref = odds_score.reference(win.pairs, 20)
    a = odds_score.skill_units(win, ref, H(6), run_only=False)
    b = odds_score.skill_units(win, ref, H(8), run_only=False)
    assert a.n_pairs == b.n_pairs and a.total == b.total
    vs = odds_score.brier_units(win, ref, H(8), H(6))
    assert vs.total == 0


def test_after_grace_a_head_without_the_added_entries_cannot_hold_or_take_the_seat():
    end = B_EPOCH + GRACE
    results, _ = live(end + 6)
    first_full = results[end - 1]
    assert first_full.window.pairs[end].scored is None    # every pair counts again
    assert odds_score.missing_rows(first_full.window, H(6), None, run_only=False) > 0
    assert odds_score.missing_rows(first_full.window, H(8), None, run_only=False) == 0
    assert results[-1].decision.champion != H(6)
    assert results[-1].decision.champion in (H(8), None)


def test_a_rebuild_across_the_block_and_grace_reproduces_the_live_window(tmp_path):
    results, reveals = live(B_EPOCH + GRACE + 2)
    write(tmp_path, reveals)
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES_B,
                                 bench_category=CATEGORY, tables={3: (PRICES_B, CATEGORY)})
    last = results[-1]
    same_window(rb.window, last.window)
    for e, ep in last.window.pairs.items():
        got = rb.window.pairs[e]
        assert got.models == ep.models
        assert (got.scored is None) == (ep.scored is None)
        if ep.scored is not None:
            assert np.array_equal(got.scored, ep.scored)
    for h, per in last.window.odds.items():
        for e, o in per.items():
            assert np.array_equal(rb.window.odds[h][e], o)
    assert rb.seat == last.decision.state


def test_the_saved_window_keeps_the_scored_mask(tmp_path):
    results, _ = live(B_EPOCH + 1)
    path = str(tmp_path / "w.npz")
    save_window(path, results[-1].window)
    back = load_window(path, 48)
    assert np.array_equal(back.pairs[B_EPOCH].scored, results[-1].window.pairs[B_EPOCH].scored)
    assert back.pairs[B_EPOCH - 1].scored is None


def test_without_menu_b_nothing_changes():
    """No block: every epoch is menu A and nothing is masked, published or
    priced differently (the release before the block, byte for byte)."""
    with_b, rv_b = live(B_EPOCH - 1)
    without, rv = live(B_EPOCH - 1, b_block=None)
    assert rv_b == rv
    assert [r.weights for r in with_b] == [r.weights for r in without]
