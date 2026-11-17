"""Protocol v2, one epoch: from verified proofs to the seat and the weight vector.

The pure modules (shared_grades, champion, payout, evidence's E'' rule) know
nothing about proofs, metagraphs or files. This module is the seam: it turns
what the validator has at the end of verification into their inputs, and
turns their outputs into a weight vector, reveal fields and persisted state.
Everything here is deterministic given the epoch's inputs (I1) and every
input is in the published reveal, so the window and the seat are rebuilt from
reveals exactly.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from fugal_subnet.champion import (
    Candidate,
    EpochPairs,
    EpochSlice,
    Ghost,
    HeadEpoch,
    OddsParams,
    Pending,
    SeatDecision,
    SeatParams,
    SeatState,
    Window,
    seat_step,
)
from fugal_subnet.pricing import policy_cost

HARD_CATEGORY = "hard"


def category_of(question: Mapping, bench_category: Mapping[str, str] | None) -> str:
    """A question's category: the benchmark's category when one is configured,
    else the benchmark itself (so the bootstrap still stratifies by benchmark)."""
    bench = str(question.get("benchmark", ""))
    if bench_category:
        return str(bench_category.get(bench, bench))
    return bench


def build_epoch_slice(
    epoch: int,
    questions: list[Mapping],
    pool_models: list[str],
    cells: Mapping[tuple[str, str], float],
    cell_tokens: Mapping[tuple[str, str], float],
    prices: Mapping[str, tuple[float, float]],
    bench_category: Mapping[str, str] | None = None,
) -> EpochSlice:
    """The epoch's slice in the global model index space.

    A cell's spend is the pool's prompt tokens for the question and the cell's
    coldkey-median attested completion tokens, at the pinned prices: measured,
    shared, and never read from the frame. Cells nobody answered are NaN in
    both arrays; a head's own non-failed answer always has its cell.
    """
    q = len(questions)
    m = len(pool_models)
    grades = np.full((q, m), np.nan, dtype=np.float64)
    spends = np.full((q, m), np.nan, dtype=np.float64)
    for i, question in enumerate(questions):
        qid = question["question_id"]
        prompt = question.get("prompt", "")
        for j, model in enumerate(pool_models):
            g = cells.get((qid, model))
            if g is None or model not in prices:
                continue
            grades[i, j] = float(g)
            spends[i, j] = policy_cost(prompt, model, prices, float(cell_tokens.get((qid, model), 0.0)))
    cats = tuple(category_of(x, bench_category) for x in questions)
    return EpochSlice(
        epoch=int(epoch),
        question_ids=tuple(x["question_id"] for x in questions),
        categories=cats,
        hard=tuple(c == HARD_CATEGORY for c in cats),
        grades=grades,
        spends=spends,
    )


def head_epoch(proof, question_ids: tuple[str, ...], model_index: Mapping[str, int]) -> HeadEpoch | None:
    """One verified proof's scored answers aligned to the slice, or None if they do not cover it.

    A verified proof answers exactly the assigned slice (verify_proof checks
    the question set), so a gap here is a bug upstream; it is refused rather
    than guessed at.
    """
    by_q = {r.question_id: r for r in proof.scored_results}
    routed = np.empty(len(question_ids), dtype=np.int64)
    failed = np.zeros(len(question_ids), dtype=bool)
    for i, qid in enumerate(question_ids):
        r = by_q.get(qid)
        if r is None or r.routed_model not in model_index:
            return None
        routed[i] = model_index[r.routed_model]
        failed[i] = bool(r.failed)
    return HeadEpoch(routed=routed, failed=failed)


# --- persistence -------------------------------------------------------------
#
# The window is arrays, so it is kept beside the JSON state file as an .npz
# (loaded with allow_pickle=False: the file is local, but the rule has no
# exceptions). The seat is small and goes into the JSON state.


def save_window(path: str, window: Window) -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = {"size": window.size, "slices": {}, "heads": {}}
    for e, sl in sorted(window.slices.items()):
        meta["slices"][str(e)] = {
            "question_ids": list(sl.question_ids),
            "categories": list(sl.categories),
            "hard": [bool(x) for x in sl.hard],
        }
        arrays[f"g_{e}"] = sl.grades
        arrays[f"s_{e}"] = sl.spends
    for i, (h, per) in enumerate(sorted(window.heads.items())):
        meta["heads"][h] = {"key": i, "epochs": sorted(per)}
        for e, he in sorted(per.items()):
            arrays[f"r_{i}_{e}"] = he.routed
            arrays[f"f_{i}_{e}"] = he.failed
        # v3: the head's odds on that epoch's pooled pairs.
        odds_epochs = sorted(window.odds.get(h, {}))
        if odds_epochs:
            meta["heads"][h]["odds_epochs"] = odds_epochs
            for e in odds_epochs:
                arrays[f"p_{i}_{e}"] = window.odds[h][e]
    if window.pairs:
        meta["pairs"] = {}
        for e, ep in sorted(window.pairs.items()):
            meta["pairs"][str(e)] = {
                "question_ids": list(ep.question_ids), "models": list(ep.models),
                "benchmarks": list(ep.benchmarks), "categories": list(ep.categories),
            }
            arrays[f"x_y_{e}"] = ep.y
            arrays[f"x_p_{e}"] = ep.prompt_tokens
            arrays[f"x_c_{e}"] = ep.completion_tokens
            if ep.weight is not None:
                arrays[f"x_w_{e}"] = ep.weight
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_window(path: str, size: int) -> Window:
    if not os.path.exists(path):
        return Window.empty(size)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        slices = {
            int(e): EpochSlice(
                epoch=int(e),
                question_ids=tuple(v["question_ids"]),
                categories=tuple(v["categories"]),
                hard=tuple(bool(x) for x in v["hard"]),
                grades=np.array(z[f"g_{e}"], dtype=np.float64),
                spends=np.array(z[f"s_{e}"], dtype=np.float64),
            )
            for e, v in meta["slices"].items()
        }
        heads = {
            h: {
                int(e): HeadEpoch(
                    routed=np.array(z[f"r_{v['key']}_{e}"], dtype=np.int64),
                    failed=np.array(z[f"f_{v['key']}_{e}"], dtype=bool),
                )
                for e in v["epochs"]
            }
            for h, v in meta["heads"].items()
        }
        pairs = {
            int(e): EpochPairs(
                question_ids=tuple(v["question_ids"]), models=tuple(v["models"]),
                benchmarks=tuple(v["benchmarks"]), categories=tuple(v["categories"]),
                y=np.array(z[f"x_y_{e}"], dtype=np.int32),
                prompt_tokens=np.array(z[f"x_p_{e}"], dtype=np.int32),
                completion_tokens=np.array(z[f"x_c_{e}"], dtype=np.int32),
                weight=(np.array(z[f"x_w_{e}"], dtype=np.int64) if f"x_w_{e}" in z.files else None),
            )
            for e, v in meta.get("pairs", {}).items()
        }
        odds = {
            h: {int(e): np.array(z[f"p_{v['key']}_{e}"], dtype=np.int16)
                for e in v["odds_epochs"]}
            for h, v in meta["heads"].items() if v.get("odds_epochs")
        }
    # A changed window size is a changed rule; shrink (or keep) to the
    # configured size from the newest epoch, never silently extend.
    return Window(size=int(size), slices=slices, heads=heads, pairs=pairs, odds=odds)


def seat_to_dict(state: SeatState) -> dict:
    out = {
        "champion": state.champion,
        "champion_since": state.champion_since,
        "history": list(state.history),
        "ghosts": [{"head_hash": g.head_hash, "last_epoch": g.last_epoch, "reason": g.reason}
                   for g in state.ghosts],
        "high_water": state.high_water,
        "high_water_hard": state.high_water_hard,
    }
    # The confirmation rule's state, only when there is any: before its
    # activation block there is none, and the published seat (which is in the
    # consensus view, v2_sync) stays byte-identical to the one before the rule.
    if state.pending:
        out["pending"] = [{"head_hash": p.head_hash, "since_epoch": p.since_epoch, "bar": p.bar}
                          for p in state.pending]
    if state.confirm_from is not None:
        out["confirm_from"] = state.confirm_from
    return out


# A published seat is someone else's file (I2). At most one pending record per
# registered hotkey can exist; this only bounds what a hand-made reveal can load.
MAX_PENDING = 4096


def seat_from_dict(raw: Mapping | None) -> SeatState:
    if not raw:
        return SeatState()
    pending = raw.get("pending") or ()
    if not isinstance(pending, (list, tuple)) or len(pending) > MAX_PENDING:
        raise ValueError("seat state: `pending` must be a list of at most "
                         f"{MAX_PENDING} records")
    confirm_from = raw.get("confirm_from")
    if confirm_from is not None and (isinstance(confirm_from, bool)
                                     or not isinstance(confirm_from, int)):
        raise ValueError("seat state: `confirm_from` must be an epoch index")
    return SeatState(
        champion=raw.get("champion"),
        champion_since=raw.get("champion_since"),
        history=tuple(raw.get("history", ())),
        ghosts=tuple(Ghost(str(g["head_hash"]), int(g["last_epoch"]), str(g["reason"]))
                     for g in raw.get("ghosts", ())),
        high_water=float(raw.get("high_water", 0.0)),
        high_water_hard=float(raw.get("high_water_hard", 0.0)),
        pending=tuple(Pending(str(p["head_hash"]), int(p["since_epoch"]), str(p["bar"]))
                      for p in pending),
        confirm_from=confirm_from,
    )


def anchor_spend_per_question(model: str) -> float:
    """The anchor's expected spend per question: the reviewed v2 token
    manifest's mean input and output tokens for `model`, at the pinned v2
    prices. Pure function of pinned files, so every validator agrees."""
    import math

    from fugal_subnet.routing_protocol import manifest_path, price_table

    manifest = json.loads(manifest_path(2).read_text())
    rows = {r["id"]: r for r in manifest.get("models", [])}
    prices = price_table(2)
    if model not in rows or model not in prices:
        raise KeyError(f"anchor model {model!r} is not in the v2 token manifest and price table")
    r = rows[model]
    rate_in, rate_out = prices[model]
    spend = math.fsum([float(r["mean_in_tokens"]) * rate_in,
                       float(r["mean_out_tokens"]) * rate_out])
    if not spend > 0:
        raise ValueError(f"anchor model {model!r} has no positive expected spend")
    return spend


def seat_params_from_config(boundary_block: int | None = None,
                            confirm_block: int | None = None, *,
                            protocol_version: int = 2,
                            odds_seat_block: int | None = None) -> SeatParams:
    """The seat rules for the epoch whose boundary is `boundary_block`.

    The fresh-data confirmation rule and the shorter retired-ghost window are
    in force from the first epoch whose boundary is at or after
    `confirm_block` (config.confirm_activation_block). With either argument
    None they are not, and the result is the rule set from before them.

    Protocol v3 prices one extra correct answer at a pinned number of dollars
    (CHAMPION_VALUE_PER_CORRECT_USD) instead of a multiple of the anchor
    model's spend, and from `odds_seat_block` decides the seat on odds
    (OddsParams), with net value kept as a guard.
    """
    from fugal_subnet import config

    confirm = (confirm_block is not None and boundary_block is not None
               and int(boundary_block) >= int(confirm_block))
    v3 = int(protocol_version) >= 3
    on_odds = (v3 and odds_seat_block is not None and boundary_block is not None
               and int(boundary_block) >= int(odds_seat_block))
    if v3:
        # V = accuracy_value x anchor_spend (champion.pair_units); pinning V in
        # dollars is accuracy_value 1 at an "anchor spend" of V itself.
        accuracy_value, anchor = 1.0, float(config.CHAMPION_VALUE_PER_CORRECT_USD)
        if not anchor > 0:
            raise ValueError("CHAMPION_VALUE_PER_CORRECT_USD must be positive")
    else:
        accuracy_value = config.CHAMPION_ACCURACY_VALUE
        anchor = anchor_spend_per_question(config.CHAMPION_ANCHOR_MODEL)
    odds = OddsParams(
        margin_points=config.BRIER_MARGIN, min_pairs=config.ODDS_MIN_PAIRS,
        nv_noninferiority=config.NV_NONINFERIORITY, ref_prior=config.ODDS_REF_PRIOR,
        confirm_pairs=config.ODDS_CONFIRM_PAIRS,
    ) if on_odds else None
    return SeatParams(
        margin_points=config.CHAMPION_MARGIN,
        confidence=config.CHAMPION_CONFIDENCE,
        resamples=config.CHAMPION_BOOTSTRAP_RESAMPLES,
        accuracy_value=accuracy_value,
        hard_weight=config.CHAMPION_HARD_WEIGHT,
        hard_gate_min_questions=config.CHAMPION_HARD_GATE_MIN_QUESTIONS,
        ghost_off=config.CHAMPION_GHOST_OFF,
        ghost_retire=(config.CHAMPION_GHOST_RETIRE_CONFIRM if confirm
                      else config.CHAMPION_GHOST_RETIRE),
        wilson_confidence=config.WILSON_CONFIDENCE,
        min_accuracy=config.CHAMPION_MIN_ACCURACY,
        min_paired_questions=config.CHAMPION_MIN_PAIRED_QUESTIONS,
        anchor_spend=anchor,
        confirm_questions=config.CHAMPION_CONFIRM_QUESTIONS if confirm else 0,
        confirm_open_epochs=config.CHAMPION_CONFIRM_OPEN_EPOCHS if confirm else 0,
        odds=odds,
    )


def v2_state_for_reveal(
    *,
    seat: SeatState,
    owners: Mapping[str, str],
    serving: Mapping[str, int],
    burn_in: Mapping[str, float],
    extra: Mapping[str, object] | None = None,
) -> dict:
    """The post-epoch snapshot of everything the next v2 epoch reads.

    Published in every v2 reveal as `v2_state`, so a validator that lost its
    state, or has just joined, can rebuild it from the newest reveal, with the
    window replayed from the last CHAMPION_WINDOW_EPOCHS reveals (the rebuild
    from reveals); until it does, a validator whose window was lost fails
    closed (no first or take decisions). `extra` is where later v2 state (the copy rule's first-commit
    seniority, ownership and ghosts) is published: one block, one channel, so
    the rebuild reads one place. Keys are sorted for byte-identical reveals.
    """
    block = {
        "seat": seat_to_dict(seat),
        "owners": {h: owners[h] for h in sorted(owners)},
        "serving": {h: int(serving[h]) for h in sorted(serving)},
        "burn_in": {h: float(burn_in[h]) for h in sorted(burn_in)},
    }
    for key in sorted(extra or {}):
        if key in block:
            raise ValueError(f"v2_state key {key!r} is reserved")
        block[key] = extra[key]
    return block


def decision_for_reveal(decision: SeatDecision) -> dict:
    return {
        "event": decision.event,
        "champion": decision.champion,
        "bar": decision.bar,
        "seat": seat_to_dict(decision.state),
        "detail": [[h, why] for h, why in decision.detail],
    }


@dataclass(frozen=True)
class OddsInput:
    """What a protocol v3 epoch needs beyond a v2 one: the exploration answers
    of every verified proof, and what turns a head into odds on them."""

    nonce: bytes
    # Every verified proof's exploration results (explore_pool.ExploreAnswer).
    answers: list
    # {question_id: benchmark} for every explored question.
    benchmark_of: Mapping[str, str]
    # {uid: (W, b, model ids)} of each verified head, as loaded and bounded.
    heads: Mapping[int, tuple]
    # question_id -> its embedding row from the pinned embeddings file.
    hidden_of: object
    # Coldkeys the validator's exploration detectors flagged this epoch.
    flagged: frozenset[str] = frozenset()
    coldkey_cap: int = 36
    implausible_z: int = 4
    implausible_min_samples: int = 8
    implausible_min_trials: int = 50
    ref_prior: int = 20
    single_model_min_pairs: int = 300
    category_targets: Mapping[str, float] | None = None
    # {entry: inverse-probability weight} of the epoch's cost-aware exploration
    # (explore_weights.weights of epoch_plan.explore_shares); None weighs one.
    explore_weights: Mapping[str, int] | None = None
    # The protocol's menu. A head whose rows do not cover every entry makes no
    # prediction at all (every pair -1), so it is not eligible for the odds
    # seat (champion._no_full_menu). None checks only the pairs drawn.
    menu: tuple[str, ...] | None = None


def odds_b64(odds: np.ndarray) -> str:
    """A head's odds as published: little-endian int16, base64."""
    import base64

    return base64.b64encode(np.asarray(odds, dtype="<i2").tobytes()).decode("ascii")


def odds_from_b64(text: str, n: int) -> np.ndarray:
    """Inverse of `odds_b64`, validated: exactly `n` values in -1..ODDS_SCALE."""
    import base64

    from fugal_subnet.head_odds import ODDS_SCALE

    raw = base64.b64decode(text, validate=True)
    if len(raw) != 2 * n:
        raise ValueError(f"head odds hold {len(raw) // 2} values for {n} pairs")
    out = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    if n and (int(out.min()) < -1 or int(out.max()) > ODDS_SCALE):
        raise ValueError("head odds outside -1..ODDS_SCALE")
    return out


def odds_digest(pool_rows: list, odds_by_head: Mapping[str, str]) -> str:
    """One hash over an epoch's pooled pairs and every head's odds on them.

    It goes into `v2_state` (as `odds_digest`), which is the consensus view two
    validators are compared on, so validators that pooled different pairs or
    computed different odds differ visibly in the epoch it happened."""
    import hashlib

    canonical = json.dumps({"pool": pool_rows, "odds": {h: odds_by_head[h] for h in sorted(odds_by_head)}},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pool_and_odds(window: Window, oi: OddsInput, serving: Mapping[str, int],
                   answered: Mapping[str, HeadEpoch], bench_category):
    """(EpochPairs, {head: odds}, flagged coldkeys with reasons) for a v3 epoch."""
    from fugal_subnet import explore_pool, head_odds, odds_score

    cells = {(a.question_id, a.model): str(oi.benchmark_of.get(a.question_id, ""))
             for a in oi.answers}
    expected = odds_score.expected_odds(window.pairs, oi.ref_prior, cells,
                                        min_model_pairs=oi.implausible_min_trials)
    implausible = explore_pool.private_implausible(
        oi.answers, expected, z=oi.implausible_z, min_samples=oi.implausible_min_samples)
    pooled = explore_pool.pool_pairs(
        oi.answers, nonce=oi.nonce, coldkey_cap=oi.coldkey_cap,
        flagged=set(oi.flagged) | set(implausible), weights=oi.explore_weights)
    benches = tuple(str(oi.benchmark_of.get(p.question_id, "")) for p in pooled)
    pairs = EpochPairs(
        question_ids=tuple(p.question_id for p in pooled),
        models=tuple(p.model for p in pooled),
        benchmarks=benches,
        categories=tuple(category_of({"benchmark": b}, bench_category) for b in benches),
        y=np.array([p.y for p in pooled], dtype=np.int32),
        prompt_tokens=np.array([p.prompt_tokens for p in pooled], dtype=np.int32),
        completion_tokens=np.array([p.completion_tokens for p in pooled], dtype=np.int32),
        weight=(np.array([p.weight for p in pooled], dtype=np.int64)
                if oi.explore_weights is not None else None),
    )
    hidden = [oi.hidden_of(q) for q in pairs.question_ids]
    odds: dict[str, np.ndarray] = {}
    for h, uid in sorted(serving.items()):
        if h not in answered or uid not in oi.heads:
            continue
        W, b, models = oi.heads[uid]
        if oi.menu is not None and not set(oi.menu) <= set(models):
            # From v3 a head gives odds for every menu entry. One that leaves
            # an entry out is published with no odds anywhere (-1), which the
            # reveal shows and a rebuild reads the same way.
            odds[h] = np.full(len(pairs.models), -1, dtype=np.int16)
            continue
        odds[h] = head_odds.head_odds(W, b, list(models), hidden, list(pairs.models))
    return pairs, odds, implausible, pooled


def _odds_shadow(window: Window, ref_prior: int, serving: Mapping[str, int],
                 champion: str | None) -> dict:
    """Where every scored head's odds stand, published each v3 epoch whether or
    not odds decide the seat yet: its mean skill over the reference on its
    current run, and its mean Brier gain over the champion."""
    from fractions import Fraction

    from fugal_subnet import odds_score

    ref = odds_score.reference(window.pairs, ref_prior)
    out = {}
    for h in sorted(serving):
        if h not in window.odds:
            continue
        skill, n = odds_score.solo_skill(window, ref, h)
        row = {"uid": int(serving[h]), "pairs": n, "skill_points": odds_score.points(skill)}
        if champion is not None and champion != h and champion in window.odds:
            bu = odds_score.brier_units(window, ref, h, champion)
            if bu.n_pairs:
                row["vs_champion"] = {
                    "pairs": bu.n_pairs,
                    "brier_gain_points": odds_score.points(Fraction(bu.total, bu.n_pairs)),
                }
        out[h] = row
    return out


@dataclass(frozen=True)
class V2EpochResult:
    window: Window
    decision: SeatDecision
    copy_state: object          # copy_rule.CopyState after this epoch
    uids: list[int]
    weights: list[float]
    reveal: dict

    @property
    def owners(self) -> dict[str, str]:
        """{head hash: the hotkey that first committed it}, for logs and the reveal."""
        return {r.head_hash: r.first_hotkey for r in self.copy_state.heads}


def run_v2_epoch(
    *,
    epoch_index: int,
    epoch_id: str,
    boundary_block: int,
    blocks_per_epoch: int,
    questions: list[Mapping],
    pool_models: list[str],
    prices: Mapping[str, tuple[float, float]],
    verified_proofs: Mapping[int, object],
    head_hashes: Mapping[int, str],
    coldkey_of: Mapping[int, str],
    hotkey_of: Mapping[int, str],
    records: Mapping[int, object],
    commitments: list,
    window: Window,
    seat: SeatState,
    copy_state,
    burn_uid: int,
    params: SeatParams,
    shares: tuple[float, ...],
    burn_in_questions: int,
    bench_category: Mapping[str, str] | None = None,
    allow_take: bool = True,
    seniority_floor: int | None = None,
    seniority_salt: str = "",
    odds_input: OddsInput | None = None,
    rules: Mapping | None = None,
    state_rules: Mapping | None = None,
) -> V2EpochResult:
    """One v2 epoch, from verified proofs to the weight vector. Pure.

    Ownership, seniority, retirement and the senior bars are the copy rule's
    (fugal_subnet/copy_rule.py). `commitments` is the boundary snapshot,
    one `copy_rule.Commitment` per registered hotkey (with its owning coldkey),
    read pinned at `boundary_block`; `copy_state` is the copy rule's state
    after the previous epoch. A head hash belongs to the coldkey that first
    committed it, only its canonical owner UID serves it (`classify`), its
    seniority is its first commit, a champion is retired when its owner stops
    committing it, and a head takes a seat only by also beating every senior
    seat holder by the margin (`senior_bars`). Copies, duplicates and unknown
    hashes are not candidates at all: they are neither paid, nor measured,
    nor able to retire anything. Nothing here reads the frame, and
    `paid_fraction` is 1.

    `odds_input` (protocol v3) adds the exploration pool and every scored
    head's odds on it to the window and the reveal. Whether they decide the
    seat is `params.odds`; they are computed and published either way.

    `rules` names what this epoch is scored under (protocol, and whether the
    seat is on odds); `state_rules` what the incoming window and seat were
    built under. Where they differ the state is not carried over: a new
    protocol starts from an empty window and an empty seat (its questions and
    its model list are different ones), and a new seat objective starts from
    an empty seat on the window it has. `rules` is published in `v2_state`, so
    a rebuilt validator knows what the state it loaded was built under.
    """
    from fugal_subnet import copy_rule
    from fugal_subnet.champion import window_accuracy
    from fugal_subnet.evidence import burn_in_complete
    from fugal_subnet.payout import seat_weights
    from fugal_subnet.shared_grades import (
        ScoredAnswer,
        cells_for_reveal,
        shared_cell_grades,
        shared_cell_tokens,
    )

    if rules is not None and dict(rules) != dict(state_rules or {}):
        if (state_rules or {}).get("protocol") != rules.get("protocol"):
            window = Window.empty(window.size)
        seat = SeatState()

    answers = [
        ScoredAnswer(
            question_id=r.question_id, model=r.routed_model, coldkey=coldkey_of[uid],
            correct=bool(r.correct), failed=bool(r.failed),
            completion_tokens=int(r.completion_tokens),
        )
        for uid in sorted(verified_proofs)
        for r in verified_proofs[uid].scored_results
    ]
    cells = shared_cell_grades(answers)
    tokens = shared_cell_tokens(answers)
    sl = build_epoch_slice(epoch_index, questions, pool_models, cells, tokens, prices,
                           bench_category)
    model_index = {m: i for i, m in enumerate(pool_models)}

    # The copy rule's fold of this boundary's commitments. Idempotent across a
    # retry of the same epoch: a state already folded through this epoch (a
    # --once re-run after the state was written) is used as it is.
    if copy_state.epoch is not None and int(epoch_index) <= copy_state.epoch:
        new_copy = copy_state
    else:
        new_copy = copy_rule.observe(copy_state, epoch_index, boundary_block, commitments,
                                     ghost_retire=params.ghost_retire)

    # Who serves each head: only its canonical owner UID.
    committed_by = {c.hotkey: c for c in copy_rule.valid_commitments(commitments, boundary_block)}
    served = {}
    for uid in sorted(verified_proofs):
        c = committed_by.get(hotkey_of[uid])
        if c is not None and c.head_hash == head_hashes[uid]:
            served[uid] = copy_rule.Commitment(hotkey_of[uid], coldkey_of[uid],
                                               head_hashes[uid], c.block)
    verdicts = copy_rule.classify(new_copy, served)
    serving: dict[str, int] = {
        head_hashes[uid]: uid for uid, v in sorted(verdicts.items()) if v == "owner"
    }

    answers_by_head: dict[str, HeadEpoch] = {}
    for h, uid in sorted(serving.items()):
        he = head_epoch(verified_proofs[uid], sl.question_ids, model_index)
        if he is not None:
            answers_by_head[h] = he
    pairs = head_odds_now = pooled = None
    implausible: dict[str, str] = {}
    if odds_input is not None:
        pairs, head_odds_now, implausible, pooled = _pool_and_odds(
            window, odds_input, serving, answers_by_head, bench_category)
    new_window = window.add_epoch(sl, answers_by_head, pairs, head_odds_now)

    candidates: dict[str, Candidate] = {}
    for h, uid in sorted(serving.items()):
        if h not in answers_by_head:
            continue
        rec = records.get(uid)
        since = copy_rule.first_epoch(new_copy, h, blocks_per_epoch)
        candidates[h] = Candidate(
            head_hash=h,
            seniority=copy_rule.v2_seniority(new_copy, h, floor_block=seniority_floor,
                                             salt=seniority_salt),
            eligible=True,
            since_epoch=int(since if since is not None else epoch_index),
            burned_in=burn_in_complete(getattr(rec, "evidence", None), burn_in_questions),
            first_block=(new_copy.get(h).first_block if new_copy.get(h) is not None else None),
        )

    # Retired: the champion or a standing ghost whose OWNER no longer commits
    # it. A missed ghost that becomes retired is upgraded by seat_step.
    watched = {g.head_hash for g in seat.ghosts} | ({seat.champion} if seat.champion else set())
    # A watched head with no copy record at all has been retired too: the copy
    # rule drops a record once its FIRST retirement is older than ghost_retire,
    # so a champion whose head was retired once, re-committed and then dropped
    # again has no record the epoch its owner moves on, and was read as merely
    # missed (a 3-epoch ghost instead of the 24-epoch retired one; theme-2
    # review, 2026-11-08).
    retired = (set(copy_rule.retired_hashes(new_copy))
               | {h for h in watched if h not in new_copy}) & watched

    bars = copy_rule.senior_bars(
        new_copy, epoch_index, new_window, candidates,
        holders=copy_rule.seat_holders(seat, len(shares)),
        ghost_off=params.ghost_off, ghost_retire=params.ghost_retire,
        floor_block=seniority_floor, salt=seniority_salt,
    )
    decision = seat_step(seat, epoch_index, epoch_id, new_window, candidates,
                         retired=retired, params=params, allow_take=allow_take,
                         senior_bars=bars)

    # Payout eligibility is the HOLD rule: live, canonical owner, window
    # accuracy at or above the floor. Burn-in is not required, so a champion
    # that resumes after a miss (its E'' counter restarted) is paid at once.
    payable = set()
    for h in candidates:
        acc, n = window_accuracy(new_window, h)
        if n > 0 and acc >= params.min_accuracy:
            payable.add(h)
    # Seat 1 is the champion's, and a champion that is gone but still barring
    # (a ghost) keeps it: its share burns, and seats 2 and 3 keep paying the
    # two champions before it.
    ghost_hashes = {g.head_hash for g in decision.state.ghosts}
    seat_one = decision.champion
    history = tuple(decision.state.history)
    if seat_one is None and history and history[0] in ghost_hashes:
        seat_one, history = history[0], history[1:]
        # The ghost holds the seat; it is not paid for it, even when its head
        # is live again (a retired champion that recommitted its old hash
        # serves as its own ghost and was paid seat 1 every epoch it served;
        # v2 attack review, 2026-11-09).
        payable = payable - {seat_one}
    uids, weights = seat_weights(
        seat_one, history, serving, payable,
        shares=shares, paid_fraction=1.0, burn_uid=burn_uid,
    )
    owners = {r.head_hash: r.first_hotkey for r in new_copy.heads}
    extra = {copy_rule.REVEAL_KEY: new_copy.to_reveal()}
    if rules is not None:
        extra["rules"] = dict(rules)
    v3_fields: dict = {}
    if odds_input is not None:
        from fugal_subnet import explore_pool
        from fugal_subnet.single_model import single_model_table

        pool_rows = explore_pool.pairs_for_reveal(pooled, odds_input.benchmark_of)
        odds_rows = {h: odds_b64(o) for h, o in sorted(head_odds_now.items())}
        extra["odds_digest"] = odds_digest(pool_rows, odds_rows)
        v3_fields = {
            "exploration_pool": pool_rows,
            "head_odds": odds_rows,
            "exploration_excluded_coldkeys": {
                ck: why for ck, why in sorted(implausible.items())},
            "odds_shadow": _odds_shadow(new_window, odds_input.ref_prior, serving,
                                        decision.champion),
            "single_model_table": single_model_table(
                new_window.pairs, prices, odds_input.category_targets or {},
                min_pairs=odds_input.single_model_min_pairs),
        }
    reveal = {
        "shared_grades": cells_for_reveal(cells, tokens),
        "failed": {
            str(uid): sorted(r.question_id for r in verified_proofs[uid].scored_results if r.failed)
            for uid in sorted(verified_proofs)
        },
        "copy_verdicts": {str(uid): v for uid, v in sorted(verdicts.items())},
        # What ordered pre-flip commits: the activation block they tie at and
        # the boundary hash that breaks the tie (anyone can check it on chain).
        "v2_seniority": {"floor_block": seniority_floor, "salt": seniority_salt},
        "champion": decision_for_reveal(decision),
        "v2_state": v2_state_for_reveal(
            seat=decision.state, owners=owners, serving=serving,
            burn_in={
                h: float(getattr(getattr(records.get(uid), "evidence", None),
                                 "burn_in_progress", 0.0) or 0.0)
                for h, uid in serving.items()
            },
            extra=extra,
        ),
        **v3_fields,
    }
    return V2EpochResult(new_window, decision, new_copy, uids, weights, reveal)


__all__ = [
    "Candidate", "OddsInput", "V2EpochResult", "build_epoch_slice", "odds_b64", "odds_digest",
    "odds_from_b64", "run_v2_epoch", "category_of", "decision_for_reveal", "head_epoch",
    "load_window", "save_window", "seat_from_dict", "seat_params_from_config", "seat_to_dict",
    "v2_state_for_reveal",
]
