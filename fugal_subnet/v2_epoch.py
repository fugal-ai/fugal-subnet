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
    EpochSlice,
    Ghost,
    HeadEpoch,
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
    w = Window(size=int(meta["size"]), slices=slices, heads=heads)
    if w.size != int(size):
        # A changed window size is a changed rule; shrink (or keep) to the
        # configured size from the newest epoch, never silently extend.
        w = Window(size=int(size), slices=w.slices, heads=w.heads)
    return w


def seat_to_dict(state: SeatState) -> dict:
    return {
        "champion": state.champion,
        "champion_since": state.champion_since,
        "history": list(state.history),
        "ghosts": [{"head_hash": g.head_hash, "last_epoch": g.last_epoch, "reason": g.reason}
                   for g in state.ghosts],
        "high_water": state.high_water,
        "high_water_hard": state.high_water_hard,
    }


def seat_from_dict(raw: Mapping | None) -> SeatState:
    if not raw:
        return SeatState()
    return SeatState(
        champion=raw.get("champion"),
        champion_since=raw.get("champion_since"),
        history=tuple(raw.get("history", ())),
        ghosts=tuple(Ghost(str(g["head_hash"]), int(g["last_epoch"]), str(g["reason"]))
                     for g in raw.get("ghosts", ())),
        high_water=float(raw.get("high_water", 0.0)),
        high_water_hard=float(raw.get("high_water_hard", 0.0)),
    )


def seat_params_from_config() -> SeatParams:
    from fugal_subnet import config

    return SeatParams(
        margin_points=config.CHAMPION_MARGIN,
        confidence=config.CHAMPION_CONFIDENCE,
        resamples=config.CHAMPION_BOOTSTRAP_RESAMPLES,
        accuracy_value=config.CHAMPION_ACCURACY_VALUE,
        hard_weight=config.CHAMPION_HARD_WEIGHT,
        hard_gate_min_questions=config.CHAMPION_HARD_GATE_MIN_QUESTIONS,
        ghost_off=config.CHAMPION_GHOST_OFF,
        ghost_retire=config.CHAMPION_GHOST_RETIRE,
        wilson_confidence=config.WILSON_CONFIDENCE,
        min_accuracy=config.CHAMPION_MIN_ACCURACY,
        min_paired_questions=config.CHAMPION_MIN_PAIRED_QUESTIONS,
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
    new_window = window.add_epoch(sl, answers_by_head)

    candidates: dict[str, Candidate] = {}
    for h, uid in sorted(serving.items()):
        if h not in answers_by_head:
            continue
        rec = records.get(uid)
        since = copy_rule.first_epoch(new_copy, h, blocks_per_epoch)
        candidates[h] = Candidate(
            head_hash=h,
            seniority=copy_rule.v2_seniority(new_copy, h),
            eligible=True,
            since_epoch=int(since if since is not None else epoch_index),
            burned_in=burn_in_complete(getattr(rec, "evidence", None), burn_in_questions),
        )

    # Retired: the champion or a standing ghost whose OWNER no longer commits
    # it. A missed ghost that becomes retired is upgraded by seat_step.
    watched = {g.head_hash for g in seat.ghosts} | ({seat.champion} if seat.champion else set())
    retired = set(copy_rule.retired_hashes(new_copy)) & watched

    bars = copy_rule.senior_bars(
        new_copy, epoch_index, new_window, candidates,
        holders=copy_rule.seat_holders(seat, len(shares)),
        ghost_off=params.ghost_off, ghost_retire=params.ghost_retire,
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
    uids, weights = seat_weights(
        seat_one, history, serving, payable,
        shares=shares, paid_fraction=1.0, burn_uid=burn_uid,
    )
    owners = {r.head_hash: r.first_hotkey for r in new_copy.heads}
    reveal = {
        "shared_grades": cells_for_reveal(cells, tokens),
        "failed": {
            str(uid): sorted(r.question_id for r in verified_proofs[uid].scored_results if r.failed)
            for uid in sorted(verified_proofs)
        },
        "copy_verdicts": {str(uid): v for uid, v in sorted(verdicts.items())},
        "champion": decision_for_reveal(decision),
        "v2_state": v2_state_for_reveal(
            seat=decision.state, owners=owners, serving=serving,
            burn_in={
                h: float(getattr(getattr(records.get(uid), "evidence", None),
                                 "burn_in_progress", 0.0) or 0.0)
                for h, uid in serving.items()
            },
            extra={copy_rule.REVEAL_KEY: new_copy.to_reveal()},
        ),
    }
    return V2EpochResult(new_window, decision, new_copy, uids, weights, reveal)


__all__ = [
    "Candidate", "V2EpochResult", "build_epoch_slice", "run_v2_epoch", "category_of", "decision_for_reveal", "head_epoch",
    "load_window", "save_window", "seat_from_dict", "seat_params_from_config", "seat_to_dict",
    "v2_state_for_reveal",
]
