#!/usr/bin/env python3
"""Experiment: what fresh-data confirmation of a seat take buys and costs.

Simulated with the SHIPPED champion.seat_step and paired_test (no API spend,
no chain). One settled champion and k challengers answer the same
300-question slices (the v2 category mix, hard weight 2, margin 0.2 at 99%,
window 48, 1,500 paired and 1,500 fresh questions). A challenger's outcome
differs from the champion's on a share of questions set so that the
per-question spread of net value matches pairs measured on mainnet reveals
(about 14 points for near-copies, 27 for independently trained heads), and it
wins a differing question with a probability set so that its true edge is
`delta` points.

For the rule off and on it reports the share of runs in which a challenger
took the seat, and the epoch it did: for challengers with NO edge that is a
false take; for challengers with a real edge it is the time to the seat.

    uv run python scripts/experiments/experiment_seat_confirmation.py 120 400 out.json

Arguments: runs per row, bootstrap resamples (the shipped value is 2,000; 400
keeps a full table to about half an hour on six cores), output file. The
figures in docs/INVARIANTS.md ("a take is confirmed on fresh questions") are
this script at 120 runs and 400 resamples.
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from fugal_subnet.champion import (
    Candidate,
    EpochSlice,
    HeadEpoch,
    SeatParams,
    SeatState,
    Window,
    seat_step,
)

CATS = ["knowledge"] * 120 + ["math"] * 75 + ["hard"] * 90 + ["instruction"] * 15
HARD = tuple(c == "hard" for c in CATS)
Q = len(CATS)
# Share of questions that differ, set so the per-question sd of net value matches
# mainnet pairs: near-copies about 14 points, independently trained heads about 27.
NEAR, INDEP = 0.0105, 0.038

def params(confirm, resamples):
    return SeatParams(margin_points=0.2, confidence=0.99, resamples=resamples, accuracy_value=30,
                      hard_weight=2, hard_gate_min_questions=100, ghost_off=3, ghost_retire=8,
                      wilson_confidence=0.95, min_accuracy=0.30, min_paired_questions=1500,
                      anchor_spend=0.0163, confirm_questions=1500 if confirm else 0,
                      confirm_open_epochs=5 if confirm else 0)

def run(args):
    seed, k, d, delta, epochs, confirm, resamples = args
    rng = np.random.default_rng(seed)
    eps = delta / (260.0 * d)
    p = params(confirm, resamples)
    names = ["c" * 64] + [f"{i:064x}" for i in range(1, k + 1)]
    window = Window.empty(48)
    state = SeatState(champion=names[0], champion_since=0)
    routed = {h: np.full(Q, i, dtype=np.int64) for i, h in enumerate(names)}
    nofail = np.zeros(Q, dtype=bool)
    for e in range(1, epochs + 1):
        g = np.empty((Q, k + 1))
        g[:, 0] = rng.random(Q) < 0.92
        for j in range(1, k + 1):
            g0 = g[:, 0]
            # P(pick a champion-wrong question to differ on) vs champion-right:
            # weights so that P(challenger wins | differs) = 0.5 + eps
            w_wrong = (0.5 + eps) / max(1e-9, 1 - 0.92)
            w_right = (0.5 - eps) / 0.92
            pr = np.where(g0 == 0, w_wrong, w_right) * d
            differ = rng.random(Q) < np.clip(pr, 0, 1)
            g[:, j] = np.where(differ, 1.0 - g0, g0)
        sl = EpochSlice(epoch=e, question_ids=tuple(f"q{e}_{i}" for i in range(Q)),
                        categories=tuple(CATS), hard=HARD, grades=g,
                        spends=np.full((Q, k + 1), 0.004))
        window = window.add_epoch(sl, {h: HeadEpoch(routed[h], nofail) for h in names})
        cands = {h: Candidate(h, (i,), True, 1, burned_in=e >= 5) for i, h in enumerate(names)}
        dec = seat_step(state, e, f"e{e:08d}", window, cands, params=p)
        state = dec.state
        if dec.event == "take":
            return e
    return None

def main():
    trials = int(sys.argv[1]); resamples = int(sys.argv[2])
    out = {}
    with Pool(6) as pool:
        for label, k, d, delta, epochs in [
            ("edge +0.5 near-copy", 1, NEAR, 0.5, 150), ("edge +0.85 near-copy", 1, NEAR, 0.85, 150),
            ("edge +1.5 near-copy", 1, NEAR, 1.5, 150), ("edge +2 independent", 1, INDEP, 2.0, 150),
            ("edge +4 independent", 1, INDEP, 4.0, 150),
            ("null k=10 near-copy", 10, NEAR, 0.0, 60), ("null k=10 independent", 10, INDEP, 0.0, 60),
            ("null k=30 near-copy", 30, NEAR, 0.0, 60),
        ]:
            for confirm in (False, True):
                t = time.time()
                n = trials if k <= 10 else max(20, trials // 3)
                res = pool.map(run, [(1000 + s, k, d, delta, epochs, confirm, resamples) for s in range(n)])
                took = [r for r in res if r is not None]
                row = {"runs": n, "took": len(took), "share": round(len(took) / n, 3),
                       "median_epoch": (int(np.median(took)) if took else None),
                       "p90_epoch": (int(np.percentile(took, 90)) if took else None)}
                out[f"{label} | confirm={'on' if confirm else 'off'}"] = row
                print(f"{label:28s} confirm={'on ' if confirm else 'off'} {row}  ({time.time()-t:.0f}s)", flush=True)
    json.dump(out, open(sys.argv[3], "w"), indent=1)

if __name__ == "__main__":
    main()
