#!/usr/bin/env python3
"""The v4 seat's headline figures, simulated with the real seat step
(seat_v4.seat_step) at modest scale and a fixed seed
(docs/decisions/SEAT_MINIMAL_V4.md 3.9, 5.2, 10).

The design's model: each epoch every comparison gets n labels, each head's
Brier gain over the bar on a label drawn with mean d and SD about 8 points.
Here the labels are real `labels_v4.LabelPairs` with outcomes at 50%, the
reference R is fitted by `labels_v4.reference` as in the live epoch, and each
head's odds on a label lean toward the outcome with probability f and away
from it otherwise, by s (`odds = 0.5 +- 0.4 s`): its gain over R on the label
is +-40 s - 16 s^2 points, mean (2f - 1) 40 s - 16 s^2, SD about 40 s. The
heads answer the scored questions with net value 0, so rule B passes once a
head's run holds 1,000 questions and only rule A and rule C decide.

Two figures:

1. **No zero-edge head is seated.** 30 heads whose gain over R has mean 0
   and SD 8 answer every epoch for 300 epochs on an empty seat
   (design: "30 heads with no edge over 300 simulated epochs: none seated").
2. **Epochs to seat a head 0.5 points better** than the holder, at the
   recommended label budget: 815 labels a comparison an epoch (ten hotkeys at
   112 calls, 5.2), per-label SD of the paired gain about 8 (design: about
   8 epochs; rule B's 1,000 questions alone take five).

Run: python scripts/experiments/experiment_seat_v4.py [--zero-heads 30
--zero-epochs 300 --runs 50]. No network, no paid call.
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fugal_subnet import labels_v4  # noqa: E402
from fugal_subnet.attacks.seat_v4_world import SCALE, World  # noqa: E402

LABELS_AT_RECOMMENDED = 815          # 5.2: ten hotkeys, $20 a day, labels in one test


def mix_for(mean_points: float, s: float) -> float:
    """f such that a head leaning by s, toward the outcome with probability
    f, has mean gain `mean_points` over a 50% predictor."""
    return 0.5 + (mean_points + 16 * s * s) / (80 * s)


class MixWorld(World):
    """The seat's synthetic world with heads drawn from (s, f) mixtures."""

    def __init__(self, *, seed: int = 0, **kw):
        super().__init__(seed=seed, **kw)
        self.seed = seed
        self.mix: dict[str, tuple[float, float]] = {}

    def add_mix(self, name: str, mean_points: float, s: float, **kw):
        self.mix[name] = (s, mix_for(mean_points, s))
        return self.add(name, **kw)

    def odds(self, h, pairs: labels_v4.LabelPairs, epoch: int) -> np.ndarray:
        s, f = self.mix[h.name]
        y = pairs.y.astype(np.int64)
        # Which labels a head leans the right way on: its own stream per
        # run, head and epoch (a head's gain sign does not depend on y).
        key = f"{self.seed}|{h.name}|{epoch}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        toward = np.random.default_rng(seed).random(len(y)) < f
        right = np.where(y >= SCALE // 2, SCALE, 0)
        target = np.where(toward, right, SCALE - right)
        o = np.rint(SCALE // 2 + s * 0.8 * (target - SCALE // 2)).astype(np.int64)
        return o.astype(np.int16)


def zero_edge(heads: int = 30, epochs: int = 300, labels: int = 500, seed: int = 1) -> dict:
    """`heads` heads with no edge over R answer every epoch on an empty
    seat; how many epochs any of them holds the seat."""
    w = MixWorld(labels=labels, seed=seed)
    names = [f"z{i:02d}" for i in range(heads)]
    for i, n in enumerate(names):
        w.add_mix(n, 0.0, 0.2, block=3 + i)
    seated, best = [], None
    for e in range(1, epochs + 1):
        d = w.epoch(e, served=names, answered=names)
        if d.champion is not None:
            seated.append((e, w.name(d.champion)))
        for h, doc in d.evidence["heads"].items():
            vb = doc.get("vs_bar")
            if vb and vb["lower"] is not None and (best is None or vb["lower"] > best):
                best = vb["lower"]
    return {"heads": heads, "epochs": epochs, "labels": labels, "seated": seated,
            "best_lower_points": None if best is None else best / 1e6}


def epochs_to_seat(edge: float = 0.5, labels: int = LABELS_AT_RECOMMENDED, seed: int = 0,
                   limit: int = 60) -> int | None:
    """A holder 2 points over R; a challenger `edge` points better than it,
    committed and answering from the epoch after the holder is seated. The
    epochs from the challenger's first label to its take (None: not in
    `limit` epochs)."""
    w = MixWorld(labels=labels, seed=seed)
    w.add_mix("h", 2.0, 0.15, block=3)
    e = 0
    while w.holder() != "h":
        e += 1
        w.epoch(e, served=["h"], answered=["h"])
        if e > 30:
            raise RuntimeError("the holder was never seated")
    w.add_mix("c", 2.0 + edge, 0.15, block=(e + 1) * 10 - 5)
    first = e + 1
    for k in range(first, first + limit):
        w.epoch(k, served=["h", "c"], answered=["h", "c"])
        if w.holder() == "c":
            return k - first + 1
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--zero-heads", type=int, default=30)
    p.add_argument("--zero-epochs", type=int, default=300)
    p.add_argument("--zero-labels", type=int, default=500)
    p.add_argument("--runs", type=int, default=50)
    a = p.parse_args()
    t = time.time()
    z = zero_edge(a.zero_heads, a.zero_epochs, a.zero_labels)
    print(f"zero edge: {z['heads']} heads x {z['epochs']} epochs at {z['labels']} labels: "
          f"seated in {len(z['seated'])} epochs; highest bound over R "
          f"{z['best_lower_points']:+.3f} points (margin 0.1)  [{time.time() - t:.0f}s]")
    t = time.time()
    got = [epochs_to_seat(seed=s) for s in range(a.runs)]
    done = sorted(x for x in got if x is not None)
    print(f"0.5 points better at {LABELS_AT_RECOMMENDED} labels an epoch: {len(done)} of "
          f"{a.runs} seated; median {statistics.median(done) if done else '-'} epochs, "
          f"range {done[0] if done else '-'}-{done[-1] if done else '-'}  [{time.time() - t:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
