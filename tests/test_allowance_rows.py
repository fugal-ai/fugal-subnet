"""Protocol v4: does deleting a row ever beat keeping it, on the published field?
(docs/decisions/SEAT_MINIMAL_V4.md 4.2; data/observations/allowance_rows_e00026475-e00026509.json)

A head without a row for a new entry is scored, during the entry's
allowance, as if its row said one number for every question: the entry's
measured rate q_e. Deleting a row therefore swaps the row's odds for a
constant. The rule itself is exact (tests/test_v4_labels.py: an omitted row
and the one-number row at q_e hold the same odds on every label), so what
deletion can gain is only what that constant would have earned over the row.

This file measures that on what heads really predicted for the last entries
that joined a menu: v3 menu B's seven entries, from their join (e00026475)
through e00026509, 549 pooled labels, 28 heads with odds on them, 25 of them
on at least 40 (the design's set). The constant is the strongest one a
measured rate can be: each entry's own rate over all its labels, in
hindsight. Losses are the score's: (p - y)^2 in odds units, summed exactly;
"weighted" multiplies each by the pair's published exploration weight, as
labels_v4.gain_items does.

What holds, and what does not (each asserted below):

- Per head, all its rows of the seven entries together, per label
  unweighted: every one of the 25 heads' rows beats the hindsight constant
  (by 2.0 Brier points a label on average, 0.16 at the least). This is the
  design's figure, reproduced.
- In the score's own units, with the pair weights: 21 of the 25 do; four
  would have scored better with their rows deleted, by 0.42 to 0.86 points
  a label of those entries, had q_e been the hindsight constant.
- Per row (one head, one entry): 56 of the 175 rows score worse than the
  hindsight constant on that head's labels of the entry, weighted or not.

So "deleting a row is never better than keeping it" is what the published
field shows per head and per label; it is not true of every row, nor of
every head in the score's weighted units. What is true of every row is the
identity: deleting a row is scored exactly as committing the one-number row
at q_e, which any head may write.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from fugal_subnet.explore_weights import WEIGHT_SCALE, apply

PATH = (Path(__file__).resolve().parents[1] / "data" / "observations"
        / "allowance_rows_e00026475-e00026509.json")
MIN_LABELS = 40


def _load():
    doc = json.loads(PATH.read_text(encoding="ascii"))
    assert doc["format"] == "allowance-rows-v1"
    return doc


def _loss(p: int, y: int, w: int, weighted: bool) -> int:
    return int(apply(np.array([(p - y) ** 2]), np.array([w if weighted else WEIGHT_SCALE]))[0])


def _hindsight(doc, weighted: bool) -> dict[int, int]:
    """Each entry's rate over all its labels, rounded half up: the best
    constant there could be for the entry, as the design reads it."""
    out = {}
    for e in range(len(doc["entries"])):
        ls = [lab for lab in doc["labels"] if lab[1] == e]
        sw = sum(lab[3] if weighted else 1 for lab in ls)
        m = Fraction(sum((lab[3] if weighted else 1) * lab[2] for lab in ls), sw)
        out[e] = int((2 * m + 1) // 2)
    return out


def _per_head(doc, weighted: bool):
    """[(labels, kept loss, deleted loss)] per head with MIN_LABELS labels."""
    const = _hindsight(doc, weighted)
    out = []
    for rows in doc["heads"]:
        if len(rows) < MIN_LABELS:
            continue
        keep = delete = 0
        for i, p in rows:
            _, e, y, w = doc["labels"][i]
            keep += _loss(p, y, w, weighted)
            delete += _loss(const[e], y, w, weighted)
        out.append((len(rows), keep, delete))
    return out


def test_the_observations_are_the_named_reveals():
    doc = _load()
    picks = json.loads((PATH.parent / "menu_v4_seed_picks_e00026466-e00026509.json").read_text())
    named = {r["epoch_id"]: r["sha256"] for r in picks["reveals"]}
    assert [r["epoch_id"] for r in doc["reveals"]] == [f"e{i:08d}" for i in range(26475, 26510)]
    assert all(named[r["epoch_id"]] == r["sha256"] for r in doc["reveals"])
    assert len(doc["entries"]) == 7 and len(doc["labels"]) == 549 and len(doc["heads"]) == 28
    assert sum(len(rows) >= MIN_LABELS for rows in doc["heads"]) == 25


def test_per_head_and_per_label_no_head_would_have_gained_by_deleting_its_rows():
    heads = _per_head(_load(), weighted=False)
    assert len(heads) == 25
    gains = [Fraction(delete - keep, n * 10**6) for n, keep, delete in heads]
    assert all(g > 0 for g in gains)
    assert min(gains) > Fraction(15, 100)                       # 0.16 points a label at the least
    assert Fraction(19, 10) < sum(gains) / len(gains) < Fraction(21, 10)   # 2.0 on average


def test_in_the_scores_weighted_units_four_heads_would_have_gained():
    heads = _per_head(_load(), weighted=True)
    losing = [Fraction(keep - delete, n * 10**6) for n, keep, delete in heads if keep >= delete]
    assert len(heads) - len(losing) == 21
    assert len(losing) == 4 and max(losing) < Fraction(86, 100) and min(losing) > Fraction(41, 100)


def test_per_row_a_third_of_the_rows_are_worse_than_the_hindsight_constant():
    doc = _load()
    for weighted in (False, True):
        const = _hindsight(doc, weighted)
        rows = worse = 0
        for head in doc["heads"]:
            if len(head) < MIN_LABELS:
                continue
            for e in range(len(doc["entries"])):
                mine = [(doc["labels"][i], p) for i, p in head if doc["labels"][i][1] == e]
                if not mine:
                    continue
                rows += 1
                keep = sum(_loss(p, lab[2], lab[3], weighted) for lab, p in mine)
                delete = sum(_loss(const[e], lab[2], lab[3], weighted) for lab, p in mine)
                worse += keep >= delete
        assert (rows, worse) == (175, 56)
