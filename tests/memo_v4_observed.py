"""The published v3 epochs as a v4 window, for the memorisation check's
reproduction (tests/test_memo_v4.py; scripts/build_published_v4.py labels).

data/observations/memo_v4_labels_e00026466-e00026509.json holds, from the 44
v3 reveals e00026466..e00026509, each epoch's pooled labels with the
coldkeys whose answers they pool, every head's published odds on them, each
head's owner and first commit block, and for each labelled question its set
and the first named reveal (from e00026189) that published it. This module
reads it into what memo_v4 reads: a window of labels_v4.LabelPairs with the
heads' odds, the published set, and each head's side."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OBSERVATIONS = (Path(__file__).resolve().parent.parent / "data" / "observations"
                / "memo_v4_labels_e00026466-e00026509.json")
NOW = 26509
BLOCKS_PER_EPOCH = 360


def load(path: Path = OBSERVATIONS):
    """(window, published, sides): the window of the 44 epochs as LabelPairs
    (no allowance: every row the heads lacked was certainly wrong in v3),
    the published set folded through NOW, and {head: labels_v4.Side} with
    the head's owner and its first epoch (the first boundary at or after
    its first commit)."""
    from fugal_subnet import epoch_plan, labels_v4, memo_v4
    from fugal_subnet.champion import EpochPairs, Window
    from fugal_subnet.v2_epoch import odds_from_b64

    doc = json.loads(path.read_bytes())
    sets, models, coldkeys = doc["sets"], doc["models"], doc["coldkeys"]
    category = epoch_plan.mix_v4().bench_category
    qids = [f"q{i:05d}" for i in range(len(doc["questions"]))]
    pairs, odds = {}, {}
    for e, rows, head_odds in doc["epochs"]:
        n = len(rows)
        base = EpochPairs(
            question_ids=tuple(qids[r[0]] for r in rows),
            models=tuple(models[r[1]] for r in rows),
            benchmarks=tuple(sets[doc["questions"][r[0]][0]] for r in rows),
            categories=tuple(category[sets[doc["questions"][r[0]][0]]] for r in rows),
            y=np.array([r[2] for r in rows], dtype=np.int32),
            prompt_tokens=np.zeros(n, dtype=np.int32),
            completion_tokens=np.zeros(n, dtype=np.int32),
            weight=np.array([r[3] for r in rows], dtype=np.int64))
        pairs[e] = labels_v4.label_pairs(base, [tuple(coldkeys[c] for c in r[4]) for r in rows],
                                         np.full(n, -1, dtype=np.int16))
        for h, text in head_odds.items():
            odds.setdefault(h, {})[e] = odds_from_b64(text, n)
    window = Window(size=48, slices={}, heads={}, pairs=pairs, odds=odds)
    published = memo_v4.Published({qids[i]: int(f) for i, (_, f) in enumerate(doc["questions"])},
                                  NOW)
    sides = {h: labels_v4.Side(h, coldkeys[v["owner"]], -(-int(v["first_block"]) // BLOCKS_PER_EPOCH))
             for h, v in doc["heads"].items()}
    return window, published, sides
