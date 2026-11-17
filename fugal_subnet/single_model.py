"""The single-model table: what always calling one model buys (protocol v3).

Published in every v3 reveal, from the window's pooled exploration pairs
(explore_pool.py): for each model, how often it was right on questions nobody
chose it for, how many tokens it used, and what that costs at the pinned
prices. It is the measurement that used to need an owner-paid run of every
model over a sample of the pool, and it is refreshed every epoch by the
miners' own exploration.

It is a DIAGNOSTIC. It decides no seat and no weight: a per-model estimate on
a few hundred pairs is too noisy to gate a payout on, and a gate that fails
closed would empty the seat and burn the miner share. What it is for: showing
whether the seated router beats the best single model at its price; telling a
miner what each model does; and giving the next release its token statistics
and its list of models that are never worth routing to, at no cost.

A model with fewer than `min_pairs` pairs in the window is NOT LISTED. It is
never assumed to be perfect, or anything else, until it has been measured.

Accuracy is weighted to the scored mix: each category's rate times that
category's share of the scored slice (`targets`, renormalised over the
categories the model was actually measured on). Everything is exact rational
arithmetic until the final float, so two validators publish the same table.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from fugal_subnet.champion import EpochPairs
from fugal_subnet.head_odds import ODDS_SCALE


def single_model_table(
    pairs: Mapping[int, EpochPairs],
    prices: Mapping[str, tuple[float, float]],
    targets: Mapping[str, float],
    *,
    min_pairs: int,
) -> list[dict]:
    """One row per model with at least `min_pairs` pairs, sorted by model id."""
    stats: dict[str, dict[str, list[int]]] = {}
    tokens: dict[str, list[int]] = {}
    for e in sorted(pairs):
        ep = pairs[e]
        for model, cat, y, p_tok, c_tok in zip(ep.models, ep.categories, ep.y.tolist(),
                                               ep.prompt_tokens.tolist(),
                                               ep.completion_tokens.tolist()):
            c = stats.setdefault(model, {}).setdefault(cat, [0, 0])
            c[0] += int(y)
            c[1] += 1
            t = tokens.setdefault(model, [0, 0])
            t[0] += int(p_tok)
            t[1] += int(c_tok)
    rows = []
    for model in sorted(stats):
        n = sum(c[1] for c in stats[model].values())
        if n < int(min_pairs):
            continue
        weight = {cat: Fraction(targets.get(cat, 0.0)) for cat in stats[model]}
        total_w = sum(weight.values())
        if total_w > 0:
            accuracy = sum(weight[cat] * Fraction(c[0], c[1] * ODDS_SCALE)
                           for cat, c in stats[model].items()) / total_w
        else:
            accuracy = Fraction(sum(c[0] for c in stats[model].values()), n * ODDS_SCALE)
        mean_in, mean_out = Fraction(tokens[model][0], n), Fraction(tokens[model][1], n)
        row = {
            "model": model, "pairs": n,
            "accuracy": float(accuracy),
            "accuracy_unweighted": float(Fraction(sum(c[0] for c in stats[model].values()),
                                                  n * ODDS_SCALE)),
            "mean_prompt_tokens": float(mean_in),
            "mean_completion_tokens": float(mean_out),
            "by_category": {cat: {"pairs": c[1], "accuracy": float(Fraction(c[0], c[1] * ODDS_SCALE))}
                            for cat, c in sorted(stats[model].items())},
        }
        if model in prices:
            rate_in, rate_out = prices[model]
            row["spend_per_question"] = float(mean_in * Fraction(rate_in)
                                              + mean_out * Fraction(rate_out))
        rows.append(row)
    return rows


def best_at_or_below(rows: list[dict], spend: float) -> dict | None:
    """The most accurate listed model that costs no more than `spend` a question."""
    affordable = [r for r in rows if r.get("spend_per_question") is not None
                  and r["spend_per_question"] <= spend]
    return max(affordable, key=lambda r: (r["accuracy"], r["model"]), default=None)
