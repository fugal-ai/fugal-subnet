#!/usr/bin/env python3
"""Does a trained routing head beat a random one?

This is the subnet's founding premise and it had never been tested. The whole
mechanism -- TEE proofs, evidence accumulation, weight setting -- is machinery
for paying miners to find a better routing policy. If a trained head does not
route better than chance, none of that machinery measures anything.

Method
------
Ground truth comes from `scripts/experiment_routing_dataset.py` (see its
docstring for what that data is and is not). Questions are split into disjoint
train and test halves; every strategy is fit on train only and reported on
test.

Strategies compared, all scored through the SAME metric function so the
comparison is like for like:

  trained_sft         the repository's own trainer (scripts/train_head.py
                      train_sft), fit on train-split soft targets
  trained_sft_cma     the same, plus the shipped sep-CMA-ES refinement
  trained_ce          the same architecture trained harder, by cross-entropy
                      against the cheapest-correct model. Present to separate
                      "routing is not learnable" from "the shipped trainer is
                      mis-configured" -- two findings with very different fixes
  random_head         a head with random W and b at the trainer's own init
                      scale, evaluated through the shipped routing rule
  random_routing      per-question uniform choice over models -- the honest
                      "chance" baseline (see NOTE below)
  cheapest            always the cheapest model
  best_single         always the single model with the highest train accuracy;
                      this is the subnet's own reference (reference_frame.
                      best_model), so score 1.0 is exactly "matched it"
  best_constant       the single model with the highest COMPOSITE score on
                      train. The strongest null: quality x thrift may favour a
                      cheaper model than the reference, and unlike a random
                      head it has no seed variance to get lucky on
  domain_oracle       the best model for each question's SOURCE dataset, fit
                      on train. Reads a metadata label the head never sees, so
                      it is a reference rather than a competitor -- a head that
                      only matches it has learned domain identity, not routing
  oracle              cheapest model that got the question right -- the ceiling

NOTE on "random". A random *head* is not the same as random *routing*. Over
L2-normalised embeddings that occupy a narrow cone, a random linear map's
argmax is very nearly constant, so a random head behaves like "always model k
for an arbitrary k" rather than like a coin flip. Both are reported, because
"beats a random head" and "beats chance" are different claims and only the
second one is interesting.

Routing decisions for every head-based strategy come from
`fugal_subnet.head_eval.evaluate_head`, which applies the same rule the miner's
harness does -- softmax, quantise, argmax (`tee/harness._route_question`).

Scoring does NOT come from that function. `evaluate_head` drops questions no
model answered; the consensus path does not, because `proof.accuracy` divides
by every scored question. `score_decisions` below follows the consensus path.
See its docstring -- the difference is worth 7 points of apparent score.

No paid API call happens anywhere in this file.

Usage
-----
    python scripts/experiment_head_efficacy.py --json results/head_efficacy.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# isort: off
# Order is load-bearing: numpy and torch read the CPU-dispatch env vars once,
# at import, so determinism has to come first.
import fugal_subnet.determinism  # noqa: F401,E402

import numpy as np  # noqa: E402

from fugal_subnet.config import (  # noqa: E402
    HEAD_HIDDEN_DIM,
)

# The LEGACY score this experiment measured, kept verbatim so the numbers in
# docs/HEAD_EFFICACY.md stay reproducible. Scoring moved to headroom above the
# constant-policy frontier (fugal_subnet/frontier.py) because of what this
# experiment found; these constants no longer exist in config.
SCORE_QUALITY_CAP = 2.0
SCORE_QUALITY_EXPONENT = 0.9
SCORE_THRIFT_CAP = 10.0
from fugal_subnet.head_eval import HeadArtifact, evaluate_head  # noqa: E402
from fugal_subnet.scoring import wilson_lower_bound  # noqa: E402
from fugal_subnet.soft_targets import compute_soft_targets  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_routing_dataset import load_experiment_data  # noqa: E402

logger = logging.getLogger("fugal.experiment.efficacy")


def _load_trainer():
    """Import scripts/train_head.py as a module.

    Deliberate: the claim under test is about the trainer this repository
    ships, so the experiment must call that code and not a reimplementation
    of it.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_preference_training.py")
    spec = importlib.util.spec_from_file_location("fugal_train_head", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]  # train_head parses argv at import time only in main()
    spec.loader.exec_module(mod)
    return mod


# -- metrics --

def score_decisions(
    decisions: np.ndarray,
    matrix: np.ndarray,
    cost: np.ndarray,
    ref_idx: int,
    acc_best: float,
) -> dict:
    """Score a routing decision vector the way the LIVE subnet scores a miner.

    Every question counts, including ones no model answered.

    That is deliberate and it is the definition the consensus path uses:
    `proof.n_total` is `len(scored_results)` and `proof.accuracy` divides by
    it, so a question nobody can do is a question the miner got wrong. The
    validator could not do otherwise -- it holds one proof and has no idea what
    the other twelve models would have said.

    `head_eval.evaluate_head` DOES exclude them ("questions no model in the
    pool answered correctly carry no routing signal"), but nothing on the TEE
    path calls it: `_proof_to_head_score` builds the HeadScore straight from
    the proof. Mixing the two definitions is not a small error. The reference's
    accuracy is a raw rate over everything it was asked, so pairing it with a
    numerator computed over answerable questions only puts a harder population
    under the denominator than the numerator -- measured here, that alone lifts
    the reference model's score against ITSELF from 0.994 to 1.065, which reads
    as "beat the best single model by 6.5%" and is entirely an artefact.
    """
    n = len(decisions)
    if n == 0:
        raise ValueError("no questions")

    q = np.arange(n)
    sel = decisions
    correct = matrix[q, sel].astype(bool)
    spend = cost[q, sel].sum()
    ref_spend = cost[q, ref_idx].sum()
    answerable = matrix.sum(axis=1) > 0
    oracle_spend = sum(
        cost[i, np.where(matrix[i] == 1)[0]].min() for i in np.where(answerable)[0]
    )

    acc = float(correct.mean())
    lcb = wilson_lower_bound(acc, n)
    quality = min(max(lcb / max(acc_best, 1e-9), 0.0), SCORE_QUALITY_CAP)
    thrift = min(max(ref_spend / max(spend, 1e-12), 0.0), SCORE_THRIFT_CAP)
    w = SCORE_QUALITY_EXPONENT
    composite = (quality ** w) * (thrift ** (1.0 - w))

    return {
        "n_scored": n,
        "accuracy": acc,
        # Routing skill given the question was answerable at all. Not what the
        # subnet scores; useful because it separates "routed badly" from "the
        # question was impossible", which the live metric cannot.
        "accuracy_on_answerable": float(
            matrix[answerable, sel[answerable]].mean()) if answerable.any() else 0.0,
        "n_answerable": int(answerable.sum()),
        "accuracy_lcb": float(lcb),
        "cost_usd": float(spend),
        "cost_per_question": float(spend / n),
        "ref_cost_usd": float(ref_spend),
        "oracle_cost_usd": float(oracle_spend),
        "quality": float(quality),
        "thrift": float(thrift),
        "composite_score": float(composite),
        "n_distinct_models_used": int(len(np.unique(sel))),
        # A head can beat a constant policy while BEING one. These two say
        # whether it actually routes: `max_model_share` near 1.0 and entropy
        # near 0 mean the head picked a favourite and stopped reading the
        # question, whatever its accuracy says.
        "max_model_share": float(np.bincount(sel).max() / len(sel)),
        "routing_entropy_bits": float(_entropy_bits(sel)),
    }


def _entropy_bits(sel: np.ndarray) -> float:
    counts = np.bincount(sel)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def head_decisions(W, b, hidden, matrix, models, soft, model_costs) -> np.ndarray:
    """Routing decisions via the shipped validator rule."""
    head = HeadArtifact(W=W.astype(np.float32), b=b.astype(np.float32),
                        models=list(models), commit_hash="")
    hs = evaluate_head(head, hidden, matrix, list(models), soft, model_costs)
    return hs.routing_decisions.astype(int)


# -- strategies --

def run(args) -> dict:
    hidden, matrix, cost, models, sources, raw = load_experiment_data(args.threshold)
    n, m = matrix.shape
    assert hidden.shape == (n, HEAD_HIDDEN_DIM), hidden.shape
    logger.info("data: %d questions x %d models, %d sources",
                n, m, len(set(sources)))

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_train = int(n * args.train_frac)
    tr, te = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    logger.info("split: %d train / %d test questions", len(tr), len(te))

    H_tr, H_te = hidden[tr], hidden[te]
    M_tr, M_te = matrix[tr], matrix[te]
    C_tr, C_te = cost[tr], cost[te]

    # -- the reference model, chosen exactly as reference_frame.best_model does:
    # highest accuracy lower bound on the evidence available, ties to cheaper.
    # Here the evidence is the train split.
    tr_acc = M_tr.mean(axis=0)
    tr_lcb = np.array([wilson_lower_bound(float(a), len(tr)) for a in tr_acc])
    mean_cost = C_tr.mean(axis=0)
    order = sorted(range(m), key=lambda i: (-tr_lcb[i], mean_cost[i]))
    ref_idx = order[0]
    acc_best = float(M_te[:, ref_idx].mean())  # ceiling, valued on the test split
    logger.info("reference model: %s (train lcb %.3f), test accuracy %.3f",
                models[ref_idx], tr_lcb[ref_idx], acc_best)

    soft_tr = compute_soft_targets(M_tr, tau=args.tau)
    soft_te = compute_soft_targets(M_te, tau=args.tau)
    model_costs = {mm: float(mean_cost[i]) for i, mm in enumerate(models)}

    trainer = _load_trainer()
    results: dict[str, dict] = {}
    decisions_by_name: dict[str, np.ndarray] = {}

    def record(name: str, dec: np.ndarray) -> dict:
        decisions_by_name[name] = dec
        results[name] = score_decisions(dec, M_te, C_te, ref_idx, acc_best)
        return results[name]

    # -- trained head: the repository's own two-stage pipeline --
    t0 = time.time()
    W, b = trainer.train_sft(
        H_tr, soft_tr, m, HEAD_HIDDEN_DIM,
        epochs=args.sft_epochs, lr=args.sft_lr,
        batch_size=args.sft_batch_size, device=args.device,
    )
    logger.info("SFT done in %.1fs", time.time() - t0)
    record("trained_sft", head_decisions(W, b, H_te, M_te, models, soft_te, model_costs))
    results["trained_sft"]["train_accuracy"] = score_decisions(
        head_decisions(W, b, H_tr, M_tr, models, soft_tr, model_costs),
        M_tr, C_tr, ref_idx, float(M_tr[:, ref_idx].mean()))["accuracy"]

    if not args.skip_cma:
        t0 = time.time()
        W2, b2 = trainer.train_cma(
            W, b, H_tr, M_tr, mean_cost, args.lam,
            generations=args.cma_generations, popsize=args.cma_popsize,
            sigma=args.cma_sigma,
        )
        logger.info("CMA done in %.1fs", time.time() - t0)
        record("trained_sft_cma",
               head_decisions(W2, b2, H_te, M_te, models, soft_te, model_costs))
        results["trained_sft_cma"]["train_accuracy"] = score_decisions(
            head_decisions(W2, b2, H_tr, M_tr, models, soft_tr, model_costs),
            M_tr, C_tr, ref_idx, float(M_tr[:, ref_idx].mean()))["accuracy"]

    # -- a head trained harder than the shipped trainer trains one --
    #
    # Here to separate two very different findings. If the shipped head loses
    # and this one wins, the architecture is fine and `train_head.py` is
    # mis-configured -- a bug with a fix. If BOTH lose, the premise itself is
    # in trouble. Cross-entropy against the cheapest-correct model, which is a
    # sharp label, where SOFT_TARGET_TAU=1.0 makes the shipped target a nearly
    # flat distribution (e^1 : e^0 is only 2.72 : 1).
    if not args.skip_ce:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from experiment_pool_memorization import fit_linear_head

        keep, lab = [], []
        for i in range(len(tr)):
            ok = np.where(M_tr[i] == 1)[0]
            if len(ok):
                keep.append(i)
                lab.append(int(ok[np.argmin(C_tr[i, ok])]))
        Wc, bc = fit_linear_head(
            H_tr[keep], np.array(lab), m, args.ce_epochs, args.ce_lr,
            args.device, weight_decay=args.ce_weight_decay,
        )
        record("trained_ce",
               head_decisions(Wc, bc, H_te, M_te, models, soft_te, model_costs))
        results["trained_ce"]["train_accuracy"] = score_decisions(
            head_decisions(Wc, bc, H_tr, M_tr, models, soft_tr, model_costs),
            M_tr, C_tr, ref_idx, float(M_tr[:, ref_idx].mean()))["accuracy"]

    # -- random head, at the trainer's own initialisation scale --
    rand_runs = []
    for s in range(args.random_seeds):
        r = np.random.RandomState(9000 + s)
        Wr = (r.randn(m, HEAD_HIDDEN_DIM) * 0.01).astype(np.float32)
        br = np.zeros(m, dtype=np.float32)
        d = head_decisions(Wr, br, H_te, M_te, models, soft_te, model_costs)
        rand_runs.append(score_decisions(d, M_te, C_te, ref_idx, acc_best))
    results["random_head"] = _aggregate(rand_runs)

    # -- random routing: genuine chance, one uniform draw per question --
    chance_runs = []
    for s in range(args.random_seeds):
        r = np.random.RandomState(4000 + s)
        d = r.randint(0, m, size=len(te))
        chance_runs.append(score_decisions(d, M_te, C_te, ref_idx, acc_best))
    results["random_routing"] = _aggregate(chance_runs)

    # -- the best CONSTANT policy: the strongest null there is --
    #
    # `best_single` is the subnet's reference model, picked on accuracy alone.
    # But the score is quality x thrift, so some other constant policy may
    # score higher than the reference does -- a slightly worse model at a
    # tenth of the price, for instance. That is the bar a head must clear to
    # be earning its place, and unlike a random head it carries no seed
    # variance: it is a deterministic max over the model list.
    #
    # Chosen on TRAIN and reported on TEST. Choosing it on test would be
    # picking the winner after seeing the answers, which flatters the null in
    # exactly the way that would make a real head look worse than it is.
    acc_best_tr = float(M_tr[:, ref_idx].mean())
    const_scores = [
        score_decisions(np.full(len(tr), k), M_tr, C_tr, ref_idx, acc_best_tr
                        )["composite_score"]
        for k in range(m)
    ]
    best_const = int(np.argmax(const_scores))
    record("best_constant", np.full(len(te), best_const))

    # -- degenerate baselines --
    cheapest = int(np.argmin(mean_cost))
    record("cheapest", np.full(len(te), cheapest))
    record("best_single", np.full(len(te), ref_idx))

    # -- domain oracle: the best model for each question's SOURCE dataset,
    # fit on the train split. Not a competitor -- it reads a metadata label the
    # head never sees -- but the most useful reference in the table. If the
    # trained head only matches this, it has learned to recognise the domain
    # and nothing finer, which is a much weaker claim than "it routes".
    def _group(s: str) -> str:
        return "ragbench" if s.startswith("rungalileo") else s

    groups = np.array([_group(s) for s in sources])
    g_tr, g_te = groups[tr], groups[te]
    fallback = int(np.argmax(M_tr.mean(axis=0)))
    per_group = {}
    for g in set(g_tr):
        rows = M_tr[g_tr == g]
        per_group[g] = int(np.argmax(rows.mean(axis=0))) if len(rows) > 20 else fallback
    record("domain_oracle", np.array([per_group.get(g, fallback) for g in g_te]))

    # -- oracle: cheapest correct model, per question --
    oracle = np.zeros(len(te), dtype=int)
    for i in range(len(te)):
        ok = np.where(M_te[i] == 1)[0]
        oracle[i] = ok[np.argmin(C_te[i, ok])] if len(ok) else 0
    record("oracle", oracle)

    # WITHIN-BENCHMARK concentration, which is the diagnostic that actually
    # separates a router from a domain classifier. A head sending 95% of MMLU
    # to model A and 95% of MATH to model B has a global max share near 0.5 and
    # looks beautifully spread, while being exactly the two-line lookup on the
    # `benchmark` field that `domain_oracle` implements. Averaging the share
    # WITHIN each source collapses that disguise; the global figure cannot.
    for name, dec in decisions_by_name.items():
        shares = []
        for g in set(g_te):
            sel_g = g_te == g
            if sel_g.sum() < 50:
                continue
            counts = np.bincount(dec[sel_g], minlength=m)
            shares.append(counts.max() / sel_g.sum())
        results[name]["within_source_max_share"] = (
            float(np.mean(shares)) if shares else float("nan"))

    # Per-source accuracy for the strategies that route, against the reference.
    # Worth having because the backbone truncates at 512 tokens: on a long RAG
    # context the head is reading the first fragment and nothing else, so a
    # domain where it fails may be a domain the EMBEDDING never saw rather than
    # one where routing is impossible.
    by_source: dict[str, dict] = {}
    for g in sorted(set(g_te)):
        sel = g_te == g
        if sel.sum() < 50:
            continue
        row = {"n": int(sel.sum()),
               "best_single": float(M_te[sel][:, ref_idx].mean()),
               "oracle": float((M_te[sel].sum(axis=1) > 0).mean())}
        for name, dec in decisions_by_name.items():
            row[name] = float(M_te[sel][np.arange(int(sel.sum())), dec[sel]].mean())
        by_source[g] = row

    meta = {
        "n_questions": n,
        "n_train": len(tr),
        "n_test": len(te),
        "n_models": m,
        "per_source": by_source,
        "threshold": args.threshold,
        "reference_model": models[ref_idx],
        "cheapest_model": models[cheapest],
        "acc_best_on_test": acc_best,
        "per_model_test_accuracy": {models[i]: float(M_te[:, i].mean()) for i in range(m)},
        "per_model_mean_cost_usd": {models[i]: float(mean_cost[i]) for i in range(m)},
        "score_quality_exponent": SCORE_QUALITY_EXPONENT,
        "sources": sorted(set(sources)),
    }
    return {"meta": meta, "results": results}


def _aggregate(runs: list[dict]) -> dict:
    out: dict = {"n_runs": len(runs)}
    for k in runs[0]:
        vals = np.array([r[k] for r in runs], dtype=np.float64)
        out[k] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


def _report(blob: dict) -> None:
    meta, res = blob["meta"], blob["results"]
    print()
    print(f"{meta['n_test']} test questions, {meta['n_models']} models, "
          f"reference = {meta['reference_model']} (acc {meta['acc_best_on_test']:.3f})")
    print()
    hdr = (f"{'strategy':<20}{'accuracy':>10}{'cost/q $':>12}{'thrift':>9}"
           f"{'score':>9}{'models':>8}{'share':>7}{'in-src':>8}{'bits':>7}")
    print(hdr)
    print("-" * len(hdr))
    for name in ["oracle", "domain_oracle", "trained_ce", "trained_sft",
                 "trained_sft_cma", "best_constant", "best_single",
                 "random_head", "random_routing", "cheapest"]:
        r = res.get(name)
        if r is None:
            continue
        std = f" +/-{r['accuracy_std']:.3f}" if "accuracy_std" in r else ""
        print(f"{name:<20}{r['accuracy']:>10.3f}{r['cost_per_question']:>12.6f}"
              f"{r['thrift']:>9.2f}{r['composite_score']:>9.3f}"
              f"{r['n_distinct_models_used']:>8.1f}{r['max_model_share']:>7.2f}"
              f"{r.get('within_source_max_share', float('nan')):>8.2f}"
              f"{r['routing_entropy_bits']:>7.2f}{std}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1105)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--sft-epochs", type=int, default=100)
    p.add_argument("--sft-lr", type=float, default=1e-3)
    p.add_argument("--sft-batch-size", type=int, default=64)
    p.add_argument("--cma-generations", type=int, default=50)
    p.add_argument("--cma-popsize", type=int, default=32)
    p.add_argument("--cma-sigma", type=float, default=0.1)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--skip-cma", action="store_true")
    p.add_argument("--skip-ce", action="store_true")
    p.add_argument("--ce-epochs", type=int, default=3000)
    p.add_argument("--ce-lr", type=float, default=0.05)
    p.add_argument("--ce-weight-decay", type=float, default=0.0)
    p.add_argument("--random-seeds", type=int, default=20)
    p.add_argument("--json", type=str, default="")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    blob = run(args)
    _report(blob)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(blob, fh, indent=2)
        logger.info("wrote %s", args.json)


if __name__ == "__main__":
    main()
