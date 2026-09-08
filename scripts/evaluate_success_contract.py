#!/usr/bin/env python3
"""Reproducible SPROUT offline mechanism experiment; never calls workers."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# isort: off
import fugal_subnet.determinism  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

# isort: on
from fugal_subnet.success_training import fit, grouped_split  # noqa: E402
from fugal_subnet.vendor import success_contract as c  # noqa: E402


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(np.sum(v > neg) + .5 * np.sum(v == neg)) / len(neg) for v in pos]))


def summary(p, y):
    observed = np.isfinite(y)
    p, y = p[observed], y[observed]
    bins = []
    for lo in np.arange(0, 1, .1):
        mask = (p >= lo) & (p < lo + .1 if lo < .9 else p <= 1)
        if mask.any():
            bins.append({"lo": float(lo), "n": int(mask.sum()), "mean_prediction": float(p[mask].mean()),
                         "success_rate": float(y[mask].mean())})
    return {"n": len(y), "brier": float(np.mean((p - y) ** 2)), "calibration": bins}


def interval(values, draws):
    estimates = np.nanmean(values[draws], axis=1)
    return {"mean": float(np.nanmean(values)), "ci95": np.quantile(estimates, [.025, .975]).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, choices=[512, 2048], required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out, src = Path(args.output), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    idx = json.loads((src / "sprout_embed_index.json").read_text())["index"]
    questions = json.loads((src / "sprout_prompts.json").read_text())["prompts"]
    questions = [questions[i] for i in idx]
    with np.load(src / "sprout_matrix.npz", allow_pickle=False) as z:
        models = z["models"].tolist()
        scores, it, ot = (z[k][idx].astype(float) for k in ("scores", "in_tokens", "out_tokens"))
        costs = z["cost_usd"][idx].astype(float)
    costs = np.where((it > 0) & np.isfinite(it) & np.isfinite(ot) & (ot >= 0), costs, np.nan)
    labels = np.where(np.isfinite(scores), (scores >= 1).astype(float), np.nan)
    splits = grouped_split(questions)
    train, val, test = splits
    profile_id = c.PROFILE_ID if args.max_length == 2048 else c.digest(dict(c.PROFILE, max_length=512, version="experimental-512"))
    cache = out / f"embeddings-{args.max_length}.npz"
    metric_file = out / f"embedding-metrics-{args.max_length}.json"
    if cache.exists():
        hidden = c.load_cache(cache, questions, profile_id)
        metrics = json.loads(metric_file.read_text())
    else:
        from transformers import AutoModel, AutoTokenizer
        c.check_backbone(args.backbone)
        tok = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
        model = AutoModel.from_pretrained(args.backbone, local_files_only=True, dtype=torch.float32).eval()
        lengths = [len(tok(c.format_question(q))["input_ids"]) for q in questions]
        order = np.argsort(lengths, kind="stable")
        hidden = np.empty((len(questions), 1024), dtype=np.float32)
        started = time.monotonic()
        for start in range(0, len(questions), args.batch_size):
            batch = order[start:start + args.batch_size]
            hidden[batch] = c.embed(tok, model, [questions[i] for i in batch], args.batch_size, args.max_length)
            if start % (args.batch_size * 25) == 0:
                print(f"{args.max_length}: {start}/{len(questions)}; {time.monotonic()-started:.1f}s", flush=True)
        seconds = time.monotonic() - started
        metrics = {"seconds": seconds, "seconds_per_question": seconds / len(questions),
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "truncated": sum(n > args.max_length for n in lengths), "n": len(questions),
                   "batch_size": args.batch_size, "threads": args.threads,
                   "device": "cpu", "dtype": "float32", "profile_id": profile_id}
        c.save_cache(cache, questions, hidden, profile_id)
        metric_file.write_text(json.dumps(metrics, indent=2))
        del model
    W, b, epoch, history = fit(hidden, labels, train, val, epochs=args.epochs)
    probs = c.predictions(W, b, hidden[test])
    y, actual_cost = labels[test], costs[test]
    # The old importer replaced missing counts with zero. Only strictly positive
    # input and finite nonnegative output observations can be counted here. This
    # still cannot establish completeness of output counts; export stays offline.
    usable = np.isfinite(it) & np.isfinite(ot) & (it > 0) & (ot >= 0)
    sources = {name: c.file_hash(src / name) for name in
               ("sprout_matrix.npz", "sprout_prompts.json", "sprout_embed_index.json")}
    manifest = {"version": "fugal-token-statistics-v1", "profile_id": c.PROFILE_ID,
                "status": "offline-incomplete-provenance", "sources": sources,
                "worker_profile": "SPROUT historical provider requests; exact generation settings unavailable in local extract",
                "models": [{"id": m, "samples": int(usable[train, j].sum()),
                            "mean_in_tokens": float(it[train, j][usable[train, j]].mean()),
                            "mean_out_tokens": float(ot[train, j][usable[train, j]].mean())}
                           for j, m in enumerate(models)]}
    # Preserve the historical experiment's assumed snapshot and label it explicitly.
    from experiment_routing_dataset import PRICES_USD_PER_MTOK
    price_rows = [{"id": m, "in": PRICES_USD_PER_MTOK[m][0], "out": PRICES_USD_PER_MTOK[m][1]} for m in models]
    prices = {r["id"]: (r["in"] / 1e6, r["out"] / 1e6) for r in price_rows}
    z = c.make_head(W, b, models, manifest, json.dumps({"sources": sources, "selected_epoch": epoch,
                                                     "profile_id": profile_id, "experiment": "offline mechanism only"}))
    # 512 weights must not masquerade as the production embedding profile.
    if args.max_length == 512:
        z["contract"] = np.asarray("experimental-success-512")
        z["profile_id"] = np.asarray(profile_id)
    expected_cost = c.estimated_cost(z, prices)
    draws = np.random.default_rng(7).integers(0, len(test), (2000, len(test)))
    baselines = {}
    for j, m in enumerate(models):
        baselines[m] = {"accuracy": interval(y[:, j], draws), "recorded_cost": interval(actual_cost[:, j], draws)}
    cheap = int(np.argmin(expected_cost))
    routes = {}
    for lam in (0, .5, 1, 2, 5):
        picked = c.rank(probs, expected_cost, lam)[:, 0]
        routed_y = y[np.arange(len(test)), picked]
        routed_cost = actual_cost[np.arange(len(test)), picked]
        routes[str(lam)] = {"accuracy": interval(routed_y, draws), "recorded_cost": interval(routed_cost, draws),
                           "observed_labels": int(np.isfinite(routed_y).sum()),
                           "paired_accuracy_minus_fixed": {m: interval(routed_y-y[:, j], draws) for j, m in enumerate(models)},
                           "selected_counts": {m: int((picked == j).sum()) for j, m in enumerate(models)}}
    # Fixed mixtures chosen using validation outcomes only, evaluated with the
    # same held-out bootstrap draws. Report the complete grid including endpoints.
    mixtures = []
    # This experiment's frontier below is explicit to keep its recorded costs
    # separate from the subnet reward frontier's evolving reference frame.
    vc, va = np.nanmean(costs[val], axis=0), np.nanmean(labels[val], axis=0)
    points = sorted((vc[j], va[j], j) for j in range(len(models)))
    hull = []
    for point in points:
        if hull and point[1] <= hull[-1][1]:
            continue
        while len(hull) >= 2:
            a, b0 = hull[-2:]
            if (b0[1]-a[1])*(point[0]-b0[0]) <= (point[1]-b0[1])*(b0[0]-a[0]):
                hull.pop()
            else:
                break
        hull.append(point)
    for left, right in zip(hull, hull[1:]):
        for weight in (0, .25, .5, .75, 1):
            j, k = left[2], right[2]
            mixture_y = (1-weight)*y[:, j] + weight*y[:, k]
            mixture_c = (1-weight)*actual_cost[:, j] + weight*actual_cost[:, k]
            mixtures.append({"left": models[j], "right": models[k], "right_weight": weight,
                             "accuracy": interval(mixture_y, draws), "recorded_cost": interval(mixture_c, draws)})
    report = {"scope": "SPROUT offline mechanism test, not live-subnet calibration; historical assumed prices",
              "sources": sources, "embedding": metrics, "label_conversion": "finite score >= 1 => 1; lower => 0; NaN remains missing",
              "token_limitations": "Original importer replaced missing token counts with zero. Positive input only used for manifest. Output zero may be missing. No deployable export.",
              "worker_profile": manifest["worker_profile"], "selected_indices": idx,
              "splits": {name: split.tolist() for name, split in zip(("train", "validation", "test"), splits)},
              "split_method": "seed 42; sorted unique exact prompts permuted; 60/20/20 groups",
              "selected_epoch": epoch, "training_history": history,
              "pooled": summary(probs, y), "per_model": {m: dict(summary(probs[:, j], y[:, j]), auc=auc(probs[:, j], y[:, j])) for j, m in enumerate(models)},
              "routes": routes, "fixed_models": baselines, "cheapest_model": models[cheap],
              "fixed_mixture_frontier": mixtures, "bootstrap": "2000 paired question resamples, seed 7; conditional on selected sample and fitted head"}
    np.savez(out / f"head-{args.max_length}.npz", **z)
    report["head_sha256"] = c.file_hash(out / f"head-{args.max_length}.npz")
    report["profile_id"] = profile_id
    report["prices"] = price_rows
    for suffix, obj in (("report", report), ("tokens", manifest), ("prices", price_rows)):
        (out / f"{suffix}-{args.max_length}.json").write_text(json.dumps(obj, indent=2, allow_nan=False))
    np.savez(out / f"head-{args.max_length}.npz", **z)
    print(json.dumps({"length": args.max_length, "epoch": epoch, "brier": report["pooled"]["brier"], "embedding": metrics}), flush=True)


if __name__ == "__main__":
    main()
