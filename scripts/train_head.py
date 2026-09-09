#!/usr/bin/env python3
"""Train independent success heads. Missing labels are NaN, never failures.

Real matrices require explicit model column IDs and either questions or a profile
cache. Export is offline; live admission additionally requires the reviewed
benchmark manifest. --synthetic is test-only and cannot produce a deployable bundle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# isort: off
import fugal_subnet.determinism  # noqa: E402,F401
import numpy as np  # noqa: E402

# isort: on
from fugal_subnet.success_training import fit, grouped_split  # noqa: E402
from fugal_subnet.vendor import success_contract as contract  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--matrix")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--manifest", help="Token observation manifest")
    p.add_argument("--hidden-states", help="Profile-tagged .npz cache")
    p.add_argument("--prices", default=str(Path(__file__).resolve().parents[1] / "data/models.json"))
    p.add_argument("--output", default="data/head.npz")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-questions", type=int, default=300)
    args = p.parse_args()
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        questions = [f"test-only-{i}" for i in range(args.n_questions)]
        h = rng.normal(size=(len(questions), 1024)).astype(np.float32)
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        y = rng.integers(0, 2, (len(h), len(args.models))).astype(float)
        manifest = {"version": "fugal-token-statistics-v1", "profile_id": contract.PROFILE_ID,
                    "status": "synthetic-test-only", "sources": ["synthetic"],
                    "worker_profile": "synthetic; no worker calls",
                    "models": [{"id": m, "samples": len(h), "mean_in_tokens": 10.,
                                "mean_out_tokens": 10.} for m in args.models]}
        provenance = "synthetic-test-only"
    else:
        if not args.manifest:
            p.error("--manifest is required; missing token observations block export")
        with np.load(args.matrix, allow_pickle=False) as z:
            if list(z["models"]) != args.models:
                p.error("matrix model-column alignment mismatch")
            questions = list(z["questions"].astype(str))
            y = z["matrix"].astype(float)
        if args.hidden_states:
            h = contract.load_cache(args.hidden_states, questions)
        else:
            from fugal_subnet.backbone import compute_hidden_states
            h = compute_hidden_states(questions)
        manifest = json.loads(Path(args.manifest).read_text())
        contract.validate_manifest(manifest)
        provenance = json.dumps({"matrix_sha256": contract.file_hash(args.matrix), "seed": args.seed})
    train, val, test = grouped_split(questions, args.seed)
    W, b, epoch, history = fit(h, y, train, val, epochs=args.epochs, seed=args.seed)
    z = contract.make_head(W, b, args.models, manifest, provenance)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **z)
    contract.check_manifest(contract.load(path.read_bytes()), manifest)
    prices = {r["id"]: (r["in"] / 1e6, r["out"] / 1e6) for r in json.loads(Path(args.prices).read_text())}
    cost = contract.estimated_cost(z, prices)
    picked = contract.rank(contract.predictions(W, b, h[test]), cost, 1.0)[:, 0]
    chosen = y[test][np.arange(len(test)), picked]
    report = {"selected_epoch": epoch,
              "routing_lambda": 1.0, "price_sha256": contract.file_hash(args.prices),
              "routed_observed_labels": int(np.isfinite(chosen).sum()),
              "routed_accuracy": float(np.nanmean(chosen)) if np.isfinite(chosen).any() else None,
              "routed_estimated_cost": float(cost[picked].mean()), "history": history,
              "splits": {k: v.tolist() for k, v in zip(("train", "validation", "test"), (train, val, test))},
              "test_brier": float(np.nanmean((contract.predictions(W, b, h[test]) - y[test]) ** 2)),
              "provenance": provenance, "head_sha256": contract.file_hash(path)}
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    path.with_suffix(".tokens.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {path}; selected epoch {epoch} by validation BCE. Test Brier {report['test_brier']:.6f}")


if __name__ == "__main__":
    main()
