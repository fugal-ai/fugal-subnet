#!/usr/bin/env python3
"""Build token-statistics candidates from recorded proof bundles. No paid calls.

Use --observations for normalized JSONL records with model, prompt_tokens,
completion_tokens. Inputs must be actual observations; zero input or absent
counts are excluded, never imputed. Human review promotes candidate to reviewed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fugal_subnet.vendor import success_contract as c  # noqa: E402


def build(paths, models, worker_profile):
    values = {m: [] for m in models}
    sources = {}
    omitted = 0
    seen = set()
    for path in paths:
        path = Path(path)
        sources[path.name] = c.file_hash(path)
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            bundle = json.loads(path.read_text())
            proof = bundle.get("proof", bundle)
            rows = proof["results"]
            proof_id = proof.get("hotkey", "") + ":" + proof.get("epoch_id", "")
            if proof_id in seen:
                raise ValueError("duplicate proof observations")
            seen.add(proof_id)
        for row in rows:
            m = row.get("model", row.get("routed_model"))
            it, ot = row.get("prompt_tokens"), row.get("completion_tokens")
            if m not in values:
                continue
            if type(it) is not int or type(ot) is not int or it <= 0 or ot < 0:
                omitted += 1
                continue
            values[m].append((it, ot))
    missing = [m for m, rows in values.items() if not rows]
    if missing:
        raise ValueError(f"missing recorded token observations: {missing}")
    return {"version": "fugal-token-statistics-v1", "profile_id": c.PROFILE_ID,
            "status": "candidate", "sources": sources, "worker_profile": worker_profile,
            "omitted_missing_or_zero_input": omitted,
            "models": [{"id": m, "samples": len(rows),
                        "mean_in_tokens": sum(it for it, _ in rows) / len(rows),
                        "mean_out_tokens": sum(ot for _, ot in rows) / len(rows)}
                       for m, rows in values.items()]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", nargs="+", required=True)
    p.add_argument("--models", default="data/models.json")
    p.add_argument("--worker-profile", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    models = [r["id"] for r in json.loads(Path(args.models).read_text())]
    manifest = build(args.observations, models, args.worker_profile)
    c.validate_manifest(manifest)
    Path(args.output).write_text(json.dumps(manifest, indent=2) + "\n")
    print(c.digest(manifest))


if __name__ == "__main__":
    main()
