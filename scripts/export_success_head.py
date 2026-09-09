#!/usr/bin/env python3
"""Verify and bundle an offline success head. Publication and winner selection are manual."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fugal_subnet.vendor import success_contract as c  # noqa: E402


def export(head, manifest, prices, report, output, deployable=False):
    head, manifest, prices, report, output = map(Path, (head, manifest, prices, report, output))
    z = c.load(head.read_bytes())
    tokens = json.loads(manifest.read_text())
    c.check_manifest(z, tokens, deployable=deployable)
    rates = {r["id"]: (r["in"] / 1e6, r["out"] / 1e6) for r in json.loads(prices.read_text())}
    c.estimated_cost(z, rates)
    evaluation = json.loads(report.read_text())
    if not evaluation or not isinstance(evaluation, dict):
        raise ValueError("evaluation report is required")
    if evaluation.get("head_sha256") != c.file_hash(head):
        raise ValueError("evaluation report does not bind this head")
    if deployable and ("synthetic" in c.scalar(z, "provenance") or tokens["status"] != "reviewed"):
        raise ValueError("test/candidate artifacts cannot be exported as deployable")
    output.mkdir(parents=True, exist_ok=False)
    for source, target in ((head, "head.npz"), (manifest, "tokens.json"),
                           (prices, "prices.json"), (report, "evaluation.json")):
        shutil.copyfile(source, output / target)
    metadata = {"contract": c.CONTRACT, "profile_id": c.PROFILE_ID,
                "provenance": c.scalar(z, "provenance"), "deployable": deployable,
                "default_lambda": float(z["lam"]), "cost_profile_id": c.scalar(z, "cost_profile_id"),
                "files": {p.name: c.file_hash(p) for p in sorted(output.iterdir())}}
    (output / "bundle.json").write_text(json.dumps(metadata, indent=2) + "\n")
    verify(output)
    return metadata


def verify(output):
    output = Path(output)
    metadata = json.loads((output / "bundle.json").read_text())
    if set(metadata["files"]) != {"head.npz", "tokens.json", "prices.json", "evaluation.json"}:
        raise ValueError("invalid bundle file list")
    for name, expected in metadata["files"].items():
        if c.file_hash(output / name) != expected:
            raise ValueError(f"bundle hash mismatch: {name}")
    z = c.load((output / "head.npz").read_bytes())
    expected = {"contract": c.CONTRACT, "profile_id": c.PROFILE_ID,
                "provenance": c.scalar(z, "provenance"), "default_lambda": float(z["lam"]),
                "cost_profile_id": c.scalar(z, "cost_profile_id")}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("bundle metadata disagrees with head")
    if not isinstance(metadata.get("deployable"), bool):
        raise ValueError("invalid deployable flag")
    c.check_manifest(z, json.loads((output / "tokens.json").read_text()), metadata["deployable"])
    prices = json.loads((output / "prices.json").read_text())
    c.estimated_cost(z, {r["id"]: (r["in"] / 1e6, r["out"] / 1e6) for r in prices})
    evaluation = json.loads((output / "evaluation.json").read_text())
    if evaluation.get("head_sha256") != c.file_hash(output / "head.npz"):
        raise ValueError("evaluation report does not bind this head")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head")
    p.add_argument("--manifest")
    p.add_argument("--prices")
    p.add_argument("--report")
    p.add_argument("--output")
    p.add_argument("--deployable", action="store_true", help="Requires reviewed recorded observations")
    p.add_argument("--verify", help="Verify an existing bundle directory")
    args = p.parse_args()
    if args.verify:
        result = verify(args.verify)
    else:
        if not all((args.head, args.manifest, args.prices, args.report, args.output)):
            p.error("export requires --head --manifest --prices --report --output")
        result = export(args.head, args.manifest, args.prices, args.report, args.output, args.deployable)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
