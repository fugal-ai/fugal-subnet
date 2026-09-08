#!/usr/bin/env python3
"""Check vendored module/fixtures and packaged price/cost snapshots against pins."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", help="Core checkout containing the pinned commit; CI supplies it")
    args = p.parse_args()
    vendor = ROOT / "fugal_subnet/vendor"
    source = json.loads((vendor / "SOURCE.json").read_text())
    for name, spec in source["files"].items():
        data = (vendor / spec["vendored"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == spec["sha256"], name
        if args.core:
            expected = subprocess.check_output(["git", "show", source["commit"] + ":" + name], cwd=args.core)
            assert data == expected, name
    for name in ("models.json", "benchmark_tokens_v1.json"):
        assert (ROOT / "data" / name).read_bytes() == (ROOT / "fugal_subnet/routing_data" / name).read_bytes(), name
    print(f"Success contract synchronized with core {source['commit']}")


if __name__ == "__main__":
    main()
