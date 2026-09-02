#!/usr/bin/env python3
"""Repin the v2 golden vector and rewrite its reviewable fixture.

Run this only after deliberately changing packaged consensus material, and
review the resulting ``tests/fixtures/v2_golden.json`` diff before committing.
A change confined to the ``material`` section is ordinary rebuild churn; any
change inside ``math`` is a consensus regression and must be investigated
rather than repinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from fugal_subnet.consensus_manifest import canonical_json

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_MODULE = ROOT / "fugal_subnet" / "v2" / "golden.py"
FIXTURE = ROOT / "tests" / "fixtures" / "v2_golden.json"


def fixture_bytes(vector: dict) -> bytes:
    """Pretty-print for reviewable diffs; the pins cover the canonical form."""
    return json.dumps(vector, indent=2, sort_keys=True, ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _replace_pin(source: str, name: str, value: str) -> str:
    pattern = re.compile(rf'^{name} = "[^"]*"$', re.MULTILINE)
    updated, count = pattern.subn(f'{name} = "{value}"', source)
    if count != 1:
        raise RuntimeError(f"expected exactly one {name} assignment, found {count}")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without writing anything",
    )
    args = parser.parse_args()

    # Imported late so a PENDING pin cannot fail the import path.
    from fugal_subnet.v2.golden import build_golden_vector

    vector = build_golden_vector()
    whole = hashlib.sha256(canonical_json(vector)).hexdigest()
    math = hashlib.sha256(canonical_json(vector["math"])).hexdigest()

    source = GOLDEN_MODULE.read_text(encoding="utf-8")
    current_fixture = FIXTURE.read_bytes() if FIXTURE.exists() else b""
    desired_fixture = fixture_bytes(vector)
    desired_source = _replace_pin(source, "EXPECTED_GOLDEN_SHA256", whole)
    desired_source = _replace_pin(desired_source, "EXPECTED_MATH_SHA256", math)

    if args.check:
        drifted = source != desired_source or current_fixture != desired_fixture
        print(f"whole-vector sha256: {whole}")
        print(f"math-only    sha256: {math}")
        print("DRIFT" if drifted else "up to date")
        return 1 if drifted else 0

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(desired_fixture)
    GOLDEN_MODULE.write_text(desired_source, encoding="utf-8")
    print(f"fixture written: {FIXTURE.relative_to(ROOT)}")
    print(f"EXPECTED_GOLDEN_SHA256 = {whole}")
    print(f"EXPECTED_MATH_SHA256   = {math}")
    print("Review the fixture diff before committing.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
