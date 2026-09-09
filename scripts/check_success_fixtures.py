#!/usr/bin/env python3
"""Run portable numeric conformance with no backbone or inference calls."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fugal_subnet.vendor import success_contract as c  # noqa: E402

fixture = json.loads((ROOT / "fugal_subnet/vendor/conformance.json").read_text())
assert fixture["profile_id"] == c.PROFILE_ID
for case in fixture["cases"]:
    assert c.rank(c.sigmoid(case["logits"]), case["costs"], case["lam"]).tolist() == case["order"], case["name"]
print(f"{len(fixture['cases'])} shared numeric fixtures passed")
