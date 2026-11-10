#!/usr/bin/env python3
"""Run portable numeric conformance with no backbone or inference calls."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fugal_subnet.vendor import success_contract as c  # noqa: E402

# Explicit exits, not `assert`: python -O strips asserts and this would pass
# having checked nothing.
fixture = json.loads((ROOT / "fugal_subnet/vendor/conformance.json").read_text())
if fixture["profile_id"] != c.PROFILE_ID:
    raise SystemExit(f"profile_id {fixture['profile_id']!r} != contract {c.PROFILE_ID!r}")
for case in fixture["cases"]:
    order = c.rank(c.sigmoid(case["logits"]), case["costs"], case["lam"]).tolist()
    if order != case["order"]:
        raise SystemExit(f"fixture {case['name']!r}: got order {order}, expected {case['order']}")
print(f"{len(fixture['cases'])} shared numeric fixtures passed")
