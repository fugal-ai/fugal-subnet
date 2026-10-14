#!/usr/bin/env python3
"""Build the shipped bootstrap reference frame from published epoch reveals.

The frame is what the constant-policy frontier is valued from, and a frontier
with no evidence pays nobody: every model is unmeasured, so the upper bound on
what a constant policy might achieve is 1.0 and no router clears it. Cold, the
subnet needs FRONTIER_MIN_TRIALS distinct observations per model before it pays
in full -- at fifteen exploration slots over seventeen models that is weeks of
epochs. A bootstrap frame measured on testnet makes epoch 1 look like epoch 300.

This script is the only sanctioned way to produce `data/reference_frame.json`.
It replays reveals through the same `rebuild_from_reveals` a validator uses
(one vote per coldkey, one sample per distinct question and model, decay per
epoch), prints what it measured so a reviewer can sanity-check every model's
accuracy against what they know of it, and writes the file. The file is
hash-pinned in scripts/check_safety_invariants.py because it is consensus
state: every validator must load the same one, so a change to it is a change
to every score and goes through review like data/models.json.

    python scripts/build_bootstrap_frame.py --reveals results/success-v1/epochs
    python scripts/build_bootstrap_frame.py --reveals /path/to/val2/epochs --out data/reference_frame.json

No network, no API spend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fugal_subnet.reference_frame import rebuild_from_reveals  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reveals", required=True,
                    help="Directory of <epoch>/reveal.json files to replay")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "reference_frame.json"))
    ap.add_argument("--min-trials", type=float, default=50.0,
                    help="Warn about any model measured fewer times than this")
    args = ap.parse_args()

    frame = rebuild_from_reveals(args.reveals)
    if not frame.trials:
        print(f"no exploration samples found under {args.reveals}", file=sys.stderr)
        return 1

    print(f"{'model':<44}{'trials':>8}{'accuracy':>10}{'out_tok':>9}")
    print("-" * 71)
    thin = []
    for model in sorted(frame.trials):
        n = frame.trials[model]
        acc = frame.successes[model] / n if n else 0.0
        out_tok = frame.completion_tokens.get(model, 0.0) / n if n else 0.0
        flag = "  <- thin" if n < args.min_trials else ""
        if flag:
            thin.append(model)
        print(f"{model:<44}{n:>8.1f}{acc:>10.3f}{out_tok:>9.0f}{flag}")
    print("-" * 71)
    print(f"{len(frame.trials)} models over {frame.epochs} epochs")
    if thin:
        print(f"{len(thin)} models under {args.min_trials:.0f} trials; the frontier will "
              "discount itself until they are measured live")

    payload = json.dumps(frame.to_dict(), indent=2, sort_keys=True) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(payload)
    digest = hashlib.sha256(payload.encode("utf-8").replace(b"\r\n", b"\n")).hexdigest()
    print(f"\nwrote {args.out}\nsha256 {digest}")
    print("Pin this digest as PINNED_BOOTSTRAP_FRAME_SHA256 in "
          "scripts/check_safety_invariants.py in the same commit, and say why.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
