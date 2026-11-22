#!/usr/bin/env python3
"""Build the observations the v4 allowance is checked on
(data/observations/allowance_rows_e00026475-e00026509.json).

docs/decisions/SEAT_MINIMAL_V4.md 4.2 claims that, on the published field,
deleting a row for a new entry is never better than keeping it when a head
without the row is scored at a single constant rate for the entry. The
claim is checked (tests/test_allowance_rows.py) on what heads really
predicted for the last entries that joined a menu: v3 menu B's seven
entries, which joined at e00026475 and were explored, pooled and given odds
by every serving head from then on.

THE FILE is built by this script from the public reveals, saved with a
generic client and no key, so anyone can repeat it:

    for e in e00026475 ... e00026509; do
      curl -A reveal-snapshot/1.0 -o "reveals/$e.json" \\
        "https://storage.googleapis.com/fugal-reveals/70/$e/reveal.json"; done
    python scripts/build_allowance_observations.py observe reveals \\
        --out data/observations/allowance_rows_e00026475-e00026509.json

It keeps each reveal's sha256 (the same bytes the seed's picks file names),
and per pooled pair of one of the seven entries: the reveal it is in, the
entry, the pooled outcome y and the pair's published exploration weight, in
the reveal's integer units. Per head that published odds on any of those
pairs: its odds on each, in ODDS_SCALE units. Heads are numbered in order of
their hash; no head, operator or answer is named.

`observe --check` rebuilds and compares with the committed file.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMAT = "allowance-rows-v1"
REVEAL_SOURCE = "https://storage.googleapis.com/fugal-reveals/70/{epoch_id}/reveal.json"
FIRST, LAST = 26475, 26509            # menu B's join through the last published v3 epoch
OUT_PATH = os.path.join(ROOT, "data", "observations",
                        "allowance_rows_e00026475-e00026509.json")


def _odds(text: str, n: int) -> list[int]:
    raw = base64.b64decode(text, validate=True)
    if len(raw) != 2 * n:
        raise ValueError(f"odds hold {len(raw) // 2} values for {n} pairs")
    return [int.from_bytes(raw[2 * i:2 * i + 2], "little", signed=True) for i in range(n)]


def observe(reveals_dir: str) -> str:
    reveals, labels, by_head = [], [], {}
    entries = None
    for i in range(FIRST, LAST + 1):
        epoch_id = f"e{i:08d}"
        path = os.path.join(reveals_dir, f"{epoch_id}.json")
        with open(path, "rb") as f:
            raw = f.read()
        r = json.loads(raw)
        if r.get("epoch_id") != epoch_id:
            raise SystemExit(f"{path} is the reveal of {r.get('epoch_id')!r}")
        reveals.append({"epoch_id": epoch_id, "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw)})
        if entries is None:
            # The entries in their grace at the join: menu B's added ones.
            entries = list(r["odds_unscored_entries"])
        pool = r["exploration_pool"]
        odds = {h: _odds(v, len(pool)) for h, v in r.get("head_odds", {}).items()}
        for j, p in enumerate(pool):
            if p["model"] not in entries:
                continue
            idx = len(labels)
            labels.append([len(reveals) - 1, entries.index(p["model"]), int(p["y"]),
                           int(p["weight"])])
            for h, o in odds.items():
                if o[j] >= 0:
                    by_head.setdefault(h, []).append([idx, o[j]])
    doc = {
        "format": FORMAT,
        "source": REVEAL_SOURCE,
        "about": ("v3 menu B's seven entries from their join (e00026475) through e00026509: "
                  "every pooled pair of one of them as [reveal, entry, y, weight], and every "
                  "head's odds on those pairs as [label, odds]; heads numbered by hash order, "
                  "none named. Built by scripts/build_allowance_observations.py."),
        "reveals": reveals,
        "entries": entries,
        "labels": labels,
        "heads": [by_head[h] for h in sorted(by_head)],
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("observe", help="published v3 reveals -> the observations file")
    p.add_argument("reveals_dir")
    p.add_argument("--out", default=OUT_PATH)
    p.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = observe(args.reveals_dir)
    if args.check:
        with open(args.out, encoding="ascii") as f:
            if f.read() != text:
                print(f"{args.out} differs from the rebuild", file=sys.stderr)
                return 1
        print(f"{args.out} matches")
        return 0
    with open(args.out, "w", encoding="ascii") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(text)} bytes, sha256 "
          f"{hashlib.sha256(text.encode('ascii')).hexdigest()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
