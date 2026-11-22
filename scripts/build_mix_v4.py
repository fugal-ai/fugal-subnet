#!/usr/bin/env python3
"""Build the protocol v4 question mix (data/pool_mix_v4.json).

The mix is the share of the scored slice, and of exploration, that each
category and each question set gets in a v4 epoch
(docs/decisions/LIVE_MENU_V4.md, section 10). It is derived once, here, from
named published v3 reveals, and pinned. It is not adaptive at run time.

Nothing here touches the network. Every input is a file, and the same files
give the same mix byte for byte:

    python scripts/build_mix_v4.py build

reads the committed observations file
(data/observations/mix_v4_cells_e00026466-e00026489.json) and the v3 pool
manifest, and writes the reviewed copy, data/pool_mix_v4.json, and the copy
the package ships, fugal_subnet/routing_data/pool_mix_v4.json. `build
--check` rebuilds and compares with both copies instead of writing them.
`--slice` is the scored slice the cycle cap is computed for (default
config.SLICE_SIZE_V4).

THE OBSERVATIONS FILE is itself built by this script from the public
reveals, saved with a generic client and no key, so anyone can repeat it:

    for e in e00026466 ... e00026489; do
      curl -A reveal-snapshot/1.0 -o "reveals/reveal_$e.json" \\
        "https://storage.googleapis.com/fugal-reveals/70/$e/reveal.json"; done
    python scripts/build_mix_v4.py observe reveals --first e00026466 --last e00026489 \\
        --out data/observations/mix_v4_cells_e00026466-e00026489.json

It keeps each reveal's sha256 and, per question set, integer sums only.

WHAT IS COUNTED (the observe step):

- An exploration cell is one (question, entry) pair of one epoch: every
  hotkey's non-failed exploration rows for it (the reveal's `exploration`),
  answered k times correctly out of n. Its outcome is k / n.
- Only entries on the menu v4 starts on (the pinned seed,
  data/menu_v4_seed.json) are counted: p is the accuracy of an entry drawn
  uniformly from that menu.
- A cell's weight is its entry's published inverse-probability weight in
  that epoch (v3: the entry's `weight` in the reveal's `exploration_pool`,
  which in v3 depends on the entry and the epoch's menu alone). The weight
  is computed with the v3 functions (epoch_plan.explore_shares(3, the
  reveal's models)), because an entry's cells are not all pooled, and
  checked against every weight the reveal publishes.
- A cell's question set is the one the same reveal publishes for the
  question (its scored questions, its pooled pairs, its shared exploration
  set), else the one its id names. An `omnimath_` id names two sets
  (`omnimath_hard`, `omnimath_mid`); one the reveal does not place is left
  out, and counted.
- Per set and per n, the file keeps: cells, the sum of weights, and the sum
  of weight x k. The weighted outcome of the set is then
  sum over n of (sum of weight x k) / n, over the sum of weights, exactly.

THE RULE (the build step; section 10):

- A category's p is the weighted outcome over all its cells; its value is
  p(1 - p), the outcome variance of a pair.
- Targets are proportional to the categories' values; a category below
  MIX_FLOOR_PCT_V4 (5%) is set to it and the others share the rest in
  proportion, until none is below it; then each target is rounded to
  MIX_STEP_PCT_V4 (5%) by largest remainder (ties by category name). The
  targets are integer per-cents summing to 100.
- Inside a category, sets split by their own p(1 - p) x pool size; a set
  with fewer than MIX_MIN_PAIRS_V4 (100) cells takes its category's value.
  A set whose share would exceed pool size / (slice x MIX_CYCLE_EPOCHS_V4)
  (96: two 48-epoch windows) is held at that cap and the rest of its
  category's share goes to the others in proportion, until none exceeds
  it. A category that cannot place its share under the caps stops the
  build.
- Pinned as integer millionths of the slice (`weight_ppm`), summing within
  a category to its target exactly: each set's exact share rounded down,
  then the leftover units to the largest remainders (ties by set name),
  never above a set's cap. The slicer reads these as the within-category
  weights, so a set's expected share of the slice is its weight_ppm.
- No question leaves the pool for being answered by every entry, and no set
  leaves the draw through the mix: a build in which a set of the pool would
  get no share stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fugal_subnet import config  # noqa: E402  (the repo root is on the path from here)

OBSERVATIONS_FORMAT = "mix-v4-cells-v1"
MIX_FORMAT = "pool-mix-v4-v1"
REVEAL_SOURCE = "https://storage.googleapis.com/fugal-reveals/70/{epoch_id}/reveal.json"
OBSERVATIONS_PATH = os.path.join(ROOT, "data", "observations",
                                 "mix_v4_cells_e00026466-e00026489.json")
MIX_PATH = os.path.join(ROOT, "data", "pool_mix_v4.json")
PACKAGE_MIX_PATH = os.path.join(ROOT, "fugal_subnet", "routing_data", "pool_mix_v4.json")
MANIFEST_PATH = os.path.join(ROOT, "fugal_subnet", "benchmarks", "pool_manifest_v3.json")
SEED_PATH = os.path.join(ROOT, "data", "menu_v4_seed.json")

PPM = 1_000_000
FLOOR_PCT = config.MIX_FLOOR_PCT_V4
STEP_PCT = config.MIX_STEP_PCT_V4
MIN_PAIRS = config.MIX_MIN_PAIRS_V4
CYCLE_EPOCHS = config.MIX_CYCLE_EPOCHS_V4

# A question id's set when the reveal does not publish it: the loaders' id
# formats, longest prefix first. `omnimath_` is not here: it names two sets.
ID_PREFIXES = (
    ("mmlupro_", "mmlupro"), ("mmlu_", "mmlu"), ("supergpqa_", "supergpqa_hard"),
    ("arc_", "arc"), ("code_cc_", "code_contests"), ("code_lc_", "code_leetcode"),
    ("code_he_", "code_humanevalplus"), ("code_mbpp_", "code_mbppplus"),
    ("gsm8k_train_", "gsm8k_train"), ("math_train_", "math_l12"), ("math_", "math_l35"),
    ("rlvr_ifeval_", "rlvr_ifeval"), ("ifeval_", "ifeval"), ("aime25_", "aime25"),
    ("aime_", "aime"), ("beyondaime_", "beyondaime"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _epoch_number(epoch_id: str) -> int:
    """The number in a published reveal's epoch id, to order and range-check
    the input files. Ids are read, never formatted, here."""
    if len(epoch_id) < 2 or epoch_id[0] != "e" or not epoch_id[1:].isdigit():
        raise SystemExit(f"{epoch_id!r} is not an epoch id")
    return int(epoch_id[1:])


def _lines(head: dict, name: str, rows: list[str]) -> str:
    """A JSON object with `rows` (already encoded) one a line under `name`."""
    out = ["{"]
    for k in sorted(head):
        out.append(f"{json.dumps(k)}:{json.dumps(head[k], sort_keys=True, separators=(',', ':'))},")
    out.append(f"{json.dumps(name)}:[")
    out.append(",\n".join(rows))
    out.append("]")
    out.append("}")
    return "\n".join(out) + "\n"


def set_of_id(question_id: str) -> str | None:
    for prefix, name in ID_PREFIXES:
        if question_id.startswith(prefix):
            return name
    return None


# --- observe: published reveals -> integer sums ------------------------------------


def _entry_weights(reveal: dict) -> dict[str, int]:
    """{entry: its pair weight} in a v3 epoch, by the v3 functions, checked
    against every weight the reveal publishes."""
    from fugal_subnet import epoch_plan
    from fugal_subnet import explore_weights as xw

    weights = xw.weights(epoch_plan.explore_shares(3, list(reveal["models"])))
    for p in reveal["exploration_pool"]:
        if weights.get(p["model"]) != p["weight"]:
            raise SystemExit(f"{reveal['epoch_id']}: the published weight of {p['model']} "
                             f"({p['weight']}) is not the v3 rule's ({weights.get(p['model'])})")
    return weights


def observe(reveals_dir: str, first: str, last: str) -> str:
    """The observations file's text, from `reveal_<epoch id>.json` files of
    every epoch from `first` to `last`."""
    lo, hi = _epoch_number(first), _epoch_number(last)
    files = {}
    for name in sorted(os.listdir(reveals_dir)):
        if name.startswith("reveal_") and name.endswith(".json"):
            files[_epoch_number(name[len("reveal_"):-len(".json")])] = os.path.join(reveals_dir, name)
    missing = [n for n in range(lo, hi + 1) if n not in files]
    if missing or hi < lo:
        raise SystemExit(f"no reveal file for epoch number(s) {missing} in {reveals_dir}")
    seed_bytes = open(SEED_PATH, "rb").read()
    on_menu = {e["id"] for e in json.loads(seed_bytes)["entries"]}
    reveals = []
    sums: dict[str, dict[int, list[int]]] = {}
    left_out = {"failed_rows": 0, "off_menu_cells": 0, "unplaced_cells": 0}
    for n in range(lo, hi + 1):
        raw = open(files[n], "rb").read()
        reveal = json.loads(raw)
        epoch_id = reveal["epoch_id"]
        if _epoch_number(epoch_id) != n or reveal.get("protocol_version") != 3:
            raise SystemExit(f"{files[n]} is not the protocol v3 reveal of epoch number {n}")
        reveals.append({"epoch_id": epoch_id, "sha256": _sha256(raw), "bytes": len(raw)})
        weights = _entry_weights(reveal)
        published = {q["question_id"]: q["benchmark"] for q in reveal["questions"]}
        published.update((p["question_id"], p["benchmark"]) for p in reveal["exploration_pool"])
        published.update(reveal.get("explore_benchmarks") or {})
        cells: dict[tuple[str, str], list[int]] = {}
        for x in reveal["exploration"]:
            if x["failed"]:
                left_out["failed_rows"] += 1
                continue
            if not isinstance(x["correct"], bool):
                raise SystemExit(f"{epoch_id}: a row's `correct` is not a boolean")
            c = cells.setdefault((x["question_id"], x["model"]), [0, 0])
            c[0] += 1 if x["correct"] else 0
            c[1] += 1
        for (qid, entry), (k, rows) in sorted(cells.items()):
            if entry not in on_menu:
                left_out["off_menu_cells"] += 1
                continue
            name = published.get(qid) or set_of_id(qid)
            if name is None:
                left_out["unplaced_cells"] += 1
                continue
            if name not in config.BENCHMARK_CATEGORY_V3:
                raise SystemExit(f"{epoch_id}: {qid} is in {name!r}, which is not a v3 set")
            if entry not in weights:
                raise SystemExit(f"{epoch_id}: {entry} answered exploration but is not on "
                                 "the epoch's menu")
            s = sums.setdefault(name, {}).setdefault(rows, [0, 0, 0])
            s[0] += 1
            s[1] += weights[entry]
            s[2] += weights[entry] * k
    head = {"format": OBSERVATIONS_FORMAT, "source": REVEAL_SOURCE, "reveals": reveals,
            "menu": {"file": "data/menu_v4_seed.json", "sha256": _sha256(seed_bytes)},
            "left_out": left_out}
    rows = [json.dumps({"benchmark": b,
                        "by_rows": [[r, *sums[b][r]] for r in sorted(sums[b])]},
                       sort_keys=True, separators=(",", ":"))
            for b in sorted(sums)]
    return _lines(head, "sets", rows)


def read_observations(path: str) -> dict[str, list[list[int]]]:
    """{set: [[rows a cell, cells, sum of weights, sum of weight x correct]]}."""
    doc = json.load(open(path, encoding="utf-8"))
    if doc.get("format") != OBSERVATIONS_FORMAT:
        raise SystemExit(f"{path} is not a {OBSERVATIONS_FORMAT} file")
    out = {}
    for row in doc["sets"]:
        name = row["benchmark"]
        if name in out:
            raise SystemExit(f"{path}: set {name} appears twice")
        for r in row["by_rows"]:
            if (len(r) != 4 or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in r)
                    or r[0] < 1 or r[3] > r[0] * r[2]):
                raise SystemExit(f"{path}: set {name} holds a row that is not counts: {r}")
        out[name] = row["by_rows"]
    return out


# --- the rule -----------------------------------------------------------------------


class SetStat:
    """One set's (or one category's) exploration cells: the cell count and
    the exactly weighted outcome."""

    def __init__(self, by_rows: list[list[int]]):
        self.cells = sum(r[1] for r in by_rows)
        self.weight = sum(r[2] for r in by_rows)
        self.correct = sum((Fraction(r[3], r[0]) for r in by_rows), Fraction(0))

    @property
    def p(self) -> Fraction:
        if self.weight <= 0:
            raise SystemExit("a set or category with no weighted cells has no p")
        return self.correct / self.weight

    @property
    def value(self) -> Fraction:
        p = self.p
        return p * (1 - p)


def floored_targets(values: dict[str, Fraction], floor: Fraction) -> dict[str, Fraction]:
    """Shares proportional to `values`, every one at least `floor`: those
    below it are set to it and the rest share what is left in proportion,
    until none is below it. Exact."""
    cats = sorted(values)
    total = sum(values.values())
    if total <= 0:
        raise SystemExit("no category has outcome variance: the targets are undefined")
    t = {c: values[c] / total for c in cats}
    while any(t[c] < floor for c in cats):
        fixed = [c for c in cats if t[c] <= floor]
        free = [c for c in cats if t[c] > floor]
        rest = 1 - floor * len(fixed)
        s = sum(values[c] for c in free)
        if not free or rest <= 0 or s <= 0:
            raise SystemExit("the floor cannot be met for every category")
        t = {**{c: floor for c in fixed}, **{c: rest * values[c] / s for c in free}}
    return t


def rounded_targets(exact: dict[str, Fraction], step_pct: int, floor_pct: int) -> dict[str, int]:
    """Integer per-cents in steps of `step_pct`, summing to 100, by largest
    remainder on `exact` (shares summing to 1, each at least the floor);
    ties to the earlier category name."""
    units = 100 // step_pct
    scaled = {c: v * units for c, v in exact.items()}
    q = {c: int(v) for c, v in scaled.items()}           # floor; v >= 0
    if sum(exact.values()) != 1 or any(q[c] * step_pct < floor_pct for c in q):
        raise SystemExit("the exact targets must sum to 1 with every one at the floor")
    order = sorted(scaled, key=lambda c: (-(scaled[c] - q[c]), c))
    for c in order[:units - sum(q.values())]:
        q[c] += 1
    out = {c: q[c] * step_pct for c in sorted(q)}
    assert sum(out.values()) == 100
    return out


def cycle_cap(pool: int, slice_size: int) -> Fraction:
    """The largest share of the slice a set of `pool` questions may have:
    drawn through no faster than once in MIX_CYCLE_EPOCHS_V4 epochs."""
    return Fraction(pool, slice_size * CYCLE_EPOCHS)


def split_category(target: Fraction, weights: dict[str, Fraction], pools: dict[str, int],
                   slice_size: int) -> tuple[dict[str, Fraction], list[str]]:
    """({set: exact share of the slice}, sets held at the cap): `target`
    split in proportion to `weights`, no set above its cycle cap, what a
    capped set cannot take going to the others in proportion."""
    share: dict[str, Fraction] = {}
    capped: list[str] = []
    left, free = target, sorted(weights)
    while True:
        s = sum(weights[b] for b in free)
        if s <= 0:
            break
        over = [b for b in free if left * weights[b] / s > cycle_cap(pools[b], slice_size)]
        if not over:
            break
        for b in over:
            share[b] = cycle_cap(pools[b], slice_size)
            capped.append(b)
            left -= share[b]
        free = [b for b in free if b not in over]
    s = sum(weights[b] for b in free)
    if left > 0 and s <= 0:
        raise SystemExit(f"a category cannot place {left} of the slice under the cycle cap "
                         f"(sets {sorted(weights)})")
    for b in free:
        share[b] = left * weights[b] / s
    return share, sorted(capped)


def to_ppm(shares: dict[str, Fraction], total_ppm: int, pools: dict[str, int],
           slice_size: int) -> dict[str, int]:
    """Integer millionths of the slice summing to `total_ppm`: each share
    rounded down, the leftover units to the largest remainders (ties to the
    earlier set name), never above a set's cycle cap."""
    exact = {b: v * PPM for b, v in shares.items()}
    out = {b: int(v) for b, v in exact.items()}
    left = total_ppm - sum(out.values())
    order = sorted(exact, key=lambda b: (-(exact[b] - out[b]), b))
    while left > 0:
        # A set held at its cap has no room for a unit; the walk goes round
        # again over the sets that have, in the same order.
        room = [b for b in order if exact[b] > 0
                and Fraction(out[b] + 1, PPM) <= cycle_cap(pools[b], slice_size)]
        if not room:
            raise SystemExit(f"the category's {total_ppm} millionths cannot be placed under "
                             "the caps")
        for b in room[:left]:
            out[b] += 1
        left -= len(room[:left])
    if left < 0:
        raise SystemExit("the shares add up to more than the category's target")
    return out


def derive(observations: dict[str, list[list[int]]], pools: dict[str, int],
           categories: dict[str, str], slice_size: int) -> dict:
    """The mix and its trace from the observations, the sets' pool sizes and
    their categories."""
    if slice_size <= 0:
        raise SystemExit("the slice size must be positive")
    unknown = sorted(set(observations) - set(categories))
    if unknown:
        raise SystemExit(f"observations for sets with no category: {unknown}")
    sets = sorted(b for b in categories if pools.get(b, 0) > 0)
    cats = sorted({categories[b] for b in sets})
    by_set = {b: SetStat(observations.get(b, [])) for b in sets}
    by_cat = {c: SetStat([r for b in sets if categories[b] == c for r in observations.get(b, [])])
              for c in cats}
    exact = floored_targets({c: by_cat[c].value for c in cats}, Fraction(FLOOR_PCT, 100))
    targets = rounded_targets(exact, STEP_PCT, FLOOR_PCT)
    weight_ppm: dict[str, int] = {}
    trace = {"categories": {}, "sets": {}}
    for c in cats:
        members = [b for b in sets if categories[b] == c]
        own = {b: by_set[b].cells >= MIN_PAIRS for b in members}
        value = {b: by_set[b].value if own[b] else by_cat[c].value for b in members}
        weights = {b: value[b] * pools[b] for b in members}
        if sum(weights.values()) <= 0:
            weights = {b: Fraction(pools[b]) for b in members}
        share, capped = split_category(Fraction(targets[c], 100), weights, pools, slice_size)
        weight_ppm.update(to_ppm(share, targets[c] * PPM // 100, pools, slice_size))
        trace["categories"][c] = {"cells": by_cat[c].cells, "p": by_cat[c].p,
                                  "value": by_cat[c].value, "exact": exact[c],
                                  "target_pct": targets[c], "capped": capped}
        for b in members:
            trace["sets"][b] = {"cells": by_set[b].cells,
                                "p": by_set[b].p if by_set[b].weight else None,
                                "value_from": "own" if own[b] else "category",
                                "value": value[b], "share": share[b],
                                "cap": cycle_cap(pools[b], slice_size), "capped": b in capped}
    # No question leaves the pool through the mix either: a set every
    # observed entry answered right (p(1 - p) = 0 on its own cells) would get
    # no share and never be drawn. That is the owner's call, not this rule's.
    empty = sorted(b for b in sets if weight_ppm[b] <= 0)
    if empty:
        raise SystemExit(f"sets {empty} would get no share of the slice: no question leaves "
                         "the pool through the mix")
    return {"targets": targets, "weight_ppm": weight_ppm, "sets": sets, "trace": trace}


# --- build -------------------------------------------------------------------------


def pool_sizes(manifest_path: str = MANIFEST_PATH) -> tuple[dict[str, int], dict[str, str], str]:
    """({set: questions}, {set: category}, the manifest's sha256) of the pinned
    v3 pool, which is v4's pool: no set is added in this release."""
    raw = open(manifest_path, "rb").read()
    per = json.loads(raw)["per_benchmark"]
    sizes = {b: int(v["n"]) for b, v in per.items()}
    cats = {b: v["category"] for b, v in per.items()}
    if cats != {b: config.BENCHMARK_CATEGORY_V3[b] for b in cats} or set(cats) != set(
            config.BENCHMARK_CATEGORY_V3):
        raise SystemExit("the v3 pool manifest's sets and categories are not config's")
    return sizes, cats, _sha256(raw)


def build(observations_path: str = OBSERVATIONS_PATH, slice_size: int | None = None,
          manifest_path: str = MANIFEST_PATH) -> tuple[str, list[str]]:
    """(the mix file's text, notes for the reviewer)."""
    slice_size = config.SLICE_SIZE_V4 if slice_size is None else int(slice_size)
    observed = read_observations(observations_path)
    pools, categories, manifest_sha = pool_sizes(manifest_path)
    mix = derive(observed, pools, categories, slice_size)
    obs_raw = open(observations_path, "rb").read()
    doc = {
        "format": MIX_FORMAT,
        "slice_size": slice_size,
        "cycle_epochs": CYCLE_EPOCHS,
        "category_targets_pct": mix["targets"],
        "benchmarks": {b: {"category": categories[b], "pool": pools[b],
                           "weight_ppm": mix["weight_ppm"][b]} for b in mix["sets"]},
        "inputs": {
            "observations": {"file": os.path.relpath(observations_path, ROOT)
                             if observations_path.startswith(ROOT) else
                             os.path.basename(observations_path),
                             "sha256": _sha256(obs_raw)},
            "pool_manifest": {"file": "fugal_subnet/benchmarks/pool_manifest_v3.json",
                              "sha256": manifest_sha},
        },
    }
    text = json.dumps(doc, sort_keys=True, indent=1) + "\n"
    notes = []
    for c, t in mix["trace"]["categories"].items():
        notes.append(f"{c}: {t['cells']} cells, p {float(t['p']):.5f}, p(1-p) "
                     f"{float(t['value']):.5f}, exact {float(t['exact']) * 100:.3f}% -> "
                     f"{t['target_pct']}%" + (f"; capped {t['capped']}" if t['capped'] else ""))
    for b, t in mix["trace"]["sets"].items():
        notes.append(f"  {b}: {t['cells']} cells, p "
                     + ("-" if t["p"] is None else f"{float(t['p']):.5f}")
                     + f", value {float(t['value']):.5f} ({t['value_from']}), pool "
                     f"{pools[b]}, share {float(t['share']) * 100:.3f}% "
                     f"({float(t['share']) * slice_size:.2f} a slice), cap "
                     f"{float(t['cap']) * 100:.3f}%" + (" CAPPED" if t["capped"] else "")
                     + f", {mix['weight_ppm'][b]} ppm")
    return text, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("observe", help="published v3 reveals -> the observations file")
    p.add_argument("reveals_dir")
    p.add_argument("--first", required=True)
    p.add_argument("--last", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build", help="the observations file -> the mix")
    p.add_argument("--observations", default=OBSERVATIONS_PATH)
    p.add_argument("--slice", type=int, default=None,
                   help=f"the scored slice size (default {config.SLICE_SIZE_V4})")
    p.add_argument("--out", default=None,
                   help="write (or check) this one file instead of the data/ copy and the "
                        "package copy")
    p.add_argument("--check", action="store_true",
                   help="compare with the file(s) instead of writing")
    args = parser.parse_args(argv)

    if args.command == "observe":
        text = observe(args.reveals_dir, args.first, args.last)
        outs = [args.out]
    else:
        text, notes = build(args.observations, args.slice)
        for line in notes:
            print(line)
        outs = [MIX_PATH, PACKAGE_MIX_PATH] if args.out is None else [args.out]
    data = text.encode("utf-8")
    for out in outs:
        shown = os.path.relpath(out, ROOT) if out.startswith(ROOT) else out
        if getattr(args, "check", False):
            if not os.path.exists(out) or open(out, "rb").read() != data:
                raise SystemExit(f"{shown} is not what these inputs build")
            print(f"{shown} rebuilds byte for byte (sha256 {_sha256(data)})")
            continue
        with open(out, "wb") as f:
            f.write(data)
        print(f"wrote {shown} ({len(data)} bytes, sha256 {_sha256(data)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
