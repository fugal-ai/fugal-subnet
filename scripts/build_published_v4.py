#!/usr/bin/env python3
"""Build protocol v4's published-set seed (data/published_v4.json).

The memorisation check (docs/decisions/SEAT_MINIMAL_V4.md, section 6;
fugal_subnet/memo_v4.py) needs, per pool question, whether a reveal has
published an outcome on it. At the v4 activation that state is seeded from
this file, derived here from named published reveals and pinned; after it,
every node folds each epoch's reveal (memo_v4.fold_reveal).

Nothing in `build` touches the network. Every input is a file, and the same
file gives the same seed byte for byte:

    python scripts/build_published_v4.py build

reads the committed observations file
(data/observations/published_v4_e00026189-e00026509.json) and writes the
reviewed copy, data/published_v4.json, and the copy the package ships,
fugal_subnet/routing_data/published_v4.json. `build --check` rebuilds and
compares with both copies instead of writing them.

THE OBSERVATIONS FILE is built by this script from the public reveals,
saved with a generic client and no key, and the pinned v3 pool (v4's pool),
so anyone can repeat it:

    for e in e00026189 ... e00026509; do
      curl -A reveal-snapshot/1.0 -o "reveals/$e.json" \\
        "https://storage.googleapis.com/fugal-reveals/70/$e/reveal.json"; done
    python scripts/build_published_v4.py observe reveals --first e00026189 \\
        --last e00026509 --out data/observations/published_v4_e00026189-e00026509.json

e00026189 is the first reveal the public store holds for netuid 70 (v1),
e00026349 the first of v2, e00026466 the first of v3. `observe` loads the
v3 pool through the loader (its pool hash is checked against the pinned
manifest), or reads `--pool-ids`, a JSON list of [question id, set] pairs
made from it. It keeps, per reveal, its sha256 and size, its protocol, how
many question ids it lists and how many pool questions it publishes first;
per question set its size and how many of it were published; and the ids of
the pool's questions no named reveal published, by set.

WHAT A REVEAL PUBLISHES is memo_v4.published_ids, the rule the v4 fold uses:
its scored questions, every exploration answer it lists, and its pooled
labels. A v1 reveal publishes outcomes for its scored questions (`matrix`)
and its exploration answers; v1's pool shares its AIME, IFEval, MMLU and
MATH (levels 3-5) ids with the v3 pool, so v1 reveals are read too.

THE SEED (the build step): the observations' unpublished ids by set, the
last named reveal (`through`), the pool they were observed on, and the
sha256 of the observations file. Every pool question of a named set that is
not listed is published as of `through` (memo_v4.seed_published); a set the
seed does not name entered the pool after it and is wholly unpublished.

THE MEMORISATION OBSERVATIONS (`labels`, for tests/test_memo_v4.py): from
the 44 v3 reveals e00026466..e00026509, each epoch's pooled labels (question,
entry, outcome, weight, the coldkeys whose answers it pools), every head's
published odds on them, each head's owner and first commit block, and for
each labelled question the first named reveal that published it (reading
every named reveal from --since):

    python scripts/build_published_v4.py labels reveals --since e00026189 \\
        --first e00026466 --last e00026509 \\
        --out data/observations/memo_v4_labels_e00026466-e00026509.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OBSERVATIONS_FORMAT = "published-v4-observations-v1"
LABELS_FORMAT = "memo-v4-labels-v1"
REVEAL_SOURCE = "https://storage.googleapis.com/fugal-reveals/70/{epoch_id}/reveal.json"
OBSERVATIONS_PATH = os.path.join(ROOT, "data", "observations",
                                 "published_v4_e00026189-e00026509.json")
LABELS_PATH = os.path.join(ROOT, "data", "observations",
                           "memo_v4_labels_e00026466-e00026509.json")
SEED_PATH = os.path.join(ROOT, "data", "published_v4.json")
PACKAGE_SEED_PATH = os.path.join(ROOT, "fugal_subnet", "routing_data", "published_v4.json")
MANIFEST_PATH = os.path.join(ROOT, "fugal_subnet", "benchmarks", "pool_manifest_v3.json")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _epoch_number(epoch_id: str) -> int:
    from fugal_subnet.memo_v4 import epoch_index

    try:
        return epoch_index(epoch_id)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def _reveal_files(reveals_dir: str, first: str, last: str) -> list[tuple[str, str]]:
    """[(epoch id, path)] for every epoch from `first` to `last`, named
    `<epoch id>.json` or `reveal_<epoch id>.json`; a missing one stops."""
    lo, hi = _epoch_number(first), _epoch_number(last)
    if hi < lo:
        raise SystemExit("--last is before --first")
    found: dict[int, tuple[str, str]] = {}
    for name in sorted(os.listdir(reveals_dir)):
        stem = name[len("reveal_"):] if name.startswith("reveal_") else name
        if not stem.endswith(".json"):
            continue
        eid = stem[:-len(".json")]
        try:
            n = _epoch_number(eid)
        except SystemExit:
            continue
        if lo <= n <= hi:
            if n in found:
                raise SystemExit(f"two files for {eid}")
            found[n] = (eid, os.path.join(reveals_dir, name))
    missing = [n for n in range(lo, hi + 1) if n not in found]
    if missing:
        raise SystemExit(f"no reveal file for {len(missing)} epochs, first {missing[0]}")
    return [found[n] for n in range(lo, hi + 1)]


def _read_reveal(eid: str, path: str) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        data = f.read()
    reveal = json.loads(data)
    if not isinstance(reveal, dict) or reveal.get("epoch_id") != eid:
        raise SystemExit(f"{path} is not the reveal of {eid}")
    return reveal, data


def _pool_ids(pool_ids: str | None) -> dict[str, str]:
    """{question id: set} of the v3 pool, checked against the pinned manifest."""
    from fugal_subnet.benchmarks import loader

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    if pool_ids:
        with open(pool_ids, encoding="utf-8") as f:
            pairs = json.load(f)
        pool = [{"question_id": q, "benchmark": b} for q, b in pairs]
    else:
        pool = loader.load_pool(3)
    if loader.pool_hash(pool) != manifest["pool_hash"]:
        raise SystemExit("the pool's ids are not the pinned v3 pool's (pool_hash)")
    return {q["question_id"]: q["benchmark"] for q in pool}


def _lines(head: dict, name: str, rows: list[str], close: str = "]") -> str:
    """A JSON object with `rows` (already encoded) one a line under `name`."""
    out = ["{"]
    for k in sorted(head):
        out.append(f"{json.dumps(k)}:{json.dumps(head[k], sort_keys=True, separators=(',', ':'))},")
    out.append(f"{json.dumps(name)}:{'[' if close == ']' else '{'}")
    out.append(",\n".join(rows))
    out.append(close)
    out.append("}")
    return "\n".join(out) + "\n"


# --- observe: reveals + the pool -> the observations file -----------------------------------


def observe(reveals_dir: str, first: str, last: str, pool_ids: str | None) -> str:
    from fugal_subnet.memo_v4 import published_ids

    pool = _pool_ids(pool_ids)
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    published: set[str] = set()
    rows = []
    for eid, path in _reveal_files(reveals_dir, first, last):
        reveal, data = _read_reveal(eid, path)
        ids = published_ids(reveal)
        new = sorted(q for q in ids if q in pool and q not in published)
        published.update(new)
        rows.append(json.dumps({"epoch_id": eid, "sha256": _sha256(data), "bytes": len(data),
                                "protocol": int(reveal.get("protocol_version") or 1),
                                "listed": len(ids), "new": len(new)},
                               sort_keys=True, separators=(",", ":")))
        del reveal, data
    per_set: dict[str, list[int]] = {}
    unpublished: dict[str, list[str]] = {}
    for q, s in pool.items():
        row = per_set.setdefault(s, [0, 0])
        row[0] += 1
        if q in published:
            row[1] += 1
        else:
            unpublished.setdefault(s, []).append(q)
    head = {
        "format": OBSERVATIONS_FORMAT,
        "source": REVEAL_SOURCE,
        "first": first,
        "last": last,
        "pool": {"pool_hash": manifest["pool_hash"], "n_questions": len(pool),
                 "per_set": {s: {"pool": n, "published": k} for s, (n, k) in sorted(per_set.items())}},
        "unpublished": {s: sorted(v) for s, v in sorted(unpublished.items())},
    }
    return _lines(head, "reveals", rows)


# --- build: the observations file -> the seed -------------------------------------------------


def build(observations: str) -> str:
    """The seed's text from the observations file, after checking that it
    adds up: per set, its pool size minus its published count is the number
    of its unpublished ids, and the sizes are the pinned manifest's."""
    from fugal_subnet.memo_v4 import SEED_FORMAT, check_seed

    with open(observations, "rb") as f:
        data = f.read()
    obs = json.loads(data)
    if obs.get("format") != OBSERVATIONS_FORMAT:
        raise SystemExit(f"{observations} is not a {OBSERVATIONS_FORMAT} file")
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    per_set = obs["pool"]["per_set"]
    if ({s: r["pool"] for s, r in per_set.items()}
            != {s: r["n"] for s, r in manifest["per_benchmark"].items()}
            or obs["pool"]["pool_hash"] != manifest["pool_hash"]
            or obs["pool"]["n_questions"] != manifest["n_questions"]):
        raise SystemExit("the observations were not made on the pinned v3 pool")
    for s, r in per_set.items():
        if r["pool"] - r["published"] != len(obs["unpublished"].get(s, [])):
            raise SystemExit(f"set {s}: its counts and its unpublished ids disagree")
    reveals = obs["reveals"]
    numbers = [_epoch_number(r["epoch_id"]) for r in reveals]
    if (numbers != list(range(_epoch_number(obs["first"]), _epoch_number(obs["last"]) + 1))
            or sum(r["new"] for r in reveals) != sum(r["published"] for r in per_set.values())):
        raise SystemExit("the observations do not cover every named reveal once, or their counts "
                         "do not add up")
    doc = {
        "format": SEED_FORMAT,
        "through": obs["last"],
        "pool": {"pool_hash": obs["pool"]["pool_hash"], "n_questions": obs["pool"]["n_questions"],
                 "per_set": {s: r["pool"] for s, r in sorted(per_set.items())}},
        "observations_sha256": _sha256(data),
        "unpublished": {s: obs["unpublished"][s] for s in sorted(obs["unpublished"])},
    }
    check_seed(doc)
    rows = [f"{json.dumps(s)}:{json.dumps(ids, separators=(',', ':'))}"
            for s, ids in doc["unpublished"].items()]
    head = {k: v for k, v in doc.items() if k != "unpublished"}
    return _lines(head, "unpublished", rows, close="}")


# --- labels: the v3 reveals -> the memorisation observations ----------------------------------


def labels(reveals_dir: str, since: str, first: str, last: str) -> str:
    from fugal_subnet.memo_v4 import published_ids

    first_seen: dict[str, int] = {}
    reveal_rows = []
    epochs: dict[int, dict] = {}
    heads: dict[str, dict] = {}
    lo = _epoch_number(first)
    for eid, path in _reveal_files(reveals_dir, since, last):
        reveal, data = _read_reveal(eid, path)
        e = _epoch_number(eid)
        for q in sorted(published_ids(reveal)):
            first_seen.setdefault(q, e)
        if e < lo:
            del reveal, data
            continue
        reveal_rows.append({"epoch_id": eid, "sha256": _sha256(data), "bytes": len(data)})
        voters: dict[tuple[str, str], set[str]] = {}
        for x in reveal.get("exploration") or []:
            if not x.get("failed"):
                voters.setdefault((x["question_id"], x["model"]), set()).add(x["coldkey"])
        pairs = []
        for p in reveal["exploration_pool"]:
            key = (p["question_id"], p["model"])
            if key not in voters:
                raise SystemExit(f"{eid}: pooled pair {key} has no answer in `exploration`")
            pairs.append([p["question_id"], p["benchmark"], p["model"], int(p["y"]),
                          int(p["weight"]), sorted(voters[key])])
        cr = reveal["v2_state"]["copy_rule"]
        for h in reveal.get("head_odds") or {}:
            heads.setdefault(h, {"owner": cr["ownership"][h],
                                 "first_block": int(cr["first_commit"][h])})
        epochs[e] = {"pairs": pairs, "odds": dict(sorted((reveal.get("head_odds") or {}).items()))}
        del reveal, data
    questions = sorted({p[0] for ep in epochs.values() for p in ep["pairs"]})
    q_index = {q: i for i, q in enumerate(questions)}
    models = sorted({p[2] for ep in epochs.values() for p in ep["pairs"]})
    m_index = {m: i for i, m in enumerate(models)}
    coldkeys = sorted({c for ep in epochs.values() for p in ep["pairs"] for c in p[5]}
                      | {h["owner"] for h in heads.values()})
    c_index = {c: i for i, c in enumerate(coldkeys)}
    bench = {}
    for ep in epochs.values():
        for p in ep["pairs"]:
            if bench.setdefault(p[0], p[1]) != p[1]:
                raise SystemExit(f"{p[0]} is placed in two sets")
    rows = []
    for e in sorted(epochs):
        ep = epochs[e]
        rows.append(json.dumps([e, [[q_index[p[0]], m_index[p[2]], p[3], p[4],
                                     [c_index[c] for c in p[5]]] for p in ep["pairs"]],
                                ep["odds"]], separators=(",", ":")))
    sets = sorted(set(bench.values()))
    s_index = {s: i for i, s in enumerate(sets)}
    head = {
        "format": LABELS_FORMAT,
        "source": REVEAL_SOURCE,
        "since": since,
        "reveals": reveal_rows,
        # Each labelled question as [its set, the first named reveal that
        # published it], in question-id order; the ids themselves are
        # sha256(json of the sorted id list), recomputable from the reveals.
        "sets": sets,
        "questions": [[s_index[bench[q]], first_seen[q]] for q in questions],
        "questions_sha256": _sha256(json.dumps(questions, separators=(",", ":")).encode()),
        "models": models,
        "coldkeys": coldkeys,
        "heads": {h: {"owner": c_index[v["owner"]], "first_block": v["first_block"]}
                  for h, v in sorted(heads.items())},
    }
    return _lines(head, "epochs", rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("observe", help="named reveals and the v3 pool -> the observations file")
    p.add_argument("reveals_dir")
    p.add_argument("--first", required=True)
    p.add_argument("--last", required=True)
    p.add_argument("--pool-ids", default=None,
                   help="a JSON list of [question id, set] of the v3 pool, instead of loading it")
    p.add_argument("--out", required=True)
    p = sub.add_parser("build", help="the observations file -> the seed")
    p.add_argument("--observations", default=OBSERVATIONS_PATH)
    p.add_argument("--out", default=None,
                   help="write (or check) this one file instead of the data/ copy and the "
                        "package copy")
    p.add_argument("--check", action="store_true",
                   help="compare with the file(s) instead of writing")
    p = sub.add_parser("labels", help="the v3 reveals -> the memorisation observations")
    p.add_argument("reveals_dir")
    p.add_argument("--since", required=True)
    p.add_argument("--first", required=True)
    p.add_argument("--last", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.command == "observe":
        text = observe(args.reveals_dir, args.first, args.last, args.pool_ids)
        outs = [args.out]
    elif args.command == "labels":
        text = labels(args.reveals_dir, args.since, args.first, args.last)
        outs = [args.out]
    else:
        text = build(args.observations)
        outs = [SEED_PATH, PACKAGE_SEED_PATH] if args.out is None else [args.out]
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
