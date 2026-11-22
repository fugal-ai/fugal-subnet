#!/usr/bin/env python3
"""Protocol v3 behaves identically on two trees: a scripted scenario, line by line.

The v4 code is built on the same branch as the live v3 rules, so a change
there can move a v3 epoch without anyone meaning it to. This script runs one
scripted scenario on each of two trees, each tree in its own interpreter
with only its own code and its own test helpers on the path, and compares
the results line by line. Each line is `<tag> <value>`, the value a short
hash of what the tree computed (or, for the commitment and namespace lines,
a small JSON verdict):

- v2: forty epochs of the paired net-value seat with takes, a miss, a late
  head and failed calls; every reveal, seat, event and weight vector; the
  rebuild from the reveals and the saved window's bytes;
- v3: odds-seat epochs with every miner serving and with gaps, the same with
  their rebuild and saved window bytes; menu A to menu B with its grace;
- proofs: a v3 proof parsed, hashed and serialised as is, with stray v4 keys
  in the v3 namespace, and in a namespace that is no v1-v3 one with
  malformed v4 fields;
- commitments: three fields, four fields (a menu hash token), and a fourth
  field that is not a token;
- the consensus digest a v1-v3 epoch publishes (taken in this environment,
  so it must be equal when both trees run in the same one);
- with `--reveals DIR`: every published reveal under DIR (a directory per
  epoch, e000NNNNN/reveal.json, or a flat directory of e000NNNNN.json)
  through consensus_view, comparable_reveal and pairs_from_reveal, and the
  seat and window rebuilt from them all. A DIR with no reveal in either
  layout is refused (exit 2) before anything runs.

Exactly two differences are intended, and each is checked to have its exact
shape on each side, not merely tolerated:

1. `commit/four_field`: a four-field commitment (`reveal:<epoch>:<sha256>:
   <menu hash token>`) parses on a tree with protocol v4 (as the epoch and
   sha256, and as the commitment with its menu hash) and does not parse on
   a tree without it.
2. `proof/v4_namespace`: a proof in a namespace that is no v1-v3 one, with
   malformed v4 fields, cannot be parsed on a tree with protocol v4 (its v4
   fields are read and refused); a tree without v4 parses it (and the
   namespace check refuses it later).

Any other difference fails the run (exit 1).

    python scripts/check_v3_equivalence.py --ref origin/main          # this checkout vs a ref
    python scripts/check_v3_equivalence.py --tree A --tree B          # two trees on disk
    python scripts/check_v3_equivalence.py --ref origin/main --reveals DIR

The first tree is the one under test (with v4), the second the reference
(without). `--ref` materialises the ref with `git archive` into a temporary
directory. No network, no chain, no spend.

Not a CI step: CI checks out one shallow commit (no `origin/main`), and once
v4 is on `main` the reference has v4 too, so the two intended differences
are no longer differences. Run it by hand before merging v4 work, against the
deployed release's tag (`--ref <tag>`), with `--reveals` over a directory of
published reveals (either layout). The release checklist names it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# A menu hash, and (`_token`) the 43 characters of unpadded base64url a
# four-field commitment carries it in, as reveal_publisher.encode_menu_hash
# writes it (computed here so a tree without that function can be asked too).
_MENU_HASH = "f1d6b6e3b8f1d2c88b5e8a2a90a0e2b5a7c9f0e3d1b2c4a6e8f0a1b3c5d7e9f1"


def _token(menu_hash: str) -> str:
    import base64

    return base64.urlsafe_b64encode(bytes.fromhex(menu_hash)).rstrip(b"=").decode()


# ── the scenario, run inside one tree ───────────────────────────────────────


def _scenario(tree: str, reveals: str | None) -> list[tuple[str, str]]:
    """Every (tag, value) line of the scenario, computed with `tree`'s code."""
    import dataclasses
    import hashlib
    import pathlib

    sys.path.insert(0, tree)
    os.chdir(tree)
    import fugal_subnet
    import tests

    for mod in (fugal_subnet, tests):
        if not os.path.abspath(mod.__file__).startswith(os.path.abspath(tree) + os.sep):
            raise SystemExit(f"{mod.__name__} was imported from {mod.__file__}, not from {tree}")
    lines: list[tuple[str, str]] = []

    def blob_hash(obj) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def feed(tag: str, obj) -> None:
        lines.append((tag, blob_hash(obj)))

    def feed_bytes(tag: str, data: bytes) -> None:
        lines.append((tag, hashlib.sha256(data).hexdigest()[:16]))

    # 1. v2: the 40-epoch scenario of the v2 equivalence script.
    import tests.test_v2_epoch as T2
    from fugal_subnet.v2_epoch import save_window, seat_params_from_config, seat_to_dict
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals
    from tests.test_v2_rebuild import reveal_of, write

    params = dataclasses.replace(
        seat_params_from_config(300 * 5, None, protocol_version=2, odds_seat_block=None),
        min_paired_questions=60)

    def policies(e):
        p = {7: T2.always("m/mid")}
        if e >= 3:
            p[5] = T2.oracle if e % 7 else T2.always("m/strong")
        if e >= 2:
            p[6] = T2.always("m/cheap") if e < 20 else T2.oracle
        if 15 <= e < 17:
            p.pop(5, None)
        if e >= 25:
            p[8] = T2.oracle
        return p

    window = seat = copy = None
    reveals_v2 = []
    for e in range(40):
        r = T2.run(e, policies(e), window=window, seat=seat, copy=copy, params=params,
                   failed={7: (1, 2, 3)} if e % 5 == 0 else None)
        window, seat, copy = r.window, r.decision.state, r.copy_state
        rv = json.loads(json.dumps(reveal_of(e, r, policies(e))))
        reveals_v2.append(rv)
        feed(f"v2/{e:02d}", {"rv": rv, "seat": seat_to_dict(seat), "event": r.decision.event,
                             "detail": r.decision.detail, "uids": list(map(int, r.uids)),
                             "w": [float(x) for x in r.weights]})
    d = pathlib.Path(tempfile.mkdtemp(prefix="v3equiv-"))
    write(d, reveals_v2)
    rb = rebuild_v2_from_reveals(d, window_size=48, prices=T2.PRICES, bench_category=T2.CATEGORY)
    feed("v2/rebuild_seat", seat_to_dict(rb.seat))
    save_window(str(d / "w.npz"), window)
    feed_bytes("v2/window_bytes", (d / "w.npz").read_bytes())

    # 2. v3: odds-seat epochs, every miner serving and with gaps, and their rebuild.
    import tests.test_v3_epoch as T3

    for name, schedule in (("all", lambda e: T3.UIDS),
                           ("gaps", lambda e: T3.UIDS if e % 4 else (5, 6))):
        results, reveals_v3 = T3.live(26, schedule=schedule)
        for e, (r, rv) in enumerate(zip(results, reveals_v3)):
            feed(f"v3/{name}/{e:02d}", {"rv": json.loads(json.dumps(rv)),
                                        "seat": seat_to_dict(r.decision.state),
                                        "event": r.decision.event, "detail": r.decision.detail,
                                        "w": [float(x) for x in r.weights]})
        d = pathlib.Path(tempfile.mkdtemp(prefix="v3equiv-"))
        write(d, reveals_v3)
        rb = rebuild_v2_from_reveals(d, window_size=48, prices=T3.PRICES,
                                     bench_category=T3.CATEGORY)
        feed(f"v3/{name}/rebuild_seat", seat_to_dict(rb.seat))
        save_window(str(d / "w.npz"), results[-1].window)
        feed_bytes(f"v3/{name}/window_bytes", (d / "w.npz").read_bytes())

    # 3. v3 menu A to menu B, with its grace.
    import tests.test_v3_menu_b as TB

    results, reveals_b = TB.live(24)
    for e, (r, rv) in enumerate(zip(results, reveals_b)):
        feed(f"menu_b/{e:02d}", {"rv": json.loads(json.dumps(rv)),
                                 "seat": seat_to_dict(r.decision.state),
                                 "event": r.decision.event, "w": [float(x) for x in r.weights]})

    # 4. v3 proofs: as is, with stray v4 keys, and in a namespace no v1-v3 one is.
    from fugal_subnet.routing_protocol import identity_v3
    from fugal_subnet.tee.proof import BenchmarkProof

    base = {"epoch_id": "e00026500", "nonce": "ab" * 32, "questions_hash": "q" * 64,
            "weights_hash": "w" * 64, "source_hash": "s" * 64, "total_cost_usd": 0.5,
            "per_model_costs": {"m/a": 0.5}, "attestation_quote": "", "timestamp": 1.5,
            "hotkey": "5Fhk", "routing_protocol": identity_v3(),
            "results": [{"question_id": f"q{i}", "routed_model": "m/a", "correct": i % 2 == 0,
                         "cost_usd": 0.01, "response_hash": "r" * 64, "prompt_tokens": 10,
                         "completion_tokens": 20, "is_exploration": i > 3, "failed": i == 5,
                         "latency_ms": 7} for i in range(8)]}
    variants = {
        "proof/v3": base,
        "proof/v3_with_v4_keys": dict(
            base, menu_hash="zz", endpoint_reads="junk", proposal=[1],
            results=[dict(r, provider_cost="x", served_by=7, measure_for="no")
                     for r in base["results"]]),
        "proof/v4_namespace": dict(
            base, routing_protocol="not-a-v1-v3-namespace",
            results=[dict(r, provider_cost=-1) for r in base["results"]]),
    }
    for tag, doc in variants.items():
        try:
            p = BenchmarkProof.from_dict(json.loads(json.dumps(doc)))
            verdict = {"parses": True, "hash": blob_hash({"hash": p.content_hash(),
                                                          "dict": p.to_dict()})}
        except Exception as e:  # noqa: BLE001 - the verdict is the line
            verdict = {"parses": False, "error": type(e).__name__}
        lines.append((tag, json.dumps(verdict, sort_keys=True, separators=(",", ":"))))

    # 5. commitments.
    from fugal_subnet import reveal_sync
    from fugal_subnet.reveal_publisher import commitment_for

    three = commitment_for("e00026500", "a" * 64)
    four = f"{three}:{_token(_MENU_HASH)}"
    commitment_of = getattr(reveal_sync, "parse_commitment", None)

    def read(data):
        got = {"pointer": reveal_sync.parse_pointer(data)}
        if commitment_of is not None:
            got["commitment"] = commitment_of(data)
        return got

    lines.append(("commit/three_field", json.dumps(
        {"text": three, "pointer": reveal_sync.parse_pointer(three)}, sort_keys=True,
        separators=(",", ":"))))
    lines.append(("commit/four_field", json.dumps(read(four), sort_keys=True,
                                                  separators=(",", ":"))))
    lines.append(("commit/bad_fourth_field", json.dumps(
        {"pointer": reveal_sync.parse_pointer(three + ":xyz")}, sort_keys=True,
        separators=(",", ":"))))

    # 6. the digest a v1-v3 epoch publishes, in this environment.
    from fugal_subnet.fingerprint import consensus_digest

    lines.append(("digest/v1_v3", consensus_digest()))

    # 7. published reveals, when given.
    if reveals:
        lines += _published(reveals, blob_hash, feed_bytes)
    return lines


def reveal_paths(src: str) -> list[tuple[str, str]]:
    """[(epoch id, path)] of every reveal under `src`, by epoch, in either
    layout in use: a directory per epoch (`e000NNNNN/reveal.json`, as a
    validator writes them) or a flat directory of `e000NNNNN.json` files (as
    published reveals are downloaded). Other files (a MANIFEST.json) are not
    reveals. Exits with a message when `src` cannot be listed or one epoch is
    there in both layouts."""
    import re

    root = os.path.abspath(src)
    named = re.compile(r"e(\d+)")
    try:
        names = sorted(os.listdir(root))
    except OSError as e:
        raise SystemExit(f"--reveals {src}: cannot list it ({e})") from None
    found: dict[str, str] = {}
    for name in names:
        path = os.path.join(root, name)
        if named.fullmatch(name) and os.path.isfile(os.path.join(path, "reveal.json")):
            eid, path = name, os.path.join(path, "reveal.json")
        elif name.endswith(".json") and named.fullmatch(name[:-5]) and os.path.isfile(path):
            eid = name[:-5]
        else:
            continue
        if eid in found:
            raise SystemExit(f"--reveals {src}: {eid} is there twice ({found[eid]} and {path})")
        found[eid] = path
    return [(eid, found[eid]) for eid in sorted(found, key=lambda e: int(e[1:]))]


def _published(src: str, blob_hash, feed_bytes) -> list[tuple[str, str]]:
    """Every published reveal under `src` through the v3 readers, and the
    seat and window rebuilt from them all."""
    import hashlib

    from fugal_subnet import config
    from fugal_subnet.routing_protocol import MENU_B, price_table
    from fugal_subnet.v2_epoch import save_window, seat_to_dict
    from fugal_subnet.v2_rebuild import pairs_from_reveal, rebuild_v2_from_reveals
    from fugal_subnet.v2_sync import comparable_reveal, consensus_view

    out: list[tuple[str, str]] = []
    found = reveal_paths(src)
    if not found:
        raise SystemExit(f"--reveals {src}: no reveal read (neither e000NNNNN/reveal.json nor "
                         "e000NNNNN.json)")
    paths = [p for _eid, p in found]
    for eid, p in found:
        with open(p) as f:
            rv = json.load(f)
        view = consensus_view(rv)
        comparable = json.dumps(comparable_reveal(rv), sort_keys=True)
        try:
            pairs = repr(pairs_from_reveal(rv, {}, config.BENCHMARK_CATEGORY_V3))
        except Exception as e:  # noqa: BLE001 - the error class is the line
            pairs = type(e).__name__
        out.append((f"reveals/{eid}",
                    hashlib.sha256((view + comparable + pairs).encode()).hexdigest()[:16]))
    tables = {2: (price_table(2), config.BENCHMARK_CATEGORY_V2),
              3: (price_table(3, MENU_B), config.BENCHMARK_CATEGORY_V3)}
    rb = rebuild_v2_from_reveals(paths, window_size=48, prices=price_table(3, MENU_B),
                                 bench_category=config.BENCHMARK_CATEGORY_V3, tables=tables)
    out.append(("reveals/rebuild_seat", blob_hash(seat_to_dict(rb.seat))))
    wp = os.path.join(tempfile.mkdtemp(prefix="v3equiv-"), "w.npz")
    save_window(wp, rb.window)
    with open(wp, "rb") as f:
        out.append(("reveals/window_bytes", hashlib.sha256(f.read()).hexdigest()[:16]))
    out.append(("reveals/count", str(len(paths))))
    return out


# ── the comparison ──────────────────────────────────────────────────────────


def _parsed(value: str):
    try:
        return json.loads(value)
    except ValueError:
        return None


def _check_four_field(new: str, old: str) -> str | None:
    """None when the four-field commitment parses on the tree with v4 (as the
    epoch and sha256, and as the commitment with its menu hash) and not on
    the tree without it; else why not."""
    a, b = _parsed(new), _parsed(old)
    want = ["e00026500", "a" * 64]
    if not isinstance(a, dict) or a.get("pointer") != want:
        return f"the tree under test does not read it as {want}: {new}"
    if a.get("commitment") != want + [_MENU_HASH]:
        return f"the tree under test does not read the menu hash from it: {new}"
    if not isinstance(b, dict) or b.get("pointer") is not None or "commitment" in b:
        return f"the reference tree reads it: {old}"
    return None


def _check_v4_namespace(new: str, old: str) -> str | None:
    """None when the proof is unparseable on the tree with v4 and parses on
    the tree without it; else why not."""
    a, b = _parsed(new), _parsed(old)
    if not isinstance(a, dict) or a.get("parses") is not False:
        return f"the tree under test parses it: {new}"
    if not isinstance(b, dict) or b.get("parses") is not True:
        return f"the reference tree does not parse it: {old}"
    return None


# The two differences protocol v4 makes on purpose, each with the check of its shape.
INTENDED = {
    "commit/four_field": ("a four-field commitment parses with v4 and not without it",
                          _check_four_field),
    "proof/v4_namespace": ("a proof outside the v1-v3 namespaces with malformed v4 fields is "
                           "unparseable with v4 and parses without it", _check_v4_namespace),
}


def compare(new: list[tuple[str, str]], old: list[tuple[str, str]]) -> tuple[list[str], bool]:
    """(report lines, ok). `new` is the tree under test's lines, `old` the
    reference's. ok when every tag is on both sides, every value is equal
    except the intended ones, and each intended one has its exact shape."""
    a, b = dict(new), dict(old)
    report: list[str] = []
    ok = len(a) == len(new) and len(b) == len(old)
    if not ok:
        report.append("UNEXPECTED: a tag is repeated within one tree's lines")
    for tag in sorted(set(a) | set(b)):
        if tag not in a or tag not in b:
            ok = False
            report.append(f"UNEXPECTED {tag}: only in the "
                          f"{'tree under test' if tag in a else 'reference tree'}")
            continue
        if tag in INTENDED:
            what, check = INTENDED[tag]
            why = check(a[tag], b[tag])
            if why is None:
                report.append(f"intended   {tag}: {what}")
                report.append(f"             under test {a[tag]}")
                report.append(f"             reference  {b[tag]}")
            else:
                ok = False
                report.append(f"UNEXPECTED {tag}: not the intended difference ({why})")
            continue
        if a[tag] != b[tag]:
            ok = False
            report.append(f"UNEXPECTED {tag}: under test {a[tag]}  reference {b[tag]}")
    for tag in INTENDED:
        if tag not in a or tag not in b:
            ok = False
            report.append(f"UNEXPECTED: the intended line {tag} was not produced")
    return report, ok


# ── running a tree ──────────────────────────────────────────────────────────


def run_tree(tree: str, reveals: str | None = None) -> list[tuple[str, str]]:
    """The scenario's lines from `tree`, computed in a fresh interpreter (-I:
    no user site, no PYTHON* variables) with only that tree on the path, and
    without any FUGAL_ variable from this environment, so each tree runs its
    own code and helpers on the shipped constants."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("FUGAL_")}
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "--child", os.path.abspath(tree)]
    if reveals:
        cmd += ["--reveals", os.path.abspath(reveals)]
    proc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(f"the scenario failed in {tree}:\n{proc.stdout[-4000:]}\n"
                         f"{proc.stderr[-4000:]}")
    out = []
    for line in proc.stdout.splitlines():
        if line.startswith("LINE "):
            tag, _, value = line[5:].partition(" ")
            out.append((tag, value))
    return out


def archive(ref: str, dest: str) -> str:
    """`ref` of this checkout, written to `dest` by git archive."""
    os.makedirs(dest, exist_ok=True)
    proc = subprocess.run(["git", "-C", REPO, "archive", ref], capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"git archive {ref} failed in {REPO}: "
                         f"{proc.stderr.decode(errors='replace').strip()}")
    data = proc.stdout
    subprocess.run(["tar", "-x", "-C", dest], input=data, check=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--tree", action="append", default=[],
                    help="a tree to run (give two: the one under test, then the reference)")
    ap.add_argument("--ref", help="the reference as a git ref of this checkout (git archive); "
                                  "the tree under test is this checkout unless --tree names it")
    ap.add_argument("--reveals", help="also re-read the published reveals under this directory")
    ap.add_argument("--lines", action="store_true", help="print every line of both trees")
    args = ap.parse_args(argv)

    if args.child:
        for tag, value in _scenario(args.child, args.reveals):
            print(f"LINE {tag} {value}", flush=True)
        return 0

    if args.reveals is not None:
        # Re-reading no reveal proves nothing about them: refuse, before the
        # scenario runs, rather than report the scenario alone as EQUIVALENT.
        n = len(reveal_paths(args.reveals))
        if not n:
            print(f"NO REVEALS: --reveals {args.reveals} holds no reveal in either layout "
                  "(e000NNNNN/reveal.json, or e000NNNNN.json); nothing was compared")
            return 2
        print(f"reveals:    {n} under {os.path.abspath(args.reveals)}", flush=True)

    with tempfile.TemporaryDirectory(prefix="v3equiv-ref-") as tmp:
        if args.ref:
            if len(args.tree) > 1:
                ap.error("--ref takes at most one --tree (the tree under test)")
            under_test = args.tree[0] if args.tree else REPO
            reference = archive(args.ref, os.path.join(tmp, "ref"))
            names = (under_test, f"{args.ref} (git archive)")
        elif len(args.tree) == 2:
            under_test, reference = args.tree
            names = (under_test, reference)
        else:
            ap.error("give --ref REF, or --tree UNDER_TEST --tree REFERENCE")
        print(f"under test: {names[0]}\nreference:  {names[1]}", flush=True)
        new = run_tree(under_test, args.reveals)
        old = run_tree(reference, args.reveals)
    if args.lines:
        for (tag, value) in new:
            print(f"  under test  {tag} {value}")
        for (tag, value) in old:
            print(f"  reference   {tag} {value}")
    report, ok = compare(new, old)
    equal = sum(1 for tag, value in new if dict(old).get(tag) == value)
    print(f"{len(new)} lines under test, {len(old)} in the reference, {equal} equal")
    for line in report:
        print(line)
    if ok:
        print(f"EQUIVALENT: protocol v3 is identical on both trees apart from the "
              f"{len(INTENDED)} intended differences")
        return 0
    print("NOT EQUIVALENT: a difference above is not one protocol v4 intends")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
