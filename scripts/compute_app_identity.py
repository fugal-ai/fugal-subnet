#!/usr/bin/env python3
"""Compute the approved-list entry for this checkout, without TDX hardware.

This is what the (base_measurement, app_identity) split is *for*. The base half
is a property of the image and can only be measured on the machine that boots
it. The app half is a property of this repository, so it can be computed here —
which means **what code the subnet accepts is reviewable in a pull request**
rather than requiring someone to hold a quote and read hex out of a register.

    python scripts/compute_app_identity.py                       # this checkout
    python scripts/compute_app_identity.py --base <measurement>  # full entry
    python scripts/compute_app_identity.py --compose app-compose.json

Two identities exist because two deployment shapes exist, and they are not
interchangeable:

  runtime identity   sha384(source || pool || grader || upstream), extended into
                     RTMR3 by the miner itself. What a Fugal miner produces today
                     on a stock image. Advisory: on an unmeasured filesystem an
                     attacker extends whatever value is expected.

  compose hash       sha256 of the normalised app-compose, extended into RTMR3 by
                     dstack's *initrd* — which is itself measured into RTMR2. That
                     is what makes it evidence rather than advisory, and it is the
                     whole reason for adopting a locked image.

UNVERIFIED: the compose-hash normalisation here follows dstack's documented rules
(sorted keys, compact separators, non-finite floats as null, UTF-8 emitted
directly) but has never been compared against a hash dstack actually extended.
Check it against a live deployment before trusting an approved entry built from
it — see docs/CODE_BENCHMARK_PLAN.md for the same warning about assuming.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def source_hash() -> str:
    """SHA256 over every .py file in the package, path-prefixed, sorted.

    Identical to neurons/miner.py's `_get_source_hash`, deliberately: a value
    computed differently here would approve something the miner cannot produce.
    """
    import fugal_subnet

    src_dir = os.path.dirname(os.path.abspath(fugal_subnet.__file__))
    h = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(src_dir)):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            h.update(fpath[len(src_dir):].encode())
            with open(fpath, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def normalise_app_compose(obj) -> str:
    """dstack's normalised app-compose form: sorted keys, compact, finite floats.

    Their documented rules are sorted keys, compact output, NaN and Infinity
    rendered as null, and UTF-8 written directly rather than escaped. json.dumps
    covers three of those; the fourth needs a pass, because Python emits `NaN`
    and `Infinity` — which are not JSON — where the spec wants `null`.
    """
    def finite(x):
        if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
            return None
        if isinstance(x, dict):
            return {k: finite(v) for k, v in x.items()}
        if isinstance(x, list):
            return [finite(v) for v in x]
        return x

    return json.dumps(
        finite(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def compose_hash(path: str) -> tuple[str, str]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    canonical = normalise_app_compose(obj)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="", help="Base measurement, to print a full entry")
    ap.add_argument("--compose", default="", help="app-compose.json, for a dstack deployment")
    ap.add_argument("--pool", default="", help="Pool JSON (default: the configured loader)")
    ap.add_argument("--upstream", default=os.getenv(
        "FUGAL_OPENROUTER_BASE", "https://openrouter.ai/api/v1"))
    args = ap.parse_args()

    if args.compose:
        digest, canonical = compose_hash(args.compose)
        print(f"normalised   {canonical[:120]}{'...' if len(canonical) > 120 else ''}")
        print(f"compose_hash {digest}")
        identity = digest
    else:
        from fugal_subnet.benchmarks.loader import load_all, pool_hash
        from fugal_subnet.graders import grader_hash
        from fugal_subnet.tee.attestation import expected_rtmr3, runtime_identity

        if args.pool:
            os.environ["FUGAL_BENCHMARK_POOL"] = os.path.abspath(args.pool)
        pool = load_all()
        src, ph, gh = source_hash(), pool_hash(pool), grader_hash()
        identity = runtime_identity(src, ph, gh, args.upstream)
        print(f"source_hash  {src}")
        print(f"pool_hash    {ph}   ({len(pool)} questions)")
        print(f"grader_hash  {gh}")
        print(f"upstream     {args.upstream}")
        print(f"\nruntime_identity {identity}")
        print(f"RTMR3 after one extend from a fresh TD: {expected_rtmr3(identity)}")

    print()
    if args.base:
        print("FUGAL_TEE_MEASUREMENTS entry:")
        print(f"  {args.base}:{identity}")
    else:
        print("Pass --base <measurement> to print the full approved entry. The base "
              "half can only be read off the machine that boots the image; this "
              "half is a property of the repository, which is the point of "
              "keeping them separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
