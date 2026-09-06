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

The compose hash is sha256 of the file's RAW BYTES. It was normalised JSON here
until a real dstack-generated app-compose.json falsified that; see
`compose_hash` for what the two hashes were and why the documented rules turned
out to govern a different object entirely.
"""
from __future__ import annotations

import argparse
import hashlib
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


def compose_hash(path: str) -> tuple[str, str]:
    """SHA-256 of app-compose.json's RAW FILE BYTES. Not a re-serialisation.

    This function previously normalised the JSON first — sorted keys, compact
    separators, non-finite floats as null — following dstack's
    `normalized-app-compose.md`. That was wrong, and wrong in a way that would
    have rejected every honest miner: an approved entry built from it names a
    compose hash dstack never extends.

    dstack measures the bytes on disk. Their own source says so, next to the
    copy that writes the file:

        the measured compose-hash is sha256 of the bytes on the shared disk, so
        re-serializing would change the hash and break an externally-built
        compose
        # raw byte copy - do NOT json.load/dump (would change hashed bytes)

    Measured against a real 564-byte app-compose.json that dstack generated:

        sha256(raw bytes)            b695e134f294905730f1704b025cef4d...  <- what is measured
        sha256(sorted + compact)     7ade910a9854391c04c8198f5e594a1c...  <- the old rules

    THE DOCUMENT WAS REAL AND DESCRIBED SOMETHING ELSE. Those normalisation
    rules govern the canonical JSON of runtime EVENTS — the
    {"name","payload","type"} object in `attestation.dstack_event_digest`, where
    they are correct and verified against dstack's test vector. They do not
    govern the compose file. Two JSON conventions in one system, and the first
    reading mapped one onto the other.

    That is the second time today a documented rule for a neighbouring thing has
    produced confident, dead code — the first cost a `extend_rtmr3` that could
    never work. The lesson is cheap to state and apparently expensive to learn:
    a specification tells you what someone intended, an artefact tells you what
    they built, and only the second one is what you are verifying against.
    """
    with open(path, "rb") as f:
        raw = f.read()
    return hashlib.sha256(raw).hexdigest(), raw.decode("utf-8", "replace")


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
        digest, raw = compose_hash(args.compose)
        print(f"file         {args.compose} ({len(raw)} chars)")
        print(f"compose_hash {digest}   (sha256 of the RAW bytes, not a re-serialisation)")
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
