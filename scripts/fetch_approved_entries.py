#!/usr/bin/env python3
"""Verify a signed release tag and print the approved entry it carries.

The validator side of docs/APPROVED_ENTRY_ROTATION.md's publication channel
(diligence 2026-10, finding 7). An approved entry is only evidence if the
operator can trace it to a signed statement by the team: this fetches the
tag, verifies its signature against the allowed signers in deploy/keys/
(SSH or GPG public keys committed to the repository, so the trust root is in
the same reviewed history as the code), reads the entry out of the release
note, and prints the environment line. It never writes to the validator's
environment: that step is the operator's, deliberately.

    python scripts/fetch_approved_entries.py --tag image-v2026.10.14
    # -> FUGAL_TEE_MEASUREMENTS=<base>:<compose_hash>

Requires `git` and `gh` (the GitHub CLI) and network access; no API spend.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = os.path.join(ROOT, "deploy", "keys")
ENTRY_RE = re.compile(r"\| entry \| `([0-9a-f]{64}:[0-9a-f]{64})` \|")


def run(*cmd, **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120, **kw).stdout


def verify_tag(tag: str) -> None:
    """git tag -v, with the committed allowed-signers as the only trust root."""
    signers = os.path.join(KEYS, "allowed_signers")
    if not os.path.exists(signers):
        raise SystemExit(f"{signers} is missing: no trust root to verify {tag} against")
    run("git", "fetch", "--tags", "origin", tag, cwd=ROOT)
    env = dict(os.environ)
    run("git", "-c", f"gpg.ssh.allowedSignersFile={signers}", "tag", "-v", tag, cwd=ROOT, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the signature check (reading only; never for approval)")
    args = ap.parse_args()

    if not args.no_verify:
        verify_tag(args.tag)
        print(f"# tag {args.tag}: signature verified against deploy/keys/allowed_signers",
              file=sys.stderr)
    notes = run("gh", "release", "view", args.tag, "--json", "body", "--jq", ".body", cwd=ROOT)
    m = ENTRY_RE.search(notes)
    if not m:
        raise SystemExit(f"release {args.tag} carries no completed approved entry")
    print(f"FUGAL_TEE_MEASUREMENTS={m.group(1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
