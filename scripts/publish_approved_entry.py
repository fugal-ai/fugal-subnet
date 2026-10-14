#!/usr/bin/env python3
"""Print the release note for an approved entry, and sign the tag that carries it.

An approved entry is a consensus parameter: `<base>:<compose_hash>` is what
every --live validator accepts. docs/APPROVED_ENTRY_ROTATION.md says each one
is published as a signed GitHub release naming the commit, the image digest,
the compose hash and the base. This script writes that note (diligence
2026-10, finding 7); the operator signs the tag with their own key and
creates the release:

    python scripts/publish_approved_entry.py --digest ghcr.io/...@sha256:... \
        --commit <sha> --compose deploy/dstack/app-compose.frozen-<date>.json \
        --base <base> --tag image-v2026.10.14 > notes.md
    git tag -s image-v2026.10.14 -m "approved entry: <base>:<compose_hash>"
    git push origin image-v2026.10.14
    gh release create image-v2026.10.14 --notes-file notes.md --verify-tag

The CI image workflow drafts the note without --base (it does not know the
cloud platform's measurement); the operator completes it after
`dstack-cloud prepare` has produced the frozen app-compose.json, whose RAW
bytes are the compose hash. scripts/fetch_approved_entries.py is the
validator side: it verifies the tag signature against deploy/keys/ and prints
the FUGAL_TEE_MEASUREMENTS line.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--digest", required=True, help="image reference by digest")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--compose", required=True,
                    help="frozen app-compose.json (hashed raw) or a docker-compose.yaml (listed only)")
    ap.add_argument("--base", default="", help="base measurement of the dstack image on the platform")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    raw = open(args.compose, "rb").read()
    compose_hash = hashlib.sha256(raw).hexdigest() if args.compose.endswith(".json") else ""
    entry = f"{args.base}:{compose_hash}" if args.base and compose_hash else "(complete after prepare)"

    print(f"# Approved entry {args.tag or ''}".rstrip())
    print()
    print("| field | value |")
    print("|---|---|")
    print(f"| entry | `{entry}` |")
    print(f"| base measurement | `{args.base or 'to be filled'}` |")
    print(f"| compose hash | `{compose_hash or 'to be filled from app-compose.json raw bytes'}` |")
    print(f"| image digest | `{args.digest}` |")
    print(f"| commit | `{args.commit}` |")
    print(f"| compose file | `{os.path.basename(args.compose)}` |")
    print(f"| published | {date.today().isoformat()} |")
    print()
    print("Validators: add the entry to `FUGAL_TEE_MEASUREMENTS` alongside the current one, "
          "restart, and retire the old entry per docs/APPROVED_ENTRY_ROTATION.md after the "
          "transition window. Verify this release's tag signature with "
          "`python scripts/fetch_approved_entries.py --tag <tag>` before approving anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
