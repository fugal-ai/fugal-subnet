#!/usr/bin/env python3
"""Build, prepare and check the signed lists of protocol v4.

The lab and host allowlist (docs/decisions/LIVE_MENU_V4.md, section 6) and the
device list (section 13.4) are data signed by a release key under the
namespace `fugal-list@fugalai.com`. This tool does everything around the
signature except make it: signing is one `ssh-keygen` command the key holder
runs (deploy/keys/README.md), and this script never reads a private key.

    # 1. the first draft: exactly the labs and hosts data/models_v3b.json pins
    python scripts/signed_list.py draft            # writes data/menu_allowlist.draft.json
                                                   # (refuses to replace a draft that differs
                                                   # from the derivation; --overwrite forces)
    python scripts/signed_list.py draft --check    # the draft on disk is that derivation, or a
                                                   # reviewed draft that contains it

    # 2. after review: the bytes to sign, with a real serial
    python scripts/signed_list.py prepare --serial 1

    # 3. sign (the key holder), then check what will ship
    ssh-keygen -Y sign -f ~/.ssh/<release key> -n fugal-list@fugalai.com \\
        fugal_subnet/signed_lists/menu_allowlist.json
    python scripts/signed_list.py verify fugal_subnet/signed_lists/menu_allowlist.json

No network, no API spend. A draft carries serial 0, which the verifier refuses
whatever signs it; `prepare` is the only step that gives a list a real serial.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fugal_subnet import menu_rules as mr  # noqa: E402

MENU = ROOT / "data" / "models_v3b.json"
DRAFT = ROOT / "data" / "menu_allowlist.draft.json"
KINDS = {"menu": mr.KIND_MENU_ALLOWLIST, "devices": mr.KIND_DEVICE_LIST}


def draft_from_menu(rows: list[dict]) -> bytes:
    """The draft allowlist a pinned menu implies, and nothing else.

    labs: the part of each row's model id before `/`. hosts: each pinned host
    tag without its quantization segment. first_party, per lab: the hosts a
    row of that lab pins without a quantization and that are not the lab's
    own name, which is what such a row needs in order to pass the rules (a
    host is first-party only for the labs it is marked for).
    """
    labs: set[str] = set()
    hosts: set[str] = set()
    first_party: dict[str, set[str]] = {}
    for row in rows:
        lab = row["model"].split("/", 1)[0]
        order = row["provider"]["order"]
        quantization = mr.pinned_quantization(order)
        if quantization is None:
            raise SystemExit(f"{row['id']}: its hosts do not pin one quantization")
        labs.add(lab)
        for tag in order:
            host = mr.host_identity(tag)
            hosts.add(host)
            if quantization == mr.UNKNOWN_QUANTIZATION and host != lab:
                first_party.setdefault(lab, set()).add(host)
    return mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, mr.DRAFT_SERIAL, labs=labs,
                                   hosts=hosts, first_party=first_party)


def _reviewed_extras(have: bytes, built: bytes):
    """(labs, hosts, first-party marks) the draft `have` holds beyond the derivation
    `built`, or None when `have` is unreadable or lacks something the derivation has."""
    try:
        kind = mr.KIND_MENU_ALLOWLIST
        a, b = mr.parse_unsigned_list(have, kind), mr.parse_unsigned_list(built, kind)
    except mr.ListRefused:
        return None
    marks_a = {(lab, h) for lab, hs in a.first_party for h in hs}
    marks_b = {(lab, h) for lab, hs in b.first_party for h in hs}
    if not (a.labs >= b.labs and a.hosts >= b.hosts and marks_a >= marks_b):
        return None
    return (len(a.labs - b.labs), len(a.hosts - b.hosts), len(marks_a - marks_b))


def cmd_draft(args) -> int:
    built = draft_from_menu(json.loads(Path(args.menu).read_text(encoding="utf-8")))
    out = Path(args.out)
    have = out.read_bytes() if out.exists() else None
    if args.check:
        if have is None:
            print(f"{out} is not the draft {args.menu} implies", file=sys.stderr)
            return 1
        if have == built:
            print(f"{out} is the draft {args.menu} implies")
            return 0
        extras = _reviewed_extras(have, built)
        if extras is None:
            print(f"{out} is not the draft {args.menu} implies, and does not contain it "
                  "either", file=sys.stderr)
            return 1
        labs, hosts, marks = extras
        print(f"reviewed draft: {out} contains the derivation plus {labs} labs and {hosts} hosts"
              + (f" (and {marks} first-party marks)" if marks else ""))
        return 0
    if have is not None and have != built and not args.overwrite:
        print(f"{out} exists and differs from the derivation of {args.menu}: it may hold "
              "reviewed additions. Not overwritten. Review it in place; to replace it with "
              "the plain derivation anyway, pass --overwrite.", file=sys.stderr)
        return 1
    out.write_bytes(built)
    print(f"wrote {out}: an UNSIGNED DRAFT (serial {mr.DRAFT_SERIAL}) for review")
    return 0


def cmd_prepare(args) -> int:
    kind = KINDS[args.kind]
    if args.serial < 1:
        raise SystemExit("--serial must be at least 1 (0 marks a draft)")
    try:
        draft = mr.parse_unsigned_list(Path(args.draft).read_bytes(), kind)
    except mr.ListRefused as e:
        raise SystemExit(f"{args.draft}: {e}") from None
    fields = {name: (draft.first_party_dict() if name == "first_party"
                     else sorted(getattr(draft, name))) for name in mr._LIST_FIELDS[kind]}
    payload = mr.canonical_list_bytes(kind, args.serial, **fields)
    mr.parse_unsigned_list(payload, kind)
    out = Path(args.out) if args.out else mr.SIGNED_LISTS_DIR / mr.SHIPPED_LISTS[kind]
    out.write_bytes(payload)
    print(f"wrote {out} (serial {args.serial}, sha256 "
          f"{mr.parse_unsigned_list(payload, kind).sha256}). It is NOT signed. Sign it with:\n"
          f"  ssh-keygen -Y sign -f ~/.ssh/<release key> -n {mr.LIST_NAMESPACE} {out}\n"
          f"then: python scripts/signed_list.py verify {out}")
    return 0


def cmd_verify(args) -> int:
    path = Path(args.list)
    signature = Path(args.signature) if args.signature else path.with_name(path.name + ".sig")
    signers = Path(args.signers).read_bytes() if args.signers else None
    try:
        got = mr.verify_signed_list(path.read_bytes(), signature.read_bytes(),
                                    kind=KINDS[args.kind], held_serial=args.held_serial,
                                    signers=signers)
    except mr.ListRefused as e:
        print(f"REFUSED {path}: {e}", file=sys.stderr)
        return 1
    print(f"verified {path}: kind {KINDS[args.kind]}, serial {got.serial}, "
          f"sha256 {got.sha256}, signed by {got.signer}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("draft", help="derive the draft allowlist from a pinned menu")
    d.add_argument("--menu", default=str(MENU))
    d.add_argument("--out", default=str(DRAFT))
    d.add_argument("--check", action="store_true",
                   help="compare, write nothing; a draft that contains the derivation plus "
                        "reviewed additions passes")
    d.add_argument("--overwrite", action="store_true",
                   help="replace an existing draft that differs from the derivation")
    d.set_defaults(run=cmd_draft)

    p = sub.add_parser("prepare", help="write the bytes to sign from a reviewed draft")
    p.add_argument("--serial", type=int, required=True)
    p.add_argument("--kind", choices=sorted(KINDS), default="menu")
    p.add_argument("--draft", default=str(DRAFT))
    p.add_argument("--out", default=None, help="default: fugal_subnet/signed_lists/<name>.json")
    p.set_defaults(run=cmd_prepare)

    v = sub.add_parser("verify", help="check a list and its signature, as a neuron would")
    v.add_argument("list")
    v.add_argument("signature", nargs="?", default=None, help="default: <list>.sig")
    v.add_argument("--kind", choices=sorted(KINDS), default="menu")
    v.add_argument("--held-serial", type=int, default=0)
    v.add_argument("--signers", default=None,
                   help="an allowed-signers file (default: the shipped, pinned one)")
    v.set_defaults(run=cmd_verify)

    args = ap.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
