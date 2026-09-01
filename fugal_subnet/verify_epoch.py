"""Command-line verifier for historical v1 and canonical v2 epoch artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from fugal_subnet.commit_reveal import verify_epoch as verify_v1_epoch
from fugal_subnet.training import load_verified_reveal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently verify a Fugal epoch artifact")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--v1-epoch-dir", help="Historical directory with v1 commit.json/reveal.json")
    source.add_argument("--v2-reveal", help="Canonical v2 reveal.json")
    parser.add_argument("--network", help="Bittensor network for exact-block historical checks")
    parser.add_argument("--netuid", type=int)
    parser.add_argument("--grader-socket", help="Isolated grader socket for code/symbolic regrading")
    parser.add_argument(
        "--allow-unverified-chain",
        action="store_true",
        help="Verify internal receipt consistency but explicitly skip historical chain state",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.v1_epoch_dir:
        if not verify_v1_epoch(args.v1_epoch_dir):
            print("v1 epoch verification FAILED")
            return 1
        print("v1 historical commitment verification PASS")
        return 0

    path = Path(args.v2_reveal)
    _, verified = load_verified_reveal(
        path,
        grader_socket=args.grader_socket,
        network=args.network,
        netuid=args.netuid,
        allow_unverified_chain=args.allow_unverified_chain,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    chain = "PASS" if verified.chain_receipts_verified else "SKIPPED EXPLICITLY"
    print(
        f"v2 epoch verification PASS epoch={verified.epoch_id} "
        f"questions={len(verified.question_ids)} models={len(verified.model_ids)} "
        f"chain={chain} sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
