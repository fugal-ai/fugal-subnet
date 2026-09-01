#!/usr/bin/env python3
"""Execute and hash the manifest-pinned real v2 backbone vector."""

from __future__ import annotations

import platform

from fugal_subnet.consensus_manifest import load_consensus_manifest
from fugal_subnet.v2.backbone import verify_backbone_golden


def main() -> None:
    manifest = load_consensus_manifest("local")
    protocol = next(item for item in manifest.protocols if item.protocol_id == "v2")
    if protocol.consensus is None:
        raise SystemExit("packaged v2 consensus material is missing")
    backbone = protocol.consensus["backbone"]
    actual = verify_backbone_golden(
        expected_prompts_sha256=backbone["golden_prompts_sha256"],
        expected_embeddings_sha256=backbone["golden_embeddings_sha256"],
    )
    print(f"v2 backbone golden passed on Python {platform.python_version()}: {actual}")


if __name__ == "__main__":
    main()
