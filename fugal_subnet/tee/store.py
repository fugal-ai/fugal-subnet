"""Proof bundle storage — upload and download via HuggingFace Hub.

Miners upload proof bundles after each epoch. Validators download and
parse them for verification. Bundles are JSON-serialized BenchmarkProofs
stored as files in a HuggingFace dataset repository.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from fugal_subnet.config import HEAD_MAX_BYTES
from fugal_subnet.tee.proof import BenchmarkProof

logger = logging.getLogger(__name__)


def upload_bundle(
    proof: BenchmarkProof,
    head_bytes: bytes,
    repo_id: str,
    hotkey: str = "",
    token: str | None = None,
) -> str:
    """Upload a proof bundle (proof + the head that produced it) and return its URL.

    The proof is stored at ``proofs/{epoch_id}/{hotkey}.json``. The head is
    stored content-addressed at ``heads/{weights_hash}.npz``, so a miner that
    keeps the same head across epochs uploads it once, and a validator can
    always locate the head a given proof ran from.

    Shipping the head is what lets a validator bind the proof to the on-chain
    commitment and re-run the head on held-out questions. Without it the
    validator only ever sees a hash the miner asserted.
    """
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        head_path = Path(tmpdir) / "head.npz"
        head_path.write_bytes(head_bytes)
        api.upload_file(
            path_or_fileobj=str(head_path),
            path_in_repo=f"heads/{proof.weights_hash}.npz",
            repo_id=repo_id,
            repo_type="dataset",
        )

        filename = f"proofs/{proof.epoch_id}/{hotkey or 'anonymous'}.json"
        proof_path = Path(tmpdir) / "proof.json"
        proof_path.write_text(
            json.dumps(proof.to_dict(), separators=(",", ":")), encoding="utf-8",
        )
        api.upload_file(
            path_or_fileobj=str(proof_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    logger.info("Uploaded bundle to %s", url)
    return url


def download_bundle(
    url: str,
    repo_id: str | None = None,
    filename: str | None = None,
    token: str | None = None,
) -> tuple[BenchmarkProof, bytes]:
    """Download and parse a proof bundle, returning (proof, head_bytes).

    Both artifacts are miner-controlled input, so the head is size-capped here
    before it is ever handed to a loader (I2, bounded ingestion). The caller
    still verifies sha256(head_bytes) against the attested weights_hash.
    """
    if repo_id is None or filename is None:
        repo_id, filename = _parse_hf_url(url)

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
    )

    with open(local_path, encoding="utf-8") as f:
        data = json.load(f)
    proof = BenchmarkProof.from_dict(data)

    head_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"heads/{proof.weights_hash}.npz",
        repo_type="dataset",
        token=token,
    )
    head_bytes = Path(head_path).read_bytes()
    if len(head_bytes) > HEAD_MAX_BYTES:
        raise ValueError(
            f"Bundled head is {len(head_bytes)} bytes (max {HEAD_MAX_BYTES})"
        )

    return proof, head_bytes


def _parse_hf_url(url: str) -> tuple[str, str]:
    """Extract repo_id and filename from a HuggingFace datasets URL.

    Expected format:
        https://huggingface.co/datasets/{org}/{repo}/resolve/main/{path}
    """
    prefix = "https://huggingface.co/datasets/"
    if not url.startswith(prefix):
        raise ValueError(f"Not a HuggingFace datasets URL: {url}")

    remainder = url[len(prefix):]
    parts = remainder.split("/")
    if len(parts) < 5 or parts[2] != "resolve":
        raise ValueError(f"Cannot parse HuggingFace URL: {url}")

    repo_id = f"{parts[0]}/{parts[1]}"
    filename = "/".join(parts[4:])
    return repo_id, filename
