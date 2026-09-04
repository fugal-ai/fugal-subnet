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

from fugal_subnet.tee.proof import BenchmarkProof

logger = logging.getLogger(__name__)


def upload_proof(
    proof: BenchmarkProof,
    repo_id: str,
    hotkey: str = "",
    token: str | None = None,
) -> str:
    """Upload a proof bundle to HuggingFace and return its download URL.

    The file is stored as ``proofs/{epoch_id}/{hotkey}.json`` inside the
    repo so each miner's proof is addressable by epoch and identity.
    """
    api = HfApi(token=token)

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    filename = f"proofs/{proof.epoch_id}/{hotkey or 'anonymous'}.json"
    content = json.dumps(proof.to_dict(), separators=(",", ":"))

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / "proof.json"
        local_path.write_text(content, encoding="utf-8")

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    logger.info("Uploaded proof to %s", url)
    return url


def download_proof(
    url: str,
    repo_id: str | None = None,
    filename: str | None = None,
    token: str | None = None,
) -> BenchmarkProof:
    """Download and parse a proof bundle from HuggingFace.

    Accepts either a full URL (parsed to extract repo_id and filename)
    or explicit repo_id + filename.
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

    return BenchmarkProof.from_dict(data)


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
