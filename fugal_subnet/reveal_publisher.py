"""Publish epoch reveals to a public store, with their hash on chain.

The reveal is the subnet's stated product: the per-question, per-model ground
truth, the exploration observations, every head's routing decisions, scores
and weights. It was written to the validator's local disk and never left it
(diligence 2026-10, additional finding A1). This module publishes it.

Two facts shape the design:

- **Delay past the weight commit-reveal window.** A reveal carries the weights
  the validator just set. Where weight commit-reveal is on, those weights are
  secret for `commit_reveal_period` tempos; publishing the reveal earlier
  would leak them. So an epoch's reveal is published only once the validator
  is `delay_epochs` epochs past it. The delay is read from the subnet's
  hyperparameters at startup, never guessed.
- **The hash goes on chain, the bytes go to a store.** The commitments pallet
  holds one commitment per hotkey, so the validator's own commitment is a
  rolling pointer: `reveal:<epoch_id>:<sha256>` for the newest reveal it has
  published. Anyone can fetch the file from the store and check it against
  the chain, and two validators' commitments for the same epoch should carry
  the same hash (I1).

The store is a public-read GCS bucket named by FUGAL_REVEAL_BUCKET. Uploads go
through `google-cloud-storage` when it is installed, otherwise through the
`gcloud storage` CLI; both are bounded by a timeout. Publishing is
operational, not consensus: nothing here reads back into a score, and an
unset bucket skips with one warning so a validator without one still runs.

A `.published` marker next to each reveal records what was uploaded and
where; the marker is written only after the upload succeeds, so a failed
upload is retried next epoch and nothing is marked that was not sent.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MARKER = ".published"


def reveal_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def epoch_index_of(epoch_id: str) -> int:
    """`e00022118` -> 22118. The only epoch-id format (slicer.epoch_id_for_block)."""
    return int(str(epoch_id).lstrip("e"))


def commitment_for(epoch_id: str, digest: str) -> str:
    return f"reveal:{epoch_id}:{digest}"


@dataclass
class GcsUploader:
    """Upload a file to gs://bucket/<key>, by library if present, else by CLI."""

    bucket: str
    timeout: float = 60.0

    def upload(self, path: str, key: str) -> str:
        url = f"gs://{self.bucket}/{key}"
        try:
            from google.cloud import storage  # type: ignore
        except ImportError:
            storage = None
        if storage is not None:
            client = storage.Client()
            blob = client.bucket(self.bucket).blob(key)
            blob.upload_from_filename(path, timeout=self.timeout)
            return url
        subprocess.run(
            ["gcloud", "storage", "cp", path, url],
            check=True, capture_output=True, text=True, timeout=self.timeout,
        )
        return url


def delay_epochs_for(subtensor, netuid: int) -> int:
    """How many epochs to hold a reveal: the weight commit-reveal period when
    it is enabled, otherwise one epoch (the weights are public as soon as they
    are set). Read from the chain, never assumed."""
    try:
        hp = subtensor.get_subnet_hyperparameters(netuid)
        enabled = bool(getattr(hp, "commit_reveal_weights_enabled", False))
        period = int(getattr(hp, "commit_reveal_period", 0) or 0)
    except Exception as e:  # noqa: BLE001 - a hold that is too short leaks weights
        logger.warning("Could not read commit-reveal hyperparameters (%s); holding "
                       "reveals for 1 epoch", e)
        return 1
    return max(1, period) if enabled else 1


def due_reveals(epoch_dir: str, current_epoch_index: int, delay_epochs: int) -> list[str]:
    """Reveal paths old enough to publish and not yet marked, oldest first."""
    due = []
    for path in sorted(glob.glob(os.path.join(epoch_dir, "*", "reveal.json"))):
        epoch_id = os.path.basename(os.path.dirname(path))
        try:
            index = epoch_index_of(epoch_id)
        except ValueError:
            continue
        if index > current_epoch_index - delay_epochs:
            continue
        if os.path.exists(os.path.join(os.path.dirname(path), MARKER)):
            continue
        due.append(path)
    return due


def publish_due_reveals(
    epoch_dir: str,
    current_epoch_index: int,
    delay_epochs: int,
    netuid: int,
    uploader,
    commit=None,
) -> list[dict]:
    """Upload every due reveal; commit the newest one's hash on chain.

    `uploader.upload(path, key) -> url`; `commit(data: str) -> bool` makes the
    on-chain commitment (one per call, so it is made once per run, for the
    newest reveal published: the pallet keeps only the latest anyway, and a
    commitment per epoch is within any rate limit). Returns one record per
    published reveal. Never raises for an upload failure: the reveal stays
    unmarked and is retried next epoch.
    """
    published: list[dict] = []
    for path in due_reveals(epoch_dir, current_epoch_index, delay_epochs):
        epoch_id = os.path.basename(os.path.dirname(path))
        with open(path, "rb") as f:
            data = f.read()
        digest = reveal_digest(data)
        key = f"{netuid}/{epoch_id}/reveal.json"
        try:
            url = uploader.upload(path, key)
        except Exception as e:  # noqa: BLE001 - reported, retried next epoch
            logger.warning("Reveal %s not published (%s); will retry", epoch_id, e)
            break  # keep order: do not publish a newer epoch before an older one
        record = {"epoch_id": epoch_id, "sha256": digest, "url": url}
        with open(os.path.join(os.path.dirname(path), MARKER), "w", encoding="utf-8") as f:
            json.dump(record, f)
        published.append(record)
        logger.info("Published reveal %s -> %s (sha256 %s)", epoch_id, url, digest[:16])
    if published and commit is not None:
        newest = published[-1]
        try:
            ok = bool(commit(commitment_for(newest["epoch_id"], newest["sha256"])))
        except Exception as e:  # noqa: BLE001
            ok, e_msg = False, str(e)
            logger.warning("Reveal hash commitment failed: %s", e_msg)
        newest["committed"] = ok
    return published
