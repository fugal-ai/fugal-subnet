"""Axon handler and Dendrite client for fetching signed v2 reports."""

# Do not add ``from __future__ import annotations``. Bittensor 10.x inspects
# the forward function below at Axon.attach() time.
import hashlib
import os
import tempfile
from pathlib import Path

from fugal_subnet.v2.protocol import MatrixReportSynapse, ReportChunk, assemble_report
from fugal_subnet.v2.reports import (
    chunk_signed_report,
    verify_chunk_signature,
    verify_report,
)


class ReportFetchError(RuntimeError):
    pass


class ReportStore:
    """Permission-restricted local report store used by one builder Axon."""

    def __init__(self, root: str | Path, current_block=None):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ReportFetchError("report store root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ReportFetchError("report store root is not a private directory")
        self.root.chmod(0o700)
        self.current_block = current_block
        self._chunks: dict[tuple[str, str], tuple[ReportChunk, ...]] = {}
        self._release_blocks: dict[tuple[str, str], int] = {}

    def _atomic_write(self, destination: Path, payload: bytes) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".report.", dir=self.root)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def publish(
        self, payload: bytes, keypair, *, release_block: int = 0,
    ) -> tuple[ReportChunk, ...]:
        if (
            not isinstance(release_block, int)
            or isinstance(release_block, bool)
            or release_block < 0
        ):
            raise ReportFetchError("report release block is invalid")
        report = verify_report(payload, expected_hotkey=keypair.ss58_address)
        chunks = tuple(chunk_signed_report(payload, keypair))
        key = (report["core"]["epoch_id"], chunks[0].artifact_hash)
        if key in self._chunks and self._chunks[key] != chunks:
            raise ReportFetchError("conflicting report artifact already published")
        destination = self.root / f"{key[0]}-{key[1]}.report"
        if destination.is_symlink():
            raise ReportFetchError("persisted report artifact cannot be a symlink")
        if destination.exists() and destination.read_bytes() != payload:
            raise ReportFetchError("persisted report artifact conflicts")
        if not destination.exists():
            self._atomic_write(destination, payload)
        release_path = destination.with_suffix(".release")
        if release_path.is_symlink():
            raise ReportFetchError("persisted report release gate cannot be a symlink")
        if release_path.exists():
            try:
                persisted_release = int(release_path.read_text(encoding="ascii"))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                raise ReportFetchError("persisted report release block is invalid") from exc
            if persisted_release != release_block:
                raise ReportFetchError("persisted report release block conflicts")
        else:
            self._atomic_write(release_path, str(release_block).encode("ascii"))
        self._chunks[key] = chunks
        self._release_blocks[key] = release_block
        return chunks

    def restore(self, keypair) -> int:
        """Rebuild signed chunks from persisted reports after a service restart."""
        restored = 0
        for source in sorted(self.root.glob("*.report")):
            if source.is_symlink() or not source.is_file():
                raise ReportFetchError("persisted report artifact is not a regular file")
            payload = source.read_bytes()
            report = verify_report(payload, expected_hotkey=keypair.ss58_address)
            chunks = tuple(chunk_signed_report(payload, keypair))
            key = (report["core"]["epoch_id"], chunks[0].artifact_hash)
            if source.name != f"{key[0]}-{key[1]}.report":
                raise ReportFetchError("persisted report filename differs from signed content")
            if key in self._chunks and self._chunks[key] != chunks:
                raise ReportFetchError("persisted report conflicts with active report")
            release_path = source.with_suffix(".release")
            if release_path.is_symlink() or not release_path.is_file():
                raise ReportFetchError("persisted report release block is missing")
            try:
                release_block = int(release_path.read_text(encoding="ascii"))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                raise ReportFetchError("persisted report release block is invalid") from exc
            if release_block < 0:
                raise ReportFetchError("persisted report release block is invalid")
            self._chunks[key] = chunks
            self._release_blocks[key] = release_block
            restored += 1
        return restored

    def get(self, epoch_id: str, artifact_hash: str, chunk_index: int) -> ReportChunk:
        chunks = self._chunks.get((epoch_id, artifact_hash))
        if chunks is None:
            raise ReportFetchError("report artifact unavailable")
        release_block = self._release_blocks[(epoch_id, artifact_hash)]
        if release_block:
            if self.current_block is None:
                raise ReportFetchError("report deadline cannot be established")
            if int(self.current_block()) < release_block:
                raise ReportFetchError("report artifact is sealed until the deadline")
        if not 0 <= chunk_index < len(chunks):
            raise ReportFetchError("report chunk index unavailable")
        return chunks[chunk_index]

    def read_payload(self, epoch_id: str, artifact_hash: str) -> bytes:
        """Read a local committed report subject to the same release gate."""
        self.get(epoch_id, artifact_hash, 0)
        return self._read_persisted(epoch_id, artifact_hash)

    def resume_payload(self, epoch_id: str, artifact_hash: str) -> bytes:
        """Load this builder's own bytes for resume without serving them early."""
        if (epoch_id, artifact_hash) not in self._chunks:
            raise ReportFetchError("report artifact unavailable")
        return self._read_persisted(epoch_id, artifact_hash)

    def _read_persisted(self, epoch_id: str, artifact_hash: str) -> bytes:
        source = self.root / f"{epoch_id}-{artifact_hash}.report"
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact_hash:
            raise ReportFetchError("persisted report artifact hash differs")
        return payload


def make_report_forward(store: ReportStore):
    async def forward(synapse: MatrixReportSynapse) -> MatrixReportSynapse:
        try:
            chunk = store.get(synapse.epoch_id, synapse.artifact_hash, synapse.chunk_index)
            if synapse.manifest_hash and synapse.manifest_hash != chunk.manifest_hash:
                raise ReportFetchError("manifest hash differs")
            for field, value in chunk.__dict__.items():
                setattr(synapse, field, value)
            synapse.error = ""
        except Exception as exc:
            synapse.payload_b64 = ""
            synapse.error = str(exc)[:256]
        return synapse

    return forward


def _query_one(dendrite, axon, request: MatrixReportSynapse, timeout: float):
    responses = dendrite.query([axon], request, timeout=timeout)
    if not isinstance(responses, list) or len(responses) != 1:
        raise ReportFetchError("report Dendrite response count differs")
    response = responses[0]
    if not isinstance(response, MatrixReportSynapse):
        raise ReportFetchError("report Dendrite returned an invalid Synapse")
    if response.error:
        raise ReportFetchError(f"builder refused report artifact: {response.error}")
    return response


def fetch_report(
    dendrite,
    axon,
    *,
    epoch_id: str,
    manifest_hash: str,
    artifact_hash: str,
    builder_hotkey: str,
    timeout: float = 30.0,
) -> bytes:
    """Fetch every committed chunk and reject refusal, drift, or bad signatures."""
    first = _query_one(
        dendrite,
        axon,
        MatrixReportSynapse(
            epoch_id=epoch_id,
            manifest_hash=manifest_hash,
            artifact_hash=artifact_hash,
            chunk_index=0,
        ),
        timeout,
    )
    count = first.chunk_count
    responses = [first]
    for index in range(1, count):
        responses.append(_query_one(
            dendrite,
            axon,
            MatrixReportSynapse(
                epoch_id=epoch_id,
                manifest_hash=manifest_hash,
                artifact_hash=artifact_hash,
                chunk_index=index,
                chunk_count=count,
            ),
            timeout,
        ))
    chunks = [ReportChunk(
        epoch_id=response.epoch_id,
        manifest_hash=response.manifest_hash,
        artifact_hash=response.artifact_hash,
        chunk_index=response.chunk_index,
        chunk_count=response.chunk_count,
        payload_b64=response.payload_b64,
        builder_hotkey=response.builder_hotkey,
        signature=response.signature,
    ) for response in responses]
    if any(
        chunk.builder_hotkey != builder_hotkey
        or chunk.epoch_id != epoch_id
        or chunk.manifest_hash != manifest_hash
        or chunk.artifact_hash != artifact_hash
        or not verify_chunk_signature(chunk)
        for chunk in chunks
    ):
        raise ReportFetchError("report chunk identity or signature differs")
    payload = assemble_report(chunks)
    verify_report(payload, expected_hotkey=builder_hotkey)
    return payload
