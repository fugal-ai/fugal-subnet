"""Fugal subnet protocol — Synapse definitions for validator-miner communication.

The proof bundle travels **inline** in the response rather than as a URL the
validator then fetches.

Bundles were originally published to a HuggingFace dataset repo, a pattern
inherited from ThirtySpokes/Chutes — where the artifact is a multi-gigabyte
model and an external store is unavoidable. Fugal's bundle is ~230KB, which is
the size at which the ecosystem does the opposite: model-training subnets use a
store plus an on-chain hash, everyone else returns the payload over the axon
they are already being queried on.

An artifact store can do three jobs, and here it only does one. Integrity comes
from the hash chain (report_data -> content_hash -> weights_hash -> on-chain
commitment); attributability comes from the hotkey-signed axon; only
availability is left — for one party, once, immediately, from a miner that is
provably online, because answering this very query is how the validator learned
the hash. So an external store bought a second round trip, a second party that
must be up, and an account per miner, in exchange for nothing.

Nothing is trusted because it arrived inline: the validator still checks the
proof's content hash against the advertised `proof_hash`, and sha256 of the head
against both the attested `weights_hash` and the on-chain commitment.
"""
import bittensor as bt
import pydantic

# 1 MB head, base64-encoded (4/3 overhead) with slack.
HEAD_B64_MAX_LEN = 1_400_000
# Serialized BenchmarkProof. ~100KB for a 300-question slice; the cap leaves
# room for a larger SLICE_SIZE while still bounding ingestion (I2) at the very
# first point miner-controlled bytes enter the process.
PROOF_JSON_MAX_LEN = 4_000_000
HASH_MAX_LEN = 128


class FugalProofSynapse(bt.Synapse):
    """TEE proof submission from miner to validator.

    The validator sends this with epoch_id and nonce filled in. The miner
    returns it with the proof and the head that produced it.
    """
    epoch_id: str = pydantic.Field(default="", max_length=64)
    nonce: str = pydantic.Field(default="", max_length=HASH_MAX_LEN)
    proof_json: str = pydantic.Field(default="", max_length=PROOF_JSON_MAX_LEN)
    head_npz_b64: str = pydantic.Field(default="", max_length=HEAD_B64_MAX_LEN)
    proof_hash: str = pydantic.Field(default="", max_length=HASH_MAX_LEN)
    weights_hash: str = pydantic.Field(default="", max_length=HASH_MAX_LEN)

    def deserialize(self) -> "FugalProofSynapse":
        return self
