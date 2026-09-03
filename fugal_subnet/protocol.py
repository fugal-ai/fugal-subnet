"""Fugal subnet protocol — Synapse definitions for validator-miner communication."""
from typing import List

import bittensor as bt
import pydantic

# 1 MB head, base64-encoded (4/3 overhead) with slack.
HEAD_B64_MAX_LEN = 1_400_000
MODEL_POOL_MAX_LEN = 64          # matches config.HEAD_MAX_MODELS
MODEL_ID_MAX_LEN = 128
HASH_MAX_LEN = 128


class FugalSynapse(bt.Synapse):
    """Head submission from miner to validator.

    The validator sends this synapse with epoch_id and benchmark_hash filled in.
    The miner populates head_npz_b64, model_pool, and head_commit_hash,
    then returns it.
    """
    epoch_id: str = pydantic.Field(default="", max_length=64)
    benchmark_hash: str = pydantic.Field(default="", max_length=HASH_MAX_LEN)
    head_npz_b64: str = pydantic.Field(default="", max_length=HEAD_B64_MAX_LEN)
    model_pool: List[pydantic.constr(max_length=MODEL_ID_MAX_LEN)] = pydantic.Field(
        default_factory=list, max_length=MODEL_POOL_MAX_LEN,
    )
    head_commit_hash: str = pydantic.Field(default="", max_length=HASH_MAX_LEN)

    def deserialize(self) -> "FugalSynapse":
        return self


class FugalProofSynapse(bt.Synapse):
    """TEE proof submission from miner to validator.

    The validator sends this synapse with epoch_id and nonce filled in.
    The miner populates proof_bundle_url, proof_hash, and weights_hash,
    then returns it.
    """
    epoch_id: str = pydantic.Field(default="", max_length=64)
    nonce: str = pydantic.Field(default="", max_length=HASH_MAX_LEN)
    proof_bundle_url: str = pydantic.Field(default="", max_length=2048)
    proof_hash: str = pydantic.Field(default="", max_length=HASH_MAX_LEN)
    weights_hash: str = pydantic.Field(default="", max_length=HASH_MAX_LEN)

    def deserialize(self) -> "FugalProofSynapse":
        return self
