"""Backbone forward pass: Qwen3-0.6B hidden state extraction.

Shared by both the validator (head evaluation) and the trainer (head training).
Frozen backbone, mean-pooling, L2-normalization.
"""
from __future__ import annotations

import ctypes
import gc
import logging
import os

# isort: off
# Order is load-bearing. numpy and torch both read the CPU-dispatch env vars
# once, at import, so this must precede both or the pinning silently does
# nothing. check_safety_invariants.py enforces it.
import fugal_subnet.determinism  # noqa: F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
# isort: on

from fugal_subnet.config import (  # noqa: E402
    BACKBONE_MODEL,
)
from fugal_subnet.vendor import success_contract as contract

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple] = {}
_determinism_configured = False


def configure_determinism() -> None:
    """Deterministic algorithms, one inter-op thread, and the intra-op thread
    count FUGAL_BACKBONE_THREADS asks for (default 1). Embeddings are miner-side
    only — see fugal_subnet/determinism.py for why the count is a knob."""
    global _determinism_configured
    if _determinism_configured:
        return
    from fugal_subnet.determinism import backbone_threads

    torch.set_num_threads(backbone_threads())
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("PyTorch inter-op threads were initialized above one")
    torch.use_deterministic_algorithms(True)
    _determinism_configured = True
    logger.info(
        "Backbone determinism configured: capability=%s, threads=%d",
        os.environ.get("ATEN_CPU_CAPABILITY", "unset"),
        torch.get_num_threads(),
    )


def get_backbone(
    model_name: str = BACKBONE_MODEL,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple:
    """Load the pinned CPU float32 reference model, checking actual local bytes."""
    if device != "cpu" or dtype not in (None, torch.float32):
        raise ValueError("success embedding profile requires CPU float32")
    if model_name == contract.MODEL_ID:
        from huggingface_hub import snapshot_download
        model_name = snapshot_download(contract.MODEL_ID, revision=contract.REVISION)
    cache_key = f"{model_name}:{contract.PROFILE_ID}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    contract.check_backbone(model_name)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32, local_files_only=True,
                                     trust_remote_code=False).eval()
    _model_cache[cache_key] = (tokenizer, model)
    return tokenizer, model


def compute_hidden_states(
    prompts: list[str],
    model_name: str = BACKBONE_MODEL,
    device: str = "cpu",
    batch_size: int = 2,
    max_length: int = 2048,
) -> np.ndarray:
    """Standalone, mask-mean-L2 embeddings under the shared success profile."""
    if max_length != 2048:
        raise ValueError("benchmark embeddings require the 2048-token profile")
    configure_determinism()
    tokenizer, model = get_backbone(model_name, device)
    return contract.embed(tokenizer, model, prompts, batch_size)


def release_backbone():
    """Release the cached backbone and return its memory to the OS.

    Dropping the cache is not enough on CPU. glibc's allocator keeps freed
    blocks in its arenas rather than returning them, so RSS stays high long
    after the model is unreachable: measured 3206 MB while embedding, 2578 MB
    after clearing the cache, and 751 MB only once the arenas are trimmed.

    That 1.8 GB matters. A miner holds the backbone only to embed the pool
    once at startup and then never needs it again, so without the trim every
    miner idles for the rest of its life holding memory it cannot use — and on
    a machine running several, the kernel starts killing them (observed:
    SIGKILL on the third concurrent miner).
    """
    global _model_cache
    _model_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()

    # glibc only; a no-op elsewhere. Not required for correctness.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

    logger.info("Backbone released")
