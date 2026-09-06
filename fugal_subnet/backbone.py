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
import torch.nn.functional as F  # noqa: E402
# isort: on

from fugal_subnet.config import (  # noqa: E402
    BACKBONE_MODEL,
    HEAD_HIDDEN_DIM,
    ROUTER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple] = {}
_determinism_configured = False


def configure_determinism() -> None:
    """Lock down torch thread counts and deterministic mode for reproducible embeddings."""
    global _determinism_configured
    if _determinism_configured:
        return
    torch.set_num_threads(1)
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
    """Load (or return cached) backbone model and tokenizer.

    dtype defaults to float16 on CUDA and float32 on CPU (fp16 matmuls are
    unsupported/slow on most CPUs). trust_remote_code stays False — Qwen3 is
    natively supported by transformers, and a validator must never execute
    code fetched from a model hub.

    THE CUDA DEFAULT IS RIGHT FOR TENSOR-CORE HARDWARE AND WRONG BELOW IT, and
    it fails silently in both directions. Measured on a consumer card without
    tensor cores: fp32 ran at 2.66 TFLOPS against fp16's 0.60 — **4.4x faster
    in the direction nobody expects** — and the batch size that fits differs by
    4x between the two, so a miner sizing a box from this default gets both the
    speed and the memory wrong.

    Neither error announces itself. There is no warning and no OOM; the job
    simply runs, and past ~95% VRAM the WSL2 driver spills to host memory and
    throughput collapses. Measured: an embedding pass at batch 32 sat at
    5,864 MiB of 6,144 and ran below 1 prompt/s; at batch 8 it used 3,428 MiB
    and ran at ~29 — 30x, same work, same card.

    That is the same failure shape as a miner starting a 13-hour embedding job
    because FUGAL_BENCHMARK_POOL was missing: healthy-looking and hours wrong.
    So if embedding the pool is taking hours, check VRAM headroom and try
    `dtype=torch.float32` before concluding the machine is too small.

    Left as the default rather than changed, because it IS correct on the
    datacentre cards a serious miner would rent, and because dtype is a
    miner-side performance choice — it does not touch consensus, which is
    computed from the routing decisions in a proof and not from embeddings.
    """
    cache_key = f"{model_name}:{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from transformers import AutoModel, AutoTokenizer

    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

    logger.info("Loading backbone: %s on %s (%s)", model_name, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=False,
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _model_cache[cache_key] = (tokenizer, model)
    return tokenizer, model


def compute_hidden_states(
    prompts: list[str],
    model_name: str = BACKBONE_MODEL,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    """Extract hidden states from prompts via frozen backbone.

    Returns (N, hidden_dim) float32 array, mean-pooled and L2-normalized.
    """
    configure_determinism()
    tokenizer, model = get_backbone(model_name, device)

    all_hidden = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        full = [f"{ROUTER_SYSTEM_PROMPT}\n\n{p}" for p in batch]

        inputs = tokenizer(
            full, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        hidden = outputs.last_hidden_state.float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        pooled = F.normalize(pooled, p=2, dim=1)
        all_hidden.append(pooled.cpu().numpy())

        if (i // batch_size) % 20 == 0 and i > 0:
            logger.info("  Backbone: %d / %d prompts", i, len(prompts))

    result = np.concatenate(all_hidden, axis=0).astype(np.float32)
    assert result.shape[1] == HEAD_HIDDEN_DIM, \
        f"Backbone hidden dim {result.shape[1]} != config {HEAD_HIDDEN_DIM}"
    return result


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
