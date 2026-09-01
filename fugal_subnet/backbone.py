"""Backbone forward pass: Qwen3-0.6B hidden state extraction.

Shared by both the validator (head evaluation) and the trainer (head training).
Frozen backbone, mean-pooling, L2-normalization.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from fugal_subnet.config import BACKBONE_MODEL, HEAD_HIDDEN_DIM, ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple] = {}


def get_backbone(
    model_name: str = BACKBONE_MODEL,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> tuple:
    """Load (or return cached) backbone model and tokenizer.

    dtype defaults to float16 on CUDA and float32 on CPU (fp16 matmuls are
    unsupported/slow on most CPUs). trust_remote_code stays False — Qwen3 is
    natively supported by transformers, and a validator must never execute
    code fetched from a model hub.
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
        model_name, torch_dtype=dtype,
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _model_cache[cache_key] = (tokenizer, model)
    return tokenizer, model


def compute_hidden_states(
    prompts: list[str],
    model_name: str = BACKBONE_MODEL,
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    """Extract hidden states from prompts via frozen backbone.

    Returns (N, hidden_dim) float32 array, mean-pooled and L2-normalized.
    """
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
    """Free GPU memory by clearing the cached model."""
    global _model_cache
    _model_cache.clear()
    torch.cuda.empty_cache()
    logger.info("Backbone released from GPU")
