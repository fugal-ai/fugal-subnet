"""Pinned Linux CPU-float32 backbone used by v2 validators and trainers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

# PyTorch selects CPU kernels from the host's widest SIMD extension, and the
# resulting float32 reduction order changes the embedding bits. An AVX-512 host
# and an AVX2 host therefore disagree, which would make honest validators reject
# each other's reveals. Three dispatchers have to be pinned, not one:
# ATEN_CPU_CAPABILITY covers ATen's own vectorized kernels, while the GEMMs run
# through MKL and oneDNN, which dispatch on the real CPU independently.
# MKL_CBWR is Intel's Conditional Bitwise Reproducibility switch and is the
# mechanism actually intended for this. All of it must stay above the torch
# import, because each is read once when the dispatch tables initialize.
CPU_CAPABILITY = "avx2"
DETERMINISM_ENV = {
    "ATEN_CPU_CAPABILITY": CPU_CAPABILITY,
    "MKL_CBWR": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
for _key, _value in DETERMINISM_ENV.items():
    os.environ.setdefault(_key, _value)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TOKENIZER_REVISION = MODEL_REVISION
SYSTEM_PROMPT = (
    "You are a routing model. Given a question, your hidden state will be used to "
    "select the best language model to answer it."
)
HIDDEN_DIM = 1024
MAX_LENGTH = 512
BATCH_SIZE = 8
ROUNDING_DECIMALS = 8
GOLDEN_PROMPTS = (
    "What is 2 + 2?",
    "Write a Python function that reverses a Unicode string.",
    "Explain why the sky appears blue in one sentence.",
    "Return a JSON object with keys alpha and beta.",
)

_lock = threading.Lock()
_cache: tuple[Any, Any] | None = None
_configured = False


class BackboneEnvironmentError(RuntimeError):
    """The host cannot execute the consensus-defined v2 backbone policy."""


@dataclass(frozen=True)
class BackboneSpec:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    tokenizer_revision: str = TOKENIZER_REVISION
    device: str = "cpu"
    dtype: str = "float32"
    max_length: int = MAX_LENGTH
    batch_size: int = BATCH_SIZE
    rounding_decimals: int = ROUNDING_DECIMALS
    cpu_capability: str = CPU_CAPABILITY

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.__dict__, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def require_supported_host() -> None:
    machine = platform.machine().lower()
    if platform.system() != "Linux" or machine not in {"x86_64", "amd64"}:
        raise BackboneEnvironmentError(
            "v2 consensus backbone requires Linux x86-64 CPU execution"
        )


def configure_determinism() -> None:
    """Apply the process-level deterministic settings required by v2."""
    global _configured
    require_supported_host()
    with _lock:
        if _configured:
            return
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch forbids changing this after parallel work begins. Accept
            # only an already-canonical value; otherwise this process is unsafe.
            if torch.get_num_interop_threads() != 1:
                raise BackboneEnvironmentError(
                    "PyTorch inter-op threads were initialized above one"
                )
        torch.use_deterministic_algorithms(True)
        # Fail closed rather than diverge silently: if torch already selected a
        # wider kernel set (another module imported it before this one, or the
        # environment overrode the pin) the embeddings will not match consensus.
        active = str(torch.backends.cpu.get_cpu_capability()).strip().upper()
        if active != CPU_CAPABILITY.upper():
            raise BackboneEnvironmentError(
                "v2 consensus requires CPU capability "
                f"{CPU_CAPABILITY.upper()}, but PyTorch selected {active}. Set "
                f"ATEN_CPU_CAPABILITY={CPU_CAPABILITY} before importing torch."
            )
        _configured = True


def get_backbone() -> tuple[Any, Any]:
    """Load exactly one pinned model/tokenizer pair on CPU as float32."""
    global _cache
    configure_determinism()
    with _lock:
        if _cache is not None:
            return _cache
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading pinned v2 backbone %s@%s on CPU float32", MODEL_ID, MODEL_REVISION)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=TOKENIZER_REVISION,
            trust_remote_code=False,
        )
        model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to("cpu").eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _cache = (tokenizer, model)
        return _cache


def compute_hidden_states(prompts: list[str]) -> np.ndarray:
    """Return canonical rounded embeddings for the exact ordered prompt list."""
    if not isinstance(prompts, list) or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a list of strings")
    if not prompts:
        return np.empty((0, HIDDEN_DIM), dtype=np.float32)
    tokenizer, model = get_backbone()
    outputs = []
    for offset in range(0, len(prompts), BATCH_SIZE):
        full = [f"{SYSTEM_PROMPT}\n\n{prompt}" for prompt in prompts[offset:offset + BATCH_SIZE]]
        inputs = tokenizer(
            full,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        with torch.inference_mode():
            result = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).to(dtype=torch.float32)
        hidden = result.last_hidden_state.to(dtype=torch.float32)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        normalized = F.normalize(pooled, p=2, dim=1)
        batch = normalized.cpu().numpy().astype(np.float32, copy=False)
        outputs.append(np.round(batch, decimals=ROUNDING_DECIMALS).astype(np.float32))
    hidden_states = np.concatenate(outputs, axis=0)
    if hidden_states.shape != (len(prompts), HIDDEN_DIM):
        raise RuntimeError("pinned backbone returned an unexpected hidden-state shape")
    if not np.all(np.isfinite(hidden_states)):
        raise RuntimeError("pinned backbone returned non-finite hidden states")
    return hidden_states


def verify_backbone_golden(
    *, expected_prompts_sha256: str, expected_embeddings_sha256: str,
) -> str:
    """Run the real pinned model and require the manifest's rounded vector."""
    prompt_bytes = json.dumps(
        list(GOLDEN_PROMPTS),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
    if prompt_hash != expected_prompts_sha256:
        raise BackboneEnvironmentError("backbone golden prompts differ from manifest")
    embeddings = compute_hidden_states(list(GOLDEN_PROMPTS))
    actual = hashlib.sha256(embeddings.tobytes(order="C")).hexdigest()
    if actual != expected_embeddings_sha256:
        # Name the host and the actual digest: this failure is nearly always a
        # dispatcher that ignored the pins above, and the message is the only
        # evidence available from another machine's CI run.
        active = str(torch.backends.cpu.get_cpu_capability())
        pins = ", ".join(f"{key}={os.environ.get(key)}" for key in sorted(DETERMINISM_ENV))
        raise BackboneEnvironmentError(
            "pinned backbone embedding vector differs from manifest: expected "
            f"{expected_embeddings_sha256}, got {actual}; torch capability "
            f"{active}, threads {torch.get_num_threads()}, pins {pins}"
        )
    return actual


def release_backbone() -> None:
    global _cache
    with _lock:
        _cache = None
