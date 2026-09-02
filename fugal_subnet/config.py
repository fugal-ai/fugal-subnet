"""Env-overridable constants for the Fugal subnet."""

import logging
import os

_log = logging.getLogger(__name__)

NETWORK = os.getenv("FUGAL_NETWORK", "test")
NETUID = int(os.getenv("FUGAL_NETUID", "1"))

# --- Epoch ---
EPOCH_INTERVAL = int(os.getenv("FUGAL_EPOCH_INTERVAL", "3600"))
SLICE_SIZE = int(os.getenv("FUGAL_SLICE_SIZE", "300"))

# --- Head constraints ---
HEAD_MAX_BYTES = int(os.getenv("FUGAL_HEAD_MAX_BYTES", str(1 * 1024 * 1024)))  # 1 MB
# Decompressed cap: a valid head is ~130 KB even at 64 models; 8 MB blocks zip bombs.
HEAD_MAX_DECOMPRESSED_BYTES = int(os.getenv("FUGAL_HEAD_MAX_DECOMPRESSED", str(8 * 1024 * 1024)))
HEAD_HIDDEN_DIM = int(os.getenv("FUGAL_HEAD_HIDDEN_DIM", "1024"))  # Qwen3-0.6B
HEAD_MAX_MODELS = int(os.getenv("FUGAL_HEAD_MAX_MODELS", "64"))

# --- Scoring ---
WILSON_CONFIDENCE = float(os.getenv("FUGAL_WILSON_CONFIDENCE", "0.95"))
COMPOSITE_W_ACC = float(os.getenv("FUGAL_W_ACC", "0.55"))
COMPOSITE_W_COST = float(os.getenv("FUGAL_W_COST", "0.35"))
COMPOSITE_W_KL = float(os.getenv("FUGAL_W_KL", "0.10"))

# --- Soft targets ---
SOFT_TARGET_TAU = float(os.getenv("FUGAL_TAU", "1.0"))

# --- Dedup ---
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("FUGAL_DEDUP_THRESHOLD", "0.99"))

# --- Liveness ---
LIVENESS_MAX_MISSED = int(os.getenv("FUGAL_MAX_MISSED_EPOCHS", "3"))

# --- Routing ---
ROUTING_LAMBDA = float(os.getenv("FUGAL_LAMBDA", "2.0"))

# --- API ---
API_TIMEOUT = int(os.getenv("FUGAL_API_TIMEOUT", "180"))
API_MAX_RETRIES = int(os.getenv("FUGAL_API_RETRIES", "3"))
API_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
API_CONCURRENCY = int(os.getenv("FUGAL_API_CONCURRENCY", "4"))

# --- Backbone ---
BACKBONE_MODEL = os.getenv("FUGAL_BACKBONE", "Qwen/Qwen3-0.6B")
ROUTER_SYSTEM_PROMPT = (
    "You are a routing model. Given a question, your hidden state will be used to "
    "predict which language model can best answer it. Read the question carefully."
)

# --- Weight stability ---
MAX_WEIGHT_DELTA = float(os.getenv("FUGAL_MAX_WEIGHT_DELTA", "0.3"))

# --- Validator budget ---
# None means "not explicitly set" — live mode requires a positive value.
_budget_raw = os.getenv("FUGAL_EPOCH_BUDGET")
if _budget_raw:
    EPOCH_BUDGET_USD: float | None = float(_budget_raw)
    if EPOCH_BUDGET_USD <= 0:
        raise ValueError(f"FUGAL_EPOCH_BUDGET must be positive, got {_budget_raw!r}")
else:
    EPOCH_BUDGET_USD = None
MAX_MODEL_POOL = int(os.getenv("FUGAL_MAX_MODEL_POOL", "30"))
MAX_MODELS_PER_MINER = int(os.getenv("FUGAL_MAX_MODELS_PER_MINER", "30"))

# --- Head commitment (anti-copy / anti-overfit) ---
# When enabled, a head is only scored if its SHA256 was committed on-chain
# (pallet_commitments) at or before the epoch boundary block. This makes
# head-copying and slice-overfitting detectable: any head changed after the
# epoch nonce is knowable has a commitment block past the boundary.
REQUIRE_COMMITMENT = os.getenv("FUGAL_REQUIRE_COMMITMENT", "1") not in ("0", "false", "False")

# --- Miner axon access control ---
# Minimum stake (TAO) a querying hotkey needs when it lacks a validator permit.
# 0 admits any registered hotkey (safe default for testnets); raise on finney.
MIN_VALIDATOR_STAKE = float(os.getenv("FUGAL_MIN_VALIDATOR_STAKE", "0"))

# --- Matrix caching ---
CACHE_STALENESS_TTL = int(os.getenv("FUGAL_CACHE_TTL", str(14 * 86400)))  # 2 weeks

# --- Grader ---
EXEC_TIMEOUT = int(os.getenv("FUGAL_EXEC_TIMEOUT", "10"))
EXEC_MAX_BYTES = int(os.getenv("FUGAL_EXEC_MAX_BYTES", str(512 * 1024)))  # 512 KB

# --- Model cost cap ---
MAX_MODEL_COST_PER_QUERY = float(os.getenv("FUGAL_MAX_MODEL_COST", "0.10"))

# --- Validation ---
_w_sum = COMPOSITE_W_ACC + COMPOSITE_W_COST + COMPOSITE_W_KL
if abs(_w_sum - 1.0) > 1e-6:
    _log.warning("Composite weights sum to %.4f, not 1.0 — scores will be scaled", _w_sum)
