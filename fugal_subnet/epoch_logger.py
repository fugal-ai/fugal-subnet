"""Structured epoch logging — JSON artifacts for auditability.

Each epoch produces a structured log entry containing timing, scores, weights,
and anomaly flags. Logs are append-only JSONL for easy ingestion by monitoring
tools. Separate from commit-reveal (which proves integrity); this captures
operational telemetry.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

EPOCH_LOG_DIR = os.getenv("FUGAL_EPOCH_LOG_DIR", "results/epoch_logs")


@dataclass
class EpochLog:
    epoch_id: str
    block_hash: str
    timestamp: float
    n_questions: int
    n_miners_queried: int
    n_heads_valid: int
    n_heads_invalid: int
    commit_hash: str = ""
    reveal_verified: bool = False
    scores: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    dedup_disqualified: list = field(default_factory=list)
    weight_capped: bool = False
    set_weights_success: bool = False
    set_weights_msg: str = ""
    # Whether the weights were read back off chain and matched what was
    # submitted. set_weights_success only says the extrinsic was included.
    weights_confirmed_on_chain: bool = False
    weights_confirm_detail: str = ""
    anomalies: list = field(default_factory=list)
    duration_s: float = 0.0


class EpochTimer:
    """Context manager for timing epoch phases."""

    def __init__(self):
        self._start = time.monotonic()
        self.phases: dict[str, float] = {}
        self._phase_start: float | None = None
        self._current_phase: str | None = None

    def start_phase(self, name: str):
        if self._current_phase:
            self.end_phase()
        self._current_phase = name
        self._phase_start = time.monotonic()

    def end_phase(self):
        if self._current_phase and self._phase_start is not None:
            self.phases[self._current_phase] = time.monotonic() - self._phase_start
        self._current_phase = None
        self._phase_start = None

    @property
    def total_s(self) -> float:
        return time.monotonic() - self._start


def detect_anomalies(
    scores: dict[int, dict],
    weights: dict[int, float],
    n_miners_queried: int,
    n_heads_valid: int,
) -> list[str]:
    """Flag suspicious patterns in epoch results."""
    anomalies = []

    if n_miners_queried > 0 and n_heads_valid == 0:
        anomalies.append("no_valid_heads: all miners failed to respond")

    if n_miners_queried > 1 and n_heads_valid == 1:
        anomalies.append("single_responder: only 1 of %d miners responded" % n_miners_queried)

    if scores:
        accs = [s.get("acc", 0) for s in scores.values()]
        if all(a > 0.95 for a in accs) and len(accs) > 1:
            anomalies.append("suspiciously_high_accuracy: all miners >95%% accuracy")
        if all(a < 0.1 for a in accs) and len(accs) > 0:
            anomalies.append("universally_low_accuracy: all miners <10%% — possible bad benchmark slice")

    weight_vals = [w for w in weights.values() if w > 0]
    if len(weight_vals) > 1:
        max_w = max(weight_vals)
        if max_w > 0.8:
            anomalies.append("weight_concentration: single miner has %.1f%% of weight" % (max_w * 100))

    return anomalies


def write_epoch_log(log: EpochLog):
    """Append epoch log as a single JSONL line."""
    os.makedirs(EPOCH_LOG_DIR, exist_ok=True)
    log_path = os.path.join(EPOCH_LOG_DIR, "epochs.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(asdict(log), default=str) + "\n")
    logger.info("Epoch %s logged (%.1fs, %d anomalies)",
                log.epoch_id, log.duration_s, len(log.anomalies))


def read_epoch_logs(limit: int = 100) -> list[dict]:
    """Read recent epoch logs for analysis."""
    log_path = os.path.join(EPOCH_LOG_DIR, "epochs.jsonl")
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines[-limit:]]
