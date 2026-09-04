"""Proof data models and shared helpers for TEE-attested benchmarks.

BenchmarkProof is what a miner produces inside the TEE after running
the routing benchmark. QuestionResult holds per-question routing
decisions and grades.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def compute_questions_hash(question_ids: list[str]) -> str:
    """Compute the canonical hash for a set of question IDs."""
    canonical = json.dumps(sorted(question_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class QuestionResult:
    question_id: str
    routed_model: str
    correct: bool
    cost_usd: float
    response_hash: str
    # Attested token counts. Carrying them makes every model's counterfactual
    # cost on this question exactly computable from the pinned price table
    # (cost_m = p_in_m * prompt_tokens + p_out_m * completion_tokens), so cost
    # comparisons need no assumed "typical question" constant anywhere.
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class BenchmarkProof:
    epoch_id: str
    nonce: str
    questions_hash: str
    weights_hash: str
    source_hash: str
    results: list[QuestionResult]
    total_cost_usd: float
    per_model_costs: dict[str, float]
    attestation_quote: bytes
    timestamp: float

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return self.n_correct / self.n_total

    def content_hash(self) -> str:
        """SHA256 of the proof content (excluding attestation).

        This is the value bound into the TDX report_data field so the
        hardware attestation covers the proof's content.
        """
        payload = {
            "epoch_id": self.epoch_id,
            "nonce": self.nonce,
            "questions_hash": self.questions_hash,
            "weights_hash": self.weights_hash,
            "source_hash": self.source_hash,
            "results": [
                {
                    "question_id": r.question_id,
                    "routed_model": r.routed_model,
                    "correct": r.correct,
                    "cost_usd": r.cost_usd,
                    "response_hash": r.response_hash,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                }
                for r in self.results
            ],
            "total_cost_usd": self.total_cost_usd,
            "per_model_costs": self.per_model_costs,
            "timestamp": self.timestamp,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "nonce": self.nonce,
            "questions_hash": self.questions_hash,
            "weights_hash": self.weights_hash,
            "source_hash": self.source_hash,
            "results": [
                {
                    "question_id": r.question_id,
                    "routed_model": r.routed_model,
                    "correct": r.correct,
                    "cost_usd": r.cost_usd,
                    "response_hash": r.response_hash,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                }
                for r in self.results
            ],
            "total_cost_usd": self.total_cost_usd,
            "per_model_costs": self.per_model_costs,
            "attestation_quote": self.attestation_quote.hex(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkProof:
        results = [QuestionResult(**r) for r in d["results"]]
        return cls(
            epoch_id=d["epoch_id"],
            nonce=d["nonce"],
            questions_hash=d["questions_hash"],
            weights_hash=d["weights_hash"],
            source_hash=d["source_hash"],
            results=results,
            total_cost_usd=d["total_cost_usd"],
            per_model_costs=d["per_model_costs"],
            attestation_quote=bytes.fromhex(d["attestation_quote"]),
            timestamp=d["timestamp"],
        )
