"""Proof data models and shared helpers for TEE-attested benchmarks.

BenchmarkProof is what a miner produces inside the TEE after running
the routing benchmark. QuestionResult holds per-question routing
decisions and grades.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from fugal_subnet.routing_protocol import identity


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
    # True for nonce-forced exploration routes. These never count toward the
    # miner's own score — being made to route randomly must not be a penalty —
    # they exist only to sample the model pool for the reference frame.
    is_exploration: bool = False
    # True when the model call did not complete (timeout, connection drop,
    # non-2xx from the proxy). Recorded as its own fact, in the attested proof,
    # because "graded wrong" and "never answered" mean different things to the
    # reference frame: a wrong answer measures the model, a failed call
    # measures the network, and a miner controls its own network. Before this
    # field existed a coalition could lower every model's measured accuracy by
    # dropping connections during exploration only (diligence 2026-10, finding
    # 2). A failed result is always `correct=False` and `verify_proof` holds
    # it to that.
    failed: bool = False

    def hash_payload(self) -> dict:
        """The fields `content_hash` covers for this result.

        `failed` is included ONLY when True. A proof with no failed calls then
        hashes exactly as it did before the field existed, so a miner and a
        validator on either side of this change agree on every honest proof;
        only a proof that actually records a failure needs both sides to know
        the field. Deterministic either way: the same object always yields the
        same payload.
        """
        payload = {
            "question_id": self.question_id,
            "routed_model": self.routed_model,
            "correct": self.correct,
            "cost_usd": self.cost_usd,
            "response_hash": self.response_hash,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "is_exploration": self.is_exploration,
        }
        if self.failed:
            payload["failed"] = True
        return payload

    def to_dict(self) -> dict:
        return {**self.hash_payload(), "failed": self.failed}

    @classmethod
    def from_dict(cls, d: dict) -> QuestionResult:
        return cls(
            question_id=d["question_id"],
            routed_model=d["routed_model"],
            correct=bool(d["correct"]),
            cost_usd=float(d["cost_usd"]),
            response_hash=d["response_hash"],
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            completion_tokens=int(d.get("completion_tokens", 0)),
            is_exploration=bool(d.get("is_exploration", False)),
            failed=bool(d.get("failed", False)),
        )


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
    # WHOSE proof this is. Defaults to "" so that a proof carrying no hotkey is
    # representable — it has to be, or the validator could not reject one — but
    # an empty value is never valid under verification. See verify_proof.
    #
    # Before this existed, a proof was bound to an epoch, a slice, a head and a
    # runtime, and to NO MINER. Any party could take an honest miner's proof off
    # its public axon response, commit the same weights_hash, and present it as
    # their own; verification passed, and what stopped it was dedup plus
    # commit-block seniority — statistical, not cryptographic.
    hotkey: str = ""
    routing_protocol: str = field(default_factory=identity)

    @property
    def scored_results(self) -> list[QuestionResult]:
        """Results the miner is judged on — exploration is excluded."""
        return [r for r in self.results if not r.is_exploration]

    @property
    def exploration_results(self) -> list[QuestionResult]:
        return [r for r in self.results if r.is_exploration]

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.scored_results if r.correct)

    @property
    def n_total(self) -> int:
        return len(self.scored_results)

    @property
    def accuracy(self) -> float:
        scored = self.scored_results
        if not scored:
            return 0.0
        return self.n_correct / len(scored)

    @property
    def scored_cost_usd(self) -> float:
        """Cost of the scored routes only.

        Exploration is a forced cost the miner did not choose, so billing it
        against their thrift would punish them for the subnet's own sampling.
        """
        return sum(r.cost_usd for r in self.scored_results)

    def content_hash(self) -> str:
        """SHA256 of the proof content (excluding attestation).

        This is the value bound into the TDX report_data field so the
        hardware attestation covers the proof's content.
        """
        payload = {
            # First, because it is the thing that was missing. content_hash is
            # bound into the TDX report_data, so including the hotkey here makes
            # the HARDWARE attest whose proof this is. Relay then fails on a
            # signature rather than on a similarity threshold.
            "hotkey": self.hotkey,
            "routing_protocol": self.routing_protocol,
            "epoch_id": self.epoch_id,
            "nonce": self.nonce,
            "questions_hash": self.questions_hash,
            "weights_hash": self.weights_hash,
            "source_hash": self.source_hash,
            "results": [r.hash_payload() for r in self.results],
            "total_cost_usd": self.total_cost_usd,
            "per_model_costs": self.per_model_costs,
            "timestamp": self.timestamp,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "hotkey": self.hotkey,
            "routing_protocol": self.routing_protocol,
            "epoch_id": self.epoch_id,
            "nonce": self.nonce,
            "questions_hash": self.questions_hash,
            "weights_hash": self.weights_hash,
            "source_hash": self.source_hash,
            "results": [r.to_dict() for r in self.results],
            "total_cost_usd": self.total_cost_usd,
            "per_model_costs": self.per_model_costs,
            "attestation_quote": self.attestation_quote.hex(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkProof:
        results = [QuestionResult.from_dict(r) for r in d["results"]]
        return cls(
            # Absent rather than empty for a proof written before this field
            # existed. Either way it fails verification; the distinction only
            # matters for reading old artifacts without an exception.
            hotkey=d.get("hotkey", ""),
            routing_protocol=d.get("routing_protocol", ""),
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
