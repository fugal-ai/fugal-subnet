"""Deterministic v2 head validation and canonical-registry evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from fugal_subnet.head_eval import HeadArtifact

MODEL_ID_MAX_BYTES = 128
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]*$")
ROUTING_LAMBDA = 2.0
ROUTING_ROUNDING_DECIMALS = 12


@dataclass
class V2HeadScore:
    accuracy: float
    cost_efficiency: float
    kl_score: float
    routing_decisions: np.ndarray
    routing_model_ids: tuple[str, ...]
    routing_distributions: np.ndarray
    correct_mask: np.ndarray


def validate_head_models(
    head: HeadArtifact,
    active_registry: list[str],
    wire_model_pool: list[str] | None = None,
) -> None:
    """Validate the NPZ model list against the canonical active registry."""
    if not head.models:
        raise ValueError("v2 head must contain at least one model")
    if len(active_registry) != len(set(active_registry)):
        raise ValueError("active registry contains duplicate model ids")
    if len(head.models) != len(set(head.models)):
        raise ValueError("head model ids must be unique")
    if len(head.models) != head.W.shape[0] or len(head.models) != head.b.shape[0]:
        raise ValueError("head model list does not match W/b rows")

    active = set(active_registry)
    for model in head.models:
        if not isinstance(model, str):
            raise ValueError("model ids must be strings")
        if len(model.encode("utf-8")) > MODEL_ID_MAX_BYTES:
            raise ValueError(f"model id exceeds {MODEL_ID_MAX_BYTES} UTF-8 bytes")
        if not MODEL_ID_PATTERN.fullmatch(model):
            raise ValueError(f"invalid model id: {model!r}")
        if model not in active:
            raise ValueError(f"head model is not active in v2 registry: {model}")

    if wire_model_pool is not None and list(wire_model_pool) != list(head.models):
        raise ValueError("wire model_pool must exactly match the NPZ model list and order")


def _rounded_softmax(logits: np.ndarray, decimals: int) -> np.ndarray:
    rounded_logits = np.round(np.asarray(logits, dtype=np.float64), decimals=decimals)
    shifted = rounded_logits - rounded_logits.max()
    exp = np.exp(shifted)
    probabilities = exp / exp.sum()
    probabilities = np.round(probabilities, decimals=decimals)
    probabilities /= probabilities.sum()
    return probabilities


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_safe = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q_safe = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def evaluate_head(
    head: HeadArtifact,
    hidden_states: np.ndarray,
    matrix: np.ndarray,
    active_registry: list[str],
    soft_targets: np.ndarray,
    canonical_costs: dict[str, float],
    *,
    wire_model_pool: list[str] | None = None,
    routing_lambda: float = ROUTING_LAMBDA,
    rounding_decimals: int = ROUTING_ROUNDING_DECIMALS,
) -> V2HeadScore:
    """Evaluate using canonical indexes, independent of head-local row order."""
    validate_head_models(head, active_registry, wire_model_pool)
    if rounding_decimals < 0 or rounding_decimals > 15:
        raise ValueError("rounding_decimals must be between 0 and 15")
    if not math.isfinite(routing_lambda) or routing_lambda < 0:
        raise ValueError("routing_lambda must be finite and non-negative")

    hidden = np.asarray(hidden_states, dtype=np.float32)
    grades = np.asarray(matrix)
    targets = np.asarray(soft_targets, dtype=np.float64)
    n_questions = hidden.shape[0]
    n_models = len(active_registry)
    if hidden.ndim != 2 or hidden.shape[1] != head.W.shape[1]:
        raise ValueError("hidden state shape does not match the head")
    if grades.shape != (n_questions, n_models):
        raise ValueError("matrix shape does not match questions/active registry")
    if targets.shape != grades.shape:
        raise ValueError("soft target shape does not match matrix")
    if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(targets)):
        raise ValueError("hidden states and soft targets must be finite")

    canonical_index = {model: index for index, model in enumerate(active_registry)}
    costs = []
    for model in head.models:
        if model not in canonical_costs:
            raise ValueError(f"canonical price missing for active model {model}")
        cost = float(canonical_costs[model])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"invalid canonical price for active model {model}")
        costs.append(cost)
    head_costs: np.ndarray = np.asarray(costs, dtype=np.float64)

    decisions = np.full(n_questions, -1, dtype=np.int32)
    distributions = np.zeros((n_questions, n_models), dtype=np.float64)
    correct_mask = np.zeros(n_questions, dtype=bool)
    total_kl = 0.0
    total_head_cost = 0.0
    total_oracle_cost = 0.0
    scoreable = 0

    for question_index in range(n_questions):
        logits = head.W @ hidden[question_index] + head.b
        local_distribution = _rounded_softmax(logits, rounding_decimals)
        aligned = distributions[question_index]
        for local_index, model in enumerate(head.models):
            aligned[canonical_index[model]] = local_distribution[local_index]
        aligned /= aligned.sum()

        utility = np.round(
            local_distribution - routing_lambda * head_costs,
            decimals=rounding_decimals,
        )
        best_utility = utility.max()
        tied_local = np.flatnonzero(utility == best_utility)
        selected_local = min(
            tied_local,
            key=lambda index: canonical_index[head.models[int(index)]],
        )
        selected_canonical = canonical_index[head.models[int(selected_local)]]
        decisions[question_index] = selected_canonical

        correct_models = np.flatnonzero(grades[question_index] == 1)
        if len(correct_models) == 0:
            continue
        scoreable += 1
        correct_mask[question_index] = grades[question_index, selected_canonical] == 1
        total_head_cost += canonical_costs[active_registry[selected_canonical]]
        total_oracle_cost += min(
            canonical_costs[active_registry[int(index)]] for index in correct_models
        )
        total_kl += _kl_divergence(targets[question_index], aligned)

    routing_ids = tuple(active_registry[int(index)] for index in decisions)
    if scoreable == 0:
        return V2HeadScore(
            accuracy=0.0,
            cost_efficiency=0.0,
            kl_score=-100.0,
            routing_decisions=decisions,
            routing_model_ids=routing_ids,
            routing_distributions=np.round(distributions, decimals=rounding_decimals),
            correct_mask=correct_mask,
        )

    return V2HeadScore(
        accuracy=round(float(correct_mask.sum()) / scoreable, rounding_decimals),
        cost_efficiency=round(
            min(1.0, total_oracle_cost / max(total_head_cost, 1e-15)),
            rounding_decimals,
        ),
        kl_score=round(-total_kl / scoreable, rounding_decimals),
        routing_decisions=decisions,
        routing_model_ids=routing_ids,
        routing_distributions=np.round(distributions, decimals=rounding_decimals),
        correct_mask=correct_mask,
    )
