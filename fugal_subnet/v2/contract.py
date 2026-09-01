"""Runtime binding between v2 manifest values and executable constants."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from fugal_subnet.v2 import backbone, dedup, head_eval, matrix, protocol, reports
from fugal_subnet.v2.rewards import MAX_WEIGHT_DELTA, WEIGHT_PRECISION
from fugal_subnet.v2.scoring import (
    COMPOSITE_ACCURACY,
    COMPOSITE_COST,
    COMPOSITE_KL,
    SCORE_ROUNDING_DECIMALS,
)
from fugal_subnet.v2.soft_targets import ROUNDING_DECIMALS, TAU
from fugal_subnet.v2.validator_state import LIVENESS_MISS_LIMIT


class ContractMismatch(RuntimeError):
    pass


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str):
        raise ContractMismatch(f"{label} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ContractMismatch(f"{label} is invalid") from exc
    if not parsed.is_finite() or str(parsed) != value:
        raise ContractMismatch(f"{label} is not canonical")
    return parsed


def verify_executable_contract(consensus: dict) -> None:
    """Require every manifest parameter consumed through a code constant."""
    committee = consensus.get("committee")
    if not isinstance(committee, dict):
        raise ContractMismatch("committee contract is missing")
    fixed_committee = {
        "maximum_builders": 5,
        "minimum_reports": 3,
        "maximum_questions": matrix.MAX_QUESTIONS,
        "maximum_models": matrix.MAX_MODELS,
        "maximum_report_chunks": protocol.REPORT_MAX_CHUNKS,
        "maximum_chunk_bytes": protocol.REPORT_CHUNK_BYTES,
        "maximum_response_bytes": matrix.MAX_RESPONSE_BYTES,
        "maximum_output_tokens": matrix.MAX_OUTPUT_TOKENS,
        "maximum_retries": matrix.MAX_RETRIES,
    }
    if any(committee.get(key) != value for key, value in fixed_committee.items()):
        raise ContractMismatch("committee bounds differ from executable constants")
    if reports.MAX_QUESTIONS != matrix.MAX_QUESTIONS or reports.MAX_MODELS != matrix.MAX_MODELS:
        raise ContractMismatch("matrix and report shape bounds differ")
    if reports.MAX_RESPONSE_BYTES != matrix.MAX_RESPONSE_BYTES:
        raise ContractMismatch("matrix and report response bounds differ")
    if reports.MAX_REPORT_BYTES != protocol.REPORT_MAX_BYTES:
        raise ContractMismatch("report artifact bounds differ")
    slice_size = committee.get("slice_size")
    concurrency = committee.get("api_concurrency")
    if (
        not isinstance(slice_size, int)
        or isinstance(slice_size, bool)
        or not 0 < slice_size <= matrix.MAX_QUESTIONS
        or not isinstance(concurrency, int)
        or isinstance(concurrency, bool)
        or not 0 < concurrency <= matrix.MAX_CONCURRENCY
    ):
        raise ContractMismatch("matrix schedule bounds differ from executable limits")

    rounding = consensus.get("rounding")
    if not isinstance(rounding, dict) or set(rounding) != {
        "embedding_decimals", "routing_decimals", "score_decimals",
        "weight_decimals", "tie_break", "evaluation",
    }:
        raise ContractMismatch("rounding/evaluation contract schema differs")
    if (
        rounding["embedding_decimals"] != backbone.ROUNDING_DECIMALS
        or rounding["routing_decimals"] != head_eval.ROUTING_ROUNDING_DECIMALS
        or rounding["routing_decimals"] != ROUNDING_DECIMALS
        or rounding["score_decimals"] != SCORE_ROUNDING_DECIMALS
        or rounding["weight_decimals"] != WEIGHT_PRECISION
        or rounding["tie_break"] != "canonical_model_then_uid_hotkey"
    ):
        raise ContractMismatch("rounding contract differs from executable constants")

    evaluation = rounding["evaluation"]
    expected_keys = {
        "accuracy_weight", "cost_weight", "kl_weight", "routing_lambda",
        "soft_target_tau", "dedup_decision_threshold",
        "dedup_distribution_threshold", "max_weight_delta",
        "liveness_miss_limit",
    }
    if not isinstance(evaluation, dict) or set(evaluation) != expected_keys:
        raise ContractMismatch("evaluation contract schema differs")
    expected_decimals = {
        "accuracy_weight": Decimal(str(COMPOSITE_ACCURACY)),
        "cost_weight": Decimal(str(COMPOSITE_COST)),
        "kl_weight": Decimal(str(COMPOSITE_KL)),
        "routing_lambda": Decimal(str(head_eval.ROUTING_LAMBDA)),
        "soft_target_tau": Decimal(str(TAU)),
        "dedup_decision_threshold": Decimal(str(dedup.DECISION_THRESHOLD)),
        "dedup_distribution_threshold": Decimal(str(dedup.DISTRIBUTION_THRESHOLD)),
        "max_weight_delta": MAX_WEIGHT_DELTA,
    }
    if any(
        _decimal(evaluation[key], key) != expected
        for key, expected in expected_decimals.items()
    ) or evaluation["liveness_miss_limit"] != LIVENESS_MISS_LIMIT:
        raise ContractMismatch("evaluation parameters differ from executable constants")
