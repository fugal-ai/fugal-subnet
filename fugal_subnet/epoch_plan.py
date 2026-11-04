"""Which questions an epoch asks: the one place the miner and the validator both call.

The scored slice and the exploration targets are consensus inputs: a miner and a validator
that compute them differently disagree on every question and every proof fails on
questions_hash or on the exploration check, with an error that names the symptom, not the
cause. So both neurons call these two functions with the epoch's protocol version
(slicer.protocol_version_for_boundary), and nothing else selects questions.

v1 delegates to the functions v1 has always used, byte for byte. v2 selects by category
(config.CATEGORY_TARGETS_V2, BENCHMARK_CATEGORY_V2, BENCHMARK_WEIGHT_V2) with the unbiased
apportionment of slicer.select_slice_v2 / select_explore_set_v2. The model list is an argument,
never read from config here, so both sides pass the same versioned list
(routing_protocol.models_for_protocol(version)).
"""
from __future__ import annotations

from fugal_subnet.benchmarks.slicer import select_explore_set_v2, select_slice, select_slice_v2
from fugal_subnet.config import BENCHMARK_CATEGORY_V2, BENCHMARK_WEIGHT_V2, CATEGORY_TARGETS_V2
from fugal_subnet.exploration import assign_explore_models, expected_exploration, explore_nonce


def scored_slice(version: int, nonce: bytes, pool: list[dict], size: int) -> list[dict]:
    """The epoch's scored questions."""
    if version == 1:
        return select_slice(nonce, pool, size)
    if version == 2:
        return select_slice_v2(nonce, pool, size, CATEGORY_TARGETS_V2, BENCHMARK_CATEGORY_V2,
                               bench_weight=BENCHMARK_WEIGHT_V2)
    raise ValueError(f"unknown protocol version {version!r}")


def exploration_plan(version: int, nonce: bytes, pool: list[dict], scored_ids: set[str],
                     models: list[str], size: int) -> dict[str, str]:
    """{question_id: required_model}: the epoch's exploration set (disjoint from the scored
    slice) and the model each question must be sent to."""
    if size <= 0:
        return {}
    if version == 1:
        return expected_exploration(nonce, pool, scored_ids, models, size)
    if version == 2:
        chosen = select_explore_set_v2(explore_nonce(nonce), pool, scored_ids, size,
                                       CATEGORY_TARGETS_V2, BENCHMARK_CATEGORY_V2,
                                       bench_weight=BENCHMARK_WEIGHT_V2)
        return assign_explore_models(nonce, chosen, models)
    raise ValueError(f"unknown protocol version {version!r}")
