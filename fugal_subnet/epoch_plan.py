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

from fugal_subnet import config
from fugal_subnet.benchmarks.slicer import (
    PoolIndex,
    select_explore_set_v2,
    select_private_explore,
    select_slice,
    select_slice_v2,
)
from fugal_subnet.config import BENCHMARK_CATEGORY_V2, BENCHMARK_WEIGHT_V2, CATEGORY_TARGETS_V2
from fugal_subnet.exploration import (
    assign_explore_models,
    expected_exploration,
    explore_nonce,
    private_nonce,
)


def scored_slice(version: int, nonce: bytes, pool: list[dict], size: int) -> list[dict]:
    """The epoch's scored questions."""
    if version == 1:
        return select_slice(nonce, pool, size)
    if version == 2:
        return select_slice_v2(nonce, pool, size, CATEGORY_TARGETS_V2, BENCHMARK_CATEGORY_V2,
                               bench_weight=BENCHMARK_WEIGHT_V2)
    if version == 3:
        return select_slice_v2(nonce, pool, size, config.CATEGORY_TARGETS_V3,
                               config.BENCHMARK_CATEGORY_V3,
                               bench_weight=config.BENCHMARK_WEIGHT_V3)
    raise ValueError(f"unknown protocol version {version!r}")


def exploration_plan(version: int, nonce: bytes, pool: list[dict], scored_ids: set[str],
                     models: list[str], size: int, hotkey: str | None = None,
                     index: PoolIndex | None = None,
                     shares: dict[str, int] | None = None) -> dict[str, str]:
    """{question_id: required_model}: the epoch's exploration set (disjoint from the scored
    slice) and the model each question must be sent to.

    v1 and v2: one set for every miner; `hotkey` is not read. v3: the shared set (every
    miner answers it) plus, when `hotkey` is given, that miner's private set. The miner
    passes its own hotkey; the validator calls this once per miner it verifies, so the
    proof check stays an exact match of what was assigned.

    v3 draws each question's entry by `shares` (explore_shares: cost-aware, the
    same numbers on every node); None draws every entry equally.
    """
    if version == 3:
        shared, private = exploration_plan_v3(nonce, pool, scored_ids, models, hotkey, index,
                                              shares=shares)
        return {**shared, **private}
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


def pool_index_v3(pool: list[dict]) -> PoolIndex:
    """The v3 pool indexed for private draws. Build it once per pool: it is what makes a
    per-miner plan cost a few dozen HMACs instead of one per question."""
    return PoolIndex(pool, config.BENCHMARK_CATEGORY_V3, config.BENCHMARK_WEIGHT_V3)


def explore_shares(version: int, models: list[str]) -> dict[str, int] | None:
    """The epoch's exploration sampling shares: v3's cost-aware shares from the
    pinned menu and token manifest (explore_weights.plan_shares), None before v3.
    The miner and the validator both call this, so they draw the same entries."""
    if version < 3:
        return None
    from fugal_subnet import explore_weights
    return explore_weights.plan_shares(list(models))


def exploration_plan_v3(nonce: bytes, pool: list[dict], scored_ids: set[str], models: list[str],
                        hotkey: str | None, index: PoolIndex | None = None,
                        shares: dict[str, int] | None = None,
                        ) -> tuple[dict[str, str], dict[str, str]]:
    """(shared, private) exploration for one miner in a v3 epoch.

    - **Shared**: `EXPLORE_SHARED_SIZE` questions and their models, chosen as v2 chooses
      its whole set, the same for every miner. Miners answering the same pairs is what
      the cross-miner agreement check reads.
    - **Private**: `EXPLORE_PRIVATE_SIZE` questions drawn for this hotkey
      (slicer.select_private_explore), disjoint from the scored slice and the shared set,
      each sent to an entry drawn by `shares` (explore_weights.assign_by_share:
      systematic sampling, so equal shares are a round-robin) with the same
      per-miner nonce. Empty when `hotkey` is None (the shared half alone).
    """
    from fugal_subnet.explore_weights import assign_by_share
    if shares is None:
        shares = dict.fromkeys(models, 1)
    elif set(shares) != set(models):
        raise ValueError("exploration shares must name exactly the epoch's models")
    shared: dict[str, str] = {}
    if config.EXPLORE_SHARED_SIZE > 0:
        chosen = select_explore_set_v2(explore_nonce(nonce), pool, scored_ids,
                                       config.EXPLORE_SHARED_SIZE, config.CATEGORY_TARGETS_V3,
                                       config.BENCHMARK_CATEGORY_V3,
                                       bench_weight=config.BENCHMARK_WEIGHT_V3)
        shared = assign_by_share(nonce, chosen, shares)
    private: dict[str, str] = {}
    if hotkey and config.EXPLORE_PRIVATE_SIZE > 0:
        mine = private_nonce(nonce, hotkey)
        chosen = select_private_explore(mine, index or pool_index_v3(pool),
                                        set(scored_ids) | set(shared),
                                        config.EXPLORE_PRIVATE_SIZE, config.CATEGORY_TARGETS_V3)
        private = assign_by_share(mine, chosen, shares)
    return shared, private
