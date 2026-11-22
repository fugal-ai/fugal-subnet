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

v4 draws on the pinned mix (mix_v4: routing_data/pool_mix_v4.json, built by
scripts/build_mix_v4.py), and an epoch's plan is built by plan_v4 alone. Nothing
on the v1-v3 path reads it.
"""
from __future__ import annotations

import functools
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
    coverage_nonce,
    expected_exploration,
    explore_nonce,
    private_nonce,
    private_parts,
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
    if version == 4:
        mix = mix_v4()
        if size != mix.slice_size:
            raise ValueError(f"the v4 mix is pinned for a {mix.slice_size}-question slice, "
                             f"not {size} (its cycle cap depends on the slice)")
        return select_slice_v2(nonce, pool, size, mix.targets, mix.bench_category,
                               bench_weight=mix.bench_weight)
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

    v4: `shares` is the epoch's explore_weights.SharesV4 (explore_shares_v4) and
    is required; `models` must be the entries it draws over. The plan is the
    shared set, then the miner's cost-aware and coverage parts
    (exploration_plan_v4).
    """
    if version == 4:
        from fugal_subnet.explore_weights import SharesV4
        if not isinstance(shares, SharesV4):
            raise ValueError("a v4 exploration plan needs the epoch's v4 shares (explore_shares_v4)")
        if sorted(models) != shares.models():
            raise ValueError("exploration shares must name exactly the epoch's models")
        shared, cost_part, coverage = exploration_plan_v4(nonce, pool, scored_ids, shares, hotkey,
                                                          index)
        return {**shared, **cost_part, **coverage}
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


# --- protocol v4 (docs/decisions/LIVE_MENU_V4.md, section 8) ---------------------
#
# Reached only with version 4, which no release activates yet.


MIX_V4_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing_data",
                           "pool_mix_v4.json")
MIX_V4_FORMAT = "pool-mix-v4-v1"
_PPM = 1_000_000


@dataclass(frozen=True)
class MixV4:
    """The pinned v4 question mix (routing_data/pool_mix_v4.json, built by
    scripts/build_mix_v4.py; docs/decisions/LIVE_MENU_V4.md, section 10).
    Integers only, so every node draws on the same exact proportions:

    - `targets`: category -> integer per-cent of the slice (sums to 100);
    - `bench_category`: question set -> category;
    - `bench_weight`: question set -> millionths of the slice, summing within
      a category to its target; the slicer's within-category weights;
    - `slice_size`: the scored slice the mix was built for (its cycle cap).
    """

    targets: Mapping[str, int]
    bench_category: Mapping[str, str]
    bench_weight: Mapping[str, int]
    slice_size: int

    def menu_mix(self) -> tuple[tuple[str, int], ...]:
        """The scored category mix as sorted (category, integer weight)
        pairs, the form menu_state.MenuParams.mix takes."""
        return tuple(sorted(self.targets.items()))


def _is_count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


@functools.lru_cache(maxsize=1)
def mix_v4() -> MixV4:
    """The pinned v4 mix, read once and checked: integer per-cent targets
    summing to 100, each at least the floor and on the step; each category's
    set weights summing to its target; no set above its cycle cap. The bytes
    are pinned by scripts/check_safety_invariants.py."""
    with open(MIX_V4_PATH, "rb") as f:
        doc = json.loads(f.read())
    if doc.get("format") != MIX_V4_FORMAT:
        raise ValueError(f"{MIX_V4_PATH} is not a {MIX_V4_FORMAT} file")
    targets = doc["category_targets_pct"]
    sets = doc["benchmarks"]
    slice_size, cycle = doc["slice_size"], doc["cycle_epochs"]
    if (not all(_is_count(v) for v in (slice_size, cycle, *targets.values())) or slice_size == 0
            or sum(targets.values()) != 100
            or any(v < config.MIX_FLOOR_PCT_V4 or v % config.MIX_STEP_PCT_V4 for v in targets.values())):
        raise ValueError(f"{MIX_V4_PATH}: the targets are not integer per-cents on the step, at "
                         "least the floor, summing to 100")
    per_cat = dict.fromkeys(targets, 0)
    for b, row in sets.items():
        if (row["category"] not in targets or not _is_count(row["weight_ppm"])
                or not _is_count(row["pool"])
                or row["weight_ppm"] * slice_size * cycle > row["pool"] * _PPM):
            raise ValueError(f"{MIX_V4_PATH}: set {b} has no category, no count or is above its "
                             "cycle cap")
        per_cat[row["category"]] += row["weight_ppm"]
    if per_cat != {c: v * _PPM // 100 for c, v in targets.items()}:
        raise ValueError(f"{MIX_V4_PATH}: the set weights do not sum to the category targets")
    return MixV4(
        targets=MappingProxyType(dict(sorted(targets.items()))),
        bench_category=MappingProxyType({b: sets[b]["category"] for b in sorted(sets)}),
        bench_weight=MappingProxyType({b: sets[b]["weight_ppm"] for b in sorted(sets)}),
        slice_size=slice_size)


def _mix_v4():
    """(category targets, benchmark -> category, benchmark weights) the v4
    scored slice and exploration sets are drawn on: the pinned mix."""
    mix = mix_v4()
    return mix.targets, mix.bench_category, mix.bench_weight


def pool_index_v4(pool: list[dict]) -> PoolIndex:
    """The v4 pool indexed for private draws (build it once per pool)."""
    _, bench_category, bench_weight = _mix_v4()
    return PoolIndex(pool, bench_category, bench_weight)


def menu_step_params_v4(min_category_accuracy_bp: int):
    """The derivation step's production parameters (menu_state.MenuParams):
    token means and the proposal gates weighted to the pinned v4 mix
    (`mix_v4().menu_mix()`), every other constant menu_state's compiled one.
    The category floor (PROPOSAL_MIN_CATEGORY_ACCURACY, section 5) is not
    fixed by the design yet, so the caller names it."""
    from fugal_subnet import menu_state
    return menu_state.MenuParams(mix=mix_v4().menu_mix(),
                                 min_category_accuracy_bp=int(min_category_accuracy_bp))


def plan_v4(pool: list[dict], slice_size: int | None = None, *, published=None,
            epoch: int | None = None):
    """The v4 epoch's question plan (td_v4.V4Plan): the one way to build it,
    for the miner's TD and for every verifier alike, so the scored slice,
    the exploration, the pool index and the targets cannot differ between
    them.

    - `scored_slice`: scored_slice(4, ...) at `slice_size`, which must be the
      size the pinned mix was built for (config.SLICE_SIZE_V4);
    - `exploration`: td_v4.exploration_v4 on this plan's pool index;
    - `pool_index`: pool_index_v4(pool); `targets`: the pinned mix's.

    With `published` (memo_v4.Published, folded through `epoch` - 2 at
    least) the plan is epoch `epoch`'s (SEAT_MINIMAL_V4.md 6.3): its pool
    index is memo_v4.reserve_index, so the scored slice and the private draw
    read only questions published through
    `epoch` - 2, and every hotkey's private set holds the epoch's fresh
    questions, the only draw that reads the unpublished reserve. Such a plan
    draws on its own pool only: another pool is refused, never drawn without
    the published set.
    """
    from fugal_subnet.tee import td_v4

    mix = mix_v4()
    size = config.SLICE_SIZE_V4 if slice_size is None else int(slice_size)
    if size != mix.slice_size:
        raise ValueError(f"the v4 mix is pinned for a {mix.slice_size}-question slice, not {size}")
    if published is None:
        if epoch is not None:
            raise ValueError("an epoch's plan reads the published set: pass it with the epoch")
        index = pool_index_v4(pool)
        drawn = pool
    else:
        from fugal_subnet import memo_v4
        if epoch is None:
            raise ValueError("a plan with the published set is one epoch's: name the epoch")
        index = memo_v4.reserve_index(published, pool, int(epoch), mix.bench_category,
                                      mix.bench_weight)
        drawn = index.published_pool

    def own(questions: list[dict]) -> bool:
        if questions is pool:
            return True
        if published is not None:
            raise ValueError("a plan with the published set draws on the pool it was built on")
        return False

    def scored(nonce: bytes, questions: list[dict], n: int) -> list[dict]:
        if n != size:
            raise ValueError(f"this plan draws a {size}-question slice, not {n}")
        return scored_slice(4, nonce, drawn if own(questions) else questions, n)

    def exploration(nonce: bytes, questions: list[dict], scored_ids: set[str], state,
                    hotkey: str) -> dict[str, str]:
        # The plan's index is the index of the pool it was built on; another
        # pool is indexed afresh, so the draw is always the pool's own.
        return td_v4.exploration_v4(nonce, questions, scored_ids, state, hotkey,
                                    index=index if own(questions) else None)

    return td_v4.V4Plan(scored_slice=scored, exploration=exploration, pool_index=index,
                        targets=mix.targets)


def challenge_nonce(nonce: bytes) -> bytes:
    """The key of an epoch's challenge list past its slice: the epoch nonce's
    HMAC under a label of its own, so the draw is the epoch's alone."""
    import hashlib
    import hmac

    return hmac.new(bytes(nonce), b"fugal-challenge-list-v4", hashlib.sha256).digest()


def challenge_list_v4(nonce: bytes, slice_questions: list[dict], index: PoolIndex,
                      targets: Mapping[str, float], size: int | None = None) -> list[dict]:
    """The epoch's challenge list (FAST_CROWN_V4.md 2, package F2): the scored
    slice first, in its order, then CHALLENGE_LIST_SIZE_V4 minus its length
    further questions on the scored mix (`targets`), drawn from `index` by
    the epoch's challenge nonce as a private set is drawn
    (slicer.select_private_explore, disjoint from the slice) and put in the
    order of their keyed digests, so every prefix past the slice is itself a
    draw on the mix. With an epoch's plan (`plan_v4` with the published set)
    `index` is its ReserveIndex, so the list reads only questions published
    through E - 2, as the slice does. A challenge of n questions answers the
    first n; a pool too small to fill the list gives a shorter one (and no
    challenge longer than it). Deterministic in public inputs."""
    import hashlib
    import hmac

    n = config.CHALLENGE_LIST_SIZE_V4 if size is None else int(size)
    first = list(slice_questions)
    if len(first) > n:
        raise ValueError("the scored slice is longer than the challenge list")
    key = challenge_nonce(nonce)
    taken = {q["question_id"] for q in first}
    rest = select_private_explore(key, index, taken, n - len(first), dict(targets))
    rest.sort(key=lambda q: hmac.new(key, f"order:{q['question_id']}".encode("utf-8"),
                                     hashlib.sha256).digest())
    return first + rest


def explore_shares_v4(state, explore_pairs=None):
    """The epoch's v4 exploration draw (explore_weights.SharesV4): shares from
    the epoch's menu state's measured costs (menu_state.costs) and its
    exploration-pair counts (`state.explore_pairs()`, section 8's n_e: each
    entry's exploration pairs in the window, as the menu publishes them). The
    miner's TD and every validator call this with the epoch's menu and
    nothing else, so they draw the same entries and weigh every pair the
    same. The coverage part is aimed first at the provisional entries that
    joined at this epoch (`provisional_first`). `explore_pairs` replaces the
    state's counts for a hypothetical
    (tests, the W0 replay); the epoch's draw never passes it."""
    from fugal_subnet import explore_weights, menu_state
    counts = state.explore_pairs() if explore_pairs is None else explore_pairs
    return explore_weights.plan_shares_v4(menu_state.costs(state), counts,
                                          first=provisional_first(state))


def provisional_first(state) -> tuple[str, ...]:
    """The entries the epoch's coverage exploration is aimed at first
    (FAST_CROWN_V4.md 18(c)): the provisional entries that joined at this
    epoch, sorted. A function of the menu alone."""
    return tuple(sorted(e.id for e in state.entries
                        if e.provisional and e.joined_epoch == state.epoch))


def exploration_plan_v4(nonce: bytes, pool: list[dict], scored_ids: set[str], shares,
                        hotkey: str | None, index: PoolIndex | None = None,
                        ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """(shared, cost-aware part, coverage part) for one miner in a v4 epoch.

    - **Shared**: `EXPLORE_SHARED_SIZE_V4` questions chosen as v3 chooses them,
      the same for every miner, each sent to an entry drawn by `shares.cost`.
      Zero under v4 (SEAT_MINIMAL_V4.md 2.3 item 33): the set is always empty.
    - **Private**: `EXPLORE_PRIVATE_SIZE_V4` questions drawn for this hotkey as
      v3 draws its private set (disjoint from the scored slice and the shared
      set), split by a keyed rank (exploration.private_parts) into the
      cost-aware part (`EXPLORE_COST_AWARE_SIZE_V4`, entries by `shares.cost`,
      keyed by the miner's nonce as v3's are) and the coverage part (the rest,
      entries by `shares.coverage`, keyed by exploration.coverage_nonce).
      Empty when `hotkey` is None.

    With the epoch's memo_v4.ReserveIndex as `index` (plan_v4 with the
    published set), the private set is the epoch's fresh questions
    (memo_v4.fresh_questions: drawn from the unpublished reserve by the
    epoch nonce, the same for every hotkey) and the rest drawn from
    published questions only; it keeps its size, and the split and the
    entry draws are the same functions, so every pair's weight is the same
    exact inverse probability.
    """
    from fugal_subnet.explore_weights import SharesV4, assign_by_share
    if not isinstance(shares, SharesV4):
        raise ValueError("a v4 exploration plan needs the epoch's v4 shares (explore_shares_v4)")
    targets, bench_category, bench_weight = _mix_v4()
    shared: dict[str, str] = {}
    if config.EXPLORE_SHARED_SIZE_V4 > 0:
        chosen = select_explore_set_v2(explore_nonce(nonce), pool, scored_ids,
                                       config.EXPLORE_SHARED_SIZE_V4, targets, bench_category,
                                       bench_weight=bench_weight)
        shared = assign_by_share(nonce, chosen, shares.cost)
    cost_part: dict[str, str] = {}
    coverage: dict[str, str] = {}
    if hotkey and config.EXPLORE_PRIVATE_SIZE_V4 > 0:
        mine = private_nonce(nonce, hotkey)
        index = index or pool_index_v4(pool)
        excluded = set(scored_ids) | set(shared)
        fresh: list[dict] = []
        from fugal_subnet import memo_v4
        if isinstance(index, memo_v4.ReserveIndex):
            # The epoch's fresh questions (memo_v4.fresh_questions), the same
            # for every hotkey, then the rest of the set from published ones.
            fresh = [q for q in memo_v4.fresh_questions(nonce, index, targets)
                     if q["question_id"] not in excluded][:config.EXPLORE_PRIVATE_SIZE_V4]
            excluded |= {q["question_id"] for q in fresh}
        chosen = fresh + select_private_explore(mine, index, excluded,
                                                config.EXPLORE_PRIVATE_SIZE_V4 - len(fresh),
                                                targets)
        first, rest = private_parts(mine, chosen, config.EXPLORE_COST_AWARE_SIZE_V4)
        if first:
            cost_part = assign_by_share(mine, first, shares.cost)
        if rest:
            coverage = assign_by_share(coverage_nonce(mine), rest, shares.coverage)
    return shared, cost_part, coverage
