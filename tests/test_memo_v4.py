"""Protocol v4: the memorisation check and the rationed unpublished reserve
(docs/decisions/SEAT_MINIMAL_V4.md 6; fugal_subnet/memo_v4.py).

- The published set: per question the first epoch a reveal published an
  outcome on it, seeded from a pinned file built from named reveals and
  folded from each reveal; a rebuild reaches the live state.
- The reserve: an epoch spends ceil(R / 1200) unpublished questions, drawn
  by the epoch nonce for the whole field and put in every hotkey's private
  set; every other draw reads published questions only.
- The check: a head's skill over R on labels of questions published before
  its row counts against the others, within benchmark, under the one bound,
  with the minimum in each part; questions are the bound's items.
- On the 44 published v3 epochs no head is refused, and only the
  within-benchmark comparison keeps two of them from being refused.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from fugal_subnet import config, epoch_plan, labels_v4, memo_v4
from fugal_subnet import menu_state as ms
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.exploration import private_nonce, private_parts
from fugal_subnet.labels_v4 import Side
from fugal_subnet.memo_v4 import SEEN, UNSEEN, Part, Published
from fugal_subnet.seat_bound import A_ALPHA, LN2_UP, log_steps

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "fugal_subnet" / "benchmarks" / "pool_manifest_v3.json"
EPOCH = 1000


# --- fixtures -----------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def manifest_pool() -> tuple[dict, ...]:
    """A pool of the v3 manifest's sets at their pinned sizes (ids only)."""
    per = json.loads(MANIFEST.read_bytes())["per_benchmark"]
    return tuple({"question_id": f"{b}_{i:05d}", "benchmark": b, "prompt": f"{b} {i}"}
                 for b in sorted(per) for i in range(per[b]["n"]))


def _h(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


@functools.lru_cache(maxsize=1)
def published_state() -> Published:
    """Two thirds of the pool published at epochs 1..EPOCH - 3, the rest not
    (a keyed draw), folded through EPOCH - 1."""
    first = {}
    for q in manifest_pool():
        k = _h(q["question_id"])
        if k % 3:
            first[q["question_id"]] = 1 + k % (EPOCH - 3)
    return Published(first, EPOCH - 1)


def nonce(k: int = 0) -> bytes:
    return hashlib.sha256(f"memo-epoch-{k}".encode()).digest()


@functools.lru_cache(maxsize=1)
def plan():
    pool = list(manifest_pool())
    return pool, epoch_plan.plan_v4(pool, published=published_state(), epoch=EPOCH)


STATE = ms.seed_state(EPOCH)


# --- the published set ---------------------------------------------------------------------


def test_a_fold_records_the_first_epoch_and_a_retry_of_the_same_epoch_changes_nothing():
    p = Published({"a": 3}, 5)
    p6 = memo_v4.fold(p, 6, ["a", "b"])
    assert dict(p6.first) == {"a": 3, "b": 6} and p6.through == 6
    assert memo_v4.fold(p6, 6, ["b"]) is p6                     # a retry
    assert memo_v4.fold(p6, 9, []).through == 9                  # an epoch with no reveal skipped
    with pytest.raises(ValueError, match="already folded"):
        memo_v4.fold(p6, 6, ["c"])
    with pytest.raises(ValueError, match="already folded"):
        memo_v4.fold(p6, 5, ["b"])                               # b was first published in 6


def test_the_view_an_epoch_reads_and_its_digest_do_not_depend_on_later_folds():
    p = Published({"a": 3, "b": 6}, 6)
    later = memo_v4.fold(memo_v4.fold(p, 7, ["c"]), 8, ["d", "a"])
    assert later.view(6) == p.view(6) and later.view(6).digest == p.digest
    assert later.digest != p.digest
    assert p.published_by("b", 6) and not p.published_by("b", 5)
    with pytest.raises(ValueError, match="folded through"):
        p.view(7)


def test_the_state_document_round_trips_and_is_read_strictly():
    p = Published({"q1": 4, "q0": 4, "q9": 2}, 7)
    doc = p.to_doc()
    assert doc == {"format": memo_v4.STATE_FORMAT, "through": 7,
                   "first": {"2": ["q9"], "4": ["q0", "q1"]}}
    assert memo_v4.published_from_doc(json.loads(json.dumps(doc))) == p
    for bad in ({**doc, "through": "7"}, {**doc, "first": {"2": ["q9"], "4": ["q1", "q0"]}},
                {**doc, "first": {"2": ["q9"], "4": ["q9"]}}, {**doc, "extra": 1},
                {**doc, "first": {"9": ["q0"]}}):
        with pytest.raises(ValueError):
            memo_v4.published_from_doc(bad)


def test_a_reveal_publishes_its_scored_questions_every_exploration_answer_and_its_labels():
    reveal = {"epoch_id": "e00000012",
              "questions": [{"question_id": "s1"}, {"question_id": "s2"}],
              "exploration": [{"question_id": "x1", "failed": True}, {"question_id": "x2"}],
              "exploration_pool": [{"question_id": "x2"}, {"question_id": "p3"}]}
    assert memo_v4.published_ids(reveal) == {"s1", "s2", "x1", "x2", "p3"}
    assert memo_v4.published_ids({"epoch_id": "e00000001"}) == set()
    with pytest.raises(ValueError, match="question id"):
        memo_v4.published_ids({"questions": [{"id": "s1"}]})
    folded = memo_v4.fold_reveal(Published({"s1": 3}, 11), reveal)
    assert dict(folded.first) == {"s1": 3, "s2": 12, "x1": 12, "x2": 12, "p3": 12}


def test_a_rebuild_from_the_reveals_reaches_the_live_state():
    reveals = [{"epoch_id": f"e{e:08d}",
                "exploration": [{"question_id": f"q{(e * 7 + i) % 40}"} for i in range(6)]}
               for e in range(20, 35) if e != 27]
    live = Published({}, 19)
    for r in reveals:
        live = memo_v4.fold_reveal(live, r)
    assert memo_v4.rebuild(Published({}, 19), list(reversed(reveals))) == live
    # a node that already folded part of the way folds only what is after it
    assert memo_v4.rebuild(memo_v4.rebuild(Published({}, 19), reveals[:5]), reveals) == live
    with pytest.raises(ValueError, match="two reveals"):
        memo_v4.rebuild(Published({}, 19), reveals + reveals[:1])


# --- the seed --------------------------------------------------------------------------------


def _script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_published_v4",
                                                  ROOT / "scripts" / "build_published_v4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_seed_rebuilds_byte_for_byte_from_the_committed_observations():
    text = _script().build(str(ROOT / "data" / "observations"
                               / "published_v4_e00026189-e00026509.json")).encode()
    assert (ROOT / "data" / "published_v4.json").read_bytes() == text
    assert Path(memo_v4.SEED_PATH).read_bytes() == text


def test_the_seed_is_the_named_reveals_on_the_pinned_pool():
    obs = json.loads((ROOT / "data" / "observations"
                      / "published_v4_e00026189-e00026509.json").read_bytes())
    seed = memo_v4.seed_doc()
    manifest = json.loads(MANIFEST.read_bytes())
    assert seed["through"] == "e00026509" and obs["first"] == "e00026189"
    assert seed["pool"]["pool_hash"] == manifest["pool_hash"]
    assert seed["pool"]["per_set"] == {b: r["n"] for b, r in manifest["per_benchmark"].items()}
    by_protocol = {}
    for r in obs["reveals"]:
        by_protocol[r["protocol"]] = by_protocol.get(r["protocol"], 0) + 1
    assert by_protocol == {1: 160, 2: 117, 3: 44}
    # 32,106 of the 48,320 questions were published by the named reveals; v1's
    # reveals published 6,737 of them first.
    assert sum(len(v) for v in seed["unpublished"].values()) == 16214
    assert sum(r["new"] for r in obs["reveals"] if r["protocol"] == 1) == 6737
    cats = epoch_plan.mix_v4().bench_category
    reserve = {}
    for s, ids in seed["unpublished"].items():
        reserve[cats[s]] = reserve.get(cats[s], 0) + len(ids)
        assert all(q.split("_")[0] in s or s.startswith(q.split("_")[0]) for q in ids)
    assert reserve == {"code": 1931, "easy_math": 8020, "everyday": 880, "hard": 1183,
                       "instruction": 1590, "knowledge": 2511, "math": 99}


def test_the_seed_is_read_for_its_pool_and_refuses_another():
    doc = {"format": memo_v4.SEED_FORMAT, "through": "e00000040",
           "pool": {"pool_hash": "x", "n_questions": 5, "per_set": {"aime": 3, "arc": 2}},
           "observations_sha256": "y", "unpublished": {"aime": ["aime_2"], "arc": []}}
    pool = [{"question_id": f"aime_{i}", "benchmark": "aime"} for i in range(3)] + [
        {"question_id": f"arc_{i}", "benchmark": "arc"} for i in range(2)] + [
        {"question_id": "mmlu_0", "benchmark": "mmlu"}]          # a set added after the seed
    p = memo_v4.seed_published(pool, doc)
    assert p.through == 40
    assert dict(p.first) == {"aime_0": 40, "aime_1": 40, "arc_0": 40, "arc_1": 40}
    with pytest.raises(ValueError, match="another pool"):
        memo_v4.seed_published([q for q in pool if q["question_id"] != "aime_2"], doc)
    for bad in ({**doc, "unpublished": {"aime": ["aime_2", "aime_1"]}},
                {**doc, "unpublished": {"gsm8k": ["g"]}}, {**doc, "through": "40"}):
        with pytest.raises(ValueError):
            memo_v4.seed_published(pool, bad)


# --- the reserve and the fresh draw ---------------------------------------------------------


def test_an_epoch_spends_the_reserve_over_1200_rounded_up():
    assert memo_v4.FRESH_RESERVE_EPOCHS == config.FRESH_RESERVE_EPOCHS_V4 == 1200
    assert [memo_v4.fresh_count(r) for r in (0, 1, 1200, 1201, 7000, 16214)] == [0, 1, 1, 2, 6, 14]


@pytest.mark.parametrize("reserve", [1200, 4200, 7000, 16214, 60000])
def test_the_reserve_lasts_past_60_days_with_no_new_questions_whatever_the_hotkeys(reserve):
    """The spend is F per epoch for the field, never per hotkey, so the number
    of hotkeys does not enter. 1,200 epochs is 60 days at 20 an epoch."""
    r = reserve
    for _ in range(1200 - 1):
        r -= memo_v4.fresh_count(r)
    assert r >= 1                                                 # still a reserve on day 60
    if reserve >= 2400:
        assert r * 4 >= reserve                                   # a quarter of it or more


def test_the_fresh_questions_are_unpublished_through_two_epochs_back_and_the_same_for_all():
    pool, p = plan()
    index = p.pool_index
    assert isinstance(index, memo_v4.ReserveIndex) and index.through == EPOCH - 2
    fresh = memo_v4.fresh_questions(nonce(), index, epoch_plan.mix_v4().targets)
    view = published_state().view(EPOCH - 2)
    reserve = sum(1 for q in pool if q["question_id"] not in view.first)
    assert index.reserve_size == reserve and index.fresh_count == memo_v4.fresh_count(reserve)
    assert len(fresh) == index.fresh_count > 0
    assert not any(q["question_id"] in view.first for q in fresh)
    # every node: another arrival order of the pool, a state folded further
    later = memo_v4.fold(published_state(), EPOCH + 3, [q["question_id"] for q in fresh])
    other = epoch_plan.plan_v4(list(reversed(pool)), published=later, epoch=EPOCH)
    assert (sorted(q["question_id"] for q in memo_v4.fresh_questions(
        nonce(), other.pool_index, epoch_plan.mix_v4().targets))
        == sorted(q["question_id"] for q in fresh))
    assert other.pool_index.digest == index.digest
    # another epoch nonce draws another set
    assert memo_v4.fresh_questions(nonce(1), index, epoch_plan.mix_v4().targets) != fresh


def test_every_private_set_holds_the_fresh_questions_and_the_rest_is_published():
    pool, p = plan()
    view = published_state().view(EPOCH - 2)
    fresh = {q["question_id"] for q in memo_v4.fresh_questions(nonce(), p.pool_index,
                                                                epoch_plan.mix_v4().targets)}
    scored = [q["question_id"] for q in p.scored_slice(nonce(), pool, config.SLICE_SIZE_V4)]
    assert len(scored) == config.SLICE_SIZE_V4 and all(q in view.first for q in scored)
    for hk in ("5Fhotkey-a", "5Fhotkey-b", "5Fhotkey-c"):
        explored = p.exploration(nonce(), pool, set(scored), STATE, hk)
        assert len(explored) == config.EXPLORE_PRIVATE_SIZE_V4
        assert fresh <= set(explored) and not set(explored) & set(scored)
        assert all(q in view.first for q in set(explored) - fresh)
    # No proof measures a proposal (FAST_CROWN_V4.md 18): the plan draws none.
    assert not hasattr(p, "measurement")


def test_the_private_set_keeps_its_parts_and_weights_so_every_pair_is_weighed_exactly():
    pool, p = plan()
    shares = epoch_plan.explore_shares_v4(STATE)
    scored = {q["question_id"] for q in p.scored_slice(nonce(), pool, config.SLICE_SIZE_V4)}
    _, cost, cover = epoch_plan.exploration_plan_v4(nonce(), pool, scored, shares, "5Fhk",
                                                    p.pool_index)
    assert len(cost) == config.EXPLORE_COST_AWARE_SIZE_V4 == 68
    assert len(cover) == config.EXPLORE_COVERAGE_SIZE_V4 == 44
    # Each part's entries are its shares' draw on its questions, the same
    # function as without the published set: a pair's weight, 1 / (M
    # pi_private), is the shares' alone.
    from fugal_subnet.exploration import coverage_nonce
    from fugal_subnet.explore_weights import assign_by_share
    mine = private_nonce(nonce(), "5Fhk")
    by_id = {q["question_id"]: q for q in pool}
    assert cost == assign_by_share(mine, [by_id[q] for q in cost], shares.cost)
    assert cover == assign_by_share(coverage_nonce(mine), [by_id[q] for q in cover],
                                    shares.coverage)


def test_a_fresh_question_falls_in_the_cost_aware_part_at_the_parts_share():
    """The split ranks the whole private set by the miner's own key, so a
    fresh question is in the cost-aware part with chance 68 / 112, as every
    other question is: what pi_private assumes."""
    pool, p = plan()
    targets = epoch_plan.mix_v4().targets
    fresh = memo_v4.fresh_questions(nonce(), p.pool_index, targets)
    done = set(published_state().view(EPOCH - 2).first)
    rest = [q for q in pool if q["question_id"] in done]
    assert len(fresh) + 98 == config.EXPLORE_PRIVATE_SIZE_V4
    hits = trials = 0
    for k in range(300):
        chosen = fresh + rest[k * 98:(k + 1) * 98]
        assert len(chosen) == config.EXPLORE_PRIVATE_SIZE_V4
        first, _ = private_parts(private_nonce(nonce(), f"5Fhk{k}"), chosen, 68)
        ids = {q["question_id"] for q in first}
        hits += sum(1 for q in fresh if q["question_id"] in ids)
        trials += len(fresh)
    share, expect = hits / trials, 68 / 112
    assert abs(share - expect) < 4 * math.sqrt(expect * (1 - expect) / trials)


def test_a_category_with_an_empty_reserve_gives_its_fresh_slots_to_the_others():
    pool = list(manifest_pool())
    cats = epoch_plan.mix_v4().bench_category
    first = dict(published_state().first)
    for q in pool:
        if cats[q["benchmark"]] == "hard":
            first.setdefault(q["question_id"], 5)                 # hard: all published
    p = epoch_plan.plan_v4(pool, published=Published(first, EPOCH - 1), epoch=EPOCH)
    fresh = memo_v4.fresh_questions(nonce(), p.pool_index, epoch_plan.mix_v4().targets)
    assert len(fresh) == p.pool_index.fresh_count
    assert not any(cats[q["benchmark"]] == "hard" for q in fresh)


def test_a_category_with_too_few_published_questions_is_left_out_of_the_published_draws():
    pool = list(manifest_pool())
    cats = epoch_plan.mix_v4().bench_category
    first = {q: e for q, e in published_state().first.items()
             if cats[q.rsplit("_", 1)[0]] != "everyday"}
    arc = [q["question_id"] for q in pool if q["benchmark"] == "arc"]
    first.update(dict.fromkeys(arc[:memo_v4.PUBLISHED_MIN_PER_CATEGORY - 1], 5))
    assert memo_v4.PUBLISHED_MIN_PER_CATEGORY == 200 + 112 + 150 + 1
    p = epoch_plan.plan_v4(pool, published=Published(first, EPOCH - 1), epoch=EPOCH)
    assert "everyday" not in p.pool_index.cats and "everyday" in p.pool_index.reserve.cats
    scored = [q for q in p.scored_slice(nonce(), pool, config.SLICE_SIZE_V4)]
    assert len(scored) == config.SLICE_SIZE_V4 and all(q["benchmark"] != "arc" for q in scored)
    explored = p.exploration(nonce(), pool, {q["question_id"] for q in scored}, STATE, "5Fhk")
    assert len(explored) == config.EXPLORE_PRIVATE_SIZE_V4
    published_arc = set(arc[:memo_v4.PUBLISHED_MIN_PER_CATEGORY - 1])
    assert not set(explored) & published_arc
    # one more published question and it is drawn again
    first[arc[memo_v4.PUBLISHED_MIN_PER_CATEGORY - 1]] = 5
    p2 = epoch_plan.plan_v4(pool, published=Published(first, EPOCH - 1), epoch=EPOCH)
    assert "everyday" in p2.pool_index.cats


def test_an_epochs_plan_reads_the_published_set_and_draws_on_its_own_pool_only():
    pool, p = plan()
    with pytest.raises(ValueError, match="the pool it was built on"):
        p.scored_slice(nonce(), list(pool), config.SLICE_SIZE_V4)
    with pytest.raises(ValueError, match="the pool it was built on"):
        p.exploration(nonce(), list(pool), set(), STATE, "5Fhk")
    with pytest.raises(ValueError, match="name the epoch"):
        epoch_plan.plan_v4(pool, published=published_state())
    with pytest.raises(ValueError, match="pass it with the epoch"):
        epoch_plan.plan_v4(pool, epoch=EPOCH)
    with pytest.raises(ValueError, match="folded through"):
        epoch_plan.plan_v4(pool, published=Published({}, EPOCH - 3), epoch=EPOCH)


# --- the check ------------------------------------------------------------------------------


def _part(questions: dict[str, tuple[int, int]]) -> Part:
    return Part(sum(n for n, _ in questions.values()), questions)


def _parts(spec):
    """{benchmark: {SEEN: Part, UNSEEN: Part}} from {benchmark: (seen items,
    unseen items)}, each item (labels, units) on its own question."""
    return {b: {SEEN: _part({f"{b}s{i}": v for i, v in enumerate(s)}),
                UNSEEN: _part({f"{b}u{i}": v for i, v in enumerate(u)})}
            for b, (s, u) in spec.items()}


def test_the_gap_is_the_weighted_within_benchmark_difference_exactly():
    spec = {"a": ([(1, 10), (2, 30), (1, -4), (1, 6), (1, 8)], [(1, 1), (1, 2), (3, 3), (1, 4), (1, 5)]),
            "b": ([(1, 100)] * 6, [(1, 50)] * 5 + [(2, 0)])}
    ev = memo_v4.evidence_from_parts(_parts(spec))
    ns, nu, ss, su = 6, 7, 50, 15                                 # a: labels and sums
    ws = Fraction(ns * nu, ns + nu)
    nbs, nbu = 6, 7
    wb = Fraction(nbs * nbu, nbs + nbu)
    want = (ws * (Fraction(ss, ns) - Fraction(su, nu)) + wb * (Fraction(600, 6) - Fraction(250, 7))) / (ws + wb)
    assert ev.gap == want
    assert ev.seen_labels == 12 and ev.unseen_labels == 14 and set(ev.benchmarks) == {"a", "b"}


def test_the_verdict_is_the_bounds_inequality_on_the_gap():
    """Refused iff (gap - margin)^2 > Var(gap) (1 + n0 / N) L(N), gap above the
    margin, with N questions, evaluated here in Fractions from the gap's own
    variance: the scaled items change nothing."""
    rng = np.random.default_rng(7)
    unit = config.BOUND_UNITS_PER_POINT_V4
    for k in range(30):
        spec = {}
        for b in range(rng.integers(1, 4)):
            shift = int(rng.integers(0, 8)) * unit
            seen = [(int(rng.integers(1, 4)), int(rng.normal(shift, 20 * unit))) for _ in range(rng.integers(5, 400))]
            unseen = [(int(rng.integers(1, 9)), int(rng.normal(0, 40 * unit))) for _ in range(rng.integers(5, 120))]
            spec[f"b{b}"] = (seen, unseen)
        ev = memo_v4.evidence_from_parts(_parts(spec))
        w = {b: Fraction(sum(n for n, _ in s) * sum(n for n, _ in u), sum(n for n, _ in s) + sum(n for n, _ in u))
             for b, (s, u) in spec.items()}
        big_w = sum(w.values())
        var = Fraction(0)
        n_items = 0
        for b, (s, u) in spec.items():
            for items, sign in ((s, 1), (u, -1)):
                c = w[b] / (big_w * sum(n for n, _ in items))
                xs = [x for _, x in items]
                m = Fraction(sum(xs), len(xs))
                var += c * c * sum((x - m) ** 2 for x in xs)
                n_items += len(xs)
        n0 = memo_v4.MEMO_MIN_LABELS
        log_term = A_ALPHA + LN2_UP * log_steps(n_items, n0)
        d = ev.gap - memo_v4.MEMO_MARGIN_UNITS
        passes = d > 0 and d * d > var * Fraction(n_items + n0, n_items) * log_term
        assert ev.bound.passes == passes, k
        assert ev.refused == (passes and ev.enough)


def test_a_benchmark_with_too_few_questions_in_a_part_is_left_out():
    spec = {"a": ([(1, 5)] * 10, [(1, 0)] * 4), "b": ([(1, 5)] * 10, [(1, 0)] * 5)}
    ev = memo_v4.evidence_from_parts(_parts(spec))
    assert set(ev.benchmarks) == {"b"} and ev.left_out == ("a",)
    assert ev.seen_labels == 10 and ev.unseen_labels == 5
    empty = memo_v4.evidence_from_parts(_parts({"a": spec["a"]}))
    assert empty.bound is None and not empty.refused and empty.gap == 0


def test_the_minimum_labels_in_each_part_gate_the_refusal():
    unit = config.BOUND_UNITS_PER_POINT_V4
    n = memo_v4.MEMO_MIN_LABELS
    assert n == config.MEMO_MIN_LABELS_V4 == 500
    rng = np.random.default_rng(3)
    big = [(1, int(30 * unit + rng.normal(0, 5 * unit))) for _ in range(4 * n)]

    def unseen(k):
        return [(1, int(rng.normal(0, 5 * unit))) for _ in range(k)]

    assert memo_v4.evidence_from_parts(_parts({"a": (big, unseen(n))})).refused
    short = memo_v4.evidence_from_parts(_parts({"a": (big, unseen(n - 1))}))
    assert short.bound.passes and not short.refused


def test_a_question_is_one_item_however_many_labels_it_has():
    """Every hotkey answers every fresh question, so one question's labels
    move together: they enter the bound as one item, and a gap carried by a
    handful of questions is not mistaken for many independent labels."""
    unit = config.BOUND_UNITS_PER_POINT_V4
    rng = np.random.default_rng(5)
    seen = [(1, int(rng.normal(3 * unit, 20 * unit))) for _ in range(3000)]
    # 60 questions, 10 labels each, each question's labels sharing one draw
    clustered = [(10, 10 * int(rng.normal(0, 18 * unit))) for _ in range(60)]
    as_clusters = memo_v4.evidence_from_parts(_parts({"a": (seen, clustered)}))
    # the same labels as if each were its own question
    flat = [(1, x // 10) for n, x in clustered for _ in range(n)]
    as_labels = memo_v4.evidence_from_parts(_parts({"a": (seen, flat)}))
    assert as_clusters.unseen_labels == as_labels.unseen_labels == 600
    assert as_clusters.gap == as_labels.gap
    assert as_clusters.lower < as_labels.lower                     # wider: fewer real items


def _window(epochs, qs_per_epoch=40, benches=("b1", "b2"), voters=("ckA", "ckB")):
    """Labels of `epochs`: each epoch's questions `e<e>q<i>` on two entries,
    alternately voted by ckA and ckB, outcome by a keyed draw."""
    pairs = {}
    for e in epochs:
        q, m, b, y, v = [], [], [], [], []
        for i in range(qs_per_epoch):
            q.append(f"e{e}q{i}")
            m.append("m/a" if i % 2 else "m/b")
            b.append(benches[i % len(benches)])
            y.append(10000 if _h(f"{e}-{i}") % 3 else 0)
            v.append((voters[i % len(voters)],))
        n = len(q)
        base = EpochPairs(question_ids=tuple(q), models=tuple(m), benchmarks=tuple(b),
                          categories=("knowledge",) * n, y=np.array(y, np.int32),
                          prompt_tokens=np.zeros(n, np.int32), completion_tokens=np.zeros(n, np.int32))
        pairs[e] = labels_v4.label_pairs(base, v, np.full(n, -1, np.int16))
    return Window(size=48, slices={}, heads={}, pairs=pairs, odds={})


def test_the_split_reads_only_the_labels_the_head_may_be_scored_on():
    w = _window(range(10, 20))
    w.odds["h"] = {e: np.full(40, 6000, np.int16) for e in range(10, 20)}
    ref = labels_v4.reference(w, now=19)
    published = Published({f"e{e}q{i}": e for e in range(10, 20) for i in range(40)}, 19)
    parts = memo_v4.split_labels(w, ref, Side("h", "ckA", 14), published, now=19)
    labels = sum(p[s].labels for p in parts.values() for s in (SEEN, UNSEEN))
    assert labels == 6 * 20                                       # epochs 14..19, ckB's half
    # each question was first published by its own label's epoch, never before the row
    assert all(p[SEEN].labels == 0 for p in parts.values())


def test_a_label_is_seen_when_its_question_was_published_before_its_row_counts():
    w = _window(range(10, 20))
    w.odds["h"] = {e: np.full(40, 6000, np.int16) for e in range(10, 20)}
    ref = labels_v4.reference(w, now=19)
    # every question of epochs 15.. was published back in epoch 3 (a repeat)
    published = Published({f"e{e}q{i}": 3 if e >= 15 else e
                           for e in range(10, 20) for i in range(40)}, 19)
    parts = memo_v4.split_labels(w, ref, Side("h", "ckZ", 12), published, now=19)
    assert sum(p[SEEN].labels for p in parts.values()) == 5 * 40
    assert sum(p[UNSEEN].labels for p in parts.values()) == 3 * 40
    # an inherited row counts from its predecessor's epoch: m/a's labels of
    # 10..14 are read too, and nothing is seen before 2
    # (m/a is the odd half of each epoch's questions; its row now predates
    # epoch 3's reveal, so its repeats of 15..19 are unseen too)
    succ = Side("h", "ckZ", 12, rows_since={"m/a": 2})
    parts = memo_v4.split_labels(w, ref, succ, published, now=19)
    assert sum(p[UNSEEN].labels for p in parts.values()) == 10 * 20 + 3 * 20
    assert sum(p[SEEN].labels for p in parts.values()) == 5 * 20
    succ_late = Side("h", "ckZ", 12, rows_since={"m/a": 2, "m/b": 2})
    parts = memo_v4.split_labels(w, ref, succ_late, published, now=19)
    assert sum(p[SEEN].labels for p in parts.values()) == 0
    with pytest.raises(ValueError, match="folded through"):
        memo_v4.split_labels(w, ref, Side("h", "ckZ", 12), Published({}, 17), now=19)
    with pytest.raises(ValueError, match="not of R"):
        memo_v4.split_labels(w, ref, labels_v4.REFERENCE, published, now=19)


# --- the published v3 epochs ----------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def observed():
    from tests.memo_v4_observed import NOW, load

    window, published, sides = load()
    ref = labels_v4.reference(window, now=NOW)
    out = {}
    for h, side in sorted(sides.items()):
        parts = memo_v4.split_labels(window, ref, side, published, now=NOW)
        out[h] = (parts, memo_v4.evidence_from_parts(parts))
    return out


def test_no_head_of_the_published_v3_epochs_is_refused_within_benchmark():
    """SEAT_MINIMAL_V4.md 6.1, reproduced from data/observations/
    memo_v4_labels_e00026466-e00026509.json (44 v3 reveals; first
    publication read from every named reveal from e00026189, v1 included):
    53 heads have a comparison, none is refused, and no head's gap is near
    its bound."""
    out = observed()
    compared = {h: ev for h, (_, ev) in out.items() if ev.bound is not None}
    assert len(out) == 53 and len(compared) == 53
    assert not any(ev.refused for ev in compared.values())
    assert all(ev.lower < memo_v4.MEMO_MARGIN_UNITS for ev in compared.values())
    gaps = sorted(float(ev.gap) / 1e6 for ev in compared.values())
    assert round(gaps[-1], 2) == 4.04 and round(gaps[0], 2) == -3.32
    assert sum(ev.seen_labels for ev in compared.values()) == 30_534
    assert sum(ev.unseen_labels for ev in compared.values()) == 28_481
    # the nearest any head comes: its bound's lower end of the gap, in points
    assert max(ev.lower for ev in compared.values()) == -1_642_493


def test_on_the_published_epochs_only_the_within_benchmark_comparison_spares_two_heads():
    """Pooled across benchmarks, two heads of the published field would be
    refused: their "gain" on repeated questions is which benchmarks repeat
    (small sets repeat sooner). Within benchmark neither is."""
    refused_flat = []
    for h, (parts, ev) in observed().items():
        merged = {SEEN: {}, UNSEEN: {}}
        for b, p in parts.items():
            for side in (SEEN, UNSEEN):
                merged[side].update({b + q: v for q, v in p[side].by_question.items()})
        flat = memo_v4.evidence_from_parts({"all": {s: _part(merged[s]) for s in (SEEN, UNSEEN)}})
        if flat.refused:
            refused_flat.append(h[:12])
        assert not ev.refused
    assert sorted(refused_flat) == ["9fcc5e6d9023", "df7758ae5c64"]
