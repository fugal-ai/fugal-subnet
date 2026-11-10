"""Known-answer and boundary tests for the pool, slicing, grading-task and protocol rules.

A planted-bug review found that each of these could change without any test noticing:
the epoch id width, the nonce payload order, the v1 HMAC rank direction and cap floor,
the v2 scored salt, the v2 cap, the v2 exploration shares, the pool loader's unpinned
escape hatch and size floor, the IFEval checker hand-off, the token estimate, the
Omni-MATH hard boundary, and the synapse size caps.

The golden values below were computed from the code as it stands and are consensus
values: a change to any of them means two validators on different builds select
different questions or price them differently. Changing one is a coordinated release,
not a test update.
"""
import gzip
import hashlib
import json
from collections import Counter

import pydantic
import pytest

from fugal_subnet.benchmarks import loader
from fugal_subnet.benchmarks.slicer import (
    _largest_remainder,
    _within_category_shares,
    category_quotas,
    derive_nonce,
    epoch_id_for_block,
    select_explore_set_v2,
    select_slice,
    select_slice_v2,
)
from fugal_subnet.config import (
    BENCHMARK_CATEGORY_V2,
    CATEGORY_TARGETS_V2,
)

BLOCK_HASH = "0x" + "ab" * 32


def _ids_hash(slice_: list[dict]) -> str:
    return hashlib.sha256(json.dumps([q["question_id"] for q in slice_]).encode()).hexdigest()


# --- epoch id and nonce: golden vectors (I1) -----------------------------------


@pytest.mark.parametrize("index, expected", [
    (0, "e00000000"),
    (7, "e00000007"),
    (1234, "e00001234"),
    (99999999, "e99999999"),
])
def test_epoch_id_for_block_golden(index, expected):
    assert epoch_id_for_block(index) == expected


@pytest.mark.parametrize("epoch_id, expected_hex", [
    ("e00001234", "40b5134e8feafe7cd657151f7de4e372c4b25b2900d071f90614427078ed966d"),
    ("e00000000", "88d9b2366b8560fcdd59c51b84e8f6e37b9153e9861c8a6b64229616d0b4e86d"),
])
def test_derive_nonce_golden(epoch_id, expected_hex):
    assert derive_nonce(epoch_id, BLOCK_HASH).hex() == expected_hex


def test_derive_nonce_is_sha256_of_epoch_then_block_hash():
    # The payload order is part of consensus: epoch id first, then the block hash.
    want = hashlib.sha256(f"e00001234:{BLOCK_HASH}".encode()).digest()
    assert derive_nonce("e00001234", BLOCK_HASH) == want


# --- v1 slice: golden ids and the cap floor ------------------------------------

V1_SIZES = {"alpha": 40, "beta": 25, "gamma": 10, "delta": 3}
V1_POOL = [{"question_id": f"{b}_{i:03d}", "benchmark": b} for b, n in V1_SIZES.items() for i in range(n)]


@pytest.mark.parametrize("k, expected", [
    (0, "53a3bc4025960a879ff6e219497b9854d319af2b437aae485038eebe6e7aead9"),
    (1, "a42e94903f61b7333463b900690f3d53f3e1ff92d256d096f0238165e663c826"),
    (2, "6aba3b19def50fba6dc81995c6ffd8a5a724fa61548f6457e4f40d418cdfd916"),
])
def test_v1_slice_ids_golden(k, expected):
    nonce = derive_nonce(epoch_id_for_block(1000 + k), BLOCK_HASH)
    sl = select_slice(nonce, V1_POOL, size=20)
    assert Counter(q["benchmark"] for q in sl) == {"alpha": 8, "beta": 8, "gamma": 3, "delta": 1}
    assert _ids_hash(sl) == expected


def test_v1_slice_takes_lowest_hmac_first_within_a_benchmark():
    import hmac
    nonce = derive_nonce(epoch_id_for_block(1000), BLOCK_HASH)
    sl = select_slice(nonce, V1_POOL, size=20)
    for bench in V1_SIZES:
        got = [q["question_id"] for q in sl if q["benchmark"] == bench]
        ranked = sorted((q["question_id"] for q in V1_POOL if q["benchmark"] == bench),
                        key=lambda qid: hmac.new(nonce, qid.encode(), hashlib.sha256).hexdigest())
        assert got == ranked[:len(got)], bench


def test_v1_tiny_benchmark_cap_floor_is_one():
    # delta is 3 of 78 questions: its cap is max(1, int(1.5 * 20 * 3 / 78)) = 1.
    # A floor of 2 would double its share of the slice.
    for k in range(5):
        nonce = derive_nonce(epoch_id_for_block(k), BLOCK_HASH)
        sl = select_slice(nonce, V1_POOL, size=20)
        assert len(sl) == 20
        assert Counter(q["benchmark"] for q in sl)["delta"] == 1


# --- v2 slice and exploration: golden ids, quotas and shares --------------------

V2_SIZES = {"mmlupro": 60, "mmlu": 30, "math_l35": 40, "omnimath_mid": 20, "supergpqa_hard": 50,
            "aime": 30, "aime25": 5, "beyondaime": 8, "omnimath_hard": 25, "ifeval": 12}
V2_POOL = [{"question_id": f"{b}_{i:03d}", "benchmark": b} for b, n in V2_SIZES.items() for i in range(n)]
# A weighted category, as a fixture: the slicer's weighted path stays pinned
# by these vectors. Production weights no category since 2026-11-09
# (config.BENCHMARK_WEIGHT_V2 = {}); these are the weights it had before.
WEIGHTED_HARD = {"supergpqa_hard": 0.5, "aime": 0.244369, "aime25": 0.007858,
                 "beyondaime": 0.026192, "omnimath_hard": 0.221582}
T, C, W = CATEGORY_TARGETS_V2, BENCHMARK_CATEGORY_V2, WEIGHTED_HARD


def _v2_nonce(k: int) -> bytes:
    return derive_nonce(epoch_id_for_block(2000 + k), BLOCK_HASH)


V2_GOLDEN = [
    # (scored ids hash, exploration ids hash, scored category quotas)
    ("890e5df4632e4f66d250977bc1d89153e9e222d276c852db139e178d431af8ab",
     "cf1132100d4364cc150de243003b607f6666a922d5e093b05db313810e128cc0",
     {"hard": 13, "instruction": 2, "knowledge": 16, "math": 10}),
    ("f822e2b4c8acfac159e5b377b7e8bc1ff0d311d973ed726fdb514ccb183dc4da",
     "ecc2849fab1f5d89b405efb21cc94686bce7bdc939f32e4ba7c2f39726ac8f89",
     {"hard": 12, "instruction": 2, "knowledge": 16, "math": 11}),
    ("81fcf4ac5d4750cc3a9c3922bc5f48378f7825898ab7d841e052e2f43c04b3da",
     "5c69856df5bdde81a12b6bf39a2833f3099e8a5ad8c7b7e4ada1892ec2d1e77b",
     {"hard": 12, "instruction": 2, "knowledge": 16, "math": 11}),
    ("592078fc58b268b1701c2bdcc2eebaaf60f640af33244aca43c263b4a5e45c9b",
     "f76f390c6577b58588901320ca7681f88176e2f61b0cbc44f224af5267036a0f",
     {"hard": 13, "instruction": 2, "knowledge": 16, "math": 10}),
]


@pytest.mark.parametrize("k", range(len(V2_GOLDEN)))
def test_v2_category_quotas_golden(k):
    assert category_quotas(41, T, set(T), _v2_nonce(k), "scored") == V2_GOLDEN[k][2]


@pytest.mark.parametrize("k", range(len(V2_GOLDEN)))
def test_v2_scored_slice_ids_golden(k):
    sl = select_slice_v2(_v2_nonce(k), V2_POOL, 41, T, C, bench_weight=W)
    assert _ids_hash(sl) == V2_GOLDEN[k][0]


@pytest.mark.parametrize("k", range(len(V2_GOLDEN)))
def test_v2_exploration_ids_golden(k):
    nonce = _v2_nonce(k)
    scored = select_slice_v2(nonce, V2_POOL, 41, T, C, bench_weight=W)
    ex = select_explore_set_v2(nonce, V2_POOL, {q["question_id"] for q in scored}, 15, T, C, bench_weight=W)
    assert _ids_hash(ex) == V2_GOLDEN[k][1]


def test_v2_scored_slice_uses_the_scored_category_quotas():
    # The scored slice and exploration draw their category quotas under different salts;
    # the scored slice must use "scored". Nonces 2 and 3 are ones where the two differ.
    for k in range(len(V2_GOLDEN)):
        nonce = _v2_nonce(k)
        sl = select_slice_v2(nonce, V2_POOL, 41, T, C, bench_weight=W)
        per_cat = Counter(C[q["benchmark"]] for q in sl)
        assert dict(per_cat) == category_quotas(41, T, set(T), nonce, "scored")
    differ = [k for k in range(len(V2_GOLDEN))
              if category_quotas(41, T, set(T), _v2_nonce(k), "scored")
              != category_quotas(41, T, set(T), _v2_nonce(k), "explore")]
    assert differ, "the fixture must include a nonce where the salts disagree"


def test_v2_within_category_split_is_unbiased_largest_remainder_when_caps_do_not_bind():
    # The cap is share_factor * want * share. The largest-remainder quota is at most
    # ceil(want * share), so with share_factor = 1.5 and want * share >= 2 the cap never
    # binds and each benchmark gets exactly its unbiased largest-remainder quota. A cap
    # without share_factor (floor(want * share)) would cut every rounded-up unit.
    pool = [{"question_id": f"{b}_{i:03d}", "benchmark": b}
            for b, n in {"kA": 70, "kB": 50, "kC": 30}.items() for i in range(n)]
    cats = {"kA": "knowledge", "kB": "knowledge", "kC": "knowledge"}
    targets = {"knowledge": 1.0}
    ranked = {b: [q for q in pool if q["benchmark"] == b] for b in ("kA", "kB", "kC")}
    share = _within_category_shares(["kA", "kB", "kC"], ranked, None)
    seen_round_up = False
    for k in range(12):
        nonce = _v2_nonce(100 + k)
        want = 23   # 23 * (7/15, 5/15, 3/15) = 10.73, 7.67, 4.6
        got = Counter(q["benchmark"] for q in select_slice_v2(nonce, pool, want, targets, cats))
        lr = _largest_remainder(want, share, nonce, "scored:knowledge")
        assert dict(got) == {b: n for b, n in lr.items() if n}
        seen_round_up |= any(lr[b] > int(want * share[b]) for b in lr)
    assert seen_round_up


def test_v2_exploration_with_one_weighted_category():
    # With only one category weighted (the fixture), the within-category shares must be
    # computed per category: computing them over every benchmark at once would mix
    # weighted and unweighted benchmarks and refuse.
    assert set(W) < set(C)
    for k in range(8):
        nonce = _v2_nonce(k)
        scored = select_slice_v2(nonce, V2_POOL, 41, T, C, bench_weight=W)
        ex = select_explore_set_v2(nonce, V2_POOL, {q["question_id"] for q in scored}, 15, T, C, bench_weight=W)
        assert len(ex) == 15
        assert dict(Counter(C[q["benchmark"]] for q in ex)) == {
            c: n for c, n in category_quotas(15, T, set(T), nonce, "explore").items() if n}


# --- pricing: the token estimate is consensus state -----------------------------


@pytest.mark.parametrize("prompt, tokens", [
    ("", 1),
    ("abc", 1),
    ("abcdef", 2),       # 1.5 rounds to 2
    ("abcdefg", 2),      # 1.75 rounds to 2, not down to 1
    ("a" * 10, 2),       # 2.5: banker's rounding
    ("a" * 400, 100),    # 4 characters per token
    ("a" * 1001, 250),
])
def test_question_input_tokens_golden(prompt, tokens):
    from fugal_subnet.pricing import question_input_tokens
    assert question_input_tokens(prompt) == tokens


# --- grading task: IFEval's constraints reach the grader ------------------------


def _ifeval_question() -> dict:
    # Shaped exactly like fugal_subnet.benchmarks.ifeval.load() output.
    return {
        "prompt": "Write a short note about rivers, all in lowercase, without commas.",
        "gold": "",
        "grader_id": "constraint_set",
        "benchmark": "ifeval",
        "question_id": "ifeval_0000",
        "metadata": {"checker": {"id": "constraint_set", "params": {"constraints": [
            {"kind": "lowercase_only"},
            {"kind": "must_exclude", "s": ","},
        ]}}},
    }


def test_ifeval_question_grades_through_build_grader_task():
    from fugal_subnet import graders
    from fugal_subnet.grading_task import build_grader_task

    task = build_grader_task(_ifeval_question())
    assert task["checker"]["params"]["constraints"]
    assert graders.grade(task, "rivers carry water to the sea and shape the land") == 1
    assert graders.grade(task, "Rivers carry water, and shape the land") == 0
    assert graders.grade(task, "rivers carry water, and shape the land") == 0


# --- v2 sets: the Omni-MATH hard boundary ---------------------------------------


def test_omnimath_difficulty_exactly_at_the_hard_threshold_is_hard(monkeypatch):
    from fugal_subnet.benchmarks import v2_sets

    thr = v2_sets.OMNI_HARD_DIFFICULTY
    rows = [{"question_id": f"omnimath_{i:05d}", "prompt": "p", "gold": "1", "grader_id": "integer_exact",
             "metadata": {"difficulty": d, "domain": "x"}}
            for i, d in enumerate([thr - 0.5, thr, thr + 0.5])]
    monkeypatch.setattr(v2_sets, "_omni_integer", lambda: [dict(r) for r in rows])
    hard = {q["question_id"] for q in v2_sets.load_omnimath_hard()}
    mid = {q["question_id"] for q in v2_sets.load_omnimath_mid()}
    assert hard == {"omnimath_00001", "omnimath_00002"}
    assert mid == {"omnimath_00000"}
    assert hard.isdisjoint(mid) and len(hard | mid) == len(rows)


# --- loader: the v2 pool's escape hatch and size floor --------------------------

SMALL_POOL = [{"question_id": f"mmlupro_{i:05d}", "benchmark": "mmlupro", "prompt": f"q{i}",
               "gold": "A", "grader_id": "letter_mcq", "metadata": {}} for i in range(20)]


@pytest.fixture
def v2_env(monkeypatch, tmp_path):
    for var in ("FUGAL_BENCHMARK_POOL_V2", "FUGAL_POOL_CACHE_V2", "FUGAL_POOL_UNPINNED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FUGAL_POOL_FILE_V2", str(tmp_path / "absent.json.gz"))
    return monkeypatch, tmp_path


def test_unpinned_declaration_does_not_skip_verification_of_a_built_pool(v2_env):
    # FUGAL_POOL_UNPINNED waives verification for a pool FILE an operator placed. A pool
    # built from Hugging Face is the consensus pool by construction and is always verified.
    monkeypatch, _ = v2_env
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")
    monkeypatch.setattr(loader, "build_pool_v2", lambda strict=True: [dict(q) for q in SMALL_POOL])
    with pytest.raises(RuntimeError, match="manifest"):
        loader.load_all_v2(strict=True)


def test_unpinned_declaration_still_waives_verification_for_a_pool_file(v2_env):
    monkeypatch, tmp_path = v2_env
    path = tmp_path / "pool_v2.json"
    path.write_text(json.dumps(SMALL_POOL))
    monkeypatch.setenv("FUGAL_BENCHMARK_POOL_V2", str(path))
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")
    assert loader.load_all_v2(strict=True) == SMALL_POOL


def _pinned_small_pool(monkeypatch, tmp_path):
    """A pool file AND a v2 manifest that pins it, so only the size floor can refuse it."""
    path = tmp_path / "pool_v2.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(SMALL_POOL, f)
    manifest = tmp_path / "pool_manifest_v2.json"
    manifest.write_text(json.dumps({"pool_hash": loader.pool_hash(SMALL_POOL),
                                    "content_hash": loader.content_hash(SMALL_POOL),
                                    "n_questions": len(SMALL_POOL)}))
    monkeypatch.setattr(loader, "manifest_path_v2", lambda: str(manifest))
    monkeypatch.setenv("FUGAL_BENCHMARK_POOL_V2", str(path))


def test_v2_pool_below_the_size_floor_is_refused_even_when_it_matches_its_manifest(v2_env):
    monkeypatch, tmp_path = v2_env
    _pinned_small_pool(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="floor"):
        loader.load_all_v2(strict=True)


def test_v2_pool_at_the_size_floor_loads(v2_env):
    monkeypatch, tmp_path = v2_env
    _pinned_small_pool(monkeypatch, tmp_path)
    import fugal_subnet.config as config
    monkeypatch.setattr(config, "BENCHMARK_POOL_MIN_SIZE", len(SMALL_POOL))
    assert loader.load_all_v2(strict=True) == SMALL_POOL


# --- protocol: every miner-supplied field is capped (I2) ------------------------


def _caps():
    from fugal_subnet import protocol as p
    return {
        "epoch_id": 64,
        "nonce": p.HASH_MAX_LEN,
        "proof_json": p.PROOF_JSON_MAX_LEN,
        "head_npz_b64": p.HEAD_B64_MAX_LEN,
        "proof_hash": p.HASH_MAX_LEN,
        "weights_hash": p.HASH_MAX_LEN,
        "event_log_json": p.EVENT_LOG_MAX_LEN,
    }


def test_every_fugal_synapse_field_has_a_pinned_cap():
    import bittensor as bt

    from fugal_subnet.protocol import FugalProofSynapse
    own = set(FugalProofSynapse.model_fields) - set(bt.Synapse.model_fields)
    assert own == set(_caps())


@pytest.mark.parametrize("field", sorted(_caps()))
def test_synapse_field_refuses_one_byte_over_its_cap(field):
    from fugal_subnet.protocol import FugalProofSynapse
    cap = _caps()[field]
    assert len(getattr(FugalProofSynapse(**{field: "x" * cap}), field)) == cap
    with pytest.raises(pydantic.ValidationError):
        FugalProofSynapse(**{field: "x" * (cap + 1)})
