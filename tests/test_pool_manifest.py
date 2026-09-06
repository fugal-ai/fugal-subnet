"""The pool's identity must track what changes a score — no less, and no more.

pool_hash covers question ids, which is what the slice is drawn from. It says
nothing about what the questions say, so a hand-edited cache or an unpinned
loader could agree on ids and still grade different answers. The manifest's
content_hash closes that.

The "no more" half is equally load-bearing and is what these tests mostly
guard. An identity that is sensitive to differences no score depends on rejects
legitimate pools, and the first casualty is the documented
FUGAL_BENCHMARK_POOL override.
"""
import json

from scripts.build_pool_manifest import content_hash


def _q(qid, prompt, gold, grader="exec_io"):
    return {"question_id": qid, "prompt": prompt, "gold": gold,
            "grader_id": grader, "benchmark": "humaneval"}


def test_json_round_trip_does_not_change_the_content_hash():
    """Eight HumanEval golds are lists of TUPLES in memory and lists of LISTS
    after a JSON round trip — which is exactly what FUGAL_BENCHMARK_POOL does.
    exec_io compares json.dumps(got) to json.dumps(gold), so the two grade
    identically, and the identity hash must agree with the grader."""
    pool = [_q("HumanEval_107", "p", [(8, 13), (4, 6)])]
    round_tripped = json.loads(json.dumps(pool))
    assert round_tripped[0]["gold"] == [[8, 13], [4, 6]]
    assert content_hash(pool) == content_hash(round_tripped)


def test_content_hash_changes_when_grading_would():
    """The "no less" half: anything a checker reads must move the hash."""
    base = [_q("q1", "prompt", [1, 2])]
    for changed in (
        [_q("q1", "DIFFERENT prompt", [1, 2])],
        [_q("q1", "prompt", [1, 3])],
        [_q("q1", "prompt", [1, 2], grader="exec_unittest")],
        [_q("q2", "prompt", [1, 2])],
    ):
        assert content_hash(base) != content_hash(changed), changed


def test_content_hash_is_order_independent():
    """Two operators whose loaders return the same questions in a different
    order hold the same pool, and must agree."""
    a = [_q("q1", "one", [1]), _q("q2", "two", [2])]
    assert content_hash(a) == content_hash(list(reversed(a)))


def test_manifest_matches_the_shipped_pool_configuration():
    """The shipped manifest must describe a pool someone can actually build."""
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "data" / "pool_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["n_questions"] > 0
    assert len(manifest["pool_hash"]) == 64
    assert len(manifest["content_hash"]) == 64
    # gpqa is gated and livecode has no revision pin; both must be declared
    # rather than silently absent, or two operators build different pools.
    assert "gpqa" in manifest["skip_benchmarks"]
    assert "livecode" in manifest["skip_benchmarks"]
    assert sum(manifest["per_benchmark"].values()) == manifest["n_questions"]
