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


# --- Where the pool could still diverge silently ---------------------------

def test_content_hash_is_importable_from_the_package_not_only_a_checkout():
    """`load_all` calls this at startup. It used to be imported from scripts/,
    which is not a package and is not shipped in the wheel — so the loader
    raised ModuleNotFoundError for anyone whose working directory was not the
    repo root, and an installed validator carried no verification at all."""
    import fugal_subnet.benchmarks.loader as loader_mod

    assert callable(loader_mod.content_hash)

    import scripts.build_pool_manifest as script_mod
    assert script_mod.content_hash is loader_mod.content_hash


def test_an_operator_who_configures_nothing_gets_the_pinned_skip_list(monkeypatch):
    """The pool is consensus state, so the default must be the pinned pool and
    not whatever this machine happens to be able to load."""
    from fugal_subnet.benchmarks.loader import _default_skip

    monkeypatch.delenv("FUGAL_SKIP_BENCHMARKS", raising=False)
    pinned = set(json.load(open("data/pool_manifest.json"))["skip_benchmarks"])
    assert _default_skip() == pinned
    assert pinned, "the manifest pins no skips; this test would prove nothing"


def test_an_explicit_setting_still_wins_including_the_empty_one(monkeypatch):
    """Unset and empty are different answers: empty means 'skip nothing', which
    an operator must be able to say."""
    from fugal_subnet.benchmarks.loader import _default_skip

    monkeypatch.setenv("FUGAL_SKIP_BENCHMARKS", "mmlu")
    assert _default_skip() == {"mmlu"}

    monkeypatch.setenv("FUGAL_SKIP_BENCHMARKS", "")
    assert _default_skip() == set()


def test_the_default_configuration_does_not_bypass_verification(monkeypatch):
    """The bug this closes: a skip-list mismatch made _verify_against_manifest
    return early, so the check that would have caught a divergent pool disabled
    itself and logged a warning. A divergence that silences its own alarm is
    worse than no check."""
    import pytest

    from fugal_subnet.benchmarks.loader import _default_skip, _verify_against_manifest

    monkeypatch.delenv("FUGAL_SKIP_BENCHMARKS", raising=False)
    wrong_pool = [{"question_id": "q1", "prompt": "x", "gold": "y",
                   "grader_id": "exec_io"}]
    with pytest.raises(RuntimeError, match="does not match"):
        _verify_against_manifest(wrong_pool, _default_skip(), strict=True)
