"""Unified benchmark loading interface.

Every loader returns list[dict] with schema:
    prompt:      str   — the question/task text sent to a model
    gold:        str   — the gold answer for grading
    grader_id:   str   — which grader to use (maps to graders.CHECKERS)
    benchmark:   str   — benchmark name (e.g. "mmlu", "math", "gsm8k")
    question_id: str   — unique stable identifier for this question
    metadata:    dict   — benchmark-specific fields (checker params, test code, etc.)

Dataset revisions are PINNED (DATASET_REVISIONS): the benchmark pool is
consensus-relevant state — two validators loading a dataset at different
times must build byte-identical pools, so loaders never track a moving
"main" branch. Bump these pins deliberately, as a coordinated release.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os

logger = logging.getLogger(__name__)

# HuggingFace dataset revision pins (commit SHAs, resolved 2026-08-24).
DATASET_REVISIONS: dict[str, str] = {
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "openai/openai_humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    "Idavidrein/gpqa": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "qq8933/AIME_1983_2024": "3e2cc86390666c5c756622afc0eeb9e6194496bc",
}

_BENCHMARKS: dict[str, str] = {
    "mmlu":      "fugal_subnet.benchmarks.mmlu",
    "math":      "fugal_subnet.benchmarks.math_",
    "gsm8k":     "fugal_subnet.benchmarks.gsm8k",
    "humaneval": "fugal_subnet.benchmarks.humaneval",
    "livecode":  "fugal_subnet.benchmarks.livecode",
    "gpqa":      "fugal_subnet.benchmarks.gpqa",
    "ifeval":    "fugal_subnet.benchmarks.ifeval",
    "aime":      "fugal_subnet.benchmarks.aime",
}


def available_benchmarks() -> list[str]:
    return sorted(_BENCHMARKS.keys())


def load_benchmark(name: str) -> list[dict]:
    if name not in _BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {available_benchmarks()}")
    mod = importlib.import_module(_BENCHMARKS[name])
    return mod.load()


def pool_hash(pool: list[dict]) -> str:
    """Identity of a question pool.

    The pool is consensus state: the slice is drawn from it, so two neurons
    holding different pools derive different slices and every proof fails on
    questions_hash — correctly, but for a reason that looks nothing like the
    cause. This makes the real cause nameable in a log line.
    """
    ids = sorted(q["question_id"] for q in pool)
    canonical = json.dumps(ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _drop_unscoreable(pool: list[dict]) -> list[dict]:
    """Remove questions no checker can score under the current harness policy.

    Derived from config.HARNESS_ALLOW_EXEC rather than hardcoded, so enabling
    execution re-includes the benchmarks in exactly one place. A question the
    harness cannot grade is not neutral: the miner still pays to answer it, and
    since the slicer equalises benchmarks it was a full sixth of every graded
    slice, punishing anyone who routed code to a capable model while no quality
    was reachable. See docs/CODE_BENCHMARK_PLAN.md for putting them back.
    """
    from fugal_subnet.config import EXECUTION_CHECKERS, HARNESS_ALLOW_EXEC

    if HARNESS_ALLOW_EXEC:
        return pool
    kept = [q for q in pool if q.get("grader_id") not in EXECUTION_CHECKERS]
    dropped = len(pool) - len(kept)
    if dropped:
        from collections import Counter
        by_bench = Counter(q.get("benchmark", "") for q in pool
                           if q.get("grader_id") in EXECUTION_CHECKERS)
        logger.warning(
            "Excluded %d questions the harness cannot score (%s): their checkers "
            "run candidate code and FUGAL_HARNESS_ALLOW_EXEC is off. They would "
            "score zero for every miner while still costing them API spend.",
            dropped, dict(by_bench),
        )
    return kept


def load_all(strict: bool = True) -> list[dict]:
    """Load the full benchmark pool.

    Both neurons call this, so both derive the same pool from the same pinned
    dataset revisions. FUGAL_BENCHMARK_POOL overrides it with a local JSON file
    — for offline work and local testnets, where reaching HuggingFace is either
    impossible or pointlessly slow. The override applies to miner and validator
    alike, which is the point: a pool one side can load and the other cannot is
    how the two end up disagreeing.

    Benchmarks named in FUGAL_SKIP_BENCHMARKS (comma-separated) are skipped
    deliberately. Any OTHER load failure raises when strict=True: a validator
    running with a silently incomplete pool would select a different slice
    than its peers and diverge on every epoch. Fail loudly instead.
    """
    override = os.getenv("FUGAL_BENCHMARK_POOL", "").strip()
    if override:
        with open(override, encoding="utf-8") as f:
            pool = json.load(f)
        pool = _drop_unscoreable(pool)
        logger.info(
            "Benchmark pool loaded from FUGAL_BENCHMARK_POOL=%s "
            "(%d questions, pool_hash=%s)",
            override, len(pool), pool_hash(pool)[:16],
        )
        # The override used to return here, before either guard below ran. So
        # the documented production path — a materialised pool file on every
        # neuron — was also the one path on which the manifest check and the
        # size floor were both silently absent. Measured: a 150-question pool
        # ran through 40+ live epochs on netuid 552 with no complaint from
        # anything. A file is not a reason to trust its contents; the same two
        # checks apply, and a deliberately non-pinned pool (a local testnet, a
        # rehearsal) has to SAY SO with FUGAL_POOL_UNPINNED=1, which is loud.
        if _unpinned_pool_declared():
            logger.warning(
                "FUGAL_POOL_UNPINNED is set: this pool is declared NOT to be the "
                "pinned consensus pool, so the manifest and size-floor checks "
                "are skipped. Correct for a local testnet or rehearsal fixture; "
                "never set it on a neuron that talks to anyone else's.",
            )
            return pool
        _verify_against_manifest(pool, _default_skip(), strict)
        _verify_pool_is_large_enough_to_be_unmemorisable(pool, strict)
        return pool

    skip = _default_skip()
    pool = []
    for name in sorted(_BENCHMARKS):
        if name in skip:
            logger.info("Skipping benchmark %s", name)
            continue
        try:
            items = load_benchmark(name)
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"Benchmark {name!r} failed to load: {e}. "
                    f"A partial pool breaks cross-validator determinism — fix the "
                    f"load or add {name!r} to FUGAL_SKIP_BENCHMARKS explicitly."
                ) from e
            logger.warning("Skipping benchmark %s: %s", name, e)
            continue
        if not items:
            # Zero questions is a divergence, not a warning. It is the same
            # hazard as a load failure and gets the same treatment: a benchmark
            # that yields nothing here yields its full contents for anyone who
            # has a local copy of it, so two operators build different pools,
            # derive different slices, and every proof between them fails on
            # questions_hash — an error that names the symptom and never the
            # cause. LiveCodeBench does exactly this on current `datasets`
            # versions. Absence has to be declared to be reproducible.
            if strict:
                raise RuntimeError(
                    f"Benchmark {name!r} loaded 0 questions. An empty benchmark "
                    f"is a silent pool divergence: anyone holding a local copy "
                    f"of it builds a different pool and no proof between you "
                    f"will verify. Add {name!r} to FUGAL_SKIP_BENCHMARKS to "
                    f"declare the absence, or fix the load."
                )
            logger.warning("Benchmark %s loaded 0 questions", name)
        pool.extend(items)
    pool = _drop_unscoreable(pool)
    logger.info("Benchmark pool: %d questions, pool_hash=%s",
                len(pool), pool_hash(pool)[:16])
    _verify_against_manifest(pool, skip, strict)
    _verify_pool_is_large_enough_to_be_unmemorisable(pool, strict)
    return pool


def _unpinned_pool_declared() -> bool:
    """Whether the operator has declared the override pool as deliberately
    NOT the pinned consensus pool. Unset and "0" both mean no."""
    return os.getenv("FUGAL_POOL_UNPINNED", "0") not in ("0", "", "false", "False")


def _verify_pool_is_large_enough_to_be_unmemorisable(
    pool: list[dict], strict: bool,
) -> None:
    """The pool's SIZE is what makes a head route rather than memorise.

    Checked here rather than in a test because the failure arrives through
    configuration, not through code: `FUGAL_SKIP_BENCHMARKS` can shrink the
    pool on a single validator with no diff to review. Measured, a linear head
    fits random labels on 2,000 questions perfectly and on 21,717 only 16.9%,
    so dropping one large benchmark moves the subnet a long way toward
    measuring nothing — and every other check would still pass, because the
    pool loaded fine and its hash changed exactly as a legitimate change would.
    """
    from fugal_subnet.config import BENCHMARK_POOL_MIN_SIZE

    if len(pool) >= BENCHMARK_POOL_MIN_SIZE:
        return
    msg = (
        f"Benchmark pool has {len(pool)} questions, below the "
        f"{BENCHMARK_POOL_MIN_SIZE} floor. Pool size is a security parameter: "
        "a linear head memorises a small pool instead of learning to route "
        "(measured: 100% random-label fit at 2,000 questions, 16.9% at "
        "21,717). Scores from a pool this small do not measure routing. If the "
        "shrink is deliberate, raise FUGAL_POOL_MIN_SIZE and say why."
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning("%s", msg)


def content_hash(pool: list[dict]) -> str:
    """Hash over everything a validator grades against, in id order.

    Deliberately covers the four fields that decide an outcome: which question
    it is, what the miner is asked, what counts as right, and which checker
    decides. A change in any of them changes a score, so a change in any of them
    must change this hash.

    Lives HERE and not in scripts/. It is called by `load_all` at startup, and
    scripts/ is not a package and is not shipped in the wheel — so importing it
    from there crashed the loader for anyone whose working directory was not the
    repo root, and would have shipped no verification at all to an installed
    validator. Consensus code belongs in the package; the build script imports
    it from here.
    """
    h = hashlib.sha256()
    for q in sorted(pool, key=lambda x: x["question_id"]):
        h.update(json.dumps([
            q.get("question_id", ""),
            q.get("prompt", ""),
            # Canonicalised the way the GRADER canonicalises it, not with str().
            # exec_io compares json.dumps(got) to json.dumps(gold), so a tuple
            # and a list of the same values grade identically — and eight
            # HumanEval golds are tuples in memory and lists after a JSON round
            # trip, which is what FUGAL_BENCHMARK_POOL does. Hashing str() made
            # this manifest reject the documented override for a difference no
            # score depends on. An identity hash must be sensitive to exactly
            # what changes an outcome: no less, and no more.
            json.dumps(q.get("gold", ""), sort_keys=True, default=str),
            q.get("grader_id", ""),
        ], separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _manifest_path() -> str:
    """Locate the pinned manifest as PACKAGE DATA, not by counting directories.

    It used to be three dirnames up from this file plus data/pool_manifest.json,
    which resolves inside a checkout and to site-packages/data/... in an
    installed one. `data/` is not in the wheel, so on an installed validator the
    manifest was simply absent — and absence was the silent branch, so the
    default skip list fell back to empty and the content check returned without
    logging. The consensus verification was inert in exactly the deployment that
    most needs it, and said nothing.

    importlib.resources answers the question the file system cannot: where is
    this file relative to the PACKAGE, however the package was installed.
    """
    from importlib.resources import files

    return str(files("fugal_subnet.benchmarks") / _MANIFEST_NAME)


_MANIFEST_NAME = "pool_manifest.json"


def _default_skip() -> set:
    """Which benchmarks to skip, defaulting to the pinned manifest's list.

    The pool is consensus state, so "an operator who configures nothing" must
    get the pinned pool rather than whatever their machine happens to be able to
    load. Before this, the default was empty: an operator with a HuggingFace
    token loaded GPQA, produced a larger pool than everyone else, and — worse —
    the skip-list mismatch made `_verify_against_manifest` return early, so the
    one check that would have caught it disabled itself and logged a warning.
    A divergence that silences its own alarm is the failure mode this codebase
    keeps finding.

    An explicitly set FUGAL_SKIP_BENCHMARKS still wins, including when it is set
    to empty to mean "skip nothing". Unset and empty are different answers here,
    so `getenv(...) is None` is the test rather than truthiness.
    """
    env = os.getenv("FUGAL_SKIP_BENCHMARKS")
    if env is not None:
        return set(env.split(",")) - {""}

    path = _manifest_path()
    try:
        with open(path, encoding="utf-8") as f:
            pinned = set(json.load(f).get("skip_benchmarks", []))
    except FileNotFoundError:
        # Not fatal here — _verify_against_manifest is where the strict refusal
        # lives — but never silent. This returning empty is what made an
        # installed validator load a different pool from everyone else.
        logger.warning(
            "No pool manifest at %s; defaulting to skipping nothing. Set "
            "FUGAL_SKIP_BENCHMARKS explicitly if that is not what you want.",
            path,
        )
        return set()
    except Exception as e:  # noqa: BLE001 - a broken manifest must not be fatal here
        logger.warning("Could not read pool manifest %s: %s", path, e)
        return set()
    if pinned:
        logger.info("Skipping %s, per the pinned pool manifest",
                    ",".join(sorted(pinned)))
    return pinned


def _verify_against_manifest(pool: list[dict], skip: set, strict: bool) -> None:
    """Check the loaded pool against the pinned manifest.

    pool_hash covers question ids, which is the right identity for slice
    selection but says nothing about what the questions SAY. Revisions are
    pinned so content cannot drift on its own — but it can drift by hand: an
    edited cache, a manually placed data/benchmarks/livecode.json, or a
    benchmark whose loader carries no revision pin. Any of those otherwise
    produces a pool that agrees on ids, selects the same slice, and grades
    different things, failing at verification with a hash that names the
    symptom rather than the cause.

    Absent manifest is not an error — a fresh checkout or a bespoke pool is a
    legitimate state. A manifest that DISAGREES is, under strict.
    """
    manifest_path = _manifest_path()
    if not os.path.exists(manifest_path):
        # Loud. A fresh checkout legitimately has no manifest, but so does a
        # broken install, and the two need different reactions from a human.
        # Silence made them indistinguishable.
        msg = (
            f"No pool manifest at {manifest_path}. The pool is consensus state "
            "and nothing is verifying it: this process cannot tell whether its "
            "pool matches everybody else's."
        )
        if strict:
            raise RuntimeError(
                msg + " Refusing to run strict without it — build one with "
                "scripts/build_pool_manifest.py, or install a build that ships it."
            )
        logger.warning("%s", msg)
        return
    try:
        with open(manifest_path, encoding="utf-8") as f:
            pinned = json.load(f)
    except Exception as e:  # noqa: BLE001 - a broken manifest must not be fatal
        logger.warning("Could not read pool manifest %s: %s", manifest_path, e)
        return

    if sorted(pinned.get("skip_benchmarks", [])) != sorted(skip):
        logger.warning(
            "Pool manifest was built with FUGAL_SKIP_BENCHMARKS=%s but this "
            "process has %s — skipping content verification. The pools are "
            "different by construction and will not agree.",
            ",".join(pinned.get("skip_benchmarks", [])) or "(none)",
            ",".join(sorted(skip)) or "(none)",
        )
        return

    actual_ids, actual_content = pool_hash(pool), content_hash(pool)
    if actual_ids == pinned.get("pool_hash") and actual_content == pinned.get("content_hash"):
        logger.info("Pool matches the pinned manifest (content_hash=%s)",
                    actual_content[:16])
        return

    detail = (
        f"pinned pool_hash={pinned.get('pool_hash','?')[:16]} "
        f"content_hash={pinned.get('content_hash','?')[:16]} "
        f"n={pinned.get('n_questions')}; "
        f"loaded pool_hash={actual_ids[:16]} content_hash={actual_content[:16]} "
        f"n={len(pool)}"
    )
    if strict:
        raise RuntimeError(
            "Benchmark pool does not match the pinned manifest. The pool is "
            "consensus state: a pool that differs from every other operator's "
            "selects a different slice or grades different answers, and every "
            f"proof will fail on a hash that names none of this. {detail}. "
            "Rebuild with scripts/build_pool_manifest.py if the change is "
            "intended, and say why in the commit."
        )
    logger.warning("Benchmark pool does not match the pinned manifest: %s", detail)
