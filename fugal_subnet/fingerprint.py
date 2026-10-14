"""Environment fingerprint: what a score was actually computed with.

Two validators that disagree on a head's score have to be able to find out why.
Without a record of what each was running, "the subnet is behaving oddly" is
unfalsifiable; with one, it is a diff.

The fingerprint goes into every reveal artifact (so divergence is diagnosable
after the fact) and is asserted at validator startup (so the common causes fail
loudly at boot instead of silently skewing weights for hours).

Everything here is cheap and side-effect free. Importing this module must not
pull in torch — a fingerprint is also taken by tooling that has no business
loading a 2GB library — so torch and transformers are probed lazily.
"""
from __future__ import annotations

import hashlib
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import fugal_subnet
from fugal_subnet.determinism import DETERMINISM_ENV

# Every package whose version can change a computed score. numpy and torch do
# the arithmetic; transformers decides tokenization; datasets decides which
# questions load; math-verify decides whether an answer is graded correct.
CONSENSUS_PACKAGES = (
    "numpy",
    "torch",
    "transformers",
    "datasets",
    "math-verify",
)


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in CONSENSUS_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "missing"
    return out


def _numpy_blas() -> dict[str, str]:
    """Which BLAS numpy will use — the library that runs head_eval's matmul."""
    try:
        import numpy as np

        cfg = np.show_config("dicts") or {}
        blas = (cfg.get("Build Dependencies") or {}).get("blas") or {}
        return {
            "name": str(blas.get("name", "unknown")),
            "version": str(blas.get("version", "unknown")),
        }
    except Exception:
        return {"name": "unknown", "version": "unknown"}


def _torch_runtime() -> dict[str, str]:
    """Torch's view of the world, only if torch is already loaded."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {"loaded": "false"}
    try:
        return {
            "loaded": "true",
            "version": str(torch.__version__),
            "cuda_available": str(bool(torch.cuda.is_available())).lower(),
            "num_threads": str(torch.get_num_threads()),
        }
    except Exception:
        return {"loaded": "true", "version": "unknown"}


def grader_hash() -> str:
    """The canonical grader version string.

    Delegates to fugal_subnet.graders rather than recomputing the digest. A
    second implementation of the same hash is a thing that can silently drift
    from the first, and this value is published in every reveal.
    """
    from fugal_subnet.graders import grader_hash as canonical

    return canonical()


def environment_fingerprint() -> dict:
    """A JSON-serializable record of everything that can change a score."""
    from fugal_subnet import config
    from fugal_subnet.config import (
        EPOCH_COLLECT_FRACTION,
        EPOCH_INTERVAL,
        EXPLORE_FRACTION,
        ROUTING_DECISION_QUANTUM,
        SLICE_SHARE_FACTOR,
        SLICE_SIZE,
    )

    return {
        "fugal_version": fugal_subnet.__version__,
        "routing_protocol": __import__("fugal_subnet.routing_protocol", fromlist=["identity"]).identity(),
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": _package_versions(),
        "numpy_blas": _numpy_blas(),
        "torch_runtime": _torch_runtime(),
        # Recorded as actually in effect, not as intended: an operator who
        # exported one of these keeps their value, and that is exactly the
        # kind of divergence this record has to be able to explain.
        "cpu_dispatch": {k: os.environ.get(k, "unset") for k in DETERMINISM_ENV},
        "grader_sha256": grader_hash(),
        "routing": {
            "decision_quantum": ROUTING_DECISION_QUANTUM,
        },
        # Epoch shape. These are env-overridable and consensus-critical: two
        # neurons with different slice sizes select different questions and
        # every proof fails on questions_hash, with nothing in the error naming
        # the cause. A miner running the default 300 against a validator set to
        # 50 produced an identical questions_hash every epoch — it was slicing
        # the entire pool — and the mismatch read only as "questions hash
        # mismatch".
        "epoch": {
            "slice_size": SLICE_SIZE,
            "slice_share_factor": SLICE_SHARE_FACTOR,
            "epoch_interval_s": EPOCH_INTERVAL,
            "explore_fraction": EXPLORE_FRACTION,
            "collect_fraction": EPOCH_COLLECT_FRACTION,
        },
        # Every constant that decides a score or which proofs verify, as in
        # effect right now (read off the module, so an override is recorded
        # as the value it produced). Diligence 2026-10, finding A2: these were
        # env-overridable and absent from the digest, so a validator with one
        # exported scored the field differently and nothing said why.
        "scoring": {
            name: getattr(config, name)
            for name in _SCORING_CONSTANTS
        },
        "approved_entries": sorted(config.TEE_APPROVED_MEASUREMENTS),
    }


# Module attribute names, in a fixed order, so the digest is stable.
_SCORING_CONSTANTS = (
    "WILSON_CONFIDENCE", "SCORE_QUALITY_FLOOR", "FRONTIER_MIN_TRIALS",
    "BURN_IN_QUESTIONS", "EXPLORE_FAIL_MARGIN", "EXPLORE_DISAGREE_MAX",
    "EXPLORE_DISAGREE_MIN_SAMPLES", "EXPLORE_DISAGREE_MIN_VOTERS",
    "DEDUP_SIMILARITY_THRESHOLD", "DEDUP_PARAM_DISTANCE",
    "DEDUP_EPSILON_AGREEMENT", "DEDUP_EPSILON", "EVIDENCE_HALF_LIFE",
    "LIVENESS_MAX_MISSED_EVIDENCE", "FRAME_HALF_LIFE", "FRAME_PRIOR_STRENGTH",
    "FRAME_PRIOR_ACCURACY", "FRAME_DEFAULT_COMPLETION_TOKENS",
    "BENCHMARK_POOL_MIN_SIZE", "MAX_WEIGHT_DELTA", "REQUIRE_COMMITMENT",
    "HARNESS_ALLOW_EXEC", "HEAD_HIDDEN_DIM", "HEAD_MAX_MODELS", "HEAD_MAX_BYTES",
    "HEAD_MAX_DECOMPRESSED_BYTES", "HEAD_MAX_MODEL_ID_LEN",
)


def consensus_digest(fingerprint: dict | None = None) -> str:
    """One short hash over the score-affecting subset of the fingerprint.

    Platform and thread counts are excluded — they differ legitimately between
    honest validators. What remains is what must match: package versions, BLAS,
    kernel dispatch, the grader, and the routing constants. Two validators
    whose digests agree should produce identical scores; two that disagree have
    an explanation for why they did not.
    """
    fp = fingerprint if fingerprint is not None else environment_fingerprint()
    material = {
        "epoch": fp["epoch"],
        "scoring": fp.get("scoring", {}),
        # The contract, profile, lambda, token manifest and price table in one
        # id; a validator pointed at a different price file via
        # FUGAL_MODEL_PRICES diverges here.
        "routing_protocol": fp.get("routing_protocol", ""),
        "packages": fp["packages"],
        "numpy_blas": fp["numpy_blas"],
        "cpu_dispatch": fp["cpu_dispatch"],
        "grader_sha256": fp["grader_sha256"],
        "routing": fp["routing"],
    }
    import json

    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _expected_pins() -> dict[str, str] | None:
    """Exact versions from pyproject, the single source of truth.

    Returns None when pyproject.toml is not readable — a wheel installed into
    site-packages has no sibling pyproject. That means version pins cannot be
    verified, which is worth a warning, but it must never stop a validator
    from starting: the CPU-dispatch checks below are the load-bearing half and
    they do not depend on this file.
    """
    import re

    try:
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    except (OSError, IndexError):
        return None
    found = dict(re.findall(r'"([A-Za-z0-9_.-]+)==([^"]+)"', block))
    return {k: v for k, v in found.items() if k in CONSENSUS_PACKAGES}


def check_environment() -> list[str]:
    """Return human-readable reasons this host would diverge. Empty means OK."""
    problems: list[str] = []

    installed = _package_versions()
    pins = _expected_pins()
    if pins is None:
        problems.append(
            "pyproject.toml is not readable, so dependency pins could not be "
            "verified — confirm this host matches the locked versions"
        )
    else:
        for name, expected in pins.items():
            actual = installed.get(name, "missing")
            if actual != expected:
                problems.append(
                    f"{name} {actual} is installed but pyproject pins {expected} — "
                    "a different build can change computed scores"
                )

    for key, expected in DETERMINISM_ENV.items():
        actual = os.environ.get(key)
        if actual != expected:
            problems.append(
                f"{key}={actual!r} but consensus expects {expected!r} — "
                "CPU kernel dispatch or thread count is not pinned"
            )

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.get_num_threads() != 1:
                problems.append(
                    f"torch is using {torch.get_num_threads()} threads; "
                    "reduction order varies above one"
                )
        except Exception:
            pass

    # A consensus constant set in the environment is a validator that scores
    # the field by its own rules (diligence 2026-10, finding A2). The value is
    # in the digest either way; refusing at startup is what keeps the digest
    # from being the only place the divergence shows up.
    from fugal_subnet.config import CONSENSUS_ENV_VARS
    for name in CONSENSUS_ENV_VARS:
        if name in os.environ:
            problems.append(
                f"{name} is set in the environment; consensus constants cannot be "
                "overridden on a live validator (unset it, or change config.py "
                "through review)"
            )

    return problems


def assert_environment(strict: bool = True) -> None:
    """Fail loudly at startup rather than skewing weights quietly for hours.

    A validator running mismatched libraries does not crash — it computes
    subtly different scores, sets divergent weights, and looks healthy the
    whole time. In a consensus system a loud failure is always the better
    trade, so this raises by default.
    """
    problems = check_environment()
    if not problems:
        return
    detail = "\n".join(f"  - {p}" for p in problems)
    message = (
        "Environment does not match the consensus-pinned configuration:\n"
        f"{detail}\n"
        "Install the locked dependencies (`uv sync --locked`) and do not "
        "override the CPU dispatch variables. Set FUGAL_ALLOW_ENV_DRIFT=1 to "
        "downgrade this to a warning (mock and development runs only)."
    )
    if strict and os.getenv("FUGAL_ALLOW_ENV_DRIFT") != "1":
        raise RuntimeError(message)
    import logging

    logging.getLogger(__name__).warning(message)


def chain_genesis_hash(subtensor) -> str | None:
    """The connected chain's genesis block hash, or None if it cannot be read."""
    for getter in (
        lambda: subtensor.substrate.get_block_hash(0),
        lambda: subtensor.get_block_hash(0),
    ):
        try:
            value = getter()
        except Exception:  # noqa: BLE001 - any failure means "unknown", handled by the caller
            continue
        if value:
            return str(value)
    return None


def assert_mock_allowed(genesis_hash: str | None, mock: bool) -> None:
    """Refuse --mock on mainnet.

    Mock accepts unattested proofs: it is the mode for local testnets and the
    dress rehearsal, and on finney it is a validator that pays whatever any
    miner claims (diligence 2026-10, finding A3). The chain is identified by
    its genesis hash, which a --network URL cannot disguise. An unreadable
    genesis is not a pass: in mock the caller is told to use --live, which
    needs no such check.
    """
    if not mock:
        return
    from fugal_subnet.config import FINNEY_GENESIS_HASH

    if genesis_hash is None:
        raise RuntimeError(
            "--mock refused: the chain's genesis hash could not be read, so this "
            "cannot be shown not to be finney. Use --live, or point --network at a "
            "local or test chain that answers."
        )
    if str(genesis_hash).lower() == FINNEY_GENESIS_HASH.lower():
        raise RuntimeError(
            "--mock refused on finney (mainnet): mock accepts unattested proofs. "
            "Run with --live."
        )
