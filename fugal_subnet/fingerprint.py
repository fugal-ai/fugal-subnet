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
    from fugal_subnet.config import ROUTING_DECISION_QUANTUM, ROUTING_LAMBDA

    return {
        "fugal_version": fugal_subnet.__version__,
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
            "lambda": ROUTING_LAMBDA,
            "decision_quantum": ROUTING_DECISION_QUANTUM,
        },
    }


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
