"""Protocol v5's part of the consensus digest (skeleton).

Like consensus_v4: v5's constants and pinned files form a separate `v5`
section of the fingerprint, present only for an epoch at or after the v5
activation block, so a v1-v4 epoch's digest is byte for byte what it was.

    activation  the compiled finney block and the variable other chains read
    constants   every name in REGISTERED, read off its module, canonical JSON
    files       sha256 of every pinned file v5 reads (PINNED_FILES); None for
                a file this build does not ship yet

`problems()` lists what keeps the section from being complete: a registered
name its module lacks, a required pinned file missing. The completeness scan
over the v5 import closure (consensus_v4.problems' scan) is wired in at
integration, when the v5 entry points exist.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent

REGISTERED: dict[str, tuple[str, ...]] = {
    "fugal_subnet.config": (
        # the score (v5/DESIGN.md 1)
        "ARENA_A_REF_V5", "Y_SCALE_V5",
        # the exam (DESIGN 5) and the TD's call budget (operational)
        "EXAM_QUESTIONS_V5", "EXAM_VOTES_V5", "EXAM_BASELINE_QUESTIONS_V5",
        "EXAM_CALLS_PER_HOTKEY_V5", "COVERAGE_CALLS_V5", "SPEC_LAG_V5", "HEAD_LAM_MAX_V5",
        "EXAM_BUDGET_S_V5", "EXAM_CALL_DEADLINE_S_V5",
        # the seat (DESIGN 3)
        "WINDOW_EPOCHS_V5", "N_MIN_V5", "N0_V5", "TAKE_MARGIN_V5", "UPGRADE_MARGIN_V5",
        "CONTENDERS_MAX_V5", "PAYOUT_SEAT_SHARES", "STANDING_EPOCHS_V4",
        # B* and the vote plausibility test (v5_epoch)
        "BASELINE_MIN_PAIRS_V5", "EXPLORE_PRIVATE_Z", "EXPLORE_PRIVATE_MIN_SAMPLES_V4",
        # the one bound (shared with v4)
        "BOUND_A_ALPHA_V4", "BOUND_LN2_UP_V4", "BOUND_UNITS_PER_POINT_V4",
    ),
    "fugal_subnet.seat_v5": (
        "TAKE_MARGIN", "UPGRADE_MARGIN", "N_MIN", "N0", "CONTENDERS_MAX", "SEAT_SHARES",
        "HISTORY_MAX", "TAKE_UNITS", "UPGRADE_UNITS",
    ),
    "fugal_subnet.v5_epoch": (
        "WINDOW_EPOCHS", "SPEC_LAG", "STANDING_EPOCHS", "Y_SCALE", "BASELINE_MIN_PAIRS",
        "IMPLAUSIBLE_Z", "IMPLAUSIBLE_MIN_SAMPLES", "NANO",
        "SPEC_KEY", "COSTS_KEY", "RECORD_KEY", "SEAT_KEY", "BASELINE_KEY", "HEADS_KEY",
    ),
}

# role -> (path under fugal_subnet/, required). Names are placeholders until
# lane P lands the files (graders_arena.py, the v5 pool manifest, the
# RouterArena exclusion list, the exam pool's embeddings).
PINNED_FILES: dict[str, tuple[str, bool]] = {
    "graders_arena": ("graders_arena.py", True),
    "pool_manifest_v5": ("benchmarks/pool_manifest_v5.json", True),
    "exclusions_routerarena": ("benchmarks/exclusions_routerarena.json", True),
    "embeddings_v5": ("routing_data/embeddings_v5.sha256", True),
}


def constants() -> dict:
    from fugal_subnet.consensus_v4 import canonical

    out = {}
    for module, names in sorted(REGISTERED.items()):
        try:
            mod = importlib.import_module(module)
        except Exception as e:  # noqa: BLE001 - recorded, never raised
            for name in names:
                out[f"{module}:{name}"] = f"unavailable: {type(e).__name__}"
            continue
        for name in names:
            try:
                out[f"{module}:{name}"] = canonical(getattr(mod, name))
            except (AttributeError, TypeError) as e:
                out[f"{module}:{name}"] = f"unavailable: {type(e).__name__}"
    return out


def file_hashes(package: Path | None = None) -> dict:
    root = PACKAGE if package is None else Path(package)
    out = {}
    for role, (rel, _required) in sorted(PINNED_FILES.items()):
        try:
            out[role] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError:
            out[role] = None
    return out


def activation() -> dict:
    from fugal_subnet import config

    finney = getattr(config, "FINNEY_V5_ACTIVATION_BLOCK", None)
    env = getattr(config, "V5_ACTIVATION_BLOCK", None)
    return {"finney": finney, "env": None if env == finney else env}


def section() -> dict:
    """The fingerprint's `v5` section."""
    return {"activation": activation(), "constants": constants(), "files": file_hashes()}


def problems(package: Path | None = None) -> list[str]:
    out = [f"{k} is {v}" for k, v in sorted(constants().items())
           if isinstance(v, str) and v.startswith("unavailable")]
    root = PACKAGE if package is None else Path(package)
    out += [f"pinned file {rel} ({role}) is missing" for role, (rel, required)
            in sorted(PINNED_FILES.items()) if required and not (root / rel).is_file()]
    return out
