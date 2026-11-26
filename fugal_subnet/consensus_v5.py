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
        "MINER_PAID_FRACTION_V5",
        # B* and the vote plausibility test (v5_epoch)
        "BASELINE_MIN_PAIRS_V5", "EXPLORE_PRIVATE_Z", "EXPLORE_PRIVATE_MIN_SAMPLES_V4",
        # the memorisation rule: the exam's fresh draw (v5_epoch)
        "FRESH_RESERVE_EPOCHS_V5", "EPOCH_INTERVAL",
        # v5's environment variables (kept out of the v4 section's two lists)
        "CONSENSUS_ENV_INPUTS_V5", "OPERATIONAL_ENV_VARS_V5",
        # the one bound (shared with v4)
        "BOUND_A_ALPHA_V4", "BOUND_LN2_UP_V4", "BOUND_UNITS_PER_POINT_V4",
        # the pool's exam/coverage split and the routing embeddings' pin
        "COVERAGE_PARTITION_V5", "EMBEDDINGS_V5_SHA256",
        # v0.8.1: raw commitment blocks from this block
        "FINNEY_V5_COMMIT_FIX_BLOCK",
        # the protocol schedule (docs/decisions/SIGNED_SCHEDULE.md): where the
        # fold starts, its lead and the switches it may set
        "FINNEY_SCHEDULE_FROM_BLOCK", "SCHEDULE_LEAD_EPOCHS", "SCHEDULE_SWITCHES",
    ),
    # The schedule document's format and the commitment pointer's form: every
    # node must parse and refuse a schedule alike (protocol_schedule.fold).
    "fugal_subnet.protocol_schedule": (
        "KIND", "FORMAT", "DRAFT_SERIAL", "MAX_SERIAL", "MAX_BLOCK", "MAX_DOCUMENT_BYTES",
        "MAX_SIGNATURE_BYTES", "MAX_ACTIVATIONS", "NAME_PATTERN", "POINTER_PREFIX",
    ),
    # The v5 pool's keyed order and exam/coverage split (its content is pinned
    # by pool_manifest_v5.json's content hash, below).
    "fugal_subnet.benchmarks.arena_sets": ("POOL_SALT",),
    "fugal_subnet.seat_v5": (
        "TAKE_MARGIN", "UPGRADE_MARGIN", "N_MIN", "N0", "CONTENDERS_MAX", "SEAT_SHARES",
        "HISTORY_MAX", "TAKE_UNITS", "UPGRADE_UNITS", "PAID_FRACTION",
    ),
    # The v5 menu step's one category (validator._v5_menu_category_of).
    "neurons.validator": ("V5_MENU_CATEGORY",),
    "fugal_subnet.v5_epoch": (
        "WINDOW_EPOCHS", "SPEC_LAG", "STANDING_EPOCHS", "Y_SCALE", "BASELINE_MIN_PAIRS",
        "IMPLAUSIBLE_Z", "IMPLAUSIBLE_MIN_SAMPLES", "NANO", "EPOCH_INTERVAL",
        "FRESH_RESERVE_EPOCHS", "BLOCKS_PER_EPOCH", "SENIORITY_BASE",
        "SPEC_KEY", "COSTS_KEY", "RECORD_KEY", "SEAT_KEY", "BASELINE_KEY", "HEADS_KEY",
    ),
}

# role -> (path under fugal_subnet/, required): the arena grader, RouterArena's
# prompt templates, the v5 pool manifest (its content hash covers every
# prompt, gold and route_text) and the RouterArena exclusion list. The routing
# embeddings are pinned by config.EMBEDDINGS_V5_SHA256 (a registered constant).
PINNED_FILES: dict[str, tuple[str, bool]] = {
    "graders_arena": ("graders_arena.py", True),
    "arena_prompts": ("benchmarks/arena_prompts.json", True),
    "pool_manifest_v5": ("benchmarks/pool_manifest_v5.json", True),
    "exclusions_routerarena": ("benchmarks/exclusions_routerarena.json", True),
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


def problems(package: Path | None = None, cfg=None) -> list[str]:
    out = [f"{k} is {v}" for k, v in sorted(constants().items())
           if isinstance(v, str) and v.startswith("unavailable")]
    root = PACKAGE if package is None else Path(package)
    out += [f"pinned file {rel} ({role}) is missing" for role, (rel, required)
            in sorted(PINNED_FILES.items()) if required and not (root / rel).is_file()]
    if cfg is None:
        from fugal_subnet import config as cfg
    pin = getattr(cfg, "EMBEDDINGS_V5_SHA256", None)
    if not pin or pin == "TBD":
        out.append("config.EMBEDDINGS_V5_SHA256 is not pinned (TBD): build and publish the v5 "
                   "routing embeddings, then pin their sha256 and EMBEDDINGS_V5_URL")
    return out
