"""Protocol v4's part of the consensus digest: every constant and pinned file
that decides a v4 score, the v4 menu, exploration, the proof schema's bounds
or verification, in one registry.

The digest two validators compare (`fingerprint.consensus_digest`) covers
what a v1-v3 epoch is scored with, and a v1-v3 epoch must keep publishing
exactly the digest it published before v4 was built. So v4's constants are
not added to that material: they form a separate `v4` section of the
fingerprint, present only in the fingerprint of an epoch at or after the v4
activation block (`fingerprint.environment_fingerprint(protocol_version=4)`).
A v4 epoch's digest is then the v3 material plus this section, and a v3
epoch's digest is byte for byte what it was.

The section has three parts:

- `activation`: the compiled finney block and the variable other chains read,
  as the epoch block records v3's;
- `constants`: every name in `REGISTERED`, read off its module, as
  `<module>:<NAME>` with its value in a canonical JSON form;
- `files`: sha256 of every pinned file v4 reads (`PINNED_FILES`), None for
  an optional file the release does not ship.

What keeps the registry complete is `problems()` (run by
`tests/test_consensus_v4.py` and `scripts/check_safety_invariants.py`): every
module-level binding of an UPPER_CASE name, in any statement form, and every
class-level constant, of every module in the import closure of the v4 entry
points (the scan section below says which modules and forms) must be
registered, or named in `NOT_V4` with the reason it decides nothing a v4
epoch publishes, or be a name the v3 release had with the same statements
(`PRE_V4`, pinned by consensus_v4_pre.json). A constant added by a later
package fails that check until it is classified; moving a name between
classes needs `CLASS_PINS` updated in the same commit.
Importing this module imports nothing else; the section is computed lazily.
"""
from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib
import json
import re
from fractions import Fraction
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent

# Every constant that decides a v4 score, the menu, exploration, the proof
# schema's bounds or verification, by module. Order is irrelevant: the
# section's JSON is canonical.
REGISTERED: dict[str, tuple[str, ...]] = {
    "fugal_subnet.config": (
        # exploration (LIVE_MENU_V4.md 8; SEAT_MINIMAL_V4.md 5.3, 8)
        "EXPLORE_SHARED_SIZE_V4", "EXPLORE_COST_AWARE_SIZE_V4", "EXPLORE_COVERAGE_SIZE_V4",
        "EXPLORE_PRIVATE_SIZE_V4", "EXPLORE_COLDKEY_CAP_V4", "EXPLORE_SPEND_CAP_USD_V4",
        "EXPLORE_COVERAGE_TARGET_V4", "EXPLORE_COVERAGE_DAMP_PAIRS_V4",
        "EXPLORE_COVERAGE_SPEND_CAP_USD_V4",
        # the scored slice and the question mix (LIVE_MENU_V4.md 10)
        "SLICE_SIZE_V4", "MIX_FLOOR_PCT_V4", "MIX_STEP_PCT_V4", "MIX_MIN_PAIRS_V4",
        "MIX_CYCLE_EPOCHS_V4",
        # proposals (LIVE_MENU_V4.md 5)
        "PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4",
        # the one bound, the labels, the real-picks check (SEAT_MINIMAL_V4.md 3)
        "BOUND_A_ALPHA_V4", "BOUND_LN2_UP_V4", "BOUND_UNITS_PER_POINT_V4",
        "SEAT_WINDOW_EPOCHS_V4", "PICKS_MARGIN_POINTS_V4", "PICKS_MIN_QUESTIONS_V4",
        "FAIL_MAX_BP_V4",
        # the fast crown (FAST_CROWN_V4.md 0, 2, 12): the bound's tuning, the
        # challenge's sizes, its list and whose picks it pairs with
        "BOUND_N0_V4", "CHALLENGE_SIZES_V4", "CHALLENGE_LIST_SIZE_V4", "CHALLENGE_HOLDER_LAG_V4",
        # a challenge's holder-side failures and void challenges in a run
        "HOLDER_FAIL_CAP_BP_V4", "VOID_MAX_IN_RUN_V4",
        # the seat: rule A, standing, the late odds a reveal carries, and the
        # plausibility check's minimum sample (SEAT_MINIMAL_V4.md 3.3, 3.7)
        "ODDS_MARGIN_POINTS_V4", "ODDS_MIN_PAIRS_V4", "STANDING_EPOCHS_V4", "LATE_ODDS_SLOTS_V4",
        "EXPLORE_PRIVATE_MIN_SAMPLES_V4",
        # the memorisation check and its reserve (SEAT_MINIMAL_V4.md 6)
        "MEMO_MARGIN_POINTS_V4", "MEMO_MIN_LABELS_V4", "MEMO_MIN_QUESTIONS_V4",
        "FRESH_RESERVE_EPOCHS_V4",
        # whose commitment names each epoch's menu (LIVE_MENU_V4.md 4.3)
        "FINNEY_ROOT_HOTKEYS_V4",
    ),
    "fugal_subnet.menu_state": (
        "PICO", "MILLI", "PPM", "BP", "MAX_INT", "MAX_STRING", "MENU_LAG", "MENU_CAP",
        "MENU_MIN_ENTRIES", "MENU_MAX_PROVISIONAL", "MENU_MAX_JOINS_PER_EPOCH",
        "MENU_JOINS_SMALL_FIELD", "MENU_JOINS_FIELD_HOTKEYS", "PROVISIONAL", "ENTRY_SHIELD_EPOCHS",
        "RATE_SCALE", "ENDPOINT_MIN_VOTERS",
        "TOKEN_MIN_PAIRS", "MENU_REFRESH_EPOCHS", "MENU_PRICE_STEP", "PROPOSAL_MIN_ANSWERED_BP",
        "PROPOSAL_MIN_CATEGORY_GRADED", "PROPOSAL_USEFUL_MIN_PAIRS", "SIBLING_MARGIN_PPM",
        "SIBLING_CHEAPER_BP", "PROPOSAL_COOLDOWN_EPOCHS", "RETIRE_WINDOW_EPOCHS", "RETIRE_MIN_CALLS", "RETIRE_FAIL_RATE_BP",
        "RETIRE_FAIL_MIN_EPOCHS", "RETIRE_FAIL_MIN_COLDKEYS", "RETIRE_OUTAGE_RATE_BP",
        "RETIRE_MIN_FLOOR",
        "RETIRE_GONE_EPOCHS", "RETIRE_USE_FLOOR_BP", "RETIRED_KEEP", "MENU_MAX_BYTES",
        "PICK_COST_PRIOR_CELLS_ENTRY", "PICK_COST_PRIOR_CELLS_CATEGORY", "PICK_MAX_CATEGORIES",
        "PICK_CATEGORY_MAX_CHARS",
        # The published reasons and the state document's keys are in every
        # menu's bytes, so in its hash, which names every proof's namespace.
        "REASON_FAILING", "REASON_GONE", "REASON_LAB_REMOVED", "REASON_HOST_REMOVED",
        "REASON_DISPLACED", "REASON_GATE_ANSWERED", "REASON_GATE_CATEGORY",
        "REASON_GATE_USEFUL", "REASON_GATE_COST", "REASON_GATE_SIBLING", "REASON_PRICE_CEILING",
        "REASONS",
        "HEALTH_REASONS", "_ENTRY_REASONS", "_GATE_REASONS", "LIST_REFUSAL_REASONS",
        "_REQUEST_KEYS", "_ENTRY_KEYS", "_PROVISIONAL_KEYS", "_PICK_KEYS", "_PICK_CATEGORY_KEYS",
        "READER_PARAMS", "_REFRESHED", "_IDENTITY",
    ),
    # Proposals filed on chain and the root's join delta (FAST_CROWN_V4.md
    # 18): the pinned read's margin, the documents' and the delta's shapes
    # and bounds, the commitment fields' forms and the verdicts.
    "fugal_subnet.chain_proposals": (
        "JOIN_READ_MARGIN_BLOCKS", "PROPOSAL_DOMAIN", "DELTA_DOMAIN", "PROPOSAL_MAX_BYTES",
        "MAX_CONSIDERED", "DELTA_MAX_BYTES", "MAX_SERVING", "MAX_HOTKEY_LEN",
        "MAX_SUPPORTED_PARAMETERS",
        "PROPOSAL_FIELD_PREFIX", "JOIN_FIELD_PREFIX", "_HEX64", "_PROPOSAL_FIELD", "_JOIN_FIELD",
        "JOINED", "NOT_SERVED", "UNFETCHED", "UNREAD", "MALFORMED", "COLDKEY_ONCE", "NO_PLACE",
        "REFUSED", "_VERDICTS", "SERVED_FIELD",
    ),
    # A provisional entry's calls in the TD: their deadline and the budget
    # past which none starts (what a proof reports as failed).
    "fugal_subnet.tee.harness": ("PROVISIONAL_CALL_DEADLINE_S", "PROVISIONAL_BUDGET_S",
                                 "PROVISIONAL_SPEND_CAP_USD", "PROVISIONAL_PROMPT_OVERHEAD_TOKENS"),
    "fugal_subnet.menu_rules": (
        "MENU_MAX_HOSTS", "MAX_ANSWER_TOKENS", "MAX_PRICE_IN_PICO", "MAX_PRICE_OUT_PICO",
        "REASONING_EFFORTS", "QUANTIZATIONS",
        "UNKNOWN_QUANTIZATION", "HOST_TIER_SUFFIXES", "ALLOWLIST_REFUSALS", "MAX_MODEL_ID_LEN",
        "MAX_LAB_LEN", "MAX_HOST_LEN", "MAX_HOST_TAG_LEN", "_NAME", "_LAB_RE", "_MODEL_RE",
        "_HOST_RE", "_HOST_TAG_RE", "LIST_NAMESPACE", "LIST_FORMATS", "KIND_MENU_ALLOWLIST",
        "KIND_DEVICE_LIST", "DRAFT_SERIAL", "MAX_SERIAL", "MAX_LIST_BYTES", "MAX_LIST_ITEMS",
        "MAX_SIGNATURE_BYTES", "MAX_SIGNERS_BYTES", "LIST_SIGNERS_SHA256", "SHIPPED_LISTS",
        "_DEVICE_RE", "_LIST_FIELDS", "_KEY_TYPE", "_SIGNERS_LINE", "_ARMOR_BEGIN", "_ARMOR_END",
        "_MAGIC", "_HASH",
    ),
    "fugal_subnet.menu_inputs": (
        "Z_95", "_SQRT_SCALE", "_VOTE_KEYS", "_CALL_KEYS", "_PAIR_KEYS", "_USE_KEYS",
        "_GRADED_KEYS", "_SUMMARY_STRINGS",
    ),
    # The reveal's `menu` field shape ({epoch, hash, source: {epoch, hash}}),
    # in the view validators are compared on.
    "fugal_subnet.menu_publish": ("REVEAL_SUMMARY", "LIST_KINDS", "MENU_FIELD_KEYS",
                                  "MENU_SOURCE_KEYS"),
    "fugal_subnet.seat_bound": ("A_ALPHA", "LN2_UP", "UNITS_PER_POINT"),
    "fugal_subnet.picks_v4": (
        "UNITS_PER_POINT", "BP", "FAIL_MAX_BP", "PICKS_MARGIN_POINTS", "PICKS_MARGIN_UNITS",
        "PICKS_MIN_QUESTIONS", "REVEAL_KEY", "DIGEST_KEY", "BOUND_N0", "INT32_MIN",
        "INT32_MAX", "_HEX", "_MAX_ID", "NO_ITEM", "HOLDER_FAIL_CAP_BP", "VOID_MAX_IN_RUN",
    ),
    "fugal_subnet.labels_v4": ("SEAT_WINDOW_EPOCHS", "PROVISIONAL_LABEL"),
    # The seat (SEAT_MINIMAL_V4.md 3.3 to 3.8): the rules' values as the
    # module compiles them, the state's bounds (what a published seat state
    # may hold, so what a rebuild accepts), the evidence's keys and the
    # reference's name in it, and the patterns its state's lines are read with.
    "fugal_subnet.seat_v4": (
        "UNITS_PER_POINT", "STANDING_EPOCHS", "ODDS_MARGIN_POINTS", "ODDS_MARGIN_UNITS",
        "ODDS_MIN_PAIRS", "PICKS_MARGIN_UNITS", "PICKS_MIN_QUESTIONS", "BOUND_N0", "HISTORY_KEEP",
        "MAX_PREDECESSORS", "MAX_STATE_HEADS", "MAX_ROWS", "MAX_ENTRY_ID", "REVEAL_KEY",
        "DIGEST_KEY", "R", "_HASH", "_ROW_ITEM", "_EPOCH_ITEM", "HOLDER_LAG", "_EPOCH_KEY",
        "_INDEX_ITEM",
    ),
    # The memorisation check (SEAT_MINIMAL_V4.md 6): its compiled values,
    # the published set's lag and its seed and state formats, the evidence's
    # part names, and the reserve's per-category minimum.
    "fugal_subnet.memo_v4": (
        "SEED_FORMAT", "STATE_FORMAT", "FRESH_RESERVE_EPOCHS", "MEMO_MARGIN_UNITS",
        "MEMO_MIN_LABELS", "MEMO_MIN_QUESTIONS", "READ_LAG", "SEEN", "UNSEEN",
        "PUBLISHED_MIN_PER_CATEGORY", "_FORMER_MEASURE_ROOM",
        # The most bytes a node reads of the published set a reveal names
        # (its snapshot): past it the set cannot be had, as a menu past
        # MENU_MAX_BYTES cannot.
        "MAX_SNAPSHOT_BYTES",
    ),
    "fugal_subnet.epoch_plan": ("MIX_V4_FORMAT", "_PPM"),
    "fugal_subnet.explore_weights": ("WEIGHT_SCALE", "SHARE_SCALE", "_SQRT_SCALE"),
    "fugal_subnet.head_eval": (
        "V4_REQUIRED_FIELDS", "V4_IGNORED_FIELDS", "HEAD_MAX_ROWS_V4", "_V4_DIM", "_V4_MAX_ID_LEN",
        "V4_PARAM_DTYPES",
    ),
    "fugal_subnet.head_odds": ("ODDS_SCALE",),
    "fugal_subnet.tee.td_v4": ("WIRE_KEYS", "WIRE_PROVIDER_KEYS", "CHALLENGE_CHOICES"),
    "fugal_subnet.tee.endpoint_reads": (
        "PICO", "MAX_LISTING_BYTES", "MAX_LISTING_ENDPOINTS", "MAX_READ_MODELS", "MAX_READ_HOSTS",
        "MAX_PRICE_PICO", "MAX_ANSWER_LIMIT", "MAX_ABS_STATUS", "MAX_MODEL_LEN", "MAX_HOST_LEN",
        "MAX_QUANTIZATION_LEN", "MAX_DECIMAL_LEN", "_NAME", "MODEL_RE", "HOST_RE", "_QUANT_RE",
        "_DECIMAL_RE", "_HOST_KEYS",
    ),
    # The billed cost each v4 call result carries, exact (nano-dollars).
    "fugal_subnet.tee.runtime": ("NANO", "_COST_RE"),
    # An operator's own v4 options as a TD takes them (the proposal it
    # serves and commits, the size of the challenge it answers).
    "fugal_subnet.tee.v4_options": (
        "MAX_PROPOSAL_CHARS", "MAX_CHALLENGE_CHARS", "MAX_CHALLENGE_USD_CHARS", "_PRINTABLE",
        "_PROPOSAL_KEYS",
    ),
    "fugal_subnet.tee.proof": (
        "V4_RESULT_KEYS", "V4_PROOF_KEYS", "MAX_PROVIDER_COST_NANO", "MAX_SERVED_BY_LEN",
        "CHALLENGE_SIZES", "_CHALLENGE_KEYS",
    ),
    # Pre-v4 modules whose constants a v4 epoch also reads, and which the
    # v1-v3 digest does not carry (compiled, never in the scoring block).
    "fugal_subnet.explore_pool": ("_VARIANCE_CLAMP",),
    "fugal_subnet.copy_rule": ("MAX_COMMITMENTS", "MAX_STATE_HEADS"),
    "fugal_subnet.reveal_publisher": ("COMMITMENT_MAX_BYTES", "_MENU_TOKEN"),
    "fugal_subnet.reveal_sync": ("MAX_JOIN_FETCH_BYTES",),
    "fugal_subnet.routing_protocol": ("REQUEST_FIELDS",),
}

# Constants of the scanned modules that decide nothing a v4 epoch publishes,
# each with the reason. A path is here because the bytes it names are in
# PINNED_FILES; a local bound is here because it decides how long a node
# waits or how much it keeps, never what any epoch is scored on.
#
# Config names v1-v3 read and a v4 epoch does not, since the seat package
# removed their rules from the v4 path (SEAT_MINIMAL_V4.md 12). They stay in
# the v1-v3 digest's scoring block, which a v4 digest also covers.
_V1_V3_ONLY = (
    "the v1-v3 seat's parameters (v2_epoch.seat_params_from_config); a v4 epoch "
    "decides by seat_v4 and reads none of them",
    "the v1-v3 burn-in counter; a v4 reveal publishes none and the v4 seat reads none",
    "the cross-miner agreement detector, which a v4 epoch does not run (SEAT_MINIMAL_V4.md "
    "2.3 item 32); a v4 epoch's plausibility minimum is EXPLORE_PRIVATE_MIN_SAMPLES_V4",
    "v1's copy detection (dedup.find_duplicates), which a v4 epoch does not run",
)
# Protocol v5's names (~/projects/fugal-v2-handoff/v5/DESIGN.md): no v4 epoch
# reads them (a v5 proof field is absent from every v4 proof, and the v5 TD
# path runs only from the v5 activation); the v5 digest registers them.
_V5_ONLY = "protocol v5 only, registered by the v5 digest (consensus_v5); no v4 epoch reads it"
NOT_V4: dict[str, dict[str, str]] = {
    "fugal_subnet.config": {
        "V4_ACTIVATION_BLOCK_ENV": "recorded in the section's `activation` part",
        "FINNEY_V4_ACTIVATION_BLOCK": "recorded in the section's `activation` part",
        "CHAMPION_MARGIN": _V1_V3_ONLY[0],
        "CHAMPION_CONFIDENCE": _V1_V3_ONLY[0],
        "CHAMPION_BOOTSTRAP_RESAMPLES": _V1_V3_ONLY[0],
        "CHAMPION_HARD_WEIGHT": _V1_V3_ONLY[0],
        "CHAMPION_HARD_GATE_MIN_QUESTIONS": _V1_V3_ONLY[0],
        "CHAMPION_GHOST_OFF": _V1_V3_ONLY[0],
        "CHAMPION_GHOST_RETIRE": _V1_V3_ONLY[0],
        "CHAMPION_GHOST_RETIRE_CONFIRM": _V1_V3_ONLY[0],
        "CHAMPION_MIN_ACCURACY": _V1_V3_ONLY[0],
        "CHAMPION_MIN_PAIRED_QUESTIONS": _V1_V3_ONLY[0],
        "CHAMPION_CONFIRM_QUESTIONS": _V1_V3_ONLY[0],
        "CHAMPION_CONFIRM_OPEN_EPOCHS": _V1_V3_ONLY[0],
        "CHAMPION_VALUE_PER_CORRECT_USD": _V1_V3_ONLY[0],
        "BRIER_MARGIN": _V1_V3_ONLY[0],
        "NV_NONINFERIORITY": _V1_V3_ONLY[0],
        "ODDS_MIN_PAIRS": _V1_V3_ONLY[0],
        "ODDS_CONFIRM_PAIRS": _V1_V3_ONLY[0],
        "V2_BURN_IN_QUESTIONS": _V1_V3_ONLY[1],
        "EXPLORE_DISAGREE_MAX": _V1_V3_ONLY[2],
        "EXPLORE_DISAGREE_MIN_SAMPLES": _V1_V3_ONLY[2],
        "EXPLORE_DISAGREE_MIN_VOTERS": _V1_V3_ONLY[2],
        "DEDUP_EPSILON": _V1_V3_ONLY[3],
        "DEDUP_EPSILON_AGREEMENT": _V1_V3_ONLY[3],
        "DEDUP_PARAM_DISTANCE": _V1_V3_ONLY[3],
        "DEDUP_SIMILARITY_THRESHOLD": _V1_V3_ONLY[3],
        "CHALLENGE_BUDGET_S_V4": "the TD's time budget for a challenge's calls: no validator "
                                 "reads it; what it cuts is in the proof (own calls by the "
                                 "reported size, holder-side calls marked not started)",
        "CHALLENGE_SPEND_USD_V4": "the default of the TD's dollar ceiling on a challenge's "
                                  "realised spend: no validator reads it; what it cuts is in "
                                  "the proof",
        "CHALLENGE_SPEND_MAX_USD_V4": "the largest dollar ceiling an operator may give its TD: "
                                      "no validator reads it",
        "MAX_WEIGHT_DELTA": "the v1-v3 per-epoch weight cap; no v4 vector is capped, the "
                            "all-burn vector of an epoch with no valid proof included "
                            "(SEAT_MINIMAL_V4.md 2.2 #28)",
    },
    "fugal_subnet.menu_state": {"SEED_PATH": "a path; the seed's bytes are in PINNED_FILES"},
    "fugal_subnet.menu_rules": {
        "SIGNED_LISTS_DIR": "a path; the lists' bytes are in PINNED_FILES",
        "SIGNERS_PATH": "a path; the signers' bytes are in PINNED_FILES",
        "_ITEM_RULES": "functions (code), not values",
    },
    "fugal_subnet.menu_publish": {
        "PUBLISH_BUDGET_S": "how long the root waits on its store (timing)",
        "REBUILD_MAX_LINKS": "how far one walk goes before it stops (a local bound)",
        "MAX_AUDIT_DIFFERENCES": "the size of a follower's audit, a per-validator field",
        "MAX_HELD_EPOCHS": "how many menus a node keeps on disk",
        "MAX_REVEAL_READ_BYTES": "how much of a reveal a node reads when anchoring",
    },
    "fugal_subnet.labels_v4": {
        "REFERENCE": "a sentinel object (the reference's side)",
        "_INDEXES": "a per-process cache of what each epoch's labels index to (derived "
                    "from the labels, holds no value of its own)",
        "_REFERENCES": "a per-process cache of R for the same epochs' labels objects (derived "
                       "from them, holds no value of its own)",
        # gain_strata_many's exact integer packing and overflow guards: they
        # choose how the same sums are computed, never their values
        "_SHIFT": "the bit position of the packed loss (an exact integer encoding)",
        "_Q_LIMIT": "the packed range; past it an epoch is summed the per-pair way, to the "
                    "same integers",
        "_INT64_MAX": "the int64 range, which sizes the chunks of exact int64 sums",
        "_LIMB": "the split of a sum of squares into two exact int64 limbs",
        "_LIMB_MASK": "the split of a sum of squares into two exact int64 limbs",
        "_ROW_BLOCK": "how many pairs are computed together (an array size)",
        "_CHUNK": "how many labels one array pass reads (an array size)",
        "_TILE": "how many rows one cache-sized pass reads (an array size)",
        "_X_GROUP": "how many heads one pass computes together (an array size)",
        "_TILE_CHUNK": "how many labels one tile pass reads (an array size)",
        "_THREADS": "how many epochs are summed at once; the integer sums do not depend on it",
    },
    "fugal_subnet.seat_v4": {
        "RULE_C_FIRST_ROUND": "how many of rule C's comparisons are computed in decide's first "
                              "batch; the rule reads and publishes the same ones whatever it is",
    },
    "fugal_subnet.seat_bound": {
        "INT64_LIMIT": "the int64 range, which picks int64 or Python-int sums; both give "
                       "the same exact sum, so it changes no v4 value",
    },
    "fugal_subnet.memo_v4": {"SEED_PATH": "a path; the seed's bytes are in PINNED_FILES"},
    # How the published set moves between nodes: where it is kept and
    # published, how far a node replays in one go. What the
    # set is, is memo_v4's (registered above); a snapshot is read only when
    # it hashes to the digest a reveal names.
    "fugal_subnet.memo_sync": {
        "SNAPSHOT_PREFIX": "where a snapshot is stored; readers check its bytes by digest",
        "STATE_SUFFIX": "a file name beside the validator's state",
        "TD_FILE": "a file name on the TD's disk",
        "HELD_FORMAT": "the format of a node's own state file (never published)",
        "PUBLISH_BUDGET_S": "how long the root waits on its store (timing)",
        "MAX_FOLD_EPOCHS": "how many boundaries a node replays before it anchors (a local bound)",
        "ANCHOR_HISTORY_EPOCHS": "the chain history a node's endpoint needs to re-anchor (a "
                                 "message and the operators' guides; the anchor is the root's "
                                 "snapshot, checked by digest)",
        "_PRUNED_MARKERS": "how a node tells a pruned read from a transient one (when it "
                           "re-anchors on the root's snapshot; the set it reaches is the root's)",
        "_HEX64": "the shape of a sha256 digest, a reader's check",
    },
    "fugal_subnet.epoch_plan": {"MIX_V4_PATH": "a path; the mix's bytes are in PINNED_FILES"},
    "fugal_subnet.head_odds": {
        "HEAD_MAX_ABS_PARAM": "in the v1-v3 digest's scoring block",
        "THRESHOLDS_SHA256": "in the v1-v3 digest's scoring block",
        "_THRESHOLDS": "a cache of the pinned table",
        "_U": "float64's unit roundoff in odds_matrix's error band, which only chooses which "
              "path computes an odds integer; both give the same one",
    },
    "fugal_subnet.tee.td_v4": {
        "MAX_HOLDER_REVEAL_BYTES": "how much of a reveal the TD reads for its own choice",
        "MAX_CATCH_UP_EPOCHS": "how far the TD walks before it re-anchors (a local bound)",
        "RESUME_MAX_EPOCHS": "how old a kept menu the TD resumes from (a local bound)",
        "MAX_HELD_COMMITMENTS": "how many commitments the TD keeps in memory",
        "MENU_FILE": "a file name on the TD's disk",
    },
    "fugal_subnet.join_step": {
        "FETCH_WORKERS": "how many fetches the root makes at once (timing)",
        "RESPONSE_MAX_BYTES": "how much of one proposer's answer the root reads (a fetch "
                              "bound: past it the proposal is unfetched and waits, as a "
                              "timeout does)",
        "AUDIT_FIELD": "the name of a validator's own audit field in its reveal (not compared)",
        "LISTING_URL": "where the root fetches a listing; what it read is stated in the delta",
    },
    "fugal_subnet.chain_proposals": {
        "JOIN_FETCH_BUDGET_S": "the root's own time for fetching documents and listings",
        "JOIN_FETCH_TIMEOUT_S": "the root's own wait on one fetch",
        "JOIN_PUBLISH_BUDGET_S": "the root's own time for publishing the delta",
        "JOIN_COMMIT_SLACK_BLOCKS": "how close to the boundary the root still commits (timing)",
    },
    "fugal_subnet.tee.harness": {
        "CHALLENGE_KINDS": "the TD's own labels for a challenge's calls; the proof says what "
                           "each result is (`holder_pick`), and verification reads that",
    },
    "fugal_subnet.tee.provision": {
        "REMOVED_FIELDS": "the reason a push naming a field an earlier v4 build took is "
                          "refused (it is refused whatever the text says): an operator's "
                          "message, never an input of a TD's work",
    },
    "fugal_subnet.tee.proof": {
        "TPM_INTERMEDIATES_MAX": "the TPM half of attestation, v1-v3 verification",
        "TPM_INTERMEDIATE_MAX_BYTES": "the TPM half of attestation, v1-v3 verification",
        "V5_RESULT_KEYS": _V5_ONLY,
        "V5_PROOF_KEYS": _V5_ONLY,
    },
    "fugal_subnet.tee.td_v5": {
        "SPEC_LAG": _V5_ONLY,
        "COVERAGE_CALLS": _V5_ONLY,
        "Y_SCALE": _V5_ONLY,
        "EXAM_BUDGET_S": _V5_ONLY,
        "EXAM_CALL_DEADLINE_S": _V5_ONLY,
        "REVEAL_SPEC_KEY": _V5_ONLY,
        "MAX_SPEC_REVEAL_BYTES": _V5_ONLY,
        "EXAM_KINDS": _V5_ONLY,
    },
    # Read by v1-v3's seat and not by v4's (seat_v4 replaces it; the seat
    # package's deletions, SEAT_MINIMAL_V4.md 12).
    "fugal_subnet.champion": {
        "_BLOCK": "a v1-v3 bootstrap block size",
        "NV_UNITS_PER_POINT": "net value's units, which only v1-v3's seat tests compute",
        "UNIT_CLIP": "the clip on net value's units, which only v1-v3's seat tests compute",
        "HISTORY_KEEP": "the v1-v3 seat's history bound; the v4 seat's is seat_v4.HISTORY_KEEP",
    },
    "fugal_subnet.odds_score": {
        "BRIER_UNITS_PER_POINT": "points for v3's odds shadow table and logs; a v4 epoch's points "
                                 "are in BOUND_UNITS_PER_POINT_V4 (labels_v4, seat_v4)",
    },
    "fugal_subnet.v2_epoch": {
        "HARD_CATEGORY": "the hard-subset flag of a slice, read only by v1-v3's hard gate",
        "MAX_PENDING": "the bound on v1-v3's pending records, which a v4 seat state has none of",
        "V4_WINDOW_FORMAT": "the format tag of the window file on this node's disk",
        "V4_WINDOW_MAX_BYTES": "how much a window file on this node's disk may make the loader "
                               "allocate (a local bound)",
        "V4_WINDOW_ARRAYS": "the array names and dtypes of the window file on this node's disk",
    },
    "fugal_subnet.copy_rule": {
        "REVEAL_KEY": "a reveal field name, in the v1-v3 reveal already",
        "MAX_ADDRESS_LEN": "an input bound of v1-v3, unchanged",
        "_HASH": "a regex of v1-v3, unchanged",
        "_EPOCH_ID": "a regex of v1-v3, unchanged",
    },
    "fugal_subnet.reveal_publisher": {"MARKER": "a local marker file name"},
    # The one read bound of a reveal, which the TD's holder read and the root
    # summaries' read name too (both above, NOT_V4 for the same reason).
    "fugal_subnet.reveal_sync": {
        "MAX_REVEAL_BYTES": "how much of a reveal a node reads (a local bound every reader "
                            "names; a v4 reveal at 256 UIDs is at most three quarters of it, "
                            "tests/test_v4_reveal_budget.py); no reveal's content depends on it",
    },
    # A re-anchor's replay of this node's own epochs (v2_sync, v4_replay): it
    # runs run_v2_epoch on recorded inputs and the caller's memo for each
    # epoch, and decides no rule of its own.
    "fugal_subnet.v4_replay": {
        "FORMAT": "the format tag of the epoch records on this node's disk",
        "_SUFFIX": "a file name suffix on this node's disk",
        "_SHA": "a pattern the head files on this node's disk are named by",
    },
    "fugal_subnet.tee.runtime": {
        "ENDPOINT_READ_DEADLINE_S": "how long the TD waits on the listing (timing)"},
    "fugal_subnet.tee.v4_options": {
        "FIELDS": "the names of provisioning fields (the operator's channel into its TD)",
        "PARSERS": "the names of this module's parsers",
    },
}

# What the scan of the import closure of the v4 entry points found (the
# modules neurons/ runs, beyond the v4 modules themselves), classified by
# reading where each name is used on the v4 path. Kept apart from the tables
# above so that the packages that add v4 constants do not edit the same lines.
_POOL_BUILD = ("builds the benchmark pool; a v4 epoch reads the pool only through the loader, which "
               "checks it against the pinned manifest (pool_manifest_v3.json, named by the pinned "
               "pool_mix_v4.json), so a change here that altered a question fails that check")
_POOL_PATH = ("a path or file name of the pool's data; the bytes it names are checked against the "
              "pinned manifest")
_CLOSURE_REGISTERED: dict[str, tuple[str, ...]] = {
    # the cost of a call and the spend reservation, the draw of the private slice, the blocks an
    # epoch lasts
    "fugal_subnet.api": ("OR_URL", "PROMPT_TOKEN_OVERHEAD"),
    "fugal_subnet.pricing": ("CHARS_PER_TOKEN",),
    "fugal_subnet.benchmarks.slicer": ("BLOCK_TIME_S", "_PRIVATE_DRAW_TRIES"),
    # the lengths a synapse field may have (what a validator accepts of a miner's proof and head)
    "fugal_subnet.protocol": ("EVENT_LOG_MAX_LEN", "HASH_MAX_LEN", "HEAD_B64_MAX_LEN",
                              "PROOF_JSON_MAX_LEN", "PROPOSAL_DOC_MAX_LEN"),
    # what the grader is shown: which domain each benchmark is graded as
    "fugal_subnet.grading_task": ("_BENCH_TO_DOMAIN",),
    "fugal_subnet.code_gate": ("CODE_CHECKERS",),
    "fugal_subnet.reference_frame": ("_Z",),
    "fugal_subnet.routing_protocol": ("MENU_A", "MENU_B"),
    # proof verification: the quote's layout and the TPM structures
    "fugal_subnet.scale": ("DSTACK_TPM_MAGIC",),
    "fugal_subnet.tee.attestation": (
        "_FIELDS", "_MIN_QUOTE_LEN", "_RTMR_DIGEST_BYTES", "_TDX_QUOTE_VERSION", "_TEE_TYPE_TDX",
        "_DSTACK_EVENT_TYPE", "_EVENT_NAME"),
    "fugal_subnet.tee.tpm": (
        "TPM_ALG_ECDSA", "TPM_ALG_SHA256", "TPM_GENERATED", "TPM_ST_ATTEST_QUOTE",
        "_GOOGLE_CA_ADDRESS", "_ROOT_SHA256"),
    # the provisioning channel's fields
    "fugal_subnet.tee.provision": ("ALLOWED_FIELDS",),
    # which TD measurements a validator approves
    "fugal_subnet.validator_setup": ("PINNED_BASE_MEASUREMENTS", "SUPERSEDED_COMPOSES", "_FROZEN"),
    # the head contract: the backbone, its profile, the head's bounds
    "fugal_subnet.vendor.success_contract": (
        "BACKBONE_FILES", "CONTRACT", "MAX_BYTES", "MAX_EXPANDED_BYTES", "MODEL_ID", "PROFILE",
        "PROFILE_ID", "QUANTUM", "REVISION", "SYSTEM_PROMPT"),
    # reveal fields and what is compared between validators and fetched by followers
    "fugal_subnet.menu_publish": ("REVEAL_NEXT", "REVEAL_NEXT_HASH", "REVEAL_EVENTS",
                                  "REVEAL_LISTS", "REVEAL_AUDIT"),
    "fugal_subnet.v2_sync": ("MENU_FIELD", "MENU_ROLE_FIELDS", "MENU_SUMMARY_FIELD",
                             "PER_VALIDATOR_FIELDS"),
    "fugal_subnet.reveal_sync": ("MAX_LIST_FETCH_BYTES", "MAX_MENU_FETCH_BYTES",
                                 "MAX_SIGNATURE_FETCH_BYTES", "_LIST_KIND", "_POINTER"),
    # the variable names classed consensus input or operational (added to by v4)
    "fugal_subnet.config": ("CONSENSUS_ENV_INPUTS", "OPERATIONAL_ENV_VARS"),
}
_CODE_SETS = ("a pool-building constant of the code benchmarks: " + _POOL_BUILD)
_CLOSURE_NOT_V4: dict[str, dict[str, str]] = {
    "fugal_subnet.axon_allowlist": {
        "VALIDATOR_ADVERTISED_PORT": "the port the validator's axon advertises on the chain; "
                                     "no score, proof or reveal depends on it"},
    "fugal_subnet.benchmarks.code_sets": {
        **dict.fromkeys(
            ("CANDIDATES", "CC_COLUMNS", "CC_SAMPLE", "CC_TRAIN_SHARDS", "CODE_LICENCES",
             "DATASET_REVISIONS_CODE", "FLOATISH", "FUNC_SUFFIX", "ID_HASH_CHARS", "LOADERS_CODE",
             "MAX_CANDIDATE_TESTS", "MAX_TEST_CHARS", "MIN_TESTS", "MULTI_ANSWER", "PYTHON3",
             "STDIO_SUFFIX"), _CODE_SETS),
        "LEETCODE_TRAIN_FILE": _POOL_PATH,
    },
    "fugal_subnet.benchmarks.gpqa": {n: "a pool-building constant of GPQA: " + _POOL_BUILD
                                     for n in ("_CHOICE_COLS", "_LABELS")},
    "fugal_subnet.benchmarks.livecode": {"_CACHE_PATH": _POOL_PATH},
    "fugal_subnet.benchmarks.loader": {
        **dict.fromkeys(("DATASET_REVISIONS", "_BENCHMARKS"), "the loader's " + _POOL_BUILD),
        **dict.fromkeys(("DEFAULT_POOL_FILE", "DEFAULT_POOL_FILE_V2", "DEFAULT_POOL_FILE_V3",
                         "_EXCLUSIONS_NAME_V3", "_MANIFEST_NAME", "_MANIFEST_NAME_V2",
                         "_MANIFEST_NAME_V3"), _POOL_PATH),
    },
    "fugal_subnet.benchmarks.math_": {"_CONFIGS": "a pool-building constant of MATH: " + _POOL_BUILD},
    "fugal_subnet.benchmarks.mmlu": {"_CHOICES": "a pool-building constant of MMLU: " + _POOL_BUILD},
    "fugal_subnet.benchmarks.v2_sets": dict.fromkeys(
        ("INTEGER_HINT", "LETTERS", "LOADERS_V2", "MMLU_V2_SIZE", "OMNI_HARD_DIFFICULTY",
         "UNGRADEABLE_IFEVAL_KINDS", "V2_LICENCES", "_GRADEABLE_INTEGER"),
        "a pool-building constant of the v2 sets: " + _POOL_BUILD),
    "fugal_subnet.benchmarks.v3_sets": dict.fromkeys(
        ("ARC_CONFIGS", "BBH_BINARY_TASKS", "BBH_DROPPED_TASKS", "BBH_INTEGER_TASKS",
         "BBH_LETTER_TASKS", "BBH_SCRIPT_REVISION", "BBH_TASKS", "DATASET_REVISIONS_V3",
         "HELD_BACK_LICENCES", "HELD_BACK_REVIEW", "ID_HASH_CHARS", "LOADERS_HELD_BACK",
         "LOADERS_V3", "MATH_L12_LEVELS", "MIN_FORBIDDEN_WORD_CHARS", "NUMERIC_SUFFIX",
         "RLVR_REPEAT_FUNC", "V3_LICENCES", "WORD_COUNT_NO_MAX", "_BBH_OPTIONS"),
        "a pool-building constant of the v3 sets: " + _POOL_BUILD),
    "fugal_subnet.code_gate": {
        "_PROBES": "the startup self-test's probe questions (a TD or validator checks its own "
                   "code grader before it runs); they decide whether a node starts, never what "
                   "an epoch is scored on"},
    "fugal_subnet.collect": {
        "COLLECT_CONNECTION_LIMIT": "the size of the collection connection pool (networking)"},
    "fugal_subnet.commit_reveal": {
        "EPOCH_DIR": "a directory (FUGAL_EPOCH_DIR, operational): where a node keeps its own "
                     "epoch files"},
    "fugal_subnet.epoch_logger": {
        "EPOCH_LOG_DIR": "a directory (FUGAL_EPOCH_LOG_DIR, operational): where a node writes "
                         "its logs"},
    "fugal_subnet.determinism": {
        "DETERMINISM_ENV": "in the v1-v3 digest already: fingerprint's cpu_dispatch records each "
                           "of its variables as the process has them"},
    "fugal_subnet.head_archive": {
        "_HEX64": "the shape of a head file's name in this node's own archive"},
    "fugal_subnet.head_publisher": dict.fromkeys(
        ("EPOCH_MARKER", "FIRST_SUFFIX", "HEADS_PREFIX", "LISTED_SUFFIX", "MAX_FAILURES_PER_RUN",
         "PUBLISH_BUDGET_S", "RETRY_EPOCHS", "REVEAL_MARKER", "SENT_SUFFIX", "SIDECAR_NAME",
         "SPOOL_MAX_HEADS", "VERIFY_NOTE", "_EPOCH_ID", "_HEX64"),
        "the publication of verified heads beside the reveal (head_publisher: operational, "
        "nothing in it is read by a score, a weight, the reveal or the digest): file names, "
        "bounds and timing of this node's own uploads"),
    "fugal_subnet.logging_setup": {"_FORMAT": "the log line format"},
    "fugal_subnet.reference_frame": {
        "BOOTSTRAP_FRAME_PATH": "a path (FUGAL_FRAME_BOOTSTRAP, operational) to the v1-v3 "
                                "reference frame's bootstrap file; a v4 epoch runs no "
                                "agreement detector"},
    "fugal_subnet.tee.attestation": {
        "COLLATERAL_PROBE_TIMEOUT_S": "how long the startup probe of the collateral endpoint "
                                      "waits (timing)"},
    "fugal_subnet.tee.dstack_client": {
        "DSTACK_SOCKET": "the path of the TD's local dstack socket",
        "_MAX_RESPONSE": "the most the TD reads from its own local dstack daemon",
        "_TIMEOUT": "how long the TD waits on its own local dstack daemon (timing)"},
    "fugal_subnet.tee.tpm": {
        "_FETCH_DEADLINE_S": "how long a node waits fetching an issuing certificate (timing)",
        "_ROOTS_DIR": "a path; the pinned root's bytes are checked against _ROOT_SHA256 "
                      "(registered)"},
    "fugal_subnet.validator_setup": {
        "FINNEY_DEFAULT_ARCHIVE": "the default chain endpoint a validator connects to",
        "FINNEY_DEFAULT_FALLBACK": "the default fallback chain endpoint"},
    "fugal_subnet.config": {
        "V4_ACTIVATION_BLOCK_MALFORMED": "a FUGAL_V4_ACTIVATION_BLOCK that is not a block number: "
                                         "refused at startup off finney and ignored on finney "
                                         "while no v4 block is compiled, so no epoch runs on it "
                                         "(config.v4_activation_block)",
        "MENU_MIRRORS": "where menus are fetched from (FUGAL_MENU_MIRRORS, operational); a "
                        "fetched menu is accepted only by its hash"},
    "neurons.miner": {"METAGRAPH_REFRESH_S": "how often the miner re-reads the metagraph (timing)"},
    "neurons.validator": {
        "STATE_PATH": "a path (FUGAL_STATE_PATH, operational): where the validator keeps its state",
        "_ACTIVATION_HASH": "a per-process cache of the hash of each activation block's block",
        "_COLLECTION_LINE_UIDS": "how many uids a log line lists"},
}
for _module, _names in _CLOSURE_REGISTERED.items():
    REGISTERED[_module] = tuple(REGISTERED.get(_module, ())) + _names
for _module, _reasons in _CLOSURE_NOT_V4.items():
    NOT_V4.setdefault(_module, {}).update(_reasons)

# Protocol v5's names: read only by a v5 epoch (from config.v5_activation_block),
# which consensus_v5 registers; a v4 epoch reads none of them.
_V5_ONLY = ("a protocol v5 constant: read only from the v5 activation block on and "
            "registered by consensus_v5; a v4 epoch reads none of it")
_V5_NAMES: dict[str, tuple[str, ...]] = {
    "fugal_subnet.config": (
        "FINNEY_V5_ACTIVATION_BLOCK", "V5_ACTIVATION_BLOCK_ENV", "V5_ACTIVATION_BLOCK_MALFORMED",
        "ARENA_A_REF_V5", "EXAM_QUESTIONS_V5", "EXAM_BASELINE_QUESTIONS_V5", "EXAM_VOTES_V5",
        "EXAM_CALLS_PER_HOTKEY_V5", "COVERAGE_CALLS_V5", "WINDOW_EPOCHS_V5", "N_MIN_V5", "N0_V5",
        "TAKE_MARGIN_V5", "UPGRADE_MARGIN_V5", "CONTENDERS_MAX_V5", "SPEC_LAG_V5",
        "HEAD_LAM_MAX_V5", "Y_SCALE_V5", "EXAM_BUDGET_S_V5", "EXAM_CALL_DEADLINE_S_V5",
        "BASELINE_MIN_PAIRS_V5"),
    "fugal_subnet.arena_score": (
        "Y_SCALE", "UNITS_PER_POINT", "A_REF", "NANO", "ARENA_BETA", "ARENA_COST_MAX_PER_1K",
        "ARENA_COST_MIN_PER_1K", "NANO_PER_QUERY_PER_USD_PER_1K", "COST_MAX_ND", "COST_MIN_ND",
        "LOG2_BITS", "LN2", "LOG2_SPAN"),
    "fugal_subnet.exam_v5": (
        "EXAM_QUESTIONS", "EXAM_BASELINE_QUESTIONS", "EXAM_VOTES", "EXAM_CALLS_PER_HOTKEY",
        "HEAD_LAM_MAX", "Y_SCALE", "FAILED_CELL_MIN_VOTERS", "_NO_PICK"),
}
for _module, _names in _V5_NAMES.items():
    NOT_V4.setdefault(_module, {}).update(dict.fromkeys(_names, _V5_ONLY))

# The names each pre-v4 module had in the v3 release. They are classified
# by the v1-v3 mechanisms (the scoring block, the protocol identity, the
# environment lists); v4 reads them through those. A name added since is
# not here, and must be registered or put in NOT_V4: problems() holds every
# name here to consensus_v4_pre.json, which records the release's value.
_PRE_V4_NAMES: dict[str, frozenset[str]] = {
    "fugal_subnet.config": frozenset("""
        API_MAX_RETRIES API_RETRYABLE_STATUS API_TIMEOUT ARCHIVE_ENDPOINTS BACKBONE_BATCH_SIZE
        BACKBONE_MODEL BENCHMARK_CATEGORY_V2 BENCHMARK_CATEGORY_V3 BENCHMARK_POOL_MIN_SIZE
        BENCHMARK_WEIGHT_V2 BENCHMARK_WEIGHT_V3 BURN_IN_QUESTIONS CATEGORY_TARGETS_V2
        CATEGORY_TARGETS_V3 CHAMPION_ACCURACY_VALUE CHAMPION_ANCHOR_MODEL CHAMPION_TIME_GATE_FACTOR
        CHAMPION_WINDOW_EPOCHS COMMIT_AFTER_BOUNDARY_BLOCKS COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS
        COMMIT_RETRY_BLOCKS CONFIRM_ACTIVATION_BLOCK_ENV CONSENSUS_ENV_VARS
        EMBEDDINGS_SHA256 EMBEDDINGS_URL EPOCH_COLLECT_FRACTION EPOCH_INTERVAL EVIDENCE_HALF_LIFE
        EXECUTION_CHECKERS EXPLORE_COLDKEY_CAP EXPLORE_COST_FLOOR EXPLORE_COST_MAX_HALVES
        EXPLORE_FAIL_MARGIN EXPLORE_FRACTION EXPLORE_PRIVATE_MIN_TRIALS EXPLORE_PRIVATE_SIZE
        EXPLORE_PRIVATE_Z EXPLORE_SHARED_SIZE EXPLORE_SPEND_CAP_USD FALLBACK_ENDPOINTS
        FINNEY_CONFIRM_ACTIVATION_BLOCK FINNEY_GENESIS_HASH FINNEY_MENU_B_BLOCK
        FINNEY_MENU_GRACE_EPOCHS FINNEY_ODDS_SEAT_BLOCK FINNEY_ROOT_HOTKEY
        FINNEY_V2_ACTIVATION_BLOCK FINNEY_V3_ACTIVATION_BLOCK FRAME_DEFAULT_COMPLETION_TOKENS
        FRAME_HALF_LIFE FRAME_PRIOR_ACCURACY FRAME_PRIOR_STRENGTH FRONTIER_MIN_TRIALS
        HARNESS_ALLOW_EXEC HARNESS_CALL_TIMEOUT_S HARNESS_CONCURRENCY HARNESS_MAX_TOKENS
        HARNESS_SYSTEM_PROMPT_V3 HARNESS_WAIT_MARGIN_S HEAD_ARCHIVE_DIR HEAD_HIDDEN_DIM
        HEAD_MAX_BYTES HEAD_MAX_DECOMPRESSED_BYTES HEAD_MAX_MODELS HEAD_MAX_MODEL_ID_LEN
        LATE_START_MARGIN_BLOCKS LIVENESS_MAX_MISSED_EVIDENCE MENU_B_BLOCK_ENV
        MENU_GRACE_EPOCHS MENU_GRACE_EPOCHS_ENV MIN_VALIDATOR_STAKE ODDS_REF_PRIOR
        ODDS_SEAT_BLOCK_ENV PAYOUT_SEAT_SHARES PROOF_MAX_TOKENS
        PROXY_CALL_DEADLINE_S PROXY_CALL_DEADLINE_V3_DEFAULT_S PROXY_CALL_DEADLINE_V3_S
        PUBLISH_HEADS PUBLISH_HEADS_AFTER_EPOCHS REFERENCE_BUCKET REQUIRE_COMMITMENT REVEAL_BUCKET
        REVEAL_DELAY_EPOCHS ROOT_HOTKEY_ENV ROOT_VALIDATOR ROUTER_SYSTEM_PROMPT
        ROUTING_DECISION_QUANTUM SANDBOXED_CODE_CHECKERS SCORE_QUALITY_FLOOR SINGLE_MODEL_MIN_PAIRS
        SLICE_SHARE_FACTOR SLICE_SIZE SOFT_TARGET_TAU TEE_APPROVED_MEASUREMENTS
        TEE_COLLATERAL_EPOCH_BUDGET TEE_COLLATERAL_MAX_TIMEOUTS TEE_COLLATERAL_TIMEOUT
        TEE_MODEL_PRICES_PATH TEE_PCCS_URL TEE_PROXY_PORT TRAINING_COST_LAMBDA V2_ACTIVATION_BLOCK
        V3_ACTIVATION_BLOCK_ENV VALIDATOR_IP WEIGHTS_SET_RETRIES WILSON_CONFIDENCE
        """.split()),
    "fugal_subnet.v2_rebuild": frozenset({"_SNAPSHOT_KEYS"}),
    "fugal_subnet.v2_sync": frozenset({"DRIFT_CHECKS"}),
    "fugal_subnet.routing_protocol": frozenset({
        "ROOT", "MANIFEST_PATH", "PRICES_PATH", "PRICES_PATH_V2", "MANIFEST_PATH_V2",
        "PRICES_PATH_V3", "MANIFEST_PATH_V3", "PRICES_PATH_V3B", "MANIFEST_PATH_V3B",
        "BENCHMARK_LAMBDA"}),
    "fugal_subnet.tee.verify": frozenset({"_HEX64", "_TDX_QUOTE_VERSIONS"}),
    "fugal_subnet.tee.harness": frozenset({"PROXY_RECONNECT_WAIT_S", "_CODE_SANDBOX",
                                           "_CODE_SANDBOX_LOCK"}),
    "fugal_subnet.reveal_sync": frozenset({
        "GCS", "FINNEY_REFERENCE_BUCKET", "MAX_LISTING_BYTES", "MAX_LISTING_PAGES", "REQUEST_TIMEOUT_S", "_BUCKET", "_EPOCH_ID", "_SHA256",
        "MAX_POINTERS"}),
    "fugal_subnet.tee.runtime": frozenset({
        "_OPENROUTER_BASE", "PROXY_RETRYABLE_STATUS", "PROXY_OPERATOR_STATUS",
        "OPENROUTER_402_TRANSIENT", "OPENROUTER_402_TOO_LARGE", "PROXY_MAX_ATTEMPTS",
        "PROXY_RETRY_BUDGET_S", "PROXY_MAX_WAIT_S", "PROXY_REFUSAL_MARKERS", "_CONFIGFS_TSM"}),
    "fugal_subnet.payout": frozenset(),
    "fugal_subnet.single_model": frozenset(),
    # The provisioning channel (it checks the v4 options it carries,
    # tee/v4_options). ALLOWED_FIELDS is pinned by tests/test_provision_schema.py.
    "fugal_subnet.tee.provision": frozenset({
        "NONCE_BYTES", "PROVISION_PORT", "_ATTEST_PATH", "_PUSH_PATH",
        "_STATUS_PATH", "_LOGS_PATH", "_HKDF_INFO_KEY", "_HKDF_INFO_LOG_TOKEN",
        "_AEAD_NONCE_BYTES", "_MAX_PENDING_NONCES", "_MAX_BODY_BYTES", "_HANDLER_TIMEOUT_S",
        "LOG_RING_LINES"}),
}

_PRE_V4_REASON = ("a constant of the v3 release (origin/main) with the same value: classified by "
                  "v1-v3's mechanisms (the scoring block, the protocol identity, the environment "
                  "lists); the pin file consensus_v4_pre.json holds what the release has")
PRE_V4: dict[str, dict[str, str]] = {
    module: dict.fromkeys(sorted(names), _PRE_V4_REASON) for module, names in _PRE_V4_NAMES.items()}

# A name may be NOT_V4 or PRE_V4 only with a reason, and moving a name into
# either class must be deliberate: per module, the hash of the sorted names
# classified NOT_V4 and the hash of those classified PRE_V4 (class_hashes()).
# A name added to or moved between those classes changes a hash, and
# problems() fails until the line here is updated in the same commit.
CLASS_PINS: dict[str, tuple[str, str]] = {
    "fugal_subnet.arena_score": ("1ac5026690406c9b", "4f53cda18c2baa0c"),
    "fugal_subnet.axon_allowlist": ("250f7ed2869c7fa2", "4f53cda18c2baa0c"),
    "fugal_subnet.benchmarks.code_sets": ("74febd3366d2f6df", "4f53cda18c2baa0c"),
    "fugal_subnet.benchmarks.gpqa": ("00069def9aadd4c0", "4f53cda18c2baa0c"),
    "fugal_subnet.benchmarks.livecode": ("1ac5675e8cb17526", "4f53cda18c2baa0c"),
    "fugal_subnet.benchmarks.loader": ("d5a9f08b55acc9aa", "4f53cda18c2baa0c"),
    "fugal_subnet.benchmarks.math_": ("d9a4e097379678c2", "4f53cda18c2baa0c"),
    "fugal_subnet.benchmarks.mmlu": ("35614e83f6cc1105", "4f53cda18c2baa0c"),
    "fugal_subnet.benchmarks.v2_sets": ("688a22339353c2e5", "4f53cda18c2baa0c"),
    "fugal_subnet.benchmarks.v3_sets": ("9aff5afe74f4321e", "4f53cda18c2baa0c"),
    "fugal_subnet.chain_proposals": ("5eff42c0edb6c03c", "4f53cda18c2baa0c"),
    "fugal_subnet.champion": ("1ce6bc87921a5bf0", "4f53cda18c2baa0c"),
    "fugal_subnet.code_gate": ("2745c98ee9a82e5e", "4f53cda18c2baa0c"),
    "fugal_subnet.collect": ("bac0db6b872fcae9", "4f53cda18c2baa0c"),
    "fugal_subnet.commit_reveal": ("353e9a600cb5d95c", "4f53cda18c2baa0c"),
    # v5: EXAM_BUDGET_S_V5, EXAM_CALL_DEADLINE_S_V5 and BASELINE_MIN_PAIRS_V5
    # added to NOT_V4 (v5 only)
    "fugal_subnet.config": ("df3813905eae1773", "2b9b61905b4c5c4f"),
    "fugal_subnet.copy_rule": ("5191ef53fd3d75ce", "4f53cda18c2baa0c"),
    "fugal_subnet.determinism": ("ea226adaa11ca2b1", "4f53cda18c2baa0c"),
    "fugal_subnet.epoch_logger": ("eaa876bbf2003294", "4f53cda18c2baa0c"),
    "fugal_subnet.epoch_plan": ("924655747f80455e", "4f53cda18c2baa0c"),
    "fugal_subnet.exam_v5": ("324515b25eb537cb", "4f53cda18c2baa0c"),
    "fugal_subnet.head_archive": ("1ee99a516c64ffb9", "4f53cda18c2baa0c"),
    "fugal_subnet.head_odds": ("ae8ff68a53819d51", "4f53cda18c2baa0c"),
    "fugal_subnet.head_publisher": ("9a42d3792535d56c", "4f53cda18c2baa0c"),
    "fugal_subnet.join_step": ("2c5665a618be04e3", "4f53cda18c2baa0c"),
    "fugal_subnet.labels_v4": ("732ff1197514ee96", "4f53cda18c2baa0c"),
    "fugal_subnet.logging_setup": ("5589929081090786", "4f53cda18c2baa0c"),
    "fugal_subnet.memo_sync": ("cc229d7269dec539", "4f53cda18c2baa0c"),
    "fugal_subnet.memo_v4": ("619b58eb22d883b1", "4f53cda18c2baa0c"),
    "fugal_subnet.menu_publish": ("32ae87c52a8e7157", "4f53cda18c2baa0c"),
    "fugal_subnet.menu_rules": ("8ea7d2c6618ce6d0", "4f53cda18c2baa0c"),
    "fugal_subnet.menu_state": ("619b58eb22d883b1", "4f53cda18c2baa0c"),
    "fugal_subnet.odds_score": ("0152d266727f25dc", "4f53cda18c2baa0c"),
    "fugal_subnet.payout": ("4f53cda18c2baa0c", "4f53cda18c2baa0c"),
    "fugal_subnet.reference_frame": ("9676975d97467997", "4f53cda18c2baa0c"),
    "fugal_subnet.reveal_publisher": ("5839d7d7dcd91d30", "4f53cda18c2baa0c"),
    "fugal_subnet.reveal_sync": ("54b76052a244ca58", "9227ff058e52688c"),
    "fugal_subnet.routing_protocol": ("4f53cda18c2baa0c", "cad4775dc98bc524"),
    "fugal_subnet.seat_bound": ("94cda8bcda4e0604", "4f53cda18c2baa0c"),
    "fugal_subnet.seat_v4": ("0b5e67941110d21f", "4f53cda18c2baa0c"),
    "fugal_subnet.single_model": ("4f53cda18c2baa0c", "4f53cda18c2baa0c"),
    "fugal_subnet.tee.attestation": ("1dadc7b329998b57", "4f53cda18c2baa0c"),
    "fugal_subnet.tee.dstack_client": ("a08d24a047e69d88", "4f53cda18c2baa0c"),
    "fugal_subnet.tee.harness": ("51a04a98ca3b4992", "778863ca476ad168"),
    # 2026-11-24: V5_RESULT_KEYS and V5_PROOF_KEYS added NOT_V4 (protocol v5 only).
    "fugal_subnet.tee.proof": ("1883d2ea8683f259", "4f53cda18c2baa0c"),
    "fugal_subnet.tee.provision": ("ae01f2857fd671ae", "6bd87895b607fada"),
    "fugal_subnet.tee.runtime": ("da960f72963c78a0", "863f9b513d6a0d0e"),
    "fugal_subnet.tee.td_v4": ("f31b734466ec3059", "4f53cda18c2baa0c"),
    # 2026-11-24: the v5 TD module; its constants are protocol v5 only (NOT_V4).
    "fugal_subnet.tee.td_v5": ("7c1aa8462bbc9796", "4f53cda18c2baa0c"),
    "fugal_subnet.tee.tpm": ("9bcdc262923867b7", "4f53cda18c2baa0c"),
    "fugal_subnet.tee.v4_options": ("1ee788d99f45a154", "4f53cda18c2baa0c"),
    "fugal_subnet.tee.verify": ("4f53cda18c2baa0c", "c6becece9cf88d68"),
    "fugal_subnet.v2_epoch": ("311f331a70b8f46f", "4f53cda18c2baa0c"),
    "fugal_subnet.v2_rebuild": ("4f53cda18c2baa0c", "ff22ecff3f39a026"),
    "fugal_subnet.v2_sync": ("4f53cda18c2baa0c", "4acf1e1289ad1865"),
    "fugal_subnet.v4_replay": ("4acd41d1fd7315e3", "4f53cda18c2baa0c"),
    "fugal_subnet.validator_setup": ("9af859d9fe09d19f", "4f53cda18c2baa0c"),
    "neurons.miner": ("cd54bd02c0ef3108", "4f53cda18c2baa0c"),
    "neurons.validator": ("fc0d1bd92ad244c9", "4f53cda18c2baa0c"),
}

# Every module whose constants are classified above: the v4 modules, and the
# pre-v4 modules that read or are read by v4 code (where the seat and the
# memorisation packages add their constants).
SCANNED: tuple[str, ...] = tuple(sorted(set(REGISTERED) | set(NOT_V4) | set(PRE_V4)))

# Modules that import v4 code but hold no constant of a v4 epoch: tooling
# and this registry's own reader.
NOT_SCANNED: dict[str, str] = {
    "fugal_subnet.fingerprint": "reads this registry",
    "fugal_subnet.consensus_v4": "this registry",
    "fugal_subnet.embeddings_file": "a hash-pinned file reader (the v3 embeddings)",
    "fugal_subnet.graders": "hash-pinned whole (GRADER_SHA256, checked in CI): every byte of it is "
                            "in the digest's grader_sha256, constants included",
    "fugal_subnet.graders_code": "hash-pinned whole (the code grader's sha256 is part of the "
                                 "digest's grader_sha256, `+code:`): every byte of it is covered, "
                                 "constants included",
    "fugal_subnet.attacks.run_miner_attacks": "an attack suite (tooling)",
    "fugal_subnet.attacks.run_tee_attacks": "an attack suite (tooling)",
    "fugal_subnet.attacks.seat_v4_world": "an attack suite's world (tooling): its constants are "
                                          "fixtures no validator reads",
    "fugal_subnet.attacks.memo_seat_world": "an attack suite's world (tooling): its constants are "
                                            "fixtures no validator reads",
    "fugal_subnet.attacks.seat_v5_world": "an attack suite's world (tooling): its constants are "
                                          "fixtures no validator reads",
    "fugal_subnet.seat_v5": "protocol v5 only: decides nothing a v4 epoch publishes; its constants "
                            "are in the v5 section (consensus_v5.REGISTERED)",
    "fugal_subnet.consensus_v5": "the v5 registry",
    "fugal_subnet.v5_epoch": "protocol v5 only: a v5 epoch's votes, window, B*, seat and next "
                             "exam; its constants are in the v5 section (consensus_v5.REGISTERED)",
}

# The pinned files a v4 epoch reads, by role: (path under fugal_subnet/,
# required). An optional file is recorded as None when the release does not
# ship it. A package that adds a pinned v4 file (the published-set seed of
# the memorisation check) adds it here; tests/test_consensus_v4.py fails on
# a v4 data file under routing_data/ or signed_lists/ that is not listed.
PINNED_FILES: dict[str, tuple[str, bool]] = {
    "menu_v4_seed": ("routing_data/menu_v4_seed.json", True),
    "pool_mix_v4": ("routing_data/pool_mix_v4.json", True),
    "list_signers": ("signed_lists/allowed_signers", True),
    "menu_allowlist": ("signed_lists/menu_allowlist.json", False),
    "device_list": ("signed_lists/device_list.json", False),
    # The memorisation check's published-set seed (SEAT_MINIMAL_V4.md 6.3),
    # also pinned by check_safety_invariants (PINNED_PUBLISHED_V4_SHA256).
    "published_v4": ("routing_data/published_v4.json", True),
}


# The data files of the v1-v3 releases (the v3 release's routing_data/): each
# is hashed into the v1-v3 digest through the routing identities
# (routing_protocol.identity, identity_v2, identity_v3), which a v4 digest
# also covers. Every other file under routing_data/ or signed_lists/ is in
# PINNED_FILES or fails problems().
_V1_V3_DATA_REASON = ("a v1-v3 routing file, hashed into the v1-v3 digest through "
                      "routing_protocol's identities")
V1_V3_DATA: dict[str, str] = dict.fromkeys((
    "routing_data/benchmark_tokens_protocol_v2.json", "routing_data/benchmark_tokens_protocol_v3.json",
    "routing_data/benchmark_tokens_protocol_v3b.json", "routing_data/benchmark_tokens_v1.json",
    "routing_data/models.json", "routing_data/models_v2.json", "routing_data/models_v3.json",
    "routing_data/models_v3b.json"), _V1_V3_DATA_REASON)


def canonical(value):
    """A constant's value as JSON: exact (a Fraction as "n/d", bytes as hex,
    a pattern as its source), with unordered collections sorted. Raises
    TypeError for a value that has no such form (code, an object)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": int(value.flags)}
    if isinstance(value, (frozenset, set)):
        return sorted((canonical(v) for v in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    raise TypeError(f"{type(value).__name__} has no canonical form")


def constants() -> dict:
    """{"<module>:<NAME>": canonical value} of every registered constant. A
    value that cannot be read is recorded as such (a fingerprint never
    fails); the registry test holds that none is."""
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
                value = mod
                for part in name.split("."):      # `<Class>.<NAME>` for a class constant
                    value = getattr(value, part)
                out[f"{module}:{name}"] = canonical(value)
            except (AttributeError, TypeError) as e:
                out[f"{module}:{name}"] = f"unavailable: {type(e).__name__}"
    return out


def file_hashes(package: Path | None = None) -> dict:
    """{role: sha256 of the pinned file, or None when an optional one is not shipped}."""
    root = PACKAGE if package is None else Path(package)
    out = {}
    for role, (rel, _required) in sorted(PINNED_FILES.items()):
        try:
            out[role] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError:
            out[role] = None
    return out


def activation() -> dict:
    """The v4 block as the section records it: finney's compiled value, and
    the variable another chain reads (left out when it repeats finney's)."""
    from fugal_subnet import config

    finney = getattr(config, "FINNEY_V4_ACTIVATION_BLOCK", None)
    env = getattr(config, "V4_ACTIVATION_BLOCK_ENV", None)
    return {"finney": finney, "env": None if env == finney else env}


def section() -> dict:
    """The fingerprint's `v4` section (fingerprint.environment_fingerprint)."""
    return {"activation": activation(), "constants": constants(), "files": file_hashes()}


# --- the registry's completeness (tests/test_consensus_v4.py, check_safety_invariants) ---
#
# What the scan sees. A name is a constant when it is UPPER_CASE (one
# leading underscore allowed). Every binding of such a name is read off the
# source, wherever it stands at module level: a plain, tuple, starred,
# annotated or augmented assignment, a `for`, `with`, `except` or `match`
# target, a walrus, an import, a `def` or `class`, a `global` declaration in
# a function, all of them also inside `if`, `try`, `with` and loops; and the
# UPPER_CASE names a class defined in the module sets in its body, as
# `<Class>.<NAME>`. A module that writes names dynamically (`globals()[..]`,
# `exec`, `sys.modules[__name__]`) is refused: the scan cannot see what it
# binds.
#
# Which modules it reads. The transitive import closure of the v4 entry
# points: every module of `neurons/`, every module whose name says v4 and
# every module that imports one of the v4 core modules (_V4_CORE). An import
# is followed wherever it stands in the file (function bodies and
# conditional imports included), as `import a.b`, `from a import b` (b a
# module or a name), a relative import, and a string literal that names a
# module (`importlib.import_module("fugal_subnet.x")`, a table of module
# names). A module that imports by a name it computes is refused unless
# listed in DYNAMIC_IMPORTS. A module is not scanned only by an entry in
# NOT_SCANNED, which states why; the imports of such a module are not
# followed.

_CONSTANT = re.compile(r"_?[A-Z][A-Z0-9_]*")
_TOP_PACKAGES = ("fugal_subnet", "neurons")
# The modules that hold v4's rules: a module that imports one of them is v4
# code, or reads it, so it is an entry point of the closure.
_V4_CORE = frozenset((
    "menu_state", "menu_rules", "menu_inputs", "menu_publish", "seat_bound", "picks_v4",
    "labels_v4", "td_v4", "endpoint_reads", "head_odds", "epoch_plan", "explore_weights",
    "seat_v4", "memo_v4", "consensus_v4", "v4_options"))
_MODULE_NAME = re.compile(r"(?:fugal_subnet|neurons)(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_MUTATORS = frozenset(("update", "setdefault", "pop", "popitem", "clear", "__setitem__"))

# Modules that import by a name they compute (importlib.import_module(x)),
# each with how the scan still reaches what they load: the modules named by
# string literals in the file are followed.
DYNAMIC_IMPORTS: dict[str, str] = {
    "fugal_subnet.benchmarks.loader": "imports the benchmark modules named in its _BENCHMARKS "
                                      "table; each name is a string literal in the file, so the "
                                      "closure follows it",
}

# The names the v3 release (origin/main) had, by module, with a hash of the
# statements that bind each: the file beside this one that PRE_V4 is checked
# against (and, where the release is available, tests/test_consensus_v4.py
# checks the file against it).
PRE_V4_PIN = "consensus_v4_pre.json"


def _names(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _names(elt)
    elif isinstance(target, ast.Starred):
        yield from _names(target.value)


class _Binder(ast.NodeVisitor):
    """The names a block of module-level statements binds, with the
    statements that bind them. Does not enter a function or a lambda (except
    to read its `global` declarations) or a comprehension's own target."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.bound: dict[str, list[str]] = {}
        self.stmt: ast.AST | None = None
        self.augmented: set[str] = set()

    def bind(self, name: str, stmt) -> None:
        self.bound.setdefault(self.prefix + name, []).append(
            ast.dump(stmt) if stmt is not None else "")

    def visit(self, node):
        if isinstance(node, ast.stmt):
            outer, self.stmt = self.stmt, node
            try:
                return super().visit(node)
            finally:
                self.stmt = outer
        return super().visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bind(node.id, self.stmt)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.augmented.add(self.prefix + node.target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        # `NAME: type` with no value declares a field and binds nothing.
        if node.value is not None:
            self.visit(node.target)
            self.visit(node.value)
        self.visit(node.annotation)

    def visit_FunctionDef(self, node):
        self.bind(node.name, node)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Global):
                for name in sub.names:
                    # a name a function declares global is a module name
                    self.bound.setdefault(self.prefix + name, []).append(ast.dump(sub))
        for decorator in node.decorator_list:
            self.visit(decorator)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        return None

    def visit_ClassDef(self, node):
        self.bind(node.name, node)
        inner = _Binder(prefix=f"{self.prefix}{node.name}.")
        for stmt in node.body:
            inner.visit(stmt)
        for name, dumps in inner.bound.items():
            self.bound.setdefault(name, []).extend(dumps)
        self.augmented |= inner.augmented

    def visit_comprehension(self, node):
        self.visit(node.iter)
        for cond in node.ifs:
            self.visit(cond)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bind(node.name, self.stmt)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.bind((alias.asname or alias.name).split(".")[0], node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.bind(alias.asname or alias.name, node)

    def visit_MatchAs(self, node):
        if node.name:
            self.bind(node.name, self.stmt)
        self.generic_visit(node)

    def visit_MatchStar(self, node):
        if node.name:
            self.bind(node.name, self.stmt)

    def visit_MatchMapping(self, node):
        if node.rest:
            self.bind(node.rest, self.stmt)
        self.generic_visit(node)


def module_bindings(path: Path) -> dict[str, list[str]]:
    """{name: [dump of each statement that binds it]} for the UPPER_CASE
    names a module binds at module level, and `<Class>.<NAME>` for those a
    class defined in it binds in its body."""
    return source_bindings(Path(path).read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1024)
def _parse(source: str):
    return ast.parse(source)


@functools.lru_cache(maxsize=1024)
def _bind(source: str) -> _Binder:
    binder = _Binder()
    for stmt in _parse(source).body:
        binder.visit(stmt)
    return binder


def augmented_constants(source: str) -> list[str]:
    """The constants a module rebinds by an augmented assignment (`X += 1`)."""
    return sorted(n for n in _bind(source).augmented if _CONSTANT.fullmatch(n.rsplit(".", 1)[-1]))


def source_bindings(source: str) -> dict[str, list[str]]:
    """module_bindings of a module's source text."""
    binder = _bind(source)
    # A name the module only imports (`from x import NAME`) is the value of
    # the module that defines it, which is scanned and classifies it; an
    # import that also rebinds a name elsewhere in the module is not "only".
    return {name: dumps for name, dumps in binder.bound.items()
            if _CONSTANT.fullmatch(name.rsplit(".", 1)[-1])
            and not all(d.startswith(("Import(", "ImportFrom(")) for d in dumps)}


def module_constants(path: Path) -> list[str]:
    """The names `module_bindings` finds, sorted."""
    return sorted(module_bindings(path))


def _is_call_to(node, names) -> bool:
    return (isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id in names)
                 or (isinstance(node.func, ast.Attribute) and node.func.attr in names)))


def _is_module_object(node) -> bool:
    """`sys.modules[...]`, or `sys.modules.get(...)`: a module held by name."""
    if isinstance(node, ast.Subscript):
        node = node.value
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        node = node.func.value
    return isinstance(node, ast.Attribute) and node.attr == "modules"


def dynamic_bindings(tree) -> list[str]:
    """Forms by which a module binds names the scan cannot read off its
    source: a write through globals() or vars(), exec/eval, a setattr or an
    attribute write on a module taken from sys.modules."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("exec", "eval"):
            out.append(f"line {node.lineno}: {node.func.id}()")
        elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del))
              and _is_call_to(node.value, ("globals", "vars"))):
            out.append(f"line {node.lineno}: a write through globals()")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in _MUTATORS
              and _is_call_to(node.func.value, ("globals", "vars"))):
            out.append(f"line {node.lineno}: globals().{node.func.attr}()")
        elif _is_call_to(node, ("setattr", "delattr")) and node.args \
                and (_is_module_object(node.args[0])
                     or (isinstance(node.args[0], ast.Name) and node.args[0].id == "__name__")):
            out.append(f"line {node.lineno}: setattr on a module")
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) \
                and _is_module_object(node.value):
            out.append(f"line {node.lineno}: an attribute written on a module from sys.modules")
    return out


def _module_file(root: Path, module: str) -> Path | None:
    base = root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def all_modules(root: Path) -> list[str]:
    """Every module of the packages the scan covers, as dotted names (a
    package's __init__.py is the package's name)."""
    out = []
    for top in _TOP_PACKAGES:
        for path in sorted((root / top).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            parts = list(path.relative_to(root).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            out.append(".".join(parts))
    return out


def _with_parents(root: Path, module: str):
    """The module and the packages Python imports to reach it, those that exist."""
    parts = module.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if _module_file(root, name) is not None:
            yield name


def module_imports(root: Path, module: str, tree) -> tuple[set[str], bool]:
    """(the modules of the scanned packages `module` imports, whether it
    imports by a name it computes). See the comment above for what counts."""
    path = _module_file(root, module)
    package = module if path is not None and path.name == "__init__.py" \
        else module.rpartition(".")[0]
    found: set[str] = set()

    def add(name: str) -> None:
        if name.split(".")[0] in _TOP_PACKAGES:
            found.update(_with_parents(root, name))

    computed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parts = parts[:len(parts) - (node.level - 1)]
                base = ".".join(parts + ([base] if base else []))
            add(base)
            for alias in node.names:
                add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and _MODULE_NAME.fullmatch(node.value):
            add(node.value)
        elif _is_call_to(node, ("import_module", "__import__")):
            if not (node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                computed = True
    return found, computed


def import_graph(root: Path) -> tuple[dict[str, set[str]], set[str]]:
    """({module: the modules it imports}, the modules that import by a name
    they compute) over every module of the scanned packages."""
    graph: dict[str, set[str]] = {}
    computed: set[str] = set()
    for module in all_modules(root):
        tree = _parse(_module_file(root, module).read_text(encoding="utf-8"))
        graph[module], was_computed = module_imports(root, module, tree)
        if was_computed:
            computed.add(module)
    return graph, computed


def scanned_modules(root: Path | None = None, graph=None) -> set[str]:
    """The modules whose constants must be classified: the transitive import
    closure of the entry points (`neurons/`, the v4-named modules, the
    importers of the v4 core), and the modules with a table here, less
    NOT_SCANNED."""
    root = PACKAGE.parent if root is None else Path(root)
    graph = import_graph(root)[0] if graph is None else graph
    entries = {m for m, imported in graph.items()
               if m.split(".")[0] == "neurons" or "v4" in m.rsplit(".", 1)[-1]
               or any(i.rsplit(".", 1)[-1] in _V4_CORE for i in imported)}
    reached: set[str] = set()
    todo = sorted(entries)
    while todo:
        module = todo.pop()
        if module in reached or module in NOT_SCANNED:
            continue
        reached.add(module)
        todo.extend(graph.get(module, ()))
    return (reached | set(SCANNED)) - set(NOT_SCANNED)


def pre_v4_hashes(root: Path, module: str) -> dict[str, str]:
    """{name: hash of the statements binding it} for a module's constants,
    as the PRE_V4 pin records them."""
    path = _module_file(root, module)
    return {} if path is None else binding_hashes(source_bindings(path.read_text(encoding="utf-8")))


def binding_hashes(bindings: dict[str, list[str]]) -> dict[str, str]:
    return {name: hashlib.sha256(json.dumps(dumps).encode()).hexdigest()[:16]
            for name, dumps in sorted(bindings.items())}


def pre_v4_pin(root: Path | None = None) -> dict:
    """The pin file's {module: {name: hash}}, or {} when it is not there."""
    root = PACKAGE.parent if root is None else Path(root)
    try:
        return json.loads((root / "fugal_subnet" / PRE_V4_PIN).read_text(encoding="utf-8"))["modules"]
    except (OSError, ValueError, KeyError):
        return {}


def _class_hash(names) -> str:
    return hashlib.sha256(json.dumps(sorted(names)).encode()).hexdigest()[:16]


def class_hashes() -> dict[str, tuple[str, str]]:
    """{module: (hash of its NOT_V4 names, hash of its PRE_V4 names)}, to
    compare with CLASS_PINS."""
    return {m: (_class_hash(NOT_V4.get(m, {})), _class_hash(PRE_V4.get(m, {})))
            for m in sorted(set(NOT_V4) | set(PRE_V4))}


def problems(root: Path | None = None) -> list[str]:
    """Why the registry is not complete, or []: a constant of a scanned
    module with no classification, a classified name no module defines, a name
    classified twice or with no reason, a name moved between classes
    (CLASS_PINS), a PRE_V4 name the v3 release does not have as it is now
    (the pin file), a module that writes names the scan cannot read or
    imports by a computed name, a v4 data file not in PINNED_FILES, a required
    pinned file missing. `root` is the checkout (the directory holding
    fugal_subnet/), for the tests."""
    root = PACKAGE.parent if root is None else Path(root)
    out: list[str] = []
    graph, computed = import_graph(root)
    scanned = scanned_modules(root, graph)
    pin = pre_v4_pin(root)
    for module in sorted(scanned):
        path = _module_file(root, module)
        if path is None:
            out.append(f"{module}: scanned but missing")
            continue
        bindings = module_bindings(path)
        defined = set(bindings)
        registered = set(REGISTERED.get(module, ()))
        not_v4 = set(NOT_V4.get(module, {}))
        pre = set(PRE_V4.get(module, {}))
        for a, b, what in ((registered, not_v4, "REGISTERED and NOT_V4"),
                           (registered, pre, "REGISTERED and PRE_V4"),
                           (not_v4, pre, "NOT_V4 and PRE_V4")):
            for name in sorted(a & b):
                out.append(f"{module}:{name} is in both {what}")
        for name in sorted(defined - registered - not_v4 - pre):
            out.append(f"{module}:{name} has no protocol v4 classification: register it in "
                       "consensus_v4.REGISTERED (it decides a v4 score, the menu, exploration, a "
                       "proof or its verification) or name it in NOT_V4 with the reason it does not")
        for name in sorted((registered | not_v4 | pre) - defined):
            out.append(f"{module}:{name} is classified but not defined (renamed or deleted?)")
        have = pre_v4_hashes(root, module) if pre else {}
        for name in sorted(pre & defined):
            if name not in pin.get(module, {}):
                out.append(f"{module}:{name} is in PRE_V4 but not in {PRE_V4_PIN}: the v3 release "
                           "does not have it, so it must be registered or put in NOT_V4")
            elif pin[module][name] != have[name]:
                out.append(f"{module}:{name} is in PRE_V4 but is not what the v3 release has "
                           f"({PRE_V4_PIN}): register it or put it in NOT_V4")
        for name in augmented_constants(path.read_text(encoding="utf-8")):
            out.append(f"{module}:{name} is rebound by an augmented assignment: a constant is "
                       "bound by one statement, so its classification covers its value")
        for what in dynamic_bindings(_parse(path.read_text(encoding="utf-8"))):
            out.append(f"{module}: binds names the registry scan cannot read ({what}); write "
                       "them as plain assignments")
    for module in sorted(computed - set(DYNAMIC_IMPORTS) - set(NOT_SCANNED)):
        out.append(f"{module} imports by a name it computes, which the scan cannot follow: "
                   "import by a literal name, or list it in consensus_v4.DYNAMIC_IMPORTS")
    for table, label in ((NOT_V4, "NOT_V4"), (PRE_V4, "PRE_V4")):
        for module, reasons in sorted(table.items()):
            for name, reason in sorted(reasons.items()):
                if not isinstance(reason, str) or not reason.strip():
                    out.append(f"{module}:{name} is in {label} without a reason")
    for table, label in ((NOT_SCANNED, "NOT_SCANNED"), (DYNAMIC_IMPORTS, "DYNAMIC_IMPORTS")):
        for module, reason in sorted(table.items()):
            if not isinstance(reason, str) or not reason.strip():
                out.append(f"{module} is in {label} without a reason")
    for module in sorted(set(NOT_SCANNED) & set(SCANNED)):
        out.append(f"{module} is in NOT_SCANNED and also has a classification")
    hashes = class_hashes()
    for module, expected in sorted(CLASS_PINS.items()):
        if hashes.get(module) != expected:
            out.append(f"{module}: the names classified NOT_V4 or PRE_V4 changed (CLASS_PINS has "
                       f"{expected}, they hash to {hashes.get(module)}); a name moved between "
                       "classes needs that pin updated in the same commit, with the reason")
    for module in sorted(set(hashes) - set(CLASS_PINS)):
        out.append(f"{module} has NOT_V4 or PRE_V4 names and no entry in CLASS_PINS")
    for module in sorted(set(pin) - set(PRE_V4)):
        out.append(f"{PRE_V4_PIN} names {module}, which PRE_V4 does not")
    for module in sorted(PRE_V4):
        for name in sorted(set(pin.get(module, {})) - set(PRE_V4[module])):
            out.append(f"{PRE_V4_PIN} has {module}:{name} and PRE_V4 does not")
    return out + _data_file_problems(root)


def _data_file_problems(root: Path) -> list[str]:
    out = []
    package = root / "fugal_subnet"
    listed = {rel for rel, _required in PINNED_FILES.values()}
    listed |= set(V1_V3_DATA)
    found = {f"routing_data/{p.name}" for p in (package / "routing_data").glob("*") if p.is_file()}
    found |= {f"signed_lists/{p.relative_to(package / 'signed_lists').as_posix()}"
              for p in (package / "signed_lists").rglob("*")
              if p.is_file() and not p.name.endswith(".sig")}
    for rel in sorted(found - listed):
        out.append(f"fugal_subnet/{rel} is a data file in neither consensus_v4.PINNED_FILES (so "
                   "in the v4 digest) nor V1_V3_DATA (the v1-v3 files, hashed in the v1-v3 digest)")
    for rel in sorted(set(V1_V3_DATA) - found):
        out.append(f"fugal_subnet/{rel} is listed in V1_V3_DATA and is not there")
    for rel, reason in sorted(V1_V3_DATA.items()):
        if not isinstance(reason, str) or not reason.strip():
            out.append(f"{rel} is in V1_V3_DATA without a reason")
    for role, (rel, required) in sorted(PINNED_FILES.items()):
        if required and not (package / rel).is_file():
            out.append(f"fugal_subnet/{rel} ({role}) is required and missing")
    return out
