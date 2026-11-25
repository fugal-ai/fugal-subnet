"""Env-overridable constants for the Fugal subnet."""

import logging
import os

_log = logging.getLogger(__name__)

# Network and netuid are CLI concerns: neurons/validator.py and neurons/miner.py
# read FUGAL_NETWORK and FUGAL_NETUID as click defaults. Nothing in the package
# needs them, so they are not constants here.

# --- Epoch ---
EPOCH_INTERVAL = int(os.getenv("FUGAL_EPOCH_INTERVAL", "3600"))
SLICE_SIZE = int(os.getenv("FUGAL_SLICE_SIZE", "300"))
# A benchmark's share of the slice may not exceed this factor times its share
# of the pool. Equal per-benchmark quotas gave AIME, GSM8K and IFEval -- 2,511
# questions, about what a linear head can memorise outright (measured,
# docs/decisions/HEAD_EFFICACY.md) -- 180 of 300 slots, so 60% of every score was a
# lookup table away (diligence 2026-10, finding 6). Pure proportional draw
# (factor 1) hands MMLU 65% of the slice and lets it dominate the routing
# signal, which is what stratification existed to prevent. At 1.5 the three
# small benchmarks hold 17% of the slice, MMLU 48%, and a 2,500-question
# memoriser captures about 21% of a score instead of 64%; the table for other
# factors is in docs/decisions/HEAD_EFFICACY.md, experiment 4. Consensus-critical (two
# neurons with different values select different questions), so it is in the
# fingerprint's epoch block.
SLICE_SHARE_FACTOR = float(os.getenv("FUGAL_SLICE_SHARE_FACTOR", "1.5"))
# How far into the epoch validators collect proofs, as a fraction of it.
#
# A miner cannot begin benchmarking until the boundary block exists — the block
# hash is what seeds the question slice, and hiding the slice until that moment
# is the whole anti-overfitting design. So there is an unavoidable gap between
# an epoch starting and any proof existing for it, and a validator that queries
# at the boundary asks before anyone can possibly answer.
#
# Retrying until something appears would fix the symptom and break I1: how long
# a validator happened to wait would decide which miners it scored, so two
# honest validators would grade different fields and publish different weights.
# The collection point is therefore derived from the block number, identically
# for everyone, and lives in the consensus digest — a validator collecting at a
# different offset diverges visibly instead of silently.
#
# Half the epoch splits it evenly: miners get half to benchmark, validators get
# half to verify, score, set weights and publish the reveal. At the default
# 3600s epoch that is 150 blocks (~30 min) for a 300-question slice — 6s per
# question, ample for serial API calls.
EPOCH_COLLECT_FRACTION = float(os.getenv("FUGAL_COLLECT_FRACTION", "0.5"))

# --- Grading policy ---
# Whether the benchmark harness may EXECUTE model-produced code to grade it.
#
# Off, deliberately. graders.py's sandbox is process-level — rlimits, kill-tree,
# DEVNULL, a size cap — with no filesystem or namespace isolation. Under --live
# the miner controls where the metering proxy points, so it controls what
# "model output" the harness receives; executing that inside the TD that
# produces the attested proof would let a miner run chosen code in its own
# enclave and walk away with a genuine Intel signature over forged results.
# That is a total break of I8, not a contained risk.
#
# The cost of having it off was invisible and large. exec_io and exec_unittest
# both return 0 unless permitted, and they are the checkers for humaneval and
# livecode — so those questions could never score above zero, while miners paid
# real API money attempting them. Because the slicer equalises benchmarks, that
# was not humaneval's 0.8% share of the pool but a FULL SIXTH of every graded
# slice, and routing those questions to a capable model was punished on thrift
# with no possible quality gain. The scoring was teaching heads to route code to
# the cheapest model available.
#
# So the pool now excludes what the harness cannot score (see loader.load_all),
# derived from this flag rather than hardcoded: turning execution on re-includes
# the benchmarks in one place. Doing that safely is docs/decisions/CODE_BENCHMARK_PLAN.md.
HARNESS_ALLOW_EXEC = os.getenv("FUGAL_HARNESS_ALLOW_EXEC", "0") not in ("0", "", "false", "False")

# How many model calls the miner's benchmark runs at once. Miner-side only:
# nothing in consensus depends on how fast a miner answers, only on whether
# its proof exists at the collection block.
#
# Measured 2026-09-07 on the first live epoch on real TDX: 315 serial calls to
# the three cheapest models with 2048-token completions ran past 49 minutes —
# the collection point is 36 minutes after the boundary. The config note above
# assumed "6 s per question, ample for serial API calls"; real completions on
# math prompts run ~10 s, and one 180 s timeout eats 5% of the window on its
# own. Eight in flight brings a 315-question slice to a few minutes and makes
# a stalled call cost one worker rather than the epoch.
#
# Raised to 24 for protocol v2 (2026-11-05). The 16,384-token answer budget
# lets reasoning models run long: in the v2 token run, glm-5.3 and mimo-v2.6-pro
# had p90 completions at the cap, and several other cheap reasoners at 11k-15k.
# A latency model calibrated on that run (757 calls, wall time reproduced)
# puts the p95 epoch of a router leaning on cheap reasoners at 33 min with 8 in
# flight (44 min if providers are 30% slower), a missed proof every epoch; with
# 24 it is 14 min (18 min). The floor is one call at PROXY_CALL_DEADLINE_S.
# More in flight also holds more of an OpenRouter in-flight budget; see
# runtime.OPENROUTER_402_TRANSIENT and docs/MINER_GUIDE.md.
# Raised to 40 for protocol v3 (2026-11-16). v3 answer budgets are each entry's
# real limit (up to 128k) and high-effort calls may run 30 min, so the worst
# exploration epoch (all 32 drawn calls at the 1,800 s deadline plus a 300-call
# slice at 60 s) is 75,600 call-seconds: 3,150 s at 24 in flight, past the
# 2,160 s collection point; 1,890 s at 40. Calls wait on the network, not the
# CPU, so 40 threads cost the TD little. Each running call holds input +
# max_tokens at its entry's prices against OpenRouter's in-flight budget (low
# balance and newer accounts only): ~$1.07 per call averaged over the v3 menu,
# ~$43 for 40 spread evenly, up to ~$205 if all 40 went to Claude Fable 5.2.
# docs/MINER_GUIDE.md tells miners what balance to keep.
HARNESS_CONCURRENCY = max(1, int(os.getenv("FUGAL_HARNESS_CONCURRENCY", "40")))

# --- Protocol v2 benchmark pool (approved scope, 2026-11-04) ---
# Consensus constants from V2_ACTIVATION_BLOCK, written into the v2 pool
# manifest and checked against it (scripts/check_safety_invariants.py).
# Targets apply to the scored slice and to exploration alike
# (slicer.select_slice_v2 / select_explore_set_v2).
# Decided 2026-11-04: instruction is 5% in v2.0 (only 211 IFEval items are checkable
# by the pinned constraint kinds and gradeable by the pinned grader, after the 48
# removed 2026-11-09; at 15 a slice they repeat every ~14 epochs).
# The generated instruction set that lets it grow again is v2.1.
CATEGORY_TARGETS_V2 = {"knowledge": 0.40, "math": 0.25, "hard": 0.30, "instruction": 0.05}
BENCHMARK_CATEGORY_V2 = {
    "mmlupro": "knowledge", "mmlu": "knowledge",
    "math_l35": "math", "omnimath_mid": "math",
    "supergpqa_hard": "hard", "aime": "hard", "aime25": "hard", "beyondaime": "hard", "omnimath_hard": "hard",
    "ifeval": "instruction",
}
# Within-category weights: none. Every category, hard included, splits by
# pool share. The hard category used to be weighted 50% open multiple-choice
# (supergpqa_hard, 7,050 questions) and 50% integer olympiad math (1,907):
# 45 olympiad slots an epoch, the whole set every 42 epochs, and with the hard
# weight of 2 the small sets (olympiad and IFEval, 2,118 questions) carried
# 27% of the seat test's net value. A head can memorise about 2,000-2,500
# questions (INVARIANTS, pool size) and every cell's grade is published, so
# that share was a lookup table worth more than any routing edge (Fable
# review 2026-11-09, item 4). By pool share: about 19 olympiad slots, the set
# every 99 epochs (longer than the 48-epoch window), 14% of net value.
# A consensus constant (in the pool manifest, whose hash is in the digest).
BENCHMARK_WEIGHT_V2: dict[str, float] = {}

# --- Protocol v3 benchmark pool and exploration ---
# Decided 2026-11-16 (owner). Scoring rewards honest odds, which only questions
# where models DISAGREE can teach, so the saturated knowledge share falls and
# chat-like instruction following, everyday reasoning and code (most of real
# router traffic) come in. Easy maths stays small: nearly every model gets it
# right. Pinned in pool_manifest_v3.json (whose hash is in the digest).
CATEGORY_TARGETS_V3 = {
    "knowledge": 0.20, "hard": 0.20, "math": 0.10, "instruction": 0.15,
    "everyday": 0.15, "easy_math": 0.05, "code": 0.15,
}
BENCHMARK_CATEGORY_V3 = {
    "mmlupro": "knowledge", "mmlu": "knowledge",
    "supergpqa_hard": "hard", "aime": "hard", "aime25": "hard", "beyondaime": "hard",
    "omnimath_hard": "hard",
    "math_l35": "math", "omnimath_mid": "math",
    "ifeval": "instruction", "rlvr_ifeval": "instruction",
    # BIG-Bench Hard stays held back (its only split is its published test set), so
    # everyday reasoning is ARC (train split) in v3.0.
    "arc": "everyday",
    "gsm8k_train": "easy_math", "math_l12": "easy_math",
    "code_leetcode": "code", "code_mbppplus": "code", "code_humanevalplus": "code",
    "code_contests": "code",
}
BENCHMARK_WEIGHT_V3: dict[str, float] = {}
# Exploration in v3 (epoch_plan.exploration_plan_v3). Every miner answers the
# SHARED pairs: eight, which is EXPLORE_DISAGREE_MIN_SAMPLES, the fewest the
# cross-miner agreement check judges a miner on. Each miner also answers its
# own PRIVATE pairs, drawn from the epoch nonce and its hotkey: 24, so about
# twenty operators measure about 500 distinct (question, model) pairs an
# epoch and the odds seat test has its 1,500 pairs in three or four epochs.
# 32 exploration calls against v2's 15 is about 5% more calls for a miner.
EXPLORE_SHARED_SIZE = int(os.getenv("FUGAL_EXPLORE_SHARED_SIZE", "8"))
EXPLORE_PRIVATE_SIZE = int(os.getenv("FUGAL_EXPLORE_PRIVATE_SIZE", "24"))
# The most PRIVATE pairs one operator (coldkey) contributes to an epoch's
# pool, however many hotkeys it runs (explore_pool.capped). One and a half
# hotkeys' worth: an operator with one hotkey is never cut, and one with fifty
# supplies a twelfth of a twenty-operator pool, not most of it.
EXPLORE_COLDKEY_CAP = int(os.getenv("FUGAL_EXPLORE_COLDKEY_CAP", "36"))
# Cost-aware exploration (protocol v3; fugal_subnet/explore_weights.py). Menu
# entries are drawn for exploration with a probability that falls with their
# expected cost (pinned prices x token means), never below EXPLORE_COST_FLOOR
# of a uniform share, at the most uniform exponent (0, 1/2, ... up to
# EXPLORE_COST_MAX_HALVES / 2) whose expected spend for a miner's
# EXPLORE_SHARED_SIZE + EXPLORE_PRIVATE_SIZE calls is within
# EXPLORE_SPEND_CAP_USD. Pooled pairs carry inverse-probability weights, so the
# odds score still estimates what uniform sampling would have. At the
# provisional v3 token means (2026-11-16) uniform exploration costs a miner
# $0.316 per epoch; the $0.20 cap selects exponent 1/2 ($0.163, design effect
# 1.32). The cap is EXPECTED spend: a sampling rule that stopped at a realized
# total would make an entry's chance depend on the draws before it, and the
# weights would no longer undo it. Compiled constants: they decide which pairs
# every miner is assigned and how every pair is weighted (consensus).
EXPLORE_SPEND_CAP_USD = "0.20"
EXPLORE_COST_FLOOR = "1/4"
EXPLORE_COST_MAX_HALVES = 4
# An operator whose private answers are more than this many standard
# deviations from what the reference expects of those models on those kinds of
# question has its pairs left out of the epoch's pool
# (explore_pool.private_implausible). Never a score.
EXPLORE_PRIVATE_Z = int(os.getenv("FUGAL_EXPLORE_PRIVATE_Z", "4"))
# An answer counts toward that judgement only once the window holds this many
# pooled pairs of its model. Before that the reference is its prior, one half,
# and every honest operator would be "implausibly" right. About four epochs
# for a 33-entry menu at 500 pairs an epoch.
EXPLORE_PRIVATE_MIN_TRIALS = int(os.getenv("FUGAL_EXPLORE_PRIVATE_MIN_TRIALS", "50"))
# Exploration in protocol v4 (docs/decisions/LIVE_MENU_V4.md, section 8, at
# the sizes of docs/decisions/SEAT_MINIMAL_V4.md, sections 5.3 and 8;
# explore_weights.plan_shares_v4, epoch_plan.exploration_plan_v4). Read only
# under protocol v4, which no release activates yet. There is no shared set
# (EXPLORE_SHARED_SIZE_V4 = 0, SEAT_MINIMAL_V4.md 2.3 item 33: its only reader
# was the cross-miner agreement detector, and a pair every operator voted on
# judges no head). A miner's private set is EXPLORE_PRIVATE_SIZE_V4 calls in
# two parts:
#   - the cost-aware part, EXPLORE_COST_AWARE_SIZE_V4 calls drawn by v3's
#     cost-aware shares on the menu state's measured costs, the exponent the
#     most uniform whose expected spend over the shared set (none) and this
#     part is within EXPLORE_SPEND_CAP_USD_V4 (floor and exponents as v3's);
#   - the coverage part, EXPLORE_COVERAGE_SIZE_V4 calls, drawn toward entries
#     with fewer than EXPLORE_COVERAGE_TARGET_V4 exploration pairs in the
#     window as published MENU_LAG epochs back. Only the fraction
#     min(1, sum of deficits / EXPLORE_COVERAGE_DAMP_PAIRS_V4) follows the
#     deficits, so one entry with a deficit does not take every coverage
#     call; the part is then mixed toward uniform until its expected spend is
#     within EXPLORE_COVERAGE_SPEND_CAP_USD_V4.
# Both caps are on EXPECTED spend, as v3's is. The owner's label budget is
# 112 calls a hotkey-epoch (SEAT_MINIMAL_V4.md 5.3, about $20 a day with the
# confidential VM): W0's setting B (24 + 16 calls with the 8 shared ones,
# from its replay of 24 published epochs, 2026-11-18) scaled by 112 / 40
# with the shared set removed, 68 + 44. The caps keep v3's headroom over
# what those calls cost at measured costs ($0.35 cost-aware, $0.43 uniform
# coverage), so the coverage cap binds only on a spike; the damping is
# setting B's 500 scaled by the coverage calls. EXPLORE_COLDKEY_CAP_V4 is 1.5x
# the private size, as v3's is. Compiled constants, not read from the
# environment.
EXPLORE_SHARED_SIZE_V4 = 0
EXPLORE_COST_AWARE_SIZE_V4 = 68
EXPLORE_COVERAGE_SIZE_V4 = 44
EXPLORE_PRIVATE_SIZE_V4 = EXPLORE_COST_AWARE_SIZE_V4 + EXPLORE_COVERAGE_SIZE_V4
EXPLORE_COLDKEY_CAP_V4 = 168
EXPLORE_SPEND_CAP_USD_V4 = "0.55"
EXPLORE_COVERAGE_TARGET_V4 = 100
EXPLORE_COVERAGE_DAMP_PAIRS_V4 = 1400
EXPLORE_COVERAGE_SPEND_CAP_USD_V4 = "1.10"
# The question mix in protocol v4 (docs/decisions/LIVE_MENU_V4.md, section
# 10). The mix itself is data, derived once from named published reveals by
# scripts/build_mix_v4.py and pinned (routing_data/pool_mix_v4.json, read by
# epoch_plan.mix_v4); these are the rule's parameters, read by that script
# and by the checks of the pinned file. A category's share is proportional to
# the outcome variance p(1 - p) of a uniformly drawn entry on its exploration
# pairs, at least MIX_FLOOR_PCT_V4, rounded to MIX_STEP_PCT_V4; inside a
# category sets split by their own p(1 - p) x pool size (a set with fewer
# than MIX_MIN_PAIRS_V4 exploration pairs takes its category's value), and no
# set's share of the slice exceeds pool size / (slice x MIX_CYCLE_EPOCHS_V4),
# so none is drawn through faster than two 48-epoch windows.
# SLICE_SIZE_V4 is the scored slice the mix is built for: 200
# (SEAT_MINIMAL_V4.md 3.4 and 8: the real-picks check prices picks at the
# measured pick-cost table, so 1,000 questions, five epochs of 200, are its
# minimum; v3's slice is 300). The mix is rebuilt for it, and at 200 no
# set's cycle cap binds. Compiled constants, not read from the environment.
SLICE_SIZE_V4 = 200
MIX_FLOOR_PCT_V4 = 5
MIX_STEP_PCT_V4 = 5
MIX_MIN_PAIRS_V4 = 100
MIX_CYCLE_EPOCHS_V4 = 96
# A proposed entry joins only if, on its 300-question measurement, some
# category reaches this accuracy on at least menu_state's
# PROPOSAL_MIN_CATEGORY_GRADED graded questions, beside the 95% answered gate
# (docs/decisions/LIVE_MENU_V4.md, section 5), in basis points. 5000 (50%) is
# the lead's choice pending the seat-rule package: the design names the gate
# and not its value. Compiled, not read from the environment; a live v4
# validator refuses to start without it (validator._v4_menu_setup).
PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4 = 5000
# The one bound of protocol v4 (docs/decisions/SEAT_MINIMAL_V4.md, 3.1;
# seat_bound.lower_bound, champion.lower_bound): every v4 seat comparison is a
# one-sided normal-mixture confidence sequence at alpha = 1% on integer items
# of 1 / BOUND_UNITS_PER_POINT_V4 of a point. Its logarithms are replaced by
# pinned rationals that are never below them, which only widens it:
# BOUND_A_ALPHA_V4 >= 2 ln(1 / alpha) = 9.21034..., and BOUND_LN2_UP_V4 >= ln 2
# times the bit length of ceil(1 + n / n0) for ln(1 + n / n0). Exact rationals,
# compiled, not read from the environment. Read only under protocol v4.
BOUND_A_ALPHA_V4 = "9211/1000"
BOUND_LN2_UP_V4 = "6932/10000"
BOUND_UNITS_PER_POINT_V4 = 1_000_000
# The labels a v4 comparison reads are those of the last SEAT_WINDOW_EPOCHS_V4
# epochs (docs/decisions/SEAT_MINIMAL_V4.md 3.2, condition 1; labels_v4): the
# same 48 as v3's CHAMPION_WINDOW_EPOCHS, compiled, not read from the
# environment. Read only under protocol v4.
SEAT_WINDOW_EPOCHS_V4 = 48
# The real-picks check of protocol v4 (docs/decisions/SEAT_MINIMAL_V4.md, 3.4
# and 8; picks_v4.picks_check): a head's net value over the bar on the scored
# questions of its answered run, at 1 / routing_protocol.BENCHMARK_LAMBDA
# dollars a correct answer and each answered call at its entry's pick cost in
# the question's category, every question weighted 1, passes the bound above
# with margin -PICKS_MARGIN_POINTS_V4 points on at least
# PICKS_MIN_QUESTIONS_V4 paired questions (n0 BOUND_N0_V4). FAIL_MAX_BP_V4 is
# the absolute failure bound, in basis points: a proof with more than it of
# its exploration calls failed is not counted, except in an outage epoch
# (menu_inputs.outage); a challenge's failed calls fail closed instead
# (picks_v4.challenge_items). Compiled, not read from the environment; read only
# under protocol v4.
PICKS_MARGIN_POINTS_V4 = "3/2"
PICKS_MIN_QUESTIONS_V4 = 500
FAIL_MAX_BP_V4 = 3400
# The fast crown (docs/decisions/FAST_CROWN_V4.md 0, 2, 12): rules A and B tune
# the bound's mixture to BOUND_N0_V4 items (100) instead of each rule's
# minimum, which is worth half again the questions at the decided sizes; rule
# A's minimum is ODDS_MIN_PAIRS_V4 (300) and rule B's PICKS_MIN_QUESTIONS_V4
# (500). A challenger answers the first 500 or 1,000 questions
# (CHALLENGE_SIZES_V4) of the epoch's challenge list of CHALLENGE_LIST_SIZE_V4
# questions (epoch_plan.challenge_list_v4: the scored slice first), pairing
# its picks with those of the seat holder the root's reveal of epoch
# E - CHALLENGE_HOLDER_LAG_V4 names (the newest reveal at E's boundary: the
# same lag as menu_state.MENU_LAG, for the same reason). Compiled, not read
# from the environment; read only under protocol v4.
BOUND_N0_V4 = 100
CHALLENGE_SIZES_V4 = (500, 1000)
CHALLENGE_LIST_SIZE_V4 = 1000
CHALLENGE_HOLDER_LAG_V4 = 2
# The TD's time budget for a challenge (FAST_CROWN_V4.md 3): no challenge
# call is started more than CHALLENGE_BUDGET_S_V4 seconds after the epoch's
# first call. The challenger's own calls are made first, in list order; when
# some are not started the challenge is reported at the largest size whose own
# calls were all made (none below 500: no challenge, an unanswered attempt), and
# a holder-side call not started is no item (HOLDER_FAIL_CAP_BP_V4). A TD
# constant: no validator reads it.
CHALLENGE_BUDGET_S_V4 = 1800
# A challenge's holder-side calls (picks_v4.challenge_verdict): a call the
# challenger's TD did not start (cut by its time or dollar budget) is no item;
# one made that failed scores as the holder answering right. When more than
# HOLDER_FAIL_CAP_BP_V4 basis points (10%) of the challenge's holder-side calls
# failed or were not made, the challenge is void: counted for and against
# nobody, published with that reason. The measured rates (FAST_CROWN_V4.md 1):
# no honest head-epoch failed more than 3 calls of about 1,000; the flaky
# entries the gates admit fail 5-10%; more than 10% is an outage, a cut or a
# deliberate failure, and then the holder's side says nothing about routing,
# while failing it up to the cap moves the items by at most about 1.5 points
# against the challenger (scripts/experiments/experiment_fast_crown_v4.py v).
# A run passes through at most VOID_MAX_IN_RUN_V4 void epoch: a second one
# restarts it like a gap, so voiding cannot select a run's good epochs.
HOLDER_FAIL_CAP_BP_V4 = 1000
VOID_MAX_IN_RUN_V4 = 1
# The TD's dollar ceiling for one challenge (the operator's
# --v4-challenge-usd, FUGAL_V4_CHALLENGE_USD, provisioning field
# `v4_challenge_usd`: whole dollars, 1 to CHALLENGE_SPEND_MAX_USD_V4, default
# CHALLENGE_SPEND_USD_V4). It bounds REALISED spend, with an ESTIMATE
# pre-check. Before its first call the TD estimates the challenge's cost from
# the epoch's pick-cost table (td_v4.challenge_estimate: every own call and
# every holder-side call where the picks differ, each at its entry's pick
# cost, what the challenge's items are priced at) and makes no challenge whose
# estimate exceeds the ceiling (logged with both figures). While running it
# starts no challenge call once the challenge's realised spend plus the
# pick-cost estimate of its calls in flight reaches the ceiling; what is not
# started is treated as a cut by the time budget (own-side prefix, holder-side
# `not_started`, HOLDER_FAIL_CAP_BP_V4). Calls in flight when the ceiling is
# reached finish and are paid: at most HARNESS_CONCURRENCY calls, each bounded
# by its entry's max_tokens at the price ceiling (docs/MINER_GUIDE.md states
# the estimate at seed prices and this overshoot). TD constants: no
# validator reads them.
CHALLENGE_SPEND_USD_V4 = 100
CHALLENGE_SPEND_MAX_USD_V4 = 50000
# The memorisation check of protocol v4 (docs/decisions/SEAT_MINIMAL_V4.md 6;
# memo_v4): a head is refused when the bound says its skill over R on labels
# of questions published before its commit exceeds its skill on the others,
# within benchmark, by more than MEMO_MARGIN_POINTS_V4 Brier points, with
# MEMO_MIN_LABELS_V4 labels in each part (also the bound's n0) and benchmarks
# with fewer than MEMO_MIN_QUESTIONS_V4 questions in either part left out.
# The unpublished reserve is rationed: an epoch draws ceil(R / FRESH_RESERVE_
# EPOCHS_V4) fresh questions for the whole field, R the reserve, so a reserve
# of 1,200 or more lasts 60 days at 20 epochs a day (of 2,400 or more, a
# quarter is left then), whatever the number of hotkeys. The minimum is 500 (the design's 1,000 is out of reach of a
# 48-epoch window once the reserve is small and the field is a few operators;
# SEAT_MINIMAL_V4.md 6.3). Compiled, not read from the environment. Read only
# under protocol v4.
MEMO_MARGIN_POINTS_V4 = 1
MEMO_MIN_LABELS_V4 = 500
MEMO_MIN_QUESTIONS_V4 = 5
FRESH_RESERVE_EPOCHS_V4 = 1200
# The seat of protocol v4 (docs/decisions/SEAT_MINIMAL_V4.md 3.3 to 3.8;
# seat_v4). Rule A, the odds test: X beats Y when the bound above passes on
# X's Brier gain over Y with margin ODDS_MARGIN_POINTS_V4 Brier points on at
# least ODDS_MIN_PAIRS_V4 labels (n0 BOUND_N0_V4): v3's margin, the bootstrap
# replaced by the bound. STANDING_EPOCHS_V4 (G) is how long an absent head
# keeps its place and the holder its seat: a head, the holder included,
# stands while it was verified in this epoch or one of the G - 1 before it
# (owner decision: 3; FAST_CROWN_V4.md 6).
# LATE_ODDS_SLOTS_V4 bounds the late odds one reveal publishes (head-epochs,
# newest epoch first, the heads in turn), so the reveal stays inside the
# miner's holder read at 256 UIDs (tests/test_v4_reveal_budget.py); the rest
# are computed and published in the epochs after. EXPLORE_PRIVATE_MIN_SAMPLES_V4
# is the plausibility check's minimum sample under v4, the value v1-v3 read
# from EXPLORE_DISAGREE_MIN_SAMPLES, whose detector v4 does not run (2.3 item
# 32). Compiled, not read from the environment; read only under protocol v4.
ODDS_MARGIN_POINTS_V4 = "1/10"
ODDS_MIN_PAIRS_V4 = 300
STANDING_EPOCHS_V4 = 3
LATE_ODDS_SLOTS_V4 = 32
EXPLORE_PRIVATE_MIN_SAMPLES_V4 = 8

# --- Protocol v3: activation, and the seat decided on odds ---
# Protocol v3 (a new pool, a new menu, per-miner exploration, odds computed
# and published for every head) applies from the first epoch whose boundary is
# at or after V3's block; the seat is decided on odds from the first epoch at
# or after the odds seat block, and on net value until then. As for the
# confirmation rule: finney's blocks are compiled into the release that
# schedules them (None = not scheduled), the variables are for other chains,
# and on finney a variable that says anything else is refused.
# v0.5.0 (2026-11-17): v3 from epoch 26466, block 9,527,760 (~2026-11-17 17:34
# UTC); the seat on odds 8 epochs later, block 9,530,640 (~2026-11-18 03:10 UTC).
FINNEY_V3_ACTIVATION_BLOCK: int | None = 9527760
FINNEY_ODDS_SEAT_BLOCK: int | None = 9530640
_v3_activation = os.getenv("FUGAL_V3_ACTIVATION_BLOCK", "").strip()
V3_ACTIVATION_BLOCK_ENV = int(_v3_activation) if _v3_activation else None
# Protocol v4 (docs/decisions/LIVE_MENU_V4.md) on finney: v0.7.0 compiles the
# block below (an epoch boundary); on finney the variable may only repeat it,
# and the finney compose must set it to the same value (the miner reads it
# from there; check_safety_invariants). Another chain (a local chain, the
# rehearsals, tests) reads FUGAL_V4_ACTIVATION_BLOCK. Set it only with
# `scripts/set_v4_activation_block.py <block>`, which changes this line, the
# finney compose and the release notes together.
FINNEY_V4_ACTIVATION_BLOCK: int | None = 9572760
_v4_activation = os.getenv("FUGAL_V4_ACTIVATION_BLOCK", "").strip()
# A value that is not a block number never stops an import (v3's released code
# ignores this variable altogether, so a stale or mistyped one on a finney
# neuron must not matter before a release compiles a v4 block): it is kept
# aside and refused by name where the neuron decides its protocol,
# v4_activation_block, off finney.
V4_ACTIVATION_BLOCK_ENV: int | None = None
V4_ACTIVATION_BLOCK_MALFORMED: str | None = None
_v4_digits = _v4_activation[1:] if _v4_activation.startswith("-") else _v4_activation
if _v4_digits.isascii() and _v4_digits.isdigit():
    V4_ACTIVATION_BLOCK_ENV = int(_v4_activation)
elif _v4_activation:
    V4_ACTIVATION_BLOCK_MALFORMED = _v4_activation
_odds_seat = os.getenv("FUGAL_ODDS_SEAT_BLOCK", "").strip()
ODDS_SEAT_BLOCK_ENV = int(_odds_seat) if _odds_seat else None
# --- Protocol v3, menu B: entries added at a block, with a grace period ---
# The v3 menu grows at MENU_B's block without a reset: menu A (models_v3.json,
# 53 entries) before it, menu B (models_v3b.json: A unchanged plus the added
# entries, appended to A's index space) from it. Heads built for menu A stay
# valid; seats, burn-in and the window carry over. For MENU_GRACE_EPOCHS from
# the block the added entries are drawn for exploration (so miners get data)
# but left out of the odds score for every head alike; after it a head must
# give odds for every menu B entry to take or hold the odds seat
# (docs/decisions/V3_MENU_B.md). Same finney/env rule as the blocks above.
# v0.6.0 (2026-11-17): menu B from epoch 26475, block 9,531,000 (~2026-11-18
# 04:22 UTC), one epoch after the odds seat block; grace 20 epochs, to
# ~2026-11-19 04:22 UTC.
FINNEY_MENU_B_BLOCK: int | None = 9531000
_menu_b = os.getenv("FUGAL_MENU_B_BLOCK", "").strip()
MENU_B_BLOCK_ENV = int(_menu_b) if _menu_b else None
# The grace length is consensus like the blocks (it decides which pairs every
# head is scored on): finney's is compiled and a variable may only repeat it;
# another chain reads the variable (default the finney value). Positive only.
# Resolve it with menu_grace_epochs(genesis); MENU_GRACE_EPOCHS is the
# variable-or-default value for code that has no chain (tests, tools).
FINNEY_MENU_GRACE_EPOCHS: int = 20
_menu_grace = os.getenv("FUGAL_MENU_GRACE_EPOCHS", "").strip()
MENU_GRACE_EPOCHS_ENV = int(_menu_grace) if _menu_grace else None
MENU_GRACE_EPOCHS = MENU_GRACE_EPOCHS_ENV if MENU_GRACE_EPOCHS_ENV is not None else FINNEY_MENU_GRACE_EPOCHS
# The odds seat test (champion._challenge_odds, odds_score.py). The margin is
# in Brier points (0.01 of Brier score): a challenger's lower bound, at
# CHAMPION_CONFIDENCE, must clear it on at least ODDS_MIN_PAIRS pairs both
# heads were scored on. Sized on the per-pair spread of Brier differences
# measured on mainnet exploration (about 0.03 to 0.05): a head half a point
# better takes the seat in three or four epochs of 500 pairs, one a point
# better at the floor, and an edge of 0.2 points needs 17 to 50 epochs, which
# is the price of near-equal heads not trading the seat on noise.
BRIER_MARGIN = float(os.getenv("FUGAL_BRIER_MARGIN", "0.1"))
ODDS_MIN_PAIRS = int(os.getenv("FUGAL_ODDS_MIN_PAIRS", "1500"))
# Fresh pairs (drawn after its first pass) the confirmation of an odds take needs.
ODDS_CONFIRM_PAIRS = int(os.getenv("FUGAL_ODDS_CONFIRM_PAIRS", "1500"))
# A head that takes the seat on odds may not route worse than the head it
# replaces by more than this many net-value points (the lower bound of its
# paired net value must be above minus this).
NV_NONINFERIORITY = float(os.getenv("FUGAL_NV_NONINFERIORITY", "0.5"))
# Strength of the reference's shrinkage of a (model, benchmark) rate toward
# the model's overall rate, in pseudo-pairs (odds_score.reference).
ODDS_REF_PRIOR = int(os.getenv("FUGAL_ODDS_REF_PRIOR", "20"))
# A model is listed in the published single-model table once the window holds
# this many pooled pairs of it (single_model.py). Diagnostic: it decides no seat.
SINGLE_MODEL_MIN_PAIRS = int(os.getenv("FUGAL_SINGLE_MODEL_MIN_PAIRS", "300"))
# From v3 the value of one extra correct answer is a pinned number of dollars,
# not CHAMPION_ACCURACY_VALUE times a model's expected spend. It is what v2's
# rule gave on the day it was pinned (30 x the anchor's $0.0163 a question),
# so nothing moves at the switch; what changes is that the menu can then
# change, the anchor model included, without moving the exchange rate between
# accuracy and cost.
CHAMPION_VALUE_PER_CORRECT_USD = float(os.getenv("FUGAL_CHAMPION_VALUE_PER_CORRECT_USD", "0.49"))
# sha256 of the v3 embeddings file: one row per v3 pool question, computed
# once with the pinned backbone profile. The miner's TD routes from it and
# every validator computes odds from it, so it is a consensus input like the
# pool it belongs to, and a file with another hash is refused. None until the
# v3 pool is pinned and its embeddings are computed; a live validator with v3
# scheduled and no pin refuses to start. Where the file is on disk is
# FUGAL_EMBEDDINGS_FILE (operational: the bytes are what is checked).
# Pinned 2026-11-16: 48,320 rows (pool_manifest_v3.json) x 1024, 214,347,992
# bytes; built by scripts/build_embeddings_v3.py (one prompt per row, batch 1,
# one thread, pinned dispatch) and verified bit for bit on two CPUs.
EMBEDDINGS_SHA256: str | None = "f64ca354a5c5fca191abd2bf6930f50e65d6af90390a46b9ebd15fc14cbe4e17"
# Where the pinned file is published (an HTTPS URL, or file:// in tests). The
# image build fetches it into the image (so its bytes are measured) and a
# validator fetches it once at setup: `python -m fugal_subnet.embeddings_file
# fetch`. The URL is only a location; the pinned hash is what is trusted, and a
# download with another hash is refused before anything is written in place.
# None until the v3 release publishes the file.
EMBEDDINGS_URL: str | None = (
    "https://storage.googleapis.com/fugal-reveals/embeddings/v3/embeddings-f64ca354a5c5fca191abd2bf6930f50e65d6af90390a46b9ebd15fc14cbe4e17.npz")


def _chain_block(genesis_hash: str | None, finney_value: int | None, env_value: int | None,
                 name: str) -> int | None:
    """A rule's activation block on the chain with this genesis: finney's is
    the compiled value and the variable may only repeat it; any other chain
    reads the variable (confirm_activation_block states why)."""
    if env_value is not None and env_value < 0:
        raise ValueError(f"{name} must not be negative, got {env_value}")
    if genesis_hash is not None and str(genesis_hash).lower() != FINNEY_GENESIS_HASH.lower():
        return env_value
    if env_value is not None and env_value != finney_value:
        where = ("on finney" if genesis_hash is not None else
                 "and the chain's genesis hash could not be read, so this cannot be shown "
                 "not to be finney")
        raise ValueError(f"{name}={env_value} is set {where}, where the block is fixed by "
                         f"the release ({finney_value}). Unset it.")
    return finney_value


def v3_activation_block(genesis_hash: str | None) -> int | None:
    return _chain_block(genesis_hash, FINNEY_V3_ACTIVATION_BLOCK, V3_ACTIVATION_BLOCK_ENV,
                        "FUGAL_V3_ACTIVATION_BLOCK")


def v4_activation_block(genesis_hash: str | None) -> int | None:
    """The block protocol v4 applies from on this chain, or None. Finney has
    none in any release yet (FINNEY_V4_ACTIVATION_BLOCK), so there the
    variable is refused unless unset; another chain reads it. Never before v3.

    A value that is not a block number (V4_ACTIVATION_BLOCK_MALFORMED) is
    refused by name on any chain but finney, and on finney while no release
    compiles a v4 block it is ignored, as the v3 release ignores the variable."""
    malformed = V4_ACTIVATION_BLOCK_MALFORMED
    if malformed is not None:
        known_finney = (genesis_hash is not None
                        and str(genesis_hash).lower() == FINNEY_GENESIS_HASH.lower())
        if known_finney and FINNEY_V4_ACTIVATION_BLOCK is None:
            return None
        raise ValueError(f"FUGAL_V4_ACTIVATION_BLOCK={malformed!r} is not a block number "
                         "(digits only). Unset it or set the block.")
    block = _chain_block(genesis_hash, FINNEY_V4_ACTIVATION_BLOCK, V4_ACTIVATION_BLOCK_ENV,
                         "FUGAL_V4_ACTIVATION_BLOCK")
    if block is not None:
        v3 = v3_activation_block(genesis_hash)
        if v3 is None or block < v3:
            raise ValueError(f"the v4 activation block ({block}) must not be before the v3 "
                             f"activation block ({v3})")
    return block


def menu_b_block(genesis_hash: str | None) -> int | None:
    """The block from which the v3 menu is menu B. Never before v3."""
    block = _chain_block(genesis_hash, FINNEY_MENU_B_BLOCK, MENU_B_BLOCK_ENV,
                         "FUGAL_MENU_B_BLOCK")
    v3 = v3_activation_block(genesis_hash)
    if block is not None and (v3 is None or block < v3):
        raise ValueError(f"the menu B block ({block}) must not be before the v3 "
                         f"activation block ({v3})")
    return block


def menu_grace_epochs(genesis_hash: str | None) -> int:
    """Menu B's grace in epochs on the chain with this genesis (the same rule
    as the blocks: finney's is compiled; v0.6.0 review, finding 2)."""
    env = MENU_GRACE_EPOCHS_ENV
    if env is not None and env < 1:
        raise ValueError(f"FUGAL_MENU_GRACE_EPOCHS must be a positive number of epochs, got {env}")
    if genesis_hash is not None and str(genesis_hash).lower() != FINNEY_GENESIS_HASH.lower():
        return env if env is not None else FINNEY_MENU_GRACE_EPOCHS
    if env is not None and env != FINNEY_MENU_GRACE_EPOCHS:
        where = ("on finney" if genesis_hash is not None else
                 "and the chain's genesis hash could not be read, so this cannot be shown "
                 "not to be finney")
        raise ValueError(f"FUGAL_MENU_GRACE_EPOCHS={env} is set {where}, where the grace is "
                         f"fixed by the release ({FINNEY_MENU_GRACE_EPOCHS}). Unset it.")
    return FINNEY_MENU_GRACE_EPOCHS


def protocol_for_chain(genesis_hash: str | None) -> int:
    """The protocol a miner image serves on this chain: from the release's
    compiled finney blocks (the variables only repeat them there, or set them
    on another chain). A compose that leaves a variable out can no longer
    leave a miner on the protocol before it (2026-11-17: a compose without
    FUGAL_V3_ACTIVATION_BLOCK would have kept every miner on v2). v4 only
    where a v4 block applies, which on finney no release sets."""
    if v4_activation_block(genesis_hash) is not None:
        return 4
    if v3_activation_block(genesis_hash) is not None:
        return 3
    return 2 if V2_ACTIVATION_BLOCK is not None else 1


def odds_seat_block(genesis_hash: str | None) -> int | None:
    """The block from which the seat is decided on odds. Never before v3."""
    block = _chain_block(genesis_hash, FINNEY_ODDS_SEAT_BLOCK, ODDS_SEAT_BLOCK_ENV,
                         "FUGAL_ODDS_SEAT_BLOCK")
    v3 = v3_activation_block(genesis_hash)
    if block is not None and (v3 is None or block < v3):
        raise ValueError(f"the odds seat block ({block}) must not be before the v3 "
                         f"activation block ({v3})")
    return block

# Miner-side only (never a score): an epoch is not started with fewer than this
# many blocks left before its collection block (~12 min at 12 s/block).
LATE_START_MARGIN_BLOCKS = 60
# Miner-side only: the v2 image commits its head this many blocks after the last
# v1 boundary, clear of the boundary block where validators read commitments.
COMMIT_AFTER_BOUNDARY_BLOCKS = 10
# The pre-flip commit is retried every COMMIT_RETRY_BLOCKS until this many blocks
# before the activation block. A single failed transaction would otherwise leave
# the commit to the post-flip fallback, and validators reading commitments at the
# flip boundary would reject this miner's first v2 epoch as committed late.
COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS = 20
COMMIT_RETRY_BLOCKS = 2

# --- Harness answer budget and call deadlines (image constants, protocol v2) ---
# Part of the measured image, not env-overridable: exploration (the reference
# frame's source) runs on the same harness, so a miner and the frame ask for the
# same answer.
#
# HARNESS_MAX_TOKENS bounds the whole completion, reasoning included (reasoning
# tokens count inside max_tokens on most providers). v1 asked for 2048, and
# measured on the July matrix that cap was the reason most long reasoners were
# graded wrong: 118 of one model's 145 wrong answers stopped at it. At 8192,
# 4.1% of all answers still ended at the cap, nearly all wrong, mostly on
# competition math. 16384 recovers most of those.
HARNESS_MAX_TOKENS = 16384
#
# Deadlines nest, innermost first, so each layer reports before the next gives
# up: one upstream attempt <= the proxy's per-call deadline < the harness's wait.
# A 16k answer from a slow model (~50 tok/s) takes ~330 s.
# PROXY_CALL_DEADLINE_S is wall-clock (runtime._post_with_deadline): a per-read
# socket timeout is not a deadline, because OpenRouter keeps a non-streamed
# request alive while the model reasons (one such call was measured at 917 s).
# The system prompt every v3 model call carries, identical in the harness and
# in whatever serves the winning head (fugal-core), so a model is measured the
# way it is used. An image constant like the budgets above, hash-pinned in
# scripts/check_safety_invariants.py (SYSTEM_PROMPT_V3_SHA256): a byte change
# moves every v3 grade. Source: fugal-core's WORKER_SYSTEM_PROMPT (fugal/
# router.py), with one change: its "Lead with the answer, then supporting
# detail" fought two graders (numeric_final reads the LAST number; letter_mcq
# lost replies that open with a bold letter and then explain, measured
# 2026-11-10 on 26% of Fable 5.2's correct SuperGPQA answers), so it is
# replaced by "a requested answer format wins". fugal-core must carry the same
# text. DRAFT 2026-11-16, for review before the v3 image is frozen.
HARNESS_SYSTEM_PROMPT_V3: str | None = (
    "You are Fugal, a helpful AI assistant. Each message may be handled by a different "
    "underlying model, so keep one consistent voice and identity: never claim to be, or "
    "speculate about, any specific model or the company that built you. Be concise and "
    "direct. Use Markdown for structure and fenced code blocks with a language tag for "
    "code. If you are unsure or lack the information to answer, say so plainly instead of "
    "guessing. When a request says how to give the answer, such as a single letter, a "
    "final number at the end, a \\boxed{} result or one code block, follow that format "
    "exactly; it takes precedence over everything above."
)
PROXY_CALL_DEADLINE_S = 420.0      # total time the proxy spends on one call, retries included
HARNESS_CALL_TIMEOUT_S = 450.0     # the harness's wait on the proxy; > PROXY_CALL_DEADLINE_S

# Protocol v3: one deadline per menu entry, by its reasoning effort. A v3
# entry carries its own answer budget (up to ~128k tokens, models_v3.json), and
# a long answer is what the product would really return, so cutting it off at
# 420 s would teach heads that slow, careful entries fail. The live check of
# 2026-11-16 measured 100-300 s for high/max entries at a 16k budget and one
# low-effort cheap reasoner at 1,117 s.
#
# The ceiling is the collection point, not the epoch: validators collect at
# EPOCH_COLLECT_FRACTION (0.5) of a 72-minute epoch, 36 min after the
# boundary, and the proof must be built and attested before then. 30 min for
# the slow efforts leaves ~6 min for the rest of the proof.
#
# Throughput, worst case (docs/decisions/V3_MENU_BUDGETS.md has the
# arithmetic): 300 scored calls at a 60 s mean plus all 32 exploration calls
# at the 30-min deadline is 75,600 s of calls; at HARNESS_CONCURRENCY 24 that is
# 3,150 s, past the 2,160 s collection point; it needs >= 38 in flight. The
# expected case (exploration mostly fast) is ~1,470 s at 24.
PROXY_CALL_DEADLINE_V3_S = {
    "minimal": 600.0, "low": 600.0, "medium": 900.0,
    "high": 1800.0, "xhigh": 1800.0, "max": 1800.0,
}
PROXY_CALL_DEADLINE_V3_DEFAULT_S = 900.0   # a v3 entry without an effort (thinking on, or none)
HARNESS_WAIT_MARGIN_S = 30.0               # the harness waits this much longer than the proxy


def call_deadline_s(request: dict) -> float:
    """The proxy's wall-clock deadline for one upstream call, from its request.

    A v3 request names pinned providers (routing_protocol.request_for); its
    deadline comes from its reasoning effort. Any other request (v1, v2) keeps
    PROXY_CALL_DEADLINE_S, byte-for-byte the old behaviour.
    """
    if not isinstance(request, dict) or "provider" not in request:
        return PROXY_CALL_DEADLINE_S
    reasoning = request.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    if effort is None:
        return PROXY_CALL_DEADLINE_V3_DEFAULT_S
    return PROXY_CALL_DEADLINE_V3_S.get(str(effort), PROXY_CALL_DEADLINE_V3_DEFAULT_S)

# Checkers that grade by running the candidate's code, and therefore return 0
# unless HARNESS_ALLOW_EXEC. Named here rather than in graders.py, which is
# hash-pinned and must stay byte-identical.
EXECUTION_CHECKERS = frozenset({"exec_io", "exec_unittest"})
# Protocol v3's code checkers (graders_code.py). They also run the candidate's
# code, but in the seccomp sandbox, and are gated by its self-test
# (graders_code.sandbox_selftest, run by harness.code_sandbox_status), NOT by
# HARNESS_ALLOW_EXEC: the loader keeps their questions, and a harness whose
# sandbox failed refuses to produce a proof for an epoch that has any.
SANDBOXED_CODE_CHECKERS = frozenset({"exec_stdio", "exec_func"})

# --- Proof bounds (I2) ---
# The largest token count a proof result may carry, prompt or completion. Not
# env-overridable: it decides which proofs verify. Honest calls are bounded by
# the harness's answer budget and the pool's prompts, orders of magnitude
# below this; the bound exists because a JSON integer is unbounded and every
# value a validator computes with must be bounded first (I2).
PROOF_MAX_TOKENS = 10_000_000

# --- Head constraints ---
HEAD_MAX_BYTES = int(os.getenv("FUGAL_HEAD_MAX_BYTES", str(1 * 1024 * 1024)))  # 1 MB
# Decompressed cap: a valid head is ~130 KB even at 64 models; 8 MB blocks zip bombs.
HEAD_MAX_DECOMPRESSED_BYTES = int(os.getenv("FUGAL_HEAD_MAX_DECOMPRESSED", str(8 * 1024 * 1024)))
HEAD_HIDDEN_DIM = int(os.getenv("FUGAL_HEAD_HIDDEN_DIM", "1024"))  # Qwen3-0.6B
HEAD_MAX_MODELS = int(os.getenv("FUGAL_HEAD_MAX_MODELS", "64"))
# Real OpenRouter IDs are well under 100 chars.
# Without a cap a miner can pad IDs to megabytes: the array still fits the
# decompressed budget, and the strings land verbatim in the published
# reveal artifact. Amplification, not RCE, but free to close.
HEAD_MAX_MODEL_ID_LEN = int(os.getenv("FUGAL_HEAD_MAX_MODEL_ID_LEN", "256"))

# --- Scoring ---
WILSON_CONFIDENCE = float(os.getenv("FUGAL_WILSON_CONFIDENCE", "0.95"))

# score = headroom above the constant-policy frontier, ramped in over burn-in.
#
# A router is paid for the value of reading the question. Anything a policy can
# do WITHOUT reading it — always call model k, or flip a coin between two — is
# free to build, so the subnet must not pay for it. The reference is therefore
# the whole curve of what constant policies buy at each price (the upper convex
# hull of the per-model (cost, accuracy) points from the reference frame, plus
# the origin; see fugal_subnet/frontier.py), and the score is
#
#     headroom = wilson_lcb(miner accuracy) - frontier(miner cost per question)
#
# Every constant policy has zero headroom by construction. A router earns
# exactly the accuracy it adds over the best non-routing alternative at its own
# price. Cheap-and-smart routing is where the frontier is low and steep, so
# that is where headroom is largest — which is the product.
#
# WHY THIS REPLACED quality^0.9 * thrift^0.1 AGAINST THE BEST SINGLE MODEL.
# Measured (docs/decisions/HEAD_EFFICACY.md): a head with W=0 that always called one
# mid-priced model scored 1.183 against the best trained router's 1.057. Any
# quality-versus-cost tradeoff scored against ONE point of the frontier hands a
# cheaper point a cost advantage for free; no exponent fixes that, because the
# constant policy sits on the curve the exponent is trying to trade along. The
# derivation of w=0.9 and the break-even analysis are kept in
# docs/decisions/design-decisions.md as the record of why.
#
# THE QUALITY FLOOR. "Match frontier quality" is the product claim, so a router
# that gives up too much of the best model's accuracy has not delivered it
# however much headroom it shows at a low price. Below this fraction of the
# frontier's maximum accuracy the score is zero. 0.8 is a product constant, not
# a derived one, and is stated as such.
SCORE_QUALITY_FLOOR = float(os.getenv("FUGAL_SCORE_QUALITY_FLOOR", "0.8"))

# How many DISTINCT decayed exploration observations every model on the hull
# needs before the frontier is trusted at face value. Scores scale with
# least_trials/this until it is met. A trial is one (question, model) per
# epoch however many miners answered it (reference_frame.aggregate_observations),
# so this is a count of what has been measured and does not fill faster in a
# bigger field. It was 50, and thirty miners' duplicate answers reached that in
# a couple of epochs on two distinct questions per model, before the prior had
# washed out (diligence 2026-10, finding 3). At fifteen exploration slots over
# seventeen models a cold frame reaches 300 in roughly 340 epochs; the shipped
# bootstrap frame (data/reference_frame.json) is what makes epoch 1 pay.
FRONTIER_MIN_TRIALS = float(os.getenv("FUGAL_FRONTIER_MIN_TRIALS", "300"))

# A fresh artifact's score ramps in over this many scored questions. This is
# what makes evidence reset symmetric: resetting clears accumulated PENALTIES
# as readily as accumulated credit, so without a ramp any miner could wash a
# bad record by flipping one weight bit. With it, climbing back costs exactly
# what earning the position cost in the first place.
BURN_IN_QUESTIONS = int(os.getenv("FUGAL_BURN_IN_QUESTIONS", "3000"))
# Protocol v2's burn-in (the E'' counter, evidence.burn_in_complete): scored
# questions in an unbroken run before a head can TAKE a seat, first or by
# challenge. 1,500 = five epochs of a 300-question slice. Every head is new at
# the flip, so the seat stays empty and the miner share burns for this long
# after it: about six hours, chosen 2026-11-04 over v1's 3,000 (~12 h). v2-only: v1
# epochs keep BURN_IN_QUESTIONS. Holding or resuming a seat needs no burn-in;
# a challenger also needs CHAMPION_MIN_PAIRED_QUESTIONS from its unbroken run.
V2_BURN_IN_QUESTIONS = int(os.getenv("FUGAL_V2_BURN_IN_QUESTIONS", "1500"))

# --- Soft targets ---
SOFT_TARGET_TAU = float(os.getenv("FUGAL_TAU", "1.0"))

# --- Exploration integrity ---
# A proof whose exploration calls fail more often than its scored calls, by
# more than this margin, is rejected. Failed calls are excluded from the
# reference frame (they measure the network, not the model), so without this a
# miner could skip exploration -- 5% of its inference bill -- by blocking its
# own network during that phase and pay nothing for it. 0.34 is six of the
# fifteen default exploration slots; an honest per-call failure rate of a few
# percent reaches it with probability well under one in a million. Diligence
# 2026-10, finding 2.
EXPLORE_FAIL_MARGIN = float(os.getenv("FUGAL_EXPLORE_FAIL_MARGIN", "0.34"))
# Cross-miner agreement. Every miner answers the same exploration question on
# the same nonce-chosen model, so honest answers mostly agree; a miner that
# disagrees with the majority of coldkeys on more than this share of its
# answers (over at least EXPLORE_DISAGREE_MIN_SAMPLES of them, on questions
# with at least EXPLORE_DISAGREE_MIN_VOTERS coldkeys) is flagged and its
# samples are left out of the reference frame for the epoch. Loose on purpose:
# OpenRouter routes one model id to several providers, so temperature 0 is
# not identical output and some honest disagreement is normal. Diligence
# 2026-10, finding 4.
EXPLORE_DISAGREE_MAX = float(os.getenv("FUGAL_EXPLORE_DISAGREE_MAX", "0.4"))
EXPLORE_DISAGREE_MIN_SAMPLES = int(os.getenv("FUGAL_EXPLORE_DISAGREE_MIN_SAMPLES", "8"))
EXPLORE_DISAGREE_MIN_VOTERS = int(os.getenv("FUGAL_EXPLORE_DISAGREE_MIN_VOTERS", "3"))

# --- Dedup ---
# Behavioural clause: two heads whose routing decisions agree on this share of
# the slice are one head. A copier can pass it by nudging a bias term until a
# handful of decisions change (diligence 2026-10, finding 5), which is why the
# two clauses below exist alongside it.
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("FUGAL_DEDUP_THRESHOLD", "0.99"))
# Parameter clause: two heads whose weights sit closer than this relative
# Frobenius distance (fugal_subnet/head_distance.py) are one head whatever
# their decisions say. Measured 2026-10-14 by
# scripts/experiments/experiment_head_similarity.py on the cached SPROUT data: eight heads
# each trained on its own 60% of the questions sit 0.42-0.61 from each other
# (and agree on only 85-92% of decisions); copies of one head with 10%
# multiplicative weight noise sit at 0.10 and change 0-15 of 300 decisions,
# which the behavioural clause alone misses. 0.2 is the geometric midpoint of
# that gap (docs/decisions/HEAD_EFFICACY.md, "Experiment 3").
DEDUP_PARAM_DISTANCE = float(os.getenv("FUGAL_DEDUP_PARAM_DISTANCE", "0.2"))
# Epsilon clause (Subnet 9's rule): a head that agrees with a SENIOR head on
# at least DEDUP_EPSILON_AGREEMENT of decisions must beat its composite score
# by DEDUP_EPSILON (relative) or it is disqualified. Copy-and-tweak then has to
# be copy-and-improve, which is the thing the subnet pays for anyway.
DEDUP_EPSILON_AGREEMENT = float(os.getenv("FUGAL_DEDUP_EPSILON_AGREEMENT", "0.95"))
DEDUP_EPSILON = float(os.getenv("FUGAL_DEDUP_EPSILON", "0.005"))

# --- Evidence accumulation ---
EVIDENCE_HALF_LIFE = int(os.getenv("FUGAL_EVIDENCE_HALF_LIFE", "200"))
LIVENESS_MAX_MISSED_EVIDENCE = int(os.getenv("FUGAL_MAX_MISSED_EVIDENCE", "10"))

# --- Exploration (recovers the counterfactual the TEE architecture removes) ---
# Fraction of the scored slice size, answered with a nonce-chosen model the
# miner does not pick. These answers are never scored against the miner; they
# are the only unbiased samples of the model pool anyone has.
EXPLORE_FRACTION = float(os.getenv("FUGAL_EXPLORE_FRACTION", "0.05"))

# --- Reference frame ---
# Per-model evidence decays on its own half-life. Longer than the per-miner
# half-life on purpose: the frame describes the model pool, which changes far
# more slowly than a miner's head does, and a steadier reference means scores
# stay comparable across time.
FRAME_HALF_LIFE = int(os.getenv("FUGAL_FRAME_HALF_LIFE", "500"))
# Beta prior strength, in pseudo-observations. Makes the frame well-defined at
# epoch 1 with zero samples, and washes out once real evidence exceeds it.
#
# Calibrated by measurement, not taste. The prior is deliberately neutral (0.5)
# and therefore *wrong* for real models, so while it dominates it biases the
# ceiling low — and it dominates for longer in a small field, which makes a
# miner's score depend on how many other miners are online. Measured ceiling
# spread between a 3-miner and a 50-miner field, 30-model pool:
#
#     epoch:        10      25      50     100     200     400
#     K=200:     0.113   0.145   0.155   0.128   0.100   0.063
#     K=50:      0.112   0.103   0.104   0.076   0.041   0.019
#     K=20:      0.054   0.046   0.053   0.052   0.020   0.005
#
# All three converge to the same true value — field size changes how fast, not
# where — but K=20 gets there soonest and is the least field-sensitive at every
# horizon. Its cost is a noisier ceiling in the first few epochs, which weight
# capping and the burn-in ramp already absorb.
FRAME_PRIOR_STRENGTH = float(os.getenv("FUGAL_FRAME_PRIOR_STRENGTH", "20"))
# Deliberately neutral: the subnet does not claim to know any model's accuracy
# before measuring it. Under a flat prior every model ties, and best_model()
# breaks the tie toward the cheapest — the coherent reference for quality per
# dollar among equals. Recalibrate from testnet data, do not guess.
FRAME_PRIOR_ACCURACY = float(os.getenv("FUGAL_FRAME_PRIOR_ACCURACY", "0.5"))
# Fallback completion length for a model the frame has never observed.
FRAME_DEFAULT_COMPLETION_TOKENS = float(
    os.getenv("FUGAL_FRAME_DEFAULT_COMPLETION_TOKENS", "256")
)

# --- TEE (Trusted Execution Environment) ---
TEE_APPROVED_MEASUREMENTS = [
    m.strip() for m in os.getenv("FUGAL_TEE_MEASUREMENTS", "").split(",") if m.strip()
]
# No bundle store: proofs travel inline in the synapse response. See
# fugal_subnet/protocol.py for why an external artifact store earned nothing
# at this payload size.
TEE_PROXY_PORT = int(os.getenv("FUGAL_TEE_PROXY_PORT", "8199"))
TEE_MODEL_PRICES_PATH = os.getenv("FUGAL_MODEL_PRICES", "")

# Where DCAP collateral is fetched from. Pinned HERE, deliberately, rather than
# left to whatever `dcap-qvl` defaults to.
#
# `get_collateral_and_verify(quote)` with no url resolves to
# `PHALA_PCCS_URL = "https://pccs.phala.network"` inside the library. That is
# how every --live validator came to depend on Phala without anyone choosing
# it, and how INVARIANTS came to claim the collateral came from Intel: the
# dependency was invisible because nothing in this repo named it.
#
# The value below is the same endpoint, so behaviour is unchanged. What changes
# is who decides: a `dcap-qvl` bump can no longer move every validator's
# collateral source, because the url is now an argument this project supplies
# rather than a default it inherits. `check_safety_invariants` enforces that
# `verify_dcap` never calls the library without it.
#
# This is NOT a trust root. Collateral is TCB info, QE identity and CRLs, all
# Intel-signed and checked against a root CA compiled into `dcap-qvl` — a
# mirror can withhold or serve stale, never forge. So the endpoint is an
# availability, privacy and latency choice, and the measured cost of that
# choice is ~690 ms per proof from GCP us-central1 (docs/INVARIANTS.md).
#
# Operators wanting neither the third party nor the round trip should point
# this at a local caching PCCS, which re-serves the same Intel-signed bytes.
TEE_PCCS_URL = os.getenv("FUGAL_PCCS_URL", "https://pccs.phala.network").strip()

# How long one collateral fetch may take before the validator gives up on it.
#
# This is a HANG GUARD, not a latency target. The measured round trip is 788 ms
# cold and ~690 ms warm (GCP us-central1 -> Phala), so the default below is
# roughly 12x the worst measured case. It exists for the failure that has no
# natural bound: a PCCS that accepts the connection and then never answers.
#
# Without it a stalled upstream blocks the epoch outright -- no weights, and no
# log line either, because the code that records the failure is downstream of
# the block and never runs. Every validator hits it at the same moment, since
# the collection point is a deterministic block.
#
# A timeout here raises CollateralUnavailable, NOT "invalid". That distinction
# is the prerequisite for this timeout existing at all: before the unverifiable
# path was built, a bare timeout converted a network outage into a false
# accusation of forgery against every honest miner at once.
#
# RESIDUAL, stated because it is easy to assume otherwise: this bounds ONE
# fetch, and the verify loop is serial over every UID. 256 stalled proofs still
# cost 256 x this value, which overruns the 1800 s post-collection budget. The
# per-proof bound stops the unbounded hang; it does not by itself satisfy I6 at
# field scale. Bounding the aggregate is tracked in docs/decisions/OPEN_WORK.md.
# --- Benchmark pool size is a SECURITY parameter ---
#
# Nothing about the number 21,717 says "security", which is exactly why this
# constant exists. What stops a miner memorising the pool instead of learning
# to route is that a linear head does not have the parameters to do it — and
# that is a property of pool SIZE, not of the head.
#
# Measured, random-label fit on isotropic Gaussian embeddings (the most
# favourable geometry, so an upper bound on any real pool):
#
#     2,000 questions -> 100%   memorisation is total, the subnet measures
#                               nothing at all
#     4,000           ->  79%
#     8,000           ->  35%
#    21,717           ->  16.9% the real pool today
#
# So the pool must not be allowed to shrink quietly. It already nearly did:
# excluding one benchmark the way the code-benchmark work excluded others would
# take it to ~7,675 and roughly double what a head can memorise, and no test
# would have noticed — the load would succeed, the hash would change as
# expected, and every downstream check would pass.
#
# Slicing and nonce-derived exploration do NOT defend this. They hide WHICH
# questions are asked; they cannot hide the pool, because they draw from it.
# Head capacity against pool size is the real defence -- and the slice SHAPE
# decides how much of a score that capacity can buy: equal per-benchmark
# quotas let 2,511 questions fill 60% of every slice (SLICE_SHARE_FACTOR).
BENCHMARK_POOL_MIN_SIZE = int(os.getenv("FUGAL_POOL_MIN_SIZE", "12000"))

TEE_COLLATERAL_TIMEOUT = float(os.getenv("FUGAL_COLLATERAL_TIMEOUT", "10"))

# The aggregate ceiling: total seconds one epoch may spend fetching collateral,
# across all proofs. This is the half that actually satisfies I6.
#
# The per-call bound above stops ONE fetch hanging. It does not stop the epoch
# hanging, because the verify loop is serial over every UID -- 256 proofs at
# 10 s each is 2560 s against an 1800 s post-collection window, so the epoch
# returns having already missed the window it exists to hit.
#
# Sized from measurement, not taste. A healthy field costs 256 x ~690 ms
# measured RTT = ~178 s, so the default leaves roughly 3.4x headroom over the
# normal case while capping the pathological one at a third of the
# post-collection budget -- leaving scoring, weight-setting and the reveal the
# ~1200 s they share.
#
# Raising this does not buy more verified proofs when the upstream is healthy;
# it only buys patience with an upstream that is not. Lowering it converts
# proofs to `unverifiable` sooner, which is a real cost: those miners fall
# through to apply_miss.
TEE_COLLATERAL_EPOCH_BUDGET = float(
    os.getenv("FUGAL_COLLATERAL_EPOCH_BUDGET", "600")
)

# How many fetches may be abandoned before this validator stops dialling the
# PCCS for the rest of the epoch.
#
# A bounded fetch cannot be cancelled -- cancelling unwinds a Rust future
# inside pyo3 and crashes the interpreter -- so it is ABANDONED instead, and an
# abandoned fetch keeps its socket until the connection dies on its own.
# Measured at 1.02 leaked file descriptors per abandoned fetch. A container
# inheriting the common 1024 default would exhaust it in a few epochs against a
# persistently stalled PCCS, and lose its axon and subtensor sockets with it --
# an I6 failure arriving by a different road than the hang.
#
# So a run of consecutive timeouts is treated as evidence about the ENDPOINT
# rather than about the proof. Consecutive, not cumulative: an upstream that
# answers between failures — even with an error — is slow, not down, and the
# counter resets.
#
# WHY 10 AND NOT 3, because 3 is the intuitive answer and it is wrong. A
# breaker converts a graded failure into an all-or-nothing cliff, and where the
# cliff sits decides how far two honest validators on the SAME PCCS diverge.
# Simulated over 400 validator-epochs per cell at the real defaults (256
# proofs, T=10, B=600, RTT 0.788 s), counting proofs verified, p10..p90 spread:
#
#     p(stall)   no breaker    K=3      K=10
#     0.02            6          6        6
#     0.05            9         10       11
#     0.10           11        136       11
#     0.20           58        184       58
#
# At K=3 and a 10% stall rate one validator trips at proof ~5 and verifies 100
# while its neighbour never trips and verifies 236 — the breaker is free when
# the endpoint is healthy and maximally divergent exactly when it is degraded,
# which is the only case it exists for. K=10 is indistinguishable from having
# no breaker at all on divergence, while still capping a genuinely dead PCCS at
# 10 abandoned fetches and 100 s of the 600 s ceiling.
#
# The fd cost of the higher threshold is smaller than it first appeared:
# abandoned descriptors ARE reclaimed once the connection dies, by peer close
# or by Linux's own retransmit timeout (~15 min, well inside a 3600 s epoch).
# The leak is transient rather than cumulative across epochs. Measured at
# 1.02 fds per abandoned fetch at a short bound and 5.3 at a longer one, so
# "10 fds" is really 10-53. Still small, and worth paying to stop the breaker
# manufacturing a fork.
TEE_COLLATERAL_MAX_TIMEOUTS = int(
    os.getenv("FUGAL_COLLATERAL_MAX_TIMEOUTS", "10")
)

# --- Routing ---
# New benchmark routing uses the shared success contract at lambda=1. The
# constant below is retained ONLY for historical preference-head experiments;
# no live success path reads it. See routing_protocol.BENCHMARK_LAMBDA.
TRAINING_COST_LAMBDA = float(os.getenv("FUGAL_LAMBDA", "2.0"))

# Routing utilities are quantized to this step before argmax picks a model.
# Deliberately NOT env-overridable: it is consensus-critical, and two validators
# using different quanta would disagree on every near-tie.
#
# Why it exists: the routing decision is argmax(sigmoid(W@h + b) - lam*costs),
# a discontinuity with no tolerance. Any float difference between two validators
# — a different BLAS kernel, CPU generation, or library build — flips the
# decision whenever two models' utilities are close, and across 300 questions
# and 30 models near-ties are possible. Quantizing reduces sensitivity to small
# numerical differences; it does not guarantee agreement at every rounding
# boundary. Conformance fixtures check selections explicitly. Exact ties retain
# artifact row order.
ROUTING_DECISION_QUANTUM = 1e-4

# --- API ---
API_TIMEOUT = int(os.getenv("FUGAL_API_TIMEOUT", "180"))
API_MAX_RETRIES = int(os.getenv("FUGAL_API_RETRIES", "3"))
API_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# --- Backbone ---
BACKBONE_MODEL = os.getenv("FUGAL_BACKBONE", "Qwen/Qwen3-0.6B")
# Two questions bound memory for the 2048-token CPU profile. Mask-aware pooling
# removes padding from the mean; numerical batch/single conformance is checked
# separately. Larger batches require more memory and conformance verification.
BACKBONE_BATCH_SIZE = int(os.getenv("FUGAL_BACKBONE_BATCH_SIZE", "2"))
# torch intra-op threads for the backbone. Miner-side only: validators never
# run the backbone, and embeddings shape only the miner's own routing choices,
# so this is a throughput knob and not consensus state (see
# fugal_subnet/determinism.py). Default 1 keeps every other tool's behaviour;
# a TD miner sets 0 (all vCPUs) in its compose. Read by determinism.py before
# torch imports, so it is documented here and consumed there.
# FUGAL_BACKBONE_THREADS=1
ROUTER_SYSTEM_PROMPT = (
    "You are a routing model. Given a question, your hidden state will be used to "
    "predict which language model can best answer it. Read the question carefully."
)

# --- Weight stability ---
MAX_WEIGHT_DELTA = float(os.getenv("FUGAL_MAX_WEIGHT_DELTA", "0.3"))

# --- Protocol v2: activation ---
# The block at which every validator switches from the v1 rules to the v2
# rules. It must be an epoch boundary (a multiple of blocks_per_epoch), and the
# rules an epoch is scored under are decided by that epoch's BOUNDARY block
# (slicer.protocol_version_for_boundary), never by the block a validator
# happens to be at. So a retried epoch, or a validator that reaches an epoch
# late, applies the same rules as every peer. Unset means v2 never activates
# and the v1 rules apply for good. A consensus input, like the epoch length:
# set once per deployment, identical on every neuron, and in the consensus
# digest's epoch block, so a validator with a different or missing value
# shows a different digest from the first epoch. The miner image reads the
# same variable to hold its axon until the flip. docs/INVARIANTS.md, "I1 —
# protocol v2 activates at one named boundary".
_v2_activation = os.getenv("FUGAL_V2_ACTIVATION_BLOCK", "").strip()
V2_ACTIVATION_BLOCK = int(_v2_activation) if _v2_activation else None
# Finney's flip block, as it happened (2026-11-11). A live finney validator
# whose FUGAL_V2_ACTIVATION_BLOCK is unset or different is refused at startup
# (neurons/validator.py, _check_chain_configuration): unset, it would score
# every epoch by the v1 rules with nothing in its log to say so.
FINNEY_V2_ACTIVATION_BLOCK = 9485640

# --- Protocol v2: champion scoring (paired net value) ---
# From the activation block, a challenger is compared with the current
# champion on the SAME questions, by net value per question:
#
#     NV_i = u_i * (g_challenger,i - g_champion,i) - (s_challenger,i - s_champion,i) / V
#     V    = CHAMPION_ACCURACY_VALUE * the pinned anchor's expected spend per
#            question (CHAMPION_ANCHOR_MODEL, below)
#     u_i  = CHAMPION_HARD_WEIGHT on the hard-reasoning category, else 1
#     NV   = 100 * mean_i NV_i                       (percentage points)
#
# The challenger takes the seat when the one-sided paired-bootstrap lower
# bound of NV, at CHAMPION_CONFIDENCE over CHAMPION_BOOTSTRAP_RESAMPLES
# resamples, exceeds CHAMPION_MARGIN points. The champion keeps the seat with
# no margin, so two near-equal heads cannot trade it on noise. This margin
# also replaces the earlier draft's absolute headroom margin (0.004) wherever
# a head must beat a senior or the champion: there is one margin in v2, in NV
# points.
#
# CHAMPION_ACCURACY_VALUE (m) prices one extra correct answer at m questions'
# worth of the anchor's spend. At 30, accuracy dominates and cost
# decides between heads of near-equal accuracy. It is a published policy
# constant. With no champion there is no comparison: the eligible head with
# the highest measured net value takes the empty seat (champion.solo_value).
# At today's prices (V about $0.49) the whole v2 price range is worth about 4
# points of accuracy. Kept at 30 at launch for lack of v2 accuracy data;
# revisited after the first week (MAINNET_LAUNCH, "revisit m").
CHAMPION_ACCURACY_VALUE = float(os.getenv("FUGAL_CHAMPION_ACCURACY_VALUE", "30"))
CHAMPION_HARD_WEIGHT = float(os.getenv("FUGAL_CHAMPION_HARD_WEIGHT", "2"))
# The anchor that prices one extra correct answer: V = CHAMPION_ACCURACY_VALUE
# x this model's expected spend per question, from the reviewed v2 token
# manifest (mean input and output tokens) at the pinned v2 prices
# (v2_epoch.anchor_spend_per_question). Pinned, so the exchange rate between
# accuracy and cost does not depend on who holds the seat: with the
# champion's own spend as the anchor, a cheap champion made every pricier,
# more accurate router pay for its spend at the cheap champion's rate.
CHAMPION_ANCHOR_MODEL = os.getenv("FUGAL_CHAMPION_ANCHOR_MODEL", "anthropic/claude-fable-5.1")
CHAMPION_MARGIN = float(os.getenv("FUGAL_CHAMPION_MARGIN", "0.2"))
# 99%, because the test is re-run every epoch on a growing, then sliding,
# window: a one-sided test at confidence c lets a head with NO real edge take
# the seat with probability about 1 - c at EACH epoch, and those chances add
# up. Simulated with the shipped paired_test and constants (2026-11-08): at
# 90%, a zero-edge near-copy challenger (20% different routing) took the seat
# in 22% of 30-epoch runs (an independent router 35%; one of 10 sybil heads
# 70%); at 99%, 3% (independent 6%; 10 sybils 27%), while a challenger 2
# points better still took it in every run (1 point better: 63%).
CHAMPION_CONFIDENCE = float(os.getenv("FUGAL_CHAMPION_CONFIDENCE", "0.99"))
CHAMPION_BOOTSTRAP_RESAMPLES = int(os.getenv("FUGAL_CHAMPION_BOOTSTRAP_RESAMPLES", "2000"))
# Quality gates on a challenger, besides the margin: accuracy at or above the
# high-water mark (overall, and on the hard subset once the comparison holds
# at least CHAMPION_HARD_GATE_MIN_QUESTIONS hard questions; the marks start at
# the first champion's lower bound, never fall while a champion or its ghost
# stands, and reset when the last ghost expires with the seat empty), and
# per-call time within
# CHAMPION_TIME_GATE_FACTOR of the champion's. Time is a gate only; it is not
# priced into NV, and it is not applied until proofs carry a hash-bound
# per-call latency.
# An absolute accuracy floor on a head's window accuracy (shared grades, the
# last CHAMPION_WINDOW_EPOCHS), applied to taking AND holding the seat. It
# replaces v1's frame-based quality floor, because in v2 nothing that pays may
# read the frame. Its job is narrow: a broken head (every call failing, say)
# must not be seated first and collect the champion share until someone
# beats it. 0.30 is far below any working router on any plausible v2 list,
# but it is UNMEASURED on the v2 benchmarks and is to be revisited after the
# first v2 week.
CHAMPION_MIN_ACCURACY = float(os.getenv("FUGAL_CHAMPION_MIN_ACCURACY", "0.30"))
# Insurance on the paired test: a take needs at least this many paired
# questions against the bar, whatever the bootstrap says. The v2 burn-in
# (V2_BURN_IN_QUESTIONS, 1,500 in an unbroken run) is the same count, so a
# burned-in head normally meets it already; this keeps a short
# overlap with a ghost or a champion that only just appeared from deciding
# the seat on a few hundred questions.
CHAMPION_MIN_PAIRED_QUESTIONS = int(os.getenv("FUGAL_CHAMPION_MIN_PAIRED_QUESTIONS", "1500"))
CHAMPION_HARD_GATE_MIN_QUESTIONS = int(os.getenv("FUGAL_CHAMPION_HARD_GATE_MIN_QUESTIONS", "100"))
CHAMPION_TIME_GATE_FACTOR = float(os.getenv("FUGAL_CHAMPION_TIME_GATE_FACTOR", "2"))
# A champion that stops being eligible leaves a ghost bar: for this many
# epochs a head takes the seat only by beating the ghost by the margin, and
# otherwise the miner share burns. GHOST_OFF when it simply missed proofs,
# GHOST_RETIRE when its owner committed a different head. Kept as short as
# the copy rule allows, because every ghost epoch with no qualifying head
# burns.
CHAMPION_GHOST_OFF = int(os.getenv("FUGAL_CHAMPION_GHOST_OFF", "3"))
CHAMPION_GHOST_RETIRE = int(os.getenv("FUGAL_CHAMPION_GHOST_RETIRE", "24"))
# The paired comparison reads the last CHAMPION_WINDOW_EPOCHS epochs by
# calendar, counting only epochs in which both heads were scored and, for the
# challenger, only from its first commit. Bounded so the per-question state
# every validator holds (and rebuilds from reveals) is bounded: 48 epochs is
# up to 14,400 paired questions, about two and a half days.
CHAMPION_WINDOW_EPOCHS = int(os.getenv("FUGAL_CHAMPION_WINDOW_EPOCHS", "48"))

# --- Protocol v2: fresh-data confirmation of a take ---
# From the confirmation activation block (below), beating a settled champion
# on the paired test once does not take its seat. The epoch of that first
# pass is recorded, and the challenger takes the seat when it passes the SAME
# test (same margin, same confidence) on questions drawn after that epoch: at
# least CHAMPION_CONFIRM_QUESTIONS of them, while the first pass is still in
# the comparison window. The champion holds, and is paid, meanwhile.
#
# Why. The take test runs every epoch for every challenger, so its false
# positives add up over epochs AND over challengers: k heads with no real
# edge pass by luck about k times as often as one, which makes registering
# more hotkeys worth something. The first pass is therefore a selection, and
# a selection is confirmed on data it did not see. A head with no edge has to
# be lucky twice, on two samples that share no question.
#
# 1,500 fresh questions is five epochs and the same count as the burn-in and
# CHAMPION_MIN_PAIRED_QUESTIONS, so "enough to judge on" means one thing.
# Measured on the mainnet reveals e00026349-e00026384 (36 epochs): the one
# take in them (+0.83 points at its first pass) measured +1.06 on its first
# 1,500 fresh questions and confirms at 2,400 (eight epochs).
CHAMPION_CONFIRM_QUESTIONS = int(os.getenv("FUGAL_CHAMPION_CONFIRM_QUESTIONS", "1500"))
# A head seated from an EMPTY seat (the first fill, or over a retired ghost)
# beat nobody, and the pick is the best of several noisy measurements. So for
# this many epochs after such a fill the seat can still be taken by the plain
# test, as before the rule; only then is the champion settled and a challenger
# confirmed against it. Without this the confirmation would also protect a
# lucky first fill. A champion that was itself confirmed is settled at once.
CHAMPION_CONFIRM_OPEN_EPOCHS = int(os.getenv("FUGAL_CHAMPION_CONFIRM_OPEN_EPOCHS", "5"))
# From the same block, a RETIRED champion's ghost bars for this many epochs
# instead of CHAMPION_GHOST_RETIRE. A retired head answers no new questions,
# so a head committed after it retired can never be paired with it, and for
# the whole window the seat can only go to a head that already overlapped it;
# otherwise the miner share burns (24 epochs is about 29 hours). The window
# exists so that the owner's replacement head can finish burn-in and compete
# for the empty seat on equal terms with copies of the retired one, which is
# V2_BURN_IN_QUESTIONS (five epochs) plus slack for one early miss.
CHAMPION_GHOST_RETIRE_CONFIRM = int(os.getenv("FUGAL_CHAMPION_GHOST_RETIRE_CONFIRM", "8"))
# The activation block: the two rules above apply to every epoch whose
# boundary block is at or after it (the boundary decides, as for v2, so a
# retried or late epoch applies the same rules as every peer).
#
# Finney's value is COMPILED IN, not read from the environment. The v2 flip
# asked every validator operator to set a variable, and a validator that had
# not set it ran the old rules with nothing to say so. A validator that runs
# this release agrees on the block by construction, and one that sets the
# variable on finney to anything else is refused at startup
# (confirm_activation_block). None means not scheduled: the release that
# schedules it sets the number here. Other chains (testnet, the rehearsals)
# have no compiled value and read FUGAL_CONFIRM_ACTIVATION_BLOCK.
# The single v3 upgrade sets this together with the v3 and odds seat blocks:
# v0.5.0 (2026-11-17), the same block as v3.
FINNEY_CONFIRM_ACTIVATION_BLOCK: int | None = 9527760
_confirm_activation = os.getenv("FUGAL_CONFIRM_ACTIVATION_BLOCK", "").strip()
CONFIRM_ACTIVATION_BLOCK_ENV = int(_confirm_activation) if _confirm_activation else None


def confirm_activation_block(genesis_hash: str | None) -> int | None:
    """The confirmation rule's activation block on the chain with this genesis.

    Finney: the compiled value, and the environment variable is refused unless
    it says the same. Any other chain: the environment variable. An unreadable
    genesis with the variable unset is treated as finney (the compiled value),
    which is the safe reading: it can only be a chain this release knows.
    """
    env = CONFIRM_ACTIVATION_BLOCK_ENV
    if env is not None and env < 0:
        raise ValueError(f"FUGAL_CONFIRM_ACTIVATION_BLOCK must not be negative, got {env}")
    if genesis_hash is not None and str(genesis_hash).lower() != FINNEY_GENESIS_HASH.lower():
        return env
    if env is not None and env != FINNEY_CONFIRM_ACTIVATION_BLOCK:
        where = ("on finney" if genesis_hash is not None else
                 "and the chain's genesis hash could not be read, so this cannot be shown "
                 "not to be finney")
        raise ValueError(
            f"FUGAL_CONFIRM_ACTIVATION_BLOCK={env} is set {where}, where the block is "
            f"fixed by the release ({FINNEY_CONFIRM_ACTIVATION_BLOCK}). Unset it."
        )
    return FINNEY_CONFIRM_ACTIVATION_BLOCK


# --- Protocol v2: payout ---
# The miner share is split across the last three champions: current, previous,
# the one before. A seat pays only while its head is live and passes the
# gates; an empty seat's share goes to the current champion.
PAYOUT_SEAT_SHARES = tuple(
    float(x) for x in os.getenv("FUGAL_PAYOUT_SEAT_SHARES", "0.85,0.10,0.05").split(",")
)

# --- Validator budget ---
# Deliberately absent. Under the TEE architecture the validator never calls a
# model, so it has no API budget to cap and no shared model pool to bound.
# Miners pay for their own inference inside the TEE, which is what resolved the
# I5 cost-asymmetry problem — a $1 registration could previously waste $30+ of
# validator inference per epoch. tests/test_paid_safety.py asserts the
# validator has no epoch_budget parameter so this cannot creep back.

# --- Head commitment (anti-copy / anti-overfit) ---
# When enabled, a head is only scored if its SHA256 was committed on-chain
# (pallet_commitments) at or before the epoch boundary block. This makes
# head-copying and slice-overfitting detectable: any head changed after the
# epoch nonce is knowable has a commitment block past the boundary.
REQUIRE_COMMITMENT = os.getenv("FUGAL_REQUIRE_COMMITMENT", "1") not in ("0", "false", "False")

# --- Miner axon access control ---
# Minimum stake a querying hotkey needs in --live, permit or not. The unit is
# the chain's `total_stake` for the UID as `metagraph.S` reports it: alpha on
# this subnet plus root TAO at the chain's 0.18 root weight, after child/parent
# key inheritance -- the same quantity subtensor uses for permits. A permit
# alone does not admit: the chain grants one to any nonzero stake while fewer
# than max_allowed_validators hotkeys are staked (review 2026-11-02). The axon
# answers with the miner's FULL head, so an open axon is a free download of
# the leader's router for anyone with a registration (diligence 2026-10,
# finding 5); dedup, not this floor, is the defence, so the floor is set where
# it never starves a real validator (50, decided 2026-11-03) rather than where
# it would stop a determined copier. It ships inside the measured image, so a
# change is an image rotation. Mock ignores it and admits any registered
# hotkey, for local testnets and the dress rehearsal; a live miner refuses to
# start with it at 0 (neurons/miner.py, caller_allowed).
MIN_VALIDATOR_STAKE = float(os.getenv("FUGAL_MIN_VALIDATOR_STAKE", "50"))
# This validator's public egress IP, recorded on chain at startup so miners can
# admit only validators at their cloud firewall (fugal_subnet/axon_allowlist.py,
# scripts/axon_firewall.py). Operational: it decides who can reach a miner, not
# any score.
VALIDATOR_IP = os.getenv("FUGAL_VALIDATOR_IP", "").strip()

# Grader execution limits live in graders.py itself (EXEC_TIMEOUT_SECS and the
# output cap): that file is hash-pinned consensus code and cannot read an
# env-overridable value without making every validator's grades depend on its
# environment.


# --- Reveal publishing (operational; nothing here reads back into a score) ---
# Public-read GCS bucket each epoch's reveal.json is uploaded to, delayed past
# the weight commit-reveal window, with its sha256 committed on chain by the
# validator hotkey (fugal_subnet/reveal_publisher.py). Unset: publishing is
# skipped with one warning. Diligence 2026-10, finding A1.
REVEAL_BUCKET = os.getenv("FUGAL_REVEAL_BUCKET", "").strip()
# Override the hold (epochs) read from the chain's commit-reveal period. 0 =
# read it from the chain.
REVEAL_DELAY_EPOCHS = int(os.getenv("FUGAL_REVEAL_DELAY_EPOCHS", "0"))

# --- Protocol v4 menu mirrors (operational; every byte read is checked by hash) ---
# Where a reader also looks for a menu's bytes and the signed lists it names,
# after the root's bucket: https base URLs, comma-separated, each serving the
# root's content-addressed names (<netuid>/menus/<hash>.json, <netuid>/lists/...).
# Any validator with a bucket mirrors what it accepted (menu_publish.Mirror).
# Nothing depends on a mirror existing; a mirror that serves other bytes is skipped.


def parse_menu_mirrors(text: str) -> tuple[str, ...]:
    return tuple(m.strip().rstrip("/") for m in str(text).split(",") if m.strip())


MENU_MIRRORS = parse_menu_mirrors(os.getenv("FUGAL_MENU_MIRRORS", ""))

# --- The v2 star: root and followers (operational) ---
# Every validator but the root verifies its v2 seat state against the root's
# published reveals and re-anchors on them after a missed epoch
# (fugal_subnet/v2_sync.py). A follower fetches those reveals itself
# (fugal_subnet/reveal_sync.py): on finney from the root's public bucket with
# nothing to set, elsewhere from FUGAL_REFERENCE_BUCKET. The root is the
# validator with FUGAL_ROOT=1, or the one whose FUGAL_REVEAL_BUCKET is the
# reference bucket. FUGAL_FRAME_BOOTSTRAP (a directory the operator keeps in
# sync) still works and takes precedence over the built-in fetch.
# FUGAL_ROOT=0 says the opposite, for the one case that needs it: a root that
# lost its state follows another validator's reveals until it has caught up
# (docs/VALIDATOR_GUIDE.md, "The root"), although it publishes to the
# reference bucket. Unset: decided by the bucket.
# Only the explicit values count; anything else is refused at import, so a
# typo such as FUGAL_ROOT=off cannot make a validator the root.


def _parse_optional_flag(name: str, raw: str) -> bool | None:
    value = str(raw or "").strip().lower()
    if not value:
        return None
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    raise ValueError(f"{name}={raw!r} is not one of 1, true, yes, 0, false, no "
                     "(or unset). Fix it before starting the validator.")


def parse_root_flag(raw: str) -> bool | None:
    return _parse_optional_flag("FUGAL_ROOT", raw)


ROOT_VALIDATOR: bool | None = parse_root_flag(os.getenv("FUGAL_ROOT", ""))
REFERENCE_BUCKET = os.getenv("FUGAL_REFERENCE_BUCKET", "").strip()

# The root validator's hotkey. Followers verify the reveals they download
# against this hotkey's on-chain commitment (`reveal:<epoch_id>:<sha256>`,
# reveal_publisher), so the bucket alone cannot change what they re-anchor
# on. Built into the release on finney (uid 0 of netuid 70, the subnet
# owner's validator), where the variable cannot change it; other chains name
# their root in FUGAL_ROOT_HOTKEY, and without it a follower fetches
# unverified, as before. Not a consensus input: it decides which published
# files a follower accepts, never how an epoch is scored.
FINNEY_ROOT_HOTKEY = "5E4x811vM4DLkrTwMhrMX1deUp9zPt3GJLinxZneBvAMTL5U"
ROOT_HOTKEY_ENV = os.getenv("FUGAL_ROOT_HOTKEY", "").strip()


def root_hotkey(finney: bool, env: str | None = None) -> str | None:
    """The hotkey whose reveal commitment a follower verifies against.

    On finney the release's; a FUGAL_ROOT_HOTKEY that names another hotkey is
    refused, so a typo or a stale value cannot point followers elsewhere.
    Elsewhere the variable, or None (unverified, as before this check)."""
    value = (ROOT_HOTKEY_ENV if env is None else env).strip()
    if finney:
        if value and value != FINNEY_ROOT_HOTKEY:
            raise ValueError(
                f"FUGAL_ROOT_HOTKEY={value} is set on finney, where the root's hotkey is "
                f"fixed by the release ({FINNEY_ROOT_HOTKEY}). Unset it.")
        return FINNEY_ROOT_HOTKEY
    return value or None


# Protocol v4 reads each epoch's menu from the root's reveal commitment, so a
# node that rebuilds past menus (menu_publish.rebuild) must read each epoch's
# commitment under the hotkey that was the root's for that epoch, not under
# today's: read under one hotkey, a walk across a change of the root's key
# ends silently on another menu. The table is (first epoch, hotkey), sorted:
# the commitment for epoch E (the root's reveal commitment for E - MENU_LAG,
# read at E's boundary) is read under the last row whose first epoch is at or
# before E. A release that rotates the root's key appends a row and keeps
# every older one; the new root's first commitment is its reveal for that
# row's epoch minus MENU_LAG. Finney's is compiled; elsewhere
# FUGAL_ROOT_HOTKEY names one hotkey for every epoch.
FINNEY_ROOT_HOTKEYS_V4: tuple[tuple[int, str], ...] = ((0, FINNEY_ROOT_HOTKEY),)


def root_hotkeys_v4(finney: bool, env: str | None = None) -> tuple[tuple[int, str], ...]:
    """The root's hotkeys for protocol v4's menu commitments, as
    ((first epoch, hotkey), ...): finney's compiled table (a
    FUGAL_ROOT_HOTKEY that names another hotkey is refused, as in
    `root_hotkey`), or the variable's one hotkey from epoch 0, or () when
    none is known."""
    hk = root_hotkey(finney, env)
    if finney:
        return FINNEY_ROOT_HOTKEYS_V4
    return ((0, hk),) if hk else ()


# --- Head publication (operational; nothing here reads back into a score) ---
# Whether this validator uploads the bytes of every head it verified to its
# reveal bucket, by sha256, with or after the reveal that lists them
# (fugal_subnet/head_publisher.py; when, below). Unset: the root does, nobody else. 1: this
# validator does too (it needs FUGAL_REVEAL_BUCKET). 0: off, the root
# included. It changes no reveal, hash or score: the locations go in a
# sidecar beside the reveal. Parsed like FUGAL_ROOT, so a typo is refused at
# import and cannot switch it either way.


def parse_publish_heads_flag(raw: str) -> bool | None:
    return _parse_optional_flag("FUGAL_PUBLISH_HEADS", raw)


PUBLISH_HEADS: bool | None = parse_publish_heads_flag(os.getenv("FUGAL_PUBLISH_HEADS", ""))


def publish_heads_enabled(root: bool, flag: bool | None = None) -> bool:
    """Whether verified heads are published: the switch when it is set,
    otherwise only on the root. Call it with `flag=PUBLISH_HEADS`."""
    return bool(root) if flag is None else bool(flag)


# Which heads are published when. A head that holds or has held a seat is
# published with the reveal that says so. Any other verified head is published
# once it is this many epochs past the epoch it first verified in, so a head
# is measured before its bytes are public. 32 epochs is about 38 hours; a
# head that passes the seat test at once is seated in roughly 15 epochs at
# the current pool rate (ODDS_MIN_PAIRS, then ODDS_CONFIRM_PAIRS), and in
# about 30 if it misses one epoch on the way. 0 publishes
# every head with the first reveal that lists it. Operational, like the
# switch above: it decides when bytes are uploaded, never a reveal or a score.


def parse_publish_heads_after_epochs(raw: str) -> int:
    value = str(raw or "").strip()
    if not value:
        return 32
    try:
        epochs = int(value)
    except ValueError:
        epochs = -1
    if epochs < 0:
        raise ValueError(f"FUGAL_PUBLISH_HEADS_AFTER_EPOCHS={raw!r} is not a whole number of "
                         "epochs, 0 or more (or unset). Fix it before starting the validator.")
    return epochs


PUBLISH_HEADS_AFTER_EPOCHS: int = parse_publish_heads_after_epochs(
    os.getenv("FUGAL_PUBLISH_HEADS_AFTER_EPOCHS", ""))


# --- Head archive (operational; nothing here reads back into a score) ---
# Directory where the validator keeps every VERIFIED miner head, raw bytes
# named by sha256, plus an index.jsonl (fugal_subnet/head_archive.py). Private:
# created 0700, never uploaded (head publication, above, keeps its own
# outbox). Unset (default): heads are not archived.
HEAD_ARCHIVE_DIR = os.getenv("FUGAL_HEAD_ARCHIVE_DIR", "").strip()

# --- Chain endpoints (operational) ---
# The validator pins three reads to old blocks every epoch (commitments and
# the owner hotkey at the boundary block, the metagraph at the collection
# block). A non-archive node serves roughly the last 300 blocks, so a validator
# that starts late in an epoch, or retries one, can be refused those reads. Name
# archive-capable endpoints here; the SDK uses them for historical queries and
# the fallbacks for everything else when the primary drops.
ARCHIVE_ENDPOINTS = [
    e.strip() for e in os.getenv("FUGAL_ARCHIVE_ENDPOINTS", "").split(",") if e.strip()
]
FALLBACK_ENDPOINTS = [
    e.strip() for e in os.getenv("FUGAL_FALLBACK_ENDPOINTS", "").split(",") if e.strip()
]
# How many times one epoch may re-submit set_weights after a failure, while
# blocks remain before the next boundary and the chain's rate limit allows.
# Each SDK call already retries up to five times internally; this bounds the
# outer loop so a transient websocket drop costs seconds, not the epoch.
WEIGHTS_SET_RETRIES = int(os.getenv("FUGAL_WEIGHTS_SET_RETRIES", "3"))

# --- Chain identity ---
# Genesis block hash of Bittensor mainnet (finney). Not configurable: it is the
# one fact about the chain a --network URL cannot disguise, and it is what
# both neurons compare against to refuse --mock on mainnet (diligence 2026-10,
# finding A3). Verified 2026-10-29 at finney block ~9,391,998 on two independent
# endpoints (lite.sub.latent.to, bittensor-finney.api.onfinality.io). Re-verify with
# `subtensor.substrate.get_block_hash(0)` before launch; a wrong value here
# makes the guard inert, never wrongly blocking.
FINNEY_GENESIS_HASH = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"

# --- Every FUGAL_ environment variable this module reads, classified ---
#
# Diligence 2026-10, additional finding A2: scoring constants could be changed
# by environment variable and were not in the consensus digest, so a validator
# with FUGAL_SCORE_QUALITY_FLOOR=0.5 exported scored the field differently
# from every other validator and nothing recorded why. Three classes:
#
# CONSENSUS_ENV_VARS   decide a score or which proofs verify. Their VALUES are
#                      in `fingerprint.consensus_digest`, and a live validator
#                      refuses to start with any of them set in its
#                      environment (`fingerprint.check_environment`). Mock
#                      warns, so local testnets and the dress rehearsal can
#                      shrink slices.
# CONSENSUS_ENV_INPUTS must agree across neurons and are SUPPOSED to be set
#                      per deployment (the epoch length is the chain's tempo;
#                      the approved entries are the published list). In the
#                      fingerprint so a divergence is attributable.
# OPERATIONAL_ENV_VARS change timing, ports, paths or a miner's own costs,
#                      never a validator's scores.
#
# scripts/check_safety_invariants.py fails if a FUGAL_ variable is read
# ANYWHERE under fugal_subnet/ or neurons/ and listed in none of the three, so
# a new constant cannot be added unclassified. Until 2026-11-05 only this file
# was scanned, and the pool's skip list and the unpinned-pool declaration
# (read by benchmarks/loader.py) were consensus-affecting, undigested and not
# refused on finney (review finding F-027).
CONSENSUS_ENV_VARS = (
    "FUGAL_SLICE_SIZE",
    "FUGAL_SLICE_SHARE_FACTOR",
    "FUGAL_COLLECT_FRACTION",
    "FUGAL_EXPLORE_FRACTION",
    "FUGAL_HARNESS_ALLOW_EXEC",
    "FUGAL_WILSON_CONFIDENCE",
    "FUGAL_SCORE_QUALITY_FLOOR",
    "FUGAL_FRONTIER_MIN_TRIALS",
    "FUGAL_BURN_IN_QUESTIONS",
    "FUGAL_V2_BURN_IN_QUESTIONS",
    "FUGAL_EXPLORE_SHARED_SIZE",
    "FUGAL_EXPLORE_PRIVATE_SIZE",
    "FUGAL_EXPLORE_COLDKEY_CAP",
    "FUGAL_EXPLORE_PRIVATE_Z",
    "FUGAL_EXPLORE_PRIVATE_MIN_TRIALS",
    "FUGAL_BRIER_MARGIN",
    "FUGAL_ODDS_MIN_PAIRS",
    "FUGAL_ODDS_CONFIRM_PAIRS",
    "FUGAL_NV_NONINFERIORITY",
    "FUGAL_ODDS_REF_PRIOR",
    "FUGAL_SINGLE_MODEL_MIN_PAIRS",
    "FUGAL_CHAMPION_VALUE_PER_CORRECT_USD",
    "FUGAL_EXPLORE_FAIL_MARGIN",
    "FUGAL_EXPLORE_DISAGREE_MAX",
    "FUGAL_EXPLORE_DISAGREE_MIN_SAMPLES",
    "FUGAL_EXPLORE_DISAGREE_MIN_VOTERS",
    "FUGAL_DEDUP_THRESHOLD",
    "FUGAL_DEDUP_PARAM_DISTANCE",
    "FUGAL_DEDUP_EPSILON_AGREEMENT",
    "FUGAL_DEDUP_EPSILON",
    "FUGAL_EVIDENCE_HALF_LIFE",
    "FUGAL_MAX_MISSED_EVIDENCE",
    "FUGAL_FRAME_HALF_LIFE",
    "FUGAL_FRAME_PRIOR_STRENGTH",
    "FUGAL_FRAME_PRIOR_ACCURACY",
    "FUGAL_FRAME_DEFAULT_COMPLETION_TOKENS",
    "FUGAL_POOL_MIN_SIZE",
    "FUGAL_MAX_WEIGHT_DELTA",
    "FUGAL_REQUIRE_COMMITMENT",
    "FUGAL_HEAD_HIDDEN_DIM",
    "FUGAL_HEAD_MAX_MODELS",
    "FUGAL_HEAD_MAX_BYTES",
    "FUGAL_HEAD_MAX_DECOMPRESSED",
    "FUGAL_HEAD_MAX_MODEL_ID_LEN",
    "FUGAL_MODEL_PRICES",
    "FUGAL_CHAMPION_ACCURACY_VALUE",
    "FUGAL_CHAMPION_HARD_WEIGHT",
    "FUGAL_CHAMPION_ANCHOR_MODEL",
    "FUGAL_CHAMPION_MARGIN",
    "FUGAL_CHAMPION_CONFIDENCE",
    "FUGAL_CHAMPION_BOOTSTRAP_RESAMPLES",
    "FUGAL_CHAMPION_HARD_GATE_MIN_QUESTIONS",
    "FUGAL_CHAMPION_TIME_GATE_FACTOR",
    "FUGAL_CHAMPION_GHOST_OFF",
    "FUGAL_CHAMPION_GHOST_RETIRE",
    "FUGAL_CHAMPION_WINDOW_EPOCHS",
    "FUGAL_CHAMPION_MIN_ACCURACY",
    "FUGAL_CHAMPION_MIN_PAIRED_QUESTIONS",
    "FUGAL_CHAMPION_CONFIRM_QUESTIONS",
    "FUGAL_CHAMPION_CONFIRM_OPEN_EPOCHS",
    "FUGAL_CHAMPION_GHOST_RETIRE_CONFIRM",
    "FUGAL_PAYOUT_SEAT_SHARES",
    # Read by benchmarks/loader.py: both change which questions a neuron
    # scores (or whether the pool is verified at all), so they are refused on
    # finney and recorded in the digest's epoch block.
    "FUGAL_SKIP_BENCHMARKS",
    "FUGAL_POOL_UNPINNED",
    # Read by tee/runtime.py: the model upstream. A settable upstream is the
    # upstream-substitution exploit (INVARIANTS I8); the compose never sets
    # it and check_safety_invariants refuses a compose that does. Classified
    # consensus so a neuron with it set is refused on finney and visibly
    # different elsewhere.
    "FUGAL_OPENROUTER_BASE",
)
CONSENSUS_ENV_INPUTS = (
    "FUGAL_EPOCH_INTERVAL",
    "FUGAL_TEE_MEASUREMENTS",
    # Read by validator_setup: 0 makes FUGAL_TEE_MEASUREMENTS the only source
    # of approved entries (default: the release's frozen composes are added).
    "FUGAL_APPROVED_FROM_CODE",
    "FUGAL_V2_ACTIVATION_BLOCK",
    # Read off finney only: there the block is compiled in
    # (FINNEY_CONFIRM_ACTIVATION_BLOCK) and a different value is refused.
    "FUGAL_CONFIRM_ACTIVATION_BLOCK",
    "FUGAL_V3_ACTIVATION_BLOCK",
    # Off finney only: finney has no v4 block, and a set variable is refused there.
    "FUGAL_V4_ACTIVATION_BLOCK",
    "FUGAL_ODDS_SEAT_BLOCK",
    "FUGAL_MENU_B_BLOCK",
    "FUGAL_MENU_GRACE_EPOCHS",
)
OPERATIONAL_ENV_VARS = (
    "FUGAL_VALIDATOR_IP",
    "FUGAL_HARNESS_CONCURRENCY",
    "FUGAL_TAU",
    "FUGAL_LAMBDA",
    "FUGAL_TEE_PROXY_PORT",
    "FUGAL_PCCS_URL",
    "FUGAL_COLLATERAL_TIMEOUT",
    "FUGAL_COLLATERAL_EPOCH_BUDGET",
    "FUGAL_COLLATERAL_MAX_TIMEOUTS",
    "FUGAL_API_TIMEOUT",
    "FUGAL_API_RETRIES",
    "FUGAL_BACKBONE",
    "FUGAL_BACKBONE_BATCH_SIZE",
    "FUGAL_MIN_VALIDATOR_STAKE",
    "FUGAL_REVEAL_BUCKET",
    "FUGAL_REVEAL_DELAY_EPOCHS",
    "FUGAL_ROOT",
    "FUGAL_REFERENCE_BUCKET",
    "FUGAL_ROOT_HOTKEY",
    "FUGAL_MENU_MIRRORS",
    "FUGAL_PUBLISH_HEADS",
    "FUGAL_PUBLISH_HEADS_AFTER_EPOCHS",
    "FUGAL_HEAD_ARCHIVE_DIR",
    "FUGAL_ARCHIVE_ENDPOINTS",
    "FUGAL_FALLBACK_ENDPOINTS",
    "FUGAL_WEIGHTS_SET_RETRIES",
    # Read outside this module. Pool SOURCES are operational because the
    # loaded pool is verified against the pinned manifest whatever file it
    # came from (only FUGAL_POOL_UNPINNED, above, switches that off); the
    # rest are paths, ports, addresses, chain selection and offline tooling.
    "FUGAL_BENCHMARK_POOL",
    "FUGAL_POOL_FILE",
    "FUGAL_BENCHMARK_POOL_V2",
    "FUGAL_POOL_FILE_V2",
    "FUGAL_BENCHMARK_POOL_V3",
    "FUGAL_POOL_FILE_V3",
    "FUGAL_POOL_CACHE_V3",
    "FUGAL_FRAME_BOOTSTRAP",
    "FUGAL_EMBEDDINGS_FILE",
    "FUGAL_BACKBONE_THREADS",
    "FUGAL_EPOCH_DIR",
    "FUGAL_EPOCH_LOG_DIR",
    "FUGAL_STATE_PATH",
    "FUGAL_NETWORK",
    "FUGAL_NETUID",
    "FUGAL_WALLET_PATH",
    "FUGAL_MINER_PORT",
    # Protocol v4, read by neurons/miner.py: the operator's own proposal and
    # the ceiling on what it spends measuring it. A miner's own costs; the
    # compiled rules and every validator judge the filing itself.
    "FUGAL_V4_PROPOSAL",
    # Where the TD keeps its last accepted menu and measurement ledger, and
    # a menu file the operator hands in as a last-resort start: every menu
    # is checked against the chain and the compiled rules whatever its source.
    "FUGAL_V4_STATE_DIR",
    "FUGAL_V4_ANCHOR",
    # Whether the TD answers the scored questions (auto, on or off): a
    # miner's own spend; the proof says which it did and is judged on that.
    "FUGAL_V4_CHALLENGE",
    # The TD's dollar budget for one challenge: a miner's own spend; what it
    # leaves out is in the proof and judged by the rules every challenge is.
    "FUGAL_V4_CHALLENGE_USD",
    "FUGAL_AXON_IP",
    "FUGAL_PROVISIONED_WALLET_DIR",
    "FUGAL_EMBEDDING_CACHE",
    "FUGAL_POOL_CACHE_V2",
    "FUGAL_API_KEY",
    "FUGAL_ALLOW_ENV_DRIFT",
    "FUGAL_MIN_VALIDATORS",
    "FUGAL_SCORE_DIVERGENCE",
    "FUGAL_WEIGHT_DIVERGENCE",
)

# Protocol v5's environment variables, kept in their own lists: CONSENSUS_ENV_INPUTS
# and OPERATIONAL_ENV_VARS are registered constants of the v4 consensus section
# (consensus_v4.REGISTERED), so a name added to them would move the v4 digest
# that v0.7.0 validators publish before the v5 block. These lists are hashed by
# the v5 section only (consensus_v5) and classified like the two above
# (check_safety_invariants' env classification reads all four).
CONSENSUS_ENV_INPUTS_V5 = (
    # Off finney only, as v4's: finney compiles FINNEY_V5_ACTIVATION_BLOCK.
    "FUGAL_V5_ACTIVATION_BLOCK",
)
OPERATIONAL_ENV_VARS_V5 = (
    # The pool is verified against pool_manifest_v5's content hash and the
    # embeddings against EMBEDDINGS_V5_SHA256 whatever file or cache they came
    # from; FUGAL_ARENA_CACHE_DIR is only where a build downloads its sources.
    "FUGAL_BENCHMARK_POOL_V5",
    "FUGAL_POOL_FILE_V5",
    "FUGAL_POOL_CACHE_V5",
    "FUGAL_ARENA_CACHE_DIR",
    "FUGAL_EMBEDDINGS_V5_FILE",
)


# --- v5 ---
# Protocol v5 (~/fugal-v2-handoff/v5/DESIGN.md; docs/INVARIANTS.md, the v5
# entries): a head is scored on the value of its real picks the way
# RouterArena scores a router, S = accuracy - ln(mean cost per query) /
# k (arena_score.k_pair), paired against the holder and the best single entry on the same
# exam questions, with labels voted by other operators (arena_score.py,
# exam_v5.py). Nothing before the v5 block reads these.
#
# Activation: as v4's. Finney's block is compiled (None: no release schedules
# v5) and the variable may only repeat it there; another chain reads
# FUGAL_V5_ACTIVATION_BLOCK. Never before v4.
FINNEY_V5_ACTIVATION_BLOCK: int | None = 9587880
_v5_activation = os.getenv("FUGAL_V5_ACTIVATION_BLOCK", "").strip()
V5_ACTIVATION_BLOCK_ENV: int | None = None
V5_ACTIVATION_BLOCK_MALFORMED: str | None = None
_v5_digits = _v5_activation[1:] if _v5_activation.startswith("-") else _v5_activation
if _v5_digits.isascii() and _v5_digits.isdigit():
    V5_ACTIVATION_BLOCK_ENV = int(_v5_activation)
elif _v5_activation:
    V5_ACTIVATION_BLOCK_MALFORMED = _v5_activation
# RouterArena's exchange rate between accuracy and cost, per compared pair
# (arena_score.k_pair): Arena(a, c) = 100 (1 + b) a C / (b a + C), b = 0.1,
# C = log2(200 / c) / log2(200 / 0.0044) with c the cost per 1K queries
# clamped to [$0.0044, $200]; its local trade-off between accuracy and
# ln(cost) is k = C^2 ln2 log2(200 / 0.0044) / (b a^2). C is evaluated at the
# pair's logarithmic-mean pick cost (label-free), a at this reference
# accuracy (RouterArena's routers run at 0.79-0.86). Re-derived each release.
ARENA_A_REF_V5 = "82/100"
# One epoch's exam: at most this many questions, the first
# EXAM_BASELINE_QUESTIONS_V5 of them also labelled for the best single entry.
# Every exam question carries B*'s cell: the seat's bound over B* needs
# N_MIN_V5 paired questions, and at 100 a question an epoch no seat could fill
# for ~21 epochs after the first exam (about a day); at the full exam it fills
# in a few epochs, for about $0.2 more a hotkey-epoch.
EXAM_QUESTIONS_V5 = 500
EXAM_BASELINE_QUESTIONS_V5 = 500
# Distinct coldkeys that vote on each exam cell (one hotkey each).
EXAM_VOTES_V5 = 3
# Exam calls one verified hotkey makes per epoch (the budget that sizes the
# exam: cells x votes <= this x hotkeys), and its coverage calls.
EXAM_CALLS_PER_HOTKEY_V5 = 320
COVERAGE_CALLS_V5 = 48
# The seat's evidence window, minimum paired questions and the bound's n0.
WINDOW_EPOCHS_V5 = 48
N_MIN_V5 = 2000
N0_V5 = 2000
# Margins in S-points (exact Fractions): a challenger must beat the holder
# by TAKE_MARGIN_V5; a same-coldkey successor replaces its own holder unless
# it is SHOWN worse: the holder's bound over it passes -UPGRADE_MARGIN_V5
# (+1/4; never before N_MIN_V5 paired questions). The successor must also be
# payable (its bound over the best single entry passes 0).
TAKE_MARGIN_V5 = "1/4"
UPGRADE_MARGIN_V5 = "-1/4"
# The share of the miner emission v5 pays out (owner decision 2026-11-24):
# the seat vector (85/10/5 to the holder and history) is scaled by this, and
# the rest goes to the burn uid every epoch. 1/2 = miners keep half of their
# 41%; the other half burns.
MINER_PAID_FRACTION_V5 = "1/2"
# Contenders scored per epoch (the newest standing head per coldkey).
CONTENDERS_MAX_V5 = 24
# The spec of epoch E's exam is published in the root's reveal of E - this.
SPEC_LAG_V5 = 2
# A v5 head's own lam (the price a pick weighs cost at): in (0, this].
HEAD_LAM_MAX_V5 = 10000
# Labels are integers 0..Y_SCALE_V5 (RouterArena metrics may be fractional).
Y_SCALE_V5 = 10000
# A TD's exam/coverage calls: none starts later than EXAM_BUDGET_S_V5 seconds
# after the epoch's first call and none waits longer than
# EXAM_CALL_DEADLINE_S_V5, so the proof is ready before the validators'
# collection (boundary + half of a 360-block epoch, ~36 minutes). Operational
# for the TD; a call past either is recorded not_started or failed.
EXAM_BUDGET_S_V5 = 1200
EXAM_CALL_DEADLINE_S_V5 = 600
# B* (the best single full entry, by RouterArena's score on the window's
# coverage votes) is chosen only among entries with at least this many votes.
BASELINE_MIN_PAIRS_V5 = 100
# The exam draws only the pool's unpublished reserve (no exam question's
# labels are public when it is drawn: the memorisation rule, v5_epoch), and
# ceil(R / FRESH_RESERVE_EPOCHS_V5) of it an epoch (at most EXAM_QUESTIONS_V5),
# so the reserve shrinks by about 1/240 an epoch: a 60,000-question reserve
# gives 250 an epoch at first and keeps 37% of itself after 12 days. A release
# that adds questions raises it at once.
FRESH_RESERVE_EPOCHS_V5 = 240
# The v5 pool (benchmarks/arena_sets.py, pinned by pool_manifest_v5.json) is
# split once, by a keyed hash of each question id (arena_sets.in_coverage),
# into the exam part and the coverage part (this share of it). Exams draw only
# the exam part's unpublished reserve; coverage (the training labels every
# miner buys) draws only the coverage part and reuses it freely. Coverage
# labels therefore never shrink the exam's reserve and the memorisation rule
# needs nothing from them.
COVERAGE_PARTITION_V5 = "1/5"
# The v5 pool's routing embeddings: one row per question of route_text (the
# question without RouterArena's instruction template, arena_sets.route_text),
# built like the v3 file (one prompt per row, batch 1, one thread). 63,411
# rows x 1024; built 2026-11-24 on two CPU types, a 24-row sample recomputed
# bit for bit on a third. Published at the URL below; the hash is what is trusted.
EMBEDDINGS_V5_SHA256: str = "3b6cd118f0e4e597b792f2b04af0e08a14bd8e072b689a46d98d492b3b955b04"
EMBEDDINGS_V5_URL: str | None = (
    "https://storage.googleapis.com/fugal-reveals/embeddings/v5/embeddings-3b6cd118f0e4e597b792f2b04af0e08a14bd8e072b689a46d98d492b3b955b04.npz")


def v5_activation_block(genesis_hash: str | None) -> int | None:
    """The block protocol v5 applies from on this chain, or None. As
    v4_activation_block: finney's is compiled and the variable may only repeat
    it; another chain reads it; a malformed value is refused by name except on
    finney while no release compiles a v5 block. Never before v4."""
    malformed = V5_ACTIVATION_BLOCK_MALFORMED
    if malformed is not None:
        known_finney = (genesis_hash is not None
                        and str(genesis_hash).lower() == FINNEY_GENESIS_HASH.lower())
        if known_finney and FINNEY_V5_ACTIVATION_BLOCK is None:
            return None
        raise ValueError(f"FUGAL_V5_ACTIVATION_BLOCK={malformed!r} is not a block number "
                         "(digits only). Unset it or set the block.")
    block = _chain_block(genesis_hash, FINNEY_V5_ACTIVATION_BLOCK, V5_ACTIVATION_BLOCK_ENV,
                         "FUGAL_V5_ACTIVATION_BLOCK")
    if block is not None:
        v4 = v4_activation_block(genesis_hash)
        if v4 is None or block < v4:
            raise ValueError(f"the v5 activation block ({block}) must not be before the v4 "
                             f"activation block ({v4})")
    return block
