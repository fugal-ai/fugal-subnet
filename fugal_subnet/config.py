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
# the benchmarks in one place. Doing that safely is docs/CODE_BENCHMARK_PLAN.md.
HARNESS_ALLOW_EXEC = os.getenv("FUGAL_HARNESS_ALLOW_EXEC", "0") not in ("0", "", "false", "False")

# Checkers that grade by running the candidate's code, and therefore return 0
# unless HARNESS_ALLOW_EXEC. Named here rather than in graders.py, which is
# hash-pinned and must stay byte-identical.
EXECUTION_CHECKERS = frozenset({"exec_io", "exec_unittest"})

# --- Head constraints ---
HEAD_MAX_BYTES = int(os.getenv("FUGAL_HEAD_MAX_BYTES", str(1 * 1024 * 1024)))  # 1 MB
# Decompressed cap: a valid head is ~130 KB even at 64 models; 8 MB blocks zip bombs.
HEAD_MAX_DECOMPRESSED_BYTES = int(os.getenv("FUGAL_HEAD_MAX_DECOMPRESSED", str(8 * 1024 * 1024)))
HEAD_HIDDEN_DIM = int(os.getenv("FUGAL_HEAD_HIDDEN_DIM", "1024"))  # Qwen3-0.6B
HEAD_MAX_MODELS = int(os.getenv("FUGAL_HEAD_MAX_MODELS", "64"))
# Real OpenRouter IDs are well under 100 chars and save_head writes U100.
# Without a cap a miner can pad IDs to megabytes: the array still fits the
# decompressed budget, and the strings land verbatim in the published
# reveal artifact. Amplification, not RCE, but free to close.
HEAD_MAX_MODEL_ID_LEN = int(os.getenv("FUGAL_HEAD_MAX_MODEL_ID_LEN", "256"))

# --- Scoring ---
WILSON_CONFIDENCE = float(os.getenv("FUGAL_WILSON_CONFIDENCE", "0.95"))

# score = quality^w * thrift^(1-w), both measured against the best single model.
#
# A weighted GEOMETRIC mean, not a weighted sum. Additive terms substitute: a
# miner trades accuracy for cost at whatever rate the designer picked, and each
# degenerate strategy (all-cheapest, all-frontier) collects its own term's
# weight regardless of the other. Under a product, neither axis can rescue the
# other and both degenerates score badly.
#
# w is DERIVED from the product claim, not chosen. The claim is "match frontier
# quality at a fraction of the cost", so quality is a near-constraint rather
# than a freely tradeable axis: a router that gives up 40% of quality has not
# delivered the product, however cheap, and must not outscore simply matching
# the best model at the best model's price. That is
#
#     0.6^w * R^(1-w) < 1      =>      w > ln R / (ln R - ln 0.6)
#
# An unweighted sqrt (w=0.5) fails this at any interesting R.
#
# R IS THE COST RATIO, AND IT IS NOT 6. The original derivation solved this at
# R=6, the saving the product targets, and got w > 0.778. But R is not bounded
# by what the product targets — it is bounded by what the scoring function
# PERMITS, which is SCORE_THRIFT_CAP. Solving the same inequality at the cap:
#
#     w > ln 10 / (ln 10 + ln(1/0.6)) = 0.8184
#
# w=0.8 fails this. It scores the degraded-but-cheap router 1.0532 against the
# 1.000 of a full quality match — so the documented claim was already false at
# the old value, by a margin the 6x derivation hid.
#
# Two live runs produced it independently: a 63% router beating a 93% one at
# 13x cheaper, and a 46% router beating a 62% one. Neither was a corner case.
#
# WHY 0.9 AND NOT 0.8184. The claim above compares a miner to a fixed baseline,
# but weights are set by comparing miners to EACH OTHER, and a pairwise gap
# spans the whole band rather than half of it: one miner at the cap and another
# at the floor is a ratio of cap^2 = 100, needing
#
#     w > ln 100 / (ln 100 + ln(1/0.6)) = 0.9002
#
# 0.9 clears the absolute claim with room (0.795 vs 1.000) and the observed live
# case with room (that needed 0.8412), but sits 0.00015 BELOW the pairwise
# bound. It is the knife edge of that bound, not a margin above it — at exactly
# a 40% quality gap and exactly a 100x cost gap the two tie. Going to 0.91 buys
# that margin; going to 0.95 (which holds to ~16000x) would make the subnet
# nearly indifferent to cost and defeat the point of it.
#
# The alternative fix is to shrink the thrift band so R can never exceed 6 and
# the original 0.778 holds. Rejected: it would pay a miner who found a genuinely
# 10x-cheaper route no more than one who found 2.45x, discarding exactly the
# signal this subnet exists to find.
#
# NOTE ON READING THIS NUMBER. w does not set influence on its own; the
# exponent times the RANGE does. Quality is a ratio against the best model, so
# it realistically spans ~3x; thrift spans 100x. That is why w=0.8 read as
# "quality dominates 4:1" while quality and thrift actually contributed 2.41x
# and 2.51x of effective range — the cost term had marginally MORE pull than
# the accuracy term. tests/test_scoring_tradeoff.py pins both claims so this
# cannot silently regress if either the cap or the exponent moves.
SCORE_QUALITY_EXPONENT = float(os.getenv("FUGAL_SCORE_QUALITY_EXPONENT", "0.9"))

# Caps stop a degenerate running away with an unbounded ratio — a near-free
# model would otherwise drive thrift toward infinity. The thrift cap is well
# above the ~6x saving the product targets, so a genuinely frugal router is
# rewarded for all of its advantage rather than having it truncated; the
# quality exponent, not the cap, is what keeps cheap-and-wrong from winning.
#
# These two constants are coupled and must move together. The exponent is
# derived from the widest cost ratio the caps permit (see above), so RAISING
# SCORE_THRIFT_CAP WITHOUT RAISING THE EXPONENT reopens the gap where a
# cheap-and-wrong router wins.
SCORE_QUALITY_CAP = float(os.getenv("FUGAL_SCORE_QUALITY_CAP", "2.0"))
SCORE_THRIFT_CAP = float(os.getenv("FUGAL_SCORE_THRIFT_CAP", "10.0"))

# A fresh artifact's score ramps in over this many scored questions. This is
# what makes evidence reset symmetric: resetting clears accumulated PENALTIES
# as readily as accumulated credit, so without a ramp any miner could wash a
# bad record by flipping one weight bit. With it, climbing back costs exactly
# what earning the position cost in the first place.
BURN_IN_QUESTIONS = int(os.getenv("FUGAL_BURN_IN_QUESTIONS", "3000"))

# --- Soft targets ---
SOFT_TARGET_TAU = float(os.getenv("FUGAL_TAU", "1.0"))

# --- Dedup ---
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("FUGAL_DEDUP_THRESHOLD", "0.99"))

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
# field scale. Bounding the aggregate is tracked in docs/OPEN_WORK.md.
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
# Head capacity against pool size is the only real defence.
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
# The routing rule is argmax(softmax(W@h + b)) — no cost term, no exchange rate.
#
# The old rule was `p - lambda*cost`, which mixes a probability with dollars and
# so implicitly asserts what a correct answer is worth (lambda=2.0 asserted
# $0.50). The subnet has no business asserting that. It states the objective —
# quality per dollar against the best single model — and lets each miner's head
# discover its own tradeoff. Judging the outcome instead of dictating the rule
# removes the last hardcoded exchange rate from consensus.
#
# This remains available to miners as a TRAINING hyperparameter: a head still
# has to learn cost-awareness, it just is not handed the tradeoff.
TRAINING_COST_LAMBDA = float(os.getenv("FUGAL_LAMBDA", "2.0"))

# Routing utilities are quantized to this step before argmax picks a model.
# Deliberately NOT env-overridable: it is consensus-critical, and two validators
# using different quanta would disagree on every near-tie.
#
# Why it exists: the routing decision is argmax(softmax(W@h + b) - lam*costs),
# a discontinuity with no tolerance. Any float difference between two validators
# — a different BLAS kernel, CPU generation, or library build — flips the
# decision whenever two models' utilities are close, and across 300 questions
# and 30 models near-ties are certain. Quantizing turns "every validator must
# produce identical bits" into "every validator must agree to within 1e-4",
# which survives library and hardware changes. Exact ties then resolve by
# lowest index (numpy argmax), which is deterministic everywhere.
ROUTING_DECISION_QUANTUM = 1e-4

# --- API ---
API_TIMEOUT = int(os.getenv("FUGAL_API_TIMEOUT", "180"))
API_MAX_RETRIES = int(os.getenv("FUGAL_API_RETRIES", "3"))
API_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# --- Backbone ---
BACKBONE_MODEL = os.getenv("FUGAL_BACKBONE", "Qwen/Qwen3-0.6B")
# Consensus-critical: tokenizer padding is per-batch, so hidden states depend on
# how prompts are grouped. Two hosts using different batch sizes can produce
# different embeddings for the same question, which flips near-tie routing
# decisions. Pin it here rather than leaving it a call-site default.
BACKBONE_BATCH_SIZE = int(os.getenv("FUGAL_BACKBONE_BATCH_SIZE", "8"))
ROUTER_SYSTEM_PROMPT = (
    "You are a routing model. Given a question, your hidden state will be used to "
    "predict which language model can best answer it. Read the question carefully."
)

# --- Weight stability ---
MAX_WEIGHT_DELTA = float(os.getenv("FUGAL_MAX_WEIGHT_DELTA", "0.3"))

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
# Minimum stake (TAO) a querying hotkey needs when it lacks a validator permit.
# 0 admits any registered hotkey (safe default for testnets); raise on finney.
MIN_VALIDATOR_STAKE = float(os.getenv("FUGAL_MIN_VALIDATOR_STAKE", "0"))

# Grader execution limits live in graders.py itself (EXEC_TIMEOUT_SECS and the
# output cap): that file is hash-pinned consensus code and cannot read an
# env-overridable value without making every validator's grades depend on its
# environment.
