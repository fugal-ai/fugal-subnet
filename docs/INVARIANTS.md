# Consensus Invariants and Threat Model

This subnet is a mechanism-design problem with a distributed determinism
requirement. Almost every risk follows from two facts:

1. **Money flows from a score.** Every miner is motivated to find the cheapest
   path to a high score. Anything that raises a score without improving routing
   is an exploit, whether or not it is a "bug".
2. **Independent validators must agree.** Scores are computed on machines
   nobody controls centrally. If two honest validators disagree, weights split,
   the incentive signal degrades, and miners can farm the disagreement.

So the question to ask of any change is not "is this code correct?" but
**"what must be true, and what makes it false?"**

This file is the answer to the first half. It exists because a real
consensus bug (`I1`, below) survived five separate code reviews: the property
was never written down, so nobody checked it, and a line that had been correct
for months silently contradicted a newly added claim in the README. Reviews
sample. Written invariants with executable checks hold.

**Adding a consensus-affecting change means adding or updating an invariant
here, and a check that enforces it.**

## The invariants

| | Invariant | Enforced by |
|---|---|---|
| **I1** | **Determinism.** Same epoch inputs ⟹ byte-identical scores on any honest validator. | `scripts/check_determinism.py` (both modes, in CI — 13 stages covering the live path), `fugal_subnet/determinism.py`, `check_safety_invariants.check_epoch_id_single_source`, `slicer.protocol_version_for_boundary` (`tests/test_v2_activation.py`), TEE proof verification (all validators verify the same attested proof) |
| **I2** | **Bounded ingestion.** No miner-supplied bytes reach deserialization, allocation, or execution without size, shape, and value bounds. | `run_miner_attacks.py`, `tests/test_head_properties.py`, `check_safety_invariants.py` (no-pickle) |
| **I3** | **Monotonic incentive.** A miner cannot raise its score except by routing better or more cheaply, and **no policy that does not read the question can score above zero**. Artifact-keyed evidence with miss=0 prevents selective publication; the burn-in ramp prevents penalty-washing by reset. | Constant-policy frontier (`frontier.py`, `tests/test_frontier.py`, `tests/test_degenerate_constant_policy.py`), the v2 copy rule (`copy_rule.py`, `tests/test_copy_rule.py`), the v2 champion rule (`champion.py`, `tests/test_champion.py`; constant policies are no longer zero by construction in v2, see its entry), commit-reveal, behavioural dedup (global model index), evidence accumulation, `run_attacks.py`, `run_tee_attacks.py` |
| **I4** | **Non-interference.** A miner cannot lower another miner's score, prevent them being scored, or move the reference they are scored against — including by being present or absent. | `tests/test_non_interference.py`, TEE architecture (no shared model pool), nonce-derived exploration targets, reference frame pooled over time, coldkey-median shared grades from v2 (`tests/test_shared_grades.py`) |
| **I5** | **Bounded spend.** No miner behavior can make a validator exceed its budget. Validators verify proofs — zero inference cost. | TEE architecture (miners pay their own inference), `tests/test_paid_safety.py` |
| **I6** | **Liveness.** No miner behavior **and no external dependency** can stop a validator completing an epoch and setting weights, and the validator collects at a point where proofs can exist. Every call that leaves the process on the epoch path is bounded. | `run_miner_attacks.py`, property test P1, TEE proof timeout, `slicer.collect_block_for_epoch`, `tests/test_collection_point.py`, `tests/test_collateral_timeout.py`, `check_safety_invariants.check_external_calls_bounded` |
| **I7** | **Auditability.** Any divergence between two validators is diagnosable after the fact from published artifacts. | `fugal_subnet/fingerprint.py`, `environment` block in every `reveal.json` |
| **I8** | **TEE integrity.** Every claim a proof makes is bound to something the miner cannot forge: the hardware's own measurement registers, or a hash chain rooted in the attestation. No miner-influenced code executes inside the enclave that produces the proof. | `fugal_subnet/tee/verify.py`, `attestation.measurement_id`, `run_tee_attacks.py` (32 cases, in CI), `check_tee_safety`, `config.HARNESS_ALLOW_EXEC`, `tests/test_grader_policy.py` |
| **I9** | **Reference-frame agreement.** Every validator derives the same reference frame from the same published exploration samples, and no single miner can materially move it. | `fugal_subnet/reference_frame.py` (order-independent accumulation), `check_determinism.py` `frame` stage, `tests/test_non_interference.py` |

## How TEE resolves prior gaps

### I3 — RESOLVED 2026-09-08: the score is headroom above the constant-policy frontier

**Status: decided (`docs/decisions/I3_DECISION.md`, option 3) and implemented.** The
score is no longer `quality^0.9 * thrift^0.1` against the best single model.
It is the accuracy a miner achieves **above what any non-routing policy buys at
the miner's own cost**, where the reference is the upper convex hull of the
per-model (cost, accuracy) points from the reference frame plus the origin
(`fugal_subnet/frontier.py`, `fugal_subnet/scoring.py`). Every constant
policy — one model, or a random mixture of models — has zero headroom by
construction; a router earns exactly the value of reading the question. A
quality floor (`SCORE_QUALITY_FLOOR`, 0.8 of the frontier's maximum accuracy)
keeps "match frontier quality" a requirement rather than a suggestion, and the
frontier's own confidence (`FRONTIER_MIN_TRIALS` decayed exploration trials on
every hull model) scales scores so that a cold frame pays nobody rather than
paying constant policies for the frame's ignorance. Until the frontier is
trusted, the whole weight vector burns to UID 0 — deliberately.

Enforced by `tests/test_frontier.py` (every constant policy and every mixture
scores zero; a router above the hull scores its headroom; the frontier is
unmoved by other miners; two validators build identical frontiers; a cold
frame pays nothing; skipping epochs does not make a miner look cheaper) and
`tests/test_degenerate_constant_policy.py` (the measured finding, pinned in
the form "constant policies score zero, a trained router beats them all").
The unconfident share of emission burns in `compute_weights(paid_fraction=frontier.confidence)`, after normalising — inside `composite` the factor cancels (measured on the first live frontier epoch; `test_the_unconfident_share_of_emission_burns_after_normalisation`).

### I3 — the frontier is valued at the raw mean, not the prior (fixed 2026-10-14)

The 2026-10-14 diligence review found the frontier paying constant policies
at launch, and the repo's own code reproduces it. The frame valued every model
at its posterior mean under a neutral 0.5 prior worth 20 samples; after 50
real samples a 90% model read as 78.6%, which was exactly where
`FRONTIER_MIN_TRIALS` (then 50) let the frontier pay headroom in full. A head
that always calls that model was scored at its true 90% against a frontier
that believed 78.6% was the best constant policy, and collected the
difference. Duplicate exploration answers counted as fresh samples, so thirty
miners reached full payout in a few epochs on two distinct questions per
model. And `test_every_constant_policy_scores_zero` scored each policy at the
frame's *own* estimate of its accuracy, which is zero against a frontier built
from that estimate whatever the estimate is — a test that could not fail.

| samples per model | top model, frame estimate | best constant-policy score |
|---|---|---|
| 50 | 0.786 | 0.109 |
| 200 | 0.864 | 0.031 |
| 650 | 0.888 | 0.006 |
| 2000 | 0.896 | 0.000 |

Three changes:

1. **Valuation uses the raw sample mean**, `ReferenceFrame.frontier_accuracy`,
   with no prior. The miner is judged on its lower bound, so a constant policy
   at its true accuracy sits on the frontier and its lower bound below it:
   zero headroom at every sample count, by construction. An unmeasured model
   is valued at 1.0 — nothing is known, nothing is paid above it. An upper
   confidence bound was tried first and rejected: its width (four points at
   500 samples) taxed every honest router for a certainty the miner-side
   lower bound already provides. The prior still serves `accuracy` and
   `best_model`, which select and report; it no longer values.
2. **`FRONTIER_MIN_TRIALS` is 300 distinct observations**, where a trial is
   one (question, model) per epoch however many miners answered (I9). The
   ramp is a count of what has been measured and cannot be filled by
   duplicates.
3. **A live validator starts from the shipped bootstrap frame**,
   `data/reference_frame.json`, hash-pinned in `check_safety_invariants.py`
   (`check_bootstrap_frame_pinned`). It is measured evidence, not a prior:
   exploration counts, valued exactly as live samples are. Without it
   epoch 1 pays nobody and full payout arrives after roughly 340 epochs at
   fifteen exploration slots over seventeen models. `load_bootstrap(require=
   True)` makes a missing file a startup error in live, because one validator
   starting cold is one validator scoring the field against a different
   frontier (I1). Two builders produce the same structure: `scripts/
   build_bootstrap_frame.py` replays published reveals (attested samples), and
   `scripts/build_frame_from_calls.py` makes the exploration calls directly,
   through the miner's own proxy, prompts and graders, and commits the call log
   beside the frame (re-gradeable, not attested). **The 2026-11-03 launch frame
   is the second kind**: 17 models × 300 nonce-stratified questions,
   `data/observations/bootstrap_frame_calls_v1.jsonl`, chosen so the subnet
   could launch with a full-confidence frontier in a day instead of waiting two
   days of chain epochs. The trade is stated in the script's docstring and
   here: the bootstrap's provenance is Fugal's logged calls, which anyone can
   re-issue and re-grade, rather than TEE quotes. Live exploration folds in from
   epoch 1 and the bootstrap decays out at `FRAME_HALF_LIFE`.

Enforced by `tests/test_finding3_prior_bias.py` and the rewritten
`test_every_constant_policy_scores_zero`, both of which score constant policies
at their TRUE accuracy against frames built from exact-count samples at
several sizes.

What follows is the analysis as it stood when the defect was open, kept
because it is the reason the reference is a curve and not a point.

**Status when found: measured, unfixed, and a product decision rather than a tuning one.**

I3 says a miner cannot raise its score except by routing better or more
cheaply. On the shipped constants that holds in the letter and fails in
substance. Measured on 5,000 held-out questions with the shipped trainer and
the shipped routing rule:

| strategy | score |
|---|---|
| **always gpt-4o-mini** — `W=0` plus a one-hot bias | **1.183** |
| trained head (SFT) | 1.058 |
| trained head (+CMA) | 1.060 |
| per-benchmark lookup | 1.036 |
| best single model *(the reference)* | 0.985 |
| random head | 0.868 |
| always cheapest | 0.479 |

**The head is not the problem.** It genuinely routes — 13 models used,
within-source top share 0.45 where a benchmark lookup would be 1.00 by
construction — and it beats chance by 0.24, the reference by 0.07, and a domain
lookup. On MATH it reaches 0.895 against best-single's 0.775 by routing to a
model 12x cheaper and better at maths. That is the product working. The problem
is that a head which never reads the question beats it by 12%, costs nothing to
build, and any miner finds it on day one.

**Why the existing defence misses it.** `scoring.py` argues the geometric mean
stops degenerates because "neither axis can rescue the other". That was tested
at the ENDS of the price range and holds there — always-cheapest scores 0.479.
The winner is in the MIDDLE. `config.py` derives `w` from "a router that gives
up **40%** of quality must not outscore a quality match"; this one gives up
**5%** and wins. The constraint was applied to the wrong corner.

**It is not an artefact of the dataset it was found on.** The break-even is a
property of the scoring constants alone:

    acc_k / acc_best  >  thrift ^ (-(1-w)/w)

At `w=0.9` and `SCORE_THRIFT_CAP=10`, a constant policy beats the reference on
**77.4%** of its accuracy and beats the trained head on **82.3%**. On
`data/models.json` the 17 pinned models span **168x** in cost and **8 of 17**
sit at or beyond the thrift cap. Establishing whether any pinned model clears
82.3% costs ~$688 of real inference and has not been run — so this is treated
as LIVE on the pinned table.

**Three resolutions, and they are not the same kind of change:**

1. **Re-tune** (exponent, thrift cap). Cheapest to act on, and constrained:
   the pairwise bound already forces `w > 0.9002`, the cap is *already* binding
   at 10 against gpt-4o-mini's raw 14.7 and still leaves the exploit
   profitable, and pushing `w` toward 0.96 leaves the subnet nearly
   indifferent to cost — the thing it exists to measure.
2. **Accept it.** The subnet's claim is quality per dollar, and by that measure
   a cheap-and-decent model honestly *is* better value than a router costing
   9x more for 5% more accuracy. If so, routing is not worth doing on this
   model set, and no exponent hides that.
3. **Change the reference.** Scoring against the best *single* model hands a
   cheap model a large thrift ratio for free. A cost-matched frontier prices it
   differently. This is a consensus redesign.

Recorded here rather than patched because (1) is the cheapest to reach for and
the analysis does not support it being obviously right.

`tests/test_degenerate_constant_policy.py` pins the break-even so the constants
cannot move without meeting this. **The check that would have caught it
originally is 13 evaluations** — assert a trained head beats every constant
policy — and it would have failed on the commit that moved `w` to 0.9.

### I1 — two accuracy definitions exist, and only one is consensus

They differ by about **7 points**, and both are defensible:

| | Denominator | Where |
|---|---|---|
| `proof.accuracy` | every scored question | `proof.py` — **this is what scoring reads**, via `_proof_to_head_score` |
| `head_eval.evaluate_head` | scored questions *minus* those no model in the pool answered correctly | `head_eval.py:203` — "carry no routing signal" |

Excluding unanswerable questions is reasonable for measuring routing skill:
nobody could have got them right, so they say nothing about the router. Keeping
them is reasonable for scoring: it is the miner's realised accuracy on the slice
it was given. The problem is not that either is wrong. It is that **two exist
and one is called the obvious name.**

`evaluate_head` is not on the consensus path — verified, it has zero callers
under `fugal_subnet/` or `neurons/`, and every reference is in tests, scripts or
comments. So there is no bug in the shipped subnet. What there is, is a trap:
it is the function you would reach for to reason about head scoring, and it is
not the scoring function. Someone did reach for it and got a reference model
scoring **1.065 against itself** — impossible under the shipped definitions,
where the reference's thrift is exactly 1 and its quality cannot exceed 1.

`check_one_accuracy_definition_on_the_consensus_path` fails CI if anything under
`fugal_subnet/` or `neurons/` imports or calls it. A docstring was considered and
rejected: docstrings are read by people already looking at the function, and the
failure mode is someone grepping for a scoring function and finding one with the
right name. Wiring it in would move every miner's apparent accuracy by ~7 points
while reading as a refactor.

The invariant is not "use this function". It is that **every validator divides
by the same denominator**, and the denominator comes from the proof.

### I1 — matrix agreement (resolved by architecture)

**Previous gap:** Two validators calling the same model on the same question
get different responses. The matrix diverges, scores diverge, validators
disagree. LLM APIs are not deterministic even at temperature 0.

**Resolution:** With TEE, validators no longer compute their own matrices.
Miners run benchmarks inside Intel TDX confidential VMs and produce
hardware-attested proofs. All validators verify the same attested proof, so
they agree by construction. The I1 gap is closed.

**Measured across architectures.** The whole validator scoring path — slice,
exploration, routing decisions, proof, verification, reference frame, scores,
weights — was run on x86_64 (AVX2) and on aarch64 (Ampere) and produced
byte-identical digests at all nine stages. That is the property consensus
actually rests on, and it is now evidence rather than assertion.

The backbone is a separate question and a weaker one. `check_determinism.py`
feeds the pipeline `rng.randn` embeddings, not Qwen3-0.6B — the stage is named
`synthetic_hidden` for that reason. Measured directly, the real forward pass
differs between x86_64 and aarch64 in the last bits of float32 (~2e-9 per
element). This does **not** fork consensus:

- Validators never run the backbone. Only miners do, and only for their own
  routing; a validator recomputes nothing float-dependent from the head.
- Under `--live` the architecture is pinned by the approved TDX measurement,
  and TDX is Intel-only, so every attested miner is x86_64 by construction.

The residual is that two miners running an identical head on different
architectures could in principle diverge on a routing decision whose top two
logits are within 1e-9, which would make a copied head marginally harder to
deduplicate. Under `--live` that path does not exist. It is recorded here
rather than fixed because the fix — pinning a cross-architecture backbone — buys
nothing that the measurement does not already buy.

### I6/I9 — when a validator collects proofs is consensus state

A miner cannot begin benchmarking until its epoch's boundary block exists,
because that block's hash is what selects the questions — hiding the slice
until that moment is the whole anti-overfitting design. There is therefore an
unavoidable gap between an epoch starting and any proof existing for it.

A validator that queries at the boundary asks before anyone can answer. With
real models on a 300-question slice the miner needs minutes; the validator
collects nothing, and then marks the epoch processed and never returns to it.
Every epoch, forever, logged as `no valid proofs` as though the miners were at
fault. That is a liveness failure (I6) wearing the costume of a miner problem.

Retrying until proofs appear would fix the symptom and break I1. How long a
validator happened to wait would decide which miners it scored, so two honest
validators would grade different fields and publish different weights — a
divergence with no bug behind it and nothing in the artifacts to explain it.

So the collection point is derived, not chosen: `collect_block_for_epoch`
returns `boundary + fraction × blocks_per_epoch`, clamped strictly inside the
epoch, and it is the only place that computes it (`tests/test_collection_point.py`
asserts that structurally, because a test that derives the value once and hands
it to both sides cannot catch a second implementation — which is exactly how
the epoch id came to be formatted two different ways). The metagraph is read at
that same block, so every validator queries the same axons at the same chain
state. `EPOCH_COLLECT_FRACTION` is in the consensus digest, so a validator
collecting at a different offset diverges visibly instead of silently.

Half the epoch is the default: miners get half to benchmark, validators get
half to verify, score, set weights and publish the reveal.

### I4 — pool manipulation (resolved by architecture)

**Previous vulnerability:** Sybil registrations declaring cheap models could
evict a victim's models from the shared pool, zeroing the victim's accuracy.

**Resolution:** With TEE, each miner runs their own benchmark inside their
own TEE VM. There is no shared model pool to manipulate. The attack is
eliminated by architecture, not by a code fix.

### I5 — cost asymmetry (resolved by architecture)

**Previous concern:** A $1 miner registration could waste $30+ of validator
inference per epoch. The validator computing the matrix was the wrong
architecture — every other Bittensor subnet has miners pay for expensive work.

**Resolution:** Validators verify proofs, never call models. Zero validator
inference cost. Miners pay for their own API calls inside the TEE, metered by
the attested MeteringProxy.

### I8 — image identity and code identity are two claims, and only one is in the measurement

The two are routinely collapsed, in this repo's own documentation twice, in
opposite directions at the same time. `docs/MINER_GUIDE.md` simultaneously
over-claimed what the base measurement proves ("running a modified harness
produces an unapproved measurement" — it does not) and mis-stated which
registers compose it ("MRTD, RTMR0-2" — RTMR0 is excluded). One document, one
section apart, wrong both ways. That is not carelessness so much as a sign that
the distinction was never named as a pair, so there was nothing to be
consistent with.

Named as a pair:

| Claim | What proves it | What it does NOT prove |
|---|---|---|
| **Image identity** — which OS image booted | `measurement_id` = `sha256(MRTD ‖ RTMR1 ‖ RTMR2)`, filled by the CPU and covered by the Intel signature | Which Fugal code ran. On a stock VM the repo is git-cloned onto an unmeasured filesystem; **measured directly**, editing `harness.py` left the value byte-identical |
| **Code identity** — which application ran | The `compose-hash` event replayed out of RTMR3 and checked at `verify.py:366`, which covers the image the compose names | Anything, unless the image is pinned by digest — a moving tag leaves the hash stable while the code beneath it changes |

Three consequences follow, and each has previously been rediscovered rather
than read:

- **RTMR0 is excluded and its exclusion is load-bearing.** It records the TDVF
  configuration the host builds — CPU count, memory size, device config — which
  the cloud provider chooses, not us. Including it would fork the approved list
  by instance size while proving nothing about the code. This is why miners may
  pick any machine shape, and a document that lists RTMR0 as measured silently
  revokes that permission.
- **RTMR3 is excluded from `measurement_id` but is not ignorable.** It carries
  code identity. Excluding it from the *image* identity is correct; treating it
  as unused is the error that would make dstack adoption buy nothing.
- **`--live` requires dstack for this reason and no other.** Not because a
  stock TDX VM cannot produce a genuine quote — it can — but because nothing on
  it binds the code to the quote.

The general form, which is the reason this is here rather than only in the
guide: **a measurement is evidence for exactly the claim its inputs cover, and
the inputs are not in the sentence.** "Attested" reads as covering whatever the
reader had in mind. Whenever this document or any other asserts that something
is attested, it should name the registers, because two readings that differ by
one register differ by the whole security property.

`check_measured_registers_documented` enforces the membership half. It derives
the expected register set from `measurement_id`'s own body rather than
hardcoding it, so changing the code to include RTMR3 for dstack fails the check
until the documentation follows — which is the direction the failure should
point. Falsified against both the sentence this audit removed and a plausible future
drift naming RTMR3 in place of RTMR2; it fires on each and passes clean.

Note the shape of its one limitation, discovered by it firing on the sentence
above when that drift was written out in full: **the check cannot tell an
assertion from an illustration.** A membership claim quoted as an example of
what NOT to write reads identically to the claim itself. That is the same
precision-over-recall trade as the `s.get` false positive, and the cost is
borne here rather than by loosening the pattern — prose that needs to discuss a
wrong register set should describe it rather than format it as a claim.

### I8 — BROKEN AS IMPLEMENTED: the measurement does not cover the miner's code

**`measurement_id` proves which operating system booted. It does not prove which
Fugal code ran.** Measured directly on a live TD: a line was appended to
`fugal_subnet/tee/harness.py` and the TD re-measured.

    before   a68d0ccd3473a6c44fa9689cfdcf22066bc83e232efeaf33cc0341d0c532ae65
    after    a68d0ccd3473a6c44fa9689cfdcf22066bc83e232efeaf33cc0341d0c532ae65

Identical. **Those two values are stale and must not be copied into an approved
list.** They were computed while `measurement_id` still included RTMR0. The same
image, same machine shape, now measures
`a1ecb6273d38bd8b5ad629edf12f48d1c3b861fa8f506d1e1e303783e68461f3`. They are
kept here because they are what the experiment produced, and the finding is
unaffected either way: editing the harness moves neither the old value nor the
new one.

`measurement_id` is sha256 over MRTD, RTMR1 and RTMR2 — the image, the kernel,
the cmdline and initrd — and `attestation.measurement_id` excludes RTMR3
deliberately, because RTMR3 is application-extendable and including runtime data
would mean no image could stay on an approved list. That reasoning is sound and
its consequence is fatal: the repo is git-cloned onto an **unmeasured
filesystem** at runtime, and nothing measures it.

So a miner can edit the line in `harness.py` that records results, inside a
genuine TD running a genuinely approved image, and produce a proof that passes
every check in `verify.py`. The chain this module documents:

    measurement_id(quote) in approved_measurements
      -> the image that ran is the published one          <- OS only
    report_data == proof.content_hash()
      -> the proof body is exactly what that image produced  <- FALSE

The second claim does not hold. It is "exactly what *something running on that
OS* produced". `verify.py` correctly rejects `proof.source_hash` as
self-declared and names the exact attack — *"DCAP verification passes for an
attacker who simply runs modified code on real hardware"* — then defends against
it with a measurement that does not include the code. The right kind of defence,
over the wrong bytes.

**What the real-hardware negative control actually proved.** A second VM on the
same machine type but a **different OS image**
(`ubuntu-2204-jammy-v20260826`, measurement `498288bb2464…`) produced a genuine
Intel-signed quote that passed DCAP and was still rejected as an unapproved
runtime image. That is real and worth having: a different boot chain is refused.
It is *not* what an earlier revision of this document claimed — that modified
*code* is refused. Modified code on an approved image is accepted.

**What would fix it.** The code has to be inside the measured boot chain:
baked into a purpose-built guest image so it lands in MRTD/RTMR1/RTMR2, or on a
dm-verity volume whose roothash sits on the kernel command line, which the
bootloader measures. Extending RTMR3 *from the application itself* cannot work —
an attacker who controls the code controls what it extends.

**RTMR3 is not categorically useless, and this matters for the dstack decision.**
`measurement_id()` excludes RTMR3 with this reasoning: it is the
application-extendable register, so including runtime data would mean no image
could ever stay on an approved list. That is correct for a register extended by
the workload at arbitrary times with arbitrary values. It is **wrong** for a
register extended once, at boot, by measured launch code, with a deterministic
hash of the application — which is exactly what dstack does, extending RTMR3
with the app/compose hash before the workload runs.

So adopting dstack **without changing `measurement_id()` would reproduce this
exact gap at the cost of weeks**: a better-built base image, still not binding
the miner's code, because the register that binds it is the one being thrown
away. Any decision to adopt dstack must include the consensus change to
`attestation.py` that brings RTMR3 into the identity — deliberately, with the
distinction above written down, not discovered afterwards.

The general form, worth keeping: *an exclusion justified by an assumption
survives the assumption's death unless someone re-reads the justification.*

### I8 — two changes made now, ahead of the image work

Neither closes the gap above. Both are prerequisites for closing it, and both
are cheap enough that deferring them to the image migration only means doing
them under more pressure.

**1. RTMR0 is no longer part of `measurement_id()`.** It records the TDVF
configuration the host builds — virtual hardware setup, CPU count, memory size,
device layout — which the cloud provider chooses and we do not. It is not a
per-image discriminator: the same pinned image on two machine shapes produces
two identities, so an approved list forks by instance size while proving
nothing about the code. Measured on live TDX, and dstack documents the same
behaviour. Thirty Spokes (Bittensor SN99), whose attestation patterns this
module was forked from, excludes RTMR0 for exactly this reason; the fork kept
RTMR0 and dropped RTMR3, which is backwards on both counts.

The identity is now `sha256(MRTD || RTMR1 || RTMR2)` — the image, the kernel,
and the cmdline/initrd. **Every previously published measurement value changes.**
Nothing is live on the old values, so there is no migration; a `--live` subnet
would have needed a coordinated approved-list update.

Checks: `tests/test_tee.py::test_measurement_id_ignores_rtmr0_machine_shape`,
`::test_measurement_id_ignores_rtmr3_application_register`,
`::test_measurement_id_tracks_the_boot_chain`.

**2. The runtime now extends RTMR3 at startup, and this is ADVISORY.**
`attestation.runtime_identity()` is `sha384(source_hash || pool_hash ||
grader_hash)` — the three things that decide what a proof means — and
`extend_rtmr3()` writes it through the kernel's tsm-mr interface
(`/sys/class/misc/tdx_guest/mr/rtmr3`) before any proof exists.

It proves nothing today, for the reason stated above: an attacker who controls
the code controls what it extends. It is written anyway because when the image
is locked, the same value is extended from a measured initrd instead of from
userspace, and at that point it becomes evidence without a new mechanism. The
value is deliberately free of per-epoch data — no nonce, no slice, no results —
so it is fixed for a deployment and computable off-hardware from the repo,
which makes an approved list of it reviewable in a pull request rather than
requiring someone to hold a quote.

`extend_rtmr3()` returns False rather than raising when the interface is absent,
and the miner logs a warning naming what is not bound. A silent success would be
indistinguishable from a binding that never happened — the same failure shape as
the humaneval zero.

**Verified on hardware**, and the verification found a bug that testing could
not. On a live `c3-standard-4` TD (6.17.0-1022-gcp):

- The interface is at
  `/sys/class/misc/tdx_guest/measurements/rtmr3:sha384`, **not**
  `/sys/class/misc/tdx_guest/mr/rtmr3` as first written. The directory is
  `measurements` and the register name carries its hash algorithm. With the
  wrong path `extend_rtmr3` returned False forever and logged *"this kernel may
  predate the tsm-mr interface"* — a wrong diagnosis of a path typo,
  indistinguishable from the feature genuinely being absent. Unit tests passed
  throughout, because a missing file is exactly what they assert on a
  non-TDX host.
- The write succeeds, and the new RTMR3 **appears in the next Intel-signed
  quote** — sysfs value and quote value identical.
- `measurement_id` is unchanged by the extend, as designed.

**Removed 2026-11-08.** The userspace extend never became evidence: the image
binding went to the compose hash in dstack's RTMR3 log instead. On the live
compose the miner container cannot write `/sys` (Docker mounts it read-only),
so the extend failed and logged a misleading warning at every start; on a
compose that could write it, the extra extend would be missing from dstack's
log and every honest proof would stop replaying. `neurons/miner.py` no longer
extends RTMR3, `extend_rtmr3` is gone, and
`tests/test_tee.py::test_nothing_writes_rtmr3_from_userspace` keeps any
shipped code from writing a measurement register.

### I8 — verified on hardware: the upstream binding, and a fresh TD

The upstream is genuinely bound, measured rather than argued. Two boots of one
c3-standard-4, everything held constant except `FUGAL_OPENROUTER_BASE`:

| upstream | RTMR3 after one extend |
|---|---|
| `https://openrouter.ai/api/v1` | `9c16a3b5987a572b…` |
| `http://127.0.0.1:8799` | `61f579c30c23d544…` |

In both cases `expected_rtmr3(identity)` reproduced the value inside the
Intel-signed quote exactly. A miner who repoints the upstream at a server of
their own can no longer present the same runtime identity as an honest one.
Advisory until the image is locked, like everything in this register — but the
mechanism is measured, not asserted.

**A fresh TD starts from zero.** A GCP confidential VM stop/start produces a new
Trust Domain with RTMR3 cleared — confirmed on both boots above. So a miner
cannot accumulate extends across restarts to walk the register toward a chosen
value: the chain restarts from 48 zero bytes every boot. That closes the obvious
question about an advisory, userspace extend.

### I8 — the TPM half on GCP, and why it is not a service call

On GCP, dstack's attestation is a distinct variant: a TDX quote **and** a TPM
quote. Verifying only the TDX half is not an option, and if some validators
check the TPM half while others do not they disagree on validity — a consensus
fork with no bug behind it. It has to be one rule for everyone.

`tpm-qvl` does not exist on PyPI, so the rule has to be built. Three paths were
examined and the middle one is a trap:

| | |
|---|---|
| **`cryptography`, hand-rolled — CHOSEN** | Parse `TPMS_ATTEST`, verify the quote signature against the vTPM attestation key, chain the AK certificate to Google's root. ~200 lines, deterministic, no native dependency, no external service, pinnable and auditable the way `graders.py` is |
| **`google-cloud-confidentialcomputing` — REJECTED** | Google's Attestation Verifier: hand it the attestation, receive a signed verdict |
| `tpm2-pytss` | CFFI over `libtss2`. Every validator would need the native library installed, to marshal structures we can parse ourselves. Fine as a cross-check, poor as the dependency |

**Why the service call is rejected**, since it is the easy option and looks
harmless: it would make Google a **trust root and a per-proof availability
dependency for every validator**, which is the exact property that ruled out
GCP Confidential Space when the stack was chosen. It is strictly worse than the
Intel PCS dependency already accepted — PCS serves *static collateral* every
validator can cache, whereas a per-proof verdict from a live service can in
principle differ between callers or over time, and two validators disagreeing on
validity is a consensus fork. A subnet whose notion of a valid proof is "what
Google's API said this morning" is not verifying anything.

**Budget it as real work.** ~200 lines of security-critical binary parsing plus
certificate-chain validation, consensus-affecting, and it needs the same pinning
and attack-suite discipline as `dcap-qvl`. It is not a dependency bump.

### I8 — built: the TPM half, and the second trust root it adds

`fugal_subnet/tee/tpm.py`, verified against two real dstack attestations from
two different GCP Confidential VMs (`tests/fixtures/attestation_{A,B}.bin`).
Both real ECDSA P-256 signatures verify; a single flipped bit in the quote
fails; one machine's quote presented under the other's certificate fails.

**This adds Google's CA as a trust root, alongside Intel's.** Stated plainly
because it is a change to the threat model, not an implementation detail: the
subnet now believes a TPM quote because a certificate chains to
`CN=EK/AK CA Root, O=Google LLC`. Intel attests the TD; Google attests that the
machine underneath is a genuine Confidential VM. Neither replaces the other and
a validator must check both — checking one and not the other is a fork.

**The chain is pinned, never fetched.** The AK certificate's AIA extension
points at `http://privateca-content-<uuid>.storage.googleapis.com/.../ca.crt`.
Fetching it per proof is refused on three grounds, in increasing order of
seriousness: it is plain HTTP; it is a live availability dependency for every
validator; and two validators fetching at different moments can get different
answers, which is a consensus fork with no bug behind it — the same objection
that ruled out Google's Attestation Verifier service. The root is vendored at
`fugal_subnet/tee/roots/google_ek_ak_root.der`, hash-pinned in **two** places
that `check_safety_invariants.py` forces to agree (`tpm._ROOT_SHA256` and
`PINNED_TPM_ROOT_SHA256`). A substituted anchor does not fail loudly on its
own — it just starts accepting a different CA's attestations — so it gets the
grader treatment.

**Intermediates rotate, so they are not the anchor.** The observed intermediate
was issued 2026-05-19 and lives at a per-CA-instance URL that cannot be
enumerated in advance; pinning it would reject miners the day Google rotates,
and pinning a per-region set would reject any region not yet vendored. Instead
a proof carries its own intermediates in `BenchmarkProof.tpm_intermediates`
(DER), which `verify_proof` passes through `unwrap_attestation` to
`verify_ak_chain(..., extra_certs=...)`, exactly as TLS does: a supplied
certificate is worthless unless it chains to the pinned root by signature
(`run_tee_attacks`: a rogue CA named as Google's intermediate, and one named as
the root, are both refused). Determinism is preserved because the bytes are in
the proof rather than on the network, and so no validator's verdict depends on
when it asked. The field is **outside** `content_hash`: it is untrusted input
checked against the anchor, so forging with it is impossible, and keeping it
out changed no `report_data`, no `proof_hash` and no existing proof.

**The gap that was here is closed for v2 proofs (2026-11-08).** Outside
`content_hash`, the intermediates were not bound to `proof_hash` or
`report_data`, so a miner could send a valid chain to one validator and junk to
another under one `proof_hash` (I1). Now `content_hash` includes
`tpm_intermediates` **only when non-empty**, so a proof without them keeps its
hash byte for byte (`tests/test_tpm_proof_intermediates.py` pins one), and a
proof with them binds them through `report_data`. The miner half is built:
`tpm.intermediates_for` runs once at startup inside the TD, takes each issuer
from the vendored intermediates when one signed the AK and otherwise fetches it
from the certificate's AIA caIssuers URL, re-verifies the whole chain
(signatures and validity) against the pinned root, and returns `[]` rather than
carry anything that does not chain. The miner attaches the list to **v2 proofs
only**, so a v1 proof from the v2 image is byte-identical for a validator that
has not upgraded yet. Bounded in `from_dict` (at most 4 certificates of 8 KB;
I2); the fetch is bounded to the same 8 KB.

**Issuers are chosen by signature, not by name (2026-11-08).** Google gives
every EK/AK intermediate the same subject (`CN=EK/AK CA Intermediate, OU=Google
Cloud, ...`), one per CA instance. `verify_ak_chain` used to take the first
certificate whose subject matched the leaf's issuer, which is the vendored
us-central1 intermediate, so a leaf from any other CA instance failed its
signature check and was refused **even when its proof carried the right
intermediate**: the proof-carried path could never have worked outside
us-central1. The old rotation test removed the vendored file, which hid it.
`tpm._issuer_of` now tries every same-named candidate in a fixed order (the
root, the vendored files sorted by name, then the proof's own in proof order)
and takes the first that may issue and whose signature verifies, so the
verdict is deterministic (I1) and order-independent in outcome.

**Signature and chain are one check, not two.** Either alone proves nothing: a
signature without a chain accepts a quote the miner signed with their own key
and their own certificate; a chain without a signature accepts a genuine
certificate stapled to somebody else's quote. `verify_tpm_quote` exists so no
caller can do one half and believe it did both. `tests/test_tpm.py` defeats each
half separately and shows neither defeat survives composition.

**Certificate validity is the one time-dependent input**, so it is a parameter
(`at=`) rather than a hidden call to the clock, and two validators evaluating a
certificate across its expiry boundary would disagree. Observed leaves are valid
for 30 years, so the window is wide — but this was not theoretical while writing
it: the fixture's `notBefore` was 56 minutes in the past, and the same test
written an hour earlier would have failed for a reason that has nothing to do
with correctness.

**A proof is not anonymous.** The AK certificate's subject is
`CN=<instance id>, OU=<GCP project>, O=Google Compute Engine, L=<zone>`. Every
miner's project name, instance id and zone are public in their proof. Not a
vulnerability, but miners are told in `MINER_GUIDE.md` rather than discovering
it.

**Wired into `verify_proof`, and the shape is decided by the bytes.** A miner
under dstack sends the whole attestation envelope, not a bare quote.
`unwrap_attestation` reads a u16 little-endian version at offset 0: 4 or 5 is a
bare TDX quote and passes through unchanged, anything else is decoded as a
dstack blob. Which shape arrived is therefore never a configuration question —
two validators configured differently would otherwise disagree about identical
bytes.

**If an envelope carries a TPM quote, it is verified there and then.** Not
optionally, not later: a validator checking only the TDX half accepts proofs
another validator rejects. A genuine TDX quote inside an envelope whose TPM half
fails is rejected whole, and there is a test that does exactly that — flips one
bit inside TPMS_ATTEST, leaves the TDX quote untouched, and requires rejection.
A missing `cryptography` raises rather than returning invalid, for the same
reason a missing `dcap-qvl` does: it is the operator's misconfiguration, and
downgrading it to "this proof is bad" lets a --live validator reject the entire
field while appearing healthy.

### I8 — the TPM half is bound to the TDX half (fixed 2026-11-02)

The section above says a GCP attestation has two halves and a validator must
check both. It checked both *in isolation*: the TPM quote's signature and its
chain to Google's root, and the TDX quote's DCAP chain — and nothing tied the
two together. dstack's agent writes `sha256(tdx_quote)` into
`TPMS_ATTEST.extra_data`, measured equal on both real fixtures, and that field
was parsed and ignored. So fixture A's TDX quote with fixture B's TPM quote
passed `unwrap_attestation`: two genuine signatures from two different
machines, accepted as one attestation. The second trust root proved only that
some Confidential VM's vTPM existed somewhere.

`unwrap_attestation` now requires `extra_data == sha256(tdx_quote)` and
rejects otherwise. Consensus-affecting in the direction every verification
rule is: a validator without it accepts what one with it rejects, so it ships
to all validators together, and no honest dstack blob changes outcome.
`tests/test_tpm.py::test_the_tpm_half_is_bound_to_its_own_tdx_quote` holds the
measured fact; `::test_a_tpm_quote_from_another_machine_cannot_be_stapled_onto_a_tdx_quote`
holds the refusal and was checked to fail against the previous code.

### I1 — one entry point selects an epoch's questions (protocol v2)

The scored slice and the exploration targets are consensus inputs. A miner and a validator that
compute them differently disagree on every proof, and the failure names `questions_hash`, not
the cause. Both neurons therefore call `fugal_subnet.epoch_plan.scored_slice(version, ...)` and
`exploration_plan(version, ...)` with the epoch's protocol version
(`slicer.protocol_version_for_boundary`), and nothing else selects questions.
- v1 delegates byte for byte to the functions v1 always used (`tests/test_epoch_plan.py` pins it).
- v2 selects by category with unbiased apportionment.
- The model list is an argument, so both sides pass the same versioned list
  (`routing_protocol.models_for_protocol`).
- The round-robin model assignment exists once (`exploration.assign_explore_models`).

### I8 — per-call latency is attested (protocol v2)

The v2 time gate compares a router's per-call time with the anchor's, so the time has to be
a fact the miner cannot choose. `QuestionResult.latency_ms` is the harness's own monotonic
wall time for the call, retries included, written inside the TD and covered by
`content_hash` like every other result field. It follows the `failed` precedent: present
only when set, so a proof without it hashes exactly as before and v1 miners and validators
keep agreeing on every honest proof. `from_dict` keeps the integer exactly as carried and
rejects a bool, float, string or negative value (the #120 lesson: re-typing a number changes
the hash). A validator treats a missing field as "time gate not applied". Enforced by
`tests/test_proof_latency.py`.

### I8 — a proof names the miner it was produced for

`content_hash` covers the epoch, the slice, the head and the runtime — and,
now, the **hotkey**. It is bound into the TDX `report_data`, so the hardware
attests whose proof this is.

**What it was before.** A proof was bound to no miner at all. The attack, which
is now `a_relayed_proof` in the TEE suite:

    A commits weights_hash_A, runs an honest TD, serves head + proof.
    B reads A's head off A's own axon response — it is served inline, publicly.
    B commits weights_hash_A as its own; nothing prevents committing another
      miner's hash.
    B relays A's proof verbatim.
    verify_proof PASSED.

What stopped it was `dedup.find_duplicates` plus seniority (registration
block, then commit block): B's routing vector is identical and B is junior, so
B is disqualified.
That defence is real and still stands as defence in depth — but it is a cosine
threshold and a block ordering, which is a **statistical** answer to a question
with a **cryptographic** one.

**Two moves, two different failures.** Relaying A's proof under B's hotkey fails
the identity check. Rewriting the field to B's hotkey fails `report_data`,
because the quote was signed over the original `content_hash`. Both are in the
attack suite, and the second is the canary: if it ever reports EXPLOITED, the
hotkey has fallen out of `content_hash` and the binding is decorative.

**Enforced in every mode**, like the `report_data` binding and for the same
reason: this is a property of the proof's content, not of the silicon, so a
local testnet catches relay too. Mock mode is the absence of hardware checks,
not a weaker set of rules.

**A missing `expected_hotkey` raises rather than skipping.** Verifying without
it silently restores the property this removes, which is the failure shape this
document exists to catch. Callers that genuinely have no metagraph — attack
fixtures, determinism runs — pass `mock=True`, which is the honest way to say
"not checking hardware".

**The hotkey is public, so binding it costs no confidentiality.** It is on
chain. It reaches the TD over the provisioning channel as `hotkey_ss58`, which
is MAY-CROSS by the channel's own rule. A miner who pushes somebody else's
hotkey produces proofs bound to *that* hotkey, which their own uid cannot use.

**Checked first**, before unwrapping or DCAP: it is the cheapest check that can
reject, it needs no hardware, and a relayed proof should not cost a DCAP
verification to refuse.

### I8 — measured on real hardware: what a live dstack CVM actually reports

Three questions that two sessions failed to settle by reading, answered by one
run on a `c3-standard-8` dstack CVM in GCP us-central1. **Measured**, not
modelled and not read.

**`.status` is `UpToDate`, `advisory_ids` empty**, across three consecutive
`verify()` calls. So a real dstack TDX platform on GCP c3 is up to date, and
none of `OUT_OF_DATE`, `SW_HARDENING_NEEDED` or `CONFIGURATION_NEEDED` arose.

**Whether `verify` raises on `Revoked` is STILL UNANSWERED.** No revoked
platform was observed, so the question did not arise — and an absent observation
is not a null result. Recorded as open rather than quietly closed, because the
temptation after a clean run is to treat "we did not see it" as "it does not
happen".

**`verify` did not raise at all on this path** — three calls, zero exceptions,
every one returning a report. Combined with `verify_dcap` discarding `.status`,
the operational consequence is exact and it is the finding rather than the
timing:

> **An `UpToDate` box and a `REVOKED` box are currently indistinguishable to
> this validator.**

**Timing, GCP us-central1 to Phala** (a validator on a home connection or
another continent will see something else, so the number is labelled with its
network):

    788.39 ms   cold, first call after boot
    687.85 ms
    698.75 ms
    quote generation 252.12 ms, blob 39,467 bytes — measured separately so the
    network cost is isolated

Against the 2.68 ms local verify cost measured elsewhere, **the Phala round trip
is ~99.6% of the cost of verifying a proof.** Serially, 100 miners is roughly
70 seconds of verification per epoch. So collateral-in-proof is a **throughput**
argument as well as a trust-surface one — which neither the fork analysis nor
the local CPU measurement showed on its own.

### I8 — the base measurement is independent of instance shape AND application

`measurement_id` is `sha256(MRTD ‖ RTMR1 ‖ RTMR2)`. Byte-identical across three
real deploys in September 2026:

    12a1f2f56907f80576be553f3d71031ec86f2848877ac05c82db5d8627fc9141

and across two fresh deploys on 2026-11-03:

    f4be9383e31c6cc87707b2b01ace9114ed958435b524425677e63d65065aa427

The two differ because MRTD (and RTMR0) changed between the dates while RTMR1
and RTMR2 did not: GCP rolled the TDX virtual firmware, the image stayed. The
invariant (shape- and application-independent) holds on both dates; what it
does not promise is platform-independence over time, which is why an entry is
published from a live measurement and a firmware roll is a base rotation.

    fixture A   c3-standard-4, nginx/socat test app
    fixture B   c3-standard-4, nginx/socat test app
    rehearsal   c3-standard-8, the real Fugal miner image

Two different instance shapes and two different applications, one value. This
was reasoned when RTMR0 was dropped and explicitly flagged as unverified; it is
now measured, and it confirms the `base_measurement:app_identity` split doing
exactly what it was designed for — the base identifies **what booted**, the app
half identifies **what ran**, and neither leaks into the other.

Consequence for miners: **instance size is a free choice.** It is stated in
MINER_GUIDE as a permission, because a miner who assumes they must match a
reference shape will over-provision for no reason.

### I8 — an approved image must be publicly pullable

An approved-list entry is `<base_measurement>:<app_identity>`, and the app half
is a compose hash. **If the image that hash names cannot be pulled by anyone,
the approved list is an assertion rather than evidence.**

That is the reason, and it outranks the practical one. The practical one is that
dstack has no private registry authentication at all — its only registry feature
sets `registry-mirrors` in `daemon.json`, which is a mirror list, not
credentials, so there is nowhere for a pull secret to live.

The point of computing the app identity off-hardware (`compute_app_identity.py`)
is that **what the subnet accepts is reviewable in a pull request**. A private
image breaks that at the last step: the compose file is reviewable, the hash is
reproducible, and the thing the compose actually runs is opaque. Reviewing the
first two while the third is unavailable is the shape of check this document
exists to reject — one that reports success while examining nothing.

Nothing in a Fugal miner's image is secret. The API key arrives at runtime and
is deliberately excluded from the compose because `/v1/Info` publishes it; the
head is pushed after boot. So publishing costs nothing that is not already
public by design.

### I8 — the app-identity chain, closed against hardware

Every link verified on a real deploy rather than argued, in
`tests/test_tpm.py::test_the_compose_file_bytes_reach_the_signed_register`:

    sha256(app-compose.json RAW BYTES)  ==  the compose-hash event in the log
    the event log                       replays to RTMR3
    RTMR3                               is inside the Intel-signed quote

`ced4501de5b82ea9...`, from `tests/fixtures/app-compose_A.json` through to
`attestation_A.bin`. The compose file's provenance matters: it is the
`app_compose` string the **guest** returned from `/v1/Info` — the bytes the
guest itself hashed, not a local reconstruction — so the two fixtures are
evidence about each other rather than two copies of one assumption.

This is what makes an approved entry computable off-hardware, and it is the
check that was missing when the compose hash was computed from normalised JSON.
That version produced a hash dstack never extends: it would have rejected every
honest miner while looking correct.

**Do not re-serialise.** Fields are in insertion order, not sorted. dstack's
bytes happen to equal `json.dumps(obj, indent=2)` exactly, and that is asserted
so a change is noticed — but nothing depends on it. Hashing a re-serialisation
instead of the bytes is guessing at their writer again, and that has already
been paid for once.

**`report_data` is right-zero-padded by the guest, not hashed.** Measured: a
5-byte value came back as those bytes followed by 59 zeros, and the guest
neither hashes nor rejects a short value. A 32-byte content hash therefore
appears in the quote as `content_hash || 32 zero bytes`, which is exactly what
`verify_proof` compares against. A different convention here would fail every
live proof on a binding check that looks like tampering rather than a mismatch.

### I8 — the miner's per-instance data channel (decided: the attested push; was an OPEN DECISION)

*Status 2026-11-06:* decided and built. Per-miner data (head, hotkey keyfile,
coldkeypub, API key) crosses an attested, encrypted push from the operator to
the TD (`tee/provision.py`; the entries "the wallet crosses the attested
channel" and "the attested channel was authenticated, not confidential"
below), and nothing per-miner is in the compose. The analysis that led there
is kept as written.

**At the time: unresolved, and it changes what a miner's `app-compose.json`
looks like, so it blocks packaging.** Recorded here rather than decided,
because it is a product question and not a packaging one.

The constraint chain, each link verified in dstack's source rather than inferred:

    encrypted env vars       are decrypted with a key fetched from KMS at boot
    therefore .env           is refused unless key_provider == "kms"
                             (a hard client-side raise, hit in practice)
    key_provider == "kms"    boot-loops against Phala's public KMS
    key_provider == "local"  needs dstack-vmm, which a GCP CVM does not have
    therefore                key_provider == "tpm", and .env is unavailable

So `allowed_envs` — the obvious answer, and one this guide briefly stated as a
requirement before it was falsified — **cannot deliver the key**.

**Putting it in the compose is not the fallback.** `app-compose.json` is
returned in full by the guest agent's `/v1/Info` as the `app_compose` field;
that is how this repo's reference fixture was obtained. A key there is readable
by anyone who reaches the agent and becomes part of an approved entry's
provenance. Unwithdrawable.

Three options, none free, none chosen:

| | Cost |
|---|---|
| **Self-hosted KMS** — dstack-cloud can stand one up (`kms deploy`/`attest`), restoring encrypted env with `key_provider=kms` | Either every miner runs a KMS, or the subnet runs one and miners trust it — a new centralisation, which is the thing dstack was chosen to avoid |
| **Miner fetches its key at runtime** from an endpoint it controls, after boot | Keeps the key out of the measurement entirely, but makes the fetch endpoint miner-controlled input reaching inside the TD — the `FUGAL_OPENROUTER_BASE` problem in a different coat |
| **The subnet supplies the key** | Changes who pays for inference; an economic decision, not a deployment one |

Note the first two converge: a self-hosted KMS *is* an attested key-fetch
endpoint. The real axis is who operates it and who is trusted.

**Nothing should be improvised here.** A miner who invents a fourth answer will
most likely put the key somewhere `/v1/Info` publishes it.

#### The question is bigger than the key, and that makes it easier

An API key is not the only per-miner value a miner needs inside the TD. From
`neurons/miner.py`, the minimum is:

| | |
|---|---|
| the **wallet keyfile** | a secret, and a different one per miner |
| the **head artifact** (`--head-path`) | per miner, and it *changes between epochs* — updating it is the entire competitive activity |
| the **API key** | a secret, per miner |
| netuid, network, port | uniform across miners; these can live in the compose |

Two of those three cannot be in `app-compose.json` for a reason that has nothing
to do with secrecy: **the compose hash must be identical across every honest
miner.** It is the app identity in the approved list. Put anything per-miner in
that file and every miner has a distinct hash, the approved list needs an entry
per miner, and the whole single-approved-application model collapses. The head
settles it on its own — a value that changes every epoch can never live in a
measured file.

So a post-boot channel for per-instance data is **required regardless of how the
key question is answered**. The key is a passenger on a channel that has to
exist anyway. That is the useful reframing: we are not choosing a secret-delivery
mechanism, we are choosing the miner's data channel, and encrypted env was only
ever one candidate for it.

It also means the "miner-controlled input inside the TD" objection is not an
argument against the channel — the channel is unavoidable — but a constraint on
what may travel it. The existing pattern already handles this correctly and
should be extended rather than replaced: the head arrives as untrusted bytes and
is *bound* by an on-chain `weights_hash` committed before the nonce. Untrusted
input that is bound is safe; unbound input is not. Concretely, what may cross:

  - **secrets and per-miner artifacts** — wallet, head, API key. Bound where
    consensus depends on them (the head already is), irrelevant where it does
    not (the wallet and key are the miner's own).
  - **NOT the upstream URL, the pool, the grader, or anything else consensus
    reads.** Those are in the measurement precisely so a miner cannot choose
    them, which is what `runtime_identity(source, pool, grader, upstream)` is
    for. A channel that can carry `FUGAL_OPENROUTER_BASE` hands a miner the
    upstream-substitution exploit that `scripts/stub_upstream.py` demonstrates.

That line — secrets and bound artifacts yes, consensus inputs never — is the
thing to decide and then enforce, and it is narrower than "how do we do KMS".

#### Resolved on hardware: the channel exists, and it is not confidential

Measured on a GCP `c3-standard-4` dstack CVM, not read from documentation.

**`key_provider` is `tpm`.** It boots clean — no KMS contact, no restart loop —
and `tpm-attest` maps both `Platform::Dstack` and `Platform::Gcp` to the dstack
PCR policy, sealing a 32-byte seed into the vTPM so app keys survive a redeploy
and are bound to the measured boot state. `kms` boot-loops against the public
KMS, `local` needs a `dstack-vmm` a confidential VM does not have. It is the
only workable value and also the best one.

**`key-provider` is its own RTMR3 event**, carrying `{"name":...,"id":...}`.
So a validator can require a provider by reading the *authenticated log*,
independently of the compose hash — meaning a provider change does not mint a
new app identity to approve. Verified against both fixtures, which carry
`"none"`. No rule is enforced yet; the field is asserted so that deciding to
require it later is a small change rather than a discovery.

**The compose bytes survive upload unchanged**, diffed local-vs-guest on a real
boot: 807 bytes, identical, `sha256(local) == guest compose_hash`. Nothing
re-serialises on the way in, so `compute_app_identity.py --compose` is
trustworthy in a pull request. Worth moving to dstack's `app_compose_file` mode
regardless, so the property holds by construction rather than by observation.

**`.user-config` is the per-instance channel.** Operator-supplied JSON shipped
on the shared disk, read only when `requirements.launch_token_hash` is set, and
**not part of the compose hash** — exactly the unmeasured per-miner shape the
head requires. The shared-disk file list is fixed (`app-compose.json`,
`.sys-config.json`, `.instance_info`, plus `.encrypted-env` and `.user-config`),
so nothing arbitrary rides along.

**But the shared disk is plain FAT32 and is not encrypted.** It is built
locally, uploaded to GCS and registered as a GCP disk, so anything in
`.user-config` is readable by Google and by anyone with read access to the
miner's project. Against the may-cross list above:

| Payload | Verdict |
|---|---|
| **head artifact** | **Fits.** Not secret, changes every epoch, and already bound by the on-chain `weights_hash` committed before the nonce. This is the payload that made a channel unavoidable, and this is the channel for it. |
| **wallet hotkey** | **No.** Plaintext to Google, and it signs the miner's on-chain identity. |
| **API key** | **No.** Plaintext to Google, and it is the miner's money. |

So the channel is free and solves the hard payload, and **the open question is
now narrowed to secret delivery only** — a self-hosted KMS, or a post-boot fetch
the miner authenticates. Those converge: a self-hosted KMS *is* an attested
key-release endpoint.

`/dstack/persistent` is **not** an ingress path, despite appearances: the guest
LUKS-formats it with a key derived from the app keys, which under `tpm` come
from the TPM-sealed seed. The operator holds no key and cannot pre-populate it.
It is persistence *for* the TD — useful for the embedding cache across restarts,
useless for delivery.

**The ceiling is 50 MB, enforced in the guest.** Written as a number because
"plenty of room" ages badly and "50 MB" does not. `dstack-util`'s `HostShared::copy`
size-checks every host-shared file as it copies it in and bails rather than
truncating:

| File | Limit |
|---|---|
| `app-compose.json` | 50 MB |
| `.user-config` | **50 MB** |
| `.encrypted-env` | 256 KB |
| `.sys-config.json` | 32 KB |
| `.instance_info` | 10 KB |

The 8 MB figure that appeared here first was a floor, not a ceiling, and reading
it as a ceiling was backwards: the shared disk is sized
`max(8 MB, contents + 4 MB)` and grows to fit what is put on it. Against a real
head — `data/rehearsal/head_m1.npz` is 77,234 bytes — that is roughly **679x
headroom**, so a head would have to grow three orders of magnitude before the
channel is threatened. Settled, not pending.

Note `.encrypted-env` caps at 256 KB. That is the real constraint on the KMS
path if it is ever revisited, and it is far tighter than the one that was
worried about here.

**Five unidentified trailing bytes.** Real `TPMT_SIGNATURE` fields carry five
bytes past the structure (`0000010000`). They are not identified and are not
guessed at. Ignoring them is safe because `r` and `s` are read from fixed
offsets at the front, so nothing appended can change what is verified. Recorded
so that nobody later mistakes silence for knowledge.

### I8 — the wallet crosses the attested channel

Found on 2026-09-06 while preparing the first `--live` run against merged
`main`: **a dstack TD had no permitted way to receive its hotkey.** The miner
signs `serve_axon` and the head commitment itself, so the keyfile must be inside
the TD, and every route into a TD had been ruled out in turn — the compose is
public, `.user-config` is plaintext to the cloud provider, and the attested
channel's allow-list carried only the head, the public hotkey address and the
API key. The earlier "TDs" that did register on 552 got their hotkeys by
`gcloud compute scp` onto a stock VM, which dstack forbids. So the design said
"wallet, head, API key may cross" while the code allowed two of the three.

Resolved the way the design already implied: `hotkey_keyfile_b64` and
`coldkeypub_b64` join `ALLOWED_FIELDS`. The keyfile is the same trust class as
the API key — a secret, and only the miner's own; a miner who pushes it to a TD
they have not verified has handed over their own identity and nobody else's.
`coldkeypub` is public and is there only because the SDK's `serve_axon` reads
the coldkey **address** off the wallet, so a hotkey alone does not serve
(measured: `bt.Wallet` with hotkey only signs fine and raises on
`coldkeypub`). The TD writes both under a tmpfs root (`/dev/shm`, 0700/0600)
and opens that as its wallet path; nothing reaches the data volume, so a key
never outlives the attestation that released it.

Checks: `tests/test_provision_schema.py` (the allow-list literal),
`tests/test_provision_transport.py::test_a_pushed_wallet_lands_in_tmpfs_with_private_modes`.
Operator side: `scripts/provision_td.py`, which refuses an encrypted hotkey (a
TD cannot answer a password prompt, and a miner hung on one is indistinguishable
from a healthy unprovisioned miner) and never prints a secret.

### I8 — the attested channel was authenticated, not confidential

Found the same day, hours after the first real pushes, by reading `push()` and
`serve()` side by side while designing an operator log path. The pusher
verified an Intel-signed quote over its nonce and then POSTed the payload — the
head, the **hotkey keyfile**, the **OpenRouter key** — to `http://<td>:8092` in
**plaintext**. A quote authenticates; it does not encrypt anything. Everyone on
the path between the operator and the cloud edge could read both secrets, and
the firewall on the port narrows who may *connect*, not who may *observe*. The
word "attested" had been doing the work "encrypted" was needed for.

**Fix, measured by test.** The quote now covers a key to encrypt to. The
receiver generates an ephemeral X25519 key when it starts and puts
`sha256(pubkey)` in the second 32 bytes of `report_data`, next to the
operator's 32-byte nonce, so the Intel signature says "this TD, on this
measured image, holds this key". The pusher checks exactly that
(`report_data_for`), then seals the payload — X25519 with a fresh operator key,
HKDF-SHA256 salted with the nonce, ChaCha20-Poly1305 with the nonce as
associated data — and sends only the envelope. The TD refuses plaintext,
refuses an envelope whose nonce it never attested, consumes each nonce once
(replay), and reports one error class for every decryption failure so a
rejection leaks nothing. A TD that returns no key is refused by the pusher as
an old receiver that would accept plaintext.

The canary is `test_refuses_a_key_the_hardware_did_not_sign_for`: the quote is
genuine and over our nonce and the TD's real key, but the TD *offers* a
different public key — the shape of a man-in-the-middle substituting their key
after a genuine attestation. If that test ever passes, the seal encrypts to
whoever asks.

**The same secret closes the black-box finding.** The push's session key is
kept on both sides (in memory in the TD, in a 0600 file on the operator's
machine — `provision_td.py --session-file`), and `GET /provision/logs` returns
the miner's recent log lines sealed under it to a caller presenting a bearer
token derived from it. So an approved miner's own operator can read its logs
with `public_logs=false`, and nobody who can merely reach the port learns
anything (`test_logs_are_operator_only_and_encrypted`).

What this does not change: every check *before* the push — nonce, base
measurement, event-log replay, compose hash, instance id — is as it was, and
`ALLOWED_FIELDS` is unchanged (the payload is validated after decryption). It
does change the miner image, so the entry approved for netuid 552 on
2026-09-06 (`cf31a1a4…`) is the last one whose receiver speaks plaintext; the
next image built from `main` requires a new approved entry, as any image
change does.

### I8 — there is no hourly TDX bare metal, so there is no quick fallback

Recorded because the obvious escape hatch from the Google trust root does not
exist in the form everyone assumes. Checked, and both first guesses were wrong:
Equinix Metal was sunset 30 June 2026, and Latitude.sh's confidential compute is
AMD SEV-SNP only, not TDX. Providers that do rent TDX bare metal — OpenMetal,
Hydra Host, OVHcloud — bill **monthly**, and OpenMetal states outright that
there is no per-hour tier. Realistic figure for an Emerald Rapids box:
$500–1500/mo, some of it behind a sales conversation.

Two consequences, and the second is the one that matters:

**Bare metal cannot be spun up to dodge a problem.** It is a migration with lead
time, not an afternoon's work. Anything in the GCP path that turns out to be a
dead end is a real schedule event.

**Therefore the Google trust root is a permanent, named cost, not a temporary
one.** The three properties above — Google's CA as a second trust root, its
intermediate served over plain HTTP, the GCP project and instance named in every
proof — are properties of the design we are committing to, not inconveniences we
are passing through on the way to something else. That is also the argument for
permissionlessness: a subnet whose miners must sign a monthly contract and talk
to a salesperson has no miners.

**The recorded alternative, not proposed and not built.** AMD SEV-SNP bare metal
is roughly a third of the price (~$207/mo observed) and dstack supports SNP on
its stable line, with measurements derivable the same way. That would replace
Intel with AMD as the sole silicon root and remove Google from the chain
entirely. The blocker is ours: `attestation.py` hard-checks
`tee_type == TDX (0x81)` and `parse_quote` rejects anything else, so it is a
porting job rather than a configuration change. Written down so that if the
Google trust root ever becomes a problem, nobody concludes the only alternative
costs $1000/mo and drops the question.

### I8 — why we believe `dcap-qvl==0.6.3` is safe

CVE-2026-22696 (GHSA-796p-j2gh-9m2q, critical): `dcap-qvl` lacked mandatory
Quoting Enclave identity validation, so a verifier could accept quotes from an
unauthorised QE — a direct attack on this invariant. The advisory names <0.3.9
as affected and 0.3.9 as patched, and says nothing about 0.6.x.

**Version ordering is not evidence**, so the source was read rather than
inferred: v0.6.3 carries a `qe_identity` module and calls
`verify_qe_identity_signature` unconditionally in the main verify path with `?`,
so a failure is fatal; it rejects on `mr_signer` and `isv_prod_id` mismatch and
gates the expected QE id to `TD_QE`. The fix is present and mandatory.

Recorded because the next person will ask, and "we pin a version above the
patched one" is a weaker answer than "we read it". A pinned security-critical
verifier that silently misses a fix is a bad failure mode, and pinning is what
makes it possible.

### I8 — why `cryptography` is pinned at 50.0.1, and what pinning cost

The first pin was `46.0.3`, chosen because it was what the resolver produced.
CI rejected it: `pip-audit` found **11 known vulnerabilities** in that version,
with fixes spread across 46.0.5, 46.0.6, 46.0.7, 48.0.1, 49.0.0 and 50.0.0.
Clearing all of them requires >= 50.0.0, so the pin is 50.0.1.

**This is the failure mode the `dcap-qvl` note warns about, caught in the act.**
An exact pin freezes a security-critical dependency at whatever the resolver
happened to pick, and it stays frozen while advisories accumulate behind it.
Pinning is still right — a behaviour change in the TPM verifier is a consensus
change and must arrive deliberately — but it is only safe when something
actively re-checks the pinned version. `pip-audit` in CI is that something, and
it is why the audit step is not optional decoration.

The bump was verified, not assumed: 46 -> 50 is four major versions, and the
verifier does real X.509 chain building and ECDSA verification against real
Google certificates. All TPM, SCALE and dstack-client tests pass under 50.0.1
against the committed fixtures, so the artifacts confirm the API surface we use
is unchanged.

### I8 — the approved entry is a PAIR, not a hash

An approved-list entry is `<base_measurement>:<app_identity>`. (Written
2026-09; since 2026-10-14 a bare `<base_measurement>` is **refused** outside
mock — see "The model upstream is miner-controlled" under Known gaps — and the
app half that is enforced is the sha256 of the frozen `app-compose.json`'s raw
bytes, replayed out of the RTMR3 log. The sha384 `runtime_identity` below is the
advisory userspace extend; the shape argument is unchanged.)

    base_measurement = sha256(MRTD || RTMR1 || RTMR2)     what booted
    app_identity     = sha384(source || pool || grader || upstream)   what ran

**Kept as two halves rather than one hash, deliberately.** They have different
lifecycles and different approvers: the base rotates when the image or kernel
changes, on the cloud provider's schedule; the app identity rotates when our
code, pool, grader or upstream changes, on ours. Hashing them together would
force one rotation for either event and make "what code is approved"
unreadable. Apart, the app identity is computable off-hardware straight from the
repo — so **what code the subnet accepts is reviewable in a pull request**
instead of requiring someone to hold a quote.

Rotation follows from the shape rather than needing a mechanism of its own:
publish the new value, approve both during the transition window, retire the
old. Each half rotates independently. A bare base remains valid and means "this
image is approved, nothing is required of RTMR3" — which is the honest state on
an unlocked image, where an RTMR3 match proves nothing because an attacker
running modified code simply extends the expected value.

**What locking the image changes is which half is trustworthy, not what the
halves are.** That is why this definition is not dstack-specific: dstack moves
the extend into a measured initrd and adds entries to the log, and the pair
above is unchanged.

**It is an extend, not a set.** The hardware computes
`RTMR = SHA384(RTMR || input)` from 48 zero bytes at boot, so the register never
holds the value written. Confirmed by replay: writing
`sha384(b"fugal-runtime-test")` to a fresh TD produced
`f6fdca9f66372d80685dc6be023d9e6ae25a42334f3b59de24c78da941883a64…`, equal to
`sha384(bytes(48) + input)`.

The consequence is a design constraint for the verifier: **RTMR3 can never be
compared to `runtime_identity()` directly.** A verifier must replay the chain of
extends from zero and check the result equals the quote's RTMR3 — the same
event-log replay dstack requires, now confirmed as a property of the hardware
rather than a dstack convention.

Checks: `tests/test_tee.py::test_runtime_identity_is_register_width_and_deterministic`,
`::test_rtmr3_is_replayed_not_compared`.

**Until then `--live` binds less than it appears to**, and no rotation procedure
changes that: rotating an approved list of measurements that do not cover the
workload rotates a value proving only which OS booted.

Operational consequence for validators: **collateral is fetched from Phala's
PCCS, not from Intel.** This was recorded here as Intel for months and was
wrong. `verify_dcap` calls `get_collateral_and_verify(quote)` with no
`pccs_url`, and `dcap-qvl` 0.6.3 resolves that as:

    url = (pccs_url or "").strip() or PHALA_PCCS_URL   # https://pccs.phala.network

So a `--live` validator needs outbound HTTPS to `pccs.phala.network`, and a
validator firewalled to `api.trustedservices.intel.com` on the strength of the
old sentence would fail every verification while its configuration looked
correct.

**What Phala is and is not.** It is an availability dependency and a privacy
leak: every miner's quote is sent to a third party, and a PCCS outage degrades
verification for every validator at once. It is **not** a correctness trust
root. Collateral is Intel-signed TCB info, QE identity and CRLs, and `verify`
checks those signatures against an Intel root CA compiled into the library
(`IntelSGXRootCA.der`; `verify_with_root_ca` exists to supply a different one).
A malicious mirror cannot forge collateral — it can only withhold it. That
distinction is why this is an operational finding rather than a repeat of the
Google Attestation Verifier decision, which was rejected because a live service
returned a *verdict*. A mirror returns *evidence* we verify ourselves.

**Two related defects in the same function**, found with it. *Status
2026-11-06:* the first is fixed (every collateral fetch runs under a
per-proof allowance and a per-epoch budget, and a timeout records the proof
as `unverifiable`; I6, "a bare timeout trades a visible halt for a silent
fork"). The second is a standing policy, not a fix: every DCAP status is
accepted so that cloud firmware lag does not drop honest miners
(`docs/decisions/OPEN_WORK.md` §4), and whether `dcap-qvl` 0.6.3 raises on `Revoked` is
still unanswered. As found:

  - The `timeout=30` guards only the `ThreadPoolExecutor` branch, which runs
    when an event loop is already running. On the validator's synchronous path
    `loop.is_running()` is False, so `run_until_complete` is used with **no
    timeout at all**. Measured: a stub sleeping 60s blocked `verify_dcap` for
    the full 60s.
  - `verify_dcap` **discards the verdict**. `get_collateral_and_verify` returns
    a `VerifiedReport` with `.status` and `.advisory_ids`; the function logs it
    and returns True for anything that does not raise. So **at minimum,
    out-of-date, configuration-needed and software-hardening-needed platforms
    pass** — those are observable return values that are thrown away.
    **Whether `Revoked` also passes is UNCONFIRMED.** The compiled library
    contains the literal "TCB status is invalid: Revoked", which would mean
    `verify` raises and `verify_dcap` returns False via its except branch — but
    that string cannot be disambiguated from a string table, because Rust
    concatenates adjacent literals and `Revoked` is also an enum name. Pulling
    the other way: `.status` is documented as returning `REVOKED`, and
    `QuotePolicy.allow_status` would be pointless if plain `verify` already
    rejected everything but UpToDate. **Settling it needs a real quote**, not
    more reading. Recorded at the strength the evidence supports rather than the
    strength that makes the better warning.

**How the wrong endpoint got recorded, because the shape recurs.** Nobody
measured Intel. The observation was "no local PCCS is configured and
verification succeeds", and the conclusion drawn was "therefore it goes to
Intel directly". The real explanation was a library default nobody looked for.
An inference was written down as a measurement, and an endpoint appeared in
this document that had never been seen in a packet or a line of source. The
correction came from reading `dcap_qvl/__init__.py`, not from observing traffic.
Fourth instance today of a real observation describing a neighbouring thing.

### I1/I6 — the DCAP verdict is already a function of network luck

The deepest version of the PCCS findings, and the one that decides what the fix
has to be. `verify_dcap` ends:

    except Exception:
        return False

A transient network failure is therefore not "unverifiable", it is **invalid**.
Two validators verifying the *same proof* over *different network luck* reach
*different verdicts*. That is a consensus fork with no bug and no attacker
behind it, and it exists **today**, with no caching and no status enforcement
anywhere. At 256 miners fetching per proof, transient failures are not an edge
case; they are expected.

**So "does caching introduce a consensus hazard?" is the wrong question.** The
verdict is already nondeterministic across validators. Caching, status
enforcement and the time source are not three independent choices — they are
three faces of one requirement:

> **The verification verdict must be a function of the proof, not of the
> validator's network or clock.**

Anything short of that leaves a fork surface. Note that discarding `.status`
currently *hides* this: every non-exception collapses to True, so the only
divergence left is the raise/no-raise boundary. The discard is load-bearing for
determinism by accident, which is why enforcing status without fixing the fetch
would make things worse, not better.

**The fix that satisfies it, and it is the shape this codebase already chose
once.** `dcap-qvl` exposes `verify(quote, collateral, now_secs)` separately from
`get_collateral`, and `QuoteCollateralV3` has `to_json` / `from_json`. So
collateral can travel **with the proof**:

  - the miner fetches collateral once and ships it in the bundle;
  - the validator calls `verify` directly, with **no network in the epoch loop
    at all**;
  - a miner cannot forge it, because collateral is Intel-signed TCB info, QE
    identity and CRLs, checked against the Intel root CA compiled into the
    library. Untrusted carrier, pinned anchor.

That is exactly the decision already taken for Google's AK intermediate
certificates, which rotate at per-CA-instance URLs and therefore travel with the
proof rather than being fetched or pinned. Same problem, same answer, and the
precedent is evidence the pattern fits rather than a coincidence.

It removes the fork, the hang, the availability dependency, the privacy leak and
the throughput cost in one change, because all five are consequences of fetching
inside the epoch.

**The argument for it is NOT "Intel signs it, so the carrier does not matter."**
That is true and insufficient. Collateral includes `root_ca_crl` and `pck_crl`
— **the revocation lists**. So collateral-in-proof hands delivery of the
revocation mechanism to the platform holder, and revocation is the one control
in the entire chain specifically designed to operate *against* them.

The attack, and its exact limit:

  - A miner whose PCK certificate has been revoked ships a **stale but validly
    Intel-signed CRL** from before the revocation. Every signature checks out,
    every chain reaches the pinned root, the quote verifies.
  - **They CAN also supply the certificate chain. Demonstrated, not inferred.**
    An earlier draft of this entry claimed they could not, on the evidence that
    `QuoteCollateralV3` exposes nine Python properties and none is a PCK chain.
    That described the **PyO3 wrapper**, not the Rust struct that `to_json` /
    `from_json` serialise:

        $ strings -a _dcap_qvl.abi3.so | grep -o "struct QuoteCollateralV3 with [0-9]* elements"
        struct QuoteCollateralV3 with 10 elements

    The tenth is `pck_certificate_chain`, and the field-name run in the binary
    places it immediately after `qe_identity_signature`. Confirmed by
    experiment rather than by reading:

        from_json(json | {"pck_certificate_chain": "ATTACKER"})  -> accepted
        ...round-tripped through to_json                          -> "ATTACKER"
        from_json(json | {"totally_made_up_field": "x"})          -> dropped

    Unknown keys are discarded, so a key that **survives** the round trip is a
    real struct field. A miner shipping collateral as JSON therefore supplies
    the PCK certificate chain **and** the CRL — the full chain, not an old
    revocation list alone.

    Supporting evidence pointing the same way: `get_collateral`'s own docstring
    says the returned collateral "has the PCK certificate chain attached, so it
    works for quotes with any supported certification data type (including types
    2 and 3 where the PCK cert isn't embedded in the quote)", and the binary
    contains both "Failed to extract PCK certificates from collateral" and
    "...from quote" — both paths are live.

  - **Still open:** whether `verify` *prefers* the collateral's chain over the
    quote's for certification type 5, which is what a dstack TDX miner produces.
    That decides how bad this is, and it cannot be answered from the stub — it
    needs the Rust source or a live quote.

  - **The sanitiser already exists, and it is the wrapper.** The nine-argument
    Python constructor builds a collateral whose `pck_certificate_chain` is
    `None` (verified). So a validator must **never hand miner JSON to
    `from_json`**; it should parse the JSON itself, take the nine known fields,
    and rebuild through the constructor — discarding anything else, including a
    supplied chain. The field the wrapper hides is exactly the field that must
    be dropped, which makes the wrapper a usable choke point rather than the
    liability it first appeared to be. Any implementation that skips this step
    is the full exploit.

    Demonstrated end to end — hostile JSON in, chain gone, the nine honest
    fields byte-identical:

        via from_json  -> pck_certificate_chain present
        via sanitiser  -> pck_certificate_chain ABSENT, nine fields preserved

    **The encoding is pinned, not guessed.** The constructor is type-strict
    where `from_json` is not: four fields are `bytes` (`root_ca_crl`, `pck_crl`,
    `tcb_info_signature`, `qe_identity_signature`) and five are `str`. Passing
    strings throughout raises `TypeError: Can't extract 'str' to 'Vec'`. In the
    JSON the four are **lowercase hex** — `bytes([0,1,254,255])` serialises as
    `"0001feff"` — so they are rebuilt with `bytes.fromhex`, not base64 and not
    a byte array. Splatting nine JSON values into the constructor does not work,
    which is where a "cheap and unconditional" sanitiser stops being either.

    **A real dstack quote carries its own PCK chain, so stripping the
    collateral's is safe in practice.** Measured on `attestation_A.bin`:
    `cert_chain_pem_bytes` is **3677 bytes** and `ca` is `platform`, i.e. the
    certification data embeds the chain in the quote itself. So forcing the
    quote's chain cannot break a legitimate dstack miner — which is what makes
    the sanitiser viable rather than merely correct. Whether `verify` *prefers*
    a collateral-supplied chain when both are present is still open, but the
    sanitiser is safe either way for the quotes this subnet actually receives.

    **It is the security boundary, so it needs an attack case, not a unit
    test.** `run_miner_attacks` must feed hostile collateral through the real
    path and assert the chain is gone. A test that only checks the honest
    round trip would pass against an implementation that forwards the miner's
    JSON untouched.

**Therefore the freshness bound is not a staleness nuisance with defence in
depth behind it. With collateral-in-proof it is the ENTIRE security of
revocation, single-layered** — and it only holds at all if the supplied
certificate chain is stripped on receipt, per the sanitiser note above. Written at that strength deliberately: stated as
"bound the collateral age", someone later relaxes it for miner convenience;
stated as "this is the only thing standing between the subnet and revoked
hardware", nobody does.

**Freshness is implementable** — `tcb_info` is exposed as a JSON string, so
`issueDate` / `nextUpdate` / `tcbEvaluationDataNumber` parse in pure Python with
no binding change. That question is settled, not open.

**The real argument for the trade** is structural, not performance. Today a PCCS
problem is a **globally correlated validator-side failure**: every validator
hits one host at the same deterministic block, so one outage degrades everyone
simultaneously. Afterwards it is an **uncorrelated per-miner failure**: a miner
who cannot fetch their own collateral fails alone. That is the same philosophy
as I5 — miners bear their own costs — and it, not the saved milliseconds, is
why the trade is good. What is being traded is a **liveness and privacy
dependency for a revocation-freshness dependency**.

**Collateral would be miner-supplied input, and I2 applies.** It reaches a Rust
JSON parser via `from_json` carrying certificate chains and CRLs: size caps
before parse, and cases in `run_miner_attacks`. Note explicitly that collateral
is **not covered by `content_hash` / `report_data`** — it is unattested data
riding inside an attested bundle. Everything else in that bundle is attested, so
the assumption that this is too will be made unless it is written down. (Binding
it into `content_hash` would stop an outside process swapping it after the TD
produced the proof, but would do nothing about staleness, because the TD is the
miner's own.)

**Two things it does NOT remove, and both must be decided with it:**

  - **Stale collateral.** A miner may ship old but validly-signed collateral
    that predates a revocation of their platform. This needs a freshness bound
    checked against `tcb_info`'s issue/next-update fields — and that bound must
    be a **pinned constant every validator shares**, not a local TTL each
    operator tunes, or it reintroduces the divergence it was meant to remove.
  - **`now_secs`, and this is load-bearing rather than tidy-up.** The freshness
    bound is `next_update > now`. With local clocks, collateral near expiry is
    valid for validator A and expired for validator B — so the fork *moves*
    rather than closes. Epoch-block time is what makes the bound consensus-safe
    at all, which means the two land together or the change is net-negative.
    Same reasoning that made certificate validity an explicit `at=` parameter in
    the TPM verifier rather than a hidden call to the clock.

Not built. Recorded because the throughput work will otherwise fix the symptom
that was measured rather than the property that is wrong.

### I4/I6 — no fetch before the free checks pass (fixed 2026-11-02)

The section below says the fetch target is derived from the miner's own
bytes and the fetch is charged to a budget every miner shares. Until
2026-11-02 `verify_proof` ran that fetch FIRST, before the report_data,
nonce, slice, hotkey and advertised-hash comparisons that cost nothing, so a
proof copied from a public source (this repository's own fixtures hold two
real quotes), or last epoch's proof replayed, bought a fetch from every
validator before being refused by ten lines of comparison. Every local
binding now precedes DCAP and the measurement match; a fetch only ever
follows a fresh quote over a fresh content hash that names this miner, this
epoch and this slice. `tests/test_verify_order.py` counts the fetches; four
of its six cases fail against the previous order (the hotkey check was
already first, and a sound proof fetched once either way). The residual (a
quote that parses but carries an unknown FMSPC) is unchanged and still needs
collateral-in-proof.

### I4 — "unverifiable" is a privilege, and the miner picks the fetch target

If a proof can be classified *unverifiable* rather than *invalid*, that outcome
must be unreachable by miner action. Otherwise a miner triggers it deliberately
and becomes **unscoreable at will** — neither rewarded nor punished, immune to
the attack suite, and invisible in the invalid count. That is an I4 problem
(non-interference) wearing an I6 costume.

**Two facts make this harder than picking the right exception, and both were
checked in the library rather than assumed:**

  - **The fetch target is derived from the miner's own bytes.**
    `get_collateral(pccs_url, raw_quote)` parses the quote and extracts the
    FMSPC to build the request. A miner supplying a well-formed quote with an
    unknown FMSPC causes a fetch that fails for a reason that is entirely their
    doing — and at fetch time nothing has verified that FMSPC yet.
  - **dcap-qvl collapses every failure into one exception type.** Its own
    docstring: *"ValueError: If the quote is invalid, the HTTP client can't be
    built, or the PCCS / PCS fetch fails (all Rust-side errors are surfaced as
    ``ValueError`` for consistency with the rest of this module)."* So
    "Failed to find Fmspc" (the miner's bytes) and "Failed to get collateral"
    (the network) are **indistinguishable at the Python boundary**.

**The exception type carries zero information — demonstrated, not inferred.**
With every fetch pointed at a closed local port so nothing left the host,
garbage bytes, a truncated header and a mock quote all produced the identical
`ValueError`. Branching on the type is impossible; branching on the message is
the "never branch on prose" failure the flag exists to prevent.

**But splitting the calls filters more than first stated.** All three of those
failed at **parse**, before any network call — the unreachable URL never
mattered. `parse_quote` is separately exposed, so a parse-first split cleanly
classifies every malformed-bytes case as *invalid*, and those are the cases a
miner reaches by sending rubbish. Confirmed against a **real dstack TDX quote**
as well as against garbage: it parses, `fmspc` is `00806F050000`, `ca` is
`platform`.

**The residual is one case, and its size is unmeasured:** a quote that *parses*
but carries an FMSPC the upstream does not know, producing a fetch-time 404
that is the miner's doing. That stands as **reasoned, not run** — demonstrating
it needs a real upstream, and crafted quotes are not worth sending to a third
party to find out. The binary does carry "Failed to find Fmspc" as a literal
distinct from "Failed to parse quote", so the case is *classifiable in
principle* if the library ever surfaced a code; it is not inherently
indistinguishable, it is merely inaccessible without branching on prose.

So the accurate claim is **not** that item 1 is unimplementable. It is that
item 1 is implementable with a residual hole of unknown size that cannot be
closed through this API.

**The conclusion is unchanged: this is another consequence that only
collateral-in-proof resolves.** With collateral supplied by the miner
there is no fetch, therefore no fetch-failure class, and `unverifiable` narrows
to genuinely local conditions — a missing dependency, which no miner can cause.
The I4 hole closes by construction rather than by classification.

Whatever lands, it needs a `run_miner_attacks` case: hostile proof in, assert
the outcome is **invalid** and never **unverifiable**.

### I1/I6 — a bare timeout trades a visible halt for a silent fork

The obvious fix for the hang is a timeout on the collateral fetch. **It is not
safe on its own**, and the trace is four lines long:

    TimeoutError inside get_collateral_and_verify
      -> verify_dcap's `except Exception: return False`
      -> VerifyResult(False, "DCAP attestation verification failed")
      -> validator: n_invalid += 1; continue

So a timeout converts a network problem into **a false accusation**: the miner
is logged and scored as having failed verification, indistinguishable from a
forged quote. And because it fires per proof, a slow-but-alive PCCS produces a
**partial field** — some proofs verified, some timed out — which is the
divergence case already identified as worse than a total outage. Two validators
with different network luck publish different weights and neither logs anything
unusual.

**"Unverifiable" must therefore be a distinct outcome from "invalid" before any
timeout lands.** That is the prerequisite, not the follow-up.

**But it does not, by itself, close the fork — and this is the part that is easy
to miss.** Whether a fetch succeeds is a per-validator, per-proof network event.
Two validators will hold different sets of verifiable proofs no matter how
honestly each labels them. Distinguishing the outcomes stops the slander; it
does not make the weights agree.

What decides that is the **granularity of the fail-closed**:

| Granularity | Consequence |
|---|---|
| **Per miner** — skip the unverifiable ones | Validator A scores 256, B scores 255. Divergent weight vectors. Smaller and more honest than a false accusation, but still a fork. |
| **Per epoch** — publish nothing unless everything verified | **Refuted: this is the most dangerous of the three, not the safe one.** It was recorded here as "consensus-safe with an abstention cost". That was wrong. I6 says no *miner* behaviour may stop a validator completing an epoch and setting weights — and with the crafted-FMSPC path open, per-epoch abandon hands **every miner a subnet kill switch**, fired simultaneously across all validators because the collection point is a deterministic block. One miner, one malformed FMSPC, no weights anywhere. A threshold hybrid inherits the same property at the price of N+1 registrations. |

**So skip-the-miner is the only I6-compatible answer**, and that is what
`validator.py` already does — `if not result.valid: n_invalid += 1; continue`.
The epoch already completes and weights already get set. What remains is
therefore **accounting, not behaviour**: count unverifiable separately from
invalid, so an outage reads as an outage rather than as mass fraud.

That still does not make the weights agree — **no local handling of a
per-validator network failure can, because the divergence is in the input and
not in the handling.** Error handling cannot repair a nondeterministic input.
Skip-the-miner is the least bad option available locally, not a fix.

So items like the timeout, the `unverifiable` outcome, widening I6 and making
`pccs_url` explicit are **honesty and hygiene improvements, and all of them are
worth doing** — but none restores determinism. The only fix for the fork is to
remove the per-validator network dependency from the verdict entirely, which is
what collateral-in-proof does. Anything short of that is choosing which bad
outcome to prefer.

### I6 — the invariant was wrong, not merely unenforced

I6 read "no **miner** behaviour can stop a validator completing an epoch". The
hang that halted the subnet did not arrive through a miner. It arrived through
a third party, so **no reading of the old invariant was violated** while every
validator in the field blocked simultaneously on a stalled PCCS.

That is worth stating plainly, because the tempting conclusion — "we forgot a
timeout" — is the wrong one and leads to the wrong fix. A timeout on the one
call that happened to be unbounded repairs today. It does nothing about the
next call added inside the epoch loop, and there will be one. What failed was
the **scope of the property**: the invariant named the adversary (a miner)
rather than the requirement (the epoch completes). Anything outside the process
can stall, whether or not it is hostile, and whether or not it is a miner.

So I6 now reads **"no miner behaviour and no external dependency"**, and
`check_external_calls_bounded` enforces it syntactically: every `_run_coro`
call and every blocking network primitive under `fugal_subnet/` and `neurons/`
must pass a `timeout`. The check asserts the argument is *present*, not that
its value is *right* — a wrong budget is a tuning bug, an absent budget is
unbounded, and only the second kind stops the subnet.

**What the timeout actually cost to get right, and why it is recorded here.**
The pre-existing `timeout=30` was not merely inapplicable to the validator's
synchronous path. It was **ineffective on both branches**:

- `run_until_complete` had no bound of its own. Measured: a stub sleeping 60 s
  blocked for the full 60 s.
- `Future.result(timeout=30)` does raise on time, but `with
  ThreadPoolExecutor()` calls `shutdown(wait=True)` on exit, so the
  `TimeoutError` cannot propagate until the hung call finishes anyway.
  Measured: a 2 s bound over a 6 s stall returned at **6.01 s**.

Reading the code found a timeout and stopped there. Only running it showed the
timeout was decorative — the same lesson recorded in the method note of
`docs/decisions/OPEN_WORK.md`, arrived at again by the same route. The fix is
`asyncio.wait_for` **inside** the coroutine, because it *cancels* the work
rather than abandoning the wait: the loop unwinds, the worker thread finishes,
and shutdown has nothing to wait for. One mechanism, both branches.

`tests/test_collateral_timeout.py` therefore asserts the bound in **wall-clock
time**. An assertion that a timeout argument was *passed* passes against the
broken code; only a clock distinguishes a bound that fires from one that is
merely present. Falsified against the pre-fix implementation: the suite takes
120 s and fails, and 5 s and passes after.

**The bound must ABANDON the fetch, not cancel it — and this was found by
crashing the interpreter, not by reading.** The obvious implementation,
`asyncio.wait_for`, bounds the call correctly and then cancels the coroutine.
Cancelling *this* coroutine unwinds a Rust future inside pyo3, whose tokio
worker then touches the interpreter during finalization:

    thread 'tokio-rt-worker' panicked at pyo3/src/interpreter_lifecycle.rs:
    assertion failed: The Python interpreter is not initialized

Measured against a local socket that accepts and never answers: the bound fires
correctly at 2.01 s, the process stays usable, and then exits 134 or 139 — **4
runs in 10**. `neurons/validator.py` contains no `async def`, so it takes that
path for every proof of every epoch. A hang had been traded for an intermittent
crash.

The whole test suite passed throughout, because every case stubbed
`get_collateral` with an `asyncio.sleep`. **A Python coroutine cancels cleanly;
the Rust future does not.** The stub differed from the real thing at exactly
the point under test — the same error as the decorative `timeout=30` those
tests were written to catch, committed while writing them. So
`tests/test_collateral_timeout.py` now runs the real binding in a subprocess
against a stalled socket and asserts the **exit code**, six times, because the
broken version passed six runs in ten.

The fix is a daemon thread with a bounded `join`. On timeout the thread is left
running: the fetch finishes on its own, or the process exits without it.
Nothing is cancelled, so pyo3 is never unwound. Measured 12/12 clean.

**The cost of abandoning is a leaked file descriptor per abandoned fetch**
(measured: 1.02), because the socket outlives the wait. A container on the
common 1024 default would exhaust it within a few epochs against a stalled
PCCS and lose its axon and subtensor sockets — an I6 failure by a different
road. The only way to leak fewer is to dial fewer times, so `CollateralBudget`
carries a circuit breaker: `TEE_COLLATERAL_MAX_TIMEOUTS` (10; see below for why not 3) consecutive
timeouts and this validator stops dialling that endpoint for the rest of the
epoch.

**The threshold is 10, and 3 — the intuitive answer — was measurably wrong.** A
breaker turns a graded failure into an all-or-nothing cliff, so where the cliff
sits decides how far two honest validators *on the same PCCS* diverge.
Simulated over 400 validator-epochs per cell at the real defaults, the p10..p90
spread in proofs verified was 11 with no breaker and 11 at K=10, but **136 at
K=3** once one call in ten stalls: one validator trips at proof ~5 and verifies
100 while its neighbour never trips and verifies 236. The breaker was free when
the endpoint was healthy and maximally divergent exactly when it was degraded,
which is the only case it exists for. K=10 is indistinguishable from no breaker
on divergence and still caps a dead PCCS at 10 abandoned fetches.

The fd cost of the higher threshold is also smaller than the first measurement
suggested: abandoned descriptors **are** reclaimed once the connection dies, by
peer close or by the kernel's retransmit timeout (~15 min, well inside a 3600 s
epoch). The leak is transient, not cumulative across epochs. The original
"exhausts 1024 in four epochs" came from a stall server that held every
connection open forever, which no real PCCS does. Consecutive rather than cumulative — an upstream answering in between is
slow, and slow is what the time budget is for. That bounds the leak at 10 per
epoch instead of one per miner.

**The per-proof bound does not by itself satisfy I6 at field scale**, and that
is worth stating because "the fetch has a timeout now" reads like the problem
is solved. The verify loop is serial over every UID, so a slow-but-alive PCCS
costs N x the per-call bound: 256 proofs at 10 s is 2560 s against an 1800 s
post-collection window. The epoch completes in the sense that the function
returns, and misses its weight-setting window regardless — an I6 failure
wearing the costume of a successful run.

So the bound is **two bounds**. `CollateralBudget`
(`config.TEE_COLLATERAL_EPOCH_BUDGET`, 600 s) caps what the whole phase may
spend, is charged on every exit rather than only on success, and refuses
without dialling once spent. Sized at ~3.4x the measured healthy cost
(256 x 690 ms = 178 s) and a third of the post-collection window.

Two consequences that are not obvious:

- **Verification order became a fairness question.** When the ceiling binds,
  whoever is last in line goes unverified. In UID order that is the same miners
  every epoch. The order is therefore derived from the epoch nonce — identical
  across validators, not miner-influenceable, and it rotates the exposure.
- **A time ceiling biases persistently, not just once.** A count cap would cut
  at the same index on every validator; a time budget cuts where each
  validator's network luck puts it. That difference is not noise — a validator
  on a poor link verifies systematically fewer proofs **every epoch**, so the
  divergence is correlated with infrastructure quality rather than averaging
  out. It is a real cost of choosing time over count, it is accepted because a
  count cap cannot bound time at all (to avoid binding a healthy epoch it must
  exceed the field size, which is the 2560 s overrun being fixed), and only
  collateral-in-proof removes it.
- **A shared ceiling is a shared resource**, so I4 applies to it. A malformed
  quote cannot drain it, because parsing happens before fetching and is judged
  locally; that is now asserted by a test rather than left as a property of the
  code's current shape. The crafted-FMSPC residual still reaches the fetch, but
  buys one fetch per miner per epoch, so exhausting the ceiling also requires
  the upstream to be slow.

  That last clause turned out to rest on a quantity **nobody has measured**:
  how long a PCCS takes to refuse an FMSPC it does not recognise. A fast 404
  (~0.7 s) needs >850 hostile UIDs to drain 600 s and the attack is impossible
  at 256; a stall to the full per-call bound needs ~60, and with the circuit
  breaker as few as three landing consecutively in the permutation would end
  verification for the rest of the field. Registration is at the floor
  (~$0.12/UID).

  So the allowance enforces a **per-proof fair share** —
  `min(per_call, remaining, remaining/proofs_left)` — which bounds what any one
  proof can take from the miners behind it regardless of that timing. The
  design is no longer conditional on an unmeasured fact. Measuring it is still
  worth doing, but it decides tuning now rather than safety.

### I6 — a PCCS hang halts the subnet, and does it invisibly

Two facts recorded separately above are far worse together, so they are stated
here as one:

  - every validator fetches collateral for every proof from **one host**, with
    no caching, and they do it **simultaneously** because the collection point
    is a deterministic block (I6/I9);
  - the `timeout=30` does not apply on the validator's synchronous path, so a
    slow PCCS blocks for as long as the HTTP client allows, per proof, serially.

**The outage is the safe case; the hang is the dangerous one.** That is the
counterintuitive part and it is why this needs writing down:

| Phala state | What happens |
|---|---|
| **Down** — fast connection error | `verify_dcap` catches, returns False, every proof invalid, the epoch is skipped and logged with the `no_valid_proofs` anomaly. Visible, recoverable, no weights set. |
| **Hanging** — accepts and stalls | The validator blocks with no timeout. The epoch never completes. No weights, **no log line, no anomaly** — the code that would record the failure is downstream of the block and never runs. |

So the failure that looks less severe is the one that stops the subnet, and it
stops it silently, on every validator at once.

**I6 as written does not cover this.** It says *"No **miner** behavior can stop
a validator completing an epoch and setting weights"*, and its guard is the TEE
proof timeout — which bounds the miner query, not the collateral fetch. The
hostile miner who hangs an epoch was anticipated and defended; the same hang
arriving through a third-party dependency was not, because the invariant names
miners rather than naming the property. **A liveness invariant scoped to one
source of delay is not a liveness invariant.** Whatever fix lands here should
widen I6 to "no external party" and give the fetch a bound that holds on both
code paths.

### I1/I6 — one collateral fetch per platform per epoch (2026-11-08)

`verify_dcap` fetched collateral once per PROOF. Collateral depends only on
the platform (FMSPC), the PCK issuing CA and the quote type, and both real
deploys share one platform (FMSPC `00806F050000`), so the epoch's
`CollateralBudget` now keeps what it fetched per platform and every later
quote on that platform verifies with no network call. About 256 round trips
become about one: most of the verify time, most of the network-luck
divergence between validators (the verdict on a cached platform no longer
depends on each proof's own fetch), and the per-proof PCCS dependency go
with them. No trust is added: the collateral is Intel-signed and checked on
every use; only successful fetches are kept.

**The trap, found by running it.** `get_collateral` copies the asking
quote's PCK certificate chain into the collateral, and `verify` judges a quote
by the collateral's chain when one is present (a garbage chain fails a
genuine quote; no chain falls back to the quote's own and passes). Shared as
fetched, every miner would be judged by the FIRST miner's PCK certificate:
honest miners on other machines refused, and one miner choosing the chain
the rest are judged by (I4). The chain is dropped before the collateral is
used or kept (`_without_pck_chain`), so each quote is judged by the chain it
carries, which every dstack quote does.

**Stale collateral is an outage, not an accusation.** Measured on recorded
collateral: past its window, `verify` fails every genuine quote ("TCBInfo
expired", "CrlExpired"), and before its issue date too. Those failures read
as every miner's proof being bad at once. When `verify` refuses a quote and
the collateral is outside `collateral_window` (latest issue date of TCB info
and QE identity; earliest next update of those and both CRLs; the PCK CRL
expires first, by minutes), `verify_dcap` raises `CollateralUnavailable`
instead of returning False. The window only ever explains a refusal; it never
refuses what the library accepted.

**Residual, accepted.** An abandoned fetch keeps its thread and socket until
the connection dies. The breaker bounds abandonment per epoch; across epochs
nothing does, so a PCCS that accepts connections and never answers leaks a
few descriptors per epoch. A process-wide cap was built and dropped: once the
PCCS recovered, connections that never close would keep it engaged and the
validator would stop dialling for good, a silent stop, whereas descriptor
exhaustion ends in a crash that the service manager restarts.

`tests/test_dcap_verdict.py` (real library, real fixtures, collateral
recorded from the configured PCCS on 2026-11-09, pinned time, no network):
genuine quotes pass, one changed byte is refused, a signature failure is a
refusal (until then no test saw `verify_dcap` say no: every attack case stubs
it), stale and unreadable collateral are outages, one fetch serves a
platform, and each quote is judged by its own chain.

### I8 — what "attested" actually means

DCAP verification proves a quote is genuine and Intel-signed. It proves the
*hardware* is real. It says nothing about whether the code inside it is the
published code — an attacker who owns a TDX machine produces a perfectly valid
quote while running a modified harness.

For a period this subnet checked `proof.source_hash`, a field the workload
writes about itself, against the approved-image list. That is not an attestation
of anything: the attacker simply writes an approved value. The check now reads
`measurement_id(quote)`, derived from MRTD, RTMR1 and RTMR2 (RTMR0 was dropped later; see above), which the CPU fills in
and the Intel signature covers. RTMR3 is excluded because it is
application-extendable, so including it would make an image identity change
with runtime data and no image could stay on an approved list.

The full chain a proof must satisfy:

```
Intel DCAP signature          -> the quote is genuine, from real TDX hardware
TDATTRIBUTES.DEBUG clear      -> the host could not read or change the TD
measurement_id(quote)         -> the image that ran is the published one
report_data == content_hash   -> the proof body is what that image produced
weights_hash == commitment    -> the head that ran was committed before the nonce
sha256(bundled head)          -> the head shipped is the head attested
result ids == assigned slice  -> those answers are to the questions we asked
exploration == nonce targets  -> the sampling quota was actually performed
per-question costs == total   -> the cost figures are internally consistent
```

The DEBUG line is checked by `attestation.td_is_debug` in `verify_proof`, in
every mode, before any fetch (2026-11-08). Until then it rested on two things
outside the repo: Google offering no debug TDs (every live proof carries
Google's TPM half), and the DCAP library probably refusing them (its binary
carries "Debug mode is enabled"; never observed on a real debug quote).
`tests/test_tee.py::test_verify_proof_refuses_a_debug_td`, and
`tests/test_dstack_blob.py` confirms both real GCP quotes read as non-debug.

Each line was, at some point, absent — and `run_tee_attacks.py` keeps an
executable exploit for each, because every one of them fails *silently*: the
proof verifies, the miner is paid, and nothing in a log says otherwise.

### I4 — the reference frame must not move with the field

The constant-policy frontier needs an estimate of how good each model is,
and that estimate is built from miners' exploration samples. If it tracked the
miner population, every miner would move every other miner's score and the
same head would be worth more in a thin epoch than a busy one.

Three things keep it from doing so. The frame accumulates over *time*, not
over miners, so one epoch's samples are a single decayed contribution to a
long-running estimate. Since 2026-10-14 an epoch contributes one observation
per (question, model) however many miners answered it (I9), so the number of
trials — and with it the frontier's confidence — no longer depends on field
size at all. And every frontier point is valued at the raw sample *mean*
rather than a confidence bound (either direction): a bound's width shrinks as
evidence accumulates, which made the old ceiling a function of sample count
(measured, when sample count tracked field size: 0.14 of score between a
3-miner and a 50-miner field). A thin frame is discounted once, through the
frontier's confidence factor (`FRONTIER_MIN_TRIALS`), which is the same for
every miner in the epoch and so cannot reorder them.

Residual: a small subnet gathers distinct observations no more slowly than a
large one now — exploration slots per epoch are fixed — but a field below a
handful of honest operators per question loses the median's protection (I9).

### I4 — seniority squatting (residual, accepted; v1 only)

Superseded from protocol v2 by first-commit seniority (the v2 copy rule).
Under v1, dedup seniority is `commitments.seniority_key(registration block, commitment
block)`, both read from chain state at the epoch boundary. A squatter who
registered *before* its victim therefore still holds earlier seniority. This is
far weaker than the copy-and-outrank bug it replaced — it requires predicting
the victim well in advance — but it is not zero.

### I1 — commitments and seniority come from the boundary block (fixed 2026-10-14)

Two things the 2026-10-14 diligence review found under "additional findings".
On-chain commitments were read at the chain head, at whatever moment in the
epoch a validator reached that phase, so two validators could see different
commitment sets for the same epoch. And seniority was the earliest commitment
block each validator had ever *seen* for a hotkey, persisted in its own state
file: deterministic for one validator and different for each, so two
validators could disqualify different members of the same dedup cluster. Both
are I1 breaks with no bug behind them.

`get_commitments_with_blocks(subtensor, netuid, block=boundary_block)` pins
the read. Seniority is `seniority_key(registration block, commitment block)`,
the registration block from the metagraph pinned at the collection block and
the commitment block from the pinned read; the per-validator
`first_commit_blocks` map is gone from the state file. Registration comes
first so a retrained author keeps rank over whoever copied its previous head
(`tests/test_seniority.py`, and the original guard in
`tests/test_non_interference.py`).

### I9 — the frame is a vote of operators, not a sum of answers (fixed 2026-10-14)

The 2026-10-14 diligence review rated this critical, and reproduced it with the
repo's own code: every miner answers the same exploration questions, each copy
was added to the frame as a fresh trial, and a failed model call was graded as
a wrong answer. A coalition that dropped its own network during exploration —
which ran as a separate pass after the scored questions, so it was a separate
thing to block — pulled every model's measured accuracy down by exactly its
share of the field. At 10% of miners, 9 of 17 always-one-model policies became
profitable; at a third, 15 of 17.

| colluding share | frontier peak | constant policies paid |
|---|---|---|
| 0% | 0.898 | 0 of 17 |
| 10% | 0.809 | 9 of 17 |
| 20% | 0.719 | 12 of 17 |
| 33% | 0.603 | 15 of 17 |

Four changes, which ship together because they share the sample schema:

1. **A failed call is attested as failed.** `QuestionResult.failed` is set by
   the harness when the call did not complete (None from `_call_model`, as
   distinct from "" for a reply with no text), is hash-bound, and is held to
   its meaning by `verify_proof`: a failed result cannot be correct and cannot
   carry a reply. `failed` enters `content_hash` only when True, so a proof
   with no failures hashes as it did before the field existed.
2. **Failing exploration is not free.** `verify_proof` rejects a proof whose
   exploration failure rate exceeds its scored failure rate by more than
   `EXPLORE_FAIL_MARGIN` (0.34, six of fifteen slots). It is the difference
   that is judged, so an outage that takes both down is a bad epoch and not a
   rejection.
3. **Exploration is interleaved with scored calls** in nonce order, so the
   network sees one stream. This removes the cheap form of the attack; TLS
   record sizes could still in principle single out a request, which is why
   it is 1 and 2 that close it and this is defence in depth.
4. **One sample per (question, model) per epoch**, via
   `reference_frame.aggregate_observations`: failed answers dropped, one vote
   per coldkey (hotkeys under one coldkey are one operator), the median across
   coldkeys. `frame.trials` therefore counts what has been measured — thirty
   identical answers are one observation, which is also what finding 3 (the
   prior's confidence filling on duplicates) needed. Reveals carry `coldkey`
   and `failed` per exploration entry so `rebuild_from_reveals` regroups the
   same way, and `frame_excluded_coldkeys`, the operators the detectors kept
   out of the frame that epoch, so the replay drops the same voters (without
   it a rebuilt validator re-admitted them and valued the frontier from a
   different frame, found 2026-11-05); entries written before those fields
   existed are each their own voter, which is what they were.

**The assumption, stated plainly.** The frame is robust while honest coldkeys
are a majority of those answering each question. Sybils are not a lever — the
coldkey is the unit — and each coldkey costs a registration and a TDX VM that
runs the whole benchmark. Below a majority a coalition cannot move a single
observation; above it the frame is theirs, and the two-sided detector
(finding 4) is what makes that attributable.

Enforced by `tests/test_finding2_frame_manipulation.py`,
`tests/test_proof_failed_flag.py`, the `run_tee_attacks.py` cases "block the
network during exploration only" and "a failed call that is also correct", and
the `frame` stage of `check_determinism.py`, whose synthetic miners share
exploration questions so the aggregation is what is hashed.

### I4 — the detector is two-sided, baseline-free, and enforces for the frame (fixed 2026-10-14)

`implausible_exploration` flagged a miner only when its exploration accuracy
beat the frame's expectation by more than 35 points, compared against the
frame the attack of finding 2 was moving, and changed nothing: a log line and
an anomaly entry. The 2026-10-14 diligence review rated this high. Now:

**Numbers in a proof hash as the miner serialised them (2026-11-03, mainnet
day one).** `content_hash` canonicalises `hash_payload()` with `json.dumps`,
and JSON tells `0` from `0.0`. The harness records a failed call's
`cost_usd` as the integer `0`; the validator's `QuestionResult.from_dict`
coerced every cost to `float`, so for any proof containing a failed call the
validator re-hashed `"cost_usd":0.0` where the miner had hashed
`"cost_usd":0`, and `report_data == content_hash` failed on a genuine quote.
Both seed miners' first mainnet epoch (e00026188, 42 and 39 failed calls,
nearly all `mistral-large-2512` upstream errors) was rejected this way; no
rehearsal had ever produced a proof with a failed call. Fix: `from_dict`
keeps each JSON number as it arrived (int stays int, float stays float) and
rejects bools, strings and non-finite values instead of reinterpreting them.
Rule, stated generally: **the validator must hash the bytes the miner hashed;
any parse that normalises a value changes the hash of some honest proof.**
Enforced by `tests/test_proof_json_number_types.py`. The miner-side
normalisation (record failed costs as `0.0`) is deliberately NOT part of this
change: it alters the image and therefore the approved entry, and belongs to
the next rotation.

1. **Both directions.** `|observed - expected| > margin`, against the frame
   as it stood before the epoch's samples are folded in (the call order in
   `neurons/validator.py`), where `expected` is each model's RAW mean and a
   model counts only once the frame holds 8 trials of it (2026-11-05: judged
   against the prior-pulled posterior from one trial, a cold frame flagged
   every honest miner on some epochs and dropped their samples). A miner
   serving itself gold and a miner dropping its network during exploration
   are the same distance from plausible.
2. **Cross-miner agreement, with no baseline.** `disagreeing_exploration`:
   while exploration is shared, every coldkey asks the same model the same
   question; per (question, model) with at least `EXPLORE_DISAGREE_MIN_VOTERS`
   coldkeys the majority of per-coldkey votes is the consensus, and a miner
   that disagrees with it on more than `EXPLORE_DISAGREE_MAX` of at least
   `EXPLORE_DISAGREE_MIN_SAMPLES` answers is flagged. Nothing the attacker can
   move is the reference here; the threshold is loose because OpenRouter
   serves one model id from several providers.
3. **The consequence is exclusion from the frame, never a score.** A flagged
   operator's observations are passed to `aggregate_observations` as
   `exclude_coldkeys` for that epoch. I4 says a miner's score may not depend
   on other miners; the frame is built from everyone and is the one place that
   is already true of, so keeping a flagged miner's samples out of it protects
   every other miner's reference and moves nobody's score. The flag is in the
   published epoch log as `exploration_excluded_from_frame`.

Enforced by `tests/test_finding4_two_sided_detector.py`.

### I3 — copy-and-tweak (fixed 2026-10-14)

Any registered hotkey could pull a miner's full head off its axon, because the
stake needed to query defaulted to zero; dedup compared routing decisions at
cosine ≥ 0.99, and a copy that rerouted 3 of 300 questions passed (measured
with `find_duplicates`); weight was proportional to score, so N near-copies
collected close to N shares for one TDX VM each. Diligence 2026-10, finding 5.

1. **Access.** In `--live` the miner's axon admits stake ≥
   `MIN_VALIDATOR_STAKE` (default 50, in the chain's `total_stake` units for
   the UID: alpha plus 0.18 × root TAO, childkey-inherited) and nothing else;
   a validator permit alone was dropped 2026-11-03 because the chain grants one
   to any nonzero stake on a young subnet. It refuses to start with the
   threshold at zero (`neurons/miner.py`, `caller_allowed`). Mock stays open.
2. **Parameter clause.** `head_distance.relative_distance`, rows aligned by
   model id; below `DEDUP_PARAM_DISTANCE` (0.2) two heads are one. Measured
   (`docs/decisions/HEAD_EFFICACY.md`, experiment 3): independent heads sit at 0.42+,
   a copy nudged until 15 decisions change sits at 0.10. Renaming rows to
   dodge alignment fails at load, because every model id must be priced.
3. **Epsilon rule.** `dedup.epsilon_disqualified`: a head agreeing with a
   senior one on ≥ `DEDUP_EPSILON_AGREEMENT` (0.95) of decisions must beat
   its composite by `DEDUP_EPSILON` (0.5%) or is disqualified. Independent
   heads agree on at most 92%, so the rule lives only where copies live.

A consequence worth knowing: the shipped trainer is deterministic from a zero
start, so two honest miners training it on the same public data produce one
head and are deduped as one. Independence is data.

Enforced by `tests/test_finding5_near_copy_dedup.py`.

### I1 — every scoring constant is in the digest, and none can be overridden live (fixed 2026-10-14)

Diligence 2026-10, additional finding A2. `config.py` reads its constants from
the environment, and the consensus digest covered five of them. A validator
with `FUGAL_SCORE_QUALITY_FLOOR=0.5` exported scored the field by its own
rules, and the digest two validators compare to explain a divergence said
nothing. Now every `FUGAL_` variable read anywhere under `fugal_subnet/` or
`neurons/` is in exactly one of three lists (`CONSENSUS_ENV_VARS`,
`CONSENSUS_ENV_INPUTS`, `OPERATIONAL_ENV_VARS`;
`check_every_env_constant_classified` fails CI on an unclassified one — until
2026-11-05 it scanned `config.py` alone, and `FUGAL_SKIP_BENCHMARKS`,
`FUGAL_POOL_UNPINNED` and `FUGAL_OPENROUTER_BASE`, read elsewhere, were
consensus-affecting and neither digested nor refused on finney), the
fingerprint carries a `scoring` block with every consensus constant's value
as in effect, an `epoch` block that now also names the pinned pool manifests,
the skip list in effect and whether pool verification is switched off, and a
`code` block (`source_sha256`, the miner's own `source_hash` algorithm, and
the git commit) that is in the reveal but deliberately outside the digest, so
two validators on different commits are a divergence only when a
score-affecting input differs, `consensus_digest` hashes it along
with the routing-protocol identity (which covers the price table), and
`assert_consensus_overrides_allowed` refuses any consensus constant set in the
environment on finney, identified by genesis hash as `--mock` is, in both
neurons, and `FUGAL_ALLOW_ENV_DRIFT` does not lift that. On a test or local
chain the override is allowed with a warning that names it and the digest
(2026-11-03): the netuid-552 warm-up runs `FUGAL_EXPLORE_FRACTION=0.5` on every
neuron, miners' compose included, so the reference frame reaches
`FRONTIER_MIN_TRIALS` in days, and the frame does not depend on the fraction.
Inputs — the epoch length, the approved entries — are set per deployment on
purpose and are recorded, not refused.

### I8 — the validator is live by default, and nothing runs mock on finney (fixed 2026-10-14)

Diligence 2026-10, additional finding A3. `--live/--mock` defaulted to mock
on the validator, so the mode that accepts unattested proofs was the one a
missing flag produced. The validator makes no model calls and spends nothing,
so there was no reason for it to share the miner's mock default; it is live
by default now. Both neurons read the chain's genesis block hash after
connecting and refuse `--mock` when it is finney's
(`config.FINNEY_GENESIS_HASH`, `fingerprint.assert_mock_allowed`); an
unreadable genesis is refused too, since it cannot be shown not to be
mainnet. The miner keeps `--mock` as its default because `--live` spends
real money.

### I7 — the burn UID is read from chain (fixed 2026-10-14)

Diligence 2026-10, additional finding A5. Unassigned weight burned to a
hard-coded UID 0, which is the owner's UID by convention only. The validator
now resolves `get_subnet_owner_hotkey(netuid, block=boundary_block)` each
epoch and maps it through the pinned metagraph (`resolve_burn_uid`);
`compute_weights` takes `burn_uid` as a required keyword. Live refuses to
guess on a failed lookup, because burning to the wrong UID pays someone;
mock falls back to 0 for local chains without an owner query.

### I7 — reveals are published, hash on chain (fixed 2026-10-14)

Diligence 2026-10, additional finding A1. The reveal — ground truth,
exploration observations, routing decisions, scores and weights — is the
subnet's stated product and was written to the validator's local disk and
never left it. `fugal_subnet/reveal_publisher.py` uploads each epoch's
`reveal.json` to a public-read GCS bucket (`FUGAL_REVEAL_BUCKET`) once the
validator is past the weight commit-reveal window (the hold is read from the
subnet's hyperparameters, so a reveal never leaks weights the chain is still
hiding), and commits `reveal:<epoch_id>:<sha256>` on chain with the validator
hotkey. Anyone can fetch the file and check it against the chain. Two
validators' reveals for the same epoch agree on everything except
`environment` (each records its own Python, platform and packages), so their
file hashes differ; `scripts/watch_subnet.py` compares them with `environment`
removed (I1).
A `.published` marker is written only after the upload, so failures are
retried and nothing is marked that was not sent. Operational, not consensus:
an unset bucket skips with a warning. `tests/test_reveal_publisher.py`.

### I1/I6 — an epoch is applied once, or not at all (fixed 2026-11-02)

Found by reading the loop, not by a failure, which is the point of recording
it: the shape was invisible to every test because no test re-entered an epoch.
`update_scores` mutated records in place and the frame was rebound as soon as
it was computed, before the epoch was durable; an exception anywhere between
scoring and `save_state` — a dropped websocket in `set_weights`, a failed
owner-hotkey read, a disk error on the reveal — landed in the outer handler
with the loop-level state already advanced, and the retry of the SAME epoch
applied it again: evidence accumulated twice, misses applied twice, the frame
decayed twice, then persisted. One chain blip at the wrong moment made one
validator's evidence differ from every peer's until it decayed (half-life 200
epochs). Now `update_scores` is copy-on-write and the loop rebinds its durable
names only at the commit point; `tests/test_epoch_atomicity.py` asserts that
structurally against the AST and fails against the previous loop.

The same pass moved the pinned commitments read to the moment the boundary
block is known. It sat after the collection wait and the 120 s query, by
which time the boundary block was ~250 behind the head and a non-archive node
(which keeps ~300) was one slow epoch from refusing it — and the refusal was
swallowed into an empty map, which under `REQUIRE_COMMITMENT` rejected every
proof and marked the epoch processed with no weights while peers that read in
time set theirs. In live a failed read now raises and the epoch is retried.
A validator should still run against an archive endpoint; this removes the
window, not the dependency.

### I6 — a submitted weight vector that the chain will discard (fixed 2026-11-02)

Three parameters decide whether a `set_weights` the SDK reports as
submitted ever reaches Yuma, and none of them is visible in the result. The
chain checks the commit-reveal VERSION at reveal time
(`IncorrectCommitRevealVersion`), not at commit, so a validator committing
with the wrong version logs "Weights set successfully" every epoch while
every reveal is rejected; `WeightsVersionKey` must be at or below the SDK's
`version_key`; and `min_allowed_weights` / `max_weights_limit` bound the
vector. `assert_chain_weight_parameters` reads all of them once at startup,
logs them, and refuses live on the first two. The SDK also returns
`(False, None)` without attempting anything while the rate limit has not
elapsed, which the loop logged as a failure with no reason and then marked the
epoch done; `set_weights_with_retry` names the rate limit when that is the
cause and otherwise retries a transient failure while the epoch has blocks
left. Finney at runtime 486 (2026-11-02): `CommitRevealWeightsVersion` 4, a
fresh subnet's `weights_version` 0, `min_allowed_weights` 1,
`max_weights_limit` 65535. `tests/test_validator_chain_guards.py`.

### I1 — protocol v2 activates at one named boundary (v2 release; not yet active)

v2 changes how the miner share is decided (a champion chosen by a paired
net-value comparison, seats for the last three champions, a new copy rule, a
new miss rule). Old and new rules cannot be mixed within one epoch, or across
validators in one epoch, without splitting weights. So the switch is a single
block, `FUGAL_V2_ACTIVATION_BLOCK`, and three properties hold:

1. **The epoch's boundary block decides, not the validator's current
   block.** `slicer.protocol_version_for_boundary(boundary_block,
   activation_block)` is the only place the version is computed. Every block
   inside an epoch maps to the same boundary, so a retried epoch (I1/I6, "an
   epoch is applied once") and a validator that reaches the epoch late apply
   the same rules as every peer. v2 applies from the activation boundary
   itself.
2. **The activation block is a boundary.** `validate_activation_block`
   refuses, at validator startup and before any chain connection, a block that
   is not a multiple of `blocks_per_epoch`, rather than rounding it: a block
   inside an epoch would give that epoch rules that depend on where a
   validator looked.
3. **It is in the consensus digest.** The value sits in the fingerprint's
   `epoch` block. It is a consensus *input* (`CONSENSUS_ENV_INPUTS`, like the
   epoch length), so it is set per deployment and recorded, not refused, but a
   validator configured to flip at a different block, or never, shows a
   different digest from its first epoch.

Unset means v2 never activates: a validator running the v2 code without the
variable scores every epoch under v1. That is the safe failure for code
deployed ahead of the flip, and the dangerous one after it, so the activation
block is part of the release instructions and the digest is what makes a
missing value visible.

The v2 scoring constants (`CHAMPION_*`, `PAYOUT_SEAT_SHARES`) are ordinary
consensus constants from the start: in `CONSENSUS_ENV_VARS`, in the
fingerprint's `scoring` block, and refused as live overrides on finney like
every other. They have no effect before the activation block. The promotion
margin, `CHAMPION_MARGIN` (0.2 net-value points at 99% paired-bootstrap
confidence), is the single margin of v2: it replaces the absolute headroom
margin (0.004) an earlier draft used, both for a challenger taking the seat
and for a junior head beating a senior one.

Enforced by `tests/test_v2_activation.py`.

### I4 — shared grades: a cell is a vote of operators (v2 release; from the activation block)

From protocol v2 a challenger is compared with the champion question by
question, and every head is graded from **one shared grade per (question,
model) cell** of the epoch's scored slice
(`fugal_subnet/shared_grades.py`): failed calls dropped, one vote per coldkey
(the mean over its hotkeys), the median across coldkeys. A head's grade on a
question is the grade of the cell it routed to, except that its own failed
call stays a wrong answer: a miner whose call failed did not answer and does
not collect the answer other miners paid for.

**Why.** Two heads that route a question to the same model are asking the
same thing of the same model, so a difference between their own answers there
is the provider answering differently on the second call, not routing.
Measured on mainnet reveals, two independent answers to one cell disagree on
about 8% of cells, and the paired comparison's variance falls to 0.48 of the
unpaired comparison with own answers and to 0.23 with shared cells. That is
the difference between a promotion margin the field can clear and one it
cannot.

**What this costs I4, stated plainly.** A head's grade now depends on other
operators' answers. What they contribute is attested model output graded by
the pinned grader, so it is a measurement of the model on that question, not
a judgement of the head, and it is the same measurement for every head that
routed there. A cell moves only when at least half of its voting coldkeys
answer differently; at exactly half it moves halfway, and a cell answered by
one coldkey is that coldkey's own answer. Hotkeys are not a lever: the
coldkey is the unit, as in the frame (I9). Residual: a cell with few voting
coldkeys is only as robust as those few, so on a thin field a single operator
can move a cell it shares with one other operator halfway. The majority
assumption is the frame's, applied per cell.

**Accepted residual: several voters grade a cell by majority.** The median of
three or more votes is their majority, so a cell several coldkeys answered is
graded at the majority answer and a cell one coldkey answered at its single
answer. Where a model is right more often than wrong on a question, more
voters raise the cell's expected grade (right 80% of the time: 0.80 with one
vote, 0.90 with three, 0.94 with five); where it is more often wrong they
lower it. Measured on mainnet v1 reveals (2026-11-10: 123 epochs, 1,653 cells
with 20 or more independent answers, two answers disagreeing on 7.7% of
cells, the figure above): a three-voter cell is graded +0.39 points above a
one-voter cell on average (95% interval +0.20 to +0.59), a five-voter cell
+0.58 (+0.32 to +0.85). It acts only on the questions two heads route
differently (the top heads differ on a median 24% of questions), so between
typical heads it is worth about 0.1 to 0.15 points of net value, below the
margin, and more the more differently two heads route. Two consequences: an
operator running one routing under several coldkeys gains that much on its
own routes (each coldkey costs a registration, a TDX VM and the epoch's
inference), and a head routing where nobody else does is measured that much
below a route the field shares. With the shipped `paired_test` and constants
and the measured cells, a challenger with no real edge that routes 30% of
questions differently took the seat in 1 of 60 30-epoch runs alone and in 3
of 60 with three or with five coldkeys. A mean across coldkeys does not
depend on the number of voters; it is a consensus change, to be decided on
v2 reveals (v2.1).

**Determinism.** Every step is order-independent and the published form is
sorted (`cells_for_reveal`); the `shared_grades` stage of
`check_determinism.py` feeds proofs in the opposite order to the frame stage.
From the activation boundary the reveal carries `protocol_version` and
`shared_grades`, so anyone can recompute every head's v2 grades from the
reveal alone; a v1 reveal is byte-for-byte unchanged.

Enforced by `tests/test_shared_grades.py` and the `shared_grades` stage of
`scripts/check_determinism.py`.

### I3 — one champion; taking the seat needs the margin, holding it needs none (v2 release; from the activation block)

From protocol v2 the miner share is not proportional to score. It goes to a
**champion** (and, from the payout change, to the last three champions), and
the seat changes hands by a paired test (`fugal_subnet/champion.py`):

- **Net value per question** of a challenger over the champion,
  `NV_i = u_i·(g_challenger − g_champion) − (s_challenger − s_champion)/V`,
  in points. g is the shared cell grade (I4, shared grades) and s the policy
  spend from pinned prices and the frame's verbosity, so neither side is a
  miner's self-report. `V = CHAMPION_ACCURACY_VALUE ×` a pinned anchor's
  expected spend per question: `CHAMPION_ANCHOR_MODEL` (Claude Fable 5.1) at
  the reviewed v2 token manifest's mean input and output tokens and the
  pinned v2 prices (`v2_epoch.anchor_spend_per_question`, about $0.0163, so V
  is about $0.49 at m = 30), recorded in the digest as
  `CHAMPION_ANCHOR_SPEND_PER_Q`. One extra correct answer is worth m of the
  anchor's questions, whoever holds the seat.
  **Why pinned.** With the champion's own spend as the anchor, the champion
  set the exchange rate between accuracy and cost: a cheap champion (about
  $0.0004 a question) priced every pricier, more accurate router's spend at
  its own rate, and a router five points more accurate at $0.01 a question
  lost about 80 points of NV to spend and could never take the seat. The rate
  depended on history, and it ran against accuracy first. **What it costs:**
  a router that only spends less, at the champion's accuracy, now gains a
  fraction of a point (saving $0.0015 a question is 0.2 points), so cost
  decides between heads of near-equal accuracy and rarely on its own.
  `u_i = CHAMPION_HARD_WEIGHT` on the hard-reasoning category, fixed by the
  benchmark's category and therefore the same for every head and active from
  the first epoch.
**Why 99% (2026-11-08).** The take test is re-run every epoch, so its false
positives accumulate: at 90%, a challenger with no real edge took the seat in
22-35% of simulated 30-epoch runs, and one of ten sybil heads in 70%. At 99%
those fall to 3-6% and 27%, while a head 2 net-value points better still takes
the seat in every run (simulation: shipped `paired_test`, window 48, 1,500
paired questions, hard weight 2).

- **Take.** A challenger takes the seat only if its point estimate AND the
  one-sided paired-bootstrap lower bound (`CHAMPION_CONFIDENCE`,
  `CHAMPION_BOOTSTRAP_RESAMPLES` resamples within category, both heads'
  answers to a question kept together) exceed `CHAMPION_MARGIN` points, and it
  has at least `CHAMPION_MIN_PAIRED_QUESTIONS` (1,500) paired questions
  against the bar, and passes the accuracy gates: overall accuracy at or above
  a high-water mark,
  and the same on the hard subset once it has
  `CHAMPION_HARD_GATE_MIN_QUESTIONS` questions. The marks are zero until the
  first champion, then that champion's lower bound (Wilson, at
  `WILSON_CONFIDENCE`), then each new champion's. They never fall while a
  champion or its ghost stands; when the last ghost expires with the seat
  empty they reset to zero, so a departed champion's bar cannot keep the
  share burning and the next eligible head can be first. The point
  estimate is part of the rule, not only a shortcut: no bootstrap runs for a
  mean at or below the margin.
- **Eligibility.** To hold a seat, or resume one inside its ghost window, a
  head must be live, not be an exact copy, and have window accuracy (shared
  grades over the window) at or above `CHAMPION_MIN_ACCURACY` (0.30). That
  absolute floor replaces v1's frame-based quality floor, because nothing that
  pays in v2 reads the frame; its job is only to keep a broken head out of the
  seat, and the value is unmeasured on the v2 benchmarks. To
  *take* a seat (first or by challenge) it must pass the same floor and also
  have completed burn-in
  since its last reset or miss (E''), so a champion that misses an epoch
  keeps its seat while a head that is not seated re-earns burn-in first.
- **Hold.** A live, eligible champion keeps the seat with no margin, so two
  near-equal heads cannot trade it on noise. An identical head has NV exactly
  0, so copies of the champion never take the seat, and any number of equal
  juniors earn nothing.
- **First.** With no champion and no ghost, the empty seat goes to the
  eligible head (full burn-in, quality floor, not an exact copy) that passes
  the high-water gates and the copy rule with the highest MEASURED net value
  over its current run (`champion.solo_value`: the seat test's integer NV
  units against an empty baseline, compared as an exact fraction, I1). An
  exact tie, which in practice only a copy produces, goes to the earlier REAL
  first-commit block, then to seniority. Until 2026-11-09 it went to the most
  senior such head, and since every pre-flip commit ties at the floor the
  first seat was a salted draw: N coldkeys bought N tickets and a junk head
  above the floor could win (v2 attack review). Until any
  head is eligible the champion share burns, as in v1. **Pre-flip commits
  tie.** Every first commit at or before the activation block is ranked at
  the activation block, and the tie is broken by sha256 over the activation
  boundary's block hash, the head and its owner (`copy_rule.seniority`,
  `floor_block`/`salt`; published per epoch as the reveal's `v2_seniority`).
  So the first seat among heads committed before the flip is decided by that
  tie-break, not by who committed earliest: committing days ahead buys
  nothing, and the order cannot be ground in advance, because the salt does
  not exist until the activation block and every pre-flip commitment is
  fixed by then. Commits after the flip keep their real block. Any better
  head still displaces the first champion by the margin.
- **Ghost.** A champion that stops being live or eligible leaves a ghost bar
  for `CHAMPION_GHOST_OFF` epochs (missed) or `CHAMPION_GHOST_RETIRE` (its
  owner serves a different head). While it stands, a challenger must beat the
  ghost by the margin, the seat is otherwise empty and its share burns, and a
  champion that only missed resumes its seat if it returns inside the window.
  A missed ghost whose owner then commits a different head is upgraded to the
  retire window, counted from when it first stood, and no longer resumes.
  The windows are as short as the copy rule allows, because an empty seat
  burns.
  **Accepted residual: a champion kept offline.** Absent beyond its ghost,
  a champion loses the seat to the best other head, and a re-serialised copy
  of it (new hash, same routing: its routing is published every epoch, so a
  behavioural copy needs no access to its bytes) that ties it keeps the seat,
  because the original can never beat a tie by the margin; the original is
  paid seat 2. An attacker therefore needs a copy AND to keep the champion's
  axon offline for more than `CHAMPION_GHOST_OFF` epochs (about 3.6 h)
  without a break. The defence is operational, not a seat rule: only
  validators need to reach a miner, each validator records its egress IP on
  chain at startup (`FUGAL_VALIDATOR_IP`, `axon_allowlist.publish_validator_ip`:
  a serve_axon with nothing listening; a validator needs no inbound traffic),
  and `scripts/axon_firewall.py` keeps a GCP firewall rule that admits only
  the active validators' IPs (permit, stake at or above
  `MIN_VALIDATOR_STAKE`, weights in the last three epochs: stake alone is not
  a validator, under dTAO an earning miner holds stake too) (MINER_GUIDE,
  "Protect your axon"). It refuses to apply while any
  active validator has not published one: a miner that one validator cannot
  reach looks offline to that validator alone, and validators would disagree. Why not a seat rule: unlike subnets
  whose validators evaluate a stored model, a Fugal head runs inside the
  miner's TD and is scored only while it serves, so "offline because
  knocked" and "offline by choice" look the same. Every rule that kept an
  absent champion's claim (four designs, eight independent review rounds,
  2026-11-09) either let a rarely-present champion block better heads, let
  copies game the claim, or split validators; a claim limited to heads that
  route like the absent champion is the candidate for a later release.
  `tests/test_v2_attacks.py` (A3).

**What I3 no longer guarantees, stated plainly.** v1's frontier made every
constant policy score zero. v2 has no constant-policy reference: a head that
does not read the question (always one model) can be the first champion, and
holds the seat until a router beats it by the margin on the same questions.
That is accepted. Every later promotion is a paired, margin-sized gain in net
value over the head it replaces, so the subnet still pays only for routing
that measurably beats what is seated, and a constant-policy champion is the
easiest seat there is to take. What still holds: no head raises its standing
except by beating the champion, and extra registrations do not buy seats.

**Accepted residual: withholding.** The comparison pairs only epochs in
which both heads were scored, and an operator sees its own grades before it
serves. A champion can therefore withhold an epoch it did badly on, and that
epoch never enters any challenger's comparison. It is accepted because it is
bounded and expensive: every withheld epoch empties the seat for that epoch,
so the champion forgoes its whole seat share. Measured (v2 attack review,
2026-11-09; `tests/test_v2_attacks.py`): with a champion that knows each
epoch's paired NV, withholding out-earned honest play against a challenger
up to about 2.5x the margin better (0.30 points: 2.3x the earnings, the take
delayed from epoch 15 to 85; 0.53 points: 1.1x) and gained nothing from about
0.6 points up. Counting a champion's missed
epoch as wrong answers in the pairing was considered and rejected: axons are
public addresses, so knocking the champion offline for one collection would
then be cheaper than building a better router.

**Accepted residual: serving the luckiest of several runs (best-of-N).**
A proof binds the hotkey, the epoch's nonce and the head, but nothing makes
it the only run: an operator can run N TDs on one hotkey, each producing a
valid proof, and serve the one that scored best. Unlike withholding it never
misses an epoch, so the miss rule does not price it; it helps a challenger
take the seat and a champion keep it, and different proofs to different
validators split them (one lost proof's worth each, which verify-and-resync
now heals). Measured with the shipped `paired_test` and constants for a
challenger with no real edge, grading each run on its own answers (Fable
review 2026-11-09, item 3): one run 0%, two 6%, three 16%, five 28%, ten 50%
of 30-epoch runs took the seat. That is an upper bound: a head is graded on
the shared cell, the coldkey median (I4), so a lucky run moves a grade only
where few other coldkeys answer the same (question, model). It costs N times
the inference. Closing it needs "one run per epoch": the TD's identity
committed with the head and a measured harness that refuses a second run of
an epoch, a v2.1 image and consensus change.

**The challenger is measured on its current run.** A challenger's paired
epochs are only its current unbroken run of served epochs (back from the
newest epoch to the first one it missed, and from its first commit). An
unseated head is not paid, so sitting out a bad epoch would otherwise be free:
it could withhold its bad epochs and then serve a burn-in's worth in a row,
and be compared on epochs it chose. A miss restarts both its burn-in (E'')
and its pairing, and its take gates read the same run. The champion's side is
every window epoch it was scored in (its withholding is the accepted residual
above). An epoch with no proofs at all is not in the window and breaks no run.

**The window.** The comparison reads the last `CHAMPION_WINDOW_EPOCHS`
epochs by calendar, only epochs in which both heads were scored, and for the
challenger only its current run. Per epoch the window holds what is
common to every head (slice, categories, hard flags, every cell's grade and
spend) and per head only its routing and its failed calls, so it is bounded
and reconstructable from the published reveals.

**The bar is never the field.** What a challenger must beat is the champion
or its ghost and, under the copy rule (next entry), the senior seat holders.
No junior head's presence moves it (I4).

**Determinism (I1).** Each NV_i is computed elementwise and quantised to an
integer number of `NV_UNITS_PER_POINT` units, so every sum is exact on any
CPU; V is a `math.fsum` mean. The resampling stream is numpy's PCG64 seeded
from sha256(epoch id, bar, challenger). (`Generator.integers` is stable within
a numpy version, not across them; numpy's version is in the consensus digest
every reveal carries, so a validator on another version shows it at once.) The lower bound is an order statistic
of integer sums (the k-th smallest, k = round((1 − confidence) × resamples),
0-based: 20 of 2,000 at 99%), and the margin comparison is done in the same
integer units. Per-question NV is clipped to ±`UNIT_CLIP` before it becomes an
integer, so no sum can overflow, and a NaN is refused. Former champions kept
in the seat state are bounded (`HISTORY_KEEP`). Candidates are visited in
seniority order and ties go to the most senior, so
the decision does not depend on the order proofs arrived.

**Superseded margin.** An earlier draft used an absolute headroom margin
(0.004) for both taking the seat and beating a senior head. v2 has one
margin, `CHAMPION_MARGIN`, in NV points.

**Not yet enforced.** The per-call time gate (`CHAMPION_TIME_GATE_FACTOR`)
needs a hash-bound per-call latency in the proof. Until proofs carry it the
gate is not applied, and the published policy says so.

Enforced by `tests/test_champion.py`.

### I3 — seats, not shares: the v2 payout (v2 release; from the activation block)

From protocol v2 the miner share is split across **seats** held by the last three champions
(`fugal_subnet/payout.py`, `PAYOUT_SEAT_SHARES` 0.85 / 0.10 / 0.05: current,
previous, the one before), never across the field.

- **A seat pays only while its head is live and eligible** this epoch
  (verified, window accuracy at or above `CHAMPION_MIN_ACCURACY`, not an exact
  copy: the hold rule, so burn-in is not required and a champion that resumes
  after a miss is paid at once). An unpaid seat's
  share goes to the current champion; with no current champion it burns,
  along with the champion's own share. A seat no champion has filled yet is
  an empty seat and pays the champion.
- **Seats are paid to the UID serving the seat's head.** An owner (the
  coldkey that first committed the head; the copy rule) keeps every seat its
  heads hold. One operator holds several seats only by having won each in
  turn, so registering more hotkeys buys nothing: under proportional pay an
  operator with eight passing heads took eight shares, and under seats it
  takes at most the seats it has won.
- **The burn UID is never a paid seat.** Its share goes to the champion.
- **A ghost keeps seat 1.** While the champion is gone but its ghost bar
  stands, seat 1 is the ghost's: its share burns, and seats 2 and 3 keep
  paying the two champions before it. The seats do not shift down for the
  ghost window (`v2_epoch.run_v2_epoch`,
  `test_a_ghost_keeps_seat_one_and_the_other_seats_do_not_shift`).
- The weight-change cap (`MAX_WEIGHT_DELTA`) still applies to the vector
  afterwards, as in v1, so a change of champion takes several epochs to
  complete.

`validate_seat_shares` refuses a split that does not sum to one, has a
negative share, or pays a later seat more than an earlier one.

Enforced by `tests/test_payout.py`.

### I3 — the v2 miss rule (E'') (v2 release; from the activation block)

v1's miss rule (`apply_miss`) is replaced from protocol v2: it recovers too
slowly from an honest outage. A miss cannot be free either: an operator sees
its own grades before it serves and could withhold a bad epoch. From protocol v2 (`update_scores(...,
protocol_version=2)`):

- **Before burn-in is complete** (`Evidence.burn_in_progress` <
  `V2_BURN_IN_QUESTIONS`, 1,500, counted in scored questions since the last reset or
  miss, undecayed), a miss **resets the evidence fully**
  (`evidence.apply_miss_v2`). A young head cannot shape its record by
  withholding, because withholding restarts it.
- **After burn-in**, a miss **decays the evidence with no phantom zeros** and
  **restarts the burn-in counter**. The record survives an honest outage, and
  the head must re-earn burn-in before it can take a seat. A seated champion
  keeps its seat (holding needs no burn-in; I3, the champion rule).
- **Migration.** `Evidence.burn_in_n` is new; a record written before it reads
  `-1` and counts as its `n_total`, so at the flip every head keeps the
  burn-in its evidence already shows. A state file without the field loads.
  Before the flip, a validator running this code writes `burn_in_n: -1.0` into
  its local state file for every record (v1 accumulation never sets it). That
  is local state only: it is in no reveal, no hash and no score, and it reads
  as "not yet counted".

Simulated over 400 runs of 300 epochs: withholding below-mean epochs earns
3.07 against honest play's 3.19 under E'', where plain decay without the reset
let a young head out-earn honest play by 6.3%. An isolated honest outage costs
about 4.5 epochs. v1 epochs keep `apply_miss` unchanged.

Enforced by `tests/test_miss_rule_v2.py`.

### I3/I4 — v2 pays from the paired rule alone; the frame is informational (v2 release; from the activation block)

From the activation boundary the validator runs one v2 epoch
(`fugal_subnet/v2_epoch.run_v2_epoch`) after verification: shared cells,
the comparison window, the seat (`champion.seat_step`) and the seat payout
(`payout.seat_weights`). Three rules keep it independent of the reference
frame, which in v2 has no paid bootstrap and fills at roughly 0.8 trials per
model per epoch:

1. **Nothing burns for frame confidence.** v2 passes `paid_fraction = 1`.
   Under v1 the unconfident share of a cold frame burns; with an empty v2
   frame that would burn most of the miner share for about a month. v2 burns
   only when the seat is empty.
2. **Eligibility does not read the frame, nor v1 dedup.** v1's quality floor (a
   fraction of the frontier's maximum accuracy) is replaced by
   `CHAMPION_MIN_ACCURACY` on window shared-grade accuracy, and v1's dedup
   clauses (routing similarity, parameter distance, the epsilon clause) do not
   decide v2 eligibility. In v2 a copy is decided by the copy rule
   (`copy_rule.py`): an exact hash owned by another coldkey, or a junior head
   that fails the margin against a senior seat holder.
3. **Spend is measured, not modelled (I4).** A head's spend on a question is
   the pool's prompt tokens plus the cell's coldkey-median attested completion
   tokens (`shared_grades.shared_cell_tokens`), at the pinned prices, not
   the frame's per-model verbosity. Tokens are aggregated exactly like
   grades: failed calls dropped, one vote per coldkey, the median across
   coldkeys, so a coalition below half of a cell's voting coldkeys cannot
   move its spend either. A failed call is graded 0 and costs 0.
   **Accepted residual: thin cells.** The cells that decide a paired test are
   the ones where the two heads route differently, and those are thin by
   construction: often the only other voter is one head. A coalition of as
   many coldkeys as a cell's honest voters moves it halfway, one more flips
   it (`tests/test_v2_attacks.py`, A1). The lever is weaker than the test's
   upper bound: grades and tokens are attested model output a voter can only
   bias (through provider choice on its own account), not choose. Using a
   head's own answer when a cell has fewer than K other coldkeys is the
   candidate fix for a later release.
   **Accepted residual: the miner's account chooses the provider.** Measured
   2026-11-09 (Fable review item 7, a few metered calls): one model is served
   by many providers at different precisions (qwen3.8-27b 24, from 4-bit to
   16-bit; deepseek-v4.1-flash 37), identical harness requests were answered
   by different providers, and OpenRouter applies the ACCOUNT's default sort
   and allowed/ignored provider lists to every request. The harness pins
   model, URL and token budget, not the provider, and does not record which
   one answered. So an operator can steer its own calls to the best or the
   cheapest precision. It moves only that operator's vote in a cell (the
   coldkey median), never the price (pinned) and never the model. The fix,
   for v2.1: a pinned provider per model in the request (`provider.order`,
   `allow_fallbacks: false`), so an account setting can only make that
   operator's own calls fail, and the answering provider recorded in the
   proof; it needs an availability decision per model (single-provider
   outages fail every call to it).

**What still reads the frame in v2, all informational:** the v1 composite,
headroom, quality and thrift (computed and published as diagnostics, deciding
nothing); the exploration detectors (they decide only which samples enter the
frame); and `best_model` / `reference_cost` (they feed only the diagnostics).

**Ownership, seniority and retirement are the copy rule's.** Each epoch the
validator folds the boundary snapshot of commitments (every committed hotkey
with its owning coldkey, read pinned at the boundary) into the copy rule's
state (`copy_rule.observe`). A hash belongs to the coldkey that first
committed it; only its canonical owner UID serves it (`copy_rule.classify`),
so a UID serving another coldkey's bytes is a copy and is not paid as it, not
measured as it, and cannot retire it. A head's seniority and the first epoch
of its comparison are its first commit (`v2_seniority`, `first_epoch`), never
a registration block. A champion is *retired* (24-epoch ghost) only when its
owner stops committing it (`retired_hashes`), and *missed* (3-epoch ghost)
otherwise; a standing missed ghost is checked every epoch and upgraded. A
watched head (the champion or a ghost) with no copy record at all is treated
as retired: the copy rule drops a record whose first retirement is older than
the retire window, and reading that as "missed" shortened a retired ghost to
three epochs (2e66391). A
head takes a seat only by also beating every senior seat holder in the
comparison set by the margin (`senior_bars`, holders from `seat_holders`).
The fold is idempotent across a retried epoch. Payout eligibility is the hold
rule (live, canonical owner, window accuracy at or above the floor), so a
champion that resumes after a miss is paid at once.

**The reveal.** From the activation boundary it carries `protocol_version`,
`shared_grades` (each cell's grade and median completion tokens), each UID's
failed questions, and the seat decision (event, champion, bar, seat state,
and the comparison detail), so every head's grades, spends and NV, and the
seat decision, can be recomputed from the published reveals (the window from
the last `CHAMPION_WINDOW_EPOCHS` of them, the rest from the newest). It also
carries a
`v2_state` block (`v2_epoch.v2_state_for_reveal`): the post-epoch snapshot of
everything the next epoch reads (seat, head owners, the UID chosen for each
head, each head's burn-in progress), the single place later v2 state is
published too, and what a rebuild from reveals reads. The `v2_epoch` stage of
`check_determinism.py` runs the whole v2 epoch on the synthetic proofs in
reverse order.

**State.** The seat and head owners are in the JSON state; the window is an
`.npz` beside it (loaded with `allow_pickle=False`), written before the JSON.
Adding an epoch to the window is idempotent (it replaces that epoch), so a
crash between the two writes, or a `--once` re-run, retries the epoch to the
same result instead of failing it. A seat with no window behind it (the
`.npz` lost) **fails closed**: hold, resume and ghosts continue, but no first
or take decision is made until the window is rebuilt from reveals or has
refilled, and the validator says so every epoch.
All three are durable names, rebound only at the commit point
(`tests/test_epoch_atomicity.py`). On an epoch with no proofs at all the seat
is not advanced; ghost windows are counted in epochs and still expire on
time.

**The weight-change cap still applies.** `MAX_WEIGHT_DELTA` (0.3 per epoch)
limits the vector after the seat payout, as in v1, so a change of champion
takes about three to four epochs to reach the chain in full, with weight split
between old and new during the ramp. The seat itself changes at once.

**Bounded token counts (I2).** `verify_proof` refuses a result whose prompt or
completion token count is outside `0..PROOF_MAX_TOKENS` (ten million): a JSON
integer is unbounded, and every value a validator computes with must be
bounded before it is used (I2).
`run_tee_attacks.py` cases "a token count no call can produce" and "a negative
token count".

**Hard needs the category map.** The hard weight applies to the category
`config.BENCHMARK_CATEGORY_V2` labels `"hard"`; a live validator with the
activation block set and no category map refuses to start.

Enforced by `tests/test_v2_epoch.py`, `tests/test_shared_grades.py`,
`tests/test_champion.py` (re-adding an epoch, no take on an untrusted window)
and the `v2_epoch` stage of `scripts/check_determinism.py`.

### I1 — every epoch input is chosen by the boundary-keyed version (v2 release; from the activation block)

A v2 epoch must be scored on v2's questions, v2's prices and v2's model index
space, by the miner and every validator alike, or every proof fails on
`questions_hash` and the v2 window is indexed in the wrong model space. So
every epoch input that differs by version is chosen by the epoch's
`protocol_version` (the boundary block, I1 above), in one place each:

- **Questions:** `fugal_subnet/epoch_plan.scored_slice(version, …)` and
  `exploration_plan(version, …)`, the only selection functions either neuron
  calls; v1 delegates to the v1 functions byte for byte.
- **Pool:** `loader.load_pool(version)`, v1 and v2 each with its own pinned
  manifest and pool file.
- **Prices and model space:** `routing_protocol.price_table(version)` and
  `models_for_protocol(version)`.
- **Proof namespace and head cost profile:** `verify_proof(...,
  protocol_version=)` requires the proof's `routing_protocol` to be the
  version's (`identity()` for v1, `identity_v2()` for v2, which the harness
  stamps per version) and checks the head with
  `benchmark_costs_for(head, version)`; a v1 proof does not verify in a v2
  epoch, nor the reverse.

The validator loads v1's inputs as before and, when an activation block is
set, v2's as well, at startup; a live validator that cannot load v2's refuses
to start rather than fail at the flip. Each epoch it picks one set by version
before selecting anything. The v2 comparison window starts empty at the flip,
so the two model spaces never meet in one window.

Enforced by `tests/test_v2_epoch_inputs.py` and the `epoch_plan_v2` stage of
`scripts/check_determinism.py`.

### I3/I4 — the v2 copy rule: a head must beat every senior seat holder by the margin (v2 release; from the activation block)

v1 recognised copies by resemblance (`dedup.py`) and ordered seniority by
registration (`commitments.seniority_key`). v2 replaces both: seniority is
the first commit of the head, and a junior head must be measurably better
than every senior seat holder rather than look different from it. From
protocol v2
(`fugal_subnet/copy_rule.py`, enforced inside `champion.seat_step`):

- **Seniority is the head's first commit.** A head's age is the block at
  which its exact weights hash was first committed by any hotkey on the
  subnet, read from the commitments pinned at each boundary. Ties inside one
  block go to the lowest sha256("fugal-v2-owner:" ‖ hash ‖ ":" ‖ coldkey),
  which no hotkey address steers. A head never seen committed is junior to
  every head that was. At the flip, every head committed at or before the
  activation block ties at that block, broken by a salted hash
  (`seniority(..., floor_block, salt)`; I3, "Pre-flip commits tie").
- **A hash belongs to the coldkey that first committed it** (`classify`).
  The same hash under another coldkey is an exact copy and is ineligible; a
  hash a copier still commits stays on record as the author's, so a copier
  never inherits its age. Ownership is by coldkey, so an operator whose
  hotkey is pruned and re-registers keeps its head. A chain coldkey swap
  moves the first committer's hotkey to a new coldkey; ownership follows that
  hotkey (`is_exact_copy` is False for it, and `observe` moves the record),
  while any other hotkey serving the hash under a coldkey that is not the
  owner is still a copy. Two hotkeys of the owner
  serving one hash are one head: the earliest commitment (then hotkey, then
  UID) is canonical and the other earns nothing.
- **A junior must beat every senior seat holder by the champion margin.**
  To take a seat, first or by challenge, a head must beat each senior SEAT
  HOLDER in the comparison set by `CHAMPION_MARGIN` on the same paired test
  (`paired_test`, at least `CHAMPION_MIN_PAIRED_QUESTIONS` paired questions).
  There is no second margin and no resemblance test: a copy is simply a head
  that is not measurably better than its source, whatever it looks like. The
  ranking among heads that qualify is still their margin over the champion or
  its ghost; the copy rule is pass/fail. The v1 clauses stay v1-only.
  **Seat holders** (`seat_holders`) are the heads the payout pays: the
  champion and the former champions in seats 2–3, plus the champion's
  ghosts. Only seats pay, so a copy of a head that holds no seat earns
  nothing anyway; barring it against every older head would turn one 90%
  test into many and let noise against weak seniors block a genuinely better
  head.
- **The comparison set** (from which the seat holders are taken) is every
  head with answers in the champion
  window that is live this epoch; or that its owner still commits and that
  was scored within `CHAMPION_GHOST_OFF` epochs; or that is **retired** (its
  owner no longer commits it: it committed a different head, or its hotkey
  left the subnet) within `CHAMPION_GHOST_RETIRE` epochs of its FIRST
  retirement. These ghost incumbents are unpaid. They exist so that perturbed
  copies of a retired seat holder cannot become incumbents the epoch its
  author moves on: a
  copy that served alongside the head has to beat it on that pairing, and a
  copy that only started after it retired has no pairing and is barred until
  the ghost expires. After that a copy can win, if it is then the best live
  router; that is accepted.

**What the copy rule costs honest miners, stated plainly.** A head that only
ties a senior seat holder earns nothing, even if it is independent. Rewarding ties
is what made copying pay.

**I4: a senior seat holder can delay a junior, and only one can.** The bar
now includes seats 2–3 as well as the champion, so their presence can stop a
junior taking a seat. It is bounded. Only seat holders committed before the
junior count; a head never scored has no evidence and bars nobody, so committing a
hash early buys nothing; a head that misses bars for at most
`CHAMPION_GHOST_OFF` epochs after its last scored epoch, and every epoch it
does serve adds to the pairing, after which a worse senior is beaten; and a
retired head bars for at most `CHAMPION_GHOST_RETIRE` epochs over its
lifetime, because the retirement epoch is the first one and re-committing the
hash does not restart it. A head that retires must have been scored, so each
such delay costs its operator a served epoch.

**Accepted residual: front-running a commitment (until commitments are bound
to the coldkey).** A commitment is the plain public weights hash. Whoever
gets a hash on chain first is its first committer and its owner, so an
operator who sees an author's commitment before it lands (or lands in the
same block) can claim the hash. Bounded: the claimant cannot serve the head
until the author serves it, and the author can re-commit a retrained head.
A commitment bound to the coldkey (`sha256(weights_hash ‖ coldkey)`) closes
it and is planned for a later release.

**State and rebuild (I1, I2).** First-commit blocks, owners and retirements
are history, so every validator must hold the same record. The state is a
fold (`observe`) over the boundary snapshots the validator already reads
pinned at the boundary block (hotkeys registered at the boundary, each with
its owning coldkey), so it depends only on chain state. It is published
after every epoch as plain JSON (`CopyState.to_reveal`: `epoch`,
`first_commit`, `ownership`, `first_hotkey`, `active`, `retired_at`) in the reveal's single
v2 state block, and a validator that loses its state loads the newest
reveal's (`rebuild_from_reveals`, or `from_rebuild` on the result of
`v2_rebuild.rebuild_v2_from_reveals`, which carries it verbatim) or re-derives
it from an archive node (`replay`). Bounds: a commitment is accepted only as 64 lowercase hex
characters with ss58-length hotkey and coldkey and a block in
(0, boundary]; a snapshot is capped at `MAX_COMMITMENTS` entries in
(block, hotkey) order; a record is kept only while some hotkey commits the
hash or while it is a ghost, so the state is bounded by hotkeys ×
(`CHAMPION_GHOST_RETIRE` + 1); and a loaded state is validated field by field
and refused above `MAX_STATE_HEADS`. Every function is order-independent, and
the `copy_rule` stage of `scripts/check_determinism.py` feeds the snapshot in
reverse under `--perturb`.

Enforced by `tests/test_copy_rule.py` and the `copy_rule` stage of
`scripts/check_determinism.py`.

### I9 — v2: the frame is estimated on the scored mix (v2 release; from the activation block)

Exploration draws a handful of slots an epoch, so its benchmark mix rounds
away from the scored slice's (measured on v1: one benchmark was 6.7% of
exploration against 1.7% of the scored slice), and the frame's per-model
accuracy and verbosity were estimates of the wrong mix. From protocol v2 each
aggregated exploration observation (one per (question, model), I9) is
weighted by its benchmark's scored share over its exploration share that
epoch (`slicer.mix_weights`, the same epoch's scored slice and exploration
set), and the frame keeps **weighted sums** (`successes_w`, `trials_w`,
`prompt_tokens_w`, `completion_tokens_w`) beside the raw ones. Accuracy and
verbosity read the weighted sums; the **raw trial count is never weighted**,
because it counts how much has been measured, and that is what the
frontier's confidence reads. A v1 frame has no weighted sums and serialises
byte for byte as before; the first weighted epoch seeds the weighted sums
from the raw history (every past trial at weight 1), so no estimate jumps at
the transition (found 2026-11-05: without the seed a model with 300 raw
trials was valued at its single newest sample), and once a frame is weighted
every later epoch keeps the weighted sums current.

In v2 the frame is informational (I3/I4, "v2 pays from the paired rule
alone"): this keeps the exploration detectors and the published diagnostics
honest, and nothing that pays reads it. The weights are published in the v2
reveal (`frame_mix_weights`, and `explore_benchmarks` for each explored
question), so a rebuild from reveals reproduces them.

Enforced by `tests/test_frame_mix_v2.py`.

### I1 — v2 state is rebuilt from reveals: snapshot + replay (v2 release; from the activation block)

A validator that lost its state, or has just joined, must reach the v2 state
its peers hold, or it seats a different champion (I1). The v2 decision state
cannot be replayed from reveals alone: some of what decided it (dedup
verdicts, per-head evidence, chain commitments, seniority) is not in them. So
`fugal_subnet/v2_rebuild.rebuild_v2_from_reveals` rebuilds it in two parts,
ordered by epoch id and never by filesystem order:

- **Snapshot.** The seat, head owners, the UID chosen for each head, each
  head's E'' burn-in progress, and every later v2 field (the copy rule's
  state) are read from the `v2_state` block of the newest v2 reveal.
- **Replay.** The comparison window is re-derived from the last
  `CHAMPION_WINDOW_EPOCHS` v2 reveals: each epoch's questions, its
  `shared_grades` (grades and median tokens) at the pinned prices, and each
  served head's routing and failed calls. The reference frame's rebuild
  applies the same v2 mix weights (`frame_mix_weights`,
  `explore_benchmarks`).

The validator uses it at startup when it has no local v2 state, or has a
seat but lost its window, and `FUGAL_FRAME_BOOTSTRAP` names a directory of
reveals. Per-head v1 evidence is not rebuilt (as before); only the burn-in
counter is restored, into fresh records.

**A malformed reveal is refused by name.** The replay validates every field it
reads (lists where lists are expected, one routing index per question, every
index inside the model space, well-formed epoch ids) and refuses the file by
path and epoch rather than crashing or, for a non-list `models`, silently
accepting it; a window missing one epoch is a different window from every
peer's, so the operator fixes the file.

**Exact only when the reveals are current.** Published reveals lag by the
commit-reveal hold. A rebuild is exact when the newest reveal is the epoch
just before the one the validator resumes at (its own reveals after a lost
state, or a peer's once the hold has passed). If they lag by k epochs, the
window lacks those epochs and the seat is as of the newest reveal; the
validator then stays degraded (hold, resume and ghosts only; no first or take
decision) until its window has refilled, and says so. Start a rebuilding
validator after the hold.

**The trust model, stated plainly.** A validator that rebuilds trusts the
publishing validator's snapshot, as it trusts a frame rebuilt from someone
else's reveals. That trust is bounded: every reveal's sha256 is committed on
chain by the validator that published it, so a rebuilt snapshot is pinned to
a specific, attributable artifact that cannot be altered after the fact; and
the window part is not trusted at all, because anyone can re-derive it from
the cells and routing every reveal carries. Rebuilding from your own reveals
after a lost disk is self-trust.

**Verify-and-resync: a validator that missed epochs agrees again by itself
(2026-11-09).** Each epoch's proofs can only be observed live, so a validator
that missed a v2 epoch (it was down, or it scored nothing) carries a seat
state that differs from its peers' and, through the seat's history, keeps
differing: before this, it never agreed again. Its peers' state is published
in their reveals only after the hold, so some lag is unavoidable.
`fugal_subnet/v2_sync.py` and `neurons/validator._v2_sync_step`:
- **Shape: a star.** One ROOT validator (the subnet owner's) sets no
  reference; every other validator points `FUGAL_FRAME_BOOTSTRAP` at a
  directory synced from the root's bucket (`<netuid>/` prefix).
- **Gap.** A v2 epoch a follower did not observe (or scored nothing in), or a
  startup rebuild that ends before the present, opens a gap (`gap_until`,
  persisted). A gap changes NOTHING the validator decides: blocking its takes
  only guaranteed it diverged whenever its peers took (review 2). It marks the
  validator's reveals `v2_sync.unverified` (outside the compared view) and
  makes it re-check itself against the reference.
- **Re-anchor, once.** When the reference's reveals reach an epoch k at or
  after the gap that the reference had itself verified, and its consensus
  view (`v2_state`, `weights`) differs from this validator's reveal for k (or
  there is none), the follower adopts the reference's state at k
  (`snapshot_epoch_id`), keeps its own window entries after k, carries the
  burn-in counters over them by the `update_scores` rule, and records the
  epoch (`anchor_at`). Durable at once; a retry recomputes it.
- **Verify.** The gap closes on a match at an epoch at or after both the gap
  and the last re-anchor; a mismatch there re-anchors again. A gap is bounded
  by one window, after which it closes with a warning.
- **Followers converge on the root; the root never marks itself
  unverified.** A follower with no gap that disagrees with its reference logs
  an alarm; once it has differed on `v2_sync.DRIFT_CHECKS` (3) consecutive
  verified reference epochs (`v2_sync.drifted`) it treats that as a gap at
  the newest one and re-anchors. That is how a LOST PROOF heals: a miner's
  reply that reached the root but not this validator (or the other way
  round) leaves no gap, since the epoch was observed, but leaves the burn-in
  counters, and sometimes the seat, apart for good (Fable review, item 5:
  28 of 30 simulated cases never agreed again). In a star the root is the
  arbiter, and agreement is what Yuma pays. More than one check, so a
  difference that heals by itself (the weight cap converging after an
  outage) is not followed. A reveal published while unverified is never used
  to verify or to detect drift, so two validators with gaps cannot confirm
  each other's wrong state.
- **An epoch with no valid proof burns.** In v2 a seat pays only while its
  head is live, and with no proof no head is, so the validator sets the
  capped all-burn vector (`neurons/validator._burn_epoch_weights`); it used
  to set nothing, which left the last vector on chain and, at the flip, kept
  paying the v1 winners until the first v2 proof verified (Fable review,
  item 6). v1 epochs keep main's behaviour.
- **Never stops an epoch (I6).** Any failure (a malformed reference reveal
  included) keeps the state and the gap.
- **Zero weights are not compared.** The weight cap keeps, at 0.0, every
  uid a validator's own previous weights held, so a validator started from a
  rebuild (no previous weights) lists none of them; the SDK drops zero
  weights before `set_weights`, so the chain never sees the difference.
  Comparing them kept a rebuilt validator re-anchoring every few epochs until
  its gap timed out (final review, finding 1; `v2_sync.consensus_view`).
  Agreement checks (`scripts/watch_subnet.py`, `rehearsal_v2`) use the same
  rule via `v2_sync.comparable_reveal`, which also ignores `v2_sync`.
Measured (`tests/test_v2_sync.py`): without it a validator that missed one
epoch never agreed again; with it, it agrees exactly again from `hold` epochs
after the gap after one re-anchor (holds 1-6), including a validator started
from a rebuild of the root's reveals. Seat or copy-rule events inside the
re-anchor span are applied late, so **convergence needs a stretch of about
twice the hold with no seat change**: with a change every 6 epochs and a hold
of 4 it took 14-34 epochs and 3-7 re-anchors; with a change every 4 epochs
and a hold of 2-3 it did not converge inside the window (the gap then closes
with a warning, and the follower alarms). A take needs at least five epochs
of paired questions, so changes that fast are not expected. A network-wide
epoch with no proofs changes nobody's agreement (everyone burns). One lost
proof heals within about the hold plus `DRIFT_CHECKS` epochs (12 simulated
cases, holds 1 and 3). **Accepted residuals:** the root's outages count for
the whole network: an epoch the root missed is followed by every follower
within a few epochs (a champion not seen leaves a ghost and resumes; a
challenger's burn-in run restarts), so the root is kept up and monitored. A
root that lost its state rebuilds from a follower (VALIDATOR_GUIDE, "The
root"; simulated: it agrees again, followers never move). Validators
referencing each other that miss overlapping epochs wait out the window
instead of recovering (use a star).
Nothing here changes a scoring rule: it chooses which state the rules run
from, so it can be improved without a coordinated upgrade.

Enforced by `tests/test_v2_rebuild.py`: the rebuild equals the live state
after 30 epochs with intermittent misses, a validator continuing from a
rebuild produces the same next epoch (weights and reveal) as one continuing
from its own state, reveal order does not matter, later v2 fields are carried
verbatim, and the frame rebuild reproduces the weighted frame.

### I3 — v2 burn-in is 1,500 questions, and the flip starts with an empty seat (v2 release; from the activation block)

Every miner serves a new head at the flip, so every evidence record starts
from zero and no head can take a seat until it has completed v2 burn-in:
`V2_BURN_IN_QUESTIONS`, 1,500 scored questions (five epochs of a
300-question slice) in an unbroken run since its last reset or miss. Until
then the seat is empty and the miner share burns, for about six hours after
the flip. v1 epochs keep `BURN_IN_QUESTIONS` (3,000).

The shorter burn-in does not shorten what a seat demands. The first seat goes
to the best measured eligible head that passes the copy rule (exact ties to
the earlier real commit); and a challenger still needs
`CHAMPION_MIN_PAIRED_QUESTIONS` (1,500) paired questions from its unbroken run
against the champion and every senior seat holder, at the margin, before it
can take a seat. Holding or resuming a seat needs no burn-in.

Enforced by `tests/test_miss_rule_v2.py` (the v2 constant is used in v2 epochs,
v1 keeps its own) and `tests/test_v2_activation.py` (in the digest).

## Attack surface by actor

**Miners** control: the bytes their axon returns, their head weights, when they
commit, and how many identities they register. TEE constrains them: results are
hardware-attested, costs are metered, and the runtime image is measurement-pinned.

**Other validators** control: their own published reveals and weights. Yuma
consensus plus `MAX_WEIGHT_DELTA` bound the damage; `consensus.py` detects it.

**TEE escape:** If a miner breaks out of TDX (extremely unlikely — Intel
patches are fast), they could fabricate results. Defense: measurement pinning
detects tampered runtime images, and DCAP verification validates the attestation
chain against Intel's infrastructure.

**Benchmark datasets** are pinned by revision, so upstream changes cannot
silently alter the question pool.

## Known gaps

**Pool memorization (open, bounded 2026-10-14).** The question pool is public
and finite (21,553 questions in v1; 28,986 in v2, where the slice fills
category quotas first and the same share cap applies within a category). A miner willing to pay once to evaluate every model on
every question could publish a head that encodes the resulting lookup rather
than a routing policy. It would score well here and generalize to nothing.

What bounds it is head capacity: a linear head memorises about 2,000–2,500
questions (`docs/decisions/HEAD_EFFICACY.md`, experiment 2). What the 2026-10-14
diligence review found (finding 6) is that the slicer undid that bound: equal
per-benchmark quotas gave AIME, GSM8K and IFEval — 2,511 questions, confirmed
memorisable as a set (experiment 4) — 180 of 300 slots, so a lookup table the
size of the head's capacity bought 60% of every score. `SLICE_SHARE_FACTOR`
(1.5) now caps a benchmark's share of the slice at 1.5× its share of the pool:
the small three hold 17% of the slice, MMLU 48%, and a 2,500-question
memoriser buys 17% of a score. That is the accepted residual; closing it
needs questions that are not in the public pool, which the design rejects.
**v2 (2026-11-09, Fable review item 4):** v2's small sets are the olympiad
math sets and IFEval (2,118 questions). With the hard category weighted 50%
to them and the hard weight of 2 they carried 27% of the seat test's net
value, and every cell's grade is published, so the lookup table costs
nothing. The hard category now splits by pool size (`BENCHMARK_WEIGHT_V2` is
empty): about 19 olympiad slots an epoch instead of 45, the olympiad set
every 99 epochs (longer than the 48-epoch window) instead of 42, and 14% of
net value (`tests/test_pool_v2.py` holds it at 15% or less).

The intended defence is held-out evaluation: the validator runs the head on
questions the miner has not been scored on and checks it still routes sensibly.
The head is already bound and shipped in the bundle for exactly this — but
running it requires backbone embeddings on the validator, and that is the part
not done. It is deliberately deferred rather than half-built, because it
reintroduces per-validator floating-point computation into consensus, which is
precisely what the TEE architecture removed. Two honest validators whose
embeddings differ in the last bits would disagree on near-tie routing and
diverge. Closing this properly needs either an agreed embedding artifact or
held-out questions that are not in the public pool.

**Benchmark pool reproducibility (closed by the pinned manifest, 2026-09-06;
residual below).** The pool is consensus state, and the default loader did
not produce the same one for everybody. GPQA is a gated
Hugging Face dataset, so `load_all()` raises for an operator without a token and
returns a *larger* pool for one with it. LiveCodeBench loads zero questions on
current `datasets` versions but loads normally for anyone who has manually
placed `data/benchmarks/livecode.json`. Either divergence produces proofs that
fail on `questions_hash`, an error that names the symptom and never the cause.
The pinned manifest (`benchmarks/pool_manifest.json`) catches either
divergence at startup. What remains (diligence 2026-10, finding 8) is the
build itself: `scripts/export_pool.py` materialises the pool once as
`data/benchmarks/pool.json.gz`, which `load_all()` prefers when present and
verifies against the manifest like any override, so a neuron needs Hugging
Face to build the pool and never to start. The file is not a release asset —
the datasets carry their own licences — so operators build it and copy it;
the residual is that a removed upstream revision stops NEW builds.

**The override path was the unguarded one (fixed 2026-09-06).** `load_all`
returned a `FUGAL_BENCHMARK_POOL` file before either the manifest check or the
size floor ran, so the documented production path was exactly the one on which
both guards were absent. Measured: a 150-question pool served 40+ live epochs on
netuid 552 and nothing objected. The override now runs the same two checks; a
pool that is deliberately not the pinned one — a local testnet, a rehearsal
fixture — must declare it with `FUGAL_POOL_UNPINNED=1`, which logs a warning
naming itself. `tests/test_pool_override_verified.py` holds it.

**The approved measurement is reproducible but not durable (superseded).**
*Status 2026-11-06:* the record below predates two changes. RTMR0 was dropped
from `measurement_id` (I8, "two changes made now"), so the machine shape no
longer moves the base; and the durability answer is the dstack guest image
plus the rotation procedure in `docs/APPROVED_ENTRY_ROTATION.md`, under which
`--live` has run on mainnet since 2026-11-03 (the base did move once, with a
GCP firmware roll, and was rotated by that procedure). Kept as the record of
the gap. At the time: `measurement_id` was sha256 over MRTD and RTMR0-2. Measured on real
TDX hardware rather than reasoned about: two independently created
`c3-standard-4` instances, both pinned to `ubuntu-2404-noble-amd64-v20260903`,
created 15 minutes apart with the first deleted in between, produced a
**bit-identical** measurement — every register matching. Under the current
formula that image measures
`a1ecb6273d38bd8b5ad629edf12f48d1c3b861fa8f506d1e1e303783e68461f3`.

So the value is reproducible today, provided two variables are pinned:

- **the exact image version.** `--image-family` resolves to whatever is newest
  at create time, so two runs a week apart silently measure differently.
- **the machine shape.** RTMR0 moves with it; a 4-vCPU and an 8-vCPU instance
  measure differently from identical software.

What remains open is durability, not reproducibility. A cloud provider
eventually retires a pinned image version, so an approved list built this way
has an expiry date, and rotating it means changing a consensus parameter on a
schedule set by someone else. There is no rotation procedure, and that is what
blocks `--live` on mainnet. A reproducibly-built guest image (dstack, or one
built in-house) removes the dependency entirely and remains the right long-term
answer — but it is not required to obtain a working measurement now.

**Code benchmarks are excluded (open, tracked).** The harness grades with
`allow_exec=False`, and `exec_io`/`exec_unittest` — the checkers for humaneval
and livecode — return 0 unless permitted. So those questions scored zero for
every miner regardless of the answer, and because the slicer equalises
benchmarks that was a full sixth of every graded slice, not humaneval's 0.8%
share of the pool. Worse than dead weight: miners paid API cost for an
impossible score, so routing code to a capable model was punished on thrift with
no reachable quality, training heads toward the cheapest model on exactly the
domain where routing matters most.

`allow_exec=False` is correct as it stands. The sandbox in `graders.py` is
process-level with no filesystem, namespace or network isolation, and under
`--live` the miner controls what the metering proxy returns — so executing it
inside the TD would let a miner run chosen code in its own enclave and obtain a
genuine attestation over forged results. The pool now excludes what the harness
cannot score, derived from `config.HARNESS_ALLOW_EXEC` so that enabling
execution re-includes the benchmarks in one place.
`tests/test_grader_policy.py` fails if the policy and the pool drift apart
again, in either direction. Putting them back is
[CODE_BENCHMARK_PLAN.md](decisions/CODE_BENCHMARK_PLAN.md); its final phase needs the
entry rotation procedure, which now exists (`docs/APPROVED_ENTRY_ROTATION.md`).

**The model upstream is miner-controlled (fixed 2026-10-14; was OPEN, breaks I3).**

*Status.* Closed by making the application binding mandatory. The upstream is
fixed by the dstack compose (the frozen `app-compose.json` has `allowed_envs:
[]` and does not set `FUGAL_OPENROUTER_BASE`, so the default in `runtime.py`
is the only value a measured TD can have); the compose hash is replayed out of
the RTMR3 event log in `verify_proof` and compared against the
`<base>:<compose_hash>` entry. What was missing, and what the 2026-10-14
diligence review called critical, was that a bare `<base>` entry skipped that
check entirely and a live validator only *warned* about one. Now:

- `parse_approved(..., require_app=True)` refuses a bare base, and the live
  validator calls it that way at startup (`scripts/check_safety_invariants.py`
  `check_live_requires_app_identity` reads the AST for the keyword);
- `verify_proof` refuses a bare-base match outside mock whatever the caller
  configured, so scripts that verify directly get the same answer;
- `check_upstream_is_inside_the_measurement` fails CI if any frozen compose
  allows an env, references an image by tag, or sets the upstream, and if the
  proxy ever stops validating the upstream certificate;
- `run_tee_attacks.py` carries the case ("bare base entry, miner-chosen
  application"), and `tests/test_finding1_upstream_binding.py` pins both halves.

Mock mode still accepts a bare base: a local testnet binds no application, and
dress rehearsal depends on it. The § "I8 — verified on hardware: the upstream
binding" section above describes the single-extend `runtime_identity`, which
remains advisory on an unlocked image; the compose-hash path is the enforced
one. The analysis below is kept as the record of the gap.

`fugal_subnet/tee/runtime.py` reads the endpoint the metering proxy calls from
`FUGAL_OPENROUTER_BASE`, an environment variable, inside the miner's own TD. The
pool the TD loads carries `gold` for every question. Before 2026-10-14 nothing
in `proof.py` or `verify.py` bound the upstream unless the operator chose to.

So a miner points that variable at a server they run, which answers every
question with its own gold answer and reports two tokens of usage. The result is
perfect accuracy at near-zero attested cost, and every other check still passes:
questions_hash, nonce, weights_hash, content_hash, report_data, DCAP, and the
approved measurement. The proof is *honest about a dishonest computation*, and
the miner takes the subnet.

**This survives a perfect TDX deployment.** A fully measured image with the
upstream delivered as a runtime environment variable is still broken, which
makes it a requirement on the measured image and not merely a bug behind it:

- the upstream must be inside what is measured (for dstack, `app-compose.json`,
  which is hashed into RTMR3 — *not* dstack's encrypted env, which is delivered
  after attestation and is therefore unmeasured by design)
- container images must be referenced by digest, never by tag, or the hash is
  stable while the code under it is not
- TLS validation must remain intact, or a miner who controls the VM's network
  reaches the same outcome by pointing DNS at themselves

Until 2026-10-14 nothing in the repository prevented this. The only thing
resembling a control was a line in `docs/LIVE_API_VALIDATION.md` asking the
operator not to set the variable — a request addressed to the adversary — and
a validator-side warning when a bare base entry was active, which is the same
request in a different log.

**Why it went unnoticed for so long is the instructive part.**
`scripts/stub_upstream.py` *is* this attack. It was written as a no-spend
testing tool, it works exactly as intended, and nobody asked what it meant that
it worked. A mechanism that lets a local test substitute the model layer is the
same mechanism that lets a miner substitute it in production. Building the
exploit as a convenience and never naming it as one is how a hole this size
stays invisible.

### I3 — the scored cost is the policy's, not the miner's

`thrift = reference_cost / miner_cost`. The denominator was already computed
from shared data — pinned rates plus the reference frame's measured per-model
verbosity — and the numerator was whatever the miner reported it had spent. Even
the denominator's prompt tokens came from the miner's attested counts. One side
of the ratio was unfakeable and the other was a self-report, so a miner could
raise its score by shrinking numbers rather than by routing better. On unmeasured
code with a miner-settable upstream, those numbers are entirely theirs.

Both sides are now computed by the validator:

    cost(question, model) = tokens_in(question) x rate_in(model)
                          + typical_completion(model) x rate_out(model)

Every input is already consensus state: the question pool, the pinned price
table, and the frame's pooled, decayed per-model observations.

**What is being priced is a policy, not a benchmark run.** The product's
beneficiary is a future user of the head, who cares what the policy costs
*them* — not what one miner happened to spend on one run in one hour. So the
scored cost depends only on which model was chosen for each question, which is
the thing being measured, and on nothing a miner can vary.

An estimate, not an invoice, and not trying to be one: the pinned table already
diverges from real billing by design so that two miners benchmarking hours apart
face the same denominator. Input tokens use a fixed characters-per-token
convention rather than a real tokenisation, because every model tokenises
differently, a validator should not gain a tokeniser dependency to price a
proof, and the figure is a relative signal where consistency beats precision.

The miner's reported costs stay in the proof and are still checked for internal
consistency — a miner lying about them is worth knowing — but they decide no
score. That check is now defence in depth rather than load-bearing.

**Price table accuracy still matters, for a different reason.** It no longer
guards against cheating; it defines what "cheap" means, so an error steers which
models the subnet learns to prefer. An under-priced model looks cheaper than it
is and gets over-routed. That is why rates are corrected on their own merits —
see the glm-5.2 correction — and why the five models whose real billing cannot
be modelled by any per-token pair are a product question rather than a scoring
one.

**Price table staleness.** `data/models.json` is hash-pinned, so scoring is
deterministic, but it does not track provider price changes on its own. The
metering proxy records the provider's reported cost alongside the table price
so drift is *detectable*; acting on it is a manual, deliberate update.

## The dress rehearsal

Everything above runs in-process against a mocked chain. `scripts/dress_rehearsal.py`
runs the shipped binaries — `neurons/miner.py` and `neurons/validator.py` as
real OS processes — against a real local subtensor node, with real wallets,
real registration, real axon/dendrite traffic and real `set_weights`.

That distinction is not cosmetic. The TEE pipeline shipped with three fatal
bugs behind a green CI because CI exercised a different path than production
did, and the first real-chain run surfaced eight more that no in-process test
could see: the neurons' own logging silently disabled by importing bittensor,
`--once` never exiting, a crash in the reveal block, weights reported as set
while the chain held none, a miner rendering itself unreachable by ordering two
extrinsics wrongly, the pool re-embedded every epoch, the two neurons loading
different question pools, and epoch geometry duplicated across both.

| | Asserts |
|---|---|
| **A** | A proof verifies against a real chain; weights land **and are confirmed** |
| **B** | Dedup disqualifies a copy and not the original; a real router outranks an always-cheapest one |
| **C** | Two independent validators produce byte-identical weights and frames (I1, I9) |
| **D** | Evidence accumulates, the frame fills, weight capping engages, weights confirm every epoch |
| **E** | The real backbone path works end to end and a full bundle round-trips over a real axon |

```bash
python scripts/dress_rehearsal.py --scenario all
```

### The production rehearsal on netuid 552 (2026-09-06/07)

The record is `docs/decisions/REHEARSAL_2026-09-06.md`, every line labelled measured or
read. In one sentence: two `--live` validators on different CPU architectures
scored three consecutive epochs from two dstack TDX miners' real, DCAP-verified
proofs with byte-identical scores, weights and reveals, set those weights on
chain at the subnet's tempo, rejected a mock miner's proof and a genuine proof
against a retired entry, and did so while the entry was rotated four times.

It found and fixed, in order: a TD with no way to receive its hotkey (I8); a
pool-file path that skipped both pool guards (I4); a provisioning channel that
authenticated the TD and then sent secrets in plaintext (I8); a miner whose
whole epoch aborted on one null model reply (I6, miner side); a serial
benchmark that could not fit the collection window on real providers (I6,
miner side); and an operator tool that had drifted from the verifier's
contract. None of them was reachable by any test in this repository, because
every one lived in the gap between a stub and a provider, a guide and a tool,
or a design and its allow-list.

## Running the checks

```bash
uv run python scripts/check_safety_invariants.py           # structural invariants + TEE safety
uv run python scripts/check_safety_invariants.py --release # + the launch gate (composes frozen)
uv run python scripts/check_determinism.py                 # I1, same-host
uv run python scripts/check_determinism.py --perturb       # I1, simulated second host
uv run python -m fugal_subnet.attacks.run_attacks          # I3, hostile model output
uv run python -m fugal_subnet.attacks.run_miner_attacks    # I2/I6, hostile miner input
uv run python -m fugal_subnet.attacks.run_tee_attacks      # I8/I3, forged proofs
uv run pytest -q                                           # I4, I9, evidence, TEE, end-to-end
```

All of these run in CI on every push. None requires a chain, a network, or API
spend.

## Before mainnet

See [OPEN_WORK.md](decisions/OPEN_WORK.md) for what is unfinished in the PCCS/collateral
work and what to do next, and [MAINNET_LAUNCH.md](MAINNET_LAUNCH.md) for the
operational sequence. Item 1
below was done on netuid 552 and turned up a defect no local run could: the
subnet had never been activated. `start_call` was never made, so it had no
first-emission block — staking was rejected, no validator earned a permit, and
every `set_weights` failed while the neurons reported success.

1. ~~Deploy on testnet with two validators and verify they produce identical
   weights from the same set of TEE proofs~~ — **done 2026-09-07/08**, three
   consecutive epochs, identical scores, weights, reveals and frames (I1, I9).
2. ~~Run miners on real Intel TDX VMs and verify DCAP attestation
   end-to-end~~ — **done**, GCP `c3-standard-4` under dstack 0.6.0-rc0.
3. ~~Publish approved entries and their rotation~~ — **done 2026-11-03**:
   the procedure is `docs/APPROVED_ENTRY_ROTATION.md`, the channel is
   `.github/workflows/image.yml` + `scripts/publish_approved_entry.py` +
   `scripts/fetch_approved_entries.py`, the first signed releases are
   `image-v2026.11.03b` and `03c`, and the 03b → 03c rotation ran to its
   published retirement date (2026-11-06).
4. ~~Decide I3 (`docs/decisions/I3_DECISION.md`) before launch~~ — **decided
   2026-09-08** (frontier scoring) and **re-fixed 2026-10-14** (raw-mean
   valuation, distinct-trial confidence; I3 subsections above).
5. Second independent validator operator; coldkeys encrypted and off every
   neuron host; monitoring with an alert channel. *2026-11-06:* four
   validators hold permits on netuid 70 and the alerting watch ships
   (`deploy/systemd/fugal-watch.*`); whether the operators are independent is
   an organisational fact, not a repository one.
6. ~~Recalibrate the reference-frame prior from real testnet data.~~ The prior
   no longer values the frontier (I3, 2026-10-14). ~~What remains is to ship
   `data/reference_frame.json` and pin it.~~ Shipped 2026-11-03, built from
   direct logged calls (`scripts/build_frame_from_calls.py`, item 3 above) and
   pinned.
7. ~~Close or consciously accept the pool-memorization gap above.~~ Bounded
   2026-10-14 (`SLICE_SHARE_FACTOR`; a memoriser buys ~17% of a score) and
   the residual is accepted — see "Pool memorization" under Known gaps.
8. ~~Validate real TDX attestation on a confidential VM~~ — **done** on GCP
   `c3` under dstack 0.6.0-rc0 (`docs/TDX_VALIDATION.md`): proofs verified
   `UpToDate`, a genuine proof refused against a retired entry; live on
   mainnet since 2026-11-03.
9. Validate the cost path against real OpenRouter —
   `docs/LIVE_API_VALIDATION.md`. **Measured once** at account level (within
   ~5% over four miner-epochs) and per model in the v2 token run
   (`data/benchmark_tokens_protocol_v2.json`, `billing_check`). Repeat on the
   pinned v2 pool before the flip (MAINNET_LAUNCH, "The v2 release").
10. ~~First signed approved-entry release~~ — **done 2026-11-03** (item 3).
11. ~~A public-read reveal bucket on every validator~~ — **done**: reveals
    publish to `gs://fugal-reveals/70/<epoch_id>/reveal.json`
    (`docs/MINER_GUIDE.md`, "Reading the published reveals").
12. ~~Confirm `config.FINNEY_GENESIS_HASH` against a finney node~~ —
    **confirmed 2026-11-05** (`get_block_hash(0)` on finney matches the
    constant; review F-078).
13. Protocol v2 (added 2026-11-06): `docs/MAINNET_LAUNCH.md`, "The v2
    release, in order". A v2 epoch has run only on the local chain
    (`dress_rehearsal.py --scenario v2_flip`, 16/16 on 2026-11-05); the
    testnet rehearsal of the flip is what verifies it on a public chain.

## Success protocol transition

Success heads share core's versioned CPU embedding and sigmoid-minus-dollar selector.
Benchmark λ is exactly 1. Token means must equal the versioned recorded manifest;
prices remain pinned. Contract/profile/λ/manifest/prices are bound into the protocol
identity, runtime identity, proof content, saved validator state and frame/reveals.
Unknown or pre-transition identities fail admission; no old evidence is resumed.
`tests/test_success_contract.py` enforces these boundaries, and the reference module
is checked against its exact core source in CI. The candidate manifest blocks live
startup/export pending review. See `SUCCESS_CONTRACT.md` for the reset runbook.
