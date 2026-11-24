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
| **I1** | **Determinism.** Same epoch inputs ⟹ byte-identical scores on any honest validator. | `scripts/check_determinism.py` (both modes, in CI — 35 stages: the live path, and `menu_derivation`, `seat_v4`, `explore_shares_v4`, `endpoint_vote`, `token_means`, `provisional_gates`, `retirement`, `menu_publication`, `head_odds_v4`, `pool_mix_v4`, `bound_v4`, `pick_cost_v4`, `picks_v4`, `memo_v4`, `v4_replay` and `published_v4` for the v4 menu state, the labels, odds and seat, exploration, the menu step's inputs, the menu's publication, heads, the question mix, the seat's one bound, the pick-cost table, the real-picks check, the memorisation check, a re-anchor's replay of its own epochs and the published set as nodes move it, built ahead of v4), `fugal_subnet/determinism.py`, `check_safety_invariants.check_epoch_id_single_source`, `slicer.protocol_version_for_boundary` (`tests/test_v2_activation.py`), TEE proof verification (all validators verify the same attested proof) |
| **I2** | **Bounded ingestion.** No miner-supplied bytes reach deserialization, allocation, or execution without size, shape, and value bounds. | `run_miner_attacks.py`, `tests/test_head_properties.py`, `tests/test_head_v4.py`, `check_safety_invariants.py` (no-pickle) |
| **I3** | **Monotonic incentive.** A miner cannot raise its score except by routing better or more cheaply, and **no policy that does not read the question can score above zero**. Artifact-keyed evidence with miss=0 prevents selective publication; the burn-in ramp prevents penalty-washing by reset. | Constant-policy frontier (`frontier.py`, `tests/test_frontier.py`, `tests/test_degenerate_constant_policy.py`), the v2 copy rule (`copy_rule.py`, `tests/test_copy_rule.py`), the v2 champion rule (`champion.py`, `tests/test_champion.py`; constant policies are no longer zero by construction in v2, see its entry), commit-reveal, behavioural dedup (global model index), evidence accumulation, `run_attacks.py`, `run_tee_attacks.py` |
| **I4** | **Non-interference.** A miner cannot lower another miner's score, prevent them being scored, or move the reference they are scored against — including by being present or absent. | `tests/test_non_interference.py`, TEE architecture (no shared model pool), nonce-derived exploration targets, reference frame pooled over time, coldkey-median shared grades from v2 (`tests/test_shared_grades.py`) |
| **I5** | **Bounded spend.** No miner behavior can make a validator exceed its budget. Validators verify proofs — zero inference cost. | TEE architecture (miners pay their own inference), `tests/test_paid_safety.py` |
| **I6** | **Liveness.** No miner behavior **and no external dependency** can stop a validator completing an epoch and setting weights, and the validator collects at a point where proofs can exist. Every call that leaves the process on the epoch path is bounded. | `run_miner_attacks.py`, property test P1, TEE proof timeout, `slicer.collect_block_for_epoch`, `tests/test_collection_point.py`, `tests/test_collateral_timeout.py`, `check_safety_invariants.check_external_calls_bounded` |
| **I7** | **Auditability.** Any divergence between two validators is diagnosable after the fact from published artifacts. | `fugal_subnet/fingerprint.py`, `environment` block in every `reveal.json` |
| **I8** | **TEE integrity.** Every claim a proof makes is bound to something the miner cannot forge: the hardware's own measurement registers, or a hash chain rooted in the attestation. No miner-influenced code executes inside the enclave that produces the proof. | `fugal_subnet/tee/verify.py`, `attestation.measurement_id`, `tests/test_app_identity_chain.py` (the app-identity chain through `verify_proof`), `run_tee_attacks.py` (34 cases, in CI), `check_tee_safety`, `config.HARNESS_ALLOW_EXEC`, `tests/test_grader_policy.py` |
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

**The operator's own check builds the same chain (2026-11-18).** A validator
reads the chain from the proof. The operator has no proof yet when it checks a
new TD (`provision.push`, `provision_td.py --inspect`), and held the TD to the
vendored us-central1 intermediate alone: a TD in any other region was refused
before it could be given its key and head, so it never produced a proof for a
validator to accept (issues 129 and 179). `provision.operator_tpm_chain` now
builds the chain a proof would carry, with `tpm.intermediates_for`, and
`provision.unwrap_td_attestation` holds the TD to the pinned root through it.
Nothing about what a validator accepts changed, and nothing a validator runs
fetches (`tests/test_tpm_other_region.py` reads `verify.py` and
`neurons/validator.py` for it). Three things are particular to the operator's
side:

- The issuer address comes from a certificate that a machine not yet
  identified handed over, and the request is made from the machine that holds
  the wallet. Only Google's CA content addresses are asked
  (`tpm._GOOGLE_CA_ADDRESS`, the whole address matched), and a redirect is
  refused, so the answer to `/provision/attest` cannot aim a request at a local
  file, a metadata service or the operator's network. The rule decides where a
  request may go and nothing else: anyone can publish a CA on Google's CA
  service, and what is believed is still only what chains to the pinned root
  by signature (`run_tee_attacks`: own CA published at a Google CA address;
  issuer address aimed at the operator's machine).
- `--tpm-intermediate FILE` supplies the intermediate and skips the fetch.
  It is untrusted like the proof's own: a file that does not chain changes no
  verdict.
- Inside the TD the miner's fetch keeps an unrestricted host (a host rule
  compiled into the image would stop every miner outside the vendored set the
  day Google moved its CA content, until a new image) and takes http(s) only.
  A miner with no chain asks again before each epoch
  (`neurons.miner._tpm_intermediates_for_epoch`): one failed fetch at boot
  used to mean refused proofs for the life of the process.

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
    each event's digest                 binds its (name, payload), either encoding
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

**The second link was missing until 2026-11-05.** `replay_event_log` proved
that the SEQUENCE OF DIGESTS reproduces RTMR3 and then read the `(name,
payload)` pairs beside them as if the two were one thing. They are not: a
miner keeps every genuine digest, rewrites the `compose-hash` payload to an
approved value, and the log still replays to the register the CPU signed —
verified on the real fixtures, where the forged log passed `verify_proof`.
dstack computes each digest from the event's type, name and payload (two
encodings; the GCP fixtures use the first), so the replay now recomputes it
and refuses an event whose digest binds neither.

**Each digest has one reading, and each name one value.** The first encoding
joins type, name and payload with `:`, so a name that holds `:` can absorb
payload bytes: a miner varies its own compose until the hash starts with
`:`, presents the genuine `compose-hash` digest as the name `compose-hash:`,
and a later extend under `compose-hash` becomes the only one, over the same
register. So event names are an allow-list, `[A-Za-z0-9._-]{1,64}` (every
name dstack extends fits; checked on both real logs): no `:`, nothing empty,
nothing non-ASCII, where `json.dumps` and the canonical JSON of the second
encoding would part ways. A name that appears twice is refused too: the
initrd extends each event once before any container runs, and a second one
can only be the application's. `tests/test_tee.py`, `tests/test_dstack_blob.py`
(on the real logs) and four `run_tee_attacks` cases hold it, and
`tests/test_app_identity_chain.py` holds every link of the chain through
`verify_proof` itself, including the log the live validator actually reads
(the one inside the miner's attestation): each check, disabled on its own,
fails that file without the attack suite.

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
bytes, replayed out of the RTMR3 log. That is the only live app half:
`parse_approved(require_app=True)` accepts exactly 64 hex in each half, and
`verify_proof` checks nothing but a compose hash read from a log that replays
to the quote, so a proof with no log is refused. The sha384 `runtime_identity`
below was a userspace extend the miner's own code performs, so it is evidence
of nothing and is not accepted live; the shape argument is unchanged.)

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
   the harness when there is no reply to grade (None from `_call_model`, as
   distinct from "" for a reply with no text): the call did not complete, or
   what the upstream returned carries no `choices`. The second case is an
   error inside a 200 body (`{"error": ...}`), or a body that is not JSON or
   is cut short. The metering proxy retries it within the same attempts and
   wall-clock deadline as a 429/5xx status and relays only a reply that has
   `choices`; an operator status as the body's error code (401, 403, and 402
   other than OpenRouter's temporary in-flight-budget refusal), or an error
   code or type naming a content-policy or moderation refusal, is not
   retried. A reply with `choices` and no text is still a wrong answer.
   Any 200 body that reports tokens used is metered, answered or not, so a
   failed result can carry a cost. `failed` is hash-bound, and is held to
   its meaning by `verify_proof`: a failed result cannot be correct and cannot
   carry a reply. `failed` enters `content_hash` only when True, so a proof
   with no failures hashes as it did before the field existed. Enforced on
   the miner side by `tests/test_proxy_error_bodies.py`.
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

*Protocol v4:* v1 dedup is not run for a v4 epoch (`validator.main`); a near-copy is held by rule C, precedence (the entry "I3 — v4: the seat in three rules" (package S4)).

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
The heads a published reveal names go to the same store by hash ("verified
heads are published by hash", below).

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

**After the flip, finney refuses the dangerous failures at startup**
(`neurons/validator._check_chain_configuration`, once the chain's genesis
hash is read; live only). Each was a configuration a mainnet validator
actually ran with, healthy by its own log:
- `FUGAL_V2_ACTIVATION_BLOCK` unset, or not finney's flip block
  (`config.FINNEY_V2_ACTIVATION_BLOCK`): every epoch scored by the v1 rules.
- No approved entry for a v2 miner image (`v2_compose_hashes`: the frozen
  composes in the checkout that set the activation block; an approved entry's
  second half is the sha256 of one). Every v2 proof then fails the
  runtime-image check, the seat stays empty and all weight burns.
- No public IPv4 in `FUGAL_VALIDATOR_IP`. Miners admit only the validator
  addresses recorded on chain, so the validator verifies fewer proofs than
  its peers and its seat can differ from theirs.
Other chains and mock keep their warnings: a rehearsal legitimately runs
without some of these. `tests/test_follower_setup.py`.

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

**The letter rule reads one letter: a marker, then a letter line, then the
tail (changed 2026-11-10, before the flip; v0.3.1).** `letter_mcq`
(`graders.py`) grades MMLU, MMLU-Pro and SuperGPQA: 573 of the 900 questions
in the first three v2 slices on testnet 552. It tries, in order, and takes the
last match of the first rule that matches:
1. an explicit marker with the letter a capital standing alone: "answer is
   X", "Answer: X", "answer is option X", or a heading ending in "answer"
   with a marked-up letter opening the next line;
2. a line that is only the letter (heading marks, markdown, brackets);
3. a marked-up letter opening the reply, with text after it on the line;
4. the last standalone capital A to J in the final 40 characters.

Until then it was rule 1 without regard to case and without requiring the
letter to stand alone, then rule 4. "the answer is approximately 4.2 eV …
option C" read A, "The answer is clearly (B)" read C, and a reply that put
the letter on its own first line and added working after it was read from
its last 40 characters only. A miner reported it for
`anthropic/claude-fable-5.2` on SuperGPQA (121 of 500 correct answers graded
0). **Measured 2026-11-10** on 1,703 real replies (the 19 v2 models, 50
SuperGPQA + 30 MMLU-Pro + 10 MMLU questions each, through `MeteringProxy` and
`harness._call_model` with each model's pinned reasoning setting):

| | graded correct, old rule | new rule |
|---|---|---|
| All replies (1,703) | 1,321 | 1,363 |
| SuperGPQA (944) | 684 | 713 |
| MMLU-Pro (569) | 476 | 489 |
| MMLU (190) | 161 | 161 |
| claude-fable-5.2, SuperGPQA (50) | 29 | 39 (its stated letter equals gold in 39) |

No reply graded correct by the old rule is graded wrong by the new one. Ten
models gain (fable-5.2 12, mimo-v2.6-flash 7, glm-5.3-flash 6, glm-5.3 5,
minimax-m3.1-flash 4, opus-5.5 2, qwen3.8-27b 2, mimo-v2.6-pro 2, sonnet-5.5
1, haiku-5.5 1); every gain was read by hand and is the model's own stated
letter. The old rule under-measured exactly the models that explain their
answer, so routers trained on it would have avoided them on a quarter of the
slice.

**Why before the flip, and why it is confined.** `grade()` is called only in
the TEE harness, inside the miner image; a validator never grades, it carries
the grader's hash in its epoch commitment and in the consensus digest. So the
change is this file, its pin (`check_safety_invariants.py`, `GRADER_SHA256`)
and a new miner image with new approved entries; no validator scoring code
changes. Done before v2 activates, it costs one image and one notice instead
of a second coordinated flip after it, and no v2 head is trained against the
old readings. v1 epochs are graded by v1 images until the activation block
and are unaffected; a validator on this release shows the new grader hash in
its commitment and digest from the moment it upgrades.

**What it still does not read (accepted):** a reply with no text (37 of the
1,703: reasoning used the whole budget) and "none of the options" grade 0,
correctly; an answer naming two letters reads at most one; a letter given
only inside running text more than 40 characters from the end, with no
marker and no letter line, is not found (none in the sample: the 27 replies
graded 0 that mention the gold letter were read by hand, and each gives a
different final answer or two letters); rule 4 still reads the pronoun in "..., I think" as I (the attack
suite's KNOWN case). Exactly one letter is ever read, so naming several
cannot score, and a reply is attested model output a miner cannot shape.

Enforced by `tests/test_letter_mcq.py` (shapes from the measured replies),
the letter cases of `fugal_subnet/attacks/run_attacks.py`, and the grader pin.

**Determinism.** Every step is order-independent and the published form is
sorted (`cells_for_reveal`); the `shared_grades` stage of
`check_determinism.py` feeds proofs in the opposite order to the frame stage.
From the activation boundary the reveal carries `protocol_version` and
`shared_grades`, so anyone can recompute every head's v2 grades from the
reveal alone; a v1 reveal is byte-for-byte unchanged.

Enforced by `tests/test_shared_grades.py` and the `shared_grades` stage of
`scripts/check_determinism.py`.

### I3 — one champion; taking the seat needs the margin, holding it needs none (v2 release; from the activation block)

*Protocol v4:* none of this entry's v2/v3 rules decides a v4 epoch's seat (the bootstrap test, ghosts and resume, the high-water marks, the accuracy floor, the hard weight, the first-fill ranking): the entry "I3 — v4: the seat in three rules" (package S4). It stays while a v3 epoch can still be rebuilt.

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
  **Accepted residual: the marks are absolute.** The seat is accuracy-first:
  net value decides only among heads at or above the mark, so a head that is
  cheaper but below it does not take the seat at any price (intended). The
  mark is fixed when a champion is seated (about 1.6 points under its
  accuracy for a champion seated on 1,500 questions at 90%) and does not
  follow the field down. If every head's measured accuracy later falls, as
  when a provider changes a model, challengers are measured against the old
  mark: simulated with the shipped `seat_step` and constants, a challenger
  one point more accurate than the champion took the seat in 20 of 20
  60-epoch runs with no drop and in 12 of 20 after a 3-point drop for
  everyone. A mark measured on the paired questions would follow the field;
  it is a consensus change (v2.1).
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
  owner serves a different head; `CHAMPION_GHOST_RETIRE_CONFIRM` from the
  confirmation activation block, next entry). While it stands, a challenger must beat the
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

### I3 — a take is confirmed on fresh questions (from the confirmation activation block)

*Protocol v4:* there is no confirmation phase, pending record or open period in a v4 epoch: the bound is valid at every look (the entry "I3 — v4: the seat in three rules" (package S4); I1/I3 "v4: one bound").

**The property.** Registering more heads must not buy more chances at the
seat. The take test above is re-run every epoch for every challenger, so its
false positives add up over epochs and over challengers: k heads with no real
edge over the champion pass it by luck about k times as often as one does
(the best-of-N figures above: one of ten heads with no edge took the seat in
27% of 30-epoch runs at 99%). A pass is therefore a *selection*, and a
selection is only evidence on data it did not see.

**The rule** (`champion.seat_step`, `Pending`, `_live_pending`). From the
first epoch whose boundary is at or after the confirmation activation block:

- **Pass.** A challenger that passes every condition of a take against a
  settled champion does not take the seat. The epoch is recorded: a *pending*
  record (head, epoch of the pass, the champion it passed against) in the
  seat state. The champion holds the seat and is paid.
- **Confirm.** At a later epoch the challenger takes the seat when it again
  passes every condition of a take on its whole run, AND passes the same
  paired test (the same margin, confidence and resample count; a resampling
  stream of its own, seeded with the epoch id suffixed `:confirm`) on the
  questions of epochs AFTER the recorded one only, with at least
  `CHAMPION_CONFIRM_QUESTIONS` (1,500) of them. Those questions were drawn
  after the pass, so the selection never saw them. There is still one margin.
- **A record stands** while the champion it passed against holds the seat (or
  is a ghost that has only missed), its head is verified every epoch since
  the pass (the comparison reads the current run only, so a miss leaves
  nothing fresh to confirm on), and the epoch of the pass is still inside the
  comparison window. Once the pass has left the window every question in the
  window would count as fresh, and the second test would be the first one
  over again; the head is recorded anew the next time it passes. A new
  champion clears every record: each was a pass against the previous holder.
- **A seat filled from empty is open first.** The first fill, and a take over
  a retired ghost, beat nobody: the first fill is the best of several noisy
  measurements. Such a seat is settled from epoch fill +
  `CHAMPION_CONFIRM_OPEN_EPOCHS` (5) (`SeatState.confirm_from`); in the four
  epochs between the fill and then, it can be taken by the plain test,
  exactly as before this rule, and a head that takes it that way gets the
  same open period from its own epoch. A champion that was itself
  confirmed is settled at once. Without the open period the rule would also
  protect a lucky first fill.
- **A champion that only missed is still confirmed against.** While its ghost
  stands as "missed" the bar is the same as when it is live, so a short
  outage does not reopen a settled seat to one lucky pass. A **retired**
  ghost answers no new questions, so nothing fresh can be paired with it: it
  is beaten by the plain test.

**What it buys, measured**
(`scripts/experiments/experiment_seat_confirmation.py`: the shipped
`seat_step` and `paired_test` with the shipped constants, except 400
bootstrap resamples; window 48; per-question spread matched to mainnet pairs;
60 epochs against a settled champion). Share of runs in which a challenger
with NO edge took the seat, before and with the rule: ten near-copies, 8.3%
and 0 of 120 runs; ten independently trained heads, 17.5% and 0 of 120; thirty
near-copies, 22.5% and 2.5% (1 of 40).

**What it costs, measured** (the same script, 120 runs each). A challenger
with a real edge still takes the seat, in about twice the epochs (every run
inside 150 epochs at +0.85 and above; 94% at +0.5): +1.5 points over a near-copy champion, epoch
5 becomes 10; +0.85, median 8 becomes 18; +0.5, median 31 becomes 57 (94% of
runs inside 150 epochs); an independently trained head +2 points better, 5
becomes 11. On the mainnet reveals e00026349 to e00026384, the one take in
them (+0.83 points at its pass) measured +1.06 on its first 1,500 fresh
questions and confirms at 2,400 (eight epochs after the pass).

**What it does not do.** It does not tell copies from originals, and it does
not look at who owns a head: it is the same for everybody. Among heads of
truly equal value, which one fills an EMPTY seat is still the best of their
measurements. And the withholding and best-of-N residuals above apply to the
fresh questions as they do to the run.

**The retired ghost's window is shorter from the same block:**
`CHAMPION_GHOST_RETIRE_CONFIRM` (8) epochs instead of `CHAMPION_GHOST_RETIRE`
(24). A retired head answers nothing, so a head committed after it retired
can never be paired with it, and for the whole window the seat can go only to
a head that already overlapped it; otherwise the miner share burns, for up to
about 29 hours at 24 epochs. What the window is for is giving the owner's
replacement head time to finish burn-in (five epochs) and compete for the
empty seat on equal terms with copies of the retired one; eight epochs is
that plus slack for one early miss. A ghost created before the block keeps
the expiry it was given.

**Activation.** `config.confirm_activation_block(genesis)`: finney's block is
compiled into the release (`FINNEY_CONFIRM_ACTIVATION_BLOCK`; `None` until the
protocol v3 release sets it with the v3 and odds seat blocks: 9,527,760 in v0.5.0), and `FUGAL_CONFIRM_ACTIVATION_BLOCK` set to anything
else on finney is refused at startup. Other chains read the variable. The
v2 flip relied on every operator setting a variable, and a validator that had
not set it ran the old rules with nothing in its log to say so. Which rule
set an epoch uses is decided by its boundary block, as for v2
(`v2_epoch.seat_params_from_config`). Both the compiled block and the
variable are in the consensus digest (`epoch.confirm_activation_block`), with
the three constants (`scoring`).

**Determinism and rebuild (I1).** The pending records and `confirm_from` are
part of the seat, so they are published in every reveal's `v2_state.seat`,
are in the consensus view two validators are compared on (`v2_sync`), and are
restored by a rebuild or a re-anchor; a validator rebuilt while a challenger
waits confirms it at the same epoch as one that never stopped. They are
published only when there are any, so before the block the seat is
byte-identical to the one before this rule. Records are kept sorted by head
hash, and a published seat is validated and bounded when read
(`v2_epoch.seat_from_dict`).

Enforced by `tests/test_seat_confirmation.py` and the `seat_confirmation`
stage of `scripts/check_determinism.py`.

### I3 — seats, not shares: the v2 payout (v2 release; from the activation block)

*Protocol v4:* the shares are the same 85 / 10 / 5; a seat pays while its head is live (verified; the holder with its scored answers), R does not beat it and the memorisation verdict does not refuse it; no ghost keeps seat 1, there is no weight ramp, and `MAX_WEIGHT_DELTA` is not applied to a v4 vector (the entry "I3 — v4: the seat in three rules" (package S4)).

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

*Protocol v4:* there is no burn-in counter and no miss rule in a v4 epoch: odds are computed for every epoch after a head's commit whether or not it served, and rule B reads the head's answered run (the entry "I3 — v4: the seat in three rules" (package S4)). The reveal carries no `burn_in`.

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

*Protocol v4:* the senior bars are not a separate rule in a v4 epoch; ownership, first commit and the salted tie at the (v4) activation block stay, and rule C protects every earlier contender, not only seat holders (the entry "I3 — v4: the seat in three rules" (package S4)).

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
seat but lost its window, and it has reveals to rebuild from: a follower's
reference reveals (below), or the directory a root's operator names in
`FUGAL_FRAME_BOOTSTRAP` to rebuild from its own. Per-head v1 evidence is not rebuilt (as before); only the burn-in
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
  reference; every other validator verifies against the root's published
  reveals (`<netuid>/` prefix of its bucket).
  **A follower fetches them itself** (`fugal_subnet/reveal_sync.py`): at
  startup and before every v2 epoch, over plain HTTPS from the public bucket,
  into a directory beside its state. On finney the bucket is built in and a
  follower sets nothing; the root is the validator with `FUGAL_ROOT=1`, or
  the one whose `FUGAL_REVEAL_BUCKET` is the reference bucket. Before this
  the operator ran a `gsutil rsync` on a timer and pointed
  `FUGAL_FRAME_BOOTSTRAP` at the result (still honoured as it is); a mainnet
  validator without either started with an empty seat, burned its weight and
  logged no error. **Followers verify downloaded reveals against the root's
  on-chain commitment** (`reveal:<epoch_id>:<sha256>`, which the root's
  hotkey commits for the newest reveal it publishes): the reveal at the
  committed epoch must match it; a reveal newer than the commitment is not
  taken (and removed from disk) until the root commits it; every commitment
  read is recorded beside the reference directory, and an older reveal with a
  recorded commitment must match it; without a readable commitment nothing is
  fetched. The root's hotkey is in the release on finney
  (`config.FINNEY_ROOT_HOTKEY`, refused if `FUGAL_ROOT_HOTKEY` names another);
  elsewhere `FUGAL_ROOT_HOTKEY`, unset meaning unverified. **Known limit:** the
  pallet keeps one commitment per hotkey, so a reveal published before the
  follower first ran has no recorded commitment and is accepted as before;
  reveals that each carry the previous reveal's hash would close this
  (planned with the protocol v3 reset). Operational, not consensus: it
  decides which files a follower accepts, never how an epoch is scored, and
  it changes no reveal and no digest. Otherwise the fetch adds no trust (the
  model is the paragraph above): a file that fails the listing's size or MD5,
  does not parse, or names another epoch never reaches disk; object names are
  matched exactly;
  every request and the sync as a whole are bounded; and in the epoch loop a
  failed fetch is a log line, never the epoch's failure (I6). At startup, a
  live follower with no reference reveals at all and no way to fetch any is
  refused. `tests/test_follower_setup.py`.
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
- **Under protocol v4 a re-anchor replays its own epochs; it never keeps
  them** (2026-11-20; `fugal_subnet/v4_replay.py`,
  `neurons/validator._v4_reanchor`). A v4 epoch's label pool depends on the
  window it is pooled on (the plausibility check's reference R is fitted on
  the window's earlier pairs), and so do its late odds and its seat step, so
  an epoch first run on a window that differed is a different epoch: kept,
  it carried the difference past every re-anchor (a follower that missed
  the first v4 epoch re-anchored every two epochs and agreed again only when
  the root's exclusions stopped changing). So a v4 follower records each
  epoch's inputs to `run_v2_epoch` beside its state once it is scored (the
  slice and menu tables, every verified proof's scored results, every
  exploration answer with its coldkey and private flag, every verified
  head's bytes once by hash, the boundary commitments, the nonce, the
  weight tables, the menu state, the real-picks input, the seniority floor
  and salt; the pinned embeddings and question sets are the release's; the
  reveal publishes only the pool, not the answers it was drawn from, and no
  head bytes), and a re-anchor on the reference's state at k rebuilds the
  reference's whole window at k (its reveals k-47 .. k) and runs each own
  epoch k+1 .. now-1 again through `run_v2_epoch`, in order, on it. The
  result is the state a validator that ran every epoch holds, copy-rule and
  seat events included; the replayed epochs' weights and reveals are
  dropped (nothing is republished, recommitted or set again: the reveals
  published for them stand as published), and the reveal of the epoch the
  state was replaced in says so (`v2_sync.reanchored`: the reference epoch
  and the epochs replayed; per-validator, outside the compared view). A
  span longer than the bound below, or with an epoch whose record or heads
  do not read back, replaces nothing: the state is kept, the error log names
  the epochs, and the gap moves past them, so the next re-anchor is on a
  later reference epoch (never a silent keep). Records are kept for the
  window plus two epochs; a replay holds one epoch's inputs at a time and
  costs one `run_v2_epoch` per replayed epoch. **It replays at most three
  epochs, or the reveal hold if longer** (`neurons/validator._v4_replay_span`;
  the steady state's span is the hold). The arithmetic: one replayed epoch
  costs what the live epoch costs, measured at 256 UIDs on the adversarial
  shapes of `scripts/experiments/time_v4_epoch.py` (2026-11-21, after the
  batched comparisons, vectorised odds and the v4 window file) at 6.5
  minutes (the ladder) to about 7.5 (every served head a rule C candidate
  against all 768 kept heads, `--served-last`); with 8 minutes an epoch,
  three replayed epochs and the live step take about 32 of the 72 minutes,
  under half the epoch, and a fourth would leave less than half for
  everything else the epoch does. A window's worth (48) could never run
  inside an epoch. A longer span (the reference's reveals behind, or read
  late) waits for a later reference epoch. **Bound, measured in the shipped loops**
  (`tests/test_v4_resync_replay.py`, `tests/test_v4_flip.py`): a follower
  whose last missed epoch is b publishes the root's whole consensus state
  and sets the root's weights from b + 2 on, after one re-anchor, whether
  it missed the first v4 epoch, a later one in which the root's check left
  operators out, five in a row, or nine while the root's seat was filled
  two epochs after its return (before this, that follower filled the seat an
  epoch early and set different weights). Epoch b + 1 is run before the
  root's reveal of b can be read, so it may differ. v1-v3 re-anchors keep
  their own epochs as before (same code path, same bytes).
  **Under v4 two more events are re-anchor points, without waiting for
  drift detection** (`neurons/validator._root_came_back`, `reanchor_now`):
  (1) the root's first reveal after a gap in its own reveals (an epoch this
  follower ran and the root did not: the root was down). During the outage
  each follower publishes and sets its own best knowledge, as any node with
  no reference does; the root's reveal of its first epoch back, k, is read
  from k + 2, and a follower re-anchors on it and replays its own epochs
  since, whether or not the views compared at k differ (its window holds
  epochs the root's does not), so from k + 2 it publishes the root's state
  and weights (a take the followers saw while the root was down and the root
  did not is undone then: the root is the reference). (2) A follower restarted
  without its window file (its state kept) rebuilds from the root's newest
  reveal and at once re-anchors on it, replaying its own epochs since from
  their records, so it publishes the root's state from its first epoch back;
  its own reveals from before the restart verify nothing (`anchor_at` is the
  epoch it came back in). `tests/test_v4_reanchor_points.py`.
  **A replayed epoch's holder head** comes, when the replayed seat names a
  holder the epoch's record does not hold (the reference's holder, not this
  node's), from the node's head store and then the root's head publication,
  checked by hash, as the live epoch seeks it (`v4_replay._holder_heads`);
  without it the epoch is `Missing` and nothing is replaced.
  The determinism stage `v4_replay` (I1) replays twenty live v4 epochs
  (a seat fill, late odds, an entry joining and leaving, each epoch's
  memorisation memo) from three anchor epochs; each replay must end in the
  live state byte for byte, and the stage's digest must be the same on both
  hosts.
  **Each replayed epoch runs with the memo the epoch loop built for it**
  (`neurons/validator._v4_memo_for`, the replay's `memo_for(epoch, its
  recorded nonce)`): `memo_v4.epoch_memo` on the published set as of the
  epoch (`memo_sync.ChainSet.view`: the chain's word at the boundaries
  through the epoch's, the same whatever the node has read since; the set
  is brought forward to the current epoch before the sync step), the plan
  `epoch_plan.plan_v4(pool, published=<that view>, epoch=j)` on the epoch's
  pool object, and the nonce the record kept. Nothing is folded during a
  replay, and the set never comes from the node's own reveals; a set not
  read through the epoch's boundary is `Missing` (nothing replaced). Held
  in the shipped loops: every epoch a follower replays ran with the root's
  memo for it (`tests/v4_world.check_replayed_memos`) in the flip and resync
  worlds and in the memorisation scenario with the follower down for five
  epochs while the root's check first refuses the holder, where it agrees
  from b + 2 (`tests/test_v4_replay_seat.py`); with the function removed
  each of those tests fails. A refusal opens the seat without changing its
  state, so in these runs the follower's state and weights agree even
  without the memo; the memo decides the replayed epoch's seat evidence and
  any take of the opened seat.
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
  item 6). v1 epochs keep main's behaviour. A v4 epoch sets it uncapped (no
  v4 vector is capped): the burn uid at 1.0, the same on every validator
  whatever each set before.
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
from a rebuild of the root's reveals. Through v3, seat or copy-rule events
inside the re-anchor span are applied late (v4 replays them on time, above),
so **convergence needs a stretch of about
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

*Protocol v4:* no burn-in; the v4 activation starts an empty seat, filled by the first qualifier (the entry "I3 — v4: the seat in three rules" (package S4)).

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

### I1 — protocol v3 activates at one named boundary, and the seat objective at another (v3 release; not scheduled)

Protocol v3 changes what is asked (a new pool and a new menu), what every
miner measures (its own exploration pairs), and what the seat is decided on
(a head's odds). As for v2, an epoch's rules are decided by its BOUNDARY block:
`slicer.protocol_version_for_boundary(boundary, v2_block, v3_block)` returns 3
from the v3 block, and `v2_epoch.seat_params_from_config(...,
odds_seat_block=)` puts the seat on odds from the odds seat block, which is
never before the v3 block. Between the two the seat is decided on net value
while every head's odds are computed and published, so the new objective is
visible, to miners and across validators, before it decides anything.

- **Finney's blocks are compiled in** (`config.FINNEY_V3_ACTIVATION_BLOCK`,
  `FINNEY_ODDS_SEAT_BLOCK`; v0.5.0 schedules them: v3 from block 9,527,760, the
  seat on odds from block 9,530,640), and the
  variables (`FUGAL_V3_ACTIVATION_BLOCK`, `FUGAL_ODDS_SEAT_BLOCK`) cannot
  move them there; other chains read the variables
  (`config.v3_activation_block`, `odds_seat_block`). Both pairs are in the
  consensus digest.
- **A state built under other rules is not carried over**
  (`run_v2_epoch(rules=, state_rules=)`). The first v3 epoch starts from an
  empty window and an empty seat: v3's questions and model list are different
  ones, so a v2 window cannot be compared with and a v2 champion was seated on
  other questions. The first epoch on odds keeps the window and empties the
  seat, which is then filled by the best measured skill. What a state was
  built under is published (`v2_state.rules`) and kept with the local state,
  so a validator that was down across either block, or rebuilt from a reveal,
  recognises a stale state instead of carrying it. A rebuild does what the
  live epoch did (v3 review, 2026-11-16): it replays only the reveals of the
  newest reveal's protocol, prices them at THAT protocol's table and
  categories (`rebuild_v2_from_reveals(tables=)`; a v3 window priced at v2's
  table left every v3 cell unpriced and the next epoch failed), and a
  re-anchor whose reference snapshot predates the protocol change is refused,
  state and gap kept, until the reference publishes an epoch after it
  (`tests/test_v3_rebuild_flip.py`).
- **Nothing is shipped inert by accident.** A live validator with v3
  scheduled loads the v3 pool, the v3 menu and the embeddings file at
  startup and refuses to start without them, as for v2. Until a release pins
  them (`routing_data/models_v3.json`, its token manifest, the v3 pool
  manifest, `config.EMBEDDINGS_SHA256`) no chain can schedule v3.

`tests/test_v3_wiring.py`, `tests/test_v3_epoch.py`.

### I1/I3 — v3 menu B applies from one named boundary, with a grace period (v0.6.0)

Menu B adds entries to the v3 menu without a protocol change. Every per-epoch
reader picks the epoch's menu from its BOUNDARY block through
`routing_protocol.menu_for_epoch(3, boundary, config.menu_b_block(genesis))`:
the validator's slice, exploration, pricing and proof checks, and the miner's
run inside the TD. Menu A is used before the block and menu B from it.
Finney's block is compiled in (`FINNEY_MENU_B_BLOCK` = 9,531,000, one epoch
after the odds seat block). The variable `FUGAL_MENU_B_BLOCK` cannot move it
there; other chains read the variable. The block, `MENU_GRACE_EPOCHS` and both
menu B files are in the consensus digest. The rationale is in
`docs/decisions/V3_MENU_B.md`.

- **Menu A is unchanged and menu B extends it.** Menu B's first 53 entries and
  token rows are menu A's, byte for byte, and its index space is menu A's
  followed by the added entries. A head built for menu A stays valid on menu B.
  Its seat, burn-in, seniority and window carry across the block, and nothing
  resets. Before the block every output (reveals, seats, the saved window,
  exploration draws, requests) is identical to v0.5.0. Both menu B files are
  hash-pinned, and the extension property is checked
  (`check_safety_invariants.check_menu_v3b_pinned`).
- **During grace the added entries are explored but scored for nobody.** For
  `MENU_GRACE_EPOCHS` epochs from the block, the added entries' pooled pairs
  are marked unscored (`EpochPairs.scored`, published as
  `odds_unscored_entries`) for every head alike, then and afterwards. The odds
  seat's full-menu rule covers menu A only during this time. Scoring a missing
  row at R's odds instead would let a head omit exactly the rows where R beats
  it. After grace the full-menu rule covers all of menu B: a head with no row
  for an added entry may not take or hold the odds seat.
- **A rebuild reproduces it.** A rebuild prices every v3 reveal at menu B's
  table (a superset with menu A's prices) and reads the mask from the reveal.
  A saved window keeps the mask. `check_determinism` runs the odds stage with a
  grace mask.

`tests/test_v3_menu_b.py`.

### I3 — v4 grace is per entry (REPLACED for v4 by "new entries are usable at once", package S2)

This entry described a v4 rule that no longer exists. Protocol v4 had made
menu B's grace general: each entry's pairs pooled but scored for no head
until its own `scored_from`, and a head without its row disqualified from
the odds seat after it. `docs/decisions/SEAT_MINIMAL_V4.md` (4.1, 12)
replaces it, and the v4 code no longer has it: `routing_protocol.grace_excluded`
and `scored_menu` refuse a v4 epoch (and any menu state), a v4 odds input
names no `menu` and no `unscored` entries, and no v4 pair is masked. What
holds instead is the I3/I4 entries "v4: the labels a head is scored on" and
"v4: new entries are usable at once" below. For v1-v3 the two functions are
what they were: v3 menu B's tests run unchanged, and a v3 run across the menu
B block and its grace publishes the same reveals and weights byte for byte
(`tests/test_v4_labels.py::test_v3_epochs_score_byte_for_byte_as_before`).
`scored_from` stays in the hashed menu state with a new meaning: the end of
the entry's allowance. The `grace_scoring` determinism stage is replaced by
`seat_v4`.

### I1 — the menu of an epoch is a pure function of published inputs, read through one function (v4; built ahead of v4, nothing reads it before a v4 activation block)

Protocol v4 (`docs/decisions/LIVE_MENU_V4.md`, section 4) makes the menu a
value, `MenuState`, that changes by one step an epoch without a release. Its
hash is in every v4 proof's namespace, so two nodes that hold different menus
for an epoch refuse each other's proofs. `fugal_subnet/menu_state.py` is that
value and the two functions that decide it. No v1-v3 code path imports it, no
release has set a v4 block, and `identity()`, `identity_v2()`, `identity_v3()`
and `identity_for` are unchanged. This entry records what must stay true when
the rest of v4 is wired to it.

1. **One value, one encoding, one hash.** A state holds integers and strings
   only (two fixed booleans aside: `allow_fallbacks`, always false, and a
   `reasoning` of `enabled`): prices are integer picodollars per token, token
   means integer milli-tokens, every integer below 2^53. `canonical_json` is
   sorted keys, no whitespace, ASCII; `menu_hash` is its sha256; `from_json`
   accepts those bytes and no others (no fractional number, no repeated
   field, no other layout), so a hash names exactly one state.
   `routing_protocol.identity_v4(menu_hash)` is the proof namespace of the
   epoch that runs on it.
2. **One step, pure and exact.**
   `MenuState(k + MENU_LAG) = step(MenuState(k + MENU_LAG - 1), inputs(k), params)`
   reads its three arguments and nothing else: no clock, network, chain, file
   or environment variable. The inputs are plain values (the signed list the
   node holds, voted endpoint reads, token sums, health statistics, the
   gates' reads of provisional entries, the epoch's nonce), read by key, so
   the order they arrive in cannot change the result; a key given twice is
   an error, never resolved by order. All arithmetic is on integers and
   Fractions and every rounding is half up. No entry joins in a step: new
   entries join at the boundary, from the root's join delta (the entry "v4:
   the menu lifecycle" below), and the step decides them.
3. **The compiled rules are applied, never reported.** The step is handed the
   signed lab and host list (the one the state names, or a newer one) and
   calls `menu_rules` itself: when a newer list is adopted, `check_entry` on
   every entry, provisional ones included. (A proposal is judged under the
   list in force by `chain_proposals.admit`, at the boundary.)
   Which entries a list refuses is never an input, so two validators cannot
   be told different things about the same list. The list's hash is
   recomputed from its content before it is used, so the hash the state
   names is the list the rules were applied under. A state that names no
   list (the seed) adopts one in its first step; no step runs without one.
4. **One rule for which menu an epoch runs on**, `menu_state.menu_for_epoch`:
   with no commitment by the root for the epoch, the previous menu carried
   unchanged; with a commitment to the carried menu, that menu; with a
   commitment to another, the state whose hash it names, accepted from any
   source because the bytes are checked against the hash, **and only if it
   can follow the previous one**: at most `MENU_MAX_BYTES` (refused before
   it is parsed, by the reader and by every lookup of the list the bytes
   name), this epoch, the next serial, the lists in force or newer ones
   (never none, never an older one), within the compiled caps
   (`check_caps`: at least `MENU_MIN_ENTRIES` (8) full entries, provisional
   ones not counted, and at most `MENU_CAP`
   entries, every entry with an expected cost a call above zero; at most
   `MENU_MAX_PROVISIONAL` (8) provisional entries, each with its record
   (`MenuState.provisional`); at most `RETIRED_KEEP` records in `retired` and
   `coldkey_cooldown_cap` (400) coldkeys in cooldown, so every list in a
   menu is capped and the largest document stays under `MENU_MAX_BYTES`),
   one step on from the previous menu (`check_transition`:
   no entry joins in a step (`joins in a step`); a provisional entry that
   joined by the scored epoch is decided in it, full from the new epoch
   (`scored_from`, its measured rate from its labels) or retired with its
   gate's record, and one that joined after it stays provisional; no full
   entry's `joined_epoch` or `scored_from` moves; an entry made full starts
   at its call cost (no pick table); an entry leaves only with the
   `retired` record a retirement writes (why, this epoch, the full
   cooldown), at most `health_retirement_bound` (max(2, a tenth of the
   menu)) of them for health, a displacement only from a menu the step
   leaves one below its cap and never of a shielded entry (provisional, or
   full for fewer than `ENTRY_SHIELD_EPOCHS`, 20); `retired` and the
   coldkeys' cooldowns are exactly what the step leaves (`_records_follow`:
   a record new or changed only for an id that left in this step, at this
   epoch, for a reason the step gives that, with exactly the compiled
   cooldown; a coldkey's only by a failed gate in this step; a record
   dropped only by the step's own pruning or because its id joined), so no
   running cooldown is dropped or shortened and none is invented or
   lengthened;
   prices, answer limits, token means and exploration-pair counts move
   only at the refresh cadence, a price by at most `MENU_PRICE_STEP` (4) up
   or down, item 8), and the compiled rules passed under the list it names,
   where what leaves for a list reason is what that list refuses for that
   reason (`_list_verdicts`, which needs the list). All but the list's
   verdicts and the rules are decided on the bytes and the previous menu
   alone. A newer
   list, lab and host or device, is adopted only with the list itself
   (`held_list`, `held_device_list`: its content hashes to the reference),
   never from a reference. **Any bytes that are not a menu that passes
   these checks are `MenuRefused`**, whatever they make a check raise (a
   value of another type anywhere is a refusal, never an error), which
   names the carried menu as the one the epoch runs on: the verdict depends
   on the bytes, the previous menu and the list alone, so every node holding
   them carries the same menu and logs an alarm. A miner's TD checks the
   carried menu under the list the menu it holds names (`td_v4.epoch_menu`'s
   `held_allowlist`), never under one the refused bytes name, so a refused
   menu costs every node one epoch of menu changes and nothing else. A node
   without the bytes for the committed hash, or without the list the menu
   names, gets `MenuUnavailable` and no menu: it never guesses, and a
   miner's TD in that state produces no valid proof for the epoch.
   `MenuUnavailable` means only that. A second way to obtain an
   epoch's menu is a consensus break of the same kind as a second epoch-id
   format. Every menu the step derives is accepted by this reader. What the
   step derives from the root's own proofs (prices, answer limits, token
   means, each entry's exploration-pair count `explore_pairs`, retirements,
   gate verdicts) is not checked against anything a node could recompute:
   it is the publisher's, and other validators' `menu_audit` shows a
   difference. `LIVE_MENU_V4.md` 4.3 lists what the root can still do.
5. **Who may propose is part of the hashed state.** The ids in cooldown
   (`retired`) and the proposers' coldkeys in cooldown (`coldkey_cooldowns`)
   are in the state, so two validators cannot disagree on whether a
   proposal is refused. `retired` has a fixed size and never drops a
   cooldown that is still running: only records whose cooldown has run out
   make room, and a proposal whose id would have no place in the record
   finds no place. A coldkey's cooldown is written only by a failed gate
   and leaves when it can no longer refuse a join, so at most
   `coldkey_cooldown_cap` are held. Every node holds a committed menu's
   records and cooldowns to exactly those the step leaves (item 4), so one
   published menu cannot ban an id or a coldkey beyond the compiled
   cooldown.
6. **No proposer measures, and nothing waits.** A proposal is measured by
   the field's own exploration in its join epoch (coverage aimed first at
   it); there is no measurement report, spend ceiling, make-up epoch,
   pending list or queue. An epoch the root published no step for leaves a
   provisional entry provisional (it is decided by the next step that runs,
   on the labels of its join epoch onward), and a boundary with no join
   pointer adds nothing: the proposals stay on chain and are considered at
   the next boundary.
7. **The seed is pinned.** The first v4 epoch's menu is
   `routing_data/menu_v4_seed.json` at that epoch (`seed_state`). The file is
   hash-pinned, mirrored under `data/`, and rebuilt byte for byte by
   `scripts/build_menu_v4_seed.py` from two committed input files (a saved
   public listing and integer sums from named published reveals). Every
   entry in it passes the compiled rules as an entry on the menu, against
   that listing and the lab and host list (the signed list a release ships;
   the draft under review until the first is signed). Its entries are
   full from the first v4 epoch: a seed entry is never provisional. No v4 menu
   document other than the seed ships in the package
   (`check_safety_invariants.check_no_other_v4_menu_ships`).
8. **Refreshes follow a compiled cadence.** Prices, answer limits, token
   means, exploration-pair counts and pick-cost tables (and provisional
   entries' prices and limits) move in the published state only in a step
   whose new epoch is a multiple of `menu_state.MENU_REFRESH_EPOCHS` (20,
   about a day). The one exception is the entry the gates make full in that
   step, which takes its own measured values and no pick-cost table (the
   W13 entry); a structural change does not license a refresh of anything
   else. A refreshed price moves by at most a factor of
   `MENU_PRICE_STEP` (4) up or down (`price_step`), so one published state
   can do that much to a price and a larger real move arrives over
   consecutive refreshes. A refreshed price above the price ceiling
   (`menu_rules.check_price`: $25 / $100 per million prompt / completion
   tokens) is never taken: the entry leaves the menu (reason
   `price_ceiling`, its id's cooldown), unless that would leave fewer than
   `min_entries`, when nothing about it moves; the reader accepts that
   reason only in a refresh step whose price move could pass the ceiling.
   An entry the gates make full is not priced above it either (the next
   refresh retires it). A voted price of 0 is no change, and token means
   that would make a call cost nothing are not taken. The cadence is read
   from the state's own epoch, so the step stays pure, and every node checks
   it from the two menus (`check_transition`). It keeps the menu hash, and
   so the bytes a miner's box must fetch, unchanged in most epochs.
9. **Every menu the reader accepts can run an epoch.** The exploration
   draw (`epoch_plan.explore_shares_v4`) and every head's picks
   (`head_odds.picks_and_odds_v4`) need a non-empty menu and a positive
   expected cost for every entry; `check_caps` holds both, so they compute
   for any menu it passes. The step keeps them: it never leaves fewer than
   `MENU_MIN_ENTRIES` full entries. Provisional entries do not count toward
   the minimum, since the gates may retire them all in one step whatever
   else it does: health and price-ceiling retirements and displacement of
   a full entry wait when they would go below it (`retire_waits`,
   `menu_full`), it retires at most
   `health_retirement_bound` entries for health a step, holds back a newer
   list that would leave fewer than the minimum of full entries (joins meanwhile are
   judged under the list in force, the base menu's, and the newer list is
   adopted in the first step where it leaves enough; until then the entries
   it would refuse stay on the menu), keeps one place free for a join on a
   full menu by displacing its least-used unshielded entry, and fails a
   provisional entry whose measured call costs nothing (`gate_cost`). "Gone"
   has an outage guard like "failing": an epoch whose decided listing reads
   show more than `RETIRE_OUTAGE_RATE_BP` of hosts unlisted is not counted
   (`menu_inputs.listing_outage`).

Enforced by `tests/test_menu_refresh.py` (the cadence in the step and the
reader, and an honest walk across a refresh epoch), `tests/test_menu_reader_checks.py` (each cap, each transition
check and each list rule, every menu an honest step derives over twenty
steps read back through them, and a one-value mutation fuzz of committed
bytes: the menu or `MenuRefused`, nothing else), `tests/test_menu_root_limits.py`
(each thing 4.3 says a root cannot do), `tests/test_menu_liveness.py`
(item 9, with the property that every menu `check_caps` passes draws and
picks), the root's-menu cases of `run_miner_attacks` (through the miner's
own TD session: a moved scored epoch, a refresh outside the cadence, a
dropped cooldown, an entry added in a step, bytes over the limit unread),
`tests/test_menu_state.py` and
`tests/test_menu_step.py` (one
encoding per state, the same bytes under key reordering, input reordering and
in a second process; every derived state passes the compiled rules and the
reader), `tests/test_v4_budget.py` (filed at k, on the menu of k + 1,
full from k + 3), `tests/test_menu_v4_seed.py`, the
`menu_derivation` stage of `scripts/check_determinism.py` (the seed and ten
steps across a refresh epoch, each followed by the root's join step at the
boundary and the delta every node checks, with every input collection reversed under
`--perturb`, every derived menu read back through `menu_for_epoch` from the
menu it followed, and a guard that the steps exercise a refresh epoch, a
quiet step that refreshes nothing and structural steps) and
`check_safety_invariants.check_menu_v4_seed_pinned`.

Not yet enforced, because the code does not exist yet: that no code path
other than `menu_state.menu_for_epoch` produces a menu (a static
single-reader check, like the one for epoch ids; the chain read that feeds
it on a validator is now `menu_publish.read_menu_commitment`, the entry
below); that the validator hands the step the aggregations over its
verified proofs (`menu_inputs.step_inputs`, with its own determinism
stages: `endpoint_vote`, `token_means`, `provisional_gates`,
`retirement`); that the lists handed to the step and to the reader came
from `verify_signed_list` (both kinds must be the list itself, with a
signer named and a hash that matches the content, which a caller could
still construct; a bare reference is refused; on a validator they come
from `menu_publish.ListBook`, which holds only lists it verified);
and that the step's constants (`menu_state.MENU_LAG`, `MENU_CAP`,
`MENU_REFRESH_EPOCHS` and the rest) are in the consensus digest. Until those
land this module is inert. Publish before commit is enforced (the entry
below).

### I1/I6/I7 — the v4 menu is published before its hash is committed, read at the boundary, audited, and rebuilt by walking the chain (v4; built ahead of v4, nothing reaches it before a v4 activation block)

`fugal_subnet/menu_publish.py` moves the menu between nodes
(`LIVE_MENU_V4.md` 4.3). The validator reaches it only in its `_v4_`
helpers, under protocol version 4 and a v4 activation block that no release
sets; nothing on the v1-v3 path imports it.

1. **The commitment.** The root's reveal commitment for epoch k is
   `reveal:<epoch_id>:<reveal sha256>:<menu hash token>`
   (`reveal_publisher.commitment_for`): the menu hash as its 32 bytes in
   unpadded base64url, 43 characters, one token per hash
   (`decode_menu_hash` refuses any other spelling). In hex it would be 146
   bytes, and `set_commitment` writes one `Raw<len>` field of at most 128
   bytes (`COMMITMENT_MAX_BYTES`); the token form is 125 for every epoch id
   of the next 10^8 epochs. The first three fields are v3's, so a follower
   reads the reveal pointer of either form (`reveal_sync.parse_pointer`). A
   commitment without the fourth field, for another epoch, or that does
   not parse, names no menu.
2. **Publish before commit.** `publish_due_reveals` asks its `menu` hook
   (`MenuPublisher`) for a hash only for the newest reveal it commits, only
   when that reveal carries `menu_next`. The hook returns the hash only
   when the document is the one the reveal names by hash, for epoch k + 2;
   it follows, through `menu_for_epoch`, from the menu this node read from
   the chain for k + 1 (the root never commits a menu its own reader would
   refuse or could not check); the menu bytes and every signed list it
   newly names are put in the store create-only, named by hash
   (`<netuid>/menus/<hash>.json`, `<netuid>/lists/<kind>/<sha256>.json` and
   `.json.sig`); and each is read back from the public store, byte for
   byte, within `PUBLISH_BUDGET_S`. Anything else, or any exception, and the
   commitment goes out without the hash: the menu carries, and no epoch,
   reveal or weight waits on it (I6). The marker records the hash and
   whether the commitment was made, which is how the root knows which menu
   the chain will hold at k + 1 when it steps for k.
3. **The boundary read.** Every node reads the root's commitment for
   E - MENU_LAG pinned at E's boundary block (`read_menu_commitment`), finds
   bytes for a changed hash (its own copy, the store, a mirror:
   `reveal_sync.fetch_menu_bytes` checks them against the hash), and calls
   `menu_for_epoch` (`read_epoch_menu`): no commitment, carried; refused,
   carried with an alarm; no bytes or no list, no menu, every offered proof
   counted unverifiable, and the epoch log says `menu_unavailable`. A chain
   read that fails is not "no commitment" (that would carry on this node
   while its peers moved): it fails the epoch, which is retried, as for head
   commitments.
4. **The root steps after it publishes.** In a v4 epoch the root publishes
   its due reveals before it writes reveal k, then steps from MenuState(k+1)
   as the chain will hold it (MenuState(k) and its own recorded commitment
   for k - 1). A prediction the chain does not bear out (a commitment that
   landed after the boundary) is caught by item 2 at the next publication:
   the menu is not committed, never refused.
5. **The audit.** Every other validator keeps epoch k's step inputs and, at
   the boundary of k + 2, steps from MenuState(k + 1) and publishes how its
   result differs from the root's committed menu as `menu_audit` (paths
   into the published document, no secret), with an alarm when it differs.
   With no commitment for k + 2 (the root published no step for k: it was
   down, or its publication failed) the epoch runs on the carried menu by
   rule, and there is no root step to compare with: the audit compares the
   menu the epoch runs on with the carry of MenuState(k + 1), says why in
   `why`, and still publishes its own step's hash (`own`); an alarm there
   means the node's own record is inconsistent. It runs on the root's menu
   either way. `tests/test_menu_audit_carried.py`; `scripts/rehearsal_v4.py`
   fails when any validator logs an alarm in a scenario that expects none.
   `menu`, the menu an epoch ran on, is `{epoch, hash, source: {epoch,
   hash}}`: `source` is the last published menu it was carried from (the
   first epoch at which its content, `epoch` aside, was held: a committed
   menu, in the store before its commitment, or the pinned seed, which the
   root also puts there), so any reader rebuilds a carried epoch's menu from
   one reveal and the store and checks it against `hash`
   (`menu_from_field`). Each node derives it from the menus it held
   (`MenuBook.source_of`, kept in `sources.json` beside `held.json`, and
   built link by link by a rebuild), so root and followers publish the same
   bytes (`tests/test_menu_source.py`). `menu`, the menu an epoch ran on, joins the view validators
   are compared on (`v2_sync.consensus_view`); the role fields (`menu_next`,
   its hash, the events, the lists, the audit) are left out of it.
6. **The rebuild.** A node with no menu for E - 1 walks the root's
   commitments forward from the last menu it holds, or from the pinned seed
   at the first v4 epoch (`rebuild`), each link a pinned read through
   `read_epoch_menu`, at most `REBUILD_MAX_LINKS` a call. Each epoch's
   commitment is read under the root hotkey in force for that epoch: the
   root's hotkeys are a compiled table of (first epoch, hotkey)
   (`config.FINNEY_ROOT_HOTKEYS_V4`, `root_hotkeys_v4`;
   `menu_publish.hotkey_for_epoch`), so a walk across a change of the
   root's key reaches the menus the live nodes ran on. A link it cannot
   verify (a read that fails, an epoch no row of the table covers, bytes or
   a list it cannot get) stops the walk there and says why: it holds every
   menu before and none from it on, since every later menu follows from
   that one, never goes on from a menu it could not verify, and goes on from
   there at the next boundary. The root also puts the bytes of the menu
   each reveal's epoch ran on in its store (`MenuPublisher`, best effort,
   after the menu it commits), so a TD can anchor on its newest reveal.
7. **Startup.** A validator refuses to run v4 (live) when the chain holds
   reveals longer than MENU_LAG - 1 epochs, when a root delays its own
   reveals longer, when no root hotkey is known, when the commitment would
   not fit one field, when a root has no bucket, or when the step's mix is
   not pinned (`startup_refusals`, `_v4_menu_setup`).

Enforced by `tests/test_menu_publish.py` (on a fake store and a fake chain
that keeps one commitment per hotkey: the order upload, put, read back,
commit, and a chain that refuses any commitment whose bytes a reader cannot
fetch at that moment; a store that fails, reads back other bytes or hangs,
and a list the root cannot publish, each costing only the menu change; an
absent root; a menu every node refuses; a missing link; a follower with the
root's proofs and with another proof set; a restart; a rebuild after a gap;
the cadence across a refresh epoch, where a follower fetches bytes once;
a v3 reveal published and committed exactly as before) and the
`menu_publication` stage of `scripts/check_determinism.py` (two validators
given the same proofs in opposite orders derive the same `menu_next` bytes,
and a follower's audit of the honest root is empty, across a refresh epoch).

The step's inputs are W4a's (`menu_inputs`, `tee/verify.expected_v4`,
`explore_pool.pool_pairs` with both weight tables), called in the
validator's `_v4_` helpers in this order: each proof verified against its
hotkey's `expected_v4` on MenuState(k); the epoch's pairs pooled with v4's
weights and operator cap; the epoch summarised (`summarize_epoch`, each
proof with the coldkey of the metagraph pinned at the collection block) and
kept in a window beside the validator's state; and the step's inputs
(`step_inputs(start, MenuState(k), window, proofs_k, ...)`) built as a
function of `start`, which the root calls with the MenuState(k + 1) it
predicts and every other validator with the one the chain gave, at k + 2.
The window is not rebuilt after a lost state: it refills over
CHAMPION_WINDOW_EPOCHS, and until then this validator's own step (the
root's menu, or another validator's audit) reads fewer epochs than its
peers'. The step's mix and floor (`config.MENU_STEP_PARAMS_V4`) and the
scored category mix (`config.CATEGORY_TARGETS_V4`) are not pinned yet; a
live v4 validator refuses to start without them. Mirrors are a parameter of
the read helpers and are not configured on a validator.

### I1 — odds are computed exactly by the validator (v3)

A head's prediction for a (question, model) pair is a function of two pinned
things: the head's bytes (attested by hash) and the question's embedding (the
pinned embeddings file). So the validator computes it, and the proof carries
no new field (`fugal_subnet/head_odds.py`).

It must be the same integer on every host, because per-pair losses are summed
exactly and compared with a margin. A float sigmoid of a BLAS dot product is
not: `exp` is the platform's, and the sum depends on kernel, SIMD width and
FMA. So p, an integer in 0..10,000, is computed with no rounding that depends
on the host:

- each product w_i·h_i is one IEEE-754 multiply, elementwise (for float32
  inputs the product is exact in a double);
- the sum is `math.fsum`, the correctly rounded sum of its arguments in any
  order;
- no exponential is evaluated: z is compared with a table of 10,000
  thresholds, logit((k + 0.5)/10,000), built with `decimal` arithmetic and
  pinned by digest (`THRESHOLDS_SHA256`, in the consensus digest). A host
  that builds a different table raises instead of scoring.

Under protocol v4 a validator computes millions of these an epoch, so the v4
callers (`pair_odds_v4`, `pair_odds_v4_many`, `picks_and_odds_v4`,
`labels_v4.odds_on`) compute them many at a time (`head_odds.odds_matrix`,
`odds_grid`): every logit by float64 matrix products (any BLAS, any order,
any threads), with a proven bound E on its distance from the exact path's
logit (2u(n + 8)(||w||_1 ||h||_inf + |b|) doubled, plus 2^-50 |z| and an
absolute term for flushed subnormals; the proof is in `head_odds`). Where no
threshold lies within E of the float logit its count of thresholds is the
exact one; elsewhere (and wherever anything is not finite) the exact path
above computes it. So every integer is the exact path's by construction;
the v1-v3 path (`head_odds.head_odds`) is unchanged. Enforced by
`tests/test_head_odds_vectorised.py` (320,000 random cells, heads at the
10^6 limits and built to cancel, logits on and beside thresholds, which must
take the exact path, and the pinned embeddings file when present).

`HEAD_MAX_ABS_PARAM` (10^6) bounds a v3 head's weights and biases, checked
when the proof is verified (a head outside it is refused), and the embeddings
file is checked to hold finite values no larger than a unit vector's, so
every logit is finite. A constant row is not refused: it is a base rate, and
the score prices it at zero skill.

The pooled pairs and every head's odds are published in the reveal
(`exploration_pool`, `head_odds`), their hash is in `v2_state.odds_digest`,
and `v2_state` is the consensus view validators are compared on, so two
validators that pooled different pairs or computed one odds value differently
differ visibly in that epoch. A rebuild reads the published odds
(`v2_rebuild.pairs_from_reveal`, validated and bounded), so it needs neither
the heads nor the embeddings file.

`tests/test_head_odds.py`, `tests/test_v3_epoch.py`; `head_odds`,
`explore_pool` and `odds_seat` stages of `scripts/check_determinism.py`.

### I4 — every miner measures the models (v3)

In v2 every miner answers the same 15 exploration pairs, so the field
measures 15 pairs an epoch whatever its size. In v3
(`epoch_plan.exploration_plan_v3`):

- **A shared set** of `EXPLORE_SHARED_SIZE` (8) pairs, chosen as v2 chooses
  its whole set, the same for every miner. Miners answering the same pairs is
  what the cross-miner agreement check reads.
- **A private set** of `EXPLORE_PRIVATE_SIZE` (24) pairs per miner, drawn from
  the epoch nonce and the miner's hotkey (`exploration.private_nonce`,
  `slicer.select_private_explore`): disjoint from the scored slice and the
  shared set, on the scored slice's category mix, each sent to an entry drawn
  by the epoch's cost-aware shares (below) with the same per-miner nonce. A
  proof is bound to its hotkey, and the validator derives the same set from
  the hotkey it queried, so the proof check is still an exact match of what
  was assigned. A miner learns its set only when the epoch nonce exists.

**Cost-aware exploration, unbiased by weights** (`fugal_subnet/explore_weights.py`).
Drawn uniformly over the v3 menu (53 entries, answer budgets up to ~128k
tokens), exploration would charge a miner as much for a Fable 5.2 @max call
(~$5 at 128k) as for hundreds of cheap ones. So entry e is drawn with
probability pi_e = (1 - floor) * r_e / sum(r) + floor / M, with
r_e = (c_max / c_e)^alpha and c_e its expected dollars per call (pinned prices x
token means). `EXPLORE_COST_FLOOR` (1/4) keeps every entry at a quarter of a
uniform share or more; alpha is the most uniform of 0, 1/2, ... 2 whose
EXPECTED spend for a miner's 32 calls is within `EXPLORE_SPEND_CAP_USD`
($0.20). The questions take their entries by systematic sampling on the
share line (uniform shares give a round-robin), so each question's entry has
probability exactly pi_e. Each pooled pair carries w_e = 1 / (M * pi_e)
(`weight`, WEIGHT_SCALE units, in the reveal and the window file), and
`odds_score.brier_units` / `skill_units` multiply each pair's units by it:
under the sampling distribution E[w] = 1, so the seat's paired test estimates
what uniform sampling would have counted (Horvitz-Thompson), at a variance
cost bounded by w <= 1 / floor. Pair COUNTS (min_pairs) stay counts.
Per-entry statistics (the reference R's rate per entry and benchmark, the
single-model table) condition on the entry and take no weight. The cap is on
expected spend, deliberately: a rule that stopped at a realized total would
make an entry's chance depend on the draws before it, and the weights would
no longer undo the sampling. Everything is exact: Fraction costs, an integer
square root, integer shares and weights (I1; `explore_weights` stage of
`scripts/check_determinism.py`). `tests/test_explore_weights.py`.

The pooled pairs (`fugal_subnet/explore_pool.py`) are built the way a shared
grade is: failed calls dropped; one vote per coldkey per pair; the outcome
the median across coldkeys. Two rules bound what one operator contributes:

- **The cap.** An operator's private pairs are cut to `EXPLORE_COLDKEY_CAP`
  (36) an epoch, keeping those with the lowest HMAC of the epoch nonce, the
  coldkey and the pair, so it cannot choose which. Fifty hotkeys supply a
  twelfth of a twenty-operator pool, not most of it.
- **The plausibility check** (`private_implausible`). An operator whose
  private answers are more than `EXPLORE_PRIVATE_Z` (4) standard deviations
  from what the reference expects of those models on those kinds of question,
  in either direction, has its pairs left out of the epoch's pool. The
  expectation is built from EARLIER epochs only, and an answer counts only
  once the window holds `EXPLORE_PRIVATE_MIN_TRIALS` (50) pairs of its model:
  with no history the reference is one half, and every honest operator on a
  90% model would be thrown out for being right.

A flag, like the cap, only decides what enters the pool. It never touches the
flagged miner's own score: its head is still scored on everyone else's pairs.

**Accepted residual: a private pair usually has one voter.** Its outcome is
then that operator's attested answer: a real model call made inside its TD,
graded by the pinned grader. The operator cannot choose the pair, the reply or
the grade. It can make calls fail (dropped, and bounded per proof by
`EXPLORE_FAIL_MARGIN`), and an operator that controls where its TD's traffic
goes is the upstream-binding question I8 already answers. What a dishonest
operator could add is bounded by the cap and the check above.

**Accepted residual: published outcomes.** Every pooled pair is published, so
a head can memorise them. It is scored only on pairs drawn after it was
committed (next entry), and the pool it is drawn from is the whole question
pool times the menu, so a memorised pair almost never comes up again.

`tests/test_explore_private.py`, `tests/test_odds_seat.py`.

### I3 — the seat goes to honest, sharp odds (v3; from the odds seat block)

*Protocol v4:* a v4 epoch's seat is decided by `seat_v4`, not by this entry's rules (the net-value guard as described here, the senior bars, ghosts, the high-water marks): the entry "I3 — v4: the seat in three rules" (package S4).

**The property.** What is paid for is what the product serves: for any
question, each model's chance of answering it correctly. With that, any
trade between cost and quality picks correctly; a router that only names a
model cannot be used at any other price point.

**The score** (`fugal_subnet/odds_score.py`). On a pooled pair nobody chose, a
head said p and the model was right or wrong (y): its loss is (p − y)², the
Brier score, a proper scoring rule. Integer units throughout: p and y are in
0..10,000, a pair's loss is an integer, and 0.01 of Brier score ("one Brier
point") is exactly the paired test's unit, so `champion.paired_test` is used
unchanged with a margin in Brier points.

**The reference R** is what a head that does not read the question can say:
each model's success rate on each benchmark, from the pooled pairs of the
OTHER epochs in the window (leave one epoch out), shrunk toward the model's
overall rate with strength `ODDS_REF_PRIOR` (20). A head's **skill** is how
much lower its loss is than R's on the same pairs. A table of rates has about
zero skill by construction (`test_reference_skill_gate`).

**The rules** (`champion._challenge_odds`, `seat_step` with
`SeatParams.odds`):

- **Take.** All of: the challenger's odds beat the champion's by
  `BRIER_MARGIN` (0.1 Brier points) at `CHAMPION_CONFIDENCE`, on at least
  `ODDS_MIN_PAIRS` (1,500) pairs both were scored on; its odds beat R (the
  same test, no margin); its net value on the scored questions has a lower
  bound above −`NV_NONINFERIORITY` (0.5) points, so the new champion is not a
  worse router; and the accuracy gates and burn-in as before. Fresh-data
  confirmation applies as for net value, on `ODDS_CONFIRM_PAIRS` (1,500)
  pairs drawn after the first pass.
- **Hold.** A seated head loses the seat when R's odds beat its own on the
  window's pairs by that same test (no margin). A test, not a point estimate:
  a head whose skill is near zero would otherwise lose and regain the seat on
  noise, and an epoch without a champion burns the miner share.
- **First.** An empty seat goes to the highest measured skill among heads
  with at least `ODDS_MIN_PAIRS` pairs, odds for every menu entry, and skill
  over R that is significantly positive: the one-sided test with no margin
  that a challenger's odds pass against R, at `CHAMPION_CONFIDENCE`. Until
  the v3 review (2026-11-16) a point estimate of skill ≥ 0 was enough, so a
  head with no skill could fill the seat on noise and then hold it until R
  beat it significantly (`test_an_empty_seat_is_filled_on_odds_only_by_significant_skill`).
- **Senior bars** (the copy rule) use the same test.

**Fresh pairs only.** A head is scored on the pairs of epochs in its current
unbroken run. It was committed on chain before each of those epochs' nonces
existed, so it cannot have been fitted to their outcomes; earlier pairs are
published history and are never read for it
(`test_a_head_is_scored_only_on_pairs_from_its_own_run`).

**A head gives odds for every menu entry.** A head whose rows do not cover
the menu is published with no odds at all (every pair -1,
`v2_epoch._pool_and_odds`), a pair whose model a head has no row for is
scored as certainly wrong (`odds_score.head_odds_on`), and a head with any
such pair is not eligible for the odds seat: it cannot take, confirm, fill
or hold it (`champion._no_full_menu`, stated in its reveal detail). Until the
v3 review (2026-11-16) a missing row was scored at R's odds. R is fitted on
the window's outcomes, so a copy of the champion with its weakest row
removed beat the champion with no skill of its own
(`test_a_copy_of_the_champion_with_a_row_removed_does_not_take_the_seat`).
R stays what heads are compared against, never what fills a head's gaps.

**V is pinned in dollars from v3** (`CHAMPION_VALUE_PER_CORRECT_USD`, 0.49:
what the v2 rule gave on the day it was pinned), so the menu can change, the
former anchor model included, without moving the exchange rate between
accuracy and cost.

**Accepted residual: the seat can be empty.** If no head's odds beat a table
of rates, no head takes an empty seat, and the miner share burns until one
does. That is the rule doing what it says; it is the reason the odds are
published for every head from the v3 block, before they decide.

**Accepted residual: published odds.** Every scored head's odds on every
pooled pair are published, so a head can be fitted to another head's
published odds without its bytes. A faithful copy has no edge over the head
it copies, so it does not take the seat from it (the margin, and
confirmation); a copy that drops rows is not eligible at all (above); the
copy rule's ownership is by exact hash, as before. A copy that ALTERS rows
gains only where its own odds are better, which is real skill.

`tests/test_odds_seat.py`, `tests/test_v3_epoch.py`.

### I9 — the single-model table is measured, published, and decides nothing (v3)

Every v3 reveal carries `single_model_table` (`fugal_subnet/single_model.py`):
per model, from the window's pooled pairs, its accuracy weighted to the
scored mix, its mean attested tokens and its spend per question at the pinned
prices. A model with fewer than `SINGLE_MODEL_MIN_PAIRS` (300) pairs is not
listed: it is never assumed to be perfect, or anything else, until measured.
The table is a diagnostic. No seat and no weight reads it, because a
per-model estimate on a few hundred pairs is too noisy to gate a payout on
and a gate that failed closed would empty the seat. It is also where the next
release's token statistics and menu pruning come from, at no cost to anyone.

### I1/I4 — the v3 pool: what is in it, what is left out, and what pins it (v3)

`loader.load_pool(3)` is the v2 sets (v2's IFEval mapping replaced by v3's), the
v3 sets (`v3_sets.LOADERS_V3`) and the code sets (`code_sets.LOADERS_CODE`), then
two explicit exclusion steps from `fugal_subnet/benchmarks/exclusions_v3.json`,
each counted in the manifest (`exclusion_counts`):

1. **RouterArena.** RouterArena is an evaluation-only set: routers trained on its
   questions are rejected. Every pool question whose text matches one of its
   8,400, and the LeetCode problems it takes from LiveCodeBench (by slug), are
   left out, so the subnet never trains on the exam it will be measured on.
2. **All-fail.** Questions every graded model failed on random v2 exploration
   calls, with at least four models and two strong ones failing, are left out:
   most are wrong answer keys, which no router can learn and the Brier score
   punishes as noise.

**Code questions are a pure function of pinned inputs.** `code_sets` turns each
source's rows into candidates without running anything; the pinned keep-list
`code_keep_v3.json` (built by `scripts/build_code_keep_v3.py`, which runs the
sources' reference solutions in the grader's own sandbox) names the problems and
test indices kept. Nothing executes at load time.

**The v3 manifest hashes every field a grader reads** (`loader.content_hash_v3`):
the id, prompt, gold, checker id, benchmark and the checker parameters
`build_grader_task` passes on (constraints, code tests, function names,
tolerances). `exclusions_v3.json` and `code_keep_v3.json` are also pinned by
sha256 in `check_safety_invariants.py` and recorded in the manifest.

**Categories** (`config.CATEGORY_TARGETS_V3`): knowledge .20, hard .20, math .10,
instruction .15, everyday .15, easy maths .05, code .15. `tests/test_pool_v3.py`
holds that the 2,500 questions with the highest share of net value (what a head
can memorise) carry at most 15% of it.

**Grading identity.** Commitments, reveals and the consensus digest carry
`fingerprint.grader_hash()`, which names both `graders.py` and `graders_code.py`.
A miner's harness grades a code question only after `sandbox_selftest()` passed
(`harness.require_code_sandbox`), checked before any model call; the miner
refuses to start on a pool with code questions if it fails.

### I8 — a v3 menu entry is a pinned request (v3)

From v3 a menu entry's id is `<model>@<effort>` when a model is offered at
more than one reasoning effort. The id is what a head has a row for and what
prices, the token manifest and every reveal are keyed on; what goes on the
wire is the entry's request: the provider's model id, one reasoning setting,
the answer budget and a provider order with fallbacks forbidden
(`routing_protocol.request_for`, `validate_menu_v3`). The harness sends it as
pinned, with one shared system prompt (`config.HARNESS_SYSTEM_PROMPT_V3`), so a
model is measured the way it is called. The table is part of the measured image
and of the consensus digest, like v2's.

**The shared system prompt is consensus text.** Every v3 call carries it, so a
byte change moves every v3 grade and every token mean. It is pinned as
`SYSTEM_PROMPT_V3_SHA256` in `scripts/check_safety_invariants.py` (an empty
prompt also fails) and ships in the measured image. The product (fugal-core)
must send the same text, or the subnet measures models under a prompt users
never get. It may not fight a graded answer format: fugal-core's "lead with the
answer, then supporting detail" was dropped because `numeric_final` reads the
last number and `letter_mcq` lost replies that open with a bold letter and then
explain; the v3 text says a format the request asks for takes precedence.

### I7 — verified heads are published by hash (operational; v0.6.1)

A reveal names every verified head by its sha256 (`head_hashes`) and
publishes how it routed and, from v3, its odds; the bytes themselves reached
only the validators that queried the miner. The validator that publishes
reveals now publishes the bytes too (`fugal_subnet/head_publisher.py`), so
the product, or anyone, can fetch the champion's head from a public place and
check it against the reveal and the miner's on-chain commitment. The root
does this by default; any validator with a reveal bucket may
(`FUGAL_PUBLISH_HEADS`).

**What is published when.** Every verified head is published, at one of two
times:

- a head that **holds or has held a seat** (the reveal's seat: the champion,
  the former champions in its history, a champion's ghost; a challenger
  awaiting confirmation is not one) is published with the reveal that says
  so, at that reveal's hold;
- **any other verified head** is published once it is
  `FUGAL_PUBLISH_HEADS_AFTER_EPOCHS` epochs (default 32, about 38 hours) past
  the epoch it first verified in on the publishing validator: with the reveal
  of that epoch, or the first one published after it, whether or not the head
  is still served. So a head is measured for that long before its bytes are
  public, and the top result is public as soon as it is the top result.

`0` publishes every head with the first reveal that lists it. The variable is
operational, like the switch. A reveal's sidecar names the hashes of that
reveal that are not due yet (`not_yet_due`, each with the index of the epoch
from whose reveal it is due) instead of leaving them out
(`test_a_head_that_never_held_a_seat_waits_the_hold_after_it_first_verified`,
`test_a_head_that_holds_or_has_held_a_seat_is_public_with_the_reveal`,
`test_a_held_head_that_stops_being_served_is_still_published_when_due`,
`test_over_many_epochs_with_a_hold_nothing_is_early_and_nothing_due_is_missing`).

    gs://<reveal bucket>/heads/<netuid>/<sha256>.npz      the bytes the miner served
    gs://<reveal bucket>/<netuid>/<epoch_id>/heads.json   beside the reveal: where each hash is

**It is not consensus, and nothing consensus moves.** No score, weight, seat
decision, rebuild or follower reads a published head or the sidecar. The
locations are in a sidecar and not in the reveal because reveals are what
validators are compared on: a field present only on validators that publish
heads would read as a divergence. With the feature on or off an epoch's
commit hash, its `consensus_digest` and the reveal file itself are byte for
byte the same, and the switch is classified operational
(`tests/test_head_publisher.py`:
`test_the_commit_hash_the_digest_and_the_reveal_are_identical_with_the_feature_on_and_off`,
`test_the_switch_does_not_move_the_consensus_digest_or_the_commit_hash`).
Head objects are kept outside the `<netuid>/` prefix: followers list that
prefix every epoch within a page limit (`reveal_sync.list_reveals`), and how
many distinct heads exist is decided by miners.

**A head is never public before the reveal that scores it.** An epoch's heads
are uploaded only after that epoch's reveal has its `.published` marker and
the epoch is past the reveal's hold (`reveal_publisher.delay_epochs_for`;
both conditions are checked by the head publisher itself), and only the
hashes that published reveal lists. A head first verified in epoch `k` is
therefore public no sooner than reveal `k` is, and later than that unless it
holds a seat (above). The epoch a head first verified in is recorded when it
enters the outbox and is not moved when the head is seen again; a head with
no record is not assumed to be old
(`test_a_head_is_not_public_before_its_reveal_is`,
`test_only_hashes_a_published_reveal_lists_are_uploaded`,
`test_the_first_verified_epoch_is_kept_and_a_head_without_one_waits`).

**Only verified bytes, within the caps (I2).** The outbox is filled from the
same `verified_heads` the reveal's `head_hashes` come from. The sha256 is
recomputed when a head enters the outbox and again before upload,
`HEAD_MAX_BYTES` is applied both times, a file that fails is removed, and a
head is never parsed on this path. Objects are created only if absent, so one
validator cannot replace another's object, and they carry no metadata beyond
their name. Heads verified before the feature was switched on are not
uploaded from the private archive (`FUGAL_HEAD_ARCHIVE_DIR`); they are
published only if a miner serves them again, and for a head that holds no
seat the hold then counts from the first epoch it verified in after the
switch. The outbox is bounded (`SPOOL_MAX_HEADS`, 4,096); past it the heads
seen longest ago are dropped and are not published unless served again.

**It cannot cost an epoch (I6).** Uploading is the last step of an epoch,
after the weights are set, the reveal is published and the state is saved. It
runs in a worker thread under a time budget (`PUBLISH_BUDGET_S`, 120 s). A
store that fails, is slow or never answers produces a warning; unsent heads
stay in the outbox and the epoch stays open for `RETRY_EPOCHS` (20) epochs,
and a head that is still served is kept again every epoch and goes up with a
later reveal (`test_a_failing_store_costs_nothing_and_is_retried_on_a_later_epoch`,
`test_a_hanging_store_never_holds_the_epoch_longer_than_the_budget`).

Under protocol v4 heads are files as they are under v3, and the step is the
same one: the root's v4 epoch spools the heads it verified and publishes the
due ones last, after its reveals and saved state (`tests/test_v4_loop.py`
checks that every head the published v4 reveals list is in the store by
its hash, with its sidecar, and no other head is).

**Readers do not trust the store.** A fetched head must have the sha256 the
reveal names (which is the object's name and the miner's commitment) and must
be loaded with pickle disabled, `np.load(..., allow_pickle=False)`, within
the size limits (`head_eval.load_head_from_npz`). `reveal_sync.fetch_head`
fetches and checks the hash. The reveal itself is checked against the
publishing validator's on-chain commitment, as before.

**What changes for miners.** Before this, the bytes of a head were held by
its miner and by the validators that queried it; what was public was its
hash, its routing and its odds. Now every verified head becomes public: with
the reveal (one reveal hold) once it holds or has held a seat, otherwise
`FUGAL_PUBLISH_HEADS_AFTER_EPOCHS` epochs after it first verified. The seat
rules are unchanged and are what applies to a head made from published
bytes: the same bytes under another coldkey are an exact copy and are
ineligible; different bytes are a different head, scored only on pairs drawn
after it was committed, which takes a seat only by beating every senior seat
holder by the margin and gains only where its own odds are better. The
accepted residuals recorded above for published routing ("a champion kept
offline") and published odds are to be read with this: a copy can now be made
from the bytes rather than fitted to what was published.

### I8 — v4 entry rules and signed lists (built ahead of v4; nothing reads them before a v4 activation block)

Protocol v4 (`docs/decisions/LIVE_MENU_V4.md`) moves the menu out of the image
and into published state, so the image carries the **rules** an entry must
pass instead of the entries. The first piece is in the tree and inert:
`fugal_subnet/menu_rules.py`. No v1-v3 code path imports it, and no release
has set a v4 block. This entry records what it must keep true when the rest
of v4 is wired to it.

1. **A rule's verdict is a function of its arguments.** `check_entry`,
   `check_menu` and `check_proposal` read no clock, no network and no
   environment variable, and their constants are compiled, not
   env-overridable. The same entry, list and state view give the same
   `Refusal` (code and detail) on every node, because the refusal of a
   proposal is published in reveals. Each constant has one definition: the
   rules' (`MAX_ANSWER_TOKENS`, `MENU_MAX_HOSTS`) here, the state's
   (`MENU_CAP`, `MENU_MAX_PROVISIONAL`, `MENU_LAG`, `ENTRY_SHIELD_EPOCHS`) in
   `menu_state`, and a test holds that neither module restates the other's.
   A new proposal and an entry already on the menu differ in one rule only:
   a proposal needs every pinned host listed with an answer limit
   (`check_request`, `check_proposal`), while an entry is limited by its
   pinned hosts that are still listed, provided one is (`check_entry`).
2. **A list is trusted only through `verify_signed_list`.** It checks, in
   process and offline, an OpenSSH signature over the list's exact bytes:
   namespace `fugal-list@fugalai.com`, a key that
   `fugal_subnet/signed_lists/allowed_signers` lists for that namespace by
   name, Ed25519 over SHA-512. The check is not delegated to an installed
   `ssh-keygen`: a verdict that depended on each node's OpenSSH would be a
   consensus input nobody pinned. `cryptography` is pinned exactly, for the
   reason given under "why `cryptography` is pinned".
3. **One content, one byte string.** A list is refused unless its bytes are
   the canonical encoding of its content, with no repeated key or item, so
   its sha256 names it and two parsers cannot read it differently.
4. **Serials only go up.** A list is adopted only above the serial held;
   serial 0 is a draft and is refused whatever signs it. The kind of list is
   inside the signed bytes, so a device list is never a menu allowlist.
5. **The list signers are a trust root.** They are a byte-identical copy of
   `deploy/keys/allowed_signers`, pinned in `menu_rules.LIST_SIGNERS_SHA256`
   and in `scripts/check_safety_invariants.py`, and inside the measured image.
   Changing who may sign a list is a release and a new image.

Enforced by `tests/test_menu_rules.py`, `tests/test_signed_lists.py` (which
holds the verifier to OpenSSH's own verdicts on real `ssh-keygen` output) and
`check_safety_invariants.check_signed_lists`
(`tests/test_signed_list_safety.py`). The menu's own two functions call the
rules themselves (`menu_state.step` and `menu_state.menu_for_epoch`, the I1
entry above), and the `menu_derivation` determinism stage runs them. The TD
reaches its menu and files its proposal through them (the next entry, W3).
Not yet enforced, because the code does not exist yet: that the validator does
the same (work package W4).

### I2/I5/I8 — the v4 TD runs on its menu, reads the listing, and attests what it was billed (built ahead of v4; nothing reaches it before a v4 activation block)

Work package W3 of `docs/decisions/LIVE_MENU_V4.md`: the miner's side of
protocol v4, in `fugal_subnet/tee/td_v4.py`, `fugal_subnet/tee/endpoint_reads.py`
and the protocol v4 path of `tee/harness.py` (`run_benchmark(protocol_version=4,
v4=...)`). No release sets a v4 block, the miner's `main()` never selects
protocol 4, and the v1-v3 path imports none of the v4 menu modules
(`tests/test_harness_v4.py` checks the import graph). What must stay true:

1. **The TD runs an epoch on the menu its state names, or on none** (I8).
   Before any model call or listing read, the harness reads the epoch's menu
   through `menu_state.menu_for_epoch` against the committed hash, from bytes
   of any source, and then applies the compiled rules to every entry and
   every proposal itself (`td_v4.check_menu_in_td`; a state that names no
   signed list must be the pinned seed). A committed menu the reader refuses
   is carried, as on every node, and checked in the TD under the list the
   menu it holds names (`V4EpochInputs.held_allowlist`, which the session
   picks itself), never under one the refused bytes name. A menu the TD
   cannot check
   (`MenuUnavailable`: no bytes for the hash, or not the list the menu names)
   or that fails the rules inside the TD (`MenuRejected`: an operator-held
   menu with a forbidden entry) ends the epoch with no call and no proof. The
   lists themselves, lab and host and (for a menu that newly names one)
   device, are trusted only through `menu_rules.verify_signed_list`. Bytes
   that carry the committed hash reach the reader whatever their size, so a
   document over `MENU_MAX_BYTES` is refused and carried in the TD exactly
   as on every node, never treated as missing.
2. **What goes on the wire is the entry's named fields** (I8): `model`,
   `reasoning`, `max_tokens`, and `provider` with the pinned order and
   fallbacks forbidden (`td_v4.wire_request`, equal to `menu_state.request_for`
   for an entry), plus the harness's `messages`. Every request is checked for
   exactly those keys before it is sent.
3. **The listing read is bounded and keyless** (I2, I6). One GET per model
   under the TD's one base URL, with no key, the model id checked before it
   enters the path, no redirect followed, the body capped and the read held
   to a deadline (`runtime.fetch_endpoints_page`). A page is parsed strictly
   (`endpoint_reads.parse_listing`): size, endpoint count, one entry per host,
   string shapes and number ranges; prices are decimal strings converted to
   integer picodollars with integer arithmetic. A read that fails leaves no
   record (no vote) and the epoch goes on.
4. **Billed cost is read exactly** (I1). `usage.cost` is read from the
   reply's own text, not through a float, as integer nanodollars rounded half
   up (`runtime.billed_nanodollars`); the serving host is the provider the
   reply names, bounded. Both are attested per result; neither decides a score.
5. **A TD measures no proposal; a provisional entry's calls are bounded**
   (I3, I5, I6). There is no measurement in a v4 proof and no spending
   ceiling: a proof that carries a proposal, or a result with
   `measure_for`, is refused as it is read (`proof.BenchmarkProof.from_dict`).
   A provisional entry is called like any other (its exploration draw,
   aimed first at it in its join epoch), with three limits of the TD's own
   (`harness._call_all_v4`): each call asks the proxy for
   `PROVISIONAL_CALL_DEADLINE_S` (300 s, shorter than any entry's own; the
   proxy honours only a shorter deadline, `runtime._shorter_deadline`), a
   failed call is retried once, and no call starts once
   `PROVISIONAL_BUDGET_S` (1200 s) has passed since the epoch's calls began;
   a call not made is reported failed. An entry whose price on the TD's own
   listing read is more than four times its menu price is not called
   (`td_v4.overpriced`). Validators leave every provisional call out of
   the exploration failure bound, the outage rule and the health counts
   (`menu_inputs`), so a slow or broken new host moves only its own gates.
6. **A TD that starts or falls behind does not replay history** (I6).
   `td_v4.V4Session` starts each epoch's walk from the first of: the last
   menu it accepted (kept in memory and as `<state dir>/menu.json`, its
   canonical bytes), when at most `RESUME_MAX_EPOCHS` old; the menu the
   root's newest reveal says its epoch ran on (the reveal verified against
   the root's commitment read pinned at the epoch's boundary, its `menu`
   field's bytes against the hash, the compiled caps and the rules in the
   TD: `menu_publish.read_root_anchor`); the operator's anchor
   (`--v4-anchor`); the kept menu however old; the seed. It then walks
   forward with the full reader checks. Each pinned commitment read is
   kept, and the epoch's own menu is kept when it is read, so a node that
   serves only recent blocks costs at most the epoch whose older boundary
   it cannot read; the session never goes on from a menu it could not
   verify. The miner connects with the archive and fallback endpoints the
   validator uses (`FUGAL_ARCHIVE_ENDPOINTS`, `FUGAL_FALLBACK_ENDPOINTS`).
   **How far back a v4 node reads, pinned.** Every v4 chain read (the
   published set's commitment reads, the menu walk and the join pointer,
   the window fill's reveal hashes, the root's join read at B, the TD's
   anchor) goes through the neuron's one `bt.Subtensor`, built with the
   archive endpoints, whose connection switches to the archive when a node
   answers that a block's state is discarded (the real-chain rehearsal's
   "the localnet node refuses, the archive serves"). Steady state: at most
   one epoch back (the current boundary, B). After missed epochs, where a
   read of an old boundary fails as pruned (`memo_sync.is_pruned`, told apart
   from a transient failure, which still raises as before), the published
   set re-anchors once on the root's newest snapshot named by the current
   commitment (`ChainSet.advance_to`), and a validator's menu walk on the
   menu the root's newest reveal names (`MenuNode._anchor`, the TD's
   anchor rule, held to the caps and the compiled rules): each needs pinned
   reads `memo_sync.ANCHOR_HISTORY_EPOCHS` (3) epochs back; when even that
   fails the set is `PublishedUnavailable` (no proof checked or made that
   epoch) and the menu unavailable, saying an archive endpoint is needed,
   and the next epoch tries again from a newer anchor. Across the v4
   activation (no snapshot named yet) the set is read from the release's
   seed, up to `PUBLISHED_V4_SEED_MAX_AGE_EPOCHS` (150) epochs back: an
   archive. A node on a full-history endpoint never re-anchors and reaches
   the same states as before (tests/test_memo_sync.py,
   tests/test_menu_publish.py, tests/test_v4_pruned_loop.py).
   Trust: an anchored TD takes the root's word for the menu its newest
   reveal's epoch ran on (the same root whose commitments name every
   menu); the menu is still held to the caps and the compiled rules.
7. **A TD applies the root's join delta as every node does.** The epoch's
   menu is the base menu (the committed or carried one) with the entries of
   the delta the root's join pointer names at the boundary
   (`td_v4.epoch_menu`, `chain_proposals.check_delta` on the delta's own
   contents); no pointer, a pointer to no joins, or a delta the check
   refuses is the base menu. A TD without the bytes of a named delta has no
   menu (no proof), like one without a committed menu's bytes.
8. **A v4 field never changes an earlier proof** (I1, I7). Each v4 field
   (`menu_hash`, `endpoint_reads`; per result `provider_cost`, `served_by`)
   is hashed and serialised only when present,
   bounded, and has one encoding. A v1, v2 or v3 proof keeps its bytes and
   its `content_hash` (pinned at their pre-v4 values), and the largest v4
   proof fits `PROOF_JSON_MAX_LEN`. A proof whose `routing_protocol` names a
   v1-v3 namespace is read as before the fields existed: its v4 keys are not
   read at all, so a validator on this release and one on the release before
   reach the same verdict on it whatever such keys hold.

Enforced by `tests/test_proof_v4_fields.py`, `tests/test_endpoint_reads.py`
(every page of the committed listing snapshot, and hostile pages),
`tests/test_runtime_v4.py` (a loopback upstream: redirects, oversize, deadline,
no key), `tests/test_harness_v4.py` (including a run through the real proxy),
`tests/test_td_v4_session.py` and `run_miner_attacks.py` (v4 proof fields,
listing pages, menu bytes). The validator's side of these checks (the v4
fields required, the namespace against the epoch's menu) is the entry
"I1/I3/I4 — a v4 proof is checked against the epoch's
menu" below. Not yet enforced, because the code does not exist yet: the
slice mix (W8), which the harness takes through `td_v4.V4Plan`. The exploration a
TD runs is W5's draw on the epoch's menu alone (`td_v4.exploration_v4`,
`epoch_plan.explore_shares_v4(state)`).

### I8 — a TD miner's own v4 options cross the attested channel, checked at the boundary (built ahead of v4; nothing reaches it before a v4 activation block)

A miner in a TD reads nothing of its own from its compose: the frozen compose
is every miner's, its hash is the approved identity, and it allows no
environment (`allowed_envs: []`, I8 above). Its two v4 options, the
proposal it serves and commits (`FUGAL_V4_PROPOSAL`) and whether it answers
the scored questions (`FUGAL_V4_SCORED`), therefore join `ALLOWED_FIELDS` as
`v4_proposal` and `v4_scored` (`tee/v4_options.FIELDS`), strings, each
optional. (A spending ceiling was an option of v4 drafts; with no proposer
measurement there is none, and the field is refused as unknown.)

1. **Checked at the boundary, whole.** `provision.validate_payload` checks
   each value with `tee/v4_options` before the store keeps anything, on the
   operator's side before sending and in the TD on receipt, and refuses the
   whole push when one fails, naming the option and the rule, never the
   value. A proposal: printable ASCII, at most `MAX_PROPOSAL_CHARS`, JSON
   with no key twice and no NaN, exactly its four keys, and every compiled
   entry rule that does not depend on a list (`menu_rules.check_request`:
   the model's form, the reasoning setting, at most two hosts, none twice,
   no service tier, one quantization). It is not checked against a signed
   lab and host list at provisioning, on either side: lists change without
   a release, so a lab or host a newer signed list adds can be proposed from
   an image that predates it. The list check is the root's, at admission
   (item 2). Scored: `auto`, `on` or `off`. A field sent twice in the sealed payload is
   refused (`unseal`), for every field.
2. **Their effect is the option's and nothing else.** None is an input of the
   grader, the pool, the slice, the menu or the cost model. The miner serves
   the proposal as a document (`chain_proposals.ProposalDoc`, over the
   `FugalProposalSynapse`) and commits its hash beside its head in one
   transaction; the root admits it only under the signed list in force at
   that boundary (`chain_proposals.admit`). Where the image ships a list,
   the READY line says whether that list allows the proposal and why not
   (`v4_options.listed_refusal`, informational); the scored mode is visible
   in every proof (`scored`). The compose hash and the image do not depend on them.
3. **One source, one push.** The miner takes them from the push only under
   `--await-provisioning` and refuses to start when one is also set on its
   own command line or environment (`neurons/miner._v4_pushed_options`). The
   TD takes one push per start, so changing an option is a stop/start and a
   new push.
4. **The channel before v4 is unchanged.** A push without them is the push
   before v4, accepted by the old receiver; the new receiver accepts the old
   push and the miner runs with no proposal and `auto`. An old
   receiver refuses a push that carries one, as an unknown field.

Enforced by `tests/test_td_v4_options.py` (every rule, the real receiver,
the miner's resolution, both directions against the v0.6.0 module where the
checkout has its tag), `tests/test_provision_schema.py` (the allow-list
literal), the `v4 option:` cases of `fugal_subnet/attacks/run_tee_attacks.py`,
and the release gate (`check_safety_invariants --release-v4`,
`_td_option_gaps`): every `FUGAL_V4_` option the miner reads, other than the
two with an image default, must map to a field in `v4_options.FIELDS`, be in
`ALLOWED_FIELDS`, have a parser in `v4_options.PARSERS`, and be handed on and
applied by the miner's provisioning path; a new option without that path is a
blocker (`tests/test_release_gate_v4.py`).

### I4/I5 — v4 exploration measures every entry, and each pair is weighted by its own draw (built ahead of v4; nothing reads it before a v4 activation block)

v3's exploration (the I4 entry "every miner measures the models") draws dear
entries at a quarter of a uniform share, so on a 60-entry menu a dear entry
collects about one pair an epoch. Protocol v4
(`docs/decisions/LIVE_MENU_V4.md`, section 8, at the sizes of
`docs/decisions/SEAT_MINIMAL_V4.md`, sections 5.3 and 8: the owner's label
budget of 112 calls a hotkey-epoch) drops v3's shared set and draws each
miner's private set in a cost-aware part and a coverage part. The code is in
the tree and reached only with protocol version 4
(`epoch_plan.exploration_plan(4, ...)`, `explore_shares_v4`,
`exploration_plan_v4`; `explore_weights.plan_shares_v4`):

- **No shared set** (`EXPLORE_SHARED_SIZE_V4` = 0; v3's `EXPLORE_SHARED_SIZE`,
  8, is read by v1-v3 only). Its only reader was the cross-miner agreement
  detector, and a pair every operator voted on judges no head (SEAT_MINIMAL_V4
  2.3 item 33).
- **Cost shares** are v3's rule on the epoch's menu state's measured costs
  (`menu_state.costs`), with `EXPLORE_SPEND_CAP_USD_V4` ($0.55) over the
  calls they draw: the cost-aware part (`EXPLORE_COST_AWARE_SIZE_V4`, 68)
  and the shared set (none).
- **Coverage shares** (`explore_weights.coverage_draw`, 44 calls):
  d_e = max(0, 100 - n_e), with n_e the entry's exploration pairs (never a
  proposer's measurement pairs) in the window as published `MENU_LAG` epochs
  back, read from the epoch's menu (`MenuState.explore_pairs()`, which
  `explore_shares_v4(state)` reads and nothing else); f = min(1, sum d / 1,400); q_e = f d_e / sum d + (1 - f) / M; then
  pi_e = t q_e + (1 - t) / M with t the largest value in [0, 1] whose
  expected spend is within `EXPLORE_COVERAGE_SPEND_CAP_USD_V4` ($1.10). When
  uniform coverage alone costs more than the cap, no t meets it and t is the
  end with the lower expected spend.
- **The split.** A miner's 112 private questions are drawn as v3 draws its 24,
  then ranked by HMAC(its exploration nonce, the question id): the first 68
  take entries by the cost shares, the rest by the coverage shares
  (`exploration.private_parts`, `coverage_nonce`).

**Weights stay exact inverse probabilities (I1, I3).** A shared pair (none
under v4; the table is kept, and pooling refuses a pair answered in both
sets) carries v3's weight, 1 / (M pi_cost). A private question is in the
cost-aware part with probability 68/112 whatever its benchmark (the split is
keyed on its id alone), so its entry was drawn with
pi_private = (68 pi_cost + 44 pi_cov) / 112 exactly, and a private pair
carries 1 / (M pi_private). Weighting a private
pair by its part's probability alone would not be unbiased: a coverage part
can give an entry probability zero (f = 1 and the entry at target), and two
miners can answer the same pair from different parts. Every probability is a
Fraction and every share an integer (`exact_shares` puts the coverage
probabilities over their common denominator, so the cap holds exactly). No
weight exceeds 112 / (68 x 1/4) = 6.6; at the W0 replay's measured costs the
largest is 4.8 (the dearest entry at target while the others are short) and
in steady state the design effect of the pool is 1.11 against v3's 1.31.

**Spend (I5).** A miner's expected exploration spend is at most $0.55 for the
calls drawn by the cost shares and $1.10 for the coverage part, whenever the
menu's uniform cost allows each cap to be met (v3's cost-aware cap has the
same limit: with no exponent within it, the largest exponent is used). That
and its VM is the whole bill of an odds-only miner (the scored questions
off, the default for a head that does not hold the seat: the S3 entry
below); the scored questions add about $4.65 an epoch, only in the epochs a
miner answers them.
Both caps are on expected spend, for the reason v3 gives: a rule that stopped
at a realised total would make an entry's chance depend on the draws before
it, and the weights would no longer undo the sampling. At the replay's costs
(W0's model re-run on these sizes, SEAT_MINIMAL_V4.md 5.6) the cost-aware part
costs $0.35, uniform coverage $0.43, the steady state $0.78 an epoch
(0.67-0.91), and the dearest entry joining costs $1.15 at its worst epoch:
its damped draw ($0.81 of coverage) is under the cap, which at these sizes
binds only when a dozen or more of the menu's dearest entries are short at
once.

The validator pools a pair with the weight of the set it was drawn in:
`explore_pool.pool_pairs(weights=SharesV4.shared_weights(),
private_weights=SharesV4.private_weights())` gives a pair on a question of
the miner's private set (`verify.V4Expected.private_ids`) the private
mixture's weight and any other the shared set's, and refuses a pair answered
in both sets (every node draws them disjoint). `explore_pool.coldkey_cap(4)`
is `EXPLORE_COLDKEY_CAP_V4` (168, 1.5 private sets). Without `private_weights`
pooling is v3's, byte for byte.

n_e is each entry's `explore_pairs`, published in the epoch's `MenuState`
(`MenuState.explore_pairs()`, which `epoch_plan.explore_shares_v4(state)`
reads and nothing else; measurement pairs never count). The validator's
epoch passes both weight tables and `coldkey_cap(4)` to the menu step's
pool (`validator._v4_pool`) and to the seat's (`validator._v4_odds_input`).
The sets are drawn on the pinned v4 mix (the next entry).

**Not yet enforced, because the code does not exist yet:** the v4 constants
are compiled and not yet in the consensus digest (W11 adds them at
activation).

Enforced by `tests/test_explore_v4.py` (the sizes and caps, probabilities,
exact weights as a property test, the damping and the cap on the replay's
worked cases, v3's weights against every weight published in reveals
e00026466..e00026489), `tests/test_v4_budget.py` (section 8's sizes end to
end),
`tests/test_explore_pool_v4.py` (each set's weight on a pooled pair, the v4
cap, and v3's pooled rows under every published v3 weight, pinned before the
v4 tables were added), the `explore_shares_v4` stage of
`scripts/check_determinism.py`, and its v4 input stage, which stops unless
every pooled pair weighs its own set's table and some private pair weighs a
private weight that differs from the shared one.

### I1/I3/I4 — a v4 proof is checked against the epoch's menu, and the menu step's inputs are votes of operators (built ahead of v4; nothing reads it before a v4 activation block)

Work package W4 of `docs/decisions/LIVE_MENU_V4.md`, its verification and
aggregation half: `verify_proof(protocol_version=4, v4=...)` in
`fugal_subnet/tee/verify.py`, and `fugal_subnet/menu_inputs.py`. No release
sets a v4 block, and nothing on the v1-v3 path imports `menu_inputs`,
`menu_state` or `td_v4` (`tests/test_menu_inputs.py`,
`tests/test_harness_v4.py` check the import graph). What must stay true:

1. **A v4 proof is judged against the epoch's menu and nothing the miner
   says about it** (A7). `verify.expected_v4(MenuState(E), ...)` gives one
   hotkey's `V4Expected`: the menu and its hash, the exploration the nonce,
   the hotkey and the menu's shares assign (the shared set, the cost-aware
   part, the coverage part, aimed first at the entries that joined
   provisional in E: the TD's own `td_v4.exploration_v4`). Under protocol 4
   `verify_proof` requires it (a caller error otherwise) and refuses a proof
   whose `menu_hash` is not the epoch's, whose namespace is not
   `identity_v4` of it, or that carries no `endpoint_reads`; a scored route
   to an id off the menu; exploration that is not exactly the assigned set;
   a proposal or a measurement in the proof (refused as it is read); a
   scored set that is not the
   whole slice or none, or that disagrees with the proof's `scored` field
   (the S3 entry below); and a head with no row on the menu. v1-v3 proofs
   are verified as before.
2. **One coldkey is one voter.** Every vote and count an operator could
   inflate with hotkeys is taken per coldkey, the coldkey being the one the
   chain names for the proof's hotkey (`menu_inputs.VerifiedProof`; no proof
   field names one). The endpoint vote (section 7.1): per coldkey, its
   hotkeys' reads of a host are one vote (listed when at least half read it
   listed, then the median of their values); across coldkeys the host is
   listed when at least half vote it listed, `voters` is then the coldkeys
   voting it listed and each value their median, rounded half up to a whole
   picodollar. The step moves nothing on fewer than `ENDPOINT_MIN_VOTERS`.
   A read that is not in the field's one shape (a host twice, a value that
   is not an integer in range) is no vote, here as in the parser.
3. **Health is the replay's rule** (section 9.2). Over the last
   `RETIRE_WINDOW_EPOCHS` (12) counted epochs, an epoch being an outage and
   not counted when its median coldkey failed more than
   `RETIRE_OUTAGE_RATE_BP` (30%) of all its exploration calls: a coldkey
   with at least `RETIRE_MIN_CALLS` (5) calls to an entry's current revision
   is eligible, and failing when more than `RETIRE_FAIL_RATE_BP` (30%) of
   them failed in at least `RETIRE_FAIL_MIN_EPOCHS` (3) distinct epochs; the
   step (`menu_state.health_retires`) retires the entry when at least
   `RETIRE_FAIL_MIN_COLDKEYS` (4) are eligible and more than half fail. Gone
   is `RETIRE_GONE_EPOCHS` (6) consecutive epochs, ending with the scored
   one, in which the vote shows every pinned host unlisted by at least
   `ENDPOINT_MIN_VOTERS` coldkeys. In a field smaller than those minimums
   need, both follow it (`menu_state.retire_minimum`): a majority of the
   coldkeys serving, never more than the compiled minimum and never fewer
   than `RETIRE_MIN_FLOOR` (2), so one operator never retires an entry
   alone (its listing reads are attested; it can only drop its own calls,
   one failing coldkey). An epoch whose decided reads show more
   than `RETIRE_OUTAGE_RATE_BP` of all hosts unlisted is a listing outage
   and is not counted (`listing_outage`). The step retires at most
   `health_retirement_bound` entries for health a step, never below
   `MENU_MIN_ENTRIES`; the rest wait for later steps. A published summary
   is read strictly (`summary_from_dict`: each field of exactly its type),
   so one that is not well-typed is refused whole and its epoch is absent
   from a window rebuilt from it. A provisional entry's calls count toward
   its gates only (each coldkey's calls and failures, `gate_stats`), never
   toward health or the outage rule.
4. **Pairs belong to one run of an id.** An entry's exploration-pair count
   (section 8's n_e) is its pooled exploration pairs of its current revision
   in the window, and its token sums and accuracy are over those. Pairs from
   before an entry's `joined_epoch` belong to an earlier run of the id and
   are not counted.
5. **Exact and order-free.** Medians of Fractions rounded half up once; the
   useful gate's bounds are Wilson's at z = 1.645 on the mix-weighted rate
   with the stratified effective size, through an integer square root
   (lower bound rounded down, upper rounded up, in ppm); every collection is
   read sorted by its key. A window summary (`EpochSummary`) has one JSON
   form, read back strictly.
6. **A proposer's coldkey is the chain's** at the root's pinned read
   (`chain_proposals.ChainRead.coldkey_of`), and whether a proposal passes
   the compiled rules (no duplicate, no cooldown, room) is the admission's
   verdict, which every node re-runs on the delta (the lifecycle entry).

Enforced by `tests/test_verify_v4.py` (the honest proof; each refusal, and
each attack verifying once the one rule that refuses it is taken out),
`tests/test_menu_inputs.py`, `tests/test_v4_input_attacks.py` (each step-input
attack breaking once its rule is taken out), `tests/test_menu_step.py` (the
step's health rule), `run_tee_attacks.py` (stale and foreign menus, a route
off the menu, a proof that carries a proposal or a measurement) and
`run_miner_attacks.py` (one operator's hotkeys on a price and a listing,
forged reads, one coldkey failing an entry, three epochs of failed calls),
`tests/test_menu_retire_field.py` (the field-sized minimums), and the
`endpoint_vote`, `token_means`, `provisional_gates` and
`retirement` stages of `scripts/check_determinism.py`.

Not yet enforced, because the code does not exist yet: that the validator's
epoch builds `V4Expected` for every hotkey it verifies, keeps an
`EpochSummary` per epoch in its window and hands the step
`menu_inputs.step_inputs(...)` (the wiring, W4b); which heads count for use
(`counted_heads`: those that held the seat or were senior bars) is the seat
machine's to name.

### I1/I2 — a v4 head's odds and picks are one exact function of its bytes and the epoch's menu (built ahead of v4; nothing reaches it before a v4 activation block)

Protocol v4 keeps today's head format (`docs/decisions/LIVE_MENU_V4.md`,
11.2) and changes how a head meets the menu and its costs (11.1, 7.3). The
code is reached only from the v4 path of the harness (`run_benchmark(
protocol_version=4)`); `load_head_from_npz`, `head_odds.head_odds` and the
v3 picks are unchanged, and the vendored contract is unchanged and still
byte-equal to its upstream. What must stay true:

1. **What a v4 head is** (I2; `head_eval.load_head_v4`). Today's byte and
   archive bounds first (`HEAD_MAX_BYTES`, the expanded-size cap, at most 32
   `.npy` members, no object dtype, no pickle), through the same reader as
   v1-v3. Then, required: `W` (rows x 1024) and `b` (rows), each
   little-endian float32 or float64 (`V4_PARAM_DTYPES`: half, extended and
   quad precision, big-endian, complex, integer and object arrays are refused
   before any value is read), finite with every value
   within `HEAD_MAX_ABS_PARAM` (10^6); `models`, 1
   to `HEAD_MAX_ROWS_V4` (72: `MENU_CAP` plus `MENU_MAX_PROVISIONAL`) unique,
   non-empty ids of at most 256 characters; `contract`, `profile_id` and
   `backbone_revision` equal to the pinned values; a non-empty `provenance`;
   a finite non-negative `lam`. Optional: `mean_in_tokens`,
   `mean_out_tokens` and `cost_profile_id`, checked as today when present
   and never read. Any other field is refused. Every head valid today
   with float32 or float64 parameters is a valid v4 head.
2. **Rows are matched by entry id, to the epoch's menu**
   (`head_odds.menu_rows`). Odds are computed for the MenuState's entries in
   the state's order; a row's position in the head never matters; a row
   for an id not on the menu is never read; an entry with no row has no odds
   (-1), which the grace rule prices (the I3 per-entry grace entry), never an
   exception; a request update keeps the id and so the row.
3. **One function, exact** (I1; `head_odds.picks_and_odds_v4`). The odds are
   v3's exact integers (`head_odds.odds`). A pick maximises
   odds / 10,000 - lambda x cost over the entries with odds, where cost is
   `menu_state.pick_costs(state)` (prices x the token means of the entry's
   graded scored-question cells, shrunk toward its call cost: the W13 entry
   below; a Fraction) and lambda `routing_protocol.BENCHMARK_LAMBDA` as a Fraction: no float enters
   the comparison, and equal utilities go to the first entry in the state's
   order. The TD's harness routes with it; a validator recomputing a pick
   calls it, and scores pooled pairs with `pair_odds_v4`, which gives the
   same cells. Nothing the head says about cost is read.
4. **The TD checks the head before anything else of the epoch** (I8): a
   head that does not load as a v4 head, or has no row for any entry on the
   epoch's menu (`head_eval.v4_head_verdict`), stops the epoch before any
   listing read or model call.

Enforced by `tests/test_head_v4.py` (the fields, every bound, rows in any
order, extra and missing rows, a request update, picks that follow a state
price and not the head's own token means, exact ties, and property tests on
random menus, rows and bytes), `tests/test_harness_v4.py` (the TD's scored
routes equal the shared function's picks), the v4 cases of
`run_miner_attacks.py`, and the `head_odds_v4` stage of
`scripts/check_determinism.py`. Not yet enforced, because the code is in
other packages: that the validator's v4 head check (W4) is
`v4_head_verdict` rather than the v1-v3 loader, which refuses a head
without token means; that the v4 epoch's pooled odds (W4/W11) come from
`pair_odds_v4`; that `HEAD_MAX_ROWS_V4` is in the consensus digest (W11);
and the product's parity with these picks (W10, after fugal-core's contract
accepts a v4 head; `LIVE_MENU_V4.md` 11.2).

### I1/I4 — v4 prices a pick at the measured pick cost, and exploration at the call cost (W13; built ahead of v4, nothing reaches it before a v4 activation block)

An entry has two measured costs, and v4 keeps both in the published menu
(`docs/decisions/SEAT_MINIMAL_V4.md` 3.4 and 8; the earlier seat design's
8.1-8.4, kept as designed). The **call cost** (`menu_state.costs`: prices
x token means over the entry's exploration pairs) is what one call costs on
a question drawn at random; the **pick cost** (`menu_state.pick_costs`,
`pick_cost`) is what a call costs on the questions routers actually send the
entry. On the published v3 reveals a call cost under-states the cost of the
cells heads picked about twofold, while the pick-cost table is within about
7% of it (leaving one epoch out, e00026466..e00026502: real over predicted
1.058 on the champion's picks and 1.066 on every head's, against 2.36 and
2.04 for the exploration mean per entry; `scripts/build_menu_v4_seed.py
accuracy`, `tests/test_pick_cost_accuracy.py`). What must stay true:

1. **Who reads which.** Exploration's shares, weights and spend caps read
   the call cost (`epoch_plan.explore_shares_v4`). A head's pick
   (`head_odds.picks_and_odds_v4`: the TD routes with it, a validator
   recomputes with it, the product reads the same per-entry table) reads
   the pick cost per entry, because a pick is made without knowing a
   question's category. Net value on the scored questions prices each
   answered call at its entry's pick cost in the question's category, the
   entry's own where the table does not list the category
   (`v2_epoch.build_epoch_slice(pick_spend=)`, built by the validator's
   `_v4_tables` from the epoch's menu; a rebuild prices a v4 reveal the
   same way), never at the call's own tokens. Changing the pick table
   changes picks and not exploration; changing the call table changes
   exploration and not picks (`tests/test_menu_pick_cost.py`).
2. **The cells are votes of operators** (I4; `menu_inputs.graded_cells`).
   A graded cell is a (question, entry) of the scored questions some
   verified proof answered without failing. Each coldkey's answers to it
   are one vote (the mean over its hotkeys); its prompt and completion
   tokens are the median across coldkeys, rounded half up, and no cell
   counts more completion tokens than the entry's answer limit on the
   epoch's menu, nor more prompt tokens than the answer cap
   (`menu_rules.MAX_ANSWER_TOKENS`). So one operator's many hotkeys are one
   vote in a cell, and a cell one operator answered alone moves a table by
   a bounded amount. The epoch summary keeps them per (entry, revision,
   category) as counts and token sums (`EpochSummary.graded`, strictly typed
   by `summary_from_dict`, which refuses a row with more tokens than its
   cells hold at the cap); `pick_stats` sums the window's for each entry's
   current revision from its join epoch and hands them to the step.
3. **The table, exactly** (I1; `menu_state.pick_table`). Per token kind, in
   milli-tokens: pick = (1000 x T + 30 x the call mean) / (n + 30), and per
   category pick_c = (1000 x T_c + 20 x pick) / (n_c + 20), each rounded half
   up, the category shrinking toward the entry's published value
   (`PICK_COST_PRIOR_CELLS_ENTRY`, `PICK_COST_PRIOR_CELLS_CATEGORY`, also in
   `MenuParams`). An entry with no graded cell of its revision has no table
   and is priced at its call cost; a category the table does not list is
   priced at the entry's pick cost; means that would make a pick cost
   nothing are not taken. The table is an optional `pick` object on an
   entry (cells, means, at most `PICK_MAX_CATEGORIES` (8) categories named
   in at most `PICK_CATEGORY_MAX_CHARS` (32), sorted, each with a cell,
   adding up to the table's), absent while the entry has none, so a state
   still has one document and is hashed whole.
4. **Refreshed at the cadence, reset by a join or an update.** The step
   recomputes every table at a refresh (`refresh_due`) from `pick_stats`,
   after the token means it shrinks toward; outside a refresh every table
   stays as published. An entry that joins, and a request update, start
   with no table. The reader holds both from the two menus
   (`check_transition`: a table that moves outside a refresh epoch, or one
   on a joining or updated entry, is refused) and holds every table to
   what its cells could give (`check_caps`: between its prior's share of
   what it shrinks toward and that share plus every cell at the answer cap,
   and never a pick that costs nothing). Like the call means, a table's
   values at a refresh are the publisher's, from its own proofs; a price
   moves the pick cost and the call cost alike and is held to
   `MENU_PRICE_STEP`.
5. **The seed's tables are measured.** Each seed entry's table is the
   step's rule on its graded cells in the 44 published v3 reveals
   e00026466..e00026509 (`data/observations/
   menu_v4_seed_picks_e00026466-e00026509.json`, built by
   `scripts/build_menu_v4_seed.py picks`), the prompt side at the call mean
   (a v3 reveal publishes no prompt tokens for a cell), and the safety check
   holds the seed to the compiled caps.

Enforced by `tests/test_menu_pick_cost.py` (the shrinkage's arithmetic, the
call-cost fallback, the encoding and its bounds, the step at and outside the
cadence, the reader's checks, an honest twenty-step walk with tables, picks
against exploration), `tests/test_pick_cost_inputs.py` (the coldkey median,
failed calls, the bounds on a cell, the summary's form and types, the window
sums), `tests/test_pick_cost_net_value.py`, `tests/test_menu_v4_seed.py`,
`tests/test_pick_cost_accuracy.py`, the pick-cost cases of
`run_miner_attacks.py` (one operator's twenty hotkeys on every cell; a cell
past every bound; a root repricing the table outside a refresh epoch,
beyond what its cells could give, or on a joining entry), and the
`pick_cost_v4` stage of `scripts/check_determinism.py` (six epochs of
proofs with failures, a request update mid-window and an unrouted entry,
every input reversed under `--perturb`). Not yet enforced, because the code
is in other packages: the seat step reading the real-picks check (S3 built
it, `picks_v4.picks_check`, pricing each answered call at
`menu_state.pick_cost` per (entry, category) of its own epoch's menu; S4
wires it); fugal-core's selector reading
the per-entry table (W10); and the constants in the consensus digest with
the step's others.

### I1/I4 — the v4 question mix is derived from named reveals, pinned, and read through one plan (built ahead of v4; nothing reads it before a v4 activation block)

Protocol v4 (`docs/decisions/LIVE_MENU_V4.md`, section 10) draws its scored
slice, its exploration (shared set and both private parts) and every
proposal's measurement questions on one mix, which is data:
`routing_data/pool_mix_v4.json` (reviewed copy `data/pool_mix_v4.json`,
byte-identical, sha256 pinned in `scripts/check_safety_invariants.py`), read
by `epoch_plan.mix_v4`. Integers only: each category's target in whole
per-cents, each question set's share in millionths of the slice, summing
within a category to its target exactly, so every node draws on the same
exact proportions.

- **The rule** (`scripts/build_mix_v4.py`, run once, never at run time: a
  mix that moved with the window would be a feedback loop inside a
  consensus input). A category's target is proportional to p(1 - p), p the
  accuracy of an entry drawn uniformly from the seed menu on the category's
  exploration pairs, each pair weighted by the inverse of its published
  draw probability; at least `MIX_FLOOR_PCT_V4` (5%); rounded to
  `MIX_STEP_PCT_V4` (5%) by largest remainder. Inside a category sets split
  by their own p(1 - p) x pool size (the category's value for a set with
  fewer than `MIX_MIN_PAIRS_V4`, 100, pairs), and no set's share of the slice
  exceeds pool size / (slice x `MIX_CYCLE_EPOCHS_V4`): no set is drawn
  through faster than two 48-epoch windows. A category that cannot place its
  share under the caps, or a set that would get no share, stops the build.
  All of it is exact rational arithmetic.
- **Its inputs are files.** The observations
  (`data/observations/mix_v4_cells_e00026466-e00026489.json`) are integer
  sums reduced from the 24 published v3 reveals the W0 replay used, each
  named with its sha256; `build --check` rebuilds the mix byte for byte.
  From them the rule gives the replay's targets: hard 30, code 15, math 15,
  knowledge 15, instruction 15, easy maths 5, everyday 5.
- **Built for 200 questions, where no cycle cap binds.** `SLICE_SIZE_V4` is
  200 (SEAT_MINIMAL_V4.md 8: the real-picks check prices picks at the
  measured pick-cost table, so 1,000 questions, five epochs, are its
  minimum), and the mix was rebuilt for it (`build --slice 200`, re-pinned
  2026-11-19). The targets do not depend on the slice; inside a category the
  sets take their uncapped shares (`supergpqa_hard` 52 questions an epoch,
  the whole set about every 131 epochs). At 300 every hard set but `aime`
  and two code sets were held at their caps. A different slice size needs a
  rebuild: the mix records the slice it was built for, and
  `scored_slice(4, ...)` and `plan_v4` refuse another. The miner and the
  validator pass `SLICE_SIZE_V4`, not v1-v3's `SLICE_SIZE`, for a v4 epoch.
- **No question leaves the pool for being answered by every entry.** The
  mix holds every set of the v3 pool at its full size with a positive
  share; a saturated set is held at the floor (ARC) or keeps its share
  (`aime25`, `code_humanevalplus`, every observed pair right).
- **One plan.** `epoch_plan.plan_v4(pool)` is the one constructor of a v4
  epoch's `td_v4.V4Plan`: the scored slice, the exploration on the plan's
  pool index, and the index and targets the measurement draw reads
  (`V4Plan.measurement`, which the TD's harness calls and a verifier calls
  for the same proposal). A second way to build the index or the targets
  is a consensus break of the same kind as a second epoch-id format: the TD
  and the validator would disagree on every measured question.
- **v3 is untouched.** `scored_slice(3, ...)`, `pool_index_v3` and v3's
  exploration read `config`'s v3 mix and never the v4 file.

Not yet enforced: W4's verification passes the index and the targets as two
arguments; it should take them from `plan_v4`. The v4 token means and
proposal gates (`menu_state.MenuParams.mix`) are weighted to "the scored
mix", which under v4 is `mix_v4().menu_mix()`; the seed's token means are
still weighted to v3's and are rebuilt with the seed.

Enforced by `tests/test_pool_mix_v4.py` (the rebuild, the replay's targets,
each step of the rule on hand-made inputs, the cycle cap on a small set, no
cap binding at 200, every draw on the pinned mix, and the TD and a verifier
drawing the same measurement questions through `plan_v4`) and the
`pool_mix_v4` stage of `scripts/check_determinism.py`.

### I1/I4 — the v4 epoch runs through the shipped loops, on one menu and one pool (built ahead of v4; reached only off finney, through FUGAL_V4_ACTIVATION_BLOCK)

Protocol v4 is selected as v3 was before its block was compiled:
`config.v4_activation_block(genesis)` is `FINNEY_V4_ACTIVATION_BLOCK` on finney
(None: no release schedules v4, and a set `FUGAL_V4_ACTIVATION_BLOCK` is
refused there, as v3's variables are when they differ from the compiled
block) and the variable on any other chain, never before v3's block. A value
that is not a block number never stops an import
(`config.V4_ACTIVATION_BLOCK_MALFORMED`): where the protocol is decided it is
refused by name on every chain but finney, and on finney while no release
compiles a v4 block it is ignored, as the v3 release ignores the variable;
`slicer.protocol_version_for_boundary(..., v4_activation_block=)` returns 4
from it and `config.protocol_for_chain` gives a miner image 4. The variable is
a consensus input; the digest records it only when it is set, so every
neuron without it publishes the digest it published before.

Under protocol 4 the validator's epoch (`neurons/validator.main`) is:

- the menu at the boundary (`MenuNode.at_boundary`), and from it the epoch's
  tables (`_v4_tables`): entries and prices from the MenuState, categories
  and targets from the pinned v4 mix; the exploration shares once
  (`explore_shares_v4`);
- the scored slice from the plan built once for the v3 pool (`plan_v4`,
  `_epoch_slice_v4`), at `SLICE_SIZE_V4` (200; the miner's TD runs the same
  size), never v1-v3's env-overridable `SLICE_SIZE`; v4 runs on the v3 pool
  and its embeddings, which the mix was built on;
- each proof checked against `verify.expected_v4` with that plan and those
  shares (`verify_proof(protocol_version=4)`);
- the seat: v3's rules on v4's inputs (categories, targets and net value at
  the menu's prices; odds by `pair_odds_v4` on the menu state, pooled with
  both weight tables and `coldkey_cap(4)`), on odds from the v4 block, from
  an empty window and seat;
- the menu step or its audit (`_v4_menu_epoch`) on **the pool the seat's
  odds were scored on** (`V2EpochResult.pooled`): the frame detectors' and
  the implausibility check's exclusions both apply, so the menu's counts and
  summary never pool a set the odds left out;
- the reveal with its v4 fields, publication before commitment, weights.

The miner builds its `td_v4.V4Session` from the root's pinned menu
commitment, the root's bucket and mirrors (`neurons/miner._v4_session`) on the
v3 pool, and runs every epoch through it, applying the root's join delta at
each boundary; its proposal is `FUGAL_V4_PROPOSAL` (operational; in a TD,
pushed at provisioning, the entry on a TD miner's own v4 options above),
served as a document and committed by hash beside its head. The validator
root runs the join step after its weights and before writing its reveal;
followers audit it at the boundary. The gates' category floor is
`config.PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4` = 5000.

Enforced by `tests/test_v4_loop.py`, which runs `validator.main` for a root and
a follower concurrently over nine epochs and the miner's `_run_epoch` on the
sessions `_v4_session` builds, with stand-ins only for the chain, the
transport, the public store and the model provider: every node on one menu,
a proposal filed, measured and joined, the same seat state and weights on
both validators, the reveals published before their commitments, a failed
pinned metagraph read retried, an empty audit, a too-good TD left out of
the menu step's pool where only the implausibility check catches it, and
every head the published reveals list published by hash.

Not yet: v4 reveals replayed by a rebuild or re-anchor are priced at the
current menu's prices (`_rebuild_tables`), not each reveal's own menu; the
window's model axis follows each epoch's menu and its consumers are not yet
shown to align by entry id across a menu change; the seat rules are v3's
until the seat-rule package.

### I1 — a v4 epoch's digest covers every v4 constant and pinned file; a v1-v3 epoch's is unchanged (activation package)

Two validators compare `consensus_digest` to explain a divergence. Every
constant that decides a v4 score, the menu, exploration, the proof schema's
bounds or verification, and every pinned file v4 reads (the menu seed, the
mix, the list signers, the shipped signed lists, and any file a later
package pins), is in it for a v4 epoch: the fingerprint of an epoch at or
after the v4 block (`environment_fingerprint(protocol_version=4)`, which
`commit_reveal.reveal_epoch` takes with the epoch's version) carries a `v4`
section (`fugal_subnet/consensus_v4.section`: the activation block, the
registered constants in an exact canonical form, the files' sha256), and the
digest's material gains that section. Below v4 there is no section, and the
digest is computed from exactly the material it always was: the v3 digest
and the 19 v1-v3 determinism stages are unchanged.

The registry is held complete, not remembered. Every UPPER_CASE binding of
every module the scan reads is registered, or named in `NOT_V4` with the
reason it decides nothing a v4 epoch publishes (a path whose bytes are
pinned, a local timing or retention bound, a pool-building value the pinned
manifest checks), or is a name the v3 release (origin/main) had with the
same statements (`PRE_V4`, held to `fugal_subnet/consensus_v4_pre.json`,
generated by `scripts/pin_pre_v4_names.py`, and checked against origin/main
where git has it). A binding is read in every statement form at module level
(plain, tuple, starred, annotated and augmented assignments; `for`, `with`,
`except` and `match` targets; walrus; `def`; `global` declarations; also
under `if`/`try`/`with`), and the constants a class defined in the module
sets, as `Class.NAME`; a module that writes names the scan cannot read
(`globals()`, `exec`, `sys.modules[__name__]`) is refused. The modules read
are the transitive import closure of `neurons/`, the v4-named modules and the
importers of the v4 core (every import in the file, a string literal naming a
module included), not a list: a module leaves the scan only by an entry in
`NOT_SCANNED` with its reason (the hash-pinned graders), and one that imports
by a computed name only by an entry in `DYNAMIC_IMPORTS`. `NOT_V4` and
`PRE_V4` need a reason per name, and `CLASS_PINS` pins, per module, the hash
of the names in each class: moving a name into either class is a deliberate
edit of that pin, with the reason. Every file under `routing_data/` and
`signed_lists/` is in `PINNED_FILES` or `V1_V3_DATA`. A constant a later
package adds, a new module the scan reaches, or a new data file fails
`consensus_v4.problems` (`check_safety_invariants`,
`tests/test_consensus_v4.py`) until it is classified. `check_determinism`
asserts in every child that one differing v4 constant (a menu, a config, a
seat and a memorisation constant) moves the v4 digest and leaves the v3
digest as it is, and prints both.

The seat (`seat_v4`, its `config` values) and the memorisation check
(`memo_v4`, its `config` values) are registered, and the published-set seed
(`routing_data/published_v4.json`) is a required pinned file. The names v1-v3
read and the v4 path stopped reading when the seat package removed their
rules (the v1-v3 seat's parameters, burn-in, the agreement detector, v1's
copy detection, net value's units, the pending-record bound, the weight cap)
are in `NOT_V4`
with that reason; they stay in the v1-v3 digest's scoring block, which a v4
digest also covers. The attack suites' worlds are tooling and not scanned.

Chosen: the compiled finney v4 block is recorded in the v4 section, not in
the epoch block, so a release that compiles it changes no v3 epoch's digest.
A validator that never installed the release is therefore not visible in the
digest until the first v4 epoch, where its reveal says protocol 3 and its
digest is the v3 one.

### I1/I3/I6 — the v4 flip: what resets, what carries, and what appears only from the block (activation package)

At the first v4 epoch (the boundary at `v4_activation_block`, read through
`slicer.protocol_version_for_boundary`, the epoch id from
`slicer.epoch_id_for_block` as always):

- **Resets.** The comparison window and the seat start empty (the `rules`
  tag changes to protocol 4, as at v2 and v3; SEAT_MINIMAL_V4.md 2.1 #24):
  no v3 slice, answer, pair, odds or picks record is read by a v4 epoch. The
  first v4 epoch publishes a protocol 4 seat state (`v2_state.seat`, marked
  `"protocol": 4`) with no holder and no history, holding only the heads
  `seat_v4.observe` saw at that epoch (each with `since` and `seen` the
  first v4 epoch, no predecessor); no v3 head and no v1-v3 seat field
  (pending record, ghosts, high-water marks, burn-in) is carried, and no v4
  reveal carries burn-in. The real-picks records start with the window. The
  menu is the pinned seed (`menu_state.seed_state`), and the menu step's
  summaries, exploration counts and token means start from the seed's; no
  menu is held for any epoch before the block.
- **Carries.** The copy rule's ownership state (who first committed each
  hash, and when): a v3 head's owner still owns it. From the block a
  retired head's record is kept for `STANDING_EPOCHS_V4` epochs, not v1-v3's
  ghost window, so a v3 head its owner stopped serving at the block lapses
  that many epochs after it. The reference frame and the v1 evidence records
  carry as before; no frame or agreement detector feeds a v4 epoch's label
  pool (`OddsInput.flagged` is empty), whose only exclusion is the
  plausibility check against R.
- **Brought across from the seed.** The memorisation check's published set:
  every validator and TD starts from the release's seed and reads the root's
  commitments at every boundary after it, folding the v3 reveals they name,
  so the first v4 epoch draws on, and judges heads by, the set the chain
  gives at its boundary; every v4 reveal carries `memo_v4` (its view's
  digest, through `memo_digest` in `v2_state`) from the first v4 epoch on
  (memo_sync; the memorisation entry below, 6). Before the block a validator
  reads and publishes nothing of it.
- **Only from the block.** The four-field commitment
  (`reveal:<epoch>:<sha256>:<menu hash>`), every v4 reveal field (`menu*`,
  `odds_v4`, `labels_v4`, `late_odds`, `picks_v4`) and the fingerprint's
  `v4` section. A v3 epoch's reveal on a release that schedules v4 is the
  reveal of one that does not, except, off finney, the epoch block's record
  of `FUGAL_V4_ACTIVATION_BLOCK` (finney's compiled block is recorded only in
  the v4 section).
- **A validator with no state after the block** rebuilds from the root's
  reveals at its first v4 boundary, not at startup: a v4 reveal is priced at
  its epoch's menu, which is read there (`neurons/validator._rebuild_v2_state`).

Enforced by `tests/test_v4_flip.py` (the shipped loops across the block: a
root and a follower crossing it, a follower restarted at the block, one
that misses the first v4 epoch and one that misses the second, a follower
with no state syncing across it, a control whose release schedules no v4)
and by `scripts/check_v3_equivalence.py` (a v3 scenario on two trees).
The published set crosses the block from the seed on every node
(`test_the_published_set_crosses_the_block_from_the_seed`), and the epochs
the late followers replay run with the root's memo.

A follower that misses the first v4 epoch agrees with the root again, on its
whole consensus state and its weights, from the second v4 epoch after it
(F + 2), as one that misses any later epoch does from two epochs after it:
its first v4 epoch was pooled on a window without the root's first, and the
re-anchor replays it on the root's state instead of keeping it ("Under
protocol v4 a re-anchor replays its own epochs", in the verify-and-resync
entry). `tests/test_v4_flip.py` holds it with no allowance: the follower
that missed F agrees from F + 2, the one that missed F + 1 and the one that
started with no state at F + 2 from F + 3.

**The release gate.** `scripts/check_safety_invariants.py --release-v4` (and
`--release` once `FINNEY_V4_ACTIVATION_BLOCK` is set) refuses a v4 release
unless: the block is set, epoch-aligned and after every earlier compiled
block; a signed lab and host list ships, verifies under the shipped signers,
is not the draft, and the pinned seed names it; that list (until one
ships, the draft under review) admits every entry of the pinned seed, so
the first signed list is adopted at the first step and can never be held
back for leaving fewer than `MENU_MIN_ENTRIES` (8), which would leave the
menu naming no list and every proposal refused; every seed entry is within
the price ceiling; the proposals' category
floor and the other values a v4 validator refuses to start without are
pinned, and the published-set seed ships, matches its pin and is observed
no more than `PUBLISHED_V4_SEED_MAX_AGE_EPOCHS` (150) epochs before the
activation epoch and through no later than the epoch the first v4 epoch
reads (`memo_v4.READ_LAG`): the seed is a starting point, which validators
and TDs bring forward by folding the root's reveals the chain names at each
boundary after it (item 6 of the memorisation entry), as they do after the
activation; the newest frozen finney compose is the live
one, sets every variable the miner's protocol selection reads
(`config.protocol_for_chain`, walked in config.py) to the compiled value,
sets nothing the compose checks forbid, and runs the miner on protocol 4 at
the block when the miner's own selection is executed on its environment;
every file v4 reads is in the package the image copies; the root's hotkey
table is one the menu reads can use (integer first epochs from 0, strictly
increasing, one canonical SS58 hotkey per row) and covers the first v4 epoch
(`tests/test_release_gate_v4.py`). The published-set seed is parsed with the
validator's own reader (`memo_v4.check_seed`), the shipped signers file is
the one `menu_rules.LIST_SIGNERS_SHA256` pins, the block must be an integer,
and the frozen compose's epoch interval is finney's (4320 s). The plain
check reports only how many of these are open.

### I1/I3 — v4: one bound (built ahead of v4; nothing reads it before a v4 activation block)

Every v4 seat decision (`docs/decisions/SEAT_MINIMAL_V4.md` 3.1: the odds
test, the real-picks check, the memorisation check) will be the bound of
`champion.lower_bound` (`fugal_subnet/seat_bound.py`, re-exported by
`champion`) on exact integer items: a margin, a minimum, the stratified
variance, and the pinned rational constants `A_ALPHA` and `LN2_UP`; no random
stream. Package S1 builds the function; the seat packages wire it in. What
must stay true:

1. **One formula, exact.** Items are ints in millionths of a point
   (`config.BOUND_UNITS_PER_POINT_V4`) grouped by category (strings). With
   n items, S their sum, D = S - margin x n, V the within-category sum of
   squared deviations (a Fraction) and rho = V n0 / n, the bound passes when
   n >= n_min, D > 0 and
   D^2 > (V + rho)(`BOUND_A_ALPHA_V4` + `BOUND_LN2_UP_V4` x
   bit_length(ceil(1 + n / n0))), evaluated in integers and Fractions. No
   float, no random stream, no numpy: the verdict cannot depend on a CPU, a
   library version or the order of items or categories. A window kept as
   per-category sums (`seat_bound.Stratum`, added and subtracted epoch by
   epoch) gives exactly the bound of its items.
2. **Never tighter than the mixture bound.** It is the one-sided
   normal-mixture confidence sequence at alpha = 1% with the variance
   estimated from the items, whose logarithms `9211/1000` (>= 2 ln 100) and
   `6932/10000` x the bit length (>= ln(1 + n / n0)) only widen: whenever it
   passes, the boundary with the true logarithms passes too. From n = n0
   (every rule's minimum is its n0) the widening is at most 8% of the log
   term.
3. **For a true mean at the margin the chance of ever passing is at most
   about alpha over any number of epochs**, so it is re-run every epoch with
   no confirmation phase. A mean exactly at the margin never passes (D = 0),
   and fewer items than the minimum never pass.
4. **What is published** is `Bound.lower`, the bound itself rounded down to
   a unit; the verdict is `Bound.passes`, the inequality above exactly (a
   margin at or above `lower + 1` never passes; one below `lower` always
   does once the minimum is met).
5. **A residual of the plug-in variance**: a category with one item carries
   no spread. The rules read a handful of categories with minimums of 1,000
   and 1,500 items; a rule that stratified finely would need a per-stratum
   minimum (the memorisation check leaves out benchmarks with fewer than 5).

Enforced by `tests/test_bound_v4.py` (the design's inequality exactly; the
result's types; the same verdict under key and item permutation, from sums
kept epoch by epoch, and in a second process; the rounded bound against a
high-precision reference and the float formula within one unit; never
passing where the 80-digit mixture boundary fails; the constants never below
their logarithms; a mean at the margin, a count short of the minimum, a
translation, a raised category, a lower margin or minimum, scaling and
doubled evidence; zero variance, one label, no items, a trillion labels and
values past 64 bits, negative gains, and every malformed input refused; and
3.9's simulations at a modest scale with fixed seeds: no head with no edge
seated in 300 epochs at sigma 4, 8 and 12 points and 160 to 2,000 labels an
epoch, nor any of 30 such heads, a head at the margin passing within 300
epochs at most 1.5% of the time (1.2% of 600 runs; the design measured
1.0%), and the median epochs to a take within max(1 epoch, 10%) of 3.9's
speed table, which states no tolerance (50 / 15 / 5 at 370 labels an epoch
against 51 / 15 / 5, and 20 / 6 / 2 at 1,000 against 19 / 6 / 2)), and the
`bound_v4` stage of `scripts/check_determinism.py` (adversarial items:
margins on the bound itself, a mean at the margin, no spread, single-label
categories, values past 64 bits, negative gains, a count one short, a
sliding window of sums; every category and item reversed under
`--perturb`).

Not yet enforced, because the code is in later packages: that the
memorisation check calls it (S5; the odds test and precedence, S4, call it
through `seat_bound.bound_from_strata` in `seat_v4.decide`, and the
real-picks check, S3, through `picks_v4.picks_check`); that `BOUND_A_ALPHA_V4`, `BOUND_LN2_UP_V4` and
`BOUND_UNITS_PER_POINT_V4` are in the consensus digest (with the other v4
constants at activation, W11: adding them now would change the digest every
v3 validator publishes).

### I3/I4 — v4: the labels a head is scored on (package S2; built ahead of v4, nothing reads it before a v4 activation block)

`docs/decisions/SEAT_MINIMAL_V4.md` 3.2; `fugal_subnet/labels_v4.py`. A v4
comparison of two heads X and Y (Y may be the reference R) reads a pooled
label only if (1) it was drawn in the last `SEAT_WINDOW_EPOCHS_V4` (48)
epochs; (2) it was drawn at or after both rows it reads count from: a head's
rows count from the first epoch whose boundary is at or after its first
commit (`copy_rule.first_epoch`; the seat package passes an earlier epoch for
a row a successor inherited, `Side.rows_since`), and a missing row counts
from the head's commit; (3) no coldkey that voted on it is X's or Y's owner
(R has no owner and no commit); (4) it is no proposal measurement pair (as
built, measurement results never enter the pool). What must stay true:

1. **Evidence is a function of the head, not of its serving.** A head's odds
   are computed for every epoch's pairs from its commit on, whether or not it
   served: in the epoch it is verified, and, for the window epochs it was not
   verified in, from its bytes and the stored pairs in the epoch it is next
   verified (`labels_v4.late_odds`), published in that reveal (`late_odds`).
   They are the same integers they would have been had it served
   (`head_odds.head_odds` by entry id equals `pair_odds_v4` on the epoch's
   menu, whose pairs hold only that menu's entries). A head is given odds
   whether or not it answered the scored questions.
2. **No head is judged on labels its operator supplied.** Each pair carries
   its voting coldkeys (`explore_pool.pair_voters`, the same filtering as
   `pool_pairs`), stored with the pairs and published (`labels_v4` in the
   reveal); a label any of them voted on is read for neither head.
3. **The sums are exact.** `labels_v4.gain_items` gives per category the
   label's weighted Brier loss of Y minus X's (explore_weights.apply), ints in
   millionths of a Brier point (the bound's unit); `gain_strata` gives the
   `seat_bound.Stratum` sums `seat_bound.bound_from_strata` reads. An epoch
   for which a node holds no odds of a head yet is read once it does.
   `gain_strata_many` gives the same sums for many pairs at once (the seat's
   comparisons, the published standing): each head's packed losses are
   computed once per epoch, a pair's items are the high words of one int64
   difference (the weighted rounding is exact: the proof is in `labels_v4`),
   squares are summed in int64 chunks sized by a bound on the items and
   carried in two limbs, and an epoch whose values leave the packed range
   is summed the per-pair way. The bound itself is evaluated in Python ints
   (one quotient, no per-term Fractions). Enforced by
   `tests/test_seat_v4_batched.py`: the sums against `gain_strata`, the
   bound against the Fraction one, and every seat decision (its published
   evidence byte for byte) against the previous `decide`, kept in the test.
4. **What is stored is what is published.** Per v4 epoch the window holds
   `labels_v4.LabelPairs` (the pairs, their voters, the odds a missing row is
   scored at) and per head one int16 per pair; the window file keeps both,
   including heads with no scored answers (`v2_epoch.save_window`; a v4
   window in its own layout, `V4_WINDOW_FORMAT`: 24 concatenated arrays with
   an index and one string table, uncompressed, every member's dtype and
   declared size checked against the archive before anything is read and
   every index against what it addresses, so a damaged or hostile file is
   refused before it allocates; a v1-v3 window is written and read exactly as
   before; `tests/test_v4_window_file.py`), the reveal
   publishes both (`labels_v4`, `head_odds`, `late_odds`, with
   `labels_digest` in `v2_state`), and `v2_rebuild` reads them back, refusing
   late odds for an epoch the window already holds odds of the head for.
5. **R is the earliest head of all**: `labels_v4.reference` (each entry's rate
   per benchmark over the window's pooled labels, shrunk to its overall rate,
   leave one epoch out; `odds_score.reference`), `labels_v4.REFERENCE` as a
   side with no owner and no commit.

Enforced by `tests/test_v4_labels.py` (each condition alone; the exact items
and their sums; odds for missed epochs equal to those served and published
then, from the commit and never before, for a head with no scored answers
too; the window file and the rebuild against the live window; a reveal
repeating an epoch's odds refused) and the `seat_v4` stage of
`scripts/check_determinism.py` (twenty epochs of the live epoch with misses
and late odds, an entry joining with heads with and without its row, one
whose allowance ends, one retired in its own, a head without scored answers;
the labels, every head's odds, the late odds and every pair's strata; every
input collection reversed under `--perturb`).

Not built here: R reading the proposal measurement labels and leaving out
pairs of an entry's older revisions (4.5; the window holds neither). The seat,
succession (a successor's inherited rows count from its predecessor's,
`Side.rows_since`) and the publication of each comparison's bound are package
S4's (the entry "I3 — v4: the seat in three rules").

### I3 — v4: new entries are usable at once (REPLACED for v4 by "the menu lifecycle", FAST_CROWN_V4.md 18)

This entry described a 20-epoch allowance in which a head without a new
entry's row was scored at the entry's measured rate. Section 18 replaces it:
an entry joins provisional, its labels drawn while it is provisional are
scored for no head (`labels_v4.PROVISIONAL_LABEL`, read by no comparison),
and from the epoch the gates make it full a head without its row is scored
at odds 0, "never answers correctly", 100 y^2 a label (`labels_v4._scored`).
What holds instead is the entry "I1/I3/I4/I6 — v4: the menu lifecycle" below.
The label's own epoch still decides, through that epoch's menu, and the value
is stored with the pair (`LabelPairs.missing`); a row for an entry not on
the epoch's menu is still never read.

### I2/I3/I4/I5 — v4: the scored questions are optional, all or none; rule B prices real picks at the epoch's table; the failure bound is absolute (package S3; built ahead of v4, nothing reaches it before a v4 activation block)

(Items 1, 2 and rule B's reading here are superseded by the fast crown's
one-epoch challenge: the entry "v4: the fast crown" below.)

Protocol v4 (`docs/decisions/SEAT_MINIMAL_V4.md` 3.4, 3.7, 5.4, 8) makes the
200 scored questions optional, checks a head about to take the seat on them
(rule B, the real-picks check), and replaces v3's comparison of exploration
with scored failures, which cannot be made without scored answers, by an
absolute bound. Package S3 builds the proof field, the TD's choice, the
verification, the bound, rule B as pure functions and their publication
(`fugal_subnet/picks_v4.py`); the seat package (S4) reads them. Nothing here
decides the seat. What must stay true:

1. **All of the slice or none, and the proof says which** (I3;
   `BenchmarkProof.scored`, `verify._v4_scored_check`). Every v4 proof
   carries its exploration (and any measurement due). Its `scored` field,
   in `content_hash` and so attested, says whether the TD answered the
   epoch's scored questions: true, and the scored results are exactly the
   slice, each question once; false, and there is no scored result at all.
   A v4 proof without the field, one with part of the slice whatever it
   says, and one that says false and carries scored results (another
   epoch's answers, say) are refused. The TD draws the slice either way
   (exploration and measurement are drawn outside it; its hash is in the
   proof). v1-v3 proofs never read the field: its key on one is neither
   hashed nor refused, and their bytes and hashes are pinned
   (`tests/test_proof_v4_fields.py`).
2. **The miner chooses, and the holder answers every epoch.** `--v4-scored`
   / `FUGAL_V4_SCORED` (operational; `td_v4.ScoredPolicy`): "auto", the
   default, answers only while the newest root reveal verified at the
   epoch's boundary names this miner's head as the seat holder
   (`td_v4.seat_holder_from_reveal`: the reveal's `champion.champion`,
   its bytes checked against the root's committed sha256), keeping its last
   answer when the read fails; "on" answers every epoch (a challenger, from
   about five epochs before it expects to beat the holder on odds); "off"
   never (an odds-only miner). A head that is not the holder is off by
   default.
3. **The failure bound is absolute, and an outage waives it** (I4;
   `picks_v4.failure_refusals`, `validator._v4_failure_bound`). A v4 proof
   with more than `FAIL_MAX_BP_V4` (34%) of its exploration calls failed is
   not counted (removed from the epoch's verified proofs before the pool,
   the cells, the menu step and the seat; published in `picks_v4.
   failure_refused`), and an epoch with more than 34% of a head's scored
   calls failed is not an answered epoch for it (`picks_v4.answered`), both
   except in an outage epoch. The outage is `menu_inputs.outage`, the menu
   step's health rule (the median coldkey failed more than
   `RETIRE_OUTAGE_RATE_BP`, 30%, of all its exploration calls), taken over
   the same verified proofs with the chain's coldkeys, so one operator's
   hotkeys are one voice in it. The v1-v3 relative check is not made on a
   v4 proof.
4. **Rule B, exactly** (I1, I3; `picks_v4.picks_check`). For each v4 epoch
   the validator records every serving head that answered with its value
   on each scored question: 100 x (g - t / V) points as an integer in
   millionths of a point, rounded half to even once (`call_value`), where g
   is the exact shared cell grade (`picks_v4.cell_grades`: the coldkey
   median of Fractions), t the epoch's pick cost of the routed entry in the
   question's category (`menu_state.pick_cost` on that epoch's MenuState,
   never the call's own tokens or the head's token means) and
   V = 1 / `routing_protocol.BENCHMARK_LAMBDA`; a failed call is 0. NV_i is
   the head's value minus the bar's, every question weighted 1, so
   identical heads give exactly 0 and the sign flips with the pair. The
   items, stratified by category, over the epochs of the head's answered
   run that the bar also answered, go to `seat_bound.lower_bound` with
   margin -`PICKS_MARGIN_POINTS_V4` (1.5 points), `PICKS_MIN_QUESTIONS_V4`
   (1,000) and n0 the same; with no bar (an open seat) the check is that
   the run holds 1,000 questions (`open_seat_check` compares with the last
   holder when 1,000 questions are paired with it). An epoch is priced when
   it is scored and its record is kept: a pick-cost refresh moves the price
   of later epochs only.
5. **A missed epoch restarts the run.** The answered run is the recorded
   epochs in which the head answered, back from the newest to the first it
   missed (`answered_run`): an exploration-only proof, a refused or absent
   proof, or too many failed scored calls is a miss. An epoch with no
   verified proof at all is not recorded and breaks nobody's run (as v3's
   `Window.current_run`). A holder's missed epoch is a fact the seat
   package reads (`answered_in`, and `answered` in the published evidence);
   this package does not act on it.
6. **The same cells.** The cells rule B grades with are those the pick-cost
   table's inputs count (`menu_inputs.graded_cells`): a scored (question,
   entry) some verified proof answered without failing, one vote per
   coldkey, over the same verified proofs (after the failure bound).
7. **Published and rebuilt** (I7). The window keeps each epoch's record
   (`Window.picks`, in the window file only when present); the reveal's
   `picks_v4` field carries the outage flag, the proofs not counted, each
   answered head's values (int64, base64) and every serving head's evidence
   (whether it answered, its run's first epoch and length, its questions,
   the bar, the paired epochs and questions, `passes` and the bound's
   published `lower`), and `v2_state.picks_digest` hashes it. A rebuild
   reads the records back from the reveals (`v2_rebuild.picks_from_reveal`),
   and reads an exploration-only head's empty routing row in a v4 reveal as
   no answers (in a v1-v3 reveal it is still refused). v1-v3 windows and
   reveals carry nothing new.

**Spend (I5).** An odds-only miner's bill is its exploration (the entry
"v4 exploration measures every entry": at most $0.55 expected for the
cost-aware part and $1.10 for coverage) plus its VM; answering the scored
questions adds about $4.65 an epoch at the v4 mix (SEAT_MINIMAL_V4.md 5.1),
only in the epochs the miner chooses or holds the seat.

Enforced by `tests/test_scored_optional_v4.py` (the field, its hash and
strict read; v1-v3 proofs unread; full and exploration-only proofs verify;
partial, unsaid and mismatched ones refused; the TD's exploration-only run
making no scored call; the miner's modes and its holder read),
`tests/test_picks_v4.py` (the constants; V from `BENCHMARK_LAMBDA` alone;
the arithmetic on a hand-worked case; identical heads; failed calls; the
exact cells equal the shared grades and the pick-cost table's cells; the
run, its restart, the open seat; each epoch at its own table across a
refresh; the failure bound, the outage, one operator's hotkeys; the
published form, the window file and the rebuild; run_v2_epoch end to end;
and the design's pass rates for routers one point better, equal and one
point worse reproduced at 1,000 seeded runs within five percentage
points), `tests/test_picks_v4_attacks.py` with the five picks cases of
`run_miner_attacks.py` (a partial scored set, foreign scored results on an
exploration-only proof, a holder that stops answering, answering only the
epochs that suit, junk exploration past the bound), each broken once its
rule is taken out, `tests/test_v4_loop.py` (a holder answering every epoch,
a miner switching on mid-run and an odds-only miner through both
validators' loops, every proof verified, the evidence identical on both
and rebuilt from the root's reveals), and the `picks_v4` stage of
`scripts/check_determinism.py`.

The seat step (S4, the entry "I3 — v4: the seat in three rules") reads
`picks_check`, `answered_run` and `answered_in`; the `picks_v4` evidence is
still published against the holder going into the epoch, and the seat's
own rule B verdict, against the bar it decides, is in `seat_v4`. Not yet
enforced, because the code is in another package: the S3 constants in the
consensus digest (W11, at activation, with the other v4 constants: adding
them now would change the digest every v3 validator publishes). Odds for
heads that did not answer the scored questions are S2's (the labels entry
above): every verified head has odds every epoch, an odds-only miner
included, and has no picks record.

### I3/I4 — v4: the memorisation check, and the unpublished reserve it reads (package S5; built ahead of v4, nothing reads it before a v4 activation block)

`docs/decisions/SEAT_MINIMAL_V4.md` 6; `fugal_subnet/memo_v4.py`. A head is
small on purpose; this is the second line against one whose skill comes from
outcomes already published. What must stay true:

1. **The published set is consensus state** (I1, I7). Per pool question, the
   first epoch whose reveal published an outcome on it (`memo_v4.Published`;
   `published_ids`: a scored question, any exploration answer, a pooled
   label). It is seeded at the activation from the pinned
   `routing_data/published_v4.json` (reviewed copy `data/published_v4.json`,
   sha256 pinned in `scripts/check_safety_invariants.py`; built by
   `scripts/build_published_v4.py` from the named reveals e00026189..e00026509
   with each reveal's sha256 in `data/observations/
   published_v4_e00026189-e00026509.json`, and `build --check` rebuilds it
   byte for byte), then brought forward by the one rule of item 6: the
   root's reveals, each folded at the epoch the chain names it, on every
   node and TD alike (`memo_sync.ChainSet`; never a node's own reveals).
   Epochs fold in order; a retry of a folded epoch changes nothing and one
   that would is refused. An epoch reads the set as of E - 2
   (`Published.view`), whatever a node has folded since, and that view's
   digest is in the epoch's reveal (`memo_v4.published_digest`) and, through
   `memo_digest`, in `v2_state`, the consensus view: a node with another
   published set differs there in the epoch it diverges.
2. **The reserve is spent by one draw, rationed for the whole field.** Epoch E
   spends F = ceil(R / `FRESH_RESERVE_EPOCHS_V4`) (1,200) of the R questions
   not published through E - 2, drawn by the epoch nonce alone on the pinned
   mix over the categories that still have a reserve
   (`memo_v4.fresh_questions`), and every hotkey's private set holds all of
   them. The number of hotkeys never changes F: a reserve of 1,200 or more
   lasts 60 days with no new questions. Every other draw of the epoch reads
   only questions published through E - 2 (`epoch_plan.plan_v4` with the
   published set, whose `pool_index` is a `memo_v4.ReserveIndex`: the scored
   slice, the rest of each private set, proposal measurement), and a
   category with fewer published questions than one draw can take
   (`PUBLISHED_MIN_PER_CATEGORY`, 463) is left out of them, so no draw comes
   back short. `plan_v4` is still the one constructor, and a plan built with
   the published set refuses another pool.
3. **Weights stay exact** (I1, I3). The fresh questions are members of the
   private set like any other: it keeps its 112 questions, they are split
   68 / 44 by the miner's keyed rank of every question, and each part's
   entries are its shares' draw (`explore_weights.assign_by_share`), so a
   private pair's weight is 1 / (M pi_private) exactly, fresh or not. No miner
   chooses which of its questions are fresh: the verifier's expectation
   (`verify.expected_v4` on the epoch's plan) holds the epoch's fresh set, and
   a proof answering another set fails the exploration check.
4. **The check, exactly** (`memo_v4.memo_evidence`). A head's labels against R
   (`labels_v4.eligible`) split into seen (the question was first published
   before the epoch its row counts from, `Side.row_since`) and unseen. Within
   each benchmark with at least `MEMO_MIN_QUESTIONS_V4` (5) questions in both
   parts, the gap is mean skill on seen minus unseen labels, weighted across
   benchmarks by 1 / (1/n_seen + 1/n_unseen). The bound's items are the
   questions (every operator answers every fresh question, so a question's
   labels move together), scaled by a common integer so that
   `seat_bound.bound_from_strata` evaluates the bound's inequality on the gap
   itself, in integers and Fractions. A head is refused when, with
   `MEMO_MIN_LABELS_V4` (500) labels in each part, the bound passes at
   `MEMO_MARGIN_POINTS_V4` (1 point). Comparing within benchmark is
   load-bearing: on the published v3 epochs two heads would be refused if the
   benchmarks were pooled, none is within them.
5. **The seat reads the verdict it publishes.** When `run_v2_epoch` is
   handed the epoch's memo (`memo_v4.EpochMemo`), every head the v4 seat
   keeps after `seat_v4.observe` (the holder, the history and every
   standing head, serving this epoch or not) is judged once, on the window
   holding the epoch, through the seat's own view of it (`seat_v4.side_of`:
   a successor's inherited rows count from its predecessor's commit, so
   succession does not restart the split) (`memo_v4.verdicts`). The refused
   heads are `seat_v4.decide`'s `memo_refused`; the same evidence is the
   reveal's `memo_v4` (`memo_for_reveal`, every head the seat keeps), with
   `memo_digest` and `seat_digest` (whose evidence carries each head's
   `refused` flag) in `v2_state`. Without a memo no head is refused and no
   memorisation field is written, the epoch exactly as before.

Enforced by `tests/test_memo_v4.py` (the fold, its retry and its refusals; the
view and its digest; the state document read strictly; what a reveal
publishes; a rebuild equal to the live fold; the seed's rebuild from the
committed observations, its counts on the pinned pool, and its reading for
a pool; F and the 60-day reserve whatever the hotkeys; the fresh questions
unpublished, the same on every node and for every hotkey, in every private
set; the scored slice, the rest of the private set and measurement
published only; the parts' sizes and entry draws unchanged; a fresh
question in the cost-aware part at 68/112; a category without a reserve and
one with too few published questions; the gap and the verdict against the
formula in Fractions; benchmarks left out, the minimum, questions as items,
eligibility, seen by row; and on the 44 published v3 epochs, from
`data/observations/memo_v4_labels_e00026466-e00026509.json`, no head refused
and two refused only when pooled), `tests/test_memo_v4_attacks.py` (a planted
memoriser refused once 500 unseen labels are in while an honest head of the
same overall skill is not; an honest head skilled on one benchmark not
refused; a proof that answers other questions than the fresh ones refused;
a validator with a different published set visible in `v2_state`; each
shown to get through with its rule taken out), the `memo_v4` stage of
`scripts/check_determinism.py`, and `check_safety_invariants
.check_published_v4_pinned`.

With the optional scored questions and rule B (the S3 entry above):
`tests/test_memo_v4_with_picks.py` (the TD's harness on an epoch's plan
with the published set makes a proof with the scored questions on and one
with them off, both verified against one expectation, whose exploration,
the fresh questions included, is each TD's own draw; the absolute failure
bound counts a failed fresh question like any other exploration call; a
window with labels, odds-only heads and picks records reads back with the
same memorisation evidence, and a v1-v3 window file holds nothing of v4;
one epoch given both `picks` and `memo` publishes both blocks and digests,
each input adding only its own fields), and `tests/test_v4_loop.py` (the
loop passes the memo every v4 epoch: the root and the follower publish the
same `memo_v4` and `memo_digest`, the published set each holds holding every
question a proof carried an outcome on). The seat's reading of the verdict (5): `tests/test_v4_seat_memo.py`
(v4 epochs through `run_v2_epoch` on a constructed window,
`fugal_subnet/attacks/memo_seat_world.py`: a refused holder is not the bar,
the seat opens to R and it is not paid, and the same epoch without the memo
holds it; a refused head cannot fill the seat or stand in rule C's order; a
holder not serving is still judged, and loses the seat to an honest
qualifier; a successor is refused from its inherited rows, where its own
commit would give it almost no labels; a follower handed every input in
reverse, its window and seat read back from file, publishes the same
`memo_v4`, `seat_v4`, `v2_state` and weights), the third scenario of
`tests/test_v4_seat_loop.py` (both validators' real loops, nothing handed to
them: the seed publishes the half of the pool a sharp head is confident on,
the fresh draws spend the other half, and the holder is refused once its
unseen labels are in, the seat open and unpaid on both, byte for byte),
`tests/test_v4_reveal_budget.py` (`memo_v4` at its largest shape inside the
reveal read bound), and the third part of the `seat_v4` stage of
`scripts/check_determinism.py` (those epochs with a refused head, inputs
reversed under `--perturb`).

6. **The set advances only by root reveals the chain names, the same on every
   node, and never waits on the root** (`fugal_subnet/memo_sync.py`,
   `ChainSet`; package W16). At each epoch boundary b every node (the root,
   every other validator, every miner's TD) reads the root's reveal
   commitment pinned at b's block (the read the menu makes); when it names a
   reveal newer than every one folded, that reveal's questions (its bytes
   hashing to the committed sha256) are folded as first published at
   b - 2, else nothing is. The set as of epoch E is the state after E's
   boundary: in steady state the root's reveals through E - 2, each at its
   own epoch. One view per epoch: E's plan (`epoch_plan.plan_v4(pool,
   published=<the view>, epoch=E)`, on the one pool object every draw of
   the epoch reads) and E's verdicts (`memo_v4.epoch_memo` on the plan's
   reserve index and the same view) read the same set, and the reveal names
   it (`memo_v4.published_digest`). Consequences:
   - *A root that is down or late moves nothing*: the set stays as it was and
     every TD proves, and every validator checks, on it. Its backlog is
     folded at the first boundary whose pinned read names it, on every node
     at once. A reveal never named at a boundary (published in a batch whose
     newest alone was committed, or behind a failed commitment) is never
     folded; a commitment naming a reveal older than one folded is ignored.
   - *Kept and restarted*: the state (set, newest reveal folded, last
     boundary read; `memo_sync.HELD_FORMAT`) is beside a validator's state
     file (`<state>.published_v4.json`, written atomically before it) and in
     a TD's state directory; a restart reads only the boundaries since. A
     node with no state anchors on the snapshot the reveal its current read
     names (then reads the boundaries after it), else starts from the seed
     and reads every boundary after it (across the block: the root's v3
     reveals). At most `MAX_FOLD_EPOCHS` boundaries are read in one go.
   - *The root publishes the snapshot* (`memo_sync.SnapshotPublisher`):
     before committing a reveal whose `memo_v4` names a view, it puts that
     view's canonical bytes (`memo_v4.snapshot_bytes`, whose sha256 is the
     named digest) in its store at `<netuid>/published/<digest>.json`,
     create-only, and reads them back. A reader takes bytes only when they
     hash to the digest, are at most `memo_v4.MAX_SNAPSHOT_BYTES` (8 MiB; the
     whole pinned pool published is about 1.5 MB), are the canonical
     document, name only this pool's questions and are the view named
     (`memo_v4.published_from_snapshot`).
   - *Every node checks the root*: a reveal it folds whose named view differs
     from its own state is an ALARM (a validator goes on with the chain's
     set; a TD makes no proof on it). The only epochs a node cannot check
     are those whose named reveal it cannot obtain (no bytes from the root's
     store or a mirror), as a menu whose bytes cannot be had: the anomaly
     `published_set_unavailable`. A failed pinned read fails the epoch,
     which is retried, as for the menu.

Enforced (6) by `tests/test_memo_sync.py` (the snapshot's bytes and their
strict reading; the snapshot published before the commitment; the rule:
steady state, the root down and its backlog, a stray older commitment, a
named reveal that cannot be had, a failed read, the state kept across a
restart, a snapshot anchor and tampered snapshot bytes, an ALARM and a TD
stopping on a contradicted named view, the TD's plan per epoch), the loops:
`tests/test_v4_loop.py` (every reveal carries `memo_v4` and the same
`memo_digest` on root and follower, the named digest the chain's view, the
epoch's fresh questions in every miner's exploration, the snapshot at its
digest, no ALARM), `tests/test_v4_published_loop.py` (the root down for one,
three and eight epochs: every TD proves and every follower checks every
proof every epoch, the backlog folded when named, a failed commitment and a
stray older one, a follower restarted during the outage and one restarted
without its set, a TD restarted with nothing kept, a `--once` retry; every
node's set the chain's and every reveal's `memo_v4` the same), the root's
outage in `scripts/rehearsal_v4.py` (proofs and checks through it),
`tests/test_v4_flip.py` (the set crosses the block from the seed), the third
scenario of `tests/test_v4_seat_loop.py` (a holder that memorises is refused
and the seat opens, nothing handed to the loops), the `published_v4` stage
of `scripts/check_determinism.py`, and the published-set cases of
`fugal_subnet/attacks/run_miner_attacks.py` (snapshot bytes past the cap,
another set's bytes, malformed, foreign ids, a question twice, a reveal
naming a set that leaves out published questions; each shown to get through
with its rule taken out in `tests/test_memo_sync.py`).

Still open: one view per epoch means a question first published in the
reveal of E - 1 counts as unpublished in E's verdicts for a head whose row
counts from E (its labels on it are unseen until the next epoch's view);
the window is judged again every epoch. While the root is down, questions
the followers' own reveals publish are not in the set (only the root's
committed reveals count), and a reveal never named at a boundary is never
folded: such questions stay in the reserve and may be drawn as fresh again
(at most F a epoch, about 14 at the pinned pool). A node with no state that
anchors exactly while the root's commitments are out of order (an older
reveal committed after a newer one) takes the newest reveal folded from the
commitment at the anchor's boundary; the root's own publisher never commits
out of order. The label, odds and seat digests of a validator after a missed
epoch are verify-and-resync's to bring back ("the v4 flip" above); its
published set agrees from its first v4 epoch. The v4 constants are in the
consensus digest (activation package). Residual: the check sees only
outcomes the subnet published; a miner who pays to label reserve questions
itself is not seen (about $560 for a 7,000-question reserve on a dozen
entries).

### I1/I3/I4/I6 — v4: the menu lifecycle: proposals filed on chain join first, provisional, and the field decides them (FAST_CROWN_V4.md 18; built ahead of v4, nothing reaches it before a v4 activation block)

`fugal_subnet/chain_proposals.py` (the documents, the commitment fields,
the admission and the delta), `fugal_subnet/join_step.py` (the root's step
and the followers' audit), `menu_state` (provisional records, the gates),
`menu_publish` and `td_v4` (every reader applying the delta). Epoch by
epoch, for a proposal filed in epoch k:

- **k: filed.** The miner serves a proposal document (canonical JSON, at
  most `PROPOSAL_MAX_BYTES`: its hotkey, model, reasoning, at most two
  pinned hosts, one quantization; no price, no limit) and commits
  `proposal:<sha256>` as its commitment's second field, beside its head in
  the first, one transaction (`set_commitment_fields`). The first field is
  the one every reader before v4 reads (`decode_metadata` reads field 0
  only), so a pre-v4 node reads every commitment as before.
- **k, block B: the root's pinned read.** B is `join_margin` blocks before
  the boundary of k + 1 (min(30, max(2, blocks per epoch / 5)); on finney
  30 blocks, 360 s, against a budget of 12 s of reads, 60 s of fetches,
  90 s of publication, 48 s for the commit and a 5-block slack:
  `commit_slack`). The root reads every commitment at B, the hotkeys its
  reveal of k - 1 lists as verified (`served_v4`: a proposer must have
  served the epoch before; of the well-formed proposals of hotkeys
  registered at B that served it, the `MAX_CONSIDERED` lowest hotkeys are
  considered, the cut made after that filter, `chain_proposals.candidates`),
  fetches each document by hash from the
  proposer's axon and each model's public listing, and admits in the order
  (the documents all at once, in one event loop with its own HTTP session,
  `join_step.ProposalFetcher`: each within `JOIN_FETCH_TIMEOUT_S` and
  `RESPONSE_MAX_BYTES`, no redirect followed, the whole within
  `JOIN_FETCH_BUDGET_S`; a box that hangs, refuses or sends too much costs
  only its own proposal, `unfetched`, which waits on chain;
  tests/test_v4_join_fetch.py on loopback servers),
  HMAC(block B's hash, proposal hash), which nobody chooses: one proposal
  per coldkey an epoch; cooldowns (id and coldkey); at most one join below
  ten serving hotkeys, two from ten; the cap and `MENU_MAX_PROVISIONAL`;
  an effort only on hosts whose listing takes `reasoning_effort`; a listed
  price within the ceiling (`menu_rules.MAX_PRICE_IN_PICO`,
  `MAX_PRICE_OUT_PICO`: $25 / $100 per million tokens, against $8 / $40
  for the pinned seed's dearest entries; `refused:price_above_ceiling`,
  re-run by every node's and TD's `check_delta`); the
  compiled rules under the list the base menu names. Each verdict is in the
  delta (`joined`, `not_served`, `unfetched`, `unread`, `malformed`,
  `coldkey_once`, `no_place`, `refused:<code>`), with the documents and the
  listing reads it used.
- **k, before the boundary: published, then committed.** The delta is put
  in the store by its hash and read back, then the root commits
  `join:<epoch id>:<sha256>` beside its reveal pointer (`RootFields`: each
  write carries the newest of both). A root that is late, absent or
  refused commits nothing for k + 1: every node applies an empty delta and
  the proposals stay on chain for the next boundary.
- **k + 1: provisional.** Every node and TD reads the root's join pointer
  at the boundary and applies the delta to the base menu
  (`chain_proposals.check_delta`: bytes within `DELTA_MAX_BYTES`, at most
  `MAX_SERVING` (65,536) serving hotkeys stated, canonical,
  for this base menu and this block, and the admission re-run on the
  delta's own contents gives exactly the delta). A refused delta is an
  empty one. The new entry is provisional (`scored_from` =
  `PROVISIONAL`, a record in `MenuState.provisional`), priced from the
  root's listing read, with the menu's median token means; every miner's
  coverage is aimed first at it (`explore_weights.coverage_first`); its
  calls run on the TD's own deadline and budget and are left out of the
  failure bound, the outage rule and health; its labels score no head. A
  TD does not call it when its own listing read prices it above the menu
  price by more than the price step or above the ceiling
  (`td_v4.overpriced`), and its exploration calls stop at
  `harness.PROVISIONAL_SPEND_CAP_USD` ($40) a box an epoch of REALISED
  cost, all provisional entries together (each attempt held back at its
  worst case first: `PROXY_MAX_ATTEMPTS` x (answer limit x output price +
  prompt bound x input price), at the dearer of menu and listing price).
  Followers recompute the admission from their own chain read and fetches
  and publish the differences (`join_audit`; ALARM like the menu audit).
- **k + 2: still provisional;** the step scoring k + 1 decides it.
- **k + 3: full or gone.** The gates read the field's labels from its join
  through the scored epoch: the median coldkey's answered share (each failed
  call retried once) at least `min_answered_bp`; its measured call costs
  something; some category at the accuracy floor; no full entry clearly
  better at no greater cost; no full sibling of the same model within
  `SIBLING_MARGIN_PPM` (2 points) of its accuracy unless more than
  `SIBLING_CHEAPER_BP` (20%) cheaper. Passing, it is full from the new
  epoch: its measured rate and (from 30 pairs) token means from its labels,
  its price from the epoch's votes, a missing row then scored at odds 0,
  and the 20-epoch displacement shield runs. A missing row counts from the
  head's LINE (`labels_v4.Side.line_since`: the earliest epoch any of its
  rows counts from, which is its own commit, or for a successor its line's
  origin commit while an origin row is on the menu), so re-committing a
  successor with the same rows does not restart it; a row a head adds
  counts from its commit (`Side.added`: the line's earlier epochs are
  missing), and a changed row or a head under another coldkey inherits
  nothing (`tests/test_v4_missing_row_line.py`). Every node derives the line
  from the published seat state (`since`, `rows_since`). Failing, it is retired with
  its gate's record and its proposer's coldkey waits the cooldown.

Removed with this: the proposer's measurement, its ledger and spending
ceiling, make-up epochs, the pending list and the queue, the proof's
`proposal` and `measure_for` fields. Where each concern lives now: a
cherry-picked or partial measurement (the field's draw, exploration
verified exactly; a proof with a measurement is refused as read); a
proposer's unbounded spend (no proposer spends; a dead new host costs a TD
at most `PROVISIONAL_BUDGET_S`); one coldkey filling the pipeline (one
admission per coldkey an epoch, the join limit, the provisional cap);
re-filing at once (id and coldkey cooldowns); a filing off the list (the
admission under the list in force); a root that lies about joins (every
node re-runs the admission on the delta; followers audit the chain read).

Enforced by `tests/test_v4_chain_proposals.py`, `tests/test_v4_join_loop.py`
and `tests/test_v4_loop.py` (through the shipped loops),
`tests/test_menu_entry_listing.py` (efforts, siblings, first-party marks
and tiers against the recorded listing), `tests/test_reveal_read_limit.py`
(one reveal size limit for every reader), the root-delta and proposal cases
of `run_miner_attacks.py`, the `menu_derivation` and `provisional_gates`
stages of `scripts/check_determinism.py`, and `scripts/rehearsal_v4.py`
(filed at F+1, provisional at F+2, full from F+4).

### I3 — v4: the seat in three rules (package S4; built ahead of v4, nothing reaches it before a v4 activation block)

`docs/decisions/SEAT_MINIMAL_V4.md` 3.3 to 3.8, 4.3, 4.4;
`fugal_subnet/seat_v4.py`, called by `run_v2_epoch` for a v4 epoch. One
pure step an epoch, `seat_v4.seat_step(prev, epoch=, window=, serving=, rows=,
copy_state=, menu_ids=, blocks_per_epoch=, floor_block=, salt=,
memo_refused=, allow_take=)`: `observe` (the epoch's heads: seen, new, or a
successor), then `decide`. What must stay true:

1. **The bar.** The holder is the bar when it stands (it answered the
   scored questions, at most 34% failed, in this epoch or one of the
   `STANDING_EPOCHS_V4` - 1 = 2 before it), R does not beat it (rule A) and
   the memorisation verdict does not refuse it; otherwise the bar is R and
   the seat is open.
2. **Rule A, the odds test.** X beats Y when `seat_bound` passes on X's Brier
   gain over Y on their labels (the entry "v4: the labels a head is scored
   on"), margin `ODDS_MARGIN_POINTS_V4` (0.1 Brier points),
   `ODDS_MIN_PAIRS_V4` (1,500) labels and n0 the same.
3. **Contenders.** A head other than the holder is a contender when it
   stands (verified in this epoch or one of the two before it, itself or a
   predecessor it succeeded), answered the scored questions in one of those
   epochs, is not refused, beats the bar (A), and is not shown to route
   worse than the holder (the bound of the holder's net value over it, margin
   +1.5 points, on its answered run, does not pass). R is the earliest
   contender of all. A head that never answers the scored questions bars no
   one.
4. **Qualifiers and rule C.** A qualifier is a contender that answered the
   scored questions this epoch, passes rule B against the holder (or, with
   the seat open, against the last holder if 1,000 of its questions are
   paired with it, else the count of 1,000 alone), and beats (A) every
   contender earlier than it, R first (`seat_v4.must_beat`). Precedence is the
   first commit of the head or of its earliest predecessor; commits at or
   before the v4 activation block tie at it and are ordered by the digest
   salted with its block hash (`copy_rule.seniority`); then the head's hash.
5. **The seat.** If there are qualifiers the one committed last takes the
   seat; it is unique (every other qualifier is a contender earlier than it,
   so it beat each of them), and the holder it replaces goes to the history.
   Otherwise the holder keeps the seat, open or not: a seat is never emptied
   without a replacement. At the v4 activation the seat is empty, and the
   miner share burns until the first qualifier (about five epochs: 1,500
   labels over R and a 1,000-question run). No arrival order, no random
   stream and no ranking of margins is read.
6. **Pay.** 85 / 10 / 5 (`PAYOUT_SEAT_SHARES`) over the holder and the two
   holders before it, to the UID serving the seat's head or its successor. A
   seat pays when its head is live (verified; the holder with its scored
   answers this epoch), R does not beat it and the memorisation verdict does
   not refuse it; an unpaid seat's share goes to the holder, an unpaid
   holder's share burns. No weight ramp, and `MAX_WEIGHT_DELTA` is not
   applied to a v4 vector.
7. **Ownership.** `copy_rule.classify`, unchanged: an exact copy under
   another coldkey is not a head (not scored, seated or paid). Under v4 the
   copy rule keeps a retired head's record for `STANDING_EPOCHS_V4` epochs,
   and the seat serves only a head whose record names the first commit it
   recorded (`seat_v4.seat_serving`).
8. **Succession.** A head first seen this epoch is the successor of the
   newest head its hotkey committed before it when both belong to the same
   coldkey, that head is not served this epoch, and every row it has for an
   entry on this epoch's menu is in the new head byte for byte (per-row
   digests `sha256(entry id || 0x00 || weights || bias)`, little-endian
   float64, published in the seat state). A successor inherits the seat or
   history seat, the precedence, the standing and the evidence: each
   inherited row counts from its predecessor's (`rows_since`, so its odds are
   computed from there), and its predecessors' real-picks answers are read as
   its own (`lineage_records`, at most `MAX_PREDECESSORS` back: the evidence
   window, `SEAT_WINDOW_EPOCHS_V4` (48); each predecessor was its line's head
   for at least one epoch, so one it no longer keeps was head only in epochs
   that have left the window, however often the line is re-committed). Its
   new rows count from its own commit. A head that changes an existing row is a
   new head; one under another coldkey or hotkey inherits nothing.
   **A successor's odds before its own commit are its predecessors' stored
   odds** (`labels_v4.predecessor_odds`, read by `labels_v4._scored`, so by
   rules A and C, R's test and the memorisation check alike): for an epoch
   before its commit, the odds of the most recent of its `MAX_PREDECESSORS`
   predecessors that has odds for that epoch and was the line's head then
   (the epoch is before the `since` of the head after it; the seat state
   keeps each predecessor's own `since`, `HeadState.predecessor_since`,
   published as `"hash:since"`). They are exact on every label the successor
   is scored on: before its commit those are the labels of its inherited
   rows, each byte for byte in every head of the line since the one that
   added it, and a head's odds on a pair depend only on the row for its
   entry. A predecessor's odds for an epoch after it was succeeded (its
   bytes served again) are not read, a gap in a predecessor's own epochs is
   not filled from an older one, and an epoch no predecessor covers (a gap,
   or older than the oldest predecessor kept) is computed from the successor's
   own bytes as late odds; `v2_epoch.late_epochs` lists only those. Nothing
   new is published for an epoch read this way: the odds are the ones the
   predecessor's reveals published, which a rebuild and a replay hold for as
   long as the epoch is in the window (the window drops an epoch's odds and
   its labels together). A head with no predecessor reads and publishes
   exactly what it did before.
9. **A holder lacking a row when an allowance ends** is scored certainly
   wrong on that entry's labels; once R beats it the seat is open, and the
   latest qualifier takes it that epoch; with none the holder keeps the seat,
   unpaid while R beats it. No ghost and no burn while a qualifier exists. A
   holder that adds the row by succession keeps everything.
10. **The memorisation verdict is an input** (`memo_refused`, head hashes):
    a refused holder opens the seat, a refused head is neither a contender
    nor a qualifier, and is not paid. `run_v2_epoch` computes it when handed
    the epoch's memo, for every head `observe` keeps, through `side_of` (the
    memorisation entry above, item 5); without a memo it is empty.
11. **Published and rebuilt.** The seat state is `v2_state.seat`
    (`seat_v4.to_dict`, marked `"protocol": 4`: the holder as `champion`, the
    epoch it took the seat, the history, and every head the seat keeps with
    its owner, hotkey, first commit, `since`, `seen`, origin, inherited rows,
    predecessors and row digests); the evidence is the reveal's `seat_v4`
    block (the bar, whether the seat is open, the holder's standing and R's
    bound over it, the precedence order, and per head its precedence,
    standing, rule A against the bar, the reverse picks bound, rule B, each
    rule C comparison as `index:labels:bound:passes`, whether it qualifies
    and the reasons it did not, in words), hashed in `v2_state.seat_digest`;
    the reveal's `champion` block names the holder (`champion.champion`,
    which `td_v4.seat_holder_from_reveal` reads) and the event. A v4 reveal
    writes no `pending`, `confirm_from`, `ghosts`, `high_water`, `burn_in`,
    `odds_shadow` or `frame_excluded_coldkeys`. A rebuild from the root's
    reveals reproduces the seat state and window; a follower's re-anchor
    takes the reference's state and replays its own epochs after the
    snapshot on it (`v4_replay`, "Verify-and-resync" above;
    `seat_v4.carry_seen` serves only a re-anchor made without the epochs'
    records, which the validator does not make). Late odds are published at
    most `LATE_ODDS_SLOTS_V4` (32) head-epochs a reveal, newest first, the
    heads in an order that turns with the epoch; the rest follow in the next
    epochs. **The heads a seat decision reads take their rounds first**
    (`seat_v4.late_odds_first`, passed to `v2_epoch.late_epochs` as
    `first`): the seat holder and every head that, itself or through a
    predecessor it succeeded, has a counted challenge in this epoch or one
    of the `STANDING_EPOCHS_V4 - 1` before it (`seat_v4.challenged_recently`,
    the same predicate that makes a head a contender), in
    the schedule's order among themselves; every other head then takes its
    rounds with the slots left. A pure function of the window and the seat
    state, so every node chooses the same; with no such head, or slots for
    every head-epoch, the choice is what it was without the priority
    (`tests/test_v4_late_odds_priority.py`; the fourth part of the `seat_v4`
    determinism stage holds a returning head that answers against two that
    owe nine epochs each). A successor takes none for an epoch its predecessors' odds stand
    for (item 8). **One read bound for a reveal**: every reader of a reveal
    names `reveal_sync.MAX_REVEAL_BYTES` (96 MiB), defined there once: the
    follower sync's fetch, the validator's read of its own reveal, the
    miner's holder read (`td_v4.MAX_HOLDER_REVEAL_BYTES`) and the root
    summaries' read (`menu_publish.MAX_REVEAL_READ_BYTES`), so no node refuses
    a reveal another reads. A worst-case reveal at 256 UIDs is held to three
    quarters of it (72 MiB; `tests/test_v4_reveal_budget.py`), so growth is
    noticed with room to spare. The bound decides no reveal's content: the
    v1-v3 reveals the same fetch reads are about 0.6 MB, and a larger bound
    only lets a node read more of an object it then hashes and checks as
    before.
12. **What v4 does not run** (section 12): the bootstrap, the confirmation
    phase, ghosts and resume, the high-water marks, the accuracy floor, the
    hard weight, the first-fill ranking, the burn-in counter, the senior bars
    as a separate rule, v1 dedup, the v1-frame and agreement detectors (the
    plausibility check reads `EXPLORE_PRIVATE_MIN_SAMPLES_V4`), and the
    weight cap. The v1-v3 path is unchanged.

**Residuals, accepted (3.9).** A copy of a head absent for more than
`STANDING_EPOCHS_V4` epochs is no longer barred by it
(`tests/test_v4_seat_attacks.py` documents it); the holder withholding its
scored answers is unpaid for each such epoch; an earlier contender can fit a
later head's new row from its published odds and add it by succession (4.4).

Enforced by `tests/test_seat_v4.py` (every clause on the seat's synthetic
world, `fugal_subnet/attacks/seat_v4_world.py`, at the design's constants:
the fill, the bar, one, two and three missed epochs, an open seat with and
without a qualifier, R beating the holder, the memorisation verdict, the
latest qualifier unique and order-independent, equal heads, a near-copy
through a miss, a contender that stops answering or routes worse, an earlier
head that never answers, picks two points worse, the salted tie, an
untrusted window, 85 / 10 / 5 with unpaid shares, succession and its three
refusals, a successor contender against a later near-copy, the three cases of
4.3, the published state and its refusals, a re-anchor, and that the v4 seat
names none of the removed rules' constants), `tests/test_v4_successor_odds.py`
(the predecessors' odds two and eight deep and not beyond, a predecessor
with gaps, odds after a succession not read, a successor that added rows, a
changed row, another coldkey or hotkey reading nothing, the evidence read
through the predecessors equal to the successor's own, `late_epochs` never
listing a covered epoch, and at the design's late-odds bound a challenger
taking, and R opening the seat, in the same epoch whether the holder is
re-committed each epoch or not), `tests/test_v4_successor_odds_loop.py`
(the same take through the real loops, root and follower agreeing, the
rebuild and a replay across the successions), `tests/test_v4_seat_attacks.py`
with the four seat cases of `run_miner_attacks.py` (a near-copy of an
earlier head, the same while the original misses, a changed head borrowing
an earlier commit's precedence, a head that is re-committed each epoch
keeping its evidence), each broken once its rule is taken out (rule C, the
standing window, byte-identical succession, the predecessors' odds),
`tests/test_v4_seat_loop.py`
(the scenarios of section 11 through a root's and a follower's real loops and
real miner epochs, the follower given every input in reverse order: the fill,
an odds-only miner turning its scored questions on and taking the seat, a
near-copy and an exact copy never winning, a new entry joining with heads
with and without its row, the holder adding the row by succession, two
missed epochs kept and three lost to the latest qualifier; the rebuild from
the root's reveals, the miner's holder read), `tests/test_v4_reveal_budget.py`
(the worst-case reveal), `tests/test_seat_v4_simulation.py` (no head with
zero edge seated, and a head 0.5 points better seated in about eight epochs
at the recommended budget; `scripts/experiments/experiment_seat_v4.py`), and
the `seat_v4` stage of `scripts/check_determinism.py`.

Not yet enforced, because the code is in other packages: a memo handed to
the live epoch (the validator, its sync and its rebuild pass none today, so
`memo_refused` is empty on every live path; the memorisation entry above);
and the S4 constants (`ODDS_MARGIN_POINTS_V4`,
`ODDS_MIN_PAIRS_V4`, `STANDING_EPOCHS_V4`, `LATE_ODDS_SLOTS_V4`,
`EXPLORE_PRIVATE_MIN_SAMPLES_V4`) in the consensus digest (W11, at
activation, with the other v4 constants).

## Attack surface by actor

**Miners** control: the bytes their axon returns, their head weights, when they
commit, and how many identities they register. TEE constrains them: results are
hardware-attested, costs are metered, and the runtime image is measurement-pinned.

**Other validators** control: their own published reveals and weights. Yuma
consensus plus `MAX_WEIGHT_DELTA` bound the damage (for a v4 vector Yuma
alone: every input to it is a published consensus decision); `consensus.py`
detects it.

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

**Protocol v3 code grading (built 2026-11-16, not yet active).** v3 grades code
with a second pinned file, `fugal_subnet/graders_code.py` (`exec_stdio`,
`exec_func`), leaving `graders.py` untouched. Its child runs under a seccomp
filter (no sockets, no fork/exec, no open-for-write or path changes, no ptrace,
no signals or limits on other processes), CPU-time rlimits, `-S` (stdlib only),
a pinned hash seed and a cleared environment, and the grading process makes
itself non-dumpable so the child cannot read the pool's gold from its memory.
The grade is decided in the parent from exit status, measured CPU and stdout.
Pinned as `CODE_GRADER_SHA256` in `check_safety_invariants.py` (a byte change
fails the check); hostile programs in `fugal_subnet/attacks/run_code_attacks.py`
(CI). Determinism rests on CPU-time limits plus calibration that keeps only tests
a reference passes in ≤ 25% of the limit (`scripts/calibrate_code_tests.py`),
with shared grades absorbing the borderline residual. The harness must run
`sandbox_selftest()` at start and refuse code questions if it fails. Design:
[CODE_GRADER_V3.md](decisions/CODE_GRADER_V3.md).

The v3 review (2026-11-16) hardened it before it shipped: no priority or
scheduling call at all and no `fchmod`/`fchown` (with `PRIO_USER` and
`who=0`, `setpriority` reached every process of the uid); new threads get
8 MiB stacks, two malloc arenas and a fixed CPU count of 4, so at least 64
threads start and `ThreadPoolExecutor()` behaves the same on every TD (the
main thread's 64 MiB stack is unchanged); the wall backstop is
max(3 × CPU + 2 s, 20 s) a run and 60 s a question. **Release gate:** a pool
with `exec_stdio`/`exec_func` questions is served only by an image whose
harness grades them through `graders_code` with a working sandbox
(`fugal_subnet/code_gate.py`: the miner refuses to start otherwise, and
`check_safety_invariants` checks it when the v3 pool manifest holds code),
because `graders.grade` returns 0 for both checkers
(`tests/test_code_release_gate.py`, `tests/test_graders_code_review.py`).

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
is not accepted live (only a compose hash replayed out of the log is); the
compose-hash path is the only one. The analysis below is kept as the record of the gap.

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

### I1/I2/I3 — v4: the fast crown: a one-epoch challenge, the holder on exploration, one bound tuned to 100 (FAST_CROWN_V4.md packages F1-F7; built ahead of v4, nothing reaches it before a v4 activation block)

`docs/decisions/FAST_CROWN_V4.md` (sections 0, 2-6, 12) replaces the 200
scored questions every epoch (the S3 entry above, items 1 and 2, and rule B's
reading there) by a challenge a miner answers in one epoch, and lets the
holder stand and be paid on its exploration alone. What must stay true:

1. **One bound, tuned to 100** (I1/I3; `config.BOUND_N0_V4`). Rules A and B
   and rule B's reverse bound use n0 = 100; rule A needs
   `ODDS_MIN_PAIRS_V4` 300 labels, rule B `PICKS_MIN_QUESTIONS_V4` 500 items.
   The memorisation check keeps n0 = its minimum.
2. **The challenge is a prefix of one list** (I1/I3;
   `epoch_plan.challenge_list_v4`, `verify._v4_scored_check`). Each epoch's
   list is `CHALLENGE_LIST_SIZE_V4` questions: the scored slice first, then
   published questions (the plan's `ReserveIndex`) drawn on the scored mix by
   the epoch's challenge nonce, in keyed order. A proof with `scored` true
   carries `challenge` {size 500 or 1000, holder}; its own results are
   exactly the first `size` questions, each once; its holder-side results
   (`holder_pick` true) are distinct questions of the prefix, each to an
   entry other than its own pick; with no holder named, none. Anything else
   refuses the proof. `scored` false: no challenge, no scored or holder-side
   result. v1-v3 proofs never read either field.
3. **The holder is the epoch's, and every pick is recomputed** (I1/I3;
   `seat_v4.challenge_holder`, `v2_epoch._challenge_epoch`). A challenge in
   epoch E pairs with the holder after E - `CHALLENGE_HOLDER_LAG_V4` (2),
   which the root's newest reveal at E's boundary names; the seat state
   keeps the holders of its last two decided epochs (`holders`, published,
   read by a rebuild and a re-anchor), and a node that did not run E - 2
   reads the newest one at or before it (the seat as it last decided it). A
   TD makes a challenge only when the root's reveal of E - 2 itself is to be
   had. Every validator recomputes the challenger's own
   pick on each question from its verified head and the holder's from the
   holder's bytes (`picks_v4.challenge_picks`: `head_odds.picks_and_odds_v4`
   on the epoch's menu and pick-cost table over its FULL entries only, so a
   provisional entry is never a challenge pick on either side, in the TD,
   on every node and in `holder_picks`; a head with odds for no full entry on
   a question of its prefix makes no challenge); a challenge naming another holder, or stating another
   pick or holder-side call, is refused and not counted (published with its
   reason): the run restarts. An entry without a row is never the holder's
   pick; a question with no entry the holder has odds for has no holder
   pick. The holder's bytes come from any verified proof carrying them, else
   from the validator's replay store or the root's head publication, checked
   by hash; without them the epoch is not scored here
   (`v2_epoch.HolderHeadMissing`).
4. **Items fail closed, up to a cap** (I3; `picks_v4.challenge_items`,
   `challenge_verdict`). 0 where the two picks are the same entry; else the
   challenger's value minus the holder's, 100 x (grade - table cost), the
   challenger's failed call graded 0 and the holder's failed call 1, each at
   its table cost; no holder pick: the holder's side is 0. A holder-side
   call the challenger's TD did not start (`not_started` in the proof: its
   time or dollar budget ran out first) is no item (`NO_ITEM`, the int32
   minimum, which no real item takes: real items are held within
   INT32_MIN + 1 .. INT32_MAX). When more than `HOLDER_FAIL_CAP_BP_V4`
   (10%) of a challenge's holder-side calls failed or were not started, the
   challenge is **void**: its items are all `NO_ITEM`, it counts for and
   against nobody (no item, not `answered_in`, no mean, no cell), and
   `picks_v4.void` publishes {head: reason}. 10% is above every honest rate
   measured (no head-epoch past 3 failed calls of ~1,000; flaky entries 5-10%)
   and keeps what failing the holder's side up to the cap can cost a
   challenger to about 1.5 points (FAST_CROWN_V4.md 1;
   `experiment_fast_crown_v4.py v`). The grade is the call's own. A failed
   or unmade call never raises an item (attack case 1). Each epoch's mean
   item per entry (by the challenger's entry, over differing picks, counted
   items only) is published as [n, sum].
5. **Evidence over attempts is the answered run** (I3; `picks_check`).
   Rule B reads the unbroken run of epochs with a counted or void challenge,
   counted items paired with the bar (or a predecessor it succeeded); a gap
   restarts it. A void challenge neither counts nor breaks the run, but a
   run passes through at most `VOID_MAX_IN_RUN_V4` (1) of them: the next one
   back ends it as a gap does. Unlimited voids would let a challenger keep
   only its good epochs (the note's "never resets, bad epochs withheld":
   75% false passes at -1.75 points and 500 questions; with one void a run, 0;
   `experiment_fast_crown_v4.py v`).
6. **Standing and pay by verification** (I3; `seat_v4.decide`). Every
   head, the holder included, stands while verified (by its own coldkey:
   `seat_serving`) in one of the last `STANDING_EPOCHS_V4`; a seat is paid
   when its head is verified that epoch, R does not beat it and it is not
   refused. A contender bars later heads while it has a counted challenge
   in the last `STANDING_EPOCHS_V4` epochs; a qualifier has one this epoch.
7. **Whose serving counts** (I3; `seat_v4.seat_serving`,
   `lineage_records`). Standing, pay and challenge evidence read the seat's
   view of who owns a head: the recorded commit, or the same bytes committed
   again by the coldkey that owns the head; never another coldkey's commit of
   the same bytes after the first-commit record lapsed. A predecessor's
   challenges are read as its successor's only from epochs before the
   successor's commit (`tests/test_v4_seat_ownership.py`).
8. **The TD challenges only what it can verify, inside its budget**
   (I2/I8; `td_v4.ChallengePolicy`, `harness._run_benchmark_v4`). The
   operator's `--v4-challenge` (0, 500 or 1000; provisioning field
   `v4_challenge`, parsed exactly) sets the size; the holder is read from
   the root's reveal of E - 2 (bytes checked against the root's
   commitment) and its head bytes checked against that hash in the TD. No
   challenge when either cannot be verified. The TD starts the challenger's
   own calls first, in list order, then the holder-side calls with
   exploration in the nonce's order (`picks_v4.call_order_key`), deciding
   each start in that order; no challenge call is started more than
   `CHALLENGE_BUDGET_S_V4` after the first call, and a call started keeps its
   outcome. The dollar ceiling (`--v4-challenge-usd`, provisioning field
   `v4_challenge_usd`: whole dollars 1 to `CHALLENGE_SPEND_MAX_USD_V4`
   (50,000), default `CHALLENGE_SPEND_USD_V4` (100), parsed exactly, shown
   on the READY line) bounds realised spend with an estimate pre-check: no
   challenge whose estimate (`td_v4.challenge_estimate`: each own pick and
   each differing holder pick at its entry's pick cost, the epoch's
   pick-cost table) exceeds it, logged with both figures; and no challenge
   call started once the challenge's realised cost plus the estimate of its
   calls in flight has reached it, cut as the time budget cuts (calls in
   flight then finish and are paid: at most HARNESS_CONCURRENCY of them)
   (v0.6.0 provisioning unchanged: a push without the field is the push
   before v4). The challenge is reported at the largest size whose own calls
   were all started (below 500: no challenge, an unanswered attempt); the
   holder-side calls not started are in the proof marked `not_started`, and
   verification refuses a proof whose `not_started` calls are not the last
   of its holder-side calls in that order (`verify._v4_scored_check`). What a
   miner pays: every call started (own and holder side), whatever the
   outcome. What counts: in a counted challenge every own call and every
   holder-side call made (failed ones as the holder right); in a void one
   nothing, and its run goes on (once); with no challenge reported, nothing,
   and the run restarts.
9. **The reveal fits** (I2; `tests/test_v4_reveal_budget.py`). `picks_v4`
   publishes the holder, the list as [id, category], challengers' items as
   int32, per-entry means, the challenge calls past the slice's shared
   cells summed per (entry, category, holder-side calls not started left
   out), the holder's computed slice picks, refused challenges, void
   challenges with their reason (`void`), and the holder's routing value
   against R's routing
   (`routing_vs_r`: published only, deciding nothing). A v4 reveal carries
   no per-UID `failed` lists. At 256 challengers of 1,000 the worst case
   stays at least 3 MiB under 64 (60.0 MiB measured).
10. **The holder's picks are published every epoch it holds** (I1/I2;
   `picks_v4.holder_picks_doc`, `v2_epoch._publish_holder_picks`).
   `picks_v4.holder_picks` is {"head": the seat holder `champion.champion`
   names, "picks": base64 of one little-endian int8 per question of
   `picks_v4.list`, in list order, the index of the picked entry in the
   reveal's `models`, -1 for no pick}, computed by every validator with
   `picks_v4.challenge_picks` (full entries only) on the epoch's menu and
   pick-cost table, so
   a reader holding the head's bytes, the menu the reveal names and the
   prompts recomputes each one; `{}` while the seat is empty. It is part of
   the digested `picks_v4` field (root and followers byte-identical);
   without the holder's bytes a node does not score the epoch.

Enforced by `tests/test_fast_crown_v4.py`, `tests/test_v4_seat_ownership.py`,
the seat loops (`tests/test_v4_seat_loop.py`), the reveal budget test and
the determinism stages `seat_v4` and `picks_v4`.

### I1/I3 — v4: one account of the merged protocol (the fast crown, chain proposals and the epoch-cost work together; built ahead of v4, nothing reaches it before a v4 activation block)

Where an earlier v4 entry above says otherwise, this account holds; the
entries above give each part's detail and tests.

1. **The seat and the challenge.** A head, the holder included, stands while
   it (or a predecessor) was verified in one of the last `STANDING_EPOCHS_V4`
   epochs; the holder answers no question and is paid on verification alone.
   A contender has a counted challenge in that span
   (`seat_v4.challenged_recently`); a qualifier one this epoch. A challenge
   is the first 500 or 1,000 questions of the epoch's challenge list (the
   slice first), the challenger's own pick and the holder's where they
   differ, the holder being the one after E - 2; every validator recomputes
   both picks (`head_odds.picks_and_odds_v4`, picks chosen exactly by
   `head_odds.pick_rows`). Rules A and B tune the bound to `BOUND_N0_V4` =
   100; rule A needs 300 labels, rule B 500 items. Rule A's comparisons are
   computed in batches (`labels_v4.gain_strata_many`), held equal to the
   per-pair path (`seat_v4.decide(per_pair=True)`) on every semantic below
   (`tests/test_seat_v4_batched.py`). A validator that cannot obtain the
   holder's head does not score the epoch (`HolderHeadMissing`); it seeks
   the bytes in one order: what it holds, the bytes miners served this
   epoch, the root's head publication, which publishes a holder and its
   successor with the reveal that names them.
2. **The menu lifecycle.** A miner files a proposal by committing
   `proposal:<sha256>` as its second commitment field and serving the
   document; the root admits at its pinned read before the boundary and
   commits `join:<epoch>[:<sha256>]`; the entry is provisional for two epochs
   (coverage aimed at it, its calls outside the failure bound, outage rule
   and health, its labels scoring no head), then full or retired. No proof
   carries a proposal or a measurement. A head carrying a provisional
   entry's row may route its users to it, but a provisional entry is never a
   challenge pick, on either side (`picks_v4.challenge_picks`: the TD, every
   node and `holder_picks` compute a challenge's picks over full entries
   only); `routing_vs_r` takes the holder's challenge picks, reads no
   provisional label, and R never picks a provisional entry.
3. **Labels and missing rows.** A label drawn while its entry was
   provisional is read by no comparison. From full, a missing row is odds 0,
   counted from the head's line (`labels_v4.Side.line_since`: the line
   origin's own commit, published as the oldest predecessor's "hash:since").
   A row the line added (`HeadState.added`, published) counts from its own
   start, and the entry's labels from the line's start up to then are scored
   as a missing row is (odds 0), for every later generation of the line; a
   successor's odds before its commit are its predecessors' stored odds.
   Every reader (the per-pair sums, the batched sums, the memorisation split)
   takes each side's per-entry starts from `Side._starts` on the odds the side
   is scored from.
4. **Late odds.** At most `LATE_ODDS_SLOTS_V4` head-epochs a reveal; the
   holder and every head with a counted challenge in the last
   `STANDING_EPOCHS_V4` epochs take their rounds first (`late_odds_first`), a
   function of the window and the seat state alone.
5. **The reveal's fields and size.** One read limit, `MAX_REVEAL_BYTES`
   (96 MiB). `picks_v4` carries `holder`, `list`, `items` (int32,
   challengers only), `item_means`, `cells`, `refused`, `routing_vs_r`,
   `holder_picks` and `evidence`, and no per-UID `failed` lists; a
   validator's reveal carries `served_v4` and, on followers, `join_audit`.
   The worst case (256 challengers of 1,000, 48 predecessors, 8 provisional
   entries, the holder's picks, the largest join audit) stays within 72 MiB
   (`tests/test_v4_reveal_budget.py`).
6. **The flip.** Unchanged by the merge: a v4 commitment field 0 is the
   bare head hash a pre-v4 node reads; the proposal and join fields are
   second fields written only from the block; no pinned read asks for a
   block the chain has not produced (the loops assert it).

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

**The product reads reveals from the public store without a chain check
(open, operational).** The product that follows the seat holder reads the
root's reveals from the public bucket and checks a published head against a
reveal (head parity: the head's routing decisions on the reveal's questions
must be the reveal's), which binds a head to a reveal. Nothing on the
product's side checks a reveal against the root's on-chain reveal
commitment (`reveal:<epoch>:<sha256>[:<menu>]`), so it relies on the
bucket's contents for which reveal is the root's. Validators and TDs do read
the commitment (`reveal_sync`, `menu_publish.read_root_anchor`). Closing it
needs a chain read in the product (the commitment pinned at a recent block,
the reveal's sha256 compared before use), or a reveal list signed by the
root's hotkey that the product can verify offline. This decides no score:
no validator reads anything the product does.

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

## Protocol v5 (draft; applies only from a v5 activation block no release has set)

Protocol v5 scores a head on its real picks at RouterArena's trade-off between
accuracy and cost. An S-point is one accuracy point after the log-cost
adjustment of RouterArena's own score (`fugal_subnet/arena_score.py`); every
comparison below is a paired mean of per-question items in S-points, bounded
by `seat_bound.lower_bound` (valid at every look, alpha 1%).

### I1 — v5: the exam spec is a pure function of published data

The exam for epoch E (its questions, each contender's pick, the cells to
label, the baseline B* and its questions, the ring of hotkeys and the cell
assignment) is computed from the boundary nonce, the heads verified by E−2,
the menu state and the pool alone (`exam_v5.build_spec`). The root publishes
it in its reveal of E−2; every validator recomputes it, and a validator whose
recomputation differs from the root's treats that epoch's exam as void
(nobody is scored on it). Check: determinism stages `exam_spec_v5`,
`arena_items_v5`, `seat_v5` (`scripts/check_determinism.py --perturb`).

### I3 — v5: the seat in one rule

`fugal_subnet/seat_v5.py`, re-run every epoch on the window's items:

- **Incumbent.** The holder, if it stands, is not an exact copy, and is not
  shown worse than B*.
  Otherwise there is none and the former holder joins the history.
- **Upgrade.** A newer head of the holder's own coldkey replaces it unless it
  is *shown* worse: the holder's bound over it passes −UPGRADE_MARGIN_V5
  (+1/4). With fewer than N_MIN_V5 paired questions nothing is shown, so a
  fresh retrain is not shown worse. It must also be payable itself (its bound
  over B* passes 0). It keeps the holder's place (`holder_since`).
- **Standing.** A holder that stops standing goes to the history and the
  seat refills on merit the same epoch. A returning former holder has no
  resume right: it walks as an ordinary contender in first-commit order.
- **Walk.** The other contenders, earliest first commit first (ties by head
  hash): C replaces the incumbent when its bound over the incumbent passes
  TAKE_MARGIN_V5 (or there is none) and its bound over B* passes 0. A later
  head must beat whoever holds the seat at its turn, so neither a farm of
  near-copies nor a lucky late copy of an earlier head passes without the
  margin. An empty seat is filled only this way: on merit, never by a draw.
- **Contenders.** Per coldkey the newest standing head, plus the holder; at
  most CONTENDERS_MAX_V5, the earliest commits kept. Exact copies
  (`copy_rule` ownership) never contend.
- **Pay.** 85/10/5 to the holder and the history (`payout.seat_weights`); a
  seat is paid only while its head is live this epoch and its bound over B*
  passes 0. Every unpaid share burns. While no head beats
  the best single model, the whole weight burns.

Check: `tests/test_seat_v5.py`, the `seat_v5` determinism stage, the attack
world `fugal_subnet/attacks/seat_v5_world.py`.

### I5 — v5: no head is judged on labels it could have read

A head is fixed by its commitment, so it can only have learned the labels of
questions published before it was committed. Two rules on the published set
(`memo_v4.Published`, the chain-driven view every node holds; from v5 a
reveal's exam and coverage votes are published outcomes too,
`memo_v4.published_ids`):

- **The exam is fresh.** Its questions are drawn from the pool's unpublished
  reserve only, leaving out every question of a spec already built, at
  ceil(R / FRESH_RESERVE_EPOCHS_V5) an epoch (at most EXAM_QUESTIONS_V5)
  (`v5_epoch.next_spec`, `reserve`, `exam_size`). When a question is drawn,
  none of its labels is public.
- **Per-comparison eligibility.** A question counts in the comparison of X
  with Y (or of X with B*) only when its labels were never published or were
  first published in an epoch at or after the later of the compared heads'
  commit epochs (`v5_epoch.eligible_after`): a reveal of epoch f is public
  from epoch f + 1, so a head committed in epoch c can have read it only if
  f ≤ c − 1. This covers what the draw cannot see, a question published by
  another path between the draw and the exam.

- **Coverage never touches the exam.** The v5 pool is split once, by a keyed
  hash of each question id (`arena_sets.in_coverage`, COVERAGE_PARTITION_V5
  of it to coverage), into an exam part and a coverage part. Exams draw only
  the exam part; coverage (the training labels) draws only the coverage part
  and reuses it freely (`td_v5.coverage_v5`). Coverage labels being public
  therefore never shrinks the exam's reserve.

So memorisation gains nothing by construction; v4's statistical per-head
refusal is not used. The reserve is the evidence's budget: it shrinks by about
1/FRESH_RESERVE_EPOCHS_V5 an epoch, and a release that adds questions raises
it. Check: `tests/test_v5_memo_resync.py`, the `v5_epoch` determinism stage,
`tests/test_graders_arena.py` (the partition).

### v5: routers see the question, never RouterArena's template

The exam pool's questions carry two texts (`benchmarks/arena_sets.py`): the
templated `prompt` RouterArena's harness builds (the answering model gets it,
and grading needs its answer format) and `route_text`, the row's own content
(context when non-empty, question, options as listed) without any template
sentence, label or format instruction. Routers are scored on picks computed
from the embeddings of `route_text` only (the pinned EMBEDDINGS_V5_SHA256
file), because RouterArena forbids routing on its harness's instruction text
and a real user sends none. `route_text` is in the pool's content hash
(pool_manifest_v5.json). Check: `tests/test_graders_arena.py`.

### I4 — v5: an exam cell is a vote of operators, never of the compared ones

Each exam cell (question, menu entry) is labelled by EXAM_VOTES_V5 hotkeys of
distinct coldkeys, assigned by a keyed walk of the ring. Votes are pooled as
exploration pairs are (failed calls dropped, the plausibility check, the
median). In the comparison of X with Y, no vote from X's or Y's coldkey
counts; a question left without a usable vote gives that comparison no item.
A cell whose usable votes all failed, from at least two coldkeys
(`exam_v5.FAILED_CELL_MIN_VOTERS`), counts as a wrong answer at zero cost, as
the product would fail it. A cell on an entry that left the menu between the
spec (E−2) and its exam (E) is not called and not expected
(`tee/verify.expected_v5`, the harness): it has no vote at all, so it gives
that comparison no item and never counts as a failure.
So no operator's re-runs, withheld calls or own grading can move its own
comparisons. Check: `tests/test_exam_v5.py`.

### I8 — v5: a TD labels only the cells assigned to it

A v5 TD answers the cells the exam spec assigns to its hotkey (checked
against the root's committed spec hash) and its coverage share, nothing
else; a vote for a cell not assigned to the voting hotkey is refused by
verification. Check: `tests/test_td_v5.py`.

### v5 state: rebuilt from the root's reveals

A validator's v5 state (seat, head book, window records, the specs ahead) is
a pure function of the root's reveals (`v5_epoch.rebuild`). A node with no
state or a stale one rebuilds it before it scores (`v5_node.V5Node.resync`,
called from the epoch loop): a fresh follower holds the root's state. The
bytes of the heads in the book are kept beside the state, each checked
against its sha256 when read back, so a restart still gives the holder and
the contenders their picks. Check: `tests/test_v5_memo_resync.py`.
