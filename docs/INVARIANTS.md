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
| **I1** | **Determinism.** Same epoch inputs ⟹ byte-identical scores on any honest validator. | `scripts/check_determinism.py` (both modes, in CI — 9 stages covering the live path), `fugal_subnet/determinism.py`, `check_safety_invariants.check_epoch_id_single_source`, TEE proof verification (all validators verify the same attested proof) |
| **I2** | **Bounded ingestion.** No miner-supplied bytes reach deserialization, allocation, or execution without size, shape, and value bounds. | `run_miner_attacks.py`, `tests/test_head_properties.py`, `check_safety_invariants.py` (no-pickle) |
| **I3** | **Monotonic incentive.** A miner cannot raise its score except by routing better or more cheaply. Artifact-keyed evidence with miss=0 prevents selective publication; the burn-in ramp prevents penalty-washing by reset. | Commit-reveal, behavioural dedup (global model index), evidence accumulation, `run_attacks.py`, `run_tee_attacks.py` |
| **I4** | **Non-interference.** A miner cannot lower another miner's score, prevent them being scored, or move the reference they are scored against — including by being present or absent. | `tests/test_non_interference.py`, TEE architecture (no shared model pool), nonce-derived exploration targets, reference frame pooled over time |
| **I5** | **Bounded spend.** No miner behavior can make a validator exceed its budget. Validators verify proofs — zero inference cost. | TEE architecture (miners pay their own inference), `tests/test_paid_safety.py` |
| **I6** | **Liveness.** No miner behavior can stop a validator completing an epoch and setting weights, and the validator collects at a point where proofs can exist. | `run_miner_attacks.py`, property test P1, TEE proof timeout, `slicer.collect_block_for_epoch`, `tests/test_collection_point.py` |
| **I7** | **Auditability.** Any divergence between two validators is diagnosable after the fact from published artifacts. | `fugal_subnet/fingerprint.py`, `environment` block in every `reveal.json` |
| **I8** | **TEE integrity.** Every claim a proof makes is bound to something the miner cannot forge: the hardware's own measurement registers, or a hash chain rooted in the attestation. No miner-influenced code executes inside the enclave that produces the proof. | `fugal_subnet/tee/verify.py`, `attestation.measurement_id`, `run_tee_attacks.py` (11 cases, in CI), `check_tee_safety`, `config.HARNESS_ALLOW_EXEC`, `tests/test_grader_policy.py` |
| **I9** | **Reference-frame agreement.** Every validator derives the same reference frame from the same published exploration samples, and no single miner can materially move it. | `fugal_subnet/reference_frame.py` (order-independent accumulation), `check_determinism.py` `frame` stage, `tests/test_non_interference.py` |

## How TEE resolves prior gaps

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

### I8 — the approved entry is a PAIR, not a hash

An approved-list entry is `<base_measurement>` or `<base_measurement>:<app_identity>`.

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
`::test_extend_rtmr3_reports_failure_rather_than_pretending`.

**Until then `--live` binds less than it appears to**, and no rotation procedure
changes that: rotating an approved list of measurements that do not cover the
workload rotates a value proving only which OS booted.

Operational consequence for validators: DCAP collateral is fetched from Intel's
PCS directly, with no local caching service. A `--live` validator therefore
needs outbound HTTPS to `api.trustedservices.intel.com`, and an Intel PCS
outage degrades verification for every validator at once.

### I8 — what "attested" actually means

DCAP verification proves a quote is genuine and Intel-signed. It proves the
*hardware* is real. It says nothing about whether the code inside it is the
published code — an attacker who owns a TDX machine produces a perfectly valid
quote while running a modified harness.

For a period this subnet checked `proof.source_hash`, a field the workload
writes about itself, against the approved-image list. That is not an attestation
of anything: the attacker simply writes an approved value. The check now reads
`measurement_id(quote)`, derived from MRTD and RTMR0-2, which the CPU fills in
and the Intel signature covers. RTMR3 is excluded because it is
application-extendable, so including it would make an image identity change
with runtime data and no image could stay on an approved list.

The full chain a proof must satisfy:

```
Intel DCAP signature          -> the quote is genuine, from real TDX hardware
measurement_id(quote)         -> the image that ran is the published one
report_data == content_hash   -> the proof body is what that image produced
weights_hash == commitment    -> the head that ran was committed before the nonce
sha256(bundled head)          -> the head shipped is the head attested
result ids == assigned slice  -> those answers are to the questions we asked
exploration == nonce targets  -> the sampling quota was actually performed
per-question costs == total   -> the cost figures are internally consistent
```

Each line was, at some point, absent — and `run_tee_attacks.py` keeps an
executable exploit for each, because every one of them fails *silently*: the
proof verifies, the miner is paid, and nothing in a log says otherwise.

### I4 — the reference frame must not move with the field

Scoring against "the best single model" needs an estimate of how good that
model is, and that estimate is built from miners' exploration samples. If it
tracked the miner population, every miner would move every other miner's score
and the same head would be worth more in a thin epoch than a busy one.

Two things keep it from doing so. The frame accumulates over *time*, not over
miners, so one epoch's samples are a single decayed contribution to a
long-running estimate. And the ceiling is valued at the posterior *mean* rather
than a lower confidence bound — an LCB's pessimism shrinks as evidence
accumulates, which made the ceiling a function of sample count and therefore of
field size (measured: 0.14 of score between a 3-miner and a 50-miner field).
The LCB still *selects* which model is the reference, so a lucky model cannot
be crowned; it just does not *value* it.

Residual: convergence speed still depends on field size. A small subnet reaches
a stable frame more slowly. The prior strength was calibrated against this
directly (see `FRAME_PRIOR_STRENGTH`) rather than chosen.

### I4 — seniority squatting (residual, accepted)

Dedup seniority tracks the earliest block at which a hotkey was ever seen
committing a valid head. A squatter who registered and committed *before* its
victim therefore still holds earlier seniority. This is far weaker than the
copy-and-outrank bug it replaced — it requires predicting the victim well in
advance — but it is not zero.

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

**Pool memorization (open).** The question pool is public and finite (~21K).
A miner willing to pay once to evaluate every model on every question could
publish a head that encodes the resulting lookup rather than a routing policy.
It would score well here and generalize to nothing.

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

**Benchmark pool reproducibility (open).** The pool is consensus state, and
the default loader does not produce the same one for everybody. GPQA is a gated
Hugging Face dataset, so `load_all()` raises for an operator without a token and
returns a *larger* pool for one with it. LiveCodeBench loads zero questions on
current `datasets` versions but loads normally for anyone who has manually
placed `data/benchmarks/livecode.json`. Either divergence produces proofs that
fail on `questions_hash`, an error that names the symptom and never the cause.
Until the pool is published as a hash-pinned artifact and every neuron is
pointed at it with `FUGAL_BENCHMARK_POOL`, the loader is a consensus hazard.

**The approved measurement is reproducible but not durable (open, blocks
`--live`).** `measurement_id` is sha256 over MRTD and RTMR0-2. Measured on real
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
[CODE_BENCHMARK_PLAN.md](CODE_BENCHMARK_PLAN.md), and its final phase needs the
measurement rotation procedure that does not exist yet.

**The model upstream is miner-controlled (OPEN, EXPLOITABLE TODAY, breaks I3).**
`fugal_subnet/tee/runtime.py` reads the endpoint the metering proxy calls from
`FUGAL_OPENROUTER_BASE`, an environment variable, inside the miner's own TD. The
pool the TD loads carries `gold` for every question. Nothing in `proof.py` or
`verify.py` binds the upstream — grep them for it and there is nothing.

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

Nothing in the repository prevents this today. The only thing resembling a
control is a line in `docs/LIVE_API_VALIDATION.md` asking the operator not to
set the variable — a request addressed to the adversary.

**Why it went unnoticed for so long is the instructive part.**
`scripts/stub_upstream.py` *is* this attack. It was written as a no-spend
testing tool, it works exactly as intended, and nobody asked what it meant that
it worked. A mechanism that lets a local test substitute the model layer is the
same mechanism that lets a miner substitute it in production. Building the
exploit as a convenience and never naming it as one is how a hole this size
stays invisible.

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

## Running the checks

```bash
python scripts/check_safety_invariants.py          # structural invariants + TEE safety
python scripts/check_determinism.py                # I1, same-host
python scripts/check_determinism.py --perturb      # I1, simulated second host
python -m fugal_subnet.attacks.run_attacks         # I3, hostile model output
python -m fugal_subnet.attacks.run_miner_attacks   # I2/I6, hostile miner input
python -m fugal_subnet.attacks.run_tee_attacks     # I8/I3, forged proofs
pytest -q                                          # I4, I9, evidence, TEE, end-to-end
```

All of these run in CI on every push. None requires a chain, a network, or API
spend.

## Before mainnet

See [MAINNET_LAUNCH.md](MAINNET_LAUNCH.md) for the operational sequence. Item 1
below was done on netuid 552 and turned up a defect no local run could: the
subnet had never been activated. `start_call` was never made, so it had no
first-emission block — staking was rejected, no validator earned a permit, and
every `set_weights` failed while the neurons reported success.

1. Deploy on testnet with two validators and verify they produce identical
   weights from the same set of TEE proofs, and identical reference frames from
   the same published exploration samples (I9).
2. Run miners on real Intel TDX VMs (GCP `c3-standard` or Azure confidential
   VMs) and verify DCAP attestation end-to-end.
3. Publish approved runtime measurements (`FUGAL_TEE_MEASUREMENTS`) and
   document the process for updating them. These are `measurement_id()` values
   — sha256 over the quote's MRTD and RTMR0-2 — not source hashes.
4. Recalibrate the reference-frame prior from real testnet data. It currently
   sits at a deliberately neutral 0.5 for every model, which is honest (the
   subnet has measured nothing yet) but is wrong for real models and biases the
   ceiling low until real evidence outweighs it.
5. Close or consciously accept the pool-memorization gap above.
6. Validate real TDX attestation on a confidential VM —
   `docs/TDX_VALIDATION.md`. DCAP signature verification and `measurement_id`
   matching are the only checks no local run can make; a consumer CPU cannot
   produce a genuine quote. Mock mode is their absence, not a weaker form.
7. Validate the cost path against real OpenRouter —
   `docs/LIVE_API_VALIDATION.md`. The pinned price table is deterministic, not
   necessarily correct; only a live comparison distinguishes the two.
