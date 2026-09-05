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
| **I8** | **TEE integrity.** Every claim a proof makes is bound to something the miner cannot forge: the hardware's own measurement registers, or a hash chain rooted in the attestation. | `fugal_subnet/tee/verify.py`, `attestation.measurement_id`, `run_tee_attacks.py` (11 cases, in CI), `check_tee_safety` |
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

### I8 — verified on real hardware

The negative control has now been run on genuine TDX silicon rather than a
synthetic quote. A second confidential VM on the same machine type but a
different image (`ubuntu-2204-jammy-v20260826`, measurement `f83820ad0424…`)
produced a real Intel-signed quote that **passed** DCAP verification and was
still **rejected** by `verify_proof` with only the approved 24.04 measurement in
the set: *"Unapproved runtime image"*. Real hardware, real signature, code we
did not approve, refused. That is the property the whole attestation design
rests on, and it is now evidence rather than an argument.

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
created 15 minutes apart with the first deleted in between, produced the
**bit-identical** measurement `a68d0ccd3473a6c4…` — every register matching.

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
