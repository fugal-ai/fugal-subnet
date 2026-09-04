# Fugal Subnet — First Principles Design Decisions

Living document. Each step records the decision, the reasoning, and what to implement.

---

## Step 0: The Product

**Decision:** Routing is the product. A router that delivers equivalent quality at lower cost.

**Evidence:** User's current router matches GPT 5.5 quality at 1/6 cost. The value is in cost savings, not in beating the best model's accuracy. This means routing headroom is real and large — the subnet has a viable product.

**Implication:** The scoring formula must measure cost-adjusted quality, not raw accuracy. A router that saves money without losing quality is the goal.

---

## Step 1: What miners submit

**Decision:** Linear head (~130KB .npz) on frozen Qwen3-0.6B backbone. No change.

**Reasoning:**
- Simple, deterministic (matrix multiply, no branching)
- Inspectable — validators can run it themselves (just a matmul)
- ThirtySpokes independently converged on same design (~6K params)
- Head capacity is enormous relative to benchmark size — bottleneck is generalization, not capacity
- Making the head more complex doesn't help

---

## Step 2: Scoring formula

**Decision:** Replace hardcoded 55/35/10 composite with headroom scoring.

**Formula:**
```
headroom = (wilson_lcb(miner_accuracy) - zero_baseline) / (oracle_accuracy - zero_baseline)
score = headroom - λ * normalized_cost     (λ = 0.02, small)
```

- `wilson_lcb` = noise protection (8/8 ≠ certainty)
- `zero_baseline` = random routing accuracy at the miner's price point
- `oracle_accuracy` = perfect per-question routing accuracy
- `normalized_cost` = miner_cost / budget_ceiling
- `λ` tiny — cost only separates quality-tied miners

**Why better than current composite:**
- Measures routing DECISION, not model answer
- No hardcoded quality/cost tradeoff opinion
- Saturated pools correctly score as "nothing to route"
- On a 95%-accurate pool, old formula gives everyone ~95%. This gives 0-1 routing skill scale.

**Outcome (implemented, revised):** The headroom form was replaced by a
weighted geometric mean against the *best single model* rather than a
per-question oracle:

```
quality = wilson_lcb(accuracy) / acc_best
thrift  = ref_cost / miner_cost
score   = quality^0.8 * thrift^0.2
```

Three revisions to the sketch above, each forced by something concrete:

1. **Best single model, not per-question oracle.** The oracle needs a dense
   question-by-model matrix (~21K x 30 cells); the best model needs ~30
   marginals, converges within a few epochs, and is stable at any field size.
   It is also literally the product claim, so a score of 1.0 means "matched the
   frontier model's quality per dollar" — a number worth reporting.
2. **Product, not `headroom - lambda*cost`.** A subtraction still asserts an
   exchange rate, just a small one. A product asserts none, and stops either
   degenerate strategy from collecting its axis's weight regardless of the other.
3. **Exponent 0.8, derived not chosen.** "Match quality at a fraction of the
   cost" makes quality a near-constraint: giving up 40% of quality must not
   outscore matching the best model at its own price, which forces
   `w > ln6/(ln6-ln0.6) = 0.778`. An unweighted sqrt fails that test (1.095 vs
   1.000); 0.8 passes it (0.951).

---

## Step 3: Ground truth / who pays

**Decision:** TEE (miner-computed, hardware-attested). Global optimum.

**Why TEE over alternatives:**
- Solves cost asymmetry ($1 miner can't waste $30 of validator money)
- Solves I1 (matrix agreement) — validators verify proofs, don't compute own matrix
- Decentralized (no owner SPOF like owner-reference approach)
- Every other subnet has miners pay for expensive work
- Converts game theory problem into engineering problem — nothing to calibrate, nothing to game
- ThirtySpokes has battle-tested TDX code on mainnet (~40% directly reusable)

**Alternatives considered and rejected:**
- *Validator computes (current):* Cost asymmetry, I1 divergence unsolved
- *Owner pre-computes:* Centralized SPOF, still pays for attacker's models
- *Deposit + spot-check:* Fragile calibration, partial I1 divergence, same effort as TEE
- *Optimistic fraud proofs:* Challenge period delays, free-rider problem on challenges
- *ZK proofs:* Infeasible over network API calls
- *Multi-party:* Multiplies cost, collusion breaks it

**What to fork from ThirtySpokes (MIT licensed):**
- `tee/attestation.py` — Quote, Platform, AttestationReport data models (~80 lines)
- `koth/tdx.py` — Real Intel TDX quote generation + DCAP verification (~400 lines, hard to write)
- `tee/runtime.py` — MeteringProxy (trusted cost meter) + TEERuntime (~100 lines)
- `koth/confine.py` — No-egress network confinement via Linux namespaces
- `koth/store.py` — HuggingFace bundle store
- `koth/verify.py` — Proof verification logic
- `koth/commit.py` — On-chain commit-reveal for proofs

**What to build Fugal-specific:**
- The routing harness (head evaluation in TEE context)
- Scoring formula (headroom, not KOTH reign)
- Evidence accumulation (adapted to Fugal)
- Integration with existing validator/miner entry points

**Caveat:** TEE limits miners to Intel TDX VMs (GCP, Azure confidential VMs). Workable in practice (ThirtySpokes proved it on mainnet SN99) but excludes consumer hardware.

**Fallback:** Owner-published reference matrix if TEE implementation takes too long for launch.

---

## Step 4: Evidence accumulation

**Decision:** Artifact-keyed evidence accumulation with EWMA decay.

**How it works:**
- Each epoch's verified results for a FIXED artifact (same head weights) are pooled into a decayed binomial
- Score = Wilson LCB on the pooled counts
- EWMA decay (half-life ~200 epochs) — old evidence fades, so stale heads lose position
- Recommitting a new head RESETS the accumulator (its key changes: source_hash + weights_hash)
- Miss = 0 accounting: a miner that skips an epoch gets n_expected tasks scored as 0 correct

**Why this over alternatives:**
- *Per-epoch scoring (current):* Too noisy. Lucky epochs dethrone. Validators disagree.
- *EWMA smoothing:* Doesn't distinguish "miner changed head" from "miner got lucky." Old bad scores haunt improved miners.
- *Artifact-keyed accumulation:* Noise cancels (300q × 50 epochs = 15K effective samples). Artifact reset makes dethroning cost real work. Miss=0 prevents selective publication.

**Measured improvement (ThirtySpokes):**
- Validator divergence ↓5.5×
- Crown churn ↓20×
- Mis-crown rate: 71% → 6%
- Time-to-crown: monotone in true edge

**Tradeoff:** Updating your head is costly — you lose accumulated evidence. Miners should retrain in big steps, not continuous tweaks. This is a feature: it makes dethroning expensive and rankings stable.

**Correction found in implementation:** reset is *symmetric*. It clears
accumulated penalties exactly as readily as accumulated credit, so on its own it
was a free penalty wash — a miner with a poisoned record flips one weight bit
and is immediately back at full score. Measured: 20 bad epochs dropped a score
from 0.897 to 0.468, and recommitting restored it to 0.897 instantly.

Fixed with a burn-in ramp (`score *= min(1, n_total/BURN_IN_QUESTIONS)`) rather
than by weakening the reset. Reset stays cheap to *do* and expensive to *profit
from*: climbing back costs exactly what earning the position cost.

**Second correction:** the Wilson LCB assumed independent Bernoulli trials, but
at steady state `n_total` reaches ~86,550 over a ~21,000-question pool — about
4x reuse, not 86,550 independent draws. The effective n is now capped at the
distinct-question count.

**Implementation:** Adapt ThirtySpokes' `Evidence` dataclass. Their version handles multi-benchmark + frontier scoring. Ours is simpler (single routing benchmark).

---

## Step 5: Anti-gaming

**Attacks and defenses with TEE + accumulation:**

### Memorization
Head memorizes "question X → model Y" instead of learning routing patterns.
**Defense:** Held-out evaluation. Validator runs the head on questions the miner never saw (just a matmul, free). Memorizer scores at chance on held-out; honest router generalizes.

### Copying
Miner B downloads Miner A's head and submits it.
**Defense:** Behavioral dedup (existing) + earliest-commit seniority (existing). TEE binds source_hash into attestation — identical weight files are detectable.

### Sybil (multiple identities)
One entity registers 10 miners with similar heads.
**Defense:** Dedup catches similar heads. With TEE, each identity pays own VM costs — sybils are expensive.

### Benchmark gaming
Miner discovers which questions are in the pool and overfits.
**Defense:** Large pool (21K+ questions), unpredictable epoch slice (chain block hash seed), evidence accumulation (scored across many different slices over time), held-out evaluation.

### Pool manipulation (the old I4 attack)
**Defense:** With TEE, each miner runs their own benchmark. There IS no shared pool to manipulate. The attack is eliminated by architecture, not by a code fix.

---

## Step 6: Weight-setting and emissions

**Decision:** Proportional with minimum score threshold for launch. Consider top-K as miner base grows.

**Launch design:**
- Each miner's weight = accumulated_score / sum(all_scores)
- Miners below a quality floor earn nothing (prevents freeloading)
- Weight capping between epochs (existing MAX_WEIGHT_DELTA) provides stability
- Unassigned weight burns to UID 0

**Future consideration (post-launch, if miner base grows):**
- ThirtySpokes' 5-slot design (40/25/15/12/8 split)
- Stronger incentive gradient but fewer miners earn
- Anti-hoarding pension for ex-kings

**Key principle:** Emissions should pay for WORK, not for a seat. A miner that stops submitting valid proofs should stop earning. This is what ThirtySpokes' liveness enforcement does and what miss=0 accounting in evidence accumulation achieves.

---

## Step 7: Full epoch flow (TEE version)

```
1. EPOCH BOUNDARY
   - New block hash → deterministic nonce → question slice selected
   - Validator publishes nonce to miners

2. MINER SIDE (inside TEE)
   - TEE VM receives the nonce
   - TEE selects question slice (deterministic, same logic as validator)
   - TEE loads miner's routing head (np.load, allow_pickle=False)
   - TEE computes backbone embeddings for all questions
   - TEE runs head on embeddings → routing decisions
   - TEE calls each selected model via MeteringProxy (metered, unforgeable cost)
   - TEE grades responses against gold answers
   - TEE produces Proof: {results, costs, source_hash, weights_hash, measurement}
   - TEE hardware signs the proof (attestation quote)
   - Miner publishes proof + head to HuggingFace repo
   - Miner commits salted hash on-chain

3. VALIDATOR SIDE (verify-only, no inference)
   - Download each miner's published bundle
   - Verify hardware attestation quote (Intel DCAP chain)
   - Verify measurement matches approved runtime image
   - Verify source_hash + weights_hash match downloaded artifacts
   - Verify on-chain commit matches proof
   - Grade attested answers against public gold (deterministic, cheap)
   - Run head on held-out slice (matmul only, no API calls)
   - Compute headroom score from graded results
   - Feed score into evidence accumulator (artifact-keyed, EWMA decay)
   - Run behavioral dedup on routing decisions
   - Compute weights from accumulated scores
   - Set weights on-chain

4. BETWEEN EPOCHS
   - Miners retrain heads offline (using cached matrix data)
   - Recommitting new head resets evidence accumulator
   - Evidence for unchanged heads decays slowly (half-life ~200 epochs)
```

**What the validator NEVER does:** Call a model. Run miner code. Trust miner-reported data.

**What the TEE guarantees:** Results are real. Cost is real. Code that ran = code that was published.

---

## Implementation status

| Step | Status | Notes |
|---|---|---|
| Step 3: TEE infrastructure | **Implemented** | `fugal_subnet/tee/` — attestation, runtime, confine, proof, verify, harness. TDX patterns forked from ThirtySpokes/Chutes (MIT). |
| Step 3: TEE *bindings* | **Implemented** | Measurement from the quote's own registers, slice binding, head binding, bundle binding, cost consistency as rejection. `run_tee_attacks.py` keeps an exploit for each. |
| Step 4: Evidence accumulation | **Implemented** | EWMA-decayed binomial, artifact-keyed reset, miss=0, effective-n capped by pool size, burn-in ramp. |
| Step 2: Scoring formula | **Implemented** | `quality^0.8 * thrift^0.2` against the best single model. Exponent derived from the product claim. |
| Step 2: Cost model | **Implemented** | Pinned `data/models.json` as the consensus denominator; attested provider spend recorded alongside for drift detection. |
| — : Exploration + reference frame | **Implemented** | Nonce-derived quota recovers the counterfactual the TEE removes; frame pooled over time, not over miners. |
| Step 5: Anti-gaming (held-out) | **Deferred, deliberately** | See below. |
| Step 6: Weight-setting | Proportional | Fine for launch; revisit as the field grows. |

### The one thing deliberately not built

**Held-out evaluation.** The head is now bound and shipped in the proof bundle
specifically so a validator *could* run it on questions the miner was not scored
on. Running it was not built, and that is a decision rather than an omission.

Doing it requires the validator to compute backbone embeddings — which puts
per-validator floating-point computation back into consensus, exactly what the
TEE architecture removed. Two honest validators whose embeddings differ in the
last bits would disagree on near-tie routing and diverge. Half-building it would
trade a known gap for an unknown one.

The gap it leaves is real and recorded in `docs/INVARIANTS.md`: the question
pool is public and finite, so a miner willing to pay once to evaluate every
model on every question could publish a lookup table rather than a router.
Closing it properly needs either an agreed embedding artifact or held-out
questions outside the public pool.

## What measurement changed

Three values in this design came out different from the first guess, because
they were measured rather than reasoned about:

- **The frame prior strength** was going to be 200 (matching ThirtySpokes).
  Measured against field-size sensitivity it was far too strong: ceiling spread
  between a 3-miner and 50-miner field was still 0.100 at epoch 200. At K=20 it
  is 0.020. All values converge to the same truth; K only sets how fast.
- **The ceiling statistic** was going to be the accuracy lower bound. An LCB's
  width shrinks with sample count, so the ceiling moved with field size — 0.14 of
  score between a 3-miner and a 50-miner field. The LCB now selects the
  reference model; the posterior mean values it.
- **The scoring exponent** was going to be an unweighted sqrt. It scores a
  router that lost 40% of quality at 1.095, above a perfect quality match at
  1.000 — which contradicts the product claim it was supposed to encode.

## Open questions for further examination

- Evidence half-life is 200 epochs and the frame's is 500. Both are guesses that
  should be calibrated on testnet, the same way the frame prior was.
- The reference-frame prior accuracy is a neutral 0.5 for every model. Honest —
  nothing has been measured yet — but wrong for real models, and it biases the
  ceiling low until real evidence outweighs it. Recalibrate from testnet data.
- The thrift cap (10x) bounds how much cost saving can be rewarded. It is above
  the ~6x the product targets, but it is still a chosen number and should be
  revisited once real routers exist.
- Exploration is 5% of the slice. Enough to converge the frame in a reasonable
  number of epochs at moderate field sizes; unmeasured at very small ones.
