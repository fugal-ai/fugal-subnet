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
| **I1** | **Determinism.** Same epoch inputs ⟹ byte-identical scores on any honest validator. | `scripts/check_determinism.py` (both modes, in CI), `fugal_subnet/determinism.py`, `test_validator_embeds_on_cpu_for_consensus` |
| **I2** | **Bounded ingestion.** No miner-supplied bytes reach deserialization, allocation, or execution without size, shape, and value bounds. | `run_miner_attacks.py`, `tests/test_head_properties.py`, `check_safety_invariants.py` (no-pickle) |
| **I3** | **Monotonic incentive.** A miner cannot raise its score except by routing better. | Commit-reveal, behavioral dedup, cost cap, `run_attacks.py` |
| **I4** | **Non-interference.** A miner cannot lower another miner's score or prevent them from being scored. | `tests/test_non_interference.py`, routed-model pool (no fixed cap), coverage multiplier |
| **I5** | **Bounded spend.** No miner behavior can make a validator exceed its budget. | `SpendTracker` reserve/reconcile/forfeit, `tests/test_paid_safety.py` |
| **I6** | **Liveness.** No miner behavior can stop a validator completing an epoch and setting weights. | `run_miner_attacks.py`, property test P1 |
| **I7** | **Auditability.** Any divergence between two validators is diagnosable after the fact from published artifacts. | `fugal_subnet/fingerprint.py`, `environment` block in every `reveal.json` |

## Known gaps

### I4 — pool eviction (resolved)

**Previous vulnerability:** Two sybil registrations declaring the same 30 cheap
models could evict 100% of a victim's declared models from the union pool via
the fixed 30-model cap with declare-count priority, zeroing the victim's
accuracy. The attack cost only two registrations.

**Fix:** The model pool is now built from models that heads actually *route to*
(the union of every head's weight-matrix model list), not from a separate
declared pool. There is no fixed cap — the validator's epoch budget is the
natural limiter. When the budget cannot cover all routed models, models used by
fewer heads are dropped first (least scoring signal lost).

Each head is scored only on the models present in the matrix. A coverage
multiplier (`intersection_size / pool_size`) scales the composite score so a
head covering fewer models cannot outperform a head with broader coverage on
raw accuracy alone. This prevents the narrow-surface gaming strategy (declare
two easy models, ace them, ignore the rest).

An attacker's sybil heads that route to junk models simply add those models to
the matrix (a small cost to the validator) without affecting honest miners'
scores. The attacker's own heads score poorly (junk models answer incorrectly)
and earn nothing. The griefing vector is eliminated.

### I4 — seniority squatting (residual, accepted)

Dedup seniority tracks the earliest block at which a hotkey was ever seen
committing a valid head. A squatter who registered and committed *before* its
victim therefore still holds earlier seniority. This is far weaker than the
copy-and-outrank bug it replaced — it requires predicting the victim well in
advance — but it is not zero.

### I1 — the ground truth matrix is not reproducible across validators

Determinism work covers everything downstream of the matrix. The matrix itself
comes from OpenRouter, and LLM APIs are not deterministic even at temperature
0: two validators calling the same model on the same question can get different
responses, and byte-identical grading of *different responses* still yields
different matrices.

This is inherent, not a bug, and it is why `consensus.py` compares validators
by median rather than expecting exact agreement. **How much divergence this
actually causes is unmeasured.** Measure it on testnet with two validators
before trusting the incentive signal on mainnet.

## Attack surface by actor

**Miners** control: the bytes their axon returns, the model IDs they declare,
how long they take to respond, when they commit, and how many identities they
register. That is the whole of I2, I4, and I6, and most of I3.

**Other validators** control: their own published reveals and weights. Yuma
consensus plus `MAX_WEIGHT_DELTA` bound the damage; `consensus.py` detects it.

**Model providers** are not adversarial but are unreliable and
nondeterministic — the I1 gap above.

**Benchmark datasets** are pinned by revision, so upstream changes cannot
silently alter the question pool.

## Running the checks

```bash
python scripts/check_safety_invariants.py          # structural invariants
python scripts/check_determinism.py                # I1, same-host
python scripts/check_determinism.py --perturb      # I1, simulated second host
python -m fugal_subnet.attacks.run_attacks         # I3, hostile model output
python -m fugal_subnet.attacks.run_miner_attacks   # I2/I6, hostile miner input
pytest -q                                          # includes I4 and property tests
```

All of these run in CI on every push. None requires a chain, a network, or API
spend.

## Before mainnet

The gaps above are the ones worth closing first, in this order:

1. Measure matrix divergence between two validators on testnet. Until that
   number exists, the practical strength of I1 in production is unknown.
2. Run two validators on **different CPU generations** and diff their published
   reveals every epoch. `--perturb` approximates this on one machine; only real
   hardware diversity tests it properly.
