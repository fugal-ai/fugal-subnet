# I3 decision memo: what should the score reward?

One page, for a decision. The measurements are in `docs/HEAD_EFFICACY.md` and
the analysis in INVARIANTS § "I3 — OPEN". This says only what the choice is,
what each option costs, and which one follows from the product's own claim.

## The fact

On 5,000 held-out questions, with the shipped constants:

| policy | score |
|---|---|
| always one mid-priced model (`W = 0`, one-hot bias; never reads the question) | **1.183** |
| best trained router (reads the question, routes across 13 models) | 1.060 |
| best single model, which is the reference | 0.985 |
| always the cheapest model | 0.479 |

The router works — it beats chance, the reference and a per-benchmark lookup,
and on MATH it routes to a model 12x cheaper *and* better. But a head that never
reads the question beats it by 12%, and any miner finds that head on day one.
The break-even is a property of the constants alone: a model 10x cheaper than
the reference wins with 77% of its accuracy. Eight of the seventeen pinned
models are 10x cheaper than the top.

## The atomic truth

The subnet exists to reward **the value of reading the question**. A router's
worth is the accuracy it buys *beyond what any single model buys at the same
price*. Everything else — a cheap model being good value, a dear model being
accurate — is a fact about the model catalogue, not about routing, and the
subnet should not pay miners to discover it, because it is discoverable for
free by looking at the catalogue.

Scoring against the best *single* model measures the wrong thing: it hands a
cheap, decent model a large thrift ratio for a decision nobody made. The three
resolutions in INVARIANTS are, restated against that truth:

| Option | What it is | What it measures | Cost |
|---|---|---|---|
| **1. Re-tune** the exponent or the thrift cap | move `w` toward 0.96 | the same wrong thing, more gently; leaves the subnet nearly indifferent to cost | a constant, a test, an INVARIANTS entry — but the analysis says it is not right |
| **2. Accept** | declare the product "best value model or router, whichever wins" | value at the catalogue level; routing only when it beats a constant | none in code; a product statement; miners will submit constants and the reveal data flywheel stops carrying routing signal |
| **3. Cost-matched frontier** | reference = the best *constant policy at the router's own cost*, interpolated on the single-model quality/cost frontier built from the exploration frame; score = headroom above it | exactly the value of reading the question: every constant policy scores 0 headroom by construction | a consensus redesign of `scoring.py` and `reference_frame.py`, new tests and attack cases, an INVARIANTS entry, a burn-in for the frontier; re-scores every miner |

## Recommendation

**Option 3**, because it is the only one whose definition *is* the product
claim. Options 1 and 2 keep a reference that a catalogue lookup beats.

What it needs, honestly:

- The frontier is built from the exploration frame, which already pools
  per-model accuracy over time and is nonce-driven so no miner steers it. The
  new part is the cost axis (the policy-cost pricing already exists) and the
  interpolation rule, both of which must be pinned constants.
- A router with zero headroom earns zero. That is correct and also harsh for
  the first weeks; a small participation floor during burn-in is a separate,
  explicit choice, not a side effect.
- It must land before mainnet, not after: it re-scores every miner, and a
  subnet that changes its objective after launch teaches miners to farm the
  transition.
- The check that would have caught the original problem still applies:
  assert that a trained head outscores every constant policy on held-out data
  (13 evaluations). Under option 3 that assertion is true by construction for
  the constants and is the test that the frontier is computed correctly.

If the decision is option 2 instead, say so in INVARIANTS and in the README's
first paragraph, because the product then is "find the best-value model", and
the reveal artifact should stop describing itself as routing data.

## What this decision does not change

Attestation, the pool, the price table, the bindings, the channel, dedup,
evidence accumulation. All of tonight's rehearsal results stand under any of
the three options; only the number each proof turns into moves.
