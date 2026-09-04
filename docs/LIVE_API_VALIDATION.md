# Validating the Cost Path Against Real OpenRouter

The second thing a local run cannot establish. `scripts/dress_rehearsal.py`
exercises the real metering proxy over real HTTP, but it points that proxy at a
deterministic local stub, so it proves the *plumbing* — token accounting,
pricing arithmetic, cost reconciliation, rejection of inconsistent proofs —
without proving the *prices*.

## What is actually unverified

Costs are priced from `data/models.json`, which is hash-pinned and is the
consensus denominator: every validator prices every proof against it. That
makes scoring deterministic, which is the point. It also means the table can be
**internally consistent and externally wrong** — if a provider changes a price,
nothing local notices.

The proxy already records the provider's own reported cost alongside the table
price (`APICallRecord.provider_cost_usd`, `MeteringProxy.provider_total_cost`),
attested inside the enclave, precisely so the drift is detectable. Nobody has
looked at that number against reality yet.

So the open question is narrow: **does the pinned table match what OpenRouter
actually bills?**

## Running one live epoch

This spends real money. Nothing in the default path can — there is no API key
in any test or script, and `--mock` is the default everywhere.

```bash
export OPENROUTER_API_KEY=<key>          # the only place this is ever needed
export FUGAL_SLICE_SIZE=40               # keep the bill small
export FUGAL_BENCHMARK_POOL=$PWD/results/rehearsal/pool.json

# Miner: real models, real money. Do NOT set FUGAL_OPENROUTER_BASE — the
# default is the real endpoint.
python neurons/miner.py --netuid <N> --head-path <head>.npz --mock
```

At 40 questions across the pool's price range, expect **well under $1**. Watch
the miner's per-epoch cost line before letting it run a second epoch.

## What to check afterwards

Pull the epoch's proof and compare the two figures it carries:

```python
from fugal_subnet.tee.proof import BenchmarkProof
import json
proof = BenchmarkProof.from_dict(json.load(open("<proof>.json")))
print("priced from the pinned table:", proof.total_cost_usd)
# provider_cost_usd is recorded per call by the proxy
```

Three outcomes:

- **Within a few percent** — the table is good. Record the comparison and move
  on.
- **Systematically off for one model** — that model's entry is stale. Update
  `data/models.json` and its pin in `scripts/check_safety_invariants.py` in the
  same commit, and say why.
- **Off for everything** — the pricing arithmetic is wrong, not the table. That
  is a code bug in `MeteringProxy.price_call`, and it changes every score.

## Also worth confirming in the same run

- **Token counts are real.** The stub returns synthetic `usage` figures; a live
  run is the first time `prompt_tokens`/`completion_tokens` come from a
  provider. They feed the reference model's counterfactual cost, so they matter.
- **Replies grade sensibly.** The stub answers correctly or returns `"0"`. Real
  model output is the first genuine test of the graders against this pool.
- **Nothing logs the key.** Grep the epoch's output for it. This should be
  impossible — it is a standing constraint — but a live run is the only time it
  could actually happen.

## Cadence

Re-run this whenever `data/models.json` changes, and periodically regardless:
a pinned table is deterministic, not correct, and only a live comparison tells
you which one you have.
