# Validating the Cost Path Against Real OpenRouter

The second thing a local run cannot establish. `scripts/dress_rehearsal.py`
exercises the real metering proxy over real HTTP, but it points that proxy at a
deterministic local stub, so it proves the *plumbing* — token accounting,
pricing arithmetic, cost reconciliation, rejection of inconsistent proofs —
without proving the *prices*.

## What is actually unverified

Costs are priced from the pinned price table of the epoch's protocol
(`data/models.json` for v1, `data/models_v2.json` for v2) times the reviewed
token means (`data/benchmark_tokens_*.json`); both are hash-pinned and are
the consensus denominator: every validator prices every proof against them. That
makes scoring deterministic, which is the point. It also means the table can be
**internally consistent and externally wrong** — if a provider changes a price,
nothing local notices.

The proxy already records the provider's own reported cost alongside the table
price (`APICallRecord.provider_cost_usd`, `MeteringProxy.provider_total_cost`),
attested inside the enclave, precisely so the drift is detectable. It has been
looked at twice:

- **Account-level, 2026-09-07**, over four miner-epochs on netuid 552: within
  ~5%, with completions averaging 534 tokens against the 256 then assumed.
- **Per model, 2026-11-04**, the protocol v2 token run: 757 direct metered
  calls through the harness path, 18 models x 40 questions, provider-billed
  cost kept beside table-priced cost per row
  (`data/observations/benchmark_tokens_protocol_v2.jsonl`). Billed $5.18
  against $5.68 table-priced; five models' prices were re-pinned from the
  measured effective price (the `billing_check` record inside
  `data/benchmark_tokens_protocol_v2.json`).

Mainnet proofs since 2026-11-04 carry `provider_cost_usd` per call, so a
per-proof comparison is now possible from the validator's own epoch artifacts.
The question stays narrow: **do the pinned table and token means match what
OpenRouter actually bills, on the pinned pool?** The v2 run sampled from the
open sets at the category mix, not the pinned pool itself; the re-measure on
the pinned v2 pool before the flip is in `MAINNET_LAUNCH.md`, "The v2
release".

## Per model, the cheap way

`scripts/record_token_observations.py` is what the v2 token run used: the same
MeteringProxy and `_call_model` path as the benchmark, a deterministic
stratified sample, provider-billed and table-priced cost per row, no response
text, the key from a 0600 file, and a hard `--max-usd` stop. It defaults to
`--mock`, which refuses the real upstream.

```bash
uv run python scripts/record_token_observations.py --live --protocol 2 \
  --models data/models_v2.json --api-key-file ~/.fugal/openrouter.key \
  --per-model 40 --max-usd 12          # [PAID ~$5-12]
uv run python scripts/build_token_manifest.py --observations <out.jsonl> \
  --models data/models_v2.json --output <candidate.json> \
  --worker-profile "<how, when and on what sample it was measured>"
```

`record_token_observations.py` prints the provider-billed and table-priced
totals; the reviewer writes them into the manifest's `review` record as
`billing_check`, which is the comparison. Promoting the candidate to `reviewed`
is a consensus change (pin, `routing_data/` copy, CHANGELOG, review record).

## Running one live epoch

The whole-path alternative. This spends real money. Nothing in the default path can — there is no API key
in any test or script, and `--mock` is the default for the miner and every script that can call a model (the validator, which calls none, defaults to `--live`).

```bash
export OPENROUTER_API_KEY=<key>          # the only place this is ever needed
export FUGAL_SLICE_SIZE=40               # keep the bill small (mock warns; live would refuse)
export FUGAL_BENCHMARK_POOL=$PWD/results/rehearsal/pool.json
export FUGAL_POOL_UNPINNED=1             # a rehearsal pool is not the pinned one, say so

# Miner: real models, real money. Do NOT set FUGAL_OPENROUTER_BASE — the
# default is the real endpoint. --mock is refused on finney; use a test chain.
python neurons/miner.py --network test --netuid <N> --head-path <head>.npz --mock
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
  the price table of that protocol and its pin in
  `scripts/check_safety_invariants.py` in the same commit, and say why.
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

Re-run this whenever a price table or token manifest changes, before the
protocol v2 flip on the pinned v2 pool, and periodically regardless:
a pinned table is deterministic, not correct, and only a live comparison tells
you which one you have.
