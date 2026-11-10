# Shared success contract and testnet transition

The new protocol is question → independent sigmoid success predictions → subtract
λ × estimated dollars/query → round to the 1e-4 grid → select one model. Ties
retain original artifact row order. The benchmark fixes λ=1, with no environment
override. Core retains user overrides. Rewards, exploration assignments, graders,
and the number of paid calls per benchmark are unchanged.

The portable core implementation is vendored in
`fugal_subnet/vendor/success_contract.py` under Apache-2.0. Its source commit and
SHA256 are recorded in `SOURCE.json`; CI verifies exact synchronization with core.
Because the repositories are private, CI uses a small test-only core projection
plus its commit/tree Git objects as a Merkle proof. It recomputes object IDs
through the pinned commit to each file, then runs the real serving interface
on the subnet-exported fixture. No cross-repository token or serving runtime
dependency is introduced. A local core checkout can also be compared directly.
The serving application is not a subnet dependency.

The embedding profile pins Qwen/Qwen3-0.6B and its tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca`, the exact core routing system prompt,
and `system: …\nuser: …` formatting. Standalone questions are right-truncated to
**2048 total input tokens**, right-padded in batches, mask-mean-pooled, and L2
normalized on CPU float32. Local model/tokenizer bytes are checked against the
pinned snapshot. The answering model receives the complete benchmark prompt.
Caches carry profile identity and ordered-question hashes; old caches are rejected.

## Training and artifacts

`train_head.py` uses masked BCE against observed 0/1 success labels. NaN is missing.
Observed all-failure rows remain in training. Exact duplicate prompts stay together
in deterministic 60/20/20 train/validation/test groups; validation BCE selects the
checkpoint. No random embedding fallback exists. `--synthetic` creates test-only
fixtures. The former preference trainer is retained only as explicitly named
historical experiment helpers because an existing experiment still calls CMA.
Its dependency remains for that caller; it is absent from the reference path.

Success artifacts require W `(M,1024)`, b `(M,)`, unique ordered Unicode model IDs,
contract/profile IDs, backbone revision, provenance, default λ, aligned measured
input/output token means, and the token manifest identity. All arrays load without
pickle. Archive and expanded limits are 1 MiB and 8 MiB; NPY header shape claims
are checked before allocation. Core legacy v1/v2 heads remain supported by core.
Legacy preference heads cannot enter this subnet benchmark.

`data/models.json` is unchanged and stays hash-pinned. Estimated dollars/query
are measured token means times those rates. A miner's statistics must exactly
match `data/benchmark_tokens_v1.json`, including its identity.

## Reviewed token manifest (2026-11-02)

`data/benchmark_tokens_v1.json` (mirrored byte-for-byte at
`fugal_subnet/routing_data/`, which CI checks) is `status: reviewed`. The review
record travels inside the file as its `review` block, so it is part of the
protocol identity; this section is the prose version.

**Source.** `data/observations/benchmark_tokens_v2.jsonl`: 664 metered calls made
on 2026-11-02 by `scripts/record_token_observations.py` through the same
`MeteringProxy` and `harness._call_model` the benchmark uses (`max_tokens` 2048,
the full benchmark prompt), on a deterministic benchmark-stratified sample of the
pinned pool (`pool_hash 1779b4a5…`), 40 questions per model. Every row carries the
provider's own prompt and completion token counts, the table-priced and
provider-billed cost, the question id, the response hash, the pool hash, the
source hash and the script commit. No response text.

**Billing cross-check.** Table-priced total $2.158 against provider-billed $2.159
over the whole sample, so `data/models.json` tracks OpenRouter's billing to the
cent in aggregate; the largest per-model gaps (deepseek-v4-pro billed above table,
glm-5.2 below) come from provider-side cache and discount accounting and are not
a reason to move a pinned rate before launch.

**What changed against the candidate of 2026-09-21** (945 calls from three
pre-transition proof bundles, three samples for most models): completion means
moved far enough to change routing — kimi-k2.6 3711→965, gpt-5.5 1334→323,
claude-haiku-4.5 1374→281, minimax-m3 1853→391, glm-5.2 343→927,
gemini-3.1-pro 621→877. Medians sit far below means for terse models
(gpt-5.4-nano 5, claude-sonnet-5 3, gpt-5.4-mini 11) and the 90th percentile hits
the 2048 cap for long-reasoning ones; the manifest stores the mean because the
cost estimate needs the mean.

**Limitations, accepted.** 40 samples per model is a cost interface, not a
workload study; `mistralai/mistral-large-2512` has 24, because OpenRouter's
shared upstream pool rate-limited it for the whole session (every other attempt
returned HTTP 429, retried for two hours). Questions are benchmark questions, not
a user workload. Changing any number here changes the protocol identity and
resets all evidence, so this manifest is expected to stand for the first launch
period; the next revision is a flag day run as the runbook below describes.

**Reproduce.** The command is in `data/observations/README.md`; the builder emits
`candidate`, and `tests/test_success_export.py` asserts the shipped file equals
that output plus exactly `status` and `review`. A candidate manifest still refuses
`--live` startup and deployable export (`tests/test_success_contract.py`).

## Protocol v2 token manifest: REVIEWED (2026-11-04)

`data/benchmark_tokens_protocol_v2.json` (mirrored at `fugal_subnet/routing_data/`) is
`status: reviewed`. The review record travels inside the file as its `review` block. Reviewer:
Agent 1 (Claude Opus 5.5, Fugal leader agent), 2026-11-04. Merging the promotion PR on Jack's
approval is the human review, as for v1.

**Source.** `data/observations/benchmark_tokens_protocol_v2.jsonl`: 757 metered calls (3 more
failed), 2026-11-04, by `scripts/record_token_observations.py --protocol 2` through the v2
harness path (`max_tokens` 16384, each model's single pinned `reasoning` setting from
`data/models_v2.json`, the full benchmark prompt). 40 questions per model at the v2 category
mix: knowledge from MMLU-Pro and MMLU; math from MATH; hard from AIME 2025 and historical AIME,
standing in for the open hard sets that were not yet loaded; instruction from IFEval. Same
fields as v1's observations; no response text.

**Billing cross-check.** Table-priced $5.675 against provider-billed $5.183 in total. Five
models diverge by more than 5%, from OpenRouter routing to endpoints priced differently from
the catalogue default the table pins:

| model | billed ÷ table |
|---|---|
| z-ai/glm-5.3-flash | **2.86** |
| z-ai/glm-5.3 | **0.49** |
| deepseek/deepseek-v4-pro-0813 | 0.80 |
| minimax/minimax-m3.1-flash | 1.15 |
| qwen/qwen3.8-27b | 1.09 |

**Review decisions (2026-11-04):**
1. **Cap overrun (kept; the self-test logs overruns):** x-ai/grok-4.7 returned 17,763 completion tokens on 1 of 38 calls, against
   `max_tokens` 16,384 (`reasoning.effort: medium`; xAI accepts only `effort`). Every other model
   stayed at or under 16,384, and several hit it exactly (cut off as designed).
2. **Hidden prompt (kept, stated in the manifest):** grok-4.7's prompt tokens are 1,267–1,575 on every call (others 25–750), so
   the provider adds ~1.2k tokens per call, and the manifest's mean includes them.
3. **Pinned rates: re-pinned.** The five diverging models' rates were scaled by their measured
   billed÷table ratio (input and output together) in `data/models_v2.json`; the hash pin and its
   comment record why.
4. Hard-category questions were AIME only, so the means of long-reasoning models on SuperGPQA and
   Omni-MATH are an extrapolation.

**Amendment (2026-11-09): anthropic/claude-fable-5.2 added, 19 models.** At the public
OpenRouter price ($8 / $40 per million tokens) with fable-5.1's reasoning setting, measured on
40 calls on the pinned v2 pool (232 prompt and 442 completion tokens on average, $0.0195 per
question). It has two endpoints, both Anthropic: an accepted exception to the three-endpoint
rule (MINER_GUIDE). fable-5.1 stays the anchor. *Update 2026-11-10:* OpenRouter's public
endpoint list now shows five endpoints across four providers (Anthropic, Google, Amazon
Bedrock, Azure), so the exception no longer applies. The price table is unchanged.

**Amendment (2026-11-05): grok-4.7 removed, 18 models.** Decision 1 is reversed. Beyond the
17,763 seen here, an external measurement found 48,132 completion tokens (47,656 reasoning)
against `max_tokens` 16,384, about five times the call's budgeted cost: the provider does not
bound reasoning by `max_tokens`. The proxy does not stream, so a deadline cannot stop that
billing (OpenRouter bills a non-streamed call in full). The model and its manifest row are removed.
The other 18 rows are unchanged, and the call log keeps grok's 38 rows. grok-4.7 can return
in v2.1, once the harness streams and aborts a call at its deadline. Re-verification of the 18
means is recorded in the manifest's `review.amendments`.

## Offline export

The completed 2048-token SPROUT evaluation and explicitly deferred 512-token
comparison are recorded in
[the offline mechanism report](decisions/evaluation/success-contract/REPORT.md), with
machine-readable results, evaluated snapshots, and a nondeployable 2048-token
bundle. Its historical model roster differs from the live subnet roster.

```bash
python scripts/export_success_head.py \
  --head /path/to/head.npz --manifest /path/to/tokens.json \
  --prices /path/to/prices.json --report /path/to/report.json \
  --output /path/to/new-bundle
python scripts/export_success_head.py --verify /path/to/new-bundle
```

The bundle includes the head, evaluated token/price snapshots, provenance, hashes,
and evaluation report. `--deployable` additionally requires reviewed recorded
observations. Offline export does not choose a winner, publish a release, replace
core's shipped head, or promise calibrated probabilities.

Load via core's existing interface:

```bash
FUGAL_HEAD=/path/to/new-bundle/head.npz \
FUGAL_PRICES=/path/to/new-bundle/prices.json \
FUGAL_MODEL=/path/to/pinned/Qwen3-0.6B \
python -m fugal --route "what is 15% of 240?"
```

Use the bundled price snapshot for reproducibility. Core users can explicitly
choose refreshed prices, which may change selections. Sigmoid and compatibility
alone do not establish calibrated probabilities across workloads or λ choices.

## Fresh testnet evidence runbook — execute only after review

No deployment, measurement approval, chain reset, or release is performed by these
PRs. The transition uses one success protocol, not simultaneous policy support.

1. Review the token observations and prices. Resolve sampling concerns, promote the
   manifest to reviewed, and pin the final core/subnet commits and bundle hashes.
   Retrain under the 2048 profile and verify the exported head in core. Keep winner
   selection manual. Run all CI and the mocked rehearsal before proceeding.
2. Stop the old miners and validators at an agreed epoch boundary. Archive their
   exact runtime images, approved measurements, configuration, validator state,
   evidence, reference frame, reveals, and embedding caches. Record checksums.
   **Preserve wallets, keys, and chain registrations.** Do not reset the chain.
3. Rebuild the measured runtime from the reviewed commit. Its identity binds source,
   pool, graders, upstream, contract, embedding profile, benchmark λ, token manifest,
   and price snapshot. Generate actual TDX/dstack measurements on the target runtime;
   update approved measurements through the existing operator review process.
   Never invent a measurement or reuse approval for a different runtime.
4. Start with fresh `results/success-v1/validator_state.json` (move its `.bak`
   and any `.tmp` aside too, or the validator resumes from them) and
   `results/success-v1/epochs`, empty validator evidence and reference frame, and
   profile-tagged embedding caches. Update any service-level path overrides too.
   Old state, frame snapshots, reveals and proofs are rejected by protocol identity;
   do not strip their identity or import old accumulated observations as evidence.
5. Retrain and recommit miner heads before the new nonce is known. Confirm each head
   has exact manifest statistics and supported metadata. Start miners and validators
   on the same new runtime and approved identities. Check first proofs for protocol,
   head commitment, nonce, questions, exploration, identity and accounting. Confirm
   one selected worker per scored question and the unchanged exploration count.
6. Monitor fresh burn-in and reference-frame accumulation under existing frontier
   rewards. Do not compare raw old/new state totals as if they shared a namespace.

Rollback stops the new runtime and restores **the old runtime and its matching
archived state/configuration together**, including matching approved measurements
and archived legacy head files. Recommit
those legacy heads before the next usable nonce boundary so on-chain commitments
match the restored runtime. Keep new evidence separately archived. Preserve wallets
and registrations in both directions. Never feed success-protocol evidence to the
old runtime or vice versa.

To update the vendor after a reviewed core change:

```bash
python scripts/vendor_success_contract.py --core /path/to/core --commit <full-core-commit>
python scripts/check_success_vendor.py --core /path/to/core
```

The runtime package contains only the reference module and license/provenance
resources. `tests/core_snapshot` is a test dependency only. CI checks its actual
Git blob/tree/commit identities; changing a SHA256 manifest alongside modified code
cannot make an unrelated source file pass that proof.
