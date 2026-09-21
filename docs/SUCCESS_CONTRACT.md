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

## Candidate token manifest

The candidate comes from 945 recorded calls in three saved historical testnet
proof bundles. One zero-input record is excluded. Normalized observations are
committed in `data/observations/benchmark_tokens_v1.jsonl`, including original
bundle hashes and source runtime hashes; no response text or API secrets are
included. Some models have only one to three samples. Most observations for cheap
models are selected routes, so sampling is policy-dependent. This is an initial
cost interface fixture, not a representative estimate of future workload cost.

Reproduce the candidate with the command recorded in
`data/observations/README.md`. Missing observations for any requested model block
manifest generation. No fabricated token counts enter a deployable export.
The manifest's `candidate` status deliberately blocks **live startup and deployable
export** until its observations and sampling limitations receive review. Promotion
to `reviewed` changes protocol identity and requires rebuilding the measured runtime.
Mocked benchmarks can exercise the candidate. Synthetic fixtures are test-only.

## Offline export

The completed 2048-token SPROUT evaluation and explicitly deferred 512-token
comparison are recorded in
[the offline mechanism report](evaluation/success-contract/REPORT.md), with
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
4. Start with fresh `results/success-v1/validator_state.json` and
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
