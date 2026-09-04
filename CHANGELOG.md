# Changelog

All notable changes to this project will be documented here. Releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- TEE-attested benchmarks: miners run inside Intel TDX confidential VMs,
  validators verify hardware-attested proofs. Validators never call models —
  zero inference cost. Miners pay for their own API calls via an attested
  MeteringProxy inside the TEE.
- New `fugal_subnet/tee/` package with TDX quote parsing, DCAP verification,
  MeteringProxy, TEERuntime, network confinement, BenchmarkProof model,
  proof verification, and benchmark harness. TDX patterns forked from
  ThirtySpokes/Chutes (MIT licensed).
- New protocol: `FugalProofSynapse` — validator sends epoch_id + nonce,
  miner returns proof_bundle_url + proof_hash + weights_hash.
- Evidence accumulation (`fugal_subnet/evidence.py`): EWMA-decayed binomial
  with Wilson LCB scoring, artifact-keyed reset on retrain, miss=0 accounting
  to prevent selective publication.
- TEE safety invariant checks in `scripts/check_safety_invariants.py`
  (`check_tee_safety`): miner annotation check, harness grader import check,
  verify module model-call check.
- 24 TEE unit tests + 4 TEE attack tests in `tests/test_tee.py`.
- 11 evidence accumulation tests in `tests/test_evidence.py`.

### Changed

- Rewrote `neurons/miner.py` as TEE miner — runs benchmarks each epoch inside
  TDX, produces hardware-attested proofs. Old miner saved as
  `neurons/miner_legacy.py`.
- Rewrote `neurons/validator.py` as verify-only validator — checks TEE proofs,
  accumulates evidence, never calls models. Old validator saved as
  `neurons/validator_legacy.py`. Removed `--epoch-budget` flag (miners pay
  their own costs).
- Replaced fixed-cap model pool with routed-model pool: the matrix now includes
  only models that heads actually route to, eliminating the pool-eviction
  griefing vector (I4). Budget is the natural limiter; no fixed 30-model cap.
  When budget pressure requires dropping models, those routed by fewer heads are
  dropped first (least scoring signal lost).
- Added coverage multiplier to composite scoring: a head covering a fraction of
  the pool has its composite scaled by `intersection_size / pool_size`,
  preventing narrow-surface gaming.
- Pinned CPU kernel dispatch (ATEN_CPU_CAPABILITY, MKL_CBWR, DNNL_MAX_CPU_ISA,
  OPENBLAS_CORETYPE) and thread counts before both torch and numpy import,
  eliminating cross-machine embedding and scoring divergence between AVX2 and
  AVX-512 hosts. Pins are centralized in `fugal_subnet/determinism.py`.
- Validator backbone always runs on CPU in float32 — CUDA would select float16,
  so a GPU validator and a CPU validator would score the same head differently.
- Quantized routing utilities to a fixed grid (`ROUTING_DECISION_QUANTUM=1e-4`)
  before argmax, so near-tie routing decisions agree across validators even when
  BLAS kernels produce slightly different floats.
- Made mock operation the default and required explicit `--live` authorization
  plus a positive `--epoch-budget` for every paid OpenRouter workflow.
- Added atomic worst-case spend reservations (reserve/reconcile/forfeit),
  conservative retry accounting, and greater-of-canonical/live price protection.
  Provider overages are now charged (not raised), preventing double-payment on
  reasoning models whose completion_tokens exceed max_tokens.
- Removed the in-process validator re-exec watchdog; restart responsibility now
  belongs to systemd or the container supervisor.
- Pinned every direct runtime dependency to exact versions.
- Dedup seniority now tracks the earliest block a hotkey was ever seen
  committing, not its current commitment block. Prevents a copier from
  outranking the author after the author retrains.
- Added environment fingerprint (package versions, BLAS config, kernel dispatch,
  grader hash) to every reveal.json for post-hoc divergence diagnosis.
- Added startup environment assertion — live mode refuses to start on mismatched
  library versions or unpinned CPU dispatch.
- Added `--wallet-path` option to both miner and validator.
- Added head model-ID length cap (256 chars) and zero-model rejection.

- `scripts/check_determinism.py` — two-process differential determinism harness
  with 7 stages and `--perturb` mode simulating a second host. In CI.
- `fugal_subnet/attacks/run_miner_attacks.py` — 14 hostile head payloads (zip
  bombs, NaN, pickle, oversized IDs, etc.) verified blocked at the boundary.
- `tests/test_head_properties.py` — Hypothesis property-based tests on the head
  loader.
- `tests/test_non_interference.py` — regression tests for the dedup seniority
  inversion (I4).
- `docs/INVARIANTS.md` — the eight consensus invariants, what enforces each, and
  known gaps (including the I4 pool-eviction griefing vector).
- Doc/code flag consistency checker in `check_safety_invariants.py`.
- Ruff linting, pip-audit, CodeQL security scanning, CODEOWNERS, and structured
  issue templates.
- Bittensor Axon registration smoke test in CI.

### Security

- Added budget protection regression tests covering live-mode gates, concurrent
  reservation races, forfeit accounting, and spend-protection price computation.
- Prevent concurrent or retried API calls from passing a check-then-record race
  and overshooting the configured local epoch budget.

## [0.1.0] - 2026-08-31

### Added

- Initial validator and miner implementation for Bittensor SDK v10.
- Deterministic benchmark slicing and seven mechanical graders.
- Ground-truth matrix construction, soft targets, head evaluation, scoring, rewards, and behavioral deduplication.
- On-chain head commitments and epoch commit-reveal artifacts.
- Mock integration pipeline, adversarial grader suite, and Docker local-testnet tooling.
- Miner and validator operating guides.

[Unreleased]: https://github.com/fugal-ai/fugal-subnet/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fugal-ai/fugal-subnet/releases/tag/v0.1.0
