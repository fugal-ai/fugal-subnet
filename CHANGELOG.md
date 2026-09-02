# Changelog

All notable changes to this project will be documented here. Releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Added a hash-verified inactive v2 benchmark pool: MMLU, MATH, GSM8K, AIME,
  43 curated HumanEval tasks with at least eight exact JSON cases each, and all
  541 IFEval prompts. GPQA and LiveCodeBench are excluded from v2.
- Added corrected versioned v2 graders using exact Decimal integer comparison,
  the complete pinned Google IFEval evaluator, isolated symbolic parsing, and
  isolated parent-compared code I/O. The immutable v1 grader remains unchanged.
- Added inactive v2 deterministic builder selection, bounded signed reports,
  strict-majority quorum, exact-block namespaced commitment receipts, a
  fail-closed candidate model/price registry, and a pinned CPU backbone policy.
- Added the complete canonical v2 reveal format and offline verifier, including
  exact historical chain receipts, response regrading, quorum reconstruction,
  submitted/rejected head artifacts, canonical routing, dedup, scores, weight
  inputs, and exact final weights.
- Added a journal-resumable v2 epoch orchestrator, journal-backed matrix builder,
  restart-restorable report store, and strict nonzero abort behavior. Completed
  paid cells and finalized report commitments are not repeated after restart.
- Added the separately gated `fugal-validator-v2` entry point, including
  finalized-boundary committee selection, historical commitment adapters,
  sealed report serving, UID/hotkey liveness state, canonical reveal
  publication, idempotent weight submission, and a mandatory isolated-grader
  readiness check. The packaged manifest keeps this command inactive.
- Made v2 output completion crash-consistent: exact reveal bytes are durably
  staged before the weight operation and only published afterward, liveness
  history is previewed before scoring and committed after success, and expired
  active journals are terminally aborted.
- Added the installed `fugal-train` and `fugal-verify-epoch` commands. Training
  consumes verified v2 reveals directly; unsafe legacy NPZ input requires two
  explicit compatibility flags.
- Added byte-identical v2 golden vectors and a full 541-row IFEval semantic trace
  to the Python 3.10-3.12 test matrix.
- Split the v2 golden vector into separately pinned packaged-material and
  consensus-math sections, committed the vector as a reviewable JSON fixture,
  and added `scripts/update_v2_golden.py` as the only supported way to repin.
  Consensus-material rebuilds no longer produce an opaque digest change that is
  indistinguishable from a real math regression.
- Vendored and attributed the Apache-2.0 IFEval implementation and its pinned
  NLTK English Punkt parameters; added HumanEval MIT provenance.
- Added inactive v2-only canonical-ID head evaluation and behavioral dedup,
  UID/hotkey-bound scoring state, and exact bounded-simplex reward projection.
- Added an inactive v2 atomic epoch journal with manifest/boundary binding,
  pre-scheduling reservations, idempotent bounded response caching, conservative
  crash recovery, and persisted commitment/report progress.
- Added an inactive bounded/chunked `MatrixReportSynapse`, deterministic report
  assembly and signature-domain bytes, and a real Bittensor v10 Axon-attach CI
  smoke test covering every protocol annotation contract.
- Updated the Bittensor test mock to enforce Pydantic wire-field bounds like the
  real SDK, eliminating collection-order-dependent protocol tests.
- Removed v2's UID-0 burn behavior and specified fail-closed handling for
  infeasible caps or an empty positive-scoring eligible pool.
- Preserved the exact v0.1 grader bytes as an immutable historical contract and
  added a packaged consensus manifest with v2 explicitly incomplete, disabled,
  and unactivated.
- Added strict block/network protocol selection, local-only manifest overrides,
  versioned grader dispatch, and a hashed packaged v1 model/price fallback.
- Made mock operation the default and required explicit `--live` authorization
  plus a positive epoch budget for every paid OpenRouter workflow.
- Added atomic worst-case spend reservations, conservative retry accounting,
  and greater-of-canonical/live price protection.
- Removed the in-process validator re-exec watchdog; restart responsibility now
  belongs to systemd or the container supervisor.
- Pinned every direct runtime/build/test dependency, upgraded NLTK to 3.10.3,
  locked CPU-only PyTorch, and added clean wheel/sdist installation tests for
  every installed CLI and packaged consensus resource.
- Added non-root locked container builds, Python and container SBOM generation,
  signed GitHub build-provenance attestations on main, CodeQL, dependency audit,
  Ruff, consensus-module type checking, CODEOWNERS, and structured issue forms.
- Pinned the local Subtensor image by digest and isolated local-chain wallets,
  state, caches, logs, and heads under an explicit run root.
- Added graceful Dendrite/Subtensor shutdown and local-chain cadence handling;
  three consecutive mock epochs now complete without leaked SDK sessions or
  violating the development chain's 100-block weight rate limit.

### Fixed

- Read finalized `SubtensorModule.Weights` storage directly instead of the
  derived `metagraph.W` matrix, which Bittensor 10.5 can return empty even
  after finalized `set_weights` calls.
- Treat a pending timelocked weight commit as a completed v2 submission. On a
  subnet with `commit_reveal_weights_enabled`, `set_weights` writes an encrypted
  `TimelockedWeightCommits` entry and no plaintext row until its reveal epoch,
  so the restart-idempotency check could never see its own submission and
  resubmitted, burning the subnet weight rate limit. Only a commit made at or
  after the current epoch boundary suppresses a new submission, and an
  unreadable commit-reveal flag assumes the deferring path.

### Security

- Added a separate non-root Unix-socket grading launcher and minimal pinned-base
  OCI worker. Candidate jobs have no network or mounts, a read-only root,
  isolated bounded tmpfs, one-process PID limit, dropped capabilities,
  no-new-privileges, built-in seccomp, and CPU/memory/file/output/time limits.
- Added real OCI attacks for credential/wallet/engine-socket visibility, host
  traversal, network access, process spawning, root writes, hangs, and oversized
  output, plus an end-to-end restricted-socket test.
- Added the unit/regression pytest suite to the Python 3.10-3.12 CI matrix.
- Added deterministic reveal-tampering, restart/resume, report-refusal,
  ownership-change, integer/IFEval/HumanEval, budget-overshoot, and weight-cap
  regression coverage.
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
