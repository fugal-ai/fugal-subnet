# Changelog

All notable changes to this project will be documented here. Releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Pinned CPU kernel dispatch (ATEN_CPU_CAPABILITY, MKL_CBWR, DNNL_MAX_CPU_ISA)
  and thread counts before torch import, eliminating cross-machine embedding
  divergence between AVX2 and AVX-512 hosts.
- Made mock operation the default and required explicit `--live` authorization
  plus a positive `--epoch-budget` for every paid OpenRouter workflow.
- Added atomic worst-case spend reservations (reserve/reconcile/forfeit),
  conservative retry accounting, and greater-of-canonical/live price protection.
- Removed the in-process validator re-exec watchdog; restart responsibility now
  belongs to systemd or the container supervisor.
- Pinned every direct runtime dependency to exact versions.
- Added ruff linting, pip-audit, CodeQL security scanning, CODEOWNERS, and
  structured issue templates.
- Added `--wallet-path` option to both miner and validator.
- Added Bittensor Axon registration smoke test to CI.
- Enhanced safety invariant checker with grader hash verification and paid-call
  guard AST scan.

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
