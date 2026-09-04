# Changelog

All notable changes to this project will be documented here. Releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed — first run against a real chain

`scripts/dress_rehearsal.py` runs the shipped binaries against a real local
subtensor node. Its first run surfaced eight defects no in-process test could
see, every one of them in the gap between "the library works" and "the program
works":

- **The neurons' own logging was disabled.** Importing bittensor runs a
  dictConfig with `disable_existing_loggers` at its default of True, disabling
  every logger created before it. A miner failing every epoch logged nothing
  and looked idle; the traceback went nowhere.
- **`--once` never exited**, hanging in a websocket teardown during interpreter
  finalization — so any orchestrator, cron job or systemd oneshot waited
  forever on a validator that had already finished.
- **The reveal block crashed** on a call site missed during the scoring rework.
  The epoch verified, scored and built the frame, then died.
- **Weights were reported set while the chain held none.** The subnet uses
  commit-reveal, so values appear only after the reveal period. The new
  confirmation handles both modes and turned the claim into a fact.
- **The miner made itself unreachable**, committing its head hash immediately
  before serving its axon; per-hotkey rate limits rejected the second
  extrinsic and the axon stayed at 0.0.0.0. It also now explains that
  subtensor rejects loopback addresses, instead of retrying silently.
- **The pool was re-embedded every epoch** though embeddings never change.
  Computed once at startup now, and `release_backbone()` returns the memory to
  the OS — 3194 MB to 749 MB, measured — rather than leaving it in allocator
  arenas where the kernel eventually OOM-kills a co-tenant miner.
- **The two neurons loaded different question pools.** The miner took a file
  while the validator called `load_all()`. Both now use the same source, with
  `pool_hash()` making a mismatch nameable.
- **Epoch geometry was duplicated** across both neurons — the same class of
  divergence as the epoch_id bug — and now lives in `slicer` with it.

### Changed — proof delivery

The bundle now travels inline in the synapse instead of through a HuggingFace
dataset repo, and `tee/store.py` is deleted. The HF pattern was inherited from
ThirtySpokes, whose artifact is a multi-gigabyte model; Fugal's is ~230KB,
where the ecosystem does the opposite. An external store's only remaining job
here was availability — for one party, once, from a miner provably online —
so it bought a round trip, a second party that must be up, and an account per
miner, in exchange for nothing.


### Fixed — the TEE path had never run end to end

- **Epoch identity diverged between the neurons.** The miner built its epoch_id
  from the block hash and the validator from the epoch index. Since the nonce is
  `sha256(f"{epoch_id}:{block_hash}")`, their question slices overlapped 45/300
  and `verify_proof` rejected every proof — the subnet could not have set
  weights. Epoch identity now has one source, `slicer.epoch_id_for_block`, and a
  structural check in `check_safety_invariants.py` prevents a second one.
- **The miner imported a backbone function that does not exist**
  (`get_hidden_states`; the real name is `compute_hidden_states`). Every epoch
  died inside an `except` clause that only logged the message.
- **The TEE harness passed a raw loader dict to the grader**, which reads
  `task["checker"]["id"]` / `task["domain"]` — keys the loader schema does not
  have. `grade()` caught the KeyError and returned 0, so every TEE-graded answer
  scored wrong. The translation is now shared (`fugal_subnet/grading_task.py`)
  by both the matrix builder and the harness.

CI did not catch any of these because it drove the pre-TEE pipeline end to end
and the TEE pipeline only in pieces. `tests/test_tee_e2e.py`,
`scripts/check_determinism.py` and the docker-compose entry point now all drive
the live path.

### Fixed — attested claims were not bound to anything

Five exploit classes verified against production code, each now blocked and kept
as an executable regression in `fugal_subnet/attacks/run_tee_attacks.py`:

- Approved-image matching used `proof.source_hash`, a field the workload writes
  about itself, so an attacker with genuine TDX hardware could run a modified
  harness and pass. It now uses `measurement_id(quote)` — the CPU's own MRTD and
  RTMR0-2, covered by Intel's signature.
- Results were never checked against the assigned slice, and `gold_answers` was
  the whole 21K pool, so a miner could grade 300 easy questions of its choosing
  while copying the (public) expected questions hash.
- `proof.weights_hash` was never compared to the on-chain commitment, so a miner
  could commit one head and run another while keeping a stable evidence key.
  The head now ships in the bundle and is hash-verified.
- The downloaded bundle was never checked against the advertised `proof_hash`.
- Cost inconsistency was a warning. Understating `total_cost_usd` raises a
  miner's score, so advisory treatment paid the attacker; it is now a rejection.

### Changed — scoring

- Replaced the 0.55/0.35/0.10 composite with `quality^0.8 * thrift^0.2`,
  measured against the best single model. A score of 1.0 means "matched the best
  single model's quality per dollar". The exponent is derived from the product
  claim, not chosen: giving up 40% of quality must not outscore matching the
  best model at its own price, forcing `w > 0.778`.
- Added the exploration quota and reference frame. Under TEE a miner only calls
  the model it routed to, so the counterfactual — and therefore any honest cost
  denominator — was unobservable. Miners now answer a nonce-chosen ~5% of extra
  questions with a nonce-chosen model; the samples pool over time into a
  per-model accuracy estimate.
- `_proof_to_head_score` used `min(per_model_costs.values()) * n` as its
  denominator, which treated totals as per-query prices and drew every term from
  the miner's own proof. Any single-model router scored a perfect 1.0. Measured
  on the real price table the incentive ran exactly backwards: all-frontier
  1.000 vs genuinely cheap 0.177; it is now 0.008 vs 1.000.
- `MeteringProxy` priced every model identically (a TODO stub). Costs now come
  from the hash-pinned `data/models.json`; the provider's own reported figure is
  recorded alongside so drift is detectable.
- Dropped `ROUTING_LAMBDA` from the routing rule, which is now
  `argmax(softmax(Wh+b))`. `p - lambda*cost` mixed a probability with dollars and
  asserted what a correct answer is worth. It survives as a miner-side training
  hyperparameter.
- Dropped the KL term (a constant 0.0731 for every miner under TEE) and the
  coverage multiplier (pinned to 1.0 under TEE).
- Capped Wilson's effective n at the distinct-question count, added a burn-in
  ramp so evidence reset is no longer a free penalty wash, and bound
  `MinerRecord` to a hotkey so a recycled UID does not inherit standing.
- Dedup now indexes routing decisions in a global model space. Per-miner indices
  clustered two distinct single-model routers as clones and let a real copy
  evade detection at 96.7% identical routing.

### Removed

- `neurons/validator_legacy.py` and `neurons/miner_legacy.py`.
- `MAX_MODEL_COST_PER_QUERY`, `MAX_MODEL_POOL`, `MAX_MODELS_PER_MINER` and
  `EPOCH_BUDGET_USD`. All existed to bound a shared model pool the validator
  paid for; under TEE there is no shared pool and the miner pays.

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
