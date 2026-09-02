# Fugal v0.2 implementation record

This file is the durable handoff log for the consensus-breaking v0.2 hardening
work. It records completed phases, test evidence, and remaining rollout gates so
a new reviewer or coding session does not have to infer state from chat history.
For the concise current checkpoint and exact continuation commands, see
`docs/V0_2_HANDOFF.md`.

## Fixed decisions

- Preserve v0.1 for historical verification; all corrected live behavior is v2.
- Production target is Linux x86-64 with deterministic CPU-float32 inference.
- V2 excludes GPQA and LiveCodeBench from the canonical public benchmark pool.
- Use a versioned model/price registry and a five-validator builder committee;
  require at least three valid reports and abort closed without quorum.
- Execute untrusted grading in a networkless rootless OCI worker. The validator
  must never receive a container-engine socket or fall back to same-process exec.
- Mock mode is the default. Paid paths require `--live` and an explicit positive
  budget. Paid canaries, network activations, releases, and GitHub setting
  changes require separate user authorization.
- Publish bounded builder responses and evaluated head artifacts after an epoch.
- Testnet activation is a later rollout commit; mainnet activation remains unset.

## Baseline (2026-09-01)

- Clean `main` checkout at `e811529` on Ubuntu 24.04 WSL2, Python 3.12.3.
- Locked environment: Bittensor 10.5.0; `uv sync --frozen --extra dev` succeeded.
- Immutable v0.1 grader SHA256 (LF-normalized):
  `895809dedf0d14c45d9ec046bcbec2f50a09fcf7d31d9996a178e35f3539c55f`.
- Baseline safety invariants and integration suite passed.
- Baseline attack suite: 17 blocked, 3 documented residuals, 2 controls,
  0 surprises.
- Baseline integration loaded 21,717 questions locally; GPQA was gated and
  LiveCodeBench loaded zero, reconfirming that optional availability affects v1.
- Docker CLI 29.7.2 is installed and the WSL user can reach Docker Desktop.
  Real OCI grading tests now run locally. A historical v1 local-chain smoke ran
  three zero-spend epochs successfully; the required v2 five-validator suite
  has not yet been run.

## Phase status

### 1. Paid-operation guardrails — implemented, staged

- Validator and local-testnet launcher default to mock mode.
- Live operation requires `--live` plus `--epoch-budget` or an explicitly set
  positive budget environment variable; there is no default paid ceiling.
- `call_model()` additionally requires `live=True`, a budgeted tracker, and a
  spend-protection price, preventing accidental programmatic entry.
- Each HTTP attempt atomically reserves a conservative prompt/output maximum.
  Successful usage reconciles to actual cost; ambiguous failures forfeit the
  full reservation before any retry.
- Canonical prices remain the scoring/eligibility input. Reservations use the
  greater of canonical and current live rates and reject missing live prices.
- The paid canary refuses to run without explicit live authorization and budget.
- The in-process `os.execv` watchdog was removed; supervision is external.

Verification:

```text
python -m compileall -q fugal_subnet neurons scripts tests  PASS
pytest -q tests/test_paid_safety.py                         7 passed
python scripts/check_safety_invariants.py                  PASS
python neurons/validator.py --help                         PASS
python scripts/test_real_api.py                            REFUSED, exit 2
python scripts/launch_testnet.py --live --dry-run          REFUSED, exit 2
```

No OpenRouter completion request or chain mutation was made.

### 2. Consensus versioning and inactive v2 manifest — implemented, staged

- The unchanged `fugal_subnet/graders.py` is designated as the immutable v1
  module. The safety gate and runtime registry both reject any byte drift from
  its recorded hash.
- A strict packaged manifest records the legacy v1 contract and declares v2 as
  incomplete, disabled, and unactivated on local, test, and finney. Enabled or
  activated protocols must contain complete consensus material.
- The current canonicalized manifest hash is
  `eb17784950256e1bfae2bf350316f26d7df9308d325abbb87f93aa8338d9ea95`.
  It was recorded here as `5bf8b9c3...` before the final consensus-material
  rebuild; see phase 13 for how that staleness was found and prevented.
- Manifest overrides are accepted only in local/mock profiles. Testnet and
  mainnet always use the packaged resource.
- A versioned grader registry dispatches the immutable v1 grader and the
  packaged inactive v2 grader.
  A golden regression intentionally proves the known v1 `42.9 -> 42` behavior
  remains available for historical verification rather than being silently
  rewritten.
- The v1 fallback model/price snapshot is now a hashed package resource, so it
  is present in wheel and source distributions.

Verification:

```text
pytest -q tests/test_consensus_manifest.py                  7 passed
pytest -q tests/test_paid_safety.py                         7 passed
python scripts/check_safety_invariants.py                  PASS
uv lock --check                                             PASS
uv build                                                    PASS
wheel/sdist resource inspection                             PASS
```

The manifest now records verified dataset/model-resource/backbone/dependency and
worker-build hashes. It deliberately retains a null published worker registry
digest and an empty active model set; both must be resolved before v2 can be
enabled.

### 3. Inactive v2 routing, dedup, state, and rewards — implemented, staged

- V2 head validation requires unique bounded model IDs that are a subset of the
  active canonical registry. The wire `model_pool` must exactly equal the NPZ
  model list and order.
- Evaluation stores registry indexes/model IDs and full registry-aligned
  distributions. Utility ties use canonical registry order, so reordering head
  rows cannot change or disguise behavior.
- Dedup compares aligned decisions and distributions, builds deterministic
  components, and chooses the earliest commitment with hotkey/UID tie-breaks.
- Persisted v2 score records bind UID and hotkey; a UID ownership change resets
  all inherited score/head history, while unregistered records are removed.
- V2 rewards use Decimal box-simplex projection plus exact integer-unit
  quantization. The result sums exactly to one, respects ordinary UID change
  bounds, and gives immediate zero to invalid/duplicate/liveness-ineligible
  UIDs. Infeasible projections fail closed; no positive eligible score returns
  `None` so callers preserve chain weights. UID 0 has no burn role.
- A runtime contract checker binds every committee/transport/matrix bound and
  every routing, soft-target, dedup, scoring, liveness, cap, tie-break, and
  rounding parameter in executable code to the packaged manifest.
- A safety gate prevents the active validator from importing `fugal_subnet.v2`
  before a separately reviewed manifest activation.

Verification:

```text
pytest -q tests/test_v2_routing_rewards.py                 9 passed
python scripts/check_safety_invariants.py                  PASS
pytest -q                                                   43 passed
```

Regression vectors cover reordered-head dedup evasion, false local-index dedup,
wire/NPZ mismatch, residual disqualified weight, stretched caps, UID-0 burn,
infeasible redistribution, deterministic exact serialization, and UID ownership
changes.

CI now runs the complete pytest suite on Python 3.10, 3.11, and 3.12 in
addition to the direct integration and grader-attack entry points.

### 4. Inactive v2 epoch journal — implemented, staged

- The journal binds each epoch to its manifest hash and finalized boundary,
  uses a strict versioned JSON schema, and writes with same-directory
  `fsync`/`os.replace` atomicity under thread and `flock` serialization.
- Cell reservations are persisted before scheduling. Completed bounded response
  text, usage, cost, and SHA256 are cached idempotently; conflicting replays or
  tampering fail closed.
- A crash-left reserved cell is never rescheduled automatically. It remains
  visibly in-flight until the orchestrator conservatively forfeits its full
  reservation and aborts the epoch.
- Spend totals are recomputed from cells and cannot exceed the journal budget.
  Finalized commitment and chunked-report progress are persisted as resume state.
- Terminal and phase-regression rules prevent an incomplete or forfeited matrix
  from being marked complete.
- On startup, active journals from expired epoch boundaries are terminally
  marked aborted. Unreadable or symlinked journal state fails closed.

Verification:

```text
pytest -q tests/test_v2_journal.py                         8 passed
pytest -q                                                   43 passed
python scripts/check_safety_invariants.py                  PASS
```

### 5. Inactive v2 report transport and real Axon smoke — implemented, staged

- `MatrixReportSynapse` bounds epoch/hash/hotkey/signature metadata, chunk
  indexes/counts, and base64 payload size without deferred annotations.
- Reports are capped at 64 chunks of 256 KiB (16 MiB total). Assembly accepts
  out-of-order chunks but rejects missing/duplicate indexes, metadata drift,
  malformed base64, decoded oversize, and artifact hash mismatches.
- A canonical domain-separated signature message binds epoch, manifest,
  artifact hash, and builder hotkey. The report client and quorum layer verify
  report and per-chunk signatures before accepting an artifact.
- Safety inspection now covers every Axon-facing protocol module and every
  Synapse deserialize contract.
- A no-chain smoke test creates an ephemeral wallet and confirms real Bittensor
  10.5 `bt.Axon.attach()` registers both v1 and v2 Synapse annotations. CI runs
  this check on Python 3.10-3.12.

Verification:

```text
pytest -q tests/test_v2_protocol.py                        6 passed
python scripts/check_bittensor_axon.py                     PASS
python scripts/check_safety_invariants.py                  PASS
pytest -q                                                   49 passed
```

### 6. Deterministic v2 benchmarks and graders — implemented, staged

- The canonical normalized v2 pool contains 21,878 questions: MMLU 14,042,
  MATH 5,000, GSM8K 1,319, AIME 933, IFEval 541, and curated HumanEval 43.
  Every benchmark and the combined pool have packaged counts and SHA-256
  commitments; any difference aborts loading. GPQA and LiveCodeBench are not
  v2 members, and no skip/cache environment variable can change this loader.
- HumanEval v2 packages only exact-equality, JSON-type-preserving cases derived
  from the MIT-licensed pinned source. Every included task has at least eight
  unique cases; canonical solutions for all 43 tasks passed those cases in the
  real OCI worker. There is no `exec_unittest` fallback.
- The complete strict Google Research IFEval evaluator is vendored from commit
  `26d8ccdab6fec61b5c83ad6327ea8bda9e580288`. All 25 instruction classes and
  all 541 pinned rows load without missing semantics. NLTK 3.10.3 and its exact
  English Punkt parameters are pinned and packaged; langdetect is seeded.
- The two official rows requesting literal `#`/`!` frequency are patched to
  count that supplied character. Upstream randomly substitutes an ASCII letter
  for non-letter characters; repeated-process golden checks prove the v2
  adapter has no such grading randomness.
- The v2 grader identity hashes the complete packaged grading bundle: top-level
  dispatch, sandbox client/launcher/protocol, all vendored IFEval modules, and
  exact Punkt data. Worker build inputs and dependency lock remain separately
  committed in the manifest.
- V2 exact-integer grading uses Decimal: `42`, `42.0`, and `4.2e1` match 42,
  while `42.9` does not. Symbolic parsing and candidate code are delegated to
  the isolated service; service unavailability propagates and aborts rather
  than silently grading zero or falling back in process.
- Dataset/license evidence is recorded in the benchmark registry and NOTICE.
  AIME has no verified redistribution license and remains a rollout decision;
  MATH is referenced externally because its current dataset-card license and
  older loader metadata conflict.

Verification:

```text
canonical full-pool load                                      21,878 verified
official IFEval adapter across pinned dataset                 541/541 valid
HumanEval canonical solutions in real OCI worker              43/43 pass
pytest v2 grader/benchmark/sandbox/manifest subset             28 passed
```

### 7. Rootless networkless grading boundary — implemented, staged

- A bounded Unix-socket protocol separates the validator from a non-root OCI
  launcher. Linux peer credentials and socket permissions restrict callers;
  request bodies, responses, code, and gold values are never logged.
- Expected HumanEval outputs remain in the trusted launcher. Worker payloads
  contain only candidate code, function name, and test inputs.
- Every ephemeral worker uses no network or mounts, read-only root, non-root
  UID 65532, isolated bounded tmpfs, `pids-limit=1`, bounded memory/CPU/open
  files/output, dropped capabilities, no-new-privileges, built-in seccomp, and
  kill-on-timeout cleanup. The validator never receives an engine socket.
- The minimal worker is built from a base manifest digest and installs only a
  hash-locked symbolic parser dependency set. The local tested image ID is
  `sha256:57410e04114488e5439518597d9b51dff024a27a4f09e42d3516840e38e968d6`;
  it is test evidence, not the future published registry digest.
- Separate systemd identities and services are provided as deployment
  templates for the concrete v2 entry point. They remain inactive until the
  worker image has a published digest and the manifest rollout gates pass.

Verification:

```text
pytest -q tests/test_sandbox_protocol.py                     9 passed
python scripts/test_sandbox_oci.py                           PASS
  honest/environment/wallet/engine socket/host/network       PASS
  process spawn/read-only root/timeout/output cap             PASS
  symbolic parser/restricted socket round-trip                PASS
python scripts/check_safety_invariants.py                    PASS
```

### 8. Builder committee, signed reports, and historical receipts — implemented, staged

- Boundary hashes deterministically rank validator-permit hotkeys; at most five
  builders are selected and fewer than three permitted validators aborts.
- Builder report cores bind the epoch boundary, question/grader/manifest/model
  commitments, canonical question/model order, bounded response text and hashes,
  token usage, grades, reservations, actual spend, and builder hotkey.
- Reports and report chunks use separate domain-separated Bittensor signatures.
  Quorum accepts only committee reports, requires every committed artifact to be
  available and hash/signature valid, and rejects any canonical epoch mismatch.
- Consensus takes strict majority over all committed valid reports; even-sized
  ties deterministically grade zero.
- Question, report, and miner-head commitments use distinct compact on-chain
  namespaces. Historical receipts query the exact recorded block rather than a
  mutable latest pallet value, and submission requests inclusion plus finality.
- The candidate registry records six provider/cost-tier candidates and a fixed
  500-input/500-output-token routing-cost reference. All remain disabled and
  `owner_approval_required`, so a production load fails closed until terms and
  rollout review are explicitly completed.
- The v2 Qwen backbone is pinned to model/tokenizer revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, Linux x86-64 CPU float32,
  single-thread eager inference, fixed batching/truncation, and eight-decimal
  embedding rounding. Startup also executes four manifest-bound prompts and
  requires rounded embedding SHA-256
  `db07be511930164602fb74cc6cce8f1aa369fe146a7448eacc6c17f5b17221cd`.
  The legacy backbone remains unchanged for v1 history.
- The v2 entry point verifies every consensus-relevant installed distribution
  against the manifest before it loads benchmarks or starts its Axon.

Verification:

```text
pytest -q tests/test_v2_committee_reports.py tests/test_v2_commitments.py  13 passed
pytest -q tests/test_v2_backbone.py                                      5 passed
python scripts/check_v2_backbone_golden.py                              PASS
python scripts/check_safety_invariants.py                                PASS
```

### 9. Canonical reveal and offline verification — implemented, staged

- Canonical JSON reveals bind the manifest, questions, model/price registry,
  all signed builder responses, historical commitment receipts, consensus
  matrix, accepted and deterministically rejected head bytes/hashes, routes,
  distributions, dedup components, scores, projection inputs, and final weights.
- Offline verification rechecks exact historical state through a caller-supplied
  block resolver, signatures and artifact hashes, response grading, strict
  majority, head commitments/bytes, routing, dedup, scores, and exact weights.
- Missing, duplicated, unavailable, malformed, unsigned, hash-mismatched, or
  selectively refused committed artifacts abort verification.
- Exact reveal bytes are durably staged before weight submission and atomically
  promoted to immutable public `reveal.json` only afterward. Conflicting or
  symlinked staged/final paths fail closed.

Verification:

```text
pytest -q tests/test_v2_reveal.py tests/test_training.py     12 passed
fugal-verify-epoch --help                                   PASS
```

### 10. Resumable matrix/orchestrator/report service — implemented, staged

- A journal-backed matrix builder reserves before scheduling, persists bounded
  completed cells, and never repeats completed paid work on resume.
- The injectable epoch state machine enforces finalized question commitments
  before miner/paid calls, report commitments before the deadline, complete
  artifact retrieval, strict quorum, reveal verification, and only then weight
  submission. Every abort returns nonzero in once mode and preserves weights.
- Restarted builders reload exact committed local report bytes instead of
  rebuilding/resigning. Terminal completion is idempotent; chain hooks must also
  implement idempotent transaction submission around process-crash ambiguity.
- The Axon report store persists mode-0600 artifacts under a mode-0700 root and
  reconstructs signed chunks after service restart.
- The installed `fugal-validator-v2` command now binds the orchestrator to real
  Bittensor v10 adapters: finalized boundary selection, historical question,
  report and head receipts, committee report Axon exchange, canonical CPU
  embeddings, UID/hotkey liveness state, reveal publication, and idempotent
  finalized weight submission. The command checks the isolated launcher health
  before starting its Axon and refuses while the packaged manifest selects v1.
- Miner liveness is previewed for evaluation but persisted only after the weight
  operation succeeds. Same-epoch updates are idempotent, expired journals are
  aborted, and Dendrite/Subtensor sessions are explicitly closed at shutdown.

Verification:

```text
pytest -q tests/test_v2_matrix.py tests/test_v2_journal.py    PASS
pytest -q tests/test_v2_orchestrator.py                      PASS
pytest -q tests/test_v2_report_server.py                     PASS
pytest -q tests/test_v2_chain.py tests/test_v2_validator_state.py PASS
pytest -q tests/test_validator_v2.py                         PASS
```

### 11. Training, packaging, CI, and supply chain — implemented, staged

- `fugal-train` accepts verified v2 reveals and shares canonical prices,
  rounding, and routing evaluation with validators. Legacy NPZ import requires
  explicit compatibility authorization.
- Runtime resources ship in wheel and source distributions. Clean isolated
  installs smoke-test validator, miner, trainer, verifier, and resource hashes
  without the source checkout.
- Direct dependencies and the CPU-only PyTorch index are locked. Docker builds
  from the lock as UID 10001. The official local Subtensor image is pinned by
  digest and local test wallets/state are isolated under a run root.
- CI adds Ruff, mypy on consensus/security modules, `pip-audit`, CodeQL, Python
  and container SBOMs, signed package build-provenance attestations on main,
  Docker non-root smoke tests, Axon attach, all IFEval semantics, and identical
  golden vectors on Python 3.10-3.12.
- CODEOWNERS, structured protocol/bug issue forms, and release review guidance
  are present. Branch protection and signed release-tag creation are external
  owner actions and remain gated.

Final local zero-spend evidence on 2026-09-01:

```text
ruff                                                        PASS
mypy (32 consensus/security/runtime sources)                PASS
pytest                                                       178 passed
v0.1 integration pipeline                                   PASS
v0.1 attack suite                         17 blocked / 3 known / 2 controls
real Bittensor 10.5 Axon attach                              PASS
v2 golden (Python 3.10.20, 3.11.15, 3.12.3)                BYTE-IDENTICAL
v2 CPU-backbone golden (Python 3.10.20, 3.11.15, 3.12.3)   BYTE-IDENTICAL
v2 IFEval trace (Python 3.10.20, 3.11.15, 3.12.3)          BYTE-IDENTICAL
real OCI sandbox escape/resource suite                      PASS
pip-audit                                                    0 known findings
clean wheel and source install, all five CLIs               PASS
non-root application image build and all five CLIs          PASS
historical v1 local chain, three mock epochs                 PASS, $0 spend
```

The CPU-only `torch==2.14.0+cpu` wheel is served outside PyPI, so `pip-audit`
reports it as unauditable rather than vulnerable. The local v1 chain run proves
the launcher, wallet isolation, graceful SDK cleanup, miner/validator path, and
weight cadence; it is not a substitute for the still-gated v2 committee test.

The real v2 happy-path run produced five byte-identical, offline-verified
reveals with four builder reports, two accepted committed heads, complete
zero-spend journals, and derived weights. Its harness returned nonzero because
Bittensor 10.5 exposed an empty `metagraph.W` during the final weight
assertion. That adapter is corrected in phase 13.

### 13. Golden-vector reviewability and finalized weight reads — implemented

Two defects were found by running the committed `main` checkpoint rather than
trusting its recorded evidence. GitHub CI was red on both v0.2 commits
(runs `33570488216`, `33571201280`).

**Stale golden pin.** `tests/test_v2_golden.py` expected
`175e58fe...` and got `e0f196bd...`. CI on Python 3.10 and a local run on 3.12
produced the *same* actual hash, so cross-version determinism was never in
question: the pin had simply not been regenerated after the last
consensus-material rebuild, which also left the manifest hash above stale. The
root cause is structural — the golden vector embedded packaged material hashes
(`manifest_sha256`, and `question_commitment`, which derives from the grader
bundle hash) in the same opaque digest as the consensus math, and was stored
only as a hash, so ordinary rebuild churn was indistinguishable from a real
math regression and neither could be reviewed.

The vector is now `schema_version` 2 and split into two sections:

- `material` — `manifest_sha256`, `grader_sha256`, `question_commitment`.
  These legitimately move on every consensus-material rebuild.
- `math` — committee, slice, registry snapshot, matrix, soft targets, head
  evaluations, dedup, scores and weights. Derived only from fixed inputs, so
  any movement here is a consensus regression.

`EXPECTED_MATH_SHA256` pins the math alone and `EXPECTED_GOLDEN_SHA256` pins
the whole vector. `assert_golden()` reports a math mismatch first, because the
two failures require opposite responses: repin material churn, investigate math
drift. `tests/fixtures/v2_golden.json` commits the pretty-printed vector so
drift arrives as a readable JSON diff, and `scripts/update_v2_golden.py`
(with `--check`) is the only supported way to repin.

Before repinning, the math section was verified in full against the documented
invariants rather than accepted: five builders selected from five permitted
hotkeys; the reordered-head clone at UID 9 clustered with UID 5 and
disqualified; both heads producing identical decisions, distributions and
scores, proving row reordering cannot disguise behavior; every soft-target row
summing to exactly `1.000000000000`; accuracy `0.333333333333` confirming
all-zero questions are excluded; and final weights summing to exactly
`1.000000000000` with UID 9 forced to zero, UID 12 moving `0.20 -> 0.50` at
exactly the `0.3` cap and UID 5 moving `0.45 -> 0.50`. These values are also
independently covered by `tests/test_v2_routing_rewards.py`.

**Empty `metagraph.W`, and the commit-reveal defect behind it.** The previous
handoff diagnosed this as "Bittensor 10.5 returns an empty `metagraph.W`; read
the finalized `SubtensorModule.Weights` storage directly". That remedy was
implemented and acceptance still failed, because on a commit-reveal subnet the
weights are not in `Weights` either.

Measured directly against the local chain:

```text
commit_reveal_weights_enabled = True   (tempo 10, commit_reveal_period 1)
set_weights(...)             -> (True, 'Success')
  TimelockedWeightCommits    -> 1 row  [[214, [(hotkey, 2162, 0x91a015a3...)]]]
  Weights                    -> 0 rows
```

A successful `set_weights` writes an encrypted timelock commit; plaintext
`Weights` is populated only at the reveal epoch. After disabling commit-reveal
by sudo, the same submission produced `[(1, [(6, 43690), (7, 65535)])]` and the
adapter returned exactly `{6: 0.4, 7: 0.6}`, confirming the reader is correct.
`metagraph.W` stayed empty in both configurations, so replacing it was right
but not sufficient.

`read_finalized_weight_row()` reads the `SubtensorModule.Weights` storage map
at an explicit finalized block, normalizes the sparse u16 row, and raises
`ChainAdapterError` when the chain cannot be read; `finalized_weight_row()` is
the lenient wrapper that returns `None`, because the restart check must
resubmit rather than assume. `_chain_weights_match()` compares over the union
of chain and target UIDs at the existing `2/65_535` tolerance, so an extra
chain recipient is drift while an explicit zero weight matches whether or not
the pallet retained it.

The production consequence is the more important half. On any subnet with
commit-reveal enabled — the default here — a successful submission is invisible
to the plaintext read for the whole reveal window, so a validator restarting in
that window resubmits and burns the subnet `weights_rate_limit` (100 blocks).
`submit_exact_weights()` therefore also treats a pending
`TimelockedWeightCommits` entry from its own hotkey as an already-completed
submission, but only when that commit was made at or after the current epoch
boundary, so a stale commit from a finished epoch cannot suppress a new one. An
unreadable `CommitRevealWeightsEnabled` flag assumes the deferring path,
because a false negative there costs a duplicate submission.

Verification:

```text
pytest -q tests/test_v2_chain.py                            14 passed
pytest -q                                                  183 passed
scripts/check_v2_golden.py (3.10.20, 3.11.15, 3.12.3)   BYTE-IDENTICAL
  b0022276224630a94f895dfcd28cc61eb916047bc3cdd1159d467c22e961c8d0
EXPECTED_MATH_SHA256
  e15c8f129ebfe951685d97969729b984cd7da409b6e64a11bf32ed199b1a1a9d
ruff / mypy (29 sources) / safety invariants                PASS
```

**Local-chain acceptance result.** The five-validator harness now exits zero.
Independently checked artifacts from `/tmp/fugal-v2-final13`: five reveals with
the single SHA-256 `9a723e6c597e49299d0e30825866745ad9ae732f5d2e630465c0b9c87734e47a`,
a five-member committee, four builder reports, two accepted miner heads,
positive weights `6=0.421043925855` and `7=0.578956074145`, and five journals
terminally `complete` with actual spend `0`.

Two harness defects were fixed to get there. Offline verification loaded a
sixth pinned CPU backbone while all five validators were still resident, and
was OOM-killed on a 7.7 GiB host; validators and miners are now retired first,
with the grader launcher and chain left up. That kill produced no traceback,
and the harness reported only `stderr[-500:]`, a window the transformers load
report reliably fills, so a resource failure read as a consensus failure.
Failures now report the exit code, signal, and both streams.

**Evidence boundary — on-chain weight persistence is not covered.** This local
chain cannot demonstrate it at all: commit-reveal defers weights to an
encrypted commit, its drand-backed reveal never completes, and with ~0.3s
blocks and tempo 10 the commit expires within about a minute while offline
verification takes several, so no on-chain weight artifact survives to
assertion time. Disabling the flag is not a dependable workaround either — the
runtime rejects the admin call with `AdminActionProhibitedDuringWeightsWindow`
for most of a 10-block tempo (29 consecutive failures in the harness against a
success on the fifth attempt by hand). Acceptance therefore requires all five
reveals to record `set_weights` with identical exact weights, which is written
only after an accepted and finalized extrinsic, still asserts the plaintext row
wherever a subnet writes one, fails closed when a row is missing on a subnet
that should have one, and otherwise prints an explicit `EVIDENCE BOUNDARY`
line. Verifying post-reveal persistence requires a chain whose reveal
completes, which means the public testnet. The corresponding
`docs/RELEASE_CHECKLIST.md` item stays unchecked.

### 12. Remaining rollout gates (not implementation shortcuts)

The authoritative checklist is `docs/RELEASE_CHECKLIST.md`. In summary:

1. Resolve AIME/MATH redistribution and publication rights.
2. Explicitly approve a verified subset of model candidates and publish a
   digest-addressed grader worker image; then finalize a new manifest hash.
3. Independently review the concrete v2 Bittensor validator entry point and
   prove its transaction hooks against a five-validator local chain, including
   restart, quorum loss, refusal, UID transfer, and weight submission.
4. Meet the report-deadline SLO and run at least three zero-spend mock testnet
   epochs after a separate testnet activation commit.
5. Separately authorize any paid canary, GitHub branch settings, signed release,
   testnet activation, or eventual mainnet activation.

## Non-negotiable review checks

- Never add deferred annotations to `neurons/miner.py` or
  `fugal_subnet/protocol.py`.
- Every `np.load()` must explicitly use `allow_pickle=False`.
- `FugalSynapse.deserialize()` must return `self`.
- Never print any part of an OpenRouter key.
- Never run paid APIs without a separate explicit user approval and stated
  `[PAID ~$X]` ceiling.
- Never treat passing v1 tests as proof that documented v1 residuals are fixed;
  semantic corrections must be versioned as v2.
