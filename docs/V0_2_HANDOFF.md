# Fugal v0.2 engineering handoff

Last updated: 2026-09-01 19:25 America/New_York

This is the short operational handoff for continuing the v0.2 hardening work.
The design decisions and phase-by-phase implementation record are in
`docs/V0_2_IMPLEMENTATION.md`; release authority and unresolved owner gates are
in `docs/RELEASE_CHECKLIST.md`.

## Safety and rollout state

- The packaged v2 protocol is deliberately disabled, incomplete, and has no
  activation block on local, testnet, or mainnet.
- The packaged v2 model registry has no active models, and the worker registry
  digest is intentionally unset. Local acceptance uses private mock-only
  overrides under `/tmp`; it never edits packaged activation material.
- No OpenRouter request, paid canary, testnet/mainnet activation, release tag,
  or external GitHub setting change has been performed.
- Do not enable v2 until every applicable item in
  `docs/RELEASE_CHECKLIST.md` is satisfied and separately authorized.

## Current repository checkpoint

- Baseline commit: `e811529` on `main`.
- The complete v0.2 implementation is present in the working tree and was
  staged before the final local-chain fixes. The latest local-chain fixes and
  this handoff must be staged again with `git add -A` before committing.
- The immutable v1 grader hash remains
  `895809dedf0d14c45d9ec046bcbec2f50a09fcf7d31d9996a178e35f3539c55f`.
- The current packaged v2 grader-bundle hash is
  `0142b66ce2901eae197a55eb0e8d525cfecb50ddcda869b1052e4ba21cc3bdd4`.
- The locally tested OCI worker image ID is
  `sha256:57410e04114488e5439518597d9b51dff024a27a4f09e42d3516840e38e968d6`.
  This is local evidence, not the future published registry digest.

## Final local-chain work

The real acceptance harness is `scripts/test_v2_local_chain.py`. It starts a
disposable archive-mode Subtensor chain, creates five validators and two
miners, advertises their Axons, writes finalized question/report/head
commitments, runs the real networkless OCI worker, exchanges chunked signed
reports over real Bittensor Axons, verifies byte-identical reveals offline, and
checks finalized positive weights. It has no live/paid option.

The acceptance work found and fixed these real integration defects:

1. Candidate code exits, timeouts, malformed output, and output overflow are
   now ordinary failed grades only after a fresh trusted worker canary proves
   that the OCI boundary itself is healthy. Launcher/engine failure aborts.
2. Report-serving Axon threads no longer share a Subtensor client. The main
   chain thread advances a monotonic finalized-block release clock.
3. The local chain is started in archive mode so exact historical commitment
   state remains queryable.
4. Commitment submission and lookup use finalized blocks and direct historical
   hotkey metadata. They do not trust best-head state or repeatedly resolve a
   historical UID through a current metagraph.
5. Local miner Axons and head commitments are established before the v2 epoch
   boundary, so the accepted head set is non-empty and externally bound.
6. Question commitments are collected only after the precommit deadline is
   finalized. This prevents an early three-builder quorum snapshot from later
   rejecting a fourth builder whose valid report was committed before the
   report deadline.

Evidence before the last fix:

- One archive-mode run produced five byte-identical reveals and four builder
  reports, proving historical commitments, real Axon exchange, quorum, OCI
  grading, and reveal convergence. It had no evaluated heads because the
  miners had been advertised after the boundary; that harness bug is fixed.
- The next run included bound miner heads and two validators completed verified
  reveals plus weight submission. Three validators aborted on the question
  receipt race described above; the regression is now fixed.
- After the receipt fix, 41 focused validator/orchestrator/commitment/chain/
  sandbox tests pass, Ruff passes, mypy passes on 29 checked sources, and the
  safety invariant scanner passes.

The final8 rerun used:

```bash
uv run python scripts/test_v2_local_chain.py \
  --run-root /tmp/fugal-v2-final8 --timeout 1200
```

It completed the protocol path but the acceptance harness returned nonzero in
its final chain-weight assertion with:

```text
index 1 is out of bounds for axis 0 with size 0
```

This was after all five validators had finalized their reveal. Retained
evidence under `/tmp/fugal-v2-final8` proves:

- 5/5 byte-identical reveal files, SHA-256
  `f9a73ed9de0c2537a2b155c4162817eb7d922a02ecf183e71c00e124d7f29efd`;
- four independently committed builder reports and all four report artifacts
  received by every validator;
- two historically committed miner heads evaluated and assigned exact weights
  `0.421043989700` and `0.578956010300`;
- `set_weights: true` in the independently verified reveal;
- all five journals terminally `complete` with actual spend `0`;
- the offline reveal verifier returned success before the failing assertion.

The remaining failure is in `_verify_acceptance()`: it indexes
`subtensor.metagraph(netuid).W[validator_uid]`, but Bittensor 10.5 returned an
empty `metagraph.W` matrix on this local chain even though finalized
`set_weights()` calls succeeded. `fugal_subnet/v2/chain.py` also uses
`metagraph.W` in `_chain_weights_match()`, so the same SDK representation can
defeat restart idempotency by causing an unnecessary repeat submission. The
next implementation step is to read the finalized `SubtensorModule.Weights`
storage directly through `subtensor.weights(netuid, block=finalized_block)`,
normalize its u16 row, and use that shared adapter in both places. Add unit
coverage for an empty `metagraph.W` with a populated direct storage row, then
rerun final acceptance in a new empty `/tmp/fugal-v2-finalN` directory.

The disposable chain was cleaned up normally; diagnostics remain at
`/tmp/fugal-v2-final8`. Confirm no stale chain exists before rerunning:

```bash
docker ps -a --filter name=fugal_local_chain
```

Success must explicitly report five validators, at least three builder
reports, accepted miner heads, verified byte-identical reveal bytes, positive
finalized weights, and `$0` spend. Do not infer success merely from process
exit or partial validator logs.

## Validation already completed

Before the final local-chain fixes, the full zero-spend release suite recorded:

- 159 pytest tests passed.
- v1 integration passed.
- Attack suite: 17 blocked, 3 documented v1 residuals, 2 controls, 0 surprises.
- Real Bittensor 10.5 Axon attach passed.
- v2 general, CPU-backbone, and IFEval golden vectors were byte-identical on
  Python 3.10.20, 3.11.15, and 3.12.3.
- The real OCI escape/resource suite passed.
- `pip-audit` reported zero known findings (the CPU-only PyTorch wheel is
  unauditable rather than reported vulnerable).
- Clean wheel/sdist installs and all five CLIs passed.
- The non-root application container and all five CLIs passed.
- The historical v1 local chain completed three mock epochs with zero spend.

Because final-chain work changed consensus/security modules, rerun the release
commands from `docs/RELEASE_CHECKLIST.md` before the final commit. At minimum:

```bash
uv run ruff check fugal_subnet neurons scripts tests
uv run mypy fugal_subnet/v2 fugal_subnet/sandbox \
  fugal_subnet/training.py fugal_subnet/verify_epoch.py neurons/validator_v2.py
uv run pytest -q
uv run python tests/test_integration.py
uv run python -m fugal_subnet.attacks.run_attacks
uv run python scripts/check_safety_invariants.py
uv run python scripts/check_bittensor_axon.py
uv run python scripts/check_v2_golden.py
uv run python scripts/check_v2_backbone_golden.py
uv run python scripts/check_v2_ifeval_golden.py
uv run python scripts/test_sandbox_oci.py --skip-build \
  --image sha256:57410e04114488e5439518597d9b51dff024a27a4f09e42d3516840e38e968d6
uv run pip-audit --local --skip-editable
uv build
uv run python scripts/test_release_artifacts.py --dist-dir dist
```

Also run `git diff --check`, verify every `np.load` uses
`allow_pickle=False`, and rerun the Axon annotation safety scan before commit.

## Remaining work after a successful local-chain rerun

1. Record final command output and hashes in this file,
   `docs/V0_2_IMPLEMENTATION.md`, and `docs/RELEASE_CHECKLIST.md`.
2. Run the complete regression/release suite above and update test counts.
3. Stage all tracked/untracked project changes, review the staged diff, commit,
   and push the checkpoint requested by the owner; then record the GitHub CI
   result. Do not tag or publish v0.2 yet.
4. The real happy-path harness does not by itself close every scenario named in
   the release checklist. Quorum loss, selective artifact refusal,
   restart/resume, and UID ownership changes have deterministic unit/injected
   integration coverage. Either extend the disposable real-chain harness for
   those fault modes or leave that checklist item unchecked and document the
   evidence boundary precisely.
5. The remaining owner/external gates are still mandatory: resolve AIME/MATH
   publication rights; approve model IDs, prices, and response redistribution
   terms; publish/pin the worker image; measure a production-size deadline SLO;
   configure protected-branch settings; make a separate testnet activation
   commit; observe at least three zero-spend testnet epochs; and only then
   consider a separately authorized paid canary, signed release, or mainnet
   activation.

## Critical invariants for the next session

- No deferred annotations in `neurons/miner.py`, `fugal_subnet/protocol.py`, or
  any Axon-attached protocol module, including `fugal_subnet/v2/protocol.py`.
- Every `np.load()` uses `allow_pickle=False`.
- Every Axon Synapse `deserialize()` returns `self`.
- Never log any part of `OPENROUTER_API_KEY`.
- Never run an OpenRouter call without separate explicit approval and a stated
  positive `[PAID ~$X]` ceiling.
- Never activate packaged v2 or mutate testnet/mainnet from local acceptance
  evidence alone.
