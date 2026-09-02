# Fugal v0.2 engineering handoff

Last updated: 2026-09-01 21:05 America/New_York

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

- Baseline commit: `fe599c3` on `main`. Both v0.2 commits are pushed.
- GitHub CI was red on `fe599c3` and `64f78b7` (runs `33570488216`,
  `33571201280`); the stale golden pin that caused it is fixed.
- The immutable v1 grader hash remains
  `895809dedf0d14c45d9ec046bcbec2f50a09fcf7d31d9996a178e35f3539c55f`.
- The packaged v2 grader-bundle hash is
  `0142b66ce2901eae197a55eb0e8d525cfecb50ddcda869b1052e4ba21cc3bdd4`.
- The canonicalized manifest hash is
  `eb17784950256e1bfae2bf350316f26d7df9308d325abbb87f93aa8338d9ea95`.
- Golden pins: whole vector
  `b0022276224630a94f895dfcd28cc61eb916047bc3cdd1159d467c22e961c8d0`,
  math-only `e15c8f129ebfe951685d97969729b984cd7da409b6e64a11bf32ed199b1a1a9d`.
- The locally tested OCI worker image ID is
  `sha256:57410e04114488e5439518597d9b51dff024a27a4f09e42d3516840e38e968d6`.
  This is local evidence, not the future published registry digest.

## Local-chain acceptance: passing

`scripts/test_v2_local_chain.py` now exits zero. The last run:

```bash
uv run python scripts/test_v2_local_chain.py --run-root /tmp/fugal-v2-final13 --timeout 1200
```

Independently checked artifacts under `/tmp/fugal-v2-final13`:

- five reveals sharing SHA-256
  `9a723e6c597e49299d0e30825866745ad9ae732f5d2e630465c0b9c87734e47a`;
- a five-member committee and four independently committed builder reports;
- two accepted miner heads, weights `6=0.421043925855`, `7=0.578956074145`;
- five journals terminally `complete` with actual spend `0`.

Confirm no stale chain before rerunning:
`docker ps -a --filter name=fugal_local_chain`.

Three defects were fixed to reach this. See `docs/V0_2_IMPLEMENTATION.md`
phase 13 for the measured evidence behind each.

1. The stale golden pin, and the structure that made it unreviewable. The
   vector is now split into separately pinned `material` and `math` sections
   with a committed fixture at `tests/fixtures/v2_golden.json`; repin only
   through `scripts/update_v2_golden.py`, after reading the fixture diff. A
   `math` change is a consensus regression, never something to repin.
2. The commit-reveal weight defect. The previous handoff's remedy for the empty
   `metagraph.W` was wrong: on a commit-reveal subnet the weights are not in
   `SubtensorModule.Weights` either, they are an encrypted
   `TimelockedWeightCommits` entry until the reveal epoch. Production restart
   idempotency now also accepts a pending commit from its own hotkey made at or
   after the current epoch boundary.
3. Harness OOM during offline verification, which was misreported as a
   consensus failure because only `stderr[-500:]` was shown.

**Evidence boundary.** On-chain weight persistence is not verified and cannot
be on this chain: commit-reveal defers the weights, the drand reveal never
completes locally, and the commit expires long before verification finishes.
Disabling the flag is unreliable — the runtime rejects the admin call with
`AdminActionProhibitedDuringWeightsWindow` for most of a 10-block tempo.
Acceptance instead requires all five reveals to record `set_weights` with
identical exact weights, still asserts a plaintext row wherever one exists, and
prints an explicit `EVIDENCE BOUNDARY` line otherwise. The matching
`docs/RELEASE_CHECKLIST.md` item stays unchecked until a testnet run.

## Validation on the current tree

```text
pytest                                                      183 passed
ruff / mypy (29 sources) / safety invariants                PASS
v0.1 integration pipeline                                   PASS
v0.1 attack suite                         17 blocked / 3 known / 2 controls
real Bittensor 10.5 Axon attach                             PASS
v2 golden (Python 3.10.20, 3.11.15, 3.12.3)               BYTE-IDENTICAL
v2 CPU-backbone golden                                      PASS
v2 IFEval trace                            541 rows, 834 checks PASS
real OCI sandbox escape/resource suite                      PASS
pip-audit                        0 known findings (torch cpu wheel unauditable)
uv build + clean wheel/sdist install, all five CLIs          PASS
git diff --check / np.load allow_pickle audit                PASS
```

## Remaining work

1. Push this checkpoint and record the GitHub CI result on 3.10/3.11/3.12.
2. Fault modes still lacking real-chain coverage: quorum loss, selective
   artifact refusal, restart/resume, UID ownership change. Either extend the
   harness or leave the checklist item unchecked with the boundary stated.
3. Measure a production-size deadline SLO.
4. Owner/external gates, in the order the manifest enforces. A `complete: true`
   v2 is refused unless all three hold (`consensus_manifest.py:245-254`):
   clear `benchmarks.rollout_blockers` (AIME has no verified redistribution
   license; MATH has a dataset-card conflict, and reveals do publish question
   text, see `fugal_subnet/v2/reveal.py:289`), populate
   `model_registry.active_models`, and set a published `worker.image_digest`.
   Only then a separate reviewed testnet activation commit, branch protection,
   three zero-spend testnet epochs, and later any paid canary, signed release,
   or mainnet activation.

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
