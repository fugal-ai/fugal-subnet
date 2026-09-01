# Fugal v0.2 release and activation checklist

This is the fail-closed owner checklist for moving the inactive v0.2 code into a
published release or network activation. A checked code/test item does not grant
permission for an external, paid, or on-chain action.

Current local engineering status and continuation commands are recorded in
`docs/V0_2_HANDOFF.md`.

## Code and reproducibility

- [x] Preserve and hash-lock immutable v1 grading for historical verification.
- [x] Implement the canonical v2 benchmark/grader, sandbox, committee/report,
  journal, routing/dedup/scoring/reward, reveal/verifier, and trainer modules.
- [x] Default all commands to zero-spend mock operation; require `--live` and a
  positive hard budget for paid operation.
- [x] Pin runtime/build/test dependencies, CPU backbone revision/policy, local
  chain image, worker build inputs, and canonical serialization/rounding.
- [x] Add cross-Python golden/IFEval checks, real Axon attach, clean wheel/sdist
  installs, OCI escape/resource tests, lint, type checks, dependency audit,
  CodeQL, SBOMs, and build-provenance attestation configuration.
- [x] Implement a separately manifest-gated Bittensor v2 validator/report-server
  entry point with finalized boundaries and restart-idempotent weight hooks.
- [ ] Record a successful final CI run from the reviewed commit on Python
  3.10, 3.11, and 3.12.
- [ ] Record a successful clean release-artifact and container build from the
  reviewed release commit.

## Consensus material

- [ ] Obtain and record redistribution/publication approval for AIME question
  text or remove AIME in a new versioned registry.
- [ ] Resolve the MATH dataset license metadata conflict or remove MATH in a new
  versioned registry.
- [ ] Verify live existence, exact IDs, canonical prices, and bounded-response
  redistribution terms for each proposed active model.
- [ ] Owner-review and explicitly enable at most eight approved model entries;
  do not replace rejected entries merely to fill the cap.
- [ ] Publish the grader worker image and pin its immutable registry digest.
- [ ] Rebuild the final manifest, benchmark/model registries, golden vector,
  package artifacts, SBOMs, and provenance after every consensus-material edit.

## Network implementation and tests

- [ ] Complete independent review of the concrete v2 validator/report-server
  entry point and its exact historical-chain assumptions.
- [ ] Run the real local Subtensor suite with five validators and multiple
  miners: committee selection, historical commitments, chunked reports, quorum
  success/loss, selective refusal, restart/resume, UID ownership change, and
  weight submission.
- [ ] Demonstrate a production-size mock epoch completes before the manifest
  report deadline with documented hardware/concurrency SLO.
- [ ] Prove restart recovery never repeats a completed paid cell or finalized
  report artifact and never sets weights for an aborted epoch.

Historical v1 local-chain evidence is complete for three mock epochs with zero
spend. It does not satisfy the v2 five-validator/quorum acceptance items above.

## Owner/external gates

- [ ] Configure protected `main`, required CI/CodeQL reviews/checks, and required
  CODEOWNER review in GitHub settings.
- [ ] Create a separately reviewed local/test activation commit; keep testnet
  and mainnet activation blocks unset in the release candidate before approval.
- [ ] Run at least three healthy zero-spend mock testnet epochs.
- [ ] If desired, separately authorize one OpenRouter canary with an explicit
  `[PAID ~$X]` ceiling and verify reservation non-overshoot. No current approval
  exists.
- [ ] Create a signed annotated `v0.2.0` tag only after all release checks pass;
  attach the verified wheel, source distribution, SBOMs, and provenance.
- [ ] Observe at least three healthy activated testnet epochs before preparing a
  distinct reviewed mainnet activation release. Mainnet activation remains
  unset until that review.

## Commands for the reviewed release commit

```bash
uv sync --locked --extra dev
uv run python scripts/check_safety_invariants.py
uv run pytest -q
uv run python tests/test_integration.py
uv run python -m fugal_subnet.attacks.run_attacks
uv run python scripts/check_bittensor_axon.py
uv run python scripts/check_v2_golden.py
uv run python scripts/check_v2_backbone_golden.py
uv run python scripts/check_v2_ifeval_golden.py
uv run python scripts/test_sandbox_oci.py
uv run ruff check fugal_subnet neurons scripts tests
uv run mypy fugal_subnet/v2 fugal_subnet/sandbox \
  fugal_subnet/training.py fugal_subnet/verify_epoch.py
uv run pip-audit --local --skip-editable
uv build
uv run python scripts/test_release_artifacts.py --dist-dir dist
docker build --tag fugal-subnet:release-candidate .
```
