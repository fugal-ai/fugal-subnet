# Protocol v3 rehearsal plan (2026-11-16)

The v3 release (one upgrade: the seat-confirmation rules plus protocol v3)
is rehearsed in three steps instead of a multi-day testnet run. Each step
must pass before the next; the finney blocks are chosen after (b).

## (a1) In process: done, no chain, no spend

`python scripts/rehearsal_v3.py` drives the shipped epoch code
(`run_v2_epoch`, the validator's call) through v2 epochs → the v3 activation
→ the odds-seat block, with three validators that must publish identical
reveals and weights every epoch: A (reference), B (every exploration answer
in the opposite order, killed after an epoch and restarted from its
persisted window and seat), C (rebuilt from A's published reveals after the
odds block). 37 checks, including: the window is emptied at the v3
activation and kept at the odds block, the seat resets at both, v3 reveals
publish the pooled exploration, every head's odds and the rules they were
built under, and the predicting head takes the seat on odds.

Found and fixed by it: a rebuild from reveals replayed v2 epochs into the v3
window during the first 48 epochs after the flip, while a live validator
empties the window at the flip (I1 divergence for any follower that rebuilt
or re-anchored then). `v2_rebuild` now replays only reveals built under the
newest reveal's protocol (`tests/test_v3_epoch.py::
test_a_rebuild_across_the_v3_flip_keeps_only_the_v3_epochs`).

`python scripts/rehearsal_v2.py` still passes (the v2 flip, unchanged).

## (a2) Local chain (Docker): the shipped neurons across both v3 blocks

Needs: Docker reachable from WSL (Docker Desktop → Settings → Resources →
WSL integration), the final v3 pool and its embeddings file (the miner routes
from the file and the validator computes odds from it, so a stand-in file
must be pinned through `FUGAL_EMBEDDINGS_FILE` + a test-only hash on the
local chain).

Add `scenario_v3_flip` to `scripts/dress_rehearsal.py`, modelled on
`scenario_v2_flip`: a non-finney local chain with v2 already active,
`FUGAL_V3_ACTIVATION_BLOCK` and `FUGAL_ODDS_SEAT_BLOCK` a few epochs apart,
shortened epochs (as v2_flip), three mock miners serving v3 heads over a
v3 fixture pool that includes code questions (`exec_func` and `exec_stdio`,
graded in the real sandbox), two validators plus one killed mid-epoch and
restarted. Checks: every epoch's reveals agree byte for byte apart from
`environment`/`v2_sync`; `protocol_version: 3` from the block; `head_odds`
and `exploration_pool` present; the seat resets at both blocks; a follower
started late (`FUGAL_ROOT` unset, bucket = the root's reveal directory)
re-anchors and agrees; code questions are graded (some pass, some fail).

## (b) One real TD: the final image, before the finney blocks are chosen

[PAID: one GCP TDX VM for a few hours; ask the owner first.] The testnet
rehearsal TDs (`fugal-v2r-td*`) and `scripts/` tooling already exist; do not
reuse a running VM without the owner's OK.

1. Build the v3 image from the release commit (CI `build-v<date>` tag) and
   take its digest; `scripts/check_safety_invariants.py --release` passes
   (embeddings, URL and system prompt pinned once v3 is scheduled).
2. Provision one miner TD on testnet with the test compose carrying the
   digest and the test activation blocks (`scripts/provision_td.py
   --inspect`, then push; dstack-cloud as in MAINNET_LAUNCH step 6).
3. In the TD log: the embeddings file matched its pin, `sandbox_selftest()`
   passed (the code grader's seccomp filter installed), and the first v3
   proof built inside the collection point.
4. `scripts/verify_live_miner.py` against it: PASS with the new entry, and
   refused with the old entry (`--expect-reject`).
5. A validator on testnet with the same release verifies that miner's v3
   proof (`valid`), and the proof's scored results include graded code
   questions.
6. Record the measured base and compose hash for the approved entry
   (`scripts/publish_approved_entry.py`, signed release).

Then choose the finney blocks, freeze the composes, publish the entry and
send the notice.
