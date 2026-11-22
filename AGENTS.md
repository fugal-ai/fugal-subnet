# AGENTS.md — AI Agent Instructions

Instructions for AI coding agents working on this codebase.

## Project Overview

Fugal is a Bittensor subnet that finds the best cost-aware LLM router. A
router head (a small linear layer on a frozen Qwen3-0.6B backbone) reads a
question and picks one of the 18 models in the pinned price table. Miners run
TEE-attested benchmarks inside Intel TDX confidential VMs: they load their
head, call models via a metered proxy, grade the replies with hash-pinned
checkers, and produce hardware-attested proofs. Validators verify proofs
(never call models) and hold one grade per (question, model) cell that every
head shares.

**Protocol v2** (release/v2, from the published activation block) pays one
champion seat: a challenger takes it by beating the champion on the same
questions by a cost-adjusted margin (paired net-value test over a 48-epoch
window, after 1,500 scored questions of burn-in), a live champion holds with
no margin, copies never win, and the miner share splits 85/10/5 across the
last three champions. **Protocol v1** (live until the activation block)
scores every head as headroom above the constant-policy frontier with
Wilson-LCB evidence pooled across epochs. Both rule sets ship in one binary;
`slicer.protocol_version_for_boundary` picks per epoch.

## Critical Constraints

These are non-negotiable. Violating any of them breaks the subnet:

1. **NO `from __future__ import annotations` in `neurons/miner.py`
   or `fugal_subnet/protocol.py`.**
   Bittensor SDK v10.x `bt.Axon.attach()` inspects type annotations at runtime.
   Deferred annotations (PEP 563) break this introspection and the axon silently
   fails to register the forward function. Other files are fine.

2. **All `np.load()` calls MUST use `allow_pickle=False`.**
   Miners submit `.npz` files. `allow_pickle=True` on untrusted data is remote
   code execution. Grep for `np.load` before committing — every instance must
   have `allow_pickle=False`.

3. **Every Axon-attached Synapse `deserialize()` MUST return `self`, not a dict.**
   Bittensor's `dendrite.call()` calls `synapse.deserialize()`. If it returns a
   dict, the validator receives dicts instead of Synapse objects and attribute
   access breaks silently. `scripts/check_safety_invariants.py` checks it.

4. **NO API credit spend without explicit user approval.**
   Any code path that calls OpenRouter costs real money. Tag costs in comments
   as `[PAID ~$X]`. Never auto-run real API calls in tests or scripts without
   `--mock` being the default. The validator is the exception: it makes no
   model calls and defaults to `--live` (mock accepts unattested proofs); both
   neurons refuse `--mock` when the chain's genesis hash is finney's.

5. **NEVER print or log any part of `OPENROUTER_API_KEY`.**
   Not even the last 4 characters. Log "Key is set" or "Key not set", nothing more.

6. **`fugal_subnet/graders.py` is immutable.** Hash-pinned (SHA256 checked in
   CI); any byte change breaks consensus. It is excluded from ruff via
   `per-file-ignores`. `grading_task.py` is not the grader, but it decides what
   the grader sees: a change there changes every grade.

7. **`data/models.json` (v1) and `data/models_v2.json` (v2) are hash-pinned.**
   They are the consensus cost denominator: an unreviewed price edit silently
   re-scores every miner. Update the pin in `scripts/check_safety_invariants.py`
   in the same commit, and say why. The copies under
   `fugal_subnet/routing_data/` must stay byte-identical.

8. **Epoch identity has one source, `slicer.epoch_id_for_block`.** A second
   formatting of the epoch id is a consensus break. It happened once, and it
   stopped the subnet setting weights at all.

9. **Consensus invariants live in `docs/INVARIANTS.md`.** Read it before
   changing anything that affects scoring, determinism, or the miner interface.
   A consensus-affecting change updates it and adds an executable check.

## SDK Version

Bittensor SDK v10.x (10.0.0 - 10.x). Key behaviors:

- `bt.Wallet` (capital W), `bt.Subtensor`, `bt.Dendrite`, `bt.Axon`
- `dendrite.query()` returns a list of Synapse objects (one per axon)
- `subtensor.set_weights()` auto-uses commit-reveal-weights when the chain enables it
- `subtensor.query_map()` for reading on-chain storage (used by commitments.py)
- Wallet creation: `bt.Wallet(name=coldkey, hotkey=hotkey, path=wallet_path)`

## Architecture — What Each File Does

### Identity and constants
- `config.py` → every env-overridable constant (single source), with the reason each value is what it is. The three lists at the bottom classify every `FUGAL_` variable: consensus (digested, refused on finney), consensus input, operational.
- `fingerprint.py` → the consensus digest (what a score was computed with), the finney refusals for `--mock` and for consensus overrides, the code fingerprint in the reveal
- `consensus_v4.py` → protocol v4's section of that digest (present only in a v4 epoch's fingerprint, so a v1-v3 epoch's digest is unchanged): the activation block, every registered v4 constant in an exact canonical form, and the sha256 of every pinned v4 file (`PINNED_FILES`: the menu seed, the mix, the list signers, the signed lists, the published-set seed). `problems()` keeps the registry complete: every UPPER_CASE constant of a scanned module is registered, named in `NOT_V4` with its reason, or a pre-v4 name; a new v4 module, constant or data file fails the safety check and `tests/test_consensus_v4.py` until it is classified
- `determinism.py` → pins CPU kernel dispatch; imported before numpy and torch by both neurons
- `routing_protocol.py` / `pricing.py` → the protocol identity (price table + token manifest per protocol version, hash-pinned) and the cost of a routing policy from shared data alone
- `routing_data/` → the copies of `data/models*.json` and the token manifests that the identity hashes (must be byte-identical to `data/`; the safety check enforces it)
- `menu_rules.py` + `signed_lists/` → protocol v4 (design: `docs/decisions/LIVE_MENU_V4.md`; **nothing on the v1-v3 path imports it**): the compiled rules a menu entry must pass (a new proposal and an entry already on the menu differ only in how a pinned host missing from the listing is treated), and the signed lists they read (lab and host allowlist, device list), verified in process against the pinned copy of `deploy/keys/allowed_signers`. The rules' constants live here and nowhere else
- `vendor/success_contract.py` → the portable head contract (file format, cost estimate, ranking rule); vendored from fugal-core, byte-compared in CI
- `menu_state.py` → protocol v4 (design: `docs/decisions/LIVE_MENU_V4.md`; **nothing on the v1-v3 path imports it**): the menu as published state (`MenuState`, integers and strings only), its one canonical encoding and `menu_hash`, the pure derivation step (it is handed the signed list and applies `menu_rules` itself: what a list refuses is never an input), the two measured costs (`costs`, the call cost exploration reads; `pick_costs` / `pick_cost`, the pick-cost table picks, the product and net value read: W13, from graded scored-question cells via `menu_inputs.graded_cells`), the rule for which menu an epoch runs on (`menu_for_epoch`: a committed menu that fails the rules is `MenuRefused` and the carried one runs; without the bytes or the list, `MenuUnavailable`), and the pinned seed `routing_data/menu_v4_seed.json` (built by `scripts/build_menu_v4_seed.py`). `routing_protocol.identity_v4(menu_hash)` is the v4 proof namespace

### Questions (both neurons)
- `benchmarks/loader.py` + per-benchmark loaders, `benchmarks/v2_sets.py` → the pinned pools (v1: 5 benchmarks, v2: 10 open-licence benchmarks in 4 categories), checked against `pool_manifest*.json`
- `benchmarks/slicer.py` → epoch ids (the ONLY source), the activation boundary, the nonce, the v1 and v2 slice selection, the exploration draw
- `epoch_plan.py` → the one function both neurons call to get an epoch's scored slice and exploration set
- `exploration.py` → nonce-derived exploration assignment (the counterfactual samples the TEE would otherwise remove)
- `grading_task.py` → loader question → grader task (shared by the harness and every validator-side consumer)
- `graders.py` → the deterministic checkers (consensus-critical, hash-versioned, immutable)

### The miner's head
- `head_eval.py` → load and validate a head, evaluate it against a ground-truth matrix (success-contract heads rank by probability minus dollars); `load_head_v4` is protocol v4's acceptance (today's format, rows keyed by entry id, token means optional and never read), written beside the vendored contract
- `head_odds.py` → a head's exact integer odds (v3), and `picks_and_odds_v4`, the one function the v4 TD routes with and a validator recomputes with: odds on the epoch's menu in its order, picks priced at the menu's pick costs (`menu_state.pick_costs`) in Fractions
- `backbone.py` → Qwen3-0.6B hidden states (CPU float32 only; CUDA and float16 are refused)
- `api.py` → OpenRouter client (used by the miner inside the TEE, never by the validator)
- `scale.py` → SCALE codec, enough to read a TDX event log
- `soft_targets.py`, `success_training.py`, `head_archive.py` → reference training targets, the masked success trainer, and the validator's private archive of verified heads (operational, feeds no score)

### TEE pipeline (miner side, inside the TDX VM)
1. `tee/harness.py` → runs the benchmark: loads the head, routes, calls models via the proxy (24 in flight, 16k tokens in v2), grades, produces the proof
2. `tee/runtime.py` → MeteringProxy (records every call with tokens and cost) + TEERuntime (TDX attestation)
3. `tee/provision.py` → the attested provisioning channel (head, wallet, API key in; operator-only log pull out)
   - `tee/v4_options.py` → the operator's own v4 options (proposal, measurement ceiling, scored mode) as the provisioning push carries them: the strict parsers the miner's CLI/env path also uses, checked at the push boundary
4. `tee/proof.py` → BenchmarkProof and QuestionResult data models (the protocol v4 fields are hashed and serialised only when present, so earlier proofs keep their bytes)
5. `tee/td_v4.py` + `tee/endpoint_reads.py` → protocol v4, the TD side (**nothing on the v1-v3 path imports them**): the epoch's menu read through `menu_state.menu_for_epoch` and checked against the compiled rules before any call (no menu, no proof), wire requests from an entry's named fields, the strict keyless read of OpenRouter's endpoints listing (integer picodollars), measurement of the miner's own proposals under its spend ceiling, proposal filing, and `V4Session` (the TD's menu across epochs)
6. `tee/dstack_client.py` → the dstack guest agent socket (quotes, keys)

### Validator pipeline (verification, no model calls)
1. `protocol.py` → FugalProofSynapse wire format (epoch_id + nonce → proof_json + head_npz_b64 + proof_hash + weights_hash + event_log_json, all inline and size-capped)
2. `commitments.py` → on-chain head commitments (Commitments pallet), first-commit seniority
3. `tee/attestation.py` → TDX quote parsing, event-log replay to RTMR3 (every event's digest must bind its name and payload), DCAP verification
4. `tee/tpm.py` → the TPM half of a GCP dstack attestation
5. `tee/verify.py` → verifies a proof end to end: attestation, measurement against the approved entries, nonce, questions hash, head against commitment, cost consistency
6. `shared_grades.py` → protocol v2: one grade per (question, model) cell per epoch, shared by every head that routed there
7. `commit_reveal.py` + `reveal_publisher.py` → the epoch artifact (commit hash on chain, then the reveal), written atomically and published to the public store
8. `menu_publish.py` → protocol v4 (**nothing on the v1-v3 path imports it**): the root publishes the menu it derived (bytes and newly named signed lists put in the store and read back) before its hash goes into the reveal commitment (`reveal:<epoch>:<sha256>:<menu hash token>`); every node reads the root's commitment for E - MENU_LAG pinned at E's boundary and runs `menu_state.menu_for_epoch`; the others publish `menu_audit`; a node that missed epochs walks the chain forward (`rebuild`). The read helpers for miners and the product are in `reveal_sync` (`fetch_menu_bytes`, `fetch_signed_list`)
9. `head_publisher.py` → the root publishes every verified head by sha256: a seat holder's with the reveal that lists it, any other head after `FUGAL_PUBLISH_HEADS_AFTER_EPOCHS` epochs (operational, feeds no score; `reveal_sync.fetch_head` reads one back and checks the hash)

### Scoring, protocol v2 (`release/v2`)
- `v2_epoch.py` → one epoch end to end: verified proofs → shared grades → the comparison window → the seat decision → the weight vector → the reveal's v2 fields
- `champion.py` → the paired net-value test (integer units, deterministic bootstrap), the comparison window, the seat machine (first / hold / take / ghost / resume) and its accuracy gates
- `picks_v4.py` → protocol v4's real-picks check (rule B, `docs/decisions/SEAT_MINIMAL_V4.md` 3.4; **nothing on the v1-v3 path calls it**): each epoch's record of the heads that answered the optional scored questions, valued at the exact shared cell grade minus the epoch's pick cost, the answered run, `picks_check` on the one bound, and the absolute exploration failure bound with its outage waiver; published in the reveal's `picks_v4`, kept in the window, read back by a rebuild
- `seat_bound.py` → protocol v4's one bound (`docs/decisions/SEAT_MINIMAL_V4.md` 3.1; re-exported as `champion.lower_bound`; **nothing on the v1-v3 path calls it**): a one-sided normal-mixture confidence sequence at alpha = 1% on integer items with a stratified variance, in integers and Fractions only
- `labels_v4.py` → protocol v4's labels and odds (`docs/decisions/SEAT_MINIMAL_V4.md` 3.2, 4; `docs/decisions/FAST_CROWN_V4.md` 8, 18; **nothing on the v1-v3 path imports it**): which pooled labels a comparison of two heads reads (the last 48 epochs, after both rows count, voted by neither owner, never a label drawn while its entry was provisional), every head's odds on every epoch from its line's start (a successor's are its predecessors' stored odds before its commit; late odds for missed epochs), a missing row at odds 0 counted from the head's line and an added row's earlier labels likewise (`Side.added`, `Side.line`), the reference R as a side, the exact per-category sums the bound reads (`gain_strata`), and the same sums for many pairs at once (`gain_strata_many`, held equal to the per-pair path)
- `seat_v4.py` → protocol v4's seat (`docs/decisions/SEAT_MINIMAL_V4.md` 3.3 to 3.8, 4.4; `docs/decisions/FAST_CROWN_V4.md` 2, 6; **nothing on the v1-v3 path imports it**): `observe` (heads seen, new heads, succession by byte-identical rows under the same hotkey and coldkey, the rows a line added) then `decide` (the bar is the holder while it was verified in the last G = `STANDING_EPOCHS_V4` epochs, R does not beat it and the memorisation check does not refuse it, else R; contenders have a counted challenge in the last G epochs (`challenged_recently`, also the late-odds priority); rule A on the one bound with n0 = `BOUND_N0_V4`, the reverse picks bound, rule B on challenge items, rule C's precedence; the latest qualifier takes; comparisons in batches, or `per_pair`); the holder a challenge pairs with (`challenge_holder`, after E - 2); pay 85/10/5 on verification alone; the published seat state (`v2_state.seat`, `"protocol": 4`) and its evidence (`seat_v4`, digested in `seat_digest`). None of v1-v3's burn-in, ghosts, high-water marks, frame or agreement detectors or weight cap runs on a v4 epoch
- `chain_proposals.py` → protocol v4's chain proposals (`docs/decisions/FAST_CROWN_V4.md` 18; **nothing on the v1-v3 path imports it**): the proposal document a miner serves and commits by hash as its second commitment field (`proposal:<sha256>`), the root's pinned chain read a margin before the boundary, admission by a fixed order and quotas under the list in force, the join delta published by hash and the root's `join:<epoch>[:<sha256>]` pointer, `check_delta` (every node and TD re-runs admission on the delta's own contents) and the follower's audit
- `join_step.py` → protocol v4's join step (`docs/decisions/FAST_CROWN_V4.md` 18; **nothing on the v1-v3 path imports it**): the root's admission at its pinned read, the base menu plus the delta on every node, and each follower's `join_audit` (ALARM on a difference)
- `memo_v4.py` → protocol v4's memorisation check (`docs/decisions/SEAT_MINIMAL_V4.md` 6; **nothing on the v1-v3 path imports it**): the published set (per question its first publishing epoch, seeded from the pinned `routing_data/published_v4.json` built by `scripts/build_published_v4.py`, folded from each reveal), the rationed fresh draw from the unpublished reserve that every hotkey's private set holds (`plan_v4` with the published set), and a head's seen-against-unseen gap within benchmark under the one bound (`memo_evidence`, `.refused`)
- `memo_sync.py` → protocol v4 (**nothing on the v1-v3 path imports it**): how the published set moves between nodes. `ChainSet`, one class for the root, every other validator and a miner's TD: at each boundary the root's reveal commitment read pinned there, a newly named reveal folded (bytes checked by sha256), else nothing; the set as of E is what E's plan and verdicts both read; kept beside the state (`<state>.published_v4.json`) or in the TD's state directory; a node with no state anchors on the snapshot the named reveal names, else the seed; a reveal whose named view contradicts the node's state is an ALARM (a TD makes no proof). `SnapshotPublisher`: the root's upload of each named view by digest (`<netuid>/published/<digest>.json`) before the commitment. Nothing waits on the root
- `copy_rule.py` → first-commit seniority with the salted pre-flip tie, coldkey ownership, exact copies, ghost incumbents, the senior bars a junior must beat
- `payout.py` → 85 / 10 / 5 across the champion and the two before it; unpaid shares burn
- `v2_rebuild.py` → a validator that lost its state rebuilds the window and seat from published reveals (snapshot + replay)
- `v4_replay.py` → protocol v4's re-anchor (**nothing on the v1-v3 path calls it**): each v4 epoch's inputs to `run_v2_epoch` kept beside the state (`STATE_PATH.v4replay/`, a window plus two epochs, heads once by hash), and a re-anchor on the reference's state at k runs this validator's epochs k+1 .. now-1 again on it instead of keeping them, each with the memorisation memo the epoch loop built for it (`memo_for(epoch, recorded nonce)`: `neurons/validator._v4_memo_for`, on the published set as of that epoch from `memo_sync.ChainSet.view`; nothing folded); a span it cannot replay replaces nothing and waits for a later reference epoch
- `evidence.py` (v2 part) → the E'' burn-in counter: a miss after burn-in decays and restarts it

### Scoring, protocol v1 (live until the activation block)
- `reference_frame.py` → per-model accuracy and verbosity pooled from exploration; `frontier.py` + `scoring.py` → headroom above the constant-policy frontier, quality floor, burn-in, frontier confidence
- `evidence.py` → EWMA-decayed evidence with Wilson LCB, effective-n capped by pool size
- `rewards.py` → the weight vector and the per-epoch weight-change cap (±0.3, applies in v2 too)
- `dedup.py` + `head_distance.py` → v1 copy detection (behavioural cosine, parameter distance, the epsilon rule); v2 uses `copy_rule.py` instead

### Orchestrators
- `neurons/validator.py` → the epoch loop: block-aligned, state-persistent, retried epochs are idempotent; picks v1 or v2 per epoch
- `neurons/miner.py` → serves the head and proofs; holds a v2 head until the last v1 boundary, then commits it

### Support
- `epoch_logger.py`, `logging_setup.py` → structured JSONL epoch logs; logging that survives importing bittensor
- `consensus.py` → offline multi-validator audit tool (not in the live loop)
- `attacks/` → three executable attack suites (grader output, hostile miner input, forged TEE proofs); CI runs them. `attacks/seat_v4_world.py` and `attacks/memo_seat_world.py` are the constructed windows the v4 seat and memorisation cases (and the determinism stages) run on
- `data/models.json`, `data/models_v2.json` → the pinned price tables, one per protocol. Consensus-critical and hash-pinned like `graders.py`: a price change re-scores every miner.

## Graders Are Consensus Rules

`graders.py` checkers are NOT ordinary code. They are consensus rules — every
validator must produce byte-identical grades. A grader change is a consensus fork.

- Never "improve" a grader casually
- A semantic change requires a new grader version (applies from next epoch only)
- The grader version is `sha256(graders.py bytes)`, pinned in `scripts/check_safety_invariants.py`
- Run `python -m fugal_subnet.attacks.run_attacks` after any grader change — all 22 cases must pass with 0 surprises
- `grading_task.py` is not the grader, but it decides what the grader sees. A change there changes every grade.

## The Price Table Is Also A Consensus Rule

`data/models.json` (v1) and `data/models_v2.json` (v2) are the denominator
every validator prices every proof against. Both are hash-pinned in `scripts/check_safety_invariants.py` for the same
reason `graders.py` is: an unreviewed edit silently re-scores the whole subnet.
Changing prices means updating the pin in the same commit and saying why.

## Consensus Invariants

`docs/INVARIANTS.md` states the properties the subnet rests on (I1-I9, plus the protocol v2 section),
what enforces each, and the known gaps. A consensus-affecting change means
updating that file and adding a check that enforces the property — the reason
it exists is that a real consensus bug survived five code reviews because the
property was never written down.

`docs/decisions/OPEN_WORK.md` is the companion: what is unfinished, what was already
decided and should not be re-litigated, and what to do next. Read it before
picking up the PCCS/collateral work — the decisions in it cost several sessions
to reach.

## Testing

```bash
# The full suite, the way CI runs it (no API spend, no chain needed)
uv run pytest -q                      # -m "not slow" for a quick local loop

# Lint, safety invariants (pins, no-pickle, env classification; --release adds the launch gate)
uv run ruff check .
uv run python scripts/check_safety_invariants.py

# Scoring determinism across hosts (I1)
uv run python scripts/check_determinism.py --perturb

# Attack suite (grader verification)
python -m fugal_subnet.attacks.run_attacks

# Full dress rehearsal: the shipped neurons against a real local chain (Docker)
python scripts/dress_rehearsal.py --scenario all

# Protocol v2 flip: v1 epochs, the activation block, v2 epochs (seat, 85/10/5,
# E'' miss, restart mid-epoch). Chain half (Docker, ~30-60 min, not in "all"):
python scripts/dress_rehearsal.py --scenario v2_flip
# In-process half (seconds, no Docker; also run by tests/test_rehearsal_v2.py):
python scripts/rehearsal_v2.py

# Protocol v4, in process (no Docker, under two minutes): the shipped loops
# from v3 across the v4 block, the menu's life, and the seat's scenarios, on
# the shared fakes of tests/v4_world.py (also run by tests/test_rehearsal_v4.py)
python scripts/rehearsal_v4.py
# Its chain half (Docker, ~30 min, not in "all"): the shipped neurons across
# the v4 block on a real local chain, with an archive node, a local stand-in
# for the reveal bucket and a fixture published-set seed
# (scripts/dress_rehearsal_v4.py, scripts/rehearsal_shim.py)
python scripts/dress_rehearsal.py --scenario v4_flip
# A protocol v3 scenario on this tree and on origin/main, diffed (by hand)
python scripts/check_v3_equivalence.py --ref origin/main
# The v4 release gate: what a release that schedules v4 still lacks
uv run python scripts/check_safety_invariants.py --release-v4
```

Both v2 rehearsals probe which v2 pieces the branch has and SKIP (never fail)
the checks whose code has not landed. Both shorten burn-in and a few seat
constants so they finish quickly, and print the overrides they use.

There is no paid validator path: the validator never calls a model and has no
budget flag. Real spend only happens inside a `--live` miner, with a key pushed
over the attested provisioning channel (`scripts/provision_td.py`).

## Common Gotchas

- **Docker logging**: use `print(..., flush=True)` — bittensor's loguru swallows
  Python's logging module output in Docker containers
- **Benchmark downloads**: `datasets` library required; benchmarks download on
  first run, or once via `scripts/export_pool.py`. GPQA (gated) and LiveCodeBench
  are skipped by the pinned manifest. **Never set `FUGAL_SKIP_BENCHMARKS` on a
  neuron**: it replaces the manifest's skip list and the pool diverges silently
- **Local chain**: `scripts/dress_rehearsal.py` runs the shipped neurons as
  real processes on a pinned local subtensor (Docker); its chain helpers are in
  `scripts/setup_local_testnet.py`. It sets `FUGAL_POOL_UNPINNED=1` for its
  fixture pool
- **Weight setting on local chain**: the chain's 100-block rate limit means
  consecutive `--once` runs may get "weights already set" — normal at local
  testnet cadence, not an issue at 72-minute mainnet epochs (tempo 360 × 12 s = 4320 s, 360 blocks)
