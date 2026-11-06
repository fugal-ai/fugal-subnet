# Changelog

All notable changes to this project will be documented here. Releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed — epoch artifacts are written atomically; the publisher refuses a partial reveal

`commit.json` and `reveal.json` were written in place with `json.dump`, so a
crash or a full disk mid-write left a truncated file that `publish_due_reveals`
then uploaded and whose hash it committed on chain, and the `.published`
marker stopped any later retry from replacing it. Both files now go through
`write_json_atomic` (temp file, fsync, rename, like the state file), and the
publisher refuses to upload or commit a reveal that does not parse as a
complete reveal for its epoch.
### Fixed — the neurons price v1 proofs from the copy the protocol identity hashes

The validator priced v1 proofs from `data/models.json` while the protocol
identity it stamps on every reveal hashes `fugal_subnet/routing_data/models.json`;
the two are byte-identical today but nothing on the v1 path checked it. Both
neurons now read `routing_protocol.price_table(1)` (the miner keeps the
`FUGAL_MODEL_PRICES` override, which finney refuses), and
`scripts/check_safety_invariants.py` requires the two copies to be identical,
as it already did for the v2 table.
### Fixed — a pool skip-list mismatch is refused in strict mode

`loader._verify_against_manifest` returned early with a warning when the
process's `FUGAL_SKIP_BENCHMARKS` differed from the list the manifest was
built with, so a live neuron on its own skip list scored a pool nobody else
had and the check that would have said so switched itself off. Strict mode
(the live neurons) now refuses to start on a mismatch; non-strict keeps the
warning.
### Fixed — a frame rebuilt from reveals drops the voters the live epoch excluded

The live validator keeps a flagged operator's exploration samples out of the
reference frame (the two-sided detector) but published every observation and
no exclusion list, so `rebuild_from_reveals` re-admitted the flagged voter and
a validator that rebuilt its frame scored the field against a different
frontier from its peers (I1). Reveals now carry `frame_excluded_coldkeys`
and the replay applies it. Reveals from before the field re-admit as before.
### Fixed — every `FUGAL_` variable the package reads is classified; the pool inputs are digested; the reveal names the code

The classification check scanned `config.py` only, so `FUGAL_SKIP_BENCHMARKS`,
`FUGAL_POOL_UNPINNED` (benchmarks/loader.py) and `FUGAL_OPENROUTER_BASE`
(tee/runtime.py) changed which questions a neuron scores, or whether the pool
is verified at all, without being in the consensus digest or refused on
finney. The check now scans all of `fugal_subnet/` and `neurons/`; those three
are consensus variables; the fingerprint's `epoch` block carries the pinned
pool manifests' hashes, the skip list in effect, the unpinned flag and the
upstream override; and a new `code` block (`source_sha256`, `git_commit`)
names the code in every reveal while staying outside the digest.
**Every validator's consensus digest changes with this release**: upgrade
together.
### Changed — a live compose must be a frozen one; the activation block is part of the v2 compose

Both `deploy/dstack/docker-compose*.yaml` were edited after the 2026-11-03c
freeze (`FUGAL_POOL_CACHE_V2`) and matched none of the eight frozen
app-compose files, so a miner deploying from the checkout would compute a
compose hash no approved entry names. `scripts/check_safety_invariants.py`
now refuses a live compose that no frozen file embeds, and
`scripts/publish_approved_entry.py` refuses to write a note for a frozen file
that does not match a live compose (`--allow-drift` overrides). The docs and
`validator.env.example` now say where the activation block lives: in the
validator's environment AND in the v2 miner compose, so choosing it fixes the
v2 compose hash and the approved entry.
### Fixed — operator scripts: protocol-aware verification, a probe that can pass, tempo-derived defaults

`scripts/verify_live_miner.py` selected the slice and exploration with the v1
functions over the v1 table, so from the activation block it computed a
different slice from the validator and refused every v2 proof; it now picks
the epoch's inputs by the boundary-keyed version as the validator does.
`scripts/tdx_measurement.py --verify` built a proof with no hotkey, which
`verify_proof` refuses outside mock before the measurement check runs, and
told the operator to export a bare base that `--live` refuses; both corrected.
`scripts/provision_subnet.py` and `scripts/watch_subnet.py` defaulted the
epoch interval to 3600 (mainnet is 4320), reporting an aligned subnet as
MISALIGNED when the flag was forgotten; they now take `FUGAL_EPOCH_INTERVAL`
from the environment, else the chain's tempo x 12, and say which.
### Docs — the v2 numbers and formulas the guides and comments still had wrong

The v2 burn-in is 1500 questions (`V2_BURN_IN_QUESTIONS`), not 3000, in
MINER_GUIDE, VALIDATOR_GUIDE (new table row), the min-paired-questions comment
and `evidence.py`; V is 30 x the pinned anchor's expected spend (about $0.77),
not the champion's own spend, in MINER_GUIDE, `config.py` and `champion.py`
(whose docstring also said spend came from the frame's verbosity: it is the
cell's attested completion tokens); the Omni-MATH hard threshold is
difficulty >= 5 (`v2_sets.OMNI_HARD_DIFFICULTY`), not 7; the current published
image is `image-v2026.11.03c`, the digest the composes pin.
### Fixed — v2 rebuild refuses a malformed reveal by name; restored burn-in survives; the metagraph fallback is logged

The startup rebuild from reveals crashed with KeyError/ValueError tracebacks
on a truncated or malformed peer reveal and silently accepted a string
`models`; it now validates what it reads and refuses the file by path and
epoch. Records created to restore a head's E'' burn-in carried the head's
first committing hotkey, so an owner serving from a newer hotkey lost the
restored burn-in on its first verified proof; they now carry no hotkey (unknown,
kept). When the pinned metagraph read at the collection block fails and the
live one is used, the epoch log now records `metagraph_unpinned`, so a
divergence that epoch is attributable from the published artifacts.

### Fixed — v2 proxy: a reply that closes the connection no longer fails the call

The wall-clock call deadline set a per-read timeout on the connection's socket.
When an upstream answered with Content-Length and closed the connection (HTTP/1.0,
or HTTP/1.1 with `Connection: close`), `http.client` marked that socket closed
while the reply was still being read, and the next `settimeout` raised EBADF:
every such call failed. The watchdog now shuts the connection down through a
plain socket on a duplicated descriptor (not `sock.dup()`, which raises on the
TLS sockets every production call uses), and the per-read timeout tolerates the
closed socket. HTTP and HTTPS regression tests cover keep-alive, closing and
dripping replies.
Found by the v2_flip dress rehearsal; regression tests in tests/test_proxy_deadline.py.

### Changed — v2 model list: x-ai/grok-4.7 removed (18 models)

grok-4.7's provider does not bound reasoning by `max_tokens`: 17,763 completion
tokens were measured here, and 48,132 (47,656 reasoning) in an external
measurement, against 16,384. That is about five times the call's budgeted cost.
Removed from `data/models_v2.json` (and its copy) and from the v2 token manifest,
whose 18 other rows are unchanged; the call log keeps its rows.
`PINNED_PRICE_TABLE_V2_SHA256` and `identity_v2()` move. Consensus from
`V2_ACTIVATION_BLOCK`. grok-4.7 can return in v2.1, once the harness streams
calls and can abort one at its deadline (which stops billing on providers that
support it).

### Fixed — the proxy's call deadline is wall-clock

`PROXY_CALL_DEADLINE_S` (420 s) was a per-read socket timeout. OpenRouter keeps
a non-streamed request alive while the model reasons, so a call could run far
past it (917 s measured externally), keeping its proxy thread and upstream
connection open after the harness had given up. The proxy now sets the time
left as the timeout of every read, with a watchdog that shuts the socket down at
the deadline. The call ends by 420 s with its connection closed and is
recorded failed; the harness gets the proxy's 502 before its own 450 s wait. A
non-streamed reply is still billed in full by the provider. Miner-side only.
### Changed — protocol v2 seniority: pre-flip commits tie at the activation block

From an outside review: a UID with no v1 role could commit a v2 head days
before the flip and be the most senior head at it, taking the first seat by
timing. Every first commit at or before `FUGAL_V2_ACTIVATION_BLOCK` is now
ranked at the activation block, ties broken by sha256 over the activation
boundary's block hash, the head and its owner, a salt nobody holds while
pre-flip commitments are being made, so the order cannot be ground in advance.
Commits after the flip keep their real block. Validator-only; published in
the v2 reveal as `v2_seniority`.

### Changed — v2 miner: 24 calls in flight, the in-flight-budget 402 is retried, the pool is cached

- `HARNESS_CONCURRENCY` is 24 (was 8). At the 16,384-token v2 answer budget,
  reasoning models run long. A latency model calibrated on the v2 token run puts
  the p95 epoch of a head leaning on cheap reasoners at about 33 minutes with 8
  in flight, which misses the 36-minute collection block. With 24 it is about 14.
- An OpenRouter 402 is split by `metadata.reason`. `in_flight_budget_exhausted`
  (a low-balance or new account's in-flight holds are full) is temporary and is
  retried like a 429, honouring Retry-After. `weight_exceeds_budget` is an
  operator error. Any other 402 is the balance, never retried, as before.
- `FUGAL_POOL_CACHE_V2` (set in the dstack compose to
  `/app/results/pool_v2.json.gz`): the v2 pool is written to the results volume
  after its first manifest-verified load, and verified again on every read. A
  cache that fails is deleted and rebuilt from the pinned revisions. Restarts no
  longer need Hugging Face.

Miner-side only (an image and compose change; no consensus change).

### Fixed — the v2 miner retries its pre-flip commitment

The v2 image committed its head once, 10 blocks after the last v1 boundary. If
that one transaction failed, the commit fell back to after the flip, and the
miner's first v2 epoch(s) were rejected as committed late. It now retries every
`COMMIT_RETRY_BLOCKS` (2) until `COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS` (20) before
the activation block. It logs each failure and an ERROR if the deadline passes;
the post-flip fallback is unchanged. Miner-side only (an image change; no
consensus change).

### Changed — protocol v2 NV: V is pinned to an anchor, not the champion

`V = CHAMPION_ACCURACY_VALUE x anchor_spend_per_question(CHAMPION_ANCHOR_MODEL)`,
the anchor (Claude Fable 5.1) priced from the reviewed v2 token manifest and the
pinned v2 prices (about $0.026 a question, V about $0.77 at m = 30), instead of
the champion's own mean spend. The exchange rate between accuracy and cost no
longer depends on who holds the seat; a cheap champion no longer blocks a more
accurate, pricier router. `CHAMPION_ANCHOR_MODEL` is a consensus constant and
the derived `CHAMPION_ANCHOR_SPEND_PER_Q` is in the digest. Validator-only.

### Changed — protocol v2 burn-in is 1,500 questions (v2 only)

`V2_BURN_IN_QUESTIONS` (1,500, about five epochs) is the E'' burn-in a head
needs before it can take a v2 seat; the validator and `update_scores` use it in
v2 epochs, v1 keeps `BURN_IN_QUESTIONS` (3,000). Every head is new at the flip,
so the seat is empty, and the miner share burns, for about six hours after it.
Challengers still need `CHAMPION_MIN_PAIRED_QUESTIONS` (1,500) from an unbroken
run. Consensus constant, in the digest. Both rehearsals override it alongside
the v1 one.

### Added — the metering proxy names any reply over the answer budget (protocol v2 image)

Some providers bound reasoning tokens loosely. Measured 2026-11-04: one model returned 17,763
completion tokens against `max_tokens` 16,384. The proxy now logs the model and the overrun on
every such call. Nothing is refused: the call happened and is attested and billed either way.
This logs from every call rather than a one-off startup probe, and spends nothing.

### Changed — protocol v2 image wiring: the miner serves v2 only, holds until the flip, and skips epochs it cannot finish

- The harness takes `protocol_version`:
  - questions come from `epoch_plan`;
  - v2 heads are priced with `benchmark_costs_for(head, 2)` (the reviewed v2 token manifest and table);
  - every v2 call carries the model's pinned `reasoning` setting;
  - v2 proofs carry `identity_v2()`.
  v1 behaviour is unchanged.
- The TEE runtime's metering proxy takes the epoch's price table (`setup(prices=...)`).
- Miner: with `FUGAL_V2_ACTIVATION_BLOCK` set (the v2 compose), the pool, prices and model space
  are v2's. After embedding, the miner **holds**: it commits its head in the epoch before the
  activation block, `COMMIT_AFTER_BOUNDARY_BLOCKS` (10) after the last v1 boundary. Validators read
  commitments at the boundary block, so the last v1 epoch still sees the old head, and the flip
  boundary sees the new one. It registers its axon from the activation block. Unset means v1,
  as before.
- Miner: an epoch is not started with fewer than `LATE_START_MARGIN_BLOCKS` (60) blocks left
  before its collection block. Missing the block is a miss either way, and the calls are paid.

### Added — protocol v2 epoch inputs chosen by version in the validator

From `FUGAL_V2_ACTIVATION_BLOCK` the validator selects the scored slice and
exploration with `epoch_plan.scored_slice` / `exploration_plan`, and scores on
the v2 pool (`loader.load_pool(2)`), price table and model index space
(`routing_protocol.price_table(2)` / `models_for_protocol(2)`), all picked per
epoch by the boundary-keyed version; v1 is byte-identical. v2 inputs load at
startup when an activation block is set; live refuses to start without them.
New determinism stage `epoch_plan_v2`. INVARIANTS: "I1 — every epoch input is
chosen by the boundary-keyed version".

### Changed — protocol v2 wiring uses the copy rule for ownership and seniority

`run_v2_epoch` replaces its interim per-hotkey ownership with the copy rule
(`copy_rule.py`): it folds the boundary commitment snapshot (`observe`),
serves each head from its canonical owner UID only (`classify`), takes
seniority and the comparison's first epoch from the first commit
(`v2_seniority`, `first_epoch`), retires on the owner's commitment
(`retired_hashes`), and passes `senior_bars` (seat holders only) to
`seat_step`. The copy rule's state is persisted, published in the v2 reveal's
`v2_state` under `copy_rule`, and restored by the rebuild (`from_rebuild`).
The reveal also carries per-UID `copy_verdicts`.

### Added — protocol v2 model list: a second pinned price table, selected by protocol version (lands with v2)

- `data/models_v2.json` (identical copy in `fugal_subnet/routing_data/`): 19 models, prices
  from the public OpenRouter catalogue of 2026-11-04, each with at least three endpoints. Each
  pins exactly one OpenRouter `reasoning` setting: `effort` for OpenAI and xAI, a `max_tokens`
  budget for Anthropic, Google and Qwen, `enabled` otherwise. The unified parameter accepts
  `effort` or `max_tokens`, not both. Hash-pinned in `check_safety_invariants.py` like
  `data/models.json`.
- `routing_protocol.price_table(version)`, `models_for_protocol(version)` (the global model
  index space) and `reasoning_for(version, model)` serve the table of the epoch being scored,
  keyed on `slicer.protocol_version_for_boundary`. A release/v2 validator scores v1 epochs until
  the activation block, so the v1 table and `routing_protocol.identity()` are unchanged. The
  v2 table's hash is in the consensus fingerprint's epoch block.
- The v2 token manifest **candidate** (`data/benchmark_tokens_protocol_v2.json`, mirrored),
  built from 757 metered calls through the v2 harness. Review findings are in
  `docs/SUCCESS_CONTRACT.md`. Also: `routing_protocol.manifest_path(version)`,
  `benchmark_costs_for(head, version)` and `identity_v2()` (a new namespace; v1's
  `identity()` is unchanged).

### Fixed — protocol v2 payout: a ghost keeps seat 1

During a champion's ghost window the payout seats shifted down a seat (the
previous champion paid 5% instead of 10%, the third seat nothing), because the
ghost is the newest entry in the champion history. Seat 1 is now the ghost's:
its 85% burns and seats 2 and 3 keep paying the previous two champions.
Found by the v2 flip rehearsal.

### Added — protocol v2 state rebuilt from reveals (snapshot + replay)

`fugal_subnet/v2_rebuild.py`: the seat, owners, serving UIDs, burn-in
progress and any later v2 fields are read from the newest v2 reveal's
`v2_state`; the comparison window is replayed from the last
`CHAMPION_WINDOW_EPOCHS` v2 reveals. The frame rebuild applies the v2 mix
weights. A validator with no local v2 state and `FUGAL_FRAME_BOOTSTRAP` set
rebuilds at startup and restores burn-in counters. INVARIANTS: "I1 — v2 state
is rebuilt from reveals: snapshot + replay", including the trust model.

### Added — protocol v2: exploration weighted by the scored mix in the frame

From the activation block each aggregated exploration observation is
weighted by `slicer.mix_weights` (its benchmark's scored share over its
exploration share this epoch). `ReferenceFrame` keeps weighted sums for
accuracy and verbosity beside the raw ones; the raw trial count, which the
frontier's confidence reads, is unweighted. v1 frames serialise unchanged.
`aggregate_observations(..., with_question=True)` carries the question id.
The weights are in the v2 reveal (`frame_mix_weights`). INVARIANTS: "I9 — v2:
the frame is estimated on the scored mix".

### Added — `epoch_plan`: the single, versioned question-selection entry point for both neurons

`epoch_plan.scored_slice(version, nonce, pool, size)` and
`exploration_plan(version, nonce, pool, scored_ids, models, size)`. v1 is byte-identical to
`select_slice` and `expected_exploration`; v2 uses the category-first slicer with the v2
targets, map and weights. The exploration model assignment moves to
`exploration.assign_explore_models`, shared by both versions (v1 output unchanged).

### Added — protocol v2 wired into the validator (from the activation block)

`fugal_subnet/v2_epoch.py`: from `FUGAL_V2_ACTIVATION_BLOCK` the validator
runs one v2 epoch after verification. It computes shared cells (grades and
coldkey-median attested completion tokens) and the comparison window, steps
the seat, and sets the 85/10/5 seat vector with `paid_fraction = 1`. Nothing
that pays reads the frame: spend comes from cell tokens at pinned prices,
and the frame-based quality floor and epsilon clause do not run in v2. The
reveal carries the cells with tokens, per-UID failed questions and the seat
decision. The seat, owners and window persist (window as an `.npz`) and are
rebound only at the commit point. A failed call is graded 0 and costs 0 in v2.
New determinism stage `v2_epoch`. INVARIANTS: "I3/I4 — v2 pays from the paired
rule alone; the frame is informational".

### Added — protocol v2 model list: a second pinned price table, selected by protocol version (lands with v2)

- `data/models_v2.json` (identical copy in `fugal_subnet/routing_data/`): 19 models, prices
  from the public OpenRouter catalogue of 2026-11-04, each with at least three endpoints. Each
  pins exactly one OpenRouter `reasoning` setting: `effort` for OpenAI and xAI, a `max_tokens`
  budget for Anthropic, Google and Qwen, `enabled` otherwise. The unified parameter accepts
  `effort` or `max_tokens`, not both. Hash-pinned in `check_safety_invariants.py` like
  `data/models.json`.
- `routing_protocol.price_table(version)`, `models_for_protocol(version)` (the global model
  index space) and `reasoning_for(version, model)` serve the table of the epoch being scored,
  keyed on `slicer.protocol_version_for_boundary`. A release/v2 validator scores v1 epochs until
  the activation block, so the v1 table and `routing_protocol.identity()` are unchanged. The
  v2 table's hash is in the consensus fingerprint's epoch block.
- The v2 token manifest **candidate** (`data/benchmark_tokens_protocol_v2.json`, mirrored),
  built from 757 metered calls through the v2 harness. Review findings are in
  `docs/SUCCESS_CONTRACT.md`. Also: `routing_protocol.manifest_path(version)`,
  `benchmark_costs_for(head, version)` and `identity_v2()` (a new namespace; v1's
  `identity()` is unchanged).

### Added — protocol v2 copy rule (module; wiring follows)

`fugal_subnet/copy_rule.py`, from the activation block (v1 keeps
`commitments.seniority_key` and every `dedup.py` clause unchanged):
seniority is the block at which a head's exact weights hash was first
committed by any hotkey (same-block ties by a coldkey digest); a hash belongs
to the coldkey that first committed it, and the same hash under another
coldkey is an exact copy (`classify`; wiring names `owner`,
`first_commit_block`, `is_exact_copy`, `v2_seniority`); a retired head (its owner no longer
commits it) stays in the comparison set, unpaid, for `CHAMPION_GHOST_RETIRE`
epochs from its first retirement, and a head that missed for
`CHAMPION_GHOST_OFF`. `champion.seat_step(senior_bars=...)`: a head takes a
seat, first or by challenge, only by beating every senior SEAT HOLDER in that
set (the paid seats 1-3 and the champion's ghosts, `seat_holders`) by
`CHAMPION_MARGIN` on the same paired test; no second margin, no resemblance
test. A chain coldkey swap keeps the author's head: ownership follows the
first committer's hotkey. The state is a fold over boundary commitment snapshots, published as
plain JSON (`CopyState.to_reveal` / `from_reveal`: `epoch`, `first_commit`,
`ownership`, `first_hotkey`, `active`, `retired_at`) for the v2 reveal state block, rebuilt
from the newest reveal or replayed from an archive node (`from_rebuild` reads it from
`v2_rebuild.rebuild_v2_from_reveals`). New determinism stage
`copy_rule`. INVARIANTS: "I3/I4 — the v2 copy rule", including the accepted
front-running residual until commitments are bound to the coldkey.

### Added — protocol v2 benchmark pool: open licences only, category map and weights, its own manifest (lands with v2)

- `fugal_subnet/benchmarks/v2_sets.py`: knowledge = MMLU-Pro + a 3,000-question MMLU subset;
  math = MATH levels 3–5 + Omni-MATH integer items below difficulty 5; hard = SuperGPQA "hard"
  (all 7,050) + AIME 1983–2025 + BeyondAIME + Omni-MATH integer items at difficulty ≥ 5;
  instruction = IFEval (the mapped subset). **No gated datasets:** every miner embeds and every
  validator rebuilds the pool, and reveals publish its text. Licences per benchmark: MIT,
  Apache-2.0, ODC-BY, CC0-1.0. Subsets are chosen in sha256 order of a stable id, never load
  order. Revisions are pinned in `DATASET_REVISIONS`.
- `BENCHMARK_CATEGORY_V2`, `BENCHMARK_WEIGHT_V2` (hard: 50% the multiple-choice set, 50% integer
  olympiad math by pool size), `CATEGORY_TARGETS_V2` (40/25/30/5: instruction stays at 5% in v2.0 because only 259 IFEval items are checkable by the pinned constraint kinds, repeating every ~17 epochs at 15 a slice; a generated instruction set is v2.1).
- `loader.load_all_v2` / `load_pool(protocol_version)`: a separate pool, file
  (`data/benchmarks/pool_v2.json.gz`, `FUGAL_BENCHMARK_POOL_V2`) and manifest
  (`pool_manifest_v2.json`, with licences, categories and weights), verified the same way as v1.
  `scripts/build_pool_manifest.py --protocol 2` and `scripts/export_pool.py --protocol 2`. The v1
  pool is unchanged.
- No grader change: the new sets use `letter_mcq` (A–J) and `integer_exact` as pinned.

### Fixed — v2 slicer: unbiased apportionment (systematic sampling of the leftover units)

The v2 allocation gave leftover slots to the largest fractional parts every epoch. When a
category's or benchmark's fractions don't tie, that's a fixed bias the nonce never rotates. At
the v2.0 targets (40/25/30/5) the 15 exploration slots always split 6/4/4/1 instead of
6/3.75/4.5/0.75 on average, under-exploring the hard category by a sixth; inside a weighted
category a small benchmark could never win its fraction. Leftover units are now assigned by
systematic sampling over the fractional parts (one nonce-keyed offset), so each key's expected
quota equals its exact share. Exact rational arithmetic keeps every validator identical.
Integer allocations (e.g. 300 at 30/25/30/15) are unchanged.

### Added — protocol v2 champion scoring (module; wiring follows)

`fugal_subnet/champion.py`: the paired net-value test (NV quantised to exact
integers, V = m times the champion's own mean spend on the paired questions,
paired bootstrap within category, PCG64 seeded per pair and epoch), the
bounded comparison window (`CHAMPION_WINDOW_EPOCHS`, 48), and the seat state
machine (the most senior eligible head takes an empty seat, hold at no
margin, take at `CHAMPION_MARGIN`, high-water accuracy gates from the first
champion's lower bound, ghost bars, resume inside the ghost window), and an absolute
accuracy floor `CHAMPION_MIN_ACCURACY` (0.30, unmeasured on v2 benchmarks) for
taking and holding, which replaces the frame-based quality floor. A
challenger is measured only on its current unbroken run of served epochs; a
take needs `CHAMPION_MIN_PAIRED_QUESTIONS` (1,500) paired questions; a missed
ghost whose owner commits a new head becomes a retire ghost. Not wired
into the validator yet. INVARIANTS: "I3 — one champion; taking the seat needs
the margin, holding it needs none".

### Added — protocol v2 payout: 85/10/5 across the last three champions (module)

`fugal_subnet/payout.py`: `seat_weights` splits the miner share across seats
held by the current and two previous champions (`PAYOUT_SEAT_SHARES`). A seat
pays only while its head is live and eligible, otherwise to the current
champion, and burns when there is none; seats are paid to the serving UID, so
one hotkey keeps the seats it holds and extra registrations buy nothing. Not
wired into the validator yet. INVARIANTS: "I3 — seats, not shares".

### Added — protocol v2 miss rule E'' (from the activation block)

`Evidence.burn_in_n` counts scored questions since the last reset or miss.
From `FUGAL_V2_ACTIVATION_BLOCK`, a miss before burn-in resets the evidence
fully and a miss after it decays the evidence with no phantom zeros and
restarts burn-in (`apply_miss_v2`, `accumulate_epoch_v2`, `burn_in_complete`).
Pre-existing records migrate lazily (the counter reads as `n_total`). v1
epochs are unchanged. INVARIANTS: "I3 — the v2 miss rule (E'')".

### Added — v2 slicer: explicit benchmark weights inside a category

`select_slice_v2` and `select_explore_set_v2` take an optional `bench_weight` map. Within a
category a benchmark's share is its weight over the category total (default: pool size, as
before). The approved v2 hard category weights an open multiple-choice set and integer
olympiad math 50/50, rather than by raw pool size. A category that weights any benchmark must
weight all of them (a partial map is refused). The share cap and the exploration draw use the
same weighted shares, so expected exploration share still equals expected scored share.

### Added — protocol v2 slicer: category-first allocation; exploration matches the scored mix (not yet wired; lands with v2)

- `select_slice_v2` allocates the scored slice by category at published targets (largest
  remainder, no floor; ties keyed on the nonce), then by pool share within each category under
  the existing share cap (diligence finding 6), applied within the category.
- `select_explore_set_v2` allocates exploration the same way by category, and draws each slot's
  benchmark by a nonce-keyed weighted draw on pool share within the category. A benchmark's
  expected exploration share now equals its expected scored share at any size. Measured on
  mainnet with the v1 sampler: 15 exploration slots split five benchmarks 1:1:1:5:7, putting a
  benchmark that is 1.7% of the scored slice at 6.7% of exploration and over-weighting it ~4x
  in the reference frame.
- `mix_weights(scored, explored)` returns the per-benchmark correction (scored share ÷
  exploration share) for an epoch, so the frame can weight what 15 slots observed back to the
  scored mix.
- The v1 `select_slice` is unchanged. Dispatch by protocol version, and the benchmark→category
  map and targets, arrive with the v2 benchmark list.

### Changed — protocol v2 harness: a 16k answer budget, nested call deadlines, 403 as an operator status (image change; lands with v2)

- The harness asks for `max_tokens` = `HARNESS_MAX_TOKENS` (16384; was 2048). Measured on a
  320-question × 17-model matrix: at 2048 most long-reasoning models' wrong answers were
  cut-off answers; at 8192, 4.1% of all answers still stopped at the cap, nearly all graded
  wrong. Exploration (the reference frame's only source in v2) runs on this same harness,
  so miners and the frame ask for the same answer.
- Call deadlines nest so each layer reports before the next gives up: one upstream attempt
  ≤ the proxy's per-call deadline (`PROXY_CALL_DEADLINE_S` 420 s, was 180 s) < the harness's
  wait (`HARNESS_CALL_TIMEOUT_S` 450 s, was 200 s); no new retry after
  `PROXY_RETRY_BUDGET_S` (300 s, was 120 s). A 16k answer from a slow model takes ~330 s.
- Each result carries `latency_ms`: the harness's wall time for the call, retries included,
  attested and hashed. It's present only when set, so v1 proofs hash unchanged. The v2 time
  gate needs it.
- 403 joins 401/402 in `PROXY_OPERATOR_STATUS`: never retried, logged as
  `OPERATOR ACTION REQUIRED` (a key's total spend limit returns 403 on every call).
- Not yet in this change: the per-model reasoning setting (`reasoning.effort` or
  `reasoning.max_tokens`, one per model) and the startup output-length self-test. Both need
  the v2 model list (`data/models.json` is hash-pinned) and follow in that PR.

### Added — protocol v2 shared grades (from the activation block)

`fugal_subnet/shared_grades.py`: one coldkey-median grade per (question,
model) cell of the scored slice, with failed calls dropped. A head's own failed
call stays wrong. From `FUGAL_V2_ACTIVATION_BLOCK` the validator computes the
cells each epoch and the reveal carries `protocol_version` and
`shared_grades`; v1 reveals are unchanged. Nothing scores from them yet: the
paired champion comparison reads them in the next change. New determinism
stage `shared_grades`. INVARIANTS: "I4 — shared grades: a cell is a vote of
operators".

### Added — protocol v2 activation block and v2 constants (no behaviour change yet)

`FUGAL_V2_ACTIVATION_BLOCK` names the epoch boundary at which validators switch
to the v2 rules. The version an epoch is scored under is a function of its
boundary block (`slicer.protocol_version_for_boundary`), so retries and late
validators agree; a block that is not an epoch boundary is refused at startup;
the value is a consensus input in the digest's `epoch` block. Unset means v1
for every epoch. The v2 champion and payout constants (`CHAMPION_*`,
`PAYOUT_SEAT_SHARES`) are added as consensus constants. Nothing reads them yet;
the v2 rules land behind the activation block in later changes. INVARIANTS:
"I1 — protocol v2 activates at one named boundary".

### Docs — MINER_GUIDE: where the reveals are published; no GPU needed

The public reveal bucket (`fugal-reveals/70/<epoch_id>/reveal.json`), what a
reveal holds, and how to check it against the validator's rolling on-chain commitment. The
requirements list no longer recommends a GPU: success-contract heads embed on
CPU float32, which the contract enforces.

### Fixed — proofs can carry their own TPM intermediates (validator half)

`verify_ak_chain` accepted `extra_certs` and INVARIANTS said a proof could
carry its own intermediates, but `unwrap_attestation` never passed any, so only
the vendored us-central1 intermediate could chain: a Google rotation, or a
miner in another region, would have failed every proof at once.
`BenchmarkProof.tpm_intermediates` (optional DER list, bounded at 4 x 8 KB) is
now passed through. It is outside `content_hash`, so no `report_data`,
`proof_hash` or existing proof changes, and it is omitted when empty. A supplied
certificate still has to chain to the pinned Google root by signature; two new
`run_tee_attacks` cases pin that. Known gap: being outside `content_hash`,
the intermediates are not bound to `proof_hash`, so a miner can show different
validators different chains for its own proof (INVARIANTS). The miner half (the
harness embedding its chain, and adding it to `content_hash` when non-empty)
needs an image change and is not part of this.

### Docs — MINER_GUIDE: failed model calls score as wrong answers

New section under Scoring. `mistral-large-2512` is rate-limited upstream with a
single OpenRouter provider; the section gives the mainnet failure measurements,
says to treat availability as part of accuracy, and covers the OpenRouter
account itself (no total key limit, a low-balance alert: 403/402 are not
retried).

### Added — a private archive of verified miner heads on the validator (diagnostics only, opt-in)

Every axon response carries the miner's head; the validator checked it
against `weights_hash` and the on-chain commitment and then discarded it.
With `FUGAL_HEAD_ARCHIVE_DIR` set, heads that verified are kept as raw bytes
at `<dir>/<sha256>.npz` (written once per hash) with one `index.jsonl` row
per (epoch, uid): hotkey, coldkey, weights_hash, `first_seen` or `repeat`.
The directory is 0700 and files 0600; nothing is uploaded. Heads are never
parsed by the archive, a hash the bytes don't match is skipped (so no miner
string becomes a path), and a disk error is logged, not fatal. Unset (the
default), behaviour is unchanged. No change to scoring, any hash, the reveal
or the fingerprint.

### Added — per-UID failed-call counts in the epoch log (diagnostics only)

A failed upstream call is scored `correct=False`, so from the score alone a
miner whose calls failed looked identical to one whose answers were wrong, and
the validator kept nothing that could tell them apart (proofs are not stored;
the reveal's matrix records 0/1). `epochs.jsonl` now carries `n_failed`
(UID → scored calls the proof marks `failed`; exploration excluded, as from the
score), and the per-UID score line ends `(N/M correct, K failed)`. No change to
scoring, any hash, the reveal or the consensus digest; logs written earlier
read `n_failed` as absent.

### Added — approved entries for image-v2026.11.03c (the retry image), one per network

`image-v2026.11.03c` (signed, commit `1a0bce0`) built
`ghcr.io/fugal-ai/fugal-miner@sha256:843460e6…4a6f8`, public. Both composes pin
it; the frozen `app-compose.frozen-2026-11-03c-{finney,test}.json` embed them
byte for byte and their raw-byte sha256 are the compose halves of the entries
(finney `9ab870cf…f166`, test `c9b5c43b…0dad8`). The base half was re-measured
on the first TD booted from this image before publication (rotation table).
The 03b entries stay approved alongside for the 48-hour transition window per
`docs/APPROVED_ENTRY_ROTATION.md`; seed 1 keeps serving 03b until seed 2 has
served a verified epoch on 03c.

### Changed — the miner's proxy retries transient upstream errors; 401/402 are named as operator problems; failed results carry canonical number types (next image rotation)

Mainnet day one: one provider's 429s failed ~13% of a seed's calls and each
was scored as a wrong answer, which held an otherwise frontier-level head
0.11 below the frontier; an exhausted OpenRouter balance (402) failed ~55% of
an epoch and looked, at first, like a bad head. `MeteringProxy` now retries
408/409/425/429/5xx up to three attempts with 1/3/9 s jittered backoff,
honouring `Retry-After`, capped at 20 s per wait and 120 s per call, inside the
harness's own wait; 401 and 402 are never retried and are logged as
"OPERATOR ACTION REQUIRED". The harness writes a failed call's `cost_usd` as
`0.0` and its tokens as ints, so the JSON the validator hashes is the JSON the
miner hashed (the validator-side half is #120). All of this is inside the
measured image: it ships with the next `image-v*` tag and approved entry.

### Fixed — a proof with a failed call no longer fails verification on the validator

`QuestionResult.from_dict` coerced `cost_usd` to float; the harness records a
failed call's cost as the integer `0`; canonical JSON distinguishes the two, so
the validator's `content_hash` differed from the miner's attested `report_data`
on every proof that contained a failed call. Both seed miners' first mainnet
epoch was rejected for it (2026-11-03). `from_dict` now keeps every JSON number
as it arrived and rejects bools, strings and non-finite values. Validator-side
only; the image is unchanged. INVARIANTS states the rule.

### Added — the bootstrap reference frame, built from direct logged calls

`data/reference_frame.json` ships and is pinned (`PINNED_BOOTSTRAP_FRAME_SHA256`).
It was built by the new `scripts/build_frame_from_calls.py`: 17 models × 300
nonce-stratified pool questions, called through the miner's own `MeteringProxy`
with the harness's prompt and token cap, graded by `graders.grade` with the
harness's exec setting, folded with `aggregate_observations` and
`accumulate_exploration`. Every call is a row in
`data/observations/bootstrap_frame_calls_v1.jsonl` (model, question, correct,
token counts, provider cost, response hash, pool hash, protocol identity,
source hash, commit), so anyone with the pool and a key can re-issue and
re-grade it. The provenance trade (logged calls, not TEE quotes) is stated in
`INVARIANTS.md` item 3 and in the script. `build_bootstrap_frame.py` (from
reveals) is unchanged. OPEN_WORK records what the launch day taught: the base
measurement follows GCP's firmware, a rate-limited model caps everyone's
payout, the CUDA wheels in the image, model-list v2, and key spend limits.

### Added — the first signed release's approved entries, one per network

`image-v2026.11.03b` (signed by the key in `deploy/keys/allowed_signers`,
commit `bac2a33`) built `ghcr.io/fugal-ai/fugal-miner@sha256:3f6e3e2d…fda12`,
public. Both composes now pin that digest and their status comments describe
it (the previous comment, baked into the attested compose, still named the
September private image). The frozen files are
`deploy/dstack/app-compose.frozen-2026-11-03b-test.json` (test/552, the
warm-up, `FUGAL_EXPLORE_FRACTION=0.5` inside) and `…-finney.json` (finney/70);
they embed exactly the repo composes, were produced by `dstack-cloud prepare`
from `Phala-Network/meta-dstack-cloud` at tag `v0.6.1` (`5a1bfea`), and their
raw-byte sha256 are the compose halves of the entries. The base half was
re-measured on the first two warm-up TDs on 2026-11-03 and is
`f4be9383…aa427`, not the September `12a1f2f5…9141`: RTMR1 and RTMR2 (the
dstack image) are unchanged, MRTD and RTMR0 (GCP's TDX firmware) moved. The
entries are therefore `f4be9383…aa427:dcd0dd8d…2d7f` (test) and
`f4be9383…aa427:ed2be85f…85c4` (finney); the frozen files differ only in
`docker_compose_file`, and the two composes only in network, netuid and the
warm-up's exploration fraction. Every doc that quoted the September base as
current now quotes the measured one with the date, and `tests/test_tpm.py` pins
today's registers beside the September fixtures. The September entries are
retired in the rotation table; no mainnet validator ever carried them.

### Added — OCI labels on the miner image

`Dockerfile` now sets `org.opencontainers.image.source`, `title`,
`description`, `licenses` and `vendor`. Without them the ghcr.io package page
showed the base image's (uv's) description and was not linked to this
repository. The labels live in the image config, so they are in the digest:
the first build carrying them is a new image and a new approved entry.

### Added — the release signing trust root has its first key

`deploy/keys/allowed_signers` now exists, with one ED25519 key
(`release@fugalai.com`, fingerprint
`SHA256:MpaQWZxhrW77AJg8HAR2TxgFrag5HAo+i9oDnUiIBO8`, git namespace only).
Until now the file was absent by design and `scripts/fetch_approved_entries.py`
failed closed; from here a `git tag -s image-v<date>` by that key verifies, and
the first signed release can be cut. Adding or rotating a key stays a reviewed
pull request to this file.

### Changed — consensus overrides are refused on finney by genesis hash and allowed with a warning elsewhere; the testnet twin compose explores at 0.5

A live neuron with a consensus constant in its environment was refused at
startup on every chain, and `FUGAL_ALLOW_ENV_DRIFT=1` was the only way past,
which also waived the library and CPU-dispatch pins. Both neurons now check
the overrides against the connected chain's genesis hash, as `--mock` already
is: refused on finney, and the drift flag no longer lifts that; allowed with a
warning that names the override and the consensus digest on a test or local
chain. The reason is the netuid-552 warm-up: the reference frame needs
`FRONTIER_MIN_TRIALS` (300) distinct exploration trials per model before the
frontier pays in full, which is about 17 days at the shipped 0.05 and about 2
days at 0.5 (150 slots an epoch over 17 models), and the frame does not depend
on the fraction. `deploy/dstack/docker-compose.test.yaml` therefore sets
`FUGAL_EXPLORE_FRACTION=0.5` (the testnet entry attests it; the mainnet compose
does not set it, so the two twins now differ in that line as well as network
and netuid) and `validator.env.example` documents the matching validator line.
`MAINNET_LAUNCH.md` corrects §1 (before `start_call`, root-staked validators
do hold permits and do land weights, measured on netuid 70 at block 9422854;
only staking fails) and records that the protocol label
`fugal-testnet-success-v1` is not renamed for mainnet, since it is in the
identity the bootstrap frame is built under; the same note sits beside the
string in `routing_protocol.py`.

### Changed — the deploy files name mainnet: finney, netuid 70

The subnet is registered on finney as netuid 70 (block 9364303, 2026-10-26;
`start_call` not yet called as of block 9422441). `deploy/dstack/
docker-compose.yaml` now targets `finney`/`70`; its testnet twin for the
netuid-552 warm-up is `docker-compose.test.yaml`, identical except for
network and netuid, so one image digest yields one compose hash and one
approved entry per network. `image.yml` freezes and attaches both. The image
line in both is still the pre-transition testnet digest until the first
`image-v*` release. `MAINNET_LAUNCH.md` records the chain facts that bind
the launch (tempo, commit-reveal, the immunity end block and the pruning
rule) and its commands name netuid 70; `validator.env.example` and
`APPROVED_ENTRY_ROTATION.md` point at the right compose.

### Changed — a validator permit no longer admits a caller to a live miner's axon; the stake floor is 50

Review 2026-11-02 (miner surface), decided 2026-11-03. `caller_allowed`
admitted any hotkey with a validator permit, and subtensor grants a permit
to any nonzero stake while fewer than `max_allowed_validators` hotkeys are
staked, so on a young subnet a dust-staked hotkey could download every head.
Stake at or above `MIN_VALIDATOR_STAKE` is now required of every caller; the
permit is reported in the reason, not trusted. The quantity compared is
`metagraph.S`, the chain's `total_stake` for the UID — alpha on this subnet
plus root TAO at the chain's 0.18 root weight, after child/parent-key
inheritance (`get_stake_weights_for_network`) — which is what the chain
itself uses for permits. The default drops from 1000 to 50 in those units:
the gate is a speed bump in front of dedup, not the defence, and a floor set
too high silently starves every validator of proofs while one set too low
only cheapens a download that dedup already makes worthless. The value ships
inside the measured image, so changing it is an image rotation. A live
validator now logs a warning at startup when its own stake is below the
floor, instead of discovering it as "No valid proofs received" every epoch.

### Changed — Python 3.11 is the floor; the miner image moves to 3.12

GitHub's Dependabot alerts reported 17 open advisories on `main` (1 high,
GHSA-v4x8-3wr5-3wq8) while `pip-audit` in CI was clean: every one was
`filelock` 4.1.0, the resolution fork the lockfile kept for Python 3.10,
and every fixed `filelock` requires Python 3.11 or newer. The miner image
was built on Python 3.10 (`Dockerfile`), so it shipped the vulnerable one.
`requires-python` is now `>=3.11,<3.13`, the lockfile has one `filelock`
(4.2.6), the CI matrix is 3.11 and 3.12, ruff targets py311, and the image
base is `uv:0.8.14-python3.12-bookworm-slim`, pinned by digest. The image
digest changes, so the compose hash and approved entry change with it — as
they do for the reviewed manifest in the same release.
### Fixed — the validator checks the chain's weight parameters, retries a failed submit, and keeps a state backup

Validator review 2026-11-02, findings 3, 4 and 6. At startup a live
validator now reads `CommitRevealWeightsVersion` and the subnet's
`weights_version` and refuses to run if the SDK would commit a version the
chain rejects at reveal, or a `version_key` below the subnet's — failures
that never appear in `set_weights`'s result and would have looked healthy
for as long as nobody checked emission. `set_weights` is wrapped: the
chain's rate limit is read first so a refusal is named instead of logged as
`None`, and a failed submit is retried up to `FUGAL_WEIGHTS_SET_RETRIES`
(3) times while blocks remain in the epoch. `save_state` fsyncs before the
atomic rename and keeps the previous epoch's file as `.bak`; `load_state`
falls back to it with a loud error when the main file is unreadable, and
ignores fields a newer or older release added, so an upgrade or a power
loss no longer makes deleting the state the only recovery.
`FUGAL_ARCHIVE_ENDPOINTS` / `FUGAL_FALLBACK_ENDPOINTS` reach
`bt.Subtensor`, and a live validator without an archive endpoint is warned
at startup. `tests/test_validator_chain_guards.py`.
### Fixed — the provisioning receiver takes one push, bounds every connection, and refuses oversized bodies

The TD-side receiver (`fugal_subnet/tee/provision.py`) accepted any number of
pushes for the life of the miner, and each replaced the session key, so anyone
who reached port 8092 after provisioning could lock the operator out of the
TD's logs; the attest endpoint stayed open as a quote oracle; the server was
single-threaded with no socket timeout, so one half-open connection stalled
provisioning for good; and `Content-Length` was unbounded (security read
2026-11-02, finding 7). Now: one push per process (`409 already provisioned`
on both `/provision` and `/provision/attest` afterwards; re-provisioning is a
stop/start of the instance, which the miner guide says), a 3 MB body cap
refused before a byte is read, a per-connection timeout, and a threaded
server. `/provision/logs` is unchanged. Three new tests in
`tests/test_provision_transport.py`.
### Changed — verification runs every free check before the collateral fetch

`verify_proof` ran DCAP (a network fetch against the per-epoch collateral
budget every miner shares, whose FMSPC target is unsigned at parse time) and
the measurement match FIRST, then the free local bindings. A replayed or
stale proof — wrong nonce, wrong report_data, wrong advertised hash, wrong
slice, relayed from another hotkey — cost every validator a fetch that a
comparison would have refused (I4, I6; security read 2026-11-02, finding 5).
The order is now: hotkey, unwrap (TPM half), report_data, nonce, questions
hash, slice, exploration, head commitment and bytes, head compatibility,
advertised hash, gold, failed flags, exploration failure rate, cost
consistency, and only then DCAP and the approved image/application. No
rule changed; `tests/test_verify_order.py` counts the fetches. A leaf
certificate whose signature algorithm the pinned root's key cannot verify is
now a rejection in `verify_ak_chain` rather than an escaping `TypeError`
(finding 6).
### Changed — CONSENSUS: the token manifest is reviewed, from fresh observations

The success contract's token manifest (`data/benchmark_tokens_v1.json`) was a
`candidate` built from 945 pre-transition calls with three samples for most
models, and a live validator or miner refused to start until it was reviewed.
It is now rebuilt from 664 metered calls recorded on 2026-11-02 through the
benchmark's own proxy and call path on a stratified sample of the pinned pool
(40 per model; 24 for mistral-large, upstream-rate-limited), reviewed, and
promoted to `reviewed` with the record inside the file. **This changes the
protocol identity**: every head, state file, reveal and proof produced under
the candidate identity is rejected, the rehearsal fixture
(`tests/fixtures/success_bundle`) is regenerated, and the measured miner image
and approved entry are rebuilt from this commit. Billing cross-check on the
sample: table-priced $2.158 vs provider-billed $2.159. `docs/SUCCESS_CONTRACT.md`
has the review; `scripts/record_token_observations.py` is how to record more.

### Changed — CONSENSUS: the success contract (2026-09-21, PR #32)

Recorded here on 2026-11-02; it shipped without a changelog entry, which
CONTRIBUTING's own rule forbids for a consensus change, and it is the reason
every pre-2026-09-21 head, state file, reveal and proof is rejected.

The benchmark now routes with core's shared success contract
(`fugal_subnet/vendor/success_contract.py`, vendored from a pinned core
commit and proven against it in CI): independent per-model sigmoid success
predictions minus λ × estimated dollars per query at a fixed benchmark λ of
1, rounded on the 1e-4 grid, ties in artifact row order. The embedding profile
is pinned — Qwen3-0.6B at one tokenizer revision, the exact core system prompt,
2048 total input tokens, right-padded batches of two, mask-mean pooling,
L2-normalised, **CPU float32 only** (CUDA and float16 are refused). Heads must
carry the contract metadata (contract and profile ids, backbone revision,
provenance, default λ, per-model measured input/output token means, the token
manifest identity); legacy preference heads cannot enter the benchmark.

A protocol identity (`fugal_subnet/routing_protocol.py`) binds the contract,
profile, λ, token manifest and price snapshot. It is written into every proof,
state file, frame snapshot and reveal, and `require_identity` rejects anything
carrying a different one — so old state and reveals cannot be resumed or
replayed, and state paths moved to `results/success-v1/`. The token manifest
`data/benchmark_tokens_v1.json` is a **review candidate** built from 945
recorded calls; **a live validator or miner refuses to start until it is
promoted to `reviewed`**, and promotion changes the identity again, which
means a rebuilt measured image and a new approved entry. The transition and
rollback runbook is `docs/SUCCESS_CONTRACT.md`. Rewards, graders, exploration
and the number of paid calls per benchmark were unchanged.

### Changed — CONSENSUS: the 2026-10-14 diligence batch (PRs #68–#82)

Fourteen pull requests from an external review, each re-scoring or re-gating
miners; `docs/INVARIANTS.md` has the analysis per finding.

- **Frontier valued at the raw sample mean, not the prior** (finding 3). The
  neutral prior let a constant policy collect the gap between a model's true
  accuracy and its shrunk estimate. `FRONTIER_MIN_TRIALS` is now 300
  *distinct* observations per hull model, and a live validator starts from a
  hash-pinned bootstrap frame `data/reference_frame.json` (file pending).
- **One observation per (question, model) per epoch; one vote per coldkey;
  failed calls attested as failed and excluded** (finding 2). A proof whose
  exploration fails more than `EXPLORE_FAIL_MARGIN` above its scored calls is
  rejected; exploration is interleaved with scored calls.
- **Two-sided, baseline-free exploration detector** (finding 4); a flagged
  operator's samples are kept out of the frame for the epoch, never scored.
- **Dedup gains a parameter clause** (`DEDUP_PARAM_DISTANCE`) **and the
  epsilon rule** (`DEDUP_EPSILON`, `DEDUP_EPSILON_AGREEMENT`); seniority is
  registration block, then commitment block, read from chain at the boundary;
  the per-validator `first_commit_blocks` map is gone (findings 5, A4).
- **Live miners answer only validator permits or stake ≥
  `MIN_VALIDATOR_STAKE`** (finding 5); a live miner refuses to start with the
  threshold at 0.
- **Slice shares capped at 1.5× pool share** (`SLICE_SHARE_FACTOR`, finding 6).
- **Bare base approved entries refused outside mock**; the app half is the
  compose hash replayed from RTMR3 (finding 1). `check_upstream_is_inside_the_
  measurement` fails CI on a compose that allows an env, names an image by
  tag, or sets the upstream.
- **Every scoring constant is in the consensus digest; a live validator
  refuses to start with any of them set in its environment** (A2).
- **Validator is live by default; both neurons refuse `--mock` on finney**
  (A3; the shipped genesis hash was corrected in #87).
- **Burn UID resolved from the subnet owner on chain** at the boundary block
  (A5), not hard-coded 0.
- **Reveals published** to a public-read bucket with the hash committed on
  chain (A1).
- **Pool materialised once** by `scripts/export_pool.py` and preferred by
  `load_all()` (finding 8); **miner image published by digest and approved
  entries as signed releases** (finding 7; `deploy/keys/allowed_signers`
  pending).
### Fixed — CONSENSUS: the TPM half of a dstack attestation must be bound to its TDX quote

On GCP a proof carries two signatures: Intel's over the TDX quote and
Google's over a vTPM quote. `unwrap_attestation` verified the TPM quote's
signature and certificate chain but never checked that it was *this* TDX
quote's TPM quote. dstack puts `sha256(tdx_quote)` in `TPMS_ATTEST.extra_data`
(measured on both real fixtures), so a genuine TPM quote from any Confidential
VM could have been stapled onto any TDX quote and the second trust root proved
only that some GCP vTPM existed. The binding is now checked and a mismatch is
a rejection (I8). Every honest dstack blob already satisfies it; a validator
without this change accepts envelopes one with it rejects, so all validators
take it together. `tests/test_tpm.py` holds both the fact and the refusal.
### Fixed — an epoch is applied once, or not at all

`update_scores` mutated the validator's records in place and the reference
frame was rebound as soon as it was computed, before the epoch was durable.
Any exception between scoring and `save_state` — a dropped websocket in
`set_weights`, a failed owner-hotkey read, a disk error on the reveal — left
the loop with the state already advanced, and the retry of the same epoch
accumulated every miner's evidence twice, applied every miss twice and decayed
the frame twice, then persisted it (I1, I3). `update_scores` is now
copy-on-write and the loop holds the epoch's state in epoch-local names until
the commit point just before `save_state`. The on-chain commitments are read
at the boundary block the moment the epoch starts, not after the collection
wait and the 120 s query, and a failed read in live raises (the outer loop
retries) instead of degrading to an empty map that rejected every proof and
marked the epoch processed with no weights (I6). `tests/test_epoch_atomicity.py`
holds the shape of both fixes and was checked to fail against the previous
loop.
### Changed — `provision_subnet.py` no longer needs the owner coldkey to check or register

`--status` loads no wallet, so the read-only activation and tempo check from
MAINNET_LAUNCH §1-2 runs on a validator host, which holds a hotkey and never
the owner coldkey. `--owner-wallet` is optional: without it the owner steps
(identity, `start_call`, hyperparameters) are skipped and the script refuses
to register or stake into a subnet that is inactive or whose tempo does not
equal the epoch length, the two failures that have each silently cost epochs
before. `--create` still needs the owner. MAINNET_LAUNCH shows the owner half
and the operator half. `tests/test_provision_subnet_cli.py` drives `main()`
against a stubbed chain with wallet loading replaced by an assertion.
### Fixed — `scripts/export_pool.py` could not run

The documented way of building the consensus pool for every neuron
(MAINNET_LAUNCH §3) failed on every Linux host and nothing ran it. It set
`FUGAL_POOL_FILE=os.devnull` to mean "no file"; the loader tested the path
with `os.path.exists`, true for `/dev/null`, and fed a device node to the JSON
parser. Behind that, `gzip.open(..., mtime=0)` raised `TypeError` (only
`GzipFile` takes `mtime`), and the gzip header carried the output file name,
so the "file sha256" operators are told to compare differed by path alone.
The loader now ignores anything that is not a regular file; the script uses a
sentinel that can never be one and writes a header free of time and name.
Verified against the pinned revisions: 21,553 questions, `pool_hash` and
`content_hash` equal to the manifest, and the file reloads as the pinned pool.
Three tests in `tests/test_pool_file.py`, one of which runs the script's
`main()`.

### Changed — CONSENSUS: the score is headroom above the constant-policy frontier

Re-scores every miner; decided in `docs/I3_DECISION.md` (option 3). The
reference is no longer the best single model but the upper convex hull of what
every single model, and every random mixture of models, achieves at each cost
per question — built from the reference frame's nonce-assigned exploration
samples and the pinned price table (`fugal_subnet/frontier.py`). The score is
`max(0, wilson_lcb(accuracy) − frontier(cost per question)) × burn_in ×
frontier_confidence`, zero below `SCORE_QUALITY_FLOOR` (0.8) of the frontier's
best accuracy. Every policy that does not read the question scores zero by
construction; a router earns the accuracy it adds. Measured motivation: under
the old score a W=0 head that always called one mid-priced model beat the best
trained router by 12%. While the frontier is cold (`FRONTIER_MIN_TRIALS`
decayed trials per hull model) scores scale down and unassigned weight burns to
UID 0 rather than paying constant policies for the frame's ignorance.
*(Superseded 2026-10-14: the threshold is 300 distinct observations and the
burn goes to the subnet owner's UID read from chain — see the diligence entry
above.)*
`SCORE_QUALITY_EXPONENT`, `SCORE_QUALITY_CAP` and `SCORE_THRIFT_CAP` are gone;
`Evidence` gains `n_priced` so a skipped epoch cannot make a miner look cheaper.
New tests: `tests/test_frontier.py`; `tests/test_degenerate_constant_policy.py`
rewritten; `tests/test_scoring_tradeoff.py` removed with the exponent it pinned.

Two defects found on the first live frontier epoch (testnet 552, e00022118)
and fixed the same day: the confidence factor is common to every miner and so
cancelled when scores were normalised — nothing burned; `compute_weights` now
takes `paid_fraction` (the frontier's confidence) and burns the rest after
normalising. And evidence migrated from a pre-`n_priced` state file divided a
whole history of cost by one epoch of questions (4.4x the real cost per
question); `Evidence.priced_n` reads a zero `n_priced` as `n_total` on every
path. Both change the weight vector; both validators moved together.

### Fixed — the operator's live-proof check refused every live proof

`scripts/verify_live_miner.py` never passed `expected_hotkey` to
`verify_proof`, which the verifier requires outside mock mode (a proof not
bound to a miner can be relayed by anyone). It fetched and saved a genuine
315-result attested proof and then refused to verify it. Found on the first
real proof, 2026-09-07. The validator's own call was correct.

### Changed — the miner's benchmark calls models concurrently

Measured 2026-09-07 on the first live epoch on real TDX: 315 serial calls to
the three cheapest models with 2048-token completions were still running 49
minutes after the boundary, against a collection point 36 minutes in — real
replies on math prompts take ~10 s, not the 6 s the config note assumed, and
one 180 s timeout alone is 5% of the window. `HARNESS_CONCURRENCY` (default 8,
miner-side) keeps that many calls in flight. What it must not change is
pinned by `tests/test_harness_concurrency.py`: results stay in slice order,
each question is billed exactly its own calls (the proxy attributes records by
an `X-Fugal-Request` id, not by list position), a stalled call costs one
worker rather than the epoch, and cost totals use `math.fsum` so the attested
figures are identical whatever order the records landed in. The metering
proxy is now a threading server; grading stays on the main thread. Also: the
miner drops the SDK's `UnknownSynapseError` log spam from internet scanners
(114 probes in the first minute on a public axon).

### Fixed — one non-text model reply aborted the miner's whole epoch

Found 2026-09-07 on the first live epoch of the first real TDX miners, both of
which failed it. A chat completion carried `"content": null`; `_call_model`
returned `None`; `run_benchmark` called `.encode()` on it and raised, so no
proof existed for the epoch — for one reply out of three hundred. Every test
stub returns a string, so the path had never run. Replies that are not text
(null, missing, a parts list, an error body) now grade as a wrong answer with
the hash of the empty string, and the epoch completes.
`tests/test_harness_non_text_reply.py` drives the null reply through the real
HTTP path and through `run_benchmark`.

### Added — the validator says which entries it approved

`Approved entries: N — <base>…:<compose>…; …` at startup, before any chain
connection, and a malformed entry (a bare trailing colon would silently accept
any application) fails there instead of at the first proof. During a rotation
two validators must carry the same two entries; until this line the only
evidence of what a validator had loaded was its env file on its own host.

### Changed — the backbone thread count is a miner-side knob

`FUGAL_BACKBONE_THREADS` (default 1, unchanged; `0` = all cores) sets torch's
intra-op threads for the backbone. The single-thread pin cost a 4-vCPU TD
about 13 hours of startup and bought no consensus property: validators never
run the backbone, `check_determinism.py --perturb` already proves the scoring
path is thread-independent, and embeddings shape only the miner's own routing
choices. numpy's OpenBLAS stays at one thread. The dstack compose sets `0`.

### Fixed — the attested channel was authenticated, not confidential

The pusher verified an Intel-signed quote over its nonce and then POSTed the
head, the hotkey keyfile and the OpenRouter key over plain HTTP. Found the
same day as the first real pushes, by reading `push()` and `serve()` together.
The TD now binds an ephemeral X25519 public key into the attested
`report_data` beside the nonce; the pusher encrypts to it (HKDF, ChaCha20-
Poly1305, nonce as associated data); the TD refuses plaintext, refuses nonces
it never attested, and consumes each once. The same session key gives the
provisioning operator — and nobody else — an encrypted pull of the miner's
recent log lines (`provision_td.py --logs`), so `public_logs=false` no longer
makes a miner a black box to its own operator. INVARIANTS § "I8 — the
attested channel was authenticated, not confidential".

### Fixed — found preparing the first `--live` run on netuid 552

- **A dstack TD had no permitted way to receive its hotkey.** The miner signs
  `serve_axon` and the head commitment itself, so the keyfile must be inside
  the TD, and every route in had been ruled out: the compose is public, the
  shared disk is plaintext to the cloud provider, and the attested channel
  allowed only the head, the public hotkey address and the API key. The
  channel now carries `hotkey_keyfile_b64` and `coldkeypub_b64` (the SDK reads
  the coldkey address to serve); the TD holds them in tmpfs. Operator side:
  `scripts/provision_td.py`, which never prints a secret and refuses an
  encrypted hotkey. INVARIANTS § "I8 — the wallet crosses the attested channel".
- **`FUGAL_BENCHMARK_POOL` bypassed both pool guards.** The override returned
  before the manifest check and the size floor, so the documented production
  path was the unguarded one; a 150-question pool served 40+ live epochs
  unremarked. The override is now checked like the loader path, and a
  deliberately non-pinned pool must say so with `FUGAL_POOL_UNPINNED=1`.

### Removed — the pre-TEE pipeline's residue

Found by reference scan and executed removal (gate green after each), not by
reading: `fugal_subnet/matrix.py` (validators no longer build a matrix by
calling models), `fugal_subnet/tee/confine.py` (nothing imported it),
`scripts/launch_testnet.py` and `scripts/test_real_api.py` (both drove the
validator-calls-models design and advertised an `--epoch-budget` the validator
does not have; `scripts/dress_rehearsal.py` is the local-chain path), the empty
`fugal_subnet/v2` and `fugal_subnet/sandbox` packages, the integration test of
the removed pipeline, and nine `config.py` constants nothing read
(`NETWORK`, `NETUID`, `LIVENESS_MAX_MISSED`, `TEE_PROOF_TIMEOUT`,
`API_CONCURRENCY`, `CACHE_STALENESS_TTL`, `EXEC_TIMEOUT`, `EXEC_MAX_BYTES`,
`MAX_MODEL_COST_PER_QUERY`). `docs/VALIDATOR_GUIDE.md` no longer documents a
timeout that did nothing.

### Docs — corrected against the code and against measurement

README described validators building ground-truth matrices and a paid
`--epoch-budget` mode; TDX_VALIDATION gave the measurement formula with RTMR0;
VALIDATOR_GUIDE said DCAP needs Intel's PCS and described the approved list
without its `base:compose_hash` form; SECURITY listed a `0.1.x` support line
and code execution on the validator host; CONTRIBUTING required "an explicit
positive budget" for paid runs. Each is now what the code does. Every new
number in the guides (cache-hit verify time, base measurement) is measured and
says where.

### Added

- `deploy/systemd/`, `deploy/env/`, `deploy/dstack/` — the production shapes of
  a validator service (raised fd limit, restart, env file) and of the dstack
  miner compose (image by digest, persistent embedding cache, provisioning
  port), with the recipe that produces an approved `<base>:<compose_hash>`.
- `scripts/make_rehearsal_heads.py` — distinct, balanced heads over the three
  cheapest pinned models, for rehearsals that must spend cents rather than
  dollars. Synthetic-trained heads collapse onto one model on real embeddings
  and would be deduplicated as copies.

## [0.2.0] - 2026-09-05

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
*(Superseded 2026-10-14: seniority is the registration block, then the
commitment block, both read from chain at the boundary; no per-validator map.)*
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

[Unreleased]: https://github.com/fugal-ai/fugal-subnet/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/fugal-ai/fugal-subnet/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/fugal-ai/fugal-subnet/releases/tag/v0.1.0
