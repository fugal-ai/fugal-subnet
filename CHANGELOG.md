# Changelog

All notable changes to this project will be documented here. Releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
