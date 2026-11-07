# Mainnet Launch

The switch from testnet to mainnet is one flag: `--network finney`. Everything
in this document exists because something *else* also has to be true, and each
item is here because it was found the hard way rather than imagined.

Read [INVARIANTS.md](INVARIANTS.md) first. This is the operational sequence, not
the design.

---

## The subnet: finney netuid 70

Registered at block 9364303 (2026-10-26). **Live under protocol v1.** Read
from the chain at block 9449357 (2026-11-06 15:13 UTC): `active` True; tempo
360, so every neuron runs `FUGAL_EPOCH_INTERVAL=4320` (ALIGNED); commit-reveal
enabled with a one-epoch reveal period; registration open; 256 UIDs, 4
validators with permits, 194 routable miner axons, 144 UIDs with non-zero
incentive. The first scored mainnet epochs ran on 2026-11-04 (e00026193 and
following; `MINER_GUIDE.md`, "A failed model call scores as a wrong answer").
Network immunity from deregistration ends at block 9796303 (about 2026-12-25);
after that the chain prunes the non-immune subnet with the lowest *moving*
alpha price whenever a new subnet registers. Every value here moves — re-read
it with `--status` before acting.
The deploy files already name this target: `deploy/dstack/docker-compose.yaml`
(the testnet twin for rehearsals is `docker-compose.test.yaml`, which also
sets the warm-up's `FUGAL_EXPLORE_FRACTION=0.5`) and the `validator.env.example`
comment.

### The protocol label stays `fugal-testnet-success-v1` on mainnet

`routing_protocol.identity()` hashes the literal string
`"fugal-testnet-success-v1"` together with the success contract, the token
manifest and `data/models.json`. That digest names every reveal, and the
bootstrap frame is built from reveals produced under it on testnet. Renaming
the label to say "mainnet", or refreshing the price table between the warm-up
and the launch, is a new identity that rejects the frame. The word "testnet"
in it is a historical name, not a network setting; the network is `--network`
and `FUGAL_NETWORK`.

Protocol v2 has its own identity, `routing_protocol.identity_v2()`: the label
`fugal-success-v2` with the v2 token manifest and `data/models_v2.json`. A v2
reveal or frame cannot be mistaken for a v1 one, and the v1 identity is
untouched by the flip.

---

## Before you spend anything

### 0. The gates a neuron refuses to start behind

Each of these is a startup error, not a silent failure, and each is listed
here because an operator who meets the six items below and still cannot start
will otherwise go looking in the wrong place:

| Gate | Where it is enforced | What clears it |
|---|---|---|
| The token manifest shipped in the checkout is `reviewed`, not `candidate` | `neurons/validator.py`, `neurons/miner.py` (`--live` only) | **Cleared by #99 (2026-11-02)**: rebuilt from 664 fresh metered calls and promoted; the record is in `SUCCESS_CONTRACT.md`. Any later change to it is a protocol-identity change (image and approved entry rebuilt, evidence reset) |
| `data/reference_frame.json` exists and matches its pin | `reference_frame.load_bootstrap(require=True)`, `check_safety_invariants.py` | **Shipped and pinned 2026-11-03**, built by `scripts/build_frame_from_calls.py` from 17 × 300 direct, logged, graded calls under the current protocol identity (`data/observations/bootstrap_frame_calls_v1.jsonl`); `scripts/build_bootstrap_frame.py` rebuilds one from published reveals |
| Both live `deploy/dstack/docker-compose*.yaml` are, byte for byte, the compose a frozen `app-compose*.json` embeds | `scripts/check_safety_invariants.py --release` (plain CI only warns) | Freeze the compose for the released digest (`dstack-cloud prepare`, commit the app-compose.json) and publish its entry; the current composes drifted after the 2026-11-03c freeze (`FUGAL_POOL_CACHE_V2`) |
| `FUGAL_TEE_MEASUREMENTS` holds at least one `<base>:<compose_hash>` entry | validator startup | A signed `image-v*` release, verified by `scripts/fetch_approved_entries.py` against `deploy/keys/allowed_signers` (first key added 2026-11-03) |
| No consensus constant is set in the environment on finney | `fingerprint.assert_consensus_overrides_allowed`, by genesis hash, in both neurons | Unset it; only `FUGAL_EPOCH_INTERVAL` and `FUGAL_TEE_MEASUREMENTS` are meant to be set per deployment. On a test or local chain an override starts with a warning, which is how the netuid-552 warm-up runs `FUGAL_EXPLORE_FRACTION=0.5` on every neuron |
| `--mock` is not finney | `fingerprint.assert_mock_allowed` | Run `--live` |

Two more that are not startup errors but are just as final: a validator with
less than `FUGAL_MIN_VALIDATOR_STAKE` (50, in the chain's stake units for the
UID; a permit alone does not count) is refused by every live miner's axon and
sees "No valid proofs" forever, and is warned about it at startup; and
`FUGAL_REVEAL_BUCKET` unset only warns, so the subnet's product is never
published.

### 1. The subnet must be activated, and nothing will tell you if it isn't

`start_call` is the single most consequential extrinsic in a launch, and its
absence is invisible. Without it the subnet has no first-emission block:
staking is rejected with `SubtokenDisabled` and the subnet earns nothing.
Validators whose stake comes from root (their own or childkeyed to them,
counted at 0.18) do hold permits and do land weights before `start_call`
(measured on netuid 70 at block 9422854: three root-staked validators with
weights set and zero emission). A validator that depends on alpha staked to it
holds nothing, earns no permit, and every `set_weights` fails. Either way the
neurons run, log success, and nothing is paid.

Fugal's own testnet subnet (netuid 552) sat in exactly this state from May to
September. Its miner served an axon, its validator ran epochs, and not one
weight ever reached the chain.

```bash
python scripts/provision_subnet.py --network finney --netuid 70 --status
```

`active` must be `True`. If it is not, nothing below matters. `--status` loads
no wallet, so it runs on a validator host that holds only a hotkey — which is
every validator host, since the coldkey never goes on one (below).

### 2. Epoch length must equal the subnet tempo

Fugal's epoch is `FUGAL_EPOCH_INTERVAL // 12` blocks. The chain runs Yuma every
`tempo` blocks. When they differ, some epochs set weights that a later epoch
overwrites before any Yuma step reads them — that epoch's work never reaches
emission at all.

`tempo` is bounded by the chain (360 is the default and values as low as 120 are
refused with `TempoOutOfBounds`), and `weights_set_rate_limit` is root-only, so
in practice the alignment is achieved from Fugal's side:

| chain tempo | set on every neuron |
|---|---|
| 360 blocks (default) | `FUGAL_EPOCH_INTERVAL=4320` |
| any other T | `FUGAL_EPOCH_INTERVAL = T * 12` |

`provision_subnet.py --status` prints `ALIGNED` or `*** MISALIGNED ***`. The
epoch must also be at least `weights_rate_limit` blocks (100 by default, a
20-minute floor), or the chain refuses the extrinsic outright.

### 3. Every neuron must load the identical benchmark pool

The short version since 2026-10-14: build the pool once with
`python scripts/export_pool.py` on a machine that can reach Hugging Face,
copy `data/benchmarks/pool.json.gz` to every **validator** (or point
`FUGAL_POOL_FILE` at it), and leave `FUGAL_BENCHMARK_POOL` unset. A dstack
miner cannot take the file — `data/benchmarks/` is excluded from the image and
the pool is deliberately not on the provisioning allow-list — so a TD builds it
from the pinned revisions at first boot and needs Hugging Face egress once. The loader
prefers the file, verifies it against the pinned manifest at startup, and
never touches Hugging Face. Do not publish the file: the datasets carry their
own licences.

**Protocol v2 has its own pool, and a validator needs it before the flip.**
With `FUGAL_V2_ACTIVATION_BLOCK` set, the validator loads the v2 pool at
startup and refuses to start in `--live` if it cannot (`neurons/validator.py`).
So export it too (`python scripts/export_pool.py --protocol 2`) and place
`data/benchmarks/pool_v2.json.gz` (or point `FUGAL_POOL_FILE_V2` at it) on
every validator **before** the restart that sets the block. Without it, every
validator rebuilds 29,036 questions from Hugging Face on that restart, and an
upstream outage then is a validator that does not start.

The pool is consensus state: the slice is drawn from it, so a validator and a
miner with different pools disagree on every question and every proof fails on
`questions_hash` — an error that names the symptom and never the cause.

Two things make the default loader non-reproducible:

- **GPQA is a gated Hugging Face dataset.** `load_all()` raises without an
  authenticated token, so an operator without one cannot start at all — and an
  operator *with* one loads a larger pool than everyone else.
- **LiveCodeBench silently loads zero questions** on current `datasets`
  versions, but loads normally for anyone who has manually placed
  `data/benchmarks/livecode.json`. Two operators, two pools, no error.

Both are now closed by the pinned manifest (`fugal_subnet/benchmarks/
pool_manifest.json`): `load_all()` skips the gated and unpinned benchmarks by
name and refuses to start unless the pool it built matches the manifest's
content hash. The v2 pool (`pool_manifest_v2.json`, 29,036 questions from
open sets only, none gated) is loaded by `load_pool(2)` the same way; a dstack
miner caches it on its encrypted volume after the first verified load
(`FUGAL_POOL_CACHE_V2`) and re-verifies it on every read. Leave `FUGAL_BENCHMARK_POOL` unset on every neuron; a pool file is
accepted only if it verifies against the same manifest, and a deliberately
different pool must declare `FUGAL_POOL_UNPINNED=1` (local rehearsals only).

Confirm agreement before launch — every neuron logs `pool_hash` and "Pool
matches the pinned manifest" at startup. Measured 2026-09-06/07: an aarch64
validator, an x86_64 validator and two TDX miners all loaded
`1779b4a5218bb2ef…` from their own HuggingFace downloads.

### 4. Miners must be reachable, and the chain will not tell them otherwise

`serve_axon` succeeding means the chain recorded an address. It says nothing
about whether anyone can connect to it. A cloud VM whose provider firewall
blocks the port publishes a valid address, logs `Axon registered on chain`, and
is never contacted — earning nothing, indefinitely, with no error on either
side. Netuid 552's miner did this for months behind a closed Oracle security
list rule.

The miner now attempts to connect to its own published address and warns on
failure, but a host that does not route its public IP back to itself will fail
that check while being perfectly reachable. **Verify from a different machine:**

```bash
nc -vz <miner_public_ip> <port>
```

Open the port in the *cloud provider's* firewall, not just the guest's — they
are separate, and the guest one is the one that looks like it worked.

### 5. `--live` needs an approved entry, and here is how one is made

A validator started `--live` with no `FUGAL_TEE_MEASUREMENTS` now refuses to
start, because the alternative was worse: an empty approved set can never
contain any measurement, so it rejected every proof and set every miner to zero
weight, silently.

Getting a usable value is the real remaining work.
`measurement_id = sha256(MRTD ‖ RTMR1 ‖ RTMR2)` — see
`fugal_subnet/tee/attestation.py:measurement_id`.

**RTMR0 is deliberately excluded, so the machine shape does NOT affect the
measurement.** It records the TDVF configuration the host builds — vCPU count,
memory size, device config — which the cloud provider chooses and which proves
nothing about the code. Measured byte-identical across three real deploys
(`c3-standard-4` and `c3-standard-8`, two different applications) in September:

    12a1f2f56907f80576be553f3d71031ec86f2848877ac05c82db5d8627fc9141

and, on 2026-11-03, across two fresh deploys:

    f4be9383e31c6cc87707b2b01ace9114ed958435b524425677e63d65065aa427

The change is MRTD (GCP's TDX firmware), not the image: RTMR1 and RTMR2 are
identical on both dates. Publish the base you measure on the day, not the one
in a document.

Miners may therefore pick any instance size. An earlier version of this section
gave the formula as `sha256(MRTD‖RTMR0‖RTMR1‖RTMR2)` and concluded the subnet
would fork by instance size. That was wrong once RTMR0 was dropped, and it is
recorded here rather than silently deleted because it is the sort of claim
someone acts on immediately before spending real TAO.

What remains true: **RTMR1 and RTMR2 still move** — they cover the kernel,
cmdline and initrd — so a stock cloud image would need the approved list
republished on every kernel package update. **A fixed, reproducibly-built guest
image with a pinned kernel and initrd is still a prerequisite for `--live` on
mainnet.** dstack now supplies exactly that, and provisioning has been proven
end to end against a live TD (see [REHEARSAL_2026-09-06.md](decisions/REHEARSAL_2026-09-06.md) for what the
rehearsal did and did not demonstrate). Approved entries are
`<base>:<compose_hash>` pairs; the procedure for publishing and rotating them
is [APPROVED_ENTRY_ROTATION.md](APPROVED_ENTRY_ROTATION.md). The live entries
are those of the signed `image-v2026.11.03c` release (one per network; the
frozen app-compose files beside the composes in `deploy/dstack/` are their
bytes), carried by every mainnet validator since 2026-11-03. The v2 image gets
its own pair at the flip ("The v2 release, in order", below). With an entry
published, `--live` is the only honest mode; `--mock` accepts unattested
proofs and is for local rehearsals. See [TDX_VALIDATION.md](TDX_VALIDATION.md).

### 6. Miner startup cost

Backbone embeddings for the v1 pool (21,553 questions) under the current
2048-token profile took **3 h 20 min on a `c3-standard-8`** and about 5 to 6
hours on a `c3-standard-4` (measured on mainnet TDs, 2026-11; MINER_GUIDE,
"How to move"). The v2 pool is 29,036 questions, about 35% more. Under the
legacy profile the same pool took 13.2 hours on an x86_64 laptop and 36 hours
on a 4-core aarch64 VM, measured on real pool prompts averaging 330
characters. This is startup cost before the miner answers its first query.

Beware benchmarking this with synthetic prompts. Short ones give 2.5 q/s and
0.8 q/s — five times too optimistic — because cost scales with sequence length
and the pinned batch size pads every item in a batch to its longest member.

They are now cached to disk keyed by `pool_hash`, backbone, batch size and
machine, so it is paid once rather than on every restart — but it *is* paid
once, and the miner is unreachable for the duration. **Warm the cache before
registering, not after.**

Two consequences worth deciding on before launch, not after:

- **Any pool change costs every miner a day of compute.** The pool is consensus
  state, so it cannot be changed cheaply or often.
- **The barrier to entry is a day of CPU.** The embeddings are a pure function
  of pinned consensus inputs — the pool, the frozen backbone, the pinned batch
  size — so the subnet could publish them as a hash-pinned artifact and let
  miners verify rather than recompute. That is not built, and until it is, the
  practical entry cost for a miner is much higher than the ~14KB head suggests.

`fugal_subnet/determinism.py` used to pin the backbone to a single thread,
which cost roughly a factor of four on the figure above and bought nothing:
validators never run the backbone, and `scripts/check_determinism.py --perturb`
asserts the scoring path is identical with the pins removed. The torch thread
count is now `FUGAL_BACKBONE_THREADS` (default 1, unchanged; `0` = all cores),
and the dstack compose in `deploy/dstack/` sets `0`. Embeddings shape only the
miner's own routing choices, which it already controls through its head, so
this is a miner-side throughput knob and not consensus state. numpy's OpenBLAS
stays single-threaded.

The cache is shared by every miner on a host with the same pool, so three
miners on one machine pay it once between them.

---

## Launch sequence

The script has two halves. Identity, `start_call` and hyperparameters are
signed by the subnet **owner** coldkey; registration and staking are signed by
the wallets being registered. When the slot is held by a different party, or
by a multisig that is not on the machine running this, the owner runs their
half from wherever that key lives (`--owner-wallet <owner>`, no `--validators`
or `--miners`), and the operator runs the second half **without**
`--owner-wallet`: the script then checks that the subnet is active and
tempo-aligned and refuses to register into one that is not, because without
`start_call` staking fails, no permit is granted and every `set_weights` is
refused while the neurons log success.

```bash
# 1a. Owner half (whoever holds the owner coldkey): start_call and
#     hyperparameters. Idempotent — safe to re-run after any partial failure.
#     Without --yes it only prints what it would change. Tempo and identity
#     change only with --set-tempo / --set-identity: never on a live subnet
#     by accident.
python scripts/provision_subnet.py --network finney --netuid 70 \
  --owner-wallet <owner> --epoch-interval 4320 --yes

# 1b. Operator half (hotkey hosts; no owner coldkey anywhere near them):
#     register and stake. Refuses if the subnet is inactive or misaligned.
python scripts/provision_subnet.py --network finney --netuid 70 \
  --validators <w>/val1,<w>/val2 --miners <w>/m1 \
  --stake <TAO> --epoch-interval 4320 --yes

# 2. Confirm before starting neurons. No wallet needed.
python scripts/provision_subnet.py --network finney --netuid 70 --status

# 3. Start neurons with the aligned interval and the pinned pool file in
#    place (built once with scripts/export_pool.py; FUGAL_BENCHMARK_POOL stays
#    unset — see §3 above).
export FUGAL_EPOCH_INTERVAL=4320
export FUGAL_POOL_FILE=/opt/fugal/pool.json.gz

# 4. Watch, and assert rather than read.
python scripts/watch_subnet.py --network finney --netuid 70 --epochs 3 \
  --epoch-interval 4320 \
  --miners m1=<hotkey> --validators v1=<hotkey>,v2=<hotkey> \
  --reveals v1=/opt/v1/results/success-v1/epochs,v2=/opt/v2/results/success-v1/epochs
```

`watch_subnet.py` must report zero failures across at least two consecutive
epochs before the launch is real. It checks the things that have each silently
failed before: the subnet is active, axons are routable, commitments land at or
before the boundary block, validators hold permits, weights actually reach the
chain (read back commit-reveal-aware), and independent validators publish
identical weight vectors.

---

## The v2 release, in order

The protocol v2 flip is three things that must agree: the code on every
validator, one activation block, and one approved miner image whose compose
carries that block. In order:

Before the image (each changes what the image measures or carries):

1. **Re-measure the v2 token means on the pinned v2 pool** [PAID ~$5–12, owner
   approval]. The shipped means were measured on a stand-in sample, not the
   pinned v2 pool. `scripts/record_token_observations.py --live --protocol 2`,
   then `scripts/build_token_manifest.py`, reviewed and promoted into both
   copies (`LIVE_API_VALIDATION.md`). It goes first because the token manifest
   is hashed into the protocol identity. If a constant or a cost moves,
   re-check every doc that quotes the v2 numbers.
2. **Embed Google's TPM intermediate certificates in the v2 miner image.**
   Without them a Google certificate rotation, or a region other than
   us-central1, fails every proof.
3. **Add a second release signing key** on a separate device, and a
   key-rotation paragraph in `deploy/keys/README.md`. One key on one machine
   is one lost laptop from no signed release.

The release:

4. Merge `release/v2` into `main`; cut the CHANGELOG `[Unreleased]` block into
   `0.3.0` and bump `pyproject.toml`.
5. Build and sign the v2 image **from `main`** (`image-v*` release). The draft
   `image-v2026.11.05b` targets `main` but is tagged on `release/v2`; do not
   publish it.
6. **Rehearse the flip on testnet (netuid 552) first** [PAID: GCP TDX VMs].
   Steps 7–13 with the test compose and a testnet block a few hours out; two
   or more validators and two or three miners through the flip;
   `scripts/watch_subnet.py` must show v1 → v2, a first seat, and identical
   reveals. It is also the first real run of the TD tooling, so run each:
   `scripts/provision_td.py --inspect` (the base and compose hash the release
   names) and a push; `scripts/fetch_approved_entries.py --network test`;
   `scripts/verify_live_miner.py` against a provisioned miner (PASS) and with
   the old entry and `--expect-reject` (refused); and
   `scripts/tdx_measurement.py --verify` on a plain TDX VM. v2 has only run on
   a local chain. Choose the finney block only after this passes.
   Testnet's validator still carries the September base (`12a1f2f5…`); give it
   the test entry from the signed v2 release first.
7. Choose `<ACTIVATION_BLOCK>`: an epoch boundary (a multiple of 360 on
   finney) at least 24 h ahead of the announcement, so miners can re-provision.
8. Edit both `deploy/dstack/docker-compose*.yaml`: the v2 image digest (from
   step 5) and `FUGAL_V2_ACTIVATION_BLOCK=<ACTIVATION_BLOCK>`; drop the
   testnet twin's `FUGAL_EXPLORE_FRACTION=0.5` (v2 pays nothing from the
   frame). Uncomment `FUGAL_V2_ACTIVATION_BLOCK` in
   `deploy/env/validator.env.example` with the real value.
9. `dstack-cloud prepare` each; commit the two files as
   `app-compose.frozen-<tag>-{finney,test}.json`;
   `scripts/check_safety_invariants.py --release` must pass.
10. Measure the base on a fresh TD booted from the v2 image (it moved once when
    GCP changed firmware); publish the two entries with
    `scripts/publish_approved_entry.py` in a signed release.
11. Every validator: place the v2 pool file (`export_pool.py --protocol 2`,
    §3), upgrade to the release, set `FUGAL_V2_ACTIVATION_BLOCK` and add the
    v2 entry beside the current one, restart, before the block.
12. Every miner: re-provision on the v2 image before the block (the v2 miner
    holds its head until the last v1 boundary, then commits it).
13. After the block: `scripts/watch_subnet.py` for a few epochs; two
    validators' reveals must agree byte for byte and carry `protocol_version: 2`.

---

## After the flip: remove protocol v1

Protocol v1 stays in the tree until no v1 epoch can occur, because one binary
scores both rule sets and validators upgrade before the activation block.
Once the flip has held for a full comparison window (48 epochs), remove v1 in
one reviewed PR. Each item below changes a hash pin or a consensus path, so the
PR updates `scripts/check_safety_invariants.py` and `docs/INVARIANTS.md` with
it. Kept here, next to the launch sequence, so it is not forgotten.

- `data/models.json`, `data/benchmark_tokens_v1.json`, `data/reference_frame.json`,
  their copies under `fugal_subnet/routing_data/`, and their pins
- `data/observations/benchmark_tokens_v1.jsonl`, `benchmark_tokens_v2.jsonl`
  (v1 data despite the name; rename anything kept) and `bootstrap_frame_calls_v1.jsonl`
- `fugal_subnet/benchmarks/pool_manifest.json` and the v1-only loaders
  (`gsm8k`, `humaneval`, `livecode`, `gpqa`, the v1 `math`/`mmlu`/`aime`/`ifeval` sets)
- the v1 scoring modules: `reference_frame.py`, `frontier.py`, `scoring.py`,
  `dedup.py`, `head_distance.py`, the v1 halves of `evidence.py` and `rewards.py`
- the v1 branch of `neurons/validator.py` and the hold-until-flip logic of
  `neurons/miner.py`
- `slicer.protocol_version_for_boundary` and `FUGAL_V2_ACTIVATION_BLOCK`,
  once every validator runs v2 unconditionally
- the v1 sections of `MINER_GUIDE.md`, `VALIDATOR_GUIDE.md`, `INVARIANTS.md`
  and the README's "v1 until the activation block" sentences
- `scripts/build_frame_from_calls.py` and `scripts/build_bootstrap_frame.py`:
  the reference frame is v1-only (v2 scores on shared grades, never a frame)
- the `--protocol 1` defaults of `scripts/export_pool.py`,
  `scripts/build_pool_manifest.py` and `scripts/record_token_observations.py`:
  default to 2 at the flip, and drop the flag with v1
- the v1 rehearsal scenarios a–e in `scripts/dress_rehearsal.py`: they are the
  only chain tests of a steady-state network (two validators agreeing, many
  epochs, the real backbone), so replace them with v2 equivalents (seat,
  payout, copy rule, two-validator agreement) first, then remove the v1 ones
- `scripts/rehearsal_v2.py` and `dress_rehearsal.py --scenario v2_flip` test
  the crossing: keep their seat, payout and miss checks as plain v2 tests,
  drop the flip timeline and the `probe_features` skip machinery
- the pre-v2 frozen files `app-compose.frozen-2026-11-03c-{finney,test}.json` (03b removed 2026-11-06, retired by its published date)
  and their entries, once every validator has retired them
  (`docs/APPROVED_ENTRY_ROTATION.md`)
- release bookkeeping: the 0.3.0 CHANGELOG cut and `pyproject` version bump
  happen at the flip; the v1 removal is the release after it

---

## Wallet handling

Testnet keys in this repo's rehearsal were regenerated unencrypted so the run
could be automated. **Do not do that on mainnet.**

- The coldkey never goes on a neuron host. Only the hotkey and `coldkeypub.txt`
  are needed to serve an axon, commit a head hash, or set weights.
- Do registration, staking and provisioning from a machine that holds the
  coldkey, then ship hotkeys only.
- Keep the coldkey password-encrypted. `provision_subnet.py` will prompt once;
  the neurons never need it.

---

## What is still unverified

Honest list, as of 2026-11-06:

| | Status |
|---|---|
| Scoring path determinism across x86_64 and aarch64 | **verified** byte-identical (local, and on netuid 552 with real proofs: three consecutive scored epochs, identical weights on both hosts) |
| Full pipeline against a real chain, multi-miner, multi-validator | **verified** — local chain, and public testnet with two dstack TDX miners and two validators (`docs/decisions/REHEARSAL_2026-09-06.md`) |
| Real DCAP verification and measurement matching | **verified** on real Intel TDX (GCP `c3`, dstack 0.6.0-rc0): proofs verified `UpToDate`, a genuine proof refused against a retired entry |
| A reproducible TDX guest image with a stable measurement | **exists**: dstack 0.6.0-rc0 UKI on GCP, base `12a1f2f5…9141` in September across seven deploys and two instance sizes, `f4be9383…aa427` on 2026-11-03 after a GCP firmware change (image registers identical); the compose hash rotates with the image and the rotation procedure is `docs/APPROVED_ENTRY_ROTATION.md` |
| Real OpenRouter billing vs the pinned price table | **measured once**, account-level: within ~5% over four miner-epochs; completions averaged 534 tokens against the 256 assumed, and exploration was ~60% of a cheap head's epoch cost |
| A real trained head outscoring a random one | **measured** offline (`docs/decisions/HEAD_EFFICACY.md`) and on mainnet since 2026-11-04 (144 UIDs with non-zero incentive at block 9449357) |
| Behaviour under adversarial miners over weeks | **never executed** |
| Miner startup cost on a TD | **measured** on the current profile: 3 h 20 min (`c3-standard-8`) to 5–6 h (`c3-standard-4`) for the v1 pool; re-paid on every image rotation because the tool recreates the data disk; **unmeasured** for the v2 pool (29,036 questions) |
| An end-to-end epoch under the success-contract runtime | **running on mainnet** since 2026-11-04 (protocol v1) |
| A protocol v2 epoch on a public chain | **never executed**: rehearsed only on the local chain (`dress_rehearsal.py --scenario v2_flip`, 16/16 on 2026-11-05) |
