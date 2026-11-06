# Mainnet Launch

The switch from testnet to mainnet is one flag: `--network finney`. Everything
in this document exists because something *else* also has to be true, and each
item is here because it was found the hard way rather than imagined.

Read [INVARIANTS.md](INVARIANTS.md) first. This is the operational sequence, not
the design.

---

## The subnet: finney netuid 70

Registered at block 9364303 (2026-10-26). Read from the chain at block 9422441
(2026-11-03 02:27 UTC): `start_call` **not yet called**; tempo 360, so every
neuron runs `FUGAL_EPOCH_INTERVAL=4320`; commit-reveal enabled with a one-epoch
reveal period; registration open; owner cut 18%; subnet identity unset. Network
immunity from deregistration ends at block 9796303 (about 2026-12-25); after
that the chain prunes the non-immune subnet with the lowest *moving* alpha
price whenever a new subnet registers, and the moving price stays at 0 until
`start_call`. Every value here moves — re-read it with `--status` before acting.
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
content hash. Leave `FUGAL_BENCHMARK_POOL` unset on every neuron; a pool file is
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
end to end against a live TD (see [OPEN_WORK.md](OPEN_WORK.md) for what the
rehearsal did and did not demonstrate). Approved entries are
`<base>:<compose_hash>` pairs; the procedure for publishing and rotating them
is [APPROVED_ENTRY_ROTATION.md](APPROVED_ENTRY_ROTATION.md). **No entry exists
yet for a runtime a current validator would accept**: the four recorded there
are 2026-09-06/07 testnet images from before the success-contract transition.
The first `image-v*` release creates the first. With an entry
published, `--live` is the only honest mode; `--mock` accepts unattested
proofs and is for local rehearsals. See [TDX_VALIDATION.md](TDX_VALIDATION.md).

### 6. Miner startup cost

Backbone embeddings for the pool (21,553 questions since the code benchmarks
were excluded) took **13.2 hours** on an x86_64 laptop and **36 hours** on a
4-core aarch64 VM **under the legacy embedding profile**; the 2048-token
profile shipped since 2026-09-21 is heavier per question and has not been
timed on a TD (MINER_GUIDE, "The first boot embeds the whole pool"). Both measured on real
pool prompts, which average 330 characters: 0.46 q/s and 0.167 q/s
respectively. This is startup cost before the miner answers its first query.

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
# 1a. Owner half (whoever holds the owner coldkey): identity, start_call,
#     hyperparameters. Idempotent — safe to re-run after any partial failure.
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

Honest list, as of this branch:

| | Status |
|---|---|
| Scoring path determinism across x86_64 and aarch64 | **verified** byte-identical (local, and on netuid 552 with real proofs: three consecutive scored epochs, identical weights on both hosts) |
| Full pipeline against a real chain, multi-miner, multi-validator | **verified** — local chain, and public testnet with two dstack TDX miners and two validators (`docs/REHEARSAL_2026-09-06.md`) |
| Real DCAP verification and measurement matching | **verified** on real Intel TDX (GCP `c3`, dstack 0.6.0-rc0): proofs verified `UpToDate`, a genuine proof refused against a retired entry |
| A reproducible TDX guest image with a stable measurement | **exists**: dstack 0.6.0-rc0 UKI on GCP, base `12a1f2f5…9141` in September across seven deploys and two instance sizes, `f4be9383…aa427` on 2026-11-03 after a GCP firmware change (image registers identical); the compose hash rotates with the image and the rotation procedure is `docs/APPROVED_ENTRY_ROTATION.md` |
| Real OpenRouter billing vs the pinned price table | **measured once**, account-level: within ~5% over four miner-epochs; completions averaged 534 tokens against the 256 assumed, and exploration was ~60% of a cheap head's epoch cost |
| A real trained head outscoring a random one | **measured** offline (`docs/HEAD_EFFICACY.md`); on chain only two synthetic heads have been scored |
| Behaviour under adversarial miners over weeks | **never executed** |
| Miner startup cost on a TD | **measured on the legacy profile**: 7 h 39 m to embed the pool at four threads on a `c3-standard-4`; re-paid on every image rotation because the tool recreates the data disk. **Unmeasured** for the 2048-token profile |
| An end-to-end epoch under the success-contract runtime (protocol since 2026-09-21) | **never executed** on any chain; the runbook is `SUCCESS_CONTRACT.md` |
