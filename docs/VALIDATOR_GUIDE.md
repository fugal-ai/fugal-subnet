# Fugal Subnet — Validator Guide

Run a validator on the Fugal subnet. Validators verify TEE-attested proofs
from miners, compare every head on the same questions, and pay the champion
seat. **Validators never call models** — miners pay for their own inference
inside Intel TDX confidential VMs.

**Protocol v3 from block 9,527,760** (epoch 26466, ~2026-11-17 17:34 UTC),
release v0.5.0; the seat on odds from block 9,530,640 (~2026-11-18 03:10 UTC).
The blocks are compiled into v0.5.0: upgrade, fetch the embeddings file once
(`python -m fugal_subnet.embeddings_file fetch`), add the v0.5.0 image's
approved entry beside the current one, and restart before the block.

This guide describes **protocol v2**. Until the published activation block the
subnet scores by the v1 rules; the switch is the one procedure every validator
must perform, in "Protocol v2: flip procedure" below.

## What Validators Do

Epochs are aligned to chain blocks (`EPOCH_INTERVAL/12` blocks per epoch), so
every validator processes the same epoch on the same slice. Each epoch, your
validator:

1. Derives a nonce from the epoch **boundary block's** hash
2. Selects 300 questions by the published category quotas (40% knowledge,
   30% hard, 25% math, 5% instruction), then by benchmark share within each
3. Commits the question slice + grader hash (commit-reveal integrity)
4. Derives the epoch's exploration assignment from the same nonce
5. Reads every head commitment at the boundary block and the epoch's
   metagraph pinned to it
6. Queries all registered miners for TEE-attested benchmark proofs
7. Verifies each proof against every binding (see "TEE Proof Verification")
8. Builds the **shared grades**: one grade per (question, model) cell, the
   median across the coldkeys that routed there, so every head that chose the
   same model gets the same grade
9. Adds the epoch to the **comparison window** (the last 48 epochs) and runs
   the **seat decision**: hold, take, first, ghost or resume, by the paired
   net-value test against the champion (`docs/MINER_GUIDE.md`, "Scoring")
10. Pays the seat: 85% to the champion, 10% and 5% to the two before it, the
    rest to the burn UID; sets weights on-chain
11. Publishes the epoch reveal artifact (shared grades, the seat decision and
    its detail, every head's routing, the weights)

## Requirements

- Linux or WSL2
- Python 3.11-3.12
- CPU sufficient (no model inference required)
- TAO for subnet registration
- **At least `FUGAL_MIN_VALIDATOR_STAKE` (default 50) stake on the subnet,
  as the chain reports it for your UID: alpha on this subnet plus root TAO
  at the 0.18 root weight, after childkey inheritance.** A validator permit
  alone does not qualify. In `--live` a miner's axon answers only hotkeys at
  or above that floor; below it you receive no proofs and see "No valid
  proofs received" every epoch with no error on either side. The validator
  warns at startup when its own stake is below the floor
- Reliable server with good uptime

**No OpenRouter API key needed** — validators verify proofs, not call models.

## Setup

```bash
# Clone and install
git clone https://github.com/fugal-ai/fugal-subnet.git
cd fugal-subnet
python3 -m venv .venv
source .venv/bin/activate

# If cargo isn't installed (needed for bittensor build):
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# The DCAP verifier lives in the `tee` extra; --live refuses to start without it.
uv sync --locked --extra tee          # or: pip install -e ".[tee]"
```

## Step 1: Create a Wallet

```bash
btcli wallet create --wallet.name fugal_validator
btcli wallet create --wallet.name fugal_validator --wallet.hotkey default --type hotkey
```

## Step 2: Register on the Subnet

```bash
btcli subnet register \
  --netuid <NETUID> \
  --wallet.name fugal_validator \
  --wallet.hotkey default \
  --network finney
```

## Step 3: Run the Validator

```bash
python neurons/validator.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_validator \
  --hotkey default \
  --live
```

Before that command can start on mainnet, four things must be true, and the
validator refuses to start until each is (deliberately — the alternative was a
validator that scored the whole field zero while looking healthy):

1. `FUGAL_TEE_MEASUREMENTS` holds at least one `<base>:<compose_hash>` entry
   from a signed release (`docs/APPROVED_ENTRY_ROTATION.md`).
2. `FUGAL_EPOCH_INTERVAL` equals the chain tempo × 12 s (**4320** for the
   default tempo of 360). Misaligned, some epochs' weights are overwritten
   before Yuma reads them, silently (`docs/MAINNET_LAUNCH.md` §2).
3. `data/reference_frame.json`, the hash-pinned bootstrap reference frame, is
   present in the checkout (or `FUGAL_FRAME_BOOTSTRAP` names a directory of
   published reveals to rebuild from). Under v2 the frame is published for
   information only; the check stays until protocol v1 is removed.
4. The token manifest shipped in the checkout is `reviewed`, not `candidate`
   (`docs/SUCCESS_CONTRACT.md`).
5. The pinned embeddings file (protocol v3) is fetched by the validator itself
   at startup when it is missing or does not match `config.EMBEDDINGS_SHA256`;
   it is written in place only if it matches. To fetch it ahead of a restart:

   ```bash
   .venv/bin/python -m fugal_subnet.embeddings_file fetch   # -> data/embeddings/pool_v3.npz
   ```

   `FUGAL_EMBEDDINGS_FILE` points elsewhere if you keep it on another disk. In
   Docker, keep `data/embeddings` on a volume so it survives a container restart.

### Upgrading (every release)

```bash
git pull                       # or check out the release tag
uv sync --locked --extra tee
sudo systemctl restart fugal-validator   # or restart your container
```

Then check the startup log for one line:

```
READY: approved entries N (newest app-compose.frozen-<tag>-finney.json), v3 inputs ok (pool 48320, embeddings ok), archive ok, collateral ok, role follower (verified)
```

`NOT READY: …` names each missing item and the one command that fixes it.
Nothing else is manual: the approved entries come from the frozen composes
committed in the release (`fugal_subnet/validator_setup.py`), the embeddings
file is fetched and verified, and on finney the archive and fallback endpoints
default to OpenTensor's when unset.

### Minimum mainnet environment (finney, netuid 70)

> Everything a finney validator must set, in `/etc/fugal/validator.env`. The
> shipped example (`deploy/env/validator.env.example`) is pointed at the TEST
> network on purpose, so copying it unchanged runs against testnet with no
> error.
>
> ```
> FUGAL_NETWORK=finney
> FUGAL_NETUID=70
> FUGAL_EPOCH_INTERVAL=4320
> FUGAL_V2_ACTIVATION_BLOCK=9485640
> FUGAL_VALIDATOR_IP=<this host's public egress IPv4>
> FUGAL_STATE_PATH=/var/lib/fugal/validator_state.json
> FUGAL_EPOCH_DIR=/var/lib/fugal/epochs
> FUGAL_EPOCH_LOG_DIR=/var/lib/fugal
> ```
>
> - **Approved entries are not set by hand any more.** The validator approves
>   every frozen compose committed in its own checkout (minus any listed as
>   superseded), with the pinned base measurement. `FUGAL_TEE_MEASUREMENTS`
>   still adds entries if you set it; `FUGAL_APPROVED_FROM_CODE=0` uses it alone.
>   To check an entry against the signed release:
>   `python scripts/fetch_approved_entries.py --tag <image tag> --network finney`.
> - **Archive endpoint**: on finney, unset `FUGAL_ARCHIVE_ENDPOINTS` /
>   `FUGAL_FALLBACK_ENDPOINTS` default to `wss://archive.chain.opentensor.ai:443`
>   and `wss://entrypoint-finney.opentensor.ai:443`. Set them to use your own.
> - **`FUGAL_VALIDATOR_IP`**: `curl -s https://ifconfig.me` from this host.
> - **Nothing to set to follow the root.** A validator that is not the root
>   fetches the root's published reveals itself. Only the root sets
>   `FUGAL_ROOT=1`.
> - On finney the validator **refuses to start**, saying which, if
>   `FUGAL_V2_ACTIVATION_BLOCK` is unset or different, if no approved entry is
>   a v2 miner image, or if `FUGAL_VALIDATOR_IP` is not a public IPv4 address.
>   Each of these used to start, look healthy, and burn the validator's weight.
> - **What a healthy start logs**: one `READY: …` line (see "Upgrading" above),
>   and before it, in this order: `Approved entries: N`,
>   `Protocol v2 activation: boundary block 9485640`, `Consensus environment
>   digest: <16 hex>` (the same on every validator of one release),
>   `v2 role: follower of the reveals in …` (or `ROOT`), `validator IP <ip>
>   published on chain` (or `already on chain`), and then every epoch
>   `Following root: seat uid N (head …) …`.

### Test run (mock mode, test chain only)

`--mock` accepts unattested proofs and is **refused on finney**. To exercise
the loop against a test subnet before going live:

```bash
python neurons/validator.py \
  --netuid <NETUID> \
  --network test \
  --coldkey fugal_validator \
  --hotkey default \
  --mock \
  --once
```

`--once` exits after a single epoch.

### Running as a service

Use the unit and environment file in the repository rather than writing your
own: they carry the settings whose absence has failed before (`LimitNOFILE`
for the collateral fetch's descriptor leak, `EnvironmentFile` so approved
entries never appear in `ps`, `Restart=always` because state is persisted).

```bash
sudo cp deploy/systemd/fugal-validator.service /etc/systemd/system/
sudo install -D -m 0600 deploy/env/validator.env.example /etc/fugal/validator.env
sudoedit /etc/fugal/validator.env      # network, netuid, approved entries, epoch interval, bucket, paths
sudo systemctl daemon-reload && sudo systemctl enable --now fugal-validator
journalctl -u fugal-validator -f
```

Edit **every** value in the environment file; the example ships pointed at the
test network.

### Host reference

Measured on a validator host on 2026-11-03, so the numbers are a starting
point, not a floor.

- **Shape.** A 2 vCPU / 4 GB / 80 GB x86 cloud VM on Ubuntu 24.04 with
  Python 3.12. The validator verifies proofs and does
  arithmetic; it never runs the model. The locked venv is 5.7 GB on disk
  because `torch` is pinned from PyPI with its CUDA libraries, so give the
  disk 20 GB of headroom. 2 GB of swap is cheap insurance on a 4 GB box.
- **Users.** The service runs as a system user (`fugal`, no login shell
  access over SSH) that owns `/opt/fugal/fugal-subnet`, `/var/lib/fugal` and
  the wallet directory. A separate admin user holds the SSH key and `sudo`.
  The hotkey is regenerated on the host, as the service user, by the person
  who holds the words; it is never copied over the network.
- **Inbound.** The validator listens on nothing but SSH, so the host accepts
  **SSH only** (it records its egress IP on chain, `FUGAL_VALIDATOR_IP`, for
  miners' firewalls; nothing listens behind that record):
  provider firewall and host firewall (`ufw`) both deny everything else, root
  login and password authentication are off in `sshd`, `fail2ban` guards the
  SSH jail, and unattended security upgrades are on.
- **Outbound.** The chain endpoints (wss), the miners' axons (8091), the PCCS
  (`FUGAL_PCCS_URL`), Hugging Face for the benchmark datasets the pool is
  built from (not needed when a manifest-matching `pool.json.gz` is
  provided; the token manifest ships in the repo), GitHub and PyPI for the
  build, and the reveal bucket.
- **Reveal uploads.** `google-cloud-storage` is not in the lock, so
  `reveal_publisher` uses the `gcloud` CLI. Install it on the host and
  activate the bucket's writer service account **as the service user**
  (`gcloud auth activate-service-account --key-file=...`); the key file
  belongs to that user, mode 0600, and never enters the repo.
- **Archive endpoint per network.** `FUGAL_ARCHIVE_ENDPOINTS` must be on the
  same chain as `FUGAL_NETWORK`: a finney archive is the wrong chain for a
  testnet subnet. Probed from the host: `wss://test.finney.opentensor.ai:443`
  served state 100 000 blocks back in 1.6 s, and
  `wss://archive.chain.opentensor.ai:443` did the same on finney in 3.2 s.
  Switching a host from testnet to finney changes four values in
  `validator.env`: network, netuid, the archive endpoint, and the approved
  entry; `FUGAL_EXPLORE_FRACTION` is removed (a finney neuron refuses it).

### Hourly watch with an alert

`deploy/systemd/fugal-watch.timer` runs `scripts/watch_subnet.py` for one
epoch boundary every hour from the validator host and raises an alert when
any assertion fails or the run dies. The script's own epoch-interval default
is 3600 s; the wrapper passes `FUGAL_WATCH_EPOCH_INTERVAL` (4320) so it
watches the subnet's real boundaries.

```bash
sudo cp deploy/systemd/fugal-watch.{service,timer} /etc/systemd/system/
sudo install -m 0600 deploy/env/watch.env.example /etc/fugal/watch.env
sudoedit /etc/fugal/watch.env     # your hotkey, the epoch interval, the alert URL
sudo systemctl daemon-reload && sudo systemctl enable --now fugal-watch.timer
systemctl list-timers fugal-watch.timer
journalctl -t fugal-watch         # every FAIL, with or without an alert URL
```

`FUGAL_ALERT_URL` is an ntfy topic (`https://ntfy.sh/<topic>`, no account,
pushes to a phone) or a Discord channel webhook (`FUGAL_ALERT_KIND=discord`).
**ntfy topics are public: anyone who knows the name can read every alert, and
alerts carry the hostname and the FAIL lines.** Use a long random topic, for
example `fugal-` followed by 32 random characters, and treat it like a
password. For Telegram, create a bot with BotFather, send it one message, then
run `sudo deploy/systemd/fugal-alert-telegram-setup.sh` on the host: it asks
for the token without echo, finds the chat id, writes the three alert values
into `/etc/fugal/watch.env` and posts a test message. Left empty, failures go
to the journal only. A run waits until the current epoch is
`FUGAL_WATCH_SETTLE_BLOCKS` (340) blocks old, then judges it: a validator
sets weights only after collection (boundary + 180 blocks on a 360-block
epoch) and may spend up to its 1800 s (150-block) post-collection budget
first, so as late as about boundary + 330. A check made earlier reports "did
not set weights this epoch" for a healthy validator; 340 is about four
minutes before the next boundary. One run lasts at most one epoch, 72 minutes on a 4320 s
epoch; systemd does not start a second instance while one is active, so the
hourly tick simply skips. The alert URL (which carries the Telegram bot token)
reaches curl on stdin, never on its command line.

## Configuration

Operational settings are read from the environment (see `fugal_subnet/config.py`).
Constants that decide a score are **not** settable on a live validator: they
are consensus state, their values are in the digest every reveal carries, and
`--live` refuses to start if any of them is set in the environment
(`fingerprint.assert_environment`). They are listed separately below.

| Variable | Default | Description |
|----------|---------|-------------|
| `FUGAL_NETWORK` | `test` | Network: `finney`, `test`, `local` |
| `FUGAL_NETUID` | `1` | Subnet netuid |
| `FUGAL_EPOCH_INTERVAL` | `3600` | Seconds between epochs. **Mainnet: 4320** (tempo 360 × 12 s); must equal tempo × 12 on every neuron |
| `FUGAL_V2_ACTIVATION_BLOCK` | unset | The epoch boundary block at which protocol v2 activates. A consensus input: identical on every validator, recorded in the consensus digest; refused at startup if it is not an epoch boundary. **Finney: 9485640**; a live finney validator with any other value, or none, is refused at startup. Elsewhere, unset means v1 forever. See "Protocol v2: flip procedure" |
| `FUGAL_TEE_MEASUREMENTS` | — | Comma-separated approved entries, each `<base_measurement>:<compose_hash>`. Required for `--live`, which refuses a bare base: one would accept any application on that image, and the application fixes the model upstream (see `deploy/dstack/README.md`) |
| `FUGAL_PCCS_URL` | `https://pccs.phala.network` | Where DCAP collateral is fetched. Not a trust root; point at a local caching proxy to remove the third party from the epoch path |
| `FUGAL_COLLATERAL_TIMEOUT` | `10` | Seconds one collateral fetch may take before the proof is recorded `unverifiable` |
| `FUGAL_COLLATERAL_EPOCH_BUDGET` | `600` | Seconds the whole verification phase may spend fetching collateral (I6) |
| `FUGAL_BENCHMARK_POOL` | — | Leave unset. A materialised pool file is accepted only if it matches the pinned manifest and the size floor; a non-pinned pool must declare `FUGAL_POOL_UNPINNED=1` and is for local rehearsals only |
| `FUGAL_VALIDATOR_IP` | — | This host's public egress IPv4: the address miners SEE your queries come from (`curl -s https://ifconfig.me` from this host; behind NAT, the NAT's address, and it must be a single stable one). Recorded on chain at startup (only when it changed) so miners can admit validators, and only validators, at their firewall (`scripts/axon_firewall.py`). Nothing listens on it. Unset: miners that firewall their axon cannot be reached by this validator, so **a live finney validator without a public IPv4 here is refused at startup** |
| `FUGAL_REVEAL_BUCKET` | — | Public-read GCS bucket each epoch's `reveal.json` is published to, after the weight commit-reveal window, with its sha256 committed on chain by this hotkey (`reveal:<epoch_id>:<sha256>`). Unset: reveals stay on local disk. The ROOT must set it (the other validators read its reveals); for any other validator it is optional |
| `FUGAL_ROOT` | unset | `1` on the ROOT validator only (the subnet owner's): it follows nobody. Unset, a validator is the root only if its `FUGAL_REVEAL_BUCKET` is the reference bucket. `0` makes even that validator a follower, for a root that is recovering lost state (see "The root"). Any value other than `1`, `true`, `yes`, `0`, `false`, `no` is refused at startup |
| `FUGAL_REFERENCE_BUCKET` | finney: `fugal-reveals` | The root's public reveal bucket a follower fetches from. Built in on finney; set it on a testnet that has a root |
| `FUGAL_ROOT_HOTKEY` | finney: the root's hotkey (built in) | The root validator's hotkey. A follower verifies every reveal it fetches against this hotkey's on-chain reveal commitment. Built in on finney, where any other value is refused at startup; set it on a testnet that has a root (unset there: fetched reveals are not verified, with a warning) |
| `FUGAL_REVEAL_DELAY_EPOCHS` | `0` (read from chain) | Epochs to hold a reveal before publishing; the chain's commit-reveal period when enabled, else 1 |
| `FUGAL_FRAME_BOOTSTRAP` | — | **Not needed on a follower any more**: it fetches the root's reveals itself. Still honoured: a directory of a reference validator's reveals that the operator keeps in sync, used as the v2 reference in place of the built-in fetch. On a root, a directory of its own (or a peer's) published reveals to rebuild lost state from at startup |
| `FUGAL_POOL_FILE` | `data/benchmarks/pool.json.gz` | Where the materialised pool built by `scripts/export_pool.py` is looked for; verified against the manifest on load |
| `FUGAL_POOL_FILE_V2` | `data/benchmarks/pool_v2.json.gz` | The same for the protocol v2 pool (`scripts/export_pool.py --protocol 2`). Place it before setting `FUGAL_V2_ACTIVATION_BLOCK`: with the block set the validator loads the v2 pool at startup, and in `--live` refuses to start if it cannot |
| `FUGAL_ARCHIVE_ENDPOINTS` | — | Comma-separated archive-capable subtensor endpoints. **Set it on mainnet**: the validator pins reads to the boundary and collection blocks, and a non-archive node refuses blocks older than ~300 |
| `FUGAL_FALLBACK_ENDPOINTS` | — | Comma-separated endpoints the SDK falls back to when the primary drops |
| `FUGAL_WEIGHTS_SET_RETRIES` | `3` | In-epoch re-submits of `set_weights` after a failure, while blocks remain and the rate limit allows |
| `FUGAL_COLLATERAL_MAX_TIMEOUTS` | `10` | Consecutive abandoned collateral fetches before the validator stops dialling the PCCS for the epoch |
| `FUGAL_STATE_PATH` | `results/success-v1/validator_state.json` | Persisted scoring state. Put it on durable disk |
| `FUGAL_EPOCH_DIR` | `results/success-v1/epochs` | Where each epoch's `reveal.json` is written |
| `FUGAL_EPOCH_LOG_DIR` | `results` | Where `epoch_logs/` is written |
| `FUGAL_LAMBDA` | `2.0` | Legacy preference-training knob only. Benchmark routing uses the success contract's fixed λ=1 (`fugal_subnet/routing_protocol.py`) |
| `LOG_LEVEL` | `INFO` | Logging level |

**Consensus constants — read-only on mainnet.** Shown so you know what the
digest covers. Setting any of them makes a neuron refuse to start on finney
(identified by genesis hash, as `--mock` is); on a test or local chain it
starts with a warning naming the override, which is how the netuid-552
warm-up runs `FUGAL_EXPLORE_FRACTION=0.5` on every neuron.

| Variable | Value | What it decides |
|----------|-------|-----------------|
| `FUGAL_SLICE_SIZE` | `300` | Questions per epoch |
| `FUGAL_SLICE_SHARE_FACTOR` | `1.5` | Cap on a benchmark's share of the slice relative to its share of the pool |
| `FUGAL_COLLECT_FRACTION` | `0.5` | How far into the epoch proofs are collected |
| `FUGAL_EXPLORE_FRACTION` | `0.05` | Exploration quota, as a fraction of the slice (the testnet warm-up sets `0.5` on every neuron, miners' compose included) |
| `FUGAL_REQUIRE_COMMITMENT` | `1` | Require on-chain commitment before scoring |
| `FUGAL_EVIDENCE_HALF_LIFE` | `200` | EWMA decay half-life for evidence accumulation |
| `FUGAL_FRAME_HALF_LIFE` | `500` | Reference-frame decay half-life, in epochs |
| `FUGAL_FRAME_PRIOR_STRENGTH` | `20` | Frame prior strength, in pseudo-observations |
| `FUGAL_FRONTIER_MIN_TRIALS` | `300` | Distinct exploration observations per hull model before the frontier pays in full |
| `FUGAL_BURN_IN_QUESTIONS` | `3000` | Scored questions before a fresh head reaches full score (v1) |
| `FUGAL_V2_BURN_IN_QUESTIONS` | `1500` | Scored questions since the last reset or miss before a head can take a v2 seat (E'') |
| `FUGAL_WILSON_CONFIDENCE` | `0.95` | Wilson LCB confidence level |
| `FUGAL_SCORE_QUALITY_FLOOR` | `0.8` | Share of the frontier's best accuracy below which a head scores zero |
| `FUGAL_MAX_WEIGHT_DELTA` | `0.3` | Max weight change per UID per epoch |
| `FUGAL_CHAMPION_WINDOW_EPOCHS` | `48` | Epochs in the v2 comparison window |
| `FUGAL_CHAMPION_ACCURACY_VALUE` | `30` | One extra correct answer is worth this many questions of the anchor's spend (V) |
| `FUGAL_CHAMPION_ANCHOR_MODEL` | `anthropic/claude-fable-5.1` | The model whose expected spend per question prices V |
| `FUGAL_CHAMPION_HARD_WEIGHT` | `2` | Weight of a `hard`-category question in the net-value test |
| `FUGAL_CHAMPION_MARGIN` | `0.2` | NV points a challenger's lower bound must clear to take the seat |
| `FUGAL_CHAMPION_CONFIDENCE` | `0.99` | One-sided confidence of the paired bootstrap lower bound |
| `FUGAL_CHAMPION_BOOTSTRAP_RESAMPLES` | `2000` | Resamples in the paired bootstrap |
| `FUGAL_CHAMPION_MIN_PAIRED_QUESTIONS` | `1500` | Paired questions a take needs |
| `FUGAL_CHAMPION_MIN_ACCURACY` | `0.30` | Window-accuracy floor to take or hold a seat, or be paid as a former champion |
| `FUGAL_CHAMPION_HARD_GATE_MIN_QUESTIONS` | `100` | Hard questions in the window before the hard-accuracy gate applies |
| `FUGAL_CHAMPION_TIME_GATE_FACTOR` | `2` | Per-call time gate (not applied at launch) |
| `FUGAL_CHAMPION_GHOST_OFF` / `_RETIRE` | `3` / `24` | Epochs a missed / retired champion's ghost bar stands |
| `FUGAL_CHAMPION_CONFIRM_QUESTIONS` | `1500` | Fresh questions (drawn after its first pass) a challenger must pass the take test on again before it takes a settled champion's seat. From the confirmation activation block |
| `FUGAL_CHAMPION_CONFIRM_OPEN_EPOCHS` | `5` | Epochs after a seat is filled from empty during which it can still be taken by the plain test |
| `FUGAL_CHAMPION_GHOST_RETIRE_CONFIRM` | `8` | Epochs a retired champion's ghost bar stands from the confirmation activation block (instead of 24) |
| `FUGAL_PAYOUT_SEAT_SHARES` | `0.85,0.10,0.05` | The miner share across the champion and the two before it |
| dedup, exploration-integrity and head-bound constants | see `config.py` | `FUGAL_DEDUP_*`, `FUGAL_EXPLORE_*`, `FUGAL_HEAD_*`, `FUGAL_POOL_MIN_SIZE`, `FUGAL_MODEL_PRICES` |

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `test` | Network: `finney`, `test`, `local` |
| `--netuid` | `1` | Subnet netuid |
| `--coldkey` | `default` | Wallet coldkey name |
| `--hotkey` | `default` | Hotkey name |
| `--once` | off | Run one epoch and exit |
| `--live/--mock` | `--live` | Live (default, requires real TDX) or mock (accepts unattested proofs; refused on finney) |
| `--wallet-path` | SDK default | Bittensor wallet root directory |
| `--log-level` | `INFO` | Logging level |

## TEE Proof Verification

When a miner submits a proof, the validator checks:

1. **DCAP Attestation** — The TDX quote is cryptographically verified against
   Intel's DCAP infrastructure. The quote proves the benchmark ran inside a
   genuine Intel TDX confidential VM.

2. **Measurement Match** — `measurement_id(quote)` — sha256 over the quote's
   MRTD, RTMR1 and RTMR2, which the CPU fills in and Intel's signature covers —
   must be the base half of an approved entry, and the `compose-hash` event
   replayed out of the quote's RTMR3 log must be that entry's application half.
   RTMR0 is excluded because it records host-chosen virtual hardware and would
   fork the approved list by instance size.

   This deliberately does **not** use the proof's `source_hash` field. DCAP
   proves the hardware is real; it says nothing about the code inside it, so an
   attacker with a genuine TDX machine can run a modified harness and still
   produce a valid quote. What distinguishes them is the measurement register,
   which they cannot choose — unlike a field the workload writes about itself.

3. **Report Data Binding** — The proof's content hash is embedded in the TDX
   quote's `report_data` field. Tampering with any proof field after attestation
   invalidates the binding.

4. **Nonce and Questions** — The proof's nonce and questions hash must match
   this epoch's. Prevents replay across epochs.

5. **Assigned Slice** — The result question ids must equal the assigned slice
   exactly. The questions hash is a public value, so matching it proves nothing
   on its own: a miner can copy it while grading an easier set of its own
   choosing.

6. **Exploration Set** — Every nonce-assigned exploration question must be
   present and routed to the model the nonce assigned. Exploration costs the
   miner money and earns it nothing directly, so rejection is the only thing
   that makes it happen.

7. **Committed Head** — `proof.weights_hash` must equal the on-chain commitment
   *and* the sha256 of the head shipped in the bundle. Otherwise a miner
   commits one head, runs another, and keeps a stable evidence key while
   swapping heads every epoch.

8. **Advertised Bundle** — The downloaded proof must hash to the `proof_hash`
   the miner advertised over its axon.

9. **Cost Consistency** — Per-question and per-model costs must both reconcile
   with the attested total. This is a **rejection**, not a warning: understating
   the total lowers the miner's cost per question and so its frontier reference, and under real attestation every
   figure comes from the same metered image, so an inconsistency means the proof
   is not what it claims to be.

## Monitoring

### Epoch logs

Structured JSONL logs are written to `results/epoch_logs/`. Each entry includes:

- Epoch ID, block hash and protocol version
- Number of miners queried; valid, invalid and `unverifiable` proofs; per-UID
  failed-call counts
- `collection`: for every UID with a head committed at the boundary (and any
  UID that answered), how the query to its axon ended (the status and the
  seconds it took) and what became of its proof (`valid`, `invalid`,
  `unverifiable`, `absent`). The console says the same in one line per epoch:
  `Collection: 256 queried; 14 valid, 0 invalid, 0 unverifiable, 242 absent.
  Committed miners without a valid proof (2): uid 128 (absent; query 408
  Timeout after 120.0s), …`. This is what answers a miner who asks whether
  it was never reached, was slow, or was refused
- The seat event (`first`, `hold`, `take`, `ghost`, `empty`), the champion,
  the former champions, and the comparison detail per head (which gate
  failed, or the NV and its lower bound against the champion)
- Weight assignments (at most three miner UIDs plus the burn UID)
- Whether the metagraph was read pinned to the boundary block or fell back to
  the live one (`metagraph_unpinned`)
- Anomaly flags
- Phase timing breakdown
- Commit-reveal verification status

### Anomaly detection

The validator automatically flags:

- Epochs where no miners respond
- Weight concentration (>80% of weight on one miner)
- Suspiciously uniform accuracy
- Commit-reveal verification failures

## Safety Features

- **Zero validator spend** — validators verify proofs, never call models
- **TEE attestation** — hardware-attested proofs prevent result fabrication
- **Measurement pinning** — only approved runtime images are accepted
- **One seat, no margin to hold it** — a champion keeps the seat unless a
  challenger is measurably better on the same questions (99% lower bound of
  net value above the margin, over at least 1,500 paired questions), so the
  seat cannot change hands on noise
- **The copy rule** — a head belongs to the coldkey that first committed it;
  the same bytes under another coldkey are an exact copy and earn nothing; a
  junior head must also beat every senior seat holder by the margin
- **The E″ miss rule** — a missed epoch decays a head's evidence and restarts
  its burn-in; selective publication (skip bad epochs) is penalized
- **Weight capping** — weights change at most ±0.3 per UID per epoch
- **Commit-reveal** — questions committed before miner queries
- **State persistence** — scoring state, the seat and the comparison window
  survive restarts; a lost window is rebuilt from published reveals
- **Seniority from chain** — the first commitment block, read at the epoch
  boundary; commitments at or before the activation block tie and are ordered
  by a salted hash, so nothing can be gamed by committing early

## Protocol v2: flip procedure

Protocol v2 switches every validator from the v1 rules to the v2 rules at one
named epoch boundary, the **activation block**. The rules an epoch is scored
under depend only on that epoch's boundary block
(`slicer.protocol_version_for_boundary`), never on when a validator happens to
run it. So a late or retried epoch is scored the same way on every peer.
What changes for miners is in `docs/MINER_GUIDE.md`, "Moving from v1 to v2".

Values marked `<LIKE_THIS>` are published in the signed v2 release
(`<V2_RELEASE_NOTE>`) at the announcement, at least 24 hours before the flip.
Do not fill them in from anywhere else.

### Before the flip

1. **Upgrade the validator to the v2 release** (`<V2_RELEASE_TAG>`) any time
   in the notice period. Until the activation block it scores exactly as v1
   does: the same weights, and reveals with no v2 fields. Upgrading early is
   safe.
2. **Set `FUGAL_V2_ACTIVATION_BLOCK=<ACTIVATION_BLOCK>`** in the environment
   file, and restart. Every validator must use the **identical** value.
   - It is a consensus input like `FUGAL_EPOCH_INTERVAL`. It is written into
     the consensus digest every reveal carries, so a validator with a
     different value, or none, shows a different digest from its first epoch,
     long before the flip.
   - It must be an epoch boundary (a multiple of the blocks per epoch). The
     validator refuses to start on a block that is not
     (`FUGAL_V2_ACTIVATION_BLOCK: ... not an epoch boundary`).
   - Unset, v2 never activates. That validator keeps v1 rules after the flip
     and disagrees with every peer that flipped.
   - The startup log confirms it:
     `Protocol v2 activation: boundary block <ACTIVATION_BLOCK> (epoch index N); v1 rules before it`.
3. **Add the v2 approved entry beside the v1 entries** in
   `FUGAL_TEE_MEASUREMENTS`, from `scripts/fetch_approved_entries.py --tag
   <V2_IMAGE_TAG> --network finney` (`--network test` on testnet 552), and restart. Follow the transition window in
   `docs/APPROVED_ENTRY_ROTATION.md`: every validator carries **both** before
   any miner moves. Before the flip a v2 miner holds its axon, so no v2 proof
   arrives early. Carrying the entry early costs nothing, and carrying it late
   scores every v2 miner absent at the flip.
4. **Set `FUGAL_VALIDATOR_IP`** to this host's public egress IPv4, and
   restart. The startup log says `validator IP <ip> published on chain` (or
   `already on chain`). Miners admit only published validator IPs at their
   firewall, so a validator without one cannot reach them; and the miners'
   script will not switch the allowlist on while any active validator is
   missing, so one missing validator leaves every miner exposed.
5. **Do not set any other v2 constant.** The `FUGAL_CHAMPION_*`,
   `FUGAL_PAYOUT_SEAT_SHARES` and `FUGAL_BURN_IN_QUESTIONS` values are
   consensus constants. `--live` refuses to start on finney with any of them
   set.

A restart at any point is safe. State is written only when an epoch
completes, so a validator killed mid-epoch resumes from the previous epoch and
scores the interrupted one again, identically. From the flip, the state also
holds the champion seat and the comparison window (`<state>.v2window.npz`
beside the state file). Keep both on durable disk. A validator that has seat
state but has lost the window logs `v2 seat state present but its comparison
window is missing`. It keeps holding the seat (hold, resume and ghosts go on)
but makes no `first` or `take` decision until the window is rebuilt from
the reference reveals (a restart does it) or has refilled for
`CHAMPION_WINDOW_EPOCHS` (48) epochs.

**A missed epoch heals itself.** The validators form a star around one ROOT
validator (the subnet owner's), whose reveals are public:
- **On the root:** set `FUGAL_ROOT=1`. It references nobody.
- **On every other validator: nothing.** It fetches the root's reveals itself,
  at startup and before every epoch, over HTTPS from the public bucket
  (finney: `gs://fugal-reveals/70`), into `reference-reveals/<netuid>/` beside
  its state file. It needs outbound HTTPS to `storage.googleapis.com` and no
  credentials. Every reveal it fetches is checked against the root's
  on-chain commitment (the `reveal:<epoch>:<sha256>` the root's hotkey
  commits); one that differs is never used and is logged as an ERROR
  (`… differ from the root's on-chain commitment and were NOT used`): tell
  the subnet owner. The root's hotkey is built in on finney; a testnet names
  it in `FUGAL_ROOT_HOTKEY`. The startup log says `v2 role: follower of the reveals in …`,
  and every epoch `Root reveals from gs://…: N fetched, …` and `Following
  root: seat uid N (head …) at e…, entries N, fugal vX (root vY), reference
  lag K epoch(s); own seat …`. The lag is normally the commit-reveal hold.
  If this validator's champion has differed from the root's for longer than
  the re-sync below takes, it logs an ERROR that says what to check: the
  approved entries, `FUGAL_VALIDATOR_IP`, and the release version.
  A validator with no reference reveals and no way to fetch any refuses to
  start. (Before this release a follower ran a `gsutil rsync` timer and set
  `FUGAL_FRAME_BOOTSTRAP`; that still works and takes precedence, and both
  can be removed.)

After a v2 epoch a follower did not run (it was down, or it scored
nothing), it logs `v2 epochs ... were not run by this validator` and keeps
working as usual; once the root has published past the gap (the commit-reveal hold) it
logs `re-anchored on it`, and once a later epoch matches, `gap closed`.
Nothing to do by hand. A follower that observed every epoch but still
differs from the root (it did not receive one miner's proof that the root
did, or the other way round) logs `ALARM: ... differs from its reference's`;
if that persists for 3 reference epochs it logs `treating it as a gap` and
re-anchors the same way. The root is the arbiter: every follower converges
on it.

**The root.** Its outages count for the whole network: an epoch the root
missed is followed by everyone (a champion that was not seen leaves a ghost
and resumes when seen; a challenger's burn-in run restarts). Keep it up and
monitored.

**If the root misses epochs.** After an epoch in which it verified no proof,
the root logs a WARNING, `Root: no verified proof in the v2 epoch(s) through
e…`. If no miner served that epoch there is nothing to do: the seat state is
unchanged. It matters only when the root was down while followers scored.
Then the followers re-anchor on the root as it publishes, and converge on it.

A root that lost its state files (a dead disk) rebuilds from a follower:
1. Stop the root. Move its state files aside: `$FUGAL_STATE_PATH` and every
   `$FUGAL_STATE_PATH.*` next to it (`.v2window.npz` among them).
2. Copy a follower's reveals (its `$FUGAL_EPOCH_DIR`, or its bucket) to a
   directory on the root. Set `FUGAL_FRAME_BOOTSTRAP` to that directory and
   `FUGAL_ROOT=0`, so that for now it follows them.
3. Start the root. It rebuilds from the follower's reveals, re-anchors as the
   directory is refreshed, and logs `gap closed` after the hold.
4. Set `FUGAL_ROOT=1` again, unset `FUGAL_FRAME_BOOTSTRAP`, and restart it.
Followers need nothing.

### The seat-confirmation rule (after the flip)

From the confirmation activation block, a challenger that beats a settled
champion on the paired test is first recorded as *pending* and takes the seat
only when it passes the same test again on at least 1,500 questions drawn
after that first pass (INVARIANTS I3, "a take is confirmed on fresh
questions"). The champion is paid meanwhile. From the same block a retired
champion's ghost bars for 8 epochs instead of 24.

- **Nothing to set.** Finney's block is part of the release; the startup log
  says `Fresh-data confirmation of a seat take: from the first epoch whose
  boundary is at or after block N` (or `not scheduled on this chain`).
  Setting `FUGAL_CONFIRM_ACTIVATION_BLOCK` on finney to anything else is
  refused. Testnets set it.
- **Upgrade before the block.** Until it, the release scores exactly as the
  one before: the same weights and the same published seat. The consensus
  digest changes with the release (three new constants and the block are in
  it), so compare digests among validators on the same release.
- **In the log**, a waiting challenger shows as `v2 <head>: passed the take
  test; takes the seat when it passes again on 1500 questions drawn from the
  next epoch on`, then per epoch `vs fresh questions, <champion>: n=… NV=…
  LCB=…`, and finally `confirmed on questions drawn since epoch N` with the
  seat event `take`. The reveal's `v2_state.seat.pending` lists who is
  waiting and since when.

### At the flip

- The first epoch whose boundary block is `<ACTIVATION_BLOCK>` is scored under
  v2. The epoch header in the log reads `protocol v2`. The previous one reads
  `protocol v1`.
- **Retire the v1 entries at the flip.** v1 images are not valid after it, a
  hard date published in the release. Remove them from
  `FUGAL_TEE_MEASUREMENTS` and restart, at or after the activation block,
  never before.
- **Expect the miner share to burn for the first hours.** Every head is new at
  the flip, and no head can take the seat until it has scored
  `V2_BURN_IN_QUESTIONS` (1500, five epochs, about six hours) in an unbroken
  run. Until then each epoch logs `v2 seat: empty` and the weight goes to the
  burn UID; an epoch with no valid proof at all burns too (`No valid proofs:
  the miner share burns`). This is intended, not a fault.

### What to watch

Per epoch, in the log and in `reveal.json`:

| Signal | Healthy | If not |
|---|---|---|
| `consensus_digest` | identical to every peer's, before and after the flip | the activation block, a consensus constant or a package differs: compare `environment` in the two reveals |
| `protocol_version` in the reveal | absent before `<ACTIVATION_BLOCK>`, `2` from it | the block is unset or different on this validator |
| `v2 seat: <event>` log line, `champion` in the reveal | `empty` until a head is burned in; then `first`, then `hold` / `take` / `ghost` | weights disagree with peers' when the seat does. Find the first epoch where the `champion.detail` lines differ |
| weights | identical to every peer's. At most three miner UIDs are paid (85/10/5), plus the burn UID | see `docs/INVARIANTS.md`, I1 |
| `shared_grades` in the reveal | present in every v2 epoch | the v2 wiring is not running: check the release |
| `unverifiable` / `invalid` counts | v2 miners verify from the flip | the v2 entry is missing (a v2 proof on a validator without it is `invalid`) |

`scripts/watch_subnet.py` compares validators epoch by epoch. Run it across the
flip and for the first day after. Weights still move by at most
`FUGAL_MAX_WEIGHT_DELTA` per epoch, so a new champion's weight reaches its seat
share over a few epochs.

### Rehearse it first

The flip is rehearsed locally with the shipped neurons on a local chain, at no
cost:

```bash
python scripts/dress_rehearsal.py --scenario v2_flip   # Docker, ~30-60 min
python scripts/rehearsal_v2.py                         # in process, seconds
```

Both check that validators agree across the flip, that v1 epochs are
unchanged, the seat and the 85/10/5 split, the E″ miss rule, and a restart
mid-epoch. The chain scenario shortens burn-in and a few other constants so it
finishes in minutes, and prints the overrides it used.

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first.

**"No valid proofs received"** — No miners responded or all proofs failed
verification. The epoch is skipped and logged. If it happens every epoch,
check your stake first: in `--live` miners answer only hotkeys with at least
`FUGAL_MIN_VALIDATOR_STAKE` (default 50) stake on the subnet as the chain
reports it, a permit alone does not qualify, and a refused query looks exactly
like an absent miner from your side. The startup log says so when it applies.

**Refused at startup on finney** — the message names the cause:
`FUGAL_V2_ACTIVATION_BLOCK is … on finney` (set it to 9485640);
`FUGAL_TEE_MEASUREMENTS approves no protocol v2 miner image` (add the current
entry with `scripts/fetch_approved_entries.py`); `FUGAL_VALIDATOR_IP is …, not
a public IPv4 address`; `Could not fetch the root validator's reveals` (allow
outbound HTTPS to `storage.googleapis.com`, or set `FUGAL_ROOT=1` if this is
the root). See "Minimum mainnet environment".

**"This validator's champion has differed from the root's"** — it verifies a
different set of proofs from the root. In order of likelihood: an approved
entry is missing (compare `Approved entries: N` with the root's), miners'
firewalls do not admit this host (`FUGAL_VALIDATOR_IP` must be the address
they see, and on chain), or the release differs (the `Following root` line
shows both versions). The `Collection:` line shows which committed miners
this validator did not get a valid proof from, and how each query ended.

**Weight-setting failed** — Check that your validator has enough stake.

**DCAP verification fails** — Install the verifier with the project's own
lock (`uv sync --locked --extra tee`; `--live` refuses to start without it) and
make sure the host has outbound HTTPS to the PCCS named by `FUGAL_PCCS_URL`
(default `https://pccs.phala.network`, not Intel). The PCCS supplies
Intel-signed collateral that `dcap-qvl` verifies against a compiled-in Intel
root, so it is an availability dependency, not a trust root. A stalled PCCS
does not stop the epoch: fetches are bounded per proof and per epoch, and
proofs that could not be checked are logged as `unverifiable`, never as
invalid. To take the third party and the ~700 ms round trip out of the epoch
path, run a caching proxy in front of it and point `FUGAL_PCCS_URL` at that
(measured: a cache hit verifies in ~15 ms against ~780 ms direct).
