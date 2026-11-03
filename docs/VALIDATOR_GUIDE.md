# Fugal Subnet — Validator Guide

Run a validator on the Fugal subnet. Validators verify TEE-attested proofs
from miners and set on-chain weights. **Validators never call models** —
miners pay for their own inference inside Intel TDX confidential VMs.

## What Validators Do

Epochs are aligned to chain blocks (`EPOCH_INTERVAL/12` blocks per epoch), so
every validator processes the same epoch on the same slice. Each epoch, your
validator:

1. Derives a nonce from the epoch **boundary block's** hash
2. Selects ~300 questions, stratified across benchmarks
3. Commits the question slice + grader hash (commit-reveal integrity)
4. Derives the epoch's exploration assignment from the same nonce
5. Queries all registered miners for TEE-attested benchmark proofs
6. Verifies each proof against every binding (see "TEE Proof Verification")
7. Pools the verified exploration samples into the **reference frame**
8. Scores each miner as accuracy headroom above the constant-policy frontier (what any single model or random mixture buys at the miner's own cost)
9. Deduplicates copied heads: by routing behavior, by weight distance, and by the epsilon rule (a near-copy must beat the senior head); the senior head, by registration block then commitment block, wins
10. Computes weights, sets weights on-chain
11. Publishes the epoch reveal artifact

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
   published reveals to rebuild from).
4. The token manifest shipped in the checkout is `reviewed`, not `candidate`
   (`docs/SUCCESS_CONTRACT.md`).

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
- **Inbound.** The validator serves no axon, so the host accepts **SSH only**:
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
to the journal only. One run lasts until the next
boundary, up to 72 minutes on a 4320 s epoch; systemd does not start a
second instance while one is active, so the hourly tick simply skips.

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
| `FUGAL_TEE_MEASUREMENTS` | — | Comma-separated approved entries, each `<base_measurement>:<compose_hash>`. Required for `--live`, which refuses a bare base: one would accept any application on that image, and the application fixes the model upstream (see `deploy/dstack/README.md`) |
| `FUGAL_PCCS_URL` | `https://pccs.phala.network` | Where DCAP collateral is fetched. Not a trust root; point at a local caching proxy to remove the third party from the epoch path |
| `FUGAL_COLLATERAL_TIMEOUT` | `10` | Seconds one collateral fetch may take before the proof is recorded `unverifiable` |
| `FUGAL_COLLATERAL_EPOCH_BUDGET` | `600` | Seconds the whole verification phase may spend fetching collateral (I6) |
| `FUGAL_BENCHMARK_POOL` | — | Leave unset. A materialised pool file is accepted only if it matches the pinned manifest and the size floor; a non-pinned pool must declare `FUGAL_POOL_UNPINNED=1` and is for local rehearsals only |
| `FUGAL_REVEAL_BUCKET` | — | Public-read GCS bucket each epoch's `reveal.json` is published to, after the weight commit-reveal window, with its sha256 committed on chain by this hotkey (`reveal:<epoch_id>:<sha256>`). Unset: reveals stay on local disk |
| `FUGAL_REVEAL_DELAY_EPOCHS` | `0` (read from chain) | Epochs to hold a reveal before publishing; the chain's commit-reveal period when enabled, else 1 |
| `FUGAL_FRAME_BOOTSTRAP` | — | Directory of published reveals to rebuild the reference frame from when there is no local state; otherwise the shipped `data/reference_frame.json` is required in `--live` |
| `FUGAL_POOL_FILE` | `data/benchmarks/pool.json.gz` | Where the materialised pool built by `scripts/export_pool.py` is looked for; verified against the manifest on load |
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
| `FUGAL_BURN_IN_QUESTIONS` | `3000` | Scored questions before a fresh head reaches full score |
| `FUGAL_WILSON_CONFIDENCE` | `0.95` | Wilson LCB confidence level |
| `FUGAL_SCORE_QUALITY_FLOOR` | `0.8` | Share of the frontier's best accuracy below which a head scores zero |
| `FUGAL_MAX_WEIGHT_DELTA` | `0.3` | Max weight change per UID per epoch |
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

- Epoch ID and block hash
- Number of miners queried, valid/invalid proofs
- Per-miner scores (Wilson LCB, cost per question, frontier reference, headroom, composite)
- Weight assignments
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
- **Evidence accumulation** — scores stabilize over epochs via EWMA decay
  and Wilson LCB; selective publication (skip bad epochs) is penalized
- **Weight capping** — weights change at most ±0.3 per UID per epoch
- **Commit-reveal** — questions committed before miner queries
- **State persistence** — scoring state survives restarts
- **Dedup seniority** — the earliest-registered hotkey, then the earliest commitment, wins duplicate clusters; read from chain at the epoch boundary

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first.

**"No valid proofs received"** — No miners responded or all proofs failed
verification. The epoch is skipped and logged. If it happens every epoch,
check your stake first: in `--live` miners answer only hotkeys with at least
`FUGAL_MIN_VALIDATOR_STAKE` (default 50) stake on the subnet as the chain
reports it, a permit alone does not qualify, and a refused query looks exactly
like an absent miner from your side. The startup log says so when it applies.

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
