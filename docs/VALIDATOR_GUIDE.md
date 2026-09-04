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
8. Scores each miner as quality per dollar against the best single model
9. Deduplicates copied routing behavior (earliest on-chain commitment wins)
10. Computes weights, sets weights on-chain
11. Publishes the epoch reveal artifact

## Requirements

- Linux or WSL2
- Python 3.10-3.12
- CPU sufficient (no model inference required)
- TAO for subnet registration
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

pip install -e .
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

### Test run (mock mode)

To verify everything works before going live:

```bash
python neurons/validator.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_validator \
  --hotkey default \
  --mock \
  --once
```

The default is `--mock`, which accepts unattested proofs (no real TDX
verification). `--live` requires real TDX attestation from miners.
`--once` exits after a single epoch.

### Running as a service

```bash
sudo tee /etc/systemd/system/fugal-validator.service > /dev/null <<'EOF'
[Unit]
Description=Fugal Subnet Validator
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/fugal-subnet
Environment=FUGAL_NETWORK=finney
Environment=FUGAL_NETUID=<NETUID>
ExecStart=/path/to/fugal-subnet/.venv/bin/python neurons/validator.py \
  --netuid <NETUID> --network finney \
  --coldkey fugal_validator --hotkey default \
  --live
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now fugal-validator
```

## Configuration

All settings are configurable via environment variables (see `fugal_subnet/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FUGAL_NETWORK` | `test` | Network: `finney`, `test`, `local` |
| `FUGAL_NETUID` | `1` | Subnet netuid |
| `FUGAL_EPOCH_INTERVAL` | `3600` | Seconds between epochs |
| `FUGAL_SLICE_SIZE` | `300` | Questions per epoch |
| `FUGAL_TEE_MEASUREMENTS` | — | Comma-separated approved TDX runtime measurements |
| `FUGAL_TEE_PROOF_TIMEOUT` | `600` | Timeout for proof verification (seconds) |
| `FUGAL_REQUIRE_COMMITMENT` | `1` | Require on-chain commitment before scoring |
| `FUGAL_EVIDENCE_HALF_LIFE` | `200` | EWMA decay half-life for evidence accumulation |
| `FUGAL_LAMBDA` | `2.0` | Miner-side TRAINING hyperparameter only. The routing rule has no cost term — see `TRAINING_COST_LAMBDA` in config.py |
| `FUGAL_EXPLORE_FRACTION` | `0.05` | Exploration quota, as a fraction of the slice |
| `FUGAL_FRAME_HALF_LIFE` | `500` | Reference-frame decay half-life, in epochs |
| `FUGAL_FRAME_PRIOR_STRENGTH` | `20` | Frame prior strength, in pseudo-observations (calibrated by measurement — see config.py) |
| `FUGAL_BURN_IN_QUESTIONS` | `3000` | Scored questions before a fresh head reaches full score |
| `FUGAL_WILSON_CONFIDENCE` | `0.95` | Wilson LCB confidence level |
| `FUGAL_MAX_WEIGHT_DELTA` | `0.3` | Max weight change per UID per epoch |
| `FUGAL_STATE_PATH` | `results/validator_state.json` | Persisted scoring state |
| `LOG_LEVEL` | `INFO` | Logging level |

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `test` | Network: `finney`, `test`, `local` |
| `--netuid` | `1` | Subnet netuid |
| `--coldkey` | `default` | Wallet coldkey name |
| `--hotkey` | `default` | Hotkey name |
| `--once` | off | Run one epoch and exit |
| `--live/--mock` | `--mock` | Mock (default, accepts unattested proofs) or live (requires real TDX) |
| `--wallet-path` | SDK default | Bittensor wallet root directory |
| `--log-level` | `INFO` | Logging level |

## TEE Proof Verification

When a miner submits a proof, the validator checks:

1. **DCAP Attestation** — The TDX quote is cryptographically verified against
   Intel's DCAP infrastructure. The quote proves the benchmark ran inside a
   genuine Intel TDX confidential VM.

2. **Measurement Match** — `measurement_id(quote)` — sha256 over the quote's
   MRTD and RTMR0-2, which the CPU fills in and Intel's signature covers — must
   match one of `FUGAL_TEE_MEASUREMENTS`.

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
   the total raises the miner's thrift score, and under real attestation every
   figure comes from the same metered image, so an inconsistency means the proof
   is not what it claims to be.

## Monitoring

### Epoch logs

Structured JSONL logs are written to `results/epoch_logs/`. Each entry includes:

- Epoch ID and block hash
- Number of miners queried, valid/invalid proofs
- Per-miner scores (quality, thrift, composite)
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
- **Dedup seniority** — earliest on-chain commitment wins duplicate clusters

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first.

**"No valid proofs received"** — No miners responded or all proofs failed
verification. The epoch is skipped and logged.

**Weight-setting failed** — Check that your validator has enough stake.

**DCAP verification fails** — Ensure `dcap-qvl` is installed:
`pip install dcap-qvl`. This requires network access to Intel's PCS.
