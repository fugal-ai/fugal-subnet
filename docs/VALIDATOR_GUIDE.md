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
4. Queries all registered miners for TEE-attested benchmark proofs
5. Verifies each proof:
   - TDX attestation (DCAP certificate chain)
   - Runtime measurement matches approved image
   - Nonce and question hash match expected values
   - Cost consistency (metered via attested proxy)
6. Checks each miner's weights hash against its **on-chain commitment**
7. Scores from verified proof results using evidence accumulation
8. Deduplicates copied routing behavior (earliest on-chain commitment wins)
9. Computes composite scores and weights, sets weights on-chain
10. Publishes the epoch reveal artifact

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
| `FUGAL_LAMBDA` | `2.0` | Cost-quality routing tradeoff |
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

2. **Measurement Match** — The `source_hash` in the proof must match one of
   the approved runtime image measurements (`FUGAL_TEE_MEASUREMENTS`). This
   ensures the miner ran the official benchmark harness, not a tampered version.

3. **Report Data Binding** — The proof's content hash is embedded in the TDX
   quote's `report_data` field. Tampering with any proof field after attestation
   invalidates the binding.

4. **Nonce and Questions** — The proof's nonce and questions hash must match
   the expected values for this epoch. Prevents proof replay across epochs.

5. **Cost Consistency** — Per-question costs must sum to the reported total.
   Costs come from the attested MeteringProxy inside the TEE.

## Monitoring

### Epoch logs

Structured JSONL logs are written to `results/epoch_logs/`. Each entry includes:

- Epoch ID and block hash
- Number of miners queried, valid/invalid proofs
- Per-miner scores (accuracy, cost efficiency)
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
