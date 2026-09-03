# Fugal Subnet — Validator Guide

Run a validator on the Fugal subnet. Validators build ground truth matrices by
calling frontier LLMs on benchmark questions, evaluate miner-submitted router
heads, and set on-chain weights that determine emission distribution.

## What Validators Do

Epochs are aligned to chain blocks (`EPOCH_INTERVAL/12` blocks per epoch), so
every validator processes the same epoch on the same slice. Each epoch, your
validator:

1. Derives a nonce from the epoch **boundary block's** hash
2. Selects ~300 questions, stratified across benchmarks
3. Commits the question slice + grader hash (commit-reveal integrity)
4. Queries all registered miners for their router heads
5. Verifies each head's SHA256 against its **on-chain commitment** — heads not
   committed at or before the boundary block are rejected
6. Calls every model in the miners' union pool (priced, capped, budget-checked)
   on those questions via OpenRouter
7. Grades responses with mechanical checkers (no LLM-as-judge)
8. Builds an N-questions x M-models binary ground truth matrix
9. Evaluates each head against the matrix (accuracy, cost efficiency, KL divergence)
10. Deduplicates copied heads (earliest on-chain commitment wins), computes
    composite scores and weights, sets weights on-chain (CRW-compatible)
11. Publishes the full epoch artifact (`results/epochs/<epoch_id>/reveal.json`):
    questions, matrix, model costs, scores, weights — the data flywheel

## Requirements

- Linux or WSL2
- Python 3.10-3.12
- CPU sufficient (the Qwen3-0.6B backbone runs head evaluation on CPU in
  ~5-10 min per epoch; a small GPU makes it seconds)
- OpenRouter API key with funded account (~$25/epoch for API calls)
- HuggingFace account for the GPQA benchmark (gated dataset): run
  `huggingface-cli login` and accept the terms at
  https://huggingface.co/datasets/Idavidrein/gpqa — or add `gpqa` to
  `FUGAL_SKIP_BENCHMARKS` (all validators must then skip it identically)
- TAO for subnet registration
- Reliable server with good uptime

## Run Inside a Container (strongly recommended)

The `exec_io` grader executes model-generated Python in a subprocess sandbox
(process-group kill, rlimits, output caps) but **without network isolation**.
A model's response to a HumanEval question could contain code that exfiltrates
data or contacts external services. Run the validator inside Docker or a VM
so untrusted code can't reach your keys or network:

```bash
docker build -t fugal-subnet .
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e OPENROUTER_API_KEY="sk-or-..." \
  -e FUGAL_NETWORK=finney \
  -v fugal-state:/app/results \
  -v ~/.bittensor:/home/fugal/.bittensor:ro \
  fugal-subnet -m neurons.validator \
    --netuid <NETUID> --network finney \
    --coldkey fugal_validator --hotkey default \
    --wallet-path /home/fugal/.bittensor/wallets \
    --live --epoch-budget 50
```

`--user` is required. The image runs as UID 10001, but `btcli` writes key files
mode `0600` owned by your host user, so without it the container can traverse
the wallet directories and read nothing:

```
cat: /home/fugal/.bittensor/wallets/.../hotkeys/default: Permission denied
```

The container provides network isolation for the grading sandbox while still
allowing the validator's own OpenRouter and chain connections.

## Cost Estimate

Each epoch calls up to 30 models on 300 questions. Actual cost depends on the
models miners declare, but expect:

- **~$15-30 per epoch** with a full model pool
- **~$1,200-2,400/month** at hourly epochs
- Budget is hard-capped per epoch (set via `--epoch-budget` or `FUGAL_EPOCH_BUDGET`)
- Live mode (`--live`) requires an explicit budget; mock mode is the default
- Models with per-query cost exceeding `$0.10` are rejected for cost protection
- Prices are fetched live from OpenRouter; `data/models.json` is the local fallback

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
# Create coldkey
btcli wallet create --wallet.name fugal_validator

# Create hotkey
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

## Step 3: Set Up API Key

Get an API key from [OpenRouter](https://openrouter.ai/) and fund the account.

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Step 4: Run the Validator

```bash
python neurons/validator.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_validator \
  --hotkey default \
  --live --epoch-budget 50
```

### Test run (no API spend)

To verify everything works before spending real money:

```bash
python neurons/validator.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_validator \
  --hotkey default \
  --mock \
  --once
```

The default is `--mock`, which uses synthetic responses instead of calling
OpenRouter. `--live` enables paid API calls and requires `--epoch-budget` (a
positive USD ceiling). `--once` exits after a single epoch.

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
Environment=OPENROUTER_API_KEY=sk-or-...
Environment=FUGAL_NETWORK=finney
Environment=FUGAL_NETUID=<NETUID>
ExecStart=/path/to/fugal-subnet/.venv/bin/python neurons/validator.py \
  --netuid <NETUID> --network finney \
  --coldkey fugal_validator --hotkey default
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
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required) |
| `FUGAL_EPOCH_BUDGET` | — | Max API spend per epoch in USD (required with `--live`) |
| `FUGAL_MAX_MODEL_POOL` | `30` | Max models in union pool |
| `FUGAL_MAX_MODELS_PER_MINER` | `30` | Max models counted per miner's declared pool |
| `FUGAL_MAX_MODEL_COST` | `0.10` | Per-query cost cap for callable models (USD) |
| `FUGAL_API_CONCURRENCY` | `4` | Concurrent OpenRouter calls during matrix build |
| `FUGAL_REQUIRE_COMMITMENT` | `1` | Require on-chain head commitment before scoring |
| `FUGAL_SKIP_BENCHMARKS` | — | Comma-separated benchmarks to skip (must match across validators) |
| `FUGAL_LAMBDA` | `2.0` | Cost-quality routing tradeoff |
| `FUGAL_WILSON_CONFIDENCE` | `0.95` | Wilson LCB confidence level (diagnostic) |
| `FUGAL_MAX_WEIGHT_DELTA` | `0.3` | Max weight change per UID per epoch |
| `FUGAL_WALLET_PATH` | — | Bittensor wallet root (defaults to SDK wallet directory) |
| `FUGAL_STATE_PATH` | `results/validator_state.json` | Persisted scoring state (survives restarts) |
| `LOG_LEVEL` | `INFO` | Logging level |

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `test` | Network: `finney`, `test`, `local` |
| `--netuid` | `1` | Subnet netuid |
| `--coldkey` | `default` | Wallet coldkey name |
| `--hotkey` | `default` | Hotkey name |
| `--once` | off | Run one epoch and exit |
| `--live/--mock` | `--mock` | Mock (default) or live (paid OpenRouter calls) |
| `--epoch-budget` | — | Positive USD ceiling (required with `--live`) |
| `--wallet-path` | SDK default | Bittensor wallet root directory |
| `--log-level` | `INFO` | Logging level |

## Monitoring

### Epoch logs

Structured JSONL logs are written to `results/epoch_logs/`. Each entry includes:

- Epoch ID and block hash
- Number of miners queried, valid/invalid heads
- Per-miner scores (accuracy, cost efficiency, KL)
- Weight assignments
- Anomaly flags
- Phase timing breakdown
- Commit-reveal verification status

### Anomaly detection

The validator automatically flags:

- Epochs where no miners (or only one of several) respond
- Weight concentration (>80% of weight on one miner)
- Suspiciously uniform accuracy (all miners >95% or all <10%)
- Commit-reveal verification failures

For cross-validator auditing, `fugal_subnet/consensus.py` compares several
validators' published reveals offline (median consensus, outlier detection,
Kendall-tau rank agreement).

### Process supervision

Use systemd (`Restart=always`) or Docker (`--restart unless-stopped`) to restart
the validator on crashes. The validator persists scoring state to disk, so restarts
resume cleanly.

## Safety Features

- **Pre-flight budget estimate** — before any API call, the estimated epoch cost
  is computed and the most expensive models are dropped until it fits the budget.
- **Budget cap** — API spend per epoch is additionally hard-capped at runtime
  (default $50). `BudgetExceeded` aborts the epoch (no weights set from a
  partial matrix).
- **Priced models only** — models without an OpenRouter price are never called
  (an unpriced model would bypass both the cost cap and the budget tracker).
- **Model cost cap** — models exceeding $0.10 per query are excluded.
- **Model pool caps** — per-miner declared pools capped at 30; if the union
  exceeds 30, models declared by more miners win (deterministic — an
  alphabetical sybil can't evict everyone else's models).
- **Head validation** — size cap, decompressed-size cap (zip bombs), shape and
  finiteness checks, model-count cap; `allow_pickle` is always False.
- **Weight capping** — weights can only change ±0.3 per UID per epoch, preventing
  sudden swings.
- **Commit-reveal** — benchmark questions are committed before miner queries,
  verified and fully published on reveal.
- **State persistence** — scoring state and previous weights survive restarts
  (`results/validator_state.json`).
- **CRW compatibility** — `set_weights()` automatically uses commit-reveal-weights
  when the chain enables it.

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first.

**High API costs** — Lower `FUGAL_EPOCH_BUDGET`, reduce `FUGAL_MAX_MODEL_POOL`, or
increase `FUGAL_EPOCH_INTERVAL` to run epochs less frequently.

**"No valid heads received"** — No miners responded. The epoch is skipped and logged.
This is normal when the subnet is new and has few miners.

**Weight-setting failed** — Check that your validator has enough stake. The error
message in the epoch log will have details.

**Frequent restarts** — Check network connectivity to the subtensor endpoint and
OpenRouter API. Review epoch logs for error details.
