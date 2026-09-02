# Fugal Subnet — Miner Guide

Get started mining on the Fugal subnet. You'll train a router head and serve it
to validators, earning TAO emissions based on how well your head routes questions
to the best model for the cheapest price.

## What You're Building

A **router head** — a small linear layer (~14KB `.npz` file) that sits on top of
a frozen Qwen3-0.6B backbone. Given a question's hidden state, your head picks
which LLM should answer it. The validator scores your head on:

- **Accuracy** — did your chosen model get the question right?
- **Cost efficiency** — did you route to a cheaper model when an expensive one wasn't needed?
- **Distribution match** — does your routing distribution match the optimal soft targets?

Better heads earn more emissions.

## Requirements

- Linux or WSL2
- Python 3.10-3.12
- GPU recommended for training (CPU works but slower)
- ~4GB disk for dependencies + backbone model
- TAO for subnet registration

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
# Create a coldkey (stores your TAO — keep this safe)
btcli wallet create --wallet.name fugal_miner

# Create a hotkey (used for mining operations)
btcli wallet create --wallet.name fugal_miner --wallet.hotkey default --type hotkey
```

## Step 2: Register on the Subnet

You need some TAO in your coldkey to pay the registration fee.

```bash
# Check registration cost
btcli subnet info --netuid <NETUID> --network finney

# Register
btcli subnet register --netuid <NETUID> --wallet.name fugal_miner --wallet.hotkey default --network finney
```

Replace `<NETUID>` with the Fugal subnet's netuid (announced on launch).

## Step 3: Train a Head

### Quick start (synthetic data, no API cost)

```bash
python scripts/train_head.py \
  --synthetic --n-questions 300 \
  --models openai/gpt-5.4-mini anthropic/claude-haiku-4.5 deepseek/deepseek-v4-flash \
  --output data/my_head.npz
```

This trains on synthetic data to verify everything works. For a competitive head,
train on real matrix data from validator-published epoch artifacts.

### Competitive training (with matrix data)

Once the subnet is running, validators publish ground truth matrices. Download one
and train against it:

```bash
python scripts/train_head.py \
  --matrix data/matrix.npz \
  --models openai/gpt-5.4-mini anthropic/claude-haiku-4.5 deepseek/deepseek-v4-flash \
  --output data/my_head.npz \
  --device cuda \
  --sft-epochs 100 \
  --cma-generations 50
```

### Training stages

1. **SFT (Stage 1)** — KL divergence loss against soft target distributions. AdamW
   optimizer on W and b only. Takes seconds to minutes.
2. **sep-CMA-ES (Stage 2)** — Derivative-free evolutionary refinement on actual routing
   fitness. Takes minutes. Skip with `--skip-cma` for faster iteration.

### Model selection strategy

Your head declares which models it can route to. This is a strategic choice:

- **More models** = more routing options, but harder to learn a good mapping
- **Cheaper models** = better cost efficiency score (35% of composite weight)
- **Expensive models** = better accuracy on hard questions (55% of composite weight)
- All miners compete in a single pool ranked by composite score (accuracy 55%, cost efficiency 35%, KL divergence 10%)
- Declare only models that exist on OpenRouter **with a listed price** — the
  validator fetches prices at runtime and will not call (or score routes to)
  unpriced models. Models above the per-query cost cap (default $0.10) are
  also excluded, and each miner's declared pool is capped (default 30 models).

### Head format

The `.npz` file must contain:

| Array | Shape | Description |
|-------|-------|-------------|
| `W` | `(L, 1024)` | Weight matrix, float32. L = number of models |
| `b` | `(L,)` | Bias vector, float32 |
| `models` | `(L,)` | Model ID strings (e.g. `openai/gpt-5.4-mini`) |

Max file size: 1MB. Hidden dimension must be 1024 (Qwen3-0.6B).

## Step 4: Run the Miner

```bash
python neurons/miner.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_miner \
  --hotkey default \
  --wallet-path /path/to/.bittensor/wallets \
  --head-path data/my_head.npz \
  --port 8091
```

The miner runs an axon server that responds to validator queries. Each epoch
(~1 hour), the validator queries all miners for their heads, evaluates them against
a fresh ground truth matrix, and sets weights.

### On-chain head commitment (important)

At startup the miner automatically commits `sha256(head bytes)` on-chain via
the Commitments pallet. **Validators only score a head whose hash was
committed at or before the epoch boundary block** — this is what stops
head-copying and last-second slice overfitting. Practical consequences:

- After starting (or swapping in a new head), your head becomes scoreable
  from the **next** epoch boundary after the commitment lands. Expect to skip
  at most one epoch after every head update.
- If the commit fails (e.g. chain congestion), the miner retries every 30s
  and logs it. Your head is served but not scored until the commitment lands.

### Axon access control

By default any registered hotkey on the subnet may query your head. Set
`FUGAL_MIN_VALIDATOR_STAKE` (TAO) to only serve hotkeys holding a validator
permit or at least that much stake — recommended on mainnet so competitors
can't trivially pull your head with a throwaway registration.

### Running as a service

For production, run the miner as a systemd service so it stays up:

```bash
sudo tee /etc/systemd/system/fugal-miner.service > /dev/null <<'EOF'
[Unit]
Description=Fugal Subnet Miner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/fugal-subnet
ExecStart=/path/to/fugal-subnet/.venv/bin/python neurons/miner.py \
  --netuid <NETUID> --network finney \
  --coldkey fugal_miner --hotkey default \
  --head-path data/my_head.npz --port 8091
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now fugal-miner
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `test` | Network: `finney`, `test`, `local` |
| `--netuid` | `1` | Subnet netuid |
| `--coldkey` | `default` | Wallet coldkey name |
| `--hotkey` | `default` | Hotkey name |
| `--wallet-path` | SDK default | Optional explicit Bittensor wallet root |
| `--port` | `8091` | Axon port |
| `--head-path` | (required) | Path to `.npz` head file |
| `--log-level` | `INFO` | Logging level |

## Updating Your Head

To improve your head, retrain on newer matrix data and restart the miner with the
new `.npz` file. The restart re-commits the new hash on-chain; the new head is
scored from the next epoch boundary after the commitment lands.

Training data: every epoch the validator publishes
`results/epochs/<epoch_id>/reveal.json` containing the full question slice,
the N×M ground truth matrix, model costs, submitted heads, scores, and weights.
Use `scripts/train_head.py` to retrain against this data.

## Scoring Details

Each epoch, your head is evaluated on ~300 nonce-selected benchmark questions,
stratified across benchmarks (from a pool of ~21,900). Your score combines:

- **Accuracy** (55%) — did your routed model get it right? (Questions no model
  answered correctly are excluded for everyone.)
- **Cost efficiency** (35%) — oracle cost ÷ your routing cost, capped at 1.0.
  1.0 means you routed as cheaply as the oracle's cheapest-correct choice.
- **KL divergence** (10%) — how well your routing distribution matches the
  optimal soft targets.

Raw epoch scores are used — no smoothing. Weight capping (±0.3 per epoch per UID)
provides stability and prevents sudden swings.

## Anti-Gaming

- **On-chain commitment** — only heads whose hash was committed before the epoch
  boundary are scored. You cannot train on the epoch's questions (the nonce
  isn't knowable before the boundary block), and you cannot copy another
  miner's head mid-epoch.
- **Behavioral dedup** — identical or near-identical heads (cosine similarity >0.99
  on routing decisions) are clustered; only the head with the **earliest on-chain
  commitment block** keeps the weight. Copies are disqualified.
- **Commit-reveal** — the validator commits the benchmark slice + grader hash
  before querying miners and publishes everything after grading; anyone can
  re-derive the scores.
- **Liveness** — miss 3 consecutive epochs and you're excluded from weights
  until you respond again.

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first (`btcli subnet register`).

**"Head file too large"** — Max 1MB. Reduce the number of models in your head.

**Port already in use** — Change `--port` to a different value.

**"head rejected: hash not committed"** (in validator logs) — Your head's
on-chain commitment hasn't landed, or landed after the epoch boundary. Keep the
miner running; it retries the commit automatically and the head is scored from
the next epoch.

**Low scores** — Retrain with more data, try different model selections. Balance
accuracy against cost efficiency — both factor into your composite score.
