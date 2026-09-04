# Fugal Subnet — Miner Guide

Run a TEE miner on the Fugal subnet. Your miner runs benchmarks inside an Intel
TDX confidential VM each epoch, producing hardware-attested proofs of routing
quality. Better routers earn more emissions.

## What You're Building

A **router head** — a small linear layer (~14KB to ~130KB `.npz` file, depending
on model count) on a frozen
Qwen3-0.6B backbone. Given a question's hidden state, your head picks which LLM
should answer it. Each epoch, your miner:

1. Receives a nonce from the validator (derived from the chain block hash)
2. Selects the epoch's question slice (deterministic, ~300 questions)
3. Loads your head, computes backbone embeddings, runs routing decisions
4. Calls the routed models via a metered proxy (you pay for inference)
5. Grades responses against gold answers
6. Produces a hardware-attested proof (TDX attestation)
7. Publishes the proof for validators to verify

**You pay for your own inference.** The metered proxy inside the TEE records
exact token counts and costs. Validators verify proofs — they never call models.

## Requirements

- **Intel TDX VM** — GCP `c3-standard` or Azure confidential VMs
  (required for hardware attestation in `--live` mode; `--mock` works anywhere)
- Linux (Ubuntu 22.04+ recommended)
- Python 3.10-3.12
- GPU recommended for training heads (CPU works but slower)
- ~4GB disk for dependencies + backbone model
- TAO for subnet registration
- OpenRouter API key (for model inference inside TEE)

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
btcli wallet create --wallet.name fugal_miner
btcli wallet create --wallet.name fugal_miner --wallet.hotkey default --type hotkey
```

## Step 2: Register on the Subnet

```bash
btcli subnet register \
  --netuid <NETUID> \
  --wallet.name fugal_miner \
  --wallet.hotkey default \
  --network finney
```

## Step 3: Train a Head

### Quick start (synthetic data, no API cost)

```bash
python scripts/train_head.py \
  --synthetic --n-questions 300 \
  --models openai/gpt-5.4-mini anthropic/claude-haiku-4.5 deepseek/deepseek-v4-flash \
  --output data/my_head.npz
```

### Competitive training (with matrix data)

Once the subnet is running, download published epoch artifacts and train:

```bash
python scripts/train_head.py \
  --matrix data/matrix.npz \
  --models openai/gpt-5.4-mini anthropic/claude-haiku-4.5 deepseek/deepseek-v4-flash \
  --output data/my_head.npz \
  --device cuda \
  --use-backbone \
  --sft-epochs 100 \
  --cma-generations 50
```

`--use-backbone` is required. Without it, the trainer falls back to random
hidden states and produces a head that scores near zero.

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
OPENROUTER_API_KEY=sk-or-... python neurons/miner.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_miner \
  --hotkey default \
  --head-path data/my_head.npz \
  --benchmark-pool data/benchmark_pool.json \
  --port 8091 \
  --mock
```

### Test run (mock mode)

`--mock` (the default) runs without real TDX attestation — useful for testing
on any hardware. Everything works the same except the attestation quote is
synthetic. Validators in mock mode accept these proofs.

### Live mode (requires TDX VM)

```bash
OPENROUTER_API_KEY=sk-or-... python neurons/miner.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_miner \
  --hotkey default \
  --head-path data/my_head.npz \
  --benchmark-pool data/benchmark_pool.json \
  --port 8091 \
  --live
```

`--live` produces real TDX attestation. Requires an Intel TDX-capable VM.

### Running as a service

```bash
sudo tee /etc/systemd/system/fugal-miner.service > /dev/null <<'EOF'
[Unit]
Description=Fugal Subnet Miner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/fugal-subnet
Environment=OPENROUTER_API_KEY=sk-or-...
ExecStart=/path/to/fugal-subnet/.venv/bin/python neurons/miner.py \
  --netuid <NETUID> --network finney \
  --coldkey fugal_miner --hotkey default \
  --head-path data/my_head.npz \
  --benchmark-pool data/benchmark_pool.json \
  --port 8091 --live
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
| `--wallet-path` | SDK default | Bittensor wallet root directory |
| `--port` | `8091` | Axon port |
| `--head-path` | (required) | Path to `.npz` head file |
| `--benchmark-pool` | (required) | Path to benchmark question pool |
| `--mock/--live` | `--mock` | Mock (default) or live TDX attestation |
| `--log-level` | `INFO` | Logging level |

## Updating Your Head

Retrain on newer data and restart the miner with the new `.npz` file. The
restart re-commits the new hash on-chain.

**Important:** Recommitting a new head resets your evidence accumulator. Your
accumulated score drops to zero and must rebuild over epochs. Retrain in big
steps, not continuous tweaks. This is by design — it makes dethroning expensive
and rankings stable.

## Scoring

You are scored on **quality per dollar, against the best single model**:

```
quality = wilson_lcb(your accuracy) / accuracy of the best single model
thrift  = what the best model would have cost / what you actually spent
score   = quality^0.8 * thrift^0.2
```

**A score of 1.0 means you matched the best single model's quality per dollar.
Above 1.0 means you beat it.** That is the whole product: same answers, less
money.

Two consequences worth internalising:

- **Neither axis rescues the other.** Routing everything to the cheapest model
  scores badly (quality collapses). Routing everything to the best model scores
  badly (thrift collapses). There is no weighting you can exploit — the score is
  a product, not a sum.
- **Quality is weighted heavier than cost**, deliberately. Giving up 40% of
  quality does not pay for itself even at a 6x saving. The exponent is derived
  from that requirement, not picked.

The reference is the best model's *measured* accuracy, pooled from exploration
samples across all miners and many epochs. It is a fact about the model pool,
not about the current field — how many other miners are online does not move it.

Results are pooled across epochs via **evidence accumulation** (EWMA decay with
Wilson LCB scoring). Your score stabilizes over time — a few lucky epochs won't
rocket you to the top, and a few bad ones won't destroy you. Consistent quality
wins.

**Burn-in:** a freshly committed head ramps in over ~3000 scored questions
(roughly 10 epochs). This is what stops a miner washing a bad record by
recommitting: recovering costs the same evidence that earning the position did.

**Exploration quota:** each epoch you also answer ~5% extra questions using a
model the *nonce* chooses, not your head. These never count toward your
accuracy or your cost — a forced random route is not a penalty — but a proof
missing them, or routing them anywhere other than the assigned model, is
rejected. Budget for the ~5%.

**Miss = 0 accounting:** If your miner misses an epoch (offline, timeout, proof
verification fails), that epoch counts as 0 correct out of n_expected. You
cannot selectively skip bad epochs.

## Anti-Gaming

- **TEE attestation** — results are hardware-attested. You cannot fabricate or
  tamper with proofs after attestation.
- **Measurement pinning** — validators check the TDX quote's own measurement
  registers (MRTD, RTMR0-2) against the approved image list, not any field your
  code writes about itself. Running a modified harness on genuine TDX hardware
  produces a valid quote and an unapproved measurement.
- **Network confinement** — inside the TEE, the benchmark process can only
  communicate with the local MeteringProxy. No data exfiltration.
- **On-chain commitment** — your head hash is committed before benchmarks run.
  Prevents mid-epoch head swaps.
- **Behavioral dedup** — identical or near-identical routing behavior is
  clustered; earliest on-chain commitment wins. Copies are disqualified.
  Routing decisions are compared in a global model index space, so perturbing
  one question to renumber your own model list does not evade it.
- **Evidence accumulation** — miss=0 prevents selective publication.

## Costs

You pay for model inference each epoch. Cost depends on:

- How many questions are in the slice (~300 per epoch)
- Which models your head routes to (cheaper models = lower cost)
- Token counts per question

Typical epoch cost: varies by model selection. The MeteringProxy inside the
TEE records exact costs, which are included in the attested proof.

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first.

**"Head file too large"** — Max 1MB. Reduce the number of models.

**Port already in use** — Change `--port`.

**TDX attestation fails** — Ensure you're running on an Intel TDX-capable VM.
Check that `/usr/bin/tdx-quote-generator` is available.

**Low scores** — Retrain with more data, try different model selections.
Evidence accumulation means scores improve with consistency over epochs.
