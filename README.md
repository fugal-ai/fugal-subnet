# Fugal Subnet

[![CI](https://github.com/fugal-ai/fugal-subnet/actions/workflows/ci.yml/badge.svg)](https://github.com/fugal-ai/fugal-subnet/actions/workflows/ci.yml)
[![CodeQL](https://github.com/fugal-ai/fugal-subnet/actions/workflows/codeql.yml/badge.svg)](https://github.com/fugal-ai/fugal-subnet/actions/workflows/codeql.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Release](https://img.shields.io/github/v/release/fugal-ai/fugal-subnet)](https://github.com/fugal-ai/fugal-subnet/releases)

A Bittensor subnet for continuously improving cost-aware LLM routing.

> [!IMPORTANT]
> **Project status: experimental and pre-launch.** The default `test` network and netuid `1` are development defaults, not an announced mainnet deployment. APIs, schemas, and economics may change before launch.

**Validators** build ground truth matrices — calling frontier models on benchmark questions, grading responses with mechanical checkers. **Miners** submit trained router heads — small linear layers (~10K-73K params) on a frozen Qwen3-0.6B backbone — that route any question to the optimal model for the cheapest price.

The subnet's core output is a continuously refreshed ground-truth matrix that miners can use to train and improve routing policies.

## Architecture

```
Question → Qwen3-0.6B (frozen) → hidden state → W·h + b → softmax → select model → call model → answer
                                                  ↑
                                          trainable head (~14KB .npz)
```

## Quick Start

```bash
# Clone and install (Linux/WSL2, Python 3.10-3.12)
git clone https://github.com/fugal-ai/fugal-subnet.git
cd fugal-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Local testnet (Docker chain + full epoch pipeline, no API spend)
python scripts/launch_testnet.py --mock --epochs 3

# Train a reference head (synthetic data, no API spend)
python scripts/train_head.py --synthetic --n-questions 200 \
  --models deepseek/deepseek-v4-flash meta-llama/llama-4-maverick openai/gpt-5.4-nano \
  --output data/my_head.npz

# Run the miner (commits the head hash on-chain, then serves it)
python neurons/miner.py --netuid 1 --head-path data/my_head.npz

# Run the validator (mock mode — no API spend)
python neurons/validator.py --netuid 1 --mock
```

The commands above do not call OpenRouter. Running a validator without `--mock` makes paid API requests and uses a default maximum epoch budget of `$50`; review `FUGAL_EPOCH_BUDGET` and the [Validator Guide](docs/VALIDATOR_GUIDE.md) first.

### Requirements

- Linux or WSL2 and Python 3.10–3.12
- Docker for the full local-chain testnet
- CPU execution is supported; CUDA is optional and accelerates backbone inference
- Model, benchmark, and Docker storage requirements vary; hardware sizing has not yet been formally benchmarked

## How It Works

1. **Epochs are aligned to chain blocks**: every `EPOCH_INTERVAL/12` blocks is an epoch boundary. The boundary block's hash seeds a nonce that selects ~300 questions (stratified across benchmarks) — every honest validator gets the identical slice.
2. Miners respond to the validator's query with their `.npz` head artifact (W, b, model list). A head is only scoreable if its SHA256 was **committed on-chain at or before the boundary block** — heads swapped after the nonce is knowable are rejected.
3. The validator calls all models in the (priced, capped, budget-checked) union pool on those questions, grades responses with mechanical checkers, and builds an N×M binary matrix.
4. Each head is evaluated: routing accuracy, cost efficiency (capped at 1.0 — the oracle's cheapest-correct routing is the ceiling), and KL divergence against soft targets. Questions no model answered correctly are excluded for everyone.
5. Composite scores (accuracy 55%, cost efficiency 35%, KL 10%) determine weights. Copied heads are deduplicated — earliest on-chain commitment wins. Weights are set on-chain; emissions flow.
6. The validator publishes the full epoch artifact (`results/epochs/<epoch>/reveal.json`): questions, the complete matrix, model costs, scores, and weights. Miners download it, retrain, submit improved heads.

## Training Pipeline

Two-stage:

- **Stage 1 — SFT**: KL divergence loss against soft target distributions (AdamW on W, b only)
- **Stage 2 — sep-CMA-ES**: Derivative-free evolutionary refinement on actual routing fitness

## Benchmarks

| Benchmark | Questions | Grader |
|-----------|-----------|--------|
| MMLU | 14,042 | letter_mcq |
| MATH | 5,000 | boxed_math |
| GSM8K | 1,319 | numeric_final |
| AIME | 933 | integer_exact |
| IFEval | 259 | constraint_set |
| GPQA-Diamond | 198 | letter_mcq (gated dataset — needs HF auth) |
| HumanEval | 164 | exec_io |
| LiveCodeBench | optional | exec_io (local JSON cache, see `fugal_subnet/benchmarks/livecode.py`) |

All HuggingFace datasets are loaded at **pinned revisions** so every validator builds a byte-identical pool.

## Guides

- **[Miner Guide](docs/MINER_GUIDE.md)** — train a router head, register, commit, run the miner
- **[Validator Guide](docs/VALIDATOR_GUIDE.md)** — set up API keys, sandboxing, run the validator, monitor epochs
- **[Consensus Governance](docs/CONSENSUS.md)** — versioning and coordinated protocol upgrades
- **[Threat Model](docs/THREAT_MODEL.md)** — trust boundaries, controls, and known limitations

## Contributing and Security

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing consensus-critical behavior. Report vulnerabilities privately according to [SECURITY.md](SECURITY.md); never include API keys or wallet material in a public issue.

Release history is recorded in [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
