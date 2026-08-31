# Fugal Subnet

A Bittensor subnet that continuously produces the best LLM router.

**Validators** build ground truth matrices — calling frontier models on benchmark questions, grading responses with mechanical checkers. **Miners** submit trained router heads — small linear layers (~10K-73K params) on a frozen Qwen3-0.6B backbone — that route any question to the optimal model for the cheapest price.

The subnet's core value is the continuously-refreshed ground truth matrix. Everything downstream (training, architecture, optimization) is a known-solved step.

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

## Project Structure

```
fugal-subnet/
├── AGENTS.md                   # AI agent instructions for working on this codebase
├── LICENSE                     # MIT
├── docker-compose.yml          # Local testnet (subtensor v432)
├── pyproject.toml
├── fugal_subnet/
│   ├── config.py               # Env-overridable constants
│   ├── protocol.py             # FugalSynapse wire format
│   ├── backbone.py             # Qwen3-0.6B hidden state extraction
│   ├── graders.py              # 7 mechanical checkers (consensus-critical)
│   ├── api.py                  # OpenRouter API client + spend tracking
│   ├── matrix.py               # Ground truth matrix construction
│   ├── soft_targets.py         # Softmax distributions for KL training
│   ├── head_eval.py            # Head evaluation (accuracy, cost, KL)
│   ├── scoring.py              # Raw epoch composite scoring
│   ├── rewards.py              # Single-pool weight computation
│   ├── dedup.py                # Behavioral dedup (cosine clustering)
│   ├── commitments.py          # On-chain head-hash commitments
│   ├── commit_reveal.py        # Commit-reveal epoch integrity + artifacts
│   ├── consensus.py            # Multi-validator consensus audit tool
│   ├── epoch_logger.py         # Structured JSONL epoch logs
│   ├── benchmarks/             # 8 benchmark loaders (pinned HF revisions)
│   └── attacks/                # 22-case adversarial grader test suite
├── neurons/
│   ├── validator.py            # Epoch loop, head eval, weight-setting
│   └── miner.py                # Axon server, head commitment + submission
├── scripts/
│   ├── train_head.py           # SFT + sep-CMA-ES head trainer
│   ├── launch_testnet.py       # End-to-end local testnet (Docker chain)
│   ├── setup_local_testnet.py  # docker-compose entrypoint
│   └── test_real_api.py        # Real-API proof-of-concept (budget-capped)
├── tests/
│   ├── bt_mock.py              # Bittensor mock for unit tests
│   └── test_integration.py     # End-to-end + security/incentive tests
├── docs/
│   ├── MINER_GUIDE.md          # Train a head, register, run the miner
│   └── VALIDATOR_GUIDE.md      # API keys, sandboxing, run the validator
└── data/
    └── models.json             # Fallback price sheet (live prices from OpenRouter)
```

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

## Why This Works

**Model specialization is real.** Different models excel at different tasks — math, code,
reasoning, creative writing. A single "best model" doesn't exist. The best model depends
on the question.

**Routers rot.** New models ship monthly, prices change, fine-tunes appear. A static router
has a shelf life. The subnet makes the router self-sustaining through economic incentives.

**The training method is solved.** Sakana's TRINITY (ICLR 2026) proved the frozen backbone +
lightweight head architecture and the SFT + sep-CMA-ES training pipeline. Fugu surpassed
GPT-5.5, Claude Opus 4.8, and Gemini 3.1 Pro through routing alone. The bottleneck is
producing fresh ground truth — which is exactly what this subnet does every epoch.

## Evolution Path

- **v1** — Single-model routing (current). Head picks one model per question.
- **v2** — Role-augmented routing (TRINITY-style). Head picks model AND role.
- **v3** — Multi-step orchestration (Conductor-style). Full agentic workflows.
- **v4** — Recursive orchestration. Orchestrator calls itself as a worker.

## Guides

- **[Miner Guide](docs/MINER_GUIDE.md)** — train a router head, register, commit, run the miner
- **[Validator Guide](docs/VALIDATOR_GUIDE.md)** — set up API keys, sandboxing, run the validator, monitor epochs

## License

MIT — see [LICENSE](LICENSE).
