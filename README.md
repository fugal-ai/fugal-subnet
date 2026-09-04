# Fugal Subnet

[![CI](https://github.com/fugal-ai/fugal-subnet/actions/workflows/ci.yml/badge.svg)](https://github.com/fugal-ai/fugal-subnet/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Release](https://img.shields.io/github/v/release/fugal-ai/fugal-subnet)](https://github.com/fugal-ai/fugal-subnet/releases)

A Bittensor subnet for continuously improving cost-aware LLM routing.

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

# Or the single-command container demo (chain + miner + validator, one epoch)
docker compose up --abort-on-container-exit

# Full dress rehearsal: multi-miner, multi-validator, multi-epoch, on a real chain
python scripts/dress_rehearsal.py --scenario all

# Train a reference head (synthetic data, no API spend)
python scripts/train_head.py --synthetic --n-questions 200 \
  --models deepseek/deepseek-v4-flash meta-llama/llama-4-maverick openai/gpt-5.4-nano \
  --output data/my_head.npz

# Run the miner (commits the head hash on-chain, benchmarks, serves proofs)
python neurons/miner.py --netuid 1 \
  --head-path data/my_head.npz \
  --benchmark-pool data/pool.json

# Run the validator (mock mode — no API spend)
python neurons/validator.py --netuid 1 --mock
```

The commands above do not call OpenRouter. Mock mode is the default. Paid
operation requires both `--live` and an explicitly configured positive
`--epoch-budget` (or `FUGAL_EPOCH_BUDGET`); there is no implicit budget or paid
default. Review the [Validator Guide](docs/VALIDATOR_GUIDE.md) first.

### Requirements

- Linux or WSL2 and Python 3.10–3.12
- Docker for the full local-chain testnet
- CPU float32 inference for backbone determinism (CUDA optional for training)

## How It Works

1. **Epochs are aligned to chain blocks**: every `EPOCH_INTERVAL/12` blocks is an epoch boundary. The boundary block's hash seeds a nonce that selects ~300 questions (stratified across benchmarks) — every honest validator derives the identical slice, and neither side can know it in advance.
2. **Miners run the benchmark themselves, inside an Intel TDX enclave.** The enclave loads the miner's `.npz` head, routes each question, calls the chosen model through a metering proxy, grades the reply with the hash-pinned graders, and produces a hardware-attested `BenchmarkProof`. Miners pay for their own inference. A head is only scoreable if its SHA256 was **committed on-chain at or before the boundary block** — a head swapped after the nonce is knowable is rejected.
3. **Miners also answer a nonce-chosen ~5% of extra questions using a model they do not pick.** These are never scored against them; they are the only unbiased samples of how good each model actually is, and a proof missing them is rejected.
4. **Validators never call a model.** They verify the proof: the Intel DCAP signature, the hardware's own measurement registers against the approved runtime image, the attested content hash, the head against its on-chain commitment, the answers against the assigned slice, the exploration set against the nonce, and the cost figures against each other.
5. **Scoring is quality per dollar against the best single model.** `quality = wilson_lcb(accuracy) / acc_best`, `thrift = ref_cost / miner_cost`, and `score = quality^0.8 * thrift^0.2`. A score of 1.0 means "matched the best single model's quality per dollar"; above 1.0 means "beat it". Evidence accumulates per head artifact across epochs, so noise cancels and a lucky epoch does not dethrone. Copied heads are deduplicated — earliest on-chain commitment wins.
6. The validator publishes the full epoch artifact (`results/epochs/<epoch>/reveal.json`): questions, results, the reference frame, scores, and weights. Miners download it, retrain, and commit improved heads.

## Benchmarks

| Benchmark | Questions | Grader |
|-----------|-----------|--------|
| MMLU | 14,042 | letter_mcq |
| MATH | 5,000 | boxed_math |
| GSM8K | 1,319 | numeric_final |
| AIME | 933 | integer_exact |
| IFEval | up to 259 | partial constraint_set |
| GPQA-Diamond | 198 | letter_mcq (gated dataset — needs HF auth) |
| HumanEval | 155 exec_io + 9 legacy fallback | exec_io / exec_unittest |
| LiveCodeBench | optional | exec_io (local JSON cache, see `fugal_subnet/benchmarks/livecode.py`) |

HuggingFace dataset revisions are pinned for reproducibility. GPQA is a gated
dataset requiring `huggingface-cli login` and accepted terms, or add `gpqa` to
`FUGAL_SKIP_BENCHMARKS` to exclude it.

## Guides

- **[Miner Guide](docs/MINER_GUIDE.md)** — train a router head, register, commit, run the miner
- **[Validator Guide](docs/VALIDATOR_GUIDE.md)** — set up API keys, sandboxing, run the validator, monitor epochs
- **[Consensus Invariants](docs/INVARIANTS.md)** — the nine properties the subnet rests on, what enforces each, and the known gaps
- **[Design Decisions](docs/design-decisions.md)** — why the scoring formula, the reference frame and the TEE architecture are what they are
- **[TDX Validation](docs/TDX_VALIDATION.md)** — the two attestation checks that need real confidential hardware, and how to run them
- **[Live API Validation](docs/LIVE_API_VALIDATION.md)** — confirming the pinned price table against what OpenRouter actually bills

## Contributing and Security

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing consensus-critical behavior. Report vulnerabilities privately according to [SECURITY.md](SECURITY.md); never include API keys or wallet material in a public issue.

Release history is recorded in [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
