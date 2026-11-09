# Fugal Subnet

[![CI](https://github.com/fugal-ai/fugal-subnet/actions/workflows/ci.yml/badge.svg)](https://github.com/fugal-ai/fugal-subnet/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Release](https://img.shields.io/github/v/release/fugal-ai/fugal-subnet)](https://github.com/fugal-ai/fugal-subnet/releases)

A Bittensor subnet that finds the best cost-aware LLM router, and pays only
the router that is measurably the best.

A **router** reads a question and picks which model should answer it, so the
user gets the best answer the money can buy. **Miners** train routers and run
the benchmark themselves inside an Intel TDX confidential VM: the enclave
loads the miner's router head (a small linear layer on a frozen Qwen3-0.6B
backbone), routes each question to one of the 18 models in the pinned price
table, pays for the call, grades the reply with hash-pinned checkers, and
produces a hardware-attested proof. **Validators** never call a model: they
verify the proof, compare every head on the same questions, and pay the
champion.

The subnet's output is the champion head, and a continuously refreshed public
record of which model answered which question correctly and at what cost:
the data the next router is trained on.

This README describes **protocol v2**. The subnet runs the v1 rules until the
published activation block (`docs/VALIDATOR_GUIDE.md`, "Protocol v2: flip
procedure"); the differences a miner must act on are in
`docs/MINER_GUIDE.md`, "Moving from v1 to v2".

## Architecture

```
Question → Qwen3-0.6B (frozen) → hidden state → per-model success probability → subtract estimated dollars → select model → call model → answer
                                                        ↑
                                                trainable head (~14 KB .npz)
```

## Quick Start

```bash
# Clone and install (Linux/WSL2, Python 3.11-3.12). CI installs with uv:
git clone https://github.com/fugal-ai/fugal-subnet.git
cd fugal-subnet
uv sync --locked --extra dev          # or: python3 -m venv .venv && pip install -e ".[dev]"

# Full dress rehearsal: the shipped neurons, multi-miner, multi-validator,
# multi-epoch, against a real local chain (Docker), no API spend
uv run python scripts/dress_rehearsal.py --scenario all
# The v2 flip on that chain: v1 epochs, the activation boundary, the first
# seat, a take, a ghost and a resume
uv run python scripts/dress_rehearsal.py --scenario v2_flip

# Provision a subnet on a real network (test or finney) — idempotent
uv run python scripts/provision_subnet.py --network test --netuid <N> \
  --owner-wallet <wallet> --status

# Watch a live subnet and assert it is working, rather than reading logs
uv run python scripts/watch_subnet.py --network test --netuid <N> --epochs 3 \
  --miners m1=<hotkey> --validators v1=<hotkey>,v2=<hotkey>

# A deterministic OpenAI-compatible endpoint, so miners run with no spend
uv run python scripts/stub_upstream.py --pool <pool.json> --port 8799

# Train a v2 head over the v2 price table (test-only synthetic labels, no spend)
uv run python scripts/train_head.py --synthetic --n-questions 200 \
  --prices data/models_v2.json \
  --models anthropic/claude-haiku-5.5 deepseek/deepseek-v4.1-flash openai/gpt-6.1-terra \
  --output data/synthetic_head_test.npz

# Train a candidate from observed labels (see docs/MINER_GUIDE.md, Step 3)
uv run python scripts/train_head.py --matrix data/matrix.npz \
  --prices data/models_v2.json --manifest data/benchmark_tokens_protocol_v2.json \
  --models anthropic/claude-haiku-5.5 deepseek/deepseek-v4.1-flash openai/gpt-6.1-terra \
  --output data/my_head.npz

# Run the miner against a local or test chain (commits the head hash
# on-chain, benchmarks, serves proofs). Mock is the miner's default.
uv run python neurons/miner.py --network test --netuid <N> --head-path data/my_head.npz

# Run the validator in mock mode on a test chain (no API spend). The
# validator defaults to --live, and both neurons refuse --mock on finney.
uv run python neurons/validator.py --network test --netuid <N> --mock
```

The commands above do not call OpenRouter as long as `OPENROUTER_API_KEY` is
unset: `--mock` replaces the attestation, not the model upstream, so a mock
miner with a key in its environment spends real money. A validator has no API
key and no budget at all: it verifies proofs, it never calls a model. The only
paid path is a miner running `--live` with an OpenRouter key delivered over
the attested provisioning channel; see the [Miner Guide](docs/MINER_GUIDE.md).

### Requirements

- Linux or WSL2 and Python 3.11–3.12
- Docker for the full local-chain testnet
- CPU float32 for the shared embedding profile and reference trainer

## How It Works

1. **Epochs are aligned to chain blocks.** Every `EPOCH_INTERVAL/12` blocks
   is an epoch boundary (mainnet: tempo 360, so 360 blocks, 72 minutes). The boundary block's hash
   seeds a nonce that selects 300 questions from the pinned pool, filling the
   published category quotas (40% knowledge, 30% hard, 25% math, 5%
   instruction). Every honest validator derives the identical slice, and
   neither side can know it in advance.
2. **Miners run the benchmark themselves, inside an Intel TDX enclave.** The
   enclave loads the miner's `.npz` head, routes each question, calls the
   chosen model through a metering proxy, grades the reply with the
   hash-pinned graders, and produces a hardware-attested `BenchmarkProof`.
   Miners pay for their own inference. A head is only scoreable if its SHA256
   was **committed on-chain at or before the boundary block**: a head swapped
   after the nonce is knowable is rejected.
3. **Miners also answer a nonce-chosen 5% of extra questions using a model
   they do not pick.** These are never scored against them; they are the
   only unbiased samples of how good each model actually is, and a proof
   missing them is rejected.
4. **Validators never call a model.** They verify the proof: the Intel DCAP
   signature, the hardware's own measurement registers against the approved
   runtime image, the attested content hash, the head against its on-chain
   commitment, the answers against the assigned slice, the exploration set
   against the nonce, and the cost figures against each other. Every
   validator then holds the same per-question grade for every head: the
   **shared grades**, published in the reveal.
5. **One champion seat.** Heads are compared pairwise on the questions they
   both answered over a 48-epoch window: accuracy gained, valued at a pinned
   dollar rate, minus the extra dollars spent (hard questions count double).
   A challenger takes the seat only when the lower confidence bound of that
   net value over the champion is positive by the published margin, after
   completing burn-in (1,500 scored questions). A live champion keeps the
   seat with no margin, so two near-equal heads cannot trade it on noise. A
   copy of the champion is never measurably better than it, so copies earn
   nothing; the senior head by first on-chain commit wins ties. A champion
   that goes offline leaves a ghost bar for three epochs and resumes its seat
   if it returns. Exact rules: `docs/INVARIANTS.md`, "Protocol v2".
6. **Payout is 85 / 10 / 5.** The champion takes 85% of the miner share, the
   two champions before it 10% and 5%, provided they are still live and above
   the accuracy floor; a seat nobody holds burns. Nothing that pays reads the
   reference frame or any model's estimated accuracy: only the shared grades
   and the pinned prices.
7. **Everything is published.** Each epoch the validator commits a hash
   on-chain and then reveals the full artifact (`reveal.json`: questions,
   every head's grades and spend, the seat decision and its comparison
   detail, the weights). A validator that lost its state rebuilds the window
   from the published reveals. Miners download them, retrain, and commit
   improved heads.

## Benchmarks

The pinned v2 pool (`fugal_subnet/benchmarks/pool_manifest_v2.json`) is
**28,986 questions** in four categories:

| Category | Target | Benchmark | Questions | Grader |
|----------|-------:|-----------|----------:|--------|
| knowledge | 40% | MMLU-Pro | 12,032 | letter_mcq |
| | | MMLU | 3,000 | letter_mcq |
| math | 25% | MATH (levels 3–5) | 3,669 | boxed_math |
| | | Omni-MATH (mid) | 1,117 | boxed_math |
| hard | 30% | SuperGPQA (hard) | 7,050 | letter_mcq |
| | | AIME 1983–2024 | 932 | integer_exact |
| | | AIME 2025 | 30 | integer_exact |
| | | BeyondAIME | 100 | integer_exact |
| | | Omni-MATH (hard) | 845 | boxed_math |
| instruction | 5% | IFEval | 211 | constraint_set |

Fifty source items are left out because the pinned grader cannot judge them as
intended: 48 IFEval items whose bullet-list or JSON checks fail compliant
answers, one AIME item whose gold is "080 or 081", and one Omni-MATH item
whose gold is beyond 2^53. The v1 pool is unchanged.

Hugging Face dataset revisions are pinned for reproducibility, and the pool's
content hash is checked at startup. **Do not set `FUGAL_SKIP_BENCHMARKS` on
any neuron**: a neuron with a different pool fails every proof on
`questions_hash`. The v1 pool (`pool_manifest.json`, 21,553 questions) stays
in the repo until the activation block.

## Guides

- **[Miner Guide](docs/MINER_GUIDE.md)** — train a router head, register, commit, run the miner, and what changes at the v2 flip
- **[Validator Guide](docs/VALIDATOR_GUIDE.md)** — register, load the approved entries, run as a service, monitor epochs (no API key), the flip procedure
- **[Mainnet Launch](docs/MAINNET_LAUNCH.md)** — the operational sequence, and the things that fail silently if you skip them
- **[Consensus Invariants](docs/INVARIANTS.md)** — the properties the subnet rests on, what enforces each, and the known gaps
- **[Decision records](docs/decisions/README.md)** — why the design is what it is: the design steps (Step 8 is protocol v2), the scoring decision, the experiments, the economics model, the first hardware rehearsal
- **[TDX Validation](docs/TDX_VALIDATION.md)** — the attestation checks that need real confidential hardware, and how to run them
- **[Live API Validation](docs/LIVE_API_VALIDATION.md)** — confirming the pinned price table against what OpenRouter actually bills

## Contributing and Security

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing consensus-critical behavior. Report vulnerabilities privately according to [SECURITY.md](SECURITY.md); never include API keys or wallet material in a public issue.

Release history is recorded in [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
