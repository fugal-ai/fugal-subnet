# AGENTS.md — AI Agent Instructions

Instructions for AI coding agents working on this codebase.

## Project Overview

Fugal is a Bittensor subnet for continuous LLM router optimization. Validators
build ground truth matrices (calling models, grading with mechanical checkers),
miners submit trained router heads (linear layers on a frozen Qwen3-0.6B backbone),
and the subnet sets on-chain weights proportional to routing quality.

## Critical Constraints

These are non-negotiable. Violating any of them breaks the subnet:

1. **NO `from __future__ import annotations` in `neurons/miner.py`
   or `fugal_subnet/protocol.py`.**
   Bittensor SDK v10.x `bt.Axon.attach()` inspects type annotations at runtime.
   Deferred annotations (PEP 563) break this introspection and the axon silently
   fails to register the forward function. Other files are fine.

2. **All `np.load()` calls MUST use `allow_pickle=False`.**
   Miners submit `.npz` files. `allow_pickle=True` on untrusted data is remote
   code execution. Grep for `np.load` before committing — every instance must
   have `allow_pickle=False`.

3. **Every Axon-attached Synapse `deserialize()` MUST return `self`, not a dict.**
   Bittensor's `dendrite.call()` calls `synapse.deserialize()`. If it returns a
   dict, the validator receives dicts instead of Synapse objects and attribute
   access breaks silently. The mock in `tests/bt_mock.py` mirrors this contract.

4. **NO API credit spend without explicit user approval.**
   Any code path that calls OpenRouter costs real money. Tag costs in comments
   as `[PAID ~$X]`. Never auto-run real API calls in tests or scripts without
   `--mock` being the default.

5. **NEVER print or log any part of `OPENROUTER_API_KEY`.**
   Not even the last 4 characters. Log "Key is set" or "Key not set", nothing more.

## SDK Version

Bittensor SDK v10.x (10.0.0 - 10.x). Key behaviors:

- `bt.Wallet` (capital W), `bt.Subtensor`, `bt.Dendrite`, `bt.Axon`
- `dendrite.query()` returns a list of Synapse objects (one per axon)
- `subtensor.set_weights()` auto-uses commit-reveal-weights when the chain enables it
- `subtensor.query_map()` for reading on-chain storage (used by commitments.py)
- Wallet creation: `bt.Wallet(name=coldkey, hotkey=hotkey, path=wallet_path)`

## Architecture — What Each File Does

### Core pipeline (execution order per epoch)
1. `benchmarks/loader.py` + individual loaders → load question pool (pinned HF revisions)
2. `benchmarks/slicer.py` → HMAC-seeded stratified question selection
3. `protocol.py` → FugalSynapse wire format for miner queries
4. `commitments.py` → read/write on-chain head commitments (Commitments pallet)
5. `api.py` → call models via OpenRouter (thread-safe SpendTracker)
6. `matrix.py` → build N×M ground truth matrix (concurrent API, sequential grading)
7. `graders.py` → 7 deterministic mechanical checkers (consensus-critical, hash-versioned)
8. `soft_targets.py` → softmax distributions from matrix for KL training signal
9. `head_eval.py` → load + validate + evaluate heads against matrix
10. `scoring.py` → composite scoring (accuracy 55%, cost efficiency 35%, KL 10%)
11. `rewards.py` → weight computation (single pool, weight capping ±0.3/epoch)
12. `dedup.py` → behavioral dedup (cosine similarity on routing decisions)
13. `commit_reveal.py` → commit-reveal integrity + publish epoch artifacts

### Orchestrators
- `neurons/validator.py` → main epoch loop (block-aligned, state-persistent)
- `neurons/miner.py` → axon server with blacklist + on-chain commitment

### Support
- `config.py` → all env-overridable constants (single source)
- `backbone.py` → Qwen3-0.6B hidden state extraction (float32 on CPU, float16 on CUDA)
- `consensus.py` → offline multi-validator audit tool (not used in the live loop)
- `epoch_logger.py` → structured JSONL logging

## Graders Are Consensus Rules

`graders.py` checkers are NOT ordinary code. They are consensus rules — every
validator must produce byte-identical grades. A grader change is a consensus fork.

- Never "improve" a grader casually
- A semantic change requires a new grader version (applies from next epoch only)
- The grader version is `sha256(graders.py bytes)`, pinned in `scripts/check_safety_invariants.py`
- Run `python -m fugal_subnet.attacks.run_attacks` after any grader change — all 22 cases must pass with 0 surprises

## Testing

```bash
# Unit + integration tests (no API spend, no chain needed)
python tests/test_integration.py

# Attack suite (grader verification)
python -m fugal_subnet.attacks.run_attacks

# Full local testnet (Docker, mock mode — no API spend)
python scripts/launch_testnet.py --mock --epochs 3

# Full local testnet (real API — costs money, needs OPENROUTER_API_KEY)
OPENROUTER_API_KEY=sk-or-... python scripts/launch_testnet.py \
  --live --epoch-budget 30 --epochs 2
```

## Common Gotchas

- **Docker logging**: use `print(..., flush=True)` — bittensor's loguru swallows
  Python's logging module output in Docker containers
- **Benchmark downloads**: `datasets` library required; benchmarks download on
  first run. GPQA is a gated HF dataset requiring `huggingface-cli login` +
  accepted terms, or add `gpqa` to `FUGAL_SKIP_BENCHMARKS`
- **Local testnet**: `docker compose up --build` or `scripts/launch_testnet.py`.
  The docker-compose sets `FUGAL_REQUIRE_COMMITMENT=0` because in-container
  miners run without chain commitments
- **Weight setting on local chain**: the chain's 100-block rate limit means
  consecutive `--once` runs may get "weights already set" — normal at local
  testnet cadence, not an issue at hourly mainnet epochs
