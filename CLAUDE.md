# CLAUDE.md

Read [AGENTS.md](AGENTS.md) for the full project architecture, file map, SDK
details, and common gotchas. Everything below is a summary of the most critical
rules.

## Critical Constraints

1. **NO `from __future__ import annotations`** in `neurons/miner.py` or
   `fugal_subnet/protocol.py` — breaks Bittensor Axon registration.
2. **All `np.load()` calls MUST use `allow_pickle=False`** — untrusted miner
   data; pickle = RCE.
3. **Synapse `deserialize()` MUST return `self`**, not a dict.
4. **NO paid API calls without explicit user approval.** `--mock` is always the
   default. Tag costs as `[PAID ~$X]`.
5. **NEVER log any part of `OPENROUTER_API_KEY`.**
6. **`fugal_subnet/graders.py` is immutable.** It is hash-pinned
   (SHA256 checked in CI). Any byte change breaks consensus. It is excluded from
   ruff via `per-file-ignores`.
7. **Consensus invariants live in `docs/INVARIANTS.md`.** Read it before
   changing anything that affects scoring, determinism, or the miner
   interface. Consensus-affecting changes update it and add a check.

## Quick Reference

```bash
# Lint
ruff check .

# Unit + integration tests (no API spend)
python tests/test_integration.py

# Attack suites (grader verification, then hostile miner input)
python -m fugal_subnet.attacks.run_attacks
python -m fugal_subnet.attacks.run_miner_attacks

# Scoring determinism (I1) — --perturb simulates a second host
python scripts/check_determinism.py --perturb

# Safety invariants (grader hash, no-pickle, no paid defaults)
python scripts/check_safety_invariants.py

# Local testnet (Docker, mock mode)
python scripts/launch_testnet.py --mock --epochs 3
```

## Project Layout

See AGENTS.md § "Architecture — What Each File Does" for the full map.
Key entry points: `neurons/validator.py` (epoch loop), `neurons/miner.py` (axon server).
All constants live in `fugal_subnet/config.py` (env-overridable).
