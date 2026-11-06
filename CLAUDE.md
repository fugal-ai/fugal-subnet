# CLAUDE.md

Read [AGENTS.md](AGENTS.md) for the full project architecture, file map, SDK
details, and common gotchas. Everything below is a summary of the most critical
rules.

> This is the only **public** Fugal repo: assume anything written here is
> world-readable. Findings that are exploitable on the live subnet go through
> a GitHub security advisory, never a public issue or PR (SECURITY.md).

## Critical Constraints

1. **NO `from __future__ import annotations`** in `neurons/miner.py` or
   `fugal_subnet/protocol.py` — breaks Bittensor Axon registration.
2. **All `np.load()` calls MUST use `allow_pickle=False`** — untrusted miner
   data; pickle = RCE.
3. **Synapse `deserialize()` MUST return `self`**, not a dict.
4. **NO paid API calls without explicit user approval.** The miner, and every
   script that can call a model, defaults to `--mock`. (The validator defaults
   to `--live` since 2026-10-14: it spends nothing, and mock accepts
   unattested proofs. Both neurons refuse `--mock` on finney.) Tag costs as
   `[PAID ~$X]`.
5. **NEVER log any part of `OPENROUTER_API_KEY`.**
6. **`fugal_subnet/graders.py` is immutable.** It is hash-pinned
   (SHA256 checked in CI). Any byte change breaks consensus. It is excluded from
   ruff via `per-file-ignores`. `grading_task.py` is not the grader, but it
   decides what the grader sees — a change there changes every grade.
7. **`data/models.json` (v1) and `data/models_v2.json` (v2) are hash-pinned
   too.** They are the consensus cost denominator: an unreviewed price edit
   silently re-scores every miner. Update the pin in
   `scripts/check_safety_invariants.py` in the same commit, and say why. The
   copies under `fugal_subnet/routing_data/` must stay byte-identical.
8. **Epoch identity has one source**, `slicer.epoch_id_for_block`. A second
   formatting of the epoch id is a consensus break — it already happened once,
   and it stopped the subnet setting weights at all.
9. **Consensus invariants live in `docs/INVARIANTS.md`.** Read it before
   changing anything that affects scoring, determinism, or the miner
   interface. Consensus-affecting changes update it and add a check.

## Quick Reference

```bash
# Lint, and the full suite the way CI runs it (no API spend)
uv run ruff check .
uv run pytest -q
uv run python tests/test_integration.py

# Attack suites (grader output, hostile miner input, forged TEE proofs)
uv run python -m fugal_subnet.attacks.run_attacks
uv run python -m fugal_subnet.attacks.run_miner_attacks
uv run python -m fugal_subnet.attacks.run_tee_attacks

# Scoring determinism (I1) — --perturb simulates a second host
uv run python scripts/check_determinism.py --perturb

# Safety invariants (pins, no-pickle, env classification; --release adds the launch gate)
uv run python scripts/check_safety_invariants.py

# Shipped neurons against a real local chain (Docker, no spend)
uv run python scripts/dress_rehearsal.py --scenario all

# Protocol v2 flip rehearsal: on the local chain, and in process (no Docker)
uv run python scripts/dress_rehearsal.py --scenario v2_flip
uv run python scripts/rehearsal_v2.py
```

## Project Layout

See AGENTS.md § "Architecture — What Each File Does" for the full map.
Key entry points: `neurons/validator.py` (epoch loop), `neurons/miner.py` (axon server).
All constants live in `fugal_subnet/config.py` (env-overridable).
