# Contributing to Fugal Subnet

Thanks for contributing. Fugal combines security-sensitive artifact handling with consensus-critical grading, so changes need clear tests and a narrow scope.

## Development setup

Fugal supports Linux or WSL2 with Python 3.10–3.12.

```bash
git clone https://github.com/fugal-ai/fugal-subnet.git
cd fugal-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For an installation that exactly follows the committed dependency lock:

```bash
uv sync --locked --extra dev
```

## Before opening a pull request

```bash
python scripts/check_safety_invariants.py
python tests/test_integration.py
python -m fugal_subnet.attacks.run_attacks
```

These commands do not require a chain or paid API calls. Use mock mode for local testnet work:

```bash
python scripts/launch_testnet.py --mock --epochs 3
```

Never run real OpenRouter tests as part of automated validation. Any intentional paid run must be initiated manually, have an explicit budget, and be documented with a `[PAID ~$X]` comment.

## Consensus changes

Validators must derive identical grades and compatible scores from the same epoch inputs. The following are consensus-critical:

- Grader semantics and dispatch in `fugal_subnet/graders.py`
- Benchmark identity, normalization, revision pins, and slicing
- Epoch nonce derivation and boundary selection
- Head artifact schema and validation
- Matrix, soft-target, evaluation, scoring, deduplication, and reward formulas
- Commitment eligibility and reveal artifact schema

Do not make a semantic change to any of these as an ordinary bug fix. Open an issue first and include:

1. The proposed grader version and activation epoch.
2. Compatibility and validator-coordination impact.
3. New adversarial cases and the complete attack-suite result.
4. A migration note in `CHANGELOG.md`.

Completed epoch artifacts are immutable. A new grader version applies no earlier than its declared activation epoch. Missing or mismatched consensus metadata is an error, not an invitation to guess.

## Non-negotiable safety rules

- Do not add `from __future__ import annotations` to `neurons/miner.py` or `fugal_subnet/protocol.py`; Bittensor v10 inspects their annotations at runtime.
- Every `np.load()` call must use `allow_pickle=False`.
- `FugalSynapse.deserialize()` must return `self`.
- Never print or log any part of an OpenRouter API key.
- Never add real paid API calls to the default test path.

## Pull requests

- Keep each pull request focused and explain the operational impact.
- Add or update tests for behavioral changes.
- Update documentation for public interfaces, configuration, or deployment changes.
- Do not commit `.env` files, keys, model heads, benchmark caches, or generated epoch artifacts.
- Use a conventional, imperative commit subject such as `fix head validation bounds`.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
