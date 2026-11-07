# Contributing to Fugal Subnet

Thanks for contributing. Fugal combines security-sensitive artifact handling with consensus-critical grading, so changes need clear tests and a narrow scope.

## Development setup

Fugal supports Linux or WSL2 with Python 3.11–3.12 (3.10 was dropped on 2026-11-02: the last `filelock` with the GHSA-v4x8-3wr5-3wq8 fix needs 3.11, and the miner image is built on 3.12).

```bash
git clone https://github.com/fugal-ai/fugal-subnet.git
cd fugal-subnet
uv sync --locked --extra dev      # exactly the committed lock, as CI installs it
```

Without uv, `python3 -m venv .venv && pip install -e ".[dev]"` works but can
drift from the lockfile; see the gate below.

## Before opening a pull request

```bash
scripts/gate.sh
```

That is the whole gate, and it is the only thing worth trusting before pushing.

**Do not hand-roll a subset.** This section used to list three commands run
against whatever `python` was on the path. CI runs far more, in **several
different environments**, and the gap has already cost a red `main`: `dcap-qvl` lives in
the optional `[tee]` extra, a local `.venv` happened to have it, and tests that
imported it unconditionally passed locally nine times and failed in CI. A venv
that drifts from the lockfile hides exactly the failures CI exists to find.

`scripts/gate.sh` closes both halves of that:

- It builds each environment with `uv sync --locked`, so what you test is what
  the lockfile says — including running the suite **without** the DCAP verifier,
  which is a real code path (`verify_dcap` behaves differently when it is
  absent) and the one that broke.
- **It reads its steps out of `.github/workflows/ci.yml`** rather than listing
  them. A step added to CI is picked up on the next run with no edit here, so
  this document cannot fall behind CI again the way it just did.

`scripts/gate.sh test|tee|quality|success` runs one job if you are iterating.
Exit status is 0 only if every step of every selected job passed.

It does not run the backbone-download half of CI's `success-contract` job or
the `docker build`. A change to the backbone path, the vendored contract or the
Dockerfile is known green only once CI says so.

None of it requires a chain or paid API calls. For a local chain run of the
shipped neurons (Docker required, no spend):

```bash
uv run python scripts/dress_rehearsal.py --scenario all
uv run python scripts/dress_rehearsal.py --scenario v2_flip   # the protocol v2 flip
```

Never run real OpenRouter calls as part of automated validation. The only paid
path in the project is a `--live` miner whose key arrives over the attested
provisioning channel; any intentional paid run is started by hand and
documented with a `[PAID ~$X]` comment.

## Consensus changes

Validators must derive identical grades and compatible scores from the same epoch inputs. The following are consensus-critical:

- Grader semantics and dispatch in `fugal_subnet/graders.py`
- Benchmark identity, normalization, revision pins, and slicing (both pool
  manifests, the v2 category map and weights)
- Epoch identity, nonce derivation, boundary selection, and the v2
  activation block
- Head artifact schema and validation
- TEE proof verification: every binding in `fugal_subnet/tee/verify.py`, the
  event-log replay, the approved-entry format, and the provisioning allow-list
- Both pinned price tables and token manifests (`data/models*.json`,
  `data/benchmark_tokens_*.json`)
- Protocol v2: the shared grades, the paired net-value test and its
  constants, the comparison window, the seat machine, the copy rule, the
  payout split, the burn-in rule E'', and the rebuild from reveals
- Protocol v1 (live until the activation block): the reference-frame,
  evidence, frontier scoring, deduplication and reward formulas
- Commitment eligibility and the reveal artifact schema
- The lists in `config.py` that classify every `FUGAL_` variable as
  consensus or operational, and the consensus digest in `fingerprint.py`

Do not make a semantic change to any of these as an ordinary bug fix. Open an issue first and include:

1. The proposed grader version and activation epoch.
2. Compatibility and validator-coordination impact.
3. New adversarial cases and the complete attack-suite result.
4. A migration note in `CHANGELOG.md`.

Completed epoch artifacts are immutable. A new grader version applies no earlier than its declared activation epoch. Missing or mismatched consensus metadata is an error, not an invitation to guess.

## Non-negotiable safety rules

- Do not add `from __future__ import annotations` to `neurons/miner.py` or `fugal_subnet/protocol.py`; Bittensor v10 inspects their annotations at runtime.
- Every `np.load()` call must use `allow_pickle=False`.
- `FugalProofSynapse.deserialize()` must return `self`.
- Never print or log any part of an OpenRouter API key.
- Never add real paid API calls to the default test path.
- `fugal_subnet/graders.py` and both price tables are hash-pinned in
  `scripts/check_safety_invariants.py`; a change updates the pin in the same
  commit and says why.
- Epoch ids come from `slicer.epoch_id_for_block` only.
- A consensus-affecting change updates `docs/INVARIANTS.md` and adds an
  executable check.

## Pull requests

- Keep each pull request focused and explain the operational impact.
- Add or update tests for behavioral changes.
- Update documentation for public interfaces, configuration, or deployment changes.
- Do not commit `.env` files, keys, model heads, benchmark caches, or generated epoch artifacts.
- Use a conventional, imperative commit subject such as `fix head validation bounds`.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
