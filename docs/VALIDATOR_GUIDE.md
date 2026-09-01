# Fugal Subnet — Validator Guide

> [!WARNING]
> The released v0.1 protocol is experimental and is not suitable for funded
> production validation or mainnet. The v0.2 isolation, quorum, and
> reproducibility implementation is present but disabled pending the explicit
> legal, registry, local-chain, testnet, and activation gates.

Run a validator on the Fugal subnet. Validators build ground truth matrices by
calling frontier LLMs on benchmark questions, evaluate miner-submitted router
heads, and set on-chain weights that determine emission distribution.

## What Validators Do

Epochs are aligned to chain blocks (`EPOCH_INTERVAL/12` blocks per epoch), so
every validator processes the same epoch on the same slice. Each epoch, your
validator:

1. Derives a nonce from the epoch **boundary block's** hash
2. Selects ~300 questions, stratified across benchmarks
3. Commits the question slice + grader hash (commit-reveal integrity)
4. Queries all registered miners for their router heads
5. Verifies each head's SHA256 against its **on-chain commitment** — heads not
   committed at or before the boundary block are rejected
6. Calls every model in the miners' union pool (priced, capped, budget-checked)
   on those questions via OpenRouter
7. Grades responses with mechanical checkers (no LLM-as-judge)
8. Builds an N-questions x M-models binary ground truth matrix
9. Evaluates each head against the matrix (accuracy, cost efficiency, KL divergence)
10. Deduplicates copied heads (earliest on-chain commitment wins), computes
    composite scores and weights, sets weights on-chain (CRW-compatible)
11. Publishes the full epoch artifact (`results/epochs/<epoch_id>/reveal.json`):
    questions, matrix, model costs, scores, weights — the data flywheel

## Requirements

- Linux or WSL2
- Python 3.10-3.12
- CPU sufficient (the Qwen3-0.6B backbone runs head evaluation on CPU in
  ~5-10 min per epoch; a small GPU makes it seconds)
- Do not provision funded OpenRouter or production wallet credentials to the
  current v0.1 validator. Its historical code grader is not isolated from the
  validator identity.
- TAO for subnet registration
- Reliable server with good uptime

## Grading isolation status

Running the v0.1 validator inside one container does not isolate generated code
from credentials mounted into that same container. It is not a production
sandbox and should be used only with mock credentials.

V2 has a separate non-root launcher at `fugal_subnet.sandbox.launcher` and a
minimal worker image under `docker/grader-worker/`. The launcher alone receives
container-engine access; the validator receives only a permission-restricted
Unix socket. The service templates in `deploy/systemd/` use separate
`fugal-validator` and `fugal-grader` identities. They are deployment references
for the separately installed `fugal-validator-v2` entry point. That command is
implemented but deliberately refuses to run while the packaged manifest selects
v1; do not activate the services until the worker digest and every item in
`docs/RELEASE_CHECKLIST.md` pass.

## Cost Estimate

Each epoch calls up to 30 models on 300 questions. Actual cost depends on the
models miners declare, but expect:

- **~$15-30 per epoch** with a full model pool
- **~$1,200-2,400/month** at hourly epochs
- There is no default paid budget: live mode requires an explicit positive
  `--epoch-budget` or `FUGAL_EPOCH_BUDGET`
- Each attempt reserves a conservative maximum before it starts; ambiguous
  timeout/retry attempts forfeit that reservation so concurrency cannot exceed
  the configured local ceiling
- Models with per-query cost exceeding `$0.10` are rejected for cost protection
- Prices are fetched live from OpenRouter; `data/models.json` is the local fallback

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
# Create coldkey
btcli wallet create --wallet.name fugal_validator

# Create hotkey
btcli wallet create --wallet.name fugal_validator --wallet.hotkey default --type hotkey
```

## Step 2: Register on the Subnet

```bash
btcli subnet register \
  --netuid <NETUID> \
  --wallet.name fugal_validator \
  --wallet.hotkey default \
  --network finney
```

## Step 3: Set Up API Key

This step is intentionally disabled for the released v0.1 production path.
Only use a key after a separately reviewed v2 activation release.

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Step 4: Run the Validator in mock mode

```bash
python neurons/validator.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_validator \
  --hotkey default \
  --mock
```

### Test run (no API spend)

To verify everything works before spending real money:

```bash
python neurons/validator.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_validator \
  --hotkey default \
  --mock \
  --once
```

Mock mode is the default; `--mock` makes it explicit. It uses synthetic
responses instead of calling OpenRouter. `--once` exits after a single epoch.

### Service templates

```bash
# Review and install both templates; replace every placeholder first.
ls deploy/systemd/fugal-grader-launcher.service \
   deploy/systemd/fugal-validator.service
```

Do not activate these production templates until the v2 entry point and a
published worker digest have passed the rollout gates.

The template invokes `fugal-validator-v2`, requires the restricted grader
socket, stores wallets/state under its private systemd state directory, and
still requires an explicit positive `FUGAL_EPOCH_BUDGET` when `--live` is
present. With the release-candidate manifest it exits before serving or making
any chain mutation because v2 activation is unset.

## Configuration

All settings are configurable via environment variables (see `fugal_subnet/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FUGAL_NETWORK` | `test` | Network: `finney`, `test`, `local` |
| `FUGAL_NETUID` | `1` | Subnet netuid |
| `FUGAL_EPOCH_INTERVAL` | `3600` | Seconds between epochs |
| `FUGAL_SLICE_SIZE` | `300` | Questions per epoch |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required) |
| `FUGAL_EPOCH_BUDGET` | — | Explicit positive live API ceiling per epoch in USD |
| `FUGAL_MAX_MODEL_POOL` | `30` | Max models in union pool |
| `FUGAL_MAX_MODELS_PER_MINER` | `30` | Max models counted per miner's declared pool |
| `FUGAL_MAX_MODEL_COST` | `0.10` | Per-query cost cap for callable models (USD) |
| `FUGAL_API_CONCURRENCY` | `4` | Concurrent OpenRouter calls during matrix build |
| `FUGAL_REQUIRE_COMMITMENT` | `1` | Require on-chain head commitment before scoring |
| `FUGAL_SKIP_BENCHMARKS` | — | Comma-separated benchmarks to skip (must match across validators) |
| `FUGAL_LAMBDA` | `2.0` | Cost-quality routing tradeoff |
| `FUGAL_WILSON_CONFIDENCE` | `0.95` | Wilson LCB confidence level (diagnostic) |
| `FUGAL_MAX_WEIGHT_DELTA` | `0.3` | Max weight change per UID per epoch |
| `FUGAL_STATE_PATH` | `results/validator_state.json` | Persisted scoring state (survives restarts) |
| `LOG_LEVEL` | `INFO` | Logging level |

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `test` | Network: `finney`, `test`, `local` |
| `--netuid` | `1` | Subnet netuid |
| `--coldkey` | `default` | Wallet coldkey name |
| `--hotkey` | `default` | Hotkey name |
| `--wallet-path` | SDK default | Optional explicit Bittensor wallet root |
| `--once` | off | Run one epoch and exit |
| `--live` | off | Explicitly enable paid OpenRouter requests |
| `--mock` | on | Default no-spend mode; flag remains for clarity |
| `--epoch-budget` | — | Required positive USD ceiling with `--live` unless the environment sets it |
| `--log-level` | `INFO` | Logging level |

## Monitoring

### Epoch logs

Structured JSONL logs are written to `results/epoch_logs/`. Each entry includes:

- Epoch ID and block hash
- Number of miners queried, valid/invalid heads
- Per-miner scores (accuracy, cost efficiency, KL)
- Weight assignments
- Anomaly flags
- Phase timing breakdown
- Commit-reveal verification status

### Anomaly detection

The validator automatically flags:

- Epochs where no miners (or only one of several) respond
- Weight concentration (>80% of weight on one miner)
- Suspiciously uniform accuracy (all miners >95% or all <10%)
- Commit-reveal verification failures

For cross-validator auditing, `fugal_subnet/consensus.py` compares several
validators' published reveals offline (median consensus, outlier detection,
Kendall-tau rank agreement).

### Process supervision

The validator does not re-execute itself from an in-process timer. Use systemd
or a container restart policy for crash recovery. This avoids restarting a
legitimate long-running matrix build and repeating paid work. The manifest-gated
v2 journal and orchestrator record cell reservations, completed bounded
responses, and finalized report progress, but they are deliberately not wired
into the historical v1 loop. The separate `fugal-validator-v2` command wires
these components together but refuses to start until v2 is activated.

## Safety Features

- **Pre-flight budget estimate** — before any API call, the estimated epoch cost
  is computed and the most expensive models are dropped until it fits the budget.
- **Atomic budget reservations** — each request attempt reserves its worst-case
  prompt/output cost before scheduling. Actual usage is reconciled after a
  valid response; ambiguous failures consume the full reservation.
- **Conservative price protection** — spend reservations use the greater of the
  packaged snapshot and current live prices. A missing live price aborts before
  paid work. Canonical v2 eligibility/scoring remains inactive until rollout.
- **Model cost cap** — models exceeding $0.10 per query are excluded.
- **Model pool caps** — per-miner declared pools capped at 30; if the union
  exceeds 30, models declared by more miners win (deterministic — an
  alphabetical sybil can't evict everyone else's models).
- **Head validation** — size cap, decompressed-size cap (zip bombs), shape and
  finiteness checks, model-count cap; `allow_pickle` is always False.
- **Weight transition (v1 limitation)** — v1 attempts a ±0.3 cap before a final
  normalization step, so the final change can exceed that bound and a
  disqualified UID can retain residual weight. The inactive v2 implementation
  replaces this with exact bounded-simplex projection and immediate zeroing.
- **Commit-reveal (v1 limitation)** — v1 writes a local commitment before miner
  queries, but it is not externally time-anchored. V2's on-chain precommit and
  committee report logic is wired into the separate concrete Bittensor entry
  point; its multi-validator local-chain behavior remains to be proven before
  activation.
- **State persistence** — scoring state and previous weights survive
  supervisor-managed restarts (`results/validator_state.json`).
- **CRW compatibility** — `set_weights()` automatically uses commit-reveal-weights
  when the chain enables it.

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first.

**High API costs** — Lower `FUGAL_EPOCH_BUDGET`, reduce `FUGAL_MAX_MODEL_POOL`, or
increase `FUGAL_EPOCH_INTERVAL` to run epochs less frequently.

**"No valid heads received"** — No miners responded. The epoch is skipped and logged.
This is normal when the subnet is new and has few miners.

**Weight-setting failed** — Check that your validator has enough stake. The error
message in the epoch log will have details.

**Supervisor restart loop** — Inspect the validator logs and the service manager's
exit status. Do not add an in-process `os.execv` watchdog; incomplete-epoch
resume belongs in the versioned journal.
