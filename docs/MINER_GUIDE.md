# Fugal Subnet — Miner Guide

Run a TEE miner on the Fugal subnet. Your miner runs benchmarks inside an Intel
TDX confidential VM each epoch, producing hardware-attested proofs of routing
quality. The best router holds the champion seat and earns most of the miner
share.

> **Protocol v3 from block 9,527,760** (epoch 26466, ~2026-11-17 17:34 UTC),
> release v0.5.0; the seat is decided on odds from block 9,530,640
> (~2026-11-18 03:10 UTC). v3 replaces the menu, the question pool and the
> scoring (honest odds per menu entry, Brier on random calls); see the v3
> notes in this guide and the CHANGELOG. Re-provision on the v0.5.0 image
> before the block.

> This guide describes **protocol v2**. Until the published activation block
> the subnet scores by the v1 rules; the differences, and how to move a running
> v1 miner without losing epochs, are in [Moving from v1 to v2](#moving-from-v1-to-v2-before-the-activation-block)
> at the end. Every v1 head must be retrained for v2.

## What You're Building

A **router head** — a small linear layer (~14KB to ~130KB `.npz` file, depending
on model count) on a frozen
Qwen3-0.6B backbone. Given a question's hidden state, your head picks which of
the 19 models in the pinned v2 price table (`data/models_v2.json`) should
answer it. Each epoch, your miner:

1. Derives the epoch nonce from the boundary block's hash (the validator derives the same one)
2. Selects the epoch's question slice (deterministic, ~300 questions)
3. Loads your head, computes backbone embeddings, runs routing decisions
4. Calls the routed models via a metered proxy (you pay for inference)
5. Grades responses against gold answers
6. Produces a hardware-attested proof (TDX attestation)
7. Publishes the proof for validators to verify

**You pay for your own inference.** The metered proxy inside the TEE records
exact token counts and costs. Validators verify proofs — they never call models.

## Requirements

- **Intel TDX VM running dstack on GCP `c3`** — the only base measurement
  with an approved entry today (`f4be9383…aa427`, measured 2026-11-03); another cloud or image
  produces a different base that no validator approves. Required for `--live`;
  `--mock` works anywhere
- Linux (Ubuntu 22.04+ recommended)
- Python 3.11-3.12
- No GPU needed. Success-contract heads embed on CPU in float32 (the contract
  rejects CUDA and float16; see *The success profile requires CPU float32 and
  memory headroom* below), and training a linear head on those embeddings is
  light work on a CPU. GPU advice further down applies only to the legacy profile
- ~4GB disk for dependencies + backbone model
- TAO for subnet registration
- OpenRouter API key (for model inference inside TEE)

## Setup

```bash
# Clone and install, exactly the committed lock (how CI installs)
git clone https://github.com/fugal-ai/fugal-subnet.git
cd fugal-subnet
uv sync --locked --extra dev

# Without uv: python3 -m venv .venv && source .venv/bin/activate && pip install -e .
# bittensor builds a Rust extension; if cargo is missing:
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source "$HOME/.cargo/env"
```

## Step 1: Create a Wallet

```bash
btcli wallet create --wallet.name fugal_miner
btcli wallet create --wallet.name fugal_miner --wallet.hotkey default --type hotkey
```

## Step 2: Register on the Subnet

```bash
btcli subnet register \
  --netuid <NETUID> \
  --wallet.name fugal_miner \
  --wallet.hotkey default \
  --network finney
```

## Step 3: Train a Head

A head names models from the v2 price table and carries the v2 token
statistics; a head over the v1 models is not scoreable from the activation
block.

### Quick start (synthetic data, no API cost)

```bash
uv run python scripts/train_head.py \
  --synthetic --n-questions 300 \
  --prices data/models_v2.json \
  --models anthropic/claude-haiku-5.5 deepseek/deepseek-v4.1-flash openai/gpt-6.1-terra \
  --output data/synthetic_head_test.npz
```

Synthetic output is for training tests; use observed data and the benchmark
manifest for the miner commands below.

### Success-head training (observed binary labels)

```bash
uv run python scripts/train_head.py \
  --matrix data/matrix.npz \
  --prices data/models_v2.json --manifest data/benchmark_tokens_protocol_v2.json \
  --models anthropic/claude-haiku-5.5 deepseek/deepseek-v4.1-flash openai/gpt-6.1-terra \
  --output data/my_head.npz --epochs 100
```

The matrix contains aligned `models`, `questions`, and `matrix` arrays. Labels
are 0 or 1; NaN means unobserved. The trainer retains observed all-failure rows,
uses masked binary cross-entropy, and selects a checkpoint by validation BCE.
Exact duplicate prompts stay together in deterministic 60/20/20 splits. The
held-out test split is reported only after checkpoint selection.

There is no published label set: you label your own data, by calling the v2
models on questions from the open sets the pool is built from (the benchmarks
table below). The published reveals are the other source: every epoch's shared
grades say which model answered which question correctly.

Embeddings are computed with the pinned CPU float32 2048-token profile. An
optional `--hidden-states` argument accepts a profile-tagged NPZ cache. There is
no implicit random fallback. Synthetic training is explicitly test-only.

Heads require the versioned success contract, not just W/b/models. See
[the contract and transition guide](SUCCESS_CONTRACT.md) for metadata, archive
limits, export, and the required fresh evidence namespace. The v2 token manifest
was reviewed on 2026-11-09 from 758 metered calls at the v2 answer budget (the
record is inside the file); your head's token statistics must match it exactly,
which `--manifest data/benchmark_tokens_protocol_v2.json` does for you.

### Protocol v4 heads (from block 9,572,760, epoch 26591)

The v4 head is the same format: a linear layer `W` (rows x 1024) and bias
`b` on the same pinned embedding. Its rows are keyed by menu entry id, up to
72 of them; rows for ids not on an epoch's menu are ignored, and an entry you
have no row for is never routed to (from its `scored_from` epoch, a missing
row costs you the odds seat). Token means and the cost profile are optional
and ignored: picks are priced at the epoch's published menu costs. Details:
`docs/decisions/LIVE_MENU_V4.md`, section 11.

A training note, from an offline study on published v3 epochs: rows fitted
as one shared direction plus a per-entry part (`W_m = w0 + V_m`, trained
jointly) predicted better than rows fitted independently, by 0.40 Brier
points (95% interval 0.12 to 0.68) on the seat's pooled pairs and 0.68 (0.52
to 0.84) on raw exploration cells. Today's format holds such a head as it
is: export `w0 + V_m` as row m. A 64-wide hidden layer or a different
embedding model did not help at public-data scale.

## Step 4: Run the Miner

There are two ways to run the miner and they are not interchangeable:

| | Where | Attestation | Model calls | Who accepts the proofs |
|---|---|---|---|---|
| **Rehearsal** (`--mock`, the default) | any Linux box, **test or local chain only** | synthetic quote | whatever `FUGAL_OPENROUTER_BASE` points at — use the stub, or your key is spent | `--mock` validators |
| **Production** (`--live`) | a dstack TD on GCP `c3`, started with `--await-provisioning` | real TDX + TPM quote, compose hash in RTMR3 | real OpenRouter, key pushed over the attested channel | `--live` validators |

`--mock` replaces the attestation and nothing else. **It is refused on
finney**, and a mock miner with `OPENROUTER_API_KEY` in its environment makes
real, paid calls. A `--live` miner on a stock VM (not dstack) produces a quote
with no compose-hash event, which every live validator rejects while the miner
keeps paying for inference — so the "live on a plain TDX VM" path does not
exist.

### Rehearsal on a test chain (no spend)

```bash
# A deterministic OpenAI-compatible stand-in for OpenRouter, from the pool's
# own gold answers. No key, no spend. It needs the pool the miner uses:
python scripts/export_pool.py --protocol 2      # data/benchmarks/pool_v2.json.gz
python scripts/stub_upstream.py --pool data/benchmarks/pool_v2.json.gz --port 8799 &

FUGAL_OPENROUTER_BASE=http://127.0.0.1:8799 python neurons/miner.py \
  --netuid <NETUID> \
  --network test \
  --coldkey fugal_miner \
  --hotkey default \
  --head-path data/my_head.npz \
  --port 8091 \
  --mock
```

Leave `--benchmark-pool` off: the miner loads the same manifest-pinned pool
the validators do (`data/benchmarks/pool.json.gz` if you built it with
`scripts/export_pool.py`, else the pinned Hugging Face revisions). A smaller
local pool is for local chains only and must be declared with
`FUGAL_POOL_UNPINNED=1`.

### Production

Production is the dstack deployment in the next section, driven by
`deploy/dstack/docker-compose.yaml` and provisioned with
`scripts/provision_td.py`. The API key is never in a unit file, a compose
file or the environment of anything outside the TD — it is read from a 0600
file on your machine and pushed encrypted to the attested TD. There is no
systemd unit for a production miner because the production miner is a
container inside a confidential VM.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `test` | Network: `finney`, `test`, `local` |
| `--netuid` | `1` | Subnet netuid |
| `--coldkey` | `default` | Wallet coldkey name |
| `--hotkey` | `default` | Hotkey name |
| `--wallet-path` | SDK default | Bittensor wallet root directory |
| `--port` | `8091` | Axon port |
| `--head-path` | (optional) | Path to `.npz` head file. Required *unless* `--await-provisioning` is set, in which case the head arrives over the attested channel |
| `--await-provisioning` | off | Serve nothing until the operator pushes the head and API key over an attested channel. **This is how a `--live` miner under dstack gets its key** — see below |
| `--benchmark-pool` | (optional) | Local pool JSON. Defaults to the same `load_all()` the validator uses, which is what you want on mainnet — override only for offline or local runs. The flag routes through the same loader, so it cannot disagree with a validator reading the same file. |
| `--mock/--live` | `--mock` | Mock (default) or live TDX attestation |
| `--log-level` | `INFO` | Logging level |
| `--v4-proposal` | unset | Protocol v4 only: a menu entry this miner proposes (JSON). The miner serves it as a document and commits its hash beside its head (below); the root admits it at the next boundary when the rules pass it. Proposing costs nothing: the field measures the entry |
| `--v4-state-dir` | `results/v4` | Protocol v4 only: where the TD keeps its last accepted menu and the published set (`FUGAL_V4_STATE_DIR`) |
| `--v4-anchor` | unset | Protocol v4 only, a last resort: a menu file to start from (`FUGAL_V4_ANCHOR`) |
| `--v4-challenge` | `0` | Protocol v4 only [PAID when not 0]: the challenge the TD answers each epoch, 0 (none), 500 or 1000 questions (`FUGAL_V4_CHALLENGE`; below) |
| `--check` | off | Run the startup checks (chain, protocol, v4 options and files, pool, code sandbox, registration), print the `READY:` line and exit (0 when ready); waits for no provisioning, embeds and serves nothing. Every start logs the same line (under v4: the seed, mix and list found, the published-set seed and the reveal it is observed through, the root's hotkey for the first v4 epoch, your proposal and challenge size, and whether the list the image ships allows your proposal) |

### Moving to protocol v4 (v0.7.0, block 9,572,760)

v0.7.0 switches finney to protocol v4 at block 9,572,760 (the boundary of
epoch 26591). Until that block everything runs v3 exactly.

1. **A v4 head.** Rows keyed by the entry ids of the first v4 menu
   (`fugal_subnet/routing_data/menu_v4_seed.json`), up to 72 rows, W and b
   as float32 or float64 (above). Check it with `python neurons/miner.py
   --check --head-path <head>.npz ...`: it prints the `READY:` line the TD
   will print, or `NOT READY: ...` with each reason.
2. **A new TD on the v0.7.0 image, before the block.** Deploy it with the
   v0.7.0 frozen finney compose (the newest `deploy/dstack/
   app-compose.frozen-*-finney.json` in the release; its image is the one
   in the signed `image-v*` release of v0.7.0) and push the head, key and
   options with `provision_td.py` (below). A v4 TD started before the block
   waits for the last boundary before it, commits the v4 head then, and
   serves from the block; started after the block it commits at once and
   counts from the next boundary it is read at. `provision_td.py --logs`
   shows `READY: protocol v4 from block 9572760, ...`.
3. **Keep the v3 TD serving until the block.** Proofs from the v3 image
   fail from the block; then stop it.
4. **Endpoint.** Across the block the TD reads the published set from the
   release's seed forward, which needs an archive endpoint behind its
   connection: keep the default (finney's public archive) unless you have
   your own archive node.
5. **Spend.** Exploration only, about $0.78 an epoch of measured calls plus
   the confidential VM, unless you choose a challenge (below), whose dollar
   ceiling defaults to $100 of realised spend.

At the block the seat and the evidence window start empty; the first head
that passes the seat rules takes it, and the miner share burns until then.

### Protocol v4: setting your options in a TD (from block 9,572,760, epoch 26591)

A TD runs the frozen compose every miner shares, which sets none of your own
options and accepts no environment of its own. Your three v4 options travel
with the head and the key, over the attested, sealed provisioning push:

```bash
python scripts/provision_td.py --address http://<ip>:8092 \
    --approved <base>:<compose_hash> --instance-id <id> \
    --head <your v4 head>.npz --wallet <wallet> --hotkey <hotkey> \
    --api-key-file ~/.fugal/openrouter.key \
    --v4-proposal '{"model": "<lab>/<name>", "reasoning": ["effort", "high"],
                    "provider_order": ["<host>/<quant>"], "quantization": "<quant>"}' \
    --v4-challenge 1000
```

| Push option | Field | Accepted values |
|---|---|---|
| `--v4-proposal` (or `@file`) | `v4_proposal` | The JSON object `--v4-proposal` takes, at most 1,024 printable ASCII characters, no key twice. It must pass every compiled entry rule that does not depend on a list: a reasoning setting of the fixed set, at most two pinned hosts, none twice, with one quantization and no service tier. Its lab and hosts are checked by the root at admission against the signed list in force then, not at the push |
| `--v4-challenge` | `v4_challenge` | Exactly `0`, `500` or `1000` [PAID when not 0] |
| `--v4-challenge-usd` | `v4_challenge_usd` | Whole dollars, `1` to `50000` (default `100`): the TD's dollar ceiling on one challenge's realised spend, checked first on its estimate |

A push that still names `v4_scored` or `v4_measure_cap_usd` (fields of
earlier v4 builds) is refused, naming the field and what replaced it.

- **Each is sent only when you give it.** Leave them all out and the push
  is exactly the push before protocol v4; the TD then runs with no proposal,
  no challenge (`0`) and the default ceiling of $100.
- **The TD checks every value before it keeps anything.** A value that fails
  (an empty one, a reasoning setting outside the fixed set, a field sent
  twice, an unknown `v4_` field, `v4_measure_cap_usd` included) refuses the
  whole push, naming the
  option and the rule, never the value: the TD stays unprovisioned, and you
  push again. `provision_td.py` runs the same checks before it sends anything.
- **How a proposal joins.** The miner serves the proposal as a small
  document (your hotkey, the model, the reasoning, the pinned hosts and the
  quantization; no price) and commits its hash as the second field of its
  commitment, beside its head, in one transaction. Commit it early in an
  epoch in which your miner also served a verified proof in the epoch
  before: the root reads every commitment about 30 blocks before the next
  boundary, fetches your document from your axon by its hash and the
  model's public listing, and admits proposals in an order drawn from that
  block's hash. One proposal per coldkey an epoch; one join an epoch below
  ten serving hotkeys, two from ten. An admitted entry is on the menu of the
  next epoch as provisional: every miner's exploration is aimed at it, its
  labels score no head, and the step two epochs on decides it on the
  field's labels (answered share, a category at the accuracy floor, not
  clearly worse than an entry costing no more, not a copy of another
  setting of its model already on the menu). It is then full, or retired
  with a cooldown on its id and your coldkey. A proposal that finds no
  place stays on chain and is considered at each boundary while your
  commitment carries it.
- **The lab and host list is checked by the root, not at the push.** Signed
  lists change without a release, so you can push a proposal for a lab or
  host that a newer list adds before your image ships that list. The root
  admits it only under the list in force at that boundary; until then the
  delta it publishes names your proposal with `refused:lab_not_allowed`
  (or `host_not_allowed`, and so on). `provision_td.py` prints a note
  saying whether the list your checkout ships allows the proposal; after the
  push, the miner's `READY:` line says the same for the list the image ships:
  `proposal not allowed by the shipped list serial N (the compiled entry
  rule lab_not_allowed): admitted only under a list in force that allows
  it`. The miner stays ready either way. An effort (`["effort", ...]`) is
  admitted only on hosts whose listing takes `reasoning_effort`, and a host
  is first-party only for the labs the list marks it for: on any other host
  pin a quantization.
- **One source.** Under `--await-provisioning` the TD takes these options
  from the push only: a value given both in the push and on the miner's own
  command line or environment stops the miner at start.
- **Changing them** means a new push, which a TD takes once per start: stop
  and start the instance, then push again.
- **An image from before protocol v4** refuses a push that carries any of
  them, as an unknown field. Push such an image without them.
- **What you see.** After the push the miner logs its `READY:` line with the
  options in force and `options from provisioning`. `--check` in a TD runs
  before any push, so it says the options are pushed at provisioning.
- **What they cannot do.** None of them is an input of the grader, the pool,
  the slice, the menu or the cost model, and none changes the compose hash or
  the image your approved entry names. A proposal still joins only when the
  root's admission passes it, which every node re-runs.

### Protocol v4: what the TD keeps, and starting late (from block 9,572,760, epoch 26591)

Under protocol v4 the menu is published state that changes without a release
(`docs/decisions/LIVE_MENU_V4.md`). Your TD reads each epoch's menu from the
root validator's commitment and checks it itself. You do not need to keep it
running from the activation block onwards, and a restart does not replay
history:

- **What it keeps.** In its state directory (`--v4-state-dir`, default
  `results/v4` under the working directory, which in the image is
  `/app/results/v4` on dstack's encrypted data volume): `menu.json`, the
  canonical bytes of the last menu it accepted (with any entries the root's
  join delta added), and `published_v4.json` (below). Neither file holds a
  key, a question or an answer.
- **Where an epoch starts.** From that kept menu when it is recent (at most 8
  epochs old). Otherwise from the root's newest published reveal: the TD reads
  the root's commitment at the epoch's boundary block, fetches the reveal it
  names from the root's bucket (or a mirror in `FUGAL_MENU_MIRRORS`), checks
  it against the commitment, takes the menu that reveal's epoch ran on,
  checks those bytes against their hash, the caps and the compiled rules,
  and walks the two epochs from there with every check a validator makes. A
  miner started thousands of epochs after activation proves on its first
  full epoch.
- **Your chain endpoint.** Those reads are pinned to epoch boundary blocks, as
  a validator's are. The miner connects with `FUGAL_ARCHIVE_ENDPOINTS` and
  `FUGAL_FALLBACK_ENDPOINTS` (finney's public endpoints by default). On a
  node that keeps only the last ~300 blocks, a boundary more than one epoch
  back cannot be read: after a restart or a skipped epoch that costs at most
  one epoch, and the next epoch starts from a newer reveal. A failed read
  never costs more than its own epoch. The TD's published set does the same:
  where the endpoint no longer serves the boundaries it missed, it
  re-anchors on the root's newest snapshot, which needs pinned reads three
  epochs back (`memo_sync.ANCHOR_HISTORY_EPOCHS`, about 1,100 blocks on
  finney). **What endpoint a v4 miner needs:** one that serves pinned reads
  at least three epochs back, or an archive behind it (the default on
  finney); across the v4 activation, before the first v4 reveal names a
  snapshot, the set is read from the release's seed, up to 150 epochs back,
  which only an archive serves. The miner logs a `chain history:` line at
  startup saying whether a pinned read back to the seed's boundary (while the
  activation is near; three epochs after) is served. A read the node no
  longer keeps is read again from the archive endpoint
  (`fugal_subnet/chain_reads.py`); the TD's compose names `--network finney`,
  so it uses finney's public archive.
- **The last resort.** If neither the kept menu nor a reveal can be had (the
  root's bucket and every mirror unreachable), hand the TD a menu file with
  `--v4-anchor` (any menu's canonical bytes, for example from a validator's
  mirror). It is checked like every menu and used only when the others fail.
- **The published set.** Each epoch's questions are drawn on the questions
  whose outcomes the root's committed reveals have published
  (`docs/decisions/SEAT_MINIMAL_V4.md` 6), and every validator checks your
  proof against that draw. Your TD advances the set itself: at each epoch's
  boundary it reads the root's commitment (pinned, as for the menu) and folds
  the reveal it newly names (the one it already reads for the menu and the
  seat holder, checked against the commitment); when the root is down
  nothing moves and the TD proves on the set it holds. It keeps the set in
  `published_v4.json` in the same state directory (question ids and epochs,
  nothing secret). Started with nothing kept, it takes the root's snapshot of
  the set the named reveal names (`<netuid>/published/<digest>.json` in the
  root's bucket, about 1-1.5 MB, checked against the digest) and reads the
  boundaries after it. Across the v4 activation, before any v4 reveal names a
  snapshot, it reads every boundary from the release's seed: about 1.4 MB and
  one pinned chain read per v3 reveal (the seed is at most 150 epochs old).
  The TD never takes a set from you, and makes no proof on a root reveal that
  contradicts its own state.
- **A provisional entry's calls.** Your TD calls a provisional entry like
  any other, on a deadline of its own (300 s a call, one retry), and starts
  none of them after 20 minutes of the epoch's calls: a new entry's slow
  host never holds your proof back. Validators leave those calls out of
  your failure bound.
- **What a provisional entry can cost you, at most.** No entry joins above
  the price ceiling of $25 per million prompt tokens and $100 per million
  completion tokens (today's dearest entries list $8 / $40), and an entry
  whose price would rise above it at a refresh leaves the menu. Your TD
  also caps what a provisional entry's exploration calls actually cost, all
  provisional entries together: at most **$40 an epoch** (before each call
  it holds back the most the call could cost, its answer limit of
  completion tokens times the higher of the menu's and the listing's price,
  three upstream attempts, and counts what the call did cost once it is
  back; calls past the cap are not made and count only against the entry's
  own answered share). A real entry costs cents a call, so the cap binds
  only on an entry whose answers run to their limit at a high price. The
  exploration draw's own caps ($0.55 cost-aware, $1.10 coverage) are on
  expected cost. **Set a spending limit on your own provider key** that
  resets daily (or weekly), above your normal bill plus these worst cases,
  so a bad epoch cannot drain the account; a total limit that never resets
  fails every call once reached (see the OpenRouter notes below).

### Protocol v4: the challenge, and a holder that only explores (from block 9,572,760, epoch 26591)

Under protocol v4 (`docs/decisions/SEAT_MINIMAL_V4.md` 3, as
`docs/decisions/FAST_CROWN_V4.md` changes it) your head is scored on odds,
which validators compute themselves from its committed bytes. Real calls
routed by your head are needed only to take the seat, in a **challenge**:
`--v4-challenge` (`FUGAL_V4_CHALLENGE`) chooses, each epoch, how many
questions of the epoch's challenge list your TD answers:

| Value | What the TD does | When to use it |
|---|---|---|
| `0` (default) | No challenge; every proof is exploration only | Far from the seat, and as the holder (it answers nothing) |
| `500` | The first 500 questions [PAID, about $16 an attempt] | Cheaper attempts, for a head well ahead on odds |
| `1000` | The first 1,000 questions [PAID, about $32 an attempt] | The epochs you expect to beat the holder on odds |

- **What a challenge is.** Each epoch has a list of 1,000 published
  questions on the scored mix, the 200 scored questions first. On each of
  the first 500 or 1,000 your TD calls your head's pick and, where the seat
  holder's pick differs, the holder's: the holder is the one the root's
  reveal of two epochs back names, and your TD reads its published head
  bytes and checks them against that hash. Both picks are taken over the
  menu's full entries only: a provisional entry is never a challenge pick,
  yours or the holder's. Each validator recomputes both
  picks; a challenge that pairs with another holder or states another pick
  is not counted. A question where both picks are the same entry counts 0;
  otherwise your value minus the holder's, at $1 a correct answer and each
  call at the menu's pick cost of its entry and category. A call that fails
  counts against you: your own as wrong, the holder's as right. A
  holder-side call your TD did not start (its time or dollar budget ran out)
  counts for no one. If more than 10% of the holder-side calls failed or
  were not started, the challenge is **void**: it counts for and against no
  one, `picks_v4.void` says why, and your run goes on through it, once (a
  second void epoch in the same run restarts it, as a gap does).
- **Time.** Your TD starts your own calls first, then the holder's with its
  exploration, 40 calls in flight; no challenge call is started more than
  30 minutes after its first call of the epoch, and a call started is
  counted whenever it ends. If some of your own calls were not started the
  challenge is reported at 500 when those 500 were all made, else not at all
  (an unanswered attempt: your proof is exploration only and your run
  restarts). The TD also makes no challenge in an epoch whose holder it
  cannot read and check (the root's reveal not to be had).
- **The dollar ceiling.** `--v4-challenge-usd` (`FUGAL_V4_CHALLENGE_USD`,
  provisioning field `v4_challenge_usd`; whole dollars, default **100**, at
  most 50,000) bounds what one challenge spends. It works in two steps:
  - **Estimate first.** Before its first call the TD estimates the
    challenge's cost from the epoch's pick-cost table (the cost each
    challenge item is priced at): your pick on every question, plus the
    holder's pick wherever it differs from yours. If the estimate exceeds
    the ceiling there is no challenge that epoch (your proof is exploration
    only and your run restarts); the TD logs the estimate and the ceiling.
  - **Realised spend while running.** No challenge call is started once
    what the challenge has actually cost so far, plus the estimate of its
    calls still in flight, reaches the ceiling. What is not started is
    treated exactly as a time cut (below): your own calls by the reported
    size, the holder's as not started, and the void rule.

  The READY line shows the ceiling in force. The estimate at the seed
  menu's pick costs, with your pick and the holder's differing on 87% of
  questions (as measured):

  | Challenge | Mean seed entry ($0.0147 a call) | Every pick the median entry | Every pick the dearest entry ($0.157 a call) |
  |---|---|---|---|
  | 500 (about 935 calls) | about $14 | about $3 | about $147 |
  | 1,000 (about 1,870 calls) | about $28 | about $6 | about $294 |

  A measured challenge cost about $16 (500) or $32 (1,000). So the default
  $100 admits a typical challenge, and refuses one whose picks (yours or the
  holder's) lean on the dearest entries: raise it for those epochs.

  **What can still go over.** The ceiling stops new calls; it cannot stop
  calls already in flight, which finish and are paid. At most 40 challenge
  calls are in flight at once, and each is bounded by its entry's
  `max_tokens` at its price, which the menu holds under the price ceiling
  ($25 in / $100 out per million tokens). At the seed menu's dearest entry
  ($40 out per million, a 131,072-token limit) a call's answer costs at most
  about $5.24, so the overshoot is at most about $210 (40 calls); at the
  price ceiling it is at most about $13.11 a call, $524 for 40, and the
  proxy may retry an upstream call up to three times, each billed (up to
  about $1,573 if all 40 calls ran to their limit three times). Real answers
  are far shorter (the measured challenge spend above). **Set a spending
  limit on your own provider key** (daily or weekly, above your normal bill
  plus these figures): it is the only bound that holds whatever the TD does.
- **What you pay and what counts.** You pay for every call your TD
  started, your own and the holder's, answered or failed. A counted
  challenge counts every own call and every holder-side call made; a void
  one counts nothing but keeps your run; a challenge not reported counts
  nothing and restarts your run.
- **What an odds-only miner runs and spends.** With no challenge your TD
  still runs every epoch: the menu, the endpoint reads, its 112 exploration
  calls (68 cost-aware, 44 coverage) and any measurement of your own
  proposal. The bill is the exploration (about $0.78 an epoch at measured
  costs) plus the confidential VM (about $0.25 an epoch): about $20 a day.
- **When to challenge.** Every reveal publishes where your head stands on
  odds (`odds_v4`, and its bound against the holder in `seat_v4`). The
  real-picks check (rule B) reads your unbroken run of epochs with a counted
  challenge and needs 500 questions: one challenge of 1,000 can take the
  seat in the epoch it is answered, if your odds beat the holder's then. An
  epoch without a counted challenge restarts the run. Your head passes when
  the bound on its items over the holder is above -1.5 points. Each
  reveal's `picks_v4.evidence` shows your run and the bound; `picks_v4.refused`
  says why a challenge was not counted, `picks_v4.void` why one was void.
- **What the holder does.** Nothing beyond its exploration: it stands while
  verified in one of the last three epochs and is paid each epoch it is
  verified, R does not beat it and the memorisation check does not refuse
  it. Set `--v4-challenge 0` once you hold the seat.
- **Failures.** A proof with more than 34% of its exploration calls failed
  is not counted at all, unless the epoch is an outage (most operators
  failing more than 30% of their exploration calls).

### Protocol v4: how a head wins and keeps the seat (from block 9,572,760, epoch 26591)

Under protocol v4 (`docs/decisions/SEAT_MINIMAL_V4.md`, 3.3 to 3.8 and 4)
the seat is decided by three rules, every epoch, by every validator from
the same published data:

1. **Better odds (rule A).** Your head must beat the bar on odds by 0.1
   Brier points, on at least 300 labels drawn after both heads were
   committed and supplied by neither head's operator. The bar is the seat
   holder; when the holder is away too long, or the reference (R, the
   entry's success rate) beats it, the bar is R and the seat is open. Your
   own exploration calls never judge your own head, and your odds are
   computed for every epoch after your commit whether or not you served it,
   so skipping an epoch changes nothing about your evidence.
2. **Real picks no worse (rule B).** Your head must answer a challenge
   (above), and its items over the holder (at $1 a correct answer, each
   call priced at the menu's pick cost) must have a bound above -1.5 points
   on 500 questions of your unbroken run of challenge epochs: one challenge
   of 500 or 1,000 can be enough.
3. **Precedence (rule C).** Heads are ordered by first commit (heads
   committed at or before the v4 activation block tie there, ordered by a
   digest of its block hash). Your head must also beat on odds, by the same
   0.1 points, every earlier head still in the race (a "contender": it was
   verified in the last three epochs, has a counted challenge in one of
   them, beats the bar and is not shown to route worse than the holder).
   A copy of an earlier head, however altered, is not 0.1 points better than
   it, so it never passes it, even in an epoch the earlier head misses. An
   earlier head that never challenges bars no one.

Of the heads that pass all three this epoch, the one committed **last**
takes the seat. If none passes, the holder keeps it, even when it is not
paid: a seat is never emptied without a replacement. At the v4 activation
the seat starts empty and stays empty (the miner share burns) until a head
has 300 labels over R and a challenge of 500 questions.

- **Keeping the seat.** Keep serving your exploration proofs: the holder
  answers no challenge, and keeps its seat while verified in this epoch or
  one of the two before it: miss one or two epochs and you keep the seat,
  unpaid for those
  epochs; miss three and the seat goes that epoch to the latest head that
  passes the three rules. It also opens when R beats your odds (for
  example, when your head lacks a row for an entry whose 20-epoch
  allowance has ended) or when the memorisation check refuses your head.
- **Adding a row keeps your place (succession).** Commit, from the same
  hotkey and coldkey, a head whose existing rows are byte for byte the old
  head's (weights and bias) and which only adds rows (or drops rows of
  entries that left the menu): it is the same head for the seat (its seat,
  its precedence, its standing and its evidence carry over; the new rows
  count from the new commit). Change any existing row and it is a new head
  with a new place at the back of the order. The seat state publishes each
  head's row digests, so you can check before you commit.
- **Copies.** The same bytes under another coldkey are not a head at all
  (not scored, seated or paid); your second hotkey on your own head earns
  nothing.
- **Pay.** 85% to the holder, 10% and 5% to the two holders before it, each
  paid in an epoch only when its head is verified (the holder with its
  scored answers), R does not beat it and the memorisation check does not
  refuse it. An unpaid former holder's share goes to the holder; an unpaid
  holder's share burns. There is no ramp: a new holder is paid its full
  share from the epoch it takes the seat.
- **Where to look.** Each reveal's `seat_v4` block shows the bar, whether
  the seat is open, the precedence order, and for your head: whether it
  stands, its bound over the bar, its picks check, each earlier contender it
  was tested against, and in words why it did not qualify. The holder is
  `champion.champion`; the seat state, with every head's row digests, is
  `v2_state.seat`.

## Running Under dstack (required for `--live`)

`--live` requires a **measured** image, and a stock cloud VM is not one: the repo
is cloned onto a filesystem nothing measures, so the attestation would prove
which OS booted and nothing about which code ran. dstack solves that — its
initrd extends your application's identity into RTMR3, and that initrd is itself
measured into RTMR2 via a dm-verity roothash on the kernel command line.

Everything below was found by deploying it, not read from documentation.

### Choose the machine, and know what it commits you to

Only the **0.6.0 prerelease** line ships the UKI package the cloud path needs;
the stable 0.5.x line does not. So a GCP deployment is a prerelease deployment.
That is a deliberate project decision, not an oversight — see
[INVARIANTS.md](INVARIANTS.md) — but you should know it is what you are running.

### Three landmines, in the order you will hit them

**1. `dstack-cloud pull` is broken.** It builds a URL under a
`guest-os-v{version}` tag that does not exist. The published image is tagged
`mkosi-os-v0.6.0-rc0` with the asset `dstack-0.6.0-rc0-uki.tar.gz`. Download it
manually and point `image_search_paths` at the extracted directory.

**2. The default `key_provider` boot-loops, and the symptom is silent.** The
guest calls Phala's public KMS at `kms.tdxlab.dstack.org:12001`, gets connection
refused, `dstack-prepare` fails, and `panic=1` reboots it — about every eight
seconds, forever. There is no error surfaced to the deploy; the VM simply never
serves. Valid values are `none | kms | local | tpm`.

> **This is a consensus setting, not a preference.** `key_provider` is extended
> into RTMR3 as its own event and is part of the app-compose, so it lands in the
> compose hash and therefore in the approved entry. **Every miner must use the
> same value** or their compose hashes differ and none of them verify against
> one approved list. Use the value this guide specifies; do not pick your own.

**3. A non-empty `.env` hard-fails the deploy** unless KMS is enabled. Leave it
empty on the `none` key provider.

You also need `mtools`, `dosfstools` and `gdisk` on the deploying machine
(`mcopy`, `mkfs.fat`, `sgdisk`).

### Time budget

About 32 minutes from nothing to a serving app, of which roughly 5 is the image
download and 7.5 is uploading ~825 MB to GCS — on a residential connection that
upload dominates. **Redeploys are about 4 minutes** once the GCP image exists,
so the cost is paid once.

### The first boot embeds the whole pool

Do **not** set `FUGAL_BENCHMARK_POOL` in a production compose. The pool is
consensus state: the miner loads the same manifest-pinned pool the validators
do, through `load_all()`, and refuses to start if it does not match. A pool
file is only accepted if it verifies against the same manifest, so putting one
in the compose buys nothing on a real subnet — it merely made the compose hash
depend on a 12 MB file. (Earlier versions of this guide said the opposite,
because at the time the override skipped the manifest check and a tiny pool
was the only way to get a miner up quickly. That bypass is closed.)

**The first boot of a fresh TD needs Hugging Face.** The v2 image builds the
pool from the pinned dataset revisions, verifies it against the manifest, and
caches it on the `fugal-results` volume (`FUGAL_POOL_CACHE_V2`). Later boots
read the cache, verify it again on every read, and rebuild from Hugging Face
only if it fails. A Hugging Face outage therefore stops only a TD whose volume
is new: a fresh instance, or one recreated by `dstack-cloud deploy --delete`.
**Bring your v2 TD up days before the flip**, not at it (see "How to move").

Embedding a full pool can take hours. Historical measurements with the previous
profile were about 13 hours on one thread and 7 h 39 m on four threads for
~21,500 questions. Those are **not measurements of the new 2048-token profile**.
Use the [offline mechanism report](decisions/evaluation/success-contract/REPORT.md) and
measure your actual pool before budgeting startup time.

- `FUGAL_BACKBONE_THREADS=0` uses all available vCPUs. This is miner-side only;
  validators never run the backbone. More threads do not imply proportional
  speedups, so measure throughput on the intended host.

- Cache the embeddings on the encrypted data volume
  (`FUGAL_EMBEDDING_CACHE` on a named volume, as in
  `deploy/dstack/docker-compose.yaml`) so the pass survives container restarts
  and instance stop/start. It does not survive `dstack-cloud deploy --delete`:
  that recreates the instance and its data disk is auto-deleted with it, so an
  image upgrade re-pays the pass. Stop, do not recreate, when you can.
- Provision and start the miner **before** you expect it to earn. It commits
  its head hash only after the axon serves, and a head committed after an
  epoch's boundary block is unscoreable for that epoch by design.

For a local rehearsal that is not talking to anyone else's neurons, a small
pool is still fine — declare it with `FUGAL_POOL_UNPINNED=1`.

### The success profile requires CPU float32 and memory headroom

The success contract rejects CUDA and float16. Earlier GPU tuning advice applied
to the legacy preference profile and does not apply to these heads.

The default embedding batch is **two questions**. An eight-question batch of
long 2048-token inputs exceeded memory on the 8 GiB development host. Keep the
default on constrained machines; larger batches require enough memory and a
successful batch/single conformance check. `FUGAL_BACKBONE_BATCH_SIZE` controls
the miner batch, while the offline evaluation has `--batch-size`.

The offline evaluator checkpoints every 200 questions and verifies profile and
question identity before resuming. The miner's full-pool cache is written only
after its embedding pass completes, so provision enough memory for that pass.
Cache files from the legacy profile are rejected.

### You may choose any instance size

Your base measurement is `sha256(MRTD ‖ RTMR1 ‖ RTMR2)` and it does **not**
include the machine shape. Measured across three real deploys — `c3-standard-4`
and `c3-standard-8`, two different applications — the value was byte-identical
in September 2026 (`12a1f2f5…9141`), and on 2026-11-03 two fresh `c3-standard-4`
deploys both measure

    f4be9383e31c6cc87707b2b01ace9114ed958435b524425677e63d65065aa427

It moved because GCP changed the TDX firmware (MRTD), not because the image
changed (RTMR1/RTMR2 are identical). Run `scripts/provision_td.py --inspect`
against your own TD and compare with the entry the validators publish.

Choose an instance with enough memory for the new embedding profile. These
historical measurements do not approve the success runtime; follow the
[fresh-state runbook](SUCCESS_CONTRACT.md#fresh-testnet-evidence-runbook--execute-only-after-review)
and obtain the actual measurements for the reviewed build.

### Only the published image digest is approved

Validators approve `<base>:<compose_hash>` pairs and refuse bare bases, and the
compose hash covers the image digest. An image you build yourself has a
different digest, so a different compose hash, so no validator will accept its
proofs. Deploy the published image — the digest named in the signed
`image-v<date>` release — with the frozen
`deploy/dstack/app-compose.frozen-<date>.json` that matches the entry the
validators carry (`docs/APPROVED_ENTRY_ROTATION.md`, "Publishing").

**The current published release is `image-v2026.11.03c`**, image
`ghcr.io/fugal-ai/fugal-miner@sha256:843460e6277be6a3af2ec749b210f7ec00f05a802dd32d07427441e57954a6f8`
(the digest both `deploy/dstack/docker-compose*.yaml` pin). Verify the tag
before you deploy it:

    git -c gpg.ssh.allowedSignersFile=deploy/keys/allowed_signers verify-tag image-v2026.11.03c

The image digest is fixed by the release. The base-measurement half of the
approved entry depends on the cloud's TD firmware, and it is published with the
entry when validators start scoring. Building
your own image is for development against a local testnet in `--mock`, where
no application is bound.

### Building the image on a fresh GCP project fails twice first

Not a Fugal problem, but everyone building their own image for a local testnet
hits it, so it is here rather than left to be rediscovered. On a **new** GCP project the default
compute service account has no permissions, so `gcloud builds submit` fails with
`PERMISSION_DENIED`, and then fails again with a confusing 403 **on its own
staging bucket**. Grant these by hand first:

    roles/storage.objectAdmin
    roles/artifactregistry.writer
    roles/logging.logWriter

on `<project-number>-compute@developer.gserviceaccount.com`. Observed twice on a
fresh project; the build succeeds in about 12 minutes once they are in place.

### Your image repository must be publicly pullable

This is a requirement, not a convenience, and it has two reasons.

The practical one: **dstack has no private registry authentication.** Its only
registry feature sets `registry-mirrors` in `daemon.json`, which is a mirror
list and not credentials. There is nowhere to put a pull secret.

The one that actually matters: **an approved compose hash that names an image
nobody can pull is an unauditable approved list.** The whole point of the
`base_measurement:app_identity` pair is that anyone can check what was approved.
If the image behind an approved hash is private, "this compose hash is approved"
becomes a claim nobody outside its owner can verify, and the approved list stops
being evidence and becomes an assertion.

So publish the image. It contains no secrets — your key arrives at runtime and
never enters the compose (see above), and your head is pushed after boot.

### Pin images by digest, never by tag

The compose hash is `sha256` of the **raw bytes** of `app-compose.json`. A tag
that moves leaves the hash stable while the code beneath it changes — which is
the exact defect the measured image exists to prevent, wearing a better hat.

### What a validator checks, so you can predict rejection

Your proof is accepted when the base measurement is approved, the event log
replays to the RTMR3 in your quote, and the `compose-hash` event matches an
approved application. Note the third: **your RTMR3 will differ from every other
miner's**, because `instance-id` is extended into it and is new on every deploy.
That is expected and is not a problem — what is approved is the compose hash
inside the log, never the register.

On GCP there is a second half. Your attestation carries a TPM quote as well as
the TDX one, and validators check both — a genuine TDX quote with an unverifiable
TPM half is rejected. Nothing is required of you to make this work; it is
produced by the platform.

### Your proof is not anonymous

The attestation key certificate in your proof has a subject like:

    CN=3943619496533622875, OU=my-gcp-project, O=Google Compute Engine, L=us-central1-a

**Your GCP project name, instance id and zone are public** to anyone who reads
your proof. This is not a leak we can close — the certificate is what proves the
machine is a real Confidential VM, and it is issued by Google with those fields
in it. If your project name is something you would rather not publish, rename it
or use a dedicated project before you register. Said here rather than left for
you to discover.

Your proof carries the Google intermediate certificate(s) for your TD's
attestation key, so it verifies in any region and across a Google rotation:
the miner builds the chain at startup (from the vendored intermediates,
or from the certificate's own AIA address) and checks it against the pinned
Google root before using it. Look for "Carrying N TPM intermediate
certificate(s) in each proof" in your log. A warning there instead means the
fetch failed; the miner tries again before each epoch (from the image after
v0.6.0; an earlier image tries once, so restart it), and until it succeeds
your proofs verify only where validators have vendored your intermediate
(us-central1 today): open an issue with the log line.

**Regions.** Google signs attestation keys with a different intermediate in
each region, all under the one root every validator pins, so no region needs
approving. `provision_td.py` checks your TD the same way: it takes the
intermediate from the vendored set or fetches it from Google's CA address
named in the TD's certificate (it asks no other host), and `--inspect` prints
the key's zone and the chain it found. To verify with no fetch at all, pass
the certificate yourself with `--tpm-intermediate FILE`. What a region can
still change is the **base measurement**: it follows the TDX firmware Google
runs under your VM, not the region's name, so run `--inspect` on a new TD and
compare `base_measurement` with the approved entry before you plan around it.

### The `app-compose.json` fields that will hurt you

The compose file is hashed as raw bytes and that hash is what a validator
approves, so every field below is part of your identity — changing one changes
your compose hash and you will need the new one approved. Get them right the
first time.

| Field | Set it to | Why |
|---|---|---|
| `public_logs` | **`false`** | Your miner holds an OpenRouter API key. Public logs are public; one stray traceback with a request header in it and your key is on the internet. There is no way to un-publish it. |
| `public_sysinfo` | `false` | Leaks process and system detail about a machine whose whole purpose is being a sealed box. No upside for a miner. |
| `gateway_enabled` | `false` | The dstack gateway publishes your service. A miner is reached by validators over its axon, not through a gateway. dstack-cloud auto-disables it whenever `key_provider` is not `kms`, but set it explicitly rather than relying on that. |
| `public_tcbinfo` | `false` | Nothing in Fugal reads the agent's public TCB endpoint — validators verify the quote inside your proof — so it is surface with no consumer. The reference fixture has it `true` because that is dstack's default, not because anything needs it. |
| `no_instance_id` | `false` | Keeps the `instance-id` event in your RTMR3 log, which is what lets `scripts/provision_td.py` refuse to hand your key to a correctly-imaged TD you did not create. (The CLI's `--no-instance-id` flag defaults to `false` on every key provider; an earlier reading that it was forced on for non-KMS providers was wrong.) |
| `key_provider` | `tpm` | Confirmed on a GCP `c3-standard-4`: boots clean, no KMS contact, no restart loop. `kms` boot-loops against Phala's public KMS; `local` needs a VMM a confidential VM does not have; `none` works but seals nothing. `tpm` seals app keys into the vTPM under a PCR policy, so they survive a redeploy and are bound to your measured boot state. |

**Every field above is part of your compose hash.** Change one and your hash
changes, and the new one needs approving before your proofs verify again.

#### You cannot ship your API key in `.env` or `allowed_envs`

This is the one that will waste your afternoon if nobody tells you. dstack's
encrypted-env mechanism encrypts environment variables to an X25519 public key
and the CVM fetches the **decryption key from KMS** by remote attestation at
boot. No KMS, no decryption key, no environment variables — and the deploy tool
refuses up front rather than failing later:

    if env_path.exists() and app.key_provider != "kms":
        raise ValueError(f"{app.env_file} found but KMS is not enabled.")

Since `kms` is not usable (it boot-loops against the public KMS), **`.env` and
`allowed_envs` are not available to a miner.**

**Do not solve this by putting the key in the compose file.** `app-compose.json`
is returned *in full* by the guest agent's `/v1/Info`, as the `app_compose`
field — that is how the reference fixture in this repo was obtained. A key
written there is readable by anyone who can reach the agent, and it would also
end up in the provenance of an approved-list entry. It cannot be withdrawn once
published.

#### How you actually get your key in: `--await-provisioning`

**This is solved, and it ships.** An earlier version of this guide called it an
open question, which left a miner unable to run `--live` at all. It is not open.

Run the miner with `--await-provisioning` and it serves nothing until its
operator pushes the head and the API key over an attested channel:

```bash
python neurons/miner.py --netuid <NETUID> --network finney \
  --coldkey fugal_miner --hotkey default \
  --port 8091 --live --await-provisioning
```

Note `--head-path` is **omitted** here — under provisioning the head arrives
over the channel, which is why the flag is optional rather than required.

The direction is the part worth understanding, because it is backwards from the
obvious design: **the miner pushes to the TD, the TD does not fetch.** A TD can
produce a fresh Intel-signed quote over any nonce, so it can prove what it is
without holding a credential first. Your own process created the TD and knows
its address; it asks the TD to prove itself, checks the quote's measurement and
compose hash against the approved entry, and only then sends. Nothing per-miner
is written anywhere the cloud provider can read, and the shared disk drops out
of the critical path — which matters because `.user-config` rides a plain FAT32
image in cloud storage and is fine for a head but fatal for a key.

The mechanism and the exact list of what may cross the channel are in
`fugal_subnet/tee/provision.py`. It has been exercised end to end on real
hardware: the pusher verified a live TD's nonce, measurement, event-log replay,
compose hash and instance id, then pushed a 77 KB head, and the miner proceeded.

**Your hotkey crosses the same channel.** The TD signs `serve_axon` and the head
commitment itself, so it needs your hotkey keyfile, and the compose file and the
shared disk are both readable by the cloud provider. The push therefore carries
the head, the API key, the hotkey keyfile and `coldkeypub.txt` (the SDK reads
the coldkey *address* to serve; without it the axon never registers). The TD
holds the keyfile in tmpfs only. The hotkey must be unencrypted — a TD cannot
answer a password prompt — which is the normal state for a hotkey; the coldkey
is what stays encrypted and never leaves your machine. The operator-side tool is
`scripts/provision_td.py`; `--inspect` shows what the TD attests to before you
send anything, and nothing it prints contains a secret.

```bash
python scripts/provision_td.py --inspect --address http://<td-ip>:8092
python scripts/provision_td.py --address http://<td-ip>:8092 \
    --approved <base>:<compose_hash> --instance-id <from --inspect> \
    --head my_head.npz --wallet fugal_miner --hotkey default \
    --api-key-file ~/.fugal/openrouter.key      # a 0600 file, never argv
python scripts/provision_td.py --logs --address http://<td-ip>:8092 \
    --wallet fugal_miner --hotkey default        # your miner's own log lines
```

**The push is encrypted, and only to the TD the quote vouches for.** The TD's
attestation covers your nonce *and* the hash of an X25519 key it generated at
start; the tool encrypts your head, key and wallet to that key and sends
ciphertext. A TD that returns no key is refused. Keep the session file the push
writes (`~/.fugal/<wallet>-<hotkey>.session`, 0600): it is the only way to read
the TD's logs, and nobody else can — with `public_logs=false` the miner is
otherwise a black box even to you.

The full recipe, including the compose file and the `dstack-cloud` steps, is
`deploy/dstack/README.md`.

The reference file in `tests/fixtures/app-compose_A.json` is a **connectivity
test app** (nginx and a socat bridge), not a miner. Do not deploy it and do not
copy its values: it sets `public_logs`, `public_sysinfo`, `public_tcbinfo` and
`gateway_enabled` to `true` — those are dstack's defaults, which is fine for a
box holding no secrets and wrong for yours.

### Protect your axon

From protocol v2 a champion is scored only while its axon answers, and a
champion kept offline for more than its 3-epoch ghost (about 3.6 h) loses the
seat. Anyone who can flood your axon off the network can try exactly that,
with a copy of your head's routing ready to take the seat (INVARIANTS I3,
"Accepted residual: a champion kept offline").

Only validators ever need to reach your miner, and every validator records its
egress IP on chain (`FUGAL_VALIDATOR_IP`). Admit only the ACTIVE validators
(a validator permit, stake at or above `MIN_VALIDATOR_STAKE`, weights set in
the last three epochs) at the **cloud** firewall, where dropped packets never
reach the box:

```bash
# Dry run: what it would admit, and any rule still open to the world
python scripts/axon_firewall.py --network finney --netuid 70 --project <gcp-project> \
    --instance <vm> --zone <zone>
# Apply, then keep it current as validators come and go
python scripts/axon_firewall.py --network finney --netuid 70 --project <gcp-project> \
    --instance <vm> --zone <zone> --apply --every 600
```

- It manages one rule (`fugal-axon-validators`, tcp:8091, tag `fugal-miner`,
  on the VM's VPC) and only REPORTS other rules that still open 8091 broadly
  (`0.0.0.0/0`, split halves, `::/0`) to any tag the VM carries (`--instance`
  reads them). GCP allow rules add up, so delete the world-open one yourself
  once the allowlist is in place (`gcloud compute firewall-rules delete
  fugal-axon`). A rule aimed at service accounts is listed for a manual check.
- It refuses to apply while any active validator has not published an IP:
  that validator would see your miner as offline while the others did not.
  `--allow-partial` overrides it; prefer asking the validator to set
  `FUGAL_VALIDATOR_IP`.
- It never writes an empty allowlist, and leaves the rule as it is when it
  cannot read the chain or the GCP firewall; with `--every` a failed run is
  retried at the next one (run it as a systemd service or in a tmux pane), so
  a new validator, or one whose IP changed, is admitted within minutes.
- With the allowlist on, the miner's own startup reachability check (it
  connects to its published address) fails and warns: it is not a validator.
  Check from a validator, or temporarily `--extra-cidr` your own IP.

## Updating Your Head

Retrain on newer data, then re-provision: `dstack-cloud stop` and `start`
the instance (not `deploy --delete`, which discards the data disk and the
embedding cache), wait for the miner to log that it is waiting to be
provisioned, and push again with `scripts/provision_td.py --head <new.npz>`
(the same encrypted, attested push as the first time). The miner commits the
new hash on-chain when it serves. The receiver takes exactly one push per
process and answers `409 already provisioned` to any later one, so a push to a
running miner changes nothing: that is deliberate, it is what keeps a stranger
who reaches port 8092 from replacing your session key and locking you out of
your own logs. On a rehearsal miner, restart it with the new `--head-path`.

**A new head starts its burn-in from zero.** It must score 1,500 questions in
an unbroken run (five epochs) before it can take a seat.

**If the head you replace is the champion, read this first.** Committing a
different head retires the old one: its seat becomes a 24-epoch ghost bar
(`CHAMPION_GHOST_RETIRE`; 8 epochs, `CHAMPION_GHOST_RETIRE_CONFIRM`, from the
confirmation activation block), the champion's share burns while the bar stands,
the old head does not resume, and a head takes the seat past the bar only by
beating the ghost by the margin on at least 1,500 questions they both
answered. A head that only started after the old one retired has no such
pairing, so it waits out the 24 epochs (about 29 hours; 8 epochs, under 10
hours, from the confirmation activation block), and so does
everyone else. The way to upgrade a champion without that gap is to run the
new head **beside** the old one for at least five epochs, on a second hotkey
under the same coldkey (two hashes are two heads; only the same bytes count
as a copy): it pairs with the champion epoch by epoch, takes the seat by the
margin when it is measurably better, the old head drops to seat 2 and is
still paid 10% while it serves, and you stop it afterwards. Retrain in big
steps, not continuous tweaks: every recommit costs five epochs of burn-in.

## Scoring

**One champion seat.** The miner share does not go to the field in proportion
to a score. It goes to a **champion**, and a head takes the seat only by beating
the champion on the same questions. Every miner answers the same scored slice,
so the comparison is question by question, over the last
`CHAMPION_WINDOW_EPOCHS` (48) epochs:

```
NV_i = u_i * (g_challenger,i - g_champion,i) - (s_challenger,i - s_champion,i) / V
V    = CHAMPION_ACCURACY_VALUE (30) x the pinned anchor's expected spend per
       question (CHAMPION_ANCHOR_MODEL, claude-fable-5.1: its mean tokens in the
       reviewed v2 token manifest at the pinned v2 prices, about $0.0163, so
       V is about $0.49); the same V whoever holds the seat
u_i  = CHAMPION_HARD_WEIGHT (2) on a question in the `hard` category, else 1
NV   = 100 x mean_i NV_i                          (percentage points)
```

- `g` is the **shared grade** of the (question, model) cell you routed to: the
  median across operators (coldkeys) of every answer to that model on that
  question. Two heads that pick the same model get the same grade, so
  provider noise on a second call does not separate them. Your own failed call
  still grades 0.
- `s` is the cost of your choice under the pinned prices, from the cell's
  median attested completion tokens. It is not your self-reported spend.
- One extra correct answer is worth 30 questions of the anchor's spend.
  Accuracy decides. Cost decides between heads of near-equal accuracy.
- You take the seat when the one-sided paired-bootstrap lower bound of NV, at
  `CHAMPION_CONFIDENCE` = 99% over `CHAMPION_BOOTSTRAP_RESAMPLES` = 2000
  resamples, is above `CHAMPION_MARGIN` = **0.2 points**. The point estimate
  must clear the margin too. The champion keeps its seat with no margin, so two
  near-equal heads cannot trade it on noise.
- The comparison counts only your current unbroken run since your commitment,
  and needs at least `CHAMPION_MIN_PAIRED_QUESTIONS` (1500) questions paired
  with the champion: five epochs. A head that is clearly better takes the seat
  at that floor; a head one accuracy point better takes it in about six
  epochs; an equal head almost never does.
- **From the confirmation activation block (announced with its release), you
  pass that test twice.** The first pass against a settled champion records
  your head as *pending*: the champion keeps the seat and is paid. You take
  the seat when you pass the same test again (same margin, same 99%) on
  questions drawn after that first pass, at least
  `CHAMPION_CONFIRM_QUESTIONS` (1500) of them, so five epochs at the
  earliest.
  - Why: the test runs every epoch for every challenger, so heads with no
    real edge pass it by luck now and then, and more heads means more lucky
    passes. Questions drawn after the pass were not part of that luck.
  - What it costs a head that is really better: about twice the epochs. In
    simulation with the shipped rule, a head 1.5 points better takes the seat
    at epoch 10 instead of 5, one 0.85 points better at a median of 18 instead
    of 8, and every such head still takes it within 150 epochs (one 0.5 points
    better: 94% of runs).
  - Keep serving. Your fresh questions are the epochs since your first pass
    in your current unbroken run; one missed epoch drops the pending record,
    and your next pass starts the count again.
  - A seat that was filled from empty (the first champion after a reset, or a
    take over a retired champion's ghost) can be taken by the single test in the
    four epochs after the fill; from epoch fill + `CHAMPION_CONFIRM_OPEN_EPOCHS`
    (5) on it is settled and a take needs confirming.
  - You can see it: the reveal's `champion.detail` has your lines (`passed
    the take test; …`, then `vs fresh questions, …: n=… NV=… LCB=…`), and
    `v2_state.seat.pending` lists every waiting head and the epoch it passed.
- Gates on a challenger, besides the margin: window accuracy at or above
  `CHAMPION_MIN_ACCURACY` = **0.30**, and at or above the high-water mark (the
  current champion's Wilson lower bound when it was seated). Once the window
  holds `CHAMPION_HARD_GATE_MIN_QUESTIONS` (100) hard questions, the same
  applies to hard-category accuracy. The per-call time gate
  (`CHAMPION_TIME_GATE_FACTOR`) is not applied at launch.

**Who owns a head, and who is senior.** A head hash belongs to the **coldkey**
that first committed it. The same bytes served under another coldkey are an
exact copy: not paid, not measured, and unable to take or retire the head's
seat. Running the same head on more hotkeys earns nothing, and a hotkey swap
under the same coldkey keeps the head. Seniority is the block of the first
commitment, with one exception that matters at launch: **every head committed
at or before the activation block ties**, and the tie is broken by a hash over
the activation boundary's block hash, the head and its owner
(`copy_rule.seniority`; published each epoch as the reveal's `v2_seniority`).
Committing days ahead buys nothing, and the order cannot be known until the
activation block exists. Commits after the flip keep their real block.

**The first champion.** With nobody seated, the best eligible head takes the
seat: burned in, at or above the accuracy floor, not an exact copy, and the
highest measured net value (accuracy first, then spend) over its unbroken run.
An exact tie goes to the head committed first on chain. Registering extra
copies or cheap heads buys nothing. There is no virtual champion to beat. Until some head is eligible the miner
share burns. A better head then displaces the first champion by the margin,
like any other.

**Copies cannot win.** A head is barred from taking a seat unless it also beats
every more senior seat holder by the same margin, on the same paired test. A
copy of the champion, or a lightly edited one, is never measurably better than
its source, so it earns nothing; there is no resemblance test and no second
margin, because none is needed.

**Payout: 85/10/5 across the last three champions.** `PAYOUT_SEAT_SHARES` =
0.85 / 0.10 / 0.05 go to the current champion, the previous one and the one
before. A seat pays only while its head is live this epoch and can hold a seat
(verified, window accuracy at or above 0.30, not an exact copy). An unpaid
seat's share goes to the current champion. With no current champion, the
champion's share burns. Nothing that pays reads the reference frame: from the
flip the whole miner share is paid by this rule, and the frame is published for
information. `MAX_WEIGHT_DELTA` (0.3 per epoch) still limits how fast the
weights on chain move, so a change of champion reaches the chain in full over
about three epochs; the seat itself changes at once.

**Burn-in is to take a seat, not to hold it.** A head must score
`V2_BURN_IN_QUESTIONS` (1500, five epochs) in an unbroken run before it can
take a seat, first or by challenge. A seated champion that has to re-earn
burn-in after a miss keeps its seat and its pay while it does.

**Missed epochs (the E″ rule).** If your miner misses an epoch (offline,
timeout, proof rejected), that epoch is a miss. You cannot selectively skip bad
epochs.

- **Before burn-in**, a miss resets your evidence to zero.
- **After burn-in**, a miss decays your evidence by one epoch of the evidence
  half-life (`FUGAL_EVIDENCE_HALF_LIFE`, 200 epochs), with no zeros added, and
  restarts the burn-in counter.
- **A champion that misses** leaves a ghost bar for `CHAMPION_GHOST_OFF` (3)
  epochs. While it stands, a challenger must beat the ghost by the margin, and
  the champion's share burns. A champion that returns inside the window
  resumes its seat and is paid at once. A champion whose owner commits a
  different head leaves a ghost for `CHAMPION_GHOST_RETIRE` (24) epochs, or
  `CHAMPION_GHOST_RETIRE_CONFIRM` (8) from the confirmation activation block.

**Exploration quota:** each epoch you also answer 5% extra questions using a
model the *nonce* chooses, not your head. They never count toward your grades
or your cost, but a proof missing them, or routing them anywhere other than the
assigned model, is rejected. They feed the published reference frame, which in
v2 is information for miners, not an input to pay. Budget for the 5%.

**Every head starts from zero at the flip.** v2 is a new model list, so the
flip resets everyone's evidence. No v1 record carries over into a v2 seat.

## The v2 protocol: answers, models, benchmarks

**Longer answers.** The harness asks for up to `HARNESS_MAX_TOKENS` = 16384
completion tokens per call, reasoning included (v1: 2048). Each model runs at
one pinned reasoning setting from the model list. Call deadlines nest so each
layer reports before the next gives up: `PROXY_CALL_DEADLINE_S` = 420 s per
call (retries included) inside `HARNESS_CALL_TIMEOUT_S` = 450 s. A 401, 402 or
403 from the provider is never retried and is logged as
`OPERATOR ACTION REQUIRED` (see "Your OpenRouter account is part of your
miner", below). The one exception is OpenRouter's temporary in-flight-budget
402, which is retried (see "A low balance also throttles concurrent calls").

**More calls in flight.** `HARNESS_CONCURRENCY` is 24 (v1: 8). Reasoning models
often run long at 16k: in the v2 token run, two cheap reasoners had p90
completions at the cap. A latency model calibrated on that run puts the p95
epoch of a head leaning on cheap reasoners at about 33 minutes with 8 in flight,
which misses the collection block 36 minutes after the boundary. With 24 it is
about 14 minutes (18 if providers are 30% slower). A head that sends everything
to one slow model can still take about 30 minutes. Longer answers cost more. The cost per epoch at 16k has not
been measured yet. Measure your own first epochs before you budget.

**The model list** (`data/models_v2.json`, hash-pinned). Every model on it has at least three
OpenRouter endpoints, is text-only, and runs at exactly one reasoning setting.
The list has 19 models. `anthropic/claude-fable-5.2` was added 2026-11-09, the
day it reached OpenRouter, with two endpoints, both Anthropic. Read from
OpenRouter's public endpoint list on 2026-11-10, it has **five endpoints across
four providers** (Anthropic, Google, Amazon Bedrock, Azure), so it meets the
three-endpoint rule and an Anthropic outage alone no longer fails every call
routed to it. One Anthropic endpoint still lists twice the catalogue price
($16 / $80 per million tokens): a call OpenRouter sends there costs you more
than the pinned price your score is computed at. The scoring yardstick is
unaffected either way. x-ai/grok-4.7 was on the draft list and was removed
before release: its provider does not bound reasoning by `max_tokens`, and one
measured call billed 48,132 completion tokens against a 16,384 budget. It can
return in v2.1, once the harness streams calls and can abort one at its
deadline. A v1 head names v1 models, so **every head must be retrained** for v2. There
is no published label set to retrain from: you label your own data.

**Open benchmarks, by category.** The v2 pool is
`fugal_subnet/benchmarks/pool_manifest_v2.json`: 28,986 questions from
openly licensed sets, none gated, each with its licence in the manifest. The
pool is split into four categories by `BENCHMARK_CATEGORY_V2`:

| Category (`CATEGORY_TARGETS_V2`) | Share of the slice | Sets |
|---|---|---|
| `knowledge` | 40% | MMLU-Pro, MMLU |
| `math` | 25% | MATH levels 3–5, Omni-MATH (difficulty < 5, integer answers) |
| `hard` | 30% | SuperGPQA (hard), AIME 1983–2025, BeyondAIME, Omni-MATH (difficulty ≥ 5 on its 1–10 scale, integer answers; `v2_sets.OMNI_HARD_DIFFICULTY`) |
| `instruction` | 5% | IFEval (the constraint kinds the pinned grader checks) |

Inside every category, `hard` included, benchmarks split by pool size
(`BENCHMARK_WEIGHT_V2` is empty): about 71 of `hard`'s 90 slots are SuperGPQA
and about 19 are olympiad math. Miner and validator both choose each epoch's questions through
`fugal_subnet/epoch_plan.py` with the epoch's protocol version. For v2,
`epoch_plan.scored_slice` takes the category quotas first and then the
benchmark quotas (`slicer.select_slice_v2`), and `epoch_plan.exploration_plan`
draws exploration at the same mix. So the reference frame measures each model
on the mix that is scored.

### Reading the published reveals

Every epoch, the validator publishes a reveal: the questions, each miner's
head hash and its routing decision on every scored question, the shared
grades (every (question, model) cell's grade and median completion tokens),
each UID's failed calls, the seat decision with its comparison detail
(`champion`, `v2_state`, `v2_seniority`), the exploration observations and
the weights. It is public-read:

    https://storage.googleapis.com/fugal-reveals/70/<epoch_id>/reveal.json
    gs://fugal-reveals/70/<epoch_id>/reveal.json

List them with
`https://storage.googleapis.com/storage/v1/b/fugal-reveals/o?prefix=70/`.
An epoch's reveal is published one epoch later (the subnet's weight
commit-reveal period, currently 1). The validator's on-chain commitment is a
rolling pointer, `reveal:<epoch_id>:<sha256>`, for the newest reveal it has
published. To check an older epoch, read the commitment at a block from when it
was current. If an upload was retried, an epoch published together with a newer
one may never have been committed on its own. Head files are not in the reveal.

**Your verified head becomes public.** From v0.6.1 the root validator
publishes the bytes of every head it verified, not only the champion's.
When:

- **A head that holds or has held a seat** (the champion, a former champion
  in the seat's history, a champion's ghost): with the reveal that says so,
  one epoch after that epoch at the current commit-reveal period.
- **Any other verified head:** 32 epochs (about 38 hours) after the epoch
  the root first verified it in, with that epoch's reveal or the next one
  published. It is published then whether or not you still serve it, and
  whether or not it ever took a seat.

Before v0.6.1 only the validators that queried you held those bytes; what
was public was the hash, the routing and the odds. Anyone can now download a
published head:

    https://storage.googleapis.com/fugal-reveals/heads/70/<sha256>.npz

where `<sha256>` is the head hash in the reveal's `head_hashes` (your
on-chain commitment). `<epoch_id>/heads.json` beside each reveal lists the
heads of that reveal that are public, and names the ones that are not due
yet with the epoch they are due from. The 32 epochs are the root's setting
(`FUGAL_PUBLISH_HEADS_AFTER_EPOCHS`); `heads.json` states the value in force.
A head that never verified is not published, and a head is not published
before the reveal of the first epoch that scored it. Nothing in scoring
changed with this. The rules that already applied to copies apply to a copy
made from published bytes ("Who owns a head" and "Copies cannot win", above):
the same bytes under another coldkey are an exact copy and earn nothing, and
a modified head is a different head that takes a seat only by beating every
senior seat holder by the margin. If you fetch someone's head, check its
sha256 against the reveal and load it with pickle disabled
(`np.load(..., allow_pickle=False)`).

### When a good proof is not scored, and why it is not you

Two behaviours can make a correct proof go unscored in an epoch. Neither
changes anything you should *do*, but a miner who does not know about them will
read the result as being cheated, so they are stated here.

**Your proof can come back `unverifiable`.** To verify your TDX quote a
validator fetches DCAP collateral from a PCCS over the network. That fetch is
bounded per call and the whole verification phase has a per-epoch ceiling, so if
the validator's PCCS is slow or down, your proof is recorded as *not checked* —
explicitly **not** as invalid. Validators log these separately (`unverifiable`
vs `invalid`) precisely so an outage is not mistaken for fraud. The practical
consequence is the honest one: an unchecked proof is not scored, so it falls
through to `apply_miss` exactly like an absent miner. **You cannot cause this
and you cannot prevent it** — the PCCS is the validator's infrastructure, not
yours, and a quote your proof supplies cannot steer which host is dialled. It
is correlated across the field, so it does not move your ranking relative to
anyone else.

**Verification order is nonce-derived, not UID order.** When a validator's
collateral budget runs out, whoever is verified last goes unverified. In UID
order that would be the same miners every epoch — a permanent penalty on high
UIDs that no miner caused and none could escape. The order is instead derived
from the epoch nonce, so it is identical on every validator and no miner can
influence it, and the cost rotates. Do not read a skipped epoch as a signal
about your head.

### A failed model call scores as a wrong answer

Your head is judged on whether the question got answered. A scored question
whose model call still fails after the proxy's retries is recorded `failed` and
scored `correct=False`, exactly like a wrong answer. Two causes are common
enough to plan for.

**`mistralai/mistral-large-2512` is rate-limited upstream.** OpenRouter serves
it from a single provider, so a 429 there has no fallback. Measured:

- 2026-11-03, direct calls at concurrency 2: 42 of 275 answered.
- Mainnet e00026193 (2026-11-04), a head that routes ~17% of questions to it,
  on `image-v2026.11.03b` (no retries): 39 of 300 scored calls failed (13%).
  The two other heads scored that epoch never route to it and had 0 failed calls.
- Mainnet e00026194 (2026-11-04), the same head on `image-v2026.11.03c`,
  which retries 429s with backoff: 23 of 300 scored calls failed (7.7%), every one
  to this model. Retries cut the failures; they do not remove them.

Treat the model's availability as part of its accuracy. If your training labels
come from calls that succeeded, they overstate it. Either train on labels that
count failures, or keep your head off the model.

**Your OpenRouter account is part of your miner.** A key with a total spend
limit returns 403 once the limit is reached, and an exhausted account balance
returns 402. Neither is retried. A 402, a 403 and a 401 (an invalid key) are
logged as `OPERATOR ACTION REQUIRED` (from protocol v2's image; on v1 images a
403 shows as an ordinary upstream error). On 2026-11-03 an exhausted balance failed about 55% of one epoch's
calls. Set no total limit on the miner's key, and set a low-balance alert on the
account.

**A low balance also throttles concurrent calls.** OpenRouter keeps an
in-flight spending budget for prepaid accounts below a balance threshold and for
newer accounts: every running request holds its input plus `max_tokens` at the
endpoint's prices until it settles. From protocol v3 the harness keeps
`HARNESS_CONCURRENCY` = 40 calls in flight, each with its menu entry's own
answer budget (up to 128k tokens; v2 kept 24 at 16,384). Computed from the
pinned v3 prices and budgets (not measured on a live key), a call holds about
$1.07 on average over the menu, so 40 calls hold about $43 for a head that
spreads evenly, and up to about $205 for one that sends everything to Claude
Fable 5.2 ($5.14 per call). The hold is released when the call settles; it is
not spent. OpenRouter does not publish the fraction of the balance that makes
up the budget, or the threshold above which it stops applying. Keep a balance
of at least ~$50, and more if your head leans on the most expensive entries. When the
budget is full, OpenRouter refuses with a temporary 402
(`in_flight_budget_exhausted`). The v2 image retries that 402 like a 429 and
logs `OpenRouter in-flight budget full`. If it is still full after the retries,
the call fails and scores as a wrong answer. A 402 with `weight_exceeds_budget`
means one request is larger than the whole budget; it is logged as
`OPERATOR ACTION REQUIRED`. Add credits for either.

## Anti-Gaming

- **TEE attestation** — results are hardware-attested. You cannot fabricate or
  tamper with proofs after attestation.
- **Measurement pinning** — validators check the TDX quote's own measurement
  registers against the approved image list, not any field your code writes
  about itself. The base measurement is `sha256(MRTD ‖ RTMR1 ‖ RTMR2)`.
  **RTMR0 is deliberately excluded** — it records host-chosen virtual hardware
  config, so including it would fork the approved list by instance size (this
  is why you may pick any machine shape, above).
- **Compose-hash pinning is what binds your code**, and the distinction
  matters. The base measurement proves which *image* booted; it does not by
  itself prove which Fugal code ran, because on a stock VM the repo is cloned
  onto a filesystem nothing measures. Measured directly: editing
  `fugal_subnet/tee/harness.py` left the base measurement byte-identical. What
  binds your code is the `compose-hash` event replayed out of RTMR3, which
  covers the image your compose names — which is why `--live` requires dstack
  and why you must pin images by digest rather than by tag. See
  [INVARIANTS.md](INVARIANTS.md) § I8.
- **Network confinement** — inside the TEE, the benchmark process can only
  communicate with the local MeteringProxy. No data exfiltration.
- **On-chain commitment** — your head hash is committed before benchmarks run.
  Prevents mid-epoch head swaps.
- **The copy rule** — only seats pay, a head is barred from a seat unless it
  beats every more senior seat holder by the margin on the same questions, and
  a head hash belongs to the coldkey that first committed it (the same bytes
  under another coldkey are an exact copy and earn nothing). A copy is never
  measurably better than its source, so copying cannot pay.
- **Axon access** — in `--live` only hotkeys with at least
  `FUGAL_MIN_VALIDATOR_STAKE` (default 50) stake on the subnet, as the chain
  reports it, can query you; a validator permit alone does not. Your head is
  served in full to whoever can, so this is a speed bump in front of dedup,
  not the defence. From v0.6.1 a verified head is also published by the
  root: with the reveal once it holds or has held a seat, otherwise 32 epochs
  after it first verified ("Reading the published reveals").
- **The E″ miss rule** — a missed epoch decays your evidence and restarts
  burn-in; you cannot publish only your good epochs.

## Costs

You pay for model inference each epoch. Cost depends on which models your head
routes to and how many tokens they write: at the v2 answer budget (16,384
tokens, reasoning included) a cheap reasoning model can cost more per question
than a dearer terse one.

**Epoch cost computed from the pinned v2 prices and the reviewed token
manifest** (mean tokens per question from 40 metered calls per model on the
pinned v2 pool, re-measured 2026-11-09), for a 300-question slice, 20 epochs a
day on mainnet:

| If your head routed everything to | $/question | $/epoch | $/day |
|---|---|---|---|
| `openai/gpt-6-luna` (cheapest per question) | 0.0004 | 0.11 | 2.20 |
| `anthropic/claude-haiku-5.5` | 0.0005 | 0.14 | 2.83 |
| `deepseek/deepseek-v4.1-flash` | 0.0012 | 0.37 | 7.36 |
| `anthropic/claude-sonnet-5.5` | 0.0043 | 1.29 | 25.82 |
| an even mix of all 19 | 0.0055 | 1.65 | 32.98 |
| `anthropic/claude-opus-5.5` | 0.0116 | 3.49 | 69.83 |
| `anthropic/claude-fable-5.1` (the anchor) | 0.0163 | 4.90 | 98.06 |
| `anthropic/claude-fable-5.2` (dearest per question: it writes more) | 0.0195 | 5.86 | 117.17 |

A real head lands somewhere inside that range. Add the exploration quota (15
questions at the nonce's choice of model, about $0.08 at the even mix) and your
confidential VM (about $150/month for a GCP `c3-standard-4`). These are
computed, not measured on a live key: the manifest's means for long-reasoning
models on the hardest sets are an extrapolation (its review record says so),
so measure your own first epochs before you budget. The MeteringProxy inside
the TEE records your exact costs, which are included in the attested proof.

`docs/decisions/MINER_ECONOMICS.md` models whether this is net-positive and under what
conditions; `scripts/experiments/model_miner_economics.py` lets you put your own numbers in.

## Moving from v1 to v2 (before the activation block)

Protocol v2 switches on at one epoch boundary, the **activation block**, on
every validator at once. Until then the subnet scores by the v1 rules:
headroom above a constant-policy frontier built from the reference frame,
paid to the field in proportion to score, with a 3,000-question burn-in and
v1's own model list and pool. A v1 head names v1 models, so **every head must
be retrained**. The v1 rules are recorded in `docs/INVARIANTS.md` and in this
guide's history (`git log -- docs/MINER_GUIDE.md`); they are not repeated here.

Values marked `<LIKE_THIS>` are not final, except `<V2_MODEL_LIST>`, which is
final (`data/models_v2.json`, 19 models since anthropic/claude-fable-5.2 was
added on 2026-11-09: a head must be exported against today's list and token
manifest, and one exported before that is refused). They are published together at the
v2 announcement, at least 24 hours before the flip, in a signed release:

| Placeholder | What it will be |
|---|---|
| `<ACTIVATION_BLOCK>` | the block of the flip. It is an epoch boundary (a multiple of the blocks per epoch) and every validator sets it as `FUGAL_V2_ACTIVATION_BLOCK` |
| `<V2_IMAGE_DIGEST>` | the v2 miner image, by digest, and its approved entry (`docs/APPROVED_ENTRY_ROTATION.md`) |
| `<V2_MODEL_LIST>` | the v2 model list and prices, `data/models_v2.json`, hash-pinned |
| `<V2_RELEASE_NOTE>` | the signed release that names all of the above |

### How to move without losing epochs

The v2 image cannot reuse the v1 embedding cache. An image upgrade is
`dstack-cloud deploy --delete`, which discards the data disk, so the v2
instance **re-embeds the whole v2 pool** at startup. Measured on the v1 pool:
3 h 20 min on a `c3-standard-8`, about 5 to 6 hours on a `c3-standard-4`.
Measured on the v2 pool (28,986 questions, the 2048-token profile; testnet
552, 2026-11-10): **8 h 28 min** on a `c3-standard-8` inside TDX with 8
threads, on each of two instances. A TD started less than about 10 hours
before the flip misses the first v2 epoch. Measure your own host.

Use a second instance:

1. **Retrain** on the v2 model list (`<V2_MODEL_LIST>`; final, merged as
   `data/models_v2.json` on `main`, and pinned in the signed v2 image), labelling your own
   data. Allow one to two days if you start from no labels. The notice is 24
   hours, so start early: the v2 model list and code are already public
   on `main` (v0.3.1; the `release/v2` branch was merged and deleted).
2. **Bring up a second TD on the v2 image** (`<V2_IMAGE_DIGEST>`) **with the
   same hotkey**, beside the running v1 TD, and provision it as usual. It
   embeds and then **holds**: no axon and no commitment until the epoch before
   `<ACTIVATION_BLOCK>`. The image reads the activation block from
   `FUGAL_V2_ACTIVATION_BLOCK` in its compose, so the v2 compose (and with it
   the compose hash and the approved entry) carries the block; a compose
   without it is a v1 miner. Do this
   days before the flip, not hours: the first boot needs Hugging Face for the
   pool and then several hours for the embeddings, and both are behind you
   once the TD is holding.
3. **The v2 TD commits in the epoch before the flip and serves from it.** A
   hotkey holds one commitment. A validator reads commitments at each epoch's
   boundary block, so the v1 epoch that is already running is still checked
   against your v1 commitment, and from the flip your v2 head is the one
   committed. Do not commit the v2 head yourself earlier: during the notice
   period that would break the v1 proofs of the same hotkey. If the commit
   transaction fails, the TD retries every 2 blocks until 20 blocks before the
   flip, and logs an ERROR if it still has not landed by then. Watch for that
   line: a head committed after the flip boundary misses its first v2 epoch(s).
4. **Stop the v1 TD** once the v2 TD is serving v2 epochs. v1 approved
   entries retire at the flip.

Without a second instance you lose the embed time plus up to one epoch. Under
E″ a v2 head that has not served yet loses nothing by starting late, but it
also cannot start its burn-in until it serves.

A rehearsal of the flip with the shipped neurons on a local chain, at no cost:
`python scripts/dress_rehearsal.py --scenario v2_flip`. The in-process half,
which needs no Docker: `python scripts/rehearsal_v2.py`.

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first.

**"Head file too large"** — Max 1MB. Reduce the number of models.

**Port already in use** — Change `--port`.

**TDX attestation fails** — Under dstack the quote comes from the guest agent
socket; check `/var/run/dstack.sock` is mounted into the container (the
shipped compose does this). On a stock VM the configfs-tsm interface is tried
first and `tdx-quote-generator` is only a fallback — and a stock VM cannot
produce a proof validators accept anyway (see Step 4).

**Not taking the seat** — read the reveal's `champion` detail: it says, per
head, which gate failed or the NV and its lower bound against the champion.
Five epochs of unbroken burn-in come first; after that, only a head that is
measurably better than the champion on the same questions takes the seat.
