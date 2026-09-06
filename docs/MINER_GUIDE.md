# Fugal Subnet — Miner Guide

Run a TEE miner on the Fugal subnet. Your miner runs benchmarks inside an Intel
TDX confidential VM each epoch, producing hardware-attested proofs of routing
quality. Better routers earn more emissions.

## What You're Building

A **router head** — a small linear layer (~14KB to ~130KB `.npz` file, depending
on model count) on a frozen
Qwen3-0.6B backbone. Given a question's hidden state, your head picks which LLM
should answer it. Each epoch, your miner:

1. Receives a nonce from the validator (derived from the chain block hash)
2. Selects the epoch's question slice (deterministic, ~300 questions)
3. Loads your head, computes backbone embeddings, runs routing decisions
4. Calls the routed models via a metered proxy (you pay for inference)
5. Grades responses against gold answers
6. Produces a hardware-attested proof (TDX attestation)
7. Publishes the proof for validators to verify

**You pay for your own inference.** The metered proxy inside the TEE records
exact token counts and costs. Validators verify proofs — they never call models.

## Requirements

- **Intel TDX VM** — GCP `c3-standard` or Azure confidential VMs
  (required for hardware attestation in `--live` mode; `--mock` works anywhere)
- Linux (Ubuntu 22.04+ recommended)
- Python 3.10-3.12
- GPU recommended for training heads (CPU works but slower)
- ~4GB disk for dependencies + backbone model
- TAO for subnet registration
- OpenRouter API key (for model inference inside TEE)

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

### Quick start (synthetic data, no API cost)

```bash
python scripts/train_head.py \
  --synthetic --n-questions 300 \
  --models openai/gpt-5.4-mini anthropic/claude-haiku-4.5 deepseek/deepseek-v4-flash \
  --output data/my_head.npz
```

### Competitive training (with matrix data)

Once the subnet is running, download published epoch artifacts and train:

```bash
python scripts/train_head.py \
  --matrix data/matrix.npz \
  --models openai/gpt-5.4-mini anthropic/claude-haiku-4.5 deepseek/deepseek-v4-flash \
  --output data/my_head.npz \
  --device cuda \
  --use-backbone \
  --sft-epochs 100 \
  --cma-generations 50
```

`--use-backbone` is required. Without it, the trainer falls back to random
hidden states and produces a head that scores near zero.

### Head format

The `.npz` file must contain:

| Array | Shape | Description |
|-------|-------|-------------|
| `W` | `(L, 1024)` | Weight matrix, float32. L = number of models |
| `b` | `(L,)` | Bias vector, float32 |
| `models` | `(L,)` | Model ID strings (e.g. `openai/gpt-5.4-mini`) |

Max file size: 1MB. Hidden dimension must be 1024 (Qwen3-0.6B).

## Step 4: Run the Miner

```bash
OPENROUTER_API_KEY=sk-or-... python neurons/miner.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_miner \
  --hotkey default \
  --head-path data/my_head.npz \
  --benchmark-pool data/benchmark_pool.json \
  --port 8091 \
  --mock
```

### Test run (mock mode)

`--mock` (the default) runs without real TDX attestation — useful for testing
on any hardware. Everything works the same except the attestation quote is
synthetic. Validators in mock mode accept these proofs.

### Live mode (requires TDX VM)

```bash
OPENROUTER_API_KEY=sk-or-... python neurons/miner.py \
  --netuid <NETUID> \
  --network finney \
  --coldkey fugal_miner \
  --hotkey default \
  --head-path data/my_head.npz \
  --benchmark-pool data/benchmark_pool.json \
  --port 8091 \
  --live
```

`--live` produces real TDX attestation. Requires an Intel TDX-capable VM.

### Running as a service

```bash
sudo tee /etc/systemd/system/fugal-miner.service > /dev/null <<'EOF'
[Unit]
Description=Fugal Subnet Miner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/fugal-subnet
Environment=OPENROUTER_API_KEY=sk-or-...
ExecStart=/path/to/fugal-subnet/.venv/bin/python neurons/miner.py \
  --netuid <NETUID> --network finney \
  --coldkey fugal_miner --hotkey default \
  --head-path data/my_head.npz \
  --benchmark-pool data/benchmark_pool.json \
  --port 8091 --live
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now fugal-miner
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `test` | Network: `finney`, `test`, `local` |
| `--netuid` | `1` | Subnet netuid |
| `--coldkey` | `default` | Wallet coldkey name |
| `--hotkey` | `default` | Hotkey name |
| `--wallet-path` | SDK default | Bittensor wallet root directory |
| `--port` | `8091` | Axon port |
| `--head-path` | (required) | Path to `.npz` head file |
| `--benchmark-pool` | (optional) | Local pool JSON. Defaults to the same `load_all()` the validator uses, which is what you want on mainnet — override only for offline or local runs. The flag routes through the same loader, so it cannot disagree with a validator reading the same file. |
| `--mock/--live` | `--mock` | Mock (default) or live TDX attestation |
| `--log-level` | `INFO` | Logging level |

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

### `FUGAL_BENCHMARK_POOL` is not optional, and the failure mode is silent

**Set it in your compose.** Leave it out and the miner falls through to
`load_all()`, starts embedding the full ~21,000-question pool, and takes
**about 13 hours** single-threaded before it answers anything.

There is no error. The process is running, the logs look busy, the axon may even
be serving — and the miner produces nothing for half a day. This was hit during
a real rehearsal by someone who had been warned about it hours earlier, which is
why it is here in bold rather than in a footnote.

### You may choose any instance size

Your base measurement is `sha256(MRTD ‖ RTMR1 ‖ RTMR2)` and it does **not**
include the machine shape. Measured across three real deploys — `c3-standard-4`
and `c3-standard-8`, two different applications — the value is byte-identical:

    12a1f2f56907f80576be553f3d71031ec86f2848877ac05c82db5d8627fc9141

So pick the instance size that suits your workload and your budget. You do not
need to match a reference shape, and doing so buys you nothing.

### Building the image on a fresh GCP project fails twice first

Not a Fugal problem, but everyone building their own image hits it, so it is
here rather than left to be rediscovered. On a **new** GCP project the default
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

If you run in a region whose Google intermediate CA is not yet vendored in
`fugal_subnet/tee/roots/`, your proof can still verify — the intermediate
travels with the proof and is checked against the pinned root — but open an
issue so it can be added, because it is one fewer moving part.

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

How a miner does get its key is an open design decision, tracked in
`docs/INVARIANTS.md`. Do not improvise one.

The reference file in `tests/fixtures/app-compose_A.json` is a **connectivity
test app** (nginx and a socat bridge), not a miner. Do not deploy it and do not
copy its values: it sets `public_logs`, `public_sysinfo`, `public_tcbinfo` and
`gateway_enabled` to `true` — those are dstack's defaults, which is fine for a
box holding no secrets and wrong for yours.

## Updating Your Head

Retrain on newer data and restart the miner with the new `.npz` file. The
restart re-commits the new hash on-chain.

**Important:** Recommitting a new head resets your evidence accumulator. Your
accumulated score drops to zero and must rebuild over epochs. Retrain in big
steps, not continuous tweaks. This is by design — it makes dethroning expensive
and rankings stable.

## Scoring

You are scored on **quality per dollar, against the best single model**:

```
quality = wilson_lcb(your accuracy) / accuracy of the best single model
thrift  = what the best model would have cost / what you actually spent
score   = quality^0.8 * thrift^0.2
```

**A score of 1.0 means you matched the best single model's quality per dollar.
Above 1.0 means you beat it.** That is the whole product: same answers, less
money.

Two consequences worth internalising:

- **Neither axis rescues the other.** Routing everything to the cheapest model
  scores badly (quality collapses). Routing everything to the best model scores
  badly (thrift collapses). There is no weighting you can exploit — the score is
  a product, not a sum.
- **Quality is weighted heavier than cost**, deliberately. Giving up 40% of
  quality does not pay for itself even at a 6x saving. The exponent is derived
  from that requirement, not picked.

The reference is the best model's *measured* accuracy, pooled from exploration
samples across all miners and many epochs. It is a fact about the model pool,
not about the current field — how many other miners are online does not move it.

Results are pooled across epochs via **evidence accumulation** (EWMA decay with
Wilson LCB scoring). Your score stabilizes over time — a few lucky epochs won't
rocket you to the top, and a few bad ones won't destroy you. Consistent quality
wins.

**Burn-in:** a freshly committed head ramps in over ~3000 scored questions
(roughly 10 epochs). This is what stops a miner washing a bad record by
recommitting: recovering costs the same evidence that earning the position did.

**Exploration quota:** each epoch you also answer ~5% extra questions using a
model the *nonce* chooses, not your head. These never count toward your
accuracy or your cost — a forced random route is not a penalty — but a proof
missing them, or routing them anywhere other than the assigned model, is
rejected. Budget for the ~5%.

**Miss = 0 accounting:** If your miner misses an epoch (offline, timeout, proof
verification fails), that epoch counts as 0 correct out of n_expected. You
cannot selectively skip bad epochs.

## Anti-Gaming

- **TEE attestation** — results are hardware-attested. You cannot fabricate or
  tamper with proofs after attestation.
- **Measurement pinning** — validators check the TDX quote's own measurement
  registers (MRTD, RTMR0-2) against the approved image list, not any field your
  code writes about itself. Running a modified harness on genuine TDX hardware
  produces a valid quote and an unapproved measurement.
- **Network confinement** — inside the TEE, the benchmark process can only
  communicate with the local MeteringProxy. No data exfiltration.
- **On-chain commitment** — your head hash is committed before benchmarks run.
  Prevents mid-epoch head swaps.
- **Behavioral dedup** — identical or near-identical routing behavior is
  clustered; earliest on-chain commitment wins. Copies are disqualified.
  Routing decisions are compared in a global model index space, so perturbing
  one question to renumber your own model list does not evade it.
- **Evidence accumulation** — miss=0 prevents selective publication.

## Costs

You pay for model inference each epoch. Cost depends on:

- How many questions are in the slice (~300 per epoch)
- Which models your head routes to (cheaper models = lower cost)
- Token counts per question

Typical epoch cost: varies by model selection. The MeteringProxy inside the
TEE records exact costs, which are included in the attested proof.

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first.

**"Head file too large"** — Max 1MB. Reduce the number of models.

**Port already in use** — Change `--port`.

**TDX attestation fails** — Ensure you're running on an Intel TDX-capable VM.
Check that `/usr/bin/tdx-quote-generator` is available.

**Low scores** — Retrain with more data, try different model selections.
Evidence accumulation means scores improve with consistency over epochs.
