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
- GPU recommended for training heads (CPU works but slower). If a GPU job runs
  far slower than the card should manage, read *A slow embedding job is usually
  VRAM, not a small card* below before sizing up — the failure is silent
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
  --output data/synthetic_head_test.npz
```

Synthetic output is for training tests; use observed data and the benchmark
manifest for the miner commands below.

### Success-head training (observed binary labels)

```bash
python scripts/train_head.py \
  --matrix data/matrix.npz \
  --manifest data/benchmark_tokens_v1.json \
  --models openai/gpt-5.4-mini anthropic/claude-haiku-4.5 deepseek/deepseek-v4-flash \
  --output data/my_head.npz --epochs 100
```

The matrix contains aligned `models`, `questions`, and `matrix` arrays. Labels
are 0 or 1; NaN means unobserved. The trainer retains observed all-failure rows,
uses masked binary cross-entropy, and selects a checkpoint by validation BCE.
Exact duplicate prompts stay together in deterministic 60/20/20 splits. The
held-out test split is reported only after checkpoint selection.

Embeddings are computed with the pinned CPU float32 2048-token profile. An
optional `--hidden-states` argument accepts a profile-tagged NPZ cache. There is
no implicit random fallback. Synthetic training is explicitly test-only.

Heads require the versioned success contract, not just W/b/models. See
[the contract and transition guide](SUCCESS_CONTRACT.md) for metadata, archive
limits, export, and the required fresh evidence namespace. The checked-in token
manifest is a review candidate derived from historical recorded calls. Live
startup and deployable export reject it until it is reviewed; mocked rehearsals
can exercise the candidate without publishing or spending on inference.

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
| `--head-path` | (optional) | Path to `.npz` head file. Required *unless* `--await-provisioning` is set, in which case the head arrives over the attested channel |
| `--await-provisioning` | off | Serve nothing until the operator pushes the head and API key over an attested channel. **This is how a `--live` miner under dstack gets its key** — see below |
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

### The first boot embeds the whole pool, and that takes half a day

Do **not** set `FUGAL_BENCHMARK_POOL` in a production compose. The pool is
consensus state: the miner loads the same manifest-pinned pool the validators
do, through `load_all()`, and refuses to start if it does not match. A pool
file is only accepted if it verifies against the same manifest, so putting one
in the compose buys nothing on a real subnet — it merely made the compose hash
depend on a 12 MB file. (Earlier versions of this guide said the opposite,
because at the time the override skipped the manifest check and a tiny pool
was the only way to get a miner up quickly. That bypass is closed.)

What the earlier warning was really about is **time**: embedding the ~21,500
questions takes **about 13 hours on one thread** before the miner answers
anything, with no error — the process is running, the logs look busy, and the
miner produces nothing for half a day. Three things make that bearable:

- Give the backbone every vCPU: `FUGAL_BACKBONE_THREADS=0` in the compose
  (`deploy/dstack/docker-compose.yaml` does). This is miner-side only —
  validators never run the backbone. Measured on a `c3-standard-4`: 7 h 39 m
  at four threads against ~13 h at one — 1.7x, not 4x, because the backbone is
  memory-bound past a few cores. More vCPUs help less than more memory
  bandwidth would.

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

### A slow embedding job is usually VRAM, not a small card

Same shape as the `FUGAL_BENCHMARK_POOL` trap above — healthy-looking, silent,
and hours wrong — so it is here rather than left to be rediscovered. Both were
hit on this project's own hardware.

**Symptom:** embedding the pool crawls, well under 1 prompt/s, with no error and
no OOM. The job looks busy and the card looks fine.

**Cause:** VRAM headroom, not VRAM size. Measured on a 6,144 MiB consumer card:
at batch 32 the job sat at 5,864 MiB — about 95% — and past roughly that point
the driver spills to host memory and throughput collapses. At batch 8 the same
work used 3,428 MiB and ran ~29 prompts/s. **30x faster on the same card for the
same job**, purely from leaving headroom.

**The second half, which runs the other way from intuition:**
`backbone.get_backbone` defaults to **float16** on CUDA. That is right for the
datacentre cards a serious miner would rent. On a consumer card *without tensor
cores* it is backwards — measured fp32 at 2.66 TFLOPS against fp16's 0.60, so
**fp32 was 4.4x faster** — and the batch size that fits differs by about 4x
between the two dtypes, so sizing a box from the default gets both speed and
memory wrong.

**What to do before concluding your machine is too small:** check VRAM headroom
and drop the batch size until you are well under ~90% occupancy, then try
float32. Only then buy a bigger card.

Dtype is a miner-side performance choice and touches nothing in consensus —
scoring reads routing decisions out of your proof, never your embeddings — so
you may pick whichever is faster on your hardware. See
`fugal_subnet/backbone.py` for the full measurements and why the default stays
as it is.

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

## Updating Your Head

Retrain on newer data and restart the miner with the new `.npz` file. The
restart re-commits the new hash on-chain.

**Important:** Recommitting a new head resets your evidence accumulator. Your
accumulated score drops to zero and must rebuild over epochs. Retrain in big
steps, not continuous tweaks. This is by design — it makes dethroning expensive
and rankings stable.

## Scoring

You are scored on **the accuracy you add over the best non-routing policy at
your own price**:

```
frontier(c) = best accuracy any single model, or random mixture of models,
              achieves at cost-per-question c   (upper convex hull, from the
              reference frame's exploration samples and the pinned price table)
headroom    = wilson_lcb(your accuracy) - frontier(your cost per question)
score       = max(0, headroom) * burn_in * frontier_confidence
              (0 if wilson_lcb < 0.8 * the frontier's maximum accuracy)
```

**A constant policy scores zero.** Always calling one model, or flipping a coin
between two, reads no question and sits on the frontier; it earns nothing, at
any price. That is the point: the subnet pays for the value of reading the
question and for nothing else. An earlier score (quality per dollar against the
best single model) paid a head that always called one mid-priced model 12% more
than the best trained router; `docs/design-decisions.md` keeps that history.

Three consequences worth internalising:

- **Cheap-and-smart is where the money is.** The frontier is low and steep at
  low prices, so a router that gets frontier-quality answers out of cheap models
  has the most headroom. Matching the best model at the best model's price has
  almost none — a constant policy already does that.
- **You cannot be scored against other miners.** The frontier is built from
  the model pool (nonce-assigned exploration pooled over time), never from
  heads. Nobody joining, leaving or copying changes your reference. Emissions
  are relative afterwards, through Yuma, as on every subnet.
- **Quality is still a requirement.** Below 80% of the frontier's best accuracy
  you earn nothing however cheap you are: "same answers, less money" needs the
  same answers.

**Your own tradeoff is yours.** How much quality to give up per dollar is your
training objective (`FUGAL_LAMBDA` is a miner-side hyperparameter); the subnet
only decides what it pays for.

The frontier is built from *measured* per-model accuracy, pooled from
exploration samples across all miners and many epochs. It is a fact about the
model pool, not about the current field. While it is still cold (few
exploration samples per model) scores are scaled down and the unassigned
weight burns; nobody is paid for the frame's ignorance.

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

**Typical epoch cost, computed from the pinned price table** over real
300-question slices (22,095 input tokens on average, 256-token completions):

| If your head routed everything to | $/epoch | $/day at 24 epochs |
|---|---|---|
| `deepseek/deepseek-v4-flash` (cheapest) | 0.014 | 0.34 |
| `openai/gpt-5.4-mini` (mid) | 0.362 | 8.69 |
| an even mix of all 17 | 0.549 | 13.17 |
| `openai/gpt-5.5` (dearest) | 2.415 | 57.95 |

A real head lands somewhere inside that range. Add ~$0.027/epoch for the
exploration quota, which no routing strategy avoids, and your confidential VM
(~$150/month for a GCP `c3-standard-4`). Completion length is the biggest
uncertainty in the table and has not been measured against a live provider.

`docs/MINER_ECONOMICS.md` models whether this is net-positive and under what
conditions; `scripts/model_miner_economics.py` lets you put your own numbers in.
The MeteringProxy inside the TEE records your exact costs, which are included in
the attested proof.

## Troubleshooting

**"Hotkey not registered"** — Register on the subnet first.

**"Head file too large"** — Max 1MB. Reduce the number of models.

**Port already in use** — Change `--port`.

**TDX attestation fails** — Ensure you're running on an Intel TDX-capable VM.
Check that `/usr/bin/tdx-quote-generator` is available.

**Low scores** — Retrain with more data, try different model selections.
Evidence accumulation means scores improve with consistency over epochs.
