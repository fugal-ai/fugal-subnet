# Fugal Subnet — Miner Guide

Run a TEE miner on the Fugal subnet. Your miner runs benchmarks inside an Intel
TDX confidential VM each epoch, producing hardware-attested proofs of routing
quality. Better routers earn more emissions.

> **Protocol v2** switches on at an announced activation block and changes the
> image, the model list, the benchmarks and the payout. Read
> [Protocol v2](#protocol-v2) before the flip: every head must be retrained, and
> the move needs a second instance to avoid losing epochs.

## What You're Building

A **router head** — a small linear layer (~14KB to ~130KB `.npz` file, depending
on model count) on a frozen
Qwen3-0.6B backbone. Given a question's hidden state, your head picks which LLM
should answer it. Each epoch, your miner:

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
manifest was reviewed on 2026-11-02 from 664 fresh metered calls (the record is
in `SUCCESS_CONTRACT.md` and inside the file); your head's token statistics must
match it exactly, which `train_head.py --manifest data/benchmark_tokens_v1.json`
does for you.

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
# own gold answers. No key, no spend.
python scripts/stub_upstream.py --port 8799 &

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

Embedding a full pool can take hours. Historical measurements with the previous
profile were about 13 hours on one thread and 7 h 39 m on four threads for
~21,500 questions. Those are **not measurements of the new 2048-token profile**.
Use the [offline mechanism report](evaluation/success-contract/REPORT.md) and
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

**The current release is `image-v2026.11.03b`**, image
`ghcr.io/fugal-ai/fugal-miner@sha256:3f6e3e2d1252a340b2a7609974987aa031da7abdf6724c267982d6feeb1fda12`.
Verify the tag before you deploy it:

    git -c gpg.ssh.allowedSignersFile=deploy/keys/allowed_signers verify-tag image-v2026.11.03b

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
- **Dedup, three clauses** — heads are one head if their routing decisions
  agree on 99% of the slice, OR their weights sit within a small relative
  distance of each other (rows aligned by model id), OR they agree on 95% of
  decisions and the junior one does not beat the senior's score by 0.5%. The
  senior head — registration block, then commitment block — survives; the
  rest are disqualified. Routing decisions are compared in a global model
  index space, so renumbering your model list does not evade it, and nudging
  a bias term until a few decisions change does not either.
- **Axon access** — in `--live` only hotkeys with at least
  `FUGAL_MIN_VALIDATOR_STAKE` (default 50) stake on the subnet, as the chain
  reports it, can query you; a validator permit alone does not. Your head is
  served in full to whoever can, so this is a speed bump in front of dedup,
  not the defence.
- **Evidence accumulation** — miss=0 prevents selective publication.

## Costs

You pay for model inference each epoch. Cost depends on:

- How many questions are in the slice (~300 per epoch)
- Which models your head routes to (cheaper models = lower cost)
- Token counts per question

**Typical epoch cost, computed from the pinned price table** over real
300-question slices (22,095 input tokens on average, 256-token completions):

| If your head routed everything to | $/epoch | $/day at 24 epochs (3600 s testnet; mainnet runs 20 at 4320 s) |
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

## Protocol v2

Protocol v2 changes the image, the model list, the benchmarks and how the
miner share is paid. It switches on at one epoch boundary, the **activation
block**, on every validator at once. Everything above this section describes
v1, which applies until that block. This section says what changes at the flip
and how to move without losing epochs.

Values marked `<LIKE_THIS>` are not final. They are published together at the
v2 announcement, at least 24 hours before the flip, in a signed release:

| Placeholder | What it will be |
|---|---|
| `<ACTIVATION_BLOCK>` | the block of the flip. It is an epoch boundary (a multiple of the blocks per epoch) and every validator sets it as `FUGAL_V2_ACTIVATION_BLOCK` |
| `<V2_IMAGE_DIGEST>` | the v2 miner image, by digest, and its approved entry (`docs/APPROVED_ENTRY_ROTATION.md`) |
| `<V2_MODEL_LIST>` | the v2 model list and prices, `data/models_v2.json`, hash-pinned |
| `<V2_RELEASE_NOTE>` | the signed release that names all of the above |

### What changes at the flip

**Longer answers.** The harness asks for up to `HARNESS_MAX_TOKENS` = 16384
completion tokens per call, reasoning included (v1: 2048). Each model runs at
one pinned reasoning setting from the model list. Call deadlines nest so each
layer reports before the next gives up: `PROXY_CALL_DEADLINE_S` = 420 s per
call (retries included) inside `HARNESS_CALL_TIMEOUT_S` = 450 s. A 401, 402 or
403 from the provider is never retried and is logged as
`OPERATOR ACTION REQUIRED` (see "Your OpenRouter account is part of your
miner", above). Longer answers cost more. The cost per epoch at 16k has not
been measured yet. Measure your own first epochs before you budget.

**A new model list.** `<V2_MODEL_LIST>`. Every model on it has at least three
OpenRouter endpoints, is text-only, and runs at exactly one reasoning setting.
A v1 head names v1 models, so **every head must be retrained** for v2. There
is no published label set to retrain from: you label your own data.

**Open benchmarks, by category.** The v2 pool is
`fugal_subnet/benchmarks/pool_manifest_v2.json`: 29,036 questions from
openly licensed sets, none gated, each with its licence in the manifest. The
pool is split into four categories by `BENCHMARK_CATEGORY_V2`:

| Category (`CATEGORY_TARGETS_V2`) | Share of the slice | Sets |
|---|---|---|
| `knowledge` | 40% | MMLU-Pro, MMLU |
| `math` | 25% | MATH levels 3–5, Omni-MATH (mid difficulty, integer answers) |
| `hard` | 30% | SuperGPQA (hard), AIME 1983–2025, BeyondAIME, Omni-MATH (difficulty ≥ 7, integer answers) |
| `instruction` | 5% | IFEval (the constraint kinds the pinned grader checks) |

Inside `hard`, half the weight is the multiple-choice set and half is the
integer olympiad math (`BENCHMARK_WEIGHT_V2`). Elsewhere a category splits by
pool size. Miner and validator both choose each epoch's questions through
`fugal_subnet/epoch_plan.py` with the epoch's protocol version. For v2,
`epoch_plan.scored_slice` takes the category quotas first and then the
benchmark quotas (`slicer.select_slice_v2`), and `epoch_plan.exploration_plan`
draws exploration at the same mix. So the reference frame measures each model
on the mix that is scored.

**Champion scoring replaces headroom scoring.** In v2 the miner share does not
go to the field in proportion to score. It goes to a **champion**, and a head
takes the seat only by beating the champion on the same questions. Every miner
answers the same scored slice, so the comparison is question by question:

```
NV_i = u_i * (g_challenger,i - g_champion,i) - (s_challenger,i - s_champion,i) / V
V    = CHAMPION_ACCURACY_VALUE (30) x the champion's own mean spend per question
       over the same questions
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
- One extra correct answer is worth 30 questions of the champion's spend.
  Accuracy decides. Cost decides between heads of near-equal accuracy.
- You take the seat when the one-sided paired-bootstrap lower bound of NV, at
  `CHAMPION_CONFIDENCE` = 90% over `CHAMPION_BOOTSTRAP_RESAMPLES` = 2000
  resamples, is above `CHAMPION_MARGIN` = **0.2 points**. The point estimate
  must clear the margin too. The champion keeps its seat with no margin, so two
  near-equal heads cannot trade it on noise.
- The comparison reads the last `CHAMPION_WINDOW_EPOCHS` (48) epochs. It counts
  only your current unbroken run since your commitment, and needs at least
  `CHAMPION_MIN_PAIRED_QUESTIONS` (1500) questions paired with the champion.
- Gates on a challenger, besides the margin: window accuracy at or above
  `CHAMPION_MIN_ACCURACY` = **0.30**, and at or above the high-water mark (the
  current champion's Wilson lower bound when it was seated). Once the window
  holds `CHAMPION_HARD_GATE_MIN_QUESTIONS` (100) hard questions, the same
  applies to hard-category accuracy. The per-call time gate
  (`CHAMPION_TIME_GATE_FACTOR`) is not applied at launch.
- **The first champion.** With nobody seated, the most senior eligible head
  takes the seat: burned in, at or above the accuracy floor, and not an exact
  copy. In v2, seniority among heads is the **commitment block**: the hotkey
  that committed a head hash first owns that hash. There is no virtual
  champion to beat. Until some head is eligible the miner share burns.

**Payout: 85/10/5 across the last three champions.** `PAYOUT_SEAT_SHARES` =
0.85 / 0.10 / 0.05 go to the current champion, the previous one and the one
before. A seat pays only while its head is live this epoch and can hold a seat
(verified, window accuracy at or above 0.30, not an exact copy). An unpaid
seat's share goes to the current champion. With no current champion, the
champion's share burns. A head hash has one owner, the hotkey that committed
it first, and a known hash keeps its owner. Only the owner's UID is measured
as that head and paid for its seat. Another UID serving the same bytes is a
copy: it is not paid, and it cannot take or retire the head's seat. Running the
same head on more hotkeys earns nothing. The v2 payout does not read the reference
frame's confidence: from the flip the whole miner share is paid by this rule,
and the frame is published for information. `MAX_WEIGHT_DELTA` still limits
how fast the weights move.

**Burn-in is to take a seat, not to hold it.** A head must score
`BURN_IN_QUESTIONS` (3000, about 10 epochs) in an unbroken run before it can
take a seat, first or by challenge. A seated champion that has to re-earn
burn-in after a miss keeps its seat and its pay while it does.

**Missed epochs (the E″ rule; changed in v2).**

- **Before burn-in**, a miss resets your evidence to zero.
- **After burn-in**, a miss decays your evidence by one epoch of the evidence
  half-life (`FUGAL_EVIDENCE_HALF_LIFE`, 200 epochs), with no zeros added, and
  restarts the burn-in counter.
- **A champion that misses** leaves a ghost bar for `CHAMPION_GHOST_OFF` (3)
  epochs. While it stands, a challenger must beat the ghost by the margin, and
  the champion's share burns. A champion that returns inside the window
  resumes its seat and is paid at once. A champion whose owner commits a
  different head leaves a ghost for `CHAMPION_GHOST_RETIRE` (24) epochs.

**Every head starts from zero.** v2 is a new model list, so the flip resets
everyone's evidence. No v1 record carries over into a v2 seat.

### How to move

The v2 image cannot reuse the v1 embedding cache. An image upgrade is
`dstack-cloud deploy --delete`, which discards the data disk, so the v2
instance **re-embeds the whole v2 pool** at startup. Measured on the v1 pool:
3 h 20 min on a `c3-standard-8`, about 5 to 6 hours on a `c3-standard-4`.
Measure the v2 pool on your own host.

Use a second instance:

1. **Retrain** on the v2 model list (`<V2_MODEL_LIST>`), labelling your own
   data. Allow one to two days if you start from no labels. The notice is 24
   hours, so start early: the v2 model list and code are already public
   on the `release/v2` branch.
2. **Bring up a second TD on the v2 image** (`<V2_IMAGE_DIGEST>`) **with the
   same hotkey**, beside the running v1 TD, and provision it as usual. It
   embeds and then **holds**: no axon and no commitment until the epoch before
   `<ACTIVATION_BLOCK>`. The image reads the activation block itself.
3. **The v2 TD commits in the epoch before the flip and serves from it.** A
   hotkey holds one commitment. A validator reads commitments at each epoch's
   boundary block, so the v1 epoch that is already running is still checked
   against your v1 commitment, and from the flip your v2 head is the one
   committed. Do not commit the v2 head yourself earlier: during the notice
   period that would break the v1 proofs of the same hotkey.
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

**Low scores** — Retrain with more data, try different model selections.
Evidence accumulation means scores improve with consistency over epochs.
