# Mainnet Launch

The switch from testnet to mainnet is one flag: `--network finney`. Everything
in this document exists because something *else* also has to be true, and each
item is here because it was found the hard way rather than imagined.

Read [INVARIANTS.md](INVARIANTS.md) first. This is the operational sequence, not
the design.

---

## Before you spend anything

### 1. The subnet must be activated, and nothing will tell you if it isn't

`start_call` is the single most consequential extrinsic in a launch, and its
absence is invisible. Without it the subnet has no first-emission block:
staking is rejected with `SubtokenDisabled`, no validator ever earns a permit,
and every `set_weights` fails. The neurons run, log success, and land nothing.

Fugal's own testnet subnet (netuid 552) sat in exactly this state from May to
September. Its miner served an axon, its validator ran epochs, and not one
weight ever reached the chain.

```bash
python scripts/provision_subnet.py --network finney --netuid <N> \
  --owner-wallet <owner> --status
```

`active` must be `True`. If it is not, nothing below matters.

### 2. Epoch length must equal the subnet tempo

Fugal's epoch is `FUGAL_EPOCH_INTERVAL // 12` blocks. The chain runs Yuma every
`tempo` blocks. When they differ, some epochs set weights that a later epoch
overwrites before any Yuma step reads them — that epoch's work never reaches
emission at all.

`tempo` is bounded by the chain (360 is the default and values as low as 120 are
refused with `TempoOutOfBounds`), and `weights_set_rate_limit` is root-only, so
in practice the alignment is achieved from Fugal's side:

| chain tempo | set on every neuron |
|---|---|
| 360 blocks (default) | `FUGAL_EPOCH_INTERVAL=4320` |
| any other T | `FUGAL_EPOCH_INTERVAL = T * 12` |

`provision_subnet.py --status` prints `ALIGNED` or `*** MISALIGNED ***`. The
epoch must also be at least `weights_rate_limit` blocks (100 by default, a
20-minute floor), or the chain refuses the extrinsic outright.

### 3. Every neuron must load the identical benchmark pool

The pool is consensus state: the slice is drawn from it, so a validator and a
miner with different pools disagree on every question and every proof fails on
`questions_hash` — an error that names the symptom and never the cause.

Two things make the default loader non-reproducible:

- **GPQA is a gated Hugging Face dataset.** `load_all()` raises without an
  authenticated token, so an operator without one cannot start at all — and an
  operator *with* one loads a larger pool than everyone else.
- **LiveCodeBench silently loads zero questions** on current `datasets`
  versions, but loads normally for anyone who has manually placed
  `data/benchmarks/livecode.json`. Two operators, two pools, no error.

Publish a materialised pool file and pin its hash, and point every neuron at it:

```bash
export FUGAL_BENCHMARK_POOL=/opt/fugal/pool.json
```

Confirm agreement before launch — both neurons log `pool_hash` at startup, and
they must match.

### 4. Miners must be reachable, and the chain will not tell them otherwise

`serve_axon` succeeding means the chain recorded an address. It says nothing
about whether anyone can connect to it. A cloud VM whose provider firewall
blocks the port publishes a valid address, logs `Axon registered on chain`, and
is never contacted — earning nothing, indefinitely, with no error on either
side. Netuid 552's miner did this for months behind a closed Oracle security
list rule.

The miner now attempts to connect to its own published address and warns on
failure, but a host that does not route its public IP back to itself will fail
that check while being perfectly reachable. **Verify from a different machine:**

```bash
nc -vz <miner_public_ip> <port>
```

Open the port in the *cloud provider's* firewall, not just the guest's — they
are separate, and the guest one is the one that looks like it worked.

### 5. `--live` needs an approved measurement, and that is not yet solved

A validator started `--live` with no `FUGAL_TEE_MEASUREMENTS` now refuses to
start, because the alternative was worse: an empty approved set can never
contain any measurement, so it rejected every proof and set every miner to zero
weight, silently.

Getting a usable value is the real remaining work.
`measurement_id = sha256(MRTD ‖ RTMR1 ‖ RTMR2)` — see
`fugal_subnet/tee/attestation.py:measurement_id`.

**RTMR0 is deliberately excluded, so the machine shape does NOT affect the
measurement.** It records the TDVF configuration the host builds — vCPU count,
memory size, device config — which the cloud provider chooses and which proves
nothing about the code. Measured byte-identical across three real deploys
(`c3-standard-4` and `c3-standard-8`, two different applications):

    12a1f2f56907f80576be553f3d71031ec86f2848877ac05c82db5d8627fc9141

Miners may therefore pick any instance size. An earlier version of this section
gave the formula as `sha256(MRTD‖RTMR0‖RTMR1‖RTMR2)` and concluded the subnet
would fork by instance size. That was wrong once RTMR0 was dropped, and it is
recorded here rather than silently deleted because it is the sort of claim
someone acts on immediately before spending real TAO.

What remains true: **RTMR1 and RTMR2 still move** — they cover the kernel,
cmdline and initrd — so a stock cloud image would need the approved list
republished on every kernel package update. **A fixed, reproducibly-built guest
image with a pinned kernel and initrd is still a prerequisite for `--live` on
mainnet.** dstack now supplies exactly that, and provisioning has been proven
end to end against a live TD (see [OPEN_WORK.md](OPEN_WORK.md) for what the
rehearsal did and did not demonstrate). Until a measurement from that image is
published, `--mock` is the only honest mode, and `--mock` accepts unattested
proofs — which means the security model is not yet enforced. See
[TDX_VALIDATION.md](TDX_VALIDATION.md).

### 6. Miner startup cost

Backbone embeddings for the 21,717-question pool take **13.2 hours** on an
x86_64 laptop and **36 hours** on a 4-core aarch64 VM. Both measured on real
pool prompts, which average 330 characters: 0.46 q/s and 0.167 q/s
respectively. This is startup cost before the miner answers its first query.

Beware benchmarking this with synthetic prompts. Short ones give 2.5 q/s and
0.8 q/s — five times too optimistic — because cost scales with sequence length
and the pinned batch size pads every item in a batch to its longest member.

They are now cached to disk keyed by `pool_hash`, backbone, batch size and
machine, so it is paid once rather than on every restart — but it *is* paid
once, and the miner is unreachable for the duration. **Warm the cache before
registering, not after.**

Two consequences worth deciding on before launch, not after:

- **Any pool change costs every miner a day of compute.** The pool is consensus
  state, so it cannot be changed cheaply or often.
- **The barrier to entry is a day of CPU.** The embeddings are a pure function
  of pinned consensus inputs — the pool, the frozen backbone, the pinned batch
  size — so the subnet could publish them as a hash-pinned artifact and let
  miners verify rather than recompute. That is not built, and until it is, the
  practical entry cost for a miner is much higher than the ~14KB head suggests.

Note also that `fugal_subnet/determinism.py` pins the backbone to a single
thread. That costs roughly a factor of four in throughput, and it buys less
than it appears to: validators never run the backbone, and its output already
differs across architectures. Raising the pin to a fixed thread count above one
is the obvious lever on the figure above, but it is a consensus-adjacent change
and needs its own verification before anyone reaches for it.

The cache is shared by every miner on a host with the same pool, so three
miners on one machine pay it once between them.

---

## Launch sequence

```bash
# 1. Provision. Idempotent — safe to re-run after any partial failure.
python scripts/provision_subnet.py --network finney --netuid <N> \
  --owner-wallet <owner> \
  --validators <w>/val1,<w>/val2 --miners <w>/m1 \
  --stake <TAO> --epoch-interval 4320 --yes

# 2. Confirm before starting neurons.
python scripts/provision_subnet.py --network finney --netuid <N> \
  --owner-wallet <owner> --status

# 3. Start neurons with the aligned interval and the pinned pool.
export FUGAL_EPOCH_INTERVAL=4320
export FUGAL_BENCHMARK_POOL=/opt/fugal/pool.json

# 4. Watch, and assert rather than read.
python scripts/watch_subnet.py --network finney --netuid <N> --epochs 3 \
  --epoch-interval 4320 \
  --miners m1=<hotkey> --validators v1=<hotkey>,v2=<hotkey> \
  --reveals v1=/opt/v1/results/epochs,v2=/opt/v2/results/epochs
```

`watch_subnet.py` must report zero failures across at least two consecutive
epochs before the launch is real. It checks the things that have each silently
failed before: the subnet is active, axons are routable, commitments land at or
before the boundary block, validators hold permits, weights actually reach the
chain (read back commit-reveal-aware), and independent validators publish
identical weight vectors.

---

## Wallet handling

Testnet keys in this repo's rehearsal were regenerated unencrypted so the run
could be automated. **Do not do that on mainnet.**

- The coldkey never goes on a neuron host. Only the hotkey and `coldkeypub.txt`
  are needed to serve an axon, commit a head hash, or set weights.
- Do registration, staking and provisioning from a machine that holds the
  coldkey, then ship hotkeys only.
- Keep the coldkey password-encrypted. `provision_subnet.py` will prompt once;
  the neurons never need it.

---

## What is still unverified

Honest list, as of this branch:

| | Status |
|---|---|
| Scoring path determinism across x86_64 and aarch64 | **verified** byte-identical |
| Full pipeline against a real chain, multi-miner, multi-validator | **verified** |
| Real DCAP verification and measurement matching | **never executed** — needs TDX silicon |
| A reproducible TDX guest image with a stable measurement | **does not exist** |
| Real OpenRouter billing vs the pinned price table | **never executed** |
| A real trained head outscoring a random one | **never executed** |
| Behaviour under adversarial miners over weeks | **never executed** |
