# Consensus Invariants and Threat Model

This subnet is a mechanism-design problem with a distributed determinism
requirement. Almost every risk follows from two facts:

1. **Money flows from a score.** Every miner is motivated to find the cheapest
   path to a high score. Anything that raises a score without improving routing
   is an exploit, whether or not it is a "bug".
2. **Independent validators must agree.** Scores are computed on machines
   nobody controls centrally. If two honest validators disagree, weights split,
   the incentive signal degrades, and miners can farm the disagreement.

So the question to ask of any change is not "is this code correct?" but
**"what must be true, and what makes it false?"**

This file is the answer to the first half. It exists because a real
consensus bug (`I1`, below) survived five separate code reviews: the property
was never written down, so nobody checked it, and a line that had been correct
for months silently contradicted a newly added claim in the README. Reviews
sample. Written invariants with executable checks hold.

**Adding a consensus-affecting change means adding or updating an invariant
here, and a check that enforces it.**

## The invariants

| | Invariant | Enforced by |
|---|---|---|
| **I1** | **Determinism.** Same epoch inputs ⟹ byte-identical scores on any honest validator. | `scripts/check_determinism.py` (both modes, in CI — 9 stages covering the live path), `fugal_subnet/determinism.py`, `check_safety_invariants.check_epoch_id_single_source`, TEE proof verification (all validators verify the same attested proof) |
| **I2** | **Bounded ingestion.** No miner-supplied bytes reach deserialization, allocation, or execution without size, shape, and value bounds. | `run_miner_attacks.py`, `tests/test_head_properties.py`, `check_safety_invariants.py` (no-pickle) |
| **I3** | **Monotonic incentive.** A miner cannot raise its score except by routing better or more cheaply. Artifact-keyed evidence with miss=0 prevents selective publication; the burn-in ramp prevents penalty-washing by reset. | Commit-reveal, behavioural dedup (global model index), evidence accumulation, `run_attacks.py`, `run_tee_attacks.py` |
| **I4** | **Non-interference.** A miner cannot lower another miner's score, prevent them being scored, or move the reference they are scored against — including by being present or absent. | `tests/test_non_interference.py`, TEE architecture (no shared model pool), nonce-derived exploration targets, reference frame pooled over time |
| **I5** | **Bounded spend.** No miner behavior can make a validator exceed its budget. Validators verify proofs — zero inference cost. | TEE architecture (miners pay their own inference), `tests/test_paid_safety.py` |
| **I6** | **Liveness.** No miner behavior can stop a validator completing an epoch and setting weights, and the validator collects at a point where proofs can exist. | `run_miner_attacks.py`, property test P1, TEE proof timeout, `slicer.collect_block_for_epoch`, `tests/test_collection_point.py` |
| **I7** | **Auditability.** Any divergence between two validators is diagnosable after the fact from published artifacts. | `fugal_subnet/fingerprint.py`, `environment` block in every `reveal.json` |
| **I8** | **TEE integrity.** Every claim a proof makes is bound to something the miner cannot forge: the hardware's own measurement registers, or a hash chain rooted in the attestation. No miner-influenced code executes inside the enclave that produces the proof. | `fugal_subnet/tee/verify.py`, `attestation.measurement_id`, `run_tee_attacks.py` (11 cases, in CI), `check_tee_safety`, `config.HARNESS_ALLOW_EXEC`, `tests/test_grader_policy.py` |
| **I9** | **Reference-frame agreement.** Every validator derives the same reference frame from the same published exploration samples, and no single miner can materially move it. | `fugal_subnet/reference_frame.py` (order-independent accumulation), `check_determinism.py` `frame` stage, `tests/test_non_interference.py` |

## How TEE resolves prior gaps

### I1 — matrix agreement (resolved by architecture)

**Previous gap:** Two validators calling the same model on the same question
get different responses. The matrix diverges, scores diverge, validators
disagree. LLM APIs are not deterministic even at temperature 0.

**Resolution:** With TEE, validators no longer compute their own matrices.
Miners run benchmarks inside Intel TDX confidential VMs and produce
hardware-attested proofs. All validators verify the same attested proof, so
they agree by construction. The I1 gap is closed.

**Measured across architectures.** The whole validator scoring path — slice,
exploration, routing decisions, proof, verification, reference frame, scores,
weights — was run on x86_64 (AVX2) and on aarch64 (Ampere) and produced
byte-identical digests at all nine stages. That is the property consensus
actually rests on, and it is now evidence rather than assertion.

The backbone is a separate question and a weaker one. `check_determinism.py`
feeds the pipeline `rng.randn` embeddings, not Qwen3-0.6B — the stage is named
`synthetic_hidden` for that reason. Measured directly, the real forward pass
differs between x86_64 and aarch64 in the last bits of float32 (~2e-9 per
element). This does **not** fork consensus:

- Validators never run the backbone. Only miners do, and only for their own
  routing; a validator recomputes nothing float-dependent from the head.
- Under `--live` the architecture is pinned by the approved TDX measurement,
  and TDX is Intel-only, so every attested miner is x86_64 by construction.

The residual is that two miners running an identical head on different
architectures could in principle diverge on a routing decision whose top two
logits are within 1e-9, which would make a copied head marginally harder to
deduplicate. Under `--live` that path does not exist. It is recorded here
rather than fixed because the fix — pinning a cross-architecture backbone — buys
nothing that the measurement does not already buy.

### I6/I9 — when a validator collects proofs is consensus state

A miner cannot begin benchmarking until its epoch's boundary block exists,
because that block's hash is what selects the questions — hiding the slice
until that moment is the whole anti-overfitting design. There is therefore an
unavoidable gap between an epoch starting and any proof existing for it.

A validator that queries at the boundary asks before anyone can answer. With
real models on a 300-question slice the miner needs minutes; the validator
collects nothing, and then marks the epoch processed and never returns to it.
Every epoch, forever, logged as `no valid proofs` as though the miners were at
fault. That is a liveness failure (I6) wearing the costume of a miner problem.

Retrying until proofs appear would fix the symptom and break I1. How long a
validator happened to wait would decide which miners it scored, so two honest
validators would grade different fields and publish different weights — a
divergence with no bug behind it and nothing in the artifacts to explain it.

So the collection point is derived, not chosen: `collect_block_for_epoch`
returns `boundary + fraction × blocks_per_epoch`, clamped strictly inside the
epoch, and it is the only place that computes it (`tests/test_collection_point.py`
asserts that structurally, because a test that derives the value once and hands
it to both sides cannot catch a second implementation — which is exactly how
the epoch id came to be formatted two different ways). The metagraph is read at
that same block, so every validator queries the same axons at the same chain
state. `EPOCH_COLLECT_FRACTION` is in the consensus digest, so a validator
collecting at a different offset diverges visibly instead of silently.

Half the epoch is the default: miners get half to benchmark, validators get
half to verify, score, set weights and publish the reveal.

### I4 — pool manipulation (resolved by architecture)

**Previous vulnerability:** Sybil registrations declaring cheap models could
evict a victim's models from the shared pool, zeroing the victim's accuracy.

**Resolution:** With TEE, each miner runs their own benchmark inside their
own TEE VM. There is no shared model pool to manipulate. The attack is
eliminated by architecture, not by a code fix.

### I5 — cost asymmetry (resolved by architecture)

**Previous concern:** A $1 miner registration could waste $30+ of validator
inference per epoch. The validator computing the matrix was the wrong
architecture — every other Bittensor subnet has miners pay for expensive work.

**Resolution:** Validators verify proofs, never call models. Zero validator
inference cost. Miners pay for their own API calls inside the TEE, metered by
the attested MeteringProxy.

### I8 — BROKEN AS IMPLEMENTED: the measurement does not cover the miner's code

**`measurement_id` proves which operating system booted. It does not prove which
Fugal code ran.** Measured directly on a live TD: a line was appended to
`fugal_subnet/tee/harness.py` and the TD re-measured.

    before   a68d0ccd3473a6c44fa9689cfdcf22066bc83e232efeaf33cc0341d0c532ae65
    after    a68d0ccd3473a6c44fa9689cfdcf22066bc83e232efeaf33cc0341d0c532ae65

Identical. **Those two values are stale and must not be copied into an approved
list.** They were computed while `measurement_id` still included RTMR0. The same
image, same machine shape, now measures
`a1ecb6273d38bd8b5ad629edf12f48d1c3b861fa8f506d1e1e303783e68461f3`. They are
kept here because they are what the experiment produced, and the finding is
unaffected either way: editing the harness moves neither the old value nor the
new one.

`measurement_id` is sha256 over MRTD, RTMR1 and RTMR2 — the image, the kernel,
the cmdline and initrd — and `attestation.measurement_id` excludes RTMR3
deliberately, because RTMR3 is application-extendable and including runtime data
would mean no image could stay on an approved list. That reasoning is sound and
its consequence is fatal: the repo is git-cloned onto an **unmeasured
filesystem** at runtime, and nothing measures it.

So a miner can edit the line in `harness.py` that records results, inside a
genuine TD running a genuinely approved image, and produce a proof that passes
every check in `verify.py`. The chain this module documents:

    measurement_id(quote) in approved_measurements
      -> the image that ran is the published one          <- OS only
    report_data == proof.content_hash()
      -> the proof body is exactly what that image produced  <- FALSE

The second claim does not hold. It is "exactly what *something running on that
OS* produced". `verify.py` correctly rejects `proof.source_hash` as
self-declared and names the exact attack — *"DCAP verification passes for an
attacker who simply runs modified code on real hardware"* — then defends against
it with a measurement that does not include the code. The right kind of defence,
over the wrong bytes.

**What the real-hardware negative control actually proved.** A second VM on the
same machine type but a **different OS image**
(`ubuntu-2204-jammy-v20260826`, measurement `498288bb2464…`) produced a genuine
Intel-signed quote that passed DCAP and was still rejected as an unapproved
runtime image. That is real and worth having: a different boot chain is refused.
It is *not* what an earlier revision of this document claimed — that modified
*code* is refused. Modified code on an approved image is accepted.

**What would fix it.** The code has to be inside the measured boot chain:
baked into a purpose-built guest image so it lands in MRTD/RTMR1/RTMR2, or on a
dm-verity volume whose roothash sits on the kernel command line, which the
bootloader measures. Extending RTMR3 *from the application itself* cannot work —
an attacker who controls the code controls what it extends.

**RTMR3 is not categorically useless, and this matters for the dstack decision.**
`measurement_id()` excludes RTMR3 with this reasoning: it is the
application-extendable register, so including runtime data would mean no image
could ever stay on an approved list. That is correct for a register extended by
the workload at arbitrary times with arbitrary values. It is **wrong** for a
register extended once, at boot, by measured launch code, with a deterministic
hash of the application — which is exactly what dstack does, extending RTMR3
with the app/compose hash before the workload runs.

So adopting dstack **without changing `measurement_id()` would reproduce this
exact gap at the cost of weeks**: a better-built base image, still not binding
the miner's code, because the register that binds it is the one being thrown
away. Any decision to adopt dstack must include the consensus change to
`attestation.py` that brings RTMR3 into the identity — deliberately, with the
distinction above written down, not discovered afterwards.

The general form, worth keeping: *an exclusion justified by an assumption
survives the assumption's death unless someone re-reads the justification.*

### I8 — two changes made now, ahead of the image work

Neither closes the gap above. Both are prerequisites for closing it, and both
are cheap enough that deferring them to the image migration only means doing
them under more pressure.

**1. RTMR0 is no longer part of `measurement_id()`.** It records the TDVF
configuration the host builds — virtual hardware setup, CPU count, memory size,
device layout — which the cloud provider chooses and we do not. It is not a
per-image discriminator: the same pinned image on two machine shapes produces
two identities, so an approved list forks by instance size while proving
nothing about the code. Measured on live TDX, and dstack documents the same
behaviour. Thirty Spokes (Bittensor SN99), whose attestation patterns this
module was forked from, excludes RTMR0 for exactly this reason; the fork kept
RTMR0 and dropped RTMR3, which is backwards on both counts.

The identity is now `sha256(MRTD || RTMR1 || RTMR2)` — the image, the kernel,
and the cmdline/initrd. **Every previously published measurement value changes.**
Nothing is live on the old values, so there is no migration; a `--live` subnet
would have needed a coordinated approved-list update.

Checks: `tests/test_tee.py::test_measurement_id_ignores_rtmr0_machine_shape`,
`::test_measurement_id_ignores_rtmr3_application_register`,
`::test_measurement_id_tracks_the_boot_chain`.

**2. The runtime now extends RTMR3 at startup, and this is ADVISORY.**
`attestation.runtime_identity()` is `sha384(source_hash || pool_hash ||
grader_hash)` — the three things that decide what a proof means — and
`extend_rtmr3()` writes it through the kernel's tsm-mr interface
(`/sys/class/misc/tdx_guest/mr/rtmr3`) before any proof exists.

It proves nothing today, for the reason stated above: an attacker who controls
the code controls what it extends. It is written anyway because when the image
is locked, the same value is extended from a measured initrd instead of from
userspace, and at that point it becomes evidence without a new mechanism. The
value is deliberately free of per-epoch data — no nonce, no slice, no results —
so it is fixed for a deployment and computable off-hardware from the repo,
which makes an approved list of it reviewable in a pull request rather than
requiring someone to hold a quote.

`extend_rtmr3()` returns False rather than raising when the interface is absent,
and the miner logs a warning naming what is not bound. A silent success would be
indistinguishable from a binding that never happened — the same failure shape as
the humaneval zero.

**Verified on hardware**, and the verification found a bug that testing could
not. On a live `c3-standard-4` TD (6.17.0-1022-gcp):

- The interface is at
  `/sys/class/misc/tdx_guest/measurements/rtmr3:sha384`, **not**
  `/sys/class/misc/tdx_guest/mr/rtmr3` as first written. The directory is
  `measurements` and the register name carries its hash algorithm. With the
  wrong path `extend_rtmr3` returned False forever and logged *"this kernel may
  predate the tsm-mr interface"* — a wrong diagnosis of a path typo,
  indistinguishable from the feature genuinely being absent. Unit tests passed
  throughout, because a missing file is exactly what they assert on a
  non-TDX host.
- The write succeeds, and the new RTMR3 **appears in the next Intel-signed
  quote** — sysfs value and quote value identical.
- `measurement_id` is unchanged by the extend, as designed.

### I8 — verified on hardware: the upstream binding, and a fresh TD

The upstream is genuinely bound, measured rather than argued. Two boots of one
c3-standard-4, everything held constant except `FUGAL_OPENROUTER_BASE`:

| upstream | RTMR3 after one extend |
|---|---|
| `https://openrouter.ai/api/v1` | `9c16a3b5987a572b…` |
| `http://127.0.0.1:8799` | `61f579c30c23d544…` |

In both cases `expected_rtmr3(identity)` reproduced the value inside the
Intel-signed quote exactly. A miner who repoints the upstream at a server of
their own can no longer present the same runtime identity as an honest one.
Advisory until the image is locked, like everything in this register — but the
mechanism is measured, not asserted.

**A fresh TD starts from zero.** A GCP confidential VM stop/start produces a new
Trust Domain with RTMR3 cleared — confirmed on both boots above. So a miner
cannot accumulate extends across restarts to walk the register toward a chosen
value: the chain restarts from 48 zero bytes every boot. That closes the obvious
question about an advisory, userspace extend.

### I8 — the TPM half on GCP, and why it is not a service call

On GCP, dstack's attestation is a distinct variant: a TDX quote **and** a TPM
quote. Verifying only the TDX half is not an option, and if some validators
check the TPM half while others do not they disagree on validity — a consensus
fork with no bug behind it. It has to be one rule for everyone.

`tpm-qvl` does not exist on PyPI, so the rule has to be built. Three paths were
examined and the middle one is a trap:

| | |
|---|---|
| **`cryptography`, hand-rolled — CHOSEN** | Parse `TPMS_ATTEST`, verify the quote signature against the vTPM attestation key, chain the AK certificate to Google's root. ~200 lines, deterministic, no native dependency, no external service, pinnable and auditable the way `graders.py` is |
| **`google-cloud-confidentialcomputing` — REJECTED** | Google's Attestation Verifier: hand it the attestation, receive a signed verdict |
| `tpm2-pytss` | CFFI over `libtss2`. Every validator would need the native library installed, to marshal structures we can parse ourselves. Fine as a cross-check, poor as the dependency |

**Why the service call is rejected**, since it is the easy option and looks
harmless: it would make Google a **trust root and a per-proof availability
dependency for every validator**, which is the exact property that ruled out
GCP Confidential Space when the stack was chosen. It is strictly worse than the
Intel PCS dependency already accepted — PCS serves *static collateral* every
validator can cache, whereas a per-proof verdict from a live service can in
principle differ between callers or over time, and two validators disagreeing on
validity is a consensus fork. A subnet whose notion of a valid proof is "what
Google's API said this morning" is not verifying anything.

**Budget it as real work.** ~200 lines of security-critical binary parsing plus
certificate-chain validation, consensus-affecting, and it needs the same pinning
and attack-suite discipline as `dcap-qvl`. It is not a dependency bump.

### I8 — built: the TPM half, and the second trust root it adds

`fugal_subnet/tee/tpm.py`, verified against two real dstack attestations from
two different GCP Confidential VMs (`tests/fixtures/attestation_{A,B}.bin`).
Both real ECDSA P-256 signatures verify; a single flipped bit in the quote
fails; one machine's quote presented under the other's certificate fails.

**This adds Google's CA as a trust root, alongside Intel's.** Stated plainly
because it is a change to the threat model, not an implementation detail: the
subnet now believes a TPM quote because a certificate chains to
`CN=EK/AK CA Root, O=Google LLC`. Intel attests the TD; Google attests that the
machine underneath is a genuine Confidential VM. Neither replaces the other and
a validator must check both — checking one and not the other is a fork.

**The chain is pinned, never fetched.** The AK certificate's AIA extension
points at `http://privateca-content-<uuid>.storage.googleapis.com/.../ca.crt`.
Fetching it per proof is refused on three grounds, in increasing order of
seriousness: it is plain HTTP; it is a live availability dependency for every
validator; and two validators fetching at different moments can get different
answers, which is a consensus fork with no bug behind it — the same objection
that ruled out Google's Attestation Verifier service. The root is vendored at
`fugal_subnet/tee/roots/google_ek_ak_root.der`, hash-pinned in **two** places
that `check_safety_invariants.py` forces to agree (`tpm._ROOT_SHA256` and
`PINNED_TPM_ROOT_SHA256`). A substituted anchor does not fail loudly on its
own — it just starts accepting a different CA's attestations — so it gets the
grader treatment.

**Intermediates rotate, so they are not the anchor.** The observed intermediate
was issued 2026-05-19 and lives at a per-CA-instance URL that cannot be
enumerated in advance; pinning it would reject miners the day Google rotates,
and pinning a per-region set would reject any region not yet vendored. Instead
`verify_ak_chain(..., extra_certs=...)` lets a proof carry its own
intermediates, exactly as TLS does: a supplied certificate is worthless unless
it chains to the pinned root. Determinism is preserved because the bytes are in
the proof rather than on the network. One intermediate (us-central1) is vendored
so the common case needs nothing extra.

**Signature and chain are one check, not two.** Either alone proves nothing: a
signature without a chain accepts a quote the miner signed with their own key
and their own certificate; a chain without a signature accepts a genuine
certificate stapled to somebody else's quote. `verify_tpm_quote` exists so no
caller can do one half and believe it did both. `tests/test_tpm.py` defeats each
half separately and shows neither defeat survives composition.

**Certificate validity is the one time-dependent input**, so it is a parameter
(`at=`) rather than a hidden call to the clock, and two validators evaluating a
certificate across its expiry boundary would disagree. Observed leaves are valid
for 30 years, so the window is wide — but this was not theoretical while writing
it: the fixture's `notBefore` was 56 minutes in the past, and the same test
written an hour earlier would have failed for a reason that has nothing to do
with correctness.

**A proof is not anonymous.** The AK certificate's subject is
`CN=<instance id>, OU=<GCP project>, O=Google Compute Engine, L=<zone>`. Every
miner's project name, instance id and zone are public in their proof. Not a
vulnerability, but miners are told in `MINER_GUIDE.md` rather than discovering
it.

**Wired into `verify_proof`, and the shape is decided by the bytes.** A miner
under dstack sends the whole attestation envelope, not a bare quote.
`unwrap_attestation` reads a u16 little-endian version at offset 0: 4 or 5 is a
bare TDX quote and passes through unchanged, anything else is decoded as a
dstack blob. Which shape arrived is therefore never a configuration question —
two validators configured differently would otherwise disagree about identical
bytes.

**If an envelope carries a TPM quote, it is verified there and then.** Not
optionally, not later: a validator checking only the TDX half accepts proofs
another validator rejects. A genuine TDX quote inside an envelope whose TPM half
fails is rejected whole, and there is a test that does exactly that — flips one
bit inside TPMS_ATTEST, leaves the TDX quote untouched, and requires rejection.
A missing `cryptography` raises rather than returning invalid, for the same
reason a missing `dcap-qvl` does: it is the operator's misconfiguration, and
downgrading it to "this proof is bad" lets a --live validator reject the entire
field while appearing healthy.

### I8 — a proof names the miner it was produced for

`content_hash` covers the epoch, the slice, the head and the runtime — and,
now, the **hotkey**. It is bound into the TDX `report_data`, so the hardware
attests whose proof this is.

**What it was before.** A proof was bound to no miner at all. The attack, which
is now `a_relayed_proof` in the TEE suite:

    A commits weights_hash_A, runs an honest TD, serves head + proof.
    B reads A's head off A's own axon response — it is served inline, publicly.
    B commits weights_hash_A as its own; nothing prevents committing another
      miner's hash.
    B relays A's proof verbatim.
    verify_proof PASSED.

What stopped it was `dedup.find_duplicates` plus commit-block seniority: B's
routing vector is identical and its commit block is later, so B is disqualified.
That defence is real and still stands as defence in depth — but it is a cosine
threshold and a block ordering, which is a **statistical** answer to a question
with a **cryptographic** one.

**Two moves, two different failures.** Relaying A's proof under B's hotkey fails
the identity check. Rewriting the field to B's hotkey fails `report_data`,
because the quote was signed over the original `content_hash`. Both are in the
attack suite, and the second is the canary: if it ever reports EXPLOITED, the
hotkey has fallen out of `content_hash` and the binding is decorative.

**Enforced in every mode**, like the `report_data` binding and for the same
reason: this is a property of the proof's content, not of the silicon, so a
local testnet catches relay too. Mock mode is the absence of hardware checks,
not a weaker set of rules.

**A missing `expected_hotkey` raises rather than skipping.** Verifying without
it silently restores the property this removes, which is the failure shape this
document exists to catch. Callers that genuinely have no metagraph — attack
fixtures, determinism runs — pass `mock=True`, which is the honest way to say
"not checking hardware".

**The hotkey is public, so binding it costs no confidentiality.** It is on
chain. It reaches the TD over the provisioning channel as `hotkey_ss58`, which
is MAY-CROSS by the channel's own rule. A miner who pushes somebody else's
hotkey produces proofs bound to *that* hotkey, which their own uid cannot use.

**Checked first**, before unwrapping or DCAP: it is the cheapest check that can
reject, it needs no hardware, and a relayed proof should not cost a DCAP
verification to refuse.

### I8 — the app-identity chain, closed against hardware

Every link verified on a real deploy rather than argued, in
`tests/test_tpm.py::test_the_compose_file_bytes_reach_the_signed_register`:

    sha256(app-compose.json RAW BYTES)  ==  the compose-hash event in the log
    the event log                       replays to RTMR3
    RTMR3                               is inside the Intel-signed quote

`ced4501de5b82ea9...`, from `tests/fixtures/app-compose_A.json` through to
`attestation_A.bin`. The compose file's provenance matters: it is the
`app_compose` string the **guest** returned from `/v1/Info` — the bytes the
guest itself hashed, not a local reconstruction — so the two fixtures are
evidence about each other rather than two copies of one assumption.

This is what makes an approved entry computable off-hardware, and it is the
check that was missing when the compose hash was computed from normalised JSON.
That version produced a hash dstack never extends: it would have rejected every
honest miner while looking correct.

**Do not re-serialise.** Fields are in insertion order, not sorted. dstack's
bytes happen to equal `json.dumps(obj, indent=2)` exactly, and that is asserted
so a change is noticed — but nothing depends on it. Hashing a re-serialisation
instead of the bytes is guessing at their writer again, and that has already
been paid for once.

**`report_data` is right-zero-padded by the guest, not hashed.** Measured: a
5-byte value came back as those bytes followed by 59 zeros, and the guest
neither hashes nor rejects a short value. A 32-byte content hash therefore
appears in the quote as `content_hash || 32 zero bytes`, which is exactly what
`verify_proof` compares against. A different convention here would fail every
live proof on a binding check that looks like tampering rather than a mismatch.

### I8 — OPEN DECISION: the miner's per-instance data channel

**Unresolved, and it changes what a miner's `app-compose.json` looks like, so it
blocks packaging.** Recorded here rather than decided, because it is a product
question and not a packaging one.

The constraint chain, each link verified in dstack's source rather than inferred:

    encrypted env vars       are decrypted with a key fetched from KMS at boot
    therefore .env           is refused unless key_provider == "kms"
                             (a hard client-side raise, hit in practice)
    key_provider == "kms"    boot-loops against Phala's public KMS
    key_provider == "local"  needs dstack-vmm, which a GCP CVM does not have
    therefore                key_provider == "tpm", and .env is unavailable

So `allowed_envs` — the obvious answer, and one this guide briefly stated as a
requirement before it was falsified — **cannot deliver the key**.

**Putting it in the compose is not the fallback.** `app-compose.json` is
returned in full by the guest agent's `/v1/Info` as the `app_compose` field;
that is how this repo's reference fixture was obtained. A key there is readable
by anyone who reaches the agent and becomes part of an approved entry's
provenance. Unwithdrawable.

Three options, none free, none chosen:

| | Cost |
|---|---|
| **Self-hosted KMS** — dstack-cloud can stand one up (`kms deploy`/`attest`), restoring encrypted env with `key_provider=kms` | Either every miner runs a KMS, or the subnet runs one and miners trust it — a new centralisation, which is the thing dstack was chosen to avoid |
| **Miner fetches its key at runtime** from an endpoint it controls, after boot | Keeps the key out of the measurement entirely, but makes the fetch endpoint miner-controlled input reaching inside the TD — the `FUGAL_OPENROUTER_BASE` problem in a different coat |
| **The subnet supplies the key** | Changes who pays for inference; an economic decision, not a deployment one |

Note the first two converge: a self-hosted KMS *is* an attested key-fetch
endpoint. The real axis is who operates it and who is trusted.

**Nothing should be improvised here.** A miner who invents a fourth answer will
most likely put the key somewhere `/v1/Info` publishes it.

#### The question is bigger than the key, and that makes it easier

An API key is not the only per-miner value a miner needs inside the TD. From
`neurons/miner.py`, the minimum is:

| | |
|---|---|
| the **wallet keyfile** | a secret, and a different one per miner |
| the **head artifact** (`--head-path`) | per miner, and it *changes between epochs* — updating it is the entire competitive activity |
| the **API key** | a secret, per miner |
| netuid, network, port | uniform across miners; these can live in the compose |

Two of those three cannot be in `app-compose.json` for a reason that has nothing
to do with secrecy: **the compose hash must be identical across every honest
miner.** It is the app identity in the approved list. Put anything per-miner in
that file and every miner has a distinct hash, the approved list needs an entry
per miner, and the whole single-approved-application model collapses. The head
settles it on its own — a value that changes every epoch can never live in a
measured file.

So a post-boot channel for per-instance data is **required regardless of how the
key question is answered**. The key is a passenger on a channel that has to
exist anyway. That is the useful reframing: we are not choosing a secret-delivery
mechanism, we are choosing the miner's data channel, and encrypted env was only
ever one candidate for it.

It also means the "miner-controlled input inside the TD" objection is not an
argument against the channel — the channel is unavoidable — but a constraint on
what may travel it. The existing pattern already handles this correctly and
should be extended rather than replaced: the head arrives as untrusted bytes and
is *bound* by an on-chain `weights_hash` committed before the nonce. Untrusted
input that is bound is safe; unbound input is not. Concretely, what may cross:

  - **secrets and per-miner artifacts** — wallet, head, API key. Bound where
    consensus depends on them (the head already is), irrelevant where it does
    not (the wallet and key are the miner's own).
  - **NOT the upstream URL, the pool, the grader, or anything else consensus
    reads.** Those are in the measurement precisely so a miner cannot choose
    them, which is what `runtime_identity(source, pool, grader, upstream)` is
    for. A channel that can carry `FUGAL_OPENROUTER_BASE` hands a miner the
    upstream-substitution exploit that `scripts/stub_upstream.py` demonstrates.

That line — secrets and bound artifacts yes, consensus inputs never — is the
thing to decide and then enforce, and it is narrower than "how do we do KMS".

#### Resolved on hardware: the channel exists, and it is not confidential

Measured on a GCP `c3-standard-4` dstack CVM, not read from documentation.

**`key_provider` is `tpm`.** It boots clean — no KMS contact, no restart loop —
and `tpm-attest` maps both `Platform::Dstack` and `Platform::Gcp` to the dstack
PCR policy, sealing a 32-byte seed into the vTPM so app keys survive a redeploy
and are bound to the measured boot state. `kms` boot-loops against the public
KMS, `local` needs a `dstack-vmm` a confidential VM does not have. It is the
only workable value and also the best one.

**`key-provider` is its own RTMR3 event**, carrying `{"name":...,"id":...}`.
So a validator can require a provider by reading the *authenticated log*,
independently of the compose hash — meaning a provider change does not mint a
new app identity to approve. Verified against both fixtures, which carry
`"none"`. No rule is enforced yet; the field is asserted so that deciding to
require it later is a small change rather than a discovery.

**The compose bytes survive upload unchanged**, diffed local-vs-guest on a real
boot: 807 bytes, identical, `sha256(local) == guest compose_hash`. Nothing
re-serialises on the way in, so `compute_app_identity.py --compose` is
trustworthy in a pull request. Worth moving to dstack's `app_compose_file` mode
regardless, so the property holds by construction rather than by observation.

**`.user-config` is the per-instance channel.** Operator-supplied JSON shipped
on the shared disk, read only when `requirements.launch_token_hash` is set, and
**not part of the compose hash** — exactly the unmeasured per-miner shape the
head requires. The shared-disk file list is fixed (`app-compose.json`,
`.sys-config.json`, `.instance_info`, plus `.encrypted-env` and `.user-config`),
so nothing arbitrary rides along.

**But the shared disk is plain FAT32 and is not encrypted.** It is built
locally, uploaded to GCS and registered as a GCP disk, so anything in
`.user-config` is readable by Google and by anyone with read access to the
miner's project. Against the may-cross list above:

| Payload | Verdict |
|---|---|
| **head artifact** | **Fits.** Not secret, changes every epoch, and already bound by the on-chain `weights_hash` committed before the nonce. This is the payload that made a channel unavoidable, and this is the channel for it. |
| **wallet hotkey** | **No.** Plaintext to Google, and it signs the miner's on-chain identity. |
| **API key** | **No.** Plaintext to Google, and it is the miner's money. |

So the channel is free and solves the hard payload, and **the open question is
now narrowed to secret delivery only** — a self-hosted KMS, or a post-boot fetch
the miner authenticates. Those converge: a self-hosted KMS *is* an attested
key-release endpoint.

`/dstack/persistent` is **not** an ingress path, despite appearances: the guest
LUKS-formats it with a key derived from the app keys, which under `tpm` come
from the TPM-sealed seed. The operator holds no key and cannot pre-populate it.
It is persistence *for* the TD — useful for the embedding cache across restarts,
useless for delivery.

**The ceiling is 50 MB, enforced in the guest.** Written as a number because
"plenty of room" ages badly and "50 MB" does not. `dstack-util`'s `HostShared::copy`
size-checks every host-shared file as it copies it in and bails rather than
truncating:

| File | Limit |
|---|---|
| `app-compose.json` | 50 MB |
| `.user-config` | **50 MB** |
| `.encrypted-env` | 256 KB |
| `.sys-config.json` | 32 KB |
| `.instance_info` | 10 KB |

The 8 MB figure that appeared here first was a floor, not a ceiling, and reading
it as a ceiling was backwards: the shared disk is sized
`max(8 MB, contents + 4 MB)` and grows to fit what is put on it. Against a real
head — `data/rehearsal/head_m1.npz` is 77,234 bytes — that is roughly **679x
headroom**, so a head would have to grow three orders of magnitude before the
channel is threatened. Settled, not pending.

Note `.encrypted-env` caps at 256 KB. That is the real constraint on the KMS
path if it is ever revisited, and it is far tighter than the one that was
worried about here.

**Five unidentified trailing bytes.** Real `TPMT_SIGNATURE` fields carry five
bytes past the structure (`0000010000`). They are not identified and are not
guessed at. Ignoring them is safe because `r` and `s` are read from fixed
offsets at the front, so nothing appended can change what is verified. Recorded
so that nobody later mistakes silence for knowledge.

### I8 — there is no hourly TDX bare metal, so there is no quick fallback

Recorded because the obvious escape hatch from the Google trust root does not
exist in the form everyone assumes. Checked, and both first guesses were wrong:
Equinix Metal was sunset 30 June 2026, and Latitude.sh's confidential compute is
AMD SEV-SNP only, not TDX. Providers that do rent TDX bare metal — OpenMetal,
Hydra Host, OVHcloud — bill **monthly**, and OpenMetal states outright that
there is no per-hour tier. Realistic figure for an Emerald Rapids box:
$500–1500/mo, some of it behind a sales conversation.

Two consequences, and the second is the one that matters:

**Bare metal cannot be spun up to dodge a problem.** It is a migration with lead
time, not an afternoon's work. Anything in the GCP path that turns out to be a
dead end is a real schedule event.

**Therefore the Google trust root is a permanent, named cost, not a temporary
one.** The three properties above — Google's CA as a second trust root, its
intermediate served over plain HTTP, the GCP project and instance named in every
proof — are properties of the design we are committing to, not inconveniences we
are passing through on the way to something else. That is also the argument for
permissionlessness: a subnet whose miners must sign a monthly contract and talk
to a salesperson has no miners.

**The recorded alternative, not proposed and not built.** AMD SEV-SNP bare metal
is roughly a third of the price (~$207/mo observed) and dstack supports SNP on
its stable line, with measurements derivable the same way. That would replace
Intel with AMD as the sole silicon root and remove Google from the chain
entirely. The blocker is ours: `attestation.py` hard-checks
`tee_type == TDX (0x81)` and `parse_quote` rejects anything else, so it is a
porting job rather than a configuration change. Written down so that if the
Google trust root ever becomes a problem, nobody concludes the only alternative
costs $1000/mo and drops the question.

### I8 — why we believe `dcap-qvl==0.6.3` is safe

CVE-2026-22696 (GHSA-796p-j2gh-9m2q, critical): `dcap-qvl` lacked mandatory
Quoting Enclave identity validation, so a verifier could accept quotes from an
unauthorised QE — a direct attack on this invariant. The advisory names <0.3.9
as affected and 0.3.9 as patched, and says nothing about 0.6.x.

**Version ordering is not evidence**, so the source was read rather than
inferred: v0.6.3 carries a `qe_identity` module and calls
`verify_qe_identity_signature` unconditionally in the main verify path with `?`,
so a failure is fatal; it rejects on `mr_signer` and `isv_prod_id` mismatch and
gates the expected QE id to `TD_QE`. The fix is present and mandatory.

Recorded because the next person will ask, and "we pin a version above the
patched one" is a weaker answer than "we read it". A pinned security-critical
verifier that silently misses a fix is a bad failure mode, and pinning is what
makes it possible.

### I8 — why `cryptography` is pinned at 50.0.1, and what pinning cost

The first pin was `46.0.3`, chosen because it was what the resolver produced.
CI rejected it: `pip-audit` found **11 known vulnerabilities** in that version,
with fixes spread across 46.0.5, 46.0.6, 46.0.7, 48.0.1, 49.0.0 and 50.0.0.
Clearing all of them requires >= 50.0.0, so the pin is 50.0.1.

**This is the failure mode the `dcap-qvl` note warns about, caught in the act.**
An exact pin freezes a security-critical dependency at whatever the resolver
happened to pick, and it stays frozen while advisories accumulate behind it.
Pinning is still right — a behaviour change in the TPM verifier is a consensus
change and must arrive deliberately — but it is only safe when something
actively re-checks the pinned version. `pip-audit` in CI is that something, and
it is why the audit step is not optional decoration.

The bump was verified, not assumed: 46 -> 50 is four major versions, and the
verifier does real X.509 chain building and ECDSA verification against real
Google certificates. All TPM, SCALE and dstack-client tests pass under 50.0.1
against the committed fixtures, so the artifacts confirm the API surface we use
is unchanged.

### I8 — the approved entry is a PAIR, not a hash

An approved-list entry is `<base_measurement>` or `<base_measurement>:<app_identity>`.

    base_measurement = sha256(MRTD || RTMR1 || RTMR2)     what booted
    app_identity     = sha384(source || pool || grader || upstream)   what ran

**Kept as two halves rather than one hash, deliberately.** They have different
lifecycles and different approvers: the base rotates when the image or kernel
changes, on the cloud provider's schedule; the app identity rotates when our
code, pool, grader or upstream changes, on ours. Hashing them together would
force one rotation for either event and make "what code is approved"
unreadable. Apart, the app identity is computable off-hardware straight from the
repo — so **what code the subnet accepts is reviewable in a pull request**
instead of requiring someone to hold a quote.

Rotation follows from the shape rather than needing a mechanism of its own:
publish the new value, approve both during the transition window, retire the
old. Each half rotates independently. A bare base remains valid and means "this
image is approved, nothing is required of RTMR3" — which is the honest state on
an unlocked image, where an RTMR3 match proves nothing because an attacker
running modified code simply extends the expected value.

**What locking the image changes is which half is trustworthy, not what the
halves are.** That is why this definition is not dstack-specific: dstack moves
the extend into a measured initrd and adds entries to the log, and the pair
above is unchanged.

**It is an extend, not a set.** The hardware computes
`RTMR = SHA384(RTMR || input)` from 48 zero bytes at boot, so the register never
holds the value written. Confirmed by replay: writing
`sha384(b"fugal-runtime-test")` to a fresh TD produced
`f6fdca9f66372d80685dc6be023d9e6ae25a42334f3b59de24c78da941883a64…`, equal to
`sha384(bytes(48) + input)`.

The consequence is a design constraint for the verifier: **RTMR3 can never be
compared to `runtime_identity()` directly.** A verifier must replay the chain of
extends from zero and check the result equals the quote's RTMR3 — the same
event-log replay dstack requires, now confirmed as a property of the hardware
rather than a dstack convention.

Checks: `tests/test_tee.py::test_runtime_identity_is_register_width_and_deterministic`,
`::test_extend_rtmr3_reports_failure_rather_than_pretending`.

**Until then `--live` binds less than it appears to**, and no rotation procedure
changes that: rotating an approved list of measurements that do not cover the
workload rotates a value proving only which OS booted.

Operational consequence for validators: **collateral is fetched from Phala's
PCCS, not from Intel.** This was recorded here as Intel for months and was
wrong. `verify_dcap` calls `get_collateral_and_verify(quote)` with no
`pccs_url`, and `dcap-qvl` 0.6.3 resolves that as:

    url = (pccs_url or "").strip() or PHALA_PCCS_URL   # https://pccs.phala.network

So a `--live` validator needs outbound HTTPS to `pccs.phala.network`, and a
validator firewalled to `api.trustedservices.intel.com` on the strength of the
old sentence would fail every verification while its configuration looked
correct.

**What Phala is and is not.** It is an availability dependency and a privacy
leak: every miner's quote is sent to a third party, and a PCCS outage degrades
verification for every validator at once. It is **not** a correctness trust
root. Collateral is Intel-signed TCB info, QE identity and CRLs, and `verify`
checks those signatures against an Intel root CA compiled into the library
(`IntelSGXRootCA.der`; `verify_with_root_ca` exists to supply a different one).
A malicious mirror cannot forge collateral — it can only withhold it. That
distinction is why this is an operational finding rather than a repeat of the
Google Attestation Verifier decision, which was rejected because a live service
returned a *verdict*. A mirror returns *evidence* we verify ourselves.

**Two related defects in the same function**, found with it and not yet fixed
because both are decisions rather than typos:

  - The `timeout=30` guards only the `ThreadPoolExecutor` branch, which runs
    when an event loop is already running. On the validator's synchronous path
    `loop.is_running()` is False, so `run_until_complete` is used with **no
    timeout at all**. Measured: a stub sleeping 60s blocked `verify_dcap` for
    the full 60s.
  - `verify_dcap` **discards the verdict**. `get_collateral_and_verify` returns
    a `VerifiedReport` with `.status` and `.advisory_ids`; the function logs it
    and returns True for anything that does not raise. So **at minimum,
    out-of-date, configuration-needed and software-hardening-needed platforms
    pass** — those are observable return values that are thrown away.
    **Whether `Revoked` also passes is UNCONFIRMED.** The compiled library
    contains the literal "TCB status is invalid: Revoked", which would mean
    `verify` raises and `verify_dcap` returns False via its except branch — but
    that string cannot be disambiguated from a string table, because Rust
    concatenates adjacent literals and `Revoked` is also an enum name. Pulling
    the other way: `.status` is documented as returning `REVOKED`, and
    `QuotePolicy.allow_status` would be pointless if plain `verify` already
    rejected everything but UpToDate. **Settling it needs a real quote**, not
    more reading. Recorded at the strength the evidence supports rather than the
    strength that makes the better warning.

**How the wrong endpoint got recorded, because the shape recurs.** Nobody
measured Intel. The observation was "no local PCCS is configured and
verification succeeds", and the conclusion drawn was "therefore it goes to
Intel directly". The real explanation was a library default nobody looked for.
An inference was written down as a measurement, and an endpoint appeared in
this document that had never been seen in a packet or a line of source. The
correction came from reading `dcap_qvl/__init__.py`, not from observing traffic.
Fourth instance today of a real observation describing a neighbouring thing.

### I1/I6 — the DCAP verdict is already a function of network luck

The deepest version of the PCCS findings, and the one that decides what the fix
has to be. `verify_dcap` ends:

    except Exception:
        return False

A transient network failure is therefore not "unverifiable", it is **invalid**.
Two validators verifying the *same proof* over *different network luck* reach
*different verdicts*. That is a consensus fork with no bug and no attacker
behind it, and it exists **today**, with no caching and no status enforcement
anywhere. At 256 miners fetching per proof, transient failures are not an edge
case; they are expected.

**So "does caching introduce a consensus hazard?" is the wrong question.** The
verdict is already nondeterministic across validators. Caching, status
enforcement and the time source are not three independent choices — they are
three faces of one requirement:

> **The verification verdict must be a function of the proof, not of the
> validator's network or clock.**

Anything short of that leaves a fork surface. Note that discarding `.status`
currently *hides* this: every non-exception collapses to True, so the only
divergence left is the raise/no-raise boundary. The discard is load-bearing for
determinism by accident, which is why enforcing status without fixing the fetch
would make things worse, not better.

**The fix that satisfies it, and it is the shape this codebase already chose
once.** `dcap-qvl` exposes `verify(quote, collateral, now_secs)` separately from
`get_collateral`, and `QuoteCollateralV3` has `to_json` / `from_json`. So
collateral can travel **with the proof**:

  - the miner fetches collateral once and ships it in the bundle;
  - the validator calls `verify` directly, with **no network in the epoch loop
    at all**;
  - a miner cannot forge it, because collateral is Intel-signed TCB info, QE
    identity and CRLs, checked against the Intel root CA compiled into the
    library. Untrusted carrier, pinned anchor.

That is exactly the decision already taken for Google's AK intermediate
certificates, which rotate at per-CA-instance URLs and therefore travel with the
proof rather than being fetched or pinned. Same problem, same answer, and the
precedent is evidence the pattern fits rather than a coincidence.

It removes the fork, the hang, the availability dependency, the privacy leak and
the throughput cost in one change, because all five are consequences of fetching
inside the epoch.

**The argument for it is NOT "Intel signs it, so the carrier does not matter."**
That is true and insufficient. Collateral includes `root_ca_crl` and `pck_crl`
— **the revocation lists**. So collateral-in-proof hands delivery of the
revocation mechanism to the platform holder, and revocation is the one control
in the entire chain specifically designed to operate *against* them.

The attack, and its exact limit:

  - A miner whose PCK certificate has been revoked ships a **stale but validly
    Intel-signed CRL** from before the revocation. Every signature checks out,
    every chain reaches the pinned root, the quote verifies.
  - **They cannot swap the certificate itself.** `cert_chain_pem_bytes` is a
    field of `Quote`, not of `QuoteCollateralV3` — whose nine fields carry no
    PCK certificate chain. The certificate rides inside the attested quote. So
    the exploit is "present an old revocation list", not "present a different
    identity", which is narrower than it first appears and is what makes the
    design viable at all.

**Therefore the freshness bound is not a staleness nuisance with defence in
depth behind it. With collateral-in-proof it is the ENTIRE security of
revocation, single-layered.** Written at that strength deliberately: stated as
"bound the collateral age", someone later relaxes it for miner convenience;
stated as "this is the only thing standing between the subnet and revoked
hardware", nobody does.

**Freshness is implementable** — `tcb_info` is exposed as a JSON string, so
`issueDate` / `nextUpdate` / `tcbEvaluationDataNumber` parse in pure Python with
no binding change. That question is settled, not open.

**The real argument for the trade** is structural, not performance. Today a PCCS
problem is a **globally correlated validator-side failure**: every validator
hits one host at the same deterministic block, so one outage degrades everyone
simultaneously. Afterwards it is an **uncorrelated per-miner failure**: a miner
who cannot fetch their own collateral fails alone. That is the same philosophy
as I5 — miners bear their own costs — and it, not the saved milliseconds, is
why the trade is good. What is being traded is a **liveness and privacy
dependency for a revocation-freshness dependency**.

**Collateral would be miner-supplied input, and I2 applies.** It reaches a Rust
JSON parser via `from_json` carrying certificate chains and CRLs: size caps
before parse, and cases in `run_miner_attacks`. Note explicitly that collateral
is **not covered by `content_hash` / `report_data`** — it is unattested data
riding inside an attested bundle. Everything else in that bundle is attested, so
the assumption that this is too will be made unless it is written down. (Binding
it into `content_hash` would stop an outside process swapping it after the TD
produced the proof, but would do nothing about staleness, because the TD is the
miner's own.)

**Two things it does NOT remove, and both must be decided with it:**

  - **Stale collateral.** A miner may ship old but validly-signed collateral
    that predates a revocation of their platform. This needs a freshness bound
    checked against `tcb_info`'s issue/next-update fields — and that bound must
    be a **pinned constant every validator shares**, not a local TTL each
    operator tunes, or it reintroduces the divergence it was meant to remove.
  - **`now_secs`, and this is load-bearing rather than tidy-up.** The freshness
    bound is `next_update > now`. With local clocks, collateral near expiry is
    valid for validator A and expired for validator B — so the fork *moves*
    rather than closes. Epoch-block time is what makes the bound consensus-safe
    at all, which means the two land together or the change is net-negative.
    Same reasoning that made certificate validity an explicit `at=` parameter in
    the TPM verifier rather than a hidden call to the clock.

Not built. Recorded because the throughput work will otherwise fix the symptom
that was measured rather than the property that is wrong.

### I6 — a PCCS hang halts the subnet, and does it invisibly

Two facts recorded separately above are far worse together, so they are stated
here as one:

  - every validator fetches collateral for every proof from **one host**, with
    no caching, and they do it **simultaneously** because the collection point
    is a deterministic block (I6/I9);
  - the `timeout=30` does not apply on the validator's synchronous path, so a
    slow PCCS blocks for as long as the HTTP client allows, per proof, serially.

**The outage is the safe case; the hang is the dangerous one.** That is the
counterintuitive part and it is why this needs writing down:

| Phala state | What happens |
|---|---|
| **Down** — fast connection error | `verify_dcap` catches, returns False, every proof invalid, the epoch is skipped and logged with the `no_valid_proofs` anomaly. Visible, recoverable, no weights set. |
| **Hanging** — accepts and stalls | The validator blocks with no timeout. The epoch never completes. No weights, **no log line, no anomaly** — the code that would record the failure is downstream of the block and never runs. |

So the failure that looks less severe is the one that stops the subnet, and it
stops it silently, on every validator at once.

**I6 as written does not cover this.** It says *"No **miner** behavior can stop
a validator completing an epoch and setting weights"*, and its guard is the TEE
proof timeout — which bounds the miner query, not the collateral fetch. The
hostile miner who hangs an epoch was anticipated and defended; the same hang
arriving through a third-party dependency was not, because the invariant names
miners rather than naming the property. **A liveness invariant scoped to one
source of delay is not a liveness invariant.** Whatever fix lands here should
widen I6 to "no external party" and give the fetch a bound that holds on both
code paths.

### I8 — what "attested" actually means

DCAP verification proves a quote is genuine and Intel-signed. It proves the
*hardware* is real. It says nothing about whether the code inside it is the
published code — an attacker who owns a TDX machine produces a perfectly valid
quote while running a modified harness.

For a period this subnet checked `proof.source_hash`, a field the workload
writes about itself, against the approved-image list. That is not an attestation
of anything: the attacker simply writes an approved value. The check now reads
`measurement_id(quote)`, derived from MRTD and RTMR0-2, which the CPU fills in
and the Intel signature covers. RTMR3 is excluded because it is
application-extendable, so including it would make an image identity change
with runtime data and no image could stay on an approved list.

The full chain a proof must satisfy:

```
Intel DCAP signature          -> the quote is genuine, from real TDX hardware
measurement_id(quote)         -> the image that ran is the published one
report_data == content_hash   -> the proof body is what that image produced
weights_hash == commitment    -> the head that ran was committed before the nonce
sha256(bundled head)          -> the head shipped is the head attested
result ids == assigned slice  -> those answers are to the questions we asked
exploration == nonce targets  -> the sampling quota was actually performed
per-question costs == total   -> the cost figures are internally consistent
```

Each line was, at some point, absent — and `run_tee_attacks.py` keeps an
executable exploit for each, because every one of them fails *silently*: the
proof verifies, the miner is paid, and nothing in a log says otherwise.

### I4 — the reference frame must not move with the field

Scoring against "the best single model" needs an estimate of how good that
model is, and that estimate is built from miners' exploration samples. If it
tracked the miner population, every miner would move every other miner's score
and the same head would be worth more in a thin epoch than a busy one.

Two things keep it from doing so. The frame accumulates over *time*, not over
miners, so one epoch's samples are a single decayed contribution to a
long-running estimate. And the ceiling is valued at the posterior *mean* rather
than a lower confidence bound — an LCB's pessimism shrinks as evidence
accumulates, which made the ceiling a function of sample count and therefore of
field size (measured: 0.14 of score between a 3-miner and a 50-miner field).
The LCB still *selects* which model is the reference, so a lucky model cannot
be crowned; it just does not *value* it.

Residual: convergence speed still depends on field size. A small subnet reaches
a stable frame more slowly. The prior strength was calibrated against this
directly (see `FRAME_PRIOR_STRENGTH`) rather than chosen.

### I4 — seniority squatting (residual, accepted)

Dedup seniority tracks the earliest block at which a hotkey was ever seen
committing a valid head. A squatter who registered and committed *before* its
victim therefore still holds earlier seniority. This is far weaker than the
copy-and-outrank bug it replaced — it requires predicting the victim well in
advance — but it is not zero.

## Attack surface by actor

**Miners** control: the bytes their axon returns, their head weights, when they
commit, and how many identities they register. TEE constrains them: results are
hardware-attested, costs are metered, and the runtime image is measurement-pinned.

**Other validators** control: their own published reveals and weights. Yuma
consensus plus `MAX_WEIGHT_DELTA` bound the damage; `consensus.py` detects it.

**TEE escape:** If a miner breaks out of TDX (extremely unlikely — Intel
patches are fast), they could fabricate results. Defense: measurement pinning
detects tampered runtime images, and DCAP verification validates the attestation
chain against Intel's infrastructure.

**Benchmark datasets** are pinned by revision, so upstream changes cannot
silently alter the question pool.

## Known gaps

**Pool memorization (open).** The question pool is public and finite (~21K).
A miner willing to pay once to evaluate every model on every question could
publish a head that encodes the resulting lookup rather than a routing policy.
It would score well here and generalize to nothing.

The intended defence is held-out evaluation: the validator runs the head on
questions the miner has not been scored on and checks it still routes sensibly.
The head is already bound and shipped in the bundle for exactly this — but
running it requires backbone embeddings on the validator, and that is the part
not done. It is deliberately deferred rather than half-built, because it
reintroduces per-validator floating-point computation into consensus, which is
precisely what the TEE architecture removed. Two honest validators whose
embeddings differ in the last bits would disagree on near-tie routing and
diverge. Closing this properly needs either an agreed embedding artifact or
held-out questions that are not in the public pool.

**Benchmark pool reproducibility (open).** The pool is consensus state, and
the default loader does not produce the same one for everybody. GPQA is a gated
Hugging Face dataset, so `load_all()` raises for an operator without a token and
returns a *larger* pool for one with it. LiveCodeBench loads zero questions on
current `datasets` versions but loads normally for anyone who has manually
placed `data/benchmarks/livecode.json`. Either divergence produces proofs that
fail on `questions_hash`, an error that names the symptom and never the cause.
Until the pool is published as a hash-pinned artifact and every neuron is
pointed at it with `FUGAL_BENCHMARK_POOL`, the loader is a consensus hazard.

**The approved measurement is reproducible but not durable (open, blocks
`--live`).** `measurement_id` is sha256 over MRTD and RTMR0-2. Measured on real
TDX hardware rather than reasoned about: two independently created
`c3-standard-4` instances, both pinned to `ubuntu-2404-noble-amd64-v20260903`,
created 15 minutes apart with the first deleted in between, produced a
**bit-identical** measurement — every register matching. Under the current
formula that image measures
`a1ecb6273d38bd8b5ad629edf12f48d1c3b861fa8f506d1e1e303783e68461f3`.

So the value is reproducible today, provided two variables are pinned:

- **the exact image version.** `--image-family` resolves to whatever is newest
  at create time, so two runs a week apart silently measure differently.
- **the machine shape.** RTMR0 moves with it; a 4-vCPU and an 8-vCPU instance
  measure differently from identical software.

What remains open is durability, not reproducibility. A cloud provider
eventually retires a pinned image version, so an approved list built this way
has an expiry date, and rotating it means changing a consensus parameter on a
schedule set by someone else. There is no rotation procedure, and that is what
blocks `--live` on mainnet. A reproducibly-built guest image (dstack, or one
built in-house) removes the dependency entirely and remains the right long-term
answer — but it is not required to obtain a working measurement now.

**Code benchmarks are excluded (open, tracked).** The harness grades with
`allow_exec=False`, and `exec_io`/`exec_unittest` — the checkers for humaneval
and livecode — return 0 unless permitted. So those questions scored zero for
every miner regardless of the answer, and because the slicer equalises
benchmarks that was a full sixth of every graded slice, not humaneval's 0.8%
share of the pool. Worse than dead weight: miners paid API cost for an
impossible score, so routing code to a capable model was punished on thrift with
no reachable quality, training heads toward the cheapest model on exactly the
domain where routing matters most.

`allow_exec=False` is correct as it stands. The sandbox in `graders.py` is
process-level with no filesystem, namespace or network isolation, and under
`--live` the miner controls what the metering proxy returns — so executing it
inside the TD would let a miner run chosen code in its own enclave and obtain a
genuine attestation over forged results. The pool now excludes what the harness
cannot score, derived from `config.HARNESS_ALLOW_EXEC` so that enabling
execution re-includes the benchmarks in one place.
`tests/test_grader_policy.py` fails if the policy and the pool drift apart
again, in either direction. Putting them back is
[CODE_BENCHMARK_PLAN.md](CODE_BENCHMARK_PLAN.md), and its final phase needs the
measurement rotation procedure that does not exist yet.

**The model upstream is miner-controlled (OPEN, EXPLOITABLE TODAY, breaks I3).**
`fugal_subnet/tee/runtime.py` reads the endpoint the metering proxy calls from
`FUGAL_OPENROUTER_BASE`, an environment variable, inside the miner's own TD. The
pool the TD loads carries `gold` for every question. Nothing in `proof.py` or
`verify.py` binds the upstream — grep them for it and there is nothing.

So a miner points that variable at a server they run, which answers every
question with its own gold answer and reports two tokens of usage. The result is
perfect accuracy at near-zero attested cost, and every other check still passes:
questions_hash, nonce, weights_hash, content_hash, report_data, DCAP, and the
approved measurement. The proof is *honest about a dishonest computation*, and
the miner takes the subnet.

**This survives a perfect TDX deployment.** A fully measured image with the
upstream delivered as a runtime environment variable is still broken, which
makes it a requirement on the measured image and not merely a bug behind it:

- the upstream must be inside what is measured (for dstack, `app-compose.json`,
  which is hashed into RTMR3 — *not* dstack's encrypted env, which is delivered
  after attestation and is therefore unmeasured by design)
- container images must be referenced by digest, never by tag, or the hash is
  stable while the code under it is not
- TLS validation must remain intact, or a miner who controls the VM's network
  reaches the same outcome by pointing DNS at themselves

Nothing in the repository prevents this today. The only thing resembling a
control is a line in `docs/LIVE_API_VALIDATION.md` asking the operator not to
set the variable — a request addressed to the adversary.

**Why it went unnoticed for so long is the instructive part.**
`scripts/stub_upstream.py` *is* this attack. It was written as a no-spend
testing tool, it works exactly as intended, and nobody asked what it meant that
it worked. A mechanism that lets a local test substitute the model layer is the
same mechanism that lets a miner substitute it in production. Building the
exploit as a convenience and never naming it as one is how a hole this size
stays invisible.

### I3 — the scored cost is the policy's, not the miner's

`thrift = reference_cost / miner_cost`. The denominator was already computed
from shared data — pinned rates plus the reference frame's measured per-model
verbosity — and the numerator was whatever the miner reported it had spent. Even
the denominator's prompt tokens came from the miner's attested counts. One side
of the ratio was unfakeable and the other was a self-report, so a miner could
raise its score by shrinking numbers rather than by routing better. On unmeasured
code with a miner-settable upstream, those numbers are entirely theirs.

Both sides are now computed by the validator:

    cost(question, model) = tokens_in(question) x rate_in(model)
                          + typical_completion(model) x rate_out(model)

Every input is already consensus state: the question pool, the pinned price
table, and the frame's pooled, decayed per-model observations.

**What is being priced is a policy, not a benchmark run.** The product's
beneficiary is a future user of the head, who cares what the policy costs
*them* — not what one miner happened to spend on one run in one hour. So the
scored cost depends only on which model was chosen for each question, which is
the thing being measured, and on nothing a miner can vary.

An estimate, not an invoice, and not trying to be one: the pinned table already
diverges from real billing by design so that two miners benchmarking hours apart
face the same denominator. Input tokens use a fixed characters-per-token
convention rather than a real tokenisation, because every model tokenises
differently, a validator should not gain a tokeniser dependency to price a
proof, and the figure is a relative signal where consistency beats precision.

The miner's reported costs stay in the proof and are still checked for internal
consistency — a miner lying about them is worth knowing — but they decide no
score. That check is now defence in depth rather than load-bearing.

**Price table accuracy still matters, for a different reason.** It no longer
guards against cheating; it defines what "cheap" means, so an error steers which
models the subnet learns to prefer. An under-priced model looks cheaper than it
is and gets over-routed. That is why rates are corrected on their own merits —
see the glm-5.2 correction — and why the five models whose real billing cannot
be modelled by any per-token pair are a product question rather than a scoring
one.

**Price table staleness.** `data/models.json` is hash-pinned, so scoring is
deterministic, but it does not track provider price changes on its own. The
metering proxy records the provider's reported cost alongside the table price
so drift is *detectable*; acting on it is a manual, deliberate update.

## The dress rehearsal

Everything above runs in-process against a mocked chain. `scripts/dress_rehearsal.py`
runs the shipped binaries — `neurons/miner.py` and `neurons/validator.py` as
real OS processes — against a real local subtensor node, with real wallets,
real registration, real axon/dendrite traffic and real `set_weights`.

That distinction is not cosmetic. The TEE pipeline shipped with three fatal
bugs behind a green CI because CI exercised a different path than production
did, and the first real-chain run surfaced eight more that no in-process test
could see: the neurons' own logging silently disabled by importing bittensor,
`--once` never exiting, a crash in the reveal block, weights reported as set
while the chain held none, a miner rendering itself unreachable by ordering two
extrinsics wrongly, the pool re-embedded every epoch, the two neurons loading
different question pools, and epoch geometry duplicated across both.

| | Asserts |
|---|---|
| **A** | A proof verifies against a real chain; weights land **and are confirmed** |
| **B** | Dedup disqualifies a copy and not the original; a real router outranks an always-cheapest one |
| **C** | Two independent validators produce byte-identical weights and frames (I1, I9) |
| **D** | Evidence accumulates, the frame fills, weight capping engages, weights confirm every epoch |
| **E** | The real backbone path works end to end and a full bundle round-trips over a real axon |

```bash
python scripts/dress_rehearsal.py --scenario all
```

## Running the checks

```bash
python scripts/check_safety_invariants.py          # structural invariants + TEE safety
python scripts/check_determinism.py                # I1, same-host
python scripts/check_determinism.py --perturb      # I1, simulated second host
python -m fugal_subnet.attacks.run_attacks         # I3, hostile model output
python -m fugal_subnet.attacks.run_miner_attacks   # I2/I6, hostile miner input
python -m fugal_subnet.attacks.run_tee_attacks     # I8/I3, forged proofs
pytest -q                                          # I4, I9, evidence, TEE, end-to-end
```

All of these run in CI on every push. None requires a chain, a network, or API
spend.

## Before mainnet

See [MAINNET_LAUNCH.md](MAINNET_LAUNCH.md) for the operational sequence. Item 1
below was done on netuid 552 and turned up a defect no local run could: the
subnet had never been activated. `start_call` was never made, so it had no
first-emission block — staking was rejected, no validator earned a permit, and
every `set_weights` failed while the neurons reported success.

1. Deploy on testnet with two validators and verify they produce identical
   weights from the same set of TEE proofs, and identical reference frames from
   the same published exploration samples (I9).
2. Run miners on real Intel TDX VMs (GCP `c3-standard` or Azure confidential
   VMs) and verify DCAP attestation end-to-end.
3. Publish approved runtime measurements (`FUGAL_TEE_MEASUREMENTS`) and
   document the process for updating them. These are `measurement_id()` values
   — sha256 over the quote's MRTD and RTMR0-2 — not source hashes.
4. Recalibrate the reference-frame prior from real testnet data. It currently
   sits at a deliberately neutral 0.5 for every model, which is honest (the
   subnet has measured nothing yet) but is wrong for real models and biases the
   ceiling low until real evidence outweighs it.
5. Close or consciously accept the pool-memorization gap above.
6. Validate real TDX attestation on a confidential VM —
   `docs/TDX_VALIDATION.md`. DCAP signature verification and `measurement_id`
   matching are the only checks no local run can make; a consumer CPU cannot
   produce a genuine quote. Mock mode is their absence, not a weaker form.
7. Validate the cost path against real OpenRouter —
   `docs/LIVE_API_VALIDATION.md`. The pinned price table is deterministic, not
   necessarily correct; only a live comparison distinguishes the two.
