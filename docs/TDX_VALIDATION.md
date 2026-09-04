# Validating Real TDX Attestation

Everything in this repo can be exercised locally except two checks, and they
are the two that matter most for I8. This document is how to close them.

## What is not covered locally, and why

`scripts/dress_rehearsal.py` runs the real miner and validator against a real
chain and enforces the entire binding chain — report_data, assigned slice,
committed head, advertised bundle, exploration assignment, cost consistency.
Two things it cannot enforce:

| Check | Why not locally |
|---|---|
| **DCAP signature chain** | Needs a genuine Intel-signed quote, which needs real TDX silicon |
| **`measurement_id` match** | Needs real MRTD/RTMR registers, which only a TDX CPU fills in |

A consumer CPU cannot produce either. Intel TDX requires 4th-generation Xeon
Scalable (Sapphire Rapids) or newer; on anything else there is no
`/dev/tdx_guest` and no quote generator, so `--live` has nothing to verify.

Mock mode is not a weaker version of these checks — it is their absence. It
skips exactly DCAP and measurement matching and enforces everything else, which
is why a mock run is still worth something and why it is not sufficient.

## Provisioning a confidential VM

Either works; pick whichever account you already have.

**GCP** — a `c3-standard-4` with confidential computing enabled:

```bash
gcloud compute instances create fugal-tdx \
  --zone=us-central1-a \
  --machine-type=c3-standard-4 \
  --confidential-compute-type=TDX \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud
```

**Azure** — a DCesv5-series confidential VM:

```bash
az vm create --name fugal-tdx --resource-group <rg> \
  --size Standard_DC4es_v5 \
  --image Canonical:ubuntu-24_04-lts:cvm:latest \
  --security-type ConfidentialVM \
  --enable-vtpm true --enable-secure-boot true
```

Confirm the guest really is a TD before going further:

```bash
ls -l /dev/tdx_guest        # must exist
```

If that device is missing, confidential compute is not actually enabled and
nothing below will work.

## Setup

```bash
git clone <repo> && cd fugal-subnet
uv sync --extra tee          # installs dcap-qvl, pinned
```

`dcap-qvl` is a declared dependency of the `tee` extra, not a manual install.
A validator running `--live` without it cannot verify any attestation.

## Getting the measurement of your image

`FUGAL_TEE_MEASUREMENTS` holds `measurement_id()` values — `sha256(MRTD ||
RTMR0 || RTMR1 || RTMR2)`. It is **not** a source hash, and deliberately not
anything the workload writes about itself: the whole point is that an attacker
running modified code inside a genuine TD produces a valid quote, and only the
measurement registers distinguish them.

RTMR3 is excluded because it is application-extendable; including it would make
the identity move with runtime data and no image could stay on an approved list.

Run this on the TD to print the measurement of the image you are about to
approve:

```bash
python scripts/tdx_measurement.py
```

Add the value it prints to the validator's environment:

```bash
export FUGAL_TEE_MEASUREMENTS=<measurement_id>
```

Publish it for other validators the same way any consensus parameter is
published — a change to this list is a change to what code the subnet accepts.

## The two checks

```bash
python scripts/tdx_measurement.py --verify
```

This asserts what cannot be asserted anywhere else:

1. A genuine quote from this machine passes `verify_dcap()` against Intel's
   collateral.
2. `verify_proof(..., mock=False)` **rejects** a proof whose measurement is not
   on the approved list — the "modified harness in a real TDX VM" attack from
   `run_tee_attacks.py`, which is blocked there against a synthetic quote and
   should be blocked here against a real one.

Both must pass before mainnet. The second is the one that matters: passing DCAP
only proves the hardware is real.

## Running live

```bash
# Miner, inside the TD
python neurons/miner.py --netuid <N> --head-path head.npz --live

# Validator (does not need TDX — it verifies, it does not attest)
FUGAL_TEE_MEASUREMENTS=<id> python neurons/validator.py --netuid <N> --live
```

Validators do not need confidential hardware. Only miners do.

## What to record

Capture, for the record:

- the `measurement_id` and which image produced it
- the output of `--verify`, both checks
- one full `reveal.json` from a live epoch

That set is what lets a third party confirm the subnet was verifying
attestations rather than accepting assertions.
