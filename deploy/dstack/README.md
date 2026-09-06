# Deploying a Fugal miner under dstack on GCP

The operational recipe behind MINER_GUIDE § "Running Under dstack". Everything
here was found by deploying, not read; the landmines are listed there and are
not repeated.

## Toolchain (once per operator machine)

```bash
# The CLI is scripts/bin/dstack-cloud in meta-dstack-cloud (a 2,400-line Python
# script). Phala-Network/dstack-cloud is a fork of the Rust workspace and does
# NOT contain it — measured 2026-09-06, on every branch and in history.
git clone --depth 1 https://github.com/Phala-Network/meta-dstack-cloud
# `dstack-cloud pull` is broken for the 0.6.0 prerelease line; fetch the UKI
# image by hand and verify it. The tool then expects
# <image_search_path>/<os_image>/disk.raw:
curl -LO https://github.com/Dstack-TEE/dstack/releases/download/mkosi-os-v0.6.0-rc0/dstack-0.6.0-rc0-uki.tar.gz
echo "54e57c4b345242e28849321809dc59faa80628816f24b740b5c9dedb842bcba7  dstack-0.6.0-rc0-uki.tar.gz" | sha256sum -c
tar xzf dstack-0.6.0-rc0-uki.tar.gz
# ~/.config/dstack-cloud/config.json: set image_search_paths to the extracted dir
```

The base measurement this image produces on a GCP `c3` (any size) is

    12a1f2f56907f80576be553f3d71031ec86f2848877ac05c82db5d8627fc9141

measured on three real deploys (`fugal_subnet/tee/attestation.measurement_id`,
sha256 over MRTD‖RTMR1‖RTMR2).

## Application settings (they are part of the approved identity)

| Field | Value | Why |
|---|---|---|
| `key_provider` | `tpm` | The only value that boots on a GCP CVM; `kms` boot-loops, `local` needs a VMM |
| `public_logs` | `false` | The miner holds an API key; one traceback with a header in it and the key is public |
| `public_sysinfo` | `false` | A sealed box has no reason to describe itself |
| `gateway_enabled` | `false` | Validators reach the axon directly |
| `public_tcbinfo` | `false` | Nothing reads the agent's public TCB endpoint: validators verify the quote inside the proof, the pusher gets its quote over `/provision/attest`. Surface with no consumer |
| `no_instance_id` | `false` | The `instance-id` event is what lets the pusher refuse a correctly-imaged TD somebody else booted |
| `.env` | absent | Refused without KMS anyway; the key travels over the attested channel |

`docker-compose.yaml` in this directory is the compose. Pin the image by digest.

## Identity → approved entry

```bash
dstack-cloud new fugal-miner && cd fugal-miner        # copy docker-compose.yaml in, set fields above
dstack-cloud prepare                                   # writes shared/app-compose.json
python scripts/compute_app_identity.py --compose shared/app-compose.json \
    --base 12a1f2f56907f80576be553f3d71031ec86f2848877ac05c82db5d8627fc9141
# -> FUGAL_TEE_MEASUREMENTS=<base>:<compose_hash>  on EVERY validator
```

The compose hash is sha256 of the file's raw bytes. Do not re-serialise it.

Three facts about that file, read out of the tool's source and worth knowing
before you hash anything:

- `dstack-cloud new` writes `app.json`, a **template `docker-compose.yaml`
  (nginx)**, `prelaunch.sh` and `.user-config`. Replace the compose with this
  directory's; leave `prelaunch.sh` alone — its full text is embedded in
  `app-compose.json` and so is part of the hash.
- `prepare` writes `shared/app-compose.json` with `json.dump(..., indent=2)`,
  embedding the full compose and prelaunch text as strings.
- **`deploy` regenerates `shared/app-compose.json`** from `app.json`,
  `docker-compose.yaml` and `prelaunch.sh`. Hash the bytes from the same state
  you deploy, and re-hash after any edit, or the approved entry names a compose
  the TD never extended.
- `app.json` must keep `no_instance_id: false`. The `instance-id` event is what
  lets `provision_td.py` refuse a correctly-imaged TD somebody else booted; a
  tool version that flips it for non-KMS providers has to be corrected by hand
  before `deploy`, and `--inspect` will show whether the event is present.

## Deploy, then provision

```bash
dstack-cloud deploy            # c3-standard-4 is enough; the machine shape is not in the measurement
dstack-cloud status            # public IP
gcloud compute instances add-tags <vm> --tags fugal-miner   # firewall: 8091 world, 8092 operator-only

python scripts/provision_td.py --inspect --address http://<ip>:8092      # read instance-id, compose hash
python scripts/provision_td.py --address http://<ip>:8092 \
    --approved <base>:<compose_hash> --instance-id <from --inspect> \
    --head data/rehearsal/head_cheap_a.npz --wallet fugal_owner2 --hotkey td1 \
    --api-key-file ~/.fugal/openrouter.key
```

`deploy` refuses when the instance exists; `deploy --delete` recreates it. The
data disk is created auto-delete, so a recreate discards the LUKS disk, the
TPM-sealed seed (new instance id) and the embedding cache; `stop`/`start` keep
all three. There is no update-in-place path in this tool, so an image upgrade
means re-provisioning and re-paying the backbone pass.

Until the push, the miner serves nothing and logs "waiting to be provisioned".
After it, the miner embeds the pinned pool — about 13 hours at one thread on a
`c3-standard-4`, once, cached on the encrypted data volume — then serves its
axon, commits its head hash and starts producing proofs at the next epoch
boundary.

Verify from outside, as a validator would:

```bash
nc -vz <ip> 8091
python scripts/verify_live_miner.py --netuid 552 --uid <uid> --coldkey <any registered> \
    --measurements <base>:<compose_hash>
python scripts/verify_live_miner.py ... --measurements <base>:<wrong_hash> --expect-reject
```
