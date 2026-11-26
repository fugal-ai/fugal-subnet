# scripts/

Every script here has one job. They are grouped by who runs them. The files
stay in one flat folder until the v1 removal after the flip, because CI, the
systemd units, the guides and signed release notes name these paths
(`docs/MAINNET_LAUNCH.md`, "After the flip").

Each script's docstring is its manual: `python scripts/<name>.py --help`.

## CI checks (run on every push; `gate.sh` runs them locally)

| Script | Job |
|---|---|
| `gate.sh` | Runs CI's own steps locally, in CI's environments. The command to run before pushing |
| `check_safety_invariants.py` | Pins (grader, price tables, frame, TPM root, list signers and signed lists), no-pickle, env classification, documented flags; `--release` adds the launch gate |
| `check_determinism.py` | I1: two processes score the same inputs to the same bytes (`--perturb` imitates another host) |
| `check_bittensor_axon.py` | The synapse type survives Axon registration (CLAUDE.md rule 1) |
| `check_success_vendor.py` | The vendored success contract is byte-for-byte fugal-core's pinned commit |
| `check_success_fixtures.py` | The vendored contract reproduces core's numeric fixtures |
| `check_success_embeddings.py` | Core and subnet agree on tokens, embeddings and routes with the real backbone (CI only: 1.2 GB download) |

## Rehearsal (no spend)

| Script | Job |
|---|---|
| `dress_rehearsal.py` | The shipped miner and validator against a real local chain (Docker): scenarios a–e and `v2_flip`. Nightly CI |
| `rehearsal_v2.py` | The v2 flip in process, in seconds; also the v2 probes `dress_rehearsal` uses |
| `setup_local_testnet.py` | Local-chain helpers for `dress_rehearsal` (subnet, funds, stake, stub upstream). Not run directly |
| `stub_upstream.py` | A deterministic fake OpenRouter for `--mock` miners, v1 and v2 models |

## Builders of pinned files (the only sanctioned way to change them)

| Script | Builds |
|---|---|
| `build_pool_manifest.py` | `pool_manifest.json` / `pool_manifest_v2.json` (`--check` verifies) |
| `signed_list.py` | Protocol v4's signed lists: `draft` derives the first lab and host allowlist from the pinned v3 menu, `prepare` writes the bytes to sign, `verify` checks a list and its signature as a neuron would. Signing is the key holder's `ssh-keygen` command (`deploy/keys/README.md`) |
| `export_pool.py` | The offline pool file validators load (`data/benchmarks/pool*.json.gz`) |
| `record_token_observations.py` | Token-count observations [PAID with `--live`; `--mock` by default] |
| `build_token_manifest.py` | `benchmark_tokens_*.json` from those observations |
| `build_frame_from_calls.py` | `data/reference_frame.json` (v1 only) from a committed call log; `--build-only` reproduces the pin |
| `build_bootstrap_frame.py` | A candidate v1 frame from published reveals |
| `vendor_success_contract.py` | `fugal_subnet/vendor/` from a reviewed fugal-core commit |
| `build_menu_v4_seed.py` | `data/menu_v4_seed.json`, the protocol v4 menu seed, from a saved public listing and integer sums of named published reveals; files only, no network; writes the data/ copy and the package copy (`build --check` verifies both) |

## Operations (live chains and TEEs)

| Script | Job |
|---|---|
| `provision_subnet.py` | Launch and configure a subnet; a dry run without `--yes`; `--status` reads only |
| `compute_app_identity.py` | The compose-hash half of an approved entry, from the repo |
| `release.py` | One command for a release: `plan`, `status`, `check-block`, `blocks` (compiled), `schedule draft`/`verify` (the signed protocol schedule), `build-tag`, `freeze` (dstack-cloud prepare), `entries`, `verify-tag` (the auto-updater's strict verifier), `validator-tag`, `notice`. Dry run by default; never signs, pushes or uploads: it prints those commands for the owner |
| `publish_approved_entry.py` | The signed tag message and release note carrying both networks' entries |
| `fetch_approved_entries.py` | Validator side: verify the signed tag, print one network's entry |
| `provision_td.py` | `--inspect` a TD; push the head and keys to it over the attested channel |
| `verify_live_miner.py` | Query one miner and verify its proof as a validator does |
| `watch_subnet.py` | Per-epoch pass/fail on a live subnet (hourly via `deploy/systemd/fugal-watch.*`) |
| `tdx_measurement.py` | On a plain TDX VM: the two checks only real hardware can make (`--verify`) |

## Miner

| Script | Job |
|---|---|
| `train_head.py` | Train a head from a labelled matrix (`--synthetic`: test-only) |
| `export_success_head.py` | Bundle a head for fugal-core and the conformance fixture; not a mining step |

## experiments/

Companions to the dated records in `docs/decisions/`: the measurements behind
past decisions. Not maintained against current code; read them with their
record.

`experiments/experiment_fast_crown_v4.py` is current: it measures its inputs
from saved v3 reveals and simulates `docs/decisions/FAST_CROWN_V4.md`
(`python scripts/experiments/experiment_fast_crown_v4.py all`, or one of
`a b c e time`; `--quick` for a smoke run).
