# Production dress rehearsal — netuid 552, 2026-09-06

The first run of the shipped binaries in `--live` mode against real Intel TDX
hardware, on the public testnet, with two validators on different CPU
architectures and miners paying real money. Every line below is labelled
**measured** (observed on a running system) or **read** (taken from source or
documentation). The distinction has a track record in this project; see
OPEN_WORK.md § "A method note".

Team: three sessions — code/coordination (A), GCP infrastructure (B), ARM box
and observation (C). All infrastructure actions were authorised per step.

## Starting state (measured, before any change)

| | |
|---|---|
| `main` | `7a4baea`, CI gate green locally (`scripts/gate.sh`, both environments) |
| netuid 552 | active, owner `fugal_owner2`, tempo 360, commit-reveal on (period 1), `weights_rate_limit` 100, 7 neurons |
| Validators | uid 0 (`fugal_owner2/default`, permit) and uid 1 (`fugal_validator2/default`, permit) — both on the ARM box, `--mock`, commit `340c07e`, `FUGAL_EPOCH_INTERVAL=1800` (**misaligned** with tempo 360), 150-question `pool_tiny.json` |
| Miners | uids 2–4 mock miners on the ARM box advertising **10.0.0.202** (private); ports 8091–8093 closed from the internet; uids 5–6 stale TD hotkeys from the stock-VM run of 2026-09-05 |
| Agreement | both ARM validators byte-identical on weights and consensus digest for 40+ consecutive epochs (same host, same arch) |
| GCP `fugal-tdx-test` | no instances, no custom images, no dstack buckets, no firewall beyond defaults; AR `fugal/miner:rehearsal` public, built from a pre-PR#10 snapshot |

## Defects found before the run started

1. **A dstack TD had no permitted way to receive its hotkey** (design gap; see
   INVARIANTS § "I8 — the wallet crosses the attested channel"). Closed by
   adding the keyfile and coldkeypub to the attested channel.
2. **`FUGAL_BENCHMARK_POOL` bypassed the manifest check and the size floor**
   (the documented production path was the unguarded one). Closed; non-pinned
   pools must declare `FUGAL_POOL_UNPINNED=1`.
3. **The SDK's `serve_axon` needs `coldkeypub`, not just the hotkey**
   (measured: hotkey-only `bt.Wallet` signs but raises on `coldkeypub`).
4. **Synthetic-trained heads collapse to one model on real embeddings**
   (measured: 99.5% and 100% onto `gpt-5.4-nano`), so two of them would be
   deduplicated as copies. Rehearsal heads are now built against real
   embeddings (`scripts/make_rehearsal_heads.py`): shares 0.334/0.333/0.333,
   agreement between the two heads 0.312 (chance 0.333).

## Infrastructure (measured)

| | |
|---|---|
| dstack CLI | `scripts/bin/dstack-cloud` in `Phala-Network/meta-dstack-cloud` (not `dstack-cloud`, which is a Rust-workspace fork) |
| Firewall | `fugal-axon` tcp:8091 from anywhere, `fugal-provision` tcp:8092 from the two operator IPs only; both on tag `fugal-miner` |
| Validator #2 | `fugal-val2`, e2-medium, x86_64, `main` 7a4baea, `uv sync --locked --extra tee` 44 s; `load_all()` 21553 questions, `pool_hash 1779b4a5…b4df`, "Pool matches the pinned manifest", first load 23 s incl. HF download |

## Approved identity (measured)

| | |
|---|---|
| Miner image | `us-central1-docker.pkg.dev/fugal-tdx-test/fugal/miner@sha256:3430da97f5f9182f0d2e8f90f64da8c6ab859a995320591745d6c69d36a8541f`, Cloud Build from a clean checkout of `e284963`, 12 m 56 s |
| Compose | `deploy/dstack/app-compose.frozen-2026-09-06.json`, 3182 bytes, byte-identical for TD1 and TD2 (TD2's `app.json` `name` had to be set to `fugal-miner`; the tool copies it into the compose) |
| Compose hash | `cf31a1a469c7f3a190deb00b4e010a1a619c933d4e28edcc45bb48753dce69d2`, recomputed by the coordinator from the raw bytes with `compute_app_identity.py --compose` |
| Base measurement | `12a1f2f56907f80576be553f3d71031ec86f2848877ac05c82db5d8627fc9141` |
| `FUGAL_TEE_MEASUREMENTS` | `12a1f2f5…9141:cf31a1a4…69d2`, on both validators |
| Instance ids | **Not predictable from `app.json`.** The 0.6.0-rc0 guest logs "mixing platform per-instance binding into instance_id" and derives an id the plain `sha256(seed‖app_id)[:20]` formula (read from the on-prem source) does not produce: TD1 predicted `748279f6…`, guest reports `4079f70663edf49c38313b7062b4823527278c13`. Take the id from the console or from `provision_td.py --inspect`, never from a formula. |

## Cost (read from the Cloud Billing catalog, us-central1 list price)

| Item | Rate |
|---|---|
| c3-standard-4 (4 vCPU $0.03465/h each + 16 GiB $0.003938/h each) | $0.2016/h |
| 61 GiB pd-balanced per TD (10 boot + 50 data + 1 shared) | ≈ $0.0084/h |
| **Per TD** | **≈ $0.21/h**; two TDs for 16 h ≈ $6.72 |
| e2-medium validator + 30 GiB | ≈ $0.0375/h |
| TDX on C3 | no separate SKU — not billed extra (the confidential surcharge SKUs are N2D/SEV) |
| 6.4 GB image pull from Artifact Registry | intra-region, free |

## Rehearsal heads

| | sha256 | routes among |
|---|---|---|
| `head_cheap_a.npz` | `b0ce2fcbb26167201c2fc868cd618f1e4c00745bb768644bfbc60c863c979bcb` | deepseek-v4-flash / gpt-5.4-nano / llama-4-maverick, ~⅓ each |
| `head_cheap_b.npz` | `bb18e52302ce74a31e050fc820afd027e8076a2535faa6132d9bbebc9e7b41f3` | same three, different assignment (31% agreement with a) |

Expected spend per 300-question epoch per miner, from the pinned table:
≈ $0.014 routing + ≈ $0.027 exploration.

## Finding: the attested channel is authenticated, not confidential

Found while designing the operator-only log path, by reading `push()` and
`serve()` side by side. The pusher verifies an Intel-signed quote over its
nonce, then POSTs the payload — head, **hotkey keyfile, OpenRouter key** — to
`http://<td>:8092/provision` in **plaintext**. The quote authenticates the TD;
nothing encrypts the secrets to it. Anyone on the path between the operator
and the cloud edge reads both. The firewall narrows who can *connect*, not who
can *observe*. Tonight's pushes (19:33, 19:35) crossed this way; both secrets
are testnet-only and the key is to be revoked at the end of the run.

Fix, in this branch: the TD generates an ephemeral X25519 key at start and
binds `sha256(pubkey)` into the second 32 bytes of `report_data` alongside the
operator's nonce, so the quote proves the TD holds the key; the pusher derives
a ChaCha20-Poly1305 key from X25519(operator ephemeral, TD pubkey) via
HKDF-SHA256 salted with the nonce and sends only ciphertext; plaintext pushes
are refused. The same shared secret authorises an operator-only, encrypted log
pull, closing the black-box finding above without `public_logs`.

## Run log

All times UTC.

- **17:38** mock-era validators on ARM confirmed weights for `e00052985`; C waits
  for the next confirmation before stopping them (never mid-`set_weights`).
- **~17:50** C's session was denied the process kills and the `sudo` install by
  its permission classifier; surfaced to the user, who added an allow rule.
  Units installed on ARM (`fugal-validator`, `fugal-stub`, `fugal-mock-miner`),
  `systemd-analyze verify` clean, all disabled and inactive, env file 0600 with
  an empty `FUGAL_TEE_MEASUREMENTS`.
- **17:56** PR #13 CI: Python 3.10/3.11 and lint failed on one line — a
  backslash inside an f-string, legal only from 3.12. Local ruff (same locked
  version) had passed; the CI matrix caught it. Fixed, all 8 checks green.
- **~18:10** PR #13 merged; `main` = `e284963`. B signalled to build the miner
  image from a clean checkout of that commit.
- **18:08–18:09** mock-era validators confirmed `e00052986`; all five neurons
  and the stub stopped inside the idle window, zero survivors; logs and state
  archived as `*-mock-era` (44 and 43 epoch dirs). ARM repo fast-forwarded to
  `e284963`, clean.
- **~18:00** GCP image `dstack-0-6-0-rc0` READY (tarball sha256 verified;
  upload 365 s, image create 83 s). `fugal-val2` unit installed, disabled.
  Provisional compose hash computed; final one waits on the image digest.
  Source reading corrected: the CLI's `--no-instance-id` defaults to false on
  every key provider, so the instance-id event is present and the pusher's
  instance check is intact.
- **~18:20** ARM: `uv sync --locked --extra tee` clean, invariants pass,
  `dcap_qvl` imports. `load_all()` → 21553 questions,
  `pool_hash 1779b4a5218bb2effed85729f345f2766b010e2c33a36e6d3afe7c1a05a4b4df`,
  manifest match — **byte-identical to the x86 validator's**, so the pool agrees
  across aarch64 and x86_64 before any epoch runs. Two observations: the
  loader still makes HuggingFace metadata calls with a warm cache (4.9 s), so a
  validator needs egress at startup; and the 164 HumanEval exec questions are
  excluded on both hosts as the manifest expects. Stub and mock miner (uid 4)
  started under systemd; validators still stopped pending the approved entry.
- **18:30** ARM validator (uid 0) started `--live` with the approved entry:
  "Operation mode: LIVE", consensus environment digest `1d51078642731d50`,
  pool 21553 / `1779b4a5…`, manifest match, no permit warning, fresh frame
  from the bootstrap prior (warned, as designed).
- **19:05 — first live epoch ever observed, `e00022078`** (boundary 7948080,
  300-question slice, collection at boundary+180 after a 33-minute wait):
  queried 7 miners, one proof returned (the mock control, uid 4), rejected with
  `DCAP: quote does not parse (… Not enough data to fill buffer)` →
  "DCAP attestation verification failed"; "No valid proofs received, skipping
  epoch". Epoch log: `n_heads_valid=0, n_heads_invalid=1, n_heads_unverifiable=0`,
  anomalies `["no_valid_proofs"]`, 2096 s. Note the mock proof fails at quote
  parsing, before the approved-entry check is ever reached — the negative
  control exercises DCAP, not the measurement match. Validator persisted state
  and is waiting for boundary 7948440 (~19:42).
- **19:22** x86 validator `fugal-val2` (uid 1) started `--live`. **Same
  consensus environment digest `1d51078642731d50` as the aarch64 validator**,
  same slice hash `cdeaeb31b3ab285c…` and same commit hash `99625e33…3419`
  for `e00022078` — the first cross-architecture agreement on a real epoch's
  inputs, before any proof exists. Startup logs do not state the permit;
  asserted from the metagraph instead (uid 1, permit True, stake 195).
- **19:26–19:29** B's session could not run `dstack-cloud deploy` (hard
  denial, no prompt); the user explicitly authorised the coordinator session to
  run it. `fugal-td1` deployed in 2 m 14 s: shared-disk image and the one-off
  `dstack-data-disk` image created, TDX instance RUNNING at 35.192.213.23,
  tags `fugal-miner,fw-fugal-td1`. `deploy` regenerated `shared/app-compose.json`
  as B predicted; its hash is unchanged (`cf31a1a4…`). Ports 8091/8092 closed
  at +1 min (guest still booting / pulling the image).
- **19:29–19:30** TD1 console: kernel 6.18.40-dstack, "tdx: Guest detected",
  TDX memory encryption active; `key_provider` tpm, no sealed seed → new seed
  sealed, app keys from the TPM; LUKS2 data disk formatted (50 GB); app info
  printed with `compose_hash cf31a1a4…69d2` (matches the approved entry) and
  `app_id 118e1b18…`; containers started, 13 image layers pulling from the
  public registry with no auth errors; no restart loop. One unit failed:
  `systemd-tpm2-setup.service` (TPM SRK setup) — dstack-prepare proceeded and
  generated app keys anyway; flagged, not diagnosed. Ports went from
  "timed out" to "connection refused" once the guest network was up, so the
  firewall path is proven open before the service listens.
- **19:31** `fugal-td2` deployed at 35.193.250.187 in ~2 min (shared images
  already existed); compose hash unchanged. Console: TDX guest, tpm key
  provider, `compose_hash cf31a1a4…` match, guest instance id
  `fdb7411e275676e633dd2811e79fd035ed75b786`; same failed
  `systemd-tpm2-setup.service`; one cosmetic sysbox cleanup warning. TD1's
  6.37 GB venv layer downloaded by 77 s of uptime; extraction follows.
- **19:33** TD1 provisioning port answered ~4.5 min after create.
  `provision_td.py --inspect`: 38,248-byte attestation, 8,000-byte quote, our
  nonce honoured, base `12a1f2f5…`, event log replays to RTMR3, nine RTMR3
  events (`app-id`, `compose-hash cf31a1a4…`, `gpu-policy-hash`,
  `instance-id 4079f706…`, `key-provider {"name":"tpm","id":""}`,
  `storage-fs ext4`, …). **First push over the attested channel to a TD that
  had no wallet:** head_cheap_a, hotkey td1, coldkeypub and the API key;
  `/provision/status` → `provisioned: true`.
- **Finding (operability):** with `public_logs=false` the container's stdout
  never reaches the serial console — there is no "Waiting to be provisioned"
  line anywhere an operator can read, and there will be no embedding progress
  or traceback either. Observables that remain: the open 8092 socket and
  `/provision/status`, VM CPU utilisation (one pinned vCPU ≈ embedding
  running), port 8091 opening when the axon serves, and on-chain axon/commit
  state. A production miner needs a deliberate log path that does not publish
  the key; open item.
- **20:10 — Route A measured, not argued.** An nginx caching proxy in front of
  `pccs.phala.network` on `fugal-val2` (`127.0.0.1:8081`, 1 h TTL, stale-on-
  error), then the validator's own `verify_dcap` on the real fixture quote:

  | path | verify time | status |
  |---|---|---|
  | proxy, first call (3 of 4 collateral URLs cold) | 69 ms | UpToDate |
  | proxy, warm | **15 ms** | UpToDate |
  | direct to Phala | 707–750 ms | UpToDate |

  A cache hit is ~50x cheaper than the round trip, and one warm cache serves
  every proof from the same platform (FMSPC `00806F050000`). The four URLs
  dcap-qvl fetches per proof: `rootcacrl`, `tdx/tcb?fmspc=…`, `tdx/qe/identity`,
  `pckcrl?ca=platform`. Adds no trust root — the bytes are Intel-signed and
  verified against the compiled-in root. `fugal-val2` now runs with
  `FUGAL_PCCS_URL=http://127.0.0.1:8081`; the ARM validator stays on Phala so
  tonight's proofs compare the two paths directly.
- **20:08 — restart drill.** `systemctl restart fugal-validator` on the x86
  validator mid-epoch (while waiting for the collection block). It came back in
  16 s, re-derived `e00022079` from chain state with the same slice and commit
  hash, and resumed waiting for the same collection block. No state lost, no
  duplicate weight-set possible (the epoch had not reached scoring).
- **20:20 — first cross-validator artefact comparison on real epochs.** Pulled
  `commit.json` for `e00022078` and `e00022079` from both hosts: byte-for-byte
  identical `commit_hash`, `grader_hash` and `block_hash` on aarch64 and
  x86_64. The epoch logs differ in exactly one expected place: the ARM
  validator recorded the mock miner as `n_heads_invalid=1` while the x86 one
  recorded `0`, because only the ARM host can reach the mock miner's private
  10.x axon. Both set no weights, so the difference cannot diverge consensus —
  the shape I4 predicts for an unreachable-to-some miner whose proof is invalid
  anyway.
- **19:35** TD2 provisioning port answered ~4 min after create; `--inspect`
  matched base, compose hash and the console's instance id `fdb7411e…`; pushed
  head_cheap_b, hotkey td2, coldkeypub, key. Both TDs `provisioned: true`.
  From here both embed the pool inside the enclave (expected ~13 h at one
  pinned thread) before serving an axon or committing a head.
- **19:47–19:49** Both TDs at a flat **26% CPU** (one pinned vCPU of four),
  the embedding signature, with no console errors — the pool load over
  HuggingFace egress inside the enclave succeeded, inferred rather than read.
  TD1's trace: 59% pull/extract → 14% idle awaiting the push → 26% steady.
- **20:16** ARM validator's second live epoch `e00022079` reproduced the first
  exactly (mock miner rejected at DCAP parse, epoch skipped, `n_heads_invalid=1`,
  no weights). Next boundary 7948800.
- **20:00–21:00 — code landed during the wait.** PR #14 (lean sweep: four dead
  files, two empty packages, nine unused constants, every stale doc claim
  corrected) merged as `8befb1a`. PR #15 (sealed provisioning channel,
  operator-only encrypted log pull) opened from `2018ea0`; gate green.
- **21:05 — thread pin lifted (PR #16).** Measured on the laptop (12 cores,
  short 37-char prompts, so absolute rates are optimistic but the ratio holds):
  1 thread 2.38 q/s, 12 threads 5.99 q/s — **2.5x**, not 12x; the backbone is
  memory-bound past a few cores. On a 4-vCPU TD expect roughly 2–3x, so the
  ~13 h pass should become ~5 h. Decision (user): redeploy both TDs on the new
  image now rather than wait out the single-threaded run. Source reading by B:
  `dstack-cloud deploy` refuses on an existing instance, `--delete` recreates
  it, and the data disk is auto-delete — so a redeploy discards the cache and
  the sealed seed (new instance ids); only `stop`/`start` preserve them. The
  compose comment that claimed otherwise is corrected in the same PR.
