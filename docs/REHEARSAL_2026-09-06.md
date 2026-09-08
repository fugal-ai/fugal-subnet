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
- **21:40 — second approved entry.** Image `main-45729c2` →
  `59039d0121d074da62e3dc380e88e01a7fe1e8f5e46409392e950a660158692a`
  (build 11 m 47 s); frozen compose `12d945d278ac746655afa9487c1d63b7932e7f2029413147acd425c1c8363910`
  (3564 bytes, TD1/TD2 identical, rehashed by the coordinator from the raw
  bytes; diff vs the first compose is exactly the digest, the cache comment
  and `FUGAL_BACKBONE_THREADS=0`). Both validators carry old+new entries —
  the first exercise of the rotation window — then both TDs are recreated
  with `deploy --delete`.
- **21:06–21:10** TD1 recreated with `deploy --delete` (instance deleted in
  2 m 20 s, shared image rebuilt, new instance up at 21:10:14) — **new
  external IP 35.184.229.7**; the old data disk and sealed seed are gone with
  the old instance, as the tool's source said. TD2 recreation follows.
  Both validators restarted with old+new entries at 20:59 (ARM env sha256
  `bcc03a8a…`; x86 verified by grep); both re-derived `e00022080` with the same
  slice hash and commit `c849e407…` as their pre-restart processes — a second
  unplanned restart drill, byte-identical.
- **21:07** Branch protection on `main` (seven required checks, strict, PR
  required) caught its first case: PR #19 was green but behind `main` and
  could not merge until updated.
- **21:13–21:15 — sealed channel on real hardware.** New TD1 (35.184.229.7,
  instance id `2106b176…`) answered on 8092 3 m 38 s after create.
  `--inspect`: the receiver offers an X25519 key and the Intel-signed quote
  covers `nonce || sha256(key)`; compose `12d945d2…` (the new entry); log
  replays to RTMR3. Sealed push accepted (`provisioned: true`), session key
  saved 0600. Then **the first operator log pull from a live TD**
  (`provision_td.py --logs`, 180 lines, encrypted): the miner logged
  `Backbone determinism configured: capability=avx2, threads=4` — the thread
  knob is in effect inside the enclave — `Computing backbone embeddings for
  21553 questions`, and the backbone weights downloading from HuggingFace
  over egress. The black box is open to its operator and to nobody else.
  Cosmetic: the miner logs `Head loaded: None` under provisioning (no path);
  fix later. TD2 recreated 21:13:45, same IP 35.193.250.187, new instance.
- **21:17** New TD2 (35.193.250.187, instance id `ad80f161…`) answered on 8092
  3 m 54 s after create; `--inspect` matched key binding, compose `12d945d2…`
  and the console's instance id; sealed push accepted; session key saved.
  **Both TDs provisioned over the sealed channel; embedding with 4 threads.**
  PR #19 (validator logs its approved entries) merged as `5b50dfd` once its
  branch was brought up to date.
- **21:20–21:22** Both recreated TDs at a flat **100% CPU** (all four vCPUs)
  against 26% before — the thread knob in effect. If throughput scaled with
  cores the pass would take ~3.3 h; the laptop measurement says ~2.5x, so
  expect the axons around 01:00–02:30 UTC.
- **21:30–22:48 (laptop closed; sessions suspended).** Nothing on the boxes
  noticed. Both validators completed `e00022080` and `e00022081` with the same
  shape as before (ARM: mock miner rejected at DCAP parse, `n_heads_invalid=1`;
  x86: `n_heads_invalid=0`, it cannot reach the private axon), identical
  commit hashes on both hosts every epoch, no weights set. Both TDs at a flat
  **100% CPU from 21:22 to 22:52** (Cloud Monitoring, 5-min means), no
  container restart, no console error. Peer sessions did not survive the
  suspend; the ARM watch timer did.
- **Finding (operability), corrected within the hour.** After the resume,
  probes of `/provision/status` on both TDs timed out from the laptop, and one
  slow answer (16 s) was read as the receiver thread starving behind four torch
  threads. Wrong. Session B found the real cause: the laptop's public IP had
  changed while it was closed (71.105.160.190 → 96.246.150.17), so the
  operator-only firewall rule silently dropped every packet, and the one slow
  answer coincided with the rule being updated. Re-timed after the fix: **64 ms
  on both TDs at 100% CPU** — the receiver is not starved. The real lesson: an
  operator-IP-pinned firewall rule fails as a *timeout*, not a refusal, and a
  residential IP changes without notice; a production operator needs a stable
  egress (or a jump host) for the provisioning and log path.

## 2026-09-07

- **04:53–04:57 — both TDs finished embedding and came up.** Measured
  inside the enclaves (sealed log pull): TD1 embedded 21,553 questions in
  **7 h 39 m** (21:14:46 → 04:53:51), TD2 in 7 h 38 m — **1.7x** faster than
  the 13 h single-thread figure on 4 vCPUs, not the 3.6x an earlier line here
  claimed (a subtraction error, caught on the third pass when TD1 was only at
  11,520/21,553 after 3 h 45 m at 0.85 q/s). The backbone is memory-bound
  past a few cores; the laptop's 2.5x at 12 threads was the better predictor. Each then registered its axon at
  its public IP, served, and committed its head hash on chain
  (TD1 `b0ce2fcb…` at block 7951219, TD2 `bb18e523…` at 7951231) — the first
  time a dstack TD signed its own extrinsics with a pushed wallet.
- **04:56 / 04:59 — first epoch run FAILED on both miners.** A real bug, found
  only on real hardware with a real provider: one model call
  (`deepseek/deepseek-v4-flash` through the metering proxy) timed out after
  180 s; `harness._call_model` returned `None`; `run_benchmark` then called
  `.encode()` on it and the whole epoch aborted (`Epoch run failed (1
  consecutive)`), leaving no proof. The second attempt (`e00022087`) failed
  the same way. Every test and rehearsal stub returns a string, so nothing
  ever exercised the failure path. OpenRouter spend at that point:
  **$0.0094**. Validators saw no TD proof for `e00022086`, as they should.
- **05:39–06:00 — harness fix landed and the third entry issued.** PR #20
  merged (`d9fa3a5`); image `main-d9fa3a5` →
  `9d7916c34f176929374c577f2bd8d94a01f954cc60ecd781e7fe61b5df5ce0a9`
  (Cloud Build 19 m 32 s). The previous session's scratchpad — the deploy
  toolchain and both project directories — had been deleted with that
  session, so the CLI, the UKI image and mtools were rebuilt here and both
  projects recreated from the documented fields; the frozen compose hashed to
  `003545255166b57c991a9c0f1d3dd8fada19e40e1afb0133e37df9795251ab0d`,
  identical for TD1/TD2 and equal to the repo compose with the new digest.
  Both validators rotated to entries 2+3 (entry 1 retired: no instance ever
  verified on it) and restarted.
- **06:01–06:12 — third redeploy.** Both TDs recreated with `deploy --delete`
  (TD1 → new IP **34.122.55.116**, instance id `36abf3df…`; TD2 stays at
  35.193.250.187, instance id `92ae6841…`); post-deploy compose hash unchanged
  (`00354525…`). `--inspect` on each: sealed key offered and covered by the
  quote, base matches, log replays, compose is entry 3. Sealed pushes accepted
  at 06:09:35 and 06:12:18; session keys saved. Expected (corrected): axons
  ~13:00 UTC, first scored epoch at the next 72-minute boundary after both
  commitments.
- **13:48–13:52 — third pass done; miners live on the fixed image.** TD1
  embedded in 7 h 38 m (06:09:51 → 13:48:20), TD2 in ~7 h 40 m; both served
  their axons at their public IPs and found their head hashes already
  committed (same heads, blocks 7951219/7951231), so both are scoreable at
  once. Both started `e00022094` (TD1 13:49:36, TD2 13:52:46) — the first
  epoch runs on the non-text-reply fix. TD2 logged two 180 s model-call
  timeouts (`deepseek-v4-flash`, `gpt-5.4-nano`) and **kept going**, where the
  old harness aborted. OpenRouter spend rose from $0.0117 to $0.0657 during
  the run — real calls, priced.
- **Finding (operability):** within seconds of serving, TD1's public axon
  received 114 scanner probes (`.env`, `wp-config.php`, `actuator`, …); the
  SDK logs each as an ERROR-level `UnknownSynapseError`. Harmless, but it buries
  the miner's own lines; a production log filter for that class is warranted.
- **14:15 — `e00022094`: no TD proof on either validator.** Both TDs were still
  running the epoch at the collection block: TD1 started 10 min into the
  epoch, TD2 13 min in (they came up mid-epoch), and 31–33 min later neither
  had finished 315 serial model calls; TD2 had also spent 2 × 180 s on
  timeouts. The validators queried the TD axons and got an empty reply within
  a second (correct: "no proof yet" is not an error), and skipped the epoch.
  Whether a full slice fits the 36-minute window when a run starts *at* the
  boundary is exactly what `e00022095` measures; the run-time of this epoch's
  (uncollected) proof is the number that decides it.
- **14:42 — finding: a serial benchmark cannot fit the window.** 53 minutes
  after starting `e00022094`, neither miner had finished its 315 calls (TD1
  with zero timeouts, so this is plain latency: ~10 s per call for
  2048-token completions on math prompts, against the "6 s per question,
  ample for serial API calls" assumed where `EPOCH_COLLECT_FRACTION` is
  defined). The collection point is 36 minutes after the boundary. So even a
  miner that starts exactly at the boundary would miss it. Spend for the
  partial run: ~$0.19 across both miners. Fix (miner-side, in progress):
  `HARNESS_CONCURRENCY=8` calls in flight, attribution by request id, fsum
  totals, threading proxy; grading and result order unchanged.
- **15:03–15:15 — fourth entry.** PR #25 (concurrent benchmark, request-id
  attribution, fsum totals, threading proxy, scan-spam filter) merged as
  `37495ae`; image → `e4c2deff9c9ee2b38b3495690c7e5c5b52acb740a96e1b16664a235725736ca4`
  (Cloud Build 15 m 42 s); frozen compose
  `41de1e4c12da1edd86fdfdd4c61279d1d82f08cb242f10551d5799e58e0f9e10`
  (TD1/TD2 identical, equals the repo compose). Both validators rotated to
  entries 3+4 (entry 2 retired, never verified) and restarted; both TDs to be
  recreated. Fourth embedding pass follows (~7.6 h).
- **15:05–15:12 — fourth redeploy.** TD1 recreated → **136.64.249.27**,
  instance id `63636f20…`; `--inspect` matched entry 4 and the sealed key;
  pushed at 15:12. TD2 recreation in progress. PR #26 (entry 4) merged as
  `da6f0d9`.
- **15:16** TD2 recreated at 35.193.250.187 (instance id `d72bf92b…`),
  inspected (entry 4, sealed key bound) and provisioned. Both miners now run
  the concurrent harness; fourth embedding pass under way — expected to serve
  ~22:50 UTC, first scored epoch at the next boundary after that.
- **07:06–08:50 — finding (liveness, I6): the ARM box lost DNS for ~1 h 40 m
  and the validator rode it out.** `websockets keepalive ping failed` at
  07:06, then `ConnectionError: Temporary failure in name resolution` on every
  chain call; the epoch loop logged `Error in epoch` and retried every ~70 s
  (72 iterations) until resolution returned, then resumed at the next boundary
  on its own. Cost: epoch `e00022089` has no ARM entry (the x86 validator has
  one) — a one-validator gap, which Yuma tolerates and which would have shown
  as a weight disagreement for that epoch had there been valid proofs. No
  operator action, no restart, no state loss. Two notes for production: the
  retry has no backoff (72 tracebacks in the journal), and a validator on a
  host with flaky DNS should pin a resolver.
- **22:53–23:05 — the concurrent harness fits the window.** Fourth pass done
  (TD1 embedded 15:15 → ~22:50, 7 h 35 m). Both miners started `e00022101` at
  the boundary and finished the full 315-call slice in **~10–12 minutes**
  (TD1 22:53:12 → 23:05:02: 227/300 correct, $0.1879; TD2 22:55:15 →
  23:05:07: 215/300, $0.1554) against the 36-minute collection point — where
  the serial harness ran past 53 minutes. Real OpenRouter calls, priced from
  the pinned table, inside the enclave. Proofs held for collection at block
  7956540.
- **23:14 — `e00022101`: no TD proof on either validator, by timing.** Its
  collection block (7956540) fell at 22:39, before either miner had finished
  embedding; they ran that epoch late (22:53–23:05) and held proofs nobody
  would collect. `e00022102` (boundary 7956720, 23:15) is the first epoch both
  miners started at the boundary: TD2 done 23:22 (223/300, $0.2021), TD1
  23:27 (222/300, $0.2177), 24–29 minutes before its collection block.
- **Finding (tooling):** `scripts/verify_live_miner.py` lagged the verifier's
  contract — it never passed `expected_hotkey`, so it fetched and saved a
  genuine 315-result live proof (`results/live_proofs/proof-uid5-e00022101.json`,
  38,248-byte attestation) and then refused to verify it. The validator's
  own call was correct; only the operator tool had drifted. Fixed in the
  working tree; the positive and negative controls run against the fix.
- **23:30 — POSITIVE CONTROL PASSED.** `verify_live_miner.py` (fixed) fetched
  TD2's `e00022102` proof over its axon — 315 results, 38,247-byte dstack
  attestation — and ran the full `--live` verification path on the laptop:
  **DCAP verification passed (status=UpToDate)**, base measurement and
  compose hash matched entry 4, report_data bound the proof body, hotkey,
  slice, exploration set, head, and advertised bundle hash all matched:
  `valid: True — All checks passed`. Saved to
  `results/live_proofs/proof-uid6-e00022102.json`. The first Fugal proof from
  real Intel TDX hardware to verify end to end.
- **23:40 — NEGATIVE CONTROL PASSED.** The same genuine `e00022102` proof
  from TD1 (315 results, real quote) checked against the *retired* entry 1:
  `valid: False — Unapproved application: compose-hash 41de1e4c… not among 1
  approved for this image`. DCAP and the base measurement pass (the hardware
  is real); the application binding refuses it. A check that can say no.

## FIRST SCORED EPOCH — `e00022102`, 2026-09-07 23:51 UTC (measured)

Both validators, independently, on different CPU architectures:

| | ARM (aarch64, uid 0) | GCP x86_64 (uid 1) |
|---|---|---|
| Proofs verified | 2 (uids 5, 6), DCAP `UpToDate` both | 2, DCAP `UpToDate` both |
| Mock control (uid 4) | rejected at DCAP parse | unreachable (private IP) |
| Unverifiable | 0 | 0 |
| Scores | uid 5: acc 0.740, quality 1.2605, thrift 0.0412, score 0.0895; uid 6: acc 0.7433, quality 1.2669, thrift 0.0756, score 0.0956 | **identical to the last digit** |
| Weights | {5: 0.48372370450239816, 6: 0.5162762954976018} | **identical** |
| `set_weights` | Success; commit recorded at block 7956902 | Success; block 7956912 |
| Chain `LastUpdate` | 7956902 ≥ boundary 7956720 | 7956912 ≥ boundary |
| Reveal | verified; `reveal.json` identical to the other host's in every field but `environment` (the host fingerprint, by design) | same |
| Anomalies | 0 | 0 |
| Consensus digest | `1d51078642731d50` | `1d51078642731d50` |

That is I1 and I9 on real attested proofs, and the first weights the subnet
has ever set from a TEE-verified benchmark. Verification cost on the ARM host
(Phala collateral): ~1.6 s for two proofs including DCAP.

**What the scores say, and why they look small.** `thrift` is 0.04–0.08 because
the reference frame is two epochs old: with two trials per model the prior
ties and `best_model()` breaks the tie toward the cheapest (`gpt-5.4-nano`,
reference cost $0.0066 against the miners' $0.09–$0.16). This is the cold-start
behaviour `FRAME_PRIOR_STRENGTH` is calibrated for and it washes out with
exploration samples; it is not a scoring defect. The two heads are ranked in
the expected order (TD2's cheaper routing wins).

**Cost, real vs pinned.** TD2's proof prices its 315 calls at **$0.2021** from
the pinned table. OpenRouter's account meter rose $0.455 → $1.264 across the
two miners' two epochs, whose pinned totals sum to $0.77 — so the pinned table
tracked real billing within roughly 5% for this mix (approximate: account-level
meter, not per-proof; the proof carries no provider-cost field, which is worth
adding). Two things the economics docs assumed differently: completions
averaged **534 tokens** (max 2,049) against the 256 assumed, and the 15
exploration questions — nonce-assigned across all 16 priced models including
`gpt-5.5` and `claude-opus-4.8` — were **~60% of the epoch's cost** for a head
that routes only among the three cheapest models.

**Correction to this record.** The 2026-09-06 20:10 entry says `fugal-val2`
ran on the PCCS cache from then on. It did not: the env line was never written
(found at 23:55 when its log named Phala), so every verification so far used
Phala on both hosts. The cache itself was measured correctly (15 ms vs 730 ms
with the validator's own `verify_dcap`). Fixed at 23:58; the remaining epochs
compare the two paths.

## 2026-09-08 — three consecutive scored epochs, then stop

| epoch | ARM (Phala collateral) | x86 (local PCCS cache from `e00022103`) | agreement |
|---|---|---|---|
| `e00022102` | 2 valid, mock rejected, weights {5: 0.4837, 6: 0.5163}, set+confirmed @7956902 | 2 valid, same weights, set+confirmed @7956912 | scores, weights, commit, digest identical |
| `e00022103` | 2 valid, weights {5: 0.48878, 6: 0.51122}, **capped**, set+confirmed | same weights, capped, set+confirmed | identical |
| `e00022104` | 2 valid, weights {5: 0.49195, 6: 0.50805}, capped, set+confirmed | same, set+confirmed | identical |

Weight capping engaged from the second scored epoch (±0.3 per UID), as
designed. `n_heads_unverifiable = 0` on both hosts in all three epochs — the
number OPEN_WORK §2 asked for. Route A comparison: the x86 validator fetched
collateral from its local cache (`DCAP collateral endpoint:
http://127.0.0.1:8081`) for `e00022103`/`e00022104` while the ARM validator
used Phala, and both produced identical weights — the collateral source does
not move the verdict when both succeed. Verification of two proofs took ~1.4 s
(cache) and ~2.1 s (Phala) end to end including everything but the query.

**02:21 UTC — both TDs stopped** (`gcloud compute instances stop`: disks,
sealed seeds and embedding caches preserved; compute and OpenRouter spend
halted). Validators left running; they log `no_valid_proofs` epochs until the
TDs are started again or deleted. **OpenRouter total: $1.93 of $20.** GCP:
two `c3-standard-4` for ~29 h of instance time across four deployments plus
the e2-medium ≈ $13.

The observation phase of the rehearsal is complete.

## 2026-09-08 — I3 decided and implemented

Option 3 from `docs/I3_DECISION.md`: the score is headroom above the
constant-policy frontier (`fugal_subnet/frontier.py`). Every single model and
every random mixture of models scores zero by construction; a router earns the
accuracy it adds at its own price; a quality floor of 0.8 of the frontier's
best accuracy and a frontier-confidence factor (`FRONTIER_MIN_TRIALS`) keep
"match frontier quality" a requirement and stop a cold frame paying constant
policies for its own ignorance — while the frontier is cold, unassigned weight
burns to UID 0. `SCORE_QUALITY_EXPONENT` and the thrift/quality caps are gone.
Pinned by `tests/test_frontier.py` and the rewritten
`tests/test_degenerate_constant_policy.py`. This re-scores every miner and is
the consensus change the rehearsal's "not yet" was about.

**16:36–16:41 UTC — both live validators moved to the frontier score
together.** PR #28 merged as `e373921`. ARM (uid 0) and `fugal-val2` (uid 1)
fast-forwarded from `e284963`, `uv sync --locked --extra tee` (no dependency
change), restarted 56 s apart within the same epoch's wait window. Both logged
the same two approved entries, the same `Consensus environment digest
1d51078642731d50`, and the same commitment for `e00022116`
(`af3eea27…`). Both loaded their pre-frontier state files (the new record
fields default; measured by the epoch running, not by reading the loader).
Epoch outcome identical: the mock miner's proof rejected on ARM (`quote does
not parse`), no valid proofs on either, epoch skipped. The TDs are still
stopped, so the first frontier-scored epoch on real proofs has not happened
yet; starting a TD is what produces it.

## 2026-09-08 — first frontier-scored epoch on a real proof (e00022118)

**17:45 UTC** td1 started (`gcloud compute instances start`; new external IP
34.63.177.161). `--inspect` at 17:46: entry 4, instance id `63636f20…`
preserved across stop/start; sealed push accepted. Embedding cache hit in 6 s;
axon re-served at the new IP; head commitment already on chain (block
7951219). Joined e00022117 late — proof ready 30 s after the validators
queried — then benchmarked e00022118 in under 6 min. Measured spend:
$0.1598 (e00022117) and $0.1252 scored + exploration (e00022118), so
≈$0.16/epoch for a three-cheap-model head, not the $0.10 estimated.

**19:04–19:05 UTC, e00022118 — both validators identical**: DCAP `UpToDate`
(ARM via Phala, val2 via its local cache), frame 15 samples, best model
mistral-large-2512 at 0.629 (7 trials), frontier hull `[deepseek-v4-flash,
gpt-5.4-nano, mistral-large-2512, kimi-k2.6]`, confidence 0.08 (least-observed
hull model 4 trials). UID 5: 242/300 (0.807), headroom 0.120, composite
0.00382, weights `{5: 0.792, 6: 0.208}` (UID 6 is the stopped td2 decaying
under the ±0.3 cap). Weights set and confirmed on both (blocks 7962671,
7962672). Reveal files differ only in `environment.platform` (aarch64 vs
x86_64); `consensus_digest` identical.

**Two defects, visible only with a real proof:**

1. **Nothing burned.** Confidence 0.08 should have sent 92% of the miner
   share to UID 0; the miner got all of it. `composite` multiplies every
   miner's score by the same factor and `compute_weights` normalises, so the
   factor cancels. Fixed: `compute_weights(paid_fraction=frontier.confidence)`
   burns after normalising.
2. **Migrated evidence priced at 4.4x.** UID 5's record predates `n_priced`;
   the new epoch decayed a zero `n_priced` and divided three epochs of decayed
   cost by 300 questions: `$0.00184/q` in the reveal against `$0.000417/q`
   spent, so the frontier was read at its flat peak (0.629 instead of 0.572)
   and headroom was 0.120 instead of 0.177. Fixed: `Evidence.priced_n` reads
   a zero `n_priced` as `n_total` on every path.

Both fixes change the weight vector; both validators moved together (below).
