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

## Rehearsal heads

| | sha256 | routes among |
|---|---|---|
| `head_cheap_a.npz` | `b0ce2fcbb26167201c2fc868cd618f1e4c00745bb768644bfbc60c863c979bcb` | deepseek-v4-flash / gpt-5.4-nano / llama-4-maverick, ~⅓ each |
| `head_cheap_b.npz` | `bb18e52302ce74a31e050fc820afd027e8076a2535faa6132d9bbebc9e7b41f3` | same three, different assignment (31% agreement with a) |

Expected spend per 300-question epoch per miner, from the pinned table:
≈ $0.014 routing + ≈ $0.027 exploration.

## Run log

_(appended as the run proceeds)_
