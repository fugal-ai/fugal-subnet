# Recorded token observations

Normalized from three saved live-testnet proof bundles. Original bundle and runtime
hashes are retained per row. These are cost-source data, not new-protocol evidence.

```bash
python scripts/build_token_manifest.py \
  --observations data/observations/benchmark_tokens_v1.jsonl \
  --worker-profile 'Historical testnet harness metered OpenRouter calls, source runtimes and original proof-bundle hashes recorded per observation. Router-selected routes plus sparse nonce-assigned exploration. Exact generation limits follow harness grader profiles at each recorded source_runtime. These pre-transition proofs supply cost observations only, never success-protocol validator evidence.' \
  --output data/benchmark_tokens_v1.json
```

The script emits a review candidate, never an automatically reviewed manifest.

## v2 — 2026-11-02, the reviewed manifest's source

`benchmark_tokens_v2.jsonl` was recorded by `scripts/record_token_observations.py`
(this repository, the commit named in each row's `script_commit`) through the
benchmark's own metering proxy and call path, on a deterministic, stratified
40-question sample per model of the pinned pool (24 for mistralai/mistral-large-2512,
upstream-rate-limited). Provider-billed cost and
table-priced cost are both kept per row. No response text, no key material.

```bash
python scripts/build_token_manifest.py \
  --observations data/observations/benchmark_tokens_v2.jsonl \
  --worker-profile "$(python -c "import json;print(json.load(open('data/benchmark_tokens_v1.json'))['worker_profile'])")" \
  --output data/benchmark_tokens_v1.json
cp data/benchmark_tokens_v1.json fugal_subnet/routing_data/benchmark_tokens_v1.json
```

The builder emits `candidate`; the review record in `docs/SUCCESS_CONTRACT.md` is
what justified editing `status` to `reviewed`.

## `dense_calls_v1.jsonl`: local only, not committed

The seed heads' dense training labels (2026-11-03): the 17 models called on the
same 900 questions of the pinned pool, in two blocks, recorded as 8,840 rows (not
900 × 17 = 15,300). Only completed calls are written (there is no `failed`
field), and block 2 was cut short:
- block 1, 300 questions: 16 models on all or nearly all of them (kimi-k2.6 298,
  deepseek-v4-pro 299); `mistral-large-2512` 43 (upstream 429s).
- block 2, 600 questions, stopped at 17:44 UTC: per-model coverage 42–600
  (deepseek-v4-flash complete, mistral-large-2512 42).

Rows carry the same fields as
`bootstrap_frame_calls_v1.jsonl` (ids, tokens, provider and table cost,
`correct`, `response_hash`; no response text, no key material). It trains and
evaluates the operator's own seed heads and is **not** consensus input: the
reference frame is built from `bootstrap_frame_calls_v1.jsonl` alone
(`scripts/build_frame_from_calls.py`, pinned in `data/reference_frame.json`).

It is kept out of the repository on purpose (`.gitignore`): it is the seed
operator's training data, not protocol evidence, and it was once swept into an
unrelated PR by `git add -A`. A working copy lives with the operator's training
scripts; nothing in this repository reads it.

## `benchmark_tokens_protocol_v2.jsonl`: protocol v2 token observations (2026-11-04)

757 metered calls (19 models × 40 questions, 3 failed) through the v2 harness: `max_tokens`
16384, each model's pinned `reasoning` setting, priced on `data/models_v2.json`. Sampled at the
v2 category mix from a measurement pool of open sets. Builds the v2 token manifest:

```bash
python scripts/build_token_manifest.py --observations data/observations/benchmark_tokens_protocol_v2.jsonl \
  --models data/models_v2.json --worker-profile "$(python -c "import json;print(json.load(open('data/benchmark_tokens_protocol_v2.json'))['worker_profile'])")" \
  --output data/benchmark_tokens_protocol_v2.json
```

Same fields as `benchmark_tokens_v2.jsonl`; no response text, no key material.

## Protocol v4 menu seed inputs (2026-11-18)

Two files, the only inputs of `scripts/build_menu_v4_seed.py build` besides the
pinned v3 menu B and its token manifest. No key and no paid call was involved
in either.

- `openrouter_endpoints_2026-11-18.json`: OpenRouter's public
  `/api/v1/models/{model}/endpoints` page for each of the 25 models of
  `data/models_v3b.json`, fetched 2026-11-18T17:29Z with a generic client and
  no key. Each page is kept as served, with its sha256. The seed's prices,
  answer limits and quantizations are read from it, and every seed entry is
  checked against it under the compiled rules (`fugal_subnet/menu_rules.py`).
- `menu_v4_seed_pairs_e00026466-e00026484.json`: integer sums from the 19
  published protocol v3 reveals `e00026466`..`e00026484`
  (`https://storage.googleapis.com/fugal-reveals/70/<epoch id>/reveal.json`),
  each named with its sha256. Per entry: pooled pairs and attested prompt and
  completion tokens by benchmark over all 19 epochs, and calls and failed
  calls by operator over the last 12 (`e00026473`..`e00026484`). Operators are
  not named. The seed's token means and its health rule are computed from it.

```bash
python scripts/build_menu_v4_seed.py build --check \
  --listing data/observations/openrouter_endpoints_2026-11-18.json \
  --pairs data/observations/menu_v4_seed_pairs_e00026466-e00026484.json
```

rebuilds the seed and compares it byte for byte with `data/menu_v4_seed.json`
and with the copy the package ships, `fugal_subnet/routing_data/menu_v4_seed.json`
(the two inputs above are the defaults, so `build --check` alone does the
same). The script's
docstring gives the commands that rebuild the two input files from the public
pages.

## Protocol v4 menu seed: graded cells and picks (2026-11-19)

`menu_v4_seed_picks_e00026466-e00026509.json`: integer sums from the 44
published protocol v3 reveals `e00026466`..`e00026509`
(`https://storage.googleapis.com/fugal-reveals/70/<epoch id>/reveal.json`),
each named with its sha256 (the 19 it shares with the pairs file name the
same bytes). Per reveal, entry and question set: the graded scored-question
cells (the reveal's `shared_grades`) and their coldkey-median completion
tokens, each rounded half up and held to the answer limit the call was sent
with (v3 menu B's `max_tokens`; 16 cells were over it, by at most 210
tokens); the cells the champion picked and those every served head picked
(each head once, a failed call counting nothing) with the same tokens; and
the pooled exploration pairs and their completion tokens. No head, operator
or answer is named. The seed's pick-cost tables (W13) are computed from the
graded cells, and

```bash
python scripts/build_menu_v4_seed.py accuracy --first e00026466 --last e00026502
```

reproduces, leaving one epoch out, how well each cost table predicts what
the picks really cost (tests/test_pick_cost_accuracy.py). The docstring of
`scripts/build_menu_v4_seed.py` gives the command that rebuilds this file
from the public reveals; `build --check` now reads it as the seed's third
input.

## Protocol v4 question mix input (2026-11-19)

`mix_v4_cells_e00026466-e00026489.json`: integer sums from the 24 published
protocol v3 reveals `e00026466`..`e00026489`
(`https://storage.googleapis.com/fugal-reveals/70/<epoch id>/reveal.json`),
each named with its sha256: the epochs of the W0 replay that fixed section
10's targets. Per question set and per number of rows a cell had: exploration
cells, the sum of their entries' published inverse-probability weights, and
the sum of weight x correct rows. Only entries on the v4 seed menu are
counted; failed rows, the entry off that menu and `omnimath_` ids no reveal
placed are counted as left out. No operator, hotkey or answer is kept.

```bash
python scripts/build_mix_v4.py build --check
```

rebuilds the mix and compares it byte for byte with `data/pool_mix_v4.json`
and with the copy the package ships, `fugal_subnet/routing_data/pool_mix_v4.json`.
The script's docstring gives the commands that rebuild the observations file
from the public reveals.

## Protocol v4 allowance: heads' rows for new entries (2026-11-19)

`allowance_rows_e00026475-e00026509.json`: from the 35 published protocol v3
reveals `e00026475`..`e00026509`
(`https://storage.googleapis.com/fugal-reveals/70/<epoch id>/reveal.json`),
each named with its sha256 (the same bytes the seed's picks file names). v3
menu B's seven entries joined at `e00026475`; per pooled pair of one of them:
the reveal it is in, the entry, the pooled outcome and the pair's published
exploration weight; per head that published odds on any of those pairs, its
odds on each. Heads are numbered in order of their hash; no head, operator or
answer is named. It is what `tests/test_allowance_rows.py` measures the
protocol v4 allowance on (`docs/decisions/SEAT_MINIMAL_V4.md` 4.2).

```bash
python scripts/build_allowance_observations.py observe --check reveals
```

rebuilds it from a directory of the saved reveals and compares it byte for
byte; the script's docstring gives the command that saves them.

## Protocol v4 fast crowning: routing-test and timing inputs (2026-11-21)

`fast_crown_v4_inputs_e00026466-e00026493.json`: from the 28 published
protocol v3 reveals `e00026466`..`e00026493`
(`https://storage.googleapis.com/fugal-reveals/70/<epoch id>/reveal.json`),
each named with its sha256. How often two heads' scored picks differ; the
paired net-value items of every head against the champion, each call priced
at the v4 seed's pick-cost table and $1 a correct answer, centred per pair
and subsampled by category; the spread of a pair's epoch mean beyond its
sampling error; the odds test's per-label gains against the champion,
standardised per head and epoch and subsampled; labels per epoch and the
share left after two operators' labels are removed; and per entry its
exploration call latencies, failed calls by epoch and pooled outcomes by
category. No head, operator or answer is named; entries are public menu ids.
It is what `scripts/experiments/experiment_fast_crown_v4.py` simulates
`docs/decisions/FAST_CROWN_V4.md` on.

```bash
python scripts/experiments/experiment_fast_crown_v4.py measure --reveals <dir of e*.json>
python scripts/experiments/experiment_fast_crown_v4.py all
```

rebuild it from a directory of the saved reveals, then run every simulation
(fixed seeds, one process, about an hour; `--quick` for a smoke run).
