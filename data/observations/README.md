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
