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
