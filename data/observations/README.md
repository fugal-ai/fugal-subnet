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
