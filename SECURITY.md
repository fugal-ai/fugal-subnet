# Security Policy

Fugal processes untrusted miner artifacts, interacts with Bittensor wallets and networks, and can spend funds through OpenRouter. Please report security issues privately.

## Supported versions

| Version | Supported |
|---|---|
| `main` | Yes (protocol v1, live on finney netuid 70) |
| `release/v2` | Yes (protocol v2; the signed `image-v*` tags for the v2 image are cut from it until it merges into `main` at the activation block) |
| Anything else | No |

Support means best-effort investigation and fixes; it is not a service-level guarantee.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/fugal-ai/fugal-subnet/security/advisories/new). Do not open a public issue for a suspected vulnerability.

Include the affected commit or release, impact, reproduction steps, and a minimal proof of concept. Remove API keys, wallet mnemonics, private keys, and other credentials from every report.

Please do not exploit a vulnerability against public infrastructure, access data that is not yours, spend API credits, or disrupt a live network. We will acknowledge a complete report on a best-effort basis, coordinate remediation, and credit reporters who want attribution.

## Trust boundaries

- Miner-supplied synapse fields and `.npz` head bytes are untrusted.
- Model responses and external API metadata are untrusted inputs to deterministic graders.
- Benchmark downloads are external but pinned to declared revisions.
- Bittensor chain state is authoritative for registration, commitments, and weight submission.
- Local operator configuration, wallet files, and environment variables are trusted and must be protected by the operator.

## High-priority areas

| Threat | Controls |
|---|---|
| Pickle/code execution through `.npz` | `allow_pickle=False`, size limits, shape/dtype validation |
| Head swapping after question selection | On-chain hash commitment before the epoch boundary |
| Copied routing behavior | Protocol v2: only the champion seat pays, and a copy is never measurably better than its source on the same questions, so it cannot take the seat; the senior head by first on-chain commit (coldkey ownership, salted pre-flip tie) wins ties, and a junior must beat every senior seat holder by the margin (`copy_rule.py`). Protocol v1: three-clause deduplication: routing-decision cosine in a global model index space, relative weight distance, and the epsilon rule (a near-copy must beat its senior); the senior hotkey by registration block, then commitment block, survives. In `--live` the axon answers only validator permits or stake ≥ `FUGAL_MIN_VALIDATOR_STAKE` |
| Validator budget exhaustion | Resolved architecturally: validators never call models, so there is no budget to exhaust. Miners pay for their own inference inside the TEE |
| Forged or tampered TEE proofs | Approved-image matching against the quote's own measurement registers (MRTD, RTMR1, RTMR2; RTMR0 excluded) plus the compose-hash event replayed from the RTMR3 log against a `<base>:<compose_hash>` entry (bare bases refused outside mock), report_data binding, hotkey binding, slice binding, head-to-commitment binding, bundle-to-advertised-hash binding, cost consistency as a rejection — each with an executable exploit in `run_tee_attacks.py` |
| Understated or fabricated costs | Costs priced from the hash-pinned price table of the epoch's protocol (`data/models.json`, `data/models_v2.json`), metered inside the attested enclave, and reconciled per-question and per-model against the attested total |
| Steering the shared reference frame | Exploration questions and target models are nonce-derived, not miner-chosen; the frame pools over time rather than over miners |
| Credential disclosure | Environment-based secrets, no-key logging rule, ignored `.env` and key files |
| Malicious executable answers | Execution graders disabled inside the TD (`HARNESS_ALLOW_EXEC=0`) and the code benchmarks excluded from the pool; validators never run model output |
| Secret delivery into the TD | Head, hotkey keyfile, coldkeypub and API key cross only the attested push channel after the operator verifies nonce, base measurement, event-log replay, compose hash and instance id; a closed allow-list pinned by a test; nothing per-miner in the compose or on the cloud-readable shared disk |
| Validator disagreement | Single-source epoch identity (structurally enforced), pinned backbone batch size, CPU kernel dispatch pinning (torch + numpy/OpenBLAS), quantized routing decisions, pinned HF dataset revisions, deterministic HMAC-seeded slicing, immutable hash-pinned graders, commit-reveal epoch artifacts, environment fingerprint + startup assertion, two-process differential harness in CI |

## Known limitations

- Validators never execute model output: grading happens inside the miner's
  TD, and the code-execution graders (`exec_io`, `exec_unittest`) are disabled
  there (`FUGAL_HARNESS_ALLOW_EXEC=0`), so the code benchmarks are excluded
  from the pool. Re-enabling them needs the sandbox work in
  `docs/decisions/CODE_BENCHMARK_PLAN.md`.
- A miner's container logs are not public (`public_logs=false` is required,
  because the TD holds an API key). The operator reads them only through the
  encrypted, attested pull in `tee/provision.py`, authorised by the session
  key of the provisioning push; nobody else can.
- Exact receipt verification depends on an archive-capable Bittensor endpoint
  retaining the historical blocks referenced by a reveal.
- External model providers can be unavailable, change behavior, or return inconsistent outputs.
- The project has not yet received an independent security audit.

Operational support requests and ordinary bugs belong in the public issue tracker.
