# Security Policy

Fugal processes untrusted miner artifacts, interacts with Bittensor wallets and networks, and can spend funds through OpenRouter. Please report security issues privately.

## Supported versions

| Version | Supported |
|---|---|
| `main` | Yes |
| `0.1.x` | Yes |
| Older versions | No |

Support means best-effort investigation and fixes while the project is pre-launch; it is not a service-level guarantee.

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
| Copied routing behavior | Cosine-similarity deduplication in a global model index space; earliest commitment wins |
| Validator budget exhaustion | Resolved architecturally: validators never call models, so there is no budget to exhaust. Miners pay for their own inference inside the TEE |
| Forged or tampered TEE proofs | Approved-image matching against the quote's own measurement registers (MRTD, RTMR0-2), report_data binding, slice binding, head-to-commitment binding, bundle-to-advertised-hash binding, cost consistency as a rejection — each with an executable exploit in `run_tee_attacks.py` |
| Understated or fabricated costs | Costs priced from the hash-pinned `data/models.json`, metered inside the attested enclave, and reconciled per-question and per-model against the attested total |
| Steering the shared reference frame | Exploration questions and target models are nonce-derived, not miner-chosen; the frame pools over time rather than over miners |
| Credential disclosure | Environment-based secrets, no-key logging rule, ignored `.env` and key files |
| Malicious executable answers | Trusted parent-side output comparison (`exec_io` grader), time/resource/output caps, Docker isolation recommended for validators |
| Validator disagreement | Single-source epoch identity (structurally enforced), pinned backbone batch size, CPU kernel dispatch pinning (torch + numpy/OpenBLAS), quantized routing decisions, pinned HF dataset revisions, deterministic HMAC-seeded slicing, immutable hash-pinned graders, commit-reveal epoch artifacts, environment fingerprint + startup assertion, two-process differential harness in CI |

## Known limitations

- Code execution graders (`exec_io`, `exec_unittest`) run generated code on the
  validator host. Operators should run the validator inside a Docker container
  or VM to contain untrusted execution. The `exec_io` grader mitigates forgery
  by comparing outputs parent-side, but host isolation remains the primary
  defense.
- Exact receipt verification depends on an archive-capable Bittensor endpoint
  retaining the historical blocks referenced by a reveal.
- External model providers can be unavailable, change behavior, or return inconsistent outputs.
- The project has not yet received an independent security audit.

Operational support requests and ordinary bugs belong in the public issue tracker.
