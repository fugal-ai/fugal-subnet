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
| Copied routing behavior | Cosine-similarity deduplication; earliest commitment wins |
| Validator budget exhaustion | Per-epoch budget, model pool caps, per-query price cap, timeouts, bounded concurrency |
| Credential disclosure | Environment-based secrets, no-key logging rule, ignored `.env` and key files |
| Malicious executable answers | V2-only networkless ephemeral OCI isolation, trusted parent comparison, time/resource/output caps |
| Validator disagreement | V2 pinned datasets, deterministic slicing/grading, replicated signed reports, grader-bundle hash, reveal artifacts |

## Known limitations

- Historical v0.1 executes generated answers without the v2 isolation boundary
  and is unsuitable for funded operation. V2 uses a separate launcher identity
  and an ephemeral OCI worker with `--network none`; the validator receives only
  a restricted Unix socket and never the container-engine socket.
- The launcher identity can create containers and is therefore privileged
  relative to its worker. Operators must isolate that identity from validator
  wallets, API credentials, benchmark gold values, and unrelated host paths.
- Exact receipt verification depends on an archive-capable Bittensor endpoint
  retaining the historical blocks referenced by a reveal.
- External model providers can be unavailable, change behavior, or return inconsistent outputs.
- The project has not yet received an independent security audit.

Operational support requests and ordinary bugs belong in the public issue tracker.
