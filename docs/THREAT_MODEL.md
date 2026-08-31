# Threat Model

This document summarizes the main security boundaries in Fugal v0.1.0. It is not a security audit.

## Assets

- Validator and miner wallet material
- OpenRouter API credentials and budget
- Consensus integrity of questions, grades, scores, and weights
- Availability of validator and miner processes
- Integrity of submitted router heads and published epoch artifacts

## Trust boundaries

- Miner-supplied synapse fields and `.npz` head bytes are untrusted.
- Model responses and external API metadata are untrusted inputs to deterministic graders.
- Benchmark downloads are external but pinned to declared revisions.
- Bittensor chain state is authoritative for registration, commitments, and weight submission.
- Local operator configuration, wallet files, and environment variables are trusted and must be protected by the operator.

## Principal threats and controls

| Threat | Current controls |
|---|---|
| Pickle/code execution through `.npz` | `allow_pickle=False`, compressed and decompressed size limits, shape/dtype validation |
| Head swapping after question selection | On-chain hash commitment before the epoch boundary |
| Copied routing behavior | Cosine-similarity deduplication; earliest commitment wins |
| Validator budget exhaustion | Per-epoch budget, model pool caps, per-query price cap, timeouts, retries, bounded concurrency |
| Credential disclosure | Environment-based secrets, no-key logging rule, ignored `.env` and key files |
| Malicious executable answers | Process isolation, time/resource/output caps, deterministic grading |
| Validator disagreement | Pinned datasets, deterministic slicing/grading, grader hash, reveal artifacts |

## Known limitations

- The executable-answer sandbox limits processes and resources but does not provide network isolation by itself. Production validators should run grading inside a network-disabled container.
- External model providers can be unavailable, change behavior, or return inconsistent outputs.
- The project has not yet received an independent security audit.
- Fugal is pre-launch software and should not be treated as production-ready.

Report vulnerabilities according to [SECURITY.md](../SECURITY.md).
