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

## High-priority areas

- Unsafe `.npz` or other miner-controlled artifact handling
- Validator divergence or consensus manipulation
- Grader sandbox escapes
- Commitment, deduplication, scoring, or weight-setting bypasses
- API key, wallet, or sensitive-log exposure
- Unbounded or unauthorized OpenRouter spend

Operational support requests and ordinary bugs belong in the public issue tracker.
