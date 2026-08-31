# Consensus Governance

Fugal validators must derive identical grades and compatible routing scores from the same epoch inputs. Changes to that behavior are protocol changes, not routine refactors.

## Consensus-critical surfaces

The following are consensus-critical:

- Grader semantics and grader dispatch in `fugal_subnet/graders.py`
- Benchmark identity, normalization, revision pins, and slicing
- Epoch nonce derivation and boundary selection
- Head artifact schema and validation
- Matrix, soft-target, evaluation, scoring, deduplication, and reward formulas
- Commitment eligibility and reveal artifact schema

Transport, logging, CLI wording, and performance-only refactors are not normally consensus changes, provided their outputs remain byte-for-byte or numerically identical where required.

## Version identifiers

The current grader version is `sha256(fugal_subnet/graders.py bytes)`. Epoch artifacts must record that hash. Future protocol-level schemas should receive explicit version fields before incompatible variants are deployed.

## Change procedure

1. Open a public design issue describing the behavior, motivation, and compatibility impact.
2. Implement the change behind a new version or activation boundary; do not silently reinterpret completed epochs.
3. Add deterministic fixtures and adversarial cases.
4. Run the integration and attack suites on every supported Python version.
5. Publish the release, expected hashes, activation epoch, and operator upgrade instructions before activation.
6. Require validators to upgrade before the announced boundary. Validators on an old rule set must fail visibly rather than silently produce divergent weights.

Emergency security fixes may be disclosed privately until a coordinated release is available, but they still require an explicit version transition when semantics change.

## Compatibility policy

- Completed epoch artifacts are immutable.
- A new grader version applies no earlier than its declared activation epoch.
- Parsers may accept older artifacts for audit purposes, but live scoring uses exactly one declared version per epoch.
- Missing or mismatched consensus metadata is an error, not an invitation to guess.

## Release evidence

A consensus-affecting release should include the Git commit, grader hash, supported Bittensor SDK range, benchmark revision map, artifact schema version, activation epoch, and complete test results.
