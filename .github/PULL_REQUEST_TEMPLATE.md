## Summary

Describe the problem and the approach taken.

## Validation

- [ ] `scripts/gate.sh` is green (it runs CI's own steps, in CI's environments)
- [ ] A fix comes with a test that fails without it
- [ ] Tests ran without real OpenRouter calls or API spend

## Consensus and security checklist

- [ ] This change does not alter grader semantics, or it includes a documented consensus-version transition
- [ ] Every `np.load()` call uses `allow_pickle=False`
- [ ] `FugalProofSynapse.deserialize()` still returns the synapse instance
- [ ] No secret, credential, generated head, benchmark cache, or epoch artifact is included
- [ ] User-facing behavior and operational changes are documented

## Related issue

Closes #
