## Summary

Describe the problem and the approach taken.

## Validation

- [ ] `python scripts/check_safety_invariants.py`
- [ ] `python tests/test_integration.py`
- [ ] `python -m fugal_subnet.attacks.run_attacks`
- [ ] Tests ran without real OpenRouter calls or API spend

## Consensus and security checklist

- [ ] This change does not alter grader semantics, or it includes a documented consensus-version transition
- [ ] Every `np.load()` call uses `allow_pickle=False`
- [ ] `FugalSynapse.deserialize()` still returns the synapse instance
- [ ] No secret, credential, generated head, benchmark cache, or epoch artifact is included
- [ ] User-facing behavior and operational changes are documented

## Related issue

Closes #
