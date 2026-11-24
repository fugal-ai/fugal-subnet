"""The v4 consensus section is frozen at the v0.7.0 value.

A release that ships the next protocol runs v4 until that protocol's block,
next to validators still on v0.7.0: both must publish the same v4 digest
until then. The v4 section (consensus_v4.section(): the activation block, the
registered constants and the pinned files) is pure code and data, so its hash
is the same on every machine; the rest of the digest (packages, BLAS, CPU
dispatch, the epoch block) is the v1-v3 material, unchanged by this rule.

The pinned hash is the section v0.7.0 computes (origin/main 69e0b91) and the
one the live root published in its e00026591 reveal (`environment.v4`), whose
consensus digest is 92a9bc4ae53b9750 on finney (8d6b7f845997fadf with the
test settings). A new protocol's names go in its own registry (consensus_v5)
and are classified NOT_V4 here, never added to a v4-registered constant.
"""
import hashlib
import json

from fugal_subnet import config, consensus_v4

# sha256 of the canonical v4 section of v0.7.0 (and of the live root's e00026591 reveal).
V070_SECTION_SHA256 = "3f697c4132409e856af53e2f45a253ce4ea74faf58da672b7968bd753fa61c9b"


def _section_sha256() -> str:
    canonical = json.dumps(consensus_v4.section(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_the_v4_section_is_v070s():
    assert _section_sha256() == V070_SECTION_SHA256


def test_no_v5_name_is_in_the_v4_section():
    constants = consensus_v4.section()["constants"]
    assert not [k for k in constants if "V5" in k.split(":", 1)[1]]
    for key in ("fugal_subnet.config:CONSENSUS_ENV_INPUTS", "fugal_subnet.config:OPERATIONAL_ENV_VARS"):
        assert not [n for n in constants[key] if "V5" in n or n == "FUGAL_ARENA_CACHE_DIR"]


def test_v5_env_names_are_classified_in_their_own_lists():
    assert "FUGAL_V5_ACTIVATION_BLOCK" in config.CONSENSUS_ENV_INPUTS_V5
    assert "FUGAL_V5_ACTIVATION_BLOCK" not in config.CONSENSUS_ENV_INPUTS
    for name in config.OPERATIONAL_ENV_VARS_V5:
        assert name not in config.OPERATIONAL_ENV_VARS
        assert name not in config.CONSENSUS_ENV_INPUTS and name not in config.CONSENSUS_ENV_VARS
