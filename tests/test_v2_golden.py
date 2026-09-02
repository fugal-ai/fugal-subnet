import json
from pathlib import Path

from fugal_subnet.consensus_manifest import canonical_json
from fugal_subnet.v2.golden import (
    EXPECTED_GOLDEN_SHA256,
    EXPECTED_MATH_SHA256,
    assert_golden,
    build_golden_vector,
    golden_math_sha256,
    golden_sha256,
)

FIXTURE = Path(__file__).parent / "fixtures" / "v2_golden.json"


def test_v2_consensus_golden_vector_is_byte_identical():
    assert_golden()
    assert golden_sha256() == EXPECTED_GOLDEN_SHA256
    assert golden_math_sha256() == EXPECTED_MATH_SHA256


def test_v2_golden_matches_committed_fixture():
    """Drift shows up as a readable JSON diff, not an opaque hash mismatch."""
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert build_golden_vector() == committed
    assert canonical_json(committed["math"]) == canonical_json(
        build_golden_vector()["math"]
    )


def test_v2_golden_math_is_independent_of_packaged_material():
    """Rebuilding manifest/grader material must never move the math pin."""
    vector = build_golden_vector()
    assert set(vector["material"]) == {
        "grader_sha256",
        "manifest_sha256",
        "question_commitment",
    }
    mutated = json.loads(json.dumps(vector))
    for key in mutated["material"]:
        mutated["material"][key] = "0" * 64
    assert canonical_json(mutated["math"]) == canonical_json(vector["math"])
