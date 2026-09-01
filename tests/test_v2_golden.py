from fugal_subnet.v2.golden import EXPECTED_GOLDEN_SHA256, assert_golden, golden_sha256


def test_v2_consensus_golden_vector_is_byte_identical():
    assert_golden()
    assert golden_sha256() == EXPECTED_GOLDEN_SHA256
