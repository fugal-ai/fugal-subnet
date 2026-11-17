"""A dedup cluster whose commitment blocks are unknown (seniority inf, as in
mock with no commitments) logs cleanly. It used %d, and formatting inf with %d
raised inside logging (seen in the v3 rehearsal log)."""
import logging

import numpy as np

from fugal_subnet.dedup import find_duplicates


def test_a_cluster_with_unknown_blocks_logs_cleanly(caplog):
    same = np.array([0, 1, 2, 3, 1, 0])
    with caplog.at_level(logging.INFO, logger="fugal_subnet.dedup"):
        out = find_duplicates({1: same, 2: same.copy()}, {1: float("inf"), 2: float("inf")})
    assert out == {2}
    messages = [r.getMessage() for r in caplog.records if "Dedup cluster" in r.msg]
    assert messages and "block inf" in messages[0]
