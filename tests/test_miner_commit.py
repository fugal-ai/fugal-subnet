"""The head commitment: replaced when it names another head, timed around the flip.

From the neurons/ review (2026-11-08), two rules with no test:
- ensure_commitment must commit when the chain holds a DIFFERENT head's hash.
  Planting "any existing commitment counts" survived everything; under it, a
  miner that retrained, or switched to its v2 head at the flip, would never
  commit the new hash and every validator would reject all its proofs.
- the v2 commit window: after the last v1 boundary (an earlier commit fails
  this hotkey's last v1 epoch) and before the flip (a later one is missed at
  the flip boundary). It was inline in main(); planted changes survived.
"""
import types

import pytest

import neurons.miner as miner
from fugal_subnet import commitments


class _Chain:
    def __init__(self, existing, succeed=True):
        self.existing, self.succeed, self.written = existing, succeed, []

    def set_commitment(self, wallet, netuid, head_hash):
        self.written.append(head_hash)
        return (self.succeed, "" if self.succeed else "rate limited")


WALLET = types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address="5hk"))


@pytest.fixture
def chain(monkeypatch):
    def make(existing, succeed=True):
        c = _Chain(existing, succeed)
        monkeypatch.setattr(commitments, "get_commitments_with_blocks",
                            lambda st, netuid: ({"5hk": existing} if existing else {}))
        return c
    return make


def test_the_same_head_already_on_chain_is_not_rewritten(chain):
    c = chain(("a" * 64, 900))
    assert commitments.ensure_commitment(c, WALLET, 70, "a" * 64) and c.written == []


def test_another_heads_commitment_is_replaced(chain):
    c = chain(("old" + "0" * 61, 900))
    assert commitments.ensure_commitment(c, WALLET, 70, "b" * 64)
    assert c.written == ["b" * 64]


def test_no_commitment_yet_is_written_and_a_failed_write_is_reported(chain):
    c = chain(None, succeed=False)
    assert commitments.ensure_commitment(c, WALLET, 70, "c" * 64) is False
    assert c.written == ["c" * 64]


@pytest.mark.parametrize("bpe", [300, 360])
def test_the_v2_commit_lands_after_the_last_v1_boundary_and_before_the_flip(bpe):
    act = 50 * bpe
    commit_at, deadline = miner._v2_commit_window(act, bpe)
    assert act - bpe < commit_at < deadline < act
    from fugal_subnet.config import COMMIT_AFTER_BOUNDARY_BLOCKS, COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS
    assert commit_at == act - bpe + COMMIT_AFTER_BOUNDARY_BLOCKS
    assert deadline == act - COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS
