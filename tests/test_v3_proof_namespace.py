"""The routing-evidence namespace a validator expects is the one the harness
stamps, for every protocol version.

Found by the v3 local-chain rehearsal (2026-11-16): the harness stamped v3
proofs with routing_protocol.identity_for(3) (the v3 menu and token manifest),
while verify_proof expected identity_v2() for every version >= 2, so every v3
proof failed with "Incompatible routing protocol evidence namespace".
"""
from types import SimpleNamespace

import pytest

from fugal_subnet.routing_protocol import identity_for
from fugal_subnet.tee.verify import verify_proof

NAMESPACE_REASON = "Incompatible routing protocol evidence namespace"


def _verify(proof, version):
    """Run verify_proof far enough to pass or fail the namespace check. A proof
    with nothing but a namespace fails later checks by raising or returning
    another reason; only the namespace verdict matters here."""
    try:
        return verify_proof(proof, set(), "", "", {}, mock=True, protocol_version=version)
    except Exception:  # noqa: BLE001 - later checks need fields this stub lacks
        return None


@pytest.mark.parametrize("version", [1, 2, 3])
def test_the_namespace_the_harness_stamps_is_the_one_verify_expects(version):
    stamped = SimpleNamespace(routing_protocol=identity_for(version))
    result = _verify(stamped, version)
    assert result is None or NAMESPACE_REASON not in (result.reason or "")


@pytest.mark.parametrize("stamped, epoch", [(2, 3), (3, 2), (1, 3)])
def test_a_proof_stamped_for_another_version_is_refused(stamped, epoch):
    result = _verify(SimpleNamespace(routing_protocol=identity_for(stamped)), epoch)
    assert result is not None and not result.valid
    assert NAMESPACE_REASON in result.reason
