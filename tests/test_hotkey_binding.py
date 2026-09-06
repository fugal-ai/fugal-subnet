"""A proof names the miner it was produced for.

Before this, `content_hash` covered the epoch, the slice, the head and the
runtime — and no miner. Any party could take an honest miner's proof off its
public axon response, commit the same weights_hash, and present it as their own.
Verification passed. What stopped it was `dedup.find_duplicates` plus
commit-block seniority: a cosine threshold and a block ordering, which is a
statistical defence against a problem that has a cryptographic answer.

The hotkey is inside content_hash, and content_hash is bound into the TDX
report_data, so the HARDWARE attests whose proof this is.
"""
import pytest

from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from fugal_subnet.tee.runtime import TEERuntime
from fugal_subnet.tee.verify import verify_proof

HONEST = "5HonestMinerHotkeyAddressAAAAAAAAAAAAAAAAAAAAAAAAAA"
RELAYER = "5RelayerMinerHotkeyAddressBBBBBBBBBBBBBBBBBBBBBBBBB"


def _proof(hotkey, **kw):
    p = BenchmarkProof(
        hotkey=hotkey,
        epoch_id="e00000001",
        nonce="ab" * 32,
        questions_hash="c" * 64,
        weights_hash="d" * 64,
        source_hash="e" * 64,
        results=[QuestionResult(
            question_id="q1", routed_model="m", correct=True, cost_usd=0.01,
            response_hash="f" * 64, prompt_tokens=10, completion_tokens=5,
        )],
        total_cost_usd=0.01,
        per_model_costs={"m": 0.01},
        attestation_quote=b"",
        timestamp=1.0,
        **kw,
    )
    # Attest it the way the miner does: report_data commits to content_hash.
    p.attestation_quote = TEERuntime(mock=True).generate_attestation(
        bytes.fromhex(p.content_hash()))
    return p


def test_the_hotkey_changes_the_content_hash():
    """If it did not, it would be a field the attestation does not cover, and
    an attacker could rewrite it freely."""
    assert _proof(HONEST).content_hash() != _proof(RELAYER).content_hash()


def test_the_hotkey_survives_a_json_round_trip():
    p = _proof(HONEST)
    assert BenchmarkProof.from_dict(p.to_dict()).hotkey == HONEST
    assert BenchmarkProof.from_dict(p.to_dict()).content_hash() == p.content_hash()


def test_a_proof_written_before_this_field_existed_is_readable_but_unbound():
    """Absent, not empty. It still fails verification — the point is only that
    reading an old artifact does not raise."""
    d = _proof(HONEST).to_dict()
    del d["hotkey"]
    assert BenchmarkProof.from_dict(d).hotkey == ""


def _verify(proof, expected_hotkey, mock=True):
    return verify_proof(
        proof,
        approved_measurements=set(),
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce,
        gold_answers={},
        expected_hotkey=expected_hotkey,
        mock=mock,
    )


def test_relaying_another_miners_proof_is_refused():
    """THE ATTACK. The relayer serves the honest miner's proof verbatim: it is
    genuinely attested, for a real epoch and slice, and its weights_hash matches
    a commitment the relayer can make. Only the hotkey says whose it is."""
    stolen = _proof(HONEST)
    result = _verify(stolen, expected_hotkey=RELAYER)
    assert not result.valid
    assert "names somebody else" in result.reason


def test_the_honest_miner_still_verifies_past_this_check():
    """The same proof, checked against the hotkey it was produced for, must not
    fail HERE — it fails later on hardware checks, which is a different reason."""
    honest = _proof(HONEST)
    result = _verify(honest, expected_hotkey=HONEST)
    assert "names somebody else" not in (result.reason or "")


def test_editing_the_hotkey_after_attestation_breaks_report_data():
    """The relayer's other move: rewrite the field to their own hotkey. The
    quote was signed over the old content_hash, so this is tamper-evident."""
    stolen = _proof(HONEST)
    stolen.hotkey = RELAYER
    result = _verify(stolen, expected_hotkey=RELAYER)
    assert not result.valid
    assert "report_data mismatch" in result.reason


def test_verifying_without_an_expected_hotkey_is_a_caller_error():
    """Skipping the check when the caller forgets would silently restore the
    property this removes, so it raises instead of passing."""
    with pytest.raises(ValueError, match="without expected_hotkey"):
        _verify(_proof(HONEST), expected_hotkey="", mock=False)


def test_an_unbound_proof_is_refused():
    """A miner running pre-binding code produces hotkey="" and must not verify
    merely because the field is falsy on both sides."""
    result = _verify(_proof(""), expected_hotkey=HONEST)
    assert not result.valid
    assert "(none)" in result.reason
