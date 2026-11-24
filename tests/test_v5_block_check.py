"""The validator's v5 activation-block check counts epochs in blocks, not
seconds: finney's v5 block 9,587,880 is a boundary of 360-block epochs (4320 s),
and is not a multiple of 4320."""
from fugal_subnet.benchmarks.slicer import blocks_per_epoch, validate_activation_block


def test_finney_v5_block_is_a_boundary_in_blocks():
    from fugal_subnet import config
    block = config.FINNEY_V5_ACTIVATION_BLOCK
    validate_activation_block(block, blocks_per_epoch(4320))
    assert block % 4320 != 0  # the seconds-based check this replaces would refuse it


def test_validator_does_not_check_the_v5_block_against_seconds():
    src = open("neurons/validator.py", encoding="utf-8").read()
    assert "v5_block % EPOCH_INTERVAL" not in src
