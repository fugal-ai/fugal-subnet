"""The validator refuses, before touching the chain, a configuration that would
make it score differently from its peers or score everyone zero.

These rules lived inline in neurons/validator.py's main(), where no unit test
reached them (the 2026-11-08 coverage map: main at 0%). They are one function
now, _check_live_configuration, and the v2 token manifest is checked too: only
v1's was, so a candidate manifest from a re-measure would have been scored.
"""
import json

import click
import pytest

from fugal_subnet import routing_protocol
from neurons.validator import _check_live_configuration as check

ENTRY = "a" * 64 + ":" + "b" * 64
CATS = {"mmlu": "knowledge"}
BPE_INTERVAL = 4320          # 360 blocks


def test_live_without_approved_entries_is_refused():
    with pytest.raises(click.ClickException, match="requires FUGAL_TEE_MEASUREMENTS"):
        check(True, [], None, BPE_INTERVAL, CATS)
    check(False, [], None, BPE_INTERVAL, CATS)            # mock may run without


def test_a_bare_base_is_refused_live_and_allowed_mock():
    with pytest.raises(click.ClickException, match="bare base"):
        check(True, ["a" * 64], None, BPE_INTERVAL, CATS)
    check(False, ["a" * 64], None, BPE_INTERVAL, CATS)


def test_an_activation_block_off_the_epoch_boundary_is_refused():
    check(True, [ENTRY], 360 * 100, BPE_INTERVAL, CATS)
    with pytest.raises(click.ClickException, match="FUGAL_V2_ACTIVATION_BLOCK"):
        check(True, [ENTRY], 360 * 100 + 7, BPE_INTERVAL, CATS)


def test_v2_without_its_category_map_is_refused_live():
    with pytest.raises(click.ClickException, match="BENCHMARK_CATEGORY_V2"):
        check(True, [ENTRY], 36000, BPE_INTERVAL, None)
    check(False, [ENTRY], 36000, BPE_INTERVAL, None)       # mock warns


def test_a_candidate_v2_token_manifest_is_refused_once_v2_is_scheduled(tmp_path, monkeypatch):
    candidate = json.loads(routing_protocol.manifest_path(2).read_text())
    candidate["status"] = "candidate"
    path = tmp_path / "tokens_v2.json"
    path.write_text(json.dumps(candidate))
    real = routing_protocol.manifest_path
    monkeypatch.setattr(routing_protocol, "manifest_path",
                        lambda v: path if v == 2 else real(v))
    check(True, [ENTRY], None, BPE_INTERVAL, CATS)          # v2 not scheduled: not read
    with pytest.raises(click.ClickException, match="protocol v2 token manifest is not deployable"):
        check(True, [ENTRY], 36000, BPE_INTERVAL, CATS)
    check(False, [ENTRY], 36000, BPE_INTERVAL, CATS)        # mock does not require reviewed
