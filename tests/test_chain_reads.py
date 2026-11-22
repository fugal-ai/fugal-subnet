"""Reads pinned to a block fall back to the archive endpoint when the node no
longer keeps that block's state (chain_reads.pinned). No network."""
import logging

import pytest

from fugal_subnet import chain_proposals, chain_reads, commitments, config

FINNEY_WORDING = ("Client error: UnknownBlock: State already discarded for "
                  "0xe474852b2c8f7d3cb687eec2171e11ce390e05ccdfc754a3c707f6ae567eaa20")
ZERO = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"


class Chain:
    def __init__(self, network="finney", fail=None, rows=(("5Hk", "a" * 64),), owner="5Own"):
        self.network, self.fail, self.rows, self.owner, self.calls = network, fail, rows, owner, []

    def query_map(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        return iter([(hk, {"block": 7, "info": data}) for hk, data in self.rows])

    def get_subnet_owner_hotkey(self, netuid, block=None):
        self.calls.append(("owner", block))
        if self.fail is not None:
            raise self.fail
        return self.owner


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(chain_reads, "_connections", {})
    monkeypatch.setattr(config, "ARCHIVE_ENDPOINTS", [])


def _archives(monkeypatch, *chains):
    urls = [f"wss://archive{i}.example:443/key-{i}" for i in range(len(chains))]
    for url, chain in zip(urls, chains):
        chain_reads._connections[url] = chain
    monkeypatch.setattr(config, "ARCHIVE_ENDPOINTS", urls)


def test_a_discarded_state_commitments_read_is_served_by_the_archive(monkeypatch):
    archive = Chain()
    _archives(monkeypatch, archive)
    node = Chain(fail=RuntimeError(FINNEY_WORDING))
    got = commitments.get_commitments_with_blocks(node, 70, block=9_543_240)
    assert list(got) == ["5Hk"] and got["5Hk"][1] == 7
    want = {"module": "Commitments", "name": "CommitmentOf", "params": [70], "block": 9_543_240}
    assert node.calls == archive.calls == [want]


def test_the_join_read_of_commitment_fields_falls_back_too(monkeypatch):
    archive = Chain()
    _archives(monkeypatch, archive)
    got = chain_proposals.read_fields_at(Chain(fail=RuntimeError(FINNEY_WORDING)), 70, 1000)
    assert "5Hk" in got and archive.calls


def test_archives_are_tried_in_order_and_a_total_failure_is_an_error_naming_hosts_only(
        monkeypatch, caplog):
    down, up = Chain(fail=ConnectionError("refused")), Chain()
    _archives(monkeypatch, down, up)
    got = commitments.get_commitments_with_blocks(Chain(fail=RuntimeError(FINNEY_WORDING)), 70,
                                                  block=5)
    assert "5Hk" in got and down.calls and up.calls
    _archives(monkeypatch, Chain(fail=ConnectionError("refused")))
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="discarded"):
        commitments.get_commitments_with_blocks(Chain(fail=RuntimeError(FINNEY_WORDING)), 70,
                                                block=5)
    line = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "block 5" in line and "wss://archive0.example" in line and "key-" not in line


def test_head_reads_and_other_failures_never_fall_back(monkeypatch):
    archive = Chain()
    _archives(monkeypatch, archive)
    with pytest.raises(RuntimeError, match="timeout"):
        commitments.get_commitments_with_blocks(Chain(fail=RuntimeError("timeout")), 70, block=5)
    with pytest.raises(RuntimeError, match="discarded"):
        commitments.get_commitments_with_blocks(Chain(fail=RuntimeError(FINNEY_WORDING)), 70)
    assert archive.calls == []


def test_off_finney_with_no_archive_configured_the_error_stands():
    node = Chain(network="ws://127.0.0.1:9944", fail=RuntimeError(FINNEY_WORDING))
    with pytest.raises(RuntimeError, match="discarded"):
        commitments.get_commitments_with_blocks(node, 70, block=5)


def test_a_finney_connection_uses_the_public_archive_when_none_is_set(monkeypatch):
    from fugal_subnet.validator_setup import FINNEY_DEFAULT_ARCHIVE

    archive = Chain()
    chain_reads._connections[FINNEY_DEFAULT_ARCHIVE] = archive
    got = commitments.get_commitments_with_blocks(Chain(fail=RuntimeError(FINNEY_WORDING)), 70,
                                                  block=5)
    assert "5Hk" in got and archive.calls


def test_an_owner_read_answering_the_zero_account_is_read_from_the_archive(monkeypatch):
    from neurons import validator

    archive = Chain(owner="5Own")
    _archives(monkeypatch, archive)

    class Mg:
        hotkeys = ["5Other", "5Own"]

    assert validator.resolve_burn_uid(Chain(owner=ZERO), 70, 5, Mg(), live=True) == 1
    assert archive.calls == [("owner", 5)]
