"""The backbone thread count is a miner-side throughput knob, not consensus.

Pinning the backbone to one thread cost a 4-vCPU TD ~13 hours of startup for
no consensus benefit: validators never run the backbone, and
`scripts/check_determinism.py --perturb` already asserts the scoring path is
identical with thread pins removed. These tests pin the knob's semantics — the
default is unchanged, 0 means every core, garbage means 1 — and that numpy's
OpenBLAS stays single-threaded whatever the knob says, because the miner's
routing arithmetic runs there and gains nothing from parallelism.
"""
import os

import pytest

from fugal_subnet import determinism


@pytest.mark.parametrize("raw,expected", [
    (None, 1), ("", 1), ("1", 1), ("3", 3), ("abc", 1), ("-2", None), ("0", None),
])
def test_backbone_threads_semantics(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("FUGAL_BACKBONE_THREADS", raising=False)
    else:
        monkeypatch.setenv("FUGAL_BACKBONE_THREADS", raw)
    n = determinism.backbone_threads()
    if expected is None:          # "all cores"
        assert n == max(1, os.cpu_count() or 1)
    else:
        assert n == expected


def test_openblas_stays_single_threaded_whatever_torch_gets():
    env = determinism.determinism_env(8)
    assert env["OMP_NUM_THREADS"] == "8" and env["MKL_NUM_THREADS"] == "8"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    # The kernel-dispatch pins are untouched by the thread knob.
    assert env["ATEN_CPU_CAPABILITY"] == "avx2" and env["OPENBLAS_CORETYPE"] == "Haswell"


def test_default_env_is_the_historical_single_thread_pin(monkeypatch):
    monkeypatch.delenv("FUGAL_BACKBONE_THREADS", raising=False)
    env = determinism.determinism_env(determinism.backbone_threads())
    assert env["OMP_NUM_THREADS"] == "1" and env["MKL_NUM_THREADS"] == "1"
