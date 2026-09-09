"""Explicit synthetic fixtures for offline tests and mocked rehearsals only.

Never imported by the measured miner or validator execution paths.
"""
from __future__ import annotations

import io

import numpy as np

from fugal_subnet.vendor import success_contract as c


def manifest(models):
    return {"version": "fugal-token-statistics-v1", "profile_id": c.PROFILE_ID,
            "status": "synthetic-test-only", "sources": ["explicit synthetic fixture"],
            "worker_profile": "mocked calls; no inference",
            "models": [{"id": m, "samples": 1, "mean_in_tokens": 500., "mean_out_tokens": 300.} for m in models]}


def arrays(W, b, models):
    models = list(models)
    return c.make_head(W, b, models, manifest(models), "synthetic-test-only")


def head_bytes(W, b, models):
    buf = io.BytesIO()
    np.savez(buf, **arrays(W, b, models))
    return buf.getvalue()


def costs(head):
    c.check_manifest(head.success, manifest(head.models))
    return np.arange(1, len(head.models) + 1, dtype=float) * .001
