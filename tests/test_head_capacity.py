"""HEAD_MAX_MODELS is a capacity bound, and capacity is a security parameter.

The cap on head rows is not an interface convenience. A routing head is a
linear map, and the thing that stops a miner MEMORISING the question pool
instead of learning to route is that a linear head does not have enough
parameters to do it. Measured on isotropic Gaussian embeddings — the most
favourable geometry there is, so an upper bound on any real pool:

    questions   random-label fit
        2,000   100%      <- total memorisation
        4,000    79%
        8,000    35%
       21,717    16.9%    <- the real pool

Architectural, not under-training: four optimiser settings up to 10,000 epochs
return 0.269 to three decimals.

Two consequences follow, and both are guarded here because neither is visible
from reading the constant.
"""
import io

import numpy as np
import pytest

from fugal_subnet.config import HEAD_HIDDEN_DIM, HEAD_MAX_MODELS
from fugal_subnet.head_eval import load_head_from_npz


def _head(models):
    n = len(models)
    buf = io.BytesIO()
    np.savez(
        buf,
        W=np.zeros((n, HEAD_HIDDEN_DIM), dtype=np.float32),
        b=np.zeros(n, dtype=np.float32),
        models=np.array(models),
    )
    return buf.getvalue()


def test_rows_must_name_distinct_models():
    """The row cap bounds capacity; duplicates spend it without declaring it.

    Before this check, 64 rows all naming one model loaded cleanly — verified.
    That is 64 parameter vectors wearing the costume of a one-model head, and
    it measurably buys the adversary capacity: random-label fit on the real
    pool size rises from 26.9% with distinct models to 44.1% at 32 rows over
    8 names. Not a lookup table, and not what 64 was chosen to permit.
    """
    with pytest.raises(ValueError, match="distinct"):
        load_head_from_npz(_head(["openai/gpt-4o-mini"] * HEAD_MAX_MODELS))


def test_even_one_duplicated_pair_is_rejected():
    """No threshold. Two rows naming one model are two ways to say one route —
    a reparameterisation with no routing content — so nothing honest produces
    it and there is no size at which it becomes legitimate."""
    with pytest.raises(ValueError, match="distinct"):
        load_head_from_npz(_head(["v/a", "v/b", "v/a"]))


def test_an_honest_head_is_unaffected():
    """The tightening must not cost a real miner anything."""
    head = load_head_from_npz(_head([f"vendor/m{i}" for i in range(13)]))
    assert len(head.models) == 13
    assert len(set(head.models)) == 13
