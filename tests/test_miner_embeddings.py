"""The miner's embedding cache: a bad cache is recomputed, never fatal.

Embedding the pool takes hours, so the miner caches it. Two failures found in
the neurons/ review (2026-11-08): a cache that could not be read, or that was
built for other text, raised instead of being recomputed (in a dstack TD that
is a crash loop the operator cannot clear: the file survives restarts and there
is no shell); and the cache path was keyed on question ids, so a prompt changed
under the same id kept the path and hit that crash on every miner.
"""
import numpy as np
import pytest

import neurons.miner as miner
from fugal_subnet import backbone
from fugal_subnet.vendor import success_contract as contract


def _pool(prompts):
    return [{"question_id": f"q{i}", "prompt": p} for i, p in enumerate(prompts)]


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FUGAL_EMBEDDING_CACHE", str(tmp_path))
    calls = []
    monkeypatch.setattr(backbone, "compute_hidden_states",
                        lambda qs, batch_size=None: (calls.append(len(qs)),
                                                     np.full((len(qs), 1024), 0.5, np.float32))[1])
    monkeypatch.setattr(backbone, "release_backbone", lambda: None)
    return tmp_path, calls


def test_a_good_cache_is_reused(cache_dir):
    _, calls = cache_dir
    pool = _pool(["a?", "b?"])
    first = miner._compute_hidden_states(pool)
    second = miner._compute_hidden_states(pool)
    assert calls == [2] and np.array_equal(first, second)


def test_an_unreadable_cache_is_set_aside_and_recomputed(cache_dir):
    _, calls = cache_dir
    pool = _pool(["a?", "b?"])
    path = miner._embedding_cache_path(pool)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not an npz")
    hidden = miner._compute_hidden_states(pool)
    assert calls == [2] and hidden.shape == (2, 1024)
    assert os.path.exists(path + ".bad")


def test_a_cache_built_for_other_text_is_recomputed(cache_dir):
    _, calls = cache_dir
    pool = _pool(["a?", "b?"])
    path = miner._embedding_cache_path(pool)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    contract.save_cache(tmp, ["other?", "text?"], np.ones((2, 1024), np.float32))
    os.replace(tmp, path)
    hidden = miner._compute_hidden_states(pool)
    assert calls == [2] and float(hidden[0, 0]) == 0.5


def test_the_cache_path_follows_the_text_not_the_ids(cache_dir):
    same_ids_new_prompt = _pool(["a?", "b, reworded?"])
    assert miner._embedding_cache_path(_pool(["a?", "b?"])) == miner._embedding_cache_path(_pool(["a?", "b?"]))
    assert miner._embedding_cache_path(_pool(["a?", "b?"])) != miner._embedding_cache_path(same_ids_new_prompt)
