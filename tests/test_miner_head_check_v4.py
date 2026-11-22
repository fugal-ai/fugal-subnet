"""The miner's startup head check under protocol v4 is the v4 verdict
(head_eval.v4_head_verdict against the menu it starts from: the pinned seed),
not v3's token-manifest check (routing_protocol.benchmark_costs_for)."""
from __future__ import annotations

import click
import pytest

from fugal_subnet import menu_state as ms
from tests.test_v4_seat_odds import _head


def test_a_v4_head_without_token_means_passes_the_v4_startup_check():
    from neurons import miner

    seed = ms.load_seed()
    head = _head(list(seed.ids())[:5])
    miner._startup_head_check(head, 4, mock=False, menu_b_block=None)
    with pytest.raises(click.ClickException, match="no row"):
        miner._startup_head_check(_head(["lab-z/not-on-the-menu"]), 4, mock=False,
                                  menu_b_block=None)
    with pytest.raises(click.ClickException):
        miner._startup_head_check(_head(list(seed.ids())[:5], scale=1e6), 4, mock=False,
                                  menu_b_block=None)
    # Under v3 the same head (no token means) is refused, as before.
    with pytest.raises(Exception):
        miner._startup_head_check(head, 3, mock=True, menu_b_block=None)


def test_main_calls_the_startup_check():
    import ast
    import inspect

    from neurons import miner

    calls = [n for n in ast.walk(ast.parse(inspect.getsource(miner.main.callback)))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_startup_head_check"]
    assert len(calls) == 1


def test_a_v4_head_file_loads_under_v4_and_main_passes_the_protocol(tmp_path):
    """--head-path is read before the startup check: under v4 by the v4 loader
    (token means optional), under v1-v3 by load_head_from_npz as before. The
    v1-v3 loader alone refused every v4 head without token means, so a v4
    miner started from a head file exited at once (the local-chain v4
    rehearsal, scripts/dress_rehearsal_v4.py, found it)."""
    import ast
    import inspect

    from neurons import miner

    head = _head(list(ms.load_seed().ids())[:5])
    path = tmp_path / "head.npz"
    path.write_bytes(head)
    assert miner._load_head_file(str(path), 4) == head
    with pytest.raises(click.ClickException, match="Head failed validation"):
        miner._load_head_file(str(path), 3)
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(miner.main.callback)))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_load_head_file"]
    assert calls and all(len(c.args) == 2 and getattr(c.args[1], "id", None) == "protocol"
                         for c in calls)
