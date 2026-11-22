"""Protocol v4: one way to build an epoch's plan, and the v4 mix wherever a
number is weighted to the scored mix (docs/decisions/LIVE_MENU_V4.md, 10).

- `epoch_plan.plan_v4(pool)` is the plan the TD runs and every verifier
  checks against: `verify.expected_v4` takes the plan (its pool index, its
  targets, its `measurement`), not an index and targets of the caller's.
- A TD session built on a pool builds its plan with plan_v4.
- The derivation step's production parameters weight token means and the
  proposal gates to the pinned v4 mix (`mix_v4().menu_mix()`), and the seed's
  token means are weighted to it too.
"""
from __future__ import annotations

import inspect

import pytest

from fugal_subnet import config, epoch_plan
from fugal_subnet import menu_state as ms
from fugal_subnet.attacks import run_tee_attacks as suite
from fugal_subnet.tee import td_v4
from fugal_subnet.tee import verify as verify_mod


def test_the_expectation_takes_the_epochs_plan_and_nothing_else():
    params = inspect.signature(verify_mod.expected_v4).parameters
    assert "plan" in params and "index" not in params and "targets" not in params
    w = suite._v4_world()
    plan = epoch_plan.plan_v4(w["pool"])
    x = verify_mod.expected_v4(w["state"], epoch=suite.V4_EPOCH, nonce=w["nonce"],
                               pool=w["pool"], scored_ids=w["slice"],
                               hotkey=suite.HONEST_HOTKEY, plan=plan)
    assert x.exploration == td_v4.exploration_v4(w["nonce"], w["pool"], set(w["slice"]),
                                                 w["state"], suite.HONEST_HOTKEY)
    assert dict(plan.targets) == dict(epoch_plan.mix_v4().targets)


def test_a_td_session_on_a_pool_builds_its_plan_with_plan_v4():
    w = suite._v4_world()
    s = td_v4.V4Session(first_epoch=90, pool=w["pool"], fetch_listing=lambda m: b"",
                        read_commitment=lambda e: None, menu_bytes=lambda h: None,
                        list_bytes=lambda s, h: None)
    assert dict(s.plan.targets) == dict(epoch_plan.mix_v4().targets)
    with pytest.raises(ValueError):
        td_v4.V4Session(first_epoch=90, fetch_listing=lambda m: b"",
                        read_commitment=lambda e: None, menu_bytes=lambda h: None,
                        list_bytes=lambda s, h: None)


def test_the_production_step_parameters_are_weighted_to_the_v4_mix():
    p = epoch_plan.menu_step_params_v4(5000)
    assert isinstance(p, ms.MenuParams)
    assert p.mix == epoch_plan.mix_v4().menu_mix() and p.min_category_accuracy_bp == 5000
    assert p.refresh_epochs == ms.MENU_REFRESH_EPOCHS


def test_the_seed_token_means_are_weighted_to_the_v4_mix():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_menu_v4_seed.py"
    spec = importlib.util.spec_from_file_location("seed_builder", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.scored_mix() == epoch_plan.mix_v4().menu_mix()
    assert builder.scored_mix() != tuple(sorted(
        (c, round(w * 100)) for c, w in config.CATEGORY_TARGETS_V3.items()))


# The validator's one plan, its v4 step parameters and the v4 mix's categories
# and targets are exercised by running the validator loop under protocol 4
# (tests/test_v4_loop.py), which replaced a search of the loop's source text.
