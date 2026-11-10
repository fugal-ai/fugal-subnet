"""Shared test setup.

Importing `bittensor` (which `neurons.miner` and `neurons.validator` do at
module scope) runs a `logging.config.dictConfig` with
`disable_existing_loggers` left True, so every logger created before it —
including `fugal_subnet.*` module loggers — is silently disabled for the rest
of the process (fugal_subnet/logging_setup.py explains the production side).
In the test session that meant any caplog-based test that happened to run
after a test importing a neuron saw nothing and failed, depending on which
files were selected: `tests/test_validator_chain_guards.py` and
`tests/test_pool_manifest.py::test_an_absent_manifest_is_never_silent` failed
in subsets and passed in the full run. The neurons re-enable the loggers at
startup with `configure_logging`; the tests do the same before each test.
"""
from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _fugal_loggers_enabled():
    # Undo the ways a previous test can have silenced the package's loggers
    # for everyone after it: a global logging.disable(), a disabled logger
    # (dictConfig with disable_existing_loggers), propagation switched off, or
    # a logger LEVEL left raised — measured: after tests/test_additional_findings.py
    # the `fugal.validator` logger sat at CRITICAL (50), so every later caplog
    # assertion on a WARNING or ERROR from it saw nothing. caplog is a handler
    # on the root logger; a record must get past its own logger first.
    logging.disable(logging.NOTSET)
    for name, obj in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(obj, logging.Logger) and name.split(".", 1)[0] in ("fugal", "fugal_subnet"):
            obj.disabled = False
            obj.propagate = True
            obj.setLevel(logging.NOTSET)
    yield
