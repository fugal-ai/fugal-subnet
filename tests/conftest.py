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

import functools
import importlib.abc
import logging
import pathlib
import re
import sys

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


# --- markers CI selects by (.github/workflows/ci.yml) -------------------------
#
# Applied here, from each test module's source, rather than by hand, so a new
# test that touches the verifier or reads a doc is selected without anyone
# remembering to mark it.
#
# `tee`: the module touches the DCAP verifier. The `tests` shards install only
# --extra dev, where dcap_qvl is absent and verify_dcap raises ImportError; the
# `tee-verifier` job installs --extra tee and runs `-m tee` so the same tests
# take the present-verifier path too. A module counts if its source names
# dcap_qvl or verify_dcap, or it is listed in _TEE_EXTRA (a module that reaches
# dcap_qvl only through production code). _TEE_EXTRA is empty today: a run of
# the whole suite without the verifier, with the check below, found no such
# test. The check keeps it that way (see _tee_marker_is_complete).
#
# `docs`: the module reads a Markdown file. CI skips the shards on a
# Markdown-only pull request but still runs `-m docs`, so a doc edit cannot
# break a test that reads it. A module counts if its source has a string
# literal ending in .md or the literal "docs".
#
# `xdist_group`: the module has a module- or session-scoped fixture. Under
# pytest-xdist each worker builds such a fixture for itself, and some of these
# build whole multi-validator worlds (tests/test_v4_published_loop.py's took
# ~75 s, four times over, once per worker). CI runs `--dist loadgroup`, which
# keeps a group on one worker, so the world is built once.
_TEE_SOURCE = re.compile(r"\bdcap_qvl\b|\bverify_dcap\b")
_TEE_EXTRA: frozenset[str] = frozenset()
_DOCS_SOURCE = re.compile(r"""\.md["']|["']docs["']""")
_SHARED_FIXTURE = re.compile(r"""scope=["'](module|session)["']""")


@functools.lru_cache(maxsize=None)
def _module_markers(path: str) -> tuple[pytest.MarkDecorator, ...]:
    source = pathlib.Path(path).read_text(encoding="utf-8")
    name = pathlib.Path(path).name
    marks = []
    if _TEE_SOURCE.search(source) or name in _TEE_EXTRA:
        marks.append(pytest.mark.tee)
    if _DOCS_SOURCE.search(source):
        marks.append(pytest.mark.docs)
    if _SHARED_FIXTURE.search(source):
        marks.append(pytest.mark.xdist_group(name))
    return tuple(marks)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tee: depends on the DCAP verifier; CI reruns it with --extra tee")
    config.addinivalue_line(
        "markers", "docs: reads a Markdown file; CI runs it on Markdown-only changes")


@pytest.hookimpl(tryfirst=True)      # before xdist reads the groups
def pytest_collection_modifyitems(config, items):
    for item in items:
        for mark in _module_markers(str(item.path)):
            item.add_marker(mark)


# The `tee` marker is only as good as its completeness: a test that reaches
# dcap_qvl through production code without naming it would run in the shards
# (verifier absent) and never with the verifier. So, where dcap_qvl is NOT
# installed (CI's shards), every attempt to import it is noticed, and a test
# that makes one without the `tee` marker fails, naming the fix. Where it is
# installed the import succeeds once and is cached, so nothing is noticed
# there, which is fine: the shards are where the gap would be.
class _DcapImportWatch(importlib.abc.MetaPathFinder):
    attempted = False

    def find_spec(self, name, path=None, target=None):
        if name == "dcap_qvl" or name.startswith("dcap_qvl."):
            type(self).attempted = True
        return None


sys.meta_path.insert(0, _DcapImportWatch())


@pytest.fixture(autouse=True)
def _tee_marker_is_complete(request):
    _DcapImportWatch.attempted = False
    yield
    if _DcapImportWatch.attempted and request.node.get_closest_marker("tee") is None:
        pytest.fail(
            f"{request.node.nodeid} tried to import dcap_qvl but is not marked `tee`, so "
            "CI would never run it with the DCAP verifier installed. Add its module to "
            "_TEE_EXTRA in tests/conftest.py.", pytrace=False)
