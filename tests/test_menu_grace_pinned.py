"""The menu B grace length is consensus: on finney it is compiled into the
release and a variable may only repeat it; elsewhere the variable sets it; it
must be a positive number of epochs (v0.6.0 review, finding 2)."""
import pytest

from fugal_subnet import config

FINNEY = config.FINNEY_GENESIS_HASH
OTHER = "0x" + "ab" * 32


def test_finney_uses_the_compiled_grace(monkeypatch):
    monkeypatch.setattr(config, "MENU_GRACE_EPOCHS_ENV", None)
    assert config.menu_grace_epochs(FINNEY) == config.FINNEY_MENU_GRACE_EPOCHS == 20


def test_finney_accepts_a_variable_that_repeats_the_compiled_grace(monkeypatch):
    monkeypatch.setattr(config, "MENU_GRACE_EPOCHS_ENV", 20)
    assert config.menu_grace_epochs(FINNEY) == 20


@pytest.mark.parametrize("value", [0, 5, 21])
def test_finney_refuses_a_different_grace(monkeypatch, value):
    monkeypatch.setattr(config, "MENU_GRACE_EPOCHS_ENV", value)
    with pytest.raises(ValueError):
        config.menu_grace_epochs(FINNEY)


def test_an_unreadable_genesis_is_treated_as_finney(monkeypatch):
    monkeypatch.setattr(config, "MENU_GRACE_EPOCHS_ENV", 3)
    with pytest.raises(ValueError):
        config.menu_grace_epochs(None)


def test_another_chain_reads_the_variable(monkeypatch):
    monkeypatch.setattr(config, "MENU_GRACE_EPOCHS_ENV", 3)
    assert config.menu_grace_epochs(OTHER) == 3
    monkeypatch.setattr(config, "MENU_GRACE_EPOCHS_ENV", None)
    assert config.menu_grace_epochs(OTHER) == config.FINNEY_MENU_GRACE_EPOCHS


@pytest.mark.parametrize("value", [0, -1])
def test_the_grace_must_be_positive_on_any_chain(monkeypatch, value):
    monkeypatch.setattr(config, "MENU_GRACE_EPOCHS_ENV", value)
    with pytest.raises(ValueError):
        config.menu_grace_epochs(OTHER)


def test_the_validator_reads_the_resolved_grace_not_the_raw_variable():
    import ast
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    assert "MENU_GRACE_EPOCHS)" not in src.replace("menu_grace_epochs(", "")
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert "MENU_GRACE_EPOCHS" not in names
