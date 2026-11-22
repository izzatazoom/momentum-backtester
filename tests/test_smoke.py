"""Smoke tests: the package and every subpackage import, and the CLI runs."""

import importlib

import pytest

import momentum
from momentum.cli import main

SUBPACKAGES = ["data", "signals", "portfolio", "backtest", "metrics", "reports"]


def test_version_is_set() -> None:
    assert momentum.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"momentum.{name}")
    assert module.__doc__, f"momentum.{name} should have a docstring"


def test_cli_runs_without_arguments() -> None:
    assert main([]) == 0
