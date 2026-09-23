from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from aegis.config import Settings
from aegis.node import EdgeNode


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: run an async test")


def pytest_pyfunc_call(pyfuncitem):
    """Minimal async runner so the suite needs no pytest-asyncio."""
    func = pyfuncitem.obj
    if inspect.iscoroutinefunction(func):
        kwargs = {k: v for k, v in pyfuncitem.funcargs.items()
                  if k in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(func(**kwargs))
        return True
    return None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.policy_file = "config/policy.yaml"
    s.sync.interval_s = 0.05
    s.sync.probe_interval_s = 0.05
    # The dimensionality comes from the weights, not from configuration —
    # EdgeNode overwrites it at construction from the model bundle.
    return s


@pytest.fixture
def node(settings: Settings) -> EdgeNode:
    return EdgeNode(settings)
