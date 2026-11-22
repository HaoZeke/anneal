"""ASE registry and experimental plugin discovery for rgpot calculators."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from ase.calculators.calculator import get_calculator_class, names

import rgpot.ase as ase_module
from rgpot.ase import RgpotCalculator, RgpotRpcCalculator
from rgpot.ase._registry import register_calculator


def test_calculators_are_registered_with_ase() -> None:
    assert get_calculator_class("rgpot") is RgpotCalculator
    assert get_calculator_class("rgpot_rpc") is RgpotRpcCalculator
    assert names.count("rgpot") == 1
    assert names.count("rgpot_rpc") == 1


def _run_isolated(code: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=os.environ.copy() if env is None else env,
    )


def _require_ase_plugins() -> None:
    try:
        import ase._4.plugins  # noqa: F401
    except ImportError:
        pytest.skip("ASE does not provide the experimental plugin API")


def _plugin_entry_env(tmp_path: Path, dist_name: str) -> dict[str, str]:
    """Point a fresh interpreter at an ``ase.plugins`` entry for rgpot.ase."""
    dist_info = tmp_path / f"{dist_name}-0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: 0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        f"[ase.plugins]\n{dist_name} = rgpot.ase\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env


def test_registry_registration_is_idempotent() -> None:
    for _ in range(2):
        register_calculator("rgpot", RgpotCalculator)
        register_calculator("rgpot_rpc", RgpotRpcCalculator)
    assert get_calculator_class("rgpot") is RgpotCalculator
    assert get_calculator_class("rgpot_rpc") is RgpotRpcCalculator
    assert names.count("rgpot") == 1
    assert names.count("rgpot_rpc") == 1


def test_register_calculator_replaces_stale_class() -> None:
    # ASE sorts its global name list in place when it registers a name, so
    # the registry is only safe to mutate from one thread.
    _run_isolated(
        """
from ase.calculators.calculator import (
    Calculator,
    external_calculators,
    get_calculator_class,
    names,
)

from rgpot.ase._registry import register_calculator


class First(Calculator):
    pass


class Second(Calculator):
    pass


register_calculator("rgpot_stale_check", First)
register_calculator("rgpot_stale_check", First)
assert get_calculator_class("rgpot_stale_check") is First
assert names.count("rgpot_stale_check") == 1
register_calculator("rgpot_stale_check", Second)
assert get_calculator_class("rgpot_stale_check") is Second
assert external_calculators["rgpot_stale_check"] is Second
assert names.count("rgpot_stale_check") == 1

register_calculator("lj", First)
assert get_calculator_class("lj") is First
assert names.count("lj") == 1
"""
    )


def test_module_reload_reruns_registration_and_replaces_stale_classes() -> None:
    _run_isolated(
        """
import importlib

from ase.calculators.calculator import get_calculator_class, names

import rgpot.ase.calculator as calculator_module
import rgpot.ase.rpc as rpc_module

old_calculator = calculator_module.RgpotCalculator
old_rpc = rpc_module.RgpotRpcCalculator
assert get_calculator_class("rgpot") is old_calculator
assert get_calculator_class("rgpot_rpc") is old_rpc

for _ in range(2):
    importlib.reload(calculator_module)
    importlib.reload(rpc_module)

new_calculator = calculator_module.RgpotCalculator
new_rpc = rpc_module.RgpotRpcCalculator
assert new_calculator is not old_calculator
assert new_rpc is not old_rpc
assert get_calculator_class("rgpot") is new_calculator
assert get_calculator_class("rgpot_rpc") is new_rpc
assert names.count("rgpot") == 1
assert names.count("rgpot_rpc") == 1
"""
    )


def test_lazy_calculator_plugins_resolve() -> None:
    try:
        from ase._4.plugins import PluginCollection
    except ImportError:
        pytest.skip("ASE does not provide the experimental plugin API")

    plugins = ase_module.__ase_plugins__
    assert ase_module.__ase_plugins__ is plugins
    by_name = {
        plugin.name: plugin for plugin in PluginCollection(plugins).calculators
    }
    assert by_name["rgpot"]() is RgpotCalculator
    assert by_name["rgpot_rpc"]() is RgpotRpcCalculator


def test_ase_entry_point_scan_import_order(tmp_path: Path) -> None:
    _require_ase_plugins()
    env = _plugin_entry_env(tmp_path, "rgpot-import-order-test")
    code = """
from ase._4.plugins import plugins
import rgpot.ase as rgpot_ase

by_name = {plugin.name: plugin for plugin in plugins.calculators}
assert by_name["rgpot"]() is rgpot_ase.RgpotCalculator
assert by_name["rgpot_rpc"]() is rgpot_ase.RgpotRpcCalculator
assert rgpot_ase.__ase_plugins__ is rgpot_ase.__ase_plugins__
"""
    _run_isolated(code, env)


def test_plugins_keep_module_when_read_before_plugin_api(
    tmp_path: Path,
) -> None:
    _require_ase_plugins()
    env = _plugin_entry_env(tmp_path, "rgpot-plugin-reentry-test")
    code = """
import sys

assert "ase._4.plugins" not in sys.modules
import rgpot.ase as rgpot_ase

assert "ase._4.plugins" not in sys.modules
plugins = rgpot_ase.__ase_plugins__
assert rgpot_ase.__ase_plugins__ is plugins
ours = [
    plugin for plugin in plugins if plugin.name in {"rgpot", "rgpot_rpc"}
]
assert {plugin.name for plugin in ours} == {"rgpot", "rgpot_rpc"}
assert all(plugin.module is not None for plugin in ours)
"""
    _run_isolated(code, env)


def test_concurrent_plugin_reads_return_the_same_object(tmp_path: Path) -> None:
    _require_ase_plugins()
    env = _plugin_entry_env(tmp_path, "rgpot-plugin-threads-test")
    code = """
import sys
import threading

assert "ase._4.plugins" not in sys.modules
import rgpot.ase as rgpot_ase

assert "ase._4.plugins" not in sys.modules
count = 16
barrier = threading.Barrier(count, timeout=60)
found = []
errors = []
lock = threading.Lock()


def read():
    try:
        barrier.wait()
        plugins = rgpot_ase.__ase_plugins__
        with lock:
            found.append(plugins)
    except Exception as exc:
        with lock:
            errors.append(exc)


threads = [threading.Thread(target=read) for _ in range(count)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

assert not errors, errors
assert len(found) == count
assert all(plugins is found[0] for plugins in found)
"""
    _run_isolated(code, env)
