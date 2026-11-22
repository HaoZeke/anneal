"""ASE registry and experimental plugin discovery for rgpot calculators."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ase.calculators.calculator import get_calculator_class, names

import rgpot.ase as ase_module
from rgpot.ase import RgpotCalculator, RgpotRpcCalculator


def test_calculators_are_registered_with_ase() -> None:
    assert get_calculator_class("rgpot") is RgpotCalculator
    assert get_calculator_class("rgpot_rpc") is RgpotRpcCalculator
    assert names.count("rgpot") == 1
    assert names.count("rgpot_rpc") == 1


def test_registry_registration_is_idempotent() -> None:
    importlib.reload(ase_module)
    importlib.reload(ase_module)
    assert get_calculator_class("rgpot") is RgpotCalculator
    assert get_calculator_class("rgpot_rpc") is RgpotRpcCalculator
    assert names.count("rgpot") == 1
    assert names.count("rgpot_rpc") == 1


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
    try:
        import ase._4.plugins  # noqa: F401
    except ImportError:
        pytest.skip("ASE does not provide the experimental plugin API")

    dist_info = tmp_path / "rgpot_import_order_test-0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rgpot-import-order-test\nVersion: 0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[ase.plugins]\nrgpot-import-order-test = rgpot.ase\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    code = """
from ase._4.plugins import plugins
import rgpot.ase as rgpot_ase

by_name = {plugin.name: plugin for plugin in plugins.calculators}
assert by_name["rgpot"]() is rgpot_ase.RgpotCalculator
assert by_name["rgpot_rpc"]() is rgpot_ase.RgpotRpcCalculator
assert rgpot_ase.__ase_plugins__ is rgpot_ase.__ase_plugins__
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
