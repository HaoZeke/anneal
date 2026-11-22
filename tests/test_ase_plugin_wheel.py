"""The installed wheel exposes its ASE 3.29 calculator plugins."""

from importlib.metadata import entry_points

import pytest


def test_installed_wheel_exposes_ase_plugins() -> None:
    plugins = pytest.importorskip("ase._4.plugins")
    from ase.calculators.calculator import get_calculator_class

    import rgpot.ase as ase_module

    matches = [
        entry_point
        for entry_point in entry_points(group="ase.plugins")
        if entry_point.name == "rgpot"
    ]
    assert len(matches) == 1
    assert matches[0].value == "rgpot.ase"
    assert matches[0].load() is ase_module

    by_name = {
        plugin.name: plugin
        for plugin in plugins.PluginCollection.from_entry_points().calculators
    }
    assert by_name["rgpot"]() is ase_module.RgpotCalculator
    assert by_name["rgpot_rpc"]() is ase_module.RgpotRpcCalculator
    assert get_calculator_class("rgpot") is ase_module.RgpotCalculator
    assert get_calculator_class("rgpot_rpc") is ase_module.RgpotRpcCalculator
