# Copyright 2023--present rgpot developers

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).parents[1]


class ForceInputMessage:
    hasOrbitalKey = False
    orbitalKey = 0


class ForceInputFactory:
    @staticmethod
    def new_message() -> ForceInputMessage:
        return ForceInputMessage()


class Schema:
    ForceInput = ForceInputFactory


SCHEMA = Schema()


def _rpc_calculator_class():
    rgpot_package = types.ModuleType("rgpot")
    rgpot_package.__path__ = [str(ROOT / "python/rgpot")]
    ase_package = types.ModuleType("rgpot.ase")
    ase_package.__path__ = [str(ROOT / "python/rgpot/ase")]
    sys.modules.setdefault("rgpot", rgpot_package)
    sys.modules.setdefault("rgpot.ase", ase_package)
    for name in ("claim", "rpc"):
        qualified = f"rgpot.ase.{name}"
        spec = importlib.util.spec_from_file_location(
            qualified, ROOT / f"python/rgpot/ase/{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return sys.modules["rgpot.ase.rpc"].RgpotRpcCalculator


class KeyedPotential:
    def __init__(self) -> None:
        self.generations: dict[int, int] = {}
        self.keys: list[int] = []

    async def calculate(self, fip):
        assert fip.hasOrbitalKey
        key = int(fip.orbitalKey)
        self.keys.append(key)
        generation = self.generations.get(key, 0) + 1
        self.generations[key] = generation

        class PotentialResult:
            energy = float(generation)
            forces = [0.0, 0.0, 0.0]

        class Reply:
            result = PotentialResult()

        return Reply()


def test_python_rpc_orbital_keys_restore_per_image_state() -> None:
    RgpotRpcCalculator = _rpc_calculator_class()
    calculator = object.__new__(RgpotRpcCalculator)
    calculator._schema = SCHEMA
    calculator._pot = KeyedPotential()

    async def run() -> list[float]:
        energies = []
        for key in (7, 19, 7):
            energy, _ = await calculator._calculate_async(
                np.zeros((1, 3)), [8], np.eye(3), key
            )
            energies.append(energy)
        return energies

    assert asyncio.run(run()) == [1.0, 1.0, 2.0]
    assert calculator._pot.keys == [7, 19, 7]
