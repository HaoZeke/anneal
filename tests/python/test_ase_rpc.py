# MIT License
# Copyright 2023--present rgpot developers
"""ASE parity tests for the Cap'n Proto calculator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase.calculators.calculator import PropertyNotImplementedError
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

from rgpot.ase import RgpotCalculator, RgpotRpcCalculator
from support import lj_crystal, lj_dimer, si_crystal

import rgpot


def _server_binary() -> str:
    binary = os.environ.get("RGPOT_POTSERV_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("RGPOT_POTSERV_BIN is not set")
    return binary


def _triclinic_lj() -> Any:
    atoms = lj_crystal()
    atoms.set_cell(
        [[35.0, 0.0, 0.0], [3.0, 34.0, 0.0], [2.0, -1.0, 33.0]],
        scale_atoms=False,
    )
    return atoms


@pytest.mark.parametrize(
    ("server_kind", "local_potential", "geometry"),
    [
        ("LJ", "lj", _triclinic_lj),
        ("SWSi", rgpot.StillingerWeberSi, si_crystal),
    ],
)
def test_rpc_matches_in_process_on_triclinic_cell(
    server_kind: str, local_potential: Any, geometry: Any
) -> None:
    atoms = geometry()
    local = atoms.copy()
    potential = local_potential() if callable(local_potential) else local_potential
    local.calc = RgpotCalculator(potential)
    local.calc.calculate(local, properties=["energy", "forces", "stress"])

    remote = atoms.copy()
    with RgpotRpcCalculator.spawn(_server_binary(), potential=server_kind) as rpc:
        assert rpc.implemented_properties == [
            "energy",
            "free_energy",
            "forces",
            "stress",
        ]
        remote.calc = rpc
        rpc.calculate(remote, properties=["energy", "forces", "stress"])

        assert rpc.results["free_energy"] == rpc.results["energy"]
        np.testing.assert_allclose(
            rpc.results["energy"],
            local.calc.results["energy"],
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rpc.results["forces"],
            local.calc.results["forces"],
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rpc.results["stress"],
            local.calc.results["stress"],
            rtol=1e-10,
            atol=1e-12,
        )


def test_rpc_frechet_cell_filter_matches_in_process() -> None:
    def relax(calc: Any) -> np.ndarray:
        atoms = si_crystal()
        atoms.calc = calc
        BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-6, steps=3)
        return atoms.cell.array.copy()

    local_cell = relax(RgpotCalculator(rgpot.StillingerWeberSi()))
    with RgpotRpcCalculator.spawn(_server_binary(), potential="SWSi") as rpc:
        remote_cell = relax(rpc)
    np.testing.assert_allclose(remote_cell, local_cell, rtol=1e-10, atol=1e-12)


def test_rpc_stress_rejects_nonperiodic_cell() -> None:
    atoms = lj_dimer()
    with RgpotRpcCalculator.spawn(_server_binary(), potential="LJ") as rpc:
        atoms.calc = rpc
        with pytest.raises(
            PropertyNotImplementedError,
            match="fully periodic",
        ):
            atoms.get_stress()


def test_rpc_stress_reflects_server_capability() -> None:
    atoms = _triclinic_lj()
    with RgpotRpcCalculator.spawn(_server_binary(), potential="LJCluster") as rpc:
        assert rpc.implemented_properties == ["energy", "free_energy", "forces"]
        atoms.calc = rpc
        with pytest.raises(
            PropertyNotImplementedError,
            match="does not provide stress",
        ):
            atoms.get_stress()
