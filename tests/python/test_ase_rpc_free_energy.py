"""RgpotRpcCalculator reports free_energy, so force_consistent=True works."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from ase import Atoms


def test_rpc_force_consistent_energy_matches_energy() -> None:
    binary = os.environ.get("RGPOT_POTSERV_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("RGPOT_POTSERV_BIN is not set")
    pytest.importorskip("capnp")
    from rgpot.ase import RgpotRpcCalculator

    atoms = Atoms(
        "Cu2H2",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]],
        cell=[15.0, 15.0, 15.0],
        pbc=False,
    )
    calc = RgpotRpcCalculator.spawn(binary, potential="CuH2")
    try:
        assert "free_energy" in calc.implemented_properties
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        free_energy = atoms.get_potential_energy(force_consistent=True)
    finally:
        calc.close()
    assert free_energy == energy
