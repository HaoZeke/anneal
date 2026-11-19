"""In-process Python surface for tabulated EAM potentials."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase import Atoms

import rgpot
from rgpot.ase import RgpotCalculator, available_kinds


ROOT = Path(__file__).resolve().parents[2]
CU_FUNCFL = ROOT / "CppCore" / "tests" / "data" / "eam_table" / "Cu_u3.eam"
CU_REFERENCE = (
    ROOT / "CppCore" / "tests" / "data" / "eam_table" / "cu_u3_cubic.ref"
)

needs_eam_table = pytest.mark.skipif(
    getattr(rgpot, "has_eam_table", True) is False,
    reason="rgpot was built without with_eam_table",
)


def cu_cell():
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3))
    rng = np.random.default_rng(20261119)
    atoms.positions[:] += rng.normal(0.0, 0.04, size=atoms.positions.shape)
    return atoms


def cu_reference() -> tuple[Atoms, float, np.ndarray]:
    lines = CU_REFERENCE.read_text().splitlines()
    box = np.asarray([float(value) for value in lines[1].split()[1:]]).reshape(3, 3)
    energy = float(lines[2].split()[1])
    count = int(lines[4].split()[1])
    rows = np.asarray(
        [[float(value) for value in line.split()] for line in lines[5 : 5 + count]]
    )
    atoms = Atoms(
        numbers=rows[:, 0].astype(np.int64),
        positions=rows[:, 1:4],
        cell=box,
        pbc=True,
    )
    return atoms, energy, rows[:, 4:7]


@needs_eam_table
def test_funcfl_matches_lammps_reference_energy_and_forces() -> None:
    atoms, energy, forces = cu_reference()
    atoms.calc = RgpotCalculator(
        "eam_table",
        file=str(CU_FUNCFL),
        format="funcfl",
    )
    assert rgpot.EAMTablePot is not None
    assert rgpot.has_eam_table
    assert "eamtable" in available_kinds()
    assert atoms.calc.potential.cutoff > 0.0
    assert atoms.get_potential_energy() == pytest.approx(energy, rel=2e-12, abs=2e-12)
    np.testing.assert_allclose(
        atoms.get_forces(),
        forces,
        rtol=2e-12,
        atol=2e-12,
    )


@needs_eam_table
def test_one_calculator_evaluates_on_real_os_threads() -> None:
    atoms = cu_cell()
    calculator = RgpotCalculator(
        "eam_table",
        file=str(CU_FUNCFL),
        format="funcfl",
    )

    def evaluate(_worker: int) -> tuple[float, bytes]:
        row = calculator.__rg_evaluate__(
            atoms.positions,
            atoms.numbers,
            atoms.cell.array,
            atoms.pbc,
        )
        return float(row["energy"]), np.asarray(row["forces"]).tobytes()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate, range(8)))
    assert len({energy for energy, _forces in results}) == 1
    assert len({forces for _energy, forces in results}) == 1


@needs_eam_table
@pytest.mark.parametrize(
    ("table", "format"),
    [
        (CU_FUNCFL, "lammps"),
        (CU_FUNCFL.with_name("no_such_table.eam"), "funcfl"),
        (CU_FUNCFL.with_name("malformed_funcfl.eam"), "funcfl"),
        (CU_FUNCFL, "setfl"),
    ],
)
def test_a_bad_table_or_format_is_a_value_error(table: Path, format: str) -> None:
    with pytest.raises(ValueError):
        rgpot.EAMTablePot(str(table), format)
