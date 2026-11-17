from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rgpot
from rgpot.lammps import FixExternal, box_rows, type_numbers, unit_factors


def image(ix: int, iy: int, iz: int) -> int:
    return (ix + 512) | ((iy + 512) << 10) | ((iz + 512) << 20)


class FakeLammps:
    def __init__(
        self,
        *,
        units="metal",
        tags=(1, 2),
        types=(1, 2),
        positions=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        images=None,
        periodic=(1, 1, 1),
        world_size=1,
    ):
        self.tags = np.asarray(tags, dtype=np.int64)
        self.positions = np.asarray(positions, dtype=float)
        self.types = np.asarray(types, dtype=np.int64)
        self.images = np.asarray(
            images if images is not None else [image(0, 0, 0)] * len(tags),
            dtype=np.int64,
        )
        self.units = units
        self.periodic = periodic
        self.world_size = world_size
        self.commands = []
        self.variables = {}
        self.energy = None
        self.virial = None
        self.callback = None
        self.numpy = SimpleNamespace(extract_atom=self._extract_atom)

    def _extract_atom(self, name):
        return {"type": self.types, "image": self.images}[name]

    def extract_global(self, name):
        return {"units": self.units, "ntypes": 2}[name]

    def extract_setting(self, name):
        return {
            "world_size": self.world_size,
            "IMGMASK": 1023,
            "IMGMAX": 512,
            "IMGBITS": 10,
            "IMG2BITS": 20,
        }[name]

    def extract_box(self):
        return (
            [1.0, 2.0, 3.0],
            [10.0, 10.5, 12.4],
            1.2,
            0.9,
            -0.7,
            self.periodic,
            0,
        )

    def command(self, command):
        self.commands.append(command)
        if command.startswith("run ") and self.variables.get("rgpot_failed"):
            raise RuntimeError("fix halt")

    def set_fix_external_callback(self, _fix_id, callback, _caller):
        self.callback = callback

    def set_internal_variable(self, name, value):
        self.variables[name] = value

    def fix_external_set_energy_global(self, _fix_id, energy):
        self.energy = energy

    def fix_external_set_virial_global(self, _fix_id, virial):
        self.virial = np.asarray(virial)


class RecordingPotential:
    def __init__(self, stress=None):
        self.calls = []
        self.reported_stress = stress

    def evaluate(self, positions, numbers, cell):
        self.calls.append((positions.copy(), numbers.copy(), cell.copy()))
        forces = np.column_stack((numbers, numbers + 1, numbers + 2)).astype(float)
        return {
            "energy": 2.5,
            "forces": forces,
            "stress": self.reported_stress,
        }


def test_units_and_elements_use_rgpot_conventions():
    assert unit_factors("metal") == (1.0, 1.0)
    assert unit_factors("real") == pytest.approx(
        (1.0, rgpot.unit_conversion_factor("eV", "kcal/mol"))
    )
    assert unit_factors("electron") == pytest.approx(
        (
            rgpot.unit_conversion_factor("angstrom", "bohr"),
            rgpot.unit_conversion_factor("eV", "hartree"),
        )
    )
    np.testing.assert_array_equal(type_numbers("O H", 2), [8, 1])
    np.testing.assert_array_equal(type_numbers([1, "Si"], 2), [1, 14])
    with pytest.raises(ValueError, match="names 1 atom types"):
        type_numbers(["H"], 2)
    with pytest.raises(ValueError, match="unknown element"):
        type_numbers(["Qq", "H"], 2)


def test_triclinic_cell_unwrapping_id_order_and_virial():
    rows = box_rows([1.0, 2.0, 3.0], [10.0, 10.5, 12.4], 1.2, 0.9, -0.7)
    expected_rows = np.array(
        [[9.0, 0.0, 0.0], [1.2, 8.5, 0.0], [-0.7, 0.9, 9.4]]
    )
    np.testing.assert_array_equal(rows, expected_rows)

    stress = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    pot = RecordingPotential(stress)
    lmp = FakeLammps(
        tags=(2, 1),
        types=(2, 1),
        positions=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        images=(image(1, -1, 0), image(0, 0, 1)),
    )
    driver = FixExternal(lmp, pot, "O H")
    forces = np.zeros((2, 3))
    driver._step(2, lmp.tags, lmp.positions, forces)

    received, numbers, cell = pot.calls[0]
    local_unwrapped = lmp.positions + np.array([[1.0, -1.0, 0.0], [0, 0, 1]]) @ rows
    np.testing.assert_allclose(received, local_unwrapped[[1, 0]])
    np.testing.assert_array_equal(numbers, [8, 1])
    np.testing.assert_array_equal(cell, expected_rows)
    np.testing.assert_array_equal(forces, [[1, 2, 3], [8, 9, 10]])

    expected = -abs(np.linalg.det(rows)) * 0.5 * (stress + stress.T)
    np.testing.assert_allclose(
        lmp.virial,
        [expected[0, 0], expected[1, 1], expected[2, 2],
         expected[0, 1], expected[0, 2], expected[1, 2]],
    )


@pytest.mark.parametrize("style", ["metal", "real", "electron"])
def test_lammps_unit_conversion(style):
    pot = RecordingPotential(np.eye(3))
    lmp = FakeLammps(units=style)
    driver = FixExternal(lmp, pot, "H He")
    forces = np.zeros((2, 3))
    driver._step(2, lmp.tags, lmp.positions, forces)

    length, energy = unit_factors(style)
    np.testing.assert_allclose(pot.calls[0][0], lmp.positions / length)
    assert lmp.energy == pytest.approx(2.5 * energy)
    np.testing.assert_allclose(
        forces,
        np.array([[1, 2, 3], [2, 3, 4]]) * energy / length,
    )


def test_periodic_box_without_stress_is_refused():
    class Harmonic:
        def __call__(self, positions, _numbers, _cell):
            return 0.5 * float(np.sum(positions**2)), -positions, 0.0

    lmp = FakeLammps()
    driver = FixExternal(lmp, Harmonic(), "H He")
    with pytest.raises(RuntimeError, match="Harmonic lacks analytic stress"):
        driver._step(2, lmp.tags, lmp.positions, np.zeros((2, 3)))


def test_nonperiodic_box_without_stress_sends_zero_virial():
    class Harmonic:
        def __call__(self, positions, _numbers, _cell):
            return 0.5 * float(np.sum(positions**2)), -positions, 0.0

    lmp = FakeLammps(periodic=(0, 0, 0))
    driver = FixExternal(lmp, Harmonic(), "H He")
    driver._step(2, lmp.tags, lmp.positions, np.zeros((2, 3)))
    np.testing.assert_array_equal(lmp.virial, np.zeros(6))


def test_finite_difference_virial_is_refused():
    with pytest.raises(ValueError, match="virial must be 'auto' or 'none'"):
        FixExternal(
            FakeLammps(),
            RecordingPotential(),
            "H He",
            virial="finite-difference",
        )


def test_empty_atom_list_is_refused():
    lmp = FakeLammps(tags=(), types=(), positions=np.zeros((0, 3)))
    driver = FixExternal(lmp, RecordingPotential(), "H He")
    with pytest.raises(RuntimeError, match="atom IDs 1..N"):
        driver._step(0, lmp.tags, lmp.positions, np.zeros((0, 3)))


class RootComm:
    def __init__(self, parts):
        self.parts = parts
        self.outcome = None
        self.shares = None

    def Get_size(self):
        return 2

    def Get_rank(self):
        return 0

    def gather(self, _part, root=0):
        assert root == 0
        return self.parts

    def bcast(self, value, root=0):
        assert root == 0
        self.outcome = value
        return value

    def scatter(self, shares, root=0):
        assert root == 0
        self.shares = shares
        return shares[0]


def test_duplicate_ids_are_refused():
    lmp = FakeLammps(tags=(1, 1), types=(1, 1))
    driver = FixExternal(lmp, RecordingPotential(), "H He")
    with pytest.raises(RuntimeError, match="atom IDs 1..N"):
        driver._step(2, lmp.tags, lmp.positions, np.zeros((2, 3)))


def test_rank_zero_reraises_the_original_error():
    parts = [
        (np.array([1]), np.zeros((1, 3)), np.array([1])),
        (np.array([2]), np.zeros((1, 3)), np.array([2])),
    ]
    comm = RootComm(parts)

    class Boom:
        def __call__(self, *_args):
            raise ValueError("evaluation failed")

    lmp = FakeLammps(
        tags=(1,),
        types=(1,),
        positions=((0.0, 0.0, 0.0),),
        world_size=2,
    )
    driver = FixExternal(lmp, Boom(), "H He", comm=comm)
    with pytest.raises(ValueError, match="evaluation failed"):
        driver._step(1, lmp.tags, lmp.positions, np.zeros((1, 3)))
    assert comm.outcome[0] == "evaluation failed"


def test_import_rgpot_does_not_import_lammps():
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    pythonpath = env.get("PYTHONPATH", "")
    extra = str(root / "python")
    env["PYTHONPATH"] = extra if not pythonpath else extra + os.pathsep + pythonpath
    script = (
        "import sys, rgpot; "
        "assert 'lammps' not in sys.modules, sorted(sys.modules); "
        "assert 'rgpot.lammps' not in sys.modules; "
        "import rgpot.lammps; "
        "assert 'lammps' not in sys.modules, sorted(sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr


def test_rank_zero_gathers_by_id_and_scatters_local_forces():
    rows = box_rows([1.0, 2.0, 3.0], [10.0, 10.5, 12.4], 1.2, 0.9, -0.7)
    parts = [
        (np.array([3, 1]), np.array([[3.0, 0, 0], [1.0, 0, 0]]), np.array([8, 1])),
        (np.array([4, 2]), np.array([[4.0, 0, 0], [2.0, 0, 0]]), np.array([14, 6])),
    ]
    comm = RootComm(parts)
    pot = RecordingPotential(np.zeros((3, 3)))
    lmp = FakeLammps(
        tags=(3, 1),
        types=(1, 2),
        positions=parts[0][1],
        world_size=2,
    )
    driver = FixExternal(lmp, pot, "O H", comm=comm)
    forces = np.zeros((2, 3))
    driver._step(2, lmp.tags, lmp.positions, forces)

    np.testing.assert_array_equal(pot.calls[0][1], [1, 6, 8, 14])
    np.testing.assert_array_equal(comm.shares[0], [[8, 9, 10], [1, 2, 3]])
    np.testing.assert_array_equal(comm.shares[1], [[14, 15, 16], [6, 7, 8]])
    np.testing.assert_array_equal(forces, comm.shares[0])
    np.testing.assert_array_equal(pot.calls[0][2], rows)


def test_callback_failure_stops_run_and_reraises_original_error():
    class Broken:
        def __call__(self, *_args):
            raise ValueError("evaluation failed")

    lmp = FakeLammps()
    driver = FixExternal(lmp, Broken(), "H He")
    forces = np.ones((2, 3))
    driver._callback(None, 0, 2, lmp.tags, lmp.positions, forces)
    np.testing.assert_array_equal(forces, 0.0)
    assert lmp.variables["rgpot_failed"] == 1.0
    with pytest.raises(ValueError, match="evaluation failed"):
        driver.run(5)
    assert driver.error is None
    assert lmp.variables["rgpot_failed"] == 0.0


def test_real_lammps_smoke():
    """Energy, forces, and virial from fix external match a direct LJ evaluation."""
    lammps = pytest.importorskip("lammps")
    positions = np.array([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]])
    cell = np.diag([20.0, 20.0, 20.0])
    pot = rgpot.LJPot(cutoff=8.0)
    direct = pot.evaluate(positions, np.array([1, 1], dtype=np.int32), cell)
    stress = np.asarray(direct["stress"], dtype=float).reshape(3, 3)
    virial = -abs(float(np.linalg.det(cell))) * stress
    expected = np.array(
        [
            virial[0, 0],
            virial[1, 1],
            virial[2, 2],
            virial[0, 1],
            virial[0, 2],
            virial[1, 2],
        ]
    )
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            """
            clear
            units metal
            atom_style atomic
            atom_modify map array sort 0 0
            boundary p p p
            region box block 0 20 0 20 0 20
            create_box 1 box
            create_atoms 1 single 1 1 1
            create_atoms 1 single 2.2 1 1
            mass 1 1
            compute vrg all pressure NULL virial
            """
        )
        with FixExternal(lmp, pot, "H") as driver:
            driver.run(0)
            pe = float(lmp.get_thermo("pe"))
            forces = np.array(lmp.gather_atoms("f", 1, 3), dtype=float).reshape(-1, 3)
            pressure = np.array(lmp.numpy.extract_compute("vrg", 0, 1), dtype=float)
            volume = float(lmp.get_thermo("vol"))
            nktv2p = float(lmp.extract_global("nktv2p"))
            assert driver.evaluations == 1
        assert pe == pytest.approx(direct["energy"])
        np.testing.assert_allclose(forces, direct["forces"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(pressure, dtype=float).reshape(-1)[:6] * volume / nktv2p,
            expected,
            atol=1e-12,
            rtol=1e-12,
        )
    finally:
        lmp.close()
