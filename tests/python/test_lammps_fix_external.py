# Copyright 2023--present rgpot developers

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import threading
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
        self.callback = {}
        self.fail_command = None
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
        if command == self.fail_command:
            raise RuntimeError(f"failed command: {command}")
        if command.startswith("run ") and self.variables.get("rgpot_failed"):
            raise RuntimeError("fix halt")

    def set_fix_external_callback(self, fix_id, callback, _caller):
        self.callback[fix_id] = callback

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


def test_tag_order_cache_is_reused_and_invalidated(monkeypatch):
    calls = 0
    original_sort = np.sort

    def counted_sort(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sort(*args, **kwargs)

    monkeypatch.setattr(np, "sort", counted_sort)
    lmp = FakeLammps()
    driver = FixExternal(lmp, RecordingPotential(np.eye(3)), "H He")
    forces = np.zeros((2, 3))
    driver._step(2, lmp.tags, lmp.positions, forces)
    driver._step(2, lmp.tags, lmp.positions, forces)
    assert calls == 0

    lmp.tags = lmp.tags[::-1].copy()
    driver._step(2, lmp.tags, lmp.positions, forces)
    assert calls == 1


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

    def allreduce(self, value):
        return value + 1

    def gather(self, part, root=0):
        assert root == 0
        if not isinstance(part, tuple):
            return [part, None]
        return self.parts

    def bcast(self, value, root=0):
        assert root == 0
        self.outcome = value
        return value

    def scatter(self, shares, root=0):
        assert root == 0
        self.shares = shares
        return shares[0]


class CollectiveWorld:
    def __init__(self, size=2):
        self.size = size
        self.condition = threading.Condition()
        self.calls = {}

    def exchange(self, index, rank, kind, value, root):
        with self.condition:
            call = self.calls.setdefault(
                index,
                {"kind": kind, "root": root, "values": {}, "reads": 0},
            )
            assert call["kind"] == kind
            assert call["root"] == root
            call["values"][rank] = value
            self.condition.notify_all()
            ready = self.condition.wait_for(
                lambda: len(call["values"]) == self.size,
                timeout=2,
            )
            if not ready:
                raise TimeoutError(f"{kind} collective did not complete")
            values = call["values"]
            if kind == "allreduce":
                result = sum(values.values())
            elif kind == "gather":
                result = [values[item] for item in range(self.size)] if rank == root else None
            elif kind == "bcast":
                result = values[root]
            elif kind == "scatter":
                result = values[root][rank]
            else:
                raise AssertionError(kind)
            call["reads"] += 1
            if call["reads"] == self.size:
                del self.calls[index]
            return result


class ThreadComm:
    def __init__(self, world, rank):
        self.world = world
        self.rank = rank
        self.index = 0

    def _exchange(self, kind, value, root=0):
        index, self.index = self.index, self.index + 1
        return self.world.exchange(index, self.rank, kind, value, root)

    def Get_size(self):
        return self.world.size

    def Get_rank(self):
        return self.rank

    def allreduce(self, value):
        return self._exchange("allreduce", value)

    def gather(self, value, root=0):
        return self._exchange("gather", value, root)

    def bcast(self, value, root=0):
        return self._exchange("bcast", value, root)

    def scatter(self, value, root=0):
        return self._exchange("scatter", value, root)


def run_rank(driver, lmp):
    forces = np.zeros((len(lmp.tags), 3))
    driver._step(len(lmp.tags), lmp.tags, lmp.positions, forces)
    return forces


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
    with pytest.raises(
        RuntimeError,
        match="rank 0 failed while evaluating the potential: "
        "ValueError: evaluation failed",
    ):
        driver._step(1, lmp.tags, lmp.positions, np.zeros((1, 3)))
    assert comm.outcome == (
        "rank 0 failed while evaluating the potential: "
        "ValueError: evaluation failed"
    )


@pytest.mark.timeout(5)
def test_non_root_rank_runs_collectives_and_receives_forces():
    world = CollectiveWorld()
    lmps = [
        FakeLammps(
            tags=(1,),
            types=(1,),
            positions=((1.0, 0.0, 0.0),),
            world_size=2,
        ),
        FakeLammps(
            tags=(2,),
            types=(2,),
            positions=((2.0, 0.0, 0.0),),
            world_size=2,
        ),
    ]
    potential = RecordingPotential(np.zeros((3, 3)))
    drivers = [
        FixExternal(lmps[rank], potential, "H He", comm=ThreadComm(world, rank))
        for rank in range(2)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_rank, drivers[rank], lmps[rank]) for rank in range(2)
        ]
        forces = [future.result(timeout=3) for future in futures]
    np.testing.assert_array_equal(forces[0], [[1, 2, 3]])
    np.testing.assert_array_equal(forces[1], [[2, 3, 4]])


@pytest.mark.timeout(5)
def test_non_root_pre_collective_failure_reaches_every_rank():
    world = CollectiveWorld()
    lmps = [
        FakeLammps(
            tags=(1,),
            types=(1,),
            positions=((1.0, 0.0, 0.0),),
            world_size=2,
        ),
        FakeLammps(
            tags=(2,),
            types=(3,),
            positions=((2.0, 0.0, 0.0),),
            world_size=2,
        ),
    ]
    drivers = [
        FixExternal(
            lmps[rank],
            RecordingPotential(np.zeros((3, 3))),
            "H He",
            comm=ThreadComm(world, rank),
        )
        for rank in range(2)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_rank, drivers[rank], lmps[rank]) for rank in range(2)
        ]
        errors = []
        for future in futures:
            with pytest.raises(RuntimeError) as caught:
                future.result(timeout=3)
            errors.append(str(caught.value))
    assert errors == [errors[0], errors[0]]
    assert errors[0].startswith(
        "rank 1 failed while preparing local atoms: IndexError:"
    )


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


def test_callback_catches_base_exception_until_run_returns():
    class Interrupted:
        def __call__(self, *_args):
            raise KeyboardInterrupt("stopped")

    lmp = FakeLammps()
    driver = FixExternal(lmp, Interrupted(), "H He")
    forces = np.ones((2, 3))
    driver._callback(None, 0, 2, lmp.tags, lmp.positions, forces)
    np.testing.assert_array_equal(forces, 0.0)
    with pytest.raises(KeyboardInterrupt, match="stopped"):
        driver.run(1)


def test_root_evaluation_catches_base_exception_until_run_returns():
    parts = [
        (np.array([1]), np.zeros((1, 3)), np.array([1])),
        (np.array([2]), np.zeros((1, 3)), np.array([2])),
    ]

    class Interrupted:
        def __call__(self, *_args):
            raise KeyboardInterrupt("root stopped")

    lmp = FakeLammps(
        tags=(1,),
        types=(1,),
        positions=((0.0, 0.0, 0.0),),
        world_size=2,
    )
    driver = FixExternal(lmp, Interrupted(), "H He", comm=RootComm(parts))
    forces = np.ones((1, 3))
    driver._callback(None, 0, 1, lmp.tags, lmp.positions, forces)
    np.testing.assert_array_equal(forces, 0.0)
    with pytest.raises(
        RuntimeError,
        match="rank 0 failed while evaluating the potential: "
        "KeyboardInterrupt: root stopped",
    ):
        driver.run(1)


def test_create_false_close_removes_callback_and_prevents_later_run():
    lmp = FakeLammps()
    driver = FixExternal(
        lmp,
        RecordingPotential(np.eye(3)),
        "H He",
        create=False,
    )
    assert "fix rgpot all external pf/callback 1 1" not in lmp.commands
    assert "rgpot" in lmp.callback

    driver.close()
    assert "unfix rgpot_halt" in lmp.commands
    assert "unfix rgpot" in lmp.commands
    assert "rgpot" not in lmp.callback
    forces = np.ones((2, 3))
    for callback in lmp.callback.values():
        callback(None, 0, 2, lmp.tags, lmp.positions, forces)
    np.testing.assert_array_equal(forces, 1.0)
    with pytest.raises(RuntimeError, match="FixExternal is closed"):
        driver.run(1)

    commands = list(lmp.commands)
    driver.close()
    assert lmp.commands == commands


def test_close_preserves_state_and_reraises_unfix_failure():
    lmp = FakeLammps()
    driver = FixExternal(lmp, RecordingPotential(np.eye(3)), "H He")
    lmp.fail_command = "unfix rgpot"
    with pytest.raises(RuntimeError, match="failed command: unfix rgpot"):
        driver.close()
    assert driver._lmp is lmp
    assert "rgpot" in lmp.callback

    lmp.fail_command = None
    driver.close()
    assert lmp.commands.count("unfix rgpot_halt") == 1
    assert "rgpot" not in lmp.callback
    assert driver._lmp is None


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


def test_ids_in_order_reach_the_potential_without_a_permutation():
    pot = RecordingPotential(np.zeros((3, 3)))
    lmp = FakeLammps(tags=(1, 2), types=(2, 1))
    driver = FixExternal(lmp, pot, "O H")
    forces = np.zeros((2, 3))
    driver._step(2, lmp.tags, lmp.positions, forces)

    received, numbers, _ = pot.calls[0]
    np.testing.assert_array_equal(received, lmp.positions)
    np.testing.assert_array_equal(numbers, [1, 8])
    np.testing.assert_array_equal(forces, [[1, 2, 3], [8, 9, 10]])


def test_reversed_ids_are_permuted_both_ways():
    pot = RecordingPotential(np.zeros((3, 3)))
    lmp = FakeLammps(
        tags=(3, 2, 1),
        types=(1, 2, 1),
        positions=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    )
    driver = FixExternal(lmp, pot, "O H")
    forces = np.zeros((3, 3))
    driver._step(3, lmp.tags, lmp.positions, forces)

    received, numbers, _ = pot.calls[0]
    np.testing.assert_array_equal(received[:, 0], [3.0, 2.0, 1.0])
    np.testing.assert_array_equal(numbers, [8, 1, 8])
    np.testing.assert_array_equal(forces, [[8, 9, 10], [1, 2, 3], [8, 9, 10]])


@pytest.mark.parametrize("tags", [(1, 3), (2, 3), (0, 1)])
def test_ids_that_are_not_one_to_n_are_refused(tags):
    lmp = FakeLammps(tags=tags, types=(1, 1))
    driver = FixExternal(lmp, RecordingPotential(np.zeros((3, 3))), "H He")
    with pytest.raises(RuntimeError, match="atom IDs 1..N"):
        driver._step(2, lmp.tags, lmp.positions, np.zeros((2, 3)))


def test_a_potential_that_edits_its_inputs_leaves_lammps_untouched():
    class Greedy:
        def evaluate(self, positions, numbers, cell):
            positions += 1.0
            numbers += 1
            cell += 1.0
            return {"energy": 0.0, "forces": np.zeros_like(positions), "stress": np.eye(3)}

    lmp = FakeLammps(tags=(1, 2))
    before = lmp.positions.copy()
    driver = FixExternal(lmp, Greedy(), "H He")
    driver._step(2, lmp.tags, lmp.positions, np.zeros((2, 3)))
    np.testing.assert_array_equal(lmp.positions, before)
    box = box_rows([1.0, 2.0, 3.0], [10.0, 10.5, 12.4], 1.2, 0.9, -0.7)
    np.testing.assert_allclose(lmp.virial[:3], -abs(np.linalg.det(box)) * np.ones(3))
