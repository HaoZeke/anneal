"""ASE-shaped Anneal: DummyAtoms uses ASE force sign. Optional real ASE."""

from __future__ import annotations

import numpy as np
import pytest

from anneal.ase import Anneal


class DummyAtoms:
    """Two particles, harmonic pair. ``get_forces`` is ASE minus-grad."""

    def __init__(self, d0: float = 1.4, frozen=None):
        self.positions = np.array([[0.0, 0.0, 0.0], [d0, 0.0, 0.0]], dtype=np.float64)
        self.calc = object()
        self.frozen = None if frozen is None else list(frozen)
        self.constraints = list(self.frozen or [])

    def __len__(self):
        return 2

    def get_positions(self):
        return self.positions.copy()

    def set_positions(self, pos):
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        if self.frozen:
            pos = pos.copy()
            for idx in self.frozen:
                pos[idx] = self.positions[idx]
        self.positions = pos

    def get_potential_energy(self):
        d = float(np.linalg.norm(self.positions[0] - self.positions[1]))
        return (d - 0.74) ** 2

    def get_forces(self):
        delta = self.positions[0] - self.positions[1]
        d = float(np.linalg.norm(delta))
        if d < 1e-12:
            return np.zeros((2, 3), dtype=np.float64)
        g = 2.0 * (d - 0.74) * (delta / d)
        forces = np.vstack([-g, g])
        if self.frozen:
            for idx in self.frozen:
                forces[idx] = 0.0
        return forces


class DummyPbc(DummyAtoms):
    pbc = np.array([True, True, True])


class DummyCellFilter:
    """N+3 rows, like FrechetCellFilter."""

    def __init__(self):
        self.atoms = DummyAtoms()
        extra = np.zeros((3, 3), dtype=np.float64)
        self.positions = np.vstack([self.atoms.positions, extra])
        self.calc = object()

    def __len__(self):
        return 5

    def get_positions(self):
        return self.positions.copy()

    def set_positions(self, pos):
        self.positions = np.asarray(pos, dtype=np.float64).reshape(-1, 3)

    def get_potential_energy(self):
        return self.atoms.get_potential_energy()

    def get_forces(self):
        base = self.atoms.get_forces()
        return np.vstack([base, np.zeros((3, 3))])


def test_swap_is_bfgs_shaped():
    atoms = DummyAtoms(1.4)
    opt = Anneal(atoms, logfile=None)
    assert opt.mode == "local"
    ok = opt.run(fmax=0.05, steps=20)
    assert ok
    d = float(np.linalg.norm(atoms.positions[0] - atoms.positions[1]))
    assert d == pytest.approx(0.74, abs=0.05)
    assert opt.get_number_of_steps() >= 1


def test_constructor_keeps_named_ase_kwargs():
    atoms = DummyAtoms(1.1)
    opt = Anneal(
        atoms,
        logfile=None,
        trajectory=None,
        restart=None,
        master=True,
        append_trajectory=True,
        force_consistent=False,
        mode="local",
    )
    assert opt.append_trajectory is True
    assert opt.force_consistent is False
    assert opt.run(fmax=0.1, steps=10)


def test_unknown_kwargs_and_mode_raise():
    atoms = DummyAtoms(1.1)
    with pytest.raises(TypeError):
        Anneal(atoms, logfile=None, not_a_kw=1)
    with pytest.raises(ValueError, match="mode"):
        Anneal(atoms, logfile=None, mode="hop")


def test_lazy_export():
    import anneal

    cls = anneal.Anneal
    assert cls is Anneal


def test_attach_fires():
    atoms = DummyAtoms(1.4)
    opt = Anneal(atoms, logfile=None)
    calls = []
    opt.attach(lambda: calls.append(opt.get_number_of_steps()))
    assert opt.run(fmax=0.05, steps=20)
    assert any(n >= 1 for n in calls)


def test_already_converged_does_not_search(monkeypatch):
    atoms = DummyAtoms(0.74)
    hops = []

    def boom(*_a, **_k):
        hops.append(1)
        raise AssertionError("search must not run")

    monkeypatch.setattr("anneal.ase.cluster_search", boom)
    opt = Anneal(atoms, logfile=None, mode="search", budget=10)
    assert opt.run(fmax=0.05, steps=5)
    assert hops == []
    assert opt.get_number_of_steps() == 0


def test_auto_two_atoms_does_not_hop(monkeypatch):
    atoms = DummyAtoms(1.4)
    hops = []

    def fake(*_a, **_k):
        hops.append(1)
        return {"best": atoms.get_positions().reshape(-1), "hops": 1}

    monkeypatch.setattr("anneal.ase.cluster_search", fake)
    opt = Anneal(atoms, logfile=None, mode="auto", budget=10)
    opt.run(fmax=0.05, steps=5)
    assert hops == []


def test_search_passes_start(monkeypatch):
    atoms = DummyAtoms(1.4)
    want = atoms.get_positions().reshape(-1).copy()
    seen = {}

    def fake(*_a, **kwargs):
        seen.update(kwargs)
        return {"best": atoms.get_positions().reshape(-1), "hops": 2}

    monkeypatch.setattr("anneal.ase.cluster_search", fake)
    opt = Anneal(atoms, logfile=None, mode="search", budget=10)
    opt.step()
    assert np.allclose(seen["start"], want)
    assert seen["length_scale"] == pytest.approx(1.4)
    assert opt.get_number_of_steps() == 1


def test_trajectory_write_is_not_swallowed(monkeypatch, tmp_path):
    atoms = DummyAtoms(1.4)
    written = []

    def fake_write(path, _atoms, append=False):
        written.append((str(path), bool(append)))

    monkeypatch.setattr("ase.io.write", fake_write, raising=False)
    import types
    import sys

    ase_io = types.ModuleType("ase.io")
    ase_io.write = fake_write
    sys.modules.setdefault("ase", types.ModuleType("ase"))
    sys.modules["ase.io"] = ase_io
    dest = tmp_path / "opt.xyz"
    opt = Anneal(atoms, logfile=None, trajectory=str(dest))
    opt.run(fmax=0.05, steps=2)
    assert written
    assert written[0][0] == str(dest)


def test_run_returns_false_when_unconverged():
    atoms = DummyAtoms(1.4)
    opt = Anneal(atoms, logfile=None, mode="local")
    ok = opt.run(fmax=1e-20, steps=1)
    assert ok is False


def test_pbc_and_cell_filter_refuse_hop():
    with pytest.raises(ValueError, match="periodic"):
        Anneal(DummyPbc(1.4), logfile=None, mode="search", budget=10).step()
    with pytest.raises(ValueError, match="cell filter"):
        Anneal(DummyCellFilter(), logfile=None, mode="search", budget=10).step()
    with pytest.raises(ValueError, match="constrained"):
        Anneal(DummyAtoms(1.4, frozen=[0]), logfile=None, mode="search", budget=10).step()


def test_fixatoms_keeps_frozen_start():
    atoms = DummyAtoms(1.4, frozen=[0])
    frozen = atoms.positions[0].copy()
    opt = Anneal(atoms, logfile=None, mode="local")
    opt.run(fmax=0.05, steps=20)
    assert np.allclose(atoms.positions[0], frozen)


def test_ase_atoms_need_calculator():
    ase = pytest.importorskip("ase")
    atoms = ase.Atoms("Cu2", positions=[[0, 0, 0], [1.4, 0, 0]])
    with pytest.raises(ValueError, match="calculator"):
        Anneal(atoms, logfile=None)


def test_ase_emt_local_descends():
    ase = pytest.importorskip("ase")
    emt = pytest.importorskip("ase.calculators.emt")
    atoms = ase.Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]])
    atoms.calc = emt.EMT()
    e0 = float(atoms.get_potential_energy())
    opt = Anneal(atoms, logfile=None, mode="local")
    opt.run(fmax=0.1, steps=30)
    assert float(atoms.get_potential_energy()) <= e0
    assert opt.get_number_of_steps() >= 1
