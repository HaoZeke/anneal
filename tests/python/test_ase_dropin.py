"""RgpotCalculator as a drop-in ASE calculator.

Each section is one part of the ASE calculator contract: properties and
units, caching, finite differences, the standard ASE drivers, and
agreement with ASE's own LennardJones, MorsePotential and EAM.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk, fcc111
from ase.calculators.calculator import Calculator
from ase.calculators.fd import (
    calculate_numerical_forces,
    calculate_numerical_stress,
)
from ase.calculators.lj import LennardJones
from ase.calculators.mixing import SumCalculator
from ase.calculators.morse import MorsePotential
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.md.verlet import VelocityVerlet
from ase.mep import NEB
from ase.optimize import BFGS, FIRE
from ase.vibrations import Vibrations

import rgpot
from rgpot.ase import RgpotCalculator, available_kinds

from eam_al_reference import ase_eam_al
from support import pt7

needs_fortran = pytest.mark.skipif(
    not rgpot.has_fortran_pots, reason="this build has no Fortran potentials"
)

# Argon (Rowley, Nicholson and Parsonage) and copper (Girifalco and Weizer)
# parameters; any values would do.
AR = dict(u0=0.0104, psi=3.40, cutoff=8.5)
CU = dict(De=0.3429, a=1.3588, re=2.866)
RCUT1, RCUT2 = 1.9, 2.7


def ase_lj() -> LennardJones:
    return LennardJones(epsilon=AR["u0"], sigma=AR["psi"], rc=AR["cutoff"], smooth=False)


def rgpot_lj(**extra) -> RgpotCalculator:
    return RgpotCalculator("lj", **AR, **extra)


def ase_morse() -> MorsePotential:
    return MorsePotential(
        epsilon=CU["De"], rho0=CU["a"] * CU["re"], r0=CU["re"], rcut1=RCUT1, rcut2=RCUT2
    )


def rgpot_morse() -> RgpotCalculator:
    # ASE's fcut and rgpot's quintic switch are the same smoothstep, over
    # [rcut1 * r0, rcut2 * r0].
    return RgpotCalculator(
        "morse",
        **CU,
        cutoff=RCUT2 * CU["re"],
        switch_width=(RCUT2 - RCUT1) * CU["re"],
    )


def argon(repeat: int = 2, seed: int = 1) -> Atoms:
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(repeat)
    atoms.rattle(0.05, seed=seed)
    return atoms


def sheared(atoms: Atoms) -> Atoms:
    cell = atoms.cell.array.copy()
    side = cell[0, 0]
    cell[1, 0] = 0.08 * side
    cell[2, 1] = -0.06 * side
    cell[2, 0] = 0.04 * side
    atoms.set_cell(cell, scale_atoms=True)
    return atoms


def cluster(symbol: str, a: float, lattice: str, radius: float, seed: int = 4) -> Atoms:
    """Atoms of a bulk crystal within radius of the origin, rattled, pbc False."""
    crystal = bulk(symbol, lattice, a=a, cubic=True).repeat(4)
    crystal.positions -= crystal.positions.mean(axis=0)
    keep = np.linalg.norm(crystal.positions, axis=1) < radius
    atoms = Atoms(
        symbols=[s for s, k in zip(crystal.get_chemical_symbols(), keep) if k],
        positions=crystal.positions[keep],
        cell=[30.0, 30.0, 30.0],
        pbc=False,
    )
    atoms.rattle(0.04, seed=seed)
    return atoms


def relative(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b))) / scale


class CountingPotential:
    """A bound potential that counts kernel calls."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def caps(self):
        return self.inner.caps()

    def config(self):
        return self.inner.config()

    def evaluate(self, positions, numbers, box):
        self.calls += 1
        return self.inner.evaluate(positions, numbers, box)


# --- properties and units ----------------------------------------------------


def test_is_an_ase_calculator() -> None:
    calc = rgpot_lj()
    assert isinstance(calc, Calculator)
    for name in ("energy", "free_energy", "forces", "stress", "energies", "stresses"):
        assert name in calc.implemented_properties


@needs_fortran
def test_properties_follow_the_kernel() -> None:
    assert RgpotCalculator("sw").implemented_properties == [
        "energy",
        "free_energy",
        "forces",
        "stress",
    ]
    eam = RgpotCalculator("eam_al")
    assert eam.implemented_properties == ["energy", "free_energy", "forces"]
    assert "stress" in RgpotCalculator("eam_al", numerical_stress=True).implemented_properties


def test_compressed_crystal_has_negative_stress() -> None:
    """ASE sign: sigma = (1/V) dE/dstrain, negative when the crystal pushes out."""
    atoms = bulk("Ar", "fcc", a=4.6, cubic=True).repeat(2)
    atoms.calc = rgpot_lj()
    stress = atoms.get_stress()
    assert stress.shape == (6,)
    assert np.all(stress[:3] < 0.0)
    np.testing.assert_allclose(stress[3:], 0.0, atol=1e-12)
    full = atoms.get_stress(voigt=False)
    assert full.shape == (3, 3)
    np.testing.assert_allclose(full, full.T)


def test_voigt_order_on_a_sheared_cell() -> None:
    atoms = sheared(argon())
    atoms.calc = rgpot_lj()
    voigt = atoms.get_stress()
    full = atoms.get_stress(voigt=False)
    np.testing.assert_array_equal(
        voigt, [full[0, 0], full[1, 1], full[2, 2], full[1, 2], full[0, 2], full[0, 1]]
    )
    numeric = calculate_numerical_stress(atoms, eps=1e-6, voigt=True)
    np.testing.assert_allclose(voigt, numeric, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("make", [rgpot_lj, rgpot_morse], ids=["lj", "morse"])
def test_per_atom_values_sum_to_totals(make) -> None:
    atoms = sheared(argon()) if make is rgpot_lj else sheared(bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2))
    atoms.rattle(0.03, seed=7)
    atoms.calc = make()
    energies = atoms.get_potential_energies()
    stresses = atoms.get_stresses()
    assert energies.shape == (len(atoms),)
    assert stresses.shape == (len(atoms), 6)
    assert energies.sum() == pytest.approx(atoms.get_potential_energy(), rel=1e-12)
    np.testing.assert_allclose(stresses.sum(axis=0), atoms.get_stress(), atol=1e-14)


def test_free_energy_equals_energy() -> None:
    atoms = argon()
    atoms.calc = rgpot_lj()
    assert atoms.get_potential_energy(force_consistent=True) == atoms.get_potential_energy()


# --- check_state and caching -----------------------------------------------


def test_unchanged_atoms_do_not_call_the_potential() -> None:
    pot = CountingPotential(rgpot.LJPot(**AR))
    atoms = argon()
    atoms.calc = RgpotCalculator(pot)
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_stress()
    atoms.get_potential_energy()
    assert pot.calls == 1
    same = atoms.copy()
    same.calc = atoms.calc
    same.get_forces()
    assert pot.calls == 1


@pytest.mark.parametrize("change", ["positions", "cell", "pbc", "numbers"])
def test_each_system_change_recomputes(change: str) -> None:
    pot = CountingPotential(rgpot.LJPot(**AR))
    atoms = argon()
    calc = RgpotCalculator(pot)
    atoms.calc = calc
    before = atoms.get_potential_energy()
    if change == "positions":
        atoms.positions[0, 0] += 0.01
    elif change == "cell":
        atoms.set_cell(atoms.cell.array * 1.01, scale_atoms=True)
    elif change == "pbc":
        atoms.pbc = False
    else:
        atoms.numbers[0] = 36
    assert change in calc.check_state(atoms)
    after = atoms.get_potential_energy()
    assert pot.calls == 2
    if change != "numbers":
        assert after != before
    atoms.get_potential_energy()
    assert pot.calls == 2


def test_reset_forgets_results() -> None:
    pot = CountingPotential(rgpot.LJPot(**AR))
    atoms = argon()
    atoms.calc = RgpotCalculator(pot)
    atoms.get_potential_energy()
    atoms.calc.reset()
    assert atoms.calc.results == {}
    atoms.get_potential_energy()
    assert pot.calls == 2


# --- finite differences ----------------------------------------------------


def _fd_cases():
    cases = [
        ("lj-bulk", rgpot_lj, lambda: sheared(argon())),
        ("lj-cluster", rgpot_lj, lambda: cluster("Ar", 5.26, "fcc", 6.0)),
        ("morse-bulk", rgpot_morse, lambda: sheared(argon_like_cu())),
        ("morse-cluster", rgpot_morse, lambda: cluster("Cu", 3.6, "fcc", 4.5)),
        ("sw-bulk", lambda: RgpotCalculator("sw"), si_small),
        ("sw-cluster", lambda: RgpotCalculator("sw"), lambda: cluster("Si", 5.431, "diamond", 4.8)),
        ("eam-bulk", lambda: RgpotCalculator("eam_al", numerical_stress=True), al_small),
        ("eam-cluster", lambda: RgpotCalculator("eam_al", numerical_stress=True), lambda: cluster("Al", 4.05, "fcc", 4.5)),
    ]
    marks = {"sw": needs_fortran, "eam": needs_fortran}
    for name, calc, geometry in cases:
        kind = name.split("-")[0]
        yield pytest.param(calc, geometry, id=name, marks=[marks[kind]] if kind in marks else [])


def argon_like_cu() -> Atoms:
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=5)
    return atoms


def si_small() -> Atoms:
    """Eight-atom sheared diamond cell: narrower than twice the SW cutoff."""
    atoms = sheared(bulk("Si", "diamond", a=5.431, cubic=True))
    atoms.rattle(0.05, seed=3)
    return atoms


def al_small() -> Atoms:
    """Four-atom fcc cell, far narrower than twice the 7.1 Angstrom cutoff."""
    atoms = sheared(bulk("Al", "fcc", a=4.05, cubic=True))
    atoms.rattle(0.05, seed=3)
    return atoms


def _spread(atoms: Atoms, count: int = 4) -> list[int]:
    """Up to count atom indices spread over the range, first and last included."""
    last = len(atoms) - 1
    return sorted({round(k * last / (count - 1)) for k in range(count)})


@pytest.mark.parametrize(("make", "geometry"), list(_fd_cases()))
def test_forces_are_minus_the_energy_gradient(make, geometry) -> None:
    atoms = geometry()
    atoms.calc = make()
    analytic = atoms.get_forces()
    # Every Cartesian component of a fixed spread of atoms. Each component
    # costs two energies, and the free-threaded job repeats this body 20
    # times on every thread.
    picked = _spread(atoms)
    numeric = calculate_numerical_forces(atoms, eps=1e-5, iatoms=picked)
    scale = max(1.0, float(np.max(np.abs(numeric))))
    np.testing.assert_allclose(analytic[picked], numeric, atol=2e-7 * scale, rtol=0.0)


@pytest.mark.parametrize(("make", "geometry"), list(_fd_cases()))
def test_stress_is_the_strain_derivative(make, geometry) -> None:
    atoms = geometry()
    atoms.calc = make()
    analytic = atoms.get_stress()
    numeric = calculate_numerical_stress(atoms, eps=1e-6, voigt=True)
    scale = max(1e-3, float(np.max(np.abs(numeric))))
    np.testing.assert_allclose(analytic, numeric, atol=2e-6 * scale, rtol=0.0)


def test_slab_stress_and_forces() -> None:
    atoms = fcc111("Ar", size=(2, 2, 3), a=5.26, vacuum=6.0)
    atoms.rattle(0.03, seed=2)
    assert list(atoms.pbc) == [True, True, False]
    atoms.calc = rgpot_lj()
    np.testing.assert_allclose(
        atoms.get_forces(), calculate_numerical_forces(atoms, eps=1e-5), atol=1e-9
    )
    np.testing.assert_allclose(
        atoms.get_stress(),
        calculate_numerical_stress(atoms, eps=1e-6, voigt=True),
        atol=1e-10,
    )


# --- standard ASE drivers --------------------------------------------------


@pytest.mark.parametrize("optimizer", [BFGS, FIRE])
def test_optimizers_converge(optimizer) -> None:
    atoms = cluster("Ar", 5.26, "fcc", 6.0)
    atoms.calc = rgpot_lj()
    start = atoms.get_potential_energy()
    assert optimizer(atoms, logfile=None).run(fmax=1e-4, steps=2000)
    assert atoms.get_potential_energy() < start
    assert np.max(np.linalg.norm(atoms.get_forces(), axis=1)) < 1e-4


@pytest.mark.parametrize("cell_filter", [FrechetCellFilter, UnitCellFilter])
def test_cell_filters_relax_the_stress(cell_filter) -> None:
    atoms = bulk("Ar", "fcc", a=5.0, cubic=True)
    atoms.calc = rgpot_lj()
    assert BFGS(cell_filter(atoms), logfile=None).run(fmax=1e-5, steps=500)
    np.testing.assert_allclose(atoms.get_stress(), 0.0, atol=1e-6)
    # The 12-6 fcc lattice constant with an 8.5 Angstrom shifted cutoff,
    # from the same relaxation under ASE's LennardJones.
    reference = bulk("Ar", "fcc", a=5.0, cubic=True)
    reference.calc = ase_lj()
    BFGS(cell_filter(reference), logfile=None).run(fmax=1e-5, steps=500)
    assert atoms.cell.lengths() == pytest.approx(reference.cell.lengths(), rel=1e-6)


@needs_fortran
def test_cell_filter_on_a_many_body_potential() -> None:
    atoms = bulk("Si", "diamond", a=5.3, cubic=True)
    atoms.calc = RgpotCalculator("sw")
    assert BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-5, steps=500)
    np.testing.assert_allclose(atoms.get_stress(), 0.0, atol=1e-6)
    # Stillinger-Weber puts the diamond minimum at 5.431 Angstrom.
    assert atoms.cell.lengths() == pytest.approx([5.431] * 3, abs=2e-3)


def test_velocity_verlet_conserves_energy() -> None:
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(3)
    atoms.calc = rgpot_lj(switch_width=1.0)
    rng = np.random.default_rng(0)
    masses = atoms.get_masses()[:, None]
    momenta = rng.normal(size=(len(atoms), 3)) * np.sqrt(masses * units.kB * 40.0)
    momenta -= momenta.mean(axis=0)
    atoms.set_momenta(momenta)
    md = VelocityVerlet(atoms, timestep=5.0 * units.fs)
    totals: list[float] = []
    md.attach(lambda: totals.append(atoms.get_total_energy()), interval=1)
    md.run(1000)
    totals_arr = np.asarray(totals)
    assert atoms.get_temperature() > 5.0
    drift = np.polyfit(np.arange(len(totals_arr)), totals_arr, 1)[0] * len(totals_arr)
    assert abs(drift) / len(atoms) < 2e-6
    assert np.max(np.abs(totals_arr - totals_arr[0])) / len(atoms) < 1e-5


def _vacancy_endpoints() -> tuple[Atoms, Atoms]:
    crystal = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(2)
    del crystal[0]
    hop = int(np.argmin(np.linalg.norm(crystal.positions - [0.0, 2.63, 2.63], axis=1)))
    initial = crystal.copy()
    final = crystal.copy()
    final.positions[hop] = [0.0, 0.0, 0.0]
    for image in (initial, final):
        image.calc = rgpot_lj()
        BFGS(image, logfile=None).run(fmax=1e-4, steps=500)
    return initial, final


@pytest.mark.parametrize("shared", [False, True], ids=["own", "shared"])
def test_neb_vacancy_hop(shared: bool) -> None:
    initial, final = _vacancy_endpoints()
    images = [initial] + [initial.copy() for _ in range(3)] + [final]
    common = rgpot_lj()
    for image in images[1:-1]:
        image.calc = common if shared else rgpot_lj()
    neb = NEB(
        images, climb=True, allow_shared_calculator=shared, method="improvedtangent"
    )
    neb.interpolate()
    assert FIRE(neb, logfile=None).run(fmax=1e-3, steps=2000)
    energies = [image.get_potential_energy() for image in images]
    barrier = max(energies) - energies[0]
    assert 0.0 < barrier < 1.0
    assert energies[0] == pytest.approx(energies[-1], abs=1e-6)
    assert energies[1] == pytest.approx(energies[3], abs=1e-4)


def test_vibrations_match_the_morse_curvature(tmp_path) -> None:
    dimer = Atoms("Cu2", positions=[[0, 0, 0], [CU["re"], 0, 0]], cell=[20, 20, 20])
    dimer.calc = RgpotCalculator("morse", **CU, cutoff=9.0)
    vib = Vibrations(dimer, name=str(tmp_path / "vib"), delta=1e-3, nfree=4)
    vib.run()
    stretch = float(np.max(np.real(vib.get_energies())))
    mass = dimer.get_masses()[0]
    reduced = mass / 2.0
    # omega^2 = k / mu with k = 2 De a^2 at r = re.
    omega = np.sqrt(2.0 * CU["De"] * CU["a"] ** 2 / reduced)
    expected = units._hbar * omega * np.sqrt(units._e / units._amu) * 1e10 / units._e
    assert stretch == pytest.approx(expected, rel=1e-5)


def test_sum_calculator() -> None:
    atoms = argon()
    pieces = [rgpot_lj(), RgpotCalculator("morse", De=0.01, a=1.0, re=4.0, cutoff=8.0)]
    atoms.calc = SumCalculator(pieces)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    parts = []
    for piece in (rgpot_lj(), RgpotCalculator("morse", De=0.01, a=1.0, re=4.0, cutoff=8.0)):
        alone = atoms.copy()
        alone.calc = piece
        parts.append((alone.get_potential_energy(), alone.get_forces(), alone.get_stress()))
    assert energy == pytest.approx(parts[0][0] + parts[1][0], rel=1e-14)
    np.testing.assert_allclose(forces, parts[0][1] + parts[1][1], atol=1e-14)
    np.testing.assert_allclose(stress, parts[0][2] + parts[1][2], atol=1e-16)
    mixed = atoms.copy()
    mixed.calc = SumCalculator([rgpot_lj(), ase_lj()])
    assert mixed.get_potential_energy() == pytest.approx(2.0 * parts[0][0], rel=1e-12)


# --- pickling and copies ---------------------------------------------------


def test_pickle_round_trip_rebuilds_the_potential() -> None:
    atoms = argon()
    calc = rgpot_lj(switch_width=0.5)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    clone = pickle.loads(pickle.dumps(calc))
    assert clone.potential is not calc.potential
    assert clone.potential.config() == calc.potential.config()
    assert clone.results["energy"] == energy
    fresh = argon()
    fresh.calc = clone
    assert fresh.get_potential_energy() == energy


def test_deepcopy_and_copy_are_independent() -> None:
    calc = RgpotCalculator(rgpot.MorsePot(**CU))
    for twin in (copy.deepcopy(calc), copy.copy(calc)):
        assert twin.potential is not calc.potential
        assert twin.potential.config() == calc.potential.config()
        atoms = argon_like_cu()
        atoms.calc = twin
        reference = argon_like_cu()
        reference.calc = calc
        assert atoms.get_potential_energy() == reference.get_potential_energy()


def test_unpicklable_wrapper_says_why() -> None:
    calc = RgpotCalculator(CountingPotential(rgpot.LJPot()))
    with pytest.raises(TypeError, match="cannot pickle"):
        pickle.dumps(calc)


def test_set_rebuilds_the_potential() -> None:
    atoms = argon()
    calc = rgpot_lj()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    calc.set(u0=2.0 * AR["u0"])
    assert calc.results == {}
    assert calc.potential.config()["u0"] == 2.0 * AR["u0"]
    assert atoms.get_potential_energy() == pytest.approx(2.0 * energy, rel=1e-13)
    assert calc.todict()["potential"] == "LJPot"


# --- independent references ------------------------------------------------


def _agree(atoms: Atoms, ours: Calculator, theirs: Calculator, per_atom: bool) -> None:
    left = atoms.copy()
    right = atoms.copy()
    left.calc = ours
    right.calc = theirs
    assert relative(left.get_potential_energy(), right.get_potential_energy()) < 1e-10
    assert relative(left.get_forces(), right.get_forces()) < 1e-10
    if atoms.cell.rank == 3:
        assert relative(left.get_stress(), right.get_stress()) < 1e-10
    if per_atom:
        assert relative(left.get_potential_energies(), right.get_potential_energies()) < 1e-10
        if atoms.cell.rank == 3:
            assert relative(left.get_stresses(), right.get_stresses()) < 1e-10


@pytest.mark.parametrize("repeat", [1, 2, 3, 5])
def test_lj_matches_ase_lennard_jones(repeat: int) -> None:
    """1 and 2 repeats are narrower than twice the cutoff and are tiled."""
    _agree(sheared(argon(repeat)), rgpot_lj(), ase_lj(), per_atom=True)


def test_lj_cluster_and_slab_match_ase_lennard_jones() -> None:
    _agree(cluster("Ar", 5.26, "fcc", 7.0), rgpot_lj(), ase_lj(), per_atom=True)
    slab = fcc111("Ar", size=(2, 2, 3), a=5.26, vacuum=6.0)
    slab.rattle(0.03, seed=2)
    _agree(slab, rgpot_lj(), ase_lj(), per_atom=True)


# On free-threaded 3.13 a pass of ASE's MorsePotential slows down with every
# thread that runs it. Twenty passes of the 256-atom cell on eight threads
# exceed the 180s pytest timeout.
@pytest.mark.iterations(1)
@pytest.mark.parametrize("repeat", [1, 2, 4])
def test_morse_matches_ase_morse_potential(repeat: int) -> None:
    atoms = sheared(bulk("Cu", "fcc", a=3.6, cubic=True).repeat(repeat))
    atoms.rattle(0.05, seed=5)
    _agree(atoms, rgpot_morse(), ase_morse(), per_atom=False)
    _agree(cluster("Cu", 3.6, "fcc", 5.0), rgpot_morse(), ase_morse(), per_atom=False)


@needs_fortran
@pytest.mark.parametrize("repeat", [1, 3])
def test_eam_al_matches_ase_eam(repeat: int) -> None:
    atoms = sheared(bulk("Al", "fcc", a=4.05, cubic=True).repeat(repeat))
    atoms.rattle(0.05, seed=2)
    left, right = atoms.copy(), atoms.copy()
    left.calc = RgpotCalculator("eam_al")
    right.calc = ase_eam_al()
    assert relative(left.get_potential_energy(), right.get_potential_energy()) < 1e-10
    assert relative(left.get_forces(), right.get_forces()) < 1e-10
    blob = pt7(z=13, jitter=0.05, seed=1)
    left, right = blob.copy(), blob.copy()
    left.calc = RgpotCalculator("eam_al")
    right.calc = ase_eam_al()
    assert relative(left.get_potential_energy(), right.get_potential_energy()) < 1e-10
    assert relative(left.get_forces(), right.get_forces()) < 1e-10


# --- any bound potential ---------------------------------------------------


def test_kind_strings_cover_the_bound_classes() -> None:
    kinds = available_kinds()
    for kind in ("lj", "ljcluster", "morse"):
        assert kind in kinds
    by_kind = RgpotCalculator("Morse", **CU)
    by_class = RgpotCalculator("MorsePot", **CU)
    assert by_kind.potential.config() == by_class.potential.config()
    with pytest.raises(ValueError, match="unknown MorsePot parameters"):
        RgpotCalculator("morse", epsilon=1.0)


@needs_fortran
@pytest.mark.parametrize(
    ("kind", "symbol", "lattice", "a"),
    [
        ("sw", "Si", "diamond", 5.431),
        ("stillingerwebersi", "Si", "diamond", 5.431),
        ("edip", "Si", "diamond", 5.431),
        ("lenosky", "Si", "diamond", 5.431),
        ("tersoff", "Si", "diamond", 5.431),
        ("tersoffsic", "Si", "diamond", 5.431),
        ("eam_al", "Al", "fcc", 4.05),
    ],
)
def test_bundled_kinds_are_tiling_invariant(kind: str, symbol: str, lattice: str, a: float) -> None:
    """The energy per atom of a primitive cube equals that of a 3x3x3 supercell."""
    small = bulk(symbol, lattice, a=a, cubic=True)
    small.rattle(0.02, seed=1)
    large = small.repeat(3)
    small.calc = RgpotCalculator(kind)
    large.calc = RgpotCalculator(kind)
    per_atom = small.get_potential_energy() / len(small)
    assert large.get_potential_energy() / len(large) == pytest.approx(per_atom, rel=1e-12)
    np.testing.assert_allclose(
        large.get_forces()[: len(small)], small.get_forces(), atol=1e-11
    )


@needs_fortran
def test_call_only_binding_works() -> None:
    """TersoffSiC has __call__ but no evaluate or caps."""
    atoms = si_small()
    atoms.calc = RgpotCalculator("tersoffsic")
    reference = atoms.copy()
    reference.calc = RgpotCalculator("tersoff")
    assert atoms.get_potential_energy() == pytest.approx(
        reference.get_potential_energy(), rel=1e-12
    )
    assert "stress" not in atoms.calc.implemented_properties


def test_cluster_only_kernel_refuses_periodic_cells() -> None:
    calc = RgpotCalculator("ljcluster", **AR)
    blob = cluster("Ar", 5.26, "fcc", 6.0)
    blob.calc = calc
    reference = blob.copy()
    reference.calc = rgpot_lj()
    assert blob.get_potential_energy() == pytest.approx(
        reference.get_potential_energy(), rel=1e-12
    )
    crystal = argon()
    crystal.calc = calc
    with pytest.raises(ValueError, match="no periodic boundaries"):
        crystal.get_potential_energy()
