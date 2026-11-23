"""Tests for the pip-installable rgpot core (real package, real LJ force)."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the real package under test — not a reimplementation.
rgpot = pytest.importorskip("rgpot")


def test_version_and_exports():
    assert hasattr(rgpot, "__version__")
    assert rgpot.__version__
    assert callable(rgpot.evaluate_lj)
    assert rgpot.LJPot is not None
    assert rgpot.EMTPot is not None
    assert rgpot.LJClusterPot is not None
    assert rgpot.MorsePot is not None


def test_evaluate_lj_finite_forces():
    """Two-atom LJ: finite energy and forces with shape (2, 3)."""
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.array([0, 0], dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 20.0

    energy, forces, variance = rgpot.evaluate_lj(positions, atom_types, box)

    assert math.isfinite(float(energy)), f"energy not finite: {energy}"
    assert forces.shape == (2, 3), forces.shape
    assert np.all(np.isfinite(forces)), forces
    assert math.isfinite(float(variance))
    # Opposite forces on a pair (action-reaction) within numerical noise
    assert np.allclose(forces[0] + forces[1], 0.0, atol=1e-9), forces


def test_ljpot_class_matches_evaluate_lj():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float64
    )
    atom_types = np.zeros(3, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 30.0

    e1, f1, v1 = rgpot.evaluate_lj(positions, atom_types, box)
    pot = rgpot.LJPot()
    e2, f2, v2 = pot(positions, atom_types, box)

    assert abs(e1 - e2) < 1e-12
    assert np.allclose(f1, f2)
    assert abs(v1 - v2) < 1e-12


def _switch(t: float) -> tuple[float, float]:
    """S(t) and dS/dt of the C^2 quintic switch."""
    return 1 - 10 * t**3 + 15 * t**4 - 6 * t**5, -30 * t**2 * (1 - t) ** 2


def test_ljpot_switch_width_matches_closed_form():
    """LJPot(switch_width=w) is the unshifted 12-6 energy times S(t)."""
    u0, psi, cutoff, width, r = 1.0, 1.0, 2.5, 0.5, 2.25
    positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.zeros(2, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 30.0

    pot = rgpot.LJPot(u0=u0, cutoff=cutoff, psi=psi, switch_width=width)
    energy, forces, _ = pot(positions, atom_types, box)

    v = 4 * u0 * ((psi / r) ** 12 - (psi / r) ** 6)
    dv = 4 * u0 * (-12 * psi**12 / r**13 + 6 * psi**6 / r**7)
    s, ds_dt = _switch((r - (cutoff - width)) / width)
    assert energy == pytest.approx(v * s, rel=1e-12)
    # Atom 0 sits at x = 0, atom 1 at x = r: F_0 = dE/dr.
    assert forces[0, 0] == pytest.approx(dv * s + v * ds_dt / width, rel=1e-10)
    assert forces[1, 0] == pytest.approx(-forces[0, 0], rel=1e-12)

    shifted, _, _ = rgpot.LJPot(u0=u0, cutoff=cutoff, psi=psi)(
        positions, atom_types, box
    )
    assert shifted != pytest.approx(energy, rel=1e-6)
    with pytest.raises(ValueError):
        rgpot.LJPot(cutoff=2.5, switch_width=3.0)


def test_ljpot_defaults_are_the_cpp_config():
    positions = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.zeros(2, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 40.0
    e_default, _, _ = rgpot.LJPot()(positions, atom_types, box)
    e_explicit, _, _ = rgpot.LJPot(u0=1.0, cutoff=15.0, psi=1.0, switch_width=0.0)(
        positions, atom_types, box
    )
    assert e_default == e_explicit


@pytest.mark.skipif(
    getattr(rgpot, "ExprPot", None) is None, reason="built without -Dwith_expr"
)
def test_expr_children_take_pair_config_keys():
    positions = np.array([[0.0, 0.0, 0.0], [2.25, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.zeros(2, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 30.0
    direct, _, _ = rgpot.LJPot(cutoff=2.5, switch_width=0.5)(
        positions, atom_types, box
    )
    expr = rgpot.ExprPot(
        "lj", {"lj": {"kind": "lj", "cutoff": 2.5, "switch_width": 0.5}}
    )
    via_expr, _, _ = expr(positions, atom_types, box)
    assert via_expr == pytest.approx(direct, rel=1e-12)

    r, De, a, re, cutoff, width = 5.0, 0.7102, 1.6047, 2.8970, 6.0, 2.0
    positions[1, 0] = r
    morse = rgpot.ExprPot(
        "m", {"m": {"kind": "morse", "cutoff": cutoff, "switch_width": width}}
    )
    energy, _, _ = morse(positions, atom_types, box)
    d = 1 - math.exp(-a * (r - re))
    s, _ = _switch((r - (cutoff - width)) / width)
    assert energy == pytest.approx((De * d * d - De) * s, rel=1e-12)


def test_extension_rpath_has_no_host_build_tree():
    """Installed _core must not embed absolute build-host RUNPATH entries."""
    parent = Path(rgpot.__file__).resolve().parent
    cores = list(parent.glob("_core*.so")) + list(parent.glob("_core*.pyd"))
    assert cores, f"missing _core extension under {parent}"
    so = cores[0]
    if not shutil.which("readelf"):
        pytest.skip("readelf not available")
    out = subprocess.check_output(["readelf", "-d", str(so)], text=True)
    # Stricter: no build-tree / home build markers in RPATH/RUNPATH lines
    linux_home = f"{os.sep}home{os.sep}"
    scratch_tmp = os.sep.join(["", "scratch", "tmp"])
    for line in out.splitlines():
        low = line.lower()
        if "runpath" not in low and "rpath" not in low:
            continue
        for bad in (
            linux_home,
            "/Users/",
            "/bbdir",
            "/build-pyeon",
            "/.mesonpy-",
            scratch_tmp,
        ):
            assert bad not in line, f"host path marker {bad!r} in:\n{line}\nfull:\n{out}"


@pytest.mark.skipif(
    os.environ.get("RGPOT_SKIP_WHEEL_TEST") == "1",
    reason="RGPOT_SKIP_WHEEL_TEST=1",
)
def test_import_with_empty_ld_library_path():
    code = (
        "import os; os.environ.pop('LD_LIBRARY_PATH', None); "
        "os.environ.pop('LIBRARY_PATH', None); "
        "import rgpot; import numpy as np; "
        "e,f,v=rgpot.evaluate_lj("
        "np.array([[0.,0.,0.],[1.5,0.,0.]]),"
        "np.array([0,0],dtype=np.int32),"
        "np.eye(3)*20); "
        "assert f.shape==(2,3); print('RGPOT_FORCE_OK', float(e))"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LD_LIBRARY_PATH", "LIBRARY_PATH")
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RGPOT_FORCE_OK" in proc.stdout


def test_core_is_abi3_or_free_threaded():
    """On CPython >= 3.12 GIL builds the extension must be abi3-tagged."""
    import sys
    import sysconfig

    parent = Path(rgpot.__file__).resolve().parent
    cores = list(parent.glob("_core*.so")) + list(parent.glob("_core*.pyd"))
    assert cores, f"missing _core under {parent}"
    name = cores[0].name
    gil_disabled = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if sys.version_info >= (3, 12) and not gil_disabled:
        assert "abi3" in name, f"expected abi3 extension, got {name}"


@pytest.mark.skipif(not rgpot.has_cache, reason="built without -Dwith_cache")
def test_cache_counts_split_computed_from_served(tmp_path):
    """Cold run computes, warm run on the same store is served."""
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.array([0, 0], dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 20.0
    store = str(tmp_path / "cache")

    cold = rgpot.PotentialCache(store)
    pot = rgpot.LJPot()
    pot.set_cache(cold)
    assert cold.counts() == (0, 0)
    first = pot(positions, atom_types, box)
    pot(positions, atom_types, box)
    assert cold.counts() == (1, 1)
    del pot, cold

    warm = rgpot.PotentialCache(store)
    pot = rgpot.LJPot()
    pot.set_cache(warm)
    again = pot(positions, atom_types, box)
    assert warm.counts() == (0, 1)
    assert float(again[0]) == float(first[0])
    warm.reset_counts()
    assert warm.counts() == (0, 0)


def _cluster13() -> np.ndarray:
    """Thirteen atoms inside a few angstroms, fixed by one seed."""
    pos = np.random.default_rng(13).uniform(-2.2, 2.2, size=(13, 3))
    return np.ascontiguousarray(pos, dtype=np.float64)


def _finite_difference_forces(pot, positions, atom_types, box, delta=1.0e-6):
    """Central difference, force = -(E+ - E-) / (2 delta)."""
    numeric = np.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += delta
            minus[atom, axis] -= delta
            energy_plus = pot(plus, atom_types, box)[0]
            energy_minus = pot(minus, atom_types, box)[0]
            numeric[atom, axis] = -(energy_plus - energy_minus) / (2.0 * delta)
    return numeric


def test_lj_cluster_13_matches_shifted_pair_and_finite_difference():
    """Shifted 12-6 pair term from the free-boundary cluster kernel.

    Energy is ``4 u0 a (a - 1) - shift`` with ``a = (psi / r)^6`` and
    the shift evaluated at the cutoff. The force on atom i is
    ``scale * (r_i - r_j)`` with ``scale = 6 b (2 a - 1) / r^2`` and
    ``b = 4 u0 a``.
    """
    u0, cutoff, psi = 1.0, 15.0, 1.0
    positions = _cluster13()
    atom_types = np.zeros(len(positions), dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 40.0
    energy, forces, _ = rgpot.LJClusterPot(u0=u0, cutoff=cutoff, psi=psi)(
        positions, atom_types, box
    )

    a_cut = (psi / cutoff) ** 6
    shift = 4.0 * u0 * a_cut * (a_cut - 1.0)
    cutoff2 = cutoff * cutoff
    psi2 = psi * psi
    four_u0 = 4.0 * u0
    reference = 0.0
    reference_forces = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = positions[i] - positions[j]
            r2 = float(delta @ delta)
            if not (r2 <= cutoff2):
                continue
            inv = 1.0 / r2
            sr2 = psi2 * inv
            a = (sr2 * sr2) * sr2
            b = four_u0 * a
            reference += (b * (a - 1.0)) - shift
            scale = ((6.0 * b) * inv) * ((2.0 * a) - 1.0)
            reference_forces[i] += scale * delta
            reference_forces[j] -= scale * delta

    assert energy == pytest.approx(reference, abs=1e-9)
    assert forces == pytest.approx(reference_forces, abs=1e-9)
    numeric = _finite_difference_forces(
        rgpot.LJClusterPot(u0=u0, cutoff=cutoff, psi=psi),
        positions,
        atom_types,
        box,
    )
    assert forces == pytest.approx(numeric, abs=1e-5)


def test_morse_13_matches_shifted_pair_and_finite_difference():
    """Shifted Morse row: ``De d^2 - De - shift``, ``d = 1 - exp(-a(r-re))``.

    The force scale is ``2 De a d (d - 1) / r``, and the force on atom i
    is that scale times ``(r_i - r_j)`` after the orthogonal minimum image.
    """
    depth, width, equilibrium, cutoff = 0.7102, 1.6047, 2.897, 9.5
    positions = _cluster13()
    atom_types = np.zeros(len(positions), dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 40.0
    lengths = np.diag(box)
    energy, forces, _ = rgpot.MorsePot(
        De=depth, a=width, re=equilibrium, cutoff=cutoff
    )(positions, atom_types, box)

    d_cut = 1.0 - math.exp(-width * (cutoff - equilibrium))
    shift = depth * d_cut * d_cut - depth
    two_de_a = 2.0 * depth * width
    cutoff2 = cutoff * cutoff
    reference = 0.0
    reference_forces = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = positions[i] - positions[j]
            delta = delta - lengths * np.round(delta / lengths)
            r2 = float(delta @ delta)
            if not (r2 <= cutoff2):
                continue
            distance = math.sqrt(r2)
            d = 1.0 - math.exp(-width * (distance - equilibrium))
            reference += depth * d * d - depth - shift
            fscale = two_de_a * d * (d - 1.0) / distance
            reference_forces[i] += fscale * delta
            reference_forces[j] -= fscale * delta

    assert energy == pytest.approx(reference, abs=1e-9)
    assert forces == pytest.approx(reference_forces, abs=1e-9)
    numeric = _finite_difference_forces(
        rgpot.MorsePot(De=depth, a=width, re=equilibrium, cutoff=cutoff),
        positions,
        atom_types,
        box,
    )
    assert forces == pytest.approx(numeric, abs=1e-5)


def _water_h_13():
    """Four waters plus a lone hydrogen, last atom Z=1, only H and O."""
    n_molecules = 4
    n_atoms = 3 * n_molecules + 1
    bond = 0.9572
    half = 0.5 * 104.52 * math.pi / 180.0
    arm_x = bond * math.sin(half)
    arm_y = bond * math.cos(half)
    oxygens = (
        (3.0, 3.0, 3.0),
        (3.0, 7.0, 3.0),
        (6.5, 3.2, 3.4),
        (6.2, 6.8, 4.1),
    )
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    atom_types = np.ones(n_atoms, dtype=np.int32)
    for molecule, oxygen in enumerate(oxygens):
        ox, oy, oz = oxygen
        positions[2 * molecule] = (ox + arm_x, oy + arm_y, oz)
        positions[2 * molecule + 1] = (ox - arm_x, oy + arm_y, oz)
        positions[2 * n_molecules + molecule] = oxygen
        atom_types[2 * n_molecules + molecule] = 8
    positions[-1] = (5.0, 5.0, 4.6)
    return np.ascontiguousarray(positions), atom_types


def _water_h_reference(positions, atom_types):
    """Open-boundary legs of the lone hydrogen, in eV and eV/Angstrom.

    Dispersion, Born-Mayer, and the oxygen Morse well are the published
    defaults, scaled by ``1/96.485336``. The site force is ``-dE/dr``
    along the vector from the lone hydrogen to that site.
    """
    scale = 1.0 / 96.485336
    rm_ho, rm_hh = 4.04741, 2.19954
    c6_ho, c6_hh = 2216.76, 30.2562
    a_ho, b_ho = 4158.88, 2.63805
    a_hh, b_hh = 5563.57, 4.27352
    depth, beta, equilibrium = 287.598, 4.73771, 0.816286
    damp_ho = 1.28 * rm_ho
    damp_hh = 1.28 * rm_hh

    def dispersion(c6, onset, distance):
        if distance >= onset:
            damp, slope = 1.0, 0.0
        else:
            damp = math.exp(-((onset / distance) - 1.0) ** 2)
            slope = 2.0 * (onset**2 / distance**3 - onset / distance**2) * damp
        inv_r6 = distance ** (-6)
        energy = -damp * c6 * inv_r6
        dedr = -slope * c6 * inv_r6 + 6.0 * damp * c6 * inv_r6 / distance
        return energy, dedr

    energy = 0.0
    forces = np.zeros_like(positions)
    lone = positions[-1]
    for site in range(len(positions) - 1):
        vector = positions[site] - lone
        distance = float(np.linalg.norm(vector))
        if int(atom_types[site]) == 8:
            e_disp, d_disp = dispersion(c6_ho, damp_ho, distance)
            e_rep = a_ho * math.exp(-b_ho * distance)
            d_rep = -b_ho * e_rep
            decay = math.exp(-beta * (distance - equilibrium))
            decay2 = math.exp(-2.0 * beta * (distance - equilibrium))
            e_morse = depth * (decay2 - 2.0 * decay)
            d_morse = -2.0 * beta * depth * (decay2 - decay)
            leg_energy = e_disp + e_rep + e_morse
            dedr = d_disp + d_rep + d_morse
        else:
            e_disp, d_disp = dispersion(c6_hh, damp_hh, distance)
            e_rep = a_hh * math.exp(-b_hh * distance)
            leg_energy = e_disp + e_rep
            dedr = d_disp - b_hh * e_rep
        energy += scale * leg_energy
        site_force = -(scale * dedr) * vector / distance
        forces[site] += site_force
        forces[-1] -= site_force
    return energy, forces


def test_water_h_13_matches_leg_sum_and_finite_difference():
    if rgpot.WaterHPot is None:
        pytest.skip("built without the water plus hydrogen potential")
    positions, atom_types = _water_h_13()
    box = np.zeros((3, 3), dtype=np.float64)
    pot = rgpot.WaterHPot()
    energy, forces, _ = pot(positions, atom_types, box)
    reference, reference_forces = _water_h_reference(positions, atom_types)
    assert energy == pytest.approx(reference, abs=1e-12)
    assert forces == pytest.approx(reference_forces, abs=1e-12)

    periodic = np.eye(3, dtype=np.float64) * 30.0
    periodic_energy, periodic_forces, _ = pot(positions, atom_types, periodic)
    assert periodic_energy == pytest.approx(energy, abs=1e-12)
    assert periodic_forces == pytest.approx(forces, abs=1e-12)

    numeric = _finite_difference_forces(pot, positions, atom_types, box)
    assert forces == pytest.approx(numeric, abs=1e-5)
