"""rgpot.D2Pot against the dftd3 program in D2 mode, and as a SumPot term."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rgpot = pytest.importorskip("rgpot")

DATA = Path(__file__).resolve().parents[1] / "CppCore" / "tests" / "data" / "d2"
# CppCore/rgpot/units.hpp
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
# dftd3 V3.1 stores the b-lyp s6 as the single precision literal 1.2.
S6_DFTD3 = float(np.float32(1.2))
# dftd3's default pair threshold, rthr = 9000 bohr^2.
DFTD3_CUTOFF = 9000.0**0.5 * BOHR_TO_ANGSTROM


def fixture(name):
    z, pos, box, grad, cellgrad = [], [], None, [], []
    energy = None
    for line in (DATA / f"{name}.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, *rest = line.split()
        if key == "box":
            box = np.array(rest, dtype=np.float64).reshape(3, 3)
        elif key == "atom":
            z.append(int(rest[0]))
            pos.append([float(v) for v in rest[1:]])
        elif key == "energy":
            energy = float(rest[0])
        elif key == "grad":
            grad.append([float(v) for v in rest])
        elif key == "cellgrad":
            cellgrad.append([float(v) for v in rest])
    return (
        np.array(pos, dtype=np.float64),
        np.array(z, dtype=np.int32),
        box,
        energy,
        np.array(grad),
        np.array(cellgrad) if cellgrad else None,
    )


@pytest.mark.parametrize(
    "name", ["water_dimer", "benzene", "water_box_cubic", "water_box_sheared"]
)
def test_d2_matches_dftd3(name):
    pos, z, box, energy, grad, cellgrad = fixture(name)
    pot = rgpot.D2Pot(s6=S6_DFTD3, cutoff=DFTD3_CUTOFF)
    out = pot.evaluate(pos, z, box)
    assert out["energy"] == pytest.approx(energy * HARTREE_TO_EV, rel=0, abs=1e-11)
    np.testing.assert_allclose(
        out["forces"], -grad * HARTREE_TO_EV / BOHR_TO_ANGSTROM, rtol=0, atol=1e-9
    )
    if cellgrad is not None:
        volume = abs(np.linalg.det(box))
        # sigma = (1/V) h^T G with the rows of box the lattice vectors.
        sigma = box.T @ cellgrad * HARTREE_TO_EV / BOHR_TO_ANGSTROM / volume
        np.testing.assert_allclose(
            np.asarray(out["stress"]).reshape(3, 3), sigma, rtol=0, atol=1e-10
        )


def test_d2_functional_and_config():
    pot = rgpot.D2Pot()
    assert pot.config() == {"s6": 1.2, "d": 20.0, "sr": 1.1, "cutoff": 150.0}
    assert rgpot.D2Pot(functional="pbe").config()["s6"] == 0.75
    assert pot.caps()["stress"]
    with pytest.raises(ValueError, match="no parameters for functional"):
        rgpot.D2Pot(functional="nonesuch")
    with pytest.raises(ValueError, match="cutoff"):
        rgpot.D2Pot(cutoff=-1.0)


def test_d2_term_cancels_in_a_sum():
    pos, z, box, *_ = fixture("water_box_sheared")
    lj = rgpot.LJPot()
    e_lj, f_lj, _ = lj(pos, z, box)
    e_d2, f_d2, _ = rgpot.D2Pot(cutoff=40.0)(pos, z, box)
    pot = rgpot.SumPot(
        [
            {"kind": "lj", "name": "lj"},
            {"kind": "d2", "name": "d2", "cutoff": 40.0},
            {"kind": "d2", "name": "undo", "cutoff": 40.0, "weight": -1.0},
            {"kind": "d2", "name": "off", "functional": "nonesuch", "enabled": False},
        ]
    )
    e, f, _ = pot(pos, z, box)
    assert e == pytest.approx(e_lj, rel=1e-12)
    np.testing.assert_allclose(f, f_lj, rtol=1e-12, atol=1e-12)
    parts = pot.components()
    assert parts[1]["energy"] == pytest.approx(e_d2, rel=1e-14)
    np.testing.assert_allclose(parts[2]["forces"], f_d2, rtol=1e-14, atol=1e-15)
    assert not parts[3]["evaluated"]
    tuned = rgpot.SumPot([{"kind": "d2", "functional": "pbe", "cutoff": 40.0}])
    assert tuned(pos, z, box)[0] == pytest.approx(e_d2 * 0.75 / 1.2, rel=1e-12)


CPMD_DATA = DATA.parent / "d2cpmd"
CPMD_SYSTEMS = [
    "cubic_water",
    "cubic_below_half",
    "cubic_above_half",
    "ortho_water_methane",
    "triclinic_water_methane",
    "cubic_truncation",
]


def cpmd_fixture(name):
    z, pos, force = [], [], []
    box = energy = None
    for line in (CPMD_DATA / f"{name}.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, *rest = line.split()
        if key == "box":
            box = np.array(rest, dtype=np.float64).reshape(3, 3)
        elif key == "atom":
            z.append(int(rest[0]))
            pos.append([float(v) for v in rest[1:]])
        elif key == "energy":
            energy = float(rest[0])
        elif key == "force":
            force.append([float(v) for v in rest])
    return np.array(pos), np.array(z, dtype=np.int32), box, energy, np.array(force)


@pytest.mark.parametrize("name", CPMD_SYSTEMS)
def test_d2cpmd_matches_opencpmd(name):
    pos, z, box, energy, force = cpmd_fixture(name)
    out = rgpot.D2CpmdPot().evaluate(pos, z, box)
    assert out["energy"] == pytest.approx(energy * HARTREE_TO_EV, rel=0, abs=1e-11)
    np.testing.assert_allclose(
        out["forces"], force * HARTREE_TO_EV / BOHR_TO_ANGSTROM, rtol=0, atol=1e-9
    )
    assert out["stress"] is not None


def test_d2cpmd_config_and_swap():
    assert rgpot.D2CpmdPot().config() == {"s6": 1.2, "alpha": 20.0, "vdweps": 0.01}
    assert rgpot.D2CpmdPot(functional="pbe").config()["s6"] == 0.75
    with pytest.raises(ValueError, match="no s6 for functional"):
        rgpot.D2CpmdPot(functional="b97-d")
    pos, z, box, *_ = cpmd_fixture("cubic_water")
    e_cpmd = rgpot.D2CpmdPot()(pos, z, box)[0]
    e_d2 = rgpot.D2Pot()(pos, z, box)[0]
    swap = rgpot.SumPot(
        [{"kind": "d2cpmd", "name": "undo", "weight": -1.0}, {"kind": "d2", "name": "d2"}]
    )
    assert swap(pos, z, box)[0] == pytest.approx(e_d2 - e_cpmd, rel=1e-12)
    assert swap.components()[0]["energy"] == pytest.approx(e_cpmd, rel=1e-15)
