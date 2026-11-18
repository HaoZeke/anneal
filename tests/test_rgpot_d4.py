"""rgpot.D4Pot with BLYP against the dftd4 program, and as a SumPot term."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rgpot = pytest.importorskip("rgpot")
pytestmark = pytest.mark.skipif(not rgpot.has_dftd4, reason="built without dftd4")

DATA = Path(__file__).resolve().parents[1] / "CppCore" / "tests" / "data" / "d4"
# CppCore/rgpot/units.hpp
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988


def fixture(name):
    z, pos, box, grad, virial = [], [], None, [], []
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
        elif key == "virial":
            virial.append([float(v) for v in rest])
    return (
        np.array(pos, dtype=np.float64),
        np.array(z, dtype=np.int32),
        box,
        energy,
        np.array(grad),
        np.array(virial) if virial else None,
    )


@pytest.mark.parametrize("name", ["water_dimer", "benzene", "water_box_cubic"])
def test_d4_blyp_matches_dftd4(name):
    pos, z, box, energy, grad, virial = fixture(name)
    pot = rgpot.D4Pot(functional="blyp", periodic=virial is not None)
    out = pot.evaluate(pos, z, box)
    assert out["energy"] == pytest.approx(energy * HARTREE_TO_EV, rel=0, abs=1e-11)
    np.testing.assert_allclose(
        out["forces"], -grad * HARTREE_TO_EV / BOHR_TO_ANGSTROM, rtol=0, atol=1e-9
    )
    if virial is None:
        assert out.get("stress") is None
    else:
        volume = abs(np.linalg.det(box / BOHR_TO_ANGSTROM))
        sigma = virial / volume * HARTREE_TO_EV / BOHR_TO_ANGSTROM**3
        np.testing.assert_allclose(
            np.asarray(out["stress"]).reshape(3, 3), sigma, rtol=0, atol=1e-10
        )


def test_d4_config_and_caps():
    pot = rgpot.D4Pot(functional="blyp")
    assert pot.config() == {
        "functional": "blyp",
        "charge": 0.0,
        "atm": True,
        "periodic": False,
    }
    assert not pot.caps()["stress"]
    assert rgpot.D4Pot(periodic=True).caps()["stress"]


def test_sumpot_swaps_d2_for_d4():
    pos, z, box, *_ = fixture("water_dimer")
    box = np.diag([40.0, 38.0, 39.0])
    d2 = rgpot.D2Pot().evaluate(pos, z, box)
    d4 = rgpot.D4Pot(functional="blyp", periodic=True).evaluate(pos, z, box)
    pot = rgpot.SumPot(
        [
            {"kind": "d2", "name": "d2", "weight": -1.0},
            {"kind": "d4", "name": "d4", "functional": "blyp", "periodic": True},
        ]
    )
    out = pot.evaluate(pos, z, box)
    assert out["energy"] == pytest.approx(d4["energy"] - d2["energy"], abs=1e-12)
    np.testing.assert_allclose(
        np.asarray(out["stress"]), np.asarray(d4["stress"]) - np.asarray(d2["stress"]),
        rtol=0, atol=1e-12,
    )
    names = [c["name"] for c in pot.components()]
    assert names == ["d2", "d4"]
    assert pot.components()[1]["energy"] == pytest.approx(d4["energy"], abs=1e-12)
