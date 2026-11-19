#!/usr/bin/env python3
"""Contract tests for NWChem PotentialConfig helpers (Cap'n Proto round-trip)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import capnp

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = SCRIPT_DIR / "../CppCore/rgpot/rpc/Potentials.capnp"
pot_capnp = capnp.load(str(SCHEMA_PATH))

sys.path.insert(0, str(SCRIPT_DIR))
from nwchem_params import (  # noqa: E402
    configure_nwchem,
    make_nwchem_params,
    make_potential_config_none,
    make_potential_config_nwchem,
)


class FakePotential:
    def __init__(self):
        self.config = None

    async def configure(self, config):
        self.config = config

        class Result:
            ok = True
            message = "configured"

        return Result()


def test_make_nwchem_params_fields(monkeypatch) -> None:
    monkeypatch.setenv("RGPOT_NWCHEM_ENGINE", "/env/libnwchemc.so")
    monkeypatch.setenv("NWCHEM_TOP", "/env/nwchem")
    params = make_nwchem_params(
        pot_capnp,
        basis="6-31g*",
        theory="dft",
        scf_type="blyp",
        charge=-1,
        multiplicity=2,
        engine_path="",
        nwchem_root="",
    )
    assert params.basis == "6-31g*"
    assert params.theory == "dft"
    assert params.scfType == "blyp"
    assert params.charge == -1
    assert params.multiplicity == 2
    assert params.enginePath == "/env/libnwchemc.so"
    assert params.nwchemRoot == "/env/nwchem"


def test_make_potential_config_nwchem_arm() -> None:
    cfg = make_potential_config_nwchem(
        pot_capnp,
        basis="sto-3g",
        theory="scf",
        scf_type="rhf",
        charge=0,
        multiplicity=1,
        engine_path="/opt/libnwchemc.so",
        nwchem_root="/opt/nwchem",
    )
    assert cfg.which() == "nwchem"
    assert cfg.nwchem.basis == "sto-3g"
    assert cfg.nwchem.theory == "scf"
    assert cfg.nwchem.scfType == "rhf"
    assert cfg.nwchem.enginePath == "/opt/libnwchemc.so"
    assert cfg.nwchem.nwchemRoot == "/opt/nwchem"


def test_make_potential_config_none_arm() -> None:
    cfg = make_potential_config_none(pot_capnp)
    assert cfg.which() == "none"


def test_configure_nwchem_async_round_trip() -> None:
    pot = FakePotential()
    ok, msg = asyncio.run(
        configure_nwchem(
            pot,
            pot_capnp,
            basis="cc-pvdz",
            theory="scf",
            scf_type="uhf",
            charge=1,
            multiplicity=2,
            engine_path="/tmp/libnwchemc.so",
            nwchem_root="/tmp/nwchem",
        )
    )
    assert ok is True
    assert msg == "configured"
    assert pot.config.which() == "nwchem"
    assert pot.config.nwchem.basis == "cc-pvdz"
    assert pot.config.nwchem.theory == "scf"
    assert pot.config.nwchem.scfType == "uhf"
    assert pot.config.nwchem.charge == 1
    assert pot.config.nwchem.multiplicity == 2
    assert pot.config.nwchem.enginePath == "/tmp/libnwchemc.so"
    assert pot.config.nwchem.nwchemRoot == "/tmp/nwchem"


def test_nwchem_params_flat_serialize_round_trip() -> None:
    """Serialize NWChemParams to words and read back (schema wire format)."""
    params = make_nwchem_params(
        pot_capnp,
        basis="def2-svp",
        theory="dft",
        scf_type="pbe0",
        charge=0,
        multiplicity=1,
        engine_path="/lib/libnwchemc.so",
        nwchem_root="/nwchem",
    )
    data = params.to_bytes()
    # pycapnp returns a context manager for from_bytes in some versions
    with pot_capnp.NWChemParams.from_bytes(data) as restored:
        assert restored.basis == "def2-svp"
        assert restored.theory == "dft"
        assert restored.scfType == "pbe0"
        assert restored.enginePath == "/lib/libnwchemc.so"
        assert restored.nwchemRoot == "/nwchem"


if __name__ == "__main__":
    import pytest

    with pytest.MonkeyPatch.context() as monkeypatch:
        test_make_nwchem_params_fields(monkeypatch)
    test_make_potential_config_nwchem_arm()
    test_make_potential_config_none_arm()
    test_configure_nwchem_async_round_trip()
    test_nwchem_params_flat_serialize_round_trip()
    print("test_nwchem_params: all ok")
