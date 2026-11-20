#!/usr/bin/env python3
"""Contract tests for CPMD PotentialConfig helpers."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import capnp

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = SCRIPT_DIR / "../CppCore/rgpot/rpc/Potentials.capnp"
pot_capnp = capnp.load(str(SCHEMA_PATH))

try:
    from _temp_paths import temp_path_str
    from cpmd_params import (
        configure_cpmd,
        make_cpmd_params,
        make_potential_config_cpmd,
    )
except ImportError:
    sys.path.insert(0, str(SCRIPT_DIR))
    from _temp_paths import temp_path_str
    from cpmd_params import (
        configure_cpmd,
        make_cpmd_params,
        make_potential_config_cpmd,
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


def test_make_cpmd_params(tmp_path: Path) -> None:
    os.environ["CPMD_ROOT"] = "/env/cpmd"
    # Explicit engine_path wins over any ambient RGPOT_CPMD_* from the runner.
    params = make_cpmd_params(
        pot_capnp,
        functional="PBE",
        cut_off_ry=90.0,
        charge=-1,
        multiplicity=2,
        task="energy",
        title="water",
        memory_mb=512,
        scratch_dir=temp_path_str(tmp_path, "cpmd-scratch"),
        permanent_dir=temp_path_str(tmp_path, "cpmd-perm"),
        engine_path="/env/libcpmdc.so",
        input_blocks=["&CPMD\n  OPTIMIZE WAVEFUNCTION\n&END"],
        system_cell=[9.0, 0.0, 0.0, 0.0, 9.5, 0.0, 0.0, 0.0, 10.0],
        pseudopotentials=[{"element": "O", "path": "O_BLYP.psp", "lmax": 2}],
        set_directives=[{"key": "SYSTEM.POISSON SOLVER", "value": "HOCKNEY"}],
    )

    assert params.functional == "PBE"
    assert params.cutOffRy == 90.0
    assert params.charge == -1
    assert params.multiplicity == 2
    assert params.task == "energy"
    assert params.title == "water"
    assert params.memoryMb == 512
    assert params.scratchDir == temp_path_str(tmp_path, "cpmd-scratch")
    assert params.permanentDir == temp_path_str(tmp_path, "cpmd-perm")
    assert params.cpmdRoot == "/env/cpmd"
    assert params.enginePath == "/env/libcpmdc.so"
    assert list(params.inputBlocks) == ["&CPMD\n  OPTIMIZE WAVEFUNCTION\n&END"]

    sections = params.inputSections
    assert len(sections) == 3
    assert sections[0].which() == "system"
    assert list(sections[0].system.cell) == [
        9.0,
        0.0,
        0.0,
        0.0,
        9.5,
        0.0,
        0.0,
        0.0,
        10.0,
    ]
    assert sections[0].system.cutOffRy == 90.0
    assert sections[1].which() == "atoms"
    assert sections[1].atoms.pseudopotentials[0].element == "O"
    assert sections[1].atoms.pseudopotentials[0].path == "O_BLYP.psp"
    assert sections[1].atoms.pseudopotentials[0].lmax == 2
    assert sections[2].which() == "set"
    assert sections[2].set.key == "SYSTEM.POISSON SOLVER"
    assert sections[2].set.value == "HOCKNEY"


def test_make_cpmd_params_accepts_all_section_arms() -> None:
    params = make_cpmd_params(
        pot_capnp,
        input_sections=[
            {
                "kind": "generic",
                "name": "PIMD",
                "directives": [{"keyword": "TEMP", "args": ["300"]}],
            },
            {
                "kind": "system",
                "symmetry": 1,
                "angstrom": False,
                "cell": [8.0, 8.0, 9.0, 90.0, 90.0, 120.0],
                "cutOffRy": 88.0,
                "charge": 1,
                "multiplicity": 3,
                "scale": 0.5,
                "directives": [{"keyword": "POISSON SOLVER", "args": ["HOCKNEY"]}],
            },
            {
                "kind": "atoms",
                "pseudopotentials": [
                    {"element": "Si", "path": "Si_MT_PBE.psp", "lmax": 2}
                ],
                "directives": [{"keyword": "ISOLATED MOLECULE", "args": []}],
            },
            {
                "kind": "cpmd",
                "optimizeWavefunction": False,
                "optimizeGeometry": True,
                "molecularDynamics": True,
                "molecularDynamicsCp": True,
                "molecularDynamicsBo": True,
                "molecularDynamicsEh": True,
                "molecularDynamicsPt": True,
                "molecularDynamicsClassical": True,
                "molecularDynamicsFile": "TRAJECTORY.in",
                "convergenceOrbitals": 2.5e-7,
                "convergenceGeometry": 1.0e-4,
                "maxStep": 8,
                "maxIter": 12,
                "timestep": 4.0,
                "electronMass": 450.0,
                "nose": True,
                "noseIons": True,
                "noseElectrons": True,
                "berendsen": "300 100",
                "langevin": True,
                "annealing": "IONS 300 50",
                "quench": True,
                "rattle": True,
                "shake": True,
                "constraint": "FIX COM",
                "trotter": "8",
                "restart": True,
                "printOptions": "FORCES ON",
                "storeOptions": "WAVEFUNCTION",
                "centerMoleculeOff": True,
                "centerMoleculeOn": True,
                "diis": True,
                "odiis": True,
                "pcg": True,
                "diagonalization": True,
                "freeEnergy": True,
                "interface": True,
                "qmmm": True,
                "bicanonicalEnsemble": True,
                "cdft": True,
                "properties": True,
                "restartWavefunction": True,
                "trajectory": True,
                "directives": [{"keyword": "PRINT", "args": ["FORCES", "ON"]}],
            },
            {
                "kind": "dft",
                "functional": "PBE0",
                "lsd": True,
                "gcCutoff": 1.0e-8,
                "xcDriver": "LIBXC",
                "libxc": "GGA_X_PBE GGA_C_PBE",
                "lrKernel": "PBE",
                "refunct": "PBE",
                "mtsHighFunc": "PBE0",
                "mtsLowFunc": "PBE",
                "hfx": True,
                "hfxScreening": "0.2",
                "hubbard": "U 1 4.0",
                "alpha": 0.25,
                "beta": 0.75,
                "oldCode": True,
                "newCode": True,
                "correlation": "LYP",
                "exchange": "B88",
                "becke88": True,
                "directives": [{"keyword": "HFX", "args": ["SCREENING"]}],
            },
            {"kind": "set", "key": "CPMD.MAXSTEP", "value": "12"},
            {"kind": "raw", "text": "&VDW\n  DISPERSION\n&END"},
        ],
    )

    sections = params.inputSections
    assert len(sections) == 7
    assert sections[0].which() == "generic"
    assert sections[0].generic.name == "PIMD"
    assert sections[0].generic.directives[0].keyword == "TEMP"
    assert list(sections[0].generic.directives[0].args) == ["300"]
    assert sections[1].which() == "system"
    assert sections[1].system.symmetry == 1
    assert sections[1].system.angstrom is False
    assert list(sections[1].system.cell) == [8.0, 8.0, 9.0, 90.0, 90.0, 120.0]
    assert sections[1].system.cutOffRy == 88.0
    assert sections[1].system.charge == 1
    assert sections[1].system.multiplicity == 3
    assert sections[1].system.scale == 0.5
    assert sections[1].system.directives[0].keyword == "POISSON SOLVER"
    assert list(sections[1].system.directives[0].args) == ["HOCKNEY"]
    assert sections[2].which() == "atoms"
    assert sections[2].atoms.pseudopotentials[0].element == "Si"
    assert sections[2].atoms.pseudopotentials[0].path == "Si_MT_PBE.psp"
    assert sections[2].atoms.pseudopotentials[0].lmax == 2
    assert sections[2].atoms.directives[0].keyword == "ISOLATED MOLECULE"
    assert list(sections[2].atoms.directives[0].args) == []
    assert sections[3].which() == "cpmd"
    assert sections[3].cpmd.optimizeWavefunction is False
    assert sections[3].cpmd.optimizeGeometry is True
    assert sections[3].cpmd.molecularDynamics is True
    assert sections[3].cpmd.molecularDynamicsCp is True
    assert sections[3].cpmd.molecularDynamicsBo is True
    assert sections[3].cpmd.molecularDynamicsEh is True
    assert sections[3].cpmd.molecularDynamicsPt is True
    assert sections[3].cpmd.molecularDynamicsClassical is True
    assert sections[3].cpmd.molecularDynamicsFile == "TRAJECTORY.in"
    assert sections[3].cpmd.convergenceOrbitals == 2.5e-7
    assert sections[3].cpmd.convergenceGeometry == 1.0e-4
    assert sections[3].cpmd.maxStep == 8
    assert sections[3].cpmd.maxIter == 12
    assert sections[3].cpmd.timestep == 4.0
    assert sections[3].cpmd.electronMass == 450.0
    assert sections[3].cpmd.nose is True
    assert sections[3].cpmd.noseIons is True
    assert sections[3].cpmd.noseElectrons is True
    assert sections[3].cpmd.berendsen == "300 100"
    assert sections[3].cpmd.langevin is True
    assert sections[3].cpmd.annealing == "IONS 300 50"
    assert sections[3].cpmd.quench is True
    assert sections[3].cpmd.rattle is True
    assert sections[3].cpmd.shake is True
    assert sections[3].cpmd.constraint == "FIX COM"
    assert sections[3].cpmd.trotter == "8"
    assert sections[3].cpmd.restart is True
    assert sections[3].cpmd.printOptions == "FORCES ON"
    assert sections[3].cpmd.storeOptions == "WAVEFUNCTION"
    assert sections[3].cpmd.centerMoleculeOff is True
    assert sections[3].cpmd.centerMoleculeOn is True
    assert sections[3].cpmd.diis is True
    assert sections[3].cpmd.odiis is True
    assert sections[3].cpmd.pcg is True
    assert sections[3].cpmd.diagonalization is True
    assert sections[3].cpmd.freeEnergy is True
    assert sections[3].cpmd.interface is True
    assert sections[3].cpmd.qmmm is True
    assert sections[3].cpmd.bicanonicalEnsemble is True
    assert sections[3].cpmd.cdft is True
    assert sections[3].cpmd.properties is True
    assert sections[3].cpmd.restartWavefunction is True
    assert sections[3].cpmd.trajectory is True
    assert sections[3].cpmd.directives[0].keyword == "PRINT"
    assert list(sections[3].cpmd.directives[0].args) == ["FORCES", "ON"]
    assert sections[4].which() == "dft"
    assert sections[4].dft.functional == "PBE0"
    assert sections[4].dft.lsd is True
    assert sections[4].dft.gcCutoff == 1.0e-8
    assert sections[4].dft.xcDriver == "LIBXC"
    assert sections[4].dft.libxc == "GGA_X_PBE GGA_C_PBE"
    assert sections[4].dft.lrKernel == "PBE"
    assert sections[4].dft.refunct == "PBE"
    assert sections[4].dft.mtsHighFunc == "PBE0"
    assert sections[4].dft.mtsLowFunc == "PBE"
    assert sections[4].dft.hfx is True
    assert sections[4].dft.hfxScreening == "0.2"
    assert sections[4].dft.hubbard == "U 1 4.0"
    assert sections[4].dft.alpha == 0.25
    assert sections[4].dft.beta == 0.75
    assert sections[4].dft.oldCode is True
    assert sections[4].dft.newCode is True
    assert sections[4].dft.correlation == "LYP"
    assert sections[4].dft.exchange == "B88"
    assert sections[4].dft.becke88 is True
    assert sections[4].dft.directives[0].keyword == "HFX"
    assert list(sections[4].dft.directives[0].args) == ["SCREENING"]
    assert sections[5].which() == "set"
    assert sections[5].set.key == "CPMD.MAXSTEP"
    assert sections[5].set.value == "12"
    assert sections[6].which() == "raw"
    assert sections[6].raw == "&VDW\n  DISPERSION\n&END"


def test_make_potential_config_cpmd() -> None:
    cfg = make_potential_config_cpmd(pot_capnp, functional="BLYP")
    assert cfg.which() == "cpmd"
    assert cfg.cpmd.functional == "BLYP"


def test_configure_cpmd() -> None:
    pot = FakePotential()
    ok, message = asyncio.run(configure_cpmd(pot, pot_capnp, functional="PBE"))
    assert ok is True
    assert message == "configured"
    assert pot.config.which() == "cpmd"
    assert pot.config.cpmd.functional == "PBE"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_make_cpmd_params(Path(tmp))
    test_make_cpmd_params_accepts_all_section_arms()
    test_make_potential_config_cpmd()
    test_configure_cpmd()
