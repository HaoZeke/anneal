#!/usr/bin/env python3
"""End to end potserv RPC: configure and calculate for NWChem and CPMD.

Needs:
  RGPOT_POTSERV      path to the potserv binary
  RGPOT_NWCHEM_ENGINE / NWCHEMC_LIBRARY  path to libnwchemc (fake or real)
  RGPOT_CPMD_ENGINE  / CPMDC_LIBRARY     path to libcpmdc (fake or real)

The rpctest task and `potctl ci rpc-integ` export the paths and set
`RGPOT_RPC_E2E_REQUIRED`, so the tests run there. Outside that task a
missing binary skips and the reason is printed.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("RGPOT_RPC_E2E_REQUIRED", "").strip() == "1":
    import capnp
else:
    capnp = pytest.importorskip("capnp", reason="pycapnp is not installed")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
_SCHEMA_CACHE: dict[str, object] = {}


def _get_pot_capnp():
    cached = _SCHEMA_CACHE.get("pot_capnp")
    if cached is not None:
        return cached
    tmpdir = tempfile.TemporaryDirectory()
    schema_path = Path(tmpdir.name) / "Potentials.capnp"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "CppCore/rgpot/rpc/schema_ledger.py"),
            "apply",
            "--root",
            str(ROOT),
            "--schema",
            str(ROOT / "CppCore/rgpot/rpc/Potentials.capnp"),
            "--out",
            str(schema_path),
        ],
        check=True,
    )
    loaded = capnp.SchemaParser().load(str(schema_path))
    _SCHEMA_CACHE["tmpdir"] = tmpdir
    _SCHEMA_CACHE["pot_capnp"] = loaded
    return loaded


@pytest.fixture(scope="module")
def pot_capnp_schema():
    return _get_pot_capnp()


sys.path.insert(0, str(SCRIPT_DIR))
from cpmd_params import configure_cpmd  # noqa: E402
from nwchem_params import configure_nwchem, make_potential_config_none  # noqa: E402


_OUTSIDE_TASK = "Outside the rpctest task this case is skipped."


def _missing(reason: str) -> None:
    message = (
        f"{reason} The rpctest task and potctl ci rpc-integ export "
        f"RGPOT_POTSERV and the fake engine libraries. {_OUTSIDE_TASK}"
    )
    if os.environ.get("RGPOT_RPC_E2E_REQUIRED", "").strip() == "1":
        raise RuntimeError(message)
    pytest.skip(message)


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val or not Path(val).exists():
        _missing(f"{name} is unset or is not a potserv binary.")
    return val


def _pick_engine(missing: str, *names: str) -> str:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v and Path(v).exists():
            return v
    joined = ", ".join(names)
    _missing(f"the {missing} engine library is absent ({joined}).")


async def _connect(port: int, retries: int = 20):
    for _ in range(retries):
        try:
            return await capnp.AsyncIoStream.create_connection(
                host="127.0.0.1", port=port
            )
        except OSError:
            await asyncio.sleep(0.25)
    raise RuntimeError(f"failed to connect to potserv on port {port}")


async def _water_force_input(pot_capnp):
    fip = pot_capnp.ForceInput.new_message()
    # O-H-H water-ish (angstrom)
    pos = [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0]
    fip.init("pos", len(pos))
    for i, p in enumerate(pos):
        fip.pos[i] = p
    fip.init("atmnrs", 3)
    fip.atmnrs[0] = 8
    fip.atmnrs[1] = 1
    fip.atmnrs[2] = 1
    box = [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0]
    fip.init("box", len(box))
    for i, b in enumerate(box):
        fip.box[i] = b
    fip.lengthUnit = "angstrom"
    fip.energyUnit = "eV"
    return fip


async def _run_backend(
    potserv: str,
    port: int,
    pot_name: str,
    engine_env: dict[str, str],
    configure_fn,
    pot_capnp,
    orbital_keys: tuple[int, ...] = (),
):
    env = os.environ.copy()
    env.update(engine_env)
    proc = subprocess.Popen(
        [potserv, str(port), pot_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        connection = await _connect(port)
        client = capnp.TwoPartyClient(connection)
        pot = client.bootstrap().cast_as(pot_capnp.Potential)

        r0 = await pot.configure(make_potential_config_none(pot_capnp))
        assert r0.ok, f"configure(none) failed: {r0.message}"

        ok, msg = await configure_fn(pot)
        assert ok, f"configure({pot_name}) failed: {msg}"

        fip = await _water_force_input(pot_capnp)
        result = await pot.calculate(fip)
        energy = float(result.result.energy)
        forces = list(result.result.forces)
        assert np.isfinite(energy), f"non-finite energy: {energy}"
        assert energy != 0.0, "energy is zero (engine likely not loaded)"
        assert len(forces) == 9, f"expected 9 force components, got {len(forces)}"
        assert all(np.isfinite(f) for f in forces), f"non-finite forces: {forces}"
        # At least one non-zero force from fake engines
        assert any(abs(f) > 1e-12 for f in forces), f"all-zero forces: {forces}"
        keyed_forces = []
        for key in orbital_keys:
            fip.hasOrbitalKey = True
            fip.orbitalKey = key
            keyed = await pot.calculate(fip)
            keyed_forces.append(float(keyed.result.forces[0]))
        return energy, forces, keyed_forces
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_nwchem_rpc_configure_and_calculate(pot_capnp_schema=None) -> None:
    potserv = _require_env("RGPOT_POTSERV")
    engine = _pick_engine(
        "NWChem", "NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE", "RGPOT_NWCHEM_ENGINE"
    )
    port = int(os.environ.get("RGPOT_E2E_NWCHEM_PORT", "19101"))
    pot_capnp = pot_capnp_schema if pot_capnp_schema is not None else _get_pot_capnp()

    async def cfg(pot):
        return await configure_nwchem(
            pot,
            pot_capnp,
            basis="sto-3g",
            theory="scf",
            scf_type="rhf",
            engine_path=engine,
        )

    energy, forces, _ = asyncio.run(
        capnp.run(
            _run_backend(
                potserv,
                port,
                "NWChem",
                {
                    "NWCHEMC_LIBRARY": engine,
                    "RGPOT_NWCHEMC_ENGINE": engine,
                    "RGPOT_NWCHEM_ENGINE": engine,
                },
                cfg,
                pot_capnp,
            )
        )
    )
    # Fake nwchemc returns 0.25 Ha -> eV conversion in frontend
    assert energy > 0.0
    print(f"NWChem E2E energy={energy} forces0={forces[0]}")


def test_cpmd_rpc_configure_and_calculate(pot_capnp_schema=None) -> None:
    potserv = _require_env("RGPOT_POTSERV")
    engine = _pick_engine(
        "CPMD", "CPMDC_LIBRARY", "RGPOT_CPMDC_ENGINE", "RGPOT_CPMD_ENGINE"
    )
    port = int(os.environ.get("RGPOT_E2E_CPMD_PORT", "19102"))
    pot_capnp = pot_capnp_schema if pot_capnp_schema is not None else _get_pot_capnp()

    async def cfg(pot):
        return await configure_cpmd(
            pot,
            pot_capnp,
            functional="BLYP",
            task="gradient",
            engine_path=engine,
        )

    energy, forces, keyed_forces = asyncio.run(
        capnp.run(
            _run_backend(
                potserv,
                port,
                "CPMD",
                {
                    "CPMDC_LIBRARY": engine,
                    "RGPOT_CPMDC_ENGINE": engine,
                    "RGPOT_CPMD_ENGINE": engine,
                },
                cfg,
                pot_capnp,
                (7, 19, 7),
            )
        )
    )
    assert energy > 0.0
    assert np.allclose(keyed_forces, [0.012, 0.012, 0.013], rtol=0.0, atol=1e-15)
    print(f"CPMD E2E energy={energy} forces0={forces[0]}")


def _main() -> int:
    failed = False
    for test in (
        test_nwchem_rpc_configure_and_calculate,
        test_cpmd_rpc_configure_and_calculate,
    ):
        try:
            test()
        except pytest.skip.Exception as skipped:
            reason = getattr(skipped, "msg", None) or str(skipped)
            print(reason, file=sys.stderr)
            failed = True
    if failed:
        return 1
    print("test_rpc_e2e_c_abi: all ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
