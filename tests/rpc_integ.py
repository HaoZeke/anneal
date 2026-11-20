#!/usr/bin/env python3
import asyncio
import argparse
import math
import subprocess
import traceback
import sys
import os
from pathlib import Path
import capnp
import numpy as np

# Load schema relative to this script unless a built effective schema is available.
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_SCHEMA_PATH = Path(SCRIPT_DIR) / "../CppCore/rgpot/rpc/Potentials.capnp"
pot_capnp = None


def load_schema(server_bin):
    candidates = []
    if server_bin:
        server_path = Path(server_bin).resolve()
        if server_path.parent.name == "CppCore":
            build_schema = (
                server_path.parent.parent / "CppCore/rgpot/rpc/Potentials.capnp"
            )
            candidates.append(build_schema)
    candidates.append(SOURCE_SCHEMA_PATH.resolve())
    for candidate in candidates:
        if candidate.exists():
            print(f"Loading RPC schema: {candidate}")
            return capnp.load(str(candidate))
    raise FileNotFoundError(
        f"could not find Potentials.capnp (checked {', '.join(str(c) for c in candidates)})"
    )

try:
    from cpmd_params import configure_cpmd
    from nwchem_params import configure_nwchem, make_potential_config_none
except ImportError:
    sys.path.insert(0, SCRIPT_DIR)
    from cpmd_params import configure_cpmd
    from nwchem_params import configure_nwchem, make_potential_config_none


def potential_config_arm(name):
    """A PotentialConfig with arm `name` set to its schema defaults."""
    cfg = pot_capnp.PotentialConfig.new_message()
    field = pot_capnp.PotentialConfig.schema.fields[name]
    if field.proto.slot.type.which() == "void":
        setattr(cfg, name, None)
    else:
        cfg.init(name)
    return cfg


async def refuse_every_arm(pot, label, applied):
    """configure() with each arm outside `applied` is refused by name, not dropped."""
    arms = list(pot_capnp.PotentialConfig.schema.union_fields)
    for name in arms:
        if name == "none" or name in applied:
            continue
        reply = await pot.configure(potential_config_arm(name))
        assert not reply.ok, f"{label} accepted configure({name}): {reply.message}"
        assert name in reply.message, f"{label} configure({name}) message: {reply.message}"
    none = await pot.configure(potential_config_arm("none"))
    assert none.ok, f"{label} configure(none): {none.message}"
    print(f"{label}: configure refused {len(arms) - 1 - len(applied)} arms by name, none is a no-op")


async def run_client(port):
    # Retry connection a few times to allow server startup
    for _ in range(10):
        try:
            # Create connection (this requires the KJ loop to be active)
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("Failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)

    # Test Case: CuH2 minimal system
    fip = pot_capnp.ForceInput.new_message()

    # Cu at 0,0,0 and H at 1.5, 0, 0 (Approx distance)
    # Using semantic types from your "Pro" schema if you applied it,
    # but based on your recent "simple" checks, we'll stick to the flat list
    # unless you updated the schema.
    # Assuming you kept the struct/flat hybrid from previous steps:
    pos_data = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
    fip.init("pos", len(pos_data))
    for i, p in enumerate(pos_data):
        fip.pos[i] = p

    fip.init("atmnrs", 2)
    fip.atmnrs[0] = 29  # Cu
    fip.atmnrs[1] = 1  # H

    box_data = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
    fip.init("box", len(box_data))
    for i, b in enumerate(box_data):
        fip.box[i] = b

    print("Sending calculation request...")
    # This await requires the KJ loop
    result = await pot.calculate(fip)

    print(f"Received Energy: {result.result.energy}")
    print(f"Received Forces: {list(result.result.forces)}")
    # CuH2 is the Fortran 2018 kernel now, and it agrees with the kernel it
    # replaced to 1e-6 rather than bit for bit -- the same tolerance
    # CuH2PotTest pins. Exact equality here was asserting a guarantee the
    # rewrite never claimed.
    assert math.isclose(result.result.energy, -0.678808, rel_tol=1e-6)
    assert math.isclose(result.result.forces[0], -7.5565221, rel_tol=1e-6)
    assert math.isclose(result.result.forces[3], 7.5565221, rel_tol=1e-6)

    # Basic physical sanity check
    if result.result.energy == 0.0 or np.isnan(result.result.energy):
        print("Error: Energy is zero or NaN")
        return False

    await refuse_every_arm(pot, "CuH2", applied=())
    again = await pot.calculate(fip)
    assert again.result.energy == result.result.energy
    return True


async def run_morse_configure(port):
    """configure(morse) on a Morse server changes the energy it returns."""
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("Morse configure: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)

    defaults = pot_capnp.MorseParams.new_message()
    fip = pot_capnp.ForceInput.new_message()
    fip.init("pos", 6)
    for i, p in enumerate([10.0, 10.0, 10.0, 10.0 + defaults.re, 10.0, 10.0]):
        fip.pos[i] = p
    fip.init("atmnrs", 2)
    fip.atmnrs[0] = 78
    fip.atmnrs[1] = 78
    fip.init("box", 9)
    for i, b in enumerate([40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0]):
        fip.box[i] = b

    before = (await pot.calculate(fip)).result.energy

    cfg = pot_capnp.PotentialConfig.new_message()
    cfg.init("morse").de = 2.0 * defaults.de
    reply = await pot.configure(cfg)
    print(f"Morse configure(morse): ok={reply.ok} msg={reply.message}")
    assert reply.ok, reply.message

    after = (await pot.calculate(fip)).result.energy
    print(f"Morse energy at re: default de {before}, doubled de {after}")
    # At r = re the shifted pair energy is -de d^2, linear in de.
    d = 1.0 - math.exp(-defaults.a * (defaults.cutoff - defaults.re))
    assert math.isclose(after, -2.0 * defaults.de * d * d, rel_tol=1e-12)
    assert math.isclose(after, 2.0 * before, rel_tol=1e-12)

    await refuse_every_arm(pot, "Morse", applied=("morse",))
    assert (await pot.calculate(fip)).result.energy == after
    return True


def effective_schema(server_bin):
    """The schema potserv was built from: vendored plus the local diffs."""
    path = os.path.join(os.path.dirname(server_bin), "rgpot", "rpc", "Potentials.capnp")
    return capnp.SchemaParser().load(path)


async def run_composite_configure(port, schema):
    """configure(composite) on a Composite server sums the terms it names."""
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("Composite configure: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(schema.Potential)

    fip = schema.ForceInput.new_message()
    pos = [0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.0]
    fip.init("pos", len(pos))
    for i, p in enumerate(pos):
        fip.pos[i] = p
    fip.init("atmnrs", 3)
    for i in range(3):
        fip.atmnrs[i] = 1
    fip.init("box", 9)
    for i, b in enumerate([20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0]):
        fip.box[i] = b

    empty = (await pot.calculate(fip)).result
    assert empty.energy == 0.0, empty.energy

    async def configure(terms):
        cfg = schema.PotentialConfig.new_message()
        built = cfg.init("composite").init("terms", len(terms))
        for slot, (name, backend, weight, enabled) in zip(built, terms):
            slot.name = name
            slot.weight = weight
            slot.enabled = enabled
            slot.init("pot").init("builtin").name = backend
        reply = await pot.configure(cfg)
        print(f"Composite configure(composite): ok={reply.ok} msg={reply.message}")
        assert reply.ok, reply.message
        return (await pot.calculate(fip)).result

    lj = await configure([("lj", "lj", 1.0, True)])
    assert lj.energy != 0.0
    cancel = await configure([("lj", "lj", 1.0, True), ("minus", "lj", -1.0, True)])
    assert abs(cancel.energy) <= 1e-12 * abs(lj.energy), cancel.energy
    assert max(abs(f) for f in cancel.forces) <= 1e-12 * max(abs(f) for f in lj.forces)
    scaled = await configure([("lj", "lj", 2.5, True), ("off", "nonesuch", 1.0, False)])
    assert math.isclose(scaled.energy, 2.5 * lj.energy, rel_tol=1e-12)
    for a, b in zip(scaled.forces, lj.forces):
        assert math.isclose(a, 2.5 * b, rel_tol=1e-12, abs_tol=1e-14)

    d2 = await configure([("d2", "d2", 1.0, True)])
    assert d2.energy < 0.0
    both = await configure([("lj", "lj", 1.0, True), ("d2", "d2", 1.0, True)])
    assert math.isclose(both.energy, lj.energy + d2.energy, rel_tol=1e-12)
    undone = await configure(
        [("lj", "lj", 1.0, True), ("d2", "d2", 1.0, True), ("undo", "d2", -1.0, True)]
    )
    assert math.isclose(undone.energy, lj.energy, rel_tol=1e-12)
    for a, b in zip(undone.forces, lj.forces):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
    cpmd = await configure([("cpmd", "d2cpmd", 1.0, True)])
    assert cpmd.energy < 0.0
    swapped = await configure([("undo cpmd", "d2cpmd", -1.0, True), ("d2", "d2", 1.0, True)])
    assert math.isclose(swapped.energy, d2.energy - cpmd.energy, rel_tol=1e-12, abs_tol=1e-15)
    scaled = await configure([("lj", "lj", 2.5, True), ("off", "nonesuch", 1.0, False)])

    cfg = schema.PotentialConfig.new_message()
    bad = cfg.init("composite").init("terms", 1)[0]
    bad.name = "bad"
    bad.init("pot").init("builtin").name = "nonesuch"
    reply = await pot.configure(cfg)
    assert not reply.ok and "'bad'" in reply.message, reply.message
    assert (await pot.calculate(fip)).result.energy == scaled.energy

    for name in schema.PotentialConfig.schema.union_fields:
        if name in ("none", "composite"):
            continue
        arm = schema.PotentialConfig.new_message()
        field = schema.PotentialConfig.schema.fields[name]
        if field.proto.slot.type.which() == "void":
            setattr(arm, name, None)
        else:
            arm.init(name)
        reply = await pot.configure(arm)
        assert not reply.ok, f"Composite accepted configure({name}): {reply.message}"
        assert name in reply.message, reply.message
    print("Composite configure: weights, switches and refusals hold")
    return True


async def run_d2_configure(port, schema, arm="d2"):
    """configure(d2) on a D2 server, or d2cpmd on D2Cpmd, rebuilds the pot."""
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("D2 configure: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(schema.Potential)
    fip = schema.ForceInput.new_message()
    pos = [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 2.9, 0.1, 0.2]
    fip.init("pos", len(pos))
    for i, p in enumerate(pos):
        fip.pos[i] = p
    fip.init("atmnrs", 4)
    for i, z in enumerate([8, 1, 1, 8]):
        fip.atmnrs[i] = z
    fip.init("box", 9)
    for i, b in enumerate([9.0, 0.0, 0.0, 1.0, 9.5, 0.0, 0.0, 0.5, 10.0]):
        fip.box[i] = b

    blyp = (await pot.calculate(fip)).result
    assert blyp.energy < 0.0, blyp.energy
    cfg = schema.PotentialConfig.new_message()
    cfg.init(arm).functional = "pbe"
    reply = await pot.configure(cfg)
    print(f"D2 configure({arm}): ok={reply.ok} msg={reply.message}")
    assert reply.ok, reply.message
    pbe = (await pot.calculate(fip)).result
    assert math.isclose(pbe.energy, blyp.energy * 0.75 / 1.2, rel_tol=1e-12)

    getattr(cfg, arm).functional = "nonesuch"
    reply = await pot.configure(cfg)
    assert not reply.ok and "nonesuch" in reply.message, reply.message
    assert (await pot.calculate(fip)).result.energy == pbe.energy
    for name in schema.PotentialConfig.schema.union_fields:
        if name in ("none", arm):
            continue
        other = schema.PotentialConfig.new_message()
        field = schema.PotentialConfig.schema.fields[name]
        if field.proto.slot.type.which() == "void":
            setattr(other, name, None)
        else:
            other.init(name)
        reply = await pot.configure(other)
        assert not reply.ok, f"{arm} backend accepted configure({name}): {reply.message}"
        assert name in reply.message, reply.message
    print("D2 configure: functional table and refusals hold")
    return True


def server_lists(server_bin, backend):
    """True when potserv's usage line names @p backend."""
    done = subprocess.run([server_bin], capture_output=True, text=True)
    usage = done.stderr.split("Available PotentialTypes:", 1)[-1].splitlines()[0]
    return backend in [name.strip() for name in usage.split(",")]


async def run_d4_configure(port, schema):
    """configure(d4) on a D4 server switches functional and periodicity."""
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("D4 configure: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(schema.Potential)
    fip = schema.ForceInput.new_message()
    pos = [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 2.9, 0.1, 0.2]
    fip.init("pos", len(pos))
    for i, p in enumerate(pos):
        fip.pos[i] = p
    fip.init("atmnrs", 4)
    for i, z in enumerate([8, 1, 1, 8]):
        fip.atmnrs[i] = z
    fip.init("box", 9)
    for i, b in enumerate([40.0, 0.0, 0.0, 5.0, 38.0, 0.0, -3.0, 4.0, 39.0]):
        fip.box[i] = b

    pbe = (await pot.calculate(fip)).result
    assert pbe.energy < 0.0, pbe.energy
    cfg = schema.PotentialConfig.new_message()
    d4 = cfg.init("d4")
    d4.functional = "blyp"
    d4.periodic = True
    reply = await pot.configure(cfg)
    print(f"D4 configure(d4): ok={reply.ok} msg={reply.message}")
    assert reply.ok, reply.message
    blyp = (await pot.calculate(fip)).result
    assert blyp.energy < 0.0 and blyp.energy != pbe.energy, (blyp.energy, pbe.energy)

    cfg.d4.functional = "nonesuch"
    reply = await pot.configure(cfg)
    assert not reply.ok and "d4 arm rejected" in reply.message, reply.message
    assert (await pot.calculate(fip)).result.energy == blyp.energy
    other = schema.PotentialConfig.new_message()
    other.init("d2")
    reply = await pot.configure(other)
    assert not reply.ok and "d2" in reply.message, reply.message
    print("D4 configure: functional, periodicity and refusals hold")
    return True


async def run_rpc_stress_parity(d2_port, composite_port, schema):
    """Periodic D2 and Composite(D2) return the same stress over RPC."""
    async def connect(port):
        for _ in range(10):
            try:
                return await capnp.AsyncIoStream.create_connection(
                    host="localhost", port=port
                )
            except OSError:
                await asyncio.sleep(0.5)
        raise RuntimeError(f"failed to connect to server on port {port}")

    d2 = capnp.TwoPartyClient(await connect(d2_port)).bootstrap().cast_as(schema.Potential)
    comp = capnp.TwoPartyClient(await connect(composite_port)).bootstrap().cast_as(schema.Potential)

    cfg = schema.PotentialConfig.new_message()
    terms = cfg.init("composite").init("terms", 1)
    terms[0].name = "d2"
    terms[0].weight = 1.0
    terms[0].enabled = True
    terms[0].init("pot").init("builtin").name = "d2"
    reply = await comp.configure(cfg)
    assert reply.ok, reply.message

    fip = schema.ForceInput.new_message()
    pos = [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 2.9, 0.1, 0.2]
    fip.init("pos", len(pos))
    for i, p in enumerate(pos):
        fip.pos[i] = p
    fip.init("atmnrs", 4)
    for i, z in enumerate([8, 1, 1, 8]):
        fip.atmnrs[i] = z
    fip.init("box", 9)
    for i, b in enumerate([9.0, 0.0, 0.0, 1.0, 9.5, 0.0, 0.0, 0.5, 10.0]):
        fip.box[i] = b

    direct = (await d2.calculate(fip)).result
    summed = (await comp.calculate(fip)).result
    assert len(direct.stress) == 9
    assert len(summed.stress) == 9
    assert math.isclose(summed.energy, direct.energy, rel_tol=1e-12, abs_tol=1e-14)
    for a, b in zip(summed.forces, direct.forces):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
    for a, b in zip(summed.stress, direct.stress):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
    print("RPC stress parity: Composite(D2) matches D2 direct call")
    return True


async def run_nwchem_smoke(port):
    """Optional smoke: start NWChem server, configure(), skip calculate if no engine."""
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("NWChem smoke: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)

    # configure(none) should succeed
    cfg_none = make_potential_config_none(pot_capnp)
    r0 = await pot.configure(cfg_none)
    print(f"NWChem configure(none): ok={r0.ok} msg={r0.message}")

    ok, msg = await configure_nwchem(
        pot, pot_capnp, basis="sto-3g", theory="scf", scf_type="rhf"
    )
    print(f"NWChem configure(nwchem): ok={ok} msg={msg}")
    # ok may be False without engine; still a valid RPC round-trip
    await refuse_every_arm(pot, "NWChem", applied=("nwchem",))
    print("NWChem smoke: configure RPC completed.")
    return True


async def configure_cpmd_smoke(pot, pot_capnp):
    """Configure a CPMD backend over RPC; calculate may require an engine."""
    cfg_none = make_potential_config_none(pot_capnp)
    r0 = await pot.configure(cfg_none)
    print(f"CPMD configure(none): ok={r0.ok} msg={r0.message}")

    ok, msg = await configure_cpmd(pot, pot_capnp, functional="BLYP", task="gradient")
    print(f"CPMD configure(cpmd): ok={ok} msg={msg}")
    ok_sections, msg_sections = await configure_cpmd(
        pot,
        pot_capnp,
        functional="PBE0",
        task="gradient",
        input_sections=[
            {
                "kind": "generic",
                "name": "PIMD",
                "directives": [{"keyword": "TEMP", "args": ["300"]}],
            },
            {
                "kind": "cpmd",
                "optimizeGeometry": True,
                "molecularDynamics": True,
                "molecularDynamicsCp": True,
                "molecularDynamicsBo": True,
                "molecularDynamicsEh": True,
                "molecularDynamicsPt": True,
                "molecularDynamicsClassical": True,
                "molecularDynamicsFile": "TRAJECTORY.in",
                "convergenceGeometry": 1.0e-4,
                "maxStep": 8,
                "maxIter": 12,
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
            {"kind": "raw", "text": "&VDW\n  DISPERSION\n&END"},
        ],
    )
    print(f"CPMD configure(cpmd sections): ok={ok_sections} msg={msg_sections}")
    print("CPMD smoke: configure RPC completed.")
    return True


async def run_cpmd_smoke(port):
    """Optional smoke: start CPMD server, configure(), skip calculate if no engine."""
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("CPMD smoke: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)
    if not await configure_cpmd_smoke(pot, pot_capnp):
        return False
    await refuse_every_arm(pot, "CPMD", applied=("cpmd",))
    return True


def _force_input(point, unit):
    fip = pot_capnp.ForceInput.new_message()
    fip.init("pos", 3)
    fip.pos[0] = float(point)
    fip.pos[1] = 0.0
    fip.pos[2] = 0.0
    fip.init("atmnrs", 1)
    fip.atmnrs[0] = 1
    fip.init("box", 9)
    for i, b in enumerate([40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0]):
        fip.box[i] = b
    fip.lengthUnit = unit
    fip.energyUnit = "eV"
    return fip


async def run_hvp_unit_smoke(port):
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print("HVP unit smoke: failed to connect to server.")
        return False

    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)
    bohr_to_angstrom = 0.529177210903
    cases = [
        ("bohr", 0.75 / bohr_to_angstrom, 0.28 / bohr_to_angstrom),
        ("nm", 0.075, 0.028),
    ]
    for unit, point, direction in cases:
        fip = _force_input(point, unit)
        req = pot.calculateHvp_request()
        req.fip = fip
        req.init("direction", 3)
        req.direction[0] = float(direction)
        req.direction[1] = 0.0
        req.direction[2] = 0.0
        hvp = np.array((await req.send()).product, dtype=np.float64)
        delta = np.sqrt(np.finfo(np.float64).eps) * (
            1.0 + abs(point) + abs(direction)
        )
        fp = np.array(
            (await pot.calculate(_force_input(point + delta * direction, unit))).result.forces,
            dtype=np.float64,
        )
        fm = np.array(
            (await pot.calculate(_force_input(point - delta * direction, unit))).result.forces,
            dtype=np.float64,
        )
        fd = -(fp - fm) / (2.0 * delta)
        # calculate() and calculateHvp() both return values in caller units.
        scale = np.max(np.abs(np.concatenate([hvp, fd])))
        bar = np.cbrt(np.finfo(np.float64).eps) * (1.0 + scale)
        assert np.max(np.abs(hvp - fd)) <= bar, (
            f"{unit} HVP mismatch: got={hvp[0]} fd={fd[0]} bar={bar}"
        )
    print("HVP unit smoke: bohr and nm finite differences agree.")
    return True


async def run_model_surface_smoke(port, name):
    for _ in range(10):
        try:
            connection = await capnp.AsyncIoStream.create_connection(
                host="localhost", port=port
            )
            break
        except OSError:
            await asyncio.sleep(0.5)
    else:
        print(f"{name} smoke: failed to connect to server.")
        return False
    client = capnp.TwoPartyClient(connection)
    pot = client.bootstrap().cast_as(pot_capnp.Potential)
    fip = pot_capnp.ForceInput.new_message()
    fip.init("box", 9)
    for i, b in enumerate([40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0]):
        fip.box[i] = b
    fip.lengthUnit = "angstrom"
    fip.energyUnit = "eV"
    if name == "DoubleWell":
        pos = [0.7, -0.2, 0.0]
    else:
        pos = [0.7, -0.2, 0.0]
    fip.init("pos", len(pos))
    for i, value in enumerate(pos):
        fip.pos[i] = value
    fip.init("atmnrs", 1)
    fip.atmnrs[0] = 1
    result = await pot.calculate(fip)
    energy = float(result.result.energy)
    forces = np.array(result.result.forces, dtype=np.float64)
    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))
    assert np.max(np.abs(forces)) > np.finfo(np.float64).eps
    print(f"{name} smoke: energy={energy} force0={forces[0]}")
    return True


def main():
    global pot_capnp
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-bin", required=True, help="Path to potserv executable"
    )
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument(
        "--nwchem-smoke",
        action="store_true",
        help="Also run NWChem configure() smoke on a second server port",
    )
    parser.add_argument(
        "--cpmd-smoke",
        action="store_true",
        help="Also run CPMD configure() smoke on a separate server port",
    )
    args = parser.parse_args()
    # Use the schema that potserv was built with, including local diffs.
    pot_capnp = effective_schema(args.server_bin)

    # 1. Start Server
    print(f"Starting server: {args.server_bin} {args.port} CuH2")
    server_proc = subprocess.Popen(
        [args.server_bin, str(args.port), "CuH2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        #  Run Client
        success = asyncio.run(capnp.run(run_client(args.port)))
    except BaseException as e:
        # Kill first, then drain: reading a live server's stderr blocks until
        # it exits, which turned an assertion failure into a job that sat
        # until the six-hour CI timeout. Print the type and traceback too --
        # `{e}` alone is empty for AssertionError.
        print(f"Exception during test: {type(e).__name__}: {e}")
        traceback.print_exc()
        server_proc.kill()
        out, err = server_proc.communicate(timeout=30)
        print("Server stdout:", out.decode(errors="replace")[-4000:])
        print("Server stderr:", err.decode(errors="replace")[-4000:])
        success = False
    finally:
        # Cleanup
        print("Killing server...")
        server_proc.kill()
        server_proc.wait()

    if not success:
        sys.exit(1)
    print("Integration test passed.")

    morse_port = args.port + 3
    print(f"Starting Morse server: {args.server_bin} {morse_port} Morse")
    morse_proc = subprocess.Popen(
        [args.server_bin, str(morse_port), "Morse"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        morse_ok = asyncio.run(capnp.run(run_morse_configure(morse_port)))
    except BaseException as e:
        print(f"Morse configure exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        morse_proc.kill()
        out, err = morse_proc.communicate(timeout=30)
        print("Server stdout:", out.decode(errors="replace")[-4000:])
        print("Server stderr:", err.decode(errors="replace")[-4000:])
        morse_ok = False
    finally:
        morse_proc.kill()
        morse_proc.wait()
    if not morse_ok:
        sys.exit(1)
    print("Morse configure passed.")

    model_port = args.port + 6
    for name in ("DoubleWell", "CoupledProtonWell"):
        print(f"Starting model smoke server: {args.server_bin} {model_port} {name}")
        model_proc = subprocess.Popen(
            [args.server_bin, str(model_port), name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            model_ok = asyncio.run(capnp.run(run_model_surface_smoke(model_port, name)))
        except BaseException as e:
            print(f"{name} smoke exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            model_proc.kill()
            out, err = model_proc.communicate(timeout=30)
            print("Server stdout:", out.decode(errors="replace")[-4000:])
            print("Server stderr:", err.decode(errors="replace")[-4000:])
            model_ok = False
        finally:
            model_proc.kill()
            model_proc.wait()
        if not model_ok:
            sys.exit(1)
        model_port += 1

    hvp_port = args.port + 9
    print(f"Starting HVP unit server: {args.server_bin} {hvp_port} Eckart")
    hvp_proc = subprocess.Popen(
        [args.server_bin, str(hvp_port), "Eckart"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        hvp_ok = asyncio.run(capnp.run(run_hvp_unit_smoke(hvp_port)))
    except BaseException as e:
        print(f"HVP unit smoke exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        hvp_proc.kill()
        out, err = hvp_proc.communicate(timeout=30)
        print("Server stdout:", out.decode(errors="replace")[-4000:])
        print("Server stderr:", err.decode(errors="replace")[-4000:])
        hvp_ok = False
    finally:
        hvp_proc.kill()
        hvp_proc.wait()
    if not hvp_ok:
        sys.exit(1)

    composite_port = args.port + 4
    print(f"Starting Composite server: {args.server_bin} {composite_port} Composite")
    composite_proc = subprocess.Popen(
        [args.server_bin, str(composite_port), "Composite"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        composite_ok = asyncio.run(
            capnp.run(
                run_composite_configure(composite_port, pot_capnp)
            )
        )
    except BaseException as e:
        print(f"Composite configure exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        composite_proc.kill()
        out, err = composite_proc.communicate(timeout=30)
        print("Server stdout:", out.decode(errors="replace")[-4000:])
        print("Server stderr:", err.decode(errors="replace")[-4000:])
        composite_ok = False
    finally:
        composite_proc.kill()
        composite_proc.wait()
    if not composite_ok:
        sys.exit(1)
    print("Composite configure passed.")

    for offset, backend, arm in ((5, "D2", "d2"), (6, "D2Cpmd", "d2cpmd")):
        d2_port = args.port + offset
        print(f"Starting {backend} server: {args.server_bin} {d2_port} {backend}")
        d2_proc = subprocess.Popen(
            [args.server_bin, str(d2_port), backend],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            d2_ok = asyncio.run(
                capnp.run(
                    run_d2_configure(d2_port, pot_capnp, arm)
                )
            )
        except BaseException as e:
            print(f"{backend} configure exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            d2_proc.kill()
            out, err = d2_proc.communicate(timeout=30)
            print("Server stdout:", out.decode(errors="replace")[-4000:])
            print("Server stderr:", err.decode(errors="replace")[-4000:])
            d2_ok = False
        finally:
            d2_proc.kill()
            d2_proc.wait()
        if not d2_ok:
            sys.exit(1)
        print(f"{backend} configure passed.")

    stress_d2_port = args.port + 8
    stress_composite_port = args.port + 9
    print(f"Starting D2 server for stress parity: {args.server_bin} {stress_d2_port} D2")
    stress_d2_proc = subprocess.Popen(
        [args.server_bin, str(stress_d2_port), "D2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(
        f"Starting Composite server for stress parity: {args.server_bin} {stress_composite_port} Composite"
    )
    stress_composite_proc = subprocess.Popen(
        [args.server_bin, str(stress_composite_port), "Composite"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stress_ok = asyncio.run(
            capnp.run(
                run_rpc_stress_parity(stress_d2_port, stress_composite_port, pot_capnp)
            )
        )
    except BaseException as e:
        print(f"Stress parity exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        stress_d2_proc.kill()
        stress_composite_proc.kill()
        d2_out, d2_err = stress_d2_proc.communicate(timeout=30)
        comp_out, comp_err = stress_composite_proc.communicate(timeout=30)
        print("D2 server stdout:", d2_out.decode(errors="replace")[-4000:])
        print("D2 server stderr:", d2_err.decode(errors="replace")[-4000:])
        print("Composite server stdout:", comp_out.decode(errors="replace")[-4000:])
        print("Composite server stderr:", comp_err.decode(errors="replace")[-4000:])
        stress_ok = False
    finally:
        stress_d2_proc.kill()
        stress_d2_proc.wait()
        stress_composite_proc.kill()
        stress_composite_proc.wait()
    if not stress_ok:
        sys.exit(1)
    print("RPC stress parity passed.")

    if server_lists(args.server_bin, "D4"):
        d4_port = args.port + 7
        print(f"Starting D4 server: {args.server_bin} {d4_port} D4")
        d4_proc = subprocess.Popen(
            [args.server_bin, str(d4_port), "D4"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            d4_ok = asyncio.run(
                capnp.run(run_d4_configure(d4_port, pot_capnp))
            )
        except BaseException as e:
            print(f"D4 configure exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            d4_proc.kill()
            out, err = d4_proc.communicate(timeout=30)
            print("Server stdout:", out.decode(errors="replace")[-4000:])
            print("Server stderr:", err.decode(errors="replace")[-4000:])
            d4_ok = False
        finally:
            d4_proc.kill()
            d4_proc.wait()
        if not d4_ok:
            sys.exit(1)
        print("D4 configure passed.")
    else:
        print("D4 configure skipped: potserv built without dftd4.")

    if args.nwchem_smoke:
        nw_port = args.port + 1
        print(f"Starting NWChem smoke server: {args.server_bin} {nw_port} NWChem")
        nw_proc = subprocess.Popen(
            [args.server_bin, str(nw_port), "NWChem"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            nw_ok = asyncio.run(capnp.run(run_nwchem_smoke(nw_port)))
        except Exception as e:
            print(f"NWChem smoke exception: {e}")
            print("Server stderr:")
            print(nw_proc.stderr.read().decode())
            nw_ok = False
        finally:
            nw_proc.kill()
            nw_proc.wait()
        if not nw_ok:
            sys.exit(1)
        print("NWChem smoke passed.")

    if args.cpmd_smoke:
        cp_port = args.port + 1 + int(args.nwchem_smoke)
        print(f"Starting CPMD smoke server: {args.server_bin} {cp_port} CPMD")
        cp_proc = subprocess.Popen(
            [args.server_bin, str(cp_port), "CPMD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            cp_ok = asyncio.run(capnp.run(run_cpmd_smoke(cp_port)))
        except Exception as e:
            print(f"CPMD smoke exception: {e}")
            print("Server stderr:")
            print(cp_proc.stderr.read().decode())
            cp_ok = False
        finally:
            cp_proc.kill()
            cp_proc.wait()
        if not cp_ok:
            sys.exit(1)
        print("CPMD smoke passed.")


if __name__ == "__main__":
    main()
