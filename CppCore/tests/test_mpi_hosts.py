#!/usr/bin/env python3
"""Exercise explicit MPI hosts, ownership, and rank-local failure recovery."""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time


def run(command: list[str], env: dict[str, str], expected: int = 0) -> str:
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    print(output, end="")
    if (result.returncode == 0) != (expected == 0):
        raise AssertionError(f"{command} returned {result.returncode}")
    return output


def mpi_command(launcher: str, ranks: int, env: dict[str, str]) -> list[str]:
    version = subprocess.run([launcher, "--version"], env=env, capture_output=True,
                             text=True, timeout=10)
    command = [launcher]
    if "Open MPI" in version.stdout + version.stderr or "OpenRTE" in version.stdout + version.stderr:
        # Slurm can grant hardware threads that share a physical core, and
        # CI runners offer fewer slots than the four ranks these tests start.
        # The job's CPU mask supplies placement for these correctness tests.
        command += ["--map-by", "slot:OVERSUBSCRIBE", "--bind-to", "none"]
    return command + ["-n", str(ranks)]


def main() -> None:
    mode = sys.argv[1]
    env = os.environ.copy()
    if mode in {"owned", "borrowed"}:
        mpiexec, executable = sys.argv[2:4]
        output = run([*mpi_command(mpiexec, 4, env), executable, mode], env)
        if mode == "owned":
            for rank in range(4):
                if f"engine exit rank={rank} participants=4" not in output:
                    raise AssertionError("engine exit handler lost its live MPI runtime")
                if f"observer rank={rank} finalized=1" not in output:
                    raise AssertionError("owned MPI runtime was not finalized")
        elif output.count("borrowed contract rank=") != 4:
            raise AssertionError("borrowed communicator checks did not finish on every rank")
        return
    if mode == "force":
        mpiexec, executable, engine = sys.argv[2:5]
        command = [*mpi_command(mpiexec, 2, env), executable, "--engine", engine]
        output = run(command + ["--system", "h2"], env)
        match = re.search(r"energy_ev=([^ ]+) maxabs_f=([^ ]+)", output)
        if not match:
            raise AssertionError("force host did not report physical results")
        positions = [0.0, 0.0, -0.3707, 0.0, 0.0, 0.3707]
        energy = (0.02 + sum(0.5 * value * value for value in positions)) * 27.211386245988
        force = max(abs(0.001 * (i + 1) + 0.01 * value) for i, value in enumerate(positions)) * (27.211386245988 / 0.529177210903)
        if not math.isclose(float(match[1]), energy, rel_tol=0.0, abs_tol=1e-11):
            raise AssertionError("force host energy differs from independent oracle")
        if not math.isclose(float(match[2]), force, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("force host force differs from independent oracle")
        for rank in range(2):
            if f"fixture finalize rank={rank} mpi_live=1 calls=1" not in output:
                raise AssertionError("engine teardown did not precede owned MPI teardown")
        with tempfile.TemporaryDirectory(prefix="rgpot-mpi-host-") as temporary:
            missing = str(Path(temporary) / "missing.xyz")
            output = run(command + ["--geom", missing], env, expected=1)
            if "cannot read geometry" not in output:
                raise AssertionError("root geometry error was not returned collectively")
            geometry = Path(temporary) / "worker-failure.xyz"
            geometry.write_text("1\n1 13 0 0\n")
            output = run(command + ["--geom", str(geometry)], env, expected=1)
            if "rank 1" not in output or "fixture worker force failure" not in output:
                raise AssertionError("worker force error was not returned by the host")
        return
    if mode != "rpc":
        raise ValueError(f"unknown MPI host test mode: {mode}")
    mpiexec, server, client, engine, backend, layout = sys.argv[2:8]
    for key in ("NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE", "CPMDC_LIBRARY", "RGPOT_CPMDC_ENGINE",
                "CP2KC_LIBRARY"):
        env.pop(key, None)
    env["RGPOT_NWCHEM_ENGINE"] = engine
    env["RGPOT_CPMD_ENGINE"] = engine
    env["RGPOT_CP2K_ENGINE"] = engine
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [server, str(port), backend]
    if layout == "mpi":
        command = [*mpi_command(mpiexec, 2, env), *command]
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    raise AssertionError(f"server exited before accepting requests: {process.returncode}")
                try:
                    with socket.create_connection(("localhost", port), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise AssertionError("server did not accept a connection")
                    time.sleep(0.05)
            run([client, f"localhost:{port}", backend, layout], env)
            # The handshake refuses an incompatible protocol before any call.
            refusal = run([client, f"localhost:{port}", backend, layout, "refuse"], env)
            if "handshake refusal" not in refusal:
                raise AssertionError("the client did not report the handshake refusal")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            output.seek(0)
            print(output.read(), end="")


if __name__ == "__main__":
    main()
