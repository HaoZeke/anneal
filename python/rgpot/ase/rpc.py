"""ASE calculator backed by an rgpot Cap'n Proto potential server.

The connection belongs to this calculator (F3). A second thread that
enters :meth:`calculate` or :meth:`close` while the first is inside
raises ``RuntimeError``.

Units follow the Cap'n Proto ``ForceInput`` defaults (Angstrom / eV),
which are ASE's units.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from rgpot.ase.claim import ConnectionClaim

def _capnp() -> Any:
    """Import capnp on first use so the module imports without the package."""
    import capnp  # pyright: ignore[reportMissingImports]

    return capnp


def _schema_path() -> Path:
    here = Path(__file__).resolve().parent / "Potentials.capnp"
    if here.is_file():
        return here
    repo = (
        Path(__file__).resolve().parents[3]
        / "CppCore"
        / "rgpot"
        / "rpc"
        / "Potentials.capnp"
    )
    if repo.is_file():
        return repo
    raise FileNotFoundError("Potentials.capnp is not installed with rgpot.ase")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


class RgpotRpcCalculator(Calculator):
    """ASE calculator that evaluates energy and forces through potserv.

    Parameters
    ----------
    host, port:
        Address of a running potserv instance.
    server_proc:
        Optional managed process this calculator stops on :meth:`close`.
        Prefer :meth:`spawn`.
    """

    implemented_properties = ["energy", "free_energy", "forces"]
    # The connection is a session (F3), not the pure protocol.
    reentrant = False

    def __init__(
        self,
        host: str = "localhost",
        port: int = 12345,
        *,
        server_proc: subprocess.Popen[bytes] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._host = host
        self._port = int(port)
        self._server_proc = server_proc
        self._claim = ConnectionClaim("RgpotRpcCalculator")

        self._schema = _capnp().load(str(_schema_path()))
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._kj = _capnp().kj_loop()
        self._closed = False
        self._pot: Any = None
        self._loop.run_until_complete(self._kj.__aenter__())
        try:
            self._loop.run_until_complete(self._connect())
        except BaseException:
            self.close()
            raise

    @classmethod
    def spawn(
        cls,
        server_bin: str | None,
        potential: str = "CuH2",
        port: int | None = None,
        *,
        startup_timeout: float = 30.0,
        **kwargs: Any,
    ) -> RgpotRpcCalculator:
        """Launch a managed potserv and return a calculator bound to it."""
        del startup_timeout
        if server_bin is None:
            server_bin = os.environ.get("RGPOT_POTSERV_BIN")
        if not server_bin or not Path(server_bin).exists():
            raise FileNotFoundError(
                f"potserv binary not found: {server_bin!r} "
                "(set the path or RGPOT_POTSERV_BIN)"
            )
        chosen = port or _free_port()
        proc: subprocess.Popen[bytes] = subprocess.Popen(
            [str(server_bin), str(chosen), potential],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            return cls(host="localhost", port=chosen, server_proc=proc, **kwargs)
        except Exception:
            proc.kill()
            proc.wait()
            raise

    def close(self) -> None:
        """Disconnect and stop a managed server, if any."""
        with self._claim:
            if not self._closed:
                self._closed = True
                try:
                    self._loop.run_until_complete(self._kj.__aexit__(None, None, None))
                except Exception:
                    pass
                self._loop.close()
            if self._server_proc is not None:
                self._server_proc.kill()
                self._server_proc.wait()
                self._server_proc = None

    def __enter__(self) -> RgpotRpcCalculator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def _connect(self) -> None:
        last_err: OSError | None = None
        for _ in range(120):
            try:
                connection = await _capnp().AsyncIoStream.create_connection(
                    host=self._host, port=self._port
                )
                client = _capnp().TwoPartyClient(connection)
                self._pot = client.bootstrap().cast_as(self._schema.Potential)
                return
            except OSError as exc:
                last_err = exc
                await asyncio.sleep(0.25)
        raise RuntimeError(
            f"RgpotRpcCalculator: could not connect to {self._host}:{self._port}: {last_err}"
        )

    async def _calculate_async(
        self, positions: Any, numbers: Any, box: Any
    ) -> tuple[float, np.ndarray]:
        fip = self._schema.ForceInput.new_message()
        fip.pos = np.asarray(positions, dtype=float).ravel().tolist()
        fip.atmnrs = [int(z) for z in numbers]
        fip.box = np.asarray(box, dtype=float).ravel()[:9].tolist()
        result = await self._pot.calculate(fip)
        energy = float(result.result.energy)
        forces = np.array(result.result.forces, dtype=float).reshape(-1, 3)
        return energy, forces

    def calculate(
        self,
        atoms: Any = None,
        properties: Any = ("energy",),
        system_changes: Any = all_changes,
    ) -> None:
        with self._claim:
            super().calculate(atoms, properties, system_changes)
            frame = self.atoms
            assert frame is not None
            cell = frame.cell
            if cell.rank == 3:
                box = np.asarray(cell.array)
            else:
                box = np.eye(3) * 100.0
            energy, forces = self._loop.run_until_complete(
                self._calculate_async(
                    frame.get_positions(), frame.get_atomic_numbers(), box
                )
            )
            self.results["energy"] = energy
            self.results["free_energy"] = energy
            self.results["forces"] = forces
