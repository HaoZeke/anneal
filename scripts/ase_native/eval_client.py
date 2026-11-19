# Copyright (c) anneal developers
"""Client for the little-endian evaluator process."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT.parent / "tmp" / "anneal-target" / "release" / "examples" / "ase_native_eval"


def binary_path() -> Path:
    return Path(os.environ.get("ASE_NATIVE_EVAL", DEFAULT_BINARY))


class EvalServer:
    """One persistent ``ase_native_eval serve`` process."""

    def __init__(self, system: str, n_atoms: int, binary: Path | None = None):
        self.system = system
        self.n_atoms = n_atoms
        self.binary = binary or binary_path()
        if not self.binary.is_file():
            raise RuntimeError(f"missing evaluator {self.binary}")
        self.proc = subprocess.Popen(
            [str(self.binary), "serve", system],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._out_bytes = 8 * (1 + 3 * n_atoms)

    def evaluate(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode() if self.proc.stderr else ""
            raise RuntimeError(f"evaluator exited {self.proc.returncode}: {err[-500:]}")
        coords = np.ascontiguousarray(positions, dtype="<f8").reshape(-1)
        if coords.size != 3 * self.n_atoms:
            raise RuntimeError("coordinate length does not match the system")
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        payload = memoryview(coords.tobytes())
        sent = 0
        while sent < len(payload):
            wrote = self.proc.stdin.write(payload[sent:])
            if not wrote:
                raise RuntimeError("evaluator stopped reading")
            sent += wrote
        self.proc.stdin.flush()
        raw = bytearray()
        while len(raw) < self._out_bytes:
            chunk = self.proc.stdout.read(self._out_bytes - len(raw))
            if not chunk:
                err = ""
                if self.proc.stderr is not None:
                    err = self.proc.stderr.read().decode(errors="replace")[-500:]
                raise RuntimeError(f"short evaluator reply (status {self.proc.poll()}): {err}")
            raw.extend(chunk)
        values = np.frombuffer(raw, dtype="<f8")
        energy = float(values[0])
        gradient = np.array(values[1:], dtype=np.float64).reshape(self.n_atoms, 3)
        return energy, gradient

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class DmaxServer:
    """Persistent scalar walk for ASE's translated greedy distance."""

    def __init__(self, n_atoms: int, binary: Path | None = None):
        self.n_atoms = n_atoms
        self.binary = binary or binary_path()
        self.proc = subprocess.Popen(
            [str(self.binary), "dmax", str(n_atoms)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._in_bytes = 8 * 6 * n_atoms

    def distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        if self.proc.poll() is not None:
            raise RuntimeError("distance process exited")
        payload = np.ascontiguousarray(
            np.concatenate([np.asarray(pos1, dtype=np.float64).reshape(-1), np.asarray(pos2, dtype=np.float64).reshape(-1)]),
            dtype="<f8",
        )
        if payload.size != 6 * self.n_atoms:
            raise RuntimeError("distance geometry has the wrong length")
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        view = memoryview(payload.tobytes())
        sent = 0
        while sent < len(view):
            wrote = self.proc.stdin.write(view[sent:])
            if not wrote:
                raise RuntimeError("distance process stopped reading")
            sent += wrote
        self.proc.stdin.flush()
        raw = bytearray()
        while len(raw) < 8:
            chunk = self.proc.stdout.read(8 - len(raw))
            if not chunk:
                raise RuntimeError("short distance reply")
            raw.extend(chunk)
        return float(np.frombuffer(raw, dtype="<f8")[0])

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def read_scales(system: str, binary: Path | None = None) -> dict[str, str]:
    path = binary or binary_path()
    proc = subprocess.run(
        [str(path), "scales", system],
        check=True,
        capture_output=True,
        text=True,
    )
    line = proc.stdout.strip().splitlines()[-1]
    return dict(part.split("=", 1) for part in line.split())
