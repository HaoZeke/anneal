#!/usr/bin/env python3
"""Time a Birkholz Cope Cartesian stencil on libuma_engine.

The stencil is the gpridge default: the transition-state geometry and
one plus and minus step of 0.01 angstrom along each Cartesian
coordinate (1 + 2 * 3N geometries). The process sets OMP, MKL and
OpenBLAS to each requested thread count before torch is imported.
Each row of the CSV is one measured wall time of the whole stencil.

Example:
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python benchmarks/uma_cope_stencil.py \
      --engine build-uma/CppCore/libuma_engine.so \
      --schema CppCore/rgpot/rpc/Potentials.capnp \
      --package single=/tmp/cope-b0.pt2 \
      --package b8=/tmp/cope-b8.pt2 \
      --repeats 3 \
      --out benchmarks/uma_cope_stencil.csv
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XYZ = ROOT / "benchmarks" / "data" / "birkholz_04_cope_ts.xyz"
DISPLACEMENT_A = 0.01


def _require_threads(threads: int) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        value = os.environ.get(name)
        if value != str(threads):
            raise SystemExit(
                f"{name} must be {threads} for this benchmark (got {value!r})"
            )
    if os.environ.get("TORCH_NUM_THREADS") == "1":
        raise SystemExit("TORCH_NUM_THREADS=1 is not a setting of this benchmark")


def load_geometry(path: Path):
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    n = int(lines[0])
    symbols = []
    pos = []
    for ln in lines[2 : 2 + n]:
        parts = ln.split()
        symbols.append(parts[0])
        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    zmap = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17, "Br": 35}
    numbers = [zmap[s] for s in symbols]
    return np.asarray(pos, dtype=np.float64), np.asarray(numbers, dtype=np.int32)


def stencil(pos: np.ndarray, step: float) -> np.ndarray:
    """Center plus and minus ``step`` on each coordinate. Shape (3N, 1+2*3N)."""
    flat = pos.reshape(-1)
    cols = 2 * flat.size + 1
    out = np.repeat(flat[:, None], cols, axis=1)
    for j in range(flat.size):
        out[j, 2 * j + 1] += step
        out[j, 2 * j + 2] -= step
    return out


def package_metadata(path: Path) -> dict:
    """The exporter's contract, stored as AOTI wrapper metadata."""
    with zipfile.ZipFile(path) as archive:
        name = next(
            item
            for item in archive.namelist()
            if item.endswith("wrapper_metadata.json")
        )
        payload = json.loads(archive.read(name))
    if "batch_max" in payload:
        return payload
    nested = payload.get("aot_inductor.metadata")
    if isinstance(nested, dict):
        return nested
    raise RuntimeError(f"no batch_max in {path}")


def encode_params(
    model: Path, schema: Path, threads: int, charge: int, spin: int
) -> bytes:
    text = (
        f'(modelPath = "{model.resolve()}", taskName = "omol", '
        f'device = "cpu", charge = {charge}, spin = {spin}, '
        f"intraopThreads = {threads}, interopThreads = {threads})"
    )
    proc = subprocess.run(
        ["capnp", "encode", str(schema), "UmaParams"],
        input=text.encode(),
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode() or "capnp encode failed")
    return proc.stdout


class Engine:
    def __init__(
        self,
        library: Path,
        model: Path,
        schema: Path,
        threads: int,
        charge: int,
        spin: int,
    ):
        self.lib = ctypes.CDLL(str(library))
        self.lib.rgpot_engine_create.restype = ctypes.c_void_p
        self.lib.rgpot_engine_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.rgpot_engine_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rgpot_engine_force_batch.restype = ctypes.c_int
        self.lib.rgpot_engine_force_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.rgpot_engine_last_error.restype = ctypes.c_size_t
        self.lib.rgpot_engine_last_error.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        blob = encode_params(model, schema, threads, charge, spin)
        err = ctypes.create_string_buffer(4096)
        self.pot = self.lib.rgpot_engine_create(blob, len(blob), err, len(err))
        if not self.pot:
            raise RuntimeError(err.value.decode() or "rgpot_engine_create failed")

    def last_error(self) -> str:
        needed = self.lib.rgpot_engine_last_error(self.pot, None, 0)
        buf = ctypes.create_string_buffer(max(needed + 1, 1))
        self.lib.rgpot_engine_last_error(self.pot, buf, len(buf))
        return buf.value.decode()

    def force_batch(self, positions: np.ndarray, numbers: np.ndarray, box: np.ndarray):
        """``positions`` is (n_geom, n_atoms, 3). One shared orthorhombic box."""
        n_geom, n_atoms, _ = positions.shape
        pos = np.ascontiguousarray(positions.reshape(n_geom, n_atoms * 3))
        boxes = np.ascontiguousarray(np.repeat(box.reshape(1, 9), n_geom, axis=0))
        forces = np.zeros_like(pos)
        energies = np.zeros(n_geom, dtype=np.float64)
        c_pos = pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_z = numbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        c_f = forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_e = energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_box = boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        rc = self.lib.rgpot_engine_force_batch(
            self.pot,
            n_geom,
            n_atoms,
            c_pos,
            c_z,
            c_f,
            c_e,
            None,
            c_box,
            None,
            None,
        )
        if rc != 0:
            message = self.last_error() or f"uma force_batch returned {rc}"
            raise RuntimeError(message)
        return energies, forces.reshape(n_geom, n_atoms, 3)

    def close(self) -> None:
        if self.pot:
            self.lib.rgpot_engine_destroy(self.pot)
            self.pot = None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--schema", type=Path, required=True)
    p.add_argument(
        "--package",
        action="append",
        required=True,
        help="name=path.pt2, repeated. name 'single' is the batch_max 0 package",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--xyz", type=Path, default=XYZ)
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=1)
    p.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="OMP/MKL/OpenBLAS and torch intra-op thread counts",
    )
    p.add_argument(
        "--molecule",
        default="",
        help="CSV molecule name. Default is the xyz stem.",
    )
    args = p.parse_args()
    if os.environ.get("UMA_STENCIL_WORKER") != "1" and len(args.threads) != 1:
        parts = []
        for threads in args.threads:
            part = args.out.with_name(args.out.stem + f"-t{threads}.csv")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(threads)
            env["MKL_NUM_THREADS"] = str(threads)
            env["OPENBLAS_NUM_THREADS"] = str(threads)
            env["UMA_STENCIL_WORKER"] = "1"
            env.pop("MKL_VERBOSE", None)
            env.pop("TORCH_NUM_THREADS", None)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--engine",
                str(args.engine),
                "--schema",
                str(args.schema),
                "--repeats",
                str(args.repeats),
                "--out",
                str(part),
                "--xyz",
                str(args.xyz),
                "--charge",
                str(args.charge),
                "--spin",
                str(args.spin),
                "--molecule",
                args.molecule,
                "--threads",
                str(threads),
            ]
            for spec in args.package:
                cmd.extend(["--package", spec])
            subprocess.run(cmd, check=True, env=env)
            parts.append(part)
        rows = []
        fieldnames = None
        for part in parts:
            with part.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                rows.extend(reader)
            part.unlink()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}", flush=True)
        return 0
    threads = args.threads[0]
    _require_threads(threads)

    pos, numbers = load_geometry(args.xyz)
    n_atoms = pos.shape[0]
    columns = stencil(pos, DISPLACEMENT_A)
    n_geom = columns.shape[1]
    # (n_geom, n_atoms, 3), molecule recentering is UmaPot's when the
    # package carries a molecular box. The caller's cell is a 40 Å cube,
    # large enough that a cutoff-6 list adds no periodic image.
    frames = np.ascontiguousarray(columns.T.reshape(n_geom, n_atoms, 3))
    box = np.array([40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0])
    molecule = args.molecule or args.xyz.stem
    print(
        f"{molecule} natoms={n_atoms} geometries={n_geom} "
        f"step={DISPLACEMENT_A} threads={threads}",
        flush=True,
    )

    rows = []
    reference = None
    for spec in args.package:
        name, path_s = spec.split("=", 1)
        path = Path(path_s)
        meta = package_metadata(path)
        engine = Engine(
            args.engine, path, args.schema, threads, args.charge, args.spin
        )
        try:
            energies, forces = engine.force_batch(frames, numbers, box)
            if reference is None:
                reference = (energies.copy(), forces.copy(), name)
            else:
                de = float(np.max(np.abs(energies - reference[0])))
                df = float(np.max(np.abs(forces - reference[1])))
                print(
                    f"{name} vs {reference[2]}: max|dE|={de:.3e} max|dF|={df:.3e}",
                    flush=True,
                )
                if de > 1e-3 or df > 1e-3:
                    raise SystemExit(
                        f"{name} disagrees with {reference[2]} "
                        f"(max|dE|={de:.3e} max|dF|={df:.3e})"
                    )
            for repeat in range(args.repeats):
                t0 = time.perf_counter()
                engine.force_batch(frames, numbers, box)
                seconds = time.perf_counter() - t0
                print(f"{name} repeat {repeat} {seconds:.6f} s", flush=True)
                rows.append(
                    {
                        "molecule": molecule,
                        "natoms": n_atoms,
                        "n_geometries": n_geom,
                        "displacement_A": DISPLACEMENT_A,
                        "package": name,
                        "package_file": path.name,
                        "batch_max": int(float(meta["batch_max"])),
                        "batch_packed": int(float(meta.get("batch_packed", 0) or 0)),
                        "export_path": meta.get("export_path", ""),
                        "molecular_box": meta.get("molecular_box", ""),
                        "torch_version": meta.get("torch_version", ""),
                        "seconds": f"{seconds:.6f}",
                        "repeat": repeat,
                        "intraop_threads": threads,
                        "interop_threads": threads,
                        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
                        "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
                        "openblas_num_threads": os.environ["OPENBLAS_NUM_THREADS"],
                    }
                )
        finally:
            engine.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
