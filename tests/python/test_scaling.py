"""Strong scaling of calculate_many.

The batch is 4096 independent Pt7 copies. A second timing is one
periodic EAM-Al cell of at least 10^4 atoms. Python threads are the
BatchEvaluator width. OpenMP is a fresh process per width because
OMP_NUM_THREADS is process-global. The Fortran kernels use
``do concurrent``, so OpenMP is not expected to speed one cell.

Medians are over five interleaved repeats. Bytes of energy and forces
must match across widths.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import rgpot

pytestmark = [
    pytest.mark.thread_unsafe,
    pytest.mark.iterations(1),
    pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no EAM-Al in this build"),
    pytest.mark.skipif(
        "TSAN_OPTIONS" in os.environ,
        reason="ThreadSanitizer cannot finish this timing run inside the pytest timeout",
    ),
]

_REPEATS = 5
_WIDTHS = (1, 2, 4, 8)
_NCOPIES = 4096


def _summarise(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return statistics.median(ordered), ordered[0], ordered[-1]


def test_strong_scaling(tmp_path: Path) -> None:
    from rgpot.ase import RgpotCalculator

    from support import eam_bulk, pt7, raw_bytes

    frames = [pt7(jitter=0.015, seed=1000 + index) for index in range(_NCOPIES)]
    bulk = eam_bulk(10_000)
    assert len(bulk) >= 10_000
    calc = RgpotCalculator("eam_al")

    def batch(atoms_list, threads: int):
        started = time.perf_counter()
        rows = calc.calculate_many(
            atoms_list, properties=("energy", "forces"), threads=threads
        )
        elapsed = time.perf_counter() - started
        digest = tuple(raw_bytes(row["energy"], row["forces"]) for row in rows)
        return elapsed, digest

    def serial():
        started = time.perf_counter()
        rows = []
        for atoms in frames:
            row = calc.__rg_evaluate__(
                atoms.get_positions(),
                atoms.get_atomic_numbers(),
                atoms.cell.array,
                atoms.pbc,
            )
            rows.append(row)
        elapsed = time.perf_counter() - started
        digest = tuple(raw_bytes(row["energy"], row["forces"]) for row in rows)
        return elapsed, digest

    py_samples: dict[int, list[float]] = {width: [] for width in _WIDTHS}
    bulk_samples: dict[int, list[float]] = {width: [] for width in _WIDTHS}
    serial_samples: list[float] = []
    digest_ref = None
    bulk_ref = None
    for _repeat in range(_REPEATS):
        elapsed, digest = serial()
        serial_samples.append(elapsed)
        if digest_ref is None:
            digest_ref = digest
        assert digest == digest_ref
        for width in _WIDTHS:
            elapsed, digest = batch(frames, width)
            py_samples[width].append(elapsed)
            assert digest == digest_ref
            elapsed, digest = batch([bulk], width)
            bulk_samples[width].append(elapsed)
            if bulk_ref is None:
                bulk_ref = digest
            assert digest == bulk_ref

    omp_samples = _omp_scaling(tmp_path)
    lines = [
        f"interpreter {sys.version.split()[0]}",
        f"pt7 copies {_NCOPIES}",
        f"eam bulk atoms {len(bulk)}",
        "variant median_s min_s max_s efficiency bytes",
    ]
    serial_median, serial_min, serial_max = _summarise(serial_samples)
    lines.append(
        "serial(pre-batch N calls) "
        f"{serial_median:.6f} {serial_min:.6f} {serial_max:.6f} n/a match"
    )
    base_median = _summarise(py_samples[1])[0]
    for width in _WIDTHS:
        median, low, high = _summarise(py_samples[width])
        efficiency = (base_median / median) / width if median > 0 else 0.0
        lines.append(
            f"python-threads-{width} {median:.6f} {low:.6f} {high:.6f} "
            f"{efficiency:.4f} match"
        )
    bulk_base = _summarise(bulk_samples[1])[0]
    for width in _WIDTHS:
        median, low, high = _summarise(bulk_samples[width])
        efficiency = (bulk_base / median) / width if median > 0 else 0.0
        lines.append(
            f"bulk-threads-{width} {median:.6f} {low:.6f} {high:.6f} "
            f"{efficiency:.4f} match"
        )
    for width, values in omp_samples.items():
        median, low, high = _summarise(values)
        base = _summarise(omp_samples[1])[0]
        efficiency = (base / median) / width if median > 0 else 0.0
        lines.append(
            f"omp-threads-{width} {median:.6f} {low:.6f} {high:.6f} "
            f"{efficiency:.4f} match"
        )
    py8 = _summarise(py_samples[8])[0]
    speedup = base_median / py8 if py8 > 0 else 0.0
    lines.append(
        "limiter A Pt7 EAM call is a few microseconds. "
        "Reading positions, numbers, and the cell in Python is the same "
        "order of cost as the kernel, so 4096 copies do not speed up with "
        "more threads. "
        f"Python-thread speedup at 8 is {speedup:.3f}. "
        "The bulk cell is one configuration, so BatchEvaluator keeps it "
        "on one worker and extra threads do not split it. "
        "A vesin neighbour list with n_threads=1 does not take the "
        "process-global pool lock. "
        "OpenMP does not enter the kernel: the Fortran ports use "
        "do concurrent, not omp parallel."
    )
    text = "\n".join(lines) + "\n"
    (tmp_path / "scaling.txt").write_text(text, encoding="utf-8")
    print(text)


def _omp_scaling(tmp_path: Path) -> dict[int, list[float]]:
    worker = tmp_path / "omp_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import os, sys, time
            sys.path.insert(0, os.environ["RGPOT_SUPPORT"])
            from support import eam_bulk, pt7, raw_bytes
            from rgpot.ase import RgpotCalculator
            frames = [pt7(jitter=0.015, seed=1000 + index) for index in range(4096)]
            bulk = eam_bulk(10_000)
            calc = RgpotCalculator("eam_al")
            calc.calculate_many(frames[:1], threads=1)
            started = time.perf_counter()
            rows = calc.calculate_many(frames, threads=1)
            elapsed = time.perf_counter() - started
            bulk_rows = calc.calculate_many([bulk], threads=1)
            digest = b"".join(raw_bytes(row["energy"], row["forces"]) for row in rows)
            digest += b"".join(
                raw_bytes(row["energy"], row["forces"]) for row in bulk_rows
            )
            print(f"{elapsed:.9f} {digest.hex()}")
            """
        ).lstrip(),
        encoding="utf-8",
    )
    support = str(Path(__file__).resolve().parent)
    source = str(Path(__file__).resolve().parents[2] / "python")
    samples: dict[int, list[float]] = {width: [] for width in _WIDTHS}
    reference = None
    for _repeat in range(_REPEATS):
        for width in _WIDTHS:
            env = os.environ.copy()
            inherited = [
                part
                for part in env.get("PYTHONPATH", "").split(os.pathsep)
                if part and part != source
            ]
            env["PYTHONPATH"] = os.pathsep.join([support, *inherited])
            env["OMP_NUM_THREADS"] = str(width)
            env["RGPOT_SUPPORT"] = support
            proc = subprocess.run(
                [sys.executable, str(worker)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            assert proc.returncode == 0, proc.stderr
            line = [item for item in proc.stdout.splitlines() if item.strip()][-1]
            elapsed_s, digest = line.split()
            samples[width].append(float(elapsed_s))
            if reference is None:
                reference = digest
            assert digest == reference
    return samples
