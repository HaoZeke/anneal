#!/usr/bin/env python3
# Copyright (c) anneal developers
"""The runner refuses a second scored start, and a row cites its provenance."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "lj104_sota_runner.py"
TABLE = HERE / "mcnemar_threshold.tsv"
SCRIPT = HERE / "mcnemar_threshold.py"
SPEC = importlib.util.spec_from_file_location("lj104_sota_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
RUNNER_MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER_MOD)


def git_out(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(HERE), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


class RunnerGate(unittest.TestCase):
    def test_second_start_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock = str(Path(tmp) / "scored.lock")
            holder = subprocess.Popen(
                [
                    sys.executable,
                    str(RUNNER),
                    "--lock-file",
                    lock,
                    "hold",
                    "--seconds",
                    "30",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert holder.stdout is not None
                self.assertEqual(holder.stdout.readline().strip(), "holding")
                second = subprocess.run(
                    [
                        sys.executable,
                        str(RUNNER),
                        "--lock-file",
                        lock,
                        "emit-row",
                        "--output",
                        str(Path(tmp) / "rows.tsv"),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(second.returncode, 2)
                self.assertIn("another scored run is live", second.stderr)
                self.assertFalse((Path(tmp) / "rows.tsv").exists())
            finally:
                holder.terminate()
                holder.wait(timeout=5)
                if holder.stdout is not None:
                    holder.stdout.close()
                if holder.stderr is not None:
                    holder.stderr.close()

    def test_row_cites_runner_blob_head_and_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rows.tsv"
            lock = str(Path(tmp) / "scored.lock")
            subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--lock-file",
                    lock,
                    "emit-row",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            header, row = output.read_text(encoding="utf-8").splitlines()
            columns = header.split("\t")
            values = dict(zip(columns, row.split("\t")))
            self.assertEqual(
                values["runner_blob"],
                git_out("hash-object", "--", str(RUNNER)),
            )
            self.assertEqual(values["head"], git_out("rev-parse", "HEAD"))
            self.assertEqual(values["tree"], git_out("rev-parse", "HEAD^{tree}"))
            for name in ("runner_blob", "head", "tree"):
                self.assertTrue(values[name])

    def test_threshold_table_matches_the_script(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.stdout, TABLE.read_text(encoding="utf-8"))

    def test_panel_is_the_six_untuned_seeds(self) -> None:
        jobs = RUNNER_MOD.panel_jobs()
        self.assertEqual(len(jobs), 24)
        seeds = {job["seed"] for job in jobs}
        self.assertEqual(seeds, {6000, 6001, 6002, 6003, 6004, 6005})
        for seed in seeds:
            self.assertFalse(5000 <= seed <= 5029)
        sizes = sorted({job["n"] for job in jobs})
        self.assertEqual(sizes, [38, 55, 75, 98])

    def test_held_out_seeds_are_not_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--lock-file",
                    str(Path(tmp) / "scored.lock"),
                    "run",
                    "--code",
                    "candidate",
                    "--seeds",
                    "5000-5029",
                    "--receipt-dir",
                    str(Path(tmp) / "receipts"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 3)
            self.assertIn("seeds 5000-5029 are not run from this commit", proc.stderr)
            self.assertFalse((Path(tmp) / "receipts").exists())


if __name__ == "__main__":
    unittest.main()
