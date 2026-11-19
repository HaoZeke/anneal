# Copyright (c) anneal developers
"""Unit tests for the timing ratio, the paired bootstrap, and the runner."""

from __future__ import annotations

import importlib.util
import itertools
import os
import platform
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STATS = load("stats", HERE / "stats.py")
RUN = load("timing_run", HERE / "run_timing.py")
SUMM = load("timing_summarize", HERE / "summarize.py")


def blank_row(cell: str, arm: str, seed: int, outside: float, charged: int, hit: int = 1) -> dict[str, str]:
    order = STATS.order_for_seed(seed)
    slot = order.index(arm) + 1
    return {
        "session_id": "20261119T000000Z",
        "run_seq": "1",
        "panel": "BLJ",
        "system": cell,
        "driver": "nm3-gate-driver",
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "arm_order": STATS.order_tag(order),
        "budget": "400000",
        "reference": "-1.0",
        "hit": str(hit),
        "evals_to_hit": str(charged),
        "charged": str(charged),
        "best": "-1.0",
        "best_bits_hex": "0xbff0000000000000",
        "wall_s": f"{outside + 1.0:.6f}",
        "pot_s": "1.000000",
        "outside_s": f"{outside:.6f}",
        "outside_per_call_s": f"{outside / charged:.12f}",
        "status": "ok",
        "library": "commit:5eb79cdcb0a986984dc7b487e5a3cc93224246f9",
        "library_tree": "7a2ba49b37241a8fe42b2c45c226374de6d92274",
        "harness_blobs": "bench/nm3-land-gate-b-timing/stats.py:0",
        "trace_sha256": "",
        "xyz_sha256": "",
        "gmin_sha256": "",
        "start_sha256": "",
        "machine_id_sha256": "ab" * 32,
        "utc": "2026-11-19T00:00:00Z",
    }


def panel_rows(cell: str, scale_b: float, scale_a2: float) -> list[dict[str, str]]:
    rows = []
    for seed in STATS.SEEDS:
        rows.append(blank_row(cell, "A", seed, 1.0, 10))
        rows.append(blank_row(cell, "B", seed, scale_b, 10))
        rows.append(blank_row(cell, "A2", seed, scale_a2, 10))
    return rows


class RatioTests(unittest.TestCase):
    def test_declared_constants_and_seeds(self) -> None:
        self.assertEqual(STATS.THRESHOLD, 1.10)
        self.assertEqual(STATS.N_RESAMPLES, 10000)
        self.assertEqual(STATS.RNG_SEED, 12345)
        self.assertEqual(STATS.SEEDS, tuple(range(2100, 2124)))
        self.assertEqual(len(STATS.SEEDS), 24)
        self.assertTrue(set(STATS.SEEDS).isdisjoint(range(2000, 2012)))
        self.assertEqual(STATS.CELLS, ("blj115n38", "blj105n38", "sc126-55"))

    def test_type7_percentile_positions(self) -> None:
        sample = [float(i) for i in range(10000)]
        self.assertAlmostEqual(STATS.type7(sample, 0.025), 249.975)
        self.assertAlmostEqual(STATS.type7(sample, 0.975), 9749.025)
        self.assertEqual(STATS.type7([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_sum_ratio_is_not_the_mean_of_ratios(self) -> None:
        outside_b = [3.0, 3.0]
        charged_b = [1, 3]
        outside_a = [1.0, 1.0]
        charged_a = [1, 1]
        self.assertAlmostEqual(STATS.sum_ratio(outside_b, charged_b, outside_a, charged_a), 1.5)
        self.assertAlmostEqual(
            STATS.median_per_seed_ratio(outside_b, charged_b, outside_a, charged_a),
            2.0,
        )

    def test_bootstrap_is_paired_and_deterministic(self) -> None:
        outside_b = [2.0, 200.0]
        charged_b = [1, 1]
        outside_a = [1.0, 100.0]
        charged_a = [1, 1]
        self.assertAlmostEqual(STATS.sum_ratio(outside_b, charged_b, outside_a, charged_a), 2.0)
        first = STATS.bootstrap_ci(outside_b, charged_b, outside_a, charged_a)
        second = STATS.bootstrap_ci(outside_b, charged_b, outside_a, charged_a)
        self.assertEqual(first, (2.0, 2.0))
        self.assertEqual(first, second)

        outside_b = [1.0, 2.0, 4.0, 8.0]
        outside_a = [1.0, 1.0, 2.0, 2.0]
        charged = [1, 1, 1, 1]
        got = STATS.bootstrap_ci(outside_b, charged, outside_a, charged)
        again = STATS.bootstrap_ci(outside_b, charged, outside_a, charged)
        self.assertEqual(got, again)
        rng = random.Random(STATS.RNG_SEED)
        samples = []
        n = 4
        for _ in range(STATS.N_RESAMPLES):
            idx = [rng.randrange(n) for _ in range(n)]
            samples.append(
                STATS.sum_ratio(
                    [outside_b[i] for i in idx],
                    [charged[i] for i in idx],
                    [outside_a[i] for i in idx],
                    [charged[i] for i in idx],
                )
            )
        self.assertEqual(got, (STATS.type7(samples, 0.025), STATS.type7(samples, 0.975)))

    def test_verdict_boundaries(self) -> None:
        self.assertEqual(STATS.verdict(1.5, 1.10, 1.0, 1.2), "PASS")
        self.assertEqual(STATS.verdict(1.2, 1.11, 1.2, 1.4), "PASS")
        self.assertEqual(STATS.verdict(1.4, 1.11, 1.2, 1.4), "PASS")
        self.assertEqual(STATS.verdict(1.41, 1.11, 1.2, 1.4), "FAIL")
        self.assertEqual(STATS.verdict(1.0, 0.9, 2.0, 3.0), "PASS")

    def test_schedule_is_a_balanced_williams_rotation(self) -> None:
        self.assertEqual(len(set(STATS.PERMUTATIONS)), 6)
        self.assertEqual(
            set(STATS.PERMUTATIONS),
            set(itertools.permutations(("A", "B", "A2"))),
        )
        successors: dict[tuple[str, str], int] = {}
        slots = {slot: {arm: 0 for arm in STATS.ARMS} for slot in (0, 1, 2)}
        for order in STATS.PERMUTATIONS:
            for slot, arm in enumerate(order):
                slots[slot][arm] += 1
            for left, right in zip(order, order[1:]):
                successors[(left, right)] = successors.get((left, right), 0) + 1
        for arm in STATS.ARMS:
            for slot in (0, 1, 2):
                self.assertEqual(slots[slot][arm], 2)
        self.assertEqual(len(successors), 6)
        self.assertTrue(all(count == 2 for count in successors.values()))
        counts: dict[tuple[int, str], int] = {}
        for seed in STATS.SEEDS:
            order = STATS.order_for_seed(seed)
            self.assertEqual(STATS.order_tag(order).count("+"), 2)
            for slot, arm in enumerate(order, start=1):
                counts[(slot, arm)] = counts.get((slot, arm), 0) + 1
        self.assertTrue(all(count == 8 for count in counts.values()))
        self.assertEqual(len(counts), 9)

    def test_constant_ratio_passes_or_fails_as_declared(self) -> None:
        slow = STATS.evaluate_cell(panel_rows("blj115n38", 2.0, 1.0), "blj115n38")
        self.assertAlmostEqual(float(slow["r_ba"]), 2.0)
        self.assertEqual(slow["ba_lo"], 2.0)
        self.assertAlmostEqual(float(slow["r_aa"]), 1.0)
        self.assertEqual(slow["verdict"], "FAIL")
        self.assertEqual(slow["identical"], 24)
        inside = STATS.evaluate_cell(panel_rows("blj105n38", 1.2, 1.2), "blj105n38")
        self.assertAlmostEqual(float(inside["r_ba"]), 1.2)
        self.assertGreater(float(inside["ba_lo"]), 1.10)
        self.assertEqual(inside["verdict"], "PASS")
        mild = STATS.evaluate_cell(panel_rows("sc126-55", 1.05, 1.0), "sc126-55")
        self.assertLessEqual(float(mild["ba_lo"]), 1.10)
        self.assertEqual(mild["verdict"], "PASS")
        self.assertEqual(mild["hits"], {"A": 24, "B": 24, "A2": 24})
        self.assertEqual(mild["charged"], {"A": 240, "B": 240, "A2": 240})

    def test_identity_flags_a_bit_difference(self) -> None:
        rows = panel_rows("blj115n38", 1.0, 1.0)
        for row in rows:
            if row["arm"] == "A2" and row["seed"] == "2100":
                row["best_bits_hex"] = "0x0000000000000000"
        report = STATS.evaluate_cell(rows, "blj115n38")
        self.assertEqual(report["identical"], 23)
        self.assertEqual(report["mismatch_seeds"], [2100])

    def test_sources_omit_machine_identity(self) -> None:
        host = platform.node()
        banned = [
            "/" + "cur" + "sor" + "/",
            "/" + "agent" + "/",
            "Gro" + "k",
            "Cla" + "ude",
            "Cur" + "sor",
            "open" + "ai",
            "Co-authored" + "-by",
            "Signed-off" + "-by",
        ]
        if len(host) > 3:
            banned.append(host)
        for path in HERE.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for word in banned:
                self.assertNotIn(word, text, f"{path.name} contains {word}")


class RunnerTests(unittest.TestCase):
    def test_quick_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            RUN.main(["--quick"])

    @unittest.skipUnless(
        shutil.which("taskset") is not None and Path("/etc/machine-id").is_file(),
        "needs taskset and machine-id",
    )
    def test_runner_records_balanced_rows(self) -> None:
        trees = {RUN.ARM_A_TREE: [], RUN.ARM_B_TREE: []}
        listing = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        for block in listing.stdout.split("\n\n"):
            path = ""
            for line in block.splitlines():
                if line.startswith("worktree "):
                    path = line.split(" ", 1)[1]
            if not path:
                continue
            sha = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD^{tree}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if sha in trees:
                trees[sha].append(path)
        if len(trees[RUN.ARM_A_TREE]) < 2 or len(trees[RUN.ARM_B_TREE]) < 1:
            self.skipTest("arm worktrees are not present")
        panels = {}
        with (ROOT / "bench/nm3-land-gate-b/panels.csv").open(encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                panel, system, _driver, budget, reference = line.strip().split(",")
                if system in STATS.CELLS:
                    panels[system] = (panel, budget, reference)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ("A", "B", "A2"):
                bindir = root / f"bin-{arm}"
                bindir.mkdir()
                for name in ("nm3-gate-driver", "h2h_sc"):
                    script = bindir / name
                    lines = ["#!/bin/sh", 'case "$1" in']
                    lines.append("  budget)")
                    for system, (_panel, budget, reference) in panels.items():
                        lines.append(f'    if [ "$2" = "{system}" ]; then')
                        lines.append(f"      printf '%s\\n%s\\n' '{budget}' '{reference}'")
                        lines.append("      exit 0")
                        lines.append("    fi")
                    lines.append("    exit 2 ;;")
                    lines.append("  coords) : > \"$4\"; exit 0 ;;")
                    lines.append("  energy) printf '1.0\\n'; exit 0 ;;")
                    lines.append("  run)")
                    lines.append(
                        '    echo "budget_env=${H2H_BUDGET:-unset} omp=$OMP_NUM_THREADS '
                        'openblas=$OPENBLAS_NUM_THREADS mkl=$MKL_NUM_THREADS rayon=$RAYON_NUM_THREADS '
                        'seat=$LJOS_SEAT"'
                    )
                    lines.append(
                        "    echo system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s"
                    )
                    lines.append(
                        '    echo "$2,anneal,$3,0,true,10,10,1.000000,-1.0,0.400000,0.600000"'
                    )
                    lines.append("    ;;")
                    lines.append("esac")
                    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    script.chmod(script.stat().st_mode | stat.S_IXUSR)
            # The run row's budget must match the panel. Patch the stub to echo the real budget.
            # A single stub handles every system, so the printed budget is selected in the shell.
            for arm in ("A", "B", "A2"):
                script = root / f"bin-{arm}" / "nm3-gate-driver"
                text = script.read_text(encoding="utf-8")
                replacement = [
                    "  run)",
                    '    echo "budget_env=${H2H_BUDGET:-unset} omp=$OMP_NUM_THREADS openblas=$OPENBLAS_NUM_THREADS mkl=$MKL_NUM_THREADS rayon=$RAYON_NUM_THREADS seat=$LJOS_SEAT"',
                    "    echo system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s",
                ]
                for system, (_panel, budget, _reference) in panels.items():
                    replacement.append(f'    if [ "$2" = "{system}" ]; then')
                    replacement.append(
                        f'      echo "$2,anneal,$3,{budget},true,10,10,1.000000,-1.0,0.400000,0.600000"'
                    )
                    replacement.append("      exit 0")
                    replacement.append("    fi")
                replacement.append("    exit 2 ;;")
                start = text.index("  run)")
                script.write_text(text[:start] + "\n".join(replacement) + "\n" + "esac\n", encoding="utf-8")
                sc = root / f"bin-{arm}" / "h2h_sc"
                sc.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
                sc.chmod(sc.stat().st_mode | stat.S_IXUSR)
            os.environ["H2H_BUDGET"] = "999"
            os.environ["LJOS_SEAT"] = "benchbot"
            rows = root / "rows.csv"
            sessions = root / "sessions.csv"
            bins = root / "bins.csv"
            launcher = root / "launch.py"
            launcher.write_text(
                "import sys\n"
                f"sys.path.insert(0, {str(HERE)!r})\n"
                "import run_timing\n"
                "from argparse import Namespace\n"
                "raise SystemExit(run_timing.cmd_run(Namespace(\n"
                f"    repo={str(ROOT)!r}, cpu='0', panels=None,\n"
                f"    arm_a_bin={str(root / 'bin-A')!r},\n"
                f"    arm_b_bin={str(root / 'bin-B')!r},\n"
                f"    arm_a2_bin={str(root / 'bin-A2')!r},\n"
                f"    arm_a_tree={trees[RUN.ARM_A_TREE][0]!r},\n"
                f"    arm_a2_tree={trees[RUN.ARM_A_TREE][1]!r},\n"
                f"    arm_b_tree={trees[RUN.ARM_B_TREE][0]!r},\n"
                f"    arm_a_library={RUN.ARM_A_LIBRARY!r},\n"
                f"    arm_a2_library={RUN.ARM_A_LIBRARY!r},\n"
                f"    arm_b_library={RUN.ARM_B_LIBRARY!r},\n"
                f"    arm_a_tree_sha={RUN.ARM_A_TREE!r},\n"
                f"    arm_a2_tree_sha={RUN.ARM_A_TREE!r},\n"
                f"    arm_b_tree_sha={RUN.ARM_B_TREE!r},\n"
                f"    work={str(root / 'work')!r},\n"
                f"    logs={str(root / 'logs')!r},\n"
                f"    rows={str(rows)!r},\n"
                f"    sessions={str(sessions)!r},\n"
                f"    bin_sha256={str(bins)!r},\n"
                f"    rgpot_lib={str(HERE / 'run_timing.py')!r},\n"
                ")))\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                ["taskset", "-c", "0", sys.executable, str(launcher)],
                capture_output=True,
                text=True,
                check=False,
                cwd=HERE,
            )
            if proc.returncode != 0:
                self.fail(proc.stderr[-2000:] + proc.stdout[-2000:])
            log = (root / "logs" / "blj115n38-A-s2100.log").read_text(encoding="utf-8")
            self.assertIn("budget_env=unset omp=1 openblas=1 mkl=1 rayon=1 seat=benchbot", log)
            recorded = RUN.run_gate.read_csv(rows)
            self.assertEqual(len(recorded), 3 * 24 * 3)
            self.assertEqual(
                {int(row["seed"]) for row in recorded},
                set(STATS.SEEDS),
            )
            counts = STATS.assert_schedule(recorded)
            self.assertTrue(all(value == 24 for value in counts.values()))
            self.assertEqual({row["session_id"] for row in recorded}, {RUN.run_gate.read_csv(sessions)[0]["session_id"]})
            self.assertTrue(all(len(row["machine_id_sha256"]) == 64 for row in recorded))


if __name__ == "__main__":
    unittest.main()
