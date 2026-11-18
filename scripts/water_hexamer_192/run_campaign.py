#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Water-hexamer panel rebuild at 192 seeds, serial, append-only rows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

SYSTEM = "h2o6"
PAPER_BUDGETS = {SYSTEM: 4000}
SEEDS = tuple(range(192))
ARMS = ("anneal", "polish_fair", "crest", "crest_fair")
CREST_FAIR_RESERVE = 618
CREST_SEARCH_CAP = PAPER_BUDGETS[SYSTEM] - CREST_FAIR_RESERVE
REFERENCE_EV = -829.852228
HIT_TOL_EV = 0.001

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "calls",
    "wall_s",
    "pot_s",
    "outside_s",
    "best_eV",
    "isomer",
    "code_revision",
    "host",
    "date",
]

SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "hits",
    "hit_rate",
    "calls_median",
    "wall_median_s",
    "pot_median_s",
    "outside_median_s",
    "code_revision",
    "host",
    "date",
]

COMPARE_FIELDS = [
    "system",
    "arm",
    "n_new",
    "n_old",
    "new_hits",
    "old_hits",
    "fisher_hits_p",
    "new_calls_median",
    "old_calls_median",
    "mwu_calls_p",
    "new_wall_median_s",
    "old_wall_median_s",
    "mwu_wall_p",
    "changed_hits",
    "changed_calls",
    "changed_time",
]

SHIFT_FIELDS = [
    "first_arm",
    "second_arm",
    "metric",
    "old_direction",
    "new_direction",
    "old_significant",
    "new_significant",
    "changed",
]


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    h = 1.0 + (len(ordered) - 1) * p
    lo = int(h)
    frac = h - lo
    index = lo - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1 = a + b
    row2 = c + d
    col1 = a + c

    def table_p(x: int) -> float:
        return (
            math.comb(row1, x)
            * math.comb(row2, col1 - x)
            / math.comb(n, col1)
        )

    p0 = table_p(a)
    total = 0.0
    left = max(0, col1 - row2)
    right = min(row1, col1)
    for x in range(left, right + 1):
        px = table_p(x)
        if px <= p0 * (1.0 + 1e-12):
            total += px
    return min(1.0, total)


def mann_whitney_u_two_sided(left: list[float], right: list[float]) -> float:
    n1 = len(left)
    n2 = len(right)
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = sorted(left + right)
    ranks: dict[float, float] = {}
    index = 0
    while index < len(pooled):
        stop = index
        while stop + 1 < len(pooled) and pooled[stop + 1] == pooled[index]:
            stop += 1
        rank = (index + 1 + stop + 1) / 2.0
        ranks[pooled[index]] = rank
        index = stop + 1
    r1 = sum(ranks[value] for value in left)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie_term = 0
    index = 0
    while index < len(pooled):
        stop = index
        while stop + 1 < len(pooled) and pooled[stop + 1] == pooled[index]:
            stop += 1
        t = stop - index + 1
        tie_term += t**3 - t
        index = stop + 1
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma2 <= 0.0:
        return 1.0
    z = (u - mu) / math.sqrt(sigma2)
    return max(0.0, min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))


def env_threads(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
        }
    )
    if extra:
        env.update(extra)
    return env


def parse_kv_lines(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def append_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file() or path.stat().st_size == 0
    if not new_file:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        if header != fields:
            raise SystemExit(f"{path} has another header; rows are append-only")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_rows(rows: list[dict[str, str]], revision: str) -> dict[tuple[int, str], dict[str, str]]:
    out: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        if row.get("code_revision") != revision:
            continue
        key = (int(row["seed"]), row["arm"])
        out[key] = row
    return out


def command_wall(
    command: list[str], *, cwd: Path, env: dict[str, str], out: Path, err: Path
) -> tuple[int, float]:
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with out.open("w", encoding="utf-8") as stdout, err.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
    wall = time.perf_counter() - started
    return completed.returncode, wall


def run_start(tool: Path, template: Path, seed: int, out: Path, core: str, env: dict[str, str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset",
        "-c",
        core,
        str(tool),
        "start",
        "--seed",
        str(seed),
        "--template",
        str(template),
        "--out",
        str(out),
    ]
    subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)


def run_score(
    tool: Path, xyz: Path, isomer_dir: Path, core: str, env: dict[str, str]
) -> dict[str, str]:
    command = [
        "taskset",
        "-c",
        core,
        str(tool),
        "score",
        "--xyz",
        str(xyz),
        "--isomer-dir",
        str(isomer_dir),
    ]
    proc = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    return parse_kv_lines(proc.stdout)


def run_water_arm(
    water_bin: Path,
    crest_bin: Path,
    *,
    arm: str,
    seed: int,
    budget: int,
    template: Path,
    isomer_dir: Path,
    core: str,
    run_dir: Path,
    env: dict[str, str],
) -> dict[str, object]:
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    else:
        run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    geom_dir = run_dir / "geom"
    extra = {
        "WATER_SEED_FIRST": str(seed),
        "WATER_SEED_COUNT": "1",
        "WATER_FORCE_BUDGET": str(budget),
        "WATER_START_XYZ": str(template),
        "WATER_ISOMER_DIR": str(isomer_dir),
        "WATER_GEOM_DIR": str(geom_dir),
    }
    if arm == "polish_fair":
        extra["WATER_POLISH_EXTERNAL"] = "1"
        extra["CREST_BIN"] = str(crest_bin)
    command = ["taskset", "-c", core, str(water_bin), "gfn2", "recommended"]
    rc, wall = command_wall(
        command,
        cwd=run_dir,
        env=env_threads({**env, **extra}),
        out=run_dir / "stdout.txt",
        err=run_dir / "stderr.txt",
    )
    if rc != 0:
        tail = (run_dir / "stdout.txt").read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"{arm} seed {seed} exited {rc}\n{tail}")
    fields = parse_kv_lines((run_dir / "stdout.txt").read_text(encoding="utf-8", errors="replace"))
    if fields.get("hit") not in {"0", "1"}:
        raise RuntimeError(f"{arm} seed {seed} is unscored")
    calls = min(int(fields["force_calls"]), budget)
    return {
        "hit": int(fields["hit"]),
        "calls": calls,
        "wall_s": float(fields["wall_seconds"]),
        "pot_s": float(fields.get("potential_seconds", "0")),
        "outside_s": float(fields.get("outside_seconds", f"{max(wall, 0.0):.6f}")),
        "best_eV": float(fields["best_energy"]),
        "isomer": fields.get("isomer", "unassigned"),
    }


def crest_counter_stats(counter: Path) -> tuple[int, float]:
    if not counter.is_file():
        return 0, 0.0
    total_calls = 0
    inside_s = 0.0
    for line in counter.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 1:
            try:
                total_calls += int(parts[0])
            except ValueError:
                continue
            continue
        if len(parts) < 3:
            continue
        try:
            inside_s += float(parts[0])
            total_calls += int(parts[2])
        except ValueError:
            continue
    return total_calls, inside_s


def run_crest_capped_stage(
    *,
    crest_bin: Path,
    xtb_bin: Path,
    start_xyz: Path,
    budget: int,
    core: str,
    env: dict[str, str],
    stage_dir: Path,
    mode: str,
) -> dict[str, object]:
    if not stage_dir.is_absolute():
        stage_dir = (Path.cwd() / stage_dir).resolve()
    else:
        stage_dir = stage_dir.resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(start_xyz, stage_dir / "start.xyz")
    wrapper = stage_dir / "xtb"
    counter = stage_dir / "xtb-calls.log"
    counter_text = str(counter)
    wrapper.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "read -r started _ < /proc/uptime",
                "out=./xtb-one.out",
                f"\"{xtb_bin}\" \"$@\" >\"$out\" 2>&1",
                "rc=$?",
                "read -r ended _ < /proc/uptime",
                "python3 - \"$out\" \"$started\" \"$ended\" \"$*\" << 'PY'",
                "import re, sys",
                "text = open(sys.argv[1], errors='ignore').read()",
                "started = float(sys.argv[2])",
                "ended = float(sys.argv[3])",
                "args = sys.argv[4]",
                "if '--md' in args or '--omd' in args or '--metadyn' in args:",
                "    found = re.findall(r'(?m)^\\s*max steps\\s*:\\s*(\\d+)', text)",
                "    calls = int(found[-1]) if found else 0",
                "    kind = 'md'",
                "else:",
                "    stars = len(re.findall(r'(?m)^\\s*\\* total energy\\b', text))",
                "    colons = len(re.findall(r'(?m)^\\s*:: total energy\\b', text))",
                "    calls = stars if stars else colons",
                "    kind = 'opt' if '--opt' in args else 'other'",
                f"open({counter_text!r}, 'a', encoding='utf-8').write('{{:.6f}} {{}} {{}}\\n'.format(ended - started, kind, calls))",
                "PY",
                "cat \"$out\"",
                "rm -f \"$out\"",
                "exit $rc",
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    if mode == "search":
        preopt = stage_dir / "preopt"
        preopt.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(start_xyz, preopt / "start.xyz")
        with (preopt / "opt.out").open("w", encoding="utf-8") as stdout:
            subprocess.run(
                ["taskset", "-c", core, str(wrapper), "start.xyz", "--gfn", "2", "--opt"],
                cwd=preopt,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                env=env_threads(env),
                text=True,
                check=False,
            )
        if (preopt / "xtbopt.xyz").is_file():
            shutil.copyfile(preopt / "xtbopt.xyz", stage_dir / "start.xyz")

    if mode == "search":
        crest_cmd = [
            str(crest_bin),
            "start.xyz",
            "--gfn2",
            "-T",
            "1",
            "--legacy",
            "-notopo",
            "-shake",
            "1",
            "--noreftopo",
            "-xnam",
            str(wrapper),
        ]
    elif mode == "polish":
        crest_cmd = [
            str(crest_bin),
            "start.xyz",
            "--gfn2",
            "-T",
            "1",
            "--opt",
            "extreme",
            "-xnam",
            str(wrapper),
        ]
    else:
        raise ValueError(f"unknown mode {mode}")

    started = time.monotonic()
    with (
        (stage_dir / "crest.out").open("w", encoding="utf-8") as stdout,
        (stage_dir / "crest.err").open("w", encoding="utf-8") as stderr,
    ):
        proc = subprocess.Popen(
            ["taskset", "-c", core, *crest_cmd],
            cwd=stage_dir,
            stdout=stdout,
            stderr=stderr,
            env=env_threads(env),
            text=True,
            start_new_session=True,
        )
        capped = False
        while proc.poll() is None:
            calls_now, _ = crest_counter_stats(counter)
            if calls_now >= budget:
                capped = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)
                break
            time.sleep(0.05)
    wall = time.monotonic() - started
    calls, inside_s = crest_counter_stats(counter)
    calls = min(calls, budget)
    inside_s = min(inside_s, wall)
    if not capped and proc.returncode not in (0, None):
        err_tail = (stage_dir / "crest.err").read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"crest {mode} exited {proc.returncode}\n{err_tail}")
    best_xyz = None
    for candidate in (
        stage_dir / "crest_best.xyz",
        stage_dir / "best.xyz",
        stage_dir / "struc.xyz",
        stage_dir / "xtbopt.xyz",
        stage_dir / "start.xyz",
    ):
        if candidate.is_file():
            best_xyz = candidate
            break
    if best_xyz is None:
        raise RuntimeError(f"crest {mode} produced no geometry")
    return {
        "calls": calls,
        "pot_s": inside_s,
        "wall_s": wall,
        "outside_s": max(wall - inside_s, 0.0),
        "best_eV": float("nan"),
        "best_xyz": best_xyz,
        "capped": int(capped),
    }


def run_crest_arm(
    *,
    arm: str,
    seed: int,
    budget: int,
    starts_dir: Path,
    core: str,
    env: dict[str, str],
    run_dir: Path,
    crest_bin: Path,
    xtb_bin: Path,
    tool_bin: Path,
    isomer_dir: Path,
) -> dict[str, object]:
    start_xyz = starts_dir / f"seed-{seed}.xyz"
    if not start_xyz.is_file():
        raise RuntimeError(f"missing start file {start_xyz}")
    if arm == "crest":
        stage = run_crest_capped_stage(
            crest_bin=crest_bin,
            xtb_bin=xtb_bin,
            start_xyz=start_xyz,
            budget=budget,
            core=core,
            env=env,
            stage_dir=run_dir / "crest",
            mode="search",
        )
        score = run_score(tool_bin, Path(stage["best_xyz"]), isomer_dir, core, env_threads(env))
        return {
            "hit": int(score["hit"]),
            "calls": int(stage["calls"]),
            "wall_s": float(stage["wall_s"]),
            "pot_s": float(stage["pot_s"]),
            "outside_s": float(stage["outside_s"]),
            "best_eV": float("nan"),
            "isomer": score.get("isomer", "unassigned"),
        }
    first = run_crest_capped_stage(
        crest_bin=crest_bin,
        xtb_bin=xtb_bin,
        start_xyz=start_xyz,
        budget=CREST_SEARCH_CAP,
        core=core,
        env=env,
        stage_dir=run_dir / "search",
        mode="search",
    )
    second = run_crest_capped_stage(
        crest_bin=crest_bin,
        xtb_bin=xtb_bin,
        start_xyz=Path(first["best_xyz"]),
        budget=CREST_FAIR_RESERVE,
        core=core,
        env=env,
        stage_dir=run_dir / "polish",
        mode="polish",
    )
    best_xyz = Path(second["best_xyz"]) if Path(second["best_xyz"]).is_file() else Path(first["best_xyz"])
    score = run_score(tool_bin, best_xyz, isomer_dir, core, env_threads(env))
    return {
        "hit": int(score["hit"]),
        "calls": int(first["calls"]) + int(second["calls"]),
        "wall_s": float(first["wall_s"]) + float(second["wall_s"]),
        "pot_s": float(first["pot_s"]) + float(second["pot_s"]),
        "outside_s": float(first["outside_s"]) + float(second["outside_s"]),
        "best_eV": float("nan"),
        "isomer": score.get("isomer", "unassigned"),
    }


def old_panel_rows(path: Path) -> dict[str, dict[str, list[float] | list[int]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = set(SEEDS)
    found = {int(row["seed"]) for row in rows}
    if found != expected:
        raise RuntimeError("old panel does not hold seeds 0-191")
    out: dict[str, dict[str, list[float] | list[int]]] = {}
    for arm in ARMS:
        out[arm] = {
            "hits": [int(row[f"{arm}_hit"]) for row in rows],
            "calls": [float(row[f"{arm}_calls"]) for row in rows],
            "wall": [float(row[f"{arm}_wall_s"]) for row in rows],
        }
    return out


def write_progress(progress_path: Path, rows: list[dict[str, object]], started: float) -> None:
    if progress_path.is_file():
        return
    by_arm = {arm: [float(row["wall_s"]) for row in rows if row["arm"] == arm] for arm in ARMS}
    if not all(by_arm[arm] for arm in ARMS):
        return
    elapsed = time.time() - started
    lines = [
        "# First-hour progress",
        "",
        f"time_elapsed_s: {elapsed:.1f}",
        "",
        "| arm | observed_seeds | wall_s_per_seed | projected_total_s |",
        "| --- | ---: | ---: | ---: |",
    ]
    projected_total = 0.0
    for arm in ARMS:
        avg = statistics.mean(by_arm[arm])
        projected = avg * len(SEEDS)
        projected_total += projected
        lines.append(f"| {arm} | {len(by_arm[arm])} | {avg:.6f} | {projected:.1f} |")
    lines.extend(
        [
            "",
            f"projected_panel_total_s: {projected_total:.1f}",
            f"projected_panel_total_h: {projected_total / 3600.0:.3f}",
            "",
        ]
    )
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("\n".join(lines), encoding="utf-8")


def summarize_against_old(
    *,
    rows: list[dict[str, str]],
    revision: str,
    host: str,
    old_rows_path: Path,
    summary_path: Path,
    compare_path: Path,
    shifts_path: Path,
) -> None:
    selected = latest_rows(rows, revision)
    if len(selected) != len(SEEDS) * len(ARMS):
        missing = [
            f"{arm}:{seed}"
            for seed in SEEDS
            for arm in ARMS
            if (seed, arm) not in selected
        ]
        raise RuntimeError(f"missing rows for {len(missing)} seed-arm pairs")

    grouped: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARMS}
    for (_, arm), row in sorted(selected.items()):
        grouped[arm].append(row)

    date = dt.date.today().isoformat()
    summary_rows: list[dict[str, object]] = []
    for arm in ARMS:
        arm_rows = grouped[arm]
        hits = [int(row["hit"]) for row in arm_rows]
        calls = [float(row["calls"]) for row in arm_rows]
        wall = [float(row["wall_s"]) for row in arm_rows]
        pot = [float(row["pot_s"]) for row in arm_rows]
        outside = [float(row["outside_s"]) for row in arm_rows]
        summary_rows.append(
            {
                "system": SYSTEM,
                "arm": arm,
                "n": len(arm_rows),
                "hits": sum(hits),
                "hit_rate": f"{sum(hits) / len(hits):.6f}",
                "calls_median": f"{type7(calls):.3f}",
                "wall_median_s": f"{type7(wall):.6f}",
                "pot_median_s": f"{type7(pot):.6f}",
                "outside_median_s": f"{type7(outside):.6f}",
                "code_revision": revision,
                "host": host,
                "date": date,
            }
        )
    write_rows(summary_path, SUMMARY_FIELDS, summary_rows)

    old = old_panel_rows(old_rows_path)
    compare_rows: list[dict[str, object]] = []
    for arm in ARMS:
        new_hits = [int(row["hit"]) for row in grouped[arm]]
        new_calls = [float(row["calls"]) for row in grouped[arm]]
        new_wall = [float(row["wall_s"]) for row in grouped[arm]]
        old_hits = [int(v) for v in old[arm]["hits"]]
        old_calls = [float(v) for v in old[arm]["calls"]]
        old_wall = [float(v) for v in old[arm]["wall"]]
        nh = sum(new_hits)
        oh = sum(old_hits)
        fisher = fisher_exact_two_sided(
            nh,
            len(new_hits) - nh,
            oh,
            len(old_hits) - oh,
        )
        calls_p = mann_whitney_u_two_sided(new_calls, old_calls)
        wall_p = mann_whitney_u_two_sided(new_wall, old_wall)
        compare_rows.append(
            {
                "system": SYSTEM,
                "arm": arm,
                "n_new": len(new_hits),
                "n_old": len(old_hits),
                "new_hits": nh,
                "old_hits": oh,
                "fisher_hits_p": f"{fisher:.6g}",
                "new_calls_median": f"{type7(new_calls):.3f}",
                "old_calls_median": f"{type7(old_calls):.3f}",
                "mwu_calls_p": f"{calls_p:.6g}",
                "new_wall_median_s": f"{type7(new_wall):.6f}",
                "old_wall_median_s": f"{type7(old_wall):.6f}",
                "mwu_wall_p": f"{wall_p:.6g}",
                "changed_hits": int(fisher < 0.05),
                "changed_calls": int(calls_p < 0.05),
                "changed_time": int(wall_p < 0.05),
            }
        )
    write_rows(compare_path, COMPARE_FIELDS, compare_rows)

    def direction(delta: float, *, lower_is_better: bool) -> str:
        if abs(delta) < 1e-12:
            return "tie"
        if lower_is_better:
            return "first" if delta < 0 else "second"
        return "first" if delta > 0 else "second"

    pair_rows: list[dict[str, object]] = []
    for first, second in [
        ("polish_fair", "anneal"),
        ("crest_fair", "crest"),
        ("anneal", "crest"),
    ]:
        old_h1 = [int(v) for v in old[first]["hits"]]
        old_h2 = [int(v) for v in old[second]["hits"]]
        new_h1 = [int(row["hit"]) for row in grouped[first]]
        new_h2 = [int(row["hit"]) for row in grouped[second]]
        old_hits_p = fisher_exact_two_sided(sum(old_h1), len(old_h1) - sum(old_h1), sum(old_h2), len(old_h2) - sum(old_h2))
        new_hits_p = fisher_exact_two_sided(sum(new_h1), len(new_h1) - sum(new_h1), sum(new_h2), len(new_h2) - sum(new_h2))
        old_hits_dir = direction(sum(old_h1) / len(old_h1) - sum(old_h2) / len(old_h2), lower_is_better=False)
        new_hits_dir = direction(sum(new_h1) / len(new_h1) - sum(new_h2) / len(new_h2), lower_is_better=False)
        pair_rows.append(
            {
                "first_arm": first,
                "second_arm": second,
                "metric": "hits",
                "old_direction": old_hits_dir,
                "new_direction": new_hits_dir,
                "old_significant": int(old_hits_p < 0.05),
                "new_significant": int(new_hits_p < 0.05),
                "changed": int((old_hits_p < 0.05) != (new_hits_p < 0.05) or old_hits_dir != new_hits_dir),
            }
        )

        old_calls_1 = [float(v) for v in old[first]["calls"]]
        old_calls_2 = [float(v) for v in old[second]["calls"]]
        new_calls_1 = [float(row["calls"]) for row in grouped[first]]
        new_calls_2 = [float(row["calls"]) for row in grouped[second]]
        old_calls_p = mann_whitney_u_two_sided(old_calls_1, old_calls_2)
        new_calls_p = mann_whitney_u_two_sided(new_calls_1, new_calls_2)
        old_calls_dir = direction(type7(old_calls_1) - type7(old_calls_2), lower_is_better=True)
        new_calls_dir = direction(type7(new_calls_1) - type7(new_calls_2), lower_is_better=True)
        pair_rows.append(
            {
                "first_arm": first,
                "second_arm": second,
                "metric": "calls",
                "old_direction": old_calls_dir,
                "new_direction": new_calls_dir,
                "old_significant": int(old_calls_p < 0.05),
                "new_significant": int(new_calls_p < 0.05),
                "changed": int((old_calls_p < 0.05) != (new_calls_p < 0.05) or old_calls_dir != new_calls_dir),
            }
        )

        old_wall_1 = [float(v) for v in old[first]["wall"]]
        old_wall_2 = [float(v) for v in old[second]["wall"]]
        new_wall_1 = [float(row["wall_s"]) for row in grouped[first]]
        new_wall_2 = [float(row["wall_s"]) for row in grouped[second]]
        old_wall_p = mann_whitney_u_two_sided(old_wall_1, old_wall_2)
        new_wall_p = mann_whitney_u_two_sided(new_wall_1, new_wall_2)
        old_wall_dir = direction(type7(old_wall_1) - type7(old_wall_2), lower_is_better=True)
        new_wall_dir = direction(type7(new_wall_1) - type7(new_wall_2), lower_is_better=True)
        pair_rows.append(
            {
                "first_arm": first,
                "second_arm": second,
                "metric": "time",
                "old_direction": old_wall_dir,
                "new_direction": new_wall_dir,
                "old_significant": int(old_wall_p < 0.05),
                "new_significant": int(new_wall_p < 0.05),
                "changed": int((old_wall_p < 0.05) != (new_wall_p < 0.05) or old_wall_dir != new_wall_dir),
            }
        )
    write_rows(shifts_path, SHIFT_FIELDS, pair_rows)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    bench_dir = root / "bench" / "water-hexamer-192"
    target_root = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    if not target_root.is_absolute():
        target_root = root / target_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--water-bin",
        type=Path,
        default=target_root / "release" / "examples" / "water_hexamer",
    )
    parser.add_argument(
        "--tool-bin",
        type=Path,
        default=target_root / "release" / "water_hexamer_tools",
    )
    parser.add_argument("--crest-bin", type=Path, default=Path(os.environ.get("CREST_BIN", "crest")))
    parser.add_argument("--xtb-bin", type=Path, default=Path(os.environ.get("XTB_BIN", "xtb")))
    parser.add_argument("--rows", type=Path, default=bench_dir / "rows.csv")
    parser.add_argument("--summary", type=Path, default=bench_dir / "summary.csv")
    parser.add_argument("--compare", type=Path, default=bench_dir / "compare-vs-36086.csv")
    parser.add_argument("--conclusion", type=Path, default=bench_dir / "conclusion-shifts.csv")
    parser.add_argument("--progress", type=Path, default=bench_dir / "progress-first-hour.md")
    parser.add_argument("--inputs", type=Path, default=bench_dir / "inputs")
    parser.add_argument("--work", type=Path, default=Path(os.environ.get("WATER192_WORK", "target-water192/campaign")))
    parser.add_argument("--core", default=os.environ.get("WATER192_CORE", "0"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--old-rows", type=Path, default=root / "paper" / "jctc" / "data" / "water_v45b" / "water-v45b-perseed.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.water_bin = args.water_bin.resolve()
    args.tool_bin = args.tool_bin.resolve()
    args.crest_bin = args.crest_bin.resolve()
    args.xtb_bin = args.xtb_bin.resolve()
    args.work = args.work.resolve()
    args.rows = args.rows.resolve()
    args.summary = args.summary.resolve()
    args.compare = args.compare.resolve()
    args.conclusion = args.conclusion.resolve()
    args.progress = args.progress.resolve()
    args.inputs = args.inputs.resolve()
    args.old_rows = args.old_rows.resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()
    host = os.uname().nodename
    date = dt.date.today().isoformat()
    template = args.inputs / "water-v10.xyz"
    isomer_dir = args.inputs / "isomers"
    starts = args.work / "starts"
    starts.mkdir(parents=True, exist_ok=True)
    env = env_threads({"ASE_PYTHON": "python3"})

    all_rows = read_rows(args.rows)
    done = set(latest_rows(all_rows, revision))
    schedule = (
        [(args.seed, args.arm)]
        if args.seed is not None and args.arm is not None
        else [(seed, arm) for seed in SEEDS for arm in ARMS]
    )
    single_pair = args.seed is not None and args.arm is not None

    started = time.time()
    for seed, arm in schedule:
        if (seed, arm) in done:
            continue
        start_xyz = starts / f"seed-{seed}.xyz"
        if not start_xyz.is_file():
            run_start(args.tool_bin, template, seed, start_xyz, args.core, env)
        run_dir = args.work / "runs" / arm / f"{seed:03d}"
        print(f"start arm={arm} seed={seed}", flush=True)
        if arm in ("anneal", "polish_fair"):
            result = run_water_arm(
                args.water_bin,
                args.crest_bin,
                arm=arm,
                seed=seed,
                budget=PAPER_BUDGETS[SYSTEM],
                template=template,
                isomer_dir=isomer_dir,
                core=args.core,
                run_dir=run_dir,
                env=env,
            )
        else:
            result = run_crest_arm(
                arm=arm,
                seed=seed,
                budget=PAPER_BUDGETS[SYSTEM],
                starts_dir=starts,
                core=args.core,
                env=env,
                run_dir=run_dir,
                crest_bin=args.crest_bin,
                xtb_bin=args.xtb_bin,
                tool_bin=args.tool_bin,
                isomer_dir=isomer_dir,
            )
        row = {
            "system": SYSTEM,
            "arm": arm,
            "seed": seed,
            "budget": PAPER_BUDGETS[SYSTEM],
            "hit": result["hit"],
            "calls": result["calls"],
            "wall_s": f"{float(result['wall_s']):.6f}",
            "pot_s": f"{float(result['pot_s']):.6f}",
            "outside_s": f"{float(result['outside_s']):.6f}",
            "best_eV": f"{float(result['best_eV']):.12f}",
            "isomer": result["isomer"],
            "code_revision": revision,
            "host": host,
            "date": date,
        }
        append_rows(args.rows, ROW_FIELDS, [row])
        all_rows.append({key: str(value) for key, value in row.items()})
        done.add((seed, arm))
        write_progress(args.progress, [dict(r) for r in all_rows if r.get("code_revision") == revision], started)
        print(
            f"done arm={arm} seed={seed} hit={row['hit']} calls={row['calls']} wall_s={row['wall_s']}",
            flush=True,
        )

    if not single_pair:
        summarize_against_old(
            rows=all_rows,
            revision=revision,
            host=host,
            old_rows_path=args.old_rows,
            summary_path=args.summary,
            compare_path=args.compare,
            shifts_path=args.conclusion,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
