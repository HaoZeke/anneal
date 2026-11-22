#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the no-builders14 same-runner panel."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

JOURNAL_FIELDS = (
    "session",
    "event",
    "utc",
    "system",
    "seed",
    "block",
    "repeat",
    "arm",
    "slot",
    "detail",
)


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def append_journal(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=JOURNAL_FIELDS, lineterminator="\n"
        )
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def journal(
    path: Path,
    session: str,
    event: str,
    *,
    system: str = "",
    seed: int | str = "",
    block: str = "",
    repeat: int | str = "",
    arm: str = "",
    slot: int | str = "",
    detail: str = "",
) -> None:
    append_journal(
        path,
        {
            "session": session,
            "event": event,
            "utc": utc(),
            "system": system,
            "seed": str(seed),
            "block": block,
            "repeat": str(repeat),
            "arm": arm,
            "slot": str(slot),
            "detail": detail,
        },
    )


def parse_output(text: str) -> tuple[dict[str, str], float, str, float, str]:
    lines = text.splitlines()
    headers = [
        index
        for index, line in enumerate(lines)
        if line.startswith("system,code,seed,")
    ]
    if len(headers) != 1 or headers[0] + 1 >= len(lines):
        raise ValueError("missing unique CSV row")
    header = lines[headers[0]].split(",")
    printed = dict(zip(header, lines[headers[0] + 1].split(",")))
    full = [line for line in lines if line.startswith("energies_full ")]
    if len(full) != 1:
        raise ValueError("missing full-precision energies")
    _, best_text, best_raw, final_text, final_raw = full[0].split()
    best = float(best_text)
    final = float(final_text)
    best_bits = f"{int(best_raw):016x}"
    final_bits = f"{int(final_raw):016x}"
    if common.bits(best) != best_bits or common.bits(final) != final_bits:
        raise ValueError("energy bits do not round-trip")
    return printed, best, best_bits, final, final_bits


def status_sample(pid: int) -> tuple[str, int, int]:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return "gone", 0, 0
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key] = value.strip()

    def kib(key: str) -> int:
        words = values.get(key, "0").split()
        return int(words[0]) if words else 0

    return values.get("State", ""), kib("RssAnon"), kib("VmHWM")


def run_child(
    command: list[str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float, int, int, int]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        proc = subprocess.Popen(command, env=env, stdout=out, stderr=err)
        peak_anon = 0
        peak_hwm = 0
        while True:
            state, anon, hwm = status_sample(proc.pid)
            peak_anon = max(peak_anon, anon)
            peak_hwm = max(peak_hwm, hwm)
            if state == "gone" or state.startswith("Z"):
                break
            time.sleep(0.005)
        _, wait_status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(wait_status)
    return (
        proc.returncode,
        time.monotonic() - started,
        peak_anon,
        int(usage.ru_maxrss),
        peak_hwm,
    )


def arm_plan(seed: int) -> list[tuple[str, str, int, int]]:
    plan = [
        ("main", arm, index, index)
        for index, arm in enumerate(("A", "B", "G", "G", "B", "A"))
    ]
    if seed <= 2005:
        plan += [
            ("aa", arm, index, index + 6)
            for index, arm in enumerate(("A", "A2", "A2", "A"))
        ]
    return plan


def env_for(arm: str, sc_lib: Path) -> dict[str, str]:
    extra = {"RGPOT_SC_LIB": str(sc_lib)}
    if arm == "G":
        extra["H2H_COMPLETION"] = "fitted"
    return common.run_env(extra)


def probe(
    builds: dict[str, Path],
    sc_lib: Path,
) -> None:
    for _family, system, driver, budget, reference in common.CELLS:
        for arm in ("A", "B", "G"):
            target = builds["B" if arm == "G" else arm]
            binary = target / "release" / "examples" / driver
            done = subprocess.run(
                [str(binary), "budget", system],
                env=env_for(arm, sc_lib),
                capture_output=True,
                text=True,
                check=False,
            )
            lines = done.stdout.splitlines()
            if done.returncode != 0 or len(lines) < 2:
                raise SystemExit(f"{system} {arm}: budget probe failed")
            if int(lines[0]) != budget:
                raise SystemExit(f"{system} {arm}: budget differs")
            if abs(float(lines[1]) - reference) > 1.0e-8:
                raise SystemExit(f"{system} {arm}: reference differs")


def copy_partial(rows: Path, handoff: Path) -> None:
    handoff.mkdir(parents=True, exist_ok=True)
    target = handoff / "partial-rows.csv"
    temporary = handoff / "partial-rows.csv.tmp"
    shutil.copyfile(rows, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(target)


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["system"],
        row["seed"],
        row["block"],
        row["repeat"],
        row["arm"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="append", required=True)
    parser.add_argument("--sc-lib", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--core", default="0")
    args = parser.parse_args()

    builds = {
        name: Path(path).resolve()
        for name, path in (item.split("=", 1) for item in args.build)
    }
    if set(builds) != {"A", "A2", "B"}:
        raise SystemExit("builds must name A, A2, and B")
    for arm, target in builds.items():
        actual = (target / "SOURCE_TREE").read_text().strip()
        if actual != common.ARM_TREES[arm]:
            raise SystemExit(f"{arm}: build tree differs")
    if builds["A"] == builds["A2"]:
        raise SystemExit("A and A2 must be independent builds")

    probe(builds, args.sc_lib)
    stored = common.read_rows(args.rows)
    prior_sessions = {row["session"] for row in stored}
    if len(prior_sessions) > 1:
        raise SystemExit("stored rows contain more than one session")
    session = (
        prior_sessions.pop()
        if prior_sessions
        else dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    journal(args.journal, session, "restart", detail="orchestrator start")
    machine = common.machine_id()
    harness = common.blob_of(Path(__file__))
    done = {row_key(row) for row in stored}
    completed_cells = 0

    for family, system, driver, budget, reference in common.CELLS:
        for seed in common.SEEDS:
            for block, arm, repeat, slot in arm_plan(seed):
                key = (system, str(seed), block, str(repeat), arm)
                if key in done:
                    continue
                target = builds["B" if arm == "G" else arm]
                binary = target / "release" / "examples" / driver
                command = [
                    "taskset",
                    "-c",
                    args.core,
                    str(binary),
                    "run",
                    system,
                    str(seed),
                ]
                started_utc = utc()
                journal(
                    args.journal,
                    session,
                    "start",
                    system=system,
                    seed=seed,
                    block=block,
                    repeat=repeat,
                    arm=arm,
                    slot=slot,
                    detail=" ".join(command[:3] + [binary.name, "run", system]),
                )
                stem = f"{seed}-{block}-{repeat}-{arm}"
                stdout_path = args.logs / system / f"{stem}.stdout"
                stderr_path = args.logs / system / f"{stem}.stderr"
                code, elapsed, rss_anon, ru_maxrss, vm_hwm = run_child(
                    command,
                    env_for(arm, args.sc_lib),
                    stdout_path,
                    stderr_path,
                )
                ended_utc = utc()
                status = "ok"
                printed: dict[str, str] = {}
                best = math.nan
                final = math.nan
                best_bits = ""
                final_bits = ""
                if code == 0:
                    try:
                        printed, best, best_bits, final, final_bits = (
                            parse_output(stdout_path.read_text())
                        )
                    except (ValueError, KeyError) as error:
                        status = f"malformed:{error}"
                else:
                    status = f"exit:{code}"

                if status == "ok":
                    if (
                        printed.get("system") != system
                        or int(printed["seed"]) != seed
                        or int(printed["budget"]) != budget
                    ):
                        status = "malformed:identity"
                hit = (
                    status == "ok"
                    and math.isfinite(best)
                    and best <= reference + 1.0e-3
                )
                calls = (
                    printed.get("evals_to_hit", "inf") if hit else "inf"
                )
                charged = printed.get("charged", "")
                wall = printed.get("wall_s", f"{elapsed:.6f}")
                inside = printed.get("pot_s", "")
                outside = printed.get("outside_s", "")
                if status == "ok" and not outside:
                    outside = f"{max(float(wall) - float(inside), 0.0):.6f}"
                driver_hit = printed.get("hit", "").lower() in ("1", "true")
                if status == "ok" and driver_hit != hit:
                    status = "malformed:driver-hit"

                row = {
                    "session": session,
                    "system": system,
                    "family": family,
                    "arm": arm,
                    "seed": str(seed),
                    "block": block,
                    "repeat": str(repeat),
                    "slot": str(slot),
                    "core": args.core,
                    "hit": "1" if hit else "0",
                    "calls_to_hit": calls,
                    "charged": charged,
                    "wall_s": wall,
                    "inside_s": inside,
                    "outside_s": outside,
                    "best": repr(best),
                    "best_bits": best_bits,
                    "final": repr(final),
                    "final_bits": final_bits,
                    "budget": str(budget),
                    "reference": repr(reference),
                    "exit_code": str(code),
                    "arm_tree": common.ARM_TREES[arm],
                    "machine": machine,
                    "rss_anon_kib": str(rss_anon),
                    "ru_maxrss_kib": str(ru_maxrss),
                    "vm_hwm_kib": str(vm_hwm),
                    "harness_blob": harness,
                    "driver_blob": common.blob_of(
                        target / "drivers" / f"{driver}.rs"
                    ),
                    "started_utc": started_utc,
                    "ended_utc": ended_utc,
                    "status": status,
                }
                common.append_row(args.rows, row)
                done.add(key)
                journal(
                    args.journal,
                    session,
                    "end",
                    system=system,
                    seed=seed,
                    block=block,
                    repeat=repeat,
                    arm=arm,
                    slot=slot,
                    detail=f"exit={code} status={status}",
                )
                print(
                    f"{system} seed={seed} arm={arm} block={block} "
                    f"repeat={repeat} exit={code} status={status}",
                    flush=True,
                )
        completed_cells += 1
        if completed_cells % 4 == 0:
            copy_partial(args.rows, args.handoff)

    copy_partial(args.rows, args.handoff)
    journal(args.journal, session, "complete", detail="all planned rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
