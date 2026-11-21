#!/usr/bin/env python3
"""Build, run, and summarize the scan-safe6 release gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTER = Path(__file__).with_name("scan_safe6_gate_report.inc")
SEEDS = (2000, 2001, 2002)
BASE_COMMIT = "61fa67cf"
CANDIDATE_TREE = "68b35a35c13fd4c5ce683594c744f3b5cb2cad8d"
FIELDS = (
    "cell",
    "system",
    "driver",
    "commit",
    "tree",
    "seed",
    "arm",
    "round",
    "wall",
    "potential",
    "outside",
    "outside_per_call",
    "calls",
    "energy",
    "energy_bits",
    "hash",
    "trajectory",
    "machine_id",
    "core",
    "trace",
)
RUNS = (
    ("lj38", "lj38", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj38")),
    ("lj55", "lj55", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj55")),
    ("lj75", "lj75", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj75")),
    ("lj98", "lj98", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj98")),
    ("lj102", "lj102", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj102")),
    (
        "lj103",
        "lj103",
        "h2h_gmin_proof+lj103",
        "h2h_gmin_proof_lj103",
        ("run", "lj103"),
    ),
    ("lj104", "lj104", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj104")),
    ("lj150", "lj150", "h2h_gmin_proof", "h2h_gmin_proof", ("run", "lj150")),
    ("au55", "au55", "h2h_metal", "h2h_metal", ("run", "au55")),
    ("m80r6", "m80r6", "h2h_morse", "h2h_morse", ("run", "m80r6")),
    ("m80r14", "m80r14", "h2h_morse", "h2h_morse", ("run", "m80r14")),
    ("lj75-seed", "lj75", "h2h_seed", "h2h_seed", ()),
)
SCHEDULE = (
    (1, "A"),
    (2, "B"),
    (3, "B"),
    (4, "A"),
    (5, "A"),
    (6, "C"),
    (7, "C"),
    (8, "A"),
)


def checked(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=True, **kwargs)


def revision(source: Path) -> tuple[str, str]:
    commit = checked(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
    ).stdout.strip()
    tree = checked(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"],
        capture_output=True,
    ).stdout.strip()
    return commit, tree


def instrument(source: Path) -> None:
    shutil.copyfile(REPORTER, source / "examples/scan_safe6_gate_report.inc")
    replacements = {
        "h2h_gmin_proof.rs": (
            "    let (outcome, wall, pot_s) = search(&cfg, spec.n, &start, seed, budget, spec.reference);\n"
            "    let hit =",
            "    let (outcome, wall, pot_s) = search(&cfg, spec.n, &start, seed, budget, spec.reference);\n"
            "    scan_safe6_gate_report(&outcome, wall, pot_s);\n"
            "    let hit =",
        ),
        "h2h_metal.rs": (
            "    let (outcome, wall, pot_s) = search(&cfg, &spec, &start, seed, budget);\n"
            "    if let Ok(path)",
            "    let (outcome, wall, pot_s) = search(&cfg, &spec, &start, seed, budget);\n"
            "    scan_safe6_gate_report(&outcome, wall, pot_s);\n"
            "    if let Ok(path)",
        ),
        "h2h_morse.rs": (
            "    let (outcome, wall, pot_s) = search(&cfg, &spec, &start, seed, budget);\n"
            "    let hit =",
            "    let (outcome, wall, pot_s) = search(&cfg, &spec, &start, seed, budget);\n"
            "    scan_safe6_gate_report(&outcome, wall, pot_s);\n"
            "    let hit =",
        ),
        "h2h_seed.rs": (
            "    let outside_s = (wall - pot_s).max(0.0);\n"
            "    let seen = objective.seen.into_inner().unwrap();",
            "    let outside_s = (wall - pot_s).max(0.0);\n"
            "    scan_safe6_gate_report(&out, wall, pot_s);\n"
            "    let seen = objective.seen.into_inner().unwrap();",
        ),
    }
    for name, (old, new) in replacements.items():
        path = source / "examples" / name
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"instrumentation anchor changed in {name}")
        path.write_text(
            text.replace(old, new) + '\ninclude!("scan_safe6_gate_report.inc");\n'
        )


def build_arm(name: str, source: Path, work: Path, logs: Path) -> Path:
    instrument(source)
    target = work / "build" / name
    output = work / "bin" / name
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    command = [
        "cargo",
        "+1.88.0",
        "build",
        "--locked",
        "--release",
        "--example",
        "h2h_gmin_proof",
        "--example",
        "h2h_metal",
        "--example",
        "h2h_morse",
        "--example",
        "h2h_seed",
    ]
    log = logs / f"build-{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        checked(command, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT)
    examples = target / "release/examples"
    for binary in ("h2h_gmin_proof", "h2h_metal", "h2h_morse", "h2h_seed"):
        shutil.copy2(examples / binary, output / binary)
    shutil.copy2(output / "h2h_gmin_proof", output / "h2h_gmin_proof_lj103")
    return output


def child_env(trace: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[key] = "1"
    for key in list(env):
        if key.startswith("H2H_"):
            del env[key]
    env["SCAN_SAFE6_TRACE"] = str(trace)
    return env


def arguments(
    cell: str,
    system: str,
    driver: str,
    prefix: tuple[str, ...],
    seed: int,
    scratch: Path,
) -> list[str]:
    if driver == "h2h_seed":
        return [
            "lj",
            "75",
            "0",
            "4000000",
            str(seed),
            "recommended",
            repr(-397.492331 + 0.001),
            str(scratch / f"{cell}-{seed}"),
        ]
    return [*prefix, str(seed)]


def read_summary(trace: Path) -> dict[str, object]:
    with trace.open() as handle:
        row = json.loads(next(handle))
    if row.get("kind") != "summary":
        raise SystemExit(f"{trace}: missing summary")
    return row


def append_row(path: Path, row: dict[str, object]) -> None:
    fresh = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_gate(args: argparse.Namespace) -> None:
    sources = {"A": args.base_a, "B": args.candidate, "C": args.base_c}
    revisions = {name: revision(source) for name, source in sources.items()}
    if not revisions["A"][0].startswith(BASE_COMMIT):
        raise SystemExit("arm A is not the registered base")
    if revisions["C"] != revisions["A"]:
        raise SystemExit("the noise arm is not the registered base")
    if revisions["B"][1] != CANDIDATE_TREE:
        raise SystemExit("arm B does not have the registered tree")
    bins = {}
    for name, source in sources.items():
        existing = args.work / "bin" / name
        if all(
            (existing / binary).is_file()
            for binary in (
                "h2h_gmin_proof",
                "h2h_gmin_proof_lj103",
                "h2h_metal",
                "h2h_morse",
                "h2h_seed",
            )
        ):
            bins[name] = existing
        else:
            bins[name] = build_arm(name, source, args.work, args.logs)
    machine_id = hashlib.sha256(platform.node().encode()).hexdigest()
    scratch = args.work / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int, int, str]] = set()
    if args.rows.is_file():
        with args.rows.open(newline="") as handle:
            done = {
                (
                    row["cell"],
                    int(row["seed"]),
                    int(row["round"]),
                    row["arm"],
                )
                for row in csv.DictReader(handle)
            }
    for cell, system, driver, binary, prefix in RUNS:
        for seed in SEEDS:
            for round_number, arm in SCHEDULE:
                key = (cell, seed, round_number, arm)
                if key in done:
                    continue
                trace_rel = Path("runs") / cell / str(seed) / f"{round_number}-{arm}.jsonl"
                trace = args.logs / trace_rel
                trace.parent.mkdir(parents=True, exist_ok=True)
                run_log = trace.with_suffix(".log")
                command = [
                    "taskset",
                    "-c",
                    "0",
                    str(bins[arm] / binary),
                    *arguments(cell, system, driver, prefix, seed, scratch),
                ]
                completed = subprocess.run(
                    command,
                    cwd=sources[arm],
                    env=child_env(trace),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                run_log.write_text(completed.stdout)
                if completed.returncode != 0:
                    raise SystemExit(
                        f"{cell} seed {seed} round {round_number} exited "
                        f"{completed.returncode}"
                    )
                measured = read_summary(trace)
                wall = float(measured["wall"])
                potential = float(measured["potential"])
                calls = int(measured["calls"])
                outside = wall - potential
                if calls <= 0 or outside < 0.0:
                    raise SystemExit(f"invalid timing in {trace_rel}")
                commit, tree = revisions[arm]
                energy_bits = str(measured["energy_bits"])
                energy = struct.unpack(">d", bytes.fromhex(energy_bits))[0]
                append_row(
                    args.rows,
                    {
                        "cell": cell,
                        "system": system,
                        "driver": driver,
                        "commit": commit,
                        "tree": tree,
                        "seed": seed,
                        "arm": arm,
                        "round": round_number,
                        "wall": f"{wall:.9f}",
                        "potential": f"{potential:.9f}",
                        "outside": f"{outside:.9f}",
                        "outside_per_call": f"{outside / calls:.15g}",
                        "calls": calls,
                        "energy": format(energy, ".17g"),
                        "energy_bits": energy_bits,
                        "hash": measured["hash"],
                        "trajectory": measured["trajectory"],
                        "machine_id": machine_id,
                        "core": 0,
                        "trace": trace_rel.as_posix(),
                    },
                )
                print(
                    f"done {cell} {seed} {round_number}{arm} "
                    f"calls={calls} outside={outside:.6f}",
                    flush=True,
                )


def metric(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["energy_bits"],
        row["hash"],
        row["calls"],
        row["trajectory"],
    )


def trace_rows(path: Path) -> list[dict[str, object]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def first_difference(
    left: Path, right: Path
) -> tuple[str, object, object] | None:
    a = trace_rows(left)
    b = trace_rows(right)
    for kind, position in (("transition", "step"), ("improvement", "call")):
        aa = [row for row in a if row["kind"] == kind]
        bb = [row for row in b if row["kind"] == kind]
        for index in range(max(len(aa), len(bb))):
            left_row = aa[index] if index < len(aa) else None
            right_row = bb[index] if index < len(bb) else None
            if left_row != right_row:
                at = (
                    left_row.get(position)
                    if left_row is not None
                    else right_row.get(position)
                )
                return f"{kind} {position} {at}", left_row, right_row
    if a[0] != b[0]:
        differing = sorted(key for key in a[0] if a[0].get(key) != b[0].get(key))
        non_timing = [key for key in differing if key not in ("wall", "potential")]
        if non_timing:
            key = non_timing[0]
            if key == "calls":
                return (
                    f"final call {min(int(a[0][key]), int(b[0][key]))}",
                    a[0][key],
                    b[0][key],
                )
            return key, a[0].get(key), b[0].get(key)
    return None


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> None:
    with args.rows.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = len(RUNS) * len(SEEDS) * len(SCHEDULE)
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} rows, found {len(rows)}")
    identity_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    divergences: list[str] = []
    for cell, _system, driver, _binary, _prefix in RUNS:
        group = [row for row in rows if row["cell"] == cell]
        checks = {
            "energy": True,
            "structure": True,
            "calls": True,
            "trajectory": True,
        }
        columns = {
            "energy": "energy_bits",
            "structure": "hash",
            "calls": "calls",
            "trajectory": "trajectory",
        }
        for seed in SEEDS:
            seeded = [row for row in group if int(row["seed"]) == seed]
            base = [row for row in seeded if row["arm"] == "A"]
            candidate = [row for row in seeded if row["arm"] == "B"]
            if len({metric(row) for row in base}) != 1:
                checks = {name: False for name in checks}
                divergences.append(f"{cell} seed {seed}: repeated base rows differ")
            if len({metric(row) for row in candidate}) != 1:
                checks = {name: False for name in checks}
                divergences.append(f"{cell} seed {seed}: repeated candidate rows differ")
            for name, column in columns.items():
                if base[0][column] != candidate[0][column]:
                    checks[name] = False
            if metric(base[0]) != metric(candidate[0]):
                detail = first_difference(
                    args.logs / base[0]["trace"],
                    args.logs / candidate[0]["trace"],
                )
                divergences.append(
                    f"{cell} seed {seed}: {detail[0] if detail else 'summary'}; "
                    f"A={detail[1] if detail else metric(base[0])}; "
                    f"B={detail[2] if detail else metric(candidate[0])}"
                )
        identity_rows.append(
            {
                "cell": cell,
                "driver": driver,
                "seeds": "2000-2002",
                **{
                    name: "PASS" if passed else "FAIL"
                    for name, passed in checks.items()
                },
                "verdict": "PASS" if all(checks.values()) else "FAIL",
            }
        )
        candidate_base = [
            float(row["outside_per_call"])
            for row in group
            if row["arm"] == "A" and int(row["round"]) in (1, 4)
        ]
        candidate = [
            float(row["outside_per_call"])
            for row in group
            if row["arm"] == "B"
        ]
        noise_base = [
            float(row["outside_per_call"])
            for row in group
            if row["arm"] == "A" and int(row["round"]) in (5, 8)
        ]
        noise = [
            float(row["outside_per_call"])
            for row in group
            if row["arm"] == "C"
        ]
        base_median = statistics.median(candidate_base)
        candidate_median = statistics.median(candidate)
        noise_base_median = statistics.median(noise_base)
        noise_median = statistics.median(noise)
        ratio = candidate_median / base_median
        noise_ratio = noise_median / noise_base_median
        time_rows.append(
            {
                "cell": cell,
                "driver": driver,
                "base_ns_call": f"{base_median * 1e9:.3f}",
                "candidate_ns_call": f"{candidate_median * 1e9:.3f}",
                "candidate_base": f"{ratio:.6f}",
                "noise_base_ns_call": f"{noise_base_median * 1e9:.3f}",
                "noise_ns_call": f"{noise_median * 1e9:.3f}",
                "noise_base": f"{noise_ratio:.6f}",
                "verdict": "PASS" if ratio <= 1.10 else "FAIL",
            }
        )
    identity_path = args.rows.with_name("anneal-scan-safe6-r2-identity.csv")
    time_path = args.rows.with_name("anneal-scan-safe6-r2-time.csv")
    write_csv(
        identity_path,
        ("cell", "driver", "seeds", "energy", "structure", "calls", "trajectory", "verdict"),
        identity_rows,
    )
    write_csv(
        time_path,
        (
            "cell",
            "driver",
            "base_ns_call",
            "candidate_ns_call",
            "candidate_base",
            "noise_base_ns_call",
            "noise_ns_call",
            "noise_base",
            "verdict",
        ),
        time_rows,
    )
    gate1 = all(row["verdict"] == "PASS" for row in identity_rows)
    gate2 = all(row["verdict"] == "PASS" for row in time_rows)
    text = args.report.read_text()
    if "## Results\n" not in text:
        raise SystemExit("report result heading is missing")
    pre_registration = text.split("## Results\n", 1)[0]
    lines = [
        "## Results",
        "",
        f"- Gate 1, bit identity: **{'PASS' if gate1 else 'FAIL'}**",
        f"- Gate 2, outside time per call: **{'PASS' if gate2 else 'FAIL'}**",
        f"- Overall verdict: **{'PASS' if gate1 and gate2 else 'FAIL'}**",
        "",
        "### Gate 1 per-cell results",
        "",
        "| Cell | Driver | Energy | Structure | Calls | Trajectory | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in identity_rows:
        lines.append(
            f"| {row['cell']} | {row['driver']} | {row['energy']} | "
            f"{row['structure']} | {row['calls']} | {row['trajectory']} | "
            f"{row['verdict']} |"
        )
    lines.extend(["", "### First divergences", ""])
    lines.extend(f"- {item}" for item in divergences)
    if not divergences:
        lines.append("No divergence was observed.")
    lines.extend(
        [
            "",
            "### Gate 2 per-cell results",
            "",
            "| Cell | Driver | A ns/call | B ns/call | B/A | "
            "A2 base ns/call | A2 ns/call | A2/A | Verdict |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in time_rows:
        lines.append(
            f"| {row['cell']} | {row['driver']} | {row['base_ns_call']} | "
            f"{row['candidate_ns_call']} | {row['candidate_base']} | "
            f"{row['noise_base_ns_call']} | {row['noise_ns_call']} | "
            f"{row['noise_base']} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "Each timing entry is the pre-registered median of six raw",
            "observations. The complete append-only rows and derived CSV",
            "tables are adjacent to this report.",
            "",
        ]
    )
    args.report.write_text(pre_registration + "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-a", type=Path)
    parser.add_argument("--base-c", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if not args.summarize and not all(
        (args.base_a, args.base_c, args.candidate, args.work)
    ):
        parser.error("run mode needs all three arms and --work")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize(args)
    else:
        run_gate(args)


if __name__ == "__main__":
    main()
