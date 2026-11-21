#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial A/B gate for the no-motif SOAP patch."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_matches_reference  # noqa: E402

ROW_FIELDS = [
    "session",
    "panel",
    "system",
    "seed",
    "arm",
    "arm_order",
    "binary",
    "budget",
    "reference",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
    "utc",
]

UNSUPPORTED_FIELDS = ["panel", "system", "reason"]
DECLARED_FIELDS = ["panel", "system", "n", "reason"]
COMMAND_FIELDS = ["panel", "system", "source", "binary", "seed0", "n", "signature"]
SUPERSEDED_FIELDS = ["utc", "panel", "system", "restart_session", "reason"]

NOTES_MORSE_SYSTEMS = ("m55r6", "m55r14", "m80r6", "m80r14")
NOTES_MORSE_BUDGETS = {"m55r6": 1_000_000, "m55r14": 1_000_000, "m80r6": 4_000_000, "m80r14": 4_000_000}
NOTES_LJ_SYSTEMS = ("lj38", "lj55", "lj75", "lj98", "lj103", "lj104")
NOTES_AU_SYSTEMS = ("au55",)
NOTES_AU_BUDGETS = {"au55": 1_000_000}
NOTES_H2H_UNSET = (
    "H2H_BUDGET",
    "H2H_PRIORS",
    "H2H_GENERAL",
    "H2H_REORDER",
    "H2H_SYSTEM",
    "H2H_SEED",
    "H2H_TARGET",
    "H2H_TOL",
    "H2H_ROW",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        tail = (proc.stdout + "\n" + proc.stderr)[-2000:]
        raise RuntimeError(f"{' '.join(command)} failed with {proc.returncode}\n{tail}")
    return proc


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_session() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def single_thread_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[key] = "1"
    if extra:
        env.update(extra)
    return env


def notes_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = single_thread_env(extra)
    for key in NOTES_H2H_UNSET:
        env.pop(key, None)
    return env


def discover_cargo() -> str:
    cargo = os.environ.get("CARGO")
    if cargo and Path(cargo).exists():
        return cargo
    for candidate in ("/usr/local/cargo/bin/cargo", str(Path.home() / ".cargo" / "bin" / "cargo"), shutil.which("cargo") or ""):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("cargo not found")


def build_arm(tree: Path, build_root: Path) -> dict[str, object]:
    head = run(["git", "-C", str(tree), "rev-parse", "HEAD"]).stdout.strip()
    build_root = build_root.resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    target = build_root / "target"
    tmp = build_root / "tmp"
    target.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cargo_home = os.environ.get("CARGO_HOME", "")
    common_env = os.environ.copy()
    common_env["CARGO_TARGET_DIR"] = str(target)
    common_env["TMPDIR"] = str(tmp)
    common_env["RUSTFLAGS"] = (
        f"--remap-path-prefix={tree}=/src "
        f"--remap-path-prefix={str(Path.home())}=/user "
        f"--remap-path-prefix={cargo_home}=/cargo"
    )
    cargo = discover_cargo()
    common_env["PATH"] = str(Path(cargo).parent) + os.pathsep + common_env.get("PATH", "")
    run(
        [
            cargo,
            "+1.88.0",
            "build",
            "--release",
            "--locked",
            "-j",
            "1",
            "--example",
            "h2h_gmin_proof",
            "--example",
            "h2h_metal",
            "--example",
            "h2h_morse",
        ],
        cwd=tree,
        env=common_env,
    )
    run(
        [cargo, "+1.88.0", "build", "--release", "--locked", "-j", "1", "--features", "rgpot-ex", "--example", "h2h_sc"],
        cwd=tree,
        env=common_env,
    )
    bins = build_root / "bin"
    bins.mkdir(parents=True, exist_ok=True)
    names = ("h2h_gmin_proof", "h2h_metal", "h2h_morse", "h2h_sc")
    for name in names:
        src = target / "release" / "examples" / name
        dst = bins / name
        dst.write_bytes(src.read_bytes())
        dst.chmod(0o755)
    return {"tree": tree, "revision": head, "bins": {name: bins / name for name in names}}


def parse_budget(proc: subprocess.CompletedProcess[str]) -> tuple[int, float]:
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"budget output malformed:\n{proc.stdout}\n{proc.stderr}")
    return int(lines[0]), float(lines[1])


def parse_row(
    stdout: str, system: str, seed: int, budget: int, reference: float
) -> dict[str, str]:
    headers = [line for line in stdout.splitlines() if line.startswith("system,")]
    if not headers:
        raise RuntimeError(f"{system} seed {seed}: no CSV header in output")
    header = headers[-1].split(",")
    rows = [line for line in stdout.splitlines() if f"{system},anneal,{seed}," in line]
    if len(rows) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one row, saw {len(rows)}")
    values = rows[0].split(",")
    row = dict(zip(header, values))
    if row.get("system") != system or int(row.get("seed", "-1")) != seed:
        raise RuntimeError(f"{system} seed {seed}: bad row {rows[0]}")
    if int(float(row.get("budget", "-1"))) != budget:
        raise RuntimeError(f"{system} seed {seed}: budget mismatch {row.get('budget')} != {budget}")
    wall = float(row["wall_s"])
    pot = float(row.get("pot_s", "0"))
    outside = float(row["outside_s"]) if "outside_s" in row else max(wall - pot, 0.0)
    hit = row["hit"].strip().lower() in ("1", "true")
    best = float(row["best"])
    if not math.isfinite(best):
        raise RuntimeError(f"{system} seed {seed}: non-finite best {row['best']}")
    expected_hit = energy_matches_reference(best, reference, DEFAULT_HIT_TOL)
    if hit != expected_hit:
        raise RuntimeError(
            f"{system} seed {seed}: hit {int(hit)} with best {best} "
            f"against reference {reference} at tolerance {DEFAULT_HIT_TOL}"
        )
    return {
        "hit": "1" if hit else "0",
        "evals_to_hit": str(int(float(row["evals_to_hit"]))),
        "charged": str(int(float(row.get("charged", row["evals_to_hit"])))),
        "best": row["best"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
    }


def notes_specs(paper, sc, seed0: int, n: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for system in NOTES_LJ_SYSTEMS:
        specs.append(
            {
                "panel": "lj",
                "system": system,
                "binary": "h2h_gmin_proof",
                "source": "notes",
                "seed0": seed0,
                "n": n,
                "expected_budget": int(paper.PAPER_BUDGETS[system]),
            }
        )
    for system in NOTES_MORSE_SYSTEMS:
        specs.append(
            {
                "panel": "morse-panel",
                "system": system,
                "binary": "h2h_morse",
                "source": "notes",
                "seed0": seed0,
                "n": n,
                "expected_budget": NOTES_MORSE_BUDGETS[system],
            }
        )
    for system in NOTES_AU_SYSTEMS:
        specs.append(
            {
                "panel": "au",
                "system": system,
                "binary": "h2h_metal",
                "source": "notes",
                "seed0": seed0,
                "n": n,
                "expected_budget": NOTES_AU_BUDGETS[system],
            }
        )
    for system, budget in sc.PAPER_BUDGETS.items():
        specs.append(
            {
                "panel": "sutton-chen",
                "system": system,
                "binary": "h2h_sc",
                "source": "notes",
                "seed0": 2000,
                "n": 6,
                "expected_budget": int(budget),
            }
        )
    return specs


def repo_specs(darby, seed0: int, n: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for system, budget in darby.PAPER_BUDGETS.items():
        specs.append(
            {
                "panel": "binary-lj",
                "system": system,
                "binary": "h2h_metal",
                "source": "repo-runner",
                "seed0": seed0,
                "n": n,
                "expected_budget": int(budget),
            }
        )
    return specs


def signature(spec: dict[str, object]) -> str:
    source = str(spec["source"])
    env_tag = "notes-env" if source == "notes" else "single-thread-env"
    return (
        f"source={source};binary={spec['binary']};seed0={spec['seed0']};n={spec['n']};"
        f"budget={spec['expected_budget']};env={env_tag}"
    )


def should_skip_due_supersede(row: dict[str, str], cutoff: dict[tuple[str, str], str]) -> bool:
    key = (row.get("panel", ""), row.get("system", ""))
    restart = cutoff.get(key)
    if not restart:
        return False
    return row.get("session", "") < restart


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--unsupported", type=Path, required=True)
    parser.add_argument("--declared", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--superseded", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--seed0", type=int, default=2000)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--rgpot-sc-lib", type=Path)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)

    if socket.gethostname() != "bench":
        raise SystemExit(f"hostname is {socket.gethostname()}, expected bench")

    root = Path(__file__).resolve().parent.parent.parent
    paper = load_module("paper_0120_run_panel", root / "scripts/paper_0120/run_panel.py")
    darby = load_module("metal_run_darby", root / "scripts/metal_h2h/run_darby.py")
    sc = load_module("metal_run_sc", root / "scripts/metal_h2h/run_sc.py")

    unsupported: list[dict[str, str]] = [
        {"panel": "morse-panel", "system": "m75", "reason": "unsupported: no M75 entry in notes command list"},
        {"panel": "surface-slab", "system": "surface-slab", "reason": "unsupported: no bench input and command line in notes"},
    ]
    if not args.notes.is_file():
        unsupported.append({"panel": "meta", "system": "notes", "reason": "unsupported: notes file missing"})
    if args.rgpot_sc_lib is None:
        unsupported.append({"panel": "sutton-chen", "system": "all", "reason": "unsupported: RGPOT_SC_LIB was not provided"})
    elif not args.rgpot_sc_lib.is_file():
        unsupported.append({"panel": "sutton-chen", "system": "all", "reason": "unsupported: RGPOT_SC_LIB was requested but not found"})

    specs = notes_specs(paper, sc, args.seed0, args.n) + repo_specs(darby, args.seed0, args.n)
    commands_rows = [
        {
            "panel": str(spec["panel"]),
            "system": str(spec["system"]),
            "source": str(spec["source"]),
            "binary": str(spec["binary"]),
            "seed0": str(spec["seed0"]),
            "n": str(spec["n"]),
            "signature": signature(spec),
        }
        for spec in specs
    ]

    existing_rows = read_csv(args.rows)
    prior_commands = {(row["panel"], row["system"]): row.get("signature", "") for row in read_csv(args.commands)}
    superseded_history = read_csv(args.superseded)
    if not args.superseded.is_file():
        write_csv(args.superseded, SUPERSEDED_FIELDS, superseded_history)
    cutoff_session: dict[tuple[str, str], str] = {}
    for row in superseded_history:
        cutoff_session[(row["panel"], row["system"])] = row["restart_session"]

    restart_session = now_session()
    existing_keys = {(row.get("panel", ""), row.get("system", "")) for row in existing_rows}
    for row in commands_rows:
        key = (row["panel"], row["system"])
        prior = prior_commands.get(key)
        if prior and prior != row["signature"] and key in existing_keys:
            append_csv(
                args.superseded,
                SUPERSEDED_FIELDS,
                {
                    "utc": now_utc(),
                    "panel": row["panel"],
                    "system": row["system"],
                    "restart_session": restart_session,
                    "reason": "command signature changed",
                },
            )
            cutoff_session[key] = restart_session

    declared = [{"panel": row["panel"], "system": row["system"], "n": row["n"], "reason": f"declared pre-run ({row['source']})"} for row in commands_rows]
    write_csv(args.commands, COMMAND_FIELDS, commands_rows)
    write_csv(args.unsupported, UNSUPPORTED_FIELDS, unsupported)
    write_csv(args.declared, DECLARED_FIELDS, declared)
    if args.plan_only:
        print(f"planned systems={len(commands_rows)} unsupported={len(unsupported)}", flush=True)
        return 0

    args.work.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    arm_a = build_arm(args.a_tree.resolve(), args.work / "A")
    arm_b = build_arm(args.b_tree.resolve(), args.work / "B")
    arms = {"A": arm_a, "B": arm_b}

    sc_disabled = any(row["panel"] == "sutton-chen" and row["system"] == "all" for row in unsupported)
    supported: list[dict[str, object]] = []
    for spec in specs:
        panel = str(spec["panel"])
        system = str(spec["system"])
        binary_name = str(spec["binary"])
        if panel == "sutton-chen" and sc_disabled:
            continue
        env_extra = {"RGPOT_SC_LIB": str(args.rgpot_sc_lib)} if binary_name == "h2h_sc" and args.rgpot_sc_lib else None
        env = notes_env(env_extra) if str(spec["source"]) == "notes" else single_thread_env(env_extra)
        try:
            a_budget, a_ref = parse_budget(run([str(arm_a["bins"][binary_name]), "budget", system], cwd=Path(arm_a["tree"]), env=env))
            b_budget, b_ref = parse_budget(run([str(arm_b["bins"][binary_name]), "budget", system], cwd=Path(arm_b["tree"]), env=env))
        except RuntimeError:
            unsupported.append({"panel": panel, "system": system, "reason": "unsupported: budget probe failed"})
            continue
        expected_budget = int(spec["expected_budget"])
        if a_budget != expected_budget or b_budget != expected_budget:
            unsupported.append(
                {
                    "panel": panel,
                    "system": system,
                    "reason": f"unsupported: expected budget {expected_budget}, got A={a_budget} B={b_budget}",
                }
            )
            continue
        if abs(a_ref - b_ref) > 1e-12:
            unsupported.append({"panel": panel, "system": system, "reason": f"unsupported: A/B references differ ({a_ref} vs {b_ref})"})
            continue
        supported.append({**spec, "budget": a_budget, "reference": a_ref})

    write_csv(args.unsupported, UNSUPPORTED_FIELDS, unsupported)
    session = now_session()
    host = socket.gethostname()
    specs_by_key = {(str(spec["panel"]), str(spec["system"])): spec for spec in supported}
    done: set[tuple[str, str, int, str]] = set()
    for row in existing_rows:
        panel = row.get("panel", "")
        system = row.get("system", "")
        key = (panel, system)
        if key not in specs_by_key:
            continue
        if should_skip_due_supersede(row, cutoff_session):
            continue
        arm = row.get("arm", "")
        if arm not in ("A", "B"):
            continue
        try:
            seed = int(row.get("seed", "-1"))
        except ValueError:
            continue
        spec = specs_by_key[key]
        seed0 = int(spec["seed0"])
        n = int(spec["n"])
        if seed0 <= seed < seed0 + n:
            done.add((panel, system, seed, arm))

    rows_written = 0
    for spec in supported:
        panel = str(spec["panel"])
        system = str(spec["system"])
        binary_name = str(spec["binary"])
        budget = int(spec["budget"])
        reference = float(spec["reference"])
        seed0 = int(spec["seed0"])
        n = int(spec["n"])
        env_extra = {"RGPOT_SC_LIB": str(args.rgpot_sc_lib)} if binary_name == "h2h_sc" and args.rgpot_sc_lib else None
        env = notes_env(env_extra) if str(spec["source"]) == "notes" else single_thread_env(env_extra)
        for offset, seed in enumerate(range(seed0, seed0 + n)):
            order = ("A", "B") if offset % 2 == 0 else ("B", "A")
            for arm_name in order:
                if (panel, system, seed, arm_name) in done:
                    continue
                arm = arms[arm_name]
                command = ["taskset", "-c", "0", str(arm["bins"][binary_name]), "run", system, str(seed)]
                proc = run(command, cwd=Path(arm["tree"]), env=env, check=False)
                log = args.logs / panel / system / f"{seed}-{arm_name}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(" ".join(command) + f"\nexit={proc.returncode}\n" + proc.stdout + proc.stderr)
                if proc.returncode != 0:
                    raise SystemExit(f"{panel} {system} seed {seed} arm {arm_name} failed; see {log}")
                parsed = parse_row(proc.stdout, system, seed, budget, reference)
                row = {
                    "session": session,
                    "panel": panel,
                    "system": system,
                    "seed": str(seed),
                    "arm": arm_name,
                    "arm_order": "".join(order),
                    "binary": binary_name,
                    "budget": str(budget),
                    "reference": f"{reference:.10f}",
                    **parsed,
                    "code_revision": str(arm["revision"]),
                    "host": host,
                    "utc": now_utc(),
                }
                append_csv(args.rows, ROW_FIELDS, row)
                rows_written += 1
                done.add((panel, system, seed, arm_name))
                print(
                    f"{rows_written}: {panel} {system} seed={seed} arm={arm_name} "
                    f"hit={row['hit']} evals={row['evals_to_hit']} wall_s={row['wall_s']}",
                    flush=True,
                )

    print(f"completed new_rows={rows_written} supported_systems={len(supported)} unsupported={len(unsupported)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
