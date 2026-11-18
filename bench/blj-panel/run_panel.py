# Copyright (c) anneal developers
"""Serial A/B gate on the binary Lennard-Jones, Lennard-Jones, Morse and
Sutton-Chen hit panels.

Arm A is develop at the base commit. Arm B is the same commit with the
binary Lennard-Jones patches. Both arms run the same drivers with the
recommended preset, one fresh process per run, seeds 2000-2011, the paper
budgets, one process at a time on core 0, one thread. For each seed the two
arms run back to back, and the order flips with the seed: A first on even
seeds, B first on odd seeds. A hit is a quenched energy at or below the
global minimum plus 0.001.

Rows are appended to ``rows.csv`` after every run and never rewritten, with
``code_revision`` and ``host``. A run already stored for the same system,
arm, seed and revision is kept and its run skipped. Nothing is run twice:
a run that crashes is stored as a miss at its budget with status ``crash``;
three crashes in a row stop the session. A run whose output cannot be read,
or reports another system, seed, budget or revision, stops the session and
stores nothing.

This process refuses to run a gate unless the UTS hostname is bench and it
is pinned to core 0.

    run_panel.py --rows rows.csv --logs DIR --bin-a DIR --tree-a DIR
        --bin-b DIR --tree-b DIR [--system NAME] [--seed N]
    run_panel.py --preflight ...   check both arms without storing a row
    run_panel.py --summarize ...   rewrite the summary from the rows
    run_panel.py --self-test       check the statistics and the loop
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
HIT_TOL = 0.001
ARMS = ("A", "B")
MAX_CRASH_STREAK = 3
PREFLIGHT_BUDGET = 2000

PAPER_BUDGETS = {
    "blj105n38": 400_000,
    "blj105n55": 1_000_000,
    "blj115n38": 400_000,
    "blj115n55": 1_000_000,
    "blj130n38": 400_000,
    "blj130n55": 1_000_000,
    "blj130n98": 4_000_000,
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "m38r6": 400_000,
    "m38r14": 400_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
    "sc126-38": 400_000,
    "sc126-55": 1_000_000,
    "sc126-80": 1_000_000,
    "sc108-38": 400_000,
    "sc108-55": 1_000_000,
    "sc108-80": 1_000_000,
    "sc96-38": 400_000,
    "sc96-55": 1_000_000,
    "sc96-80": 1_000_000,
}

# System, driver and the energy of the global minimum. The drivers carry the
# same table; the preflight compares both.
PANEL = (
    ("blj105n38", "h2h_blj", -177.2090846748),
    ("blj105n55", "h2h_blj", -283.1545313810),
    ("blj115n38", "h2h_blj", -187.3676867453),
    ("blj115n55", "h2h_blj", -289.2931853058),
    ("blj130n38", "h2h_blj", -191.1840192602),
    ("blj130n55", "h2h_blj", -306.7337643721),
    ("blj130n98", "h2h_blj", -590.7874132760),
    ("lj38", "h2h_gmin_proof", -173.928427),
    ("lj55", "h2h_gmin_proof", -279.248470),
    ("lj75", "h2h_gmin_proof", -397.492331),
    ("lj98", "h2h_gmin_proof", -543.665361),
    ("lj103", "h2h_gmin_proof", -575.766131),
    ("lj104", "h2h_gmin_proof", -582.086642),
    ("m38r6", "h2h_morse", -157.477108),
    ("m38r14", "h2h_morse", -144.321054),
    ("m80r6", "h2h_morse", -378.333471),
    ("m80r14", "h2h_morse", -340.811371),
    ("sc126-38", "h2h_sc", -35419.9804),
    ("sc126-55", "h2h_sc", -52884.6806),
    ("sc126-80", "h2h_sc", -78414.6271),
    ("sc108-38", "h2h_sc", -10117.2454),
    ("sc108-55", "h2h_sc", -14814.5225),
    ("sc108-80", "h2h_sc", -21819.9952),
    ("sc96-38", "h2h_sc", -8917.7056),
    ("sc96-55", "h2h_sc", -13217.8963),
    ("sc96-80", "h2h_sc", -19560.8966),
)

# The seven-hour baseline arm of the largest cluster runs after the rest.
RUN_LAST = ("blj130n98",)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARNESS_FILES = ("run_panel.py", "h2h_blj.rs", "enter_bench.sh", "build_arms.sh", "build_rgpot.sh")

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS")

ROW_FIELDS = [
    "system",
    "arm",
    "code",
    "driver",
    "seed",
    "budget",
    "status",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "best_bits",
    "pot_s",
    "outside_s",
    "proc_s",
    "code_revision",
    "code_tree",
    "host",
    "cpu",
    "session",
    "utc",
]


class HarnessError(RuntimeError):
    """The harness, not the search, is at fault. Stops the session."""


@dataclass(frozen=True)
class Arm:
    name: str
    bin: Path
    tree: Path
    revision: str
    tree_hash: str


def drivers() -> dict[str, str]:
    return {name: driver for name, driver, _reference in PANEL}


def references() -> dict[str, float]:
    return {name: reference for name, _driver, reference in PANEL}


def run_order(systems: list[str]) -> list[str]:
    return [s for s in systems if s not in RUN_LAST] + [s for s in systems if s in RUN_LAST]


def arms_for(seed: int) -> tuple[str, ...]:
    return ARMS if seed % 2 == 0 else ARMS[::-1]


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("H2H_")}
    for key in THREAD_VARS:
        env[key] = "1"
    env.update(extra or {})
    return env


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip().replace(",", " ")
    return ""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git(tree: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=tree, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(bin_dir: Path) -> dict[str, str]:
    path = bin_dir / "MANIFEST"
    if not path.is_file():
        raise HarnessError(f"{path.name} is missing from {bin_dir.name}; build the arms first")
    found = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key and _:
            found[key] = value
    return found


def read_sums(bin_dir: Path) -> dict[str, str]:
    path = bin_dir / "SHA256SUMS"
    if not path.is_file():
        raise HarnessError(f"{path.name} is missing from {bin_dir.name}")
    sums = {}
    for line in path.read_text().splitlines():
        digest, _, name = line.partition("  ")
        if name:
            sums[name] = digest
    return sums


def verify_binaries(arm: Arm, needed: set[str]) -> None:
    sums = read_sums(arm.bin)
    for name in sorted(needed):
        binary = arm.bin / name
        if name not in sums or not binary.is_file():
            raise HarnessError(f"arm {arm.name} has no {name}")
        if sha256_of(binary) != sums[name]:
            raise HarnessError(f"arm {arm.name} {name} differs from the built binary")


def load_arm(name: str, bin_dir: Path, tree: Path, needed: set[str]) -> Arm:
    manifest = read_manifest(bin_dir)
    if manifest.get("arm") != name:
        raise HarnessError(f"{bin_dir.name} holds arm {manifest.get('arm')}, expected {name}")
    revision = git(tree, "rev-parse", "HEAD")
    tree_hash = git(tree, "rev-parse", "HEAD^{tree}")
    if revision != manifest.get("revision") or tree_hash != manifest.get("tree"):
        raise HarnessError(
            f"arm {name} tree is {revision} ({tree_hash}), binaries were built from "
            f"{manifest.get('revision')} ({manifest.get('tree')})"
        )
    if git(tree, "status", "--porcelain", "--untracked-files=no"):
        raise HarnessError(f"arm {name} tree has tracked changes")
    arm = Arm(name, bin_dir, tree, revision, tree_hash)
    verify_binaries(arm, needed)
    return arm


def parse_row(stdout: str, system: str, seed: int) -> dict[str, str]:
    """The header and the one result row a driver prints. A driver may print
    more lines after the row; they are ignored."""
    header: list[str] | None = None
    data: list[str] | None = None
    for line in stdout.splitlines():
        if line.startswith("system,"):
            if header is not None:
                raise HarnessError(f"{system} seed {seed}: two header lines")
            header = line.split(",")
        elif line.startswith(f"{system},anneal,{seed},"):
            if data is not None:
                raise HarnessError(f"{system} seed {seed}: two result rows")
            data = line.split(",")
    if header is None or data is None or len(header) != len(data):
        raise HarnessError(f"{system} seed {seed}: expected one header and one row\n{stdout[-1500:]}")
    return dict(zip(header, data, strict=True))


def as_bool(text: str) -> bool:
    if text.strip().lower() in ("1", "true"):
        return True
    if text.strip().lower() in ("0", "false"):
        return False
    raise HarnessError(f"hit field {text!r}")


def check_result(system: str, seed: int, budget: int, hit: bool, best: float, evals: int, charged: int) -> None:
    level = references()[system] + HIT_TOL
    if hit and best > level + 1e-7:
        raise HarnessError(f"{system} seed {seed}: hit with best {best} above {level}")
    if not hit and best < level - 1e-7:
        raise HarnessError(f"{system} seed {seed}: miss with best {best} below {level}")
    if not hit and evals != budget:
        raise HarnessError(f"{system} seed {seed}: a miss reports {evals} calls, budget {budget}")
    if hit and not 0 <= evals <= budget:
        raise HarnessError(f"{system} seed {seed}: a hit reports {evals} calls, budget {budget}")
    if charged < 0:
        raise HarnessError(f"{system} seed {seed}: charged {charged}")


def result_fields(arm: Arm, system: str, seed: int, budget: int, stdout: str) -> dict[str, str]:
    parsed = parse_row(stdout, system, seed)
    if int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
        raise HarnessError(f"{system} seed {seed}: row reports seed {parsed['seed']} budget {parsed['budget']}")
    if "head" in parsed and (parsed["head"] != arm.revision or parsed.get("tree") != arm.tree_hash):
        raise HarnessError(f"{system} seed {seed}: driver ran in {parsed['head']}, arm is {arm.revision}")
    hit = as_bool(parsed["hit"])
    best = float(parsed["best"])
    evals = int(parsed["evals_to_hit"])
    charged = int(parsed["charged"])
    check_result(system, seed, budget, hit, best, evals, charged)
    wall = float(parsed["wall_s"])
    pot = float(parsed["pot_s"])
    return {
        "status": "ok",
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(charged),
        "wall_s": f"{wall:.6f}",
        "best": parsed["best"],
        "best_bits": parsed.get("best_bits", ""),
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{max(wall - pot, 0.0):.6f}",
    }


def crash_fields(budget: int, proc_s: float) -> dict[str, str]:
    return {
        "status": "crash",
        "hit": "0",
        "evals_to_hit": str(budget),
        "charged": "0",
        "wall_s": f"{proc_s:.6f}",
        "best": "nan",
        "best_bits": "",
        "pot_s": "0.000000",
        "outside_s": f"{proc_s:.6f}",
    }


def command_for(arm: Arm, system: str, seed: int) -> list[str]:
    return ["taskset", "-c", "0", str(arm.bin / drivers()[system]), "run", system, str(seed)]


def run_once(arm: Arm, system: str, seed: int, log: Path) -> tuple[dict[str, str], float, str]:
    """Run one fresh process. Returns the result fields, the process wall time
    and a one-line note for a crash."""
    budget = PAPER_BUDGETS[system]
    command = command_for(arm, system, seed)
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=arm.tree, env=child_env(), capture_output=True, text=True, check=False)
    proc_s = time.perf_counter() - start
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"{' '.join(command)}\nexit {proc.returncode}\n{proc.stdout}{proc.stderr}")
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["no stderr"])[-1]
        return crash_fields(budget, proc_s), proc_s, f"exit {proc.returncode}: {tail}"
    return result_fields(arm, system, seed, budget, proc.stdout), proc_s, ""


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise HarnessError(f"{path.name} has columns {reader.fieldnames}")
        return list(reader)


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({key: row[key] for key in ROW_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def selected(args: argparse.Namespace) -> list[str]:
    known = [name for name, _driver, _reference in PANEL]
    for name in args.system or []:
        if name not in known:
            raise SystemExit(f"unknown system {name}")
    chosen = [name for name in known if not args.system or name in args.system]
    return run_order(chosen)


def write_builds(path: Path, arms: dict[str, Arm], rgpot_manifest: Path | None) -> None:
    """Provenance of the binaries as the build recorded it, and of this
    harness. A file already there must describe the binaries in use."""
    sections: list[list[str]] = []
    for name in ARMS:
        arm = arms[name]
        sections.append(
            [
                f"[arm {name}]",
                *(arm.bin / "MANIFEST").read_text().splitlines(),
                *(arm.bin / "SHA256SUMS").read_text().splitlines(),
            ]
        )
    if rgpot_manifest is not None and rgpot_manifest.is_file():
        sections.append(["[sutton-chen library]", *rgpot_manifest.read_text().splitlines()])
    if path.is_file():
        stored = path.read_text().strip().split("\n\n")
        if stored[: len(sections)] != ["\n".join(s) for s in sections]:
            raise HarnessError(f"{path.name} does not describe the binaries in use")
        return
    changed = "yes" if git(ROOT, "status", "--porcelain", "--untracked-files=no") else "no"
    sections.append(
        [
            "[harness]",
            f"revision={git(ROOT, 'rev-parse', 'HEAD')}",
            f"tracked_changes={changed}",
            *(f"{name}_sha256={sha256_of(HERE / name)}" for name in HARNESS_FILES),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join("\n".join(s) for s in sections) + "\n")


def sc_library_ok(systems: list[str]) -> None:
    if any(drivers()[s] == "h2h_sc" for s in systems):
        lib = os.environ.get("RGPOT_SC_LIB", "")
        if not lib or not Path(lib).is_file():
            raise HarnessError("RGPOT_SC_LIB does not name the Sutton-Chen library")


def gate(args: argparse.Namespace, enforce: bool = True) -> int:
    host = socket.gethostname()
    if enforce and host != "bench":
        print(f"uts hostname is {host}, expected bench", flush=True)
        return 2
    if enforce and os.sched_getaffinity(0) != {0}:
        print(f"pinned to {sorted(os.sched_getaffinity(0))}, expected core 0", flush=True)
        return 2
    systems = selected(args)
    seeds = tuple(args.seed) if args.seed else SEEDS
    needed = {drivers()[s] for s in systems}
    sc_library_ok(systems)
    arms = {
        "A": load_arm("A", args.bin_a, args.tree_a, needed),
        "B": load_arm("B", args.bin_b, args.tree_b, needed),
    }
    if arms["A"].revision == arms["B"].revision:
        raise HarnessError("arms A and B are the same commit")
    if args.builds:
        write_builds(args.builds, arms, args.rgpot_manifest)

    stored = read_rows(args.rows)
    sessions = {row["session"] for row in stored}
    if len(sessions) > 1:
        raise HarnessError(f"rows already hold sessions {sorted(sessions)}")
    session = sessions.pop() if sessions else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {(r["system"], r["arm"], r["seed"], r["code_revision"]) for r in stored}
    cpu = cpu_model()
    session_log = args.logs / "session.log"
    session_log.parent.mkdir(parents=True, exist_ok=True)

    def say(text: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {text}"
        print(line, flush=True)
        with session_log.open("a") as handle:
            handle.write(line + "\n")

    total = len(systems) * len(seeds) * len(ARMS)
    finished = sum(
        (system, name, str(seed), arms[name].revision) in done for system in systems for seed in seeds for name in ARMS
    )
    say(f"session {session} host {host} A {arms['A'].revision} B {arms['B'].revision} runs {total} stored {finished}")
    streak = 0
    for system in systems:
        verify_binaries(arms["A"], needed)
        verify_binaries(arms["B"], needed)
        for seed in seeds:
            for name in arms_for(seed):
                arm = arms[name]
                if (system, name, str(seed), arm.revision) in done:
                    continue
                fields, proc_s, note = run_once(arm, system, seed, args.logs / system / f"{name}-{seed}.log")
                row = {
                    "system": system,
                    "arm": name,
                    "code": "anneal",
                    "driver": drivers()[system],
                    "seed": str(seed),
                    "budget": str(PAPER_BUDGETS[system]),
                    "proc_s": f"{proc_s:.6f}",
                    "code_revision": arm.revision,
                    "code_tree": arm.tree_hash,
                    "host": host,
                    "cpu": cpu,
                    "session": session,
                    "utc": utc(),
                    **fields,
                }
                append_row(args.rows, row)
                done.add((system, name, str(seed), arm.revision))
                finished += 1
                say(
                    f"[{finished}/{total}] {system} seed {seed} arm {name} {row['status']} hit={row['hit']} "
                    f"calls={row['evals_to_hit']} wall={row['wall_s']}s in={row['pot_s']}s "
                    f"out={row['outside_s']}s proc={row['proc_s']}s best={row['best']}" + (f" {note}" if note else "")
                )
                streak = streak + 1 if row["status"] != "ok" else 0
                if streak >= MAX_CRASH_STREAK:
                    say(f"{streak} crashes in a row; stopping the session")
                    return 4
        summarize(args, readme=False)
    say("every run stored")
    return 0


def summarize(args: argparse.Namespace, readme: bool = True) -> None:
    import summarize_panel

    if args.rows is None or args.summary is None or args.pairs is None:
        raise HarnessError("--summarize needs --rows, --summary and --pairs")
    summarize_panel.write_all(args.rows, args.summary, args.pairs, args.readme if readme else None)


def budget_query(arm: Arm, system: str) -> tuple[int, float]:
    proc = subprocess.run(
        [str(arm.bin / drivers()[system]), "budget", system],
        cwd=arm.tree,
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    words = proc.stdout.split()
    if proc.returncode != 0 or len(words) < 2:
        raise HarnessError(f"arm {arm.name} {drivers()[system]} budget {system} exited {proc.returncode}")
    return int(words[0]), float(words[1])


def preflight(args: argparse.Namespace) -> int:
    """Check both arms against the tables here. Runs one seed per driver at a
    small budget and stores nothing."""
    systems = selected(args)
    needed = {drivers()[s] for s in systems}
    sc_library_ok(systems)
    arms = {
        "A": load_arm("A", args.bin_a, args.tree_a, needed),
        "B": load_arm("B", args.bin_b, args.tree_b, needed),
    }
    for name in ARMS:
        arm = arms[name]
        print(f"arm {name} {arm.revision} tree {arm.tree_hash}", flush=True)
        for system in systems:
            budget, reference = budget_query(arm, system)
            if budget != PAPER_BUDGETS[system] or abs(reference - references()[system]) > 1e-6:
                raise HarnessError(
                    f"arm {name} {system}: driver says {budget} calls and {reference}, "
                    f"table says {PAPER_BUDGETS[system]} and {references()[system]}"
                )
        shown: set[str] = set()
        for system in systems:
            driver = drivers()[system]
            if driver in shown:
                continue
            shown.add(driver)
            command = [str(arm.bin / driver), "run", system, "2000"]
            proc = subprocess.run(
                command,
                cwd=arm.tree,
                env=child_env({"H2H_BUDGET": str(PREFLIGHT_BUDGET)}),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise HarnessError(f"arm {name} {driver} {system} exited {proc.returncode}\n{proc.stderr[-1500:]}")
            parsed = parse_row(proc.stdout, system, 2000)
            print(
                f"arm {name} {driver} {system} small run: budget {parsed['budget']} charged {parsed['charged']} "
                f"best {parsed['best']}",
                flush=True,
            )
    print("preflight ok", flush=True)
    return 0


def self_test() -> int:
    import summarize_panel as sp

    assert set(PAPER_BUDGETS) == {name for name, _d, _r in PANEL}, "budgets and panel name different systems"
    assert len(PANEL) == 26 and len(SEEDS) == 12
    assert abs(sp.fisher(12, 12, 5, 12) - 0.004576659039) < 1e-10, sp.fisher(12, 12, 5, 12)
    assert sp.fisher(7, 12, 7, 12) == 1.0
    _u, p = sp.mann_whitney([1.0] * 12, [1.0] * 12)
    assert p == 1.0
    assert sp.type7([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    text = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s\n"
        "lj38,anneal,2000,400000,true,10,12,0.5,-173.9,0.1\n"
        "arm x\n"
    )
    assert parse_row(text, "lj38", 2000)["evals_to_hit"] == "10"
    try:
        parse_row(text.replace("lj38,anneal", "lj55,anneal"), "lj38", 2000)
    except HarnessError:
        pass
    else:
        raise AssertionError("a row for another system was accepted")
    with tempfile.TemporaryDirectory(prefix="panel-test-") as scratch:
        return loop_test(Path(scratch))


FAKE_DRIVER = """#!/bin/sh
case "$1" in
  budget) echo "$BUDGET"; echo "$REFERENCE"; exit 0 ;;
esac
system=$2
seed=$3
head=$(git rev-parse HEAD)
tree=$(git rev-parse 'HEAD^{tree}')
hit=true
calls=$((seed - 1900))
best=$HITBEST
[ "$((seed % 5))" = 4 ] && hit=false && calls=$BUDGET && best=$MISSBEST
echo "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s,best_bits"
echo "$system,anneal,$seed,$BUDGET,$hit,$calls,$calls,0.25,$best,$head,$tree,cpu,pins,0.1,0000000000000000"
"""


def loop_test(scratch: Path) -> int:
    """Run the loop on stand-in drivers: arm order, resume, the summary."""
    trees, bins = {}, {}
    for name in ARMS:
        tree = scratch / f"tree-{name}"
        tree.mkdir()
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        (tree / "file").write_text(name)
        subprocess.run(["git", "-C", str(tree), "add", "file"], check=True)
        subprocess.run(
            ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"],
            check=True,
        )
        trees[name] = tree
        bin_dir = scratch / f"bin-{name}"
        bin_dir.mkdir()
        for driver in {d for _n, d, _r in PANEL}:
            script = bin_dir / driver
            script.write_text(FAKE_DRIVER)
            script.chmod(0o755)
        (bin_dir / "MANIFEST").write_text(
            f"arm={name}\nrevision={git(tree, 'rev-parse', 'HEAD')}\ntree={git(tree, 'rev-parse', 'HEAD^{tree}')}\n"
        )
        sums = "".join(f"{sha256_of(bin_dir / d)}  {d}\n" for d in sorted({d for _n, d, _r in PANEL}))
        (bin_dir / "SHA256SUMS").write_text(sums)
        bins[name] = bin_dir
    system = "lj38"
    old = {key: os.environ.get(key) for key in ("BUDGET", "REFERENCE", "HITBEST", "MISSBEST")}
    os.environ.update(
        BUDGET=str(PAPER_BUDGETS[system]), REFERENCE="-173.928427", HITBEST="-173.928427", MISSBEST="-173.0"
    )
    try:
        args = argparse.Namespace(
            rows=scratch / "rows.csv",
            logs=scratch / "logs",
            summary=scratch / "summary.csv",
            pairs=scratch / "pairs.csv",
            readme=None,
            builds=scratch / "builds.txt",
            rgpot_manifest=None,
            bin_a=bins["A"],
            bin_b=bins["B"],
            tree_a=trees["A"],
            tree_b=trees["B"],
            system=[system],
            seed=[2000, 2001, 2002],
        )
        assert gate(args, enforce=False) == 0
        rows = read_rows(args.rows)
        assert [(r["seed"], r["arm"]) for r in rows] == [
            ("2000", "A"),
            ("2000", "B"),
            ("2001", "B"),
            ("2001", "A"),
            ("2002", "A"),
            ("2002", "B"),
        ], [(r["seed"], r["arm"]) for r in rows]
        assert {r["code_revision"] for r in rows} == {
            git(trees["A"], "rev-parse", "HEAD"),
            git(trees["B"], "rev-parse", "HEAD"),
        }
        before = args.rows.read_text()
        assert gate(args, enforce=False) == 0
        assert args.rows.read_text() == before, "a stored run was run again"
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("self-test ok", flush=True)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--builds", type=Path)
    parser.add_argument("--rgpot-manifest", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--bin-a", type=Path)
    parser.add_argument("--bin-b", type=Path)
    parser.add_argument("--tree-a", type=Path)
    parser.add_argument("--tree-b", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv[1:])
    try:
        if args.self_test:
            return self_test()
        if args.summarize:
            summarize(args)
            return 0
        needs = ["rows", "logs", "bin_a", "bin_b", "tree_a", "tree_b"]
        if not args.preflight:
            needs += ["summary", "pairs"]
        missing = [k for k in needs if getattr(args, k) is None]
        if missing:
            parser.error(f"missing {', '.join(missing)}")
        if args.preflight:
            return preflight(args)
        return gate(args)
    except HarnessError as error:
        print(f"harness error: {error}", flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
