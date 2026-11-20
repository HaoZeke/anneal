# Copyright (c) anneal developers
"""Resumable serial run of the n=24 anneal-GMIN panel.

The manifest declares the cells, seeds, budgets and analysis rules.
Pin and provenance checks finish before any search. Seeds run in order
inside each cell, and the two arms of a seed run in ABBA order. A row
already stored for that cell, seed and arm is kept. Rows are appended
and flushed after every completion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_panel  # noqa: E402

KEEP = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "TZ",
)
THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)
ROW_FIELDS = [
    "cell",
    "seed",
    "arm",
    "slot",
    "core",
    "hit",
    "best_energy",
    "calls",
    "wall_s",
    "code_revision",
    "gmin_svnrev",
    "harness_blob_ids",
    "machine_id",
    "error",
]
SESSION_FIELDS = [
    "session",
    "event",
    "utc",
    "core",
    "machine_id",
    "anneal_revision",
    "gmin_svnrev",
    "arm_order",
    "rows_done",
    "balance",
    "lj103",
    "note",
]
CMAKE = (
    "-DWITH_ALIGN=OFF -DWITH_MPI=OFF -DWITH_OPENMP=OFF -DWITH_SPARSE=OFF "
    "-DWITH_PYTHON=OFF -DWITH_MYBLAS=OFF -DWITH_MYLAPACK=OFF"
)
ATTEMPTS = 3


class RunFailed(RuntimeError):
    def __init__(self, token: str) -> None:
        super().__init__(token)
        self.token = token


class ProvenanceError(RuntimeError):
    pass


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for one arm. Thread counts are 1. Inherited extras stay out."""
    env: dict[str, str] = {}
    for key in KEEP:
        value = os.environ.get(key)
        if value:
            env[key] = value
    for key in THREADS:
        env[key] = "1"
    if extra:
        env.update(extra)
    return env


def machine_id() -> str:
    digest = hashlib.sha256(Path("/etc/machine-id").read_bytes()).hexdigest()
    return digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_out(repo: Path, args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ProvenanceError(proc.stderr.strip() or "git")
    return proc.stdout.strip()


def arm_sequence(seed: int, origin: int) -> tuple[str, str]:
    """ABBA across seeds: even offsets run anneal then GMIN."""
    if (seed - origin) % 2 == 0:
        return ("anneal", "gmin")
    return ("gmin", "anneal")


def planned_pairs(seeds: list[int], origin: int) -> list[tuple[int, str]]:
    return [(slot, arm) for seed in seeds for slot, arm in enumerate(arm_sequence(seed, origin))]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def check_manifest(manifest: dict) -> None:
    cells = [row[0] for row in run_panel.SYSTEMS]
    if list(manifest["cells"]) != cells:
        raise SystemExit("manifest cells do not match the panel")
    budgets = {name: int(value) for name, value in manifest["budgets"].items()}
    if budgets != dict(run_panel.PAPER_BUDGETS):
        raise SystemExit("manifest budgets do not match the panel")
    for row in run_panel.SYSTEMS:
        if abs(float(manifest["references"][row[0]]) - row[3]) > 1e-12:
            raise SystemExit(f"manifest reference differs for {row[0]}")
    seeds = [int(seed) for seed in manifest["seeds"]]
    if seeds != list(range(2000, 2024)):
        raise SystemExit("manifest seeds are not 2000-2023")
    if manifest["gmin_svnrev"] != run_panel.GMIN_PIN:
        raise SystemExit("manifest svn revision does not match the pin")
    if manifest["tarball_sha256"] != run_panel.GMIN_TARBALL_SHA256:
        raise SystemExit("manifest tarball hash does not match the pin")
    if manifest["arm_order"] != "abba":
        raise SystemExit("manifest arm order is not abba")
    if manifest["cmake"] != CMAKE:
        raise SystemExit("manifest cmake flags differ")
    if abs(float(manifest["hit_tol"]) - run_panel.HIT_TOL) > 0.0:
        raise SystemExit("manifest tolerance differs")


def harness_blobs(repo: Path, paths: list[str]) -> str:
    ids = []
    for path in paths:
        committed = git_out(repo, ["rev-parse", f"HEAD:{path}"])
        on_disk = git_out(repo, ["hash-object", path])
        if committed != on_disk:
            raise ProvenanceError(f"{path} differs from HEAD")
        git_out(repo, ["cat-file", "-e", committed])
        ids.append(committed)
    return " ".join(ids)


def source_map(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def budget_of(binary: Path, tree: Path, system: str) -> tuple[int, float]:
    proc = capture([str(binary), "budget", system], tree, child_env())
    if proc.returncode != 0:
        raise RunFailed("budget")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return int(lines[0]), float(lines[1])


def resolve_drivers(bin_dir: Path, tree: Path, cells: list[str]) -> tuple[dict[str, tuple[str, str]], bool]:
    found = {}
    lj103_ok = False
    for system in cells:
        _name, _kind, _n, _ref, _rho, driver, binary = run_panel.spec(system)
        if system == "lj103":
            probe = capture([str(bin_dir / "h2h_gmin_proof"), "budget", "lj103"], tree, child_env())
            if probe.returncode == 0:
                driver, binary = "h2h_gmin_proof", "h2h_gmin_proof"
                lj103_ok = True
        found[system] = (driver, binary)
    return found, lj103_ok


def check_pins(args: argparse.Namespace, manifest: dict, repo: Path) -> tuple[str, str, str, bool, dict[str, tuple[str, str]]]:
    """Every check here is outside the timed searches."""
    check_manifest(manifest)
    head = git_out(Path(args.tree), ["rev-parse", "HEAD"])
    if head != manifest["base"]:
        raise ProvenanceError("anneal tree is not the declared base")
    git_out(repo, ["cat-file", "-e", head])
    blobs = harness_blobs(repo, list(manifest["harness_paths"]))
    for patch in manifest["patches"]:
        patch_path = repo / patch["path"]
        digest = sha256_file(patch_path)
        if digest != patch["sha256"]:
            raise ProvenanceError("patch hash mismatch")
        git_out(repo, ["cat-file", "-e", git_out(repo, ["rev-parse", f"HEAD:{patch['path']}"])])
    first_patch = (repo / manifest["patches"][0]["path"]).read_text().splitlines()
    if len(first_patch) < 65 or "JP=1" not in first_patch[64]:
        raise ProvenanceError("evaluator patch is missing JP=1 on line 65")
    tarball = sha256_file(Path(args.tarball))
    if tarball != manifest["tarball_sha256"]:
        raise ProvenanceError("tarball hash mismatch")
    source = source_map(Path(args.gmin_source).read_text())
    svn_file = Path(args.svnrev).read_text().strip()
    if source.get("svnrev") != manifest["gmin_svnrev"] or svn_file != manifest["gmin_svnrev"]:
        raise ProvenanceError("built svn revision is not the pin")
    if source.get("tarball_sha256") != manifest["tarball_sha256"]:
        raise ProvenanceError("built source records another tarball")
    if source.get("cmake") != CMAKE:
        raise ProvenanceError("built cmake flags differ")
    rustc = Path(args.rustc_version).read_text().strip()
    if not rustc.startswith("rustc 1.88.0"):
        raise ProvenanceError("compiler is not 1.88.0")
    ldd = subprocess.run(["ldd", str(Path(args.bin) / "GMIN")], capture_output=True, text=True, check=True)
    linked = ldd.stdout.lower()
    if "openblas" in linked or "libgomp" in linked or "libmpi" in linked:
        raise ProvenanceError("gmin is linked to a threaded or mpi library")
    if "libblas" not in linked or "liblapack" not in linked:
        raise ProvenanceError("gmin is not linked to the reference libraries")
    if os.sched_getaffinity(0) != {0}:
        raise SystemExit("this process is not pinned to core 0")
    drivers, lj103_ok = resolve_drivers(Path(args.bin), Path(args.tree), list(manifest["cells"]))
    for system, (_driver, binary) in drivers.items():
        if system == "lj103" and not lj103_ok:
            continue
        try:
            budget, reference = budget_of(Path(args.bin) / binary, Path(args.tree), system)
        except RunFailed as exc:
            raise ProvenanceError("budget probe failed") from exc
        if budget != run_panel.PAPER_BUDGETS[system] or abs(reference - run_panel.spec(system)[3]) > 1e-8:
            raise ProvenanceError("budget or reference mismatch")
    return head, manifest["gmin_svnrev"], blobs, lj103_ok, drivers


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
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


def capture(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def write_log(path: Path, command: list[str], proc: subprocess.CompletedProcess[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stdout = proc.stdout[-20000:]
    stderr = proc.stderr[-5000:]
    path.write_text(" ".join(command) + f"\nexit={proc.returncode}\n" + stdout + stderr)


def run_anneal(bin_dir: Path, tree: Path, binary: str, system: str, seed: int, log: Path, revision: str) -> dict[str, str]:
    budget = run_panel.PAPER_BUDGETS[system]
    command = ["taskset", "-c", "0", str(bin_dir / binary), "run", system, str(seed)]
    proc = capture(command, tree, child_env())
    write_log(log, command, proc)
    if proc.returncode != 0:
        raise RunFailed("exit")
    try:
        result = run_panel.anneal_result(proc.stdout, system, seed, budget)
    except RuntimeError as exc:
        token = "counts" if any(word in str(exc) for word in ("hit", "evals", "wall", "pot")) else "parse"
        raise RunFailed(token) from exc
    head = result.pop("head")
    if head not in ("", revision):
        raise ProvenanceError("binary reported another revision")
    return result


def run_gmin(
    bin_dir: Path,
    tree: Path,
    binary: str,
    system: str,
    seed: int,
    work: Path,
    log: Path,
    lj_data: str,
    morse_data: str,
) -> dict[str, str]:
    budget = run_panel.PAPER_BUDGETS[system]
    reference = run_panel.spec(system)[3]
    work.mkdir(parents=True, exist_ok=True)
    for name in ("row.txt", "coords", "data"):
        (work / name).unlink(missing_ok=True)
    coords_command = [str(bin_dir / binary), "coords", system, str(seed), str(work / "coords")]
    coords = capture(coords_command, tree, child_env())
    if coords.returncode != 0:
        write_log(log, coords_command, coords)
        raise RunFailed("coords")
    (work / "data").write_text(run_panel.gmin_data(system, seed, lj_data, morse_data))
    env = child_env(
        {
            "H2H_BUDGET": str(budget),
            "H2H_SYSTEM": system,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{reference:.10f}",
            "H2H_TOL": f"{run_panel.HIT_TOL}",
            "H2H_ROW": str(work / "row.txt"),
        }
    )
    command = ["taskset", "-c", "0", str(bin_dir / "GMIN")]
    proc = capture(command, work, env)
    write_log(log, command, proc)
    row_path = work / "row.txt"
    if not row_path.is_file():
        raise RunFailed("no-row")
    try:
        return run_panel.gmin_result(row_path.read_text(), system, seed, budget)
    except RuntimeError as exc:
        token = "counts" if any(word in str(exc) for word in ("hit", "evals", "wall", "pot")) else "parse"
        raise RunFailed(token) from exc


def result_row(result: dict[str, str]) -> dict[str, str]:
    return {
        "hit": result["hit"],
        "best_energy": result["best"],
        "calls": result["evals_to_hit"],
        "wall_s": result["wall_s"],
        "error": "",
    }


def blank_failure(token: str) -> dict[str, str]:
    return {"hit": "", "best_energy": "", "calls": "", "wall_s": "", "error": token}


def balance_of(rows: list[dict[str, str]]) -> str:
    counts = Counter((row["slot"], row["arm"]) for row in rows if row["error"] == "")
    return " ".join(f"{slot}:{arm}={counts[(slot, arm)]}" for slot in ("0", "1") for arm in ("anneal", "gmin"))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def session_id_of(path: Path) -> str:
    rows = read_csv(path)
    if rows:
        return rows[0]["session"]
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def self_test() -> None:
    seeds = list(range(2000, 2024))
    pairs = planned_pairs(seeds, 2000)
    if [arm for _slot, arm in pairs[:4]] != ["anneal", "gmin", "gmin", "anneal"]:
        raise SystemExit(f"abba opening {pairs[:4]}")
    for slot in (0, 1):
        for arm in ("anneal", "gmin"):
            if sum(pair == (slot, arm) for pair in pairs) != 12:
                raise SystemExit(f"abba balance {slot} {arm}")
    sentinel = "PANEL_SENTINEL_NOT_FORWARDED"
    os.environ[sentinel] = "1"
    env = child_env()
    if sentinel in env or any(env[key] != "1" for key in THREADS):
        raise SystemExit("child environment")
    del os.environ[sentinel]
    if len(run_panel.SEEDS) != 12 or run_panel.SEEDS[0] != 2000:
        raise SystemExit("the committed panel seeds changed")
    print("ok")


def build_schedule(cells: list[str], seeds: list[int], lj103_ok: bool) -> list[tuple[str, int, int, str]]:
    schedule = []
    for cell in cells:
        for seed in seeds:
            for slot, arm in enumerate(arm_sequence(seed, seeds[0])):
                if cell == "lj103" and not lj103_ok:
                    continue
                schedule.append((cell, seed, slot, arm))
    return schedule


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--gmin-source", type=Path, required=True)
    parser.add_argument("--svnrev", type=Path, required=True)
    parser.add_argument("--tarball", type=Path, required=True)
    parser.add_argument("--rustc-version", type=Path, required=True)
    parser.add_argument("--lj-data", type=Path, required=True)
    parser.add_argument("--morse-data", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--pins-only", action="store_true")
    args = parser.parse_args(argv[1:])
    manifest = load_manifest(args.manifest)
    for key in THREADS:
        os.environ[key] = "1"
    revision, svnrev, blobs, lj103_ok, drivers = check_pins(args, manifest, args.root)
    if args.pins_only:
        print(f"pins ok lj103={'present' if lj103_ok else 'absent'}")
        return 0
    ident = machine_id()
    existing = read_csv(args.rows)
    done = {(row["cell"], row["seed"], row["arm"]) for row in existing}
    if len(done) != len(existing):
        raise SystemExit("rows.csv has a duplicate cell, seed and arm")
    session = session_id_of(args.sessions)
    seeds = [int(seed) for seed in manifest["seeds"]]
    schedule = build_schedule(list(manifest["cells"]), seeds, lj103_ok)
    note = "both arms scheduled" if lj103_ok else "lj103 entry absent; cell not run; patch not applied"
    append_csv(
        args.sessions,
        SESSION_FIELDS,
        {
            "session": session,
            "event": "resume" if existing else "start",
            "utc": utc_now(),
            "core": "0",
            "machine_id": ident,
            "anneal_revision": revision,
            "gmin_svnrev": svnrev,
            "arm_order": "abba",
            "rows_done": str(len(existing)),
            "balance": balance_of(existing),
            "lj103": "present" if lj103_ok else "absent",
            "note": note,
        },
    )
    state = {}
    if args.state.is_file():
        state = json.loads(args.state.read_text())
    fails: dict[str, int] = state.get("fails", {})
    lj_data = args.lj_data.read_text()
    morse_data = args.morse_data.read_text()
    lj_data += "" if lj_data.endswith("\n") else "\n"
    morse_data += "" if morse_data.endswith("\n") else "\n"
    print(f"session {session} rows {len(existing)} schedule {len(schedule)} lj103={lj103_ok}", flush=True)
    for index, (cell, seed, slot, arm) in enumerate(schedule, start=1):
        key = (cell, str(seed), arm)
        if key in done:
            print(f"skip {cell} {arm} {seed}", flush=True)
            continue
        fail_key = f"{cell}|{seed}|{arm}"
        stored = False
        for _attempt in range(ATTEMPTS):
            print(f"start {cell} {arm} {seed} slot={slot}", flush=True)
            log = args.logs / cell / f"{arm}-{seed}-slot{slot}.log"
            try:
                if arm == "anneal":
                    result = run_anneal(
                        args.bin, args.tree, drivers[cell][1], cell, seed, log, revision
                    )
                else:
                    result = run_gmin(
                        args.bin,
                        args.tree,
                        drivers[cell][1],
                        cell,
                        seed,
                        args.work / cell / str(seed),
                        log,
                        lj_data,
                        morse_data,
                    )
                payload = result_row(result)
                stored = True
                break
            except ProvenanceError:
                raise
            except RunFailed as exc:
                fails[fail_key] = fails.get(fail_key, 0) + 1
                args.state.parent.mkdir(parents=True, exist_ok=True)
                args.state.write_text(json.dumps({"fails": fails}))
                payload = blank_failure(exc.token)
                if fails[fail_key] >= ATTEMPTS:
                    stored = True
                    break
        if not stored:
            continue
        row = {
            "cell": cell,
            "seed": str(seed),
            "arm": arm,
            "slot": str(slot),
            "core": "0",
            "code_revision": revision,
            "gmin_svnrev": svnrev,
            "harness_blob_ids": blobs,
            "machine_id": ident,
            **payload,
        }
        append_csv(args.rows, ROW_FIELDS, row)
        done.add(key)
        print(
            f"progress {index}/{len(schedule)} {cell} {arm} {seed} slot={slot} "
            f"hit={row['hit']} calls={row['calls']} wall_s={row['wall_s']} error={row['error']}",
            flush=True,
        )
    final_rows = read_csv(args.rows)
    append_csv(
        args.sessions,
        SESSION_FIELDS,
        {
            "session": session,
            "event": "complete",
            "utc": utc_now(),
            "core": "0",
            "machine_id": ident,
            "anneal_revision": revision,
            "gmin_svnrev": svnrev,
            "arm_order": "abba",
            "rows_done": str(len(final_rows)),
            "balance": balance_of(final_rows),
            "lj103": "present" if lj103_ok else "absent",
            "note": note,
        },
    )
    print(f"balance {balance_of(final_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except ProvenanceError as exc:
        print(f"provenance check failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
