#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial rerun of the GOFEE tables.

One process on core 0. Seeds 2000-2011. Arms are anneal, ASE minima
hopping, and AGOX GOFEE, rotated by seed. Minima hopping and GOFEE stop
at a wall of 7200 s. Rows are appended and are not rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE_REV = "3d005d2485dc5c7ad91b3bbd4854040cf2b38aca"
CODE_TREE = "b12293b9d637dd32417be3aa482302e0f30dd594"
HIT_TOL = 0.001
WALL_CAP_S = 7200.0
SEEDS = tuple(range(2000, 2012))
METHODS = ("anneal", "ase_mh", "agox_gofee")
THREAD = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
SHA256 = {
    "h2h_gmin_proof": "aa42b5d60c1de031a5fdce8f620a9ae52ded46acf8ce35d6f776e70dd5891d8f",
    "h2h_morse": "e3ec53040b4e8df5977d08de4add6c452c70b17abe6433a666e45db4754e9cba",
    "h2h_metal": "f78d5045be0d2f1d4694f7f1fa5077f47c6328058aa95505c95b8b16f3a08f66",
    "h2h_covalent": "57d6454d3e112373beb8c5e04716a314f853db9652619f53d913dd91456427d5",
    "h2h_seed": "83e9b9a0ceceb68eddf6d89bb6d1e5d9e432f29e11b5c3a652fec9643885844d",
    "asg_eval": "a73bf4f900cb0be6f48b6723634c091afe61eebd9f42c4dc35fd63ff2042a2aa",
}
SYSTEMS = (
    {"name": "lj38", "kind": "lj", "budget": 400_000, "reference": -173.928427, "binary_system": "lj38"},
    {"name": "lj55", "kind": "lj", "budget": 1_000_000, "reference": -279.248470, "binary_system": "lj55"},
    {"name": "lj75", "kind": "lj", "budget": 4_000_000, "reference": -397.492331, "binary_system": "lj75"},
    {"name": "au55", "kind": "metal", "budget": 1_000_000, "reference": -191.03085280, "binary_system": "au55"},
    {
        "name": "cu38",
        "kind": "metal",
        "budget": 400_000,
        "reference": -108.96759055,
        "binary_system": "cu38au0",
        "pass_reference": True,
    },
    {
        "name": "cu12au26",
        "kind": "metal",
        "budget": 400_000,
        "reference": -127.21076866,
        "binary_system": "cu12au26",
        "pass_reference": True,
    },
    {"name": "sw10", "kind": "covalent", "budget": 100_000, "reference": -13.79658026, "binary_system": "sw10"},
    {"name": "tc10", "kind": "covalent", "budget": 100_000, "reference": -55.08979464, "binary_system": "tc10"},
    {
        "name": "tip4p8",
        "kind": "water",
        "budget": 2_000_000,
        "reference": -305.5183221676,
        "binary_system": "tip4p8",
        "molecules": 8,
    },
)
REFERENCES = {item["name"]: float(item["reference"]) for item in SYSTEMS}
COLUMNS = [
    "system",
    "method",
    "seed",
    "budget",
    "hit",
    "best",
    "evals_to_hit",
    "charged",
    "wall_s",
    "stop",
    "revision",
    "binary_sha256",
    "machine_id_sha256",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def machine_id_sha256() -> str:
    text = Path("/etc/machine-id").read_text().strip()
    return hashlib.sha256(text.encode()).hexdigest()


def git_text(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "git failed")
    return proc.stdout.strip()


def pin_environment(runs: Path, library: str) -> dict[str, str]:
    keep = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ")
    fresh = {key: os.environ[key] for key in keep if key in os.environ and os.environ[key]}
    fresh.update(THREAD)
    fresh["PYTHONUNBUFFERED"] = "1"
    fresh["MPLBACKEND"] = "Agg"
    fresh["MPLCONFIGDIR"] = str(runs / "mpl")
    fresh["NUMBA_CACHE_DIR"] = str(runs / "numba-cache")
    fresh["XDG_CACHE_HOME"] = str(runs / "cache")
    fresh["RGPOT_COVALENT_LIB"] = library
    # The committed search entry script sets this so the helper does not
    # write an auth file. The model is still built with its ray flag off.
    fresh["RAY_AUTH_MODE"] = "disabled"
    fresh["RAY_TMPDIR"] = str(runs / "ray")
    for path in (runs / "mpl", runs / "numba-cache", runs / "cache", runs / "tmp", runs / "ray"):
        path.mkdir(parents=True, exist_ok=True)
    fresh["TMPDIR"] = str(runs / "tmp")
    os.environ.clear()
    os.environ.update(fresh)
    return fresh


def require_core0() -> None:
    allowed = ""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list"):
            allowed = line.split(":", 1)[1].strip()
    if allowed != "0":
        raise SystemExit(f"cpu affinity is {allowed or 'missing'}, expected 0")


def require_versions() -> None:
    import importlib.metadata as metadata

    pins = {"ase": "3.29.0", "agox": "3.11.1", "numpy": "2.2.6", "numba": "0.61.2"}
    for name, want in pins.items():
        found = metadata.version(name)
        if found != want:
            raise SystemExit(f"{name} {found} is not {want}")


def require_tree(repo: Path) -> None:
    if git_text(repo, "rev-parse", f"{CODE_REV}^{{tree}}") != CODE_TREE:
        raise SystemExit("code tree does not match the pinned tree")
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", CODE_REV, "HEAD"],
        check=False,
    )
    if ancestor.returncode != 0:
        raise SystemExit("code revision is not an ancestor of HEAD")


def require_machine(hosts: Path) -> str:
    found = machine_id_sha256()
    with hosts.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or rows[0]["machine_id_sha256"] != found:
        raise SystemExit("machine id hash does not match the host file")
    return found


def require_binaries(paths: dict[str, Path]) -> dict[str, str]:
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"missing binary {name}")
        digest = sha256_file(path)
        if digest != SHA256[name]:
            raise SystemExit(f"{name} sha256 {digest} is not the pinned binary")
        hashes[name] = digest
    return hashes


def arm_order(seed: int) -> tuple[str, ...]:
    offset = (seed - 2000) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def run_command(cmd: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["taskset", "-c", "0", *cmd],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def check_ok(proc: subprocess.CompletedProcess[str], label: str) -> None:
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(f"{label} exited {proc.returncode}\n{tail}")


def budget_lines(binary: Path, system: str, env: dict[str, str], cwd: Path) -> tuple[int, float]:
    proc = run_command([str(binary), "budget", system], env, cwd)
    check_ok(proc, f"budget {system}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return int(lines[0]), float(lines[1])


def check_budgets(binaries: dict[str, Path], env: dict[str, str], cwd: Path) -> None:
    owners = {"lj": binaries["h2h_gmin_proof"], "metal": binaries["h2h_metal"], "covalent": binaries["h2h_covalent"]}
    for spec in SYSTEMS:
        if spec["kind"] == "water":
            continue
        env_one = dict(env)
        if spec.get("pass_reference"):
            env_one["H2H_REFERENCE"] = f"{spec['reference']:.12f}"
        budget, reference = budget_lines(owners[spec["kind"]], spec["binary_system"], env_one, cwd)
        if budget != spec["budget"] or abs(reference - spec["reference"]) > 1e-6:
            raise SystemExit(f"{spec['name']} budget {budget} {reference} is not the pin")
    for spec in SYSTEMS:
        if spec["kind"] == "lj":
            continue
        proc = run_command([str(binaries["asg_eval"]), "spec", spec["name"]], env, cwd)
        check_ok(proc, f"spec {spec['name']}")
        fields = proc.stdout.splitlines()[0].split()
        if int(fields[3]) != spec["budget"] or abs(float(fields[4]) - spec["reference"]) > 1e-9:
            raise SystemExit(f"{spec['name']} evaluator spec drifted")


def parse_csv_row(stdout: str) -> dict[str, str]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    for index, line in enumerate(lines[:-1]):
        if line.startswith("system,code,seed,"):
            header = line.split(",")
            values = lines[index + 1].split(",")
            if len(values) < len(header):
                break
            return dict(zip(header, values))
    raise RuntimeError("search returned no row")


def row_from_search(
    spec: dict,
    method: str,
    seed: int,
    hit: bool,
    best: float,
    evals_to_hit: int,
    charged: int,
    wall_s: float,
    stop: str,
    binary_name: str,
    machine: str,
) -> dict[str, str]:
    if not math.isfinite(best):
        raise RuntimeError(f"{spec['name']} {method} {seed} best is not finite")
    if stop not in {"hit", "budget", "wall"}:
        raise RuntimeError(f"{spec['name']} {method} {seed} stop {stop}")
    scored = stop == "hit"
    if hit != scored:
        raise RuntimeError(f"{spec['name']} {method} {seed} hit flag does not match {stop}")
    from target_tol import energy_hits

    in_window = energy_hits(best, spec["reference"], HIT_TOL)
    if scored and not in_window:
        raise RuntimeError(f"{spec['name']} {method} {seed} best is outside the hit window")
    return {
        "system": spec["name"],
        "method": method,
        "seed": str(seed),
        "budget": str(spec["budget"]),
        "hit": "1" if scored else "0",
        "best": f"{best:.10f}",
        "evals_to_hit": str(evals_to_hit) if scored else "",
        "charged": str(charged),
        "wall_s": f"{wall_s:.6f}",
        "stop": stop,
        "revision": CODE_REV,
        "binary_sha256": SHA256[binary_name],
        "machine_id_sha256": machine,
    }


def anneal_from_csv(spec: dict, seed: int, stdout: str, binary_name: str, machine: str) -> dict[str, str]:
    from target_tol import energy_hits

    fields = parse_csv_row(stdout)
    best = float(fields["best"])
    reported = fields["hit"].strip().lower() in {"1", "true"}
    scored = energy_hits(best, spec["reference"], HIT_TOL)
    if scored != reported:
        raise RuntimeError(f"{spec['name']} seed {seed} reported hit {fields['hit']} for best {best}")
    if int(fields["budget"]) != spec["budget"]:
        raise RuntimeError(f"{spec['name']} seed {seed} budget {fields['budget']}")
    return row_from_search(
        spec,
        "anneal",
        seed,
        scored,
        best,
        int(fields["evals_to_hit"]),
        int(fields["charged"]),
        float(fields["wall_s"]),
        "hit" if scored else "budget",
        binary_name,
        machine,
    )


def write_coords(binary: Path, system: str, seed: int, path: Path, env: dict[str, str], cwd: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = run_command([str(binary), "coords", system, str(seed), str(path)], env, cwd)
    check_ok(proc, f"coords {system} {seed}")


def read_xyz(path: Path) -> list[float]:
    values = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            values.extend(float(part) for part in parts[:3])
    return values


def prepare_work(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def harness_row(spec: dict, method: str, seed: int, result: dict, binary_name: str, machine: str) -> dict[str, str]:
    stop = {"capped": "wall"}.get(result["stop"], result["stop"])
    hit = bool(result["hit"]) and stop != "wall"
    evals = int(result["calls_to_hit"]) if hit and result["calls_to_hit"] != "" else 0
    return row_from_search(
        spec,
        method,
        seed,
        hit,
        float(result["best"]),
        evals,
        int(result["charged"]),
        float(result["wall_s"]),
        stop,
        binary_name,
        machine,
    )


def run_anneal(spec, seed, binaries, env, repo, work, start, machine):
    kind = spec["kind"]
    if kind == "lj":
        from campaign import run_paper

        result = run_paper(binaries["h2h_gmin_proof"], repo, spec["name"], seed, work)
        return harness_row(spec, "anneal", seed, result, "h2h_gmin_proof", machine)
    if kind == "metal":
        one = dict(env)
        if spec.get("pass_reference"):
            one["H2H_REFERENCE"] = f"{spec['reference']:.12f}"
        proc = run_command(
            [str(binaries["h2h_metal"]), "run", spec["binary_system"], str(seed), str(start)],
            one,
            repo,
        )
        check_ok(proc, f"anneal {spec['name']} {seed}")
        (work / "stdout.txt").write_text(proc.stdout)
        (work / "stderr.txt").write_text(proc.stderr)
        return anneal_from_csv(spec, seed, proc.stdout, "h2h_metal", machine)
    if kind == "covalent":
        proc = run_command(
            [str(binaries["h2h_covalent"]), "run", spec["name"], str(seed), str(work / "best.txt")],
            env,
            repo,
        )
        check_ok(proc, f"anneal {spec['name']} {seed}")
        (work / "stdout.txt").write_text(proc.stdout)
        (work / "stderr.txt").write_text(proc.stderr)
        return anneal_from_csv(spec, seed, proc.stdout, "h2h_covalent", machine)
    stop = spec["reference"] + HIT_TOL
    prefix = work / "water"
    proc = run_command(
        [
            str(binaries["h2h_seed"]),
            "tip4p",
            str(spec["molecules"]),
            "0",
            str(spec["budget"]),
            str(seed),
            "recommended",
            repr(stop),
            str(prefix),
        ],
        env,
        work,
    )
    check_ok(proc, f"anneal {spec['name']} {seed}")
    (work / "stdout.txt").write_text(proc.stdout)
    (work / "stderr.txt").write_text(proc.stderr)
    from target_tol import energy_hits

    fields = {}
    quenched = None
    for line in proc.stdout.splitlines():
        if line.startswith("h2h_row "):
            fields = dict(part.split("=", 1) for part in line.split()[1:])
        elif line.startswith("best:"):
            quenched = float(line.split(":", 1)[1])
    if not fields or quenched is None:
        raise RuntimeError(f"anneal {spec['name']} {seed} returned no row")
    reported = fields["hit"] == "1"
    scored = energy_hits(quenched, spec["reference"], HIT_TOL)
    if scored != reported:
        raise RuntimeError(
            f"anneal {spec['name']} {seed} quenched hit {scored} disagrees with reported {reported}"
        )
    return row_from_search(
        spec,
        "anneal",
        seed,
        scored,
        quenched,
        int(fields["evals_to_hit"]),
        int(fields["charged"]),
        float(fields["wall_s"]),
        "hit" if scored else "budget",
        "h2h_seed",
        machine,
    )


def run_other(spec, method, seed, binaries, work, start, machine, campaign, asg2):
    if spec["kind"] == "lj":
        if method == "ase_mh":
            result = campaign.run_ase_mh(spec["name"], seed, spec["budget"], work, start, WALL_CAP_S)
        else:
            result = campaign.run_gofee(spec["name"], seed, spec["budget"], work, WALL_CAP_S)
        return harness_row(spec, method, seed, result, "h2h_gmin_proof", machine)
    binary = binaries["asg_eval"]
    info = asg2.read_spec(binary, spec["name"])
    positions = asg2.read_start(binary, spec["name"], seed, start, info["n"])
    if method == "ase_mh":
        result = asg2.run_ase_mh(binary, info, seed, work, positions)
    else:
        result = asg2.run_gofee(binary, info, seed, work)
    return harness_row(spec, method, seed, result, "asg_eval", machine)


def release_helper() -> None:
    """Stop a helper the search library may have left on this core."""
    try:
        import ray
    except ImportError:
        return
    if ray.is_initialized():
        ray.shutdown()


def load_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0]) != COLUMNS:
        raise SystemExit("rows.csv header does not match the registered columns")
    done = set()
    for row in rows:
        if row["revision"] != CODE_REV:
            raise SystemExit("a stored row names a different revision")
        key = (row["system"], row["method"], row["seed"])
        if key in done:
            raise SystemExit(f"duplicate row {key}")
        done.add(key)
    return done


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def preflight(args, env, binaries) -> None:
    require_core0()
    require_versions()
    require_tree(args.repo)
    require_binaries(binaries)
    require_machine(args.hosts)
    check_budgets(binaries, env, args.repo)
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts / "ase_gofee_h2h"))
    sys.path.insert(0, str(scripts))
    import campaign
    import asg2

    campaign.REPO = args.repo
    campaign.require_versions()
    campaign.warmup()
    start = args.runs / "starts" / "lj38" / "seed2000.txt"
    campaign.ensure_start(binaries["h2h_gmin_proof"], args.repo, "lj38", 2000, start)
    campaign.check_potential(binaries["h2h_gmin_proof"], args.repo, start)
    asg2.check_fixture(binaries["asg_eval"])
    spec = asg2.read_spec(binaries["asg_eval"], "cu38")
    asg2.check_gradient(binaries["asg_eval"], spec, args.runs)
    return campaign, asg2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=HERE / "rows.csv")
    parser.add_argument("--summary", type=Path, default=HERE / "summary.csv")
    parser.add_argument("--hosts", type=Path, default=HERE / "hosts.csv")
    parser.add_argument("--lj-binary", type=Path, required=True)
    parser.add_argument("--metal-binary", type=Path, required=True)
    parser.add_argument("--covalent-binary", type=Path, required=True)
    parser.add_argument("--water-binary", type=Path, required=True)
    parser.add_argument("--morse-binary", type=Path, required=True)
    parser.add_argument("--eval-binary", type=Path, required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.runs = args.runs.resolve()
    args.rows = args.rows.resolve()
    args.runs.mkdir(parents=True, exist_ok=True)
    env = pin_environment(args.runs, args.library)
    binaries = {
        "h2h_gmin_proof": args.lj_binary.resolve(),
        "h2h_morse": args.morse_binary.resolve(),
        "h2h_metal": args.metal_binary.resolve(),
        "h2h_covalent": args.covalent_binary.resolve(),
        "h2h_seed": args.water_binary.resolve(),
        "asg_eval": args.eval_binary.resolve(),
    }
    campaign, asg2 = preflight(args, env, binaries)
    if args.check:
        print("check-ok", flush=True)
        return 0
    machine = require_machine(args.hosts)
    done = load_done(args.rows)
    jobs = [
        (spec, seed, method)
        for spec in SYSTEMS
        for seed in SEEDS
        for method in arm_order(seed)
    ]
    print(f"jobs {len(jobs)} stored {len(done)}", flush=True)
    for index, (spec, seed, method) in enumerate(jobs, start=1):
        key = (spec["name"], method, str(seed))
        if key in done:
            continue
        os.chdir(args.repo)
        print(f"start {index}/{len(jobs)} {spec['name']} {method} seed={seed}", flush=True)
        start = args.runs / "starts" / spec["name"] / f"seed{seed}.txt"
        work = args.runs / spec["name"] / method / f"seed{seed}"
        try:
            if spec["kind"] == "lj":
                write_coords(binaries["h2h_gmin_proof"], spec["name"], seed, start, env, args.repo)
            elif spec["kind"] == "metal":
                write_coords(binaries["h2h_metal"], spec["binary_system"], seed, start, env, args.repo)
            elif spec["kind"] == "covalent":
                write_coords(binaries["h2h_covalent"], spec["name"], seed, start, env, args.repo)
            else:
                proc = run_command([str(binaries["asg_eval"]), "start", spec["name"], str(seed)], env, args.repo)
                check_ok(proc, f"start {spec['name']} {seed}")
                start.parent.mkdir(parents=True, exist_ok=True)
                start.write_text(proc.stdout)
            prepare_work(work)
            if method == "anneal":
                row = run_anneal(spec, seed, binaries, env, args.repo, work, start, machine)
            else:
                row = run_other(spec, method, seed, binaries, work, start, machine, campaign, asg2)
        except Exception:
            traceback.print_exc()
            release_helper()
            return 1
        append_row(args.rows, row)
        done.add(key)
        release_helper()
        print(
            f"done {spec['name']} {method} seed={seed} hit={row['hit']} stop={row['stop']} "
            f"charged={row['charged']} best={row['best']} wall={row['wall_s']}",
            flush=True,
        )
    from summarize import write_summary

    write_summary(load_rows(args.rows), args.summary, [item["name"] for item in SYSTEMS], REFERENCES)
    print("finished", flush=True)
    return 0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
