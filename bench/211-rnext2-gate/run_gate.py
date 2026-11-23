#!/usr/bin/env python3
"""Run the r-next2 bit-identity and outside-time gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(2000, 2012))
CORE = "0"
HIT_TOL = 1.0e-3
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
DRIVERS = ("h2h_seed", "h2h_gmin_proof", "h2h_morse")
ARM_HEADS = {
    "A": "5abb4dedec6cd2e95e7b17472639529fd139b278",
    "A2": "5abb4dedec6cd2e95e7b17472639529fd139b278",
    "B": "4a2c4a6021fdc4aaf61e00f265a1bf6bb8b0f3cc",
}
ARM_TREES = {
    "A": "23bd3a7458d5b709e35a711389f9fb0f6a1f7264",
    "A2": "23bd3a7458d5b709e35a711389f9fb0f6a1f7264",
    "B": "b57a30db80d41fddce42b4509bc40d9004261779",
}
ORDERS = (
    ("A", "B", "A2", "B"),
    ("B", "A2", "B", "A"),
)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_DOMAIN_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "TBB_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENMP_NUM_THREADS",
)
CELLS = (
    {
        "name": "tip4p9",
        "kind": "tip4p",
        "size": 9,
        "budget": 2_000_000,
        "reference": -344.4352092012,
    },
    {
        "name": "tip4p10",
        "kind": "tip4p",
        "size": 10,
        "budget": 2_000_000,
        "reference": -391.0226842125,
    },
    {
        "name": "tip4p14",
        "kind": "tip4p",
        "size": 14,
        "budget": 2_000_000,
        "reference": -582.9911036211,
    },
    {
        "name": "tip4p16",
        "kind": "tip4p",
        "size": 16,
        "budget": 2_000_000,
        "reference": -681.1925945898,
    },
    {
        "name": "tip4p17",
        "kind": "tip4p",
        "size": 17,
        "budget": 2_000_000,
        "reference": -723.8077531390,
    },
    {
        "name": "lj38",
        "kind": "lj",
        "size": 38,
        "budget": 400_000,
        "reference": -173.928427,
    },
    {
        "name": "lj75",
        "kind": "lj",
        "size": 75,
        "budget": 4_000_000,
        "reference": -397.492331,
    },
    {
        "name": "m55r6",
        "kind": "morse",
        "size": 55,
        "budget": 1_000_000,
        "reference": -250.286609,
    },
)
ROW_FIELDS = (
    "cell",
    "arm",
    "tree",
    "seed",
    "position",
    "arm_repeat",
    "slot",
    "energy_hash",
    "n_recorded",
    "best",
    "best_bits",
    "hops",
    "basins",
    "visits",
    "unconverged",
    "hit",
    "calls",
    "charged",
    "outside_s",
    "budget",
    "reference",
    "binary_sha256",
    "energies",
)
RECORD_RE = re.compile(
    r"^gate_record unconverged=(\d+) energy_hash=([0-9a-f]{16}) "
    r"n_recorded=(\d+) best=(\S+) best_bits=([0-9a-f]{16}) "
    r"hops=(\d+) basins=(\d+) visits=(\d+-\d+-\d+) "
    r"outside_s=(\S+) charged=(\d+)$",
    re.M,
)
ENERGIES_RE = re.compile(r"^gate_energies((?: [0-9a-f]{16})*)$", re.M)
RSS_RE = re.compile(
    r"^rss_anon_self_kb=(\d+) exec_image=(\S+) affinity=(\S+)$",
    re.M,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv1a(words: list[str]) -> str:
    hash_value = FNV_OFFSET
    for word in words:
        raw = int(word, 16).to_bytes(8, "little")
        for byte in raw:
            hash_value ^= byte
            hash_value = (hash_value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_value:016x}"


def append(path: Path, fields: tuple[str, ...], row: dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def session(path: Path, *fields: object) -> None:
    with path.open("a", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(fields)
        handle.flush()
        os.fsync(handle.fileno())


def child_env(probe: Path) -> dict[str, str]:
    env = {
        "HOME": os.environ["HOME"],
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LD_PRELOAD": str(probe),
    }
    for name in THREAD_VARS:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


class Arm:
    def __init__(self, name: str, spec: str) -> None:
        bin_dir, source_dir = spec.split(":", 1)
        self.name = name
        self.bin_dir = Path(bin_dir).resolve()
        self.source = Path(source_dir).resolve()
        self.binary_sha = {
            driver: sha256_file(self.bin_dir / driver) for driver in DRIVERS
        }
        self.driver_sha = {
            driver: sha256_file(self.source / "examples" / f"{driver}.rs")
            for driver in DRIVERS
        }

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.source), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def verify(self) -> None:
        if self.git("status", "--porcelain"):
            raise RuntimeError(f"{self.name} source is dirty")
        if self.git("rev-parse", "HEAD^") != ARM_HEADS[self.name]:
            raise RuntimeError(f"{self.name} production head mismatch")
        if self.git("rev-parse", "HEAD^^{tree}") != ARM_TREES[self.name]:
            raise RuntimeError(f"{self.name} production tree mismatch")
        for driver, digest in self.binary_sha.items():
            if sha256_file(self.bin_dir / driver) != digest:
                raise RuntimeError(f"{self.name} {driver} changed")


def parse_arms(specs: list[str]) -> dict[str, Arm]:
    arms = {name: Arm(name, value) for name, value in (spec.split("=", 1) for spec in specs)}
    if set(arms) != {"A", "A2", "B"}:
        raise RuntimeError("A, A2, and B arms are required")
    for driver in DRIVERS:
        hashes = {arm.driver_sha[driver] for arm in arms.values()}
        if len(hashes) != 1:
            raise RuntimeError(f"{driver} sources differ")
    patch_ids = set()
    for arm in arms.values():
        diff = subprocess.run(
            ["git", "-C", str(arm.source), "diff", "HEAD^", "HEAD"],
            check=True,
            capture_output=True,
        )
        identified = subprocess.run(
            ["git", "patch-id", "--stable"],
            check=True,
            capture_output=True,
            input=diff.stdout,
        ).stdout.split()[0].decode()
        patch_ids.add(identified)
        arm.verify()
    if len(patch_ids) != 1:
        raise RuntimeError("harness diffs differ")
    return arms


def check_binary_table(path: Path, arms: dict[str, Arm]) -> None:
    text = path.read_text()
    found: dict[tuple[str, str], str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        kind, driver, digest = line.split()
        found[(kind, driver)] = digest
    for arm in arms.values():
        for driver in DRIVERS:
            if found.get((arm.name, driver)) != arm.binary_sha[driver]:
                raise RuntimeError(f"binary table mismatch for {arm.name} {driver}")
            if found.get(("source", driver)) != arm.driver_sha[driver]:
                raise RuntimeError(f"source table mismatch for {driver}")


def load_done(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0]) != list(ROW_FIELDS):
        raise RuntimeError(f"unexpected header in {path.name}")
    return {(row["arm"], int(row["seed"]), int(row["position"])) for row in rows}


def next_slot(out: Path, cells: list[dict[str, object]]) -> int:
    slot = 0
    for cell in cells:
        path = out / f"rows-{cell['name']}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                slot = max(slot, int(row["slot"]))
    return slot


def run_measured(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, str, str, int, str, str]:
    pid = os.fork()
    if pid == 0:
        try:
            os.chdir(cwd)
            out_fd = os.open(str(stdout_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            err_fd = os.open(str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            os.close(out_fd)
            os.close(err_fd)
            os.execvpe(command[0], command, env)
        except BaseException as error:  # noqa: BLE001
            os.write(2, f"exec failed: {error}\n".encode())
            os._exit(127)
    _, status = os.waitpid(pid, 0)
    stdout = stdout_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace")
    reports = RSS_RE.findall(stderr)
    if len(reports) != 1:
        raise RuntimeError(f"self RSS reports: {len(reports)}")
    rss, image, affinity = reports[0]
    return os.waitstatus_to_exitcode(status), stdout, stderr, int(rss), image, affinity


def parse_record(stdout: str) -> dict[str, str]:
    records = RECORD_RE.findall(stdout)
    energies = ENERGIES_RE.findall(stdout)
    if len(records) != 1 or len(energies) != 1:
        raise RuntimeError("record line missing")
    (
        unconverged,
        energy_hash,
        n_recorded,
        best,
        best_bits,
        hops,
        basins,
        visits,
        outside_s,
        charged,
    ) = records[0]
    words = energies[0].split()
    if len(words) != int(n_recorded) or fnv1a(words) != energy_hash:
        raise RuntimeError("recorded-energy hash mismatch")
    return {
        "unconverged": unconverged,
        "energy_hash": energy_hash,
        "n_recorded": n_recorded,
        "best": best,
        "best_bits": best_bits,
        "hops": hops,
        "basins": basins,
        "visits": visits,
        "outside_s": outside_s,
        "charged": charged,
        "energies": " ".join(words),
    }


def parse_driver_hit(stdout: str, cell: dict[str, object]) -> tuple[int, int]:
    if cell["kind"] == "tip4p":
        line = next(value for value in stdout.splitlines() if value.startswith("h2h_row "))
        fields = dict(token.split("=", 1) for token in line.split()[1:])
        return int(fields["hit"]), int(fields["evals_to_hit"])
    prefix = str(cell["name"]) + ","
    line = next(value for value in stdout.splitlines() if value.startswith(prefix))
    columns = line.split(",")
    hit = 1 if columns[4] in {"1", "true"} else 0
    return hit, int(columns[5])


def command(
    taskset: str,
    arm: Arm,
    cell: dict[str, object],
    seed: int,
    prefix: Path,
) -> tuple[list[str], str]:
    if cell["kind"] == "tip4p":
        binary = arm.bin_dir / "h2h_seed"
        stop = repr(float(cell["reference"]) + HIT_TOL)
        return [
            taskset,
            "-c",
            CORE,
            str(binary),
            "tip4p",
            str(cell["size"]),
            "0",
            str(cell["budget"]),
            str(seed),
            "recommended",
            stop,
            str(prefix),
        ], "h2h_seed"
    if cell["kind"] == "lj":
        binary = arm.bin_dir / "h2h_gmin_proof"
        return [taskset, "-c", CORE, str(binary), "run", str(cell["name"]), str(seed)], (
            "h2h_gmin_proof"
        )
    binary = arm.bin_dir / "h2h_morse"
    return [taskset, "-c", CORE, str(binary), "run", str(cell["name"]), str(seed)], "h2h_morse"


def confirm_budgets(arms: dict[str, Arm]) -> None:
    source = arms["A"].source
    with (source / "data" / "tip4p_ccd.csv").open(newline="") as handle:
        table = {int(row["n"]): float(row["energy"]) for row in csv.DictReader(handle)}
    for cell in CELLS:
        if cell["kind"] == "tip4p" and abs(table[int(cell["size"])] - float(cell["reference"])) > 1e-12:
            raise RuntimeError(f"{cell['name']} reference mismatch")
    for arm in arms.values():
        for cell in CELLS:
            if cell["kind"] == "tip4p":
                continue
            binary = "h2h_gmin_proof" if cell["kind"] == "lj" else "h2h_morse"
            output = subprocess.run(
                [str(arm.bin_dir / binary), "budget", str(cell["name"])],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            if int(output[0]) != int(cell["budget"]):
                raise RuntimeError(f"{arm.name} {cell['name']} budget mismatch")
            if abs(float(output[1]) - float(cell["reference"])) > 1e-9:
                raise RuntimeError(f"{arm.name} {cell['name']} reference mismatch")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--rss-probe", type=Path, required=True)
    parser.add_argument("--cells", default="")
    args = parser.parse_args(argv[1:])
    if os.sched_getaffinity(0) != {0}:
        os.sched_setaffinity(0, {0})
    selected = set(args.cells.split(",")) if args.cells else {cell["name"] for cell in CELLS}
    cells = [cell for cell in CELLS if cell["name"] in selected]
    if len(cells) != len(selected):
        raise RuntimeError("unknown cell")
    arms = parse_arms(args.arm)
    confirm_budgets(arms)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    check_binary_table(output / "binary-sha256.txt", arms)
    work = args.work.resolve()
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    probe = args.rss_probe.resolve()
    if not probe.is_file():
        raise RuntimeError("RSS probe missing")
    taskset = subprocess.run(["sh", "-c", "command -v taskset"], check=True, capture_output=True, text=True)
    taskset_bin = taskset.stdout.strip()
    env = child_env(probe)
    sessions = logs / "sessions.csv"
    session(sessions, "event", "cell", "seed", "value")
    session(sessions, "begin", "", "", "")
    slot = next_slot(output, cells)
    for cell in cells:
        rows_path = output / f"rows-{cell['name']}.csv"
        done = load_done(rows_path)
        session(sessions, "cell_begin", cell["name"], "", "")
        for seed_index, seed in enumerate(SEEDS):
            repeats = {"A": 0, "A2": 0, "B": 0}
            for position, arm_name in enumerate(ORDERS[seed % 2], 1):
                repeats[arm_name] += 1
                if (arm_name, seed, position) in done:
                    continue
                arm = arms[arm_name]
                slot += 1
                tag = f"{cell['name']}-s{seed}-p{position}-{arm_name.lower()}-{repeats[arm_name]}"
                prefix = logs / tag
                run, image_name = command(taskset_bin, arm, cell, seed, prefix)
                rc, stdout, _stderr, _rss, image, affinity = run_measured(
                    run,
                    logs,
                    env,
                    Path(f"{prefix}.out"),
                    Path(f"{prefix}.err"),
                )
                Path(f"{prefix}.rc").write_text(f"{rc}\n")
                if rc != 0 or image != image_name or affinity != CORE:
                    raise RuntimeError(f"{tag} failed rc={rc} image={image} affinity={affinity}")
                parsed = parse_record(stdout)
                hit, calls = parse_driver_hit(stdout, cell)
                if int(parsed["charged"]) <= 0:
                    raise RuntimeError(f"{tag} charged nothing")
                row: dict[str, object] = {
                    "cell": cell["name"],
                    "arm": arm_name,
                    "tree": ARM_TREES[arm_name],
                    "seed": seed,
                    "position": position,
                    "arm_repeat": repeats[arm_name],
                    "slot": slot,
                    "energy_hash": parsed["energy_hash"],
                    "n_recorded": parsed["n_recorded"],
                    "best": parsed["best"],
                    "best_bits": parsed["best_bits"],
                    "hops": parsed["hops"],
                    "basins": parsed["basins"],
                    "visits": parsed["visits"],
                    "unconverged": parsed["unconverged"],
                    "hit": hit,
                    "calls": calls,
                    "charged": parsed["charged"],
                    "outside_s": parsed["outside_s"],
                    "budget": cell["budget"],
                    "reference": repr(float(cell["reference"])),
                    "binary_sha256": arm.binary_sha[image_name],
                    "energies": parsed["energies"],
                }
                append(rows_path, ROW_FIELDS, row)
                session(sessions, "row", cell["name"], seed, tag)
        session(sessions, "cell_end", cell["name"], "", "")
    for arm in arms.values():
        arm.verify()
    session(sessions, "complete", "", "", slot)


if __name__ == "__main__":
    main(sys.argv)
