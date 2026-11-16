# Copyright (c) anneal developers
"""Score iso and iso-hop on Sutton-Chen 9-6 N=78.

Seeds 9532 through 9563, two million energy calls, the Cambridge table
energy as the stop. Starts are drawn by this script into ``build/``.
``data/sc96_78_hits.csv`` has one row per seed and arm. The how-to
figure and its table are written from that file.
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "metal_h2h"))

import run_sc  # noqa: E402
from sc96_78_common import binary, parse_run, reference_energy, write_start  # noqa: E402

SCRATCH = ROOT / "build" / "sc96-78"
OUT = ROOT / "data" / "sc96_78_hits.csv"
FIGURE = SCRATCH / "hits.png"
HOWTO = ROOT / "docs" / "orgmode" / "howto" / "sutton-chen-head-to-head.org"
SEED0 = 9532
SEED1 = 9563
BUDGET = 2_000_000
N = 78
ARMS = ("gmin", "iso", "iso-hop")
WORKERS = 6
GMIN_ROOT = ROOT / "build" / "gmin-sc"
GMIN = GMIN_ROOT / "bin" / "GMIN"
GMIN_URL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
GMIN_SHA256 = "3ad3c5ad98dbc03a19a7deec69a6b1711983cf2f2b8d57a241dafebd60412829"
GMIN_REV = "aac1548a8aec13b53ad12a8d6904625081685d2e"


def ensure_gmin() -> None:
    """Fetch the pinned Wales tarball into build/ and compile GMIN."""
    if GMIN.is_file():
        return
    fetch = ROOT / "build" / "fetch"
    fetch.mkdir(parents=True, exist_ok=True)
    tarball = fetch / "svn.tar.bz2"
    if not tarball.is_file():
        request = urllib.request.Request(GMIN_URL, headers={"User-Agent": "anneal"})
        tarball.write_bytes(urllib.request.urlopen(request, timeout=120).read())
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != GMIN_SHA256:
        raise SystemExit(f"GMIN tarball hash is {digest}, expected {GMIN_SHA256}")
    source = GMIN_ROOT / "src"
    svnrev = source / "wales" / "GMIN" / "source" / "SVNREV"
    if not svnrev.is_file():
        with tarfile.open(tarball, "r:bz2") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name.startswith(
                    (
                        "./wales/GMIN",
                        "./wales/CMakeModules",
                        "./wales/INTERFACES",
                        "./wales/SCRIPTS/all",
                        "./wales/HiRE",
                    )
                )
            ]
            archive.extractall(source, members=members)
    got = svnrev.read_text().strip()
    if got != GMIN_REV:
        raise SystemExit(f"GMIN SVNREV is {got}, expected {GMIN_REV}")
    gmin = source / "wales" / "GMIN"
    if not (gmin / "source" / "h2h_cap.f90").is_file():
        subprocess.check_call(
            ["patch", "-p1", "--forward", "--batch", "-d", str(gmin)],
            stdin=(ROOT / "bench" / "lj104-lattice" / "v2" / "h2h-gmin.patch").open(),
        )
    potential = (gmin / "source" / "potential.f90").read_text()
    if "H2H_SC_TICK" not in potential:
        subprocess.check_call(
            ["patch", "-p1", "--forward", "--batch", "-d", str(gmin)],
            stdin=(ROOT / "bench" / "metal-cluster-h2h" / "gmin-sc-time.patch").open(),
        )
    build = gmin / "build" / "gfortran"
    build.mkdir(parents=True, exist_ok=True)
    if not (build / "Makefile").is_file():
        subprocess.check_call(
            [
                "cmake", "../../source",
                "-DWITH_ALIGN=OFF", "-DWITH_MPI=OFF", "-DWITH_OPENMP=OFF",
                "-DWITH_SPARSE=OFF", "-DWITH_PYTHON=OFF", "-DWITH_MYBLAS=OFF",
                "-DWITH_MYLAPACK=OFF",
            ],
            cwd=build,
            env={**os.environ, "FC": "gfortran", "CC": "gcc", "CXX": "g++"},
        )
    subprocess.check_call(["make", "-j", str(os.cpu_count() or 1), "GMIN"], cwd=build)
    GMIN.parent.mkdir(parents=True, exist_ok=True)
    (GMIN).write_bytes((build / "GMIN").read_bytes())
    GMIN.chmod(0o755)


def gmin_data(seed: int) -> str:
    return run_sc.data_text("sc96-78", 90_000_000) + f"RANSEED {seed}\n"


def run_gmin(seed: int, reference: str, start: Path, dest: Path) -> None:
    work = SCRATCH / "gmin-work" / str(seed)
    work.mkdir(parents=True, exist_ok=True)
    (work / "data").write_text(gmin_data(seed))
    (work / "coords").write_text(start.read_text())
    row = work / "row.txt"
    env = os.environ.copy()
    env.update(
        {
            "H2H_BUDGET": str(BUDGET),
            "H2H_SYSTEM": "sc96-78",
            "H2H_SEED": str(seed),
            "H2H_TARGET": reference,
            "H2H_TOL": "0.001",
            "H2H_ROW": str(row),
            "OMP_NUM_THREADS": "1",
        }
    )
    log = work / "gmin.log"
    with log.open("w") as handle:
        proc = subprocess.run(
            [str(GMIN)], cwd=work, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True
        )
    if not row.is_file():
        raise RuntimeError(f"gmin seed {seed} exited {proc.returncode}\n{log.read_text()[-1500:]}")
    fields = {}
    for part in row.read_text().split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    hit = fields["hit"] == "1"
    calls = fields["evals_to_hit"] if hit else "none"
    dest.write_text(
        f"hit: {'true' if hit else 'false'}\n"
        f"hit_calls: {calls}\n"
        f"spent: {fields['charged']}\n"
        f"best: {fields['best']}\n"
    )


def one(task: tuple[str, int, str, str]) -> dict[str, str]:
    arm, seed, reference, exe = task
    start = SCRATCH / "starts" / f"seed{seed}.xyz"
    dest = SCRATCH / arm / f"seed{seed}.out"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not start.is_file():
        write_start(start, N, seed)
    if not dest.is_file():
        if arm == "gmin":
            run_gmin(seed, reference, start, dest)
        else:
            env = os.environ.copy()
            env["RAYON_NUM_THREADS"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            text = subprocess.check_output(
                [exe, arm, "9-6", str(N), str(BUDGET), str(seed), str(start), reference],
                cwd=ROOT,
                env=env,
                text=True,
            )
            dest.write_text(text)
    parsed = parse_run(dest.read_text())
    hit = parsed["hit"] == "true"
    return {
        "arm": arm,
        "seed": str(seed),
        "hit": "1" if hit else "0",
        "hit_calls": parsed["hit_calls"] if hit else "",
        "spent": parsed["spent"],
        "best": parsed["best"],
        "reference": reference,
    }


def write_csv(rows: list[dict[str, str]]) -> None:
    fields = ["arm", "seed", "hit", "hit_calls", "spent", "best", "reference"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUT, SCRATCH / "hits.csv"):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def score(rows: list[dict[str, str]]) -> str:
    plot = ROOT / "docs" / "scripts" / "plot_sc_hits.py"
    labels = []
    for arm in ARMS:
        labels.append(f"{arm}={SCRATCH / arm}")
    text = subprocess.check_output(
        [sys.executable, str(plot), str(SEED0), str(SEED1), str(BUDGET), str(FIGURE), *labels],
        cwd=ROOT,
        text=True,
    )
    return text


def result_paragraph(rows: list[dict[str, str]]) -> str:
    by_arm: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARMS}
    for row in rows:
        by_arm[row["arm"]].append(row)
    lines = [
        "The table is written from =data/sc96_78_hits.csv=.",
        "The figure is written to =build/sc96-78/hits.png=.",
        "GMIN is the Wales program built from the pinned tarball.",
        "Its file is the 9-6 row, temperature 10, and the quench block in =scripts/metal_h2h/run_sc.py=.",
    ]
    for arm in ARMS:
        arm_rows = by_arm[arm]
        hits = [row for row in arm_rows if row["hit"] == "1"]
        misses = len(arm_rows) - len(hits)
        calls = sorted(int(row["hit_calls"]) for row in hits)
        if not calls:
            median = "no hit"
        else:
            mid = len(calls) // 2
            mid = (calls[len(calls) // 2 - 1] + calls[len(calls) // 2]) / 2
            if len(calls) % 2:
                mid = calls[len(calls) // 2]
            median = f"{mid:.1f}".removesuffix(".0")
        lines.append(
            f"={arm}= hits {len(hits)} of {len(arm_rows)} seeds"
            + (f", median {median} calls to a hit." if hits else ".")
            + (f" It misses {misses}." if misses else "")
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_gmin()
    reference = f"{reference_energy(SCRATCH):.4f}"
    exe = str(binary("sc_h2h"))
    for seed in range(SEED0, SEED1 + 1):
        write_start(SCRATCH / "starts" / f"seed{seed}.xyz", N, seed)
    tasks = [
        (arm, seed, reference, exe)
        for arm in ARMS
        for seed in range(SEED0, SEED1 + 1)
    ]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(one, tasks))
    rows.sort(key=lambda row: (row["arm"], int(row["seed"])))
    write_csv(rows)
    table = score(rows).rstrip("\n") + "\n"
    paragraph = result_paragraph(rows)
    page = HOWTO.read_text()
    start = page.index("#+begin_example\nlabel hits")
    end = page.index("#+end_example", start) + len("#+end_example")
    page = page[:start] + "#+begin_example\n" + table + "#+end_example" + page[end:]
    old = (
        "Here =runs/gmin= has the results of Wales's global minimisation program (GMIN) on the same 32 starts.\n"
        "On all 32 seeds =iso-hop= reaches the global minimum, with a median of 138950.5 calls.\n"
        "GMIN needs a median of 398438 calls and solves 31 seeds.\n"
        "=iso= stops short on 7 seeds.\n"
    )
    if old in page:
        page = page.replace(old, paragraph, 1)
    else:
        anchors = (
            "The table and the figure are written from",
            "The table is written from",
        )
        start_p = next(page.index(anchor) for anchor in anchors if anchor in page)
        end_p = page.index("When a curve rises earlier")
        page = page[:start_p] + paragraph + page[end_p:]
    old_captions = (
        "Fraction of the 32 evaluation seeds of 9-6 N=78 at the global minimum against energy calls, "
        "for Wales's global minimisation program (GMIN) and two anneal arms.",
        "Fraction of seeds 9532 to 9563 of 9-6 N=78 at the global minimum against energy calls, for the iso and iso-hop arms.",
    )
    new_caption = (
        "Fraction of seeds 9532 to 9563 of 9-6 N=78 at the global minimum against energy calls, "
        "for GMIN, iso and iso-hop."
    )
    for caption in old_captions:
        page = page.replace(caption, new_caption)
    old_cmd = """#+begin_src bash
python3 docs/scripts/plot_sc_hits.py 9532 9563 2000000 hits.png \\
  gmin=runs/gmin iso=runs/iso iso-hop=runs/iso-hop
#+end_src
"""
    new_cmd = """#+begin_src bash
python3 scripts/sc96_78_hits.py
#+end_src

The script draws the figure with =docs/scripts/plot_sc_hits.py= from the run logs.
"""
    if old_cmd in page:
        page = page.replace(old_cmd, new_cmd, 1)
    page = page.replace("[[file:fig/sc96-78-hits.png]]\n", "")
    HOWTO.write_text(page)
    print(table)
    print(paragraph)


if __name__ == "__main__":
    main()
