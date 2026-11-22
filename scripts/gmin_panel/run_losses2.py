#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Pair-scaled Morse against the parent preset and GMIN.

The parent binary is the preset with pair scaling off. The candidate is
this tree with ``H2H_PAIR_SCALE=1`` on Morse only. Lennard-Jones and
Gupta keep the preset. Morse seeds share the parent's ``coords`` start,
so the two Morse arms differ in the length scale of the search rather
than in the draw. Rows and the summary are appended under ``$OUTPUT``
or ``/work/gmin-losses2``. A path inside this repository is refused.

``code_revision`` is the parent commit on baseline rows, this commit on
the pair-scaled rows, and the GMIN SVNREV on GMIN rows. ``host`` is the
UTS hostname. The process refuses to run unless that hostname is bench
and the affinity is core 0.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_panel as panel  # noqa: E402
from gmin_pin import KNOWN_REVISIONS  # noqa: E402

PAIR_ENV = {"H2H_PAIR_SCALE": "1"}
CODES = ("baseline", "fix2", "gmin")


def output_dir(path: Path, root: Path) -> Path:
    """A directory under ``/work`` or ``$OUTPUT``, never inside the repo."""
    resolved = path.resolve()
    root = root.resolve()
    if resolved == root or root in resolved.parents:
        raise SystemExit(f"{resolved} is inside the repository")
    return resolved


def default_output(root: Path) -> Path:
    raw = os.environ.get("OUTPUT") or "/work/gmin-losses2"
    return output_dir(Path(raw), root)


def write_start(bin_dir: Path, tree: Path, system: str, seed: int, path: Path) -> None:
    _driver, binary = panel.resolve_driver(bin_dir, tree, system)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(bin_dir / binary), "coords", system, str(seed), str(path)],
        cwd=tree,
        env=panel.single_thread_env(),
        check=True,
    )


def gmin_label(gmin_source: Path, svnrev: Path) -> str:
    """The SVNREV of a new GMIN build, refused unless it is the pin its note describes."""
    source = dict(line.split("=", 1) for line in gmin_source.read_text().splitlines() if "=" in line)
    gmin_revision = panel.read_svnrev(svnrev)
    if source.get("svnrev") != gmin_revision:
        raise SystemExit(f"{gmin_source} does not describe {svnrev}")
    if source.get("tarball_sha256") != panel.TARBALL_SHA256:
        raise SystemExit(f"{gmin_source} has another GMIN tarball")
    return gmin_revision


def selected(rows: list[dict[str, str]], system: str, code: str, revision: str, base: str) -> list[dict[str, str]]:
    wanted = KNOWN_REVISIONS if code == "gmin" else (base,) if code == "baseline" else (revision,)
    found: dict[int, dict[str, str]] = {}
    for row in rows:
        if (
            row["system"] == system
            and row["code"] == code
            and row["code_revision"] in wanted
            and row["harness_revision"] == revision
            and int(row["seed"]) in panel.SEEDS
        ):
            found[int(row["seed"])] = row
    return [found[seed] for seed in sorted(found)]


def comparison(system: str, left_name: str, right_name: str, left: list[dict[str, str]], right: list[dict[str, str]],
               revision: str, base: str) -> dict[str, str]:
    def nums(group: list[dict[str, str]], key: str) -> list[float]:
        return [float(row[key]) for row in group]

    left_hits = sum(row["hit"] == "1" for row in left)
    right_hits = sum(row["hit"] == "1" for row in right)
    _u, p_calls = panel.mann_whitney(nums(left, "evals_to_hit"), nums(right, "evals_to_hit"))
    return {
        "system": system,
        "budget": str(panel.PAPER_BUDGETS[system]),
        "left": left_name,
        "right": right_name,
        "left_hits": str(left_hits),
        "left_n": str(len(left)),
        "right_hits": str(right_hits),
        "right_n": str(len(right)),
        "fisher_p": f"{panel.fisher(left_hits, len(left), right_hits, len(right)):.6g}",
        "left_evals_median": f"{panel.type7(nums(left, 'evals_to_hit'), 0.5):.2f}",
        "right_evals_median": f"{panel.type7(nums(right, 'evals_to_hit'), 0.5):.2f}",
        "mannwhitney_p_calls": f"{p_calls:.6g}",
        "left_wall_median_s": f"{panel.type7(nums(left, 'wall_s'), 0.5):.6f}",
        "right_wall_median_s": f"{panel.type7(nums(right, 'wall_s'), 0.5):.6f}",
        "left_pot_median_s": f"{panel.type7(nums(left, 'pot_s'), 0.5):.6f}",
        "right_pot_median_s": f"{panel.type7(nums(right, 'pot_s'), 0.5):.6f}",
        "left_outside_median_s": f"{panel.type7(nums(left, 'outside_s'), 0.5):.6f}",
        "right_outside_median_s": f"{panel.type7(nums(right, 'outside_s'), 0.5):.6f}",
        "base_revision": base,
        "code_revision": revision,
        "host": " ".join(sorted({row["host"] for row in left + right})),
    }


def summarise(rows: list[dict[str, str]], revision: str, base: str) -> list[dict[str, str]]:
    out = []
    for system, *_rest in panel.SYSTEMS:
        baseline = selected(rows, system, "baseline", revision, base)
        pair = selected(rows, system, "fix2", revision, base)
        gmin = selected(rows, system, "gmin", revision, base)
        if len(baseline) == len(panel.SEEDS) and len(pair) == len(panel.SEEDS):
            out.append(comparison(system, "baseline", "fix2", baseline, pair, revision, base))
        if len(pair) == len(panel.SEEDS) and len(gmin) == len(panel.SEEDS):
            out.append(comparison(system, "fix2", "gmin", pair, gmin, revision, base))
    return out


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["system"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    root = Path(__file__).resolve().parents[2]
    work = output_dir(Path("/work/gmin-losses2"), root)
    if work != Path("/work/gmin-losses2").resolve():
        raise SystemExit("default work directory")
    try:
        output_dir(root / "bench" / "gmin-losses2", root)
    except SystemExit:
        pass
    else:
        raise SystemExit("a repository path was accepted")
    names = [row[0] for row in panel.SYSTEMS]
    if list(panel.PAPER_BUDGETS) != names or len(panel.SEEDS) < 6:
        raise SystemExit("panel budgets or seeds")
    if PAIR_ENV != {"H2H_PAIR_SCALE": "1"}:
        raise SystemExit("pair scale switch")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--base-bin", type=Path)
    parser.add_argument("--base-tree", type=Path)
    parser.add_argument("--head-bin", type=Path)
    parser.add_argument("--head-tree", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--gmin-source", type=Path)
    parser.add_argument("--svnrev", type=Path)
    parser.add_argument("--lj-data", type=Path)
    parser.add_argument("--morse-data", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv[1:])
    out = default_output(root)
    rows_path = args.rows or (out / "rows.csv")
    summary_path = args.summary or (out / "summary.csv")
    output_dir(rows_path.parent, root)
    if args.summarize:
        write_summary(summary_path, summarise(panel.read_rows(rows_path), args.revision, args.base_revision))
        return 0
    if socket.gethostname() != "bench":
        print(f"uts hostname is {socket.gethostname()}, expected bench", file=sys.stderr)
        return 2
    if os.sched_getaffinity(0) != {0}:
        print(f"pinned to {sorted(os.sched_getaffinity(0))}, expected core 0", file=sys.stderr)
        return 2
    needed = (args.base_bin, args.base_tree, args.head_bin, args.head_tree, args.work, args.logs,
              args.gmin_source, args.svnrev, args.lj_data, args.morse_data)
    if any(value is None for value in needed):
        parser.error("run needs both binaries, their trees, work, logs, the GMIN source note and both data files")
    for tree, revision in ((args.base_tree, args.base_revision), (args.head_tree, args.revision)):
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True, check=True
        ).stdout.strip()
        if head != revision:
            raise SystemExit(f"{tree} is at {head}, not {revision}")
    gmin_revision = gmin_label(args.gmin_source, args.svnrev)
    systems = [row[0] for row in panel.SYSTEMS if not args.system or row[0] in args.system]
    for system in systems:
        for bin_dir, tree in ((args.base_bin, args.base_tree), (args.head_bin, args.head_tree)):
            budget, reference = panel.binary_budget(bin_dir, tree, system)
            if budget != panel.PAPER_BUDGETS[system] or abs(reference - panel.spec(system)[3]) > 1e-8:
                raise SystemExit(f"{system}: budget {budget} reference {reference}")
    lj_data = args.lj_data.read_text()
    morse_data = args.morse_data.read_text()
    lj_data += "" if lj_data.endswith("\n") else "\n"
    morse_data += "" if morse_data.endswith("\n") else "\n"
    host = socket.gethostname()
    cpu = panel.cpu_model()
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {panel.row_key(row) for row in panel.read_rows(rows_path)}
    seeds = tuple(args.seed) if args.seed else panel.SEEDS
    print(f"session {session} revision {args.revision} base {args.base_revision}", flush=True)
    for system in systems:
        morse = panel.spec(system)[1] == "morse"
        driver, _binary = panel.resolve_driver(args.head_bin, args.head_tree, system)
        for seed in seeds:
            coords = None
            if morse:
                coords = args.work / system / f"{seed}.xyz"
                write_start(args.base_bin, args.base_tree, system, seed, coords)
            for code in CODES:
                code_revision = {
                    "baseline": args.base_revision,
                    "fix2": args.revision,
                    "gmin": gmin_revision,
                }[code]
                key = (system, code, str(seed), code_revision, args.revision)
                if key in done:
                    print(f"skip {system} {code} {seed}", flush=True)
                    continue
                print(f"start {system} {code} {seed}", flush=True)
                log = args.logs / system / f"{code}-{seed}.log"
                if code == "gmin":
                    result = panel.run_gmin(
                        args.base_bin, args.base_tree, system, seed,
                        args.work / system / str(seed), log, lj_data, morse_data,
                    )
                else:
                    bin_dir = args.base_bin if code == "baseline" else args.head_bin
                    tree = args.base_tree if code == "baseline" else args.head_tree
                    env = PAIR_ENV if code == "fix2" and morse else None
                    result = panel.run_anneal(
                        bin_dir, tree, system, seed, log,
                        env_overrides=env, coords=coords if morse else None,
                    )
                    if result.pop("head") not in ("", code_revision):
                        raise SystemExit(f"{system} {seed}: binary reports another head")
                row = {
                    "system": system,
                    "code": code,
                    "driver": "GMIN" if code == "gmin" else driver,
                    "seed": str(seed),
                    "budget": str(panel.PAPER_BUDGETS[system]),
                    **result,
                    "code_revision": code_revision,
                    "harness_revision": args.revision,
                    "host": host,
                    "cpu": cpu,
                    "session": session,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                panel.append_row(rows_path, row)
                done.add(key)
                print(
                    f"done {system} {code} {seed} hit={row['hit']} calls={row['evals_to_hit']} "
                    f"wall_s={row['wall_s']} pot_s={row['pot_s']} outside_s={row['outside_s']} best={row['best']}",
                    flush=True,
                )
    write_summary(summary_path, summarise(panel.read_rows(rows_path), args.revision, args.base_revision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
