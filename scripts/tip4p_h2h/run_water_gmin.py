#!/usr/bin/env python3
# Copyright (c) anneal developers
"""(H2O)n head-to-head, anneal's default preset against GMIN's rigid-body
TIP4P basin-hopping, at n = 8, 12, 16, 20 and 21, twelve seeds, one core.

Both arms open the same molecular centres (``h2h_seed`` writes the start, GMIN
reads it as ``coords``) and stop at 2_000_000 charged evaluations or at the
Cambridge Cluster Database header plus 0.001 kJ/mol, whichever is first.

Rows are appended to ``--rows`` with ``code_revision``, ``host``, ``cpu`` and
the sha1 of the start. A stored anneal row is reused only when its
``code_revision`` is the revision of the binary about to run; a stored GMIN
row only when its revision and start sha1 both match. The summary is built
from the current revisions' rows only.

``calls`` is ``evals_to_hit``: the charged evaluation that first reached the
target, or the budget for a miss. ``calls_median`` takes all twelve seeds,
misses at the budget; ``calls_hit_median`` takes hits only. ``fisher_p`` is
the two-sided Fisher exact test on hits, ``mwu_calls_p`` the two-sided
Mann-Whitney U test on ``calls``. Medians are Hyndman-Fan type 7.

``--sizes`` and ``--seeds`` (seeds 2000 up) widen or narrow the panel, and
``--gmin-seeds`` runs GMIN on the first that many seeds only. The summary
takes the first twelve seeds of each arm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gm_radius import gmin_block, read_minimum  # noqa: E402
from run_h2h import (  # noqa: E402
    BUDGET,
    HIT_TOL,
    SEEDS,
    VERIFY_TOL,
    fisher_exact_two_sided,
    gmin_single_point,
    mann_whitney_u_two_sided,
    run_anneal,
    run_gmin,
    type7,
)
import run_h2h  # noqa: E402

SIZES = (8, 12, 16, 20, 21)
# Cambridge Cluster Database TIP4P headers (kJ/mol) and the structure files
# they head, relative to the repository root.
CAMBRIDGE = {
    8: (-305.5183221676, "bench/tip4p-h2h/radius/TIP4P-8.xyz"),
    12: (-492.9082357122, "paper/jctc/data/system_visuals/ccd/TIP4P/xyz/TIP4P-12.xyz"),
    16: (-681.1925945898, "paper/jctc/data/system_visuals/ccd/TIP4P/xyz/TIP4P-16.xyz"),
    20: (-872.9887544952, "tests/fixtures/tip4p-20.xyz"),
    21: (-916.7074302940, "paper/jctc/data/system_visuals/ccd/TIP4P/xyz/TIP4P-21.xyz"),
}
FIELDS = [
    "system", "code", "seed", "budget", "hit", "evals_to_hit", "charged",
    "best", "wall_s", "pot_s", "outside_s", "basins", "start_sha1",
    "code_revision", "host", "cpu",
]


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def write_gm_blocks(repo: Path, gm_dir: Path) -> None:
    gm_dir.mkdir(parents=True, exist_ok=True)
    for n, (ref, rel) in CAMBRIDGE.items():
        energy, molecules = read_minimum(repo / rel)
        if f"{energy:.10f}" != f"{ref:.10f}":
            raise SystemExit(f"n={n}: {rel} header {energy} is not {ref}")
        if len(molecules) != n:
            raise SystemExit(f"n={n}: {rel} holds {len(molecules)} molecules")
        text, worst = gmin_block(molecules - molecules[:, 0, :].mean(axis=0))
        if worst > 1e-4:
            raise SystemExit(f"n={n} rigid fit off by {worst:.2e} A")
        (gm_dir / f"tip4p{n}.gm.gmin").write_text(text)


def verify(point: Path, gmin: Path, verify_data: Path, gm_dir: Path, out: Path, work: Path) -> None:
    rows = []
    run = work / "verify"
    run.mkdir(parents=True, exist_ok=True)
    for n, (ref, _) in CAMBRIDGE.items():
        block = gm_dir / f"tip4p{n}.gm.gmin"
        point_out = subprocess.run([str(point), str(n), str(block)], check=True,
                                   capture_output=True, text=True).stdout
        anneal_e = float(point_out.split("energy:")[1].split()[0])
        gmin_e = gmin_single_point(gmin, verify_data, block, run, n)
        ok = abs(anneal_e - ref) <= VERIFY_TOL and abs(gmin_e - ref) <= VERIFY_TOL
        rows.append({"n": n, "reference_kj": f"{ref:.10f}",
                     "anneal_kj": f"{anneal_e:.10f}", "gmin_kj": f"{gmin_e:.10f}",
                     "anneal_minus_ref": f"{anneal_e - ref:.2e}",
                     "gmin_minus_ref": f"{gmin_e - ref:.2e}",
                     "reproduced": int(ok)})
        print(f"verify n={n} ref={ref:.6f} anneal={anneal_e:.6f} gmin={gmin_e:.6f} ok={ok}", flush=True)
        if not ok:
            raise RuntimeError(f"n={n} not reproduced: anneal {anneal_e}, gmin {gmin_e}, ref {ref}")
    with out.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def find(rows: list[dict[str, str]], **match: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(k) == v for k, v in match.items()):
            return row
    return None


def summarise(rows: list[dict[str, str]], anneal_rev: str, gmin_rev: str) -> list[dict[str, str]]:
    out = []
    for n in SIZES:
        ref = CAMBRIDGE[n][0]
        arms = {}
        for code, rev in (("anneal", anneal_rev), ("gmin", gmin_rev)):
            by_seed = {r["seed"]: r for r in rows
                       if r["system"] == f"tip4p{n}" and r["code"] == code
                       and r["code_revision"] == rev}
            arms[code] = [by_seed[str(s)] for s in SEEDS if str(s) in by_seed]
        if any(len(v) != len(SEEDS) for v in arms.values()):
            continue
        a, g = arms["anneal"], arms["gmin"]
        a_hit = [float(r["best"]) <= ref + HIT_TOL for r in a]
        g_hit = [float(r["best"]) <= ref + HIT_TOL for r in g]
        a_calls = [float(r["evals_to_hit"]) for r in a]
        g_calls = [float(r["evals_to_hit"]) for r in g]
        ah, gh = sum(a_hit), sum(g_hit)

        def hit_median(calls: list[float], hits: list[bool]) -> str:
            kept = [c for c, h in zip(calls, hits) if h]
            return f"{type7(kept):.1f}" if kept else ""

        def med(arm: list[dict[str, str]], key: str) -> str:
            return f"{type7([float(r[key]) for r in arm]):.6f}"

        g_outside = [max(0.0, float(r["wall_s"]) - float(r["pot_s"])) for r in g]
        out.append({
            "system": f"tip4p{n}",
            "seeds": str(len(SEEDS)),
            "budget": str(BUDGET),
            "reference_kj": f"{ref:.10f}",
            "anneal_hits": f"{ah}/{len(SEEDS)}",
            "gmin_hits": f"{gh}/{len(SEEDS)}",
            "anneal_calls_median": f"{type7(a_calls):.1f}",
            "gmin_calls_median": f"{type7(g_calls):.1f}",
            "anneal_calls_hit_median": hit_median(a_calls, a_hit),
            "gmin_calls_hit_median": hit_median(g_calls, g_hit),
            "anneal_best_median_kj": med(a, "best"),
            "gmin_best_median_kj": med(g, "best"),
            "anneal_wall_median_s": med(a, "wall_s"),
            "gmin_wall_median_s": med(g, "wall_s"),
            "anneal_pot_median_s": med(a, "pot_s"),
            "anneal_outside_median_s": med(a, "outside_s"),
            "anneal_pot_total_s": f"{sum(float(r['pot_s']) for r in a):.6f}",
            "anneal_outside_total_s": f"{sum(float(r['outside_s']) for r in a):.6f}",
            "gmin_pot_median_s": med(g, "pot_s"),
            "gmin_outside_median_s": f"{type7(g_outside):.6f}",
            "fisher_p": f"{fisher_exact_two_sided(ah, len(SEEDS) - ah, gh, len(SEEDS) - gh):.6f}",
            "mwu_calls_p": f"{mann_whitney_u_two_sided(a_calls, g_calls):.6f}",
            "anneal_revision": anneal_rev,
            "gmin_revision": gmin_rev,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--point", type=Path, required=True)
    parser.add_argument("--anneal-revision", required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--gmin-revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--verify-data", type=Path, required=True)
    parser.add_argument("--gm-dir", type=Path, required=True)
    parser.add_argument("--verify-out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=SIZES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sizes", type=int, nargs="+", choices=SIZES)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--gmin-seeds", type=int)
    args = parser.parse_args()
    if socket.gethostname() != "bench":
        print(f"hostname is {socket.gethostname()}, not bench", file=sys.stderr)
        return 2
    if Path.cwd() != Path("/work"):
        print(f"cwd is {Path.cwd()}, not /work", file=sys.stderr)
        return 2
    run_h2h.CAMBRIDGE = {n: ref for n, (ref, _) in CAMBRIDGE.items()}
    args.work.mkdir(parents=True, exist_ok=True)
    write_gm_blocks(args.repo, args.gm_dir)
    verify(args.point, args.gmin, args.verify_data, args.gm_dir, args.verify_out, args.work)

    host, cpu = socket.gethostname(), cpu_model()
    stored = read_rows(args.rows)
    sizes = (args.size,) if args.size else tuple(args.sizes or SIZES)
    first = SEEDS[0]
    seeds = (args.seed,) if args.seed is not None else tuple(range(first, first + args.seeds))
    gmin_last = first + args.gmin_seeds if args.gmin_seeds is not None else None
    for n in sizes:
        for seed in seeds:
            system = f"tip4p{n}"
            start = args.work / f"tip4p{n}-s{seed}.start.gmin"
            anneal = find(stored, system=system, code="anneal", seed=str(seed),
                          code_revision=args.anneal_revision)
            if anneal is None or not start.is_file() or sha1(start) != anneal["start_sha1"]:
                print(f"anneal n={n} seed={seed}", flush=True)
                anneal = run_anneal(args.binary, n, seed, args.work)
                anneal.update(start_sha1=sha1(start), code_revision=args.anneal_revision,
                              host=host, cpu=cpu)
                append_row(args.rows, anneal)
                stored.append({k: str(v) for k, v in anneal.items()})
            else:
                print(f"anneal n={n} seed={seed} reused", flush=True)
            start_sha = sha1(start)
            if gmin_last is not None and seed >= gmin_last:
                continue
            gmin = find(stored, system=system, code="gmin", seed=str(seed),
                        code_revision=args.gmin_revision, start_sha1=start_sha)
            if gmin is None:
                print(f"gmin n={n} seed={seed}", flush=True)
                gmin = run_gmin(args.gmin, args.data, n, seed, start, args.work)
                gmin["outside_s"] = f"{max(0.0, float(gmin['wall_s']) - float(gmin['pot_s'])):.6f}"
                gmin.update(start_sha1=start_sha, code_revision=args.gmin_revision,
                            host=host, cpu=cpu)
                append_row(args.rows, gmin)
                stored.append({k: str(v) for k, v in gmin.items()})
            else:
                print(f"gmin n={n} seed={seed} reused", flush=True)

    summary = summarise(read_rows(args.rows), args.anneal_revision, args.gmin_revision)
    if summary:
        with args.summary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary)
    for row in summary:
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
