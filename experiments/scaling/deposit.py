#!/usr/bin/env python3
"""Deposit strong scaling tables from the raw cell logs.

The quartile is Hyndman-Fan type 7. Speedup and parallel efficiency use
the median wall time against one worker. Every repeat is kept.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


def quantile_type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("empty sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    if lower >= count:
        return ordered[-1]
    if lower < 1:
        return ordered[0]
    left = ordered[lower - 1]
    if fraction == 0.0:
        return left
    return left + fraction * (ordered[lower] - left)


def karp_flatt(speedup: float, workers: int) -> float | None:
    if workers <= 1 or speedup <= 0.0:
        return None
    return (1.0 / speedup - 1.0 / workers) / (1.0 - 1.0 / workers)


def check_type7() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if quantile_type7(sample, 0.25) != 2.25:
        raise SystemExit("type 7 lower quartile mismatch")
    if quantile_type7(sample, 0.5) != 3.5:
        raise SystemExit("type 7 median mismatch")
    if quantile_type7(sample, 0.75) != 4.75:
        raise SystemExit("type 7 upper quartile mismatch")


def parse_log(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    in_stderr = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "stderr:" or line.startswith("stderr:"):
            in_stderr = True
            continue
        if in_stderr and not (
            line.startswith("exit:") or line.startswith("exit_code:")
        ):
            continue
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key] = value
    return fields


def slug(cpu_model: str, logical_cpus: str) -> str:
    letters = []
    for character in cpu_model.lower():
        if character.isalnum():
            letters.append(character)
        else:
            letters.append("-")
    compact = "".join(letters)
    while "--" in compact:
        compact = compact.replace("--", "-")
    return f"{compact.strip('-')}-{logical_cpus}cpu"


def main() -> int:
    check_type7()
    root = Path("experiments/scaling/results/raw")
    logs = sorted(root.glob("*/*/r*.txt"))
    if not logs:
        print("no raw logs", file=sys.stderr)
        return 1
    rows = []
    for path in logs:
        fields = parse_log(path)
        if fields.get("exit") != "0":
            print(f"nonzero exit in {path.name}", file=sys.stderr)
            return 1
        if fields.get("scaling") != "strong":
            print(f"not a strong scaling log: {path.name}", file=sys.stderr)
            return 1
        rows.append(fields)
    required = [
        "cpu_model",
        "logical_cpus",
        "memory_kib",
        "gpu",
        "atoms",
        "walkers",
        "generations",
        "quench_steps",
        "dmc_budget",
    ]
    anchor = rows[0]
    for fields in rows[1:]:
        for key in required:
            if fields.get(key) != anchor.get(key):
                print(f"logs disagree on {key}", file=sys.stderr)
                return 1
    identity_keys = [
        "best_energy",
        "best_energy_bits",
        "evaluations",
        "decisions",
        "n_evals",
        "n_grads",
        "n_accepts",
        "best_pos_fnv",
    ]
    identity: dict[tuple[str, str], dict[str, str]] = {}
    for fields in rows:
        key = (fields["method"], fields["repeat"])
        signature = {
            name: fields[name] for name in identity_keys if name in fields
        }
        previous = identity.setdefault(key, signature)
        if previous != signature:
            print(
                f"{fields['method']} repeat {fields['repeat']} changed with the worker count",
                file=sys.stderr,
            )
            return 1
    physical_values = [
        int(fields["physical_cpus"])
        for fields in rows
        if fields.get("physical_cpus", "").isdigit()
    ]
    logical_values = [int(fields["logical_cpus"]) for fields in rows]
    cap = min(physical_values) if physical_values else min(logical_values)
    paper_rows = [fields for fields in rows if int(fields["workers"]) <= cap]
    extra_rows = [fields for fields in rows if int(fields["workers"]) > cap]
    def grouped_of(selected: list[dict[str, str]]) -> dict[tuple[str, str], list[float]]:
        grouped: dict[tuple[str, str], list[float]] = {}
        for fields in selected:
            key = (fields["method"], fields["workers"])
            grouped.setdefault(key, []).append(float(fields["wall_s"]))
        for times in grouped.values():
            if len(times) < 6:
                print("fewer than 6 repeats", file=sys.stderr)
                raise SystemExit(1)
        return grouped

    def table_text(
        selected: list[dict[str, str]],
        baselines: dict[str, float],
        intro: list[str],
    ) -> tuple[str, str, str]:
        grouped = grouped_of(selected)
        summary_lines = [
            "method,workers,n,median_wall_s,q1_wall_s,q3_wall_s,iqr_wall_s,speedup,parallel_efficiency,karp_flatt"
        ]
        long_lines = ["method,workers,repeat,wall_s"]
        for fields in selected:
            long_lines.append(
                f"{fields['method']},{fields['workers']},{fields['repeat']},{fields['wall_s']}"
            )
        methods = sorted({key[0] for key in grouped})
        body = []
        for method in methods:
            base_median = baselines[method]
            worker_ids = sorted({int(key[1]) for key in grouped if key[0] == method})
            for workers in worker_ids:
                times = grouped[(method, str(workers))]
                median = quantile_type7(times, 0.5)
                q1 = quantile_type7(times, 0.25)
                q3 = quantile_type7(times, 0.75)
                iqr = q3 - q1
                speedup = base_median / median
                efficiency = speedup / workers
                fraction = karp_flatt(speedup, workers)
                fraction_csv = "" if fraction is None else f"{fraction:.9f}"
                fraction_md = "n/a" if fraction is None else f"{fraction:.6f}"
                summary_lines.append(
                    f"{method},{workers},{len(times)},{median:.9f},{q1:.9f},{q3:.9f},"
                    f"{iqr:.9f},{speedup:.6f},{efficiency:.6f},{fraction_csv}"
                )
                body.append(
                    f"| {method} | {workers} | {len(times)} | {median:.6f} | {q1:.6f} | "
                    f"{q3:.6f} | {iqr:.6f} | {speedup:.4f} | {efficiency:.4f} | {fraction_md} |"
                )
        cpu = anchor["cpu_model"]
        logical = anchor["logical_cpus"]
        memory = anchor["memory_kib"]
        gib = int(memory) / (1024 * 1024)
        physical = anchor.get("physical_cpus", str(cap))
        header = [
            "# Strong scaling on this host",
            "",
            f"Host: {cpu}, {physical} physical cores, {logical} logical CPUs, {memory} KiB memory ({gib:.2f} GiB).",
            f"GPU: {anchor['gpu']}.",
            (
                "Fixed problem: Lennard-Jones, "
                f"{anchor['atoms']} atoms, {anchor['walkers']} walkers, "
                f"{anchor['generations']} generations, "
                f"quench length {anchor['quench_steps']}, "
                f"diffusion Monte Carlo budget {anchor['dmc_budget']}."
            ),
            *intro,
            "Six repeats sit behind each row.",
            "Hyndman-Fan type 7 supplies the median and the quartile gap.",
            "Speedup divides the one-worker median by the row median.",
            "Parallel efficiency divides that speedup by the worker count.",
            "The Karp-Flatt column is the serial fraction from those unrounded medians.",
            "",
            "| method | workers | n | median wall s | Q1 | Q3 | quartile gap | speedup | efficiency | Karp-Flatt |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        text = "\n".join(header + body) + "\n"
        return "\n".join(summary_lines) + "\n", "\n".join(long_lines) + "\n", text

    paper_grouped = grouped_of(paper_rows)
    baselines = {
        method: quantile_type7(paper_grouped[(method, "1")], 0.5)
        for method in {key[0] for key in paper_grouped}
    }
    summary, long_text, text = table_text(
        paper_rows,
        baselines,
        [
            "The worker count is the thread pool.",
            f"Published rows use at most {cap} workers, the physical core count on this host.",
        ],
    )
    out_dir = Path("experiments/scaling/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stable_csv = out_dir / "pop_strong_scaling.csv"
    stable_md = out_dir / "pop_strong_scaling.md"
    long_csv = out_dir / "pop_strong_scaling_long.csv"
    stable_csv.write_text(summary, encoding="utf-8")
    stable_md.write_text(text, encoding="utf-8")
    long_csv.write_text(long_text, encoding="utf-8")
    named = slug(anchor["cpu_model"], anchor["logical_cpus"])
    (out_dir / f"{named}.csv").write_text(summary, encoding="utf-8")
    (out_dir / f"{named}.md").write_text(text, encoding="utf-8")
    print(f"wrote {stable_csv}")
    print(f"wrote {named}.csv")
    if extra_rows:
        extra_summary, extra_long, extra_text = table_text(
            extra_rows,
            baselines,
            [
                "These worker counts exceed the physical core count on this host.",
                "The rows record oversubscription, and they stay out of the published table.",
            ],
        )
        (out_dir / "oversubscribe.csv").write_text(extra_summary, encoding="utf-8")
        (out_dir / "oversubscribe.md").write_text(extra_text, encoding="utf-8")
        (out_dir / "oversubscribe_long.csv").write_text(extra_long, encoding="utf-8")
        print("wrote oversubscribe.md")
    write_dmc_phases(rows, cap, out_dir)
    write_dmc_quality(rows, cap, out_dir)
    return 0


def write_dmc_phases(rows: list[dict[str, str]], cap: int, out_dir: Path) -> None:
    phase_keys = ["setup_s", "selection_s", "pes_s", "sync_s", "alloc_s"]
    detail_keys = ["sync_batch_s", "sync_polish_s", "sync_wave_s"]
    dmc = [fields for fields in rows if fields.get("method") == "dmc_population"]
    if not dmc:
        print("no diffusion rows", file=sys.stderr)
        raise SystemExit(1)
    logical = int(dmc[0]["logical_cpus"])
    physical = int(dmc[0].get("physical_cpus", str(cap)))
    for fields in dmc:
        for key in phase_keys + detail_keys + [
            "phase_sum_s",
            "thread_order",
            "affinity_cpus",
            "cpu_mhz_before",
            "cpu_mhz_after",
        ]:
            if key not in fields:
                print(f"missing {key}", file=sys.stderr)
                raise SystemExit(1)
        if "(R)" in " ".join(fields.values()) or "(TM)" in " ".join(fields.values()):
            print("trademark mark in a raw log", file=sys.stderr)
            raise SystemExit(1)
        wall = float(fields["wall_s"])
        total = sum(float(fields[key]) for key in phase_keys)
        if abs(total - wall) > 1e-6 or abs(float(fields["phase_sum_s"]) - wall) > 1e-6:
            print("phase walls do not sum to the run wall", file=sys.stderr)
            raise SystemExit(1)
        if fields["thread_order"] != "round-robin":
            print("thread order is not round-robin", file=sys.stderr)
            raise SystemExit(1)
        for label in ("cpu_mhz_before", "cpu_mhz_after"):
            samples = [part for part in fields[label].split(",") if part]
            if len(samples) != logical:
                print(f"{label} does not cover every logical CPU", file=sys.stderr)
                raise SystemExit(1)
        workers = int(fields["workers"])
        cpus = [int(part) for part in fields["affinity_cpus"].split(",") if part]
        if len(cpus) != workers:
            print("affinity list length mismatch", file=sys.stderr)
            raise SystemExit(1)
        expected = [index % physical for index in range(workers)]
        if cpus != expected:
            print(f"affinity {cpus} != {expected}", file=sys.stderr)
            raise SystemExit(1)
    grouped: dict[int, list[dict[str, str]]] = {}
    for fields in dmc:
        workers = int(fields["workers"])
        if workers <= cap:
            grouped.setdefault(workers, []).append(fields)
    lines = [
        "workers,n,median_wall_s,median_setup_s,median_selection_s,median_pes_s,median_sync_s,median_alloc_s,median_sync_batch_s,median_sync_polish_s,median_sync_wave_s"
    ]
    body = [
        "# Diffusion phase walls",
        "",
        "Hyndman-Fan type 7 gives every median in this table.",
        "On each run, the five phase walls add up to that run's wall.",
        "Those phases are setup, selection, potential-energy calls, synchronization, and allocation.",
        "Three later columns split synchronization into the evaluation batch, the per-control polish, and the endgame wave.",
        "Pool thread i is pinned to physical core i, wrapping after the last core.",
        "Worker counts above the physical core count stay out of this table.",
        "",
        "| workers | n | wall s | setup s | selection s | potential s | sync s | allocation s | batch sync s | polish sync s | wave sync s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for workers in sorted(grouped):
        sample = grouped[workers]
        if len(sample) < 6:
            print("fewer than 6 diffusion repeats", file=sys.stderr)
            raise SystemExit(1)
        medians = {
            key: quantile_type7([float(item[key]) for item in sample], 0.5)
            for key in ["wall_s", *phase_keys, *detail_keys]
        }
        lines.append(
            f"{workers},{len(sample)},{medians['wall_s']:.9f},{medians['setup_s']:.9f},"
            f"{medians['selection_s']:.9f},{medians['pes_s']:.9f},{medians['sync_s']:.9f},"
            f"{medians['alloc_s']:.9f},{medians['sync_batch_s']:.9f},"
            f"{medians['sync_polish_s']:.9f},{medians['sync_wave_s']:.9f}"
        )
        body.append(
            f"| {workers} | {len(sample)} | {medians['wall_s']:.6f} | {medians['setup_s']:.6f} | "
            f"{medians['selection_s']:.6f} | {medians['pes_s']:.6f} | {medians['sync_s']:.6f} | "
            f"{medians['alloc_s']:.6f} | {medians['sync_batch_s']:.6f} | "
            f"{medians['sync_polish_s']:.6f} | {medians['sync_wave_s']:.6f} |"
        )
    (out_dir / "dmc_phases.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "dmc_phases.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print("wrote dmc_phases.md")


def write_dmc_quality(rows: list[dict[str, str]], cap: int, out_dir: Path) -> None:
    dmc = [fields for fields in rows if fields.get("method") == "dmc_population"]
    grouped: dict[int, list[dict[str, str]]] = {}
    for fields in dmc:
        workers = int(fields["workers"])
        if "first_hit_work" not in fields or "wave_width" not in fields:
            print("diffusion log missing first-hit or wave width", file=sys.stderr)
            raise SystemExit(1)
        width = int(fields["wave_width"])
        if width < 1:
            print(f"wave width {width} is empty", file=sys.stderr)
            raise SystemExit(1)
        if workers <= cap:
            grouped.setdefault(workers, []).append(fields)
    lines = [
        "workers,n,median_best_energy,energy_q1,energy_q3,energy_iqr,median_first_hit,first_hit_q1,first_hit_q3,first_hit_iqr,median_wave_width"
    ]
    body = [
        "# Diffusion solution quality",
        "",
        "Lower energy is better.",
        "First-hit counts potential calls spent when the final best value first appears.",
        "Hyndman-Fan type 7 supplies the median and the quartile gap.",
        "Wave width is the distinct elite count on that run.",
        "",
        "| workers | n | median energy | energy gap | median first hit | first-hit gap | median wave width |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for workers in sorted(grouped):
        sample = grouped[workers]
        if len(sample) < 6:
            print("fewer than 6 diffusion repeats", file=sys.stderr)
            raise SystemExit(1)
        energy = [float(item["best_energy"]) for item in sample]
        hits = [float(item["first_hit_work"]) for item in sample]
        widths = [float(item["wave_width"]) for item in sample]
        energy_med = quantile_type7(energy, 0.5)
        energy_q1 = quantile_type7(energy, 0.25)
        energy_q3 = quantile_type7(energy, 0.75)
        hit_med = quantile_type7(hits, 0.5)
        hit_q1 = quantile_type7(hits, 0.25)
        hit_q3 = quantile_type7(hits, 0.75)
        width_med = quantile_type7(widths, 0.5)
        lines.append(
            f"{workers},{len(sample)},{energy_med:.16f},{energy_q1:.16f},{energy_q3:.16f},"
            f"{energy_q3 - energy_q1:.16f},{hit_med:.6f},{hit_q1:.6f},{hit_q3:.6f},"
            f"{hit_q3 - hit_q1:.6f},{width_med:.6f}"
        )
        body.append(
            f"| {workers} | {len(sample)} | {energy_med:.6f} | {energy_q3 - energy_q1:.6f} | "
            f"{hit_med:.1f} | {hit_q3 - hit_q1:.1f} | {width_med:.1f} |"
        )
    (out_dir / "dmc_quality.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "dmc_quality.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print("wrote dmc_quality.md")


if __name__ == "__main__":
    raise SystemExit(main())
