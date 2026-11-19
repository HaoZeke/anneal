#!/usr/bin/env python3
"""Tables for the water hexamer code-only A/B.

Reads runs.csv written by run_ab.py and the frame table crest_frames.csv, and
prints, as markdown, the hits and timings of A against B for every arm with
Fisher exact tests on the hits and Mann-Whitney tests on calls and times, the
seeds where A and B disagree on a CREST hit, and the sensitivity of the CREST
hits to the gate, to the reference set and to the hit rule.

The frame table is large and is kept outside git. It is read through
external_manifest.csv from a local directory that holds the files named in the
manifest; the sha256 of the file is checked first. A session that has no
manifest yet can point at the frame table directly.
"""
import argparse
import contextlib
import csv
import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import external

ARMS = ("anneal", "polish_fair", "crest", "crest_fair")
CODES = ("A", "B")
REFERENCES = ("prism", "cage", "book")
OLD_GATE = 0.784112


BUILD_NOTES = """## How the builds ran

Build B runs through the force-negation shim. The native xtb path on commit bc16d35c has a known sign defect: anneal re-negates a buffer that already holds the physical force, so unpatched B walks uphill and never leaves a dissociated state (preflight.txt). The shim hands B the negated buffer, so the B numbers are the physically correct path. A source fix is pending upstream. Build A takes the buffer as the force and gets it unchanged.

The CREST arms are a timing null: the same work, CREST on the same engine with the same call budget, runs under both builds, and only the scoring code differs.
"""


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


FRAMES_FILE = "crest_frames.csv"


def load(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_frames(args):
    if args.manifest:
        if not args.external_dir:
            raise external.ExternalError("--manifest needs --external-dir")
        manifest = external.read_manifest(args.manifest)
        return external.load_rows(manifest, FRAMES_FILE, args.external_dir)
    if args.frames:
        return load(args.frames)
    raise external.ExternalError("give --manifest and --external-dir, or --frames")


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def quantile(values, q):
    values = [v for v in values if v is not None]
    return float(np.quantile(values, q)) if values else float("nan")


def fisher(hits_a, n_a, hits_b, n_b):
    table = [[hits_a, n_a - hits_a], [hits_b, n_b - hits_b]]
    return stats.fisher_exact(table, alternative="two-sided")[1]


def mann_whitney(x, y):
    x = [v for v in x if v is not None]
    y = [v for v in y if v is not None]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if len(set(x) | set(y)) == 1:
        return 1.0
    return stats.mannwhitneyu(x, y, alternative="two-sided", method="auto").pvalue


def fmt_p(value):
    if value != value:
        return "n/a"
    return f"{value:.2e}" if value < 1e-3 else f"{value:.3f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def by_arm_code(runs):
    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["arm"], row["code"])].append(row)
    return grouped


def hits_section(grouped, seeds):
    rows, paired = [], []
    for arm in ARMS:
        a, b = grouped.get((arm, "A"), []), grouped.get((arm, "B"), [])
        if not a or not b:
            continue
        hits_a = sum(int(r["hit"]) for r in a)
        hits_b = sum(int(r["hit"]) for r in b)
        rows.append([
            arm, f"{hits_a}/{len(a)}", f"{hits_b}/{len(b)}", f"{hits_a - hits_b:+d}",
            fmt_p(fisher(hits_a, len(a), hits_b, len(b))),
        ])
        hit_a = {int(r["seed"]): int(r["hit"]) for r in a}
        hit_b = {int(r["seed"]): int(r["hit"]) for r in b}
        common = sorted(set(hit_a) & set(hit_b))
        both = sum(1 for s in common if hit_a[s] and hit_b[s])
        only_a = sum(1 for s in common if hit_a[s] and not hit_b[s])
        only_b = sum(1 for s in common if hit_b[s] and not hit_a[s])
        neither = len(common) - both - only_a - only_b
        mcnemar = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        paired.append([arm, both, only_a, only_b, neither, fmt_p(mcnemar)])
    return (
        table(["arm", "A hits", "B hits", "A - B", "Fisher exact p"], rows),
        table(["arm", "both", "only A", "only B", "neither", "McNemar exact p"], paired),
    )


def timing_section(grouped):
    medians, tests = [], []
    for arm in ARMS:
        a = [r for r in grouped.get((arm, "A"), []) if r["status"] == "ok"]
        b = [r for r in grouped.get((arm, "B"), []) if r["status"] == "ok"]
        if not a or not b:
            continue
        columns = {}
        for name in ("calls", "wall_s", "in_s", "out_s"):
            columns[name] = ([number(r[name]) for r in a], [number(r[name]) for r in b])
        for code, rows, index in (("A", a, 0), ("B", b, 1)):
            medians.append([
                arm, code, len(rows),
                f"{median(columns['calls'][index]):.0f}",
                f"{median(columns['wall_s'][index]):.2f}",
                f"{median(columns['in_s'][index]):.2f}",
                f"{median(columns['out_s'][index]):.2f}",
                f"{median([number(r['final_energy_eV']) for r in rows]):.4f}",
                f"{np.mean(columns['wall_s'][index]):.2f}",
                f"{quantile(columns['wall_s'][index], 0.9):.2f}",
                f"{max(columns['wall_s'][index]):.2f}",
            ])
        tests.append([
            arm,
            fmt_p(mann_whitney(*columns["calls"])),
            fmt_p(mann_whitney(*columns["wall_s"])),
            fmt_p(mann_whitney(*columns["in_s"])),
            fmt_p(mann_whitney(*columns["out_s"])),
        ])
    return (
        table(["arm", "code", "runs ok", "median calls", "median wall s", "median in s", "median out s",
               "median final E eV", "mean wall s", "p90 wall s", "max wall s"], medians),
        table(["arm", "MWU p calls", "MWU p wall", "MWU p in", "MWU p out"], tests),
    )


def slow_section(grouped):
    rows = []
    for arm in ARMS:
        ok = {code: [r for r in grouped.get((arm, code), []) if r["status"] == "ok"] for code in CODES}
        pooled = [number(r["wall_s"]) for code in CODES for r in ok[code]]
        pooled = [v for v in pooled if v is not None]
        if not pooled:
            continue
        limit = 1.5 * median(pooled)
        for code in CODES:
            slow = [r for r in ok[code] if (number(r["wall_s"]) or 0.0) > limit]
            rest = [r for r in ok[code] if r not in slow]

            def per_call(rs):
                return median([
                    1e3 * number(r["in_s"]) / number(r["calls"]) for r in rs
                    if number(r["in_s"]) is not None and number(r["calls"])
                ])

            rows.append([
                arm, code, len(ok[code]), len(slow), ", ".join(r["seed"] for r in slow) or "-",
                f"{per_call(slow):.2f}" if slow else "-", f"{per_call(rest):.2f}",
            ])
    return table(["arm", "code", "runs ok", "wall above 1.5 x pooled median", "seeds",
                  "median ms in the engine per call, slow runs", "same, other runs"], rows)


def energy_section(grouped, reference):
    rows = []
    for arm in ARMS:
        for code in CODES:
            mine = [r for r in grouped.get((arm, code), []) if r["status"] == "ok"]
            values = [number(r["final_energy_eV"]) for r in mine]
            values = [v for v in values if v is not None]
            if not values:
                continue
            cells = [arm, code, len(values), f"{median(values):.4f}", f"{quantile(values, 0.1):.4f}",
                     f"{quantile(values, 0.9):.4f}", f"{min(values):.4f}"]
            if reference is not None:
                cells.append(sum(1 for v in values if v <= reference + 1e-3))
            rows.append(cells)
    header = ["arm", "code", "runs", "median E eV", "p10 E eV", "p90 E eV", "lowest E eV"]
    if reference is not None:
        header.append(f"runs within 1 meV of {reference:.6f} eV")
    return table(header, rows)


def anneal_gate_section(grouped, gate_new):
    gates = (0.3, 0.5, gate_new, 0.7, OLD_GATE, 1.0)
    rows = []
    for arm in ("anneal", "polish_fair"):
        for code in CODES:
            mine = [r for r in grouped.get((arm, code), []) if r["status"] == "ok"]
            if not mine:
                continue
            rmsds = [number(r["isomer_rmsd_A"]) for r in mine]
            cells = [arm, code, len(mine)]
            for gate in gates:
                cells.append(sum(1 for v in rmsds if v is not None and v <= gate))
            rows.append(cells)
    header = ["arm", "code", "runs"] + [f"RMSD <= {g:.3f}" for g in gates]
    return table(header, rows)


def order_section(runs):
    rows = []
    for arm in ARMS:
        for code in CODES:
            first = [number(r["wall_s"]) for r in runs
                     if r["arm"] == arm and r["code"] == code and r["status"] == "ok" and r["position_in_pair"] == "0"]
            second = [number(r["wall_s"]) for r in runs
                      if r["arm"] == arm and r["code"] == code and r["status"] == "ok" and r["position_in_pair"] == "1"]
            if not first or not second:
                continue
            rows.append([arm, code, len(first), len(second), f"{median(first):.2f}", f"{median(second):.2f}",
                         fmt_p(mann_whitney(first, second))])
    return table(["arm", "code", "runs first", "runs second", "median wall first s", "median wall second s",
                  "MWU p"], rows)


def drift_section(runs, block=24):
    rows = []
    for arm in ("anneal", "polish_fair"):
        for code in CODES:
            mine = [r for r in runs if r["arm"] == arm and r["code"] == code and r["status"] == "ok"]
            for start in range(0, 192, block):
                chunk = [r for r in mine if start <= int(r["seed"]) < start + block]
                if not chunk:
                    continue
                per_call = [1000 * number(r["wall_s"]) / number(r["calls"]) for r in chunk]
                rows.append([arm, code, f"{start}-{start + block - 1}", len(chunk), f"{median(per_call):.3f}"])
    return table(["arm", "code", "seeds", "runs", "median wall ms per call"], rows)


def failures_section(runs):
    bad = [r for r in runs if r["status"] != "ok"]
    text = "All runs finished with status ok." if not bad else None
    if bad:
        counts = defaultdict(int)
        for r in bad:
            counts[(r["arm"], r["code"], r["status"])] += 1
        text = table(["arm", "code", "status", "runs"], [[*k, v] for k, v in sorted(counts.items())])
    silent = []
    for r in runs:
        energy = number(r["final_energy_eV"])
        if energy is None or energy != energy or energy in (float("inf"), float("-inf")):
            silent.append([r["seed"], r["code"], r["arm"], r["status"], r["calls"], r["wall_s"], r["out_s"],
                           r["isomer_reason"] or "-", r["hit"]])
    if silent:
        text += "\n\nRuns without a final energy (counted as misses)\n\n" + table(
            ["seed", "code", "arm", "status", "calls", "wall s", "out s", "reason", "hit"], silent)
    return text


def crest_disagreements(grouped, arm):
    a = {int(r["seed"]): r for r in grouped.get((arm, "A"), [])}
    b = {int(r["seed"]): r for r in grouped.get((arm, "B"), [])}
    only_a = sorted(s for s in a if s in b and int(a[s]["hit"]) and not int(b[s]["hit"]))
    only_b = sorted(s for s in a if s in b and int(b[s]["hit"]) and not int(a[s]["hit"]))
    return only_a, only_b, a, b


def disagreement_table(grouped, arm, frame_minima):
    only_a, only_b, a, b = crest_disagreements(grouped, arm)
    rows = []
    for seed in sorted(only_a + only_b):
        cells = [seed]
        for code, row in (("A", a[seed]), ("B", b[seed])):
            minima = frame_minima.get((arm, code, seed), {})
            nearest = min(REFERENCES, key=lambda ref: minima.get(ref, float("inf"))) if minima else "-"
            cells += [
                row["hit"], row["frames_inside_gate"], row["hit_polish"] or "-",
                row["min_rmsd_A"], nearest,
            ]
        rows.append(cells)
    header = ["seed", "A hit", "A frames in gate", "A polish hit", "A min RMSD", "A nearest",
              "B hit", "B frames in gate", "B polish hit", "B min RMSD", "B nearest"]
    return only_a, only_b, table(header, rows)


def frame_minima_by_run(frames):
    minima = defaultdict(lambda: defaultdict(lambda: float("inf")))
    best = {}
    for row in frames:
        key = (row["arm"], row["code"], int(row["seed"]))
        for ref in REFERENCES:
            value = number(row[f"rmsd_{ref}_A"])
            if value is not None and value < minima[key][ref]:
                minima[key][ref] = value
        epot, rmsd = number(row["epot_Eh"]), number(row["rmsd_A"])
        if epot is not None and (key not in best or epot < best[key][0]):
            best[key] = (epot, rmsd)
    return {k: dict(v) for k, v in minima.items()}, best


def sensitivity_section(runs_by_key, minima, best, arm, gate_new):
    lines = []
    keys = defaultdict(list)
    for (a, code, seed), row in runs_by_key.items():
        if a == arm and row["frames"] != "":
            keys[code].append(seed)
    if not keys:
        return "No frames recorded for this arm."
    sets = {
        "prism": ("prism",), "cage": ("cage",), "book": ("book",),
        "prism+cage": ("prism", "cage"), "prism+book": ("prism", "book"),
        "all three": REFERENCES,
    }
    rows = []
    for label, refs in sets.items():
        cells = [label]
        for code in CODES:
            seeds = sorted(keys[code])
            n = len(seeds)
            for gate in (gate_new, OLD_GATE):
                hits = sum(1 for s in seeds
                           if min(minima.get((arm, code, s), {}).get(r, float("inf")) for r in refs) <= gate)
                cells.append(f"{hits}/{n}")
        rows.append(cells)
    header = ["reference set"]
    for code in CODES:
        header += [f"{code}-run gate {gate_new:.3f}", f"{code}-run gate {OLD_GATE:.3f}"]
    lines.append(table(header, rows))
    def run_minimum(code, seed):
        value = min(minima.get((arm, code, seed), {}).values(), default=float("inf"))
        polished = number(runs_by_key[(arm, code, seed)].get("polished_rmsd_A")) if arm == "crest_fair" else None
        return min(value, polished) if polished is not None else value

    sweep_rows = []
    for gate in (0.3, 0.4, 0.5, gate_new, 0.7, OLD_GATE, 0.9, 1.0, 1.2, 1.5):
        cells = [f"{gate:.3f}"]
        for code in CODES:
            seeds = sorted(keys[code])
            hits = sum(1 for s in seeds if run_minimum(code, s) <= gate)
            cells.append(f"{hits}/{len(seeds)}")
        sweep_rows.append(cells)
    title = "gate A (frames and polished structure)" if arm == "crest_fair" else "gate A"
    lines.append(table([title, "A-run hits", "B-run hits"], sweep_rows))
    last_rows = []
    for code in CODES:
        seeds = sorted(keys[code])
        in_gate = sum(1 for s in seeds if best.get((arm, code, s), (None, None))[1] is not None
                      and best[(arm, code, s)][1] <= gate_new)
        any_frame = sum(1 for s in seeds
                        if min(minima.get((arm, code, s), {}).values(), default=float("inf")) <= gate_new)
        last_rows.append([code, f"{any_frame}/{len(seeds)}", f"{in_gate}/{len(seeds)}"])
    lines.append(table(["run", "any frame inside gate", "lowest-energy frame inside gate"], last_rows))
    return "\n\n".join(lines)


def trajectory_section(frames, arm, per_mtd=49):
    bins = [(0, 3), (3, 10), (10, 25), (25, per_mtd)]
    rows = []
    for code in CODES:
        cells = {b: [0, 0, []] for b in bins}
        for row in frames:
            if row["arm"] != arm or row["code"] != code:
                continue
            position = int(row["frame"]) % per_mtd
            bucket = next(b for b in bins if b[0] <= position < b[1])
            cells[bucket][0] += 1
            values = [number(row[f"rmsd_{ref}_A"]) for ref in REFERENCES]
            values = [v for v in values if v is not None]
            if values:
                cells[bucket][1] += 1
                cells[bucket][2].append(min(values))
        for (low, high), (total, scored, minima) in cells.items():
            if not total:
                continue
            rows.append([
                code, f"{low}-{high - 1}", total, f"{100 * scored / total:.0f}%",
                f"{median(minima):.3f}" if minima else "-",
                f"{quantile(minima, 0.1):.3f}" if minima else "-",
                f"{min(minima):.3f}" if minima else "-",
            ])
    return table(["run", "frame within its metadynamics run", "frames", "frames with an RMSD",
                  "median RMSD to nearest reference A", "p10 A", "lowest A"], rows)


def classifier_section(frames, runs, arm):
    differing_frames = 0
    total = 0
    for row in frames:
        if row["arm"] != arm:
            continue
        total += 1
        own = row["isomer"] not in ("", "-")
        cross = row["cross_isomer"] not in ("", "-")
        if own != cross or (own and row["isomer"] != row["cross_isomer"]):
            differing_frames += 1
    differing_runs = sum(
        1 for r in runs
        if r["arm"] == arm and r["frames_inside_gate"] != "" and r["frames_inside_gate_cross"] != ""
        and (int(r["frames_inside_gate"]) > 0) != (int(r["frames_inside_gate_cross"]) > 0)
    )
    return (f"frames scored by both builds: {total}; frames where the two builds differ on the gate decision: "
            f"{differing_frames}; runs where the hit differs between the builds: {differing_runs}")


def frame_stats_section(runs, arm):
    rows = []
    for code in CODES:
        mine = [r for r in runs if r["arm"] == arm and r["code"] == code and r["frames"] != ""]
        if not mine:
            continue
        rows.append([
            code, len(mine),
            f"{median([number(r['frames']) for r in mine]):.0f}",
            f"{median([number(r['frames_intact']) for r in mine]):.0f}",
            f"{median([number(r['frames_ira_failed']) for r in mine]):.0f}",
            f"{median([number(r['min_rmsd_A']) for r in mine]):.3f}",
            f"{quantile([number(r['min_rmsd_A']) for r in mine], 0.1):.3f}",
            f"{quantile([number(r['min_rmsd_A']) for r in mine], 0.9):.3f}",
        ])
    return table(["run", "runs", "median frames", "median intact", "median IRA failed", "median min RMSD",
                  "p10 min RMSD", "p90 min RMSD"], rows)


def old_perseed_section(old_rows, grouped):
    columns = {
        "anneal": "anneal_hit", "polish_fair": "polish_fair_hit",
        "crest": "crest_hit", "crest_fair": "crest_fair_hit",
    }
    rows, pairs = [], []
    for arm in ARMS:
        old = {int(seed): int(row[columns[arm]]) for seed, row in old_rows.items()}
        cells = [arm, f"{sum(old.values())}/{len(old)}"]
        for code in CODES:
            mine = {int(r["seed"]): int(r["hit"]) for r in grouped.get((arm, code), [])}
            cells.append(f"{sum(mine.values())}/{len(mine)}" if mine else "-")
            common = sorted(set(old) & set(mine))
            if not common:
                continue
            both = sum(1 for s in common if old[s] and mine[s])
            only_old = sum(1 for s in common if old[s] and not mine[s])
            only_new = sum(1 for s in common if mine[s] and not old[s])
            neither = len(common) - both - only_old - only_new
            test = stats.binomtest(only_old, only_old + only_new, 0.5).pvalue if only_old + only_new else 1.0
            pairs.append([arm, code, both, only_old, only_new, neither, fmt_p(test)])
        rows.append(cells)
    return (
        table(["arm", "earlier record", "A", "B"], rows),
        table(["arm", "code", "both", "only earlier", "only now", "neither", "McNemar exact p"], pairs),
    )


def frame_counts(frames, arm, code, references, gate):
    counts, seeds = defaultdict(int), set()
    for row in frames:
        if row["arm"] != arm or row["code"] != code:
            continue
        seed = int(row["seed"])
        seeds.add(seed)
        values = [number(row[f"rmsd_{ref}_A"]) for ref in references]
        values = [v for v in values if v is not None]
        if values and min(values) <= gate:
            counts[seed] += 1
    return {seed: counts.get(seed, 0) for seed in seeds}


def count_cells(counts):
    values = list(counts.values())
    if not values:
        return ["-"] * 7
    bins = [sum(1 for v in values if v == k) for k in range(4)] + [sum(1 for v in values if v >= 4)]
    return [f"{sum(1 for v in values if v):d}/{len(values)}", f"{np.mean(values):.2f}", *bins]


def old_receipts_section(directory, frames, gate_new):
    old = {}
    for path in sorted(Path(directory).glob("seed-*.json")):
        record = json.loads(path.read_text())
        old[int(record["seed"])] = int(record["frames_hit"])
    rows = [["earlier record, 180 frames per run, gate 0.784", *count_cells(old)]]
    settings = [
        ("all three references, gate %.3f" % gate_new, REFERENCES, gate_new),
        ("all three references, gate %.3f" % OLD_GATE, REFERENCES, OLD_GATE),
        ("prism and cage, gate %.3f" % OLD_GATE, ("prism", "cage"), OLD_GATE),
        ("prism and cage, gate %.3f" % gate_new, ("prism", "cage"), gate_new),
    ]
    for code in CODES:
        for label, refs, gate in settings:
            rows.append([f"{code}-run frames, {label}", *count_cells(frame_counts(frames, "crest", code, refs, gate))])
    header = ["frames of the CREST arm", "runs with a hit", "mean frames in gate", "0", "1", "2", "3", "4 or more"]
    return table(header, rows)


def report():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--manifest", help="external_manifest.csv that lists the frame table")
    parser.add_argument("--external-dir", help="local directory with the files named in the manifest")
    parser.add_argument("--frames", help="frame table written by run_ab.py, for a session without a manifest")
    parser.add_argument("--reference-energy", type=float,
                        help="energy of the prism minimum at the level of the engine, eV")
    parser.add_argument("--old-perseed", help="per-seed table of the earlier record")
    parser.add_argument("--old-receipts", help="directory with the per-seed CREST receipts of the earlier record")
    args = parser.parse_args()
    runs = load(args.runs)
    try:
        frames = load_frames(args)
    except external.ExternalError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    grouped = by_arm_code(runs)
    seeds = sorted({int(r["seed"]) for r in runs})
    gate_values = {number(r["gate_A"]) for r in runs if number(r["gate_A"]) is not None}
    gate_new = gate_values.pop() if len(gate_values) == 1 else float("nan")
    print(f"# Water hexamer A/B: {len(seeds)} seeds, {len(runs)} runs, gate {gate_new:.6f} A\n")
    print(BUILD_NOTES)
    hits, paired = hits_section(grouped, seeds)
    print("## Hits\n\n" + hits + "\n\nPaired by seed\n\n" + paired + "\n")
    medians, tests = timing_section(grouped)
    print("## Calls and times (runs with status ok)\n\n" + medians + "\n\n" + tests + "\n")
    print("## Slow runs\n\n" + slow_section(grouped) + "\n")
    print("## Final energies\n\n" + energy_section(grouped, args.reference_energy) + "\n")
    print("## Anneal arms: runs whose answer is within a gate of the nearest of the three references\n\n"
          + anneal_gate_section(grouped, gate_new) + "\n")
    print("## Run status\n\n" + failures_section(runs) + "\n")
    print("## Order within the pair\n\n" + order_section(runs) + "\n")
    print("## Wall time per call over the session\n\n" + drift_section(runs) + "\n")
    if args.old_perseed:
        old_rows = {int(r["seed"]): r for r in load(args.old_perseed)}
        old_hits, old_pairs = old_perseed_section(old_rows, grouped)
        print("## Earlier record next to this run\n\n" + old_hits + "\n\nPer seed against the earlier rows\n\n" + old_pairs + "\n")
    minima, best = frame_minima_by_run(frames)
    if args.old_receipts and any(r["arm"] == "crest" for r in runs):
        print("## crest: frames inside the gate per run\n\n"
              + old_receipts_section(args.old_receipts, frames, gate_new) + "\n")
    runs_by_key = {(r["arm"], r["code"], int(r["seed"])): r for r in runs}
    for arm in ("crest", "crest_fair"):
        if (arm, "A") not in grouped:
            continue
        only_a, only_b, text = disagreement_table(grouped, arm, minima)
        print(f"## {arm}: seeds where A and B disagree on the hit\n")
        print(f"A hit, B no hit ({len(only_a)}): {only_a}\n")
        print(f"B hit, A no hit ({len(only_b)}): {only_b}\n")
        print(text + "\n")
        print(f"## {arm}: frames\n\n" + frame_stats_section(runs, arm) + "\n")
        print(f"## {arm}: distance to the nearest reference along the metadynamics runs\n\n"
              + trajectory_section(frames, arm) + "\n")
        print(f"## {arm}: classifier effect\n\n" + classifier_section(frames, runs, arm) + "\n")
        print(f"## {arm}: gate, reference set and hit rule\n\n"
              + sensitivity_section(runs_by_key, minima, best, arm, gate_new) + "\n")
    return 0


def main():
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            status = report()
    except SystemExit:
        sys.stdout.write(buffer.getvalue())
        raise
    if status == 0:
        sys.stdout.write(buffer.getvalue().rstrip("\n") + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
