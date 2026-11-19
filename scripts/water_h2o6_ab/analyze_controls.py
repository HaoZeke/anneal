#!/usr/bin/env python3
"""Tables for the controls of the CREST arm.

Reads controls_runs.csv and controls_frames.csv from one or more directories
written by controls.py and prints, as markdown, for every start geometry,
frame source and reference variant: how many references were loaded, the gate,
how many runs have a frame inside the gate, how many frames are inside the
gate, how often the two builds disagree on a frame, and how often the other
reference orderings disagree with the ordering of the A/B on a frame.

The frame tables are kept outside git. With --manifest and --external-dir the
table of a directory that the manifest lists is read from the local directory
after its sha256 is checked; a directory that the manifest does not list is
read in place if it has a table.
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import external


def load(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def frames_of(directory, manifest, manifest_path, external_dir):
    if manifest is not None:
        try:
            relative = Path(directory).resolve().relative_to(Path(manifest_path).resolve().parent)
        except ValueError:
            relative = None
        name = f"{relative.as_posix()}/controls_frames.csv" if relative is not None else None
        if name in manifest:
            if not external_dir:
                raise external.ExternalError("--manifest needs --external-dir")
            return external.load_rows(manifest, name, external_dir)
    path = Path(directory) / "controls_frames.csv"
    return load(path) if path.exists() else []


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def runs_section(runs):
    groups = defaultdict(list)
    for row in runs:
        groups[(row["start"], row["source"], row["refs"], row["code"])].append(row)
    rows = []
    for key in sorted(groups):
        mine = groups[key]
        scored = [r for r in mine if r["hit"] != "" and int(r["frames"]) > 0]
        failed = len(mine) - len(scored)
        loaded = sorted({r["references_loaded"] for r in scored})
        gates = sorted({r["gate_A"] if r["gate_A"] != "" else "none" for r in scored})
        hits = sum(int(r["hit"]) for r in scored)
        inside = [number(r["frames_inside_gate"]) for r in scored]
        frames = sum(int(r["frames"]) for r in scored)
        unknown = sum(int(r["frames_unknown"]) for r in scored)
        one = [int(r["frames_one_cluster"]) / int(r["frames"]) for r in scored
               if r.get("frames_one_cluster") not in (None, "") and int(r["frames"])]
        first = sorted({r["first_frame_components"] for r in scored if r.get("first_frame_components") not in (None, "")})
        rows.append([
            *key, len(scored), failed, "/".join(loaded), "/".join(gates), f"{hits}/{len(scored)}",
            f"{median(inside):.0f}", f"{median([number(r['min_rmsd_A']) for r in scored]):.3f}",
            f"{unknown / frames:.2f}" if frames else "-",
            f"{median(one):.2f}" if one else "-", "/".join(first) or "-",
            f"{median([number(r['crest_wall_s']) for r in scored]):.1f}",
        ])
    header = ["start", "source", "references", "code", "runs", "failed", "references loaded", "gate A", "runs with a hit",
              "median frames in gate", "median min RMSD A", "fraction of frames unknown",
              "median fraction of frames with one oxygen cluster", "oxygen clusters in the first frame",
              "median CREST wall s"]
    return table(header, rows)


def agreement_section(frames):
    groups = defaultdict(dict)
    for row in frames:
        groups[(row["start"], row["source"], row["refs"], row["run"])].setdefault(row["code"], {})[row["frame"]] = row
    summary = defaultdict(lambda: [0, 0, 0, 0.0])
    for (start, source, refs, _), by_code in groups.items():
        if "A" not in by_code or "B" not in by_code:
            continue
        cell = summary[(start, source, refs)]
        for index, a in by_code["A"].items():
            b = by_code["B"].get(index)
            if b is None:
                continue
            cell[0] += 1
            cell[1] += int(a["isomer"] != b["isomer"])
            ra, rb = number(a["rmsd_A"]), number(b["rmsd_A"])
            if (ra is None) != (rb is None):
                cell[2] += 1
            elif ra is not None:
                cell[3] = max(cell[3], abs(ra - rb))
    rows = [[*key, cell[0], cell[1], cell[2], f"{cell[3]:.1e}"] for key, cell in sorted(summary.items())]
    return table(["start", "source", "references", "frames compared", "isomer differs", "RMSD present in one build only",
                  "largest RMSD difference A"], rows)


GATES = (0.598496, 0.700, 0.784112, 0.900, 1.000)
OLD_GATE = 0.784112


def gate_section(frames):
    groups = defaultdict(lambda: defaultdict(list))
    for row in frames:
        if row["code"] != "A" or row["refs"] != "used":
            continue
        groups[(row["start"], row["source"])][int(row["run"])].append(number(row["rmsd_A"]))
    rows = []
    for (start, source), by_run in sorted(groups.items()):
        cells = []
        for gate in GATES:
            hits = sum(1 for values in by_run.values() if any(v is not None and v <= gate for v in values))
            cells.append(f"{hits}/{len(by_run)}")
        counts = [sum(1 for v in values if v is not None and v <= OLD_GATE) for values in by_run.values()]
        histogram = [sum(1 for c in counts if c == k) for k in range(4)] + [sum(1 for c in counts if c >= 4)]
        rows.append([start, source, len(by_run), *cells, f"{statistics.mean(counts):.2f}", *histogram])
    header = ["start", "source", "runs", *[f"runs with a frame within {g:.3f} A" for g in GATES],
              "mean frames within 0.784 A", "runs with 0 such frames", "1", "2", "3", "4 or more"]
    return table(header, rows)


def ordering_section(frames):
    groups = defaultdict(dict)
    for row in frames:
        groups[(row["start"], row["source"], row["run"], row["code"], row["frame"])][row["refs"]] = row
    summary = defaultdict(lambda: [0, 0, 0, 0.0])
    for (start, source, _, code, _), by_refs in groups.items():
        base = by_refs.get("used")
        if base is None:
            continue
        for label, row in by_refs.items():
            if label == "used":
                continue
            cell = summary[(start, source, code, label)]
            cell[0] += 1
            cell[1] += int(row["isomer"] != base["isomer"])
            ra, rb = number(row["rmsd_A"]), number(base["rmsd_A"])
            if (ra is None) != (rb is None):
                cell[2] += 1
            elif ra is not None:
                cell[3] = max(cell[3], abs(ra - rb))
    rows = [[*key, cell[0], cell[1], cell[2], f"{cell[3]:.1e}"] for key, cell in sorted(summary.items())]
    return table(["start", "source", "code", "references", "frames compared", "isomer differs",
                  "RMSD present for one ordering only", "largest RMSD difference A"], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--manifest", help="external_manifest.csv that lists the frame tables")
    parser.add_argument("--external-dir", help="local directory with the files named in the manifest")
    args = parser.parse_args()
    manifest = external.read_manifest(args.manifest) if args.manifest else None
    runs, frames = [], []
    try:
        for directory in args.directories:
            runs += load(Path(directory) / "controls_runs.csv")
            frames += frames_of(directory, manifest, args.manifest, args.external_dir)
    except external.ExternalError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("## Control runs\n\n" + runs_section(runs) + "\n")
    if frames:
        print("## Build A against build B on the same frames\n\n" + agreement_section(frames) + "\n")
        print("## Other reference orderings against the ordering used in the A/B, on the same frames\n\n"
              + ordering_section(frames) + "\n")
        print("## Runs with a frame within other gates of a reference (build A, references as used)\n\n"
              + gate_section(frames) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
