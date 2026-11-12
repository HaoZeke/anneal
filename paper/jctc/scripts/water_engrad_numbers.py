# Copyright (c) anneal developers
"""Write NUMBERS.tsv for the serial water session.

Per-seed rows use %.17g. Summary rows use .10g.
Quartiles are Hyndman-Fan type 7.
pes_calls is the search ledger plus the polish ledger for anneal, and the
driver count for CREST.
final_best_evaluations is the reported-geometry call written after that sum.
engine_calls adds those two anneal counts. For CREST it is the energy+grad total.
A missing call count stays blank. Summary rows do not quote wall time.
Landed v2.6.3 evaluations and polished energies are compared when
H2H_LANDED_NUMBERS and H2H_LANDED_CREST_DIR are set. A miss is appended to
LOSSES.md.
"""

import math
import os
import re
from pathlib import Path

from crest_engrad import require_split

LOGS = Path(os.environ.get("H2H_NUMBERS_DIR", "paper/jctc/logs/water"))
LANDED = os.environ.get("H2H_LANDED_NUMBERS", "")
LANDED_CREST = os.environ.get("H2H_LANDED_CREST_DIR", "")
def seed_list(root):
    found = []
    for path in root.glob("pair-seed*.log"):
        stem = path.name.removeprefix("pair-seed").removesuffix(".log")
        if stem.isdigit():
            found.append(int(stem))
    if not found:
        raise SystemExit("no pair logs")
    return sorted(found)
ARMS = ("crest", "driver-count budget", "paper4000", "engrad_matched")
PREFIXES = {
    "crest": ("crest",),
    "driver-count budget": ("driver_count", "fair"),
    "paper4000": ("paper4000",),
    "engrad_matched": ("engrad_matched",),
}
HEADER = [
    "seed",
    "arm",
    "pes_calls",
    "final_best_evaluations",
    "engine_calls",
    "wall_s",
    "success",
    "energy_Eh",
    "polished_energy_Eh",
    "verdict",
]
ENERGY_WINDOW = 1e-8


def type7_quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        return float("nan")
    if count == 1:
        return float(ordered[0])
    position = (count - 1) * probability + 1.0
    lower = int(math.floor(position))
    fraction = position - lower
    left = ordered[lower - 1]
    if fraction == 0.0 or lower >= count:
        return float(left)
    right = ordered[lower]
    return float(left + fraction * (right - left))


def type7_iqr(values):
    return type7_quantile(values, 0.75) - type7_quantile(values, 0.25)


def g17(value):
    return format(float(value), ".17g")


def g10(value):
    return format(float(value), ".10g")


def read_fields(path):
    fields = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        fields[key] = value
    return fields


def need(fields, key):
    if key not in fields or fields[key] == "":
        raise SystemExit(f"pair log is missing {key}")
    return fields[key]


def arm_field(fields, arm, name):
    for prefix in PREFIXES[arm]:
        key = f"{prefix}_{name}"
        if key in fields and fields[key] != "":
            return fields[key]
    raise SystemExit(f"pair log is missing {name} for {arm}")


def landed_rows(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == "seed" or parts[0] == "summary":
            continue
        if len(parts) < 13:
            continue
        seed = int(parts[0])
        arm = parts[1]
        rows[(seed, arm)] = {
            "pes_calls": int(float(parts[3])),
            "polished_energy_Eh": float(parts[12]),
            "success": int(float(parts[5])),
        }
    return rows


def main():
    sample = [1, 2, 3, 4, 5, 6]
    if abs(type7_quantile(sample, 0.25) - 2.25) > 1e-12 or abs(type7_iqr(sample) - 2.5) > 1e-12:
        raise SystemExit("type 7 quartile check failed")
    body = []
    losses = []
    seeds = seed_list(LOGS)
    for seed in seeds:
        fields = read_fields(LOGS / f"pair-seed{seed}.log")
        if int(need(fields, "seed")) != seed:
            raise SystemExit(f"pair log {seed} names another seed")
        for arm in ARMS:
            raw_pes = arm_field(fields, arm, "calls")
            raw_wall = arm_field(fields, arm, "wall")
            success = int(arm_field(fields, arm, "success"))
            energy = arm_field(fields, arm, "energy")
            polished = arm_field(fields, arm, "polished_energy")
            final_best = arm_field(fields, arm, "final_best_evaluations")
            engine = arm_field(fields, arm, "engine_calls")
            pes = "nan" if raw_pes == "nan" else str(int(float(raw_pes)))
            if arm != "crest" and final_best not in ("0", "1", "nan"):
                raise SystemExit(f"seed {seed} {arm} final best is not 0 or 1")
            if (
                arm != "crest"
                and engine != "nan"
                and pes != "nan"
                and final_best in ("0", "1")
                and int(float(engine)) != int(pes) + int(final_best)
            ):
                raise SystemExit(f"seed {seed} {arm} engine_calls broke the split")
            body.append(
                [
                    str(seed),
                    arm,
                    pes,
                    final_best,
                    engine if engine != "nan" else "nan",
                    "nan" if raw_wall == "nan" else g17(raw_wall),
                    str(success),
                    g17(energy) if energy != "nan" else "nan",
                    g17(polished) if polished != "nan" else "nan",
                    arm_field(fields, arm, "verdict"),
                ]
            )
    lines = [
        f"# n={len(seeds)} serial one host one thread",
        "# quartile=Hyndman-Fan type 7",
        "# pes_calls=anneal search ledger + polish ledger, or CREST dynamics_steps+opt_steps",
        "# final_best_evaluations=anneal reported-geometry value_and_gradient after energy_calls",
        "# engine_calls=pes_calls + final_best_evaluations for anneal; CREST energy+grad total for crest",
        "# wall_s per run: " + os.environ.get("H2H_WALL_LABEL", "not comparable across sessions"),
        "# engrad_budget rows list this session's CREST energy+grad total, which is the engrad_matched budget, and the landed v2.6.3 total",
        "# " + "\t".join(HEADER),
    ]
    for row in body:
        lines.append("\t".join(row))
    lines.append(
        "# summary arm n pes_calls_median pes_calls_iqr engine_calls_median engine_calls_iqr "
        "success_median success_iqr success_rate passes"
    )
    by_arm = {}
    for row in body:
        by_arm.setdefault(row[1], []).append(row)
    for arm in ARMS:
        group = by_arm[arm]
        calls = [float(item[2]) for item in group if item[2] != "nan"]
        engines = [float(item[4]) for item in group if item[4] != "nan"]
        success = [float(item[6]) for item in group]
        lines.append(
            "summary\t{arm}\t{n}\t{cm}\t{ci}\t{em}\t{ei}\t{sm}\t{si}\t{rate}\t{passes}".format(
                arm=arm,
                n=len(group),
                cm=g10(type7_quantile(calls, 0.5)) if calls else "nan",
                ci=g10(type7_iqr(calls)) if calls else "nan",
                em=g10(type7_quantile(engines, 0.5)) if engines else "nan",
                ei=g10(type7_iqr(engines)) if engines else "nan",
                sm=g10(type7_quantile(success, 0.5)) if success else "nan",
                si=g10(type7_iqr(success)) if success else "nan",
                rate=g10(sum(success) / len(success)) if success else "nan",
                passes=int(sum(success)),
            )
        )
    commits = set()
    for seed in seeds:
        for name in (
            f"anneal-seed{seed}/run.txt",
            f"anneal-paper4000-seed{seed}/run.txt",
            f"anneal-engrad-seed{seed}/run.txt",
        ):
            path = LOGS / name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"(?m)^source_commit[:=]\s*(\S+)\s*$", text)
            if not match or match.group(1) in {"unset", "nan"}:
                raise SystemExit(f"{name} has no source_commit")
            commits.add(match.group(1))
    if len(commits) != 1:
        raise SystemExit(f"source_commit differs across logs: {sorted(commits)}")
    lines.insert(1, "# source_commit=" + next(iter(commits)))
    engrad_rows = []
    if LANDED_CREST:
        root = Path(LANDED_CREST)
        lines.append(
            "# engrad_budget seed session_engrad landed_engrad session_driver landed_driver"
        )
        for seed in seeds:
            fresh = (LOGS / f"crest-hexamer-seed{seed}.log").read_text(
                encoding="utf-8", errors="replace"
            )
            old_text = (root / f"crest-hexamer-seed{seed}.log").read_text(
                encoding="utf-8", errors="replace"
            )
            fresh_driver, fresh_total = require_split(fresh)
            old_driver, old_total = require_split(old_text)
            engrad_rows.append((seed, fresh_driver, fresh_total, old_driver, old_total))
            lines.append(
                f"engrad_budget\t{seed}\t{fresh_total}\t{old_total}\t{fresh_driver}\t{old_driver}"
            )
    out = LOGS / "NUMBERS.tsv"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)

    if not LANDED:
        return
    previous = landed_rows(LANDED)
    for row in body:
        seed = int(row[0])
        arm = row[1]
        landed_name = {"crest": "crest", "driver-count budget": "fair", "paper4000": "paper4000"}.get(arm)
        if landed_name is None or (seed, landed_name) not in previous:
            continue
        old = previous[(seed, landed_name)]
        pes = int(row[2])
        if pes != old["pes_calls"]:
            losses.append(f"Seed {seed} {arm} ledger count is {pes} here.")
            losses.append(f"The older draw is {old['pes_calls']}.")
        polished = row[8]
        if polished == "nan":
            losses.append(f"Seed {seed} {arm} has no polished energy.")
        elif abs(float(polished) - old["polished_energy_Eh"]) > ENERGY_WINDOW:
            losses.append(f"Seed {seed} {arm} polished energy differs.")
            losses.append(f"Here it is {polished}.")
            losses.append(f"The older draw is {g17(old['polished_energy_Eh'])}.")
        if int(row[6]) != old["success"]:
            losses.append(f"Seed {seed} {arm} success is {row[6]} here.")
            losses.append(f"The older draw is {old['success']}.")
    for seed, fresh_driver, fresh_total, old_driver, old_total in engrad_rows:
        if fresh_total != old_total:
            losses.append(f"Seed {seed} energy and gradient total is {fresh_total} here.")
            losses.append(f"The older draw is {old_total}.")
        if fresh_driver != old_driver:
            losses.append(f"Seed {seed} driver count is {fresh_driver} here.")
            losses.append(f"The older draw is {old_driver}.")
    loss_path = LOGS / "LOSSES.md"
    if losses:
        loss_lines = [
            "# Losses",
            "",
            "Each older row is one draw.",
            "The program was crest start.xyz -T 1.",
            "That command has no metadynamics seed.",
            "The anneal seed fixes only the start geometry.",
            "The matched arm uses this session total.",
            "",
        ]
        loss_lines.extend(f"- {item}" for item in losses)
        loss_path.write_text("\n".join(loss_lines) + "\n", encoding="utf-8")
    else:
        loss_path.write_text(
            "# Losses\n\n"
            "The landed v2.6.3 evaluation counts and polished energies reproduced "
            "No listed difference remains for the shared arms.\n",
            encoding="utf-8",
        )
    print(loss_path)


if __name__ == "__main__":
    main()
