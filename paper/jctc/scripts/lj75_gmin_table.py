#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Tally LJ75 against GMIN with core-orbit symmetrisation, one run per CSV row.

Reads rewrite_2026/data/lj75_gmin/: the recommended and derived hops of anneal
on seeds 2000 to 2199, and GMIN on the OptBench starts 1 to 200, each row one
run to the first minimum within the tolerance of the Cambridge reference or to
the cap of force calls. provenance.txt holds the digest of each CSV, the
reference, the tolerance, the cap, the host and the GMIN build, the anneal commit the
audit ran and the published rewrite of it whose tree is that commit's, and the digest of
the sealed sums of the audit that reran every run; audit/ copies the verdict of that
audit and the summary and provenance of each of its LJ75 boards. The call PAR-2 of each
arm is also held to the sealed lj_cell and gmin_lj75 rows of the results_jcc campaigns,
read through results_jcc.load, and prints as RESULTS.md prints it.

Prints the table tbl:lj75 of the main text, or with --supplement the per-run
table tbl:lj75-seeds of the Supporting Information, or with --values every
number the draft may quote from these runs, one a line.

Hits carry the Wilson score interval at z = 1.959963984540054. Call PAR-2 is
the mean over the runs of the force calls to the first hit, twice the cap for
a run that misses; the recount must equal the sealed row, whose string is printed.
The wall time is the median of the wall_seconds column of each arm's committed CSV
(the campaign run the audit compared seed for seed), on a host that is not the
paper machine; the audit's own median walls are checked but not printed.

The script stops when a CSV and provenance.txt disagree: a digest, a seed or
start out of sequence, a cap, reference or tolerance, a hit flag against the
energy it prints, a count to the hit against the count of the run, or a
configuration digest that changes within an arm, or a published snapshot that is no
full commit held to the tree of the commit the audit ran. It stops too when the audit
disagrees with them: a file its sums do not record, a verdict that is not READY for
LJ75, a board whose hits, interval, call PAR-2, seeds, conditions, anneal commit or
configuration are not those the CSVs recount, a board that found a run of the CSVs
it does not reproduce, or one on the paper machine, or when a sealed results_jcc row gives
other hits or another call PAR-2.

Usage: lj75_gmin_table.py [--supplement | --values] [data-dir]
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import results_jcc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lj75_gmin"
TABLE = "tbl:lj75"
SEEDS = "tbl:lj75-seeds"
Z = 1.959963984540054
# A printed energy this close to the bar may sit on either side of it once rounded.
PRINTED = 1e-6
# The audit writes its means and interval bounds as floats of its own summation.
AUDIT_TOLERANCE = 1e-9
# The file, the label it prints, the first seed or start, and the audit board of each arm.
ARMS = (
    ("lj75-recommended-seeds-2000-2199.csv", "anneal, recommended hop", "seed", 2000, "lj75-recommended"),
    ("lj75-derived-seeds-2000-2199.csv", "anneal, derived hop", "seed", 2000, "lj75-derived"),
    ("gmin-lj75-sym-starts-1-200.csv", "GMIN, core orbits", "index", 1, "gmin-lj75"),
)
RUNS = 200
AUDIT = "audit"
# The files of each board that audit/ copies, and those of the audit as a whole.
BOARD_FILES = ("SHA256SUMS", "PROVENANCE", "summary.json")
AUDIT_FILES = ("PROVENANCE", "verdict.json")
NOT_PAPER_MACHINE = "not paper machine"


def die(message: str) -> NoReturn:
    raise SystemExit(f"lj75_gmin_table: {message}")


def provenance(data: Path) -> tuple[dict[str, str], dict[str, str]]:
    """The keyed lines of provenance.txt, and the digest it records for each CSV."""
    keys: dict[str, str] = {}
    digests: dict[str, str] = {}
    for line in (data / "provenance.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        first, _, rest = line.partition(" ")
        if len(first) == 64 and all(c in "0123456789abcdef" for c in first):
            digests[rest.strip()] = first
        else:
            keys[first] = rest.strip()
    return keys, digests


@dataclass(frozen=True)
class Run:
    start: int
    hit: bool
    calls: int
    wall: float


@dataclass(frozen=True)
class Arm:
    label: str
    runs: tuple[Run, ...]
    budget: int
    audit_wall: float = math.nan
    sealed_par2: str = ""

    @property
    def hits(self) -> int:
        return sum(run.hit for run in self.runs)

    @property
    def wilson(self) -> tuple[float, float]:
        n = len(self.runs)
        p = self.hits / n
        scale = 1 + Z * Z / n
        centre = (p + Z * Z / (2 * n)) / scale
        margin = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / scale
        return max(0.0, centre - margin), min(1.0, centre + margin)

    @property
    def par2(self) -> float:
        return statistics.mean(run.calls if run.hit else 2 * self.budget for run in self.runs)

    @property
    def wall(self) -> float:
        """The median wall of the runs of the CSV, which the audit's rerun does not repeat."""
        return statistics.median(run.wall for run in self.runs)


def _sums(path: Path) -> dict[str, str]:
    """The digest that a SHA256SUMS file records for each file, by its path below it."""
    found = {}
    for line in path.read_text().splitlines():
        digest, _, name = line.partition("  ")
        found[name.removeprefix("./")] = digest
    return found


def _keyed(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=AUDIT_TOLERANCE)


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(data: Path, keys: dict[str, str], digests: dict[str, str], arms: list[Arm]) -> list[Arm]:
    """The arms with the median walls of the audit's rerun, once its boards recount them.

    Every file of audit/ is one the sealed sums of the audit record, and those sums are the
    ones provenance.txt records. The verdict holds LJ75 READY, and each board ran these
    seeds under these conditions from the anneal commit of provenance.txt, gives the hits,
    interval and call PAR-2 that the CSVs recount, compared itself with those CSVs and
    found every run, and ran off the paper machine.
    """
    root = data / AUDIT
    name = f"{AUDIT}/SHA256SUMS"
    if digest_of(root / "SHA256SUMS") != digests.get(name):
        die(f"{name} does not match the digest provenance.txt records")
    sealed = _sums(root / "SHA256SUMS")
    for file in AUDIT_FILES:
        if digest_of(root / file) != sealed.get(file):
            die(f"{AUDIT}/{file} is not the file the audit sealed")
    top = _keyed(root / "PROVENANCE")
    verdict = json.loads((root / "verdict.json").read_text())
    if verdict["lj75"] != "READY" or top["lj75"] != "READY" or top["anneal_commit"] != keys["anneal_commit"]:
        die("the audit does not hold LJ75 READY at the anneal commit of provenance.txt")
    boards = {board["name"]: board for board in verdict["boards"]}
    budget = int(keys["budget"])
    reference = float(keys["reference"].split()[0])
    tolerance = float(keys["tolerance"])
    hosts = set()
    audited = []
    for arm, (csv_name, _, _, first, board) in zip(arms, ARMS, strict=True):
        where = f"{AUDIT}/{board}"
        inner = _sums(root / board / "SHA256SUMS")
        if digest_of(root / board / "SHA256SUMS") != sealed.get(f"{board}/SHA256SUMS"):
            die(f"{where}/SHA256SUMS is not the file the audit sealed")
        for file in BOARD_FILES[1:]:
            if digest_of(root / board / file) != inner.get(file) or inner[file] != sealed.get(f"{board}/{file}"):
                die(f"{where}/{file} is not the file the audit sealed")
        summary = json.loads((root / board / "summary.json").read_text())
        record = _keyed(root / board / "PROVENANCE")
        saddle = summary["saddle"]
        low, high = arm.wilson
        checks = {
            "the board": summary["name"] == record["name"] == board,
            "the cluster": summary["n_atoms"] == 75,
            "the cap": summary["budget"] == budget == int(record["budget"]),
            "the reference": summary["reference"] == reference == float(record["reference"]),
            "the tolerance": summary["tolerance"] == tolerance
            and record["hit_rule"] == f"best<=reference+{keys['tolerance']}",
            "the seeds": (summary["first"], summary["last"], summary["n_seeds"]) == (first, first + RUNS - 1, RUNS),
            "the anneal commit": summary["anneal_commit"] == record["anneal_commit"] == keys["anneal_commit"],
            "the hits": summary["hits"] == arm.hits == saddle["saddle_hits"],
            "the interval": _close(summary["wilson_95_low"], low) and _close(summary["wilson_95_high"], high),
            "the call PAR-2": _close(summary["call_par2"], arm.par2) and _close(saddle["saddle_call_par2"], arm.par2),
            "the runs it compared": saddle["compared"] is True
            and saddle["saddle_rows"] == RUNS
            and saddle["scientific_mismatches"] == []
            and saddle["budget_differences"] == []
            and _close(saddle["saddle_median_wall_seconds"], arm.wall),
            "the machine": summary["machine_class"] == record["machine_class"] == NOT_PAPER_MACHINE
            and summary["cpu_model"] != summary["paper_cpu_model"],
            "one thread": record["threads_per_process"] == "1",
            "the verdict": {k: boards[board][k] for k in ("hits", "call_par2", "median_wall_seconds")}
            == {k: summary[k] for k in ("hits", "call_par2", "median_wall_seconds")},
        }
        if summary["kind"] == "anneal":
            rows = list(csv.DictReader((data / csv_name).read_text().splitlines()))
            checks["the configuration"] = summary["config_digests"] == [rows[0]["config_digest"]]
        else:
            checks["the GMIN build"] = record["gfortran"].endswith(" " + keys["gmin_build"].split()[2])
        failed = [what for what, held in checks.items() if not held]
        if failed:
            die(f"{where} does not give {failed[0]} of {csv_name}")
        hosts.add(f"Intel Xeon family {summary['cpu_family']} model {summary['cpu_model']}, {summary['hypervisor']}")
        audited.append(dataclasses.replace(arm, audit_wall=summary["median_wall_seconds"]))
    if len(hosts) != 1:
        die(f"the audit's boards ran on more than one host: {sorted(hosts)}")
    keys["audit_host"] = hosts.pop()
    return audited


def load(data: Path = DATA) -> tuple[dict[str, str], tuple[Arm, ...]]:
    keys, digests = provenance(data)
    published = keys.get("anneal_published_commit", "")
    if len(published) != 40 or set(published) - set("0123456789abcdef"):
        die("provenance.txt names no full commit of the published anneal snapshot")
    if keys.get("anneal_published_tree") != "identical to anneal_commit":
        die("provenance.txt does not hold the published snapshot to the tree of anneal_commit")
    budget = int(keys["budget"])
    reference = float(keys["reference"].split()[0])
    tolerance = float(keys["tolerance"])
    arms = []
    for name, label, column, first, _ in ARMS:
        raw = (data / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digests.get(name):
            die(f"{name} does not match the digest provenance.txt records")
        rows = list(csv.DictReader(raw.decode().splitlines()))
        if [int(row[column]) for row in rows] != list(range(first, first + RUNS)):
            die(f"{name} does not hold the {RUNS} runs from {column} {first} in order")
        if len({row.get("config_digest") for row in rows}) != 1:
            die(f"{name} changes its configuration digest between runs")
        runs = []
        for row in rows:
            where = f"{name} {column} {row[column]}"
            if int(row["budget"]) != budget or int(row["n"]) != 75:
                die(f"{where} is not LJ75 at the cap of provenance.txt")
            if float(row["reference"]) != reference or float(row["tolerance"]) != tolerance:
                die(f"{where} scores against another reference or tolerance")
            if row.get("return_code", "0") != "0":
                die(f"{where} exited with {row['return_code']}")
            best = float(row["best"])
            hit = row["hit"] == "1"
            if abs(best - (reference + tolerance)) > PRINTED and hit != (best <= reference + tolerance):
                die(f"{where} flags hit={row['hit']} at {best}")
            calls = int(row["force_count"])
            if hit and int(row["force_to_hit"]) != calls:
                die(f"{where} counts {row['force_to_hit']} calls to the hit and {calls} in all")
            if not hit and row["force_to_hit"]:
                die(f"{where} misses and still counts calls to a hit")
            if calls > budget:
                die(f"{where} spends {calls} calls over the cap")
            runs.append(Run(int(row[column]), hit, calls, float(row["wall_seconds"])))
        arms.append(Arm(label, tuple(runs), budget))
    return keys, tuple(_sealed(_audit(data, keys, digests, arms)))


# The sealed results_jcc row of each arm: its folder, and its lj_cell arm or none for GMIN.
SEALED_ROWS = {"anneal, recommended hop": ("lj_cell", "recommended"), "anneal, derived hop": ("lj_cell", "derived"),
               "GMIN, core orbits": ("gmin_lj75", None)}


def _sealed(arms: list[Arm], sealed: Path = results_jcc.DATA) -> list[Arm]:
    """The arms with the call PAR-2 of their sealed row, once that row gives their hits and recount."""
    keys, folders = results_jcc.load(sealed)
    boards = {board["arm"]: board for board in folders["lj_cell"].summary["boards"] if board["system"] == "lj75"}
    found = []
    for arm in arms:
        folder, name = SEALED_ROWS[arm.label]
        row = boards[name] if name else folders[folder].summary
        printed = keys[f"results_md_par2_lj75_{name or 'gmin'}"]
        if (row["hits"], row["n"], row["budget"]) != (arm.hits, len(arm.runs), arm.budget):
            die(f"the sealed {folder} row of {arm.label} gives other hits, seeds or cap")
        if not (Decimal(printed) == Decimal(repr(row["par2"])) and _close(row["par2"], arm.par2)):
            die(f"the sealed {folder} row of {arm.label} gives another call PAR-2")
        found.append(dataclasses.replace(arm, sealed_par2=printed))
    return found


def fixed(value: float, places: int) -> str:
    return str(Decimal(repr(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def cells(arm: Arm) -> dict[str, str]:
    low, high = arm.wilson
    return {
        "hits": f"{arm.hits}/{len(arm.runs)}",
        "interval": f"{fixed(low, 3)}--{fixed(high, 3)}",
        "par2": arm.sealed_par2,
        "wall": fixed(arm.wall, 6),
    }


def shares(arms: tuple[Arm, ...]) -> dict[str, str]:
    """The call PAR-2 of each hop as a share of that of GMIN."""
    recommended, derived, gmin = arms
    return {"recommended": fixed(recommended.par2 / gmin.par2, 3), "derived": fixed(derived.par2 / gmin.par2, 3)}


def cap(keys: dict[str, str]) -> str:
    budget = int(keys["budget"])
    exponent = int(math.log10(budget))
    if budget != int(str(budget)[0]) * 10**exponent:
        die(f"the cap {budget} is no single digit times a power of ten")
    return rf"\({str(budget)[0]}\times10^{{{exponent}}}\)"


def values(keys: dict[str, str], arms: tuple[Arm, ...]) -> list[str]:
    """Every number of these runs, their conditions and the GMIN build the draft may print."""
    found = [value for arm in arms for value in cells(arm).values()]
    found += list(shares(arms).values())
    found += [
        cap(keys),
        rf"\({keys['tolerance']}\)",
        str(RUNS),
        str(ARMS[0][3]),
        str(ARMS[0][3] + RUNS - 1),
        str(ARMS[2][3]),
        str(ARMS[2][3] + RUNS - 1),
    ]
    found += [keys["audit_host"].split(", ")[0].removeprefix("Intel Xeon ")]
    found += [keys["anneal_commit"][:8], keys["anneal_published_commit"][:8]]
    source = keys["gmin_source"].split()
    found += [source[1].rstrip(",").removesuffix(".tar.bz2"), source[-1]]
    found += [keys["gmin_build"].split()[2]]
    literature = keys["literature"]
    found += [literature.split("mean ")[1].split()[0], literature.split("from ")[1].split()[0]]
    return list(dict.fromkeys(found))


def main_table(keys: dict[str, str], arms: tuple[Arm, ...]) -> str:
    first, last = ARMS[0][3], ARMS[0][3] + RUNS - 1
    tolerance = rf"\({keys['tolerance']}\)"
    host = keys["audit_host"].split(", ")
    caption = (
        r"LJ\(_{75}\) against GMIN with core-orbit symmetrisation~"
        r"\cite{oakleySymmetrisationSchemesGlobal2013}. Every run stops at the first minimum "
        rf"within {tolerance} of the Cambridge reference~"
        rf"\cite{{walesCambridgeClusterDatabase}} or at a cap of {cap(keys)} force calls, each code "
        rf"counting its own. anneal ran seeds {first} to {last} from random "
        rf"starts, and GMIN the minimised OptBench starts {ARMS[2][3]} to "
        rf"{ARMS[2][3] + RUNS - 1}. The interval is the Wilson score interval "
        r"\eqref{eq:wilson}. Call PAR-2 is \eqref{eq:par2} counted in force calls, a miss "
        r"costing twice the cap, and the wall time is the median of the wall\_seconds column "
        r"of each arm's committed CSV of that campaign, on an "
        rf"{host[0]} host under {host[1]}, not the paper machine of Section~\ref{{sec:hardware}}."
    )
    rows = []
    for arm in arms:
        c = cells(arm)
        rows.append(f"{arm.label} & {c['hits']} & {c['interval']} & {c['par2']} & {c['wall']}\\\\")
    column = r">{\raggedright\arraybackslash}p{2.9cm}"
    return "\n".join(
        [
            "#+BEGIN_EXPORT latex",
            r"\begin{table}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{{TABLE}}}",
            rf"\begin{{tabular}}{{@{{}}>{{\raggedright\arraybackslash}}p{{2.8cm}}{column * 4}@{{}}}}",
            r"\toprule",
            r"Code, setting & Hits & 95\% interval & Call PAR-2 & Median wall (s), not paper machine\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "#+END_EXPORT",
        ]
    )


def seed_table(keys: dict[str, str], arms: tuple[Arm, ...]) -> str:
    """Every run, one row per position: the force calls of each code to its first hit."""
    recommended, derived, gmin = arms

    def head(text: str, align: str = "r") -> str:
        return rf"\multicolumn{{1}}{{{align}}}{{{text}}}"

    def spent(run: Run) -> str:
        return str(run.calls) if run.hit else "miss"

    header = [
        " & ".join(
            [head("Seed"), head("Recommended hop"), head("Derived hop"), head("Start"), head("GMIN, core orbits")]
        )
        + r"\\",
    ]
    caption = (
        r"Every LJ\(_{75}\) run of the comparison with GMIN in the main text: the force calls to "
        r"the first hit, or a miss at the cap. A row pairs an anneal seed with a GMIN start by "
        r"position only."
    )
    rows = []
    for rec, der, opt in zip(recommended.runs, derived.runs, gmin.runs, strict=True):
        if rec.start != der.start:
            die("the recommended and derived hops ran different seeds")
        rows.append(" & ".join([str(rec.start), spent(rec), spent(der), str(opt.start), spent(opt)]) + r"\\")
    return "\n".join(
        [
            "#+BEGIN_EXPORT latex",
            r"\begingroup",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{6pt}",
            r"\begin{longtable}{@{}rrrrr@{}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{SEEDS}}}\\",
            r"\toprule",
            *header,
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            *header,
            r"\midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endlastfoot",
            *rows,
            r"\end{longtable}",
            r"\endgroup",
            "#+END_EXPORT",
        ]
    )


def main(argv: list[str]) -> None:
    flags = [arg for arg in argv if arg.startswith("--")]
    rest = [arg for arg in argv if not arg.startswith("--")]
    keys, arms = load(Path(rest[0]) if rest else DATA)
    if flags == ["--supplement"]:
        print(seed_table(keys, arms))
    elif flags == ["--values"]:
        print("\n".join(values(keys, arms)))
    elif not flags:
        print(main_table(keys, arms))
    else:
        die(f"unknown options {flags}")


if __name__ == "__main__":
    main(sys.argv[1:])
