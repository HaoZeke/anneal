#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Check that the retired GOFEE comparison tables remain absent.

The manuscript and Supporting Information no longer print these tables.
Their current comparison tables have dedicated generators.

    python3 paper/jctc/scripts/h2h_tables.py --check

The module also holds the readers that scripts/paper_audit.py and
scripts/ingest_record.py call to recompute the Sutton-Chen, tuned-GMIN and
CREST summaries from the committed rows. They write no table.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import revision_reach  # noqa: E402

MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
SI = ROOT / "paper" / "jctc" / "si.tex"
SC = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12" / "results.csv"
SC_P = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12" / "pvalues.csv"
GMIN_TUNED = ROOT / "bench" / "gmin-tuned" / "rows-shard2.csv"
CREST_EQUAL = ROOT / "bench" / "crest-equal"
CREST_ARMS = ("anneal", "crest", "crest-wall")
# The call tolerance of the panel: run_panel.py counts a CREST run as at the hop arm's calls
# when it lies within 5 percent of them.
CREST_CALL_TOLERANCE = 0.05
SC_ORDER = ("sc126-38", "sc108-38", "sc96-38", "sc126-55", "sc108-55", "sc96-55", "sc126-80", "sc108-80", "sc96-80")
SC_FORM = {"126": "12-6", "108": "10-8", "96": "9-6"}

# Pins of other projects. They are not commits of this repository.
EXTERNAL_PINS = tuple(revision_reach.EXTERNAL_REVISIONS)

RETIRED = {
    "gofee": (
        MANUSCRIPT,
        ("% gofee table: start", "% gofee table: end", "\\label{tab:gofee}"),
    ),
    "gofee-si": (
        SI,
        ("% gofee-si table: start", "% gofee-si table: end", "\\label{tab:gofee-si}"),
    ),
}


def commit_held(revision: str) -> bool:
    """The revision is an ancestor of HEAD, as the audit and the ingest of a record count it."""
    return revision_reach.commit_held(revision, ROOT)


def revision_reachable(revision: str) -> bool:
    """A full SHA that is an ancestor of HEAD, an external pin, or a repository URL at a full SHA."""
    return revision_reach.revision_reachable(revision, ROOT)


def type7_median(values: list[float]) -> float:
    ordered = sorted(values)
    h = 0.5 * (len(ordered) - 1)
    lo = int(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def show_calls_median(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fisher_p(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p, the same sum the audit uses."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def sc_label(system: str) -> str:
    form, size = system.removeprefix("sc").split("-")
    return f"{SC_FORM[form]}, {size}"


def sc_pairs() -> list[tuple[dict[str, str], dict[str, str]]]:
    """The anneal row of each system and the GMIN row of the corrected data file.

    The corrected file is the GMIN harness that differs from the anneal one;
    the anneal harness's own GMIN rows ran with Lennard-Jones step and
    temperature and are not shown."""
    rows = read(SC)
    pairs = []
    for system in SC_ORDER:
        anneal_rows = [r for r in rows if r["system"] == system and r["arm"] == "anneal"]
        anneal = anneal_rows[-1]
        anneal_harnesses = {r["harness_revision"] for r in anneal_rows}
        gmin = [r for r in rows if r["system"] == system and r["arm"] == "gmin"
                and r["harness_revision"] not in anneal_harnesses]
        if not gmin:
            raise SystemExit(f"{system}: expected a corrected GMIN row")
        pairs.append((anneal, gmin[-1]))
    return pairs


def sc_tests() -> dict[str, dict[str, str]]:
    """The p-values of each system against its corrected GMIN rows, from pvalues.csv."""
    tests: dict[str, dict[str, str]] = {}
    for anneal, gmin in sc_pairs():
        rows = [r for r in read(SC_P) if r["system"] == anneal["system"]
                and r["code_revision"] == anneal["code_revision"]
                and r["harness_revision"] == gmin["harness_revision"]]
        if len(rows) != 1:
            raise SystemExit(f"{anneal['system']}: expected one pvalues.csv row, found {len(rows)}")
        tests[anneal["system"]] = rows[0]
    return tests


def sc_gmin_quotable() -> bool:
    """Every printed arm names a harness this repository holds, or an external pin."""
    pairs = sc_pairs()
    if len(pairs) != len(SC_ORDER):
        return False
    for anneal, gmin in pairs:
        for revision in (
            anneal["harness_revision"], anneal["code_revision"],
            gmin["harness_revision"], gmin["code_revision"],
        ):
            if not revision_reachable(revision):
                return False
    return True


def tuned_gmin_summary() -> list[dict[str, object]]:
    """Recompute tuned-GMIN shard 2 directly from its per-seed rows."""
    rows = read(GMIN_TUNED)
    summary = []
    for system in ("lj98", "lj104"):
        arms = {
            arm: [row for row in rows if row["system"] == system and row["code"] == arm]
            for arm in ("anneal", "gmin")
        }
        if any(len(group) != 12 for group in arms.values()):
            raise SystemExit(f"{GMIN_TUNED}: {system} needs 12 rows per arm")
        ah = sum(int(row["hit"]) for row in arms["anneal"])
        gh = sum(int(row["hit"]) for row in arms["gmin"])
        fisher = fisher_p(ah, 12, gh, 12)
        summary.append({
            "system": system,
            "anneal_hits": ah,
            "gmin_hits": gh,
            "anneal_calls": type7_median([float(row["calls_to_hit"]) for row in arms["anneal"]]),
            "gmin_calls": type7_median([float(row["calls_to_hit"]) for row in arms["gmin"]]),
            "anneal_wall": type7_median([float(row["wall_s"]) for row in arms["anneal"]]),
            "gmin_wall": type7_median([float(row["wall_s"]) for row in arms["gmin"]]),
            "fisher": fisher,
            "fisher_two": min(1.0, 2.0 * fisher),
        })
    return summary


def tuned_gmin_harness_verified(row: dict[str, str]) -> bool:
    """Verify a tuned-GMIN row by the check the audit and the ingest of a record make."""
    return revision_reach.tuned_gmin_harness_verified(row, ROOT)


def tuned_gmin_quotable() -> bool:
    """Every tuned-GMIN row the audit counts passes the harness check."""
    used = [
        row for row in read(GMIN_TUNED)
        if row["system"] in revision_reach.TUNED_SYSTEMS and row["code"] in revision_reach.TUNED_CODES
    ]
    return all(tuned_gmin_harness_verified(row) for row in used)


def crest_equal_summary() -> dict[str, dict[str, str]]:
    """The summary row of each arm at the revision of the latest rows, checked against the rows."""
    rows = read(CREST_EQUAL / "rows.csv")
    revision = rows[-1]["code_revision"]
    rows = [r for r in rows if r["code_revision"] == revision]
    summary = {r["arm"]: r for r in read(CREST_EQUAL / "summary.csv") if r["code_revision"] == revision}
    if set(summary) != set(CREST_ARMS):
        raise SystemExit(f"crest-equal: summary arms at {revision[:8]} are {sorted(summary)}")
    for arm, row in summary.items():
        runs = [r for r in rows if r["arm"] == arm]
        hits = sum(r["hit"] == "true" for r in runs)
        if (len(runs), hits) != (int(row["n"]), int(row["hits"])):
            raise SystemExit(f"crest-equal {arm}: rows give {hits}/{len(runs)}, summary {row['hits']}/{row['n']}")
    return summary


def main(_argv: list[str]) -> int:
    stale = False
    for name, (path, tokens) in RETIRED.items():
        text = path.read_text()
        if any(token in text for token in tokens):
            print(f"the retired {name} table remains in {path.name}", file=sys.stderr)
            stale = True
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
