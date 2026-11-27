# Copyright (c) anneal developers
"""Recompute the paper and benchmark-table numbers from the committed rows.

Every check reads a committed CSV or TSV, recomputes a value, and compares it
with the number printed in ``paper/jctc/*.tex`` or a bench README. Medians and
interquartile ranges are Hyndman-Fan type 7. Fisher is the two-sided exact
test. Mann-Whitney is two-sided with the tie-corrected normal approximation;
each table names whether its runner applies the continuity correction.

A printed number agrees with a recomputed one when they differ by at most
half a unit in the printed number's last digit. A bare integer below 10 is
read as exact, so a printed p of 1 cannot stand for 0.6.

Every row file read here is checked for a code revision and a host. New row
schemas that carry a physical ``machine_id`` must also carry ``cpu_model``;
the compatibility ``host`` may be a UTS-namespace label. A file without the
older provenance fields, or whose revision is not in this repository, must
be listed in ``KNOWN_GAPS``. A revision that was rebuilt on push is read
through its alias in ``ALIASES`` once the pushed equivalent is verified; one
with no pushed equivalent is listed in ``UNPUSHED`` with the reason. A printed
number may rest on a commit off HEAD only when that commit is on a record
branch listed in ``RECORD_REFS`` that origin holds at the listed SHA and tree.
Printed numbers no committed row backs are listed in ``UNTRACED``; their text
must still be where the list says. README rows on a commit nothing holds,
whose numbers the paper does not print, are listed in ``UNVERIFIABLE`` with
the revision and the reason; each must still be printed and flagged. The
revision checks need the full history and the remote
branches; a clone without them is refused, with exit status 2.

    python3 scripts/paper_audit.py            # exit 1 on any drift
    python3 scripts/paper_audit.py --report   # also print every check
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import functools
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from codata2018 import KJ_PER_KCAL  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402
import revision_reach  # noqa: E402
from gmin_pin import REVISION as GMIN_PIN_REVISION  # noqa: E402
from gmin_pin import SNAPSHOT as GMIN_PIN_SNAPSHOT  # noqa: E402
from gmin_pin import TARBALL_SHA256 as GMIN_PIN_TARBALL_SHA256  # noqa: E402

TEX = "paper/jctc/manuscript.tex"
SI = "paper/jctc/si.tex"
# The manuscript followed by the Supporting Information: a claim is checked wherever the two place it.
PAPER = "paper/jctc/manuscript.tex+si.tex"

REVISION_COLUMNS = (
    "revision", "code_revision", "harness_revision", "head", "commit",
    "anneal_head", "release_revision", "anneal_sha",
)
ALL_REVISION_COLUMNS = (
    *REVISION_COLUMNS, "anneal_revision", "anneal_commit",
    "develop_revision", "superseded_by_revision",
)
HOST_COLUMNS = ("host",)
# Trees that rows cite in place of a commit, each with the record commit whose parent has that tree.
RECORD_TREES = {"b12293b9d637dd32417be3aa482302e0f30dd594": "5c37c840ed2e9f9a2d881105363b33762b5cca31"}
MACHINE_PROVENANCE_COLUMNS = frozenset(("machine_id", "cpu_model"))
# Revisions of other codes. The GMIN build scripts refuse a tarball whose SVNREV differs.
EXTERNAL_REVISIONS = revision_reach.EXTERNAL_REVISIONS
# Local revisions that rows cite, each with the pushed commit it was rebuilt as.
ALIASES = "bench/revision-aliases.csv"
REVISION_MAP = "bench/unreachable-revisions.csv"
# Remote branches the alias table cites, plus develop. A missing one means
# the clone cannot answer the revision checks, so the audit stops.
AUDIT_REMOTE_BRANCHES = (
    "develop",
    "bench/crest-gaps",
    "bench/gmin-sc3",
    "fix/gmin-sc",
    "fix/gmin-sc2",
)
# Tips that still carry row revisions and are not named by the alias table.
# They join the remote closure when the ref is present. A missing one does
# not make the clone unauditable: the branch may have been deleted after merge.
AUDIT_OPTIONAL_REMOTES = (
    "record/repro-pin",
    "bench/cov-floor2",
    "bench/ptcu-gupta",
    "fix/conf-miss4",
    "fix/morse-gmin",
    "fix/sc-portfolio",
    "fix/sc-shelf-waypoint",
    # Pushed record tips that are not in develop. f009acc9 is on the first,
    # 0cc3814c on the second. A missing ref does not stop the audit.
    "record/rerun-tuned-gmin-r2",
    "record/g3-funnel",
    "main",
)


def audit_remote_refs() -> list[str]:
    """The remote refs whose closure counts a revision as pushed.

    The revision index and the tests that need a pushed commit both read this
    list, so a tip a test picks is one the audit knows."""
    return [f"refs/remotes/origin/{branch}" for branch in (*AUDIT_REMOTE_BRANCHES, *AUDIT_OPTIONAL_REMOTES)]


# Push-only record branches whose commits printed numbers may rest on.
RECORD_REFS = "bench/record-refs.csv"


def unreachable_reasons() -> dict[str, str]:
    """Load the one reason shared by every map row for each identifier."""
    reasons: dict[str, str] = {}
    with (ROOT / REVISION_MAP).open(newline="") as handle:
        for row in csv.DictReader(handle):
            old = reasons.setdefault(row["revision"], row["why"])
            if old != row["why"]:
                raise RuntimeError(f"{REVISION_MAP}: inconsistent reasons for {row['revision']}")
    return reasons


UNPUSHED = unreachable_reasons()

KNOWN_GAPS = {
    "data/sc96_78_hits.csv": "code_revision unrecorded on some rows; harness_revision unrecorded on some rows; host "
                             "empty or unknown on some rows",
    "paper/jctc/data/hosts/3b11c160/hosts.csv": "revision and host unrecorded (no revision or host column); cpu_model "
                                                "recorded; no machine_id column",
    "paper/jctc/data/g3_funnel/rows.csv": "revision recorded; no host column",
    "paper/jctc/data/gmin_eeec8e2c/b/host.csv": "revision recorded; no host column",
    "paper/jctc/data/gmin_eeec8e2c/c/rows.csv": "revision recorded; machine_id recorded; no cpu_model column",
    "paper/jctc/data/gmin_eeec8e2c/d/rows.csv": "revision recorded; machine_id recorded; no cpu_model column",
    "paper/jctc/data/knockouts/move-a/rows.csv": "revision recorded; no host column; machine_id recorded; no cpu_model "
                                                        "column",
    "paper/jctc/data/knockouts/pairscale/rows.csv": "revision recorded; no host column; machine_id recorded; no cpu_model "
                                                   "column",
    "paper/jctc/data/knockouts/templates/rows.csv": "revision recorded; no host column; machine_id recorded; no cpu_model "
                                                   "column",
    "paper/jctc/data/system_visuals/references.csv": "revision and host unrecorded (no revision or host column)",
    "paper/jctc/data/outside_perf/rows.csv": "revision recorded; no host column",
    "paper/jctc/data/tip4p_gmin/rows.csv": "revision recorded; no host column",
    "bench/gad-published/summary.csv": "revision recorded; no host column",
    "bench/morse-panel/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/depth-posterior-ablation/results.csv": "revision unrecorded (no revision column)",
    "results/size-tiered-selection.csv": "revision and host unrecorded (no revision or host column)",
    "bench/morse-h2h/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/fp-fingerprint/rows.csv": "revision and host unrecorded (no revision or host column)",
    "bench/tip4p-h2h/rows-h2h-r2.csv": "revision recorded; no host column",
    "bench/covalent-h2h/before/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/after/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/v2/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/v3/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/gfn2/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/gfn2/false-hits.csv": "revision recorded; no host column",
    "bench/ase-gofee-h2h/results.csv": "revision recorded; no host column; head unrecorded on some rows",
    "bench/copper-slab/rows.csv": "revision and host unrecorded (no revision or host column)",
    "bench/growpal-h2h/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/metal-cluster-h2h/ptcu-b3d03061/icosa-23.csv": "revision and host unrecorded (no revision or host column)",
    "bench/nh3-octamer/ladder.csv": "revision and host unrecorded (no revision or host column)",
    "paper/jctc/data/literature.tsv": "revision and host unrecorded (no revision or host column)",
}

UNTRACED = [
    ("bench/water-h2o20-goat/README.md", "Seed 16001, 28000 calls, same wrapper as the table above, scored −2214.506788 eV",
     "the seed 16001 rows were not committed"),
]


# README rows that print numbers from a commit no ref, kit, bundle or mbox
# holds, and that the paper does not print. Each row stays a check: it passes
# only while its exact text is in the README and the audit still flags it on
# a listed revision, and an entry that matches nothing fails.
WHY_OLD = ("no origin branch, kit, bundle or mbox holds the commit; the manuscript, SI and figure and "
           "table scripts print no number of this row")
WHY_SC = ("no origin branch, kit, bundle or mbox holds the anneal harness commit of these rerun rows, which stay "
          "in seeds.csv and results.csv; the manuscript, SI and figure and table scripts print no number of this row")
WHY_E3 = ("origin/record/bench-sc-h2h-n12 holds the harness commit of these earlier rows and no row of "
          "record-refs.csv registers that branch; the rows stay in seeds.csv, and the manuscript, SI and figure "
          "and table scripts print no number of this row")
UNVERIFIABLE = [
    ('bench/agox-lj75/README.md',
     '| current AGOX | 0/6 | 5655.108823 | -392.980553 | 4.511777 | 103.663412 | 0.009647 |',
     ('0a5136605161', 'ccc22cee607b'), WHY_OLD),
    ('bench/agox-lj75/README.md',
     '| documented AGOX | 0/6 | 14493.759020 | -372.106043 | 25.386287 | 263.511422 | 0.003795 |',
     ('0a5136605161', 'ccc22cee607b'), WHY_OLD),
    ('bench/agox-lj75/README.md',
     '| anneal | 0/6 | 55.299841 | -396.282249 | 1.210082 | 1.000000 | 1.000000 |',
     ('0a5136605161', 'ccc22cee607b'), WHY_OLD),
    ('bench/ase-lj75/README.md',
     '| basin hopping | 0/12 | 1 | 78 | 0.359317 | -393.72607097 | 3.76626003 | 89.907831 | 1.760879 | 0.567898 | 46.580477 | 43.358141 |',
     ('162d230740f0', '8173c5e2855b'), WHY_OLD),
    ('bench/ase-lj75/README.md',
     '| minima hopping | 0/12 | 1 | 78 | 0.359317 | -391.70068193 | 5.79164907 | 172.344747 | 3.375083 | 0.296295 | 0.943898 | 171.463289 |',
     ('162d230740f0', '8173c5e2855b'), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m38r6 | for_cluster vs basin hopping | 5/12 vs 0/12 | 0.0372671 | 0.324342 | 3.083165 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m38r6 | for_cluster vs minima hopping | 5/12 vs 0/12 | 0.0372671 | 0.324323 | 3.083344 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m38r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.025276 | 39.563762 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m38r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.025274 | 39.566068 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m38r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | 0.314585 | 3.178786 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m38r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | 0.314572 | 3.178919 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m38r14 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.018058 | 55.378224 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m38r14 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.018057 | 55.380532 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m55r6 | for_cluster vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.081985 | 12.197304 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m55r6 | for_cluster vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.081980 | 12.198089 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m55r6 | recommended vs basin hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.018064 | 55.359963 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m55r6 | recommended vs minima hopping | 12/12 vs 0/12 | 7.39602e-07 | 0.018062 | 55.363522 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m55r14 | for_cluster vs basin hopping | 0/12 vs 0/12 | 1 | 0.304338 | 3.285817 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m55r14 | for_cluster vs minima hopping | 0/12 vs 0/12 | 1 | 0.304323 | 3.285984 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m55r14 | recommended vs basin hopping | 5/12 vs 0/12 | 0.0372671 | 0.317431 | 3.150296 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/ase-nonlj/README.md',
     '| m55r14 | recommended vs minima hopping | 5/12 vs 0/12 | 0.0372671 | 0.317414 | 3.150456 |',
     ('90b88235fe54',), WHY_OLD),
    ('bench/covalent-h2h/README.md',
     '| Si10 | anneal G | 12/12 | 666 | 268 | -460.683738 | 33.2 | 58.43 ms | 0.010 ms |',
     ('8aeb3b0b2272', 'b22eb28536ef', 'ce1e7a46826c', 'e6edbcbabc34', 'eb7f899959de'), WHY_OLD),
    ('bench/covalent-h2h/README.md',
     '| Si10 | CREST | 11/12 | 97,521 | 8,524 | -460.683718 | 2646.6 | 26.72 ms | 0.244 ms |',
     ('8aeb3b0b2272', 'b22eb28536ef', 'e6edbcbabc34', 'eb7f899959de'), WHY_OLD),
    ('bench/covalent-h2h/README.md',
     '| C10 | anneal G | 11/12 | 6,350 | 6,077 | -571.322738 | 14.6 | 1.18 ms | 0.002 ms |',
     ('ce1e7a46826c', 'e6edbcbabc34', 'eb7f899959de'), WHY_OLD),
    ('bench/covalent-h2h/README.md',
     '| C10 | CREST | 4/12 | 78,367 | 2,880 | -571.322711 | 439.5 | 5.10 ms | 0.288 ms |',
     ('ce1e7a46826c', 'e6edbcbabc34', 'eb7f899959de'), WHY_OLD),
    ('bench/covalent-h2h/README.md',
     '| Si10 | anneal G, `ce1e7a46` | 12/12 | 665.5 | 296 | -460.683738 | 36.1 | 70.88 ms | 0.044 ms |',
     ('ce1e7a46',), WHY_OLD),
    ('bench/covalent-h2h/README.md',
     '| C10 | anneal G, `ce1e7a46` | 12/12 | 2,516.5 | 1,908 | -571.322738 | 10.9 | 4.43 ms | 0.005 ms |',
     ('ce1e7a46',), WHY_OLD),
    ('bench/covalent-h2h/README.md',
     '| C20 | anneal G, `5351f272` | 0/12 | 100,000 | 0 | -1148.292908 | 1866.6 | 18.54 ms | 0.005 ms |',
     ('5351f272',), WHY_OLD),
    ('bench/covalent-h2h/README.md',
     'no pair below 2.23 Å. Seeds 3000 and 3007 share the next,',
     ('a52726af',), WHY_OLD),
    ('bench/growpal-h2h-v2/README.md',
     '| LJ38 | yes | 21038447 | 52.6 | 560.6 | 81.3 | 479.3 | 3.87 | 22.78 |',
     ('2b46484da27f', '621761ec8e4e'), WHY_OLD),
    ('bench/growpal-h2h-v2/README.md',
     '| LJ55 | yes | 104093769 | 104.1 | 3409.8 | 614.6 | 2795.2 | 5.90 | 26.85 |',
     ('2b46484da27f', '621761ec8e4e'), WHY_OLD),
    ('bench/occat-h2h/README.md',
     '| cu38 | anneal | 12/12 | 7426.500000 | 0.155761 | 0.124461 | 0.033680 | 1.000000 |',
     ('6eda862e2e5f', '7536aeb6656b'), WHY_OLD),
    ('bench/occat-h2h/README.md',
     '| cu38 | occat | 0/12 | 400000.000000 | 370.751968 | 179.488185 | 191.263784 | 2380.261866 |',
     ('6eda862e2e5f',), WHY_OLD),
    ('bench/occat-h2h/README.md',
     '| cu38 | gmin | 0/12 | 400000.000000 | 11.972000 | 11.006452 | 0.952224 | 76.861345 |',
     ('6eda862e2e5f',), WHY_OLD),
    ('bench/occat-h2h/README.md',
     '| cu55 | anneal | 1/1 | 2341.000000 | 0.108601 | 0.076119 | 0.032482 | 1.000000 |',
     ('6eda862e2e5f',), WHY_OLD),
    ('bench/occat-h2h/README.md',
     '| cu38 | 12 | 12 | 0 | 0 | 7.39602e-07 | 7.39602e-07 | 1.02689e-05 | 3.65846e-05 | 3.65846e-05 | 3.65846e-05 | 1.02689e-05 | 3.65846e-05 | 3.65846e-05 | 3.65846e-05 |',
     ('6eda862e2e5f', '7536aeb6656b'), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| 2000-2011 (r5 record) | 9/12 | 8/12 | 1.0 | 3 | 2 | 1.0 |',
     ('795737db3e80',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| 2012-2035 (fresh) | 18/24 | 22/24 | 0.245 | 2 | 6 | 0.289 |',
     ('795737db3e80',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| lj38 | 23/24 | 23/24 | 1 | 1 | 1 | 0.998 |',
     ('5445c4f7d633',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| m38r14 | 11/24 | 19/24 | 0.036 | 1 | 9 | 0.959 |',
     ('5445c4f7d633',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| m55r14 | 18/24 | 18/24 | 1 | 5 | 5 | 0.992 |',
     ('5445c4f7d633',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| lj75 | 7/24 | 12/24 | 0.24 | 3 | 8 | 1.002 |',
     ('5445c4f7d633',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| lj98 | 16/24 | 24/24 | 0.0039 | 0 | 8 | 1.058 |',
     ('5445c4f7d633',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| lj102 | 11/24 | 18/24 | 0.075 | 3 | 10 | 1.016 |',
     ('5445c4f7d633',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| lj103 | 13/24 | 18/24 | 0.23 | 5 | 10 | 0.968 |',
     ('5445c4f7d633',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| lj104 | 11/24 | 16/24 | 0.24 | 4 | 9 | 0.993 |',
     ('5445c4f7d633',), WHY_OLD),
    ('bench/template-free2-r7/README.md',
     '| all | 110/192 | 148/192 | | 22 | 60 | |',
     ('5445c4f7d633',), WHY_OLD),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-38 | anneal | c89653df | 12/12 | 9861 | 0.414 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-38 | anneal | c89653df | 12/12 | 9762.5 | 0.427 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-38 | anneal | c89653df | 12/12 | 8185 | 0.374 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-55 | anneal | c89653df | 12/12 | 50803 | 3.216 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-55 | anneal | c89653df | 6/12 | 511393.5 | 32.844 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-55 | anneal | c89653df | 12/12 | 28767 | 1.867 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-80 | anneal | c89653df | 6/12 | 828059.5 | 104.625 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-80 | anneal | c89653df | 0/12 | 1000000 | 128.014 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-80 | anneal | c89653df | 7/12 | 981181 | 126.882 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-38 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 10344.5 | 9861 | 0.507 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-38 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 11193 | 9762.5 | 0.194 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-38 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 9177 | 8185 | 0.403 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-55 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 65397.5 | 50803 | 0.403 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-55 | e3ed7dad | c89653df | 5/12 | 6/12 | 1 | 1000000 | 511393.5 | 0.329 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-55 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 17280 | 28767 | 0.237 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-80 | e3ed7dad | c89653df | 5/12 | 6/12 | 1 | 1000000 | 828059.5 | 0.432 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-80 | e3ed7dad | c89653df | 4/12 | 0/12 | 0.0932 | 1000000 | 1000000 | 0.0367 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-80 | e3ed7dad | c89653df | 7/12 | 7/12 | 1 | 773712.5 | 981181 | 0.59 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-38 | c89653df | f9d1ac05 | 1 | 0.00111 | 0.285 | 1.139 | tie | anneal | anneal, n.s. |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-38 | c89653df | f9d1ac05 | 7.40e-07 | 1.03e-05 | 3.66e-05 | 10.087 | anneal | anneal | anneal |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-38 | c89653df | 9149d1b0 | 1 | 0.000592 | 0.341 | 1.223 | tie | anneal | anneal, n.s. |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-55 | c89653df | a50f4b02 | 1 | 0.285 | 0.000592 | 0.179 | tie | GMIN, n.s. | GMIN |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-55 | c89653df | a50f4b02 | 0.193 | 0.0424 | 1 | 0.644 | anneal, n.s. | anneal | tie |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-55 | c89653df | a50f4b02 | 1 | 0.507 | 0.0141 | 0.349 | tie | anneal, n.s. | GMIN |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-80 | c89653df | a50f4b02 | 0.0137 | 0.00705 | 0.0404 | 0.363 | anneal | anneal | GMIN |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-80 | c89653df | a50f4b02 | 1 | 1 | 3.66e-05 | 0.335 | tie | tie | GMIN |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-80 | c89653df | a50f4b02 | 0.00458 | 0.00286 | 0.000592 | 0.318 | anneal | anneal | GMIN |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-38 | anneal | c89653df | 0.414 | 0.385 | 0.030 | 33.61 | 2.86 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-38 | anneal | c89653df | 0.427 | 0.390 | 0.034 | 34.03 | 2.88 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-38 | anneal | c89653df | 0.374 | 0.342 | 0.031 | 34.21 | 2.94 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-55 | anneal | c89653df | 3.216 | 2.952 | 0.255 | 58.42 | 4.76 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-55 | anneal | c89653df | 32.844 | 29.721 | 2.650 | 59.15 | 6.08 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-55 | anneal | c89653df | 1.867 | 1.724 | 0.143 | 60.64 | 5.01 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-80 | anneal | c89653df | 104.625 | 94.912 | 9.674 | 115.00 | 11.56 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-80 | anneal | c89653df | 128.014 | 116.275 | 11.414 | 116.18 | 11.44 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-80 | anneal | c89653df | 126.882 | 114.997 | 10.898 | 117.04 | 11.12 |',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     'On sc96-55, seeds 2000-2011, the median calls move from 17280 at e3ed7dad to 28767 at c89653df, Mann-Whitney p 0.237. `sc96-55-seeds.csv` has seeds 2012-2059 under both builds from the same starts. There e3ed7dad hits 48 of 48 at a median of 24011.5 calls and c89653df hits 48 of 48 at a median of 23506.5, Mann-Whitney p 0.58; c89653df charges more calls on 19 of 48 seeds. The move on seeds 2000-2011 does not recur on these seeds, so no develop commit was bisected. Their calls are the calls charged; both builds stop at the hit, and the e3ed7dad build prints the budget for the calls to some hits.',
     ('c89653df',), WHY_SC),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-38 | anneal | e3ed7dad | 12/12 | 10344.5 | 0.518 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-38 | gmin | e3ed7dad | 1/12 | 400000 | 4.662 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-38 | anneal | e3ed7dad | 12/12 | 11193 | 0.504 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-38 | gmin | e3ed7dad | 0/12 | 400000 | 4.570 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-38 | anneal | e3ed7dad | 12/12 | 9177 | 0.399 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-38 | gmin | e3ed7dad | 0/12 | 400000 | 4.291 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-55 | anneal | e3ed7dad | 12/12 | 65397.5 | 5.104 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-55 | gmin | e3ed7dad | 0/12 | 1000000 | 22.940 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-55 | anneal | e3ed7dad | 5/12 | 1000000 | 81.861 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-55 | gmin | e3ed7dad | 0/12 | 1000000 | 24.381 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-55 | anneal | e3ed7dad | 12/12 | 17280 | 1.316 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-55 | gmin | e3ed7dad | 0/12 | 1000000 | 21.334 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-80 | anneal | e3ed7dad | 5/12 | 1000000 | 147.453 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-80 | gmin | e3ed7dad | 0/12 | 1000000 | 43.538 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-80 | anneal | e3ed7dad | 4/12 | 1000000 | 155.295 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-80 | gmin | e3ed7dad | 0/12 | 1000000 | 51.184 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-80 | anneal | e3ed7dad | 7/12 | 773712.5 | 117.332 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-80 | gmin | e3ed7dad | 0/12 | 1000000 | 43.931 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-38 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 10344.5 | 9861 | 0.507 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-38 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 11193 | 9762.5 | 0.194 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-38 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 9177 | 8185 | 0.403 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-55 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 65397.5 | 50803 | 0.403 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-55 | e3ed7dad | c89653df | 5/12 | 6/12 | 1 | 1000000 | 511393.5 | 0.329 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-55 | e3ed7dad | c89653df | 12/12 | 12/12 | 1 | 17280 | 28767 | 0.237 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc126-80 | e3ed7dad | c89653df | 5/12 | 6/12 | 1 | 1000000 | 828059.5 | 0.432 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc108-80 | e3ed7dad | c89653df | 4/12 | 0/12 | 0.0932 | 1000000 | 1000000 | 0.0367 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     '| sc96-80 | e3ed7dad | c89653df | 7/12 | 7/12 | 1 | 773712.5 | 981181 | 0.59 |',
     ('e3ed7dad',), WHY_E3),
    ('bench/metal-cluster-h2h/sc-n12/README.md',
     'On sc96-55, seeds 2000-2011, the median calls move from 17280 at e3ed7dad to 28767 at c89653df, Mann-Whitney p 0.237. `sc96-55-seeds.csv` has seeds 2012-2059 under both builds from the same starts. There e3ed7dad hits 48 of 48 at a median of 24011.5 calls and c89653df hits 48 of 48 at a median of 23506.5, Mann-Whitney p 0.58; c89653df charges more calls on 19 of 48 seeds. The move on seeds 2000-2011 does not recur on these seeds, so no develop commit was bisected. Their calls are the calls charged; both builds stop at the hit, and the e3ed7dad build prints the budget for the calls to some hits.',
     ('e3ed7dad',), WHY_E3),
]


class Audit:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str, str, str]] = []
        self.read: dict[str, list[dict[str, str]]] = {}
        self.text_cache: dict[str, str] = {}

    def record(self, ok: bool, where: str, label: str, shown: object, value: object) -> None:
        self.results.append(("OK" if ok else "DRIFT", where, label, str(shown), str(value)))

    def expect(self, where: str, label: str, shown: object, value: object) -> None:
        """A known failure that stays visible and does not fail the run."""
        self.results.append(("EXPECTED", where, label, str(shown), str(value)))

    def rows(self, path: str, delimiter: str = ",") -> list[dict[str, str]]:
        if path not in self.read:
            with (ROOT / path).open(newline="") as handle:
                lines = [line for line in handle if line.strip() and not line.startswith("#")]
            self.read[path] = list(csv.DictReader(lines, delimiter=delimiter))
        return self.read[path]

    def text(self, path: str) -> str:
        if path == PAPER:
            return self.text(TEX) + "\n" + self.text(SI)
        if path not in self.text_cache:
            self.text_cache[path] = (ROOT / path).read_text()
        return self.text_cache[path]

    def cell(self, where: str, label: str, shown: str, value: object) -> None:
        self.record(agrees(shown, value), where, label, shown, value)

    def said(self, path: str, fragment: str) -> None:
        text = self.text(path)
        if path.endswith(".tex"):
            text = tex_without_comments(text)
        self.record(collapse(fragment) in collapse(text), path, "prose", fragment, "present")

    def drifts(self) -> list[tuple[str, str, str, str, str]]:
        return [r for r in self.results if r[0] not in {"OK", "EXPECTED"}]

    def expected(self) -> list[tuple[str, str, str, str, str]]:
        return [r for r in self.results if r[0] == "EXPECTED"]


def collapse(text: str) -> str:
    return " ".join(text.split())


_COMMIT_HELD: dict[str, bool] = {}


def commit_held(rev: str) -> bool:
    """The revision is an ancestor of HEAD, by the definition the generators and the ingest share."""
    if rev not in _COMMIT_HELD:
        _COMMIT_HELD[rev] = revision_reach.commit_held(rev, ROOT)
    return _COMMIT_HELD[rev]


def tex_without_comments(text: str) -> str:
    """Remove TeX comments while preserving literal-code environments."""
    literal_environments = ("verbatim", "lstlisting")
    active: str | None = None
    cleaned = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if active is not None:
            cleaned.append(line)
            if f"\\end{{{active}}}" in body:
                active = None
            continue

        comment = len(body)
        for index, character in enumerate(body):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and body[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                comment = index
                break
        visible = body[:comment]
        cleaned.append(visible + (ending if comment == len(body) else ""))
        for environment in literal_environments:
            if f"\\begin{{{environment}}}" in visible and f"\\end{{{environment}}}" not in visible:
                active = environment
                break
    return "".join(cleaned)


def type7(values: list[float], p: float) -> float:
    ordered = sorted(float(v) for v in values)
    h = (len(ordered) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def median(values: list[float]) -> float:
    return type7(values, 0.5)


def iqr(values: list[float]) -> float:
    return type7(values, 0.75) - type7(values, 0.25)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p, summing tables no more likely than the observed."""
    hits = hits_a + hits_b
    total = n_a + n_b
    ways = math.comb(total, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def wilson_interval(hits: int, n: int) -> tuple[float, float]:
    z = statistics.NormalDist().inv_cdf(0.975)
    p = hits / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    centre = p + z * z / (2 * n)
    scale = 1 + z * z / n
    return (centre - half) / scale, (centre + half) / scale


def mcnemar(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p: twice the smaller binomial(m, 1/2) tail of the discordant seeds, capped at 1."""
    m = only_a + only_b
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, k) for k in range(min(only_a, only_b) + 1)) / 2.0**m)


def newcombe_paired_interval(both: int, only_a: int, only_b: int, neither: int) -> tuple[float, float]:
    """Method 10 of Newcombe, Stat. Med. 17, 2635 (1998), for the paired difference p_a - p_b.

    His worked example, 36, 12, 2, 0, gives 0.0569 to 0.3404."""
    n = both + only_a + only_b + neither
    pa, pb = (both + only_a) / n, (both + only_b) / n
    la, ua = wilson_interval(both + only_a, n)
    lb, ub = wilson_interval(both + only_b, n)
    product = (both + only_a) * (only_b + neither) * (both + only_b) * (only_a + neither)
    a_ = both * neither - only_a * only_b
    if a_ > 0:
        a_ = max(a_ - n / 2.0, 0.0)
    phi = a_ / math.sqrt(product) if product else 0.0
    d = pa - pb
    lower = d - math.sqrt(max((pa - la) ** 2 - 2 * phi * (pa - la) * (ub - pb) + (ub - pb) ** 2, 0.0))
    upper = d + math.sqrt(max((ua - pa) ** 2 - 2 * phi * (ua - pa) * (pb - lb) + (pb - lb) ** 2, 0.0))
    return lower, upper


def wilcoxon_signed_rank(differences: list[float]) -> tuple[int, float, float]:
    """Nonzero pairs, T+, and the two-sided exact p over all sign assignments, by counting sums of doubled ranks."""
    nonzero = sorted((abs(d), d > 0) for d in differences if d != 0)
    m = len(nonzero)
    if m == 0:
        return 0, 0.0, 1.0
    doubled: list[int] = []
    i = 0
    while i < m:
        j = i
        while j + 1 < m and nonzero[j + 1][0] == nonzero[i][0]:
            j += 1
        doubled += [i + j + 2] * (j - i + 1)
        i = j + 1
    t_plus = sum(r for r, (_, positive) in zip(doubled, nonzero) if positive) / 2.0
    counts = {0: 1}
    for r in doubled:
        nxt = dict(counts)
        for total, ways in counts.items():
            nxt[total + r] = nxt.get(total + r, 0) + ways
        counts = nxt
    center = m * (m + 1) / 4.0
    seen = abs(t_plus - center)
    far = sum(ways for total, ways in counts.items() if abs(total / 2.0 - center) >= seen - 1e-9)
    return m, t_plus, min(1.0, far / 2.0**m)


def paired_table(first: dict[str, bool], second: dict[str, bool]) -> tuple[int, int, int, int]:
    """Both, first only, second only, neither, over the seeds both arms ran; the seed sets must match."""
    if set(first) != set(second):
        raise SystemExit("paired arms do not share their seeds")
    keys = sorted(first)
    both = sum(first[k] and second[k] for k in keys)
    only_a = sum(first[k] and not second[k] for k in keys)
    only_b = sum(second[k] and not first[k] for k in keys)
    return both, only_a, only_b, len(keys) - both - only_a - only_b


def pp(p: float) -> str:
    """A p in prose, inside math: exact when four decimals hold it, else two significant figures."""
    if p >= 1.0:
        return "1"
    four = f"{p:.4f}".rstrip("0")
    if p >= 0.01 and abs(float(four) - p) < 1e-12:
        return four
    if p >= 0.01:
        return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exponent)}}}"


def mann_whitney(a: list[float], b: list[float], continuity: bool, degenerate: float | None = 1.0) -> float | None:
    """Two-sided Mann-Whitney p, normal approximation with the tie correction.

    ``degenerate`` is returned when every value ties: 1.0 for the runners that
    write 1, None for those that write scipy's nan as a blank."""
    n1, n2 = len(a), len(b)
    pooled = sorted(list(a) + list(b))
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in a) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return degenerate
    dev = abs(u - n1 * n2 / 2.0) - (0.5 if continuity else 0.0)
    return min(1.0, math.erfc(max(dev, 0.0) / math.sqrt(var) / math.sqrt(2.0)))


def mann_whitney_exact_no_ties(a: list[float], b: list[float]) -> float:
    """Exact two-sided rank-sum tail for two samples whose pooled values do not tie."""
    if len(set(a + b)) != len(a) + len(b):
        raise ValueError("exact no-ties Mann-Whitney received a tie")
    n_a, n_b = len(a), len(b)
    observed = sum(x > y for x in a for y in b)
    smaller = min(observed, n_a * n_b - observed)
    n = n_a + n_b
    max_sum = n_a * (2 * n - n_a + 1) // 2
    ways = [[0] * (max_sum + 1) for _ in range(n_a + 1)]
    ways[0][0] = 1
    for rank in range(1, n + 1):
        for count in range(min(n_a, rank), 0, -1):
            for total in range(max_sum, rank - 1, -1):
                ways[count][total] += ways[count - 1][total - rank]
    base = n_a * (n_a + 1) / 2
    extreme = sum(count for total, count in enumerate(ways[n_a])
                  if min(total - base, n_a * n_b - (total - base)) <= smaller)
    return extreme / math.comb(n, n_a)


def parse(shown: str) -> tuple[float, float, bool] | None:
    """Value, half a unit in the last printed digit, and whether it is a bare integer."""
    s = shown.strip().strip("$").replace("\\(", "").replace("\\)", "").strip()
    s = s.replace("−", "-").replace(",", "").replace("_", "").replace("\u2009", "")
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+))\s*\\times\s*10\^\{?([-+]?\d+)\}?", s)
    if m is None:
        m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+))(?:[eE]([-+]?\d+))?", s)
    if m is None:
        return None
    mantissa, exponent = m.group(1), int(m.group(2) or 0)
    places = len(mantissa.split(".")[1]) if "." in mantissa else 0
    bare = "." not in mantissa and m.group(2) is None
    return float(mantissa) * 10.0**exponent, 0.5 * 10.0 ** (exponent - places), bare


def agrees(shown: str, value: object) -> bool:
    shown = shown.strip()
    if value is None:
        return shown in ("", "-", "--", "–", "—")
    if isinstance(value, str):
        return shown == value
    parsed = parse(shown)
    if parsed is None:
        return False
    x, half, bare = parsed
    v = float(value)
    if bare and abs(x) < 10:
        return abs(x - v) <= 1e-9 * max(1.0, abs(v))
    return abs(x - v) <= half * (1 + 1e-9) + 1e-12 * abs(v)


def md_table(audit: Audit, path: str, header_start: str, after: int = 0) -> list[tuple[int, dict[str, str]]]:
    """Rows of the first pipe table at or after line ``after`` whose header starts with ``header_start``."""
    lines = audit.text(path).splitlines()
    for i, line in enumerate(lines):
        if i + 1 < after or not line.startswith("| " + header_start):
            continue
        header = [c.strip() for c in line.strip().strip("|").split("|")]
        out = []
        j = i + 2
        while j < len(lines) and lines[j].startswith("|"):
            cells = [c.strip() for c in lines[j].strip()[1:-1].split("|")]
            out.append((j + 1, dict(zip(header, cells))))
            j += 1
        return out
    raise SystemExit(f"{path}: no table starting '{header_start}' after line {after}")


def tex_table(audit: Audit, path: str, label: str) -> list[list[str]]:
    text = audit.text(path)
    start = text.index("\\label{" + label + "}")
    body = text[text.index("\\midrule", start) + len("\\midrule"): text.index("\\bottomrule", start)]
    return [[c.strip() for c in row.split("&")] for row in body.split("\\\\") if row.strip()]


def hits_of(rows: list[dict[str, str]], key: str = "hit") -> int:
    return sum(r[key].strip().lower() in ("1", "true", "yes") for r in rows)


def pick(rows: list[dict[str, str]], **where: str) -> list[dict[str, str]]:
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def frac(rows: list[dict[str, str]], key: str = "hit") -> str:
    return f"{hits_of(rows, key)}/{len(rows)}"


# --------------------------------------------------------------------------- manuscript

CLUSTER = {"LJ\\(_{38}\\)": 38, "LJ\\(_{55}\\)": 55, "LJ\\(_{75}\\)": 75, "LJ\\(_{98}\\)": 98, "LJ\\(_{104}\\)": 104}


GMIN_PANEL_REVISION = "c2de652dca0bc75e2b0cb4afffac91fd656404e4"
GMIN_PANEL_README = "bench/gmin-panel/README.md"
# The 120 rows of session 20261121T133601Z cite this anneal commit.
# The scripts that drove them are on the record commit below, whose
# parent is b8190ce6. 9837358a is an ancestor, not that parent.
GMIN_PANEL_SESSION = "9837358ae1027067ff654fac8b48969020853452"
GMIN_RECORD_REF = "refs/audit/record/gmin-184"
GMIN_RECORD_SHA = "e4341de058b73366f2d7b5f17995da283d57bad6"
GMIN_RECORD_TREE = "63d1488f465fff1e65ce9b338641ef945fad75a9"
GMIN_RECORD_PARENT = "b8190ce61a64ff6cb61ee6bfc656b0657d78494d"
GMIN_RECORD_README = "record/gmin-184/README.md"


def mann_whitney_u(a: list[float], b: list[float]) -> float:
    n1, n2 = len(a), len(b)
    pooled = sorted(list(a) + list(b))
    n = n1 + n2
    rank: dict[float, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        i = j + 1
    return sum(rank[v] for v in a) - n1 * (n1 + 1) / 2.0


def gmin_ranked_calls(rows: list[dict[str, str]], budget: int) -> list[float] | None:
    if any(r["hit"] == "0" and int(r["charged"]) < budget for r in rows):
        return None
    return [float(r["evals_to_hit"]) if r["hit"] == "1" else math.inf for r in rows]


def gmin_ranked_wall(left: list[dict[str, str]], right: list[dict[str, str]]) -> tuple[list[float], list[float]] | None:
    l_miss = [float(r["wall_s"]) for r in left if r["hit"] == "0"]
    r_miss = [float(r["wall_s"]) for r in right if r["hit"] == "0"]
    if l_miss and r_miss:
        return None
    l_hit = [float(r["wall_s"]) for r in left if r["hit"] == "1"]
    r_hit = [float(r["wall_s"]) for r in right if r["hit"] == "1"]
    if any(m < h for m in l_miss for h in r_hit) or any(m < h for m in r_miss for h in l_hit):
        return None
    return (
        [float(r["wall_s"]) if r["hit"] == "1" else math.inf for r in left],
        [float(r["wall_s"]) if r["hit"] == "1" else math.inf for r in right],
    )


def gmin_panel_groups(audit: Audit) -> tuple[object, dict[tuple[str, str], list[dict[str, str]]]]:
    """The latest panel row of each system, arm and seed, in seed order, after row validation."""
    spec = importlib.util.spec_from_file_location("gmin_panel", ROOT / "scripts/gmin_panel/run_panel.py")
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    audit.rows("bench/morse-panel/rerun-20261119/rows.csv")
    rows = audit.rows("bench/gmin-panel/rows.csv")
    references = {row[0]: row[3] for row in panel.SYSTEMS}
    latest: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    bad_rows: list[str] = []
    for index, r in enumerate(rows, start=2):
        system = r["system"]
        budget = panel.PAPER_BUDGETS.get(system)
        ref = references.get(system)
        if budget is None or ref is None or r["hit"] not in ("0", "1") or int(r["budget"]) != budget:
            bad_rows.append(f"line {index}")
            continue
        hit = r["hit"] == "1"
        best = float(r["best"])
        evals, charged = int(r["evals_to_hit"]), int(r["charged"])
        wall, pot, outside = float(r["wall_s"]), float(r["pot_s"]), float(r["outside_s"])
        ok = (
            hit == energy_hits(best, ref, panel.HIT_TOL)
            and (not hit or 0 < evals <= charged <= budget)
            and (hit or (evals == budget and charged == budget))
            and 0.0 <= pot <= wall + 2e-6
            and outside >= 0.0
            and abs(max(wall - pot, 0.0) - outside) <= 2e-6
        )
        if not ok:
            bad_rows.append(f"line {index} {system} {r['code']} {r['seed']}")
        latest.setdefault((system, r["code"]), {})[int(r["seed"])] = r
    audit.cell("bench/gmin-panel/rows.csv", "row invariants", " ".join(bad_rows) or "ok", "ok")
    groups = {}
    for system in panel.PAPER_BUDGETS:
        for code in ("anneal", "gmin"):
            by_seed = latest.get((system, code), {})
            group = [by_seed[s] for s in sorted(by_seed)]
            where = f"bench/gmin-panel/rows.csv {system} {code}"
            audit.cell(where, "seeds", " ".join(r["seed"] for r in group), " ".join(map(str, panel.SEEDS)))
            audit.cell(where, "budget", " ".join(sorted({r["budget"] for r in group})), str(panel.PAPER_BUDGETS[system]))
            groups[(system, code)] = group
        pair_sessions = {r["session"] for code in ("anneal", "gmin") for r in groups[(system, code)]}
        audit.record(len(pair_sessions) >= 1, f"bench/gmin-panel/rows.csv {system}", "paired session count is nonzero",
                     "at least one", str(len(pair_sessions)))
    hosts = {r["host"] for group in groups.values() for r in group}
    want_hosts = {"3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873", "bench"}
    audit.cell("bench/gmin-panel/rows.csv", "host", " ".join(sorted(hosts)), " ".join(sorted(want_hosts)))
    return panel, groups


def gmin_panel_summary(audit: Audit, panel: object, groups: dict) -> None:
    history = audit.rows("bench/gmin-panel/summary.csv")
    latest = {row["system"]: row for row in history}
    summary = [latest[system] for system in panel.PAPER_BUDGETS]
    references = {row[0]: row[3] for row in panel.SYSTEMS}
    audit.cell("bench/gmin-panel/summary.csv", "systems",
               " ".join(r["system"] for r in summary), " ".join(panel.PAPER_BUDGETS))
    for line in summary:
        system = line["system"]
        budget = panel.PAPER_BUDGETS[system]
        a, g = groups[(system, "anneal")], groups[(system, "gmin")]
        where = f"bench/gmin-panel/summary.csv {system}"
        a_calls = [float(r["evals_to_hit"]) for r in a]
        g_calls = [float(r["evals_to_hit"]) for r in g]
        a_hits = [float(r["evals_to_hit"]) for r in a if r["hit"] == "1"]
        g_hits = [float(r["evals_to_hit"]) for r in g if r["hit"] == "1"]
        rc_a, rc_g = gmin_ranked_calls(a, budget), gmin_ranked_calls(g, budget)
        rw = gmin_ranked_wall(a, g)
        audit.cell(where, "budget", line["budget"], float(budget))
        audit.cell(where, "reference", line["reference"], references[system])
        audit.cell(where, "anneal_hits", line["anneal_hits"], float(hits_of(a)))
        audit.cell(where, "anneal_n", line["anneal_n"], float(len(a)))
        audit.cell(where, "gmin_hits", line["gmin_hits"], float(hits_of(g)))
        audit.cell(where, "gmin_n", line["gmin_n"], float(len(g)))
        audit.cell(where, "fisher_p", line["fisher_p"], fisher(hits_of(a), len(a), hits_of(g), len(g)))
        audit.cell(where, "anneal_evals_median", line["anneal_evals_median"], median(a_calls))
        audit.cell(where, "gmin_evals_median", line["gmin_evals_median"], median(g_calls))
        audit.cell(where, "anneal_evals_iqr", line["anneal_evals_iqr"], iqr(a_calls))
        audit.cell(where, "gmin_evals_iqr", line["gmin_evals_iqr"], iqr(g_calls))
        audit.cell(where, "anneal_evals_median_to_hit", line["anneal_evals_median_to_hit"],
                   median(a_hits) if a_hits else None)
        audit.cell(where, "gmin_evals_median_to_hit", line["gmin_evals_median_to_hit"],
                   median(g_hits) if g_hits else None)
        audit.cell(where, "mannwhitney_u_calls", line["mannwhitney_u_calls"],
                   mann_whitney_u(rc_a, rc_g) if rc_a is not None and rc_g is not None else None)
        audit.cell(where, "mannwhitney_p_calls", line["mannwhitney_p_calls"],
                   mann_whitney(rc_a, rc_g, continuity=True) if rc_a is not None and rc_g is not None else None)
        audit.cell(where, "anneal_wall_median_s", line["anneal_wall_median_s"], median([float(r["wall_s"]) for r in a]))
        audit.cell(where, "gmin_wall_median_s", line["gmin_wall_median_s"], median([float(r["wall_s"]) for r in g]))
        audit.cell(where, "mannwhitney_u_wall", line["mannwhitney_u_wall"],
                   mann_whitney_u(*rw) if rw is not None else None)
        audit.cell(where, "mannwhitney_p_wall", line["mannwhitney_p_wall"],
                   mann_whitney(*rw, continuity=True) if rw is not None else None)
        for arm, group in (("anneal", a), ("gmin", g)):
            audit.cell(where, f"{arm}_pot_median_s", line[f"{arm}_pot_median_s"], median([float(r["pot_s"]) for r in group]))
            audit.cell(where, f"{arm}_outside_median_s", line[f"{arm}_outside_median_s"],
                       median([float(r["outside_s"]) for r in group]))
            audit.cell(where, f"{arm}_best_min", line[f"{arm}_best_min"], min(float(r["best"]) for r in group))
        audit.cell(where, "anneal_revision", line["anneal_revision"], " ".join(sorted({r["code_revision"] for r in a})))
        audit.cell(where, "gmin_revision", line["gmin_revision"], " ".join(sorted({r["code_revision"] for r in g})))
        audit.cell(where, "harness_revision", line["harness_revision"], " ".join(sorted({r["harness_revision"] for r in a + g})))
        audit.cell(where, "host", line["host"], " ".join(sorted({r["host"] for r in a + g})))
        audit.cell(where, "sessions", line["sessions"], " ".join(sorted({r["session"] for r in a + g})))
        audit.cell(where, "seeds", line["seeds"], " ".join(r["seed"] for r in a))


def gmin_panel_line(audit: Audit, where: str, cells: list[str], groups: dict, system: str, arm: str,
                    budget: int) -> None:
    """One table line: Budget, Arm, Hits, Median calls, IQR, Wall, Inside, Outside, Fisher p, MW p calls, MW p wall."""
    group = groups[(system, arm)]
    other = groups[(system, "gmin" if arm == "anneal" else "anneal")]

    def col(rows: list[dict[str, str]], key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    audit.cell(where, "hits", cells[2], frac(group))
    audit.cell(where, "median calls", cells[3], median(col(group, "evals_to_hit")))
    audit.cell(where, "IQR calls", cells[4], iqr(col(group, "evals_to_hit")))
    audit.cell(where, "wall median", cells[5], median(col(group, "wall_s")))
    audit.cell(where, "inside median", cells[6], median(col(group, "pot_s")))
    audit.cell(where, "outside median", cells[7], median(col(group, "outside_s")))
    if arm == "anneal":
        rc_a, rc_g = gmin_ranked_calls(group, budget), gmin_ranked_calls(other, budget)
        rw = gmin_ranked_wall(group, other)
        audit.cell(where, "budget", cells[0], float(budget))
        audit.cell(where, "Fisher p", cells[8], fisher(hits_of(group), len(group), hits_of(other), len(other)))
        audit.cell(where, "MW p calls", cells[9],
                   mann_whitney(rc_a, rc_g, continuity=True) if rc_a is not None and rc_g is not None else None)
        audit.cell(where, "MW p wall", cells[10], mann_whitney(*rw, continuity=True) if rw is not None else None)
    else:
        for label, cell in zip(("budget", "Fisher p", "MW p calls", "MW p wall"), (cells[0], *cells[8:])):
            audit.cell(where, label, cell, None)


def gmin_paired(groups: dict, system: str) -> dict[str, object]:
    """Both arms of a GMIN panel cluster paired by seed: outcome table, McNemar, Newcombe, Wilcoxon on calls and wall."""
    a = {r["seed"]: r for r in groups[(system, "anneal")]}
    g = {r["seed"]: r for r in groups[(system, "gmin")]}
    table = paired_table({k: r["hit"] == "1" for k, r in a.items()}, {k: r["hit"] == "1" for k, r in g.items()})
    seeds = sorted(a)
    return {
        "table": table,
        "mcnemar": mcnemar(table[1], table[2]),
        "interval": newcombe_paired_interval(*table),
        "calls": wilcoxon_signed_rank([float(a[k]["evals_to_hit"]) - float(g[k]["evals_to_hit"]) for k in seeds]),
        "wall": wilcoxon_signed_rank([float(a[k]["wall_s"]) - float(g[k]["wall_s"]) for k in seeds]),
    }


def gmin_tex_line(audit: Audit, where: str, cells: list[str], groups: dict, system: str, arm: str,
                  budget: int) -> None:
    """One tab:gmin line: budget, arm, hits, outcomes, arm-specific evaluation count, and wall."""
    group = groups[(system, arm)]
    stats = gmin_paired(groups, system)
    both, only_a, only_g, neither = stats["table"]

    def col(key: str) -> list[float]:
        return [float(r[key]) for r in group]

    audit.cell(where, "hits", cells[2], frac(group))
    audit.cell(where, "seeds this arm alone hits", cells[3], float(only_a if arm == "anneal" else only_g))
    audit.cell(where, "median calls", cells[6], median(col("evals_to_hit")))
    audit.cell(where, "IQR calls", cells[7], iqr(col("evals_to_hit")))
    audit.cell(where, "wall median", cells[8], median(col("wall_s")))
    audit.cell(where, "inside median", cells[9], median(col("pot_s")))
    audit.cell(where, "outside median", cells[10], median(col("outside_s")))
    if arm == "anneal":
        audit.cell(where, "budget", cells[0], float(budget))
        audit.cell(where, "both hit", cells[4], float(both))
        audit.cell(where, "neither hits", cells[5], float(neither))
    else:
        for label, cell in zip(("budget", "both", "neither"), (cells[0], cells[4], cells[5])):
            audit.cell(where, label, cell, None)


def gmin_panel_readme(audit: Audit, panel: object, groups: dict) -> None:
    rows = audit.rows("bench/gmin-panel/rows.csv")
    first_session = "20261117T035457Z"
    old_au = pick(rows, system="au55", code="gmin", session=first_session)
    new_session = groups[("au55", "gmin")][0]["session"]
    audit.said(GMIN_PANEL_README, GMIN_PANEL_REVISION)
    audit.said(GMIN_PANEL_README, panel.GMIN_REV)
    audit.said(GMIN_PANEL_README, GMIN_PANEL_SESSION)
    audit.said(GMIN_PANEL_README, GMIN_RECORD_SHA)
    audit.said(GMIN_PANEL_README, GMIN_RECORD_TREE)
    audit.said(GMIN_PANEL_README, GMIN_RECORD_PARENT)
    audit.said(GMIN_PANEL_README, GMIN_RECORD_REF)
    audit.said(GMIN_RECORD_README, GMIN_RECORD_SHA)
    audit.said(GMIN_RECORD_README, GMIN_RECORD_PARENT)
    audit.said(GMIN_RECORD_README, GMIN_RECORD_REF)
    registry = (ROOT / RECORD_REFS).read_text()
    registered = f"{GMIN_RECORD_REF},{GMIN_RECORD_SHA},{GMIN_RECORD_TREE}"
    audit.record(registered in registry, RECORD_REFS, "gmin-184 record", registered, "registered")
    session = [r for r in rows if r["harness_revision"] == GMIN_PANEL_SESSION]
    audit.record(len(session) == 120 and sum(r["code"] == "gmin" for r in session) == 60,
                 "bench/gmin-panel/rows.csv", "session harness_revision",
                 str(len(session)), GMIN_PANEL_SESSION)
    audit.said(GMIN_PANEL_README, panel.GMIN_RERUN_TARBALL_SHA256)
    for rev in sorted({r["code_revision"] for (_s, arm), rs in groups.items() if arm == "gmin" for r in rs}):
        audit.said(GMIN_PANEL_README, rev)
    for sha in (
        "63ee803397a625cb4b51218f5467f4baa2e6da6a",
        "ac5cfc3fd39c2b29cb71ed53ca1b34b39d33f4c2",
        "1c1b12384683485cf820d3a244151b48973dc0f7",
        "465e583be21e9b82befc94ff01cf931a13f8f0a9",
    ):
        audit.said(GMIN_PANEL_README, sha)
    source_note = "bench/gmin-panel/gmin-source.txt"
    patch_lines = [
        line.removeprefix("patch=").strip()
        for line in audit.text(source_note).splitlines()
        if line.startswith("patch=")
    ]
    audit.record(bool(patch_lines), source_note, "patch lines present", str(len(patch_lines)), ">= 1")
    for entry in patch_lines:
        rel_patch, _, recorded_sha = entry.partition(" ")
        actual_sha = hashlib.sha256((ROOT / rel_patch).read_bytes()).hexdigest()
        audit.record(
            recorded_sha.strip() == actual_sha,
            source_note,
            f"in-tree patch sha256 {rel_patch}",
            recorded_sha.strip(),
            actual_sha,
        )
    audit.said(
        source_note,
        "rows_patch=bench/lj104-lattice/v2/h2h-gmin.patch@465e583be21e9b82befc94ff01cf931a13f8f0a9 "
        "a31221a67883d20032a029e34e3d81bd62c5bcb786c97fff02a9c234adc34553",
    )
    audit.said(GMIN_PANEL_README, f"adds the energy {dict((r[0], r[3]) for r in panel.SYSTEMS)['lj103']:.6f}")
    rad = panel.gmin_gupta_radius(55)
    step = panel.LJ_STEP * panel.GOLD_LENGTH_PER_LJ
    avoid = panel.LJ_AVOID * panel.GOLD_LENGTH_PER_LJ
    audit.said(GMIN_PANEL_README, f"(`RADIUS {rad:.6f}`), `STEP {step:.6f} 0.51` and `AVOID {avoid:.6f} 20 F`")
    audit.said(GMIN_PANEL_README, "`TEMPERATURE 0.1` eV from Wales's `examples/GMIN/MGupta.tar.bz2`")
    audit.said(GMIN_PANEL_README,
               f"The earlier Au55 rows of session `{first_session}` ran the alloy block of "
               "`scripts/metal_h2h/enter_gold.sh` (`RADIUS 50.0`, `TEMPERATURE 1.0`, `STEP 0.40 0.51` and "
               "`NEWRESTART 40` without `AVOID ... F`), which left `AVOIDRESEEDT` true, reseeded into a "
               f"57.7 Å cube after 40 stagnant steps, and hit {frac(old_au)}; those rows stay in `rows.csv`, "
               f"and session `{new_session}` reran both arms of Au55 under the Gupta input above "
               f"({frac(groups[('au55', 'anneal')])} against {frac(groups[('au55', 'gmin')])}).")
    s0 = [r for r in rows if r["session"] == first_session]
    gaps = []
    for prev, cur in zip(s0, s0[1:]):
        t0 = datetime.datetime.strptime(prev["utc"], "%Y-%m-%dT%H:%M:%SZ")
        t1 = datetime.datetime.strptime(cur["utc"], "%Y-%m-%dT%H:%M:%SZ")
        gap = (t1 - t0).total_seconds() - float(cur["wall_s"])
        if gap > 5.0:
            gaps.append((gap, cur))
    audit.cell(GMIN_PANEL_README, "VM suspend count", str(len(gaps)), "3")
    if len(gaps) == 3:
        g0, g1, g2 = (g for g, _ in gaps)
        r0_, r1_, r2_ = (r for _, r in gaps)
        audit.said(GMIN_PANEL_README, f"lie {g0:.0f} s, {g1:.0f} s and {g2:.0f} s after the row before plus their own wall.")
        audit.said(GMIN_PANEL_README,
                   f"On LJ75 GMIN seed {r0_['seed']} (wall {float(r0_['wall_s']):.1f} s, outside {float(r0_['outside_s']):.1f} s) "
                   f"and LJ98 anneal seed {r2_['seed']} (wall {float(r2_['wall_s']):.1f} s, outside {float(r2_['outside_s']):.1f} s)")
        other_out = [float(r["outside_s"]) for r in groups[("lj98", "gmin")] if r["seed"] != r1_["seed"]]
        post = next(r for r in groups[("lj98", "gmin")] if r["seed"] == "2005")
        audit.said(GMIN_PANEL_README,
                   f"On LJ98 GMIN seed {r1_['seed']} (wall {float(r1_['wall_s']):.1f} s), `pot_s` stayed at "
                   f"{float(r1_['pot_s']):.1f} s while `outside_s` rose to {float(r1_['outside_s']):.1f} s against "
                   f"{min(other_out):.1f}-{max(other_out):.1f} s on the other eleven seeds, and LJ98 GMIN seed 2005 "
                   f"right after the third resume took {float(post['wall_s']):.1f} s (inside {float(post['pot_s']):.1f} s)")


def bisected_z975() -> float:
    low, high = 0.0, 5.0
    for _ in range(200):
        mid = (low + high) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < 0.975:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def paired_stats_module():
    """Load ``scripts/paired_stats.py`` under a name nothing else imports.

    ``spec_from_file_location("paired_stats", ...)`` would reuse whatever
    ``sys.modules["paired_stats"]`` already holds once another file of
    that basename has been imported.
    """
    name = "anneal_scripts_paired_stats"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/paired_stats.py")
    if spec is None or spec.loader is None:
        raise ImportError("scripts/paired_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAPER_CHAIN = (
    "paper/jctc/manuscript.tex", "paper/jctc/si.tex", "paper/jctc/cover-letter.tex",
    "scripts/paired_stats.py", "scripts/gmin_panel/run_panel.py", "paper/jctc/scripts/ase_table.py",
    "scripts/h2o16_crest/summarize.py",
    "paper/jctc/scripts/water_rerun.py",
    "paper/jctc/scripts/fig_water_crest.py", "paper/jctc/scripts/water_results_table.py",
    "paper/jctc/scripts/water_significance.py", "paper/jctc/scripts/lj75_gmin_table.py",
)


def paired_statistics(audit: Audit) -> None:
    """The paired tests of the audit against known answers, and against the generators' module on the panel."""
    where = "scripts/paper_audit.py paired statistics"
    audit.cell(where, "McNemar, five discordant seeds one way", "0.0625", mcnemar(5, 0))
    audit.cell(where, "McNemar, 1 against 4", "0.375", mcnemar(1, 4))
    audit.cell(where, "McNemar, 12 against 0", "0.00048828125", mcnemar(12, 0))
    audit.cell(where, "Newcombe's paired worked example, 36 12 2 0", "0.0569 0.3404",
               " ".join(f"{x:.4f}" for x in newcombe_paired_interval(36, 12, 2, 0)))
    audit.cell(where, "Wilcoxon on 1 to 5: pairs, T+, p", "5 15 0.0625",
               " ".join(f"{x:g}" for x in wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])))
    audit.cell(where, "Wilcoxon with a tie and a zero: pairs, T+", "3 1.5",
               " ".join(f"{x:g}" for x in wilcoxon_signed_rank([0.0, 1.0, -1.0, -2.0])[:2]))
    audit.said("scripts/paired_stats.py", "if probability(value) <= observed * (1.0 + 1.0e-12)")
    audit.said("scripts/paired_stats.py", "Z95 = NormalDist().inv_cdf(0.975)")
    audit.said(SI, "\\qquad z = \\Phi^{-1}(0.975),")
    shared = paired_stats_module()
    audit.cell(where, "shared z against the bisected 0.975 normal quantile", f"{bisected_z975():.12f}",
               f"{shared.Z95:.12f}")
    worst = max(abs(shared.fisher_exact(a, 12, b, 12) - fisher(a, 12, b, 12)) for a in range(13) for b in range(13))
    audit.record(worst < 1e-9, where, "the audit and scripts/paired_stats.py agree on Fisher at 12 against 12",
                 "< 1e-9", f"{worst:.1e}")
    literal = re.compile(r"(?<![\d.])1\.9(?:6|59)\d*")
    for path in PAPER_CHAIN:
        found = sorted({m.group(0) for m in literal.finditer(audit.text(path))})
        audit.cell(path, "literal normal quantile", "none", " ".join(found) or "none")
    audit.said(SI, "is 0.0625 at \\(m=5\\) and 0.03125 at \\(m=6\\).")
    audit.cell(SI, "the McNemar floor at five and six discordant seeds", "0.0625 0.03125",
               f"{mcnemar(5, 0)} {mcnemar(6, 0)}")
    rows = audit.rows("bench/gmin-panel/rows.csv")
    budgets = gmin_panel_groups(audit)[0].PAPER_BUDGETS
    misses = [r for r in rows if r["hit"] != "1" and r["system"] in budgets]
    audit.record(all(int(float(r["evals_to_hit"])) == int(r["budget"]) for r in misses), "bench/gmin-panel/rows.csv",
                 "every miss carries the budget as its calls, as the Wilcoxon test counts it", "budget",
                 "budget" if all(int(float(r["evals_to_hit"])) == int(r["budget"]) for r in misses) else "other")
    module = paired_stats_module()
    panel, groups = gmin_panel_groups(audit)
    worst = 0.0
    for system in panel.PAPER_BUDGETS:
        mine = gmin_paired(groups, system)
        a = {r["seed"]: r for r in groups[(system, "anneal")]}
        g = {r["seed"]: r for r in groups[(system, "gmin")]}
        seeds = sorted(a)
        table = module.outcome_table([a[k]["hit"] == "1" for k in seeds], [g[k]["hit"] == "1" for k in seeds])
        audit.cell(where, f"{system} outcome table agrees", " ".join(map(str, mine["table"])), " ".join(map(str, table)))
        theirs = [float(module.mcnemar_exact(table[1], table[2])), *module.newcombe_paired(*table)]
        for key in ("calls", "wall"):
            column = "evals_to_hit" if key == "calls" else "wall_s"
            theirs.append(module.wilcoxon_exact([float(a[k][column]) - float(g[k][column]) for k in seeds])[2])
        ours = [mine["mcnemar"], *mine["interval"], mine["calls"][2], mine["wall"][2]]
        worst = max(worst, *(abs(x - y) for x, y in zip(ours, theirs)))
    audit.record(worst < 1e-12, where, "the audit and scripts/paired_stats.py agree on every panel cluster", "< 1e-12",
                 f"{worst:.1e}")
    rerun_panel, rerun = gmin_rerun_groups(audit)
    develop = gmin_develop_groups(audit, rerun_panel)
    worst = 0.0
    for system, rerun in [(s, rerun) for s in GMIN_RERUN_SYSTEMS] + [(s, develop) for s in GMIN_DEVELOP_SYSTEMS]:
        mine = gmin_paired(rerun, system)
        a = {r["seed"]: r for r in rerun[(system, "anneal")]}
        g = {r["seed"]: r for r in rerun[(system, "gmin")]}
        seeds = sorted(a)
        table = module.outcome_table([a[k]["hit"] == "1" for k in seeds], [g[k]["hit"] == "1" for k in seeds])
        theirs = [float(module.mcnemar_exact(table[1], table[2])), *module.newcombe_paired(*table)]
        for column in ("evals_to_hit", "wall_s"):
            theirs.append(module.wilcoxon_exact([float(a[k][column]) - float(g[k][column]) for k in seeds])[2])
        ours = [mine["mcnemar"], *mine["interval"], mine["calls"][2], mine["wall"][2]]
        worst = max(worst, *(abs(x - y) for x, y in zip(ours, theirs)))
    audit.record(worst < 1e-12, where, "the audit and scripts/paired_stats.py agree on every rerun cluster", "< 1e-12",
                 f"{worst:.1e}")


def manuscript_gmin(audit: Audit) -> None:
    """The bench README of the earlier panel against its rows, and tab:gmin against the imported rerun."""
    panel, groups = gmin_panel_groups(audit)
    gmin_panel_summary(audit, panel, groups)
    arms = {"anneal": "anneal", "GMIN": "gmin"}
    by_label = {labels[0]: system for system, labels in panel.LABELS.items()}
    shown = []
    system = None
    for line, row in md_table(audit, GMIN_PANEL_README, "System | Budget | Arm"):
        cells = list(row.values())
        if cells[0]:
            system = by_label.get(cells[0])
        arm = arms.get(cells[2])
        where = f"{GMIN_PANEL_README}:{line} {cells[0] or system} {cells[2]}"
        if system is None or arm is None or len(cells) != 12:
            audit.record(False, where, "table line", " | ".join(cells), "a panel system and arm")
            continue
        shown.append((system, arm))
        gmin_panel_line(audit, where, cells[1:], groups, system, arm, panel.PAPER_BUDGETS[system])
    want = [(system, arm) for system in panel.PAPER_BUDGETS for arm in ("anneal", "gmin")]
    audit.cell(GMIN_PANEL_README, "gmin panel lines", " ".join(f"{s}/{a}" for s, a in shown),
               " ".join(f"{s}/{a}" for s, a in want))
    gmin_panel_readme(audit, panel, groups)
    gmin_rerun(audit)


GMIN_RERUN_REVISION = "eeec8e2c216ae1506768bca8e29846c1c039ca3b"
GMIN_TARBALL_DATE = "2026-11-19"
GMIN_TARBALL_SVNREV = "9cbec1a446352bc9ef064bd2b1a5767194655919"
GMIN_TARBALL_SHA256 = "ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776"
GMIN_RERUN_ROW_ID = "9cbec1a446352bc9ef064bd2b1a5767194655919"
GMIN_RERUN_RAW_SYSTEMS = ("lj38", "lj55", "lj75", "lj98", "lj102", "lj103", "lj104", "lj150", "m80r6")
GMIN_RERUN_SYSTEMS = GMIN_RERUN_RAW_SYSTEMS
TEMPLATE_REMOVAL = "1c1b12384683485cf820d3a244151b48973dc0f7"
SUBMITTED = "61fa67cf380e46876b569da45e7cc091647a775b"
GMIN_DEVELOP = "paper/jctc/data/gmin_develop"
GMIN_DEVELOP_HARNESS = "f009acc96bb95023eeb6e394e863630c752a5d05"
GMIN_DEVELOP_SYSTEMS = ("au55", "m55r6", "m55r14", "m80r6", "m80r14")
GMIN_DEVELOP_HOST = "e06d886d760e0a59870126b963636ff6408201594f2686f72deb66816fccadb3"
GMIN_DEVELOP_SESSION = "20261120T072444Z"
GMIN_SOURCE_NOTE = "bench/gmin-panel/gmin-source.txt"


def gmin_rows_patch(audit: Audit) -> tuple[str, str, str]:
    """The cap patch the committed GMIN rows were built with: path, commit and sha256 from rows_patch."""
    for line in audit.text(GMIN_SOURCE_NOTE).splitlines():
        if line.startswith("rows_patch="):
            located, sha = line.removeprefix("rows_patch=").split()
            path, commit = located.split("@")
            return path, commit, sha
    raise RuntimeError(f"{GMIN_SOURCE_NOTE} has no rows_patch line")


def imported(audit: Audit, folder: str, name: str = "rows.csv", keys: tuple[str, str] = ("source", "blob")) -> str:
    """An imported row file is the blob its PROVENANCE names, at that path of a commit on that record branch."""
    fields = dict(line.split(" ", 1) for line in audit.text(f"{folder}/PROVENANCE").splitlines())
    path = f"{folder}/{name}"
    source, blob = fields[keys[0]], fields[keys[1]]
    local = git("hash-object", str(ROOT / path)).stdout.strip()
    audit.cell(path, "blob of the copy", blob, local)
    audit.cell(path, "blob at the record commit", blob, git("rev-parse", f"{fields['commit']}:{source}").stdout.strip())
    on_branch = git("merge-base", "--is-ancestor", fields["commit"], f"refs/remotes/origin/{fields['branch']}").returncode == 0
    audit.record(on_branch, path, "the record commit is on the record branch of origin", fields["branch"],
                 "on it" if on_branch else "not on it")
    return path


def ancestor_of(audit: Audit, where: str, revision: str, of: str = "HEAD") -> bool:
    ok = git("merge-base", "--is-ancestor", revision, of).returncode == 0
    audit.record(ok, where, f"{revision[:8]} is an ancestor of {of[:8]}", "ancestor", "ancestor" if ok else "not")
    return ok


def imported_gmin_row_id(audit: Audit) -> str:
    """Historical identifier stored in the imported rows; the dated tarball is the source pin."""
    identifiers = {
        r["code_revision"]
        for shard in GMIN_SHARDS
        for r in audit.rows(f"{GMIN_RERUN}/{shard}/rows.csv")
        if r["code"] == "gmin"
    }
    return next(iter(identifiers)) if len(identifiers) == 1 else ""


def gmin_rerun_groups(audit: Audit) -> tuple[object, dict[tuple[str, str], list[dict[str, str]]]]:
    """The four imported shards of the eeec8e2c rerun, checked row by row, grouped by cluster and code in seed order."""
    panel = load_script("gmin_rerun_panel", "scripts/gmin_panel/run_panel.py")
    references = {row[0]: row[3] for row in panel.SYSTEMS}
    rows, bad, first = [], [], defaultdict(lambda: defaultdict(int))
    for shard in GMIN_SHARDS:
        path = imported(audit, f"{GMIN_RERUN}/{shard}")
        seen = set()
        for index, r in enumerate(audit.rows(path), start=2):
            key = (r["system"], r["seed"])
            if key not in seen:
                first[r["system"]][r["code"]] += 1
                seen.add(key)
            budget, ref = panel.PAPER_BUDGETS.get(r["system"]), references.get(r["system"])
            revision_ok = r["code_revision"] == GMIN_RERUN_REVISION if r["code"] == "anneal" else bool(r["code_revision"])
            tarball_fields = "gmin_snapshot" in r or "tarball_sha256" in r
            tarball_ok = r["code"] != "gmin" or not tarball_fields or (
                r.get("gmin_snapshot") == GMIN_TARBALL_DATE
                and r.get("tarball_sha256") == GMIN_TARBALL_SHA256
            )
            hit, best = r["hit"] == "1", float(r["best"])
            evals, charged = int(r["evals_to_hit"]), int(r["charged"])
            wall, pot, outside = float(r["wall_s"]), float(r["pot_s"]), float(r["outside_s"])
            ok = (budget is not None and int(r["budget"]) == budget and r["hit"] in ("0", "1")
                  and revision_ok and r["harness_revision"] == GMIN_RERUN_REVISION and tarball_ok
                  and hit == energy_hits(best, ref, panel.HIT_TOL)
                  and (not hit or 0 < evals <= charged <= budget) and (hit or evals == charged == budget)
                  and 0.0 <= pot <= wall + 2e-6 and abs(max(wall - pot, 0.0) - outside) <= 2e-6)
            if not ok:
                bad.append(f"{shard}:{index}")
            rows.append(r)
    audit.cell(GMIN_RERUN, "row invariants and pinned revisions", " ".join(bad) or "ok", "ok")
    keys = [(r["system"], r["code"], r["seed"]) for r in rows]
    audit.cell(GMIN_RERUN, "each cluster, code and seed once", str(len(keys)), float(len(set(keys))))
    groups = {}
    for system in GMIN_RERUN_RAW_SYSTEMS:
        for code in ("anneal", "gmin"):
            group = sorted((r for r in rows if r["system"] == system and r["code"] == code), key=lambda r: int(r["seed"]))
            audit.cell(f"{GMIN_RERUN} {system} {code}", "seeds", " ".join(r["seed"] for r in group), PANEL_SEEDS)
            groups[(system, code)] = group
        audit.cell(f"{GMIN_RERUN} {system}", "seeds with this code first and with GMIN first", "6 6",
                   f"{first[system]['anneal']} {first[system]['gmin']}")
    audit.cell(GMIN_RERUN, "raw clusters", " ".join(GMIN_RERUN_RAW_SYSTEMS),
               " ".join(sorted({r["system"] for r in rows}, key=GMIN_RERUN_RAW_SYSTEMS.index)))
    ancestor_of(audit, GMIN_RERUN, GMIN_RERUN_REVISION)
    ancestor_of(audit, GMIN_RERUN, TEMPLATE_REMOVAL, GMIN_RERUN_REVISION)
    hosts = imported(audit, f"{GMIN_RERUN}/b", "host.csv", ("host_source", "host_blob"))
    host_rows = audit.rows(hosts)
    host_gmin = [r for r in host_rows if r["code"] == "gmin"]
    audit.record(all(r["snapshot"] == GMIN_TARBALL_DATE and r["tarball_sha256"] == GMIN_TARBALL_SHA256
                     for r in host_gmin), hosts, "shard b uses the dated GMIN tarball",
                 f"{GMIN_TARBALL_DATE} {GMIN_TARBALL_SHA256}",
                 f"{GMIN_TARBALL_DATE} {GMIN_TARBALL_SHA256}" if host_gmin and
                 all(r["snapshot"] == GMIN_TARBALL_DATE and r["tarball_sha256"] == GMIN_TARBALL_SHA256
                     for r in host_gmin) else "other")
    machine = {r["system"]: {x["machine_id_sha256"] for x in audit.rows(hosts) if x["system"] == r["system"]}
               for r in audit.rows(hosts)}
    for r in rows:
        for column in ("machine_id_sha256", "machine_id"):
            if column in r:
                machine.setdefault(r["system"], set()).add(r[column])
    ids = {system: " ".join(sorted(v)) for system, v in machine.items()}
    lj150 = "9eb296774121518feb1b187b0f6a109a42de7a5dd39e62ccd37d6e7d329604b8"
    main_host = "3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873"
    for system in GMIN_RERUN_RAW_SYSTEMS:
        audit.cell(f"{GMIN_RERUN} {system}", "machine id", lj150 if system == "lj150" else main_host, ids.get(system))
    audit.said(SI, f"LJ\\(_{{104}}\\), and M\\(_{{80}}\\) of the first block ran on\n"
                   f"\\texttt{{\\seqsplit{{{main_host}}}}},")
    audit.said(SI, f"LJ\\(_{{150}}\\) on\n\\texttt{{\\seqsplit{{{lj150}}}}},\nwhose record names no processor, memory, "
                   "or kernel;")
    audit.said(SI, "in alternating order by seed: six seeds with this code first and six with GMIN first.")
    rows_patch, rows_patch_commit, rows_patch_sha = gmin_rows_patch(audit)
    for shard in GMIN_SHARDS:
        commit = dict(line.split(" ", 1) for line in audit.text(f"{GMIN_RERUN}/{shard}/PROVENANCE").splitlines())["commit"]
        for patch in ("bench/lj104-lattice/v2/h2h-gmin.patch", "scripts/morse_h2h/morse-timer.patch",
                      "bench/metal-cluster-h2h/gmin-bgupta.patch",
                      "examples/h2h_gmin_lj.data", "examples/h2h_gmin_morse.data"):
            # The in-tree cap patch was replaced after these rows; they match the one rows_patch records.
            held = rows_patch_commit if patch == rows_patch else "HEAD"
            audit.cell(f"{GMIN_RERUN}/{shard}", f"{patch} at the record commit is the one at {held[:8]}",
                       git("rev-parse", f"{held}:{patch}").stdout.strip(),
                       git("rev-parse", f"{commit}:{patch}").stdout.strip())
    printed = git_bytes("show", f"{rows_patch_commit}:{rows_patch}")
    audit.cell(GMIN_SOURCE_NOTE, "sha256 of rows_patch at its commit", rows_patch_sha,
               hashlib.sha256(printed).hexdigest())
    audit.said(SI, f"At both revisions the cap patch has sha256\n\\texttt{{\\seqsplit{{{rows_patch_sha}}}}}.")
    row_ids = {r["code_revision"] for r in rows if r["code"] == "gmin"}
    audit.cell(GMIN_RERUN, "one historical GMIN row identifier", "1", float(len(row_ids)))
    dates = {r.get("gmin_snapshot", "") for r in rows if r["code"] == "gmin" and r.get("gmin_snapshot")}
    dates.update(r["snapshot"] for r in host_gmin)
    audit.cell(GMIN_RERUN, "GMIN public tarball date", GMIN_TARBALL_DATE, " ".join(sorted(dates)))
    tarballs = {r["tarball_sha256"] for r in rows if r.get("tarball_sha256")}
    tarballs.update(r["tarball_sha256"] for r in host_gmin)
    audit.cell(GMIN_RERUN, "GMIN tarball sha256", GMIN_TARBALL_SHA256, " ".join(sorted(tarballs)))
    audit.said(SI, "The Lennard--Jones, Morse, and Gupta rows whose source fields include a\n"
                   f"tarball pin record the public tarball dated {GMIN_TARBALL_DATE}, sha256\n"
                   f"\\texttt{{\\seqsplit{{{GMIN_TARBALL_SHA256}}}}},")
    return panel, groups


def gmin_develop_groups(audit: Audit, panel: object) -> dict[tuple[str, str], list[dict[str, str]]]:
    """The imported session of the submitted revision on gold and Morse, checked row by row."""
    path = imported(audit, GMIN_DEVELOP)
    fields = dict(line.split(" ", 1) for line in audit.text(f"{GMIN_DEVELOP}/PROVENANCE").splitlines())
    audit.cell(f"{GMIN_DEVELOP}/PROVENANCE", "ancestor", SUBMITTED, fields["ancestor"])
    ancestor_of(audit, GMIN_DEVELOP, SUBMITTED)
    ancestor_of(audit, GMIN_DEVELOP, TEMPLATE_REMOVAL, SUBMITTED)
    ancestor_of(audit, GMIN_DEVELOP, SUBMITTED, GMIN_DEVELOP_HARNESS)
    ancestor_of(audit, GMIN_DEVELOP, GMIN_DEVELOP_HARNESS, fields["commit"])
    changed = git("diff", "--name-only", SUBMITTED, GMIN_DEVELOP_HARNESS).stdout.split()
    outside = [f for f in changed if not f.startswith(("scripts/paper_rerun/tuned_gmin/", "bench/paper-rerun/tuned-gmin/"))]
    audit.cell(GMIN_DEVELOP, "files the harness commit changes outside its own folders", "scripts/gmin_panel/build_gmin.sh",
               " ".join(outside))
    build = git("diff", "-U0", SUBMITTED, GMIN_DEVELOP_HARNESS, "--", "scripts/gmin_panel/build_gmin.sh").stdout
    edits = [x for x in build.splitlines() if x[:1] in "+-" and not x.startswith(("+++", "---"))]
    audit.cell(GMIN_DEVELOP, "the build script change", '-    make -j"$(nproc)" GMIN | +    make -j"${GMIN_PANEL_JOBS:-$(nproc)}" GMIN',
               " | ".join(edits))
    references = {row[0]: row[3] for row in panel.SYSTEMS}
    rows, bad, first, seen = audit.rows(path), [], defaultdict(lambda: defaultdict(int)), set()
    for index, r in enumerate(rows, start=2):
        if (r["system"], r["seed"]) not in seen:
            first[r["system"]][r["code"]] += 1
            seen.add((r["system"], r["seed"]))
        budget, ref = panel.PAPER_BUDGETS.get(r["system"]), references.get(r["system"])
        revision_ok = r["code_revision"] == SUBMITTED if r["code"] == "anneal" \
            else r["code_revision"] == imported_gmin_row_id(audit)
        hit, best = r["hit"] == "1", float(r["best"])
        evals, charged = int(r["evals_to_hit"]), int(r["charged"])
        wall, pot, outside_s = float(r["wall_s"]), float(r["pot_s"]), float(r["outside_s"])
        ok = (budget is not None and int(r["budget"]) == budget and r["hit"] in ("0", "1")
              and revision_ok and r["harness_revision"] == GMIN_DEVELOP_HARNESS
              and r["host"] == GMIN_DEVELOP_HOST and r["session"] == GMIN_DEVELOP_SESSION
              and hit == energy_hits(best, ref, panel.HIT_TOL)
              and (not hit or 0 < evals <= charged <= budget) and (hit or evals == charged == budget)
              and 0.0 <= pot <= wall + 2e-6 and abs(max(wall - pot, 0.0) - outside_s) <= 2e-6)
        if not ok:
            bad.append(str(index))
    audit.cell(GMIN_DEVELOP, "row invariants, pinned revisions, host and session", " ".join(bad) or "ok", "ok")
    keys = [(r["system"], r["code"], r["seed"]) for r in rows]
    audit.cell(GMIN_DEVELOP, "each cluster, code and seed once", str(len(keys)), float(len(set(keys))))
    audit.cell(GMIN_DEVELOP, "clusters", " ".join(GMIN_DEVELOP_SYSTEMS),
               " ".join(sorted({r["system"] for r in rows}, key=lambda s: GMIN_DEVELOP_SYSTEMS.index(s)
                               if s in GMIN_DEVELOP_SYSTEMS else 99)))
    groups = {}
    for system in GMIN_DEVELOP_SYSTEMS:
        for code in ("anneal", "gmin"):
            group = sorted((r for r in rows if r["system"] == system and r["code"] == code), key=lambda r: int(r["seed"]))
            audit.cell(f"{GMIN_DEVELOP} {system} {code}", "seeds", " ".join(r["seed"] for r in group), PANEL_SEEDS)
            groups[(system, code)] = group
        audit.cell(f"{GMIN_DEVELOP} {system}", "seeds with this code first and with GMIN first", "6 6",
                   f"{first[system]['anneal']} {first[system]['gmin']}")
    for name in ("bench/lj104-lattice/v2/h2h-gmin.patch", "scripts/morse_h2h/morse-timer.patch",
                 "bench/metal-cluster-h2h/gmin-bgupta.patch", "examples/h2h_gmin_lj.data",
                 "examples/h2h_gmin_morse.data", "scripts/gmin_panel/run_panel.py", "examples/h2h_metal.rs",
                 "examples/h2h_morse.rs"):
        audit.cell(GMIN_DEVELOP, f"{name} at the harness commit is the one of the submitted revision",
                   git("rev-parse", f"{SUBMITTED}:{name}").stdout.strip(),
                   git("rev-parse", f"{GMIN_DEVELOP_HARNESS}:{name}").stdout.strip())
    runner = git("show", f"{GMIN_DEVELOP_HARNESS}:scripts/paper_rerun/tuned_gmin/run.py").stdout
    for needle in ('panel.gmin_data(system, seed, lj_data, morse_data)', 'if key.startswith(("H2H_", "ANNEAL_")):'):
        audit.record(needle in runner, GMIN_DEVELOP, f"the harness runs {needle}", "present",
                     "present" if needle in runner else "absent")
    metal = git("show", f"{SUBMITTED}:examples/h2h_metal.rs").stdout
    audit.record('cfg.basin_stride = std::env::var("H2H_BASIN_STRIDE")' in metal and ".unwrap_or(8)" in metal,
                 GMIN_DEVELOP, "the gold driver of the submitted revision uses a basin stride of 8", "8",
                 "8" if ".unwrap_or(8)" in metal else "other")
    return groups


def gmin_develop_data(audit: Audit, panel: object) -> None:
    """The gold input of the second block, from the panel script of the submitted revision."""
    data = panel.gmin_data("au55", 2000, "", "")
    for line in ("BGUPTAT 55 0.2061 10.229 4.036 1.79 2.884", "TEMPERATURE 0.1", "NEWRESTART 40", "EDIFF 0.003",
                 f"STEP {0.40 * 2.884 / 2 ** (1 / 6):.6f} 0.51", f"AVOID {4.0 * 2.884 / 2 ** (1 / 6):.6f} 20 F",
                 f"RADIUS {panel.gmin_gupta_radius(55):.6f}"):
        audit.record(line in data, "scripts/gmin_panel/run_panel.py", f"the Au55 GMIN input sets {line}", line,
                     "set" if line in data else "absent")
    audit.said(SI, "The Au\\(_{55}\\) row runs GMIN's Gupta potential with the gold row of Darby et al., "
                   "\\(A=0.2061\\)~eV, \\(p=10.229\\), \\(q=4.036\\), \\(\\xi=1.79\\)~eV, and \\(r_0=2.884\\)~\\AA, at "
                   "\\(T=0.1\\)~eV, with step 0.40 and taboo radius 4.0 multiplied by \\(r_0/2^{1/6}\\), GMIN's "
                   "container radius, \\texttt{NEWRESTART 40}, and an energy difference of 0.003.")
    audit.said(TEX, "On Au\\(_{55}\\) GMIN runs the Gupta potential with the gold parameters of Darby et al.\\ "
                    "\\cite{darbyCuAu2002} at \\(T=0.1\\)~eV, with the step and the taboo radius scaled by "
                    "\\(r_0/2^{1/6}\\) and GMIN's container radius, and the gold driver of this code applies the basin "
                    "bias of equation~\\ref{eq:delta} on a stride of 8.")
    audit.record("H2H_" not in audit.text(TEX) and "h2h\\_" not in audit.text(TEX),
                 TEX, "driver switches stay in the Supporting Information", "absent",
                 "absent" if "H2H_" not in audit.text(TEX) and "h2h\\_" not in audit.text(TEX) else "present")
    refs = {row[0]: row[3] for row in panel.SYSTEMS}
    audit.said(SI, f"\\({refs['au55']:.8f}\\)~eV at Au\\(_{{55}}\\), \\({refs['m55r6']:.6f}\\) and "
                   f"\\({refs['m55r14']:.6f}\\) at M\\(_{{55}}\\) (\\(\\rho=6\\) and 14), and \\({refs['m80r6']:.6f}\\) and "
                   f"\\({refs['m80r14']:.6f}\\) at M\\(_{{80}}\\) (\\(\\rho=6\\) and 14).")
    audit.said(SI, f"and every cluster of the second block on\n\\texttt{{\\seqsplit{{{GMIN_DEVELOP_HOST}}}}},\nwhose rows "
                   "name an Intel Xeon processor and no memory or kernel.")
    audit.said(SI, "and for the second by the same script at the record commit \\texttt{f009acc9} on the submitted "
                   "revision, which only sets the number of parallel build jobs,")
    audit.said(SI, f"The rows of this code in the second block are the submitted revision "
                   f"\\texttt{{\\seqsplit{{{SUBMITTED}}}}}, session {GMIN_DEVELOP_SESSION} of "
                   "\\path{paper/jctc/data/gmin_develop}, copied from the record branch its \\texttt{PROVENANCE} file "
                   "names.")
    audit.said(SI, "The GMIN tables of the main text use four GMIN inputs.")


def gmin_develop_prose(audit: Audit, panel: object, groups: dict) -> None:
    """The descriptive results of the second block, from its rows."""
    paired = {s: gmin_paired(groups, s) for s in GMIN_DEVELOP_SYSTEMS}
    t = {s: paired[s]["table"] for s in GMIN_DEVELOP_SYSTEMS}
    hits = {(s, a): hits_of(groups[(s, a)]) for s in GMIN_DEVELOP_SYSTEMS for a in ("anneal", "gmin")}
    audit.cell(TEX, "GMIN hits on M55 and M80 at rho 14", "0 0", f"{hits[('m55r14', 'gmin')]} {hits[('m80r14', 'gmin')]}")
    audit.said(TEX, f"At the submitted revision, the hit counts for this code and GMIN are "
                    f"{hits[('m55r14', 'anneal')]} and {hits[('m55r14', 'gmin')]} on M\\(_{{55}}\\) at "
                    f"\\(\\rho=14\\), {hits[('m80r14', 'anneal')]} and {hits[('m80r14', 'gmin')]} on "
                    f"M\\(_{{80}}\\) at \\(\\rho=14\\), and {hits[('au55', 'anneal')]} and "
                    f"{hits[('au55', 'gmin')]} on Au\\(_{{55}}\\).")
    audit.cell(TEX, "both arms hit M55 and M80 at rho 6 on every seed", "12 0 0 0 12 0 0 0",
               " ".join(map(str, t["m55r6"] + t["m80r6"])))
    calls = median([float(r["evals_to_hit"]) for r in groups[("m80r6", "anneal")]])
    audit.said(TEX, "Both arms hit M\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed, and on M\\(_{80}\\) at "
                    f"\\(\\rho=6\\) this code takes the same median of {calls:.0f} combined\n"
                    "energy-and-gradient passes at both revisions.")
    audit.said(TEX, "Table~\\ref{tab:gmin} tests whether the template-free search reaches minima\n"
                    "that trap that walk, on Lennard-Jones, Morse, and Gupta clusters.")
    audit.said(TEX, "Each block runs both arms of each cluster on seeds 2000--2011 in alternating order, at the "
                    "numeric ceiling in the table, from the start this code writes for the seed.")
    audit.said(COVER, "At arm-specific work definitions the search reaches Lennard-Jones, Morse,\n"
                      "and rigid-water minima that trap the corresponding GMIN arms.")


def gmin_rerun(audit: Audit) -> None:
    """tab:gmin and its prose, recomputed from the imported eeec8e2c shards and the session of the submitted revision."""
    panel, groups = gmin_rerun_groups(audit)
    develop = gmin_develop_groups(audit, panel)
    by_label = {labels[1]: system for system, labels in panel.LABELS.items()}
    shown, system, block, heads = [], None, None, []
    for row in tex_table(audit, TEX, "tab:gmin"):
        cells = [c.strip() for c in row]
        if "\\multicolumn{12}" in cells[0]:
            heads.append(cells[0].replace("\\midrule", "").strip())
            block = "develop" if "Submitted revision" in cells[0] else "rerun"
            system = None
            continue
        if cells[0]:
            system = by_label.get(cells[0])
        arm = {"this work": "anneal", "GMIN": "gmin"}.get(cells[2])
        where = f"{TEX} tab:gmin {block} {system} {cells[2]}"
        source = develop if block == "develop" else groups
        if block is None or system is None or arm is None or len(cells) != 12 or (system, arm) not in source:
            audit.record(False, where, "table line", " & ".join(cells), "a rerun cluster and arm")
            continue
        shown.append((block, system, arm))
        gmin_tex_line(audit, where, cells[1:], source, system, arm, panel.PAPER_BUDGETS[system])
    audit.cell(TEX, "tab:gmin block headings",
               "\\multicolumn{12}{@{}l}{\\emph{Template-free predecessor}} | "
               "\\multicolumn{12}{@{}l}{\\emph{Submitted revision}}", " | ".join(heads))
    table_text = audit.text(TEX).split("% gmin-panel table: start", 1)[1].split("% gmin-panel table: end", 1)[0]
    cross_arm = re.search(r"p_\\mathrm\{(?:M|calls|wall)\}|Newcombe|\\Delta", table_text)
    audit.record(cross_arm is None, TEX, "tab:gmin reports no cross-arm tests", "absent",
                 "absent" if cross_arm is None else "present")
    audit.said(TEX, "The numeric ceilings in Table~\\ref{tab:gmin} use different units: one\n"
                    "combined energy-and-gradient pass for this code and one\n"
                    "\\texttt{POTENTIAL} entry for GMIN, including entries with\n"
                    "\\texttt{GRADT=false}. The committed GMIN rows do not separate entries by\n"
                    "gradient request, so no conversion or cross-arm hit, work, or wall test is\n"
                    "reported.")
    want = ([("rerun", s, a) for s in GMIN_RERUN_SYSTEMS for a in ("anneal", "gmin")]
            + [("develop", s, a) for s in GMIN_DEVELOP_SYSTEMS for a in ("anneal", "gmin")])
    audit.cell(TEX, "tab:gmin lines", " ".join("/".join(x) for x in shown), " ".join("/".join(x) for x in want))
    lj75_rows = groups[("lj75", "anneal")] + groups[("lj75", "gmin")]
    audit.cell(GMIN_RERUN, "raw LJ75 rows printed", "24", float(len(lj75_rows)))
    audit.record("LJ\\(_{75}\\)" in table_text and any(s == "lj75" for _, s, _ in shown),
                 TEX, "tab:gmin prints the committed LJ75 pair", "present",
                 "present" if "LJ\\(_{75}\\)" in table_text else "absent")
    gmin_develop_data(audit, panel)
    gmin_develop_prose(audit, panel, develop)
    def hits(system: str, arm: str) -> int:
        return hits_of(groups[(system, arm)])

    def walls(system: str, arm: str) -> float:
        return median([float(r["wall_s"]) for r in groups[(system, arm)]])

    audit.said(TEX, f"At those arm-specific ceilings, this code hits {hits('lj98', 'anneal')} of 12 seeds on\n"
                    f"LJ\\(_{{98}}\\) and {hits('lj75', 'anneal')} of 12 on LJ\\(_{{75}}\\), while GMIN hits neither; on\n"
                    f"LJ\\(_{{38}}\\), the counts are "
                    f"{hits('lj38', 'anneal')} and {hits('lj38', 'gmin')}.")
    big = ("lj102", "lj103", "lj104", "lj150")
    audit.said(TEX, "On LJ\\(_{102}\\), LJ\\(_{103}\\), LJ\\(_{104}\\), and LJ\\(_{150}\\), the hit\n"
                    "counts are " + series(f"{hits(s, 'anneal')} against {hits(s, 'gmin')}" for s in big) + ".")
    audit.cell(TEX, "both arms hit LJ55 and M80 at rho 6 on every seed", "12 12 12 12",
               f"{hits('lj55', 'anneal')} {hits('lj55', 'gmin')} "
               f"{hits('m80r6', 'anneal')} {hits('m80r6', 'gmin')}")
    audit.said(TEX, "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed.")
    lower = [s for s in GMIN_RERUN_SYSTEMS if walls(s, "anneal") < walls(s, "gmin")]
    audit.cell(TEX, "printed clusters where the median wall of this code is lower", str(len(GMIN_RERUN_SYSTEMS)),
               float(len(lower)))
    low_wall, low_system = min((walls(s, "anneal"), s) for s in GMIN_RERUN_SYSTEMS)
    high_wall, high_system = max((walls(s, "anneal"), s) for s in GMIN_RERUN_SYSTEMS)
    audit.said(TEX, f"The median wall of a run is lower for this code on all {NUMBER_WORDS[len(lower)]} clusters, "
                    f"ranging from {panel.show_s(low_wall)}~s on {panel.LABELS[low_system][1]} to "
                    f"{panel.show_s(high_wall)}~s on {panel.LABELS[high_system][1]}.")
    audit.said(TEX, "These arm-specific wall medians are descriptive and are not tested across\n"
                    "arms.")
    lj_data = audit.text("examples/h2h_gmin_lj.data")
    for line in ("TEMPERATURE 1.0", "STEP 0.40 0.51", "NEWRESTART 40", "AVOID 4.0 20 F", "EDIFF 0.003"):
        audit.record(line in lj_data and line in audit.text("examples/h2h_gmin_morse.data"), "examples/h2h_gmin_lj.data",
                     f"both GMIN inputs set {line}", line, "set" if line in lj_data else "absent")
    audit.said(TEX, "file: \\(T=1.0\\), step \\(0.40\\), a taboo list of the last \\(20\\) minima within \\(4.0\\), and a "
                    "restart after \\(40\\) steps without a new lowest minimum; the Morse rows add GMIN's native "
                    "\\texttt{MORSE} keyword and \\(\\rho\\).")
    for path in (TEX, SI):
        part = audit.text(path)
        if path == TEX:
            part = part.split("\\label{sec:protocol}", 1)[1].split("\\label{sec:transfer}", 1)[0]
        else:
            part = part.split("\\label{sec:si-gmin}", 1)[1].split("\\section{Host and commands}", 1)[0]
        for word in ("1ce3ab64", "c2de652d", "prior-on", "Fisher", "Mann--Whitney"):
            audit.record(word not in part, path, f"the GMIN comparison does not print {word}", "absent",
                         "absent" if word not in part else "present")
    for path in (TEX, SI):
        audit.record("Mann--Whitney" not in audit.text(path), path, "no Mann--Whitney test is reported", "absent",
                     "absent" if "Mann--Whitney" not in audit.text(path) else "present")
    manuscript = audit.text(TEX)
    gmin_results = manuscript.split("\\label{sec:protocol}", 1)[1].split("\\label{sec:ase}", 1)[0]
    for marker in ("gmin-panel", "tip4p-gmin"):
        start, end = f"% {marker} table: start", f"% {marker} table: end"
        before, rest = gmin_results.split(start, 1)
        _, after = rest.split(end, 1)
        gmin_results = before + after
    hit_test = re.search(r"\\\(p(?:_|=)|McNemar|Newcombe", gmin_results)
    audit.record(hit_test is None, TEX, "the GMIN results report no cross-arm hit test", "absent",
                 "absent" if hit_test is None else hit_test.group(0))
    cover_gmin = [sentence for sentence in re.split(r"(?<=[.!?])\s+", collapse(audit.text(COVER)))
                  if "GMIN" in sentence]
    cover_test = next((sentence for sentence in cover_gmin
                       if re.search(r"\\\(p(?:_|=)|McNemar|Newcombe|hits more", sentence)), None)
    audit.record(cover_test is None, COVER, "the GMIN cover prose reports no cross-arm hit test", "absent",
                 "absent" if cover_test is None else cover_test)
    gmin_patches(audit)


def series(items) -> str:
    """A list with the serial comma: ``a``, ``a and b``, ``a, b, and c``."""
    items = list(items)
    return " and ".join(items) if len(items) <= 2 else ", ".join(items[:-1]) + ", and " + items[-1]


def morse_pair(rho: float, r: float) -> float:
    """The Doye-Wales Morse pair with unit well depth and unit pair minimum."""
    e = math.exp(rho * (1.0 - r))
    return e * (e - 2.0)


KNOCKOUT_DATA = "paper/jctc/data/knockouts"
# The commits whose code each arm of a knockout block ran, with and without the switch.
KNOCKOUT_CODE = {
    "templates": ("628773d6837ea2d959a27256ff4163b21162fd4d", "1c1b12384683485cf820d3a244151b48973dc0f7"),
    "pairscale": (SUBMITTED, SUBMITTED),
    "move-a": (SUBMITTED, SUBMITTED),
}
# How the rows of each block name the revision of an arm.
KNOCKOUT_REVISION = {"templates": ("revision", "628773d6", "1c1b1238"),
                     "pairscale": ("revision", "b12293b9", "b12293b9"),
                     "move-a": ("revision", "b12293b9", "b12293b9")}
PAIRSCALE_TREE = "b12293b9d637dd32417be3aa482302e0f30dd594"
SEARCH_SOURCE = ("src", "examples", "Cargo.toml", "Cargo.lock")
# The files the default-off knockout switches add to the search source of the submitted revision.
KNOCKOUT_SWITCH_FILES = ("examples/h2h_gmin_proof.rs", "examples/h2h_metal.rs", "examples/h2h_morse.rs",
                         "src/methods/cluster_hopping.rs", "src/methods/cluster_hopping/config.rs")
WITHDRAWN_KNOCKOUTS = ("priors", "restart", "pair-scale")


def interval_of(cell: str) -> tuple[str, str]:
    low, _, high = cell.replace("\\(-\\)", "-").partition(" to ")
    return low.strip(), high.strip()


def knockout_blocks(audit: Audit) -> list[tuple[str, list[list[str]]]]:
    """The heading and the lines of each block of tab:knockouts."""
    blocks: list[tuple[str, list[list[str]]]] = []
    for cells in tex_table(audit, TEX, "tab:knockouts"):
        cells = [c.replace("\\midrule", "").strip() for c in cells]
        if cells[0].startswith("\\multicolumn"):
            blocks.append((re.search(r"\\emph\{(.*)\}\}", cells[0]).group(1), []))
        elif blocks:
            blocks[-1][1].append(cells)
    return blocks


def knockout_cells(audit: Audit, knock: object, knockout: object) -> dict[str, dict]:
    """Each system of a block recomputed from its imported rows: budget, seeds, the paired table, p, interval."""
    path = imported(audit, f"{KNOCKOUT_DATA}/{knockout.key}")
    column, with_rev, without_rev = KNOCKOUT_REVISION[knockout.key]
    arms: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    twice = []
    for r in audit.rows(path):
        if not knockout.keep(r) or r[knockout.column] not in (knockout.with_arm, knockout.without_arm):
            continue
        if r["seed"] in arms[(r["system"], r[knockout.column])]:
            twice.append(f"{r['system']}/{r['seed']}")
        arms[(r["system"], r[knockout.column])][r["seed"]] = r
        want = with_rev if r[knockout.column] == knockout.with_arm else without_rev
        if not r[column].startswith(want):
            twice.append(f"{r['system']}/{r['seed']} at {r[column][:8]}")
    audit.cell(path, f"{knockout.key}: rows at the block revisions, one per system, arm, and seed", "none",
               " ".join(twice) or "none")
    out = {}
    for system in sorted({s for s, _ in arms}):
        first, second = arms.get((system, knockout.with_arm), {}), arms.get((system, knockout.without_arm), {})
        if not first or not second:
            continue
        if sorted(first) != sorted(second):
            audit.record(False, path, f"{knockout.key} {system}: both arms ran the same seeds", "same", "different")
            continue
        hit = knockout.hit
        table = paired_table({k: first[k][hit].strip().lower() in ("1", "true") for k in first},
                             {k: second[k][hit].strip().lower() in ("1", "true") for k in second})
        budgets = {r["budget"] for r in (*first.values(), *second.values())}
        out[system] = {"budget": budgets.pop() if len(budgets) == 1 else "mixed", "n": len(first), "table": table,
                       "p": mcnemar(table[1], table[2]), "interval": newcombe_paired_interval(*table)}
    return out


MOVE_A_DRIVERS = {"lj38": "h2h_gmin_proof", "lj55": "h2h_gmin_proof", "lj75": "h2h_gmin_proof", "au55": "h2h_metal"}


def knockout_move_a(audit: Audit, cells: dict[str, dict]) -> None:
    """The move-choice block: its record, the tree and drivers its rows ran, the order, and its prose."""
    folder = f"{KNOCKOUT_DATA}/move-a"
    fields = dict(line.split(" ", 1) for line in audit.text(f"{folder}/PROVENANCE").splitlines())
    audit.cell(f"{folder}/PROVENANCE", "tree of the record commit", fields.get("tree"),
               git("rev-parse", f"{fields['commit']}^{{tree}}").stdout.strip())
    audit.cell(f"{folder}/PROVENANCE", "the rows ran the tree two commits below the record commit", PAIRSCALE_TREE,
               git("rev-parse", f"{fields['commit']}~2^{{tree}}").stdout.strip())
    ancestor_of(audit, f"{folder}/PROVENANCE", fields.get("ancestor", ""))
    ancestor_of(audit, f"{folder}/PROVENANCE", fields.get("ancestor", ""), fields["commit"])
    rows = audit.rows(f"{folder}/rows.csv")
    wrong = sorted({r["system"] for r in rows if r["driver_blob"] != git(
        "rev-parse", f"{PAIRSCALE_TREE}:examples/{MOVE_A_DRIVERS.get(r['system'], 'h2h_morse')}.rs").stdout.strip()})
    audit.cell(f"{folder}/rows.csv", "every row ran the driver of the tree the rows name for its cluster", "none",
               " ".join(wrong) or "none")
    audit.cell(f"{folder}/rows.csv", "every row names the tree", PAIRSCALE_TREE,
               " ".join(sorted({r["revision"] for r in rows})))
    audit.cell(f"{folder}/rows.csv", "machines", "1", float(len({r["machine_id"] for r in rows})))
    audit.cell(f"{folder}/rows.csv", "arms", "H2H_KNOCKOUT=move_choice stock", " ".join(sorted({r["arm"] for r in rows})))
    first: dict[tuple[str, str], str] = {}
    for r in rows:
        first.setdefault((r["system"], r["seed"]), r["arm"])
    order = [f"{s}/{seed}" for (s, seed), arm in first.items()
             if arm != ("stock" if int(seed) % 2 == 0 else "H2H_KNOCKOUT=move_choice")]
    audit.cell(f"{folder}/rows.csv", "even seeds stock first, odd seeds uniform first", "none", " ".join(order) or "none")

    def show_signed(x: float) -> str:
        return f"\\(-\\){abs(x):.2f}" if x < 0 else f"{x:.2f}"

    pooled = [0, 0, 0, 0]
    for c in cells.values():
        pooled = [a + b for a, b in zip(pooled, c["table"])]
    pooled_p = mcnemar(pooled[1], pooled[2])
    low, high = newcombe_paired_interval(*pooled)
    n = sum(c["n"] for c in cells.values())
    audit.cell(TEX, "move choice: clusters", "8", float(len(cells)))
    audit.cell(TEX, "move choice: clusters whose McNemar p is not 1", "none",
               " ".join(s for s, c in cells.items() if c["p"] != 1.0) or "none")
    audit.said(TEX, "Drawing each move from equation~\\ref{eq:uniform} changes the hits on no cluster at \\(p<0.05\\), "
                    "with a McNemar \\(p\\) of 1 on each of the eight.")
    audit.said(TEX, f"Pooled over the eight clusters, the preset hits {pooled[0] + pooled[1]} of {n} seeds and the "
                    f"uniform draw {pooled[0] + pooled[2]} (pooled McNemar \\(p={pp(pooled_p)}\\), interval "
                    f"{show_signed(low)} to {show_signed(high)}).")
    audit.said(TEX, "At 12 seeds, neither rescaling the preset lengths nor replacing adaptive\n"
                    "move choice by a uniform draw produces a detectable hit-rate change in the\n"
                    "tested panels.")
    audit.said(COVER, "At 12 seeds, neither rescaling the preset lengths nor replacing adaptive\n"
                      "move choice by a uniform draw produces a detectable hit-rate change in the\n"
                      "tested panels.")
    audit.said(SI, "\\texttt{\\seqsplit{50c532253f3ad11e6457a119bb603f05b4f0be79}} of \\path{record/knockout-move-a}, on "
                   "the same develop commit.")
    audit.said(SI, "even seeds run the preset first and odd seeds the uniform draw first.")
    audit.record("the choice among the moves, the" not in collapse(audit.text(TEX)), TEX,
                 "the move choice is not listed among the parts with no run", "absent",
                 "absent" if "the choice among the moves, the" not in collapse(audit.text(TEX)) else "present")


def knockout_pairscale(audit: Audit, cells: dict[str, dict]) -> None:
    """The length-unit block: its record, the tree its rows ran, the driver, the order, and its prose."""
    folder = f"{KNOCKOUT_DATA}/pairscale"
    fields = dict(line.split(" ", 1) for line in audit.text(f"{folder}/PROVENANCE").splitlines())
    audit.cell(f"{folder}/PROVENANCE", "tree of the record commit", fields.get("tree"),
               git("rev-parse", f"{fields['commit']}^{{tree}}").stdout.strip())
    audit.cell(f"{folder}/PROVENANCE", "the rows ran the tree of the record commit's parent", PAIRSCALE_TREE,
               git("rev-parse", f"{fields['commit']}^^{{tree}}").stdout.strip())
    ancestor_of(audit, f"{folder}/PROVENANCE", fields.get("ancestor", ""))
    ancestor_of(audit, f"{folder}/PROVENANCE", fields.get("ancestor", ""), fields["commit"])
    changed = git("diff", "--name-only", SUBMITTED, PAIRSCALE_TREE, "--", *SEARCH_SOURCE).stdout.split()
    audit.cell(folder, "the rows' tree is the submitted revision's search source with the knockout switches",
               " ".join(KNOCKOUT_SWITCH_FILES), " ".join(sorted(changed)))
    rows = audit.rows(f"{folder}/rows.csv")
    driver = git("rev-parse", f"{PAIRSCALE_TREE}:examples/h2h_morse.rs").stdout.strip()
    audit.cell(f"{folder}/rows.csv", "every row ran the Morse driver of the tree the rows name", driver,
               " ".join(sorted({r["driver_blob"] for r in rows})))
    audit.cell(f"{folder}/rows.csv", "every row names the tree", PAIRSCALE_TREE,
               " ".join(sorted({r["revision"] for r in rows})))
    first: dict[tuple[str, str], str] = {}
    for r in rows:
        first.setdefault((r["system"], r["seed"]), r["arm"])
    order = [f"{s}/{seed}" for (s, seed), arm in first.items()
             if arm != ("stock" if int(seed) % 2 == 0 else "H2H_PAIR_SCALE=1")]
    audit.cell(f"{folder}/rows.csv", "even seeds stock first, odd seeds rescaled first", "none", " ".join(order) or "none")

    def hits(c: dict) -> tuple[int, int]:
        both, only_with, only_without, _ = c["table"]
        return both + only_with, both + only_without

    def show_signed(x: float) -> str:
        return f"\\(-\\){abs(x):.2f}" if x < 0 else f"{x:.2f}"

    pooled = [0, 0, 0, 0]
    for c in cells.values():
        pooled = [a + b for a, b in zip(pooled, c["table"])]
    pooled_p = mcnemar(pooled[1], pooled[2])
    low, high = newcombe_paired_interval(*pooled)
    n = sum(c["n"] for c in cells.values())
    below = [s for s, c in cells.items() if c["p"] < 0.05]
    audit.cell(TEX, "length unit: clusters whose arms differ at p<0.05", "none", " ".join(below) or "none")
    audit.record(pooled_p >= 0.05, TEX, "length unit: the pooled arms do not differ at p<0.05", ">= 0.05", pp(pooled_p))
    m55, m80 = cells["m55r14"], cells["m80r14"]
    audit.cell(TEX, "length unit: both arms hit M55 and M80 at rho 6 on every seed", "12 12 12 12",
               " ".join(str(x) for s in ("m55r6", "m80r6") for x in hits(cells[s])))
    audit.said(TEX, f"At \\(\\rho=14\\) the preset in the Lennard-Jones diameter hits M\\(_{{55}}\\) on "
                    f"{hits(m55)[0]} of 12 seeds against {hits(m55)[1]} (\\(p={pp(m55['p'])}\\)) and M\\(_{{80}}\\) on "
                    f"{hits(m80)[0]} against {hits(m80)[1]} (\\(p={pp(m80['p'])}\\)), and pooled over the four clusters "
                    f"it hits {pooled[0] + pooled[1]} of {n} against {pooled[0] + pooled[2]} (pooled McNemar "
                    f"\\(p={pp(pooled_p)}\\), interval {show_signed(low)} to {show_signed(high)}).")
    audit.said(TEX, f"pooled over the four clusters it hits {pooled[0] + pooled[1]} of {n} against "
                    f"{pooled[0] + pooled[2]}\n"
                    f"(pooled McNemar \\(p={pp(pooled_p)}\\), interval {show_signed(low)} to {show_signed(high)}).")
    audit.said(TEX, "Rescaling the preset lengths to the Morse pair minimum changes the hits on no cluster at "
                    "\\(p<0.05\\).")
    audit.said(TEX, "Table~\\ref{tab:knockouts} gives three paired on/off tests: the former\n"
                    "structure templates, the preset length unit, and adaptive move choice.")
    audit.said(SI, "\\texttt{\\seqsplit{5c37c840ed2e9f9a2d881105363b33762b5cca31}} of "
                   "\\path{record/knockout-pairscale}, on develop "
                   "\\texttt{\\seqsplit{dd6ff9273be30ab317bfd5508596f4beb19cb3a4}}.")
    audit.said(SI, "even seeds run the Lennard-Jones diameter first and odd seeds the Morse minimum first.")
    audit.record("the length unit, the" not in collapse(audit.text(TEX)), TEX,
                 "the length unit is not listed among the parts with no run", "absent",
                 "absent" if "the length unit, the" not in collapse(audit.text(TEX)) else "present")


def manuscript_knockouts(audit: Audit) -> None:
    """tab:knockouts and tab:tip4p-gmin recomputed from the imported rows, the LJ75 rule, and their prose."""
    knock = load_script("knockout_tables_audit", "paper/jctc/scripts/knockout_tables.py")
    blocks = knockout_blocks(audit)
    audit.cell(TEX, "knockout blocks", " | ".join(k.title for k in knock.KNOCKOUTS), " | ".join(t for t, _ in blocks))
    cells_of = {}
    for knockout, (title, lines) in zip(knock.KNOCKOUTS, blocks):
        cells = knockout_cells(audit, knock, knockout)
        cells_of[knockout.key] = cells
        by_name = {knock.name(system): system for system in cells}
        printed = []
        for line in lines:
            system = by_name.get(line[0])
            where = f"{TEX} tab:knockouts {knockout.key} {line[0]}"
            if system is None or len(line) != 9:
                audit.record(False, where, "a system the rows hold", " & ".join(line), "not in the rows")
                continue
            printed.append(system)
            c = cells[system]
            both, only_with, only_without, _ = c["table"]
            audit.cell(where, "budget", line[1], float(c["budget"]) if c["budget"] != "mixed" else "mixed")
            audit.cell(where, "seeds", line[2], float(c["n"]))
            audit.cell(where, "with", line[3], f"{both + only_with}/{c['n']}")
            audit.cell(where, "without", line[4], f"{both + only_without}/{c['n']}")
            audit.cell(where, "only with", line[5], float(only_with))
            audit.cell(where, "only without", line[6], float(only_without))
            audit.cell(where, "exact McNemar p", line[7], c["p"])
            low, high = interval_of(line[8])
            audit.cell(where, "Newcombe lower", low, c["interval"][0])
            audit.cell(where, "Newcombe upper", high, c["interval"][1])
        code = KNOCKOUT_CODE[knockout.key]
        ancestors = all(git("merge-base", "--is-ancestor", rev, SUBMITTED).returncode == 0 for rev in code)
        audit.record(("lj75" in printed) == (ancestors and "lj75" in cells), TEX,
                     f"{knockout.key}: LJ75 is printed only where both arms ran the code of ancestors",
                     "printed" if ancestors and "lj75" in cells else "left out",
                     "printed" if "lj75" in printed else "left out")
        audit.cell(TEX, f"{knockout.key}: printed systems are the systems of the rows less the left-out ones",
                   " ".join(sorted(set(cells) - set(knockout.omitted))), " ".join(sorted(printed)))
        for system in knockout.omitted:
            audit.record(system == "lj75" and not ancestors, TEX, f"{knockout.key}: {system} is left out only for the "
                         "LJ75 rule", "lj75 outside history", f"{system} {'ancestors' if ancestors else 'outside'}")
    knockout_pairscale(audit, cells_of["pairscale"])
    knockout_move_a(audit, cells_of["move-a"])
    tem = cells_of["templates"]
    for rev in KNOCKOUT_CODE["templates"]:
        ancestor_of(audit, f"{KNOCKOUT_DATA}/templates", rev, SUBMITTED)
    for key in WITHDRAWN_KNOCKOUTS:
        audit.record(not (ROOT / KNOCKOUT_DATA / key).exists(), f"{KNOCKOUT_DATA}/{key}",
                     "a block with an arm outside the history of the submitted revision has no rows in the paper",
                     "absent", "absent" if not (ROOT / KNOCKOUT_DATA / key).exists() else "present")
    for gone in ("sec:k-priors", "sec:k-stall", "sec:k-units", "tab:ablation", "structural\\_priors", "f2a628c0",
                 "4f5ec0c5", "4de32289", "d12f8825", "e5883055", "1c5575a2", "a9cef902", "Thompson choice",
                 "diverse population"):
        found = [path for path in (TEX, SI, COVER) if gone in audit.text(path)]
        audit.record(not found, " ".join(found) or TEX, f"withdrawn knockout text stays out: {gone}", "absent",
                     " ".join(found) or "absent")

    def fall(c: dict) -> tuple[int, int]:
        both, only_with, only_without, _ = c["table"]
        return both + only_with, both + only_without

    m38 = tem["m38r14"]
    lo, hi = m38["interval"]
    audit.said(TEX, f"the hits on M\\(_{{38}}\\) at \\(\\rho=14\\) fall from {fall(m38)[0]} to {fall(m38)[1]} of 12 "
                    f"(\\(p={pp(m38['p'])}\\), interval {lo:.2f} to {hi:.2f}), and on the Sutton--Chen 10--8 cluster of "
                    f"38 atoms from {fall(tem['sc108-38'])[0]} to {fall(tem['sc108-38'])[1]} of 12 "
                    f"(\\(p={pp(tem['sc108-38']['p'])}\\)).")
    audit.cell(TEX, "both revisions hit M38 at rho 6 and SC 12-6 on every seed", "12 12 12 12",
               " ".join(str(x) for s in ("m38r6", "sc126-38") for x in fall(tem[s])))
    below = [s for s, c in tem.items() if c["p"] < 0.05]
    audit.cell(TEX, "the one cluster of the template run below 0.05", "m38r14", " ".join(below))
    audit.said(TEX, f"Removing the former structure templates lowers M\\(_{{38}}\\) hits at\n"
                    f"\\(\\rho=14\\) from {fall(m38)[0]} to {fall(m38)[1]} of 12")
    audit.said(TEX, f"The paired knockouts show that removing the former structure templates\n"
                    f"lowers M\\(_{{38}}\\) hits at \\(\\rho=14\\) from {fall(m38)[0]} of 12 to {fall(m38)[1]}.")
    audit.said(SI, "\\texttt{\\seqsplit{628773d6837ea2d959a27256ff4163b21162fd4d}} (structure templates since "
                   "removed in \\texttt{1c1b1238}), and the child "
                   "\\texttt{\\seqsplit{1c1b12384683485cf820d3a244151b48973dc0f7}}, both ancestors of the submitted "
                   "revision, even seeds parent first and odd seeds child first.")
    audit.said(COVER, f"Removing the former structure templates lowers M\\(_{{38}}\\) hits at\n"
                      f"\\(\\rho=14\\) from {fall(m38)[0]} to {fall(m38)[1]} of 12")

    # The switches the drivers read, and the length unit of the Morse and gold drivers.
    config = audit.text(PRIOR_CONFIG)
    for fragment in ("pub struct Knockouts {", "pub fn apply_knockouts(&mut self, knockouts: Knockouts)",
                     "self.prior_soap = false;", "self.soap_mode = SoapProposalMode::Off;",
                     "self.arm_selection = ArmSelection::Uniform;"):
        audit.record(fragment in config, PRIOR_CONFIG, "the knockout switch", fragment,
                     "present" if fragment in config else "absent")
    for driver in ("examples/h2h_gmin_proof.rs", "examples/h2h_morse.rs", "examples/h2h_metal.rs"):
        audit.record("H2H_KNOCKOUT" in audit.text(driver), driver, "the driver reads H2H_KNOCKOUT", "read",
                     "read" if "H2H_KNOCKOUT" in audit.text(driver) else "absent")
    audit.record((ROOT / "tests/knockouts.rs").exists(), "tests/knockouts.rs",
                 "the unset switch is pinned bit for bit", "present", "checked")
    audit.said(SI, "The uniform arm sets \\texttt{H2H\\_KNOCKOUT=move\\_choice}.")
    morse = audit.text("examples/h2h_morse.rs")
    audit.record('std::env::var("H2H_PAIR_SCALE").is_ok_and(|value| value == "1")' in morse, "examples/h2h_morse.rs",
                 "pair scaling only under H2H_PAIR_SCALE=1", "gated", "checked")
    audit.said(SI, "The rescaled arm sets \\texttt{H2H\\_PAIR\\_SCALE=1}.")
    audit.record("let mut cfg = Config::recommended(spec.n);" in morse, "examples/h2h_morse.rs",
                 "the Morse driver starts from Config::recommended", "recommended", "checked")
    metal = audit.text("examples/h2h_metal.rs")
    audit.record("recommended_for_pair" not in metal and "with_scales" not in metal, "examples/h2h_metal.rs",
                 "the gold driver does not rescale the preset", "unscaled", "checked")
    runner = audit.text("scripts/gmin_panel/run_panel.py")
    audit.record("H2H_PAIR_SCALE" not in runner, "scripts/gmin_panel/run_panel.py",
                 "the GMIN panel does not set H2H_PAIR_SCALE", "unset", "checked")
    audit.said(TEX, "The Morse and gold drivers of Table~\\ref{tab:gmin} run \\texttt{Config::recommended} without "
                    "rescaling its lengths to the pair minimum of their potential.")
    audit.record("scaled to that potential's pair minimum" not in audit.text(TEX), TEX,
                 "the preset is not said to be scaled to the pair minimum", "absent", "checked")

    # TIP4P against GMIN, paired by seed and start.
    path = imported(audit, "paper/jctc/data/tip4p_gmin")
    rows = audit.rows(path)
    arms: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        arms[(int(r["molecules"]), r["arm"])][r["seed"]] = r
    revisions = {r["anneal_sha"] for r in rows}
    audit.cell(path, "TIP4P rows of this code are revision e1a69a57", "e1a69a57a8dda1b44fdde9413ae31e3fc3af1889",
               revisions.pop() if len(revisions) == 1 else " ".join(sorted(revisions)))
    ancestor_of(audit, path, "e1a69a57a8dda1b44fdde9413ae31e3fc3af1889", SUBMITTED)
    audit.cell(path, "TIP4P rows per arm and size", "12", float(min(len(v) for v in arms.values())))
    shared = all(arms[(n, "anneal")][k]["start_id"] == arms[(n, "gmin")][k]["start_id"]
                 for n in {n for n, _ in arms} for k in arms[(n, "anneal")])
    audit.record(shared, path, "both TIP4P arms of a seed share the start", "shared", "shared" if shared else "not")
    sizes = {"(H\\(_2\\)O)\\(_{8}\\)": 8, "(H\\(_2\\)O)\\(_{15}\\)": 15, "(H\\(_2\\)O)\\(_{20}\\)": 20,
             "(H\\(_2\\)O)\\(_{21}\\)": 21, "(H\\(_2\\)O)\\(_{25}\\)": 25}
    stats = {}
    for line in tex_table(audit, TEX, "tab:tip4p-gmin"):
        n = sizes.get(line[0])
        where = f"{TEX} tab:tip4p-gmin {line[0]}"
        if n is None:
            audit.record(False, where, "a TIP4P size the rows hold", line[0], "unknown")
            continue
        a, g = arms[(n, "anneal")], arms[(n, "gmin")]
        seeds = sorted(a, key=int)
        if n == 25:
            text = " & ".join(line[1:])
            for arm, label in ((a, "this work"), (g, "GMIN")):
                energies = [float(arm[k]["best_energy"]) for k in seeds]
                audit.record(f"{label} \\({min(energies):.3f}\\), median \\({median(energies):.3f}\\)" in text, where,
                             f"{label} lowest and median best energy", f"{min(energies):.3f} {median(energies):.3f}",
                             text[:80])
            continue
        table = paired_table({k: a[k]["hit"] == "1" for k in seeds}, {k: g[k]["hit"] == "1" for k in seeds})
        stats[n] = (table, mcnemar(table[1], table[2]))
        lo, hi = newcombe_paired_interval(*table)
        audit.cell(where, "this work", line[1], f"{table[0] + table[1]}/12")
        audit.cell(where, "GMIN", line[2], f"{table[0] + table[2]}/12")
        audit.cell(where, "only this work", line[3], float(table[1]))
        audit.cell(where, "only GMIN", line[4], float(table[2]))
    audit.cell(TEX, "tab:tip4p-gmin sizes", "8 15 20 21", " ".join(map(str, sorted(stats))))
    t = {n: stats[n][0] for n in stats}
    audit.cell(TEX, "GMIN alone hits no seed of 15 or 20", "0 0", f"{t[15][2]} {t[20][2]}")
    audit.said(TEX, f"The hit counts for this code and GMIN are {t[8][0] + t[8][1]} and {t[8][0] + t[8][2]} on "
                    f"(H\\(_2\\)O)\\(_8\\), {t[15][0] + t[15][1]} and {t[15][0] + t[15][2]} on "
                    f"(H\\(_2\\)O)\\(_{{15}}\\), {t[20][0] + t[20][1]} and {t[20][0] + t[20][2]} on "
                    f"(H\\(_2\\)O)\\(_{{20}}\\), and {t[21][0] + t[21][1]} and {t[21][0] + t[21][2]} on "
                    f"(H\\(_2\\)O)\\(_{{21}}\\).")
    audit.cell(TEX, "neither arm hits 21", "0 0 0", " ".join(map(str, t[21][:3])))
    a25 = [float(r["best_energy"]) for r in arms[(25, "anneal")].values()]
    g25 = [float(r["best_energy"]) for r in arms[(25, "gmin")].values()]
    audit.said(TEX, f"lowest energy of this code there is \\({min(a25):.3f}\\)~kJ/mol and that of GMIN "
                    f"\\({min(g25):.3f}\\)~kJ/mol.")
    audit.said(TEX, "Table~\\ref{tab:tip4p-gmin} puts the template-free search against GMIN on\n"
                    "rigid TIP4P clusters of 8, 15, 20, 21, and 25 molecules, at\n"
                    "\\(2\\times 10^{6}\\) arm-specific evaluations on seeds "
                    "2000--2011, both arms from the same start on each seed in alternating order.")
    audit.said(TEX, "The budget counts combined energy-and-gradient passes for this code and\n"
                    "\\texttt{POTENTIAL} entries for GMIN, so no cross-arm test is reported.")
    manuscript = audit.text(TEX)
    table_text = manuscript.split("% tip4p-gmin table: start", 1)[1].split("% tip4p-gmin table: end", 1)[0]
    cross_arm = re.search(r"p_\\mathrm\{M\}|Newcombe|\\Delta", table_text)
    audit.record(cross_arm is None, TEX, "tab:tip4p-gmin reports no cross-arm test", "absent",
                 "absent" if cross_arm is None else "present")
    orders = {r["arm_order"] for r in rows}
    audit.cell(path, "TIP4P arm orders", "anneal,gmin gmin,anneal", " ".join(sorted(orders)))
    data = audit.text("examples/h2h_gmin_tip4p.data")
    for line in ("NTIP 4", "TEMPERATURE 2.0", "FIXTEMP", "STEP 0.40 0.51", "FIXSTEP"):
        audit.record(re.search(rf"(?m)^{re.escape(line)}$", data) is not None,
                     "examples/h2h_gmin_tip4p.data", f"TIP4P GMIN input sets {line}",
                     line, "set" if line in data else "absent")
    audit.record(re.search(r"(?m)^RADIUS\b", data) is None,
                 "examples/h2h_gmin_tip4p.data", "TIP4P GMIN input sets no radius",
                 "absent", "absent" if re.search(r"(?m)^RADIUS\b", data) is None else "present")
    audit.said(TEX, "fixed steps of 0.40 and 0.51, in its default Lennard-Jones container, which\n"
                    "does not hold the (H\\(_2\\)O)\\(_{20}\\) reference minimum.")
    audit.said(SI, "They set no \\texttt{RADIUS}, so GMIN applies its default Lennard-Jones\n"
                   "container: 3.8081, 4.1725, 4.3665, 4.4013, and 4.5303~\\AA{} at 8, 15, 20,\n"
                   "21, and 25 molecules.")
    audit.said(SI, "The (H\\(_2\\)O)\\(_{20}\\) Wales--Hodges minimum extends 4.6099~\\AA{} from\n"
                   "its centroid and therefore does not fit inside the 4.3665~\\AA{} container.")
    audit.said("bench/tip4p-h2h/README.md", "| (H₂O)₈ | | 1/6 | 6/6 |")
    audit.said("bench/tip4p-h2h/README.md", "With the start fixed, the radius still does not lift GMIN off 0/6 at 15 or 20.")
    audit.said(SI, "In the committed radius study, enlarging the container raises GMIN's\n"
                   "(H\\(_2\\)O)\\(_8\\) hits from 1 of 6 to 6 of 6 and does not raise its 0 of 6\n"
                   "hits at 15 or 20 molecules.")
    hosts = {r["machine_id"] for r in rows}
    cpus = {r["cpu_model"] for r in rows}
    logical = {r["logical_cpus"] for r in rows}
    audit.cell(path, "TIP4P host", "e28102384b31bbe011849b7cdfd1173eeb13f6796b540b95e19f1017ef567589",
               " ".join(sorted(hosts)))
    audit.cell(path, "TIP4P processor", "Intel(R) Xeon(R) Processor", " ".join(sorted(cpus)))
    audit.cell(path, "TIP4P logical CPUs", "8", " ".join(sorted(logical)))
    audit.said(SI, "\\texttt{\\seqsplit{e28102384b31bbe011849b7cdfd1173eeb13f6796b540b95e19f1017ef567589}},\n"
                   "an x86\\_64 machine whose rows name an Intel Xeon processor and 8 logical\n"
                   "CPUs.")


def manuscript_priors(audit: Audit) -> None:
    """Section sec:priors read against the source of the removal commit, its parent, and the current preset."""
    removal, lj_rev = "1c1b1238", "c2de652dca0bc75e2b0cb4afffac91fd656404e4"
    subject = git("log", "-1", "--format=%s", removal).stdout.strip()
    audit.cell(TEX, "the removal commit", "fix: remove structure templates from cluster search", subject)
    removed = git("show", removal).stdout
    for switch in ("prior_reseed", "prior_growth", "prior_core", "prior_handover", "prior_motif"):
        audit.record(f"-            && cfg.{switch}" in removed or f"cfg.{switch}" in "".join(
            line for line in removed.splitlines() if line.startswith("-")), TEX,
            f"{removal} removes the {switch} code path", "removed", "removed")
    on = priors_on(f"{removal}^") or []
    audit.cell(TEX, "the six priors are on in the parent of the removal", " ".join(PRIORS), " ".join(on))
    audit.said(TEX, "The earlier template-assisted preset carried six switches that use the\n"
                    "structure of close-packed clusters, all on.")
    config = git("show", f"{removal}^:{PRIOR_CONFIG}").stdout
    audit.record("reseed_hops: 5," in config, TEX, "the reseed interval is five hops before the removal", "5",
                 "5" if "reseed_hops: 5," in config else "other")
    audit.said(TEX, "at most once every five hops, closed its core orbits, and adopted the quench as the walker.")
    current = audit.text(PRIOR_CONFIG)
    audit.record("prior_soap" in current, PRIOR_CONFIG, "prior_soap remains in the code", "present",
                 "present" if "prior_soap" in current else "absent")
    audit.said(TEX, "positions, and it remains in the code.")
    audit.said(TEX, "The template-free revision removed the structure templates, the builders of\n"
                    "face-centered cubic, polyicosahedral, Mackay, and axial-spine cuts behind the\n"
                    "last five priors;")
    count_gates(audit, lj_rev)


def count_gates(audit: Audit, lj_rev: str) -> None:
    """The atom- and body-count gates the methods name, read from the current source and from c2de652d."""
    config = audit.text(PRIOR_CONFIG)
    tests = re.findall(r"if n_points\s*(<|<=|>|>=|==)\s*(\d+)\s*\{([^}]*)\}", config)
    shown = " ".join(f"{op} {n} {'surfaces' if 'cfg.surfaces = ' in body else 'other'}" for op, n, body in tests)
    audit.cell(PRIOR_CONFIG, "the atom-count tests of the presets", "< 98 surfaces < 98 surfaces", shown)
    audit.cell(PRIOR_CONFIG, "every n_points test is one of those", str(len(tests)),
               float(len(re.findall(r"(?<!self\.)\bn_points\s*(?:<|<=|>|>=|==)\s*\d", config))))
    audit.record("self.n_points >= 2," in config, PRIOR_CONFIG, "the other n_points test is the validation of the size",
                 "present", "present" if "self.n_points >= 2," in config else "absent")
    hop = audit.text("src/methods/cluster_hopping.rs")
    audit.record("n_points < 98" not in hop, "src/methods/cluster_hopping.rs", "the growth hold gate is gone",
                 "absent", "absent" if "n_points < 98" not in hop else "present")
    old = git("show", f"{lj_rev}:src/methods/cluster_hopping.rs").stdout
    audit.record("stall_growth_waits(held, incumbent) && (n_points < 98 || skipped < GROWTH_RETRY_LEAVES)" in old,
                 f"{lj_rev[:8]}:src/methods/cluster_hopping.rs", "the prior-on growth hold releases at 98 atoms",
                 "present", "present")
    bodies = re.search(r"const RIGID_HOT_BODIES: usize = (\d+);", hop)
    audit.cell("src/methods/cluster_hopping.rs", "rigid body-count gate", "10", float(bodies.group(1)) if bodies else None)
    for fragment in ("if cfg.move_library.is_rigid_body() {\n        run_search::<R, H, true>(",
                     "if RIGID && cfg.n_points > RIGID_HOT_BODIES {",
                     "if RIGID && cfg.n_points <= RIGID_HOT_BODIES &&"):
        audit.record(fragment in hop, "src/methods/cluster_hopping.rs", "the rigid gate reads the body count",
                     fragment, "present" if fragment in hop else "absent")
    assignment = audit.text("src/assignment.rs")
    audit.record("const SHORTEST_PATH_FROM: usize = 98;" in assignment and "the same optimum" in assignment,
                 "src/assignment.rs", "the assignment switches algorithm at 98 for the same optimum", "98", "98")
    audit.record("minimum_cost_assignment" in audit.text("src/continuous_symmetry.rs"), "src/continuous_symmetry.rs",
                 "the symmetry projections use that assignment", "used", "used")
    audit.said(TEX, "For atomic clusters the preset has one atom-count gate: the diameter penalty of "
                    "equation~\\ref{eq:penalty} is used only below 98 atoms.")
    audit.said(TEX, "The revisions with the structure priors on had a second, the growth hold of "
                    "equation~\\ref{eq:hold}.")
    small = re.search(r"fn rigid_factor_cap\(n_bodies: usize\) -> f64 \{\s*if n_bodies > RIGID_HOT_BODIES \{\s*"
                      r"RIGID_FACTOR_CAP_LARGE\s*\} else \{\s*([0-9.]+)\s*\}", hop)
    audit.cell("src/methods/cluster_hopping.rs", "rigid factor cap at 10 bodies or fewer", "24",
               float(small.group(1)) if small else None)
    large = re.search(r"const RIGID_FACTOR_CAP_LARGE: f64 = ([0-9.]+);", hop)
    audit.cell("src/methods/cluster_hopping.rs", "rigid factor cap above 10 bodies", "4",
               float(large.group(1)) if large else None)
    for fragment in ("rigid_factor_cap(cfg.n_points)\n    } else {\n        1.0",
                     "rigid_factor = (rigid_factor / 1.5).max(1.0);"):
        audit.record(fragment in hop, "src/methods/cluster_hopping.rs",
                     "the rigid factor starts at its cap and falls no lower than 1", fragment,
                     "present" if fragment in hop else "absent")
    audit.said(TEX, "A rigid-body run has a gate of its own: at 10 molecules or fewer a factor on its acceptance "
                    "temperature starts at 24 and moves between 1 and 24 with the share of hops that reach a new "
                    "minimum, and above 10 that factor stays at 4 and its orientation step follows the acceptance rate.")
    audit.said(TEX, "The exact assignment inside the symmetry projections also changes algorithm at 98 atoms, and both "
                    "algorithms return an assignment of the optimal cost.")
    audit.said(TEX, "In the revisions with the structure priors on, lattice growth on a plateau is held when it has "
                    "already failed at the current energy.")
    for phrase in ("two atom-count gates", "Apart from those gates no move reads the atom count"):
        audit.record(phrase not in collapse(audit.text(TEX)), TEX, f"retired gate wording stays out: {phrase}",
                     "absent", "absent" if phrase not in collapse(audit.text(TEX)) else "present")


def rigid_stall(audit: Audit) -> None:
    """The rigid-walker stall of the Methods against rigid_quench.rs, stall_archive.rs and the hop loop."""
    quench = audit.text("src/methods/rigid_quench.rs")
    archive = audit.text("src/methods/stall_archive.rs")
    hop = audit.text("src/methods/cluster_hopping.rs")
    for path, label, pattern, shown in (
        ("src/methods/rigid_quench.rs", "rigid stall hops", r"const STAGNATION_HOPS: usize = (\d+);", "1000"),
        ("src/methods/stall_archive.rs", "stall archive seats", r"pub const SEATS: usize = (\d+);", "8"),
        ("src/methods/stall_archive.rs", "barred stall destinations", r"pub const MEMORY: usize = (\d+);", "4"),
    ):
        found = re.search(pattern, quench if path.endswith("rigid_quench.rs") else archive)
        audit.cell(path, label, shown, float(found.group(1)) if found else None)
    for path, fragment in (
        ("src/methods/rigid_quench.rs", "hops.saturating_sub(hops_at_lowest) > STAGNATION_HOPS"),
        ("src/methods/stall_archive.rs", "A `None` result means every kept minimum is the walker's own or\n    /// barred, so the walker stays put."),
        ("src/methods/stall_archive.rs", ".filter(|seat| !same_minimum(seat.energy, here))"),
        ("src/methods/cluster_hopping.rs", "if let Some((energy, state)) = stall_archive.escape(e)"),
        ("src/methods/cluster_hopping.rs", "MoveLibrary::RigidBody { relocate: true, .. }"),
        ("src/methods/cluster_hopping.rs", "trial = ClusterMove::RigidRelocate {"),
    ):
        text = {"src/methods/rigid_quench.rs": quench, "src/methods/stall_archive.rs": archive}.get(path, hop)
        audit.record(fragment in text, path, "the rigid stall reads as the Methods say", fragment,
                     "present" if fragment in text else "absent")
    audit.said(TEX, "\\begin{equation} h - h_{\\star} > 1000. \\label{eq:rigid} \\end{equation}")
    audit.said(TEX, "Its archive keeps the eight lowest distinct minima among its start and the quenched minima it "
                    "has accepted, and the lowest minimum of the run joins it at every stall.")
    audit.said(TEX, "nor one of its last four stall destinations.")
    audit.said(TEX, "If no archived minimum qualifies, the walker stays where it is; no new start is drawn.")
    audit.said(TEX, "proposes, once, the relocation of one molecule onto a vacant contact, unless a new lowest "
                    "minimum comes first.")
    for gone in ("quenches a new rigid start", "moves to a lower minimum it has already quenched"):
        audit.record(gone not in collapse(audit.text(TEX)), TEX, f"retired rigid-stall wording stays out: {gone}",
                     "absent", "absent" if gone not in collapse(audit.text(TEX)) else "present")


TEMPLATE_AND_SWITCH_REMOVAL = (TEMPLATE_REMOVAL, "5499be31263b4db0217412ee82f3da47e8d0676a")
BENCHMARK_REVISIONS = {
    "tab:gmin": ("eeec8e2c216ae1506768bca8e29846c1c039ca3b", SUBMITTED),
    "tab:tip4p-gmin": ("e1a69a57a8dda1b44fdde9413ae31e3fc3af1889",),
    "tab:ase": (SUBMITTED,),
    "tab:ase-morse": ("5eb79cdcb0a986984dc7b487e5a3cc93224246f9",),
}
TABLE_PRESETS = {
    "tab:gmin": ("recommended atomic-cluster preset",),
    "tab:knockouts": ("earlier template-assisted search and its template-free child",),
    "tab:tip4p-gmin": ("recommended template-free rigid-water preset",),
    "tab:ase": ("recommended atomic preset",
                "\\texttt{Config::recommended\\_with\\_scales} at the Gupta length and well depth"),
    "tab:ase-morse": ("\\texttt{Config::recommended\\_with\\_scales} at the length scale",
                      "an ancestor of the submitted revision"),
}


TEMPLATE_ERA = ("\\label{sec:si-template-era}", "\\section{")


def outside_template_era(text: str) -> str:
    """The SI without its section that pairs the template-assisted c2de652d with the later rows."""
    start, end = TEMPLATE_ERA
    if start not in text:
        return text
    before, rest = text.split(start, 1)
    return before + (end + rest.split(end, 1)[1] if end in rest else "")


def presets_and_templates(audit: Audit) -> None:
    """Main tables name methods; the SI names revisions and drivers."""
    text = collapse(audit.text(TEX))
    audit.said(TEX, "The search uses no structure templates.")
    audit.said(TEX, "The Supporting Information names the revision, driver, host, and record of\n"
                    "every row.")
    audit.said(TEX, "Every comparison against another code follows the removal of the structure\n"
                    "templates; the one run with them is the With arm of the knockout in\n"
                    "Section~\\ref{sec:knockouts}.")
    tables = {label: chunk for chunk in text.split("\\begin{table") for label in re.findall(r"\\label\{(tab:[^}]+)\}", chunk)}
    for label, fragments in TABLE_PRESETS.items():
        for fragment in fragments:
            audit.record(collapse(fragment) in tables.get(label, ""), TEX, f"{label} names its preset", fragment,
                         "present" if collapse(fragment) in tables.get(label, "") else "absent")
    for label, revisions in BENCHMARK_REVISIONS.items():
        for revision in revisions:
            audit.record(revision in audit.text(SI),
                         SI, f"{label} revision {revision[:8]} is named in the SI", "named",
                         "named" if revision in audit.text(SI) else "absent")
            for removal in TEMPLATE_AND_SWITCH_REMOVAL:
                ancestor_of(audit, label, removal, revision)
    for gone in ("c2de652d", "default behavior", "default behaviour", "the default preset"):
        audit.record(gone not in text, TEX, f"no table is presented as {gone}", "absent",
                     "absent" if gone not in text else "present")
    hop = git("show", f"{SUBMITTED}:src/methods/cluster_hopping.rs").stdout
    changed = git("diff", "--name-only", BENCHMARK_REVISIONS["tab:gmin"][0], SUBMITTED, "--", "src").stdout.split()
    audit.cell("src", "search files changed between the two GMIN blocks",
               "src/methods/cluster_hopping.rs src/methods/cluster_hopping/config.rs", " ".join(sorted(changed)))
    audit.record("if cfg.plateau_heat {\n                heat = (heat * 2.0).min(PLATEAU_HEAT_CAP);" in hop
                 and hop.count("heat = (heat") == 1, f"{SUBMITTED[:8]}:src/methods/cluster_hopping.rs",
                 "the walk heats only under plateau_heat", "only", "only")
    config = git("show", f"{SUBMITTED}:src/methods/cluster_hopping/config.rs").stdout
    audit.record("plateau_heat: false," in config, f"{SUBMITTED[:8]}:src/methods/cluster_hopping/config.rs",
                 "plateau_heat is off in the preset", "off", "off" if "plateau_heat: false," in config else "on")
    _, groups = gmin_rerun_groups(audit)
    develop = [r for r in audit.rows(f"{GMIN_DEVELOP}/rows.csv") if r["system"] == "m80r6" and r["code"] == "anneal"]
    earlier = {r["seed"]: (r["evals_to_hit"], r["best"]) for r in groups[("m80r6", "anneal")]}
    later = {r["seed"]: (r["evals_to_hit"], r["best"]) for r in develop}
    audit.cell(GMIN_DEVELOP, "M80 rho=6 seeds with the same calls and minimum at both revisions", "12",
               float(sum(earlier.get(seed) == value for seed, value in later.items())))
    hits = {key: sum(r["hit"] == "1" for r in group) for key, group in groups.items()}
    for system in ("lj104", "lj150"):
        audit.record(hits[(system, "gmin")] > hits[(system, "anneal")], GMIN_RERUN, f"GMIN hits more seeds on {system}",
                     "more", f"{hits[(system, 'gmin')]} against {hits[(system, 'anneal')]}")
    audit.said(SI, "the earlier template-assisted revision \\texttt{\\seqsplit{628773d6837ea2d959a27256ff4163b21162fd4d}} "
                   "(structure templates since removed in \\texttt{1c1b1238})")
    audit.said(SI, "the earlier template-assisted revision\n"
                   "\\texttt{c2de652d} (structure templates since removed in\n"
                   "\\texttt{1c1b1238})")


VISUALS = "paper/jctc/data/system_visuals"
HEXAMER_XTB = "xtb version 6.7.1 (edcfbbe) compiled by 'albert@albert-system' on 2024-07-22"


def hexamer_provenance(audit: Audit) -> None:
    """The hexamers of fig:systems are the xtb 6.7.1 relaxations RECORD describes, and its provenance names commits here."""
    keys, digests = {}, {}
    for line in audit.text(f"{VISUALS}/provenance.txt").splitlines():
        first, _, rest = line.partition(" ")
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"[0-9a-f]{64}", first):
            digests[rest.strip()] = first
        else:
            keys[first] = rest.strip()
    record = {}
    for line in audit.text(f"{VISUALS}/hexamer_gfn2/RECORD").splitlines():
        first, _, rest = line.partition(" ")
        record[first] = rest
    audit.cell(f"{VISUALS}/hexamer_gfn2/RECORD", "xtb build", HEXAMER_XTB, record.get("xtb_version"))
    audit.cell(f"{VISUALS}/hexamer_gfn2/RECORD", "xtb command", "xtb input.xyz --gfn 2 --opt vtight", record.get("command"))
    audit.cell(f"{VISUALS}/hexamer_gfn2/RECORD", "xtb release sha256",
               "62a8d18778286e815292ee53d76ce447daf460a4dea3782c0f25cbac7019b5df", record.get("xtb_release_sha256"))
    base = record.get("base_commit", "")
    ancestor_of(audit, f"{VISUALS}/hexamer_gfn2/RECORD", base)
    script = "paper/jctc/scripts/hexamer_gfn2_relax.py"
    blob = git("hash-object", script).stdout.strip()
    shown = record.get("script", "").split()
    audit.cell(f"{VISUALS}/hexamer_gfn2/RECORD", "blob of the relaxation script", blob, shown[-1] if shown else None)
    audit.cell(f"{VISUALS}/provenance.txt", "blob of the relaxation script", blob,
               (keys.get("hexamer_gfn2_script") or " ").split()[-1])
    for cls in ("prism", "cage", "book", "ring"):
        name = f"system_visuals/hexamer_gfn2/{cls}.xyz"
        path = ROOT / VISUALS / "hexamer_gfn2" / f"{cls}.xyz"
        body = path.read_bytes() if path.is_file() else b""
        audit.cell(f"{VISUALS}/provenance.txt", f"digest of {cls}.xyz", digests.get(name), hashlib.sha256(body).hexdigest())
        header = body.decode().splitlines()[1].split() if body else []
        shown = record.get(cls, "").split()
        energy = shown[shown.index("relaxed_energy_Eh") + 1] if "relaxed_energy_Eh" in shown else None
        written = header[header.index("energy:") + 1] if "energy:" in header else None
        audit.record(energy is not None and written is not None and float(energy) == float(written),
                     f"{VISUALS}/hexamer_gfn2/{cls}.xyz", "energy is the one RECORD gives", energy, written)
        audit.record("xtb: 6.7.1 (edcfbbe)" in " ".join(header), f"{VISUALS}/hexamer_gfn2/{cls}.xyz",
                     "written by xtb 6.7.1", "6.7.1", "6.7.1" if "xtb: 6.7.1 (edcfbbe)" in " ".join(header) else "other")
    structures = keys.get("structures_commit", "")
    ancestor_of(audit, f"{VISUALS}/provenance.txt", structures)
    for path in sorted((ROOT / VISUALS / "structures").iterdir()):
        for revision in dict.fromkeys((structures, SUBMITTED)):
            fixture = git_bytes("show", f"{revision}:tests/fixtures/{path.name}")
            audit.record(fixture == path.read_bytes(), f"{VISUALS}/structures/{path.name}",
                         f"byte for byte tests/fixtures/ of {revision[:8]}", "identical",
                         "identical" if fixture == path.read_bytes() else "differs")
    for key in ("anneal_commit", "hexamer_gfn2_commit"):
        audit.record(key not in keys, f"{VISUALS}/provenance.txt", f"no {key} key names a commit for every file",
                     "absent", "absent" if key not in keys else "present")
    landfold_commit = keys.get("landfold_book_commit", "")
    audit.record(revision_reach.is_full_sha(landfold_commit),
                 f"{VISUALS}/provenance.txt", "landfold source identifier is recorded",
                 "40 hexadecimal digits", landfold_commit)
    audit.said(f"{VISUALS}/provenance.txt",
               "landfold_book_commit_note source identifier for the committed landfold outputs")
    landfold = {name: digest for name, digest in digests.items()
                if name.startswith("system_visuals/landfold/")}
    audit.record(bool(landfold), f"{VISUALS}/provenance.txt",
                 "committed landfold outputs have sha256 digests", "one or more", len(landfold))
    for name, digest in sorted(landfold.items()):
        body = (ROOT / "paper/jctc/data" / name).read_bytes()
        audit.cell(f"{VISUALS}/provenance.txt", f"digest of {name}", digest,
                   hashlib.sha256(body).hexdigest())
    audit.said(TEX, "for each\nclass, the lowest of the archived GFN2-xTB water hexamers after relaxation\n"
                    "with xtb 6.7.1 (\\texttt{-{}-gfn 2 -{}-opt vtight}).}")
    audit.record("relaxed at GFN2-xTB with xtb 6.7.1" not in collapse(audit.text(TEX)), TEX,
                 "the unrecorded relaxation caption stays out", "absent",
                 "absent" if "relaxed at GFN2-xTB with xtb 6.7.1" not in collapse(audit.text(TEX)) else "present")


TEMPLATE_ERA_DATA = "paper/jctc/data/template_era"
TEMPLATE_ERA_PROSE = (
    "Table~\\ref{tab:si-template-era} pairs the archived rows of this code from the GMIN panel session with the "
    "panel of the earlier template-assisted revision \\texttt{c2de652d} (structure templates since removed in "
    "\\texttt{1c1b1238}), on seeds 2000--2011 at the same budgets.",
    "Both sessions start each seed from the geometry the driver writes for it.",
    "Au\\(_{55}\\) is left out, since its GMIN build differs between the sessions.",
    "Against \\texttt{c2de652d}, the later revisions hit M\\(_{80}\\) at \\(\\rho=6\\) on 12 of 12 seeds against 2, "
    "with \\(p=2.0\\times 10^{-3}\\) and an interval from 0.46 to 0.95, and M\\(_{55}\\) at \\(\\rho=14\\) and "
    "LJ\\(_{150}\\) on 9 and 8 against 7 and 5.",
    "They hit LJ\\(_{75}\\) on 5 of 12 against 12, with \\(p=0.016\\), LJ\\(_{98}\\), LJ\\(_{102}\\), "
    "LJ\\(_{103}\\), and LJ\\(_{104}\\) on 7, 6, 6, and 5 against 10, 9, 8, and 9, and M\\(_{80}\\) at "
    "\\(\\rho=14\\) on 6 against 9.",
    "LJ\\(_{38}\\), LJ\\(_{55}\\), and M\\(_{55}\\) at \\(\\rho=6\\) are hit on every seed at both, with medians "
    "of 127839.5 against 10615.5, 5722 against 12242, and 22596 against 563 calls.",
    "Apart from M\\(_{80}\\) at \\(\\rho=6\\) and LJ\\(_{75}\\), every McNemar \\(p\\) is above 0.2.",
)
TEMPLATE_ERA_GONE = ("c2de652d", "template-era", "template-assisted rows", "instead of 2", "instead of 12",
                     "from 2 to 12", "from 12 to 5")
TEMPLATE_ERA_HITS = {"m80r6": (2, 12), "lj75": (12, 5), "lj38": (12, 12), "lj98": (10, 7), "lj102": (9, 6),
                     "lj103": (8, 6), "lj104": (9, 5), "m80r14": (9, 6), "lj150": (5, 8), "m55r14": (7, 9)}
MATCH_CLAIM = re.compile(r"\b(match\w*|reproduc\w*|recover\w*|same as|as well as|equal\w*)\b", re.IGNORECASE)


def template_era(audit: Audit) -> None:
    """tab:si-template-era pairs c2de652d, before the template removal, with the rows of tab:gmin, in the SI only."""
    path = imported(audit, TEMPLATE_ERA_DATA)
    era = load_script("template_era_table_audit", "paper/jctc/scripts/template_era_table.py")
    ancestor_of(audit, path, era.TEMPLATES)
    after = git("merge-base", "--is-ancestor", TEMPLATE_REMOVAL, era.TEMPLATES).returncode == 0
    audit.record(not after, path, "c2de652d precedes the template removal", "precedes", "follows" if after else "precedes")
    audit.cell(path, "revision of every row of this code", era.TEMPLATES,
               " ".join(sorted({r["code_revision"] for r in audit.rows(path) if r["code"] == "anneal"})))
    lines = {}
    for revision, block in era.compared():
        for line in block:
            lines.setdefault(line["system"], line)
    audit.cell(path, "clusters compared", " ".join(sorted([*TEMPLATE_ERA_HITS, "lj55", "m55r6"])),
               " ".join(sorted(lines)))
    for system, (before, now) in TEMPLATE_ERA_HITS.items():
        line = lines[system]
        audit.cell(path, f"{system} hits at c2de652d", str(before), float(sum(r["hit"] == "1" for r in line["before"])))
        audit.cell(path, f"{system} hits later", str(now), float(sum(r["hit"] == "1" for r in line["after"])))
    for system, wanted in (("m80r6", "2.0e-3"), ("lj75", "0.016")):
        audit.cell(path, f"{system} McNemar p", wanted, lines[system]["mcnemar_p"])
    low, high = lines["m80r6"]["newcombe"]
    audit.cell(path, "m80r6 Newcombe interval", "0.46", low)
    audit.cell(path, "m80r6 Newcombe interval", "0.95", high)
    for system in ("lj98", "lj102", "lj103", "lj104", "m80r14", "lj150", "m55r14"):
        audit.record(lines[system]["mcnemar_p"] > 0.2, path, f"{system} McNemar p is above 0.2", "above",
                     f"{lines[system]['mcnemar_p']:.3g}")
    for system, before, now in (("m80r6", "4000000", "62677"), ("lj38", "10615.5", "127839.5"),
                                ("lj55", "12242", "5722"), ("m55r6", "563", "22596")):
        audit.cell(path, f"{system} median calls at c2de652d", before, era.calls(lines[system]["before"], 0.5))
        audit.cell(path, f"{system} median calls later", now, era.calls(lines[system]["after"], 0.5))
    section = audit.text(SI).split(TEMPLATE_ERA[0], 1)[1].split(TEMPLATE_ERA[1], 1)[0]
    for said in ("GMIN takes the same calls", "GMIN arm", "one host", "one machine", "single machine"):
        audit.record(said not in collapse(section), SI, f"the template-era section prints no GMIN claim: {said}",
                     "absent", "absent" if said not in collapse(section) else "present")
    source = audit.text("paper/jctc/scripts/template_era_table.py")
    for said in ("takes the same calls", "GMIN differs"):
        audit.record(said not in source, "paper/jctc/scripts/template_era_table.py",
                     f"the template-era script does not require a GMIN match: {said}", "absent",
                     "absent" if said not in source else "present")
    for fragment in TEMPLATE_ERA_PROSE:
        audit.said(SI, fragment)
    audit.said(TEX, "An earlier template-assisted revision also hits LJ\\(_{75}\\) on 12 of 12\n"
                    "seeds, against 5 of 12 for the later template-free revision\n"
                    "\\((p=0.016\\), interval \\(-\\)0.81 to \\(-\\)0.22).")
    audit.said(TEX, "On LJ\\(_{38}\\), both revisions hit every seed, but their median call costs\n"
                    "are 10615.5 and 127839.5 combined energy-and-gradient passes\n"
                    "\\((p=4.9\\times 10^{-4})\\).")
    audit.said(TEX, "That comparison spans every code change between the two revisions, so it\n"
                    "supports a change across revisions rather than an isolated template effect.")
    audit.said(COVER, "across every intervening code change, LJ\\(_{75}\\)\n"
                      "hits fall from 12 to 5 of 12.")
    audit.said(TEX, "this code reports a median of 127839.5 combined energy-and-gradient passes to the minimum, while "
                    "GMIN reports 389399 \\texttt{POTENTIAL} entries; these are not the same operation.")
    for path_ in (TEX, SI, COVER):
        sentences = re.split(r"(?<=[.;])\s+", collapse(audit.text(path_)))
        claims = [s[:80] for s in sentences if re.search(r"templat", s, re.I) and MATCH_CLAIM.search(s)
                  and not re.search(r"\bnot\b", s)]
        audit.record(not claims, path_, "no sentence says the later code matches the template-assisted rows",
                     "none", "; ".join(claims) or "none")


OUTSIDE_DATA = "paper/jctc/data/outside_perf"


def outside_session(audit: Audit) -> None:
    """tab:si-outside is the two builds of 61fa67cf in record/outside-perf3-r2, recomputed from its per-seed rows."""
    fields = dict(line.split(" ", 1) for line in audit.text(f"{OUTSIDE_DATA}/PROVENANCE").splitlines())
    path = f"{OUTSIDE_DATA}/rows.csv"
    source = git("show", f"{fields['commit']}:{fields['source']}").stdout
    audit.cell(path, "blob of the record source", fields["source_blob"],
               git("rev-parse", f"{fields['commit']}:{fields['source']}").stdout.strip())
    source_rows = list(csv.DictReader(io.StringIO(source)))
    selected = [row for row in source_rows if row["arm"] in ("develop", "develop_prime")]
    projected = io.StringIO()
    writer = csv.DictWriter(projected, fieldnames=source_rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected)
    audit.cell(path, "blob of the reachable-arm projection", fields["blob"],
               git("hash-object", "--stdin", stdin=projected.getvalue()).stdout.strip())
    audit.cell(path, "blob of the committed projection", fields["blob"],
               git("hash-object", str(ROOT / path)).stdout.strip())
    audit.cell(path, "selection from the record source", "arms develop,develop_prime", fields["selection"])
    on_branch = git("merge-base", "--is-ancestor", fields["commit"],
                    f"refs/remotes/origin/{fields['branch']}").returncode == 0
    audit.record(on_branch, path, "the record commit is on the record branch of origin",
                 fields["branch"], "on it" if on_branch else "not on it")
    audit.cell(f"{OUTSIDE_DATA}/PROVENANCE", "tree of the record commit", fields["tree"],
               git("rev-parse", f"{fields['commit']}^{{tree}}").stdout.strip())
    ancestor_of(audit, path, fields["ancestor"])
    ancestor_of(audit, path, fields["ancestor"], fields["commit"])
    era = load_script("outside_table_audit", "paper/jctc/scripts/outside_table.py")
    si = audit.text(SI)
    block = si.split(era.MARKS[0], 1)[1].split(era.MARKS[1], 1)[0].strip("\n") if si.count(era.MARKS[0]) == 1 else ""
    audit.record(block == era.table(), SI, "tab:si-outside is the summariser's block",
                 "as written" if block == era.table() else "edited by hand", "as written")
    lines = era.lines()
    worst = max(abs(line["change"]) for line in lines)
    audit.cell(path, "largest change between the builds, percent", "3.0", 100.0 * worst)
    audit.cell(path, "smallest Wilcoxon p between the builds", "0.23", min(line["p"] for line in lines))
    audit.said(SI, "Both builds take the same calls to the same minimum on every seed, and their median outside "
                   "seconds differ by at most 3.0\\% on each cluster, with every \\(p\\) at or above 0.23.")
    flat = collapse(block)
    for word in ("new_outside", "\\texttt{new}", "blj", "BLJ"):
        audit.record(word not in flat, SI, f"the outside table prints no {word} arm or system", "absent",
                     "absent" if word not in flat else "present")
    rows = audit.rows(path)
    blj = [row for row in rows if row["system"] in ("blj55s130", "blj38s105")]
    audit.cell(path, "unprinted BLJ rows", "48", float(len(blj)))
    audit.cell(path, "code revisions", "61fa67cf380e46876b569da45e7cc091647a775b",
               ",".join(sorted({row["code_revision"] for row in rows})))
    audit.said(SI, "The row file is the exact \\texttt{develop} and \\texttt{develop\\_prime}\n"
                   "projection of commit \\texttt{9178884e}")
    audit.said(SI, "It also contains both builds for BLJ\\(_{55}\\) at size ratio 1.30 and\n"
                   "BLJ\\(_{38}\\) at size ratio 1.05, systems not used in this article.")


G1_DATA = "paper/jctc/data/gmin_g1"
G1_STORED = ("bench/morse-gmin/seeds.csv", "bench/morse-panel/seeds.csv")


def morse_g1(audit: Audit) -> None:
    """tab:si-morse-g1 is the anneal rows of record/g1-develop-stock, checked row by row and recomputed."""
    path = imported(audit, G1_DATA)
    fields = dict(line.split(" ", 1) for line in audit.text(f"{G1_DATA}/PROVENANCE").splitlines())
    commit = fields["commit"]
    audit.cell(f"{G1_DATA}/PROVENANCE", "tree of the record commit", fields["tree"],
               git("rev-parse", f"{commit}^{{tree}}").stdout.strip())
    audit.cell(f"{G1_DATA}/PROVENANCE", "the record ancestor is the submitted revision", SUBMITTED, fields["ancestor"])
    ancestor_of(audit, path, SUBMITTED, commit)
    same = git("diff", "--quiet", SUBMITTED, commit, "--", "src", "examples", "Cargo.lock").returncode == 0
    audit.record(same, path, "the record commit holds the search source of the submitted revision", "same",
                 "same" if same else "differs")
    era = load_script("morse_g1_table_audit", "paper/jctc/scripts/morse_g1_table.py")
    specs = era.specs()
    rows = audit.rows(path)
    bad = [f"{r['system']} {r['seed']}" for r in rows
           if (r["code"], r["code_revision"], r["config"]) != ("anneal", SUBMITTED, "recommended")
           or int(r["budget"]) != specs[r["system"]][2]
           or (r["hit"] == "1") != energy_hits(float(r["best"]), specs[r["system"]][3], DEFAULT_HIT_TOL)]
    audit.cell(path, "rows off the stock preset, the budget or the reference of the run script", "none",
               ", ".join(bad) or "none")
    audit.cell(path, "hosts", "1", float(len({r["host"] for r in rows})))
    seeds = defaultdict(list)
    for r in rows:
        seeds[r["system"]].append(int(r["seed"]))
    audit.cell(path, "clusters with seeds 2000-2011 once each", "10",
               float(sum(sorted(v) == list(range(2000, 2012)) for v in seeds.values())))
    record = list(csv.DictReader(io.StringIO(git("show", f"{commit}:bench/g1-develop-stock/rows.csv").stdout)))
    paired = [{k: v for k, v in r.items() if k != "source"} for r in record if r["code"] == "anneal"]
    audit.record(paired == [dict(r) for r in rows], path, "the paired rows of the record carry these anneal rows",
                 "same", "same" if paired == [dict(r) for r in rows] else "differs")
    stored = [r for r in record if r["code"] == "gmin"]
    sources = {path_: [r for r in csv.DictReader(io.StringIO(audit.text(path_))) if r["code"] == "gmin"]
               for path_ in G1_STORED}
    labels = {r["system"]: {x.get("host", "") for x in sources[r["source"]] if x["system"] == r["system"]}
              for r in stored}
    machines = sorted({r["system"] for r in stored
                       if any(re.fullmatch(r"[0-9a-f]{64}", h) for h in labels[r["system"]])})
    audit.cell(path, "clusters whose stored GMIN rows name a machine", "none", ", ".join(machines) or "none")
    audit.cell(path, "clusters whose stored GMIN rows name no GMIN revision", "5",
               float(len({r["system"] for r in stored if not r["code_revision"]})))
    gone = [w for w in ("24 hits", "of the 24 misses", "other calls") if w in collapse(audit.text(SI))]
    audit.cell(SI, "the SI prints no count from the stored GMIN rows of G1", "none", ", ".join(gone) or "none")
    si = audit.text(SI)
    block = si.split(era.MARKS[0], 1)[1].split(era.MARKS[1], 1)[0].strip("\n") if si.count(era.MARKS[0]) == 1 else ""
    audit.record(block == era.table(), SI, "tab:si-morse-g1 is the summariser's block",
                 "as written" if block == era.table() else "edited by hand", "as written")
    count = {name: sum(r["hit"] == "1" for r in rows if r["system"] == name) for name in seeds}
    for name in ("m30r6", "m38r6", "m80r3"):
        audit.cell(path, f"{name} hits", "12", float(count[name]))
    audit.cell(path, "fewest hits, M38 at rho 14", "4", float(count["m38r14"]))
    audit.record(min(count, key=count.get) == "m38r14" and sorted(count.values())[1] > 4, path,
                 "M38 at rho 14 alone has the fewest hits", "m38r14", min(count, key=count.get))
    audit.cell(path, "most hits", "12", float(max(count.values())))
    audit.said(SI, "It hits every seed of M\\(_{30}\\) and M\\(_{38}\\) at \\(\\rho=6\\) and of M\\(_{80}\\) at "
                   "\\(\\rho=3\\), and 4 of 12 on M\\(_{38}\\) at \\(\\rho=14\\), the fewest of the ten.")
    audit.said(SI, "The record pairs these rows with stored GMIN rows of earlier sessions, which name no machine for "
                   "any cluster and no GMIN revision for five of the ten, so the article prints this code alone here.")
    audit.said(SI, "The rows are copied from commit \\texttt{\\seqsplit{4256269394435f77787d2bfdbead2321e08340a6}} "
                   "of \\path{record/g1-develop-stock}")
    audit.said(TEX, "The Supporting Information runs the submitted revision alone on ten more Morse clusters, of 30, "
                    "38, 70 and 80 atoms at \\(\\rho\\) from 3 to 14, where it hits from 4 to 12 of 12 seeds.")
    audit.cell(f"{G1_DATA}/PROVENANCE", "the cited record commit", "4256269394435f77787d2bfdbead2321e08340a6", commit)


G3_DATA = "paper/jctc/data/g3_funnel"
G3_REVISION = "0cc3814c053c2f5a709f0f375ebae07e5373c171"
G3_TRACE_FILES = ("examples/h2h_gmin_proof.rs", "examples/h2h_morse.rs", "src/methods/cluster_hopping.rs")


def g3_session(audit: Audit) -> None:
    """The traced session of record/g3-funnel reproduces the rows of this code in the GMIN table seed by seed."""
    path = imported(audit, G3_DATA)
    fields = dict(line.split(" ", 1) for line in audit.text(f"{G3_DATA}/PROVENANCE").splitlines())
    commit = fields["commit"]
    audit.cell(f"{G3_DATA}/PROVENANCE", "tree of the record commit", fields["tree"],
               git("rev-parse", f"{commit}^{{tree}}").stdout.strip())
    ancestor_of(audit, path, SUBMITTED, G3_REVISION)
    ancestor_of(audit, path, G3_REVISION, commit)
    changed = git("diff", "--name-only", SUBMITTED, G3_REVISION, "--", "src", "examples", "Cargo.toml",
                  "Cargo.lock").stdout.split()
    audit.cell(path, "files the traced revision changes in the search source", " ".join(G3_TRACE_FILES),
               " ".join(sorted(changed)))
    removed = [line for line in git("diff", SUBMITTED, G3_REVISION, "--", "src").stdout.splitlines()
               if line.startswith("-") and not line.startswith("---")]
    audit.cell(path, "lines of src the traced revision removes", "0", float(len(removed)))
    rows = [r for r in audit.rows(path) if r["method"] == "anneal"]
    audit.cell(path, "anneal rows", "60", float(len(rows)))
    audit.cell(path, "revision of every anneal row", G3_REVISION, " ".join(sorted({r["code_revision"] for r in rows})))
    machines = sorted({r["machine"] for r in rows})
    audit.cell(path, "machine", "3b11c160", " ".join(m[:8] for m in machines))
    replayed = sum(r["replay_trace_match"] == "1" and r["replay_best_match"] == "1" for r in rows)
    audit.cell(path, "anneal runs whose replay gives the same record and lowest energy", "60", float(replayed))
    panel = load_script("g3_gmin_panel", "scripts/gmin_panel/run_panel.py")
    refs = {row[0]: row[3] for row in panel.SYSTEMS}
    bad = [f"{r['system']} {r['seed']}" for r in rows
           if (r["hit"] == "1") != energy_hits(float(r["best"]), refs[r["system"]], DEFAULT_HIT_TOL)]
    audit.cell(path, "hits off the reference of the GMIN panel", "none", ", ".join(bad) or "none")
    table = {}
    for shard in GMIN_SHARDS:
        for r in audit.rows(f"{GMIN_RERUN}/{shard}/rows.csv"):
            if r["code"] == "anneal":
                table[(r["system"], r["seed"])] = r
    develop = {(r["system"], r["seed"]): r for r in audit.rows(f"{GMIN_DEVELOP}/rows.csv") if r["code"] == "anneal"}
    keys = ("hit", "budget", "evals_to_hit", "charged")
    for label, other in (("first block", table), ("second block", develop)):
        shared = [r for r in rows if (r["system"], r["seed"]) in other]
        same = sum(all(r[k] == other[(r["system"], r["seed"])][k] for k in keys)
                   and float(r["best"]) == float(other[(r["system"], r["seed"])]["best"]) for r in shared)
        audit.cell(path, f"seeds with the calls and lowest energy of the GMIN table, {label}",
                   "60" if label == "first block" else "12", float(same))
        audit.cell(path, f"seeds shared with the GMIN table, {label}",
                   "60" if label == "first block" else "12", float(len(shared)))
    audit.said(SI, "On all 60 seeds this code takes the same calls to the same lowest energy as\n"
                   "its archived rows, and a replay of each run gives the same record of accepted\n"
                   "minima and the same lowest energy.\n"
                   "All 60 seeds are rows of this code printed in the GMIN table.")
    audit.said(SI, "Commit \\texttt{\\seqsplit{e578f267b3826cfd7e0f1737515bc45f1b8a72c6}} of \\path{record/g3-funnel}")
    audit.said(SI, "removes no line of \\path{src}, and runs with the record on.")


TIP4P_GMIN = "aac1548a8aec13b53ad12a8d6904625081685d2e"


def gmin_sources(audit: Audit) -> None:
    """Every GMIN row behind a printed GMIN number names its identifier and machine."""
    machine = re.compile(r"[0-9a-f]{64}")
    shard_b = {(r["system"], r["seed"]): r["machine_id_sha256"]
               for r in audit.rows(f"{GMIN_RERUN}/b/host.csv") if r["code"] == "gmin"}
    sources = [(f"{GMIN_RERUN}/{shard}/rows.csv", "code", "code_revision", "host", GMIN_RERUN_ROW_ID)
               for shard in GMIN_SHARDS]
    sources += [(f"{GMIN_DEVELOP}/rows.csv", "code", "code_revision", "host", GMIN_RERUN_ROW_ID),
                ("paper/jctc/data/tip4p_gmin/rows.csv", "arm", "gmin_revision", "machine_id", TIP4P_GMIN)]
    for path, code, revision, host, want in sources:
        rows = [r for r in audit.rows(path) if r[code] == "gmin"]
        revisions = sorted({r[revision] for r in rows})
        audit.cell(path, "historical identifier of every GMIN row", want, " ".join(revisions))
        hosts = {shard_b.get((r["system"], r["seed"]), "") if r[host] == "bench" and path.endswith("/b/rows.csv")
                 else r[host] for r in rows}
        named = all(machine.fullmatch(h) for h in hosts) and bool(rows)
        audit.record(named, path, "every GMIN row names its machine", "sha256 of the machine id",
                     "all" if named else ", ".join(sorted(h[:8] or "none" for h in hosts)))
    tip4p = audit.rows("paper/jctc/data/tip4p_gmin/rows.csv")
    audit.cell("paper/jctc/data/tip4p_gmin/rows.csv",
               "TIP4P rows carrying the historical GMIN identifier", "120", len(tip4p))
    audit.record(all(r["gmin_revision"] == TIP4P_GMIN for r in tip4p),
                 "paper/jctc/data/tip4p_gmin/rows.csv",
                 "every TIP4P row carries the historical GMIN identifier",
                 TIP4P_GMIN, "all" if tip4p else "no rows")
    pinned = []
    for shard, name in (("a", "rows.csv"), ("b", "host.csv"),
                        ("c", "rows.csv"), ("d", "rows.csv")):
        path = f"{GMIN_RERUN}/{shard}/{name}"
        pinned.extend((path, row) for row in audit.rows(path)
                      if row.get("tarball_sha256", ""))
    audit.cell(SI, "rows carrying the dated GMIN tarball pin", "216", len(pinned))
    audit.record(all(row["tarball_sha256"] == GMIN_TARBALL_SHA256
                     and row["svnrev"] == GMIN_TARBALL_SVNREV
                     and row.get("gmin_snapshot", row.get("snapshot", "")) == GMIN_TARBALL_DATE
                     for _, row in pinned),
                 ", ".join(sorted({path for path, _ in pinned})),
                 "tarball rows agree on date, sha256 and source identifier",
                 f"{GMIN_TARBALL_DATE} {GMIN_TARBALL_SHA256} {GMIN_TARBALL_SVNREV}",
                 "all" if pinned else "no rows")
    audit.said(SI, "The Lennard--Jones, Morse, and Gupta rows whose source fields include a\n"
                   "tarball pin record the public tarball dated 2026-11-19, sha256\n"
                   "\\texttt{\\seqsplit{ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776}},\n"
                   "and the GMIN source identifier stored in the rows' \\texttt{svnrev} field,\n"
                   "\\texttt{\\seqsplit{9cbec1a446352bc9ef064bd2b1a5767194655919}}.")
    audit.said(SI, "The 120 rigid TIP4P rows instead record GMIN source\n"
                   "\\texttt{\\seqsplit{aac1548a8aec13b53ad12a8d6904625081685d2e}} in their\n"
                   "\\texttt{gmin\\_revision} field.")


MOVE_CHOICE = re.compile(r"Thompson|move choice|choice among the moves|choosing among the moves|posterior", re.I)
MOVE_CHOICE_CLAIM = re.compile(r"\b(drives?|driven|driving|improv\w*|help\w*|responsib\w*|key to|enabl\w*|boost\w*|explain\w*|"
                               r"account\w* for|behind the|leads? to|raises?|lifts?|increas\w*)\b", re.I)


def no_move_choice_claim(audit: Audit) -> None:
    """No sentence credits the adaptive choice among the moves with a hit or a gain."""
    for path in (TEX, SI, COVER):
        sentences = re.split(r"(?<=[.;])\s+", collapse(audit.text(path)))
        claims = [s[:80] for s in sentences if MOVE_CHOICE.search(s) and MOVE_CHOICE_CLAIM.search(s)]
        audit.record(not claims, path, "no sentence credits the move choice with success", "none",
                     "; ".join(claims) or "none")


RESTART_DRIVERS = ("examples/h2h_gmin_proof.rs", "examples/h2h_morse.rs", "examples/h2h_metal.rs",
                   "examples/h2h_seed.rs", "examples/h2h_sc.rs", "examples/ase_panel.rs", "examples/ase_nonlj.rs")


def no_anneal_restart(audit: Audit) -> None:
    """The paper claims no restart of this code, and no printed revision restarts a walk under its preset."""
    for path in (TEX, SI, COVER):
        sentences = re.split(r"(?<=[.;])\s+", collapse(audit.text(path)))
        loose = [s[:80] for s in sentences if re.search(r"restart", s, re.I) and not re.search(r"GMIN|NEWRESTART", s)]
        audit.cell(path, "every restart the text names is GMIN's", "none", " | ".join(loose) or "none")
    knockout = KNOCKOUT_CODE["templates"]
    for revision in sorted({r for rs in BENCHMARK_REVISIONS.values() for r in rs} | set(knockout)):
        where = f"{revision[:8]}:{PRIOR_CONFIG}"
        config = git("show", f"{revision}:{PRIOR_CONFIG}").stdout
        code = "\n".join(line for line in config.splitlines() if not line.lstrip().startswith("//"))
        audit.record("restart_on_stall: false," in code, where, "restart_on_stall is off in Config::default",
                     "off", "off" if "restart_on_stall: false," in code else "not off")
        setters = re.findall(r"^.*\.restart_on_stall = true;", code, re.M)
        audit.cell(where, "restart_on_stall is turned on only by with_dual_orbit_core", "1", float(len(setters)))
        users = [line.strip() for line in code.splitlines() if ".with_dual_orbit_core()" in line]
        audit.cell(where, "only the opt-in dual-core preset builds on with_dual_orbit_core",
                   "Self::for_cluster(n_points).with_dual_orbit_core()", " | ".join(users))
        for driver in RESTART_DRIVERS:
            text = git("show", f"{revision}:{driver}").stdout
            hit = bool(re.search(r"restart_on_stall\s*=\s*true|with_dual_orbit_core\(|for_cluster_dual_core\(", text))
            audit.record(not hit, f"{revision[:8]}:{driver}", "the driver turns on no restart", "none",
                         "sets one" if hit else "none")


PRIORS = ("soap", "reseed", "growth", "core", "handover", "motif")
PRIOR_CONFIG = "src/methods/cluster_hopping/config.rs"


def priors_on(revision: str) -> list[str] | None:
    """The structure priors the preset at ``revision`` sets on, or None when the revision is not here."""
    shown = git("show", f"{revision}:{PRIOR_CONFIG}")
    if shown.returncode != 0:
        return None
    return [p for p in PRIORS if f"prior_{p}: true," in shown.stdout]


def gmin_patches(audit: Audit) -> None:
    """The GMIN of the panel carries the cap and timer patches and nothing else of substance."""
    metal = audit.text("examples/h2h_metal.rs")
    morse = audit.text("examples/h2h_morse.rs")
    stride = re.search(r"cfg\.basin_stride = .*?\.unwrap_or\((\d+)\)", metal, re.S)
    audit.cell("examples/h2h_metal.rs", "basin stride", "8", float(stride.group(1)) if stride else None)
    copper = re.search(
        r'"au55" => \(\s*55,\s*[-\d.]+,\s*[\d_]+,\s*Kind::CopperGold \{ n_copper: (\d+) \}',
        metal,
    )
    audit.cell("examples/h2h_metal.rs", "Au55 copper atoms", "0", float(copper.group(1)) if copper else None)
    audit.record("second_neighbour_energy() <" not in morse, "examples/h2h_morse.rs",
                 "long-ranged threshold check is removed", "absent",
                 "absent" if "second_neighbour_energy() <" not in morse else "present")
    strides = re.findall(r"(\S*)basin_stride\s*=", metal)
    audit.cell("examples/h2h_metal.rs", "the basin stride is set only on the config of this code", "cfg.",
               " ".join(sorted(set(strides))))
    audit.record("Config::recommended(spec.n)" in metal and "gmin" not in metal.lower(), "examples/h2h_metal.rs",
                 "the gold driver builds a config of this code and never runs GMIN", "this code only",
                 "this code only" if "gmin" not in metal.lower() else "mentions GMIN")
    build = audit.text("scripts/gmin_panel/build_gmin.sh")
    listed = re.search(r"PATCHES=\((.*?)\)", build, re.S)
    names = re.findall(r'"\$ROOT/([^"]+)"', listed.group(1)) if listed else []
    want = ["bench/lj104-lattice/v2/h2h-gmin.patch", "scripts/morse_h2h/morse-timer.patch",
            "bench/metal-cluster-h2h/gmin-bgupta.patch"]
    audit.cell("scripts/gmin_panel/build_gmin.sh", "GMIN patches", " ".join(want), " ".join(names))
    audit.said(SI, ", ".join(f"\\path{{{n}}}" for n in want[:2]) + f", and \\path{{{want[2]}}}")
    audit.said(SI, f"tarball pin record the public tarball dated {GMIN_TARBALL_DATE}, sha256\n"
                   f"\\texttt{{\\seqsplit{{{GMIN_TARBALL_SHA256}}}}},")
    reads_pin = all(line in build for line in (
        "PIN=$ROOT/scripts/gmin_pin.py", 'eval "$(python3 "$PIN" shell)"', "SNAPSHOT=$GMIN_PIN_SNAPSHOT",
        "REV=$GMIN_PIN_REVISION", "TARBALL_SHA256=$GMIN_PIN_TARBALL_SHA256"))
    audit.record(reads_pin, "scripts/gmin_panel/build_gmin.sh", "the GMIN build reads the one shared pin",
                 "scripts/gmin_pin.py", "scripts/gmin_pin.py" if reads_pin else "other")
    audit.cell("scripts/gmin_pin.py", "the shared pin is the dated tarball and the printed source identifier",
               f"{GMIN_TARBALL_DATE} {GMIN_TARBALL_SHA256} {GMIN_TARBALL_SVNREV}",
               f"{GMIN_PIN_SNAPSHOT} {GMIN_PIN_TARBALL_SHA256} {GMIN_PIN_REVISION}")
    audit.record("JP=1" in audit.text(want[0]), want[0], "the first patch sets the serial run index", "JP=1",
                 "JP=1" if "JP=1" in audit.text(want[0]) else "absent")
    audit.record("(\\texttt{JP=1})" in audit.text(SI), SI,
                 "the GMIN provenance names the serial run index", "JP=1",
                 "JP=1" if "(\\texttt{JP=1})" in audit.text(SI) else "absent")
    audit.record(all(f"\\path{{{name}}}" in audit.text(SI) for name in want), SI,
                 "the GMIN provenance names every patch", "all",
                 "all" if all(f"\\path{{{name}}}" in audit.text(SI) for name in want) else "incomplete")
    allowed = {"#ifndef GMIN_MPI", "JP=1", "#endif", "IF (SAVECT) COMPRESST=.TRUE."}
    # What a capped call does instead of evaluating: return a refused energy and stop the quench.
    refusal = {"RETURN", "MFLAG=.FALSE.", "NPCALL = NPCALL - 1", "EREAL = 1.0D20", "IF (GRADT) GRAD(:) = 0.0D0",
               "IF (ENERGY.GT.1.0D19) ENERGY=1.0D6"}
    for name in names:
        text = audit.text(name)
        removed = {line[1:].strip() for line in text.splitlines() if line.startswith("-") and not line.startswith("---")}
        target = ""
        stray = []
        block: list[str] | None = None
        for line in text.splitlines():
            if line.startswith("+++ "):
                target = line[4:]
                continue
            if not line.startswith("+") or target == "b/source/h2h_cap.f90":
                continue
            body = line[1:].strip()
            if block is not None:
                if body != "END IF":
                    block.append(body)
                    continue
                head, rest = block[0], block[1:]
                guarded = "H2H" in head or any("H2H" in b for b in rest)
                rewritten = head.endswith(" THEN") and head[:-len(" THEN")] + " RETURN" in removed
                plain = [b for b in rest if "H2H" not in b]
                if not (guarded and (rewritten or "H2H" in head) and set(plain) <= refusal):
                    stray.append(" / ".join(block))
                block = None
                continue
            if re.fullmatch(r"IF \(.*\) THEN", body):
                block = [body]
            elif body and "H2H" not in body and body not in allowed:
                stray.append(body)
        audit.cell(name, "added GMIN lines outside the cap and timers", "none", "; ".join(stray) or "none")
    audit.said(SI, "which cap the \\texttt{POTENTIAL} entries, time the potential, and, in the serial build, set the run")
    audit.record("Mackay cap" not in audit.text(TEX), TEX, "Mackay-cap driver prose is removed",
                 "absent", "absent" if "Mackay cap" not in audit.text(TEX) else "present")


ASE_PANEL = ("lj38", "lj55", "lj98", "lj104", "au55")
MORSE_PANEL = ("m38r6", "m38r14", "m55r6", "m55r14")
ASE_ARMS = ("ase_bh", "ase_mh")
ASE_DATA = "paper/jctc/data/ase_panel"
ASE_REVISIONS = {"lj": "61fa67cf380e46876b569da45e7cc091647a775b",
                 "morse": "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"}
PANEL_SEEDS = " ".join(str(s) for s in range(2000, 2012))


def load_script(name: str, path: str) -> object:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ase_groups(audit: Audit) -> dict[tuple[str, str], list[dict[str, str]]]:
    """The rows of each system and arm of the two ASE tables, in seed order."""
    groups = {}
    expected_blob = git(
        "rev-parse",
        "61fa67cf380e46876b569da45e7cc091647a775b:scripts/ase_panel/run_one.py",
    ).stdout.strip()
    ase_hosts: set[str] = set()
    folders = [(s, (s,), ("anneal", "recommended", *ASE_ARMS), "lj") for s in ASE_PANEL]
    folders.append(("morse", MORSE_PANEL, ("for_cluster", "recommended", *ASE_ARMS), "morse"))
    for folder, systems, arms, kind in folders:
        rows = audit.rows(f"{ASE_DATA}/{folder}/rows.csv")
        if kind == "lj":
            ase_hosts.update(r["host"] for r in rows)
            blobs = sorted({r["harness_blob"] for r in rows})
            audit.cell(f"ase panel {folder}", "harness_blob", " ".join(blobs), expected_blob)
        for system in systems:
            for arm in arms:
                group = sorted(pick(rows, system=system, arm=arm), key=lambda r: int(r["seed"]))
                where = f"ase panel {system} {arm}"
                audit.cell(where, "seeds", " ".join(r["seed"] for r in group), PANEL_SEEDS)
                audit.cell(where, "revision", " ".join(sorted({r["code_revision"] for r in group})),
                           ASE_REVISIONS[kind])
                groups[(system, arm)] = group
    audit.record(
        len(ase_hosts) == 1
        and bool(re.fullmatch(r"[0-9a-f]{64}", next(iter(ase_hosts)))),
        "ase panel",
        "single recorded host digest",
        "one sha256",
        " ".join(sorted(ase_hosts)),
    )
    ase_host = next(iter(ase_hosts)) if len(ase_hosts) == 1 else ""
    for (system, arm), group in groups.items():
        want_host = "bench" if system in MORSE_PANEL else ase_host
        audit.cell(
            f"ase panel {system} {arm}",
            "host",
            " ".join(sorted({r["host"] for r in group})),
            want_host,
        )
    return groups


def ase_cap(audit: Audit, groups: dict, system: str) -> float:
    """The one wall cap of a cluster, checked to be three times the slowest cluster-preset wall."""
    base = "for_cluster" if system in MORSE_PANEL else "anneal"
    caps = sorted({r["cap_s"] for arm in ("recommended", *ASE_ARMS) for r in groups[(system, arm)]})
    slowest = max(float(r["wall_s"]) for r in groups[(system, base)])
    audit.cell(f"ase panel {system}", "one wall cap", "1", float(len(caps)))
    audit.record(abs(float(caps[0]) - 3.0 * slowest) <= 1e-5, f"ase panel {system}", "cap is three times the slowest "
                 f"{base} wall", caps[0], f"{3.0 * slowest:.6f}")
    return float(caps[0])


def manuscript_ase(audit: Audit) -> None:
    """Tables tab:ase and tab:ase-morse, the imported rows they come from, and every sentence that reads them."""
    table = load_script("ase_table_audit", "paper/jctc/scripts/ase_table.py")
    for folder in (*ASE_PANEL, "morse"):
        path = imported(audit, f"{ASE_DATA}/{folder}")
        fields = dict(line.split(" ", 1) for line in audit.text(f"{ASE_DATA}/{folder}/PROVENANCE").splitlines())
        ancestor_of(audit, path, fields["ancestor"])
        audit.cell(path, "every row is the PROVENANCE ancestor", fields["ancestor"],
                   " ".join(sorted({r["code_revision"] for r in audit.rows(path)})))
    audit.record(not (ROOT / ASE_DATA / "lj150").exists(), ASE_DATA, "LJ150 has no imported rows", "absent", "checked")
    groups = ase_groups(audit)

    def hits(system: str, arm: str) -> int:
        return hits_of(groups[(system, arm)])

    def stops(system: str, arm: str, stop: str = "wall") -> int:
        return sum(r["stop"] == stop for r in groups[(system, arm)])

    budgets = {s: v[1] for s, v in (table.LJ | table.MORSE).items()}
    references = {s: v[2] for s, v in (table.LJ | table.MORSE).items()}
    by_atoms = {38: 400_000, 55: 1_000_000}
    atoms = {"lj38": 38, "lj55": 55, "lj98": 98, "lj104": 104, "au55": 55,
             "m38r6": 38, "m38r14": 38, "m55r6": 55, "m55r14": 55}
    for (system, arm), group in groups.items():
        audit.cell(f"ase panel {system} {arm}", "budget", " ".join(sorted({r["budget"] for r in group})),
                   str(budgets[system]))
        audit.cell(f"ase panel {system} {arm}", "budget by atom count", str(by_atoms.get(atoms[system], 4_000_000)),
                   float(budgets[system]))
        worst = max((abs(float(r["best"]) - references[system]) for r in group if r["hit"] == "1"), default=0.0)
        audit.record(worst < DEFAULT_HIT_TOL, f"ase panel {system} {arm}", "every hit within 1e-3 of its reference",
                     f"{worst:.3g}", "below 0.001")
    for system in (*ASE_PANEL, *MORSE_PANEL):
        audit.cell(f"ase panel {system}", "recommended wall stops", "0", float(stops(system, "recommended")))

    for label, systems, labels in (("tab:ase", ASE_PANEL, table.LJ), ("tab:ase-morse", MORSE_PANEL, table.MORSE)):
        by_label = {v[3]: s for s, v in labels.items()}
        shown = []
        for cells in tex_table(audit, TEX, label):
            system = by_label.get(cells[0])
            where = f"{TEX} {label} {system}"
            if system is None or len(cells) != 11:
                audit.record(False, where, "table line", " & ".join(cells), "a panel system")
                continue
            shown.append(system)
            audit.cell(where, "budget", cells[1], float(budgets[system]))
            audit.cell(where, "cap", cells[2], ase_cap(audit, groups, system))
            for cell, arm in zip(cells[3:6], ("recommended", *ASE_ARMS)):
                audit.cell(where, f"{arm} hits", cell, frac(groups[(system, arm)]))
            for cell, arm in zip(cells[6:8], ASE_ARMS):
                audit.cell(where, f"{arm} wall stops", cell, float(stops(system, arm)))
            for cell, arm in zip(cells[8:11], ("recommended", *ASE_ARMS)):
                audit.cell(where, f"{arm} median wall", cell, median([float(r["wall_s"]) for r in groups[(system, arm)]]))
        audit.cell(TEX, f"{label} lines", " ".join(shown), " ".join(systems))

    audit.said(TEX, "\\caption{The submitted search against the basin hopping (BH) and minima hopping\n"
                    "(MH) of ASE 3.29.0 at its constructor defaults, at the effort each run reports.")
    audit.said(TEX, "Table~\\ref{tab:ase} runs the submitted revision on four Lennard-Jones\n"
                    "clusters and Au\\(_{55}\\), and Table~\\ref{tab:ase-morse} runs an ancestor of\n"
                    "that revision on four Morse clusters.")
    audit.said(TEX, "These runs do not share a start with this code, and their effort is not equal: every run stopped "
                    "at a hit, at the budget, or at the recorded wall cap shown in the table.")
    audit.said(TEX, "Cap is the wall in seconds after which a run stops at its next check, three times\n"
                    "the slowest wall of the cluster preset of this code on that cluster.")
    mh_all = all(r["stop"] == "wall" for s in (*ASE_PANEL, *MORSE_PANEL) for r in groups[(s, "ase_mh")])
    audit.record(mh_all, TEX, "every ASE minima-hopping run ended at the cap", "every", "every" if mh_all else "not")
    audit.said(TEX, "ASE minima hopping is wall-capped: every one of its runs in both tables ended at the cap with "
                    "fewer calls than the budget.")
    bh_lj = sum(stops(s, "ase_bh") for s in ASE_PANEL)
    runs_lj = sum(len(groups[(s, "ase_bh")]) for s in ASE_PANEL)
    bh_morse = all(r["stop"] == "wall" for s in MORSE_PANEL for r in groups[(s, "ase_bh")])
    audit.record(bh_morse, TEX, "basin hopping reached the cap on every Morse run", "every", "every" if bh_morse else "not")
    audit.said(TEX, f"Basin hopping reached the cap on {bh_lj} of the {runs_lj} runs of Table~\\ref{{tab:ase}} and on "
                    "every run of Table~\\ref{tab:ase-morse}, and no run of this code reached it.")
    r = {s: hits(s, "recommended") for s in (*ASE_PANEL, *MORSE_PANEL)}
    audit.said(TEX, f"Under that policy this code hits LJ\\(_{{38}}\\) and LJ\\(_{{55}}\\) on {r['lj38']} of 12, "
                    f"LJ\\(_{{98}}\\) on {r['lj98']} of 12, and LJ\\(_{{104}}\\) on {r['lj104']} of 12, while ASE basin "
                    f"hopping hits LJ\\(_{{55}}\\) on {hits('lj55', 'ase_bh')} of 12 and no other cluster, and ASE "
                    "minima hopping hits none; no arm hits Au\\(_{55}\\).")
    audit.cell(TEX, "LJ38 and LJ55 share one count", "12 12", f"{r['lj38']} {r['lj55']}")
    bh_hits = [s for s in ASE_PANEL if hits(s, "ase_bh")]
    audit.cell(TEX, "basin hopping hits one cluster", "lj55", " ".join(bh_hits))
    audit.cell(TEX, "minima hopping hits none", "0", float(sum(hits(s, "ase_mh") for s in ASE_PANEL)))
    audit.cell(TEX, "no arm hits Au55", "0", float(sum(hits_of(groups[("au55", a)]) for a in
                                                       ("anneal", "recommended", *ASE_ARMS))))
    audit.cell(TEX, "Morse at rho=6 on 12 of 12", "12 12", f"{r['m38r6']} {r['m55r6']}")
    morse_ase = [r_ for s in MORSE_PANEL for a in ASE_ARMS for r_ in groups[(s, a)]]
    audit.cell(TEX, "Morse ASE hits", "0", float(hits_of(morse_ase)))
    charged = [int(r_["charged"]) for r_ in morse_ase]
    audit.said(TEX, f"On the Morse clusters this code hits M\\(_{{38}}\\) and M\\(_{{55}}\\) at \\(\\rho=6\\) on "
                    f"{r['m38r6']} of 12 and at \\(\\rho=14\\) on {r['m38r14']} and {r['m55r14']} of 12, and neither "
                    f"ASE arm hits any; each ASE run there stopped at the cap after {min(charged)} to {max(charged)} "
                    "calls.")

    # The Supporting Information.
    audit.said(SI, "on seeds 2000--2011, at \\(4\\times 10^{5}\\) calls for 38 atoms, \\(1\\times 10^{6}\\) for 55, "
                    "and \\(4\\times 10^{6}\\) for 98 atoms and above.")
    audit.said(SI, "The budget of this code counts one call per combined energy and gradient pass.")
    ase_protocol = audit.text(SI).split("\\label{sec:si-ase}", 1)[1].split("\\section{", 1)[0]
    host_claim = "one core of one host" in ase_protocol
    audit.record(not host_claim, SI, "the ASE protocol names no single host", "absent",
                 "present" if host_claim else "absent")
    morse_caption = audit.text(TEX).split("% ase-nonlj table: start", 1)[1].split("% ase-nonlj table: end", 1)[0]
    morse_host = "one core of one host" in morse_caption
    audit.record(not morse_host, TEX, "the Morse ASE caption names no single host", "absent",
                 "present" if morse_host else "absent")
    audit.said(SI, "Environment 3.29.0 at its constructor defaults")
    audit.said(SI, "\\texttt{\\seqsplit{" + ASE_REVISIONS["lj"] + "}}, and those of\n"
                   "the Morse ASE table revision\n"
                   "\\texttt{\\seqsplit{" + ASE_REVISIONS["morse"] + "}}, at the length\n"
                   "scale of the Morse pair.\n"
                   "The former is the submitted revision, and the latter is its ancestor.")
    audit.said(SI, "Each run stops at a hit, at the budget, or at its next check after a wall cap\n"
                   "of three times the slowest wall of the cluster preset of this code on that\n"
                   "cluster.")
    preset = [str(hits(s, "anneal")) for s in ASE_PANEL]
    morse_preset = [str(hits(s, "for_cluster")) for s in MORSE_PANEL]
    audit.said(SI, f"Its hits are {series(preset)} of 12 on LJ\\(_{{38}}\\), LJ\\(_{{55}}\\), LJ\\(_{{98}}\\), "
                   f"LJ\\(_{{104}}\\), and Au\\(_{{55}}\\), and {series(morse_preset)} of 12 on M\\(_{{38}}\\) and "
                   "M\\(_{55}\\) at \\(\\rho=6\\) and 14 in the order of the Morse table.")
    bh_all = [s for s in (*ASE_PANEL, *MORSE_PANEL) if stops(s, "ase_bh") == 12]
    audit.cell(SI, "basin hopping at the cap on every run", "lj38 au55 m38r6 m38r14 m55r6 m55r14", " ".join(bh_all))
    audit.said(SI, f"Minima hopping reached its cap on every run of both tables, and basin hopping on every run of "
                   f"LJ\\(_{{38}}\\), Au\\(_{{55}}\\), and the Morse clusters and on {stops('lj55', 'ase_bh')} of 12 on "
                   "LJ\\(_{55}\\).")
    spent = [s for s in ASE_PANEL if stops(s, "ase_bh", "budget") == 12]
    audit.cell(SI, "basin hopping spent the budget on every seed", "lj98 lj104", " ".join(spent))
    audit.said(SI, "On LJ\\(_{98}\\) and LJ\\(_{104}\\) basin hopping spent the full \\(4\\times 10^{6}\\) calls on "
                   "every seed and missed on every seed.")
    gaps = [f"{median([float(r_['best']) for r_ in groups[('au55', a)]]) - references['au55']:.3f}"
            for a in ("recommended", *ASE_ARMS)]
    audit.said(SI, f"On Au\\(_{{55}}\\) the median best energy lies {gaps[0]}~eV above the reference for this code, "
                   f"{gaps[1]}~eV for basin hopping, and {gaps[2]}~eV for minima hopping.")
    audit.said(SI, "No run of this code reached its cap.")
    audit.said(SI, "the random generators of the two codes do not make a shared start from one seed")
    audit.said(SI, "\\path{paper/jctc/scripts/ase_table.py} writes both ASE tables from them.")
    audit.said(SI, "# ASE: rewrite both tables from the imported rows\npython3 paper/jctc/scripts/ase_table.py")

    # Shared numeric ceilings and cap-censored outcomes are printed without a cross-arm test.
    text = audit.text(TEX)
    section = text.split("\\label{sec:ase}", 1)[1].split("% ase-nonlj table: end", 1)[0]
    audit.record("p_\\mathrm" not in section and "Fisher" not in section, TEX,
                 "sec:ase reports no cross-arm p-values", "none",
                 "none" if "p_\\mathrm" not in section and "Fisher" not in section else "present")
    conclusions = text.split("\\section{Conclusions}", 1)[1].split("\\appendix", 1)[0]
    audit.record("Mann" not in conclusions, TEX, "the conclusions report no rank test", "no rank test",
                 "no rank test" if "Mann" not in conclusions else "a rank test")
    audit.said(TEX, "Within a cluster every arm has the same numeric call ceiling and wall cap,\n"
                    "but runs reaching the shared cap are censored.")
    audit.said(TEX, "Because the starts are unpaired and cap-censored outcomes occur, no\n"
                    "cross-arm significance test is reported for the ASE tables.")
    audit.said(TEX, "No cross-arm test is reported: starts differ; every arm has the same numeric call\n"
                    "ceiling and wall cap for that cluster, and runs reaching the shared cap are censored.")
    audit.said(SI, "Every arm within a cluster has the same numeric call ceiling and wall cap.")
    audit.said(SI, "\\emph{Unpaired, equally capped arms.}")
    audit.said(SI, "The tables therefore report the hit counts, recorded caps, stop counts, and")
    for path in (TEX, SI):
        text = audit.text(path)
        false_limits = re.search(r"(?:different|unequal) call and wall limits|call limits and wall caps also differ",
                                 text)
        audit.record(false_limits is None, path, "ASE limits are not described as unequal", "absent",
                     "absent" if false_limits is None else false_limits.group(0))
    audit.record("fisher_exact" not in audit.text("paper/jctc/scripts/ase_table.py"),
                 "paper/jctc/scripts/ase_table.py", "the ASE table generator computes no cross-arm test",
                 "absent", "absent" if "fisher_exact" not in audit.text("paper/jctc/scripts/ase_table.py") else "present")
    for path in ("scripts/ase_panel/summarize.py", "scripts/ase_nonlj/summarize.py"):
        source = audit.text(path)
        audit.record("mann_whitney" not in source and "continuity" not in source, path,
                     "no continuity-corrected rank test", "absent",
                     "absent" if "mann_whitney" not in source else "present")
    for gone in ("gofee", "AGOX", "1ce3ab64", "b49341e3", "equal-call"):
        found = [path for path in (TEX, SI, "paper/jctc/cover-letter.tex") if gone in audit.text(path)]
        audit.record(not found, " ".join(found) or TEX, f"no {gone} in the paper", "absent", " ".join(found) or "absent")
    step = 1e-4

    def curvature(pair) -> float:
        return (pair(1.0 + step) - 2.0 * pair(1.0) + pair(1.0 - step)) / step**2

    morse6 = curvature(lambda x: morse_pair(6.0, x))
    lennard_jones = curvature(lambda x: x**-12 - 2.0 * x**-6)
    audit.record(abs(morse6 - lennard_jones) < 1e-3, TEX, "Morse rho=6 curvature at the pair minimum",
                 f"{lennard_jones:.3f}", f"{morse6:.3f}")


GMIN_RERUN = "paper/jctc/data/gmin_eeec8e2c"
GMIN_SHARDS = ("a", "b", "c", "d")


def panel_references(audit: Audit) -> None:
    """The panel references of Section sec:hit, and every panel hit inside the window of equation eq:abs."""
    gmin = load_script("gmin_panel_refs", "scripts/gmin_panel/run_panel.py")
    ase = load_script("ase_table_refs", "paper/jctc/scripts/ase_table.py")
    refs: dict[str, set[float]] = defaultdict(set)
    for row in gmin.SYSTEMS:
        refs[row[0]].add(row[3])
    for system, spec in (ase.LJ | ase.MORSE).items():
        refs[system].add(spec[2])
    for system, values in sorted(refs.items()):
        audit.cell("panel references", f"{system} one reference", "1", float(len(values)))
    ref = {s: next(iter(v)) for s, v in refs.items()}
    for path, rows in (("bench/gmin-panel/rows.csv", audit.rows("bench/gmin-panel/rows.csv")),
                       ("bench/ase-nonlj/rows.csv", audit.rows("bench/ase-nonlj/rows.csv")),
                       *((f"bench/ase-{s}/rows-61fa67cf.csv", audit.rows(f"bench/ase-{s}/rows-61fa67cf.csv")) for s in ASE_PANEL),
                       *((f"bench/ase-{s}/rows.csv", audit.rows(f"bench/ase-{s}/rows.csv")) for s in ASE_PANEL),
                       ("bench/ase-au55/rows-for-cluster.csv", audit.rows("bench/ase-au55/rows-for-cluster.csv")),
                       *((f"{ASE_DATA}/{s}/rows.csv", audit.rows(f"{ASE_DATA}/{s}/rows.csv"))
                         for s in (*ASE_PANEL, "morse")),
                       *((f"{GMIN_RERUN}/{shard}/rows.csv", audit.rows(f"{GMIN_RERUN}/{shard}/rows.csv"))
                         for shard in GMIN_SHARDS)):
        worst = max((abs(float(r["best"]) - ref[r["system"]]) for r in rows if hits_of([r])), default=0.0)
        audit.record(
            worst < DEFAULT_HIT_TOL,
            path,
            "every hit within 1e-3 of its reference",
            f"{worst:.3g}",
            "below 0.001",
        )

    def f6(system: str) -> str:
        return f"\\({ref[system]:.6f}\\)"

    audit.said(TEX, f"The other tables add {f6('lj102')} at 102 atoms, "
                    f"{f6('lj103')} at 103, and {f6('lj150')} at 150, \\({ref['au55']:.8f}\\)~eV for gold at 55 atoms "
                    f"in the copper--gold Gupta archive, and from the Morse table {f6('m38r6')} and {f6('m38r14')} for "
                    f"M\\(_{{38}}\\) at \\(\\rho=6\\) and \\(14\\), {f6('m55r6')} and {f6('m55r14')} for M\\(_{{55}}\\), "
                    f"and {f6('m80r6')} and {f6('m80r14')} for M\\(_{{80}}\\).")
    for system, text in (("lj38", "\\(-173.928427\\) at 38"), ("lj55", "\\(-279.248470\\) at 55"),
                         ("lj75", "\\(-397.492331\\) at 75"), ("lj98", "\\(-543.665361\\) at"),
                         ("lj104", "\\(-582.086642\\) at 104")):
        audit.cell(TEX, f"{system} reference", re.search(r"\\\((-[\d.]+)\\\)", text).group(1), ref[system])
        audit.said(TEX, text)


def generated_tables(audit: Audit) -> None:
    """Each panel table in the manuscript is the block its summariser writes from the committed rows."""
    text = audit.text(TEX)

    def block(marks: tuple[str, str]) -> str:
        start, end = marks
        if text.count(start) != 1 or text.count(end) != 1:
            return ""
        return text.split(start, 1)[1].split(end, 1)[0].strip("\n")

    gmin = load_script("gmin_table_tex", "paper/jctc/scripts/gmin_table.py")
    knock = load_script("knockout_tables_tex", "paper/jctc/scripts/knockout_tables.py")
    ase = load_script("ase_table_tex", "paper/jctc/scripts/ase_table.py")
    era = load_script("template_era_table_tex", "paper/jctc/scripts/template_era_table.py")
    for marks, path in ((("% gofee table: start", "% gofee table: end"), TEX),
                        (("% gofee-si table: start", "% gofee-si table: end"), SI)):
        audit.record(marks[0] not in audit.text(path), path, "the GOFEE table is not printed", "absent",
                     "absent" if marks[0] not in audit.text(path) else "present")
    audit.record("tab:ablation" not in text, TEX, "the move-selection table is not printed", "absent",
                 "absent" if "tab:ablation" not in text else "present")
    for label, marks, written in (("tab:gmin", gmin.MARKS, gmin.table()),
                                  ("tab:knockouts", knock.KNOCKOUT_MARKS, knock.knockout_table()),
                                  ("tab:tip4p-gmin", knock.TIP4P_MARKS, knock.tip4p_table()),
                                  ("tab:ase", ase.MARKS["ase"], ase.ase_block()),
                                  ("tab:ase-morse", ase.MARKS["ase-morse"], ase.morse_block())):
        ok = block(marks) == written
        audit.record(ok, TEX, f"{label} is the summariser's block", "as written" if ok else "edited by hand", "as written")
    si = audit.text(SI)
    era_block = si.split(era.MARKS[0], 1)[1].split(era.MARKS[1], 1)[0].strip("\n") if si.count(era.MARKS[0]) == 1 else ""
    audit.record(era_block == era.table(), SI, "tab:si-template-era is the summariser's block",
                 "as written" if era_block == era.table() else "edited by hand", "as written")
    audit.record(era.MARKS[0] not in text, TEX, "the template-era table is not in the manuscript", "absent",
                 "absent" if era.MARKS[0] not in text else "present")
    for marks in (("% gmin-panel table: start", "% gmin-panel table: end"),):
        audit.record(text.count(marks[0]) == 1, TEX, "one GMIN block", "1", str(text.count(marks[0])))
    audit.record("% trap table: start" not in text, TEX, "the endpoint-energy trap table is not printed", "absent",
                 "absent" if "% trap table: start" not in text else "present")


def sutton_chen(audit: Audit) -> None:
    """The Sutton-Chen counts in the prose, from the rows the table is written from."""
    sc_run = load_script("run_sc_step", "scripts/metal_h2h/run_sc.py")
    step = f"{sc_run.sc_step():.6f}"
    audit.said("bench/metal-cluster-h2h/sc-n12/README.md", f"the step 0.40 σ in units of a, {step}")
    hits = load_script("sc96_step", "scripts/sc96_78_hits.py")
    audit.record(step in hits.result_paragraph([], "harness"), "scripts/sc96_78_hits.py",
                 "the Sutton-Chen step is derived", step, "present")
    h2h = load_script("h2h_tables_sc", "paper/jctc/scripts/h2h_tables.py")
    pairs = h2h.sc_pairs()
    ahead = sum(int(a["hits"]) > int(g["hits"]) for a, g in pairs)
    behind = sum(int(a["hits"]) < int(g["hits"]) for a, g in pairs)
    tied = sum(int(a["hits"]) == int(g["hits"]) for a, g in pairs)
    tests = h2h.sc_tests()
    significant = [a["system"] for a, g in pairs if float(tests[a["system"]]["fisher_p"]) < 0.05]
    gmin_faster = sum(float(g["wall_median_s"]) < float(a["wall_median_s"]) for a, g in pairs)
    ratios = [float(a["pot_per_eval_s"]) / float(g["pot_per_eval_s"]) for a, g in pairs]
    if h2h.sc_gmin_quotable():
        audit.cell(PAPER, "Sutton-Chen clusters where this setting hits more", str(ahead), float(ahead))
        audit.cell(PAPER, "Sutton-Chen clusters tied on hits", str(tied), float(tied))
        audit.cell(PAPER, "Sutton-Chen clusters where GMIN hits more", str(behind), float(behind))
        audit.cell(PAPER, "Sutton-Chen clusters with Fisher p below 0.05", "sc108-38 sc126-80 sc96-80",
                   " ".join(significant))
        audit.cell(PAPER, "Sutton-Chen clusters where GMIN has the lower median wall", str(gmin_faster),
                   float(gmin_faster))
        audit.cell(PAPER, "Sutton-Chen per-call ratio inside the potential, low",
                   f"{min(ratios):.1f}", min(ratios))
        audit.cell(PAPER, "Sutton-Chen per-call ratio inside the potential, high",
                   f"{max(ratios):.1f}", max(ratios))
    if h2h.sc_gmin_quotable():
        audit.said(PAPER, f"This work has more hits on {ahead} of the {len(pairs)} systems")
        audit.said(PAPER, f"the arms tie on {tied}, and GMIN has more hits on {behind}")
        audit.said(PAPER, f"GMIN nevertheless has the lower median raw wall time on {gmin_faster} of {len(pairs)} systems")
        audit.said(COVER, f"On {spelled(len(pairs))} Sutton--Chen clusters, this work has more hits on "
                          f"{spelled(ahead)}, ties GMIN on {spelled(tied)}, and GMIN has the lower median wall on "
                          f"{spelled(gmin_faster)}.")
    else:
        for path in (TEX, SI, COVER):
            text = audit.text(path)
            for phrase in (
                "not quoted",
                "commits this repository does not hold",
                "commit this repository does not hold",
                "commits are not in this repository",
            ):
                audit.record(
                    phrase not in text,
                    path,
                    f"no withholding narration ({phrase})",
                    "absent",
                    "present" if phrase in text else "absent",
                )
    sc_seeds = audit.rows("bench/metal-cluster-h2h/sc-n12/seeds.csv")
    audit.cell("bench/metal-cluster-h2h/sc-n12/seeds.csv", "Sutton-Chen seed rows",
               str(len(sc_seeds)), float(len(sc_seeds)))
    gmin_panel = load_script("gmin_panel_sc_fmt", "scripts/gmin_panel/run_panel.py")
    by_label = {h2h.sc_label(a["system"]): (a, g) for a, g in pairs}
    if "\\label{tab:sc}" not in audit.text(PAPER):
        sutton_chen_pvalues(audit)
        return
    for cells in tex_table(audit, PAPER, "tab:sc"):
        if cells[0] not in by_label:
            continue
        a_row, g_row = by_label[cells[0]]
        sys_name = a_row["system"]
        a_calls = [float(r["evals_to_hit"]) for r in sc_seeds
                   if r["system"] == sys_name and r["arm"] == "anneal"
                   and r["harness_revision"] == a_row["harness_revision"]]
        g_calls = [float(r["evals_to_hit"]) for r in sc_seeds
                   if r["system"] == sys_name and r["arm"] == "gmin"
                   and r["harness_revision"] == g_row["harness_revision"]]
        if h2h.sc_gmin_quotable():
            want_a = gmin_panel.show_calls(median(a_calls))
            want_g = gmin_panel.show_calls(median(g_calls))
            audit.record(cells[4] == want_a, f"{PAPER} tab:sc {sys_name}", "anneal median calls formatted like tab:gmin",
                         cells[4], want_a)
            audit.record(cells[5] == want_g, f"{PAPER} tab:sc {sys_name}", "GMIN median calls formatted like tab:gmin",
                         cells[5], want_g)
        else:
            want_a = h2h.show_calls_median(h2h.type7_median(a_calls))
            audit.record(cells[2] == a_row["hits_over_n"], f"{PAPER} tab:sc {sys_name}",
                         "anneal hits", cells[2] if len(cells) > 2 else "", a_row["hits_over_n"])
            audit.record(len(cells) > 3 and cells[3] == want_a, f"{PAPER} tab:sc {sys_name}",
                         "anneal median calls", cells[3] if len(cells) > 3 else "", want_a)
    sutton_chen_pvalues(audit)


def sutton_chen_pvalues(audit: Audit) -> None:
    """Each Sutton-Chen p-value row, recomputed from the anneal seeds and the corrected GMIN seeds alone.

    The results.csv rows it supersedes stay in place with their old values."""
    panel = "bench/metal-cluster-h2h/sc-n12"
    seeds = audit.rows(f"{panel}/seeds.csv")
    results = audit.rows(f"{panel}/results.csv")
    stored = audit.rows(f"{panel}/pvalues.csv")
    where = f"{panel}/pvalues.csv"
    systems = sorted({r["system"] for r in seeds})
    h2h = load_script("h2h_tables_sc_pvalues", "paper/jctc/scripts/h2h_tables.py")
    pairs = {anneal["system"]: (anneal, gmin) for anneal, gmin in h2h.sc_pairs()}
    latest = [
        row for row in stored
        if row["system"] in pairs
        and row["code_revision"] == pairs[row["system"]][0]["code_revision"]
        and row["harness_revision"] == pairs[row["system"]][1]["harness_revision"]
    ]
    audit.cell(where, "one latest p-value row per Sutton-Chen system",
               str(len(systems)), float(len(latest)))
    for row in latest:
        system = row["system"]
        anneal_summary, _gmin_summary = pairs[system]
        anneal = [
            r for r in seeds
            if r["system"] == system and r["arm"] == "anneal"
            and r["harness_revision"] == anneal_summary["harness_revision"]
        ]
        own = {
            r["harness_revision"] for r in seeds
            if r["system"] == system and r["arm"] == "anneal"
        }
        gmin = [r for r in seeds if r["system"] == system and r["arm"] == "gmin"
                and r["harness_revision"] == row["harness_revision"]]
        audit.record(row["harness_revision"] not in own and len(anneal) == 12 and len(gmin) == 12, where,
                     f"{system} is tested against the corrected GMIN rows only", "12 corrected rows",
                     f"{len(gmin)} rows of {row['harness_revision'][:8]}")
        hits_a = sum(r["hit"] == "1" for r in anneal)
        hits_g = sum(r["hit"] == "1" for r in gmin)
        audit.cell(where, f"{system} anneal hits", row["hits_anneal"], float(hits_a))
        audit.cell(where, f"{system} GMIN hits", row["hits_gmin"], float(hits_g))
        audit.cell(where, f"{system} Fisher p", row["fisher_p"], fisher(hits_a, len(anneal), hits_g, len(gmin)))
        for column, field in (("mannwhitney_p_evals", "evals_to_hit"), ("mannwhitney_p_wall", "wall_s")):
            audit.cell(where, f"{system} {column}", row[column],
                       mann_whitney([float(r[field]) for r in anneal], [float(r[field]) for r in gmin], True))
        old = [
            r for r in results
            if r["system"] == system and r["arm"] == "anneal"
            and r["harness_revision"] == anneal_summary["harness_revision"]
        ]
        named = (
            f"results.csv system={system} arm=anneal "
            f"harness_revision={anneal_summary['harness_revision'][:8]}:"
        )
        audit.record(len(old) == 1 and row["supersedes"].startswith(named), where,
                     f"{system} names the results.csv row it supersedes, which stays", named,
                     row["supersedes"] if old else "superseded row missing")
    for row in stored:
        system = row["system"]
        anneal = [
            r for r in seeds
            if r["system"] == system and r["arm"] == "anneal"
            and r["harness_revision"] == row["code_revision"]
        ]
        gmin = [
            r for r in seeds
            if r["system"] == system and r["arm"] == "gmin"
            and r["harness_revision"] == row["harness_revision"]
        ]
        if len(anneal) == 12 and len(gmin) == 12:
            audit.cell(where, f"{system} Fisher p", row["fisher_p"],
                       fisher(sum(r["hit"] == "1" for r in anneal), len(anneal),
                              sum(r["hit"] == "1" for r in gmin), len(gmin)))
        named = (
            f"results.csv system={system} arm=anneal "
            f"harness_revision={row['code_revision'][:8]}:"
        )
        kept = [
            r for r in results
            if r["system"] == system and r["arm"] == "anneal"
            and r["harness_revision"] == row["code_revision"]
        ]
        audit.record(len(kept) == 1 and row["supersedes"].startswith(named), where,
                     f"{system} names the results.csv row it supersedes, which stays", named,
                     row["supersedes"] if kept else "superseded row missing")


def spelled(count: int) -> str:
    """A count as ACS style writes it in running text: one through nine in words, larger counts as numerals."""
    return NUMBER_WORDS[count] if 0 < count < 10 else str(count)


SI_TIP = "ce80337b7507ca12a99913da7ee8606184d4a3e1"
SI_ARCHIVE = "paper/jctc/logs/lj75-source-37535ec1.tar.xz"


def plain(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def line_of(audit: Audit, path: str, number: int) -> str:
    return audit.text(path).splitlines()[number - 1]


def resolve_named_hit_tol(expr: str, names: dict[str, float]) -> float | None:
    """Resolve a tolerance expression through ``names`` or as a numeric literal.

    ``${DEFAULT_HIT_TOL}`` and ``DEFAULT_HIT_TOL`` both follow ``names``.
    A bare number is parsed. Anything else is unresolved, so a check that
    compares the two sides cannot pass by reading the constant twice.
    """
    token = expr.strip().strip("\"'")
    if token.startswith("${") and token.endswith("}"):
        token = token[2:-1]
    if token in names:
        return float(names[token])
    try:
        return float(token)
    except ValueError:
        return None


def jcc_target_lines(tally_text: str, script_text: str) -> tuple[int | None, int | None]:
    """Line the tally names, and the script line that sets the copper target."""
    claimed = re.search(
        r"scripts/elja_jcc_molslab_ensemble\.sh line (\d+) sets that target\.",
        tally_text,
    )
    claimed_line = int(claimed.group(1)) if claimed else None
    actual = next(
        (
            index
            for index, text in enumerate(script_text.splitlines(), 1)
            if text.strip() == "TARGET=-415.971529"
        ),
        None,
    )
    return claimed_line, actual


def supporting_numbers(audit: Audit) -> None:
    """Every number of the SI, read from the row, log, archive or source file that holds it."""
    gmin = load_script("gmin_panel_si", "scripts/gmin_panel/run_panel.py")
    ref = {row[0]: row[3] for row in gmin.SYSTEMS}

    def r6(system: str) -> str:
        return f"\\({ref[system]:.6f}\\)"

    # Board constants.
    target_tol_py = audit.text("scripts/target_tol.py")
    target_tol_rs = audit.text("src/target_tol.rs")
    py_default_hit_tol = float(
        re.search(r"DEFAULT_HIT_TOL:\s*float\s*=\s*([0-9.eE+-]+)", target_tol_py).group(1)
    )
    rs_default_hit_tol = float(
        re.search(r"pub const DEFAULT_HIT_TOL: f64 = ([0-9.eE+-]+);", target_tol_rs).group(1)
    )
    audit.cell(
        SI,
        "hit tolerance of the h2h examples",
        "0.001",
        py_default_hit_tol,
    )
    audit.cell(SI, "python and rust default hit tolerances agree", "1", float(py_default_hit_tol == rs_default_hit_tol))
    audit.said(
        SI,
        "We read a hit as an energy at or below the reference plus \\(0.001\\) "
        "(inclusive \\(E \\le E_{\\mathrm{ref}} + 10^{-3}\\)),",
    )
    audit.said(SI, f"The references, in reduced units, are {r6('lj38')} for LJ\\(_{{38}}\\), {r6('lj55')} for "
                   f"LJ\\(_{{55}}\\), and {r6('lj75')} for LJ\\(_{{75}}\\). LJ\\(_{{98}}\\) uses {r6('lj98')}, "
                   f"and LJ\\(_{{104}}\\) uses {r6('lj104')}.")
    preset = "src/methods/cluster_hopping/preset.rs"
    audit.cell(SI, "temperature at preset.rs:8", "0.8",
               float(re.fullmatch(r"\s*pub\(super\) const TEMPERATURE: f64 = ([\d.]+);", line_of(audit, preset, 8)).group(1)))
    audit.said(SI, "The temperature is \\(0.8\\), the value Wales and Doye used for Lennard-Jones basin hopping and the "
                   "value stored at \\path{src/methods/cluster_hopping/preset.rs}, line 8.")
    audit.said("examples/lj_cluster_search.rs", "random_cluster(n, 0.7, cfg.min_separation, &mut rng)")
    audit.said(SI, "The start density passed to \\texttt{random\\_cluster} is \\(0.7\\).")
    audit.said("src/methods/cluster_hopping.rs", "/// Seeds a non-overlapping configuration at liquid-like density.")
    audit.said(SI, "The function comment calls the draw liquid-like.")

    # The template-assisted single-seed log is retained as raw material but
    # is not presented as a plain walk or used for a claim.
    macros = audit.text("paper/jctc/macros.tex")
    audit.record("\\section{Plain walk}" not in audit.text(SI),
                 SI, "the mislabelled plain-walk section is absent", "absent",
                 "absent" if "\\section{Plain walk}" not in audit.text(SI) else "present")
    audit.record("\\PlainEnergy" not in macros and "\\PlainCalls" not in macros,
                 "paper/jctc/macros.tex", "plain-walk macros are absent", "absent",
                 "absent" if "\\PlainEnergy" not in macros and "\\PlainCalls" not in macros else "present")

    # Published symmetrised costs.
    lit = audit.rows("paper/jctc/data/literature.tsv", "\t")
    csm = pick(lit, method="GMIN CSM BH", N="38")[0]
    audit.said(SI, f"The continuous-symmetry basin-hopping mean at 38 atoms is {csm['reported_cost']} evaluations over "
                   f"{csm['hits']} successful runs.")
    audit.cell(SI, "successful runs equal runs", csm["runs"], float(csm["hits"]))
    audit.cell(SI, "cost unit", "energy/gradient evaluations", csm["cost_unit"])
    ga = sorted(r["N"] for r in lit if r["method"].endswith(" GA") and r["citation"].startswith("Oakley"))
    audit.said(SI, "quotes Oakley, Johnston and Wales, Table 1.")
    audit.cell(SI, "genetic-algorithm rows of Table 2", "75 98", " ".join(ga))
    audit.said(SI, "The Table 2 rows of that source are genetic-algorithm means at 75 and 98 atoms. "
                   "It has no genetic-algorithm row at 104 atoms.")
    audit.record("104" not in ga, SI, "no Oakley genetic-algorithm row at 104 atoms", "none", " ".join(ga))

    # TIP4P pair constants.
    constants = audit.text("paper/jctc/logs/tip4p_constants.txt")
    a_j, c_j = re.search(r"Jorgensen A = ([\d.]+) kJ A\^12/mol, C = ([\d.]+) kJ A\^6/mol", constants).groups()
    eps, sigma = (float(v) for v in re.search(r"anneal epsilon = ([\d.]+), sigma = ([\d.]+)", constants).groups())
    audit.cell(SI, "Jorgensen A", "2510400", float(a_j))
    audit.cell(SI, "Jorgensen C", "2552.24", float(c_j))
    audit.said(SI, "records Jorgensen's \\(A = 2510400\\) and \\(C = 2552.24\\).")
    audit.said("paper/jctc/logs/tip4p_constants.txt", "GMIN C6 = 2552.24, C12 = 2.5104e+06: equal to Jorgensen C and A: True")
    audit.said(SI, "The Wales-group basin-hopping program (GMIN) matches those two numbers in \\(C_6\\) and \\(C_{12}\\).")
    audit.cell(SI, "anneal epsilon and sigma", "0.6485 3.15365", f"{eps:g} {sigma:g}")
    c6, c12 = 4 * eps * sigma**6, 4 * eps * sigma**12
    audit.said(SI, f"give the anneal pair \\(C_6 = {c6:.4f}\\) and \\(C_{{12}} = {c12:.1f}\\).")
    audit.said("paper/jctc/logs/tip4p_constants.txt", f"C6 = {c6:.4f}, C12 = {c12:.1f}")
    audit.said(SI, f"It uses the Jorgensen TIP4P constants C6 \\(= {float(c_j):g}\\) and C12 \\(= {float(a_j):.0f}\\).")
    tip4p = audit.text("src/potentials/tip4p.rs")
    audit.said("src/potentials/tip4p.rs", "the rounded 0.6485 leaves every cluster")
    audit.record(f"{eps:g}" in tip4p, "src/potentials/tip4p.rs", "epsilon is in the source", f"{eps:g}", "present")

    # The copper-slab scripts agree on the target with the tally, though no table prints the slab.
    script = "scripts/elja_hq_sci_molslab_bank.sh"
    script_lines = audit.text(script).splitlines()
    energy_line = next(
        i
        for i, text in enumerate(script_lines, 1)
        if text == "export TARGET_ENERGY=-415.971529"
    )
    tol_line = next(
        i
        for i, text in enumerate(script_lines, 1)
        if i > energy_line and text.startswith("export TARGET_TOL=")
    )
    target_energy = float(script_lines[energy_line - 1].split("=", 1)[1])
    names = {"DEFAULT_HIT_TOL": py_default_hit_tol}
    target_tol = resolve_named_hit_tol(script_lines[tol_line - 1].split("=", 1)[1], names)
    tally = "paper/jctc/scripts/hq_sci_tally.py"
    tally_text = audit.text(tally)
    copper_rows = audit.rows("bench/copper-slab/rows.csv")
    copper_targets = {row["target_eV"] for row in copper_rows}
    copper_commands = {row["command"] for row in copper_rows}
    if len(copper_targets) != 1 or len(copper_commands) != 1 or len(copper_rows) != 8:
        audit.record(False, "bench/copper-slab/rows.csv", "eight copper rows, one target, one command",
                     "8 1 1", f"{len(copper_rows)} {len(copper_targets)} {len(copper_commands)}")
        target_from_rows = None
    else:
        target_from_rows = float(next(iter(copper_targets)))
        audit.cell("bench/copper-slab/rows.csv", "producing command", "scripts/run_cuh2_fcc_rec.sh",
                   next(iter(copper_commands)))
        for row in copper_rows:
            audit.cell("bench/copper-slab/rows.csv", f"seed {row['seed']} gap", row["gap_eV"],
                       float(row["best_eV"]) - float(row["target_eV"]))
    tally_reference = load_script("hq_sci_tally_ref", tally).REFERENCE_EV
    tally_tol_match = re.search(r"^REFERENCE_TOL_EV = (.+)$", tally_text, re.M)
    tally_tolerance = (
        resolve_named_hit_tol(tally_tol_match.group(1), names) if tally_tol_match else None
    )
    audit.cell(script, "TARGET_ENERGY value", "-415.971529", target_energy)
    audit.cell(script, "TARGET_ENERGY matches tally reference", f"{tally_reference}", target_energy)
    if target_from_rows is None:
        audit.record(False, script, "TARGET_ENERGY matches the copper rows", "unresolved", f"{target_energy}")
    else:
        audit.cell(script, "TARGET_ENERGY matches the copper rows", f"{target_from_rows:.6f}", target_energy)
    if target_tol is None:
        audit.record(False, script, "TARGET_TOL value", "unresolved", "0.001")
    else:
        audit.cell(script, "TARGET_TOL value", "0.001", target_tol)
    if target_tol is None or tally_tolerance is None:
        audit.record(
            False,
            tally,
            "TARGET_TOL matches tally tolerance",
            "unresolved",
            target_tol if target_tol is not None else "unresolved",
        )
    else:
        audit.cell(
            tally,
            "TARGET_TOL matches tally tolerance",
            f"{tally_tolerance:.6g}",
            target_tol,
        )
    audit.said(tally, f"scripts/elja_hq_sci_molslab_bank.sh line {energy_line} sets it too.")
    claimed_line, jcc_target_line = jcc_target_lines(
        tally_text, audit.text("scripts/elja_jcc_molslab_ensemble.sh")
    )
    if claimed_line is None or jcc_target_line is None:
        audit.record(
            False,
            tally,
            "the jcc script's target line",
            "missing" if claimed_line is None else str(claimed_line),
            jcc_target_line if jcc_target_line is not None else "missing",
        )
    else:
        audit.cell(tally, "the jcc script's target line", str(claimed_line), float(jcc_target_line))
    if jcc_target_line is not None:
        audit.said(tally, f"scripts/elja_jcc_molslab_ensemble.sh line {jcc_target_line} sets that target.")

    si_text = audit.text(SI)
    for gone in ("campaign of 200 seeds", "lj75-source-", "lj75_ecdf", "lj75_landscape", "occupancy_landfold_lj75",
                 "anneal core developers"):
        audit.record(gone not in si_text, SI, f"the SI leaves out {gone}", "absent",
                     "absent" if gone not in si_text else "present")


# Sections whose rows carry no revision, or a revision outside the history of
# the submitted revision, and the strings that would print them again.
WITHDRAWN = (
    ("the water hexamer against CREST", "121 of 192"),
    ("the water hexamer against CREST", "\\label{sec:crest}"),
    ("the water hexamer against CREST", "\\label{sec:hexamer}"),
    ("the water hexamer against CREST", "tab:crest-counters"),
    ("the water hexamer against CREST", "tab:hexamer-gaps"),
    ("the water hexamer against CREST", "water_crest.pdf"),
    ("the water hexamer against CREST", "36086e98"),
    ("the water hexamer against CREST", "9a30f319"),
    ("the earlier campaign", "\\section{Earlier campaign}"),
    ("the earlier campaign", "campaign-scores.csv"),
    ("the earlier campaign", "Bootstrap of a median ratio"),
    ("the earlier campaign", "disconnectivity.pdf"),
    ("the earlier campaign", "lj38_sketchmap.pdf"),
    ("the earlier campaign", "lj38_landscape.pdf"),
    ("the earlier campaign", "lj98_landscape.pdf"),
    ("the earlier campaign", "occupancy_landfold"),
    ("the symmetrized LJ75 GMIN campaign", "gmin-lj75-sym-starts"),
    ("the symmetrized LJ75 GMIN campaign", "200 of 200"),
    ("the pentagonal retry", "\\section{Pentagonal retry}"),
    ("the six-seed TIP4P rates", "\\section{TIP4P on six seeds}"),
    ("the arms not kept", "\\section{Arms that were measured and not kept}"),
    ("the copper slab", "\\section{Copper slab}"),
    ("the copper slab", "-415.971529"),
    ("the hexamer engine digest", "\\section{Hardware and reproduction}"),
    ("the anneal constants on the hexamer plate", "-197.7750"),
)


def withdrawn_benchmarks(audit: Audit) -> None:
    """The withdrawn sections stay out of the manuscript, the SI, and the cover letter."""
    for what, fragment in WITHDRAWN:
        found = [path for path in (TEX, SI, COVER) if fragment in audit.text(path)]
        audit.record(not found, " ".join(found) or SI, f"{what} is not printed: {fragment}", "absent",
                     " ".join(found) or "absent")
    cover_session = "thirteen Lennard-Jones, Gupta, and Morse clusters in one session."
    audit.record(cover_session not in collapse(audit.text(COVER)), COVER,
                 "no whole-table single-session claim", "absent",
                 "absent" if cover_session not in collapse(audit.text(COVER)) else "present")
    stale_si = "same-session panel" in collapse(audit.text(SI))
    audit.record(not stale_si, SI, "no same-session panel claim in SI", "absent",
                 "present" if stale_si else "absent")
    for path in (TEX, SI, COVER):
        before = audit.text(path).split("\\section{Searches that report", 1)[0]
        audit.record("CREST" not in before, path, "CREST is named only among the searches that report no comparable "
                     "rate", "absent", "absent" if "CREST" not in before else "present")


def supporting_information(audit: Audit) -> None:
    eps, sigma = 0.6485, 3.15365
    audit.said(SI, f"\\(C_6 = {4 * eps * sigma**6:.4f}\\) and \\(C_{{12}} = {4 * eps * sigma**12:.1f}\\).")
    refs = audit.rows("paper/jctc/data/system_visuals/references.csv")
    hexamer = next(r for r in refs if r.get("label") == "TIP4P-6" or "TIP4P-6" in ",".join(r.values()))
    energy = next(v for v in hexamer.values() if v and re.fullmatch(r"-\d+\.\d+", v))
    audit.said(SI, f"The (H\\(_2\\)O)\\(_6\\) plate prints the energy \\({float(energy):.4f}\\)~kJ/mol.")


# --------------------------------------------------------------------------- bench READMEs


def paper_0120(audit: Audit) -> None:
    spec = importlib.util.spec_from_file_location("run_panel", ROOT / "scripts/paper_0120/run_panel.py")
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    path = "bench/paper-0120/README.md"
    rows = audit.rows("bench/paper-0120/rows.csv")
    groups = panel.release_groups(rows, "07f89cadbd9ccd8386e92408a8ff0e8153b180e3")
    names = {"LJ75, `h2h_seed`": ("lj75", "h2h_seed"), "LJ103": ("lj103", "h2h_gmin_proof+lj103"),
             "Au55": ("au55", "h2h_metal"), "M80 ρ=6": ("m80r6", "h2h_morse"), "M80 ρ=14": ("m80r14", "h2h_morse")}
    for line, row in md_table(audit, path, "system | budget | 0.12.0 hits"):
        system, driver = names.get(row["system"], (row["system"].lower(), "h2h_gmin_proof"))
        group = groups[(system, driver)]
        dev = panel.DEVELOP[(system, driver)]
        found = pick(audit.rows(dev["path"]), **dev["where"])
        if dev["path"] == "bench/morse-panel/seeds.csv":
            found.extend(pick(audit.rows("bench/morse-panel/template-free-rows.csv"), **dev["where"]))
        found = [r for r in found if r["system"] == system]
        where = f"{path}:{line} {row['system']}"
        calls = [float(r["evals_to_hit"]) for r in group]
        mine = hits_of(group)
        audit.cell(where, "budget", row["budget"], float(group[0]["budget"]))
        audit.cell(where, "0.12.0 hits", row["0.12.0 hits"], f"{mine}/{len(group)}")
        audit.cell(where, "median calls 0.12.0", row["median calls 0.12.0"], median(calls))
        for shown, key in zip(row["0.12.0 wall / inside / outside (s)"].split(" / "), ("wall_s", "pot_s", "outside_s")):
            audit.cell(where, key, shown, median([float(r[key]) for r in group]))
        if "summary" in dev:
            base = found[0]
            dev_hits, dev_n = int(base["hits"]), int(base["n"])
            audit.cell(where, "median calls develop", row["median calls develop"], float(base["evals_median"]))
            audit.cell(where, "MW p", row["MW p"], None)
            audit.cell(where, "same seeds", row["same seeds"], None)
        else:
            by_seed = {int(r["seed"]): r for r in found}
            base = [by_seed[s] for s in panel.SEEDS]
            dev_hits, dev_n = sum(panel.is_hit(r["hit"]) for r in base), len(base)
            base_calls = [float(r[dev["calls"]]) for r in base]
            audit.cell(where, "median calls develop", row["median calls develop"], median(base_calls))
            audit.cell(where, "MW p", row["MW p"], mann_whitney(calls, base_calls, continuity=True))
            same = sum(panel.same_seed(r, by_seed[int(r["seed"])], dev["calls"]) for r in group)
            audit.cell(where, "same seeds", row["same seeds"], float(same))
        audit.cell(where, "develop hits", row["develop hits"], f"{dev_hits}/{dev_n}")
        audit.cell(where, "Fisher p", row["Fisher p"], fisher(mine, len(group), dev_hits, dev_n))


CREST_EQUAL_README = "bench/crest-equal/README.md"
CREST_EQUAL_RGPOT_SIDECAR = "bench/crest-equal/rgpot-revisions.csv"
GFN2_RGPOT_SIDECAR = "bench/covalent-h2h/gfn2/rgpot-revisions.csv"


def sidecar_rows(audit: Audit, path: str) -> list[dict[str, str]]:
    """Read a sidecar CSV through audit.text without registering it as a benchmark result table."""
    lines = [line for line in audit.text(path).splitlines(keepends=True) if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines))


def crest_equal_bench(audit: Audit) -> None:
    """The equal-call CREST panel against its README: seeds, the budget, the ledger, and the summary."""
    panel = load_script("crest_equal_panel", "scripts/crest_equal/run_panel.py")
    where = "bench/crest-equal/rows.csv"
    rows = audit.rows(where)
    audit.record("rgpot_revision" not in rows[0], where,
                 "rows.csv keeps its original header without rgpot_revision",
                 "absent", "present" if "rgpot_revision" in rows[0] else "absent")
    sidecar = sidecar_rows(audit, CREST_EQUAL_RGPOT_SIDECAR)
    row_keys = [(r["system"], r["arm"], r["seed"], r["code_revision"], r["host"]) for r in rows]
    side_keys = [(r["system"], r["arm"], r["seed"], r["code_revision"], r["host"]) for r in sidecar]
    audit.record(row_keys == side_keys, CREST_EQUAL_RGPOT_SIDECAR,
                 "each crest-equal row has one matching rgpot-revisions.csv sidecar row",
                 f"{len(row_keys)} aligned rows",
                 f"{len(side_keys)} rows, {'aligned' if row_keys == side_keys else 'mismatched'}")
    for s_row in sidecar:
        rev = s_row["code_revision"]
        rgpot = s_row.get("rgpot_revision", "")
        if rev == "723b7c23bbb31a076a559e3ac7f4f91be9e36e1e":
            ok, want = rgpot == "", "empty"
        else:
            ok, want = revision_reach.is_full_sha(rgpot), "full commit id"
        audit.record(ok, CREST_EQUAL_RGPOT_SIDECAR,
                     f"{s_row['system']} {s_row['arm']} seed {s_row['seed']} rgpot_revision",
                     want, rgpot or "empty")
    revisions = sorted({r["code_revision"] for r in rows})
    for revision in revisions:
        mine = [r for r in rows if r["code_revision"] == revision]
        for arm in panel.ARMS:
            seeds = sorted(int(r["seed"]) for r in mine if r["arm"] == arm)
            audit.record(seeds == list(panel.SEEDS), where, f"{revision[:8]} {arm} seeds",
                         " ".join(map(str, seeds)), " ".join(map(str, panel.SEEDS)))
    audit.record({r["budget"] for r in rows} == {str(panel.BUDGET)}, where, "budget",
                 " ".join(sorted({r["budget"] for r in rows})), panel.BUDGET)
    over = sorted(int(r["charged"]) for r in rows if int(r["charged"]) > panel.BUDGET)
    audit.record(not over, where, "no row is charged above the budget", " ".join(map(str, over)) or "none", "none")
    counts_path = "bench/crest-equal/call-counts.csv"
    if counts_path in audit.read or (ROOT / counts_path).is_file():
        for index, row in enumerate(audit.rows(counts_path), start=1):
            label = f"row {index} {row.get('arm', '')} seed {row.get('seed', '')} charged"
            try:
                driver_calls = int(row["driver_calls"])
                charged = int(row["charged"])
                if row["arm"] == "anneal" and not row.get("crest_log_calls"):
                    expected: object = driver_calls
                else:
                    expected = max(driver_calls, int(row["crest_log_calls"]))
                ok = charged == expected
            except (KeyError, ValueError):
                ok = False
                expected = "valid driver_calls and crest_log_calls"
            audit.record(ok, counts_path, label, row.get("charged", ""), expected)

    def mw(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return float("nan")
        result = mann_whitney(left, right, continuity=True, degenerate=1.0)
        return float("nan") if result is None else result

    summary = panel.summarise(rows, fisher_fn=fisher, mann_whitney_fn=mw)
    committed = audit.rows("bench/crest-equal/summary.csv")
    audit.record(summary == committed, "bench/crest-equal/summary.csv", "recomputed from rows.csv",
                 "equal" if summary == committed else "differs", "equal")
    readme_text = audit.text(CREST_EQUAL_README)
    forbidden_claim = (panel.claim_sentence(summary), "hits 1 of 12", "hits 4 of 12", "1/12", "4/12")
    found_claim = [item for item in forbidden_claim if item in readme_text]
    audit.record(not found_claim, CREST_EQUAL_README,
                 "sign-affected hit-rate claim is absent from crest-equal README",
                 "absent", " | ".join(found_claim) or "absent")
    anneal = [int(r["charged"]) for r in rows if r["arm"] == "anneal"]
    centre = sorted(anneal)[len(anneal) // 2 - 1: len(anneal) // 2 + 1]
    centre = sum(centre) / len(centre)
    for arm in ("crest", "crest-wall"):
        charged = sorted(int(r["charged"]) for r in rows if r["arm"] == arm)
        within = sum(abs(value - centre) <= 0.05 * centre for value in charged)
        row = next(r for r in summary if r["arm"] == arm)
        audit.cell(where, f"{arm} within 5 percent", row["within_5pct"], within)
        audit.said(CREST_EQUAL_README, f"{within} of {len(charged)}")
        audit.said(CREST_EQUAL_README, f"{charged[0]} to {charged[-1]} calls")
        if arm == "crest":
            audit.said(CREST_EQUAL_README, f"{charged[0] / centre:.2f} to {charged[-1] / centre:.2f} of anneal's {centre:g}")
    audit.said(CREST_EQUAL_README, f"The most any CREST row was charged is {max(int(r['charged']) for r in rows if r['arm'] == 'crest')}")
    walls = {arm: sorted(float(r["wall_s"]) for r in rows if r["arm"] == arm) for arm in panel.ARMS}
    middle = {arm: (w[len(w) // 2 - 1] + w[len(w) // 2]) / 2 for arm, w in walls.items()}
    audit.said(CREST_EQUAL_README, f"Median wall is {middle['anneal']:.2f} s for anneal, {middle['crest']:.2f} s at 0.90 ps and "
                                   f"{middle['crest-wall']:.2f} s at\n0.11 ps.")
    unconverged = sorted(r["seed"] for r in rows if r["arm"] == "anneal" and not r["gap_Eh"])
    audit.said(CREST_EQUAL_README, f"Anneal seed {unconverged[0]} polished without converging")
    audit.record(len(unconverged) == 1, where, "unconverged anneal polishes", " ".join(unconverged), "one")


def gad_published(audit: Audit) -> None:
    path = "bench/gad-published/README.md"
    rows = audit.rows("bench/gad-published/rows.csv")
    seeds = audit.rows("bench/gad-published/seed-reruns.csv")
    columns = {"Median calls to hit": "evals_median", "IQR": "evals_iqr", "Best (median)": "best_median",
               "Wall (s)": "wall_median_s", "Pot (s)": "pot_median_s", "Outside (s)": "outside_median_s",
               "Inside µs/call": "inside_us", "Outside µs/call": "outside_us"}
    for line, row in md_table(audit, path, "Size | Arm | Hits"):
        system = row["Size"].lower()
        line_rows = pick(rows, system=system, arm=row["Arm"])
        base = line_rows[-1]
        where = f"{path}:{line} {system} {row['Arm']}"
        audit.cell(where, "hits", row["Hits"], f"{base['hits']}/{base['n']}")
        for label, key in columns.items():
            audit.cell(where, label, row[label], float(base[key]))
        if row["Arm"] == "anneal":
            audit.cell(where, "Fisher p", row["Fisher p"], None)
            continue
        anneal = pick(rows, system=system, arm="anneal")[-1]
        inferential = next(candidate for candidate in reversed(line_rows)
                           if candidate["fisher_p"] and candidate["mannwhitney_p"])
        p = fisher(int(base["hits"]), int(base["n"]), int(anneal["hits"]), int(anneal["n"]))
        audit.cell(where, "Fisher p (recomputed)", row["Fisher p"], p)
        audit.cell(where, "Fisher p (rows.csv)", row["Fisher p"], float(inferential["fisher_p"]))
        audit.cell(where, "Mann-Whitney p (rows.csv)", row["Mann–Whitney p"],
                   float(inferential["mannwhitney_p"]))
    for row in pick(rows, arm="gad_deposit"):
        gen, lo, hi = row["hit_generation"], row["evals_lower"], row["evals_upper"]
        audit.cell(path, f"{row['system']} lower bound, generations 0..{gen} x 64", lo, float((int(gen) + 1) * 64))
        audit.cell(path, f"{row['system']} upper bound, x 1000 FIRE steps", hi, float((int(gen) + 1) * 64 * 1000))
        size = row["system"].removeprefix("lj")
        audit.said(path, f"generation {gen} for LJ{size} (bounds {lo} and {hi} calls)")
    expected = {("lj31", "anneal"): 12, ("lj31", "gmin"): 12, ("lj31", "gad"): 12, ("lj38", "gad"): 12}
    for group, count in expected.items():
        found = pick(seeds, system=group[0], arm=group[1])
        audit.cell("bench/gad-published/seed-reruns.csv", f"{group[0]} {group[1]} seed rows", str(count), float(len(found)))
        audit.record({row["host"] for row in found} == {"bench"}, "bench/gad-published/seed-reruns.csv",
                     f"{group[0]} {group[1]} rerun host", "bench", " ".join(sorted({row["host"] for row in found})))
    hits = {row["seed"]: row["evals_to_hit"] for row in pick(seeds, system="lj31", arm="gad") if row["hit"] == "1"}
    audit.said(path, f"seed 2000 at {hits['2000']} calls and seed 2006 at {hits['2006']} calls")
    generations = {}
    for system in ("lj31", "lj38"):
        values = [float(row["generations"]) for row in pick(seeds, system=system, arm="gad")]
        generations[system] = (median(values), min(values), max(values))
    audit.said(path, "a median of "
               f"{generations['lj31'][0]:g} generations on LJ31 "
               f"({generations['lj31'][1]:g}–{generations['lj31'][2]:g}) and "
               f"{generations['lj38'][0]:g} on LJ38 "
               f"({generations['lj38'][1]:g}–{generations['lj38'][2]:g})")
    anneal_calls = [float(row["evals_to_hit"]) for row in pick(seeds, system="lj31", arm="anneal")]
    gmin_calls = [float(row["evals_to_hit"]) for row in pick(seeds, system="lj31", arm="gmin")]
    audit.cell(path, "LJ31 GMIN median calls from per-seed reruns",
               pick(rows, system="lj31", arm="gmin")[-1]["evals_median"], median(gmin_calls))
    audit.cell(path, "LJ31 GMIN IQR from per-seed reruns",
               pick(rows, system="lj31", arm="gmin")[-1]["evals_iqr"], iqr(gmin_calls))
    audit.cell(path, "LJ31 GMIN Mann-Whitney from per-seed reruns",
               pick(rows, system="lj31", arm="gmin")[-1]["mannwhitney_p"],
               mann_whitney_exact_no_ties(gmin_calls, anneal_calls))


OPENCONF_ARM = {"branch": "anneal", "anneal": "anneal", "develop": "develop", "openconf": "openconf"}


def openconf(audit: Audit) -> None:
    base = "bench/openconf-h2h"
    tables = [
        ("paired/README.md", 0, "paired/seeds.csv"),
        ("conf-fix/README.md", 0, "conf-fix/seeds.csv"),
        ("conf-fix3/README.md", 0, "conf-fix3/seeds.csv"),
        ("conf-fix4/README.md", 0, "conf-fix4/seeds.csv"),
        ("conf-fix4/README.md", 85, "conf-fix4/n24/seeds.csv"),
    ]
    for readme, after, source in tables:
        path = f"{base}/{readme}"
        by = defaultdict(list)
        for r in audit.rows(f"{base}/{source}"):
            by[(r["molecule"], r["arm"])].append(r)
        heads = ("molecule | hits", "molecule | calls to hit", "molecule | wall")
        for head in heads:
            try:
                table = md_table(audit, path, head, after)
            except SystemExit:
                continue
            if after == 0 and readme == "conf-fix4/README.md" and table[0][0] > 85:
                continue
            for line, row in table:
                mol = row["molecule"]
                for label, shown in row.items():
                    if label == "molecule":
                        continue
                    audit.cell(f"{path}:{line} {mol}", label, shown, openconf_value(label, by, mol))


def openconf_value(label: str, by: dict, mol: str) -> object:
    def group(arm: str) -> list[dict[str, str]]:
        return by[(mol, OPENCONF_ARM[arm])]

    def to_hit(g: list[dict[str, str]]) -> list[float]:
        return [float(r["calls_to_hit"]) for r in g if r["calls_to_hit"]]

    def col(g: list[dict[str, str]], key: str) -> float | None:
        values = [float(r[key]) for r in g if r[key]]
        return median(values) if values else None

    if m := re.fullmatch(r"hits (\w+)", label):
        return frac(group(m.group(1)))
    if m := re.fullmatch(r"Fisher p(?: vs (\w+))?", label):
        a, b = group("anneal"), group(m.group(1) or "openconf")
        return fisher(hits_of(a), len(a), hits_of(b), len(b))
    if m := re.fullmatch(r"Mann-Whitney p(?: vs (\w+))?", label):
        a, b = to_hit(group("anneal")), to_hit(group(m.group(1) or "openconf"))
        return mann_whitney(a, b, continuity=True, degenerate=None) if a and b else None
    if m := re.fullmatch(r"n_3kcal (\w+)", label):
        return col(group(m.group(1)), "n_3kcal")
    if m := re.fullmatch(r"ΔE_min (\w+)", label):
        return col(group(m.group(1)), "delta_min_kcal")
    if m := re.fullmatch(r"calls to hit (\w+)", label):
        c = to_hit(group(m.group(1)))
        return median(c) if c else None
    if m := re.fullmatch(r"wall (\w+) \(s\)", label):
        return col(group(m.group(1)), "wall_s")
    if m := re.fullmatch(r"(\w+) inside \(s\)", label):
        return col(group(m.group(1)), "pot_s")
    if m := re.fullmatch(r"(\w+) outside \(s\)", label):
        return col(group(m.group(1)), "outside_s")
    raise SystemExit(f"openconf column {label!r} has no rule")


CONF_MISS4 = "bench/openconf-h2h/conf-miss4"
CONF_MISS4_ARM = {"this series": "anneal", "develop 61fa67cf": "develop", "openconf": "openconf"}
CONF_MISS4_LEDGER = ("spent", "screen", "full", "check", "quenches", "converged", "capped", "screened_out", "returned",
                     "overlap_rejected", "basins")
CONF_MISS4_SCHEDULE_SEED = 20261115
CONF_MISS4_STANDARD_SEEDS = tuple(range(2000, 2012))
# This audit is merge-only: the first parent carries the measured series and
# the second carries its develop base. Checking both parent histories keeps a
# side-branch revision from passing merely because a later HEAD merged it.
CONF_MISS4_PROVENANCE_MERGE = "c8581d69319a2f960c25ecb5f0261feb34a739b3"


def derived_table(audit: Audit, path: str, delimiter: str = "\t") -> list[dict[str, str]]:
    """A table computed from row files. It carries no revision or host of its own, so it is not a row file."""
    return list(csv.DictReader([line for line in audit.text(path).splitlines() if line.strip()], delimiter=delimiter))


def binomial_half(k: int, n: int) -> float:
    """Two-sided exact binomial p of k in n at one half."""
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2.0**n)


def signed_rank_exact(diffs: list[float]) -> float:
    """Two-sided exact Wilcoxon signed-rank p of the non-zero differences, ranks of ties averaged."""
    diffs = [d for d in diffs if d != 0.0]
    if not diffs:
        return 1.0
    order = sorted(abs(d) for d in diffs)
    rank = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1] == order[i]:
            j += 1
        rank[order[i]] = (i + j + 2)
        i = j + 1
    counts = {0: 1}
    for d in diffs:
        step = rank[abs(d)]
        grown = defaultdict(int)
        for total, ways in counts.items():
            grown[total] += ways
            grown[total + step] += ways
        counts = grown
    positive = sum(rank[abs(d)] for d in diffs if d > 0)
    whole = sum(rank[abs(d)] for d in diffs)
    tail = sum(ways for total, ways in counts.items() if total <= min(positive, whole - positive))
    return min(1.0, 2.0 * tail / 2 ** len(diffs))


def conf_miss4(audit: Audit) -> None:
    """The conf-miss4 panel, paired statistics, rescore, guard and pins against its rows."""
    base = CONF_MISS4
    rows = audit.rows(f"{base}/rows.csv")
    steps = audit.rows(f"{base}/local_step.csv")
    guard = audit.rows(f"{base}/guard.csv")
    where = f"{base}/rows.csv"
    mols = list(dict.fromkeys(r["molecule"] for r in rows))
    by = {(m, a): sorted(pick(rows, molecule=m, arm=a), key=lambda r: int(r["seed"])) for m in mols for a in
          ("anneal", "develop", "openconf")}
    audit.record(all(len(g) == 24 for g in by.values()) and len(rows) == 576, where, "8 molecules, 3 arms, 24 seeds each",
                 "576 rows", f"{len(rows)} rows")
    budgets_ok = True
    for m in mols:
        for a, o, d in zip(by[(m, "anneal")], by[(m, "openconf")], by[(m, "develop")]):
            want = (int(o["energy_calls"]) + int(o["grad_calls"])) // 2
            budgets_ok &= a["seed"] == o["seed"] == d["seed"] and int(a["budget"]) == int(d["budget"]) == want
    audit.record(budgets_ok, where, "each arm's budget is (energy_calls + grad_calls) // 2 of its seed's openconf run",
                 "holds", "holds" if budgets_ok else "fails")
    audit.record(all(r["host"] == "cloud-vm" for r in rows + steps + guard), base, "the host column is the neutral label",
                 "cloud-vm", "cloud-vm" if all(r["host"] == "cloud-vm" for r in rows + steps + guard) else "other")
    cited = {r[c] for table in (rows, steps, guard) for r in table for c in ("code_revision",) if r[c]}
    paired_revisions = {r["code_revision"] for r in rows if r["arm"] in ("anneal", "openconf")}
    audit.record(len(paired_revisions) == 1, where, "anneal and openconf carry the paired runner's one revision",
                 "one revision", " ".join(sorted(r[:8] for r in paired_revisions)))
    provenance_rows = derived_table(audit, f"{base}/provenance.tsv")
    roles = {r["commit"]: r["role"] for r in provenance_rows}
    merge_exists = git("cat-file", "-e", f"{CONF_MISS4_PROVENANCE_MERGE}^{{commit}}").returncode == 0
    merge_parents = (
        git("show", "-s", "--format=%P", CONF_MISS4_PROVENANCE_MERGE).stdout.split()
        if merge_exists
        else []
    )
    audit.record(merge_exists, base, "the documented provenance merge is a commit object",
                 "commit", "commit" if merge_exists else "missing")
    audit.record(len(merge_parents) == 2, base, "the documented provenance merge has two parents",
                 "two parents", f"{len(merge_parents)} parents")
    merge_on_origin = merge_exists and bool(
        git("for-each-ref", "--count=1", "--contains", CONF_MISS4_PROVENANCE_MERGE,
            "refs/remotes/origin").stdout.strip()
    )
    audit.record(merge_on_origin, base, "the documented provenance merge is reachable from origin",
                 "reachable", "reachable" if merge_on_origin else "not reachable")
    merge_on_head = merge_exists and git(
        "merge-base", "--is-ancestor", CONF_MISS4_PROVENANCE_MERGE, "HEAD"
    ).returncode == 0
    audit.record(merge_on_head, base, "the documented provenance merge is an ancestor of HEAD",
                 "ancestor", "ancestor" if merge_on_head else "not")
    for revision in sorted(cited):
        exists = git("cat-file", "-e", f"{revision}^{{commit}}").returncode == 0
        on_origin = exists and bool(
            git("for-each-ref", "--count=1", "--contains", revision, "refs/remotes/origin").stdout.strip()
        )
        audit.record(exists, base, f"cited revision {revision[:8]} is a commit object",
                     "commit", "commit" if exists else "missing")
        audit.record(on_origin, base, f"cited revision {revision[:8]} is reachable from origin",
                     "reachable", "reachable" if on_origin else "not reachable")
        parent_index = 1 if roles.get(revision) == "develop base" else 0
        on_parent = (
            exists
            and len(merge_parents) == 2
            and git("merge-base", "--is-ancestor", revision, merge_parents[parent_index]).returncode == 0
        )
        audit.record(on_parent, base, f"cited revision {revision[:8]} is on its provenance merge parent",
                     "ancestor", "ancestor" if on_parent else "not")

    listed = {r["commit"] for r in provenance_rows}
    for r in provenance_rows:
        tree = git("rev-parse", f"{r['commit']}^{{tree}}").stdout.strip()
        audit.record(tree == r["tree"], f"{base}/provenance.tsv", f"tree of {r['commit'][:8]} ({r['role']})", r["tree"], tree)
        audit.said(f"{base}/README.md", r["commit"])
        if r["role"] != "develop base":
            audit.said(f"{base}/README.md", r["tree"])
    audit.record(cited <= listed, f"{base}/provenance.tsv", "every revision the rows cite is listed",
                 "all listed", "all listed" if cited <= listed else ", ".join(sorted(c[:8] for c in cited - listed)))

    row_pairs = {(r["molecule"], int(r["seed"])) for r in rows if r["arm"] == "anneal"}
    with (ROOT / "bench/openconf-h2h/molecules.csv").open(newline="") as handle:
        schedule_names = [r["name"] for r in csv.DictReader(handle)]
    rng = random.Random(CONF_MISS4_SCHEDULE_SEED)
    standard = [(m, s) for m in schedule_names for s in CONF_MISS4_STANDARD_SEEDS]
    rng.shuffle(standard)
    orders = {pair: rng.random() < 0.5 for pair in standard}
    extra_seeds = sorted({seed for _, seed in row_pairs} - set(CONF_MISS4_STANDARD_SEEDS))
    extra = [(m, s) for m in schedule_names for s in extra_seeds]
    rng.shuffle(extra)
    orders.update({pair: rng.random() < 0.5 for pair in extra})
    develop_first = sum(orders[pair] for pair in row_pairs)
    audit.record(develop_first == 98 and len(row_pairs) == 192, where,
                 "the seeded paired schedule runs develop first on 98 of 192 row pairs",
                 "98 of 192", f"{develop_first} of {len(row_pairs)}")
    audit.said(f"{base}/README.md", f"Develop ran first on {develop_first} of the {len(row_pairs)} pairs and anneal on "
               f"{len(row_pairs) - develop_first}.")

    def col(g: list[dict[str, str]], key: str) -> list[float]:
        return [float(r[key]) for r in g if r[key] != ""]

    panel = derived_table(audit, f"{base}/panel.tsv")
    for line in panel:
        m, a = line["molecule"], CONF_MISS4_ARM[line["arm"]]
        g = by[(m, a)]
        at = f"{base}/panel.tsv {m} {line['arm']}"
        audit.cell(at, "n", line["n"], len(g))
        audit.record(line["hits"] == frac(g), at, "hits", line["hits"], frac(g))
        audit.record(line["hits_mirror"] == frac(g, "hit_mirror"), at, "hits_mirror", line["hits_mirror"], frac(g, "hit_mirror"))
        calls = col(g, "calls_to_hit")
        audit.cell(at, "calls_to_hit_median", line["calls_to_hit_median"], median(calls) if calls else None)
        audit.cell(at, "delta_min_median_kcal", line["delta_min_median_kcal"], median(col(g, "delta_min_kcal")))
        audit.cell(at, "invalid_total", line["invalid_total"], sum(int(r["invalid"]) for r in g))
        audit.cell(at, "wall_median_s", line["wall_median_s"], median(col(g, "wall_s")))
        for ref, label in (("develop", "vs_develop"), ("openconf", "vs_openconf")):
            if a == ref:
                audit.cell(at, f"fisher_p_{label}", line[f"fisher_p_{label}"], None)
                audit.cell(at, f"mw_p_{label}", line[f"mw_p_{label}"], None)
                continue
            other = by[(m, ref)]
            audit.cell(at, f"fisher_p_{label}", line[f"fisher_p_{label}"],
                       fisher(hits_of(g), len(g), hits_of(other), len(other)))
            x, y = col(g, "calls_to_hit"), col(other, "calls_to_hit")
            audit.cell(at, f"mw_p_{label}", line[f"mw_p_{label}"],
                       mann_whitney(x, y, continuity=True, degenerate=None) if x and y else None)
    audit.record(len(panel) == 24, f"{base}/panel.tsv", "a line per molecule and arm", "24", str(len(panel)))

    ledger_ok = True
    local_step = {(r["molecule"], r["arm"], r["seed"]): r for r in steps}
    for m in mols:
        for a in ("anneal", "develop"):
            for r in by[(m, a)]:
                step = local_step[(m, a, r["seed"])]
                ledger_ok &= step["local_step"] == "lbfgs"
                if a == "develop":
                    ledger_ok &= all(step[k] == "" for k in CONF_MISS4_LEDGER)
                else:
                    v = {k: int(step[k]) for k in CONF_MISS4_LEDGER}
                    ledger_ok &= v["screen"] + v["full"] + v["check"] <= v["spent"] and v["quenches"] == v["converged"] + v["capped"]
    audit.record(ledger_ok and len(steps) == 384, f"{base}/local_step.csv",
                 "every run resolved to lbfgs; the ledger splits are consistent and absent for develop", "holds",
                 "holds" if ledger_ok and len(steps) == 384 else "fails")

    paired = derived_table(audit, f"{base}/paired.tsv")
    for line in paired:
        m = line["molecule"]
        mine = {r["seed"]: r for r in by[(m, "anneal")]}
        theirs = {r["seed"]: r for r in by[(m, "develop")]}
        seeds = sorted(set(mine) & set(theirs), key=int)
        a, b = [mine[s] for s in seeds], [theirs[s] for s in seeds]
        only_arm = sum(1 for x, y in zip(a, b) if x["hit"] == "1" and y["hit"] == "0")
        only_base = sum(1 for x, y in zip(a, b) if y["hit"] == "1" and x["hit"] == "0")
        us = [(1e6 * float(x["outside_s"]) / float(x["calls"]), 1e6 * float(y["outside_s"]) / float(y["calls"])) for x, y in zip(a, b)]
        at = f"{base}/paired.tsv {m}"
        audit.cell(at, "n", line["n"], len(seeds))
        audit.cell(at, "hits_arm", line["hits_arm"], hits_of(a))
        audit.cell(at, "hits_base", line["hits_base"], hits_of(b))
        audit.cell(at, "only_arm", line["only_arm"], only_arm)
        audit.cell(at, "only_base", line["only_base"], only_base)
        audit.cell(at, "mcnemar_p", line["mcnemar_p"], binomial_half(only_arm, only_arm + only_base))
        audit.cell(at, "invalid_arm", line["invalid_arm"], sum(int(x["invalid"]) for x in a))
        audit.cell(at, "invalid_base", line["invalid_base"], sum(int(y["invalid"]) for y in b))
        audit.cell(at, "outside_us_per_call_arm", line["outside_us_per_call_arm"], median([p for p, _ in us]))
        audit.cell(at, "outside_us_per_call_base", line["outside_us_per_call_base"], median([q for _, q in us]))
        audit.cell(at, "outside_ratio_median", line["outside_ratio_median"], median([p / q for p, q in us]))
        audit.cell(at, "wilcoxon_p", line["wilcoxon_p"], signed_rank_exact([p - q for p, q in us]))
        audit.record(not any(p == q for p, q in us), at, "no equal pair, so no pair is dropped from the Wilcoxon test",
                     "none", "none" if not any(p == q for p, q in us) else "some")
        for side, arm in (("arm", "anneal"), ("base", "develop")):
            counts = defaultdict(int)
            for r in by[(m, arm)]:
                counts[local_step[(m, arm, r["seed"])]["local_step"]] += 1
            shown = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            audit.record(line[f"local_step_{side}"] == shown, at, f"local_step_{side}", line[f"local_step_{side}"], shown)
    audit.record(len(paired) == 8, f"{base}/paired.tsv", "a line per molecule", "8", str(len(paired)))

    rescore = derived_table(audit, f"{base}/rescore.tsv")
    moved = derived_table(audit, f"{base}/rescore-cells.tsv")
    for line in rescore:
        g = by[(line["molecule"], line["arm"])]
        at = f"{base}/rescore.tsv {line['molecule']} {line['arm']}"
        audit.cell(at, "hits_after", line["hits_after"], hits_of(g))
        audit.cell(at, "hit_mirror_after", line["hit_mirror_after"], hits_of(g, "hit_mirror"))
        audit.record(line["hits_before"] == line["hits_after"] and line["hit_mirror_before"] == line["hit_mirror_after"]
                     and line["cells_differing"] == "0", at, "no hit or mirror hit moved", "unchanged", "unchanged"
                     if line["cells_differing"] == "0" else "moved")
    audit.record(len(rescore) == 24 and not moved, f"{base}/rescore-cells.tsv", "no differing cell is listed", "0", str(len(moved)))

    series = {(r["system"], r["seed"]): r for r in guard if r["code_revision"] != "61fa67cf380e46876b569da45e7cc091647a775b"}
    develop = {(r["system"], r["seed"]): r for r in guard if r["code_revision"] == "61fa67cf380e46876b569da45e7cc091647a775b"}
    keys = ("hit", "evals_to_hit", "charged", "best")
    mismatches = sum(any(series[k][c] != develop[k][c] for c in keys) for k in develop if k in series)
    audit.record(len(series) == len(develop) == 156 and mismatches == 0, f"{base}/guard.csv",
                 "156 jobs, none differs between develop and the series", "0 mismatches", f"{mismatches} mismatches")
    audit.record(hits_of(list(series.values())) == hits_of(list(develop.values())) == 104, f"{base}/guard.csv",
                 "104 of 156 jobs hit on each side", "104", f"{hits_of(list(series.values()))} {hits_of(list(develop.values()))}")

    readme = f"{base}/README.md"
    pins = (ROOT / base / "pins.txt").read_bytes()
    audit.record(hashlib.sha1(pins).hexdigest() == "83d07464f1493b3846bc92b3d1f191331dacd586" and pins.count(b"\n") == 91,
                 f"{base}/pins.txt", "91 lines with the sha1 the README gives", "83d07464", hashlib.sha1(pins).hexdigest()[:8])
    totals = {a: (sum(len(g) and hits_of(g) for k, g in by.items() if k[1] == a),
                  sum(hits_of(g, "hit_mirror") for k, g in by.items() if k[1] == a)) for a in ("anneal", "develop", "openconf")}
    audit.said(readme, f"this series {totals['anneal'][0]}, develop {totals['develop'][0]}, openconf {totals['openconf'][0]}")
    audit.said(readme, f"With `hit_mirror`: {totals['anneal'][1]}, {totals['develop'][1]} and {totals['openconf'][1]}.")
    mirror_only = {(m, a): sum(1 for r in g if r["hit_mirror"] == "1" and r["hit"] == "0") for (m, a), g in by.items()}
    audit.said(readme, "The 8 runs that count only as a mirror hit are xld (3 anneal, 3 develop) and rq3 (2 openconf)")
    audit.record({k: v for k, v in mirror_only.items() if v} == {("xld", "anneal"): 3, ("xld", "develop"): 3, ("rq3", "openconf"): 2},
                 readme, "the runs that count only as a mirror hit", "xld 3 3, rq3 2", str({k: v for k, v in mirror_only.items() if v}))
    table = {r["molecule"]: r for r in paired}
    for line, row in md_table(audit, readme, "molecule    | series"):
        t = table[row["molecule"]]
        at = f"{readme}:{line} {row['molecule']}"
        audit.cell(at, "series", row["series"], float(t["outside_us_per_call_arm"]))
        audit.cell(at, "develop", row["develop"], float(t["outside_us_per_call_base"]))
        audit.cell(at, "ratio", row["ratio"], float(t["outside_ratio_median"]))
        audit.cell(at, "p", row["p"], float(t["wilcoxon_p"]))


def and_list(items: list[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


SC_FAMILIES = ("126", "108", "96")


def sc_order(system: str) -> tuple[int, int]:
    family, n = system.removeprefix("sc").split("-")
    return int(n), SC_FAMILIES.index(family)


def number(text: str) -> float | None:
    return float(text) if text else None


def sc_ahead(anneal: dict[str, str], gmin: dict[str, str], rows: list[dict[str, str]]) -> dict[str, str]:
    """Which code leads on hits, calls and wall; n.s. is p >= 0.05.

    On calls and wall the lead is the code lower in more seed pairs, the
    direction of the Mann-Whitney test.
    """
    a_rows = pick(rows, system=anneal["system"], arm="anneal", harness_revision=anneal["harness_revision"])
    g_rows = pick(rows, system=gmin["system"], arm="gmin", harness_revision=gmin["harness_revision"])

    def lead(column: str, p: str) -> str:
        a = [float(r[column]) for r in a_rows]
        g = [float(r[column]) for r in g_rows]
        balance = sum((y < x) - (y > x) for x in a for y in g)
        if balance == 0:
            return "tie"
        who = "GMIN" if balance > 0 else "anneal"
        return who if float(p) < 0.05 else f"{who}, n.s."

    a_hits, g_hits = int(anneal["hits"]), int(gmin["hits"])
    hits = "tie" if a_hits == g_hits else ("GMIN" if g_hits > a_hits else "anneal")
    if hits != "tie" and float(gmin["fisher_p_vs_gmin"]) >= 0.05:
        hits += ", n.s."
    return {"Hits": hits, "Calls": lead("evals_to_hit", gmin["mannwhitney_p_evals"]),
            "Wall": lead("wall_s", gmin["mannwhitney_p_wall"])}


def sc_reading(anneal: dict[str, dict[str, str]], reruns: dict[str, dict[str, str]],
               rows: list[dict[str, str]]) -> list[str]:
    """The sentences of the sc-n12 README that read the rerun tests."""
    reruns = {s: r for s, r in reruns.items()
              if all(r[c] for c in ("fisher_p_vs_gmin", "mannwhitney_p_evals", "mannwhitney_p_wall"))}
    ahead = {s: sc_ahead(anneal[s], r, rows) for s, r in reruns.items()}
    level = [s for s, r in reruns.items() if int(r["hits"]) >= int(anneal[s]["hits"])]

    def where(column: str, value: str, among: list[str]) -> list[str]:
        return [s for s in among if ahead[s][column] == value]

    out = []
    if ties := where("Hits", "tie", list(reruns)):
        out.append(f"GMIN ties anneal on hits on {and_list(ties)}.")
    for who, other in (("GMIN", "anneal"), ("anneal", "GMIN")):
        sig, ns = where("Hits", who, list(reruns)), where("Hits", f"{who}, n.s.", list(reruns))
        if sig or ns:
            parts = ([f"on {and_list(sig)} at Fisher p < 0.05"] if sig else []) + \
                    ([f"on {and_list(ns)} not significantly" if sig else f"on {and_list(ns)}, not significantly"] if ns else [])
            out.append(f"{who.capitalize() if who == 'anneal' else who} hits more seeds than {other} {', and '.join(parts)}.")
    for column, noun in (("Wall", "wall time"), ("Calls", "calls")):
        if wins := where(column, "GMIN", level):
            out.append(f"GMIN beats anneal on {noun} on {and_list(wins)}, at Mann-Whitney p < 0.05.")
        if leads := where(column, "GMIN, n.s.", level):
            out.append(f"GMIN's lead on {noun} on {and_list(leads)} is not significant.")
        if losses := where(column, "anneal", level):
            out.append(f"Anneal beats GMIN on {noun} on {and_list(losses)}, at Mann-Whitney p < 0.05.")
        if trails := where(column, "anneal, n.s.", level):
            out.append(f"Anneal's lead on {noun} on {and_list(trails)} is not significant.")
    cheap = [s for s in reruns if s not in level
             and float(reruns[s]["wall_median_s"]) < float(anneal[s]["wall_median_s"])
             and float(reruns[s]["evals_median"]) == float(reruns[s]["budget"])]
    if cheap:
        out.append(f"On {and_list(cheap)} GMIN's median wall is lower while it hits fewer seeds, "
                   "and its median calls are the full budget.")
    for s, r in reruns.items():
        if r["hits"] != "0":
            continue
        bests = sorted(float(x["best"]) for x in pick(rows, system=s, arm="gmin", harness_revision=r["harness_revision"]))
        h = r["harness_revision"][:8]
        if bests and f"{bests[0]:.4f}" == f"{bests[-1]:.4f}":
            out.append(f"On {s} every {h} GMIN seed ends at {bests[0]:.4f}.")
        elif bests:
            out.append(f"On {s} the {h} GMIN seeds end between {bests[0]:.4f} and {bests[-1]:.4f}, "
                       f"against {float(r['reference']):.4f}.")
    return out


def summary_against_rows(audit: Audit, summary: str, per_seed: str, arm_key: str, calls_key: str,
                         continuity: bool, mw_column: str = "mannwhitney_p_evals", first_only: bool = False) -> None:
    """A summary file derived from per-seed rows must match them.

    With first_only, the first anneal harness of each system and the first
    line of each (system, arm, harness) are checked here. A later anneal
    harness and a line that restates a test against it are checked by
    sutton_chen_pvalues."""
    rows = audit.rows(per_seed)
    lines = audit.rows(summary)
    if first_only:
        first_anneal: dict[str, str] = {}
        for line in lines:
            if line["arm"] == "anneal":
                first_anneal.setdefault(line["system"], line["harness_revision"])
        rows = [r for r in rows if r[arm_key] != "anneal" or r["harness_revision"] == first_anneal.get(r["system"])]
        listed: set[tuple[str, str, str]] = set()
        kept = []
        for line in lines:
            key = (line["system"], line["arm"], line["harness_revision"])
            if key in listed or (line["arm"] == "anneal" and line["harness_revision"] != first_anneal[line["system"]]):
                continue
            listed.add(key)
            kept.append(line)
        lines = kept
    for line in lines:
        harness = {"harness_revision": line["harness_revision"]} if "harness_revision" in line else {}
        group = pick(rows, system=line["system"], **{arm_key: line["arm"]}, **harness)
        where = f"{summary} {line['system']} {line['arm']}" + (f" {line['harness_revision'][:8]}" if harness else "")
        if not group:
            audit.record(False, where, "per-seed rows", line["n"] if "n" in line else "?", 0)
            continue
        calls = [float(r[calls_key]) for r in group]
        audit.cell(where, "hits", line["hits"], float(hits_of(group)))
        audit.cell(where, "evals_median", line["evals_median"], median(calls))
        if "evals_iqr" in line:
            audit.cell(where, "evals_iqr", line["evals_iqr"], iqr(calls))
        audit.cell(where, "wall_median_s", line["wall_median_s"], median([float(r["wall_s"]) for r in group]))
        for r in group:
            if r["hit"] in ("1", "true") and float(r[calls_key]) > float(r["charged"]):
                audit.record(False, per_seed, f"{r['system']} {r['seed']} hit after its charged stop", r[calls_key], r["charged"])
        # A GMIN rerun under a later harness is tested against the anneal rows;
        # the anneal row carries the tests against the GMIN rows of its own harness.
        other = pick(rows, system=line["system"], **{arm_key: "gmin" if line["arm"] == "anneal" else "anneal"},
                     **(harness if line["arm"] == "anneal" else {}))
        if line.get(mw_column):
            b = [float(r[calls_key]) for r in other]
            audit.cell(where, mw_column, line[mw_column], mann_whitney(calls, b, continuity))
        if line.get("fisher_p_vs_gmin"):
            audit.cell(where, "fisher_p_vs_gmin", line["fisher_p_vs_gmin"],
                       fisher(hits_of(group), len(group), hits_of(other), len(other)))


SC_RERUN_SUMMARY = "bench/paper-rerun/sc-h2h/summary.csv"
SC_RERUN_ROWS = "bench/paper-rerun/sc-h2h/rows.csv"
SC_RERUN_MANIFEST = "bench/paper-rerun/sc-h2h/manifest.csv"
# The committed rows blob, and the commit that stopped writing the GMIN pin
# into code_revision. The exception is these two and nothing else.
SC_H2H_ROWS_BLOB = "8e6d73ea528b6c68ecf7be18b49d15254c42b83a"
SC_H2H_RELABEL = "c2afa54b00421482de4e67c70ab8a02462bf4509"
SC_H2H_HARNESS_BLOB = "ac6b3c85b4eb435c4787a8a7425ab7e6858f9b34"
SC_H2H_GMIN_PIN = "9cbec1a446352bc9ef064bd2b1a5767194655919"
SC_H2H_CODE = "61fa67cf380e46876b569da45e7cc091647a775b"
SC_H2H_README = "bench/paper-rerun/sc-h2h/README.md"
SC_N12_README = "bench/metal-cluster-h2h/sc-n12/README.md"


def side_rows(audit: Audit, path: str) -> list[dict[str, str]]:
    """A file without revision columns. A test may already have loaded it; the audit does not, so provenance stays on the row file."""
    cached = audit.read.get(path)
    if cached is not None:
        return cached
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sc_rerun_summary(audit: Audit) -> list[dict[str, str]]:
    return side_rows(audit, SC_RERUN_SUMMARY)


def sc_rerun_code(audit: Audit) -> str:
    """The one code revision every rerun row records, which must be an ancestor of HEAD."""
    revisions = sorted({r["code_revision"].strip() for r in audit.rows(SC_RERUN_ROWS)})
    audit.record(len(revisions) == 1, SC_RERUN_ROWS, "one code revision", " ".join(revisions), "one")
    code = revisions[0] if revisions else ""
    audit.record(commit_of(code) == code and ancestor_of_head(code), SC_RERUN_ROWS,
                 "the rerun code revision is an ancestor of HEAD", code, "a full SHA, ancestor of HEAD")
    return code


def sc_rerun_tests(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """The summary each system and arm of the rerun rows gives, in summariser order.

    Mann-Whitney ranks a miss after every hit, at the budget plus one call.
    """
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[(r["system"], r["arm"])].append(r)
    systems = list(dict.fromkeys(r["system"] for r in rows))

    def ranked(group: list[dict[str, str]]) -> list[float]:
        return [float(r["evals_to_hit"]) if r["hit"] == "1" else float(r["budget"]) + 1.0 for r in group]

    out = []
    for system in systems:
        anneal, gmin = grouped.get((system, "anneal"), []), grouped.get((system, "gmin"), [])
        tested = bool(anneal and gmin and len(anneal) == len(gmin))
        for arm, group in (("anneal", anneal), ("gmin", gmin)):
            if not group:
                continue
            budgets = sorted({r["budget"] for r in group})
            out.append({
                "system": system, "arm": arm, "n": str(len(group)), "hits": str(hits_of(group)),
                "hits_over_n": frac(group), "budget": " ".join(budgets),
                "calls_median": median([float(r["evals_to_hit"]) for r in group]),
                "fisher_p_vs_gmin": fisher(hits_of(anneal), len(anneal), hits_of(gmin), len(gmin))
                if tested and arm == "anneal" else None,
                "mannwhitney_p_calls": mann_whitney(ranked(anneal), ranked(gmin), True)
                if tested and arm == "anneal" else None,
            })
    return out


def sc_h2h_relabel_commit(commit: str) -> bool:
    """True when commit changed the harness from writing the GMIN pin.

    The parent of that commit holds harness blob ac6b3c85, which assigns
    GMIN_PIN on the GMIN arm. The commit itself assigns CODE_REVISION on
    every arm. Any other commit is not the recorded relabel.
    """
    if commit_of(commit) != commit:
        return False
    parent = git("rev-parse", "--verify", "--quiet", f"{commit}^").stdout.strip()
    if not parent:
        return False
    before = git("rev-parse", "--verify", "--quiet", f"{parent}:scripts/paper_rerun/sc_h2h.py").stdout.strip()
    after = git("rev-parse", "--verify", "--quiet", f"{commit}:scripts/paper_rerun/sc_h2h.py").stdout.strip()
    if before != SC_H2H_HARNESS_BLOB or not after or after == before:
        return False
    old = git("show", f"{parent}:scripts/paper_rerun/sc_h2h.py").stdout
    new = git("show", f"{commit}:scripts/paper_rerun/sc_h2h.py").stdout
    wrote_pin = "code = GMIN_PIN" in old and f'GMIN_PIN = "{SC_H2H_GMIN_PIN}"' in old
    writes_code = "code = GMIN_PIN" not in new and "CODE_REVISION" in new
    return wrote_pin and writes_code


def sc_h2h_code_relabel(audit: Audit) -> None:
    """GMIN code_revision may be the anneal SHA only for one recorded file.

    Harness blob ac6b3c85 writes the GMIN pin into code_revision. Commit
    c2afa54b relabelled the committed rows, blob 8e6d73ea, to the anneal
    SHA. A different relabel commit, or any other rows blob, is held to
    the pin and fails closed.
    """
    blob = git("hash-object", str(ROOT / SC_RERUN_ROWS)).stdout.strip()
    audit.record(blob == SC_H2H_ROWS_BLOB, SC_RERUN_ROWS, "rows blob", blob, SC_H2H_ROWS_BLOB)
    relabel = sc_h2h_relabel_commit(SC_H2H_RELABEL)
    audit.record(relabel, SC_RERUN_ROWS, "relabel commit", SC_H2H_RELABEL,
                 "the commit that writes the anneal SHA into every code_revision")
    rows = audit.rows(SC_RERUN_ROWS)
    gmin = [r for r in rows if r["arm"] == "gmin"]
    codes = sorted({r["code_revision"].strip() for r in gmin})
    pins = sorted({r["gmin_pin"].strip() for r in gmin})
    blobs = sorted({r["harness_blob"].strip() for r in rows})
    expect = SC_H2H_CODE if blob == SC_H2H_ROWS_BLOB and relabel else SC_H2H_GMIN_PIN
    audit.record(codes == [expect] and pins == [SC_H2H_GMIN_PIN] and blobs == [SC_H2H_HARNESS_BLOB],
                 SC_RERUN_ROWS, "gmin code_revision relabel", " ".join(codes) or "absent", expect)
    for path in (SC_H2H_README, SC_N12_README):
        audit.said(path, f"blob `{SC_H2H_ROWS_BLOB}`")
        audit.said(path, f"Harness blob `{SC_H2H_HARNESS_BLOB}`")
        audit.said(path, f"Commit `{SC_H2H_RELABEL}`")
        audit.said(path, f"anneal revision `{SC_H2H_CODE}`")
        audit.said(path, f"pin `{SC_H2H_GMIN_PIN}`")


def sc_h2h_harness_blob(audit: Audit) -> None:
    """The rows' harness blob is the harness file at the cited harness revision."""
    rows = audit.rows(SC_RERUN_ROWS)
    revisions = sorted({r["harness_revision"].strip() for r in rows})
    blobs = sorted({r["harness_blob"].strip() for r in rows})
    revision = revisions[0] if len(revisions) == 1 else ""
    expected = ""
    if revision:
        expected = git("rev-parse", "--verify", "--quiet",
                       f"{revision}:scripts/paper_rerun/sc_h2h.py").stdout.strip()
    audit.cell(SC_RERUN_ROWS, "harness_blob", " ".join(blobs), expected)


def sc_rerun_against_rows(audit: Audit) -> None:
    """The rerun rows are the manifest's cells once each, and the summary is what they give."""
    rows = audit.rows(SC_RERUN_ROWS)
    manifest = side_rows(audit, SC_RERUN_MANIFEST)
    audit.record(len(rows) == len(manifest), SC_RERUN_ROWS, "rows", len(rows), len(manifest))
    count: dict[tuple[str, str, str], int] = defaultdict(int)
    for r in rows:
        count[(r["system"], r["arm"], r["seed"])] += 1
    cells = {(m["system"], m["arm"], m["seed"]): m for m in manifest}
    for key, m in cells.items():
        audit.record(count.get(key, 0) == 1, SC_RERUN_ROWS, f"manifest cell {' '.join(key)}", count.get(key, 0), 1)
    for key in sorted(set(count) - set(cells)):
        audit.record(False, SC_RERUN_ROWS, f"row outside the manifest {' '.join(key)}", count[key], 0)
    for r in rows:
        m = cells.get((r["system"], r["arm"], r["seed"]))
        if m and r["budget"] != m["budget"]:
            audit.record(False, SC_RERUN_ROWS, f"budget {r['system']} {r['arm']} {r['seed']}", r["budget"], m["budget"])
    references = {m["system"]: m["reference"] for m in manifest}
    summary = {(s["system"], s["arm"]): s for s in sc_rerun_summary(audit)}
    computed = {(c["system"], c["arm"]): c for c in sc_rerun_tests(rows)}
    for key in sorted(set(summary) | set(computed)):
        where = f"{SC_RERUN_SUMMARY} {key[0]} {key[1]}"
        line, value = summary.get(key), computed.get(key)
        if line is None or value is None:
            audit.record(False, where, "summary row", "shown" if line else "missing", "rows" if value else "no rows")
            continue
        for column in ("n", "hits", "hits_over_n", "budget"):
            audit.cell(where, column, line[column], value[column])
        audit.cell(where, "reference", line["reference"], references.get(key[0], "not in the manifest"))
        for column in ("calls_median", "fisher_p_vs_gmin", "mannwhitney_p_calls"):
            audit.cell(where, column, line[column], value[column])


def sc_gmin_pin(audit: Audit) -> str:
    """The one GMIN pin the rerun rows record. Empty when they disagree."""
    pins = sorted({r["gmin_pin"].strip() for r in audit.rows(SC_RERUN_ROWS) if r["arm"] == "gmin"})
    return pins[0] if len(pins) == 1 else ""


def sc_develop_table(audit: Audit, path: str, code: str) -> None:
    """The printed Sutton-Chen counts are the develop rerun.

    Anneal rows name the code revision. GMIN rows name the GMIN pin.
    """
    summary = sc_rerun_summary(audit)
    pin = sc_gmin_pin(audit)
    shown = {}
    for line, row in md_table(audit, path, "System | Arm | Code revision"):
        shown[(row["System"], row["Arm"])] = (line, row)
    for row in summary:
        key = (row["system"], row["arm"])
        got = shown.get(key)
        audit.record(got is not None, path, f"table row {row['system']} {row['arm']}",
                     "shown" if got else "missing", "shown")
        if got is None:
            continue
        line, cell = got
        where = f"{path}:{line} {row['system']} {row['arm']}"
        shown_rev = cell["Code revision"].strip().strip("`")
        if row["arm"] == "gmin":
            # The printed GMIN cell is the pin in the rows' gmin_pin column.
            # code_revision on those rows is the anneal commit the cells
            # were rerun beside, which is not the executable GMIN ran.
            audit.record(bool(pin) and len(shown_rev) >= 7 and pin.startswith(shown_rev), where,
                         "gmin pin", shown_rev, pin[:8])
        else:
            audit.record(bool(code) and commit_of(shown_rev) == code, where,
                         "code revision", shown_rev, code[:8])
        audit.cell(where, "hits", cell["Hits"], row["hits_over_n"])
        audit.cell(where, "calls median", cell["Calls median"], float(row["calls_median"]))
        if row["fisher_p_vs_gmin"] or cell["Fisher p"].strip():
            audit.cell(where, "Fisher p", cell["Fisher p"],
                       float(row["fisher_p_vs_gmin"]) if row["fisher_p_vs_gmin"] else None)
        if row["mannwhitney_p_calls"] or cell["Mann-Whitney p calls"].strip():
            audit.cell(where, "Mann-Whitney p calls", cell["Mann-Whitney p calls"],
                       float(row["mannwhitney_p_calls"]) if row["mannwhitney_p_calls"] else None)


def head_to_head(audit: Audit) -> None:
    summary_against_rows(audit, "bench/morse-h2h/results.csv", "bench/morse-h2h/seeds.csv", "code", "evals_to_hit", True)
    summary_against_rows(audit, "bench/metal-cluster-h2h/sc-n12/results.csv", "bench/metal-cluster-h2h/sc-n12/seeds.csv",
                         "arm", "evals_to_hit", True, first_only=True)
    sc_rows = audit.rows("bench/metal-cluster-h2h/sc-n12/seeds.csv")
    sc_summary = audit.rows("bench/metal-cluster-h2h/sc-n12/results.csv")
    anneal = {r["system"]: r for r in sc_summary if r["arm"] == "anneal"}
    anneal_harnesses = {
        system: {
            row["harness_revision"] for row in sc_summary
            if row["system"] == system and row["arm"] == "anneal"
        }
        for system in anneal
    }
    reruns: dict[str, dict[str, str]] = {}
    for r in sc_summary:
        if r["arm"] == "gmin" and r["harness_revision"] not in anneal_harnesses[r["system"]]:
            reruns[r["system"]] = r
    reruns = dict(sorted(reruns.items(), key=lambda item: sc_order(item[0])))
    path = "bench/metal-cluster-h2h/sc-n12/README.md"
    shown = set()
    for line, row in md_table(audit, path, "System | Arm | Harness"):
        group = [r for r in pick(sc_rows, system=row["System"], arm=row["Arm"])
                 if r["harness_revision"].startswith(row["Harness"])]
        where = f"{path}:{line} {row['System']} {row['Arm']} {row['Harness']}"
        shown.add((row["System"], row["Arm"], row["Harness"]))
        audit.cell(where, "seeds", "12", float(len(group)))
        audit.cell(where, "hits", row["Hits"], frac(group))
        if group:
            audit.cell(where, "calls median", row["Calls median"], median([float(r["evals_to_hit"]) for r in group]))
            audit.cell(where, "wall median", row["Wall median (s)"], median([float(r["wall_s"]) for r in group]))
    tested = {}
    for line, row in md_table(audit, path, "System | Anneal harness | GMIN harness"):
        where = f"{path}:{line} {row['System']} {row['GMIN harness']}"
        r = reruns.get(row["System"])
        tested[row["System"]] = row["GMIN harness"]
        audit.cell(where, "latest GMIN harness", row["GMIN harness"], r["harness_revision"][:8] if r else "not rerun")
        if r:
            audit.cell(where, "Fisher p", row["Fisher p"], number(r["fisher_p_vs_gmin"]))
            audit.cell(where, "Mann-Whitney p calls", row["Mann-Whitney p calls"], number(r["mannwhitney_p_evals"]))
            audit.cell(where, "Mann-Whitney p wall", row["Mann-Whitney p wall"], number(r["mannwhitney_p_wall"]))
            audit.cell(where, "wall ratio", row["Wall ratio"], number(r["wall_ratio"]))
            if not all(r[c] for c in ("fisher_p_vs_gmin", "mannwhitney_p_evals", "mannwhitney_p_wall")):
                continue
            ahead_row = sc_ahead(anneal[row["System"]], r, sc_rows)
            for column in ("Hits", "Calls", "Wall"):
                audit.cell(where, f"{column} ahead", row[f"{column} ahead"], ahead_row[column])
    anneal_harness = {system: row["harness_revision"] for system, row in anneal.items()}
    for system, r in reruns.items():
        for arm, harness in (("anneal", anneal_harness[system]), ("gmin", r["harness_revision"])):
            audit.cell(path, f"table row {system} {arm} {harness[:8]}",
                       "shown" if (system, arm, harness[:8]) in shown else "missing", "shown")
        audit.cell(path, f"test row {system}", tested.get(system, "missing"), r["harness_revision"][:8])
    for system in tested:
        audit.cell(path, f"test row {system} is a rerun", "rerun" if system in reruns else "not rerun", "rerun")
    for system in anneal_harness:
        audit.cell(path, f"{system} GMIN row from the corrected file",
                   "rerun" if system in reruns else f"old file only ({anneal_harness[system][:8]})", "rerun")
    for sentence in sc_reading(anneal, reruns, sc_rows):
        audit.said(path, sentence)
    pvalues = audit.rows("bench/metal-cluster-h2h/sc-n12/pvalues.csv")
    for system, a_summary in anneal.items():
        if system not in reruns:
            continue
        g_summary = reruns[system]
        a_rows = [
            row for row in sc_rows
            if row["system"] == system and row["arm"] == "anneal"
            and row["harness_revision"] == a_summary["harness_revision"]
        ]
        g_rows = [
            row for row in sc_rows
            if row["system"] == system and row["arm"] == "gmin"
            and row["harness_revision"] == g_summary["harness_revision"]
        ]
        if not a_rows or not g_rows:
            audit.record(False, f"bench/metal-cluster-h2h/sc-n12/pvalues.csv {system}",
                         "per-seed rows", "12", f"{len(a_rows)} {len(g_rows)}")
            continue
        tests = [
            row for row in pvalues
            if row["system"] == system
            and row["code_revision"] == a_summary["code_revision"]
            and row["harness_revision"] == g_summary["harness_revision"]
        ]
        where = f"bench/metal-cluster-h2h/sc-n12/pvalues.csv {system}"
        for label, summary, rows in (("anneal", a_summary, a_rows), ("gmin", g_summary, g_rows)):
            audit.cell(where, f"{label} n", summary["n"], len(rows))
            audit.cell(where, f"{label} hits", summary["hits"], hits_of(rows))
            audit.cell(where, f"{label} calls median", summary["evals_median"],
                       median([float(row["evals_to_hit"]) for row in rows]))
            audit.cell(where, f"{label} wall median", summary["wall_median_s"],
                       median([float(row["wall_s"]) for row in rows]))
            audit.cell(where, f"{label} inside median", summary["pot_median_s"],
                       median([float(row["pot_s"]) for row in rows]))
            audit.cell(where, f"{label} outside median", summary["outside_median_s"],
                       median([float(row["outside_s"]) for row in rows]))
        audit.cell(where, "latest test rows", str(len(tests)), 1)
        if not tests:
            continue
        test = tests[0]
        audit.cell(where, "Fisher p", test["fisher_p"],
                   fisher(hits_of(a_rows), len(a_rows), hits_of(g_rows), len(g_rows)))
        audit.cell(where, "Mann-Whitney p calls", test["mannwhitney_p_evals"],
                   mann_whitney([float(row["evals_to_hit"]) for row in a_rows],
                                [float(row["evals_to_hit"]) for row in g_rows], True))
        audit.cell(where, "Mann-Whitney p wall", test["mannwhitney_p_wall"],
                   mann_whitney([float(row["wall_s"]) for row in a_rows],
                                [float(row["wall_s"]) for row in g_rows], True))
    paired = [system for system in anneal if system in reruns]
    ahead = sum(int(anneal[system]["hits"]) > int(reruns[system]["hits"]) for system in paired)
    tied = sum(int(anneal[system]["hits"]) == int(reruns[system]["hits"]) for system in paired)
    behind = len(paired) - ahead - tied
    gmin_wall = sum(
        float(reruns[system]["wall_median_s"]) < float(anneal[system]["wall_median_s"])
        for system in paired
    )
    h2h_quote = load_script("h2h_sc_quote", "paper/jctc/scripts/h2h_tables.py")
    if h2h_quote.sc_gmin_quotable():
        audit.said("paper/jctc/manuscript.tex",
                   f"This work has more hits on {ahead} of the {len(anneal)} systems")
        audit.said("paper/jctc/manuscript.tex",
                   f"the arms tie on {tied}, and GMIN has more hits on {behind}")
        audit.said("paper/jctc/manuscript.tex",
                   f"GMIN nevertheless has the lower median raw wall time on {gmin_wall} of {len(anneal)} systems")

    path = "bench/metal-cluster-h2h/sc-n12/README.md"
    sc_rerun_against_rows(audit)
    sc_h2h_harness_blob(audit)
    sc_h2h_code_relabel(audit)
    code = sc_rerun_code(audit)
    sc_develop_table(audit, path, code)
    sc_develop_table(audit, "bench/paper-rerun/sc-h2h/README.md", code)
    sc_summary = audit.rows("bench/metal-cluster-h2h/sc-n12/results.csv")
    # The stored sc-n12 rows keep the corrected-file GMIN rerun beside the
    # earlier file. The README no longer prints those revisions.
    anneal = {r["system"]: r for r in sc_summary if r["arm"] == "anneal"}
    anneal_harness = {s: r["harness_revision"] for s, r in anneal.items()}
    reruns: dict[str, dict[str, str]] = {}
    for r in sc_summary:
        if r["arm"] == "gmin" and r["harness_revision"] != anneal_harness[r["system"]]:
            reruns[r["system"]] = r
    reruns = dict(sorted(reruns.items(), key=lambda item: sc_order(item[0])))
    for system in anneal_harness:
        audit.cell(path, f"{system} GMIN row from the corrected file",
                   "rerun" if system in reruns else f"old file only ({anneal_harness[system][:8]})", "rerun")
    audit.said("bench/metal-cluster-h2h/sc-8e076b7f/README.md",
               f"`../sc-n12/README.md` has the corrected file and GMIN reruns of {and_list(reruns)}.")

    path = "bench/morse-h2h/README.md"
    results = audit.rows("bench/morse-h2h/results.csv")
    m38 = pick(results, system="m38r6", arm="anneal")[0]
    audit.said(path, f"{float(m38['evals_median']):.0f}")

    water = audit.rows("bench/water-gmin/rows.csv")
    path = "bench/water-gmin/README.md"
    sizes = {"(H₂O)₈": "tip4p8", "(H₂O)₁₂": "tip4p12", "(H₂O)₁₆": "tip4p16", "(H₂O)₂₀": "tip4p20", "(H₂O)₂₁": "tip4p21"}
    for line, row in md_table(audit, path, "Size | anneal | GMIN"):
        system = sizes[row["Size"]]
        a, g = pick(water, system=system, code="anneal"), pick(water, system=system, code="gmin")
        where = f"{path}:{line} {system}"
        audit.cell(where, "anneal hits", row["anneal"], frac(a))
        audit.cell(where, "GMIN hits", row["GMIN"], frac(g))
        for arm, rows_ in (("anneal", a), ("GMIN", g)):
            shown_all, shown_hit = row[f"{arm} calls (all / hits)"].split(" / ")
            hit_calls = [float(r["evals_to_hit"]) for r in rows_ if r["hit"] == "1"]
            audit.cell(where, f"{arm} calls all", shown_all, median([float(r["evals_to_hit"]) for r in rows_]))
            audit.cell(where, f"{arm} calls hits", shown_hit, median(hit_calls) if hit_calls else None)
        wall_a, wall_g = row["wall a / g"].split(" / ")
        audit.cell(where, "anneal wall", wall_a, median([float(r["wall_s"]) for r in a]))
        audit.cell(where, "GMIN wall", wall_g, median([float(r["wall_s"]) for r in g]))
        pot, out = row["anneal pot / out"].split(" / ")
        audit.cell(where, "anneal pot", pot, median([float(r["pot_s"]) for r in a]))
        audit.cell(where, "anneal out", out, median([float(r["outside_s"]) for r in a]))
        audit.cell(where, "GMIN pot", row["GMIN pot"], median([float(r["pot_s"]) for r in g]))
        audit.cell(where, "p(F)", row["p(F)"], fisher(hits_of(a), len(a), hits_of(g), len(g)))
        audit.cell(where, "p(U), no continuity correction", row["p(U)"],
                   mann_whitney([float(r["evals_to_hit"]) for r in a], [float(r["evals_to_hit"]) for r in g], continuity=False))
    twelve = pick(water, system="tip4p12", code="anneal")
    audit.said(path, "where the median run stops within 0.25 s.")
    audit.cell(path, "median (H2O)12 anneal wall under 0.25 s", "0.229", median([float(r["wall_s"]) for r in twelve]))
    for system in ("tip4p16", "tip4p21"):
        for code in ("anneal", "gmin"):
            audit.cell(path, f"{system} {code} all-seed calls median is the budget", "2000000",
                       median([float(r["evals_to_hit"]) for r in pick(water, system=system, code=code)]))

    rho6 = audit.rows("bench/morse-h2h/seeds-cdb82168.csv")
    hits = sorted((r["system"], r["seed"]) for r in rho6 if r["hit"] == "1" and int(r["charged"]) < int(r["budget"]))
    audit.cell("bench/morse-h2h/README-cdb82168.md", "hits stopping inside the budget",
               "m55r14 2000 2001 2003 2004 2005; m80r14 2000 2001 2002 2003",
               "; ".join(f"{s} " + " ".join(seed for sys_, seed in hits if sys_ == s) for s in ("m55r14", "m80r14")))


SC96_78_ROWS = "data/sc96_78_hits.csv"
SC96_78_HOWTO = "docs/orgmode/howto/sutton-chen-head-to-head.org"
SC96_78_BUDGET = 2_000_000


def signflip(diffs: list[float], draws: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip p of docs/scripts/plot_sc_hits.py."""
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        total = sum(v if rng.randrange(2) == 0 else -v for v in diffs)
        if abs(total / len(diffs)) >= abs(observed) - 1e-15:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def sc96_78_howto(audit: Audit) -> None:
    """The 9-6 N=78 table scores the GMIN rows of the last harness revision."""
    rows = audit.rows(SC96_78_ROWS)
    gmin = [r for r in rows if r["arm"] == "gmin" and r["harness_revision"]]
    harness = gmin[-1]["harness_revision"]
    arms = {"gmin": pick(gmin, harness_revision=harness), "iso": pick(rows, arm="iso"), "iso-hop": pick(rows, arm="iso-hop")}
    par2 = {arm: {r["seed"]: float(r["hit_calls"]) if r["hit"] == "1" else 2.0 * SC96_78_BUDGET for r in group}
            for arm, group in arms.items()}
    text = audit.text(SC96_78_HOWTO)
    start = text.index("#+begin_example\nlabel hits")
    block = text[start:text.index("#+end_example", start)].splitlines()[2:]
    audit.cell(SC96_78_HOWTO, "table arms", " ".join(line.split()[0] for line in block), " ".join(arms))
    for line in block:
        label, hits, med, mean, p = line.split()
        group = arms[label]
        where = f"{SC96_78_HOWTO} {label}"
        calls = [float(r["hit_calls"]) for r in group if r["hit"] == "1"]
        audit.cell(where, "hits", hits, frac(group))
        audit.cell(where, "median calls to a hit", med, median(calls) if calls else None)
        audit.cell(where, "mean PAR-2", mean, sum(par2[label].values()) / len(group))
        common = sorted(set(par2["gmin"]) & set(par2[label]))
        audit.cell(where, "paired sign-flip p against gmin", p,
                   None if label == "gmin" else signflip([par2["gmin"][s] - par2[label][s] for s in common]))
        audit.said(SC96_78_HOWTO, f"={label}= hits {hits_of(group)} of {len(group)} seeds")
    audit.said(SC96_78_HOWTO, f"Its rows are those of harness {harness[:8]}")
    earlier = [r for r in rows if r["arm"] == "gmin" and r["harness_revision"] != harness]
    audit.said(SC96_78_HOWTO, f"hit {hits_of(earlier)} of {len(earlier)} seeds. They are kept in the file.")


# Numbers of the GMIN rows from the misread data file, as they were cited.
STALE_GMIN = re.compile(r"\b0/32\b|\b0 of 32\b|\bp\s*[<≤]\s*\$?(?:1e-4|10\^\{?-4\}?|0\.0001)|(?<![\d.])0\.0000\b")
RETIRED = ("Only 9-6 at N=38 has been rerun", "a rerun of 9-6 at 38 atoms", "from the old file, and are not rerun",
           "This article reports no force-call count for that optimiser",
           "hits more seeds on 9 of 13 and GMIN hits more on 2", "168.9~s", "6 of the 9 wins",
           "lower for this code on 12 of the 13 clusters", "12/13")
# Sentences that cite the old rows as old rows; each is derived from the rows elsewhere.
HISTORICAL = {
    (SC96_78_HOWTO, "The earlier GMIN rows, with temperature 0.8, step 0.40 and a container radius of 30, hit 0 of 32 seeds."),
}


def documents() -> list[str]:
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.md", "*.org", "*.tex", "*.rst"],
                            capture_output=True, text=True, check=True).stdout
    return listed.split()


def stale_claims(audit: Audit, paths: list[str] | None = None) -> None:
    """No sentence about GMIN still cites a number of its old data file."""
    seen = set()
    for path in sorted(set(documents() if paths is None else paths)):
        text = audit.text(path)
        for phrase in RETIRED:
            if phrase in collapse(text):
                audit.record(False, path, "retired claim", phrase, "absent")
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
            if re.search(r"gmin", sentence, re.I) and STALE_GMIN.search(sentence):
                key = (path, collapse(sentence))
                seen.add(key)
                audit.record(key in HISTORICAL, path, "old GMIN number", collapse(sentence), "derived or absent")
    scanned = set(documents() if paths is None else paths)
    for path, sentence in sorted(h for h in HISTORICAL - seen if h[0] in scanned):
        audit.record(False, path, "historical sentence", sentence, "present")


def gfn2_ranked(rows: list[dict[str, str]]) -> list[float]:
    """Calls to the hit with a miss ranked after every hit, as gfn2_crest.py --summary ranks them."""
    return [float(r["evals_to_hit"]) if hits_of([r]) else math.inf for r in rows]


def gfn2_table(audit: Audit, path: str, rows: list[tuple[int, dict[str, str]]],
               groups: dict[tuple[str, str], list[dict[str, str]]], per_call: str) -> None:
    """``per_call`` "pooled" is summed time over summed calls; "median" is the median over runs that made a call."""
    for line, row in rows:
        group = groups[(row["system"], row["code"])]
        where = f"{path}:{line} {row['system']} {row['code']}"
        calls = [float(r["evals_to_hit"]) for r in group]
        charged = sum(float(r["charged"]) for r in group)

        def time_per_call(column: str) -> float:
            if per_call == "pooled":
                return sum(float(r[column]) for r in group) / charged * 1e3
            return median([float(r[column]) / float(r["charged"]) * 1e3 for r in group if float(r["charged"])])

        audit.cell(where, "hits", row["hits"], frac(group))
        audit.cell(where, "median calls", row["median calls"], median(calls))
        audit.cell(where, "IQR", row["IQR"], iqr(calls))
        audit.cell(where, "best", row["best / eV"], min(float(r["best"]) for r in group))
        audit.cell(where, "median wall", row["median wall / s"], median([float(r["wall_s"]) for r in group]))
        audit.cell(where, "inside / call ms", row["inside / call"].split()[0], time_per_call("pot_s"))
        audit.cell(where, "outside / call ms", row["outside / call"].split()[0], time_per_call("outside_s"))


def covalent_gfn2(audit: Audit) -> None:
    """The GFN2 tables of the covalent README.

    The first table takes anneal's rows from seeds.csv and CREST's from seeds.csv
    and rows.csv. The paper's Si10 table takes anneal's rows of rows.csv at the
    revision its code cell names, with each row false-hits.csv names counted as
    a miss at its budget and given the energy of its geometry.
    """
    path = "bench/covalent-h2h/README.md"
    header = "system | code | hits | median calls | IQR | best / eV"
    seeds = audit.rows("bench/covalent-h2h/gfn2/seeds.csv")
    rows = audit.rows("bench/covalent-h2h/gfn2/rows.csv")
    named = {(r["system"], r["code"], r["code_revision"], r["seed"]): r
             for r in audit.rows("bench/covalent-h2h/gfn2/false-hits.csv")}
    systems = {"Si10": "si10", "C10": "c10"}

    def crest(system: str) -> list[dict[str, str]]:
        return pick(seeds, system=system, code="crest") + pick(rows, system=system, code="crest")

    first = md_table(audit, path, header)
    groups = {}
    for _, row in first:
        system = systems[row["system"]]
        code = {"anneal G": "anneal-general", "CREST": "crest"}[row["code"]]
        groups[(row["system"], row["code"])] = crest(system) if code == "crest" else pick(seeds, system=system, code=code)
    gfn2_table(audit, path, first, groups, "pooled")
    a, c = groups[("Si10", "anneal G")], groups[("Si10", "CREST")]
    audit.cell(path, "GFN2 Si10 Fisher p", "1", fisher(hits_of(a), len(a), hits_of(c), len(c)))
    audit.cell(path, "GFN2 Si10 Mann-Whitney p", "3.7e-5", mann_whitney(gfn2_ranked(a), gfn2_ranked(c), continuity=True))
    audit.said(path, "CREST stops and CREST makes no call. Mann-Whitney p = 3.7e-5 over the twelve seeds of each code, "
                     "a miss ranked after every hit")
    a, c = groups[("C10", "anneal G")], groups[("C10", "CREST")]
    p = mann_whitney([float(r["evals_to_hit"]) for r in a], [float(r["evals_to_hit"]) for r in c], continuity=True)
    audit.cell(path, "GFN2 C10 Mann-Whitney p", "0.0005", p)
    audit.cell(path, "GFN2 C10 Fisher p", "0.0094", fisher(hits_of(a), len(a), hits_of(c), len(c)))
    audit.said(path, "CREST on 4 (Fisher p = 0.0094, Mann-Whitney p = 0.0005)")

    paper = md_table(audit, path, header, after=first[-1][0] + 1)
    revision = re.search(f"`({revision_reach.commit_id_pattern(7)})`", paper[0][1]["code"]).group(1)
    recorded = [r for r in pick(rows, system="si10", code="anneal-general") if r["code_revision"].startswith(revision)]
    counted = []
    for r in recorded:
        false_hit = named.get((r["system"], r["code"], r["code_revision"], r["seed"]))
        counted.append(r if false_hit is None else dict(r, hit="false", evals_to_hit=r["budget"], best=false_hit["xtb_eV"]))
    groups = {("Si10", paper[0][1]["code"]): counted, ("Si10", "CREST"): crest("si10")}
    gfn2_table(audit, path, paper, groups, "median")
    c = crest("si10")
    audit.cell(path, "GFN2 paper Si10 Fisher p", "1", fisher(hits_of(counted), len(counted), hits_of(c), len(c)))
    audit.cell(path, "GFN2 paper Si10 Mann-Whitney p", "5.3e-4",
               mann_whitney(gfn2_ranked(counted), gfn2_ranked(c), continuity=True))
    audit.said(path, "Mann-Whitney p = 5.3e-4 on calls to the hit, both misses ranked after every hit.")
    calls = [float(r["evals_to_hit"]) for r in recorded]
    audit.cell(path, "GFN2 Si10 recorded hits", "12/12", frac(recorded))
    audit.cell(path, "GFN2 Si10 recorded median calls", "720.5", median(calls))
    audit.cell(path, "GFN2 Si10 recorded IQR", "925", iqr(calls))
    audit.cell(path, "GFN2 Si10 recorded Mann-Whitney p", "3.7e-5",
               mann_whitney(gfn2_ranked(recorded), gfn2_ranked(c), continuity=True))
    audit.said(path, "anneal has 12/12, median 720.5 calls and IQR 925, and CREST has Fisher p = 1 and Mann-Whitney p = 3.7e-5")
    shown = [f"{int(r['evals_to_hit']):,}" for r in sorted(recorded, key=lambda r: int(r["seed"]))]
    audit.said(path, f"Anneal takes {', '.join(shown[:9])}, {shown[9]} (the false hit), {shown[10]} and {shown[11]} "
                     "calls on seeds 2000-2011")
    constants = load_script("paper_constants", "paper/jctc/scripts/paper_constants.py")
    proven_rgpot = constants.values()["RgpotRevision"]
    proven_engine = "4dac51020e4f1419fa930d95894b581012dabf47681b3a883cba21063f7e8672"
    audit.said("bench/covalent-h2h/gfn2/scc-false-minimum/README.md", proven_rgpot)
    audit.said("bench/covalent-h2h/gfn2/scc-false-minimum/README.md", proven_engine)
    audit.record("rgpot_revision" not in rows[0], "bench/covalent-h2h/gfn2/rows.csv",
                 "rows.csv keeps its original header without rgpot_revision",
                 "absent", "present" if "rgpot_revision" in rows[0] else "absent")
    sidecar = sidecar_rows(audit, GFN2_RGPOT_SIDECAR)
    row_keys = [(r["system"], r["code"], r["seed"], r["code_revision"], r["engine_sha256"], r["host"]) for r in rows]
    side_keys = [(r["system"], r["code"], r["seed"], r["code_revision"], r["engine_sha256"], r["host"]) for r in sidecar]
    audit.record(row_keys == side_keys, GFN2_RGPOT_SIDECAR,
                 "each covalent-h2h gfn2 row has one matching rgpot-revisions.csv sidecar row",
                 f"{len(row_keys)} aligned rows",
                 f"{len(side_keys)} rows, {'aligned' if row_keys == side_keys else 'mismatched'}")
    for s_row in sidecar:
        want_rgpot = proven_rgpot if s_row["engine_sha256"] == proven_engine else ""
        audit.record(s_row.get("rgpot_revision") == want_rgpot, GFN2_RGPOT_SIDECAR,
                     f"{s_row['system']} {s_row['code']} seed {s_row['seed']} rgpot_revision",
                     want_rgpot or "empty", s_row.get("rgpot_revision", "") or "empty")
    gfn2_revision_tables(audit, path, header, paper[-1][0] + 1, rows, crest)


GFN2_SYSTEMS = {"Si10": "si10", "C10": "c10", "C20": "c20"}
# (system, anneal revision, Fisher p, Mann-Whitney p, its sample, sentence that prints them). The sample is
# "calls", calls to the hit with a miss ranked after every hit, or "best", the best energies. The sentence
# is filled from the rows by gfn2_sentence_fields.
GFN2_REVISION_TESTS = (
    ("Si10", "ce1e7a46", "1", "3.7e-5", "calls",
     "anneal reaches the minimum on {anneal_hits} of {runs} seeds and CREST on {crest_hits} "
     "(Fisher p = {fisher_p}, Mann-Whitney p = {mw_p}, a miss ranked after every hit)"),
    ("C10", "ce1e7a46", "0.0013", "2.6e-5", "calls",
     "anneal reaches the ring on {anneal_hits} of {runs} seeds and CREST on {crest_hits} "
     "(Fisher p = {fisher_p}, Mann-Whitney p = {mw_p})"),
    ("C20", "5351f272", "1", "0.0004", "best",
     "Anneal's best is lower than CREST's on {below} of the {pairs} pairs of seeds "
     "(Mann-Whitney p = {mw_p} on the best energies)"),
)
NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def gfn2_sentence_fields(a: list[dict[str, str]], c: list[dict[str, str]]) -> dict[str, object]:
    """The counts a GFN2_REVISION_TESTS sentence prints, from anneal's rows ``a`` and CREST's ``c``."""
    def span(rows: list[dict[str, str]]) -> str:
        calls = sorted(int(r["evals_to_hit"]) for r in rows if hits_of([r]))
        return f"{calls[0]:,} to {calls[-1]:,}" if calls else "no hit"

    def word(count: int) -> str:
        return NUMBER_WORDS[count] if count < len(NUMBER_WORDS) else str(count)

    hits = hits_of(a)
    return {
        "anneal_hits": hits, "runs": len(a), "crest_hits": hits_of(c), "crest_hits_word": word(hits_of(c)),
        "anneal_share": f"all {word(len(a))}" if hits == len(a) else f"{word(hits)} of {word(len(a))}",
        "anneal_calls": span(a), "crest_calls": span(c), "pairs": len(a) * len(c),
        "below": sum(float(x["best"]) < float(y["best"]) for x in a for y in c),
    }


def gfn2_revision_tables(audit: Audit, path: str, header: str, after: int, rows: list[dict[str, str]], crest) -> None:
    """The GFN2 tables after the paper's Si10 one.

    Anneal's rows are those of rows.csv at the revision its code cell names,
    CREST's every CREST row of the system, as in the first table.
    """
    def anneal(system: str, revision: str) -> list[dict[str, str]]:
        return [r for r in pick(rows, system=system, code="anneal-general") if r["code_revision"].startswith(revision)]

    lines = audit.text(path).splitlines()
    tabled = set()
    while any(line.startswith("| " + header) for line in lines[after:]):
        table = md_table(audit, path, header, after=after)
        groups = {}
        for _, row in table:
            system = GFN2_SYSTEMS[row["system"]]
            if row["code"] == "CREST":
                groups[(row["system"], row["code"])] = crest(system)
            else:
                revision = re.fullmatch(f"anneal G, `({revision_reach.commit_id_pattern(7)})`", row["code"]).group(1)
                groups[(row["system"], row["code"])] = anneal(system, revision)
                tabled.add((row["system"], revision))
        gfn2_table(audit, path, table, groups, "pooled")
        after = table[-1][0] + 1
    for system, revision, fisher_p, mw_p, sample, sentence in GFN2_REVISION_TESTS:
        a, c = anneal(GFN2_SYSTEMS[system], revision), crest(GFN2_SYSTEMS[system])
        where = f"{path} {system} {revision}"
        # A table whose header no longer matches is not read at all, so its rows are not checked.
        audit.record((system, revision) in tabled, where, "GFN2 table row", f"anneal G, `{revision}`", "checked")
        audit.cell(where, "GFN2 Fisher p", fisher_p, fisher(hits_of(a), len(a), hits_of(c), len(c)))
        if sample == "best":
            p = mann_whitney([float(r["best"]) for r in a], [float(r["best"]) for r in c], continuity=True)
        else:
            p = mann_whitney(gfn2_ranked(a), gfn2_ranked(c), continuity=True)
        audit.cell(where, "GFN2 Mann-Whitney p", mw_p, p)
        audit.said(path, sentence.format(fisher_p=fisher_p, mw_p=mw_p, **gfn2_sentence_fields(a, c)))
    below = sum(float(x["best"]) < float(y["best"]) for x in anneal("c20", "5351f272") for y in crest("c20"))
    audit.cell(f"{path} C20 5351f272", "anneal below CREST seed pairs", "134", below)


COVALENT_SYSTEMS = {"Si10": "sw10", "Si13": "sw13", "Si20": "sw20", "C10": "tc10", "C20": "tc20"}
COVALENT_CODES = {
    "anneal R": "anneal", "anneal P": "anneal-nopriors", "anneal G": "anneal-general",
    "anneal P v1": "anneal-nopriors@v1", "anneal G v1": "anneal-general@v1",
    "anneal P v2": "anneal-nopriors@v2", "anneal G v2": "anneal-general@v2",
    "anneal R 5e-3": "anneal-sloppy", "anneal P 5e-3": "anneal-nopriors-sloppy", "anneal G 5e-3": "anneal-general-sloppy",
    "GMIN": "gmin", "GMIN 5e-3": "gmin", "GMIN 1e-6": "gmin-tight", "ASE": "ase", "ASE BH": "ase",
}


def covalent_panels(audit: Audit) -> None:
    """The panel tables of bench/covalent-h2h; GMIN and ASE of the After table ran in the before session."""
    path = "bench/covalent-h2h/README.md"
    panels = (("## Before", "before", None), ("## After", "after", "before"),
              ("## Second panel", "v2", None), ("## Third panel", "v3", None))
    lines = audit.text(path).splitlines()
    for heading, folder, fallback in panels:
        start = lines.index(heading) + 1
        seeds = audit.rows(f"bench/covalent-h2h/{folder}/seeds.csv")
        spare = audit.rows(f"bench/covalent-h2h/{fallback}/seeds.csv") if fallback else []

        def group(system: str, label: str) -> list[dict[str, str]]:
            code = COVALENT_CODES[label]
            found = pick(seeds, system=COVALENT_SYSTEMS[system], code=code)
            return found or pick(spare, system=COVALENT_SYSTEMS[system], code=code)

        for line, row in md_table(audit, path, "system | code | hits | median calls", start):
            g = group(row["system"], row["code"])
            where = f"{path}:{line} {folder} {row['system']} {row['code']}"
            calls = [float(r["evals_to_hit"]) for r in g]
            audit.cell(where, "hits", row["hits"], frac(g))
            audit.cell(where, "median calls", row["median calls"], median(calls))
            if "IQR" in row:
                audit.cell(where, "IQR", row["IQR"], iqr(calls))
            audit.cell(where, "best", row["best"], min(float(r["best"]) for r in g))
            audit.cell(where, "pot_s", row["pot_s"], median([float(r["pot_s"]) for r in g]))
            audit.cell(where, "outside_s", row["outside_s"], median([float(r["outside_s"]) for r in g]))
        if folder not in ("v2", "v3"):
            continue
        for line, row in md_table(audit, path, "system | comparison", start):
            left, right = row["comparison"].split(" vs ")
            a, b = group(row["system"], left), group(row["system"], right)
            where = f"{path}:{line} {folder} {row['system']} {row['comparison']}"
            shown_a, shown_b = row["median calls"].split(" vs ")
            audit.cell(where, "median calls left", shown_a, median([float(r["evals_to_hit"]) for r in a]))
            audit.cell(where, "median calls right", shown_b, median([float(r["evals_to_hit"]) for r in b]))
            audit.cell(where, "hits", row["hits"], f"{frac(a)} vs {frac(b)}")
            p = mann_whitney([float(r["evals_to_hit"]) for r in a], [float(r["evals_to_hit"]) for r in b],
                             continuity=True, degenerate=None)
            audit.cell(where, "Mann-Whitney p", row["Mann-Whitney p"].replace("tied", ""), p)
            audit.cell(where, "Fisher p", row["Fisher p"], fisher(hits_of(a), len(a), hits_of(b), len(b)))


def release(audit: Audit) -> None:
    version = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)
    audit.said(TEX, f"The described submitted revision postdates the newest released tag,\n"
                    f"v{version}, and is available at\n"
                    "\\url{https://github.com/HaoZeke/anneal/commit/"
                    f"{SUBMITTED}}};\n"
                    "the Zenodo archive does not contain that revision.")
    audit.record(git("cat-file", "-e", f"{SUBMITTED}^{{commit}}").returncode == 0,
                 TEX, "the submitted revision in the archive statement exists", "present", "present")
    tagged = release_tag_available(f"v{version}")
    audit.record(tagged, TEX, "release tag exists", f"v{version}", "tagged" if tagged else "no tag")


def lj_large(audit: Audit) -> None:
    audit.rows("bench/ase-gofee-h2h/lj-large.csv")
    seeds = "bench/ase-gofee-h2h/lj-large-seeds.csv"
    if not (ROOT / seeds).is_file():
        return
    audit.rows(seeds)


def ase_gofee(audit: Audit) -> None:
    rows = audit.rows("bench/ase-gofee-h2h/results.csv")
    row = pick(rows, system="lj75", arm="agox_gofee", seed="2000")[0]
    audit.cell("bench/ase-gofee-h2h/README.md", "GOFEE LJ75 seed 2000 stop", "budget", row["stop"])
    audit.said("bench/ase-gofee-h2h/README.md",
               f"it spent the full {row['charged']} calls in {float(row['wall_s']):.0f} s, best energy {float(row['best']):.6f}")


def tip4p_panel(audit: Audit) -> None:
    """The TIP4P panel is derived directly from its committed per-seed rows."""
    path = "bench/tip4p-h2h/README-h2h.md"
    rows = audit.rows("bench/tip4p-h2h/rows-h2h-r2.csv")
    sizes = {"(H₂O)₈": "tip4p8", "(H₂O)₁₅": "tip4p15", "(H₂O)₂₀": "tip4p20", "(H₂O)₂₁": "tip4p21", "(H₂O)₂₅": "tip4p25"}
    for line, row in md_table(audit, path, "Size | anneal | GMIN | calls (a/g)"):
        size = sizes[row["Size"]].removeprefix("tip4p")
        arms = {
            arm: pick(rows, molecules=size, arm=arm)
            for arm in ("anneal", "gmin")
        }
        where = f"{path}:{line} tip4p{size}"
        for arm, label in (("anneal", "anneal"), ("gmin", "GMIN")):
            group = arms[arm]
            audit.cell(where, f"{arm} row count", "12", float(len(group)))
            expected_hits = (
                "unscored" if size == "25"
                else f"{sum(int(item['hit']) for item in group)}/{len(group)}"
            )
            audit.cell(where, f"{arm} hits", row[label], expected_hits)
            best, median_best = row[f"{label} best / median"].split(" / ")
            energies = [float(item["best_energy"]) for item in group]
            audit.cell(where, f"{arm} best", best, min(energies))
            audit.cell(where, f"{arm} median best", median_best, median(energies))
        calls_a, calls_g = row["calls (a/g)"].split(" / ")
        audit.cell(where, "anneal median calls", calls_a,
                   median([float(item["calls"]) for item in arms["anneal"]]))
        audit.cell(where, "GMIN median calls", calls_g,
                   median([float(item["calls"]) for item in arms["gmin"]]))
        audit.cell(where, "p(U), no continuity correction", row["p(U)"],
                   mann_whitney(
                       [float(item["best_energy"]) for item in arms["anneal"]],
                       [float(item["best_energy"]) for item in arms["gmin"]],
                       continuity=False,
                   ))


def pinned_hf_revision(audit: Audit, path: str) -> str:
    """Read the literal dataset revision from a fetcher without importing it."""
    for node in ast.parse(audit.text(path)).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if (
            any(isinstance(target, ast.Name) and target.id == "HF_REVISION_DEFAULT"
                for target in targets)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return value.value
    raise ValueError(f"{path}: no literal HF_REVISION_DEFAULT")


def reproduction_records(audit: Audit) -> None:
    """Every requested panel has one forced command and records its execution context."""
    reproduce = "paper/jctc/REPRODUCE.md"
    reproduce_rules = {"sc_h2h_n12", "tip4p_h2h", "agox_gofee_h2h"}
    campaigns = (
        ("sc_h2h_n12", "scripts/metal_h2h/enter_sc_n12.sh", "sutton-chen-n12", 1, ""),
        ("tip4p_h2h", "scripts/tip4p_h2h/enter_h2h.sh", "tip4p-gmin", 1, ""),
        ("agox_gofee_h2h", "scripts/ase_gofee_h2h/enter_bench.sh", "agox-gofee", 1, ""),
        (
            "water_crest_fair",
            "scripts/water_crest_fair.sh",
            "water-crest-fair-n192",
            8,
            "",
        ),
        ("crest_equal", ("scripts/crest_equal/enter_bench.sh", "scripts/crest_equal/run_panel.py"), "crest-equal-h2o6", 1, ""),
    )
    snake = audit.text("workflow/Snakefile")
    for rule, runner, campaign, cores, suffix in campaigns:
        command = f"snakemake -s workflow/Snakefile {rule} --cores {cores} --force{suffix}"
        if rule in reproduce_rules:
            audit.said(reproduce, command)
        audit.record(command in snake, "workflow/Snakefile", f"{rule} records the exact forced command",
                     command, command if command in snake else "missing")
        runners = (runner,) if isinstance(runner, str) else runner
        text = "\n".join(audit.text(path) for path in runners)
        runner = " and ".join(runners)
        for field in ("sessions.jsonl", "session_metadata.py", campaign) + (("--anneal-repro",) if rule == "water_crest_fair" else ()):
            audit.record(field in text, runner, f"{rule} session metadata names {field}",
                         field, field if field in text else "missing")
    for path in (reproduce, "scripts/water_crest_fair.sh", "workflow/Snakefile", PAPER):
        fixed = "/work/anneal_repro" in audit.text(path)
        audit.record(not fixed, path, "the anneal_repro checkout is a parameter, not a fixed path", "none",
                     "/work/anneal_repro" if fixed else "none")
    audit.said("scripts/water_crest_fair.sh", "REPRO=${ANNEAL_REPRO_ROOT:-$ROOT/../anneal_repro}")
    audit.said("workflow/Snakefile", 'repro=config.get("anneal_repro", ""),')
    text = audit.text(reproduce)
    audit.record("crest_equal" not in text and "water_crest_fair" not in text, reproduce,
                 "withdrawn CREST and hexamer commands are absent from REPRODUCE.md", "absent",
                 "present" if ("crest_equal" in text or "water_crest_fair" in text) else "absent")
    audit.said(reproduce, "Run every command of this file from the repository root.")
    relative = [line.strip() for line in text.splitlines()
                if line.startswith("    python3 scripts/") or line.startswith("    python3 scripts\\")]
    audit.record("Run these from paper/jctc" not in text and not relative, reproduce,
                 "every command runs from the repository root", "none from paper/jctc",
                 " | ".join(relative) or ("paper/jctc" if "Run these from paper/jctc" in text else "none from paper/jctc"))
    for rule in ("crest_equal", "cuh2_fcc_rec"):
        command = f"snakemake -s workflow/Snakefile {rule} --cores 1 --force"
        if rule == "cuh2_fcc_rec":
            audit.said(reproduce, command)
        audit.record(f"rule {rule}:" in snake, "workflow/Snakefile", f"the Snakefile has the rule {rule}", rule,
                     rule if f"rule {rule}:" in snake else "missing")
    output_roots = {
        "SC_H2H_N12_OUTPUT": "/work/metal-sc/output",
        "TIP4P_H2H_OUTPUT": "/work/tip4p-h2h/output",
        "AGOX_GOFEE_OUTPUT": "/work/ase-gofee/output",
        "CUH2_FCC_REC_OUTPUT": "/work/cuh2-fcc-rec",
    }
    for variable, output in output_roots.items():
        audit.said(reproduce, f"{variable}={output}")
    output_flags = (
        "output", "rows-out", "sessions-out", "hosts-out", "check-out",
        "wall-out", "verify-out", "summary", "seeds-out",
    )
    output_pattern = (
        r"--(?:" + "|".join(output_flags) + r")\s+([^\s\\]+)"
    )
    output_paths = re.findall(output_pattern, text)
    tracked = set(git("ls-files").stdout.splitlines())
    unsafe_outputs = [
        path for path in output_paths
        if path in tracked and (
            path.endswith((".csv", ".tsv", ".jsonl"))
            or path.startswith("paper/jctc/logs/")
        )
    ]
    unsafe_outputs.extend(
        path for path in re.findall(r">\s*([^\s\\]+)", text)
        if path in tracked and (
            path.endswith((".csv", ".tsv", ".jsonl"))
            or path.startswith("paper/jctc/logs/")
        )
    )
    audit.record(
        not unsafe_outputs and "--write" not in text,
        reproduce,
        "REPRODUCE commands never write tracked records",
        "fresh outputs only",
        " ".join(unsafe_outputs) or (
            "--write" if "--write" in text else "fresh outputs only"
        ),
    )
    campaign_outputs = {
        "scripts/tip4p_h2h/enter_h2h.sh": (
            "$OUTPUT/sessions.jsonl", "$OUTPUT/rows-h2h.csv",
            "$OUTPUT/h2h-summary.csv",
        ),
        "scripts/metal_h2h/enter_sc_n12.sh": (
            "$OUTPUT/sessions.jsonl", "$OUTPUT/results.csv",
            "$OUTPUT/seeds.csv",
        ),
        "scripts/ase_gofee_h2h/enter_bench.sh": (
            '"\\$OUTPUT/sessions.jsonl"', '--output-dir "\\$OUTPUT"',
        ),
        "scripts/run_cuh2_fcc_rec.sh": (
            "${CUH2_FCC_REC_OUTPUT:-/work/cuh2-fcc-rec}",
        ),
    }
    for path, required in campaign_outputs.items():
        source = audit.text(path)
        missing = [token for token in required if token not in source]
        audit.record(
            not missing,
            path,
            "the reproduction campaign writes records below /work",
            "all /work outputs",
            " ".join(missing) or "all /work outputs",
        )
    recorder = audit.text("scripts/session_metadata.py")
    for field in ('"compilers": compilers()', '"python": {"version": platform.python_version()', '"openmp": openmp_runtime()',
                  '"anneal_repro": repro_checkout(anneal_repro)', '"omp_num_threads": os.environ.get("OMP_NUM_THREADS")'):
        audit.record(field in recorder, "scripts/session_metadata.py", "the session receipt records the build context",
                     field, field if field in recorder else "missing")
    audit.said(reproduce, "the versions\nof rustc, cargo, gfortran and cc (null when one is not installed), the Python\n"
                          "version, and the OpenMP libraries the dynamic loader resolves with\n`OMP_NUM_THREADS`")
    dataset = "71185263eb6a8130d1262fd76d90109f5fabdcf5"
    data_fetcher = "scripts/data/fetch_data.py"
    bench_fetcher = "paper/jctc/scripts/fetch_bench_data.py"
    audit.said(data_fetcher, 'HF_REPO = os.environ.get("HF_REPO", "HaoZeke/anneal-paper-data")')
    audit.said(bench_fetcher, '"HF_DATASET", "HaoZeke/anneal-paper-data"')
    data_revision = pinned_hf_revision(audit, data_fetcher)
    bench_revision = pinned_hf_revision(audit, bench_fetcher)
    audit.record(data_revision == bench_revision, bench_fetcher,
                 "both data fetchers pin the same dataset revision",
                 data_revision, bench_revision)
    audit.cell(data_fetcher, "the pinned paper dataset revision", dataset, data_revision)
    audit.said(reproduce, f"`HaoZeke/anneal-paper-data` at\n`{dataset}`")
    audit.said(reproduce,
               "`python3 scripts/data/fetch_data.py bench/tip4p-h2h/logs/`")
    hf_manifest = list(csv.DictReader(
        (ROOT / "data/hf-files.csv").open(newline="")
    ))
    tip4p_logs = [
        row for row in hf_manifest
        if row["path"].startswith("bench/tip4p-h2h/logs/")
    ]
    audit.record(len(tip4p_logs) == 120, "data/hf-files.csv",
                 "the HF manifest records all TIP4P head-to-head logs",
                 "120", str(len(tip4p_logs)))
    lj_csm = audit.text("paper/jctc/data/lj_csm_sizes/provenance.txt")
    sums = hashlib.sha256((ROOT / "paper/jctc/data/lj_csm_sizes/audit/SHA256SUMS").read_bytes()).hexdigest()
    lj_csm_tokens = lj_csm.replace(".", "").split()
    has_sums = sums in lj_csm_tokens and "audit/SHA256SUMS" in lj_csm_tokens
    audit.record(has_sums, "paper/jctc/data/lj_csm_sizes/provenance.txt",
                 "provenance.txt records the digest of audit/SHA256SUMS", sums,
                 sums if has_sums else "absent")
    nested = "pixi run --locked --frozen snakemake water_crest_fair --cores 8 --force"
    pinned = all(token in audit.text("scripts/water_crest_fair.sh")
                 for token in ("pixi run", "--locked", "--frozen", "water_crest_fair"))
    audit.record(
        pinned,
        "scripts/water_crest_fair.sh",
        "full CREST campaign uses the pinned Pixi workflow",
        nested,
        nested if pinned else "missing",
    )
    audit.said("scripts/tip4p_h2h/enter_h2h.sh", "--rows $OUTPUT/rows-h2h.csv")
    runner = audit.text("scripts/tip4p_h2h/run_h2h.py")
    audit.record("unlink" not in runner, "scripts/tip4p_h2h/run_h2h.py", "the TIP4P-vs-GMIN rows file is append-only",
                 "no unlink", "no unlink" if "unlink" not in runner else "unlink")


def bench_prose(audit: Audit) -> None:
    fp = audit.rows("bench/fp-fingerprint/rows.csv")
    requench = {r["system"]: r["requenches"] for r in fp}
    audit.said("bench/fp-fingerprint/README.md",
               "twelve on Au55 and LJ38, ten on LJ75 and two on TIP4P15")
    audit.cell("bench/fp-fingerprint/README.md", "re-quenches au55 lj38 lj75 tip4p15", "12 12 10 2",
               " ".join(requench[s] for s in ("au55", "lj38", "lj75", "tip4p15")))

    growpal = audit.rows("bench/growpal-h2h/seeds.csv")
    spreads = []
    for system in ("lj38", "lj75", "lj98", "lj104"):
        walls = [float(r["wall_s"]) for r in pick(growpal, system=system, arm="growpal")]
        spreads.append(f"{max(walls) - min(walls):.2f}")
    audit.said("bench/growpal-h2h/README.md",
               f"spreads by {spreads[0]} s at LJ38, {spreads[1]} s at LJ75, {spreads[2]} s at LJ98 and {spreads[3]} s at LJ104.")

    gad = audit.rows("bench/gad-published/summary.csv")
    rerun = pick(gad, system="lj38", arm="gmin", provenance="rerun")[0]
    audit.said("bench/pimc-lj38/README.md",
               f"| gmin | {rerun['hits']}/{rerun['n']} | {float(rerun['evals_median']):g} | {rerun['wall_median_s']} | "
               f"{rerun['inside_us']} | {rerun['outside_us']} | {rerun['gmin_revision'][:8]} |")

    nh3 = audit.rows("bench/nh3-octamer/ladder.csv")
    stops = {r["seed"]: r["calls"] for r in nh3 if r["budget"] == "16000" and r["calls"] != "16000"}
    audit.said("bench/nh3-octamer/README.md", f"seeds 17000 and 17002 stopped at {stops['17000']} and {stops['17002']} calls")

    icosa = audit.rows("bench/metal-cluster-h2h/ptcu-b3d03061/icosa-23.csv")
    pt = [float(r["best"]) for r in pick(icosa, system="pt6cu17")]
    audit.said("bench/metal-cluster-h2h/ptcu-b3d03061/README.md",
               f"reach −{-min(pt):.8f} for Pt6Cu17 on all six seeds")
    audit.cell("bench/metal-cluster-h2h/ptcu-b3d03061/README.md", "Pt6Cu17 seeds at that energy", "6",
               float(sum(abs(x - min(pt)) < 1e-8 for x in pt)))
    polish_path = "bench/metal-cluster-h2h/ptcu-b3d03061/polish-reruns.csv"
    polish = pick(audit.rows(polish_path), system="pt6cu17", host="bench")[0]
    audit.said("bench/metal-cluster-h2h/ptcu-b3d03061/README.md",
               f"gives −{-float(polish['energy_eV']):.8f} eV in `polish-reruns.csv`")
    audit.record(polish["code_revision"] == "be3849e7811634aca67688da70f5413f9d860ec9", polish_path,
                 "Pt6Cu17 polish rerun revision", "be3849e7", polish["code_revision"][:8])
    audit.record(polish["code_tree"] == "93b2b874d245fed5d538386e590c415029243ae5", polish_path,
                 "Pt6Cu17 polish rerun tree", "93b2b874", polish["code_tree"][:8])

    tip = "bench/tip4p-h2h/README.md"
    audit.said(tip, "18.4 µs per GMIN call. The adaptive arm takes 18.3 µs")
    audit.said(tip, "Seed 2004 reaches −623.877 against −623.336 and misses on both.")
    audit.said(tip, "Seeds 2002 and 2004, the two one-contact misses, reach −628.371.")
    audit.cell(tip, "cage wall ratio at 15 molecules", "5.237", 40.21225 / 7.6787855)


def external_gmin_identifier(row: dict[str, str], column: str) -> bool:
    """A legacy field outside the revision-like columns audited as Git ids."""
    return (
        column == "gmin_revision"
        or column == "code_revision" and (
            row.get("code", "").lower() == "gmin"
            or row.get("arm", "").lower() == "gmin"
        )
    )


def provenance(audit: Audit) -> dict[str, str]:
    record_ref_status(RECORD_REFS)
    gaps: dict[str, str] = {}
    revisions: dict[str, set[str]] = defaultdict(set)
    for path, rows in audit.read.items():
        columns = set(rows[0]) if rows else set()
        revs = [c for c in REVISION_COLUMNS if c in columns]
        hosts = [c for c in HOST_COLUMNS if c in columns]
        machines = MACHINE_PROVENANCE_COLUMNS & columns
        problems = []
        if not revs and not hosts:
            problems.append("revision and host unrecorded (no revision or host column)")
        elif not revs:
            problems.append("revision unrecorded (no revision column)")
        elif not hosts:
            problems.append("revision recorded; no host column")
        for column in revs + hosts:
            if any(r[column].strip() in ("", "unknown") for r in rows):
                what = "unrecorded" if column in revs else "empty or unknown"
                problems.append(f"{column} {what} on some rows")
        if machines:
            if machines != MACHINE_PROVENANCE_COLUMNS:
                missing = sorted(MACHINE_PROVENANCE_COLUMNS - machines)
                lead = "revision recorded; " if revs and not problems else ""
                problems.append(
                    f"{lead}{' and '.join(sorted(machines))} recorded; "
                    f"no {' or '.join(missing)} column"
                )
            elif any(not r["cpu_model"].strip() for r in rows):
                problems.append("cpu_model empty on some rows")
            if "machine_id" in columns and any(
                not all(
                    re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in r["machine_id"].strip().split(";")
                )
                for r in rows
            ):
                problems.append("machine_id is not a salted sha256 on some rows")
        trees = set()
        for column in revs:
            for r in rows:
                if external_gmin_identifier(r, column):
                    continue
                for value in r[column].strip().split(";"):
                    if value in RECORD_TREES:
                        trees.add(value)
                    elif re.fullmatch(revision_reach.commit_id_pattern(7), value) and value not in EXTERNAL_REVISIONS:
                        revisions[value].add(path)
        for value in sorted(trees):
            held = git("rev-parse", f"{RECORD_TREES[value]}^^{{tree}}").stdout.strip()
            audit.record(held == value, path, f"tree {value[:8]} is the parent tree of record {RECORD_TREES[value][:8]}",
                         value[:8], held[:8] or "absent")
        if problems:
            gaps[path] = "; ".join(problems)
    names = sorted(revisions)
    table = alias_table()
    index = alias_index(names, table)
    aliases = verified_aliases(audit, table, index)
    unpushed = set()
    for name in (n for n in names if not index.audited_exists(n)):
        if any(local.startswith(name) for local in aliases):
            continue
        unpushed.add(name)
        if name in UNPUSHED:
            continue
        for path in revisions[name]:
            note = f"revision {name[:8]} is not in this repository"
            gaps[path] = f"{gaps[path]}; {note}" if path in gaps else note
    for name in sorted(unpushed | (set(UNPUSHED) & set(names))):
        audit.record(name in unpushed and name in UNPUSHED, ", ".join(sorted(revisions[name])), f"unpushed revision {name[:8]}",
                     UNPUSHED.get(name, "not listed"), "no pushed equivalent" if name in unpushed else "resolved")
    return gaps


def git(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, input=stdin, capture_output=True, text=True, check=False)


def commit_exists(rev: str) -> bool:
    """The object store holds a commit by this name, whether or not a ref reaches it."""
    out = git("cat-file", "--batch-check", stdin=f"{rev}^{{commit}}\n").stdout.split()
    return out[1:2] == ["commit"]


def git_bytes(*args: str, stdin: bytes = b"", env: dict[str, str] | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, input=stdin, capture_output=True, check=False, env=env).stdout


class RevisionIndex:
    """The revision lookups of one audit, from a fixed number of git calls.

    ``cites`` holds, for each text, the commits that ``git log -S<text>`` over
    the tips would list: those with a file pair, as ``git log --raw`` gives
    them, whose two sides hold the text a different number of times. One
    ``git grep -F`` over every blob of those pairs finds the blobs that hold
    any text at all; only those are read and counted. When a configured diff
    textconv driver applies to a path in that history, ``cites`` comes from
    ``git log -S`` itself, once per text, and ``textconv`` names the paths.
    """

    def __init__(self, names: tuple[str, ...], texts: tuple[str, ...], tips: tuple[str, ...], patched: tuple[str, ...]) -> None:
        asked = sorted({n for n in names + tips + patched if n})
        out = git("cat-file", "--batch-check", stdin="".join(f"{n}^{{commit}}\n" for n in asked)).stdout.splitlines()
        remote_refs = [
            f"refs/remotes/origin/{branch}"
            for branch in (*AUDIT_REMOTE_BRANCHES, *AUDIT_OPTIONAL_REMOTES)
        ]
        audited_refs = ["HEAD", *(ref for ref in remote_refs
                                   if git("show-ref", "--verify", "--quiet", ref).returncode == 0),
                        *git("for-each-ref", "--format=%(refname)", "refs/tags").stdout.split()]
        # A verified record ref is a branch origin holds at the registered SHA and tree.
        records = list(_verified_record_refs())
        audited_heads = git("rev-parse", *audited_refs).stdout.split() + records
        self.commit = {n: line.split()[0] for n, line in zip(asked, out)
                       if line.split()[1:2] == ["commit"]}
        self.parents: dict[str, list[str]] = {}
        roots = sorted({*audited_refs, *records, *self.commit.values()})
        for line in git("rev-list", "--parents", *roots).stdout.splitlines():
            commit, *parents = line.split()
            self.parents[commit] = parents
        self.audited = self.closure(audited_heads)
        remote = git("for-each-ref", "--format=%(objectname)", *remote_refs).stdout.split() + records
        self.remote = self.closure(r for r in remote if r in self.parents)
        self.patch_ids: dict[str, str] = {}
        shown = sorted({self.commit[p] for p in patched if p in self.commit})
        if shown:
            for line in git("patch-id", "--stable", stdin=git("show", *shown).stdout).stdout.splitlines():
                pid, commit = line.split()
                self.patch_ids[commit] = pid
        self.cites: dict[str, set[str]] = {t: set() for t in texts}
        self.textconv: dict[str, str] = {}
        heads = sorted({self.commit[t] for t in tips if t in self.commit})
        if self.cites and heads:
            self.pickaxe(heads)

    def closure(self, starts) -> set[str]:
        seen: set[str] = set()
        stack = list(starts)
        while stack:
            commit = stack.pop()
            if commit not in seen:
                seen.add(commit)
                stack.extend(self.parents.get(commit, ()))
        return seen

    def textconv_paths(self, heads: list[str]) -> dict[str, str]:
        """Paths in the history of ``heads`` whose diff attribute names a driver with a textconv command.

        git log -S counts a text in a path's textconv output, not in its blob,
        so the blob counts of ``pickaxe`` do not hold for these paths.
        """
        config = git("config", "--null", "--get-regexp", r"^diff\..*\.textconv$").stdout
        drivers = {entry.split("\n", 1)[0][len("diff."):-len(".textconv")] for entry in config.split("\0") if entry}
        if not drivers:
            return {}
        listed = git("log", "--format=", "--name-only", "--no-renames", "-z", *heads).stdout
        paths = sorted({p for p in listed.split("\0") if p})
        out = git("check-attr", "-z", "--stdin", "diff", stdin="".join(f"{p}\0" for p in paths)).stdout.split("\0")
        return {path: value for path, value in zip(out[0::3], out[2::3]) if value in drivers}

    def pickaxe(self, heads: list[str]) -> None:
        self.textconv = self.textconv_paths(heads)
        if self.textconv:
            drivers = ", ".join(sorted(set(self.textconv.values())))
            print(f"paper_audit: diff textconv driver {drivers} applies to {len(self.textconv)} paths; "
                  "running git log -S once per revision", file=sys.stderr)
            for text in self.cites:
                self.cites[text] = set(git("log", "--format=%H", f"-S{text}", *heads).stdout.split())
            return
        pairs: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        commit = ""
        for line in git("log", "--raw", "--no-abbrev", "--format=commit %H", *heads).stdout.splitlines():
            if line.startswith("commit "):
                commit = line[len("commit "):]
            elif line.startswith(":"):
                old_mode, new_mode, old, new = line[1:].split("\t", 1)[0].split()[:4]
                pairs[commit].append((old_mode, old, new_mode, new))
        absent = "0" * 40
        blobs = sorted({b for ps in pairs.values() for m1, b1, m2, b2 in ps for m, b in ((m1, b1), (m2, b2))
                        if b != absent and m != "160000"})
        # A blobless clone leaves these blobs with the promisor remote. An
        # alternate object directory cannot fetch them, so bring them into
        # this repository before the scratch tree is built.
        if blobs:
            git("cat-file", "--batch-check", stdin="".join(f"{b}\n" for b in blobs))
        objects = git("rev-parse", "--path-format=absolute", "--git-path", "objects").stdout.strip()
        with tempfile.TemporaryDirectory() as scratch:
            env = dict(os.environ, GIT_OBJECT_DIRECTORY=scratch, GIT_ALTERNATE_OBJECT_DIRECTORIES=objects)
            listing = "".join(f"100644 blob {b}\t{b}\n" for b in blobs).encode()
            tree = git_bytes("mktree", stdin=listing, env=env).decode().strip()
            patterns = [arg for t in self.cites for arg in ("-e", t)]
            found = git_bytes("grep", "-a", "-l", "-F", *patterns, tree, env=env).decode().split()
        held = sorted(line.split(":", 1)[1] for line in found)
        counts: dict[str, dict[str, int]] = {}
        stream = git_bytes("cat-file", "--batch", stdin="".join(f"{b}\n" for b in held).encode())
        at = 0
        for blob in held:
            end = stream.index(b"\n", at)
            size = int(stream[at:end].split()[2])
            content = stream[end + 1:end + 1 + size]
            at = end + 2 + size
            counts[blob] = {t: content.count(t.encode()) for t in self.cites}
        none = dict.fromkeys(self.cites, 0)

        def held_in(mode: str, blob: str) -> dict[str, int]:
            if blob == absent:
                return none
            if mode == "160000":
                content = f"Subproject commit {blob}\n".encode()
                return {t: content.count(t.encode()) for t in self.cites}
            return counts.get(blob, none)

        for commit, ps in pairs.items():
            for old_mode, old, new_mode, new in ps:
                before, after = held_in(old_mode, old), held_in(new_mode, new)
                for text in self.cites:
                    if before[text] != after[text]:
                        self.cites[text].add(commit)

    def resolve(self, revision: str) -> str | None:
        return self.commit.get(revision) or (revision if revision in self.parents else None)

    def exists(self, revision: str) -> bool:
        return revision in self.commit

    def audited_exists(self, revision: str) -> bool:
        """The revision exists in the histories the paper audit accepts."""
        return self.resolve(revision) in self.audited

    def on_remote(self, revision: str) -> bool:
        """The commit is reachable from a ref fetched from the remote."""
        return self.resolve(revision) in self.remote

    def patch_id(self, revision: str) -> str | None:
        return self.patch_ids.get(self.resolve(revision) or "")

    def parents_of(self, revision: str) -> list[str]:
        return self.parents.get(self.resolve(revision) or "", [])

    def is_ancestor(self, ancestor: str, revision: str) -> bool:
        commit, of = self.resolve(ancestor), self.resolve(revision)
        return bool(commit and of) and commit in self.closure([of])

    def first_cites(self, text: str, revision: str) -> bool:
        """``revision`` is the oldest commit in its history whose diff changes how often ``text`` occurs."""
        citing = self.cites.get(text, set())
        return revision in citing and not citing & self.closure(self.parents.get(revision, ()))


@functools.cache
def revision_index(names: tuple[str, ...], texts: tuple[str, ...], tips: tuple[str, ...], patched: tuple[str, ...]) -> RevisionIndex:
    return RevisionIndex(names, texts, tips, patched)


def alias_table() -> list[dict[str, str]]:
    with (ROOT / ALIASES).open(newline="") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def alias_index(names: list[str], table: list[dict[str, str]]) -> RevisionIndex:
    """The index for the cited revisions ``names`` and every revision ``table`` names."""
    revisions = ("local_revision", "pushed_revision", "branch_revision")
    return revision_index(tuple(names) + tuple(r[c] for r in table for c in revisions),
                          texts=tuple(r["local_revision"] for r in table), tips=tuple(r["rows_commit"] for r in table),
                          patched=tuple(r[c] for r in table for c in ("pushed_revision", "branch_revision")))


def verified_aliases(audit: Audit, table: list[dict[str, str]], index: RevisionIndex) -> dict[str, str]:
    """Local revisions of ALIASES whose pushed equivalent checks out, mapped to it.

    Each alias is checked: the local revision is not in the repository; the
    pushed revision and the branch revision, when there is one, are, are
    reachable from a remote ref, and have the recorded patch-id; the rows
    commit is the first in its history to cite the local revision and has
    one parent. That parent is the pushed revision, or, when the alias names
    the paths in ``differs_only_in``, descends from it and differs from it
    in those paths alone.
    """
    verified = {}
    for row in table:
        local, pushed = row["local_revision"], row["pushed_revision"]
        where = f"{ALIASES} {local[:8]}"
        checks = [("local revision is not in this repository", not index.audited_exists(local))]
        for column in ("pushed_revision", "branch_revision"):
            revision = row[column]
            if not revision and column == "branch_revision":
                continue
            checks.append((f"{column} {revision[:8]} is in this repository", index.audited_exists(revision)))
            checks.append((f"{column} {revision[:8]} is on the remote", index.on_remote(revision)))
            checks.append((f"{column} {revision[:8]} has patch-id {row['patch_id'][:8]}", index.patch_id(revision) == row["patch_id"]))
        rows_commit = row["rows_commit"]
        parents = index.parents_of(rows_commit)
        allowed = row["differs_only_in"].split()
        if allowed:
            parent = parents[0] if len(parents) == 1 else ""
            descends = bool(parent) and index.is_ancestor(pushed, parent)
            changed = git("diff", "--name-only", pushed, parent, "--").stdout.split() if descends else None
            checks.append((f"rows commit {rows_commit[:8]} sits on a descendant of {pushed[:8]}", descends))
            checks.append((f"{pushed[:8]} differs from the rows commit's parent only in {' '.join(allowed)}",
                           changed is not None and set(changed) <= set(allowed)))
        else:
            checks.append((f"rows commit {rows_commit[:8]} sits on {pushed[:8]}", parents == [pushed]))
        checks.append((f"rows commit {rows_commit[:8]} first cites {local[:8]}", index.first_cites(local, rows_commit)))
        for label, ok in checks:
            audit.record(ok, where, label, "holds", "holds" if ok else "fails")
        if all(ok for _, ok in checks):
            verified[local] = pushed
    return verified


def release_tag_available(tag: str) -> bool:
    """The release tag exists locally or is advertised by origin."""
    local = git("show-ref", "--verify", "--quiet", f"refs/tags/{tag}").returncode == 0
    if local:
        return True
    return git("ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{tag}").returncode == 0


def unauditable() -> list[str]:
    """Why this clone cannot answer the revision checks, if it cannot.

    A shallow clone, or one without the remote branches, would read every
    revision as missing or unpushed; the audit stops instead of reporting that.
    """
    problems = []
    if git("rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        problems.append("the clone is shallow; fetch the full history (actions/checkout with fetch-depth: 0, "
                        "or git fetch --unshallow origin)")
    if not git("for-each-ref", "--count=1", "refs/remotes").stdout.strip():
        problems.append("there are no remote-tracking refs; fetch the branches of origin "
                        "(git fetch origin '+refs/heads/*:refs/remotes/origin/*')")
    version = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)
    tag = f"v{version}"
    if not release_tag_available(tag):
        problems.append(f"the release tag {tag} is neither local nor advertised by origin")
    for branch in AUDIT_REMOTE_BRANCHES:
        remote_ref = f"refs/remotes/origin/{branch}"
        if git("show-ref", "--verify", "--quiet", remote_ref).returncode != 0:
            problems.append(f"the remote ref origin/{branch} is not present; fetch {branch} from origin")
    wanted = [(row[c], row["branch"] if c == "branch_revision" else "develop")
              for row in alias_table() for c in ("pushed_revision", "branch_revision") if row[c]]
    out = git("cat-file", "--batch-check", stdin="".join(f"{rev}^{{commit}}\n" for rev, _ in wanted)).stdout.splitlines()
    for (rev, branch), line in zip(wanted, out):
        if line.split()[1:2] != ["commit"]:
            problems.append(f"{ALIASES} names {rev[:8]}, which this clone does not have; fetch {branch} from origin")
    return problems


RERUN_FILES = (TEX, SI, "paper/jctc/cover-letter.tex")
COVER = "paper/jctc/cover-letter.tex"
COVER_TOC = "paper/jctc/toc-graphic.tex"
# Sentences of the cover letter that carry a digit and are not abstract sentences.
COVER_OTHER = ("This submission replaces the preprint arXiv:2302.02811, deposited in 2023 under the title "
               "``Unified Software Design Patterns for Simulated Annealing.",
               "A table of contents graphic at 3.25~in by 1.75~in accompanies the submission.")


def collapsed(text: str) -> str:
    """Whitespace collapsed so a sentence compares by its words."""
    return " ".join(text.split())


def cover_letter(audit: Audit) -> None:
    """A cover-letter sentence that carries a number is an abstract sentence, word for word."""
    letter = collapsed(audit.text(COVER))
    manuscript = collapsed(re.sub(r"~?\\todo\{[^}]*\}", "", audit.text(TEX).split("\\begin{abstract}", 1)[1]
                                  .split("\\end{abstract}", 1)[0]))
    body = letter.split("Dear Editor,")[1].split("\\vspace{1em} \\noindent Sincerely")[0]
    sentences = re.split(r"(?<=\.)(?:''|\\\\)?\s+(?=[A-Z`\\])", body)
    checked = 0
    for sentence in sentences:
        bare = re.sub(r"GFN2|TIP4P|CREST|GMIN|\\url\{[^}]*\}|Email", "", sentence)
        if not re.search(r"\d", bare) or "Telephone" in sentence:
            continue
        checked += 1
        if sentence in COVER_OTHER:
            audit.record(True, COVER, "the cover-letter sentence is the preprint line", sentence[:60], sentence[:60])
            continue
        found = sentence in manuscript
        audit.record(found, COVER, "a cover-letter sentence with a number is a sentence of the abstract",
                     sentence[:70], sentence[:70] if found else "not in the manuscript")
    audit.record(checked == len(COVER_OTHER) + 2, COVER,
                 "the numbered cover-letter claims match the two abstract claims",
                 str(len(COVER_OTHER) + 2), str(checked))
    for word in ("\\rerun", "Zenodo", "DOI", "pending", "being run", "rerun"):
        audit.record(word not in audit.text(COVER), COVER, f"the cover letter carries no process note ({word})",
                     "absent", "absent" if word not in audit.text(COVER) else "present")
    toc = audit.text(COVER_TOC)
    audit.record("paperwidth=3.25in,paperheight=1.75in" in toc, COVER_TOC, "the graphic slot is 3.25 in by 1.75 in",
                 "3.25in 1.75in", "3.25in 1.75in")
    audit.said(COVER, "A table of contents graphic at 3.25~in by 1.75~in accompanies the submission.")
    for word in ("TODO", "\\todo"):
        audit.record(word not in audit.text(COVER), COVER, f"the cover letter has no {word}", "none", "none")
    for word in ("Science Institute", "University of Iceland, Reykjavik, Iceland\\\\"):
        audit.record(word not in audit.text(COVER), COVER, f"the cover letter does not list {word}", "absent",
                     "absent" if word not in audit.text(COVER) else "present")
    audit.said(COVER, "The independently started, wall-capped comparisons with the Atomic\n"
                      "Simulation Environment are reported descriptively because their arms are\n"
                      "censored and do not share starts.")
    audit.said(COVER, "Every printed number is computed from rows committed to the repository.")


REFS_BLOB = "ff495c1f0629c91a7e45ceea02df7cca033b0e4c"
# Wording the review of the first rewrite showed to overclaim; none may come back.
OVERCLAIMS = ("same trap", "confined to one funnel", "cannot become the stored best", "cannot win",
              "basin-hopping setting", "one setting", "adds nothing", "does no better", "this setting",
              "crosses funnels more often", "the setting of", "that setting", "does not come from choosing among the moves", "rebuilds a facet",
              "completes the patch", "never restarts", "is equivalent to", "across funnels")


def stack_route_clears_temperature(source: str) -> bool:
    """Whether the literal engine route avoids the literal temperature rectangle."""
    box = re.search(r"TEMPERATURE_BOX = \(([^)]+)\)", source)
    route = re.search(r"ENGINE_ROUTE = \(\n(.*?)\n\)", source, re.DOTALL)
    if box is None or route is None:
        return False
    tx, ty, tw, th = (float(value.strip()) for value in box.group(1).split(","))
    points = [tuple(map(float, pair)) for pair in re.findall(r"\(([\d.]+), ([\d.]+)\)", route.group(1))]
    return len(points) >= 2 and all(
        max(a[0], b[0]) <= tx or min(a[0], b[0]) >= tx + tw
        or max(a[1], b[1]) <= ty or min(a[1], b[1]) >= ty + th
        for a, b in zip(points, points[1:])
    )


TRAIL_ROW_REVISIONS = (
    "eeec8e2c216ae1506768bca8e29846c1c039ca3b",
    "61fa67cf380e46876b569da45e7cc091647a775b",
    "5eb79cdcb0a986984dc7b487e5a3cc93224246f9",
    "0cc3814c053c2f5a709f0f375ebae07e5373c171",
    "1c1b12384683485cf820d3a244151b48973dc0f7",
    "628773d6837ea2d959a27256ff4163b21162fd4d",
    "b12293b9d637dd32417be3aa482302e0f30dd594",
    "c2de652dca0bc75e2b0cb4afffac91fd656404e4",
)


def historical_trail_presets(audit: Audit) -> None:
    """Every revision behind a printed atomic row enabled the trail response."""
    path = "src/methods/cluster_hopping/config.rs"
    for revision in TRAIL_ROW_REVISIONS:
        source = git("show", f"{revision}:{path}")
        enabled = source.returncode == 0 and "cfg.trail_on_stall = true;" in source.stdout
        audit.record(enabled, f"{path} at {revision[:8]}",
                     "printed-row preset enables the trail response", "enabled",
                     "enabled" if enabled else "disabled or unavailable")


def science_claims(audit: Audit) -> None:
    """Overclaiming wording stays out, endpoint prose stays on energies, and every citation resolves."""
    for path in (TEX, SI, COVER):
        flat = collapse(audit.text(path)).lower()
        for phrase in ("one host", "one machine", "single machine", "same host", "same machine"):
            audit.record(phrase not in flat, path, f"no single-host claim: {phrase}", "absent",
                         "absent" if phrase not in flat else "present")
        for phrase in OVERCLAIMS:
            audit.record(phrase not in flat, path, f"no overclaim: {phrase}", "absent",
                         "absent" if phrase not in flat else "present")
    text = audit.text(TEX)
    audit.record("\\label{sec:endpoints}" not in text, TEX, "the endpoint-energy section is not printed", "absent",
                 "absent" if "\\label{sec:endpoints}" not in text else "present")
    results = text.split("\\label{sec:results}", 1)[1].split("\\begin{acknowledgement}", 1)[0]
    visible = re.sub(r"\\(?:label|ref)\{[^}]*\}", "", re.sub(r"(?m)^%.*$", "", results)).lower()
    for word in ("funnel", "trap"):
        audit.record(word in visible, TEX, f"the results and conclusions state the {word} claim", "present",
                     "present" if word in visible else "absent")
    audit.record("hypothes" not in text.lower(), TEX, "the manuscript states no hypothesis", "absent",
                 "absent" if "hypothes" not in text.lower() else "present")
    for path in (TEX, SI, COVER):
        for word in ("candidate", "unrecorded", "missing", "pending", "TBD", "TODO", "placeholder"):
            found = re.search(rf"\b{word}", audit.text(path), re.IGNORECASE)
            audit.record(found is None, path, f"the text does not say {word}", "absent",
                         "absent" if found is None else "present")
    audit.said(TEX, "Three parts of the search have a run with and without them, and Section~\\ref{sec:knockouts} "
                    "reports all three: the structure templates (Section~\\ref{sec:priors}), at two revisions in the "
                    "history of the submitted revision, and the length unit and the choice among the moves, on the "
                    "search source of the submitted revision.")
    audit.said(TEX, "\\subsection{Mechanism and design rationale}")
    audit.said(TEX, "Together these mechanisms address the two funnel difficulties of\n"
                    "Section~\\ref{sec:intro}: proposing a crossing and not spending the remaining\n"
                    "budget in the broad funnel already found.")
    audit.said(TEX, "This paragraph is a design rationale, not a causal estimate; the evidence\n"
                    "below isolates only the components with paired on/off records.")
    config = audit.text("src/methods/cluster_hopping/config.rs")
    audit.record("cfg.trail_on_stall = true;" in config, TEX,
                 "Config::recommended enables the trail response", "enabled",
                 "enabled" if "cfg.trail_on_stall = true;" in config else "disabled")
    historical_trail_presets(audit)
    audit.said(TEX, "The preset enables two responses to a stuck chain.")
    audit.said(TEX, "When no basin entry is recorded, a trail draw leaves the chain where it is;\n"
                    "the climb is a separate choice of the same draw, not a fallback.")
    audit.said(SI, "Those presets enable both the trail and climb responses to a stuck chain.")
    audit.said(TEX, "For every other proposal, the preset uses the Bayesian screen of\n"
                    "equation~\\ref{eq:bayes-screen}.")
    audit.said(TEX, "Otherwise it finishes when the Student-\\(t\\) posterior assigns probability\n"
                    "at least \\(7/15\\) to beating the best.")
    audit.said(TEX, "The non-default calibrated-margin path, which the recommended preset\n"
                    "bypasses because it enables the Bayesian screen")
    audit.said(TEX, "The arm named \\texttt{soap} in the implementation is a fivefold-residual\n"
                    "move.")
    audit.said(TEX, "Their root-mean-square displacement is \\(0.75\\) in reduced units.")
    audit.said(TEX, "If the drawn-axis proposal does not move, the arm tries the measured\n"
                    "best-axis residual")
    audit.said(TEX, "The branch is withheld only for a calculation with an active region or\n"
                    "frozen atoms.")
    manuscript = collapse(audit.text(TEX)).lower()
    for stale in ("away from the nearest packing stored so far",
                  "a structure already dominated by an icosahedral signature",
                  "a partial energy more than \\(2\\) above the stored best"):
        audit.record(stale not in manuscript, TEX, f"stale mechanism claim is absent: {stale}",
                     "absent", "absent" if stale not in manuscript else "present")
    stack = audit.text("paper/jctc/scripts/fig_algorithm_stack.py")
    arms = audit.text("paper/jctc/scripts/fig_rec_arms.py")
    for word in ("chain 40", "40 chains", "Good-Turing census",
                 "annealed-flow bank", "tbl:systems"):
        audit.record(word not in stack, "paper/jctc/scripts/fig_algorithm_stack.py",
                     f"the reported-run stack figure does not draw {word}", "absent",
                     "absent" if word not in stack else "present")
    def figure_tuple(name: str) -> tuple[str, ...]:
        match = re.search(rf"^{name}\s*=\s*\(([^)]*)\)", stack, re.M | re.S)
        return tuple(re.findall(r'"([^"]+)"', match.group(1))) if match else ()

    kernel_source = audit.text("src/potentials.rs")
    kernel_block = kernel_source.split("pub struct ClusterKernels {", 1)[1].split("\n}", 1)[0]
    kernel_fields = re.findall(r"pub\s+([a-z_]+)\s*:", kernel_block)
    kernel_labels = {
        "cluster_lj": "Lennard-Jones",
        "cambridge_morse": "Morse",
        "gupta": "Gupta",
        "binary_lj": "binary LJ",
    }
    dispatched = tuple(kernel_labels.get(field, f"unknown:{field}") for field in kernel_fields)
    drawn_rgpot = figure_tuple("RGPOT_POTENTIALS")
    audit.record(drawn_rgpot == dispatched, "paper/jctc/scripts/fig_algorithm_stack.py",
                 "the rgpot figure row is derived from ClusterKernels",
                 ", ".join(dispatched), ", ".join(drawn_rgpot))
    drawn_native = figure_tuple("NATIVE_POTENTIALS")
    native = ("TIP4P", "Sutton-Chen")
    audit.record(drawn_native == native, "paper/jctc/scripts/fig_algorithm_stack.py",
                 "the native figure row lists the in-tree search potentials",
                 ", ".join(native), ", ".join(drawn_native))
    audit.said("src/potentials/tip4p.rs", "pub struct Tip4pCluster")
    audit.said("src/potentials/sutton_chen.rs", "pub struct SuttonChen")
    audit.record("GFN2-xTB" not in stack, "paper/jctc/scripts/fig_algorithm_stack.py",
                 "the search stack excludes the offline GFN2 reference relaxation",
                 "absent", "absent" if "GFN2-xTB" not in stack else "present")
    audit.said(TEX, "The lower band separates the rgpot Lennard-Jones, Morse, Gupta, and binary\n"
                    "Lennard-Jones kernels from the native TIP4P and Sutton--Chen implementations;\n"
                    "offline reference relaxations are not part of the search stack.")
    audit.said(TEX, "the lowest of the archived GFN2-xTB water hexamers after relaxation\n"
                    "with xtb 6.7.1")
    for unused in ("EMT", "EAM", "Stillinger-Weber", "Tersoff",
                   "GFN-FF", "PBE/6-31G*", "PET-MAD"):
        audit.record(f'"{unused}"' not in stack, "paper/jctc/scripts/fig_algorithm_stack.py",
                     f"the stack figure excludes unused potential {unused}",
                     "absent", "absent" if f'"{unused}"' not in stack else "present")
    route_clear = stack_route_clears_temperature(stack)
    audit.record(route_clear, "paper/jctc/scripts/fig_algorithm_stack.py",
                 "the energies-and-forces route clears the temperature box", "clear",
                 "clear" if route_clear else "crossing")
    toc = audit.text("paper/jctc/scripts/toc_graphic.py")
    audit.record("bank" not in toc, "paper/jctc/scripts/toc_graphic.py",
                 "the TOC graphic draws no unused bank", "absent",
                 "absent" if "bank" not in toc else "present")
    arm_tuple = 'ARMS = ("single", "surface", "sym", "burst", "soap", "dls-kernel")'
    audit.record(arm_tuple in arms and '"lattice"' not in arms and '"resample"' not in arms,
                 "paper/jctc/scripts/fig_rec_arms.py",
                 "the arm figure uses the six ClusterMove names",
                 "single, surface, sym, burst, soap, dls-kernel",
                 "exact" if arm_tuple in arms and '"lattice"' not in arms and '"resample"' not in arms else "other")
    table = text.split("% gmin-panel table: start", 1)[1].split("% gmin-panel table: end", 1)[0]
    audit.record("LJ\\(_{75}\\)" in table, TEX,
                 "LJ75 is present in the GMIN comparison table", "present",
                 "present" if "LJ\\(_{75}\\)" in table else "absent")
    kept = {
        TEX: (
            "The global minima at 75 and 98 atoms",
            "LJ\\(_{75}\\) & \\(4\\times 10^{6}\\) & 12 & 5/12 & 6/12",
            "LJ\\(_{75}\\) & \\(4\\times 10^{6}\\) & this work & 5/12",
            "Published global minima of LJ\\(_{38}\\), LJ\\(_{75}\\),",
        ),
        SI: (
            "The references, in reduced units, are \\(-173.928427\\) for LJ\\(_{38}\\),",
            "The trail step has one record, on LJ\\(_{75}\\)",
            "They hit LJ\\(_{75}\\) on 5 of 12 against 12",
            "The published GOFEE curves count local optimizations on LJ\\(_{75}\\)",
            "All 60 seeds are rows of this code printed in the GMIN table.",
        ),
    }
    for path, claims in kept.items():
        for claim in claims:
            audit.said(path, claim)
    visuals = audit.text("paper/jctc/scripts/system_visuals.py")
    si_figure = audit.text("paper/jctc/scripts/fig_si_structures.py")
    audit.record('"lj75_marks"' in visuals and 's.key != "lj75_marks"' not in si_figure,
                 "paper/jctc/scripts/system_visuals.py",
                 "the manuscript and SI structure figures include LJ75", "present",
                 "present" if '"lj75_marks"' in visuals and 's.key != "lj75_marks"' not in si_figure else "absent")
    audit.said(SI, "A fingerprint bias during local optimization is reported on LJ\\(_{38}\\)\n"
                   "and LJ\\(_{75}\\).")
    trail = "666591a0"
    audit.record(git("merge-base", "--is-ancestor", trail, SUBMITTED).returncode != 0, TEX,
                 "the trail record is outside the history of the submitted revision", "outside", "checked")
    audit.record(bool(git("branch", "-r", "--contains", trail).stdout.strip()), TEX, "the trail record is on origin",
                 "present", "checked")
    audit.said(TEX, "Among comparisons that report tests, every test is two-sided and exact, so\n"
                    "no continuity correction enters.")
    audit.said(TEX, "No adjustment for multiple comparisons is made: every reported \\(p\\) is\n"
                    "that of one cluster and one pair of arms, every cluster of each printed\n"
                    "comparison is reported,\n"
                    "and the Supporting Information lists each record that is not printed and why,\n"
                    "and a difference in hits is called significant only when its own exact test\n"
                    "is below 0.05.")
    audit.record("The authors reviewed the text and take responsibility for it." not in text,
                 TEX, "the acknowledgments contain no AI-declaration tail", "absent",
                 "absent" if "The authors reviewed the text and take responsibility for it." not in text else "present")
    audit.said(SI, "No equivalence margin was set before any run, so no table supports a claim that two arms are "
                   "equivalent.")

    # Citations: the frozen bibliography, the statistics file, and every key.
    blob = git("hash-object", "paper/jctc/refs.bib").stdout.strip()
    audit.cell("paper/jctc/refs.bib", "blob", REFS_BLOB, blob)
    keys = {}
    for bib in ("paper/jctc/refs.bib", "paper/jctc/refs-stats.bib"):
        keys[bib] = set(re.findall(r"^@\w+\{([^,\s]+),", audit.text(bib), re.M))
    both = keys["paper/jctc/refs.bib"] & keys["paper/jctc/refs-stats.bib"]
    audit.cell("paper/jctc/refs-stats.bib", "keys also in refs.bib", "none", " ".join(sorted(both)) or "none")
    known = set.union(*keys.values())
    for path in (TEX, SI):
        audit.said(path, "\\bibliography{refs,refs-stats}")
        cited = {k.strip() for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", audit.text(path)) for k in group.split(",")}
        missing = sorted(cited - known)
        audit.cell(path, "cited keys without an entry", "none", " ".join(missing) or "none")
    unused = sorted(keys["paper/jctc/refs-stats.bib"] - {
        k.strip() for path in (TEX, SI) for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", audit.text(path))
        for k in group.split(",")})
    audit.cell("paper/jctc/refs-stats.bib", "entries no text cites", "none", " ".join(unused) or "none")
    audit.said(TEX, "Because the starts are unpaired and cap-censored outcomes occur, no\n"
                    "cross-arm significance test is reported for the ASE tables.")
    audit.said(TEX, "\\cite{newcombeImprovedConfidenceIntervals1998}, built from Wilson intervals "
                    "\\cite{wilsonProbableInferenceLaw1927}.")
    refs = audit.text("paper/jctc/refs.bib")
    audit.record("title = {The Reference Set Is Not the Answer Set: {GEOM-DRUGS} Is Missing Low-Energy Conformers "
                 "on Its Own Energy Surface}" in refs, "paper/jctc/refs.bib",
                 "the GEOM-DRUGS entry keeps the cited title", "exact title", "exact title")
    doye = refs.split("@article{Doye_Miller_Wales_1999,", 1)[1].split("\n}", 1)[0]
    audit.record(all(field in doye for field in ("volume = {110}", "number = {14}", "pages = {6896--6906}")),
                 "paper/jctc/refs.bib", "Doye 1999 has volume, issue, and pages",
                 "110 14 6896--6906",
                 "110 14 6896--6906" if all(field in doye for field in
                 ("volume = {110}", "number = {14}", "pages = {6896--6906}")) else "incomplete")
    audit.said(TEX, "The exact McNemar test \\cite{mcnemarNoteSamplingError1947}")
    audit.said(TEX, "the exact Wilcoxon signed-rank test\n\\cite{wilcoxonIndividualComparisons1945}")
    for key, doi in (("mcnemarNoteSamplingError1947", "10.1007/bf02295996"),
                     ("wilcoxonIndividualComparisons1945", "10.2307/3001968"),
                     ("newcombeImprovedConfidenceIntervals1998",
                      "10.1002/(sici)1097-0258(19981130)17:22<2635::aid-sim954>3.0.co;2-c"),
                     ("fisherLogicInductiveInference1935", "10.2307/2342435"),
                     ("hyndmanSampleQuantiles1996", "10.1080/00031305.1996.10473566")):
        bib = "paper/jctc/refs-stats.bib" if key in audit.text("paper/jctc/refs-stats.bib") else "paper/jctc/refs.bib"
        entry = audit.text(bib).split("{" + key + ",", 1)[-1].split("\n}", 1)[0].lower()
        audit.record(doi in entry, bib, f"{key} carries its Crossref DOI", doi,
                     doi if doi in entry else "missing")
    audit.said(TEX, "spirit of the compression of Doye \\cite{doye2000}.")
    for key in ("wilsonProbableInferenceLaw1927", "newcombeImprovedConfidenceIntervals1998",
                "mcnemarNoteSamplingError1947", "wilcoxonIndividualComparisons1945", "hyndmanSampleQuantiles1996",
                "fisherLogicInductiveInference1935"):
        audit.record(key in audit.text(SI), SI, f"the statistical methods cite {key}", "cited",
                     "cited" if key in audit.text(SI) else "uncited")

    for word in ("prior-on", "c2de652d"):
        for path in (TEX, SI, COVER):
            text = outside_template_era(audit.text(path)) if path == SI and word == "c2de652d" else audit.text(path)
            audit.record(word not in text, path, f"no row is labelled {word}", "absent",
                         "absent" if word not in text else "present")
    audit.said(TEX, "\\title{Template-free basin hopping with adaptive moves for funnelled cluster landscapes}")
    audit.said(TEX, "We present a template-free basin-hopping search")


def manuscript_new(audit: Audit) -> None:
    """The abstract split, the added panels, the definitions, the priors, and the host paragraph."""
    gmin_panel_groups(audit)

    # Constants of the definitions and of the priors, read from the source.
    preset = "src/methods/cluster_hopping/preset.rs"
    for fragment in ("TEMPERATURE: f64 = 0.8;", "BIAS_HEIGHT: f64 = 0.25;", "MERGE_RADIUS: f64 = 0.7;",
                     "NEIGHBOUR_CUTOFF: f64 = 1.6;", "SYMMETRISE_CUTOFF: f64 = 2.5;", "SOAP_RMSD: f64 = 0.35;",
                     "SOAP_CUTOFF: f64 = 3.5;", "SCREEN_MARGIN: f64 = 2.0;"):
        audit.said(preset, fragment)
    config = "src/methods/cluster_hopping/config.rs"
    for fragment in ("bias_gamma: 5.0,", "tabu_capacity: 8,", "ContinuousSymmetry::Measure { interval: 10 };",
                     "prior_soap: true,", "calibrated_screen: true,"):
        audit.said(config, fragment)
    for fragment in ("cfg.bayes_screen = true;",
                     "bayes_exploration: 0.1,",
                     "bayes_warmup: 300,"):
        audit.said(config, fragment)
    config_src = audit.text(config)
    for removed in ("reseed_hops: 5,", "pub prior_reseed:", "pub prior_growth:", "pub prior_core:",
                    "pub prior_handover:", "pub prior_motif:"):
        audit.record(removed not in config_src, config, f"inert {removed.rstrip(':,')} is removed",
                     "absent", "absent" if removed not in config_src else "present")
    audit.said("src/methods/cluster_hopping.rs", "const WINDOW: usize = 256;")
    audit.said("src/methods/cluster_hopping.rs", "const WARMUP: usize = 8;")
    audit.said("src/methods/cluster_hopping.rs", "const QUANTILE: f64 = 0.9;")
    hop = audit.text("src/methods/cluster_hopping.rs")
    audit.said("src/methods/cluster_hopping.rs",
               "let feats = Array1::from(vec![1.0, e_screen, drift, e_screen * drift]);")
    audit.said("src/methods/cluster_hopping.rs",
               "e_screen >= ledger.best\n"
               "                && !screen.decide(feats.view(), ledger.best, rng.random::<f64>())")
    audit.said("src/methods/cluster_hopping.rs",
               "let withhold = cfg.active_region.is_some() || cfg.frozen.is_some();")
    screen = "src/screen.rs"
    audit.said(screen, "if self.n < self.warmup {")
    audit.said(screen, "if u < self.exploration {")
    audit.said(screen, "let go = p >= self.threshold;")
    moves = "src/methods/cluster_hopping/moves.rs"
    audit.said(moves, "let drawn = crate::soap::step_away_fivefold(x, *rmsd, rng);")
    audit.said(moves, "crate::soap::step_away_fivefold_measured(x, *rmsd)")
    soap = "src/soap.rs"
    audit.said(soap, "const PENTAGON_CAP: f64 = 0.75;")
    audit.said(soap, "concentrate_residual(&mut dr, 5);")
    audit.said(soap, "scale_support_to_cap(x, dr, rmsd.max(PENTAGON_CAP))")
    audit.record("if frac_421 < 0.20 {" not in hop, "src/methods/cluster_hopping.rs", "motif-seeded growth gate is removed",
                 "absent", "absent" if "if frac_421 < 0.20 {" not in hop else "present")
    audit.record("if frac_422 < 0.15 {" not in hop, "src/methods/cluster_hopping.rs", "motif-seeded growth gate is removed",
                 "absent", "absent" if "if frac_422 < 0.15 {" not in hop else "present")
    audit.record("if frac_555 < 0.20 {" not in hop, "src/methods/cluster_hopping.rs", "motif-seeded growth gate is removed",
                 "absent", "absent" if "if frac_555 < 0.20 {" not in hop else "present")
    audit.record("let motif_gap = ledger.budget() / 2;" not in hop, "src/methods/cluster_hopping.rs",
                 "motif reseed gap is removed", "absent",
                 "absent" if "let motif_gap = ledger.budget() / 2;" not in hop else "present")
    audit.record("if cfg.long_ranged && !ledger.target_reached() && ledger.remaining() > 0 {" not in hop,
                 "src/methods/cluster_hopping.rs", "long-ranged Mackay-cap branch is removed", "absent",
                 "absent" if "if cfg.long_ranged && !ledger.target_reached() && ledger.remaining() > 0 {" not in hop else "present")
    morse_driver = audit.text("examples/h2h_morse.rs")
    audit.record("cfg.long_ranged = kind.second_neighbour_energy() < -0.05;" not in morse_driver,
                 "examples/h2h_morse.rs", "Morse long-ranged Mackay-cap toggle is removed", "absent",
                 "absent" if "cfg.long_ranged = kind.second_neighbour_energy() < -0.05;" not in morse_driver else "present")
    audit.said("src/bias.rs", "let w = scale * (self.w0 + entropy) * libm::exp(-self.v[i] / denom);")
    audit.said("src/bias.rs", "let denom = (self.gamma - 1.0) * temp;")
    audit.said("src/csm.rs", "Lowest-measure average from symmetry operations detected on the incumbent.")
    audit.said("src/methods/lattice_search.rs", "hollow_cutoff: 1.47 * r,")
    audit.said("src/methods/lattice_search.rs", "clearance: 0.85 * r,")
    audit.said("src/methods/lattice_search.rs", "interior_coordination: 12,")
    for fragment in ("fcc_cut(n_atoms, centre, nearest)", "binary_geometry::mackay_cut(n_atoms, nearest)",
                     "binary_geometry::polyicosahedral_cut(n_atoms, join, nearest)"):
        audit.record(fragment not in hop, "src/methods/cluster_hopping.rs", "template cut proposal is removed", "absent",
                     "absent" if fragment not in hop else "present")
    audit.record("&& cfg.prior_core" not in hop, "src/methods/cluster_hopping.rs", "prior_core branch is removed",
                 "absent", "absent" if "&& cfg.prior_core" not in hop else "present")
    audit.record("&& species.len() >= 2" not in hop, "src/methods/cluster_hopping.rs", "binary motif-gated branch is removed",
                 "absent", "absent" if "&& species.len() >= 2" not in hop else "present")
    geom = audit.text("src/methods/cluster_hopping/binary_geometry.rs")
    audit.record("pub(crate) fn mackay_cap(n: usize, nearest: f64)" not in geom,
                 "src/methods/cluster_hopping/binary_geometry.rs", "mackay_cap helper is removed", "absent",
                 "absent" if "pub(crate) fn mackay_cap(n: usize, nearest: f64)" not in geom else "present")

    # Host and commands.
    receipt = "paper/jctc/logs/cuh2-hardware-receipt.txt"
    audit.said(receipt, "Model name:                              Intel(R) Xeon(R) Processor")
    audit.said(receipt, "CPU family:                              6")
    audit.said(receipt, "Model:                                   207")
    audit.said(receipt, "CPU(s):                                  8")
    audit.said(receipt, "Hypervisor vendor:                       KVM")
    audit.said(receipt, "Ubuntu 13.3.0-6ubuntu2~24.04.1")
    audit.said(receipt, "rustc 1.88.0")
    audit.said(receipt, "2400.000")
    text = audit.text(SI)
    build = audit.text("scripts/gmin_panel/build_gmin.sh")
    for flag in ("-DWITH_ALIGN=OFF", "-DWITH_MPI=OFF", "-DWITH_OPENMP=OFF", "-DWITH_MYBLAS=OFF", "-DWITH_MYLAPACK=OFF"):
        audit.record(flag in build, "scripts/gmin_panel/build_gmin.sh", f"the GMIN build sets {flag}", "set",
                     "set" if flag in build else "absent")
    sessions = ""
    for shard in GMIN_SHARDS:
        prov = dict(line.split(" ", 1) for line in audit.text(f"{GMIN_RERUN}/{shard}/PROVENANCE").splitlines()
                    if " " in line)
        session = prov["source"].rsplit("/", 1)[0] + "/session.txt"
        sessions += git("show", f"{prov['commit']}:{session}").stdout
        audit.record(bool(git("show", f"{prov['commit']}:{session}").stdout), session, "the shard session is committed",
                     "present", "present" if sessions else "absent")
    audit.record("gfortran" not in sessions, "bench/gmin-panel/rerun-20261119", "the rerun sessions give no compiler",
                 "absent", "absent" if "gfortran" not in sessions else "present")
    audit.said(SI, "GMIN is built by \\path{scripts/gmin_panel/build_gmin.sh} with ALIGN, MPI, and OpenMP off, against "
                   "the system BLAS and LAPACK; the record does not give the compiler version.")
    hosts = audit.rows(imported(audit, "paper/jctc/data/hosts/3b11c160", "hosts.csv"))
    first = next(r for r in hosts if r["machine_id_sha256"].startswith("3b11c160"))
    arch = first["arch"].replace("_", "\\_")
    audit.said(SI, f"an {arch} machine with an Intel Xeon processor, {first['cpu_count']} "
                   f"logical CPUs, {first['mem_kib']}~KiB of memory, and Linux kernel {first['kernel']}, as its host "
                   "record in \\path{paper/jctc/data/hosts/3b11c160} gives;")
    audit.cell(SI, "the 3b11c160 processor", "Intel(R) Xeon(R) Processor", first["cpu_model"])
    for folder in ("paper/jctc/data/gmin_eeec8e2c/a", "paper/jctc/data/gmin_eeec8e2c/c", "paper/jctc/data/gmin_eeec8e2c/d"):
        named = {r.get("cpu", "") for r in audit.rows(f"{folder}/rows.csv")} - {""}
        audit.record(not any("Xeon" in c or "AMD" in c for c in named), f"{folder}/rows.csv",
                     "the rows name no processor beyond the machine id", "none", " ".join(sorted(named)) or "none")
    d_prov = dict(line.split(" ", 1) for line in
                  audit.text("paper/jctc/data/gmin_eeec8e2c/d/PROVENANCE").splitlines())
    d_root = d_prov["source"].rsplit("/", 1)[0]
    d_host = git("show", f"{d_prov['commit']}:{d_root}/host.txt").stdout
    d_session = git("show", f"{d_prov['commit']}:{d_root}/session.txt").stdout
    audit.record(not re.search(r"Xeon|AMD|model name|cpu_model", d_host + d_session, re.I),
                 f"{d_root}/host.txt", "the LJ150 record has no processor model to add",
                 "none", "none" if not re.search(r"Xeon|AMD|model name|cpu_model",
                                                d_host + d_session, re.I) else "present")
    develop = {r["cpu"] for r in audit.rows("paper/jctc/data/gmin_develop/rows.csv")}
    audit.cell("paper/jctc/data/gmin_develop/rows.csv", "the e06d886d processor", "Intel(R) Xeon(R) Processor",
               " ".join(sorted(develop)))
    audit.said(SI, "whose rows name an Intel Xeon processor and no memory or kernel.")
    audit.record("13.3.0" not in text.split("\\label{sec:host}", 1)[1].split("\\section", 1)[0], SI,
                 "the host paragraph prints no compiler version", "absent", "checked")
    commands = {
        "bash scripts/gmin_panel/build_gmin.sh": "scripts/gmin_panel/build_gmin.sh",
        "snakemake -s workflow/Snakefile gmin_panel": "workflow/Snakefile",
        "python3 paper/jctc/scripts/gmin_table.py": "paper/jctc/scripts/gmin_table.py",
        "python3 paper/jctc/scripts/knockout_tables.py": "paper/jctc/scripts/knockout_tables.py",
        "python3 paper/jctc/scripts/ase_table.py": "paper/jctc/scripts/ase_table.py",
        "python3 scripts/render_all.py": "paper/jctc/scripts/render_all.py",
        "python3 scripts/fig_algorithm_stack.py figs/algorithm_stack.pdf":
            "paper/jctc/scripts/fig_algorithm_stack.py",
        "python3 scripts/fig_rec_arms.py figs/rec_arms.pdf": "paper/jctc/scripts/fig_rec_arms.py",
        "python3 scripts/fig_si_structures.py figs/si_structures.pdf":
            "paper/jctc/scripts/fig_si_structures.py",
        "python3 scripts/fig_si_water.py figs/si_water.pdf": "paper/jctc/scripts/fig_si_water.py",
    }
    for command, path in commands.items():
        audit.record(command in text and (ROOT / path).exists(), SI, "the command names a script that exists",
                     command, path)
    audit.said("workflow/Snakefile", "rule gmin_panel:")
    for path in (TEX, SI):
        audit.record("\\section{Several chains}" not in audit.text(path) and "eq:rank" not in audit.text(path), path,
                     "the dropped several-chains section stays out", "absent", "absent")

    # Title, affiliation, spelling, and the pending DOI.
    cover = "paper/jctc/cover-letter.tex"
    title = "Template-free basin hopping with adaptive moves for funnelled cluster landscapes"
    audit.said(TEX, "\\title{" + title + "}")
    audit.said(SI, "\\title{" + title + "}")
    audit.said(cover, "``" + title)
    for path in (TEX, SI, cover):
        audit.record("Packing moves, symmetry proposals" not in audit.text(path), path, "the r3 title is gone", "absent",
                     "absent" if "Packing moves, symmetry proposals" not in audit.text(path) else "present")
    for path in (TEX, SI, cover):
        audit.record("basin-hopping setting" not in audit.text(path), path, "the old title is gone", "absent",
                     "absent" if "basin-hopping setting" not in audit.text(path) else "present")
    for path in (TEX, SI):
        audit.said(path, "\\affiliation{SURF B.V., Amsterdam}")
        audit.record("SURF BV" not in audit.text(path), path, "the affiliation is spelled one way", "SURF B.V.", "SURF B.V.")
        audit.record("Science Institute" not in audit.text(path), path, "the affiliations are SURF B.V. and TurtleTech ehf.",
                     "absent", "absent" if "Science Institute" not in audit.text(path) else "present")
        audit.said(path, "\\affiliation{SURF B.V., Amsterdam}\n\\alsoaffiliation{TurtleTech ehf., Reykjavik, Iceland}")
    availability = audit.text(TEX).split("\\section*{Data and software availability}", 1)[1]
    audit.record("\\texttt{PROVENANCE}" not in availability and "record branch" not in availability,
                 TEX, "row-level provenance stays in the Supporting Information", "absent",
                 "absent" if "\\texttt{PROVENANCE}" not in availability and "record branch" not in availability
                 else "present")
    audit.said(TEX, "The rows of this code behind every table were recorded at revisions in the\n"
                    "history of the submitted revision, except the length-unit and move-choice\n"
                    "knockouts, which ran the submitted search source plus default-off knockout\n"
                    "switches.")
    audit.said(TEX, "The described submitted revision postdates the newest released tag,\n"
                    "v0.12.0, and is available at\n"
                    "\\url{https://github.com/HaoZeke/anneal/commit/"
                    "61fa67cf380e46876b569da45e7cc091647a775b};\n"
                    "the Zenodo archive does not contain that revision.")
    audit.record("not yet" not in availability and "\\todo" not in availability, TEX,
                 "the data statement carries no pending item", "absent", "checked")
    audit.record("Zenodo record for this submission" not in availability, TEX,
                 "the data statement carries no pending-record note", "absent",
                 "absent" if "Zenodo record for this submission" not in availability else "present")
    for path in (TEX, SI, cover):
        found = sorted({m.group(0) for m in BRITISH.finditer(audit.text(path))})
        audit.record(not found, path, "American spelling", "none", " ".join(found) or "none")
    for figure in ("landscapes.pdf", "landfold_maps", "fig:minima-maps"):
        audit.record(figure not in audit.text(TEX), TEX, f"the main text leaves out {figure}", "absent",
                     "absent" if figure not in audit.text(TEX) else "present")


RERUN_ITEMS: dict[str, str] = {}
RERUN_TAG = re.compile(r"\\rerun\{([^}]*)\}")


def rerun_tags(text: str) -> list[tuple[int, str]]:
    """The line number and item of each \\rerun tag outside a comment."""
    return [(number, found.group(1)) for number, line in enumerate(text.splitlines(), 1)
            for found in RERUN_TAG.finditer(line) if not line.lstrip().startswith("%")]


# Results of the third rewrite that a rerun replaced. None may come back, fenced or not.
REPLACED_CLAIMS = (
    (SI, ("hit LJ\\(_{75}\\) on 0 of 12 seeds", "this code hits more seeds on 9 of 13 clusters",
          "Paired by seed, the exact McNemar test is below 0.05 on 5 clusters", "The 4 are LJ\\(_{75}\\)",
          "The other 5 wins of this code are", "9 of 12 against 5 of 12 and 12 of 12 against 2 of 12",
          "In the panel GMIN hits LJ\\(_{75}\\) and LJ\\(_{98}\\) on 0 of 12 seeds",
          "Minima hopping stopped at the cap on all 72 runs", "Both ASE arms hit 0 of 12 on all 4 Morse clusters",
          "ASE minima hopping hits LJ\\(_{38}\\) on 1 of 6, LJ\\(_{55}\\) on 6 of 6",
          "hits 121 of 192 seeds for the default hop and 97 of 192 for CREST",
          "CREST with the polish under the same budget",
          "\\section{Results that wait on reruns}")),
    (TEX, ("so the data do not attribute the hits to any single mechanism",
           "it does not yet test any of the four hypotheses separately",
           "No row was run with the shell and lattice moves switched off",
           "No row of this code was run with its symmetry projections switched off",
           "No row was run with the stall rules replaced by a random restart",
           "No row was run with the basin bias or the tabu list switched off",
           "Which component carries them, if any single one does, is not yet measured",
           "does not separate the methods from their settings",
           "Whether they hold against GMIN with symmetrization and an input for each size is not yet measured",
           "needs the endpoint structures and the occupancy along each run",
           "the occupancy trajectories will test it",
           "No Lennard-Jones row was run at the current preset, which has no structure template",
           "Whether the Lennard-Jones gains hold at the current preset, without the structure priors, is not yet measured",
           "is not yet deposited", "the per-seed rows of a rerun will allow the paired test",
           "and the results that wait on reruns.")),
)
TODO_ITEMS: dict[str, str] = {}
TODO_TAG = re.compile(r"\\todo\{([^}]*)\}")
# Wording that defers a claim to a run not made, or fences off a part of the paper as its limits.
HEDGES = ("not yet", "will test", "will show", "will allow", "future work", "in future", "remains to be",
          "\\section{Limits}", "\\section{Limitations}", "\\section*{Limits}", "\\subsection{Limits}")
# Modal words that leave a statement about the code or the rows open.
MODAL = re.compile(r"\b(may|might|could|perhaps|possibly|likely)\b", re.IGNORECASE)
# Every printed row is the run itself, so no rendered text or path calls one a rerun.
RERUN_WORD = re.compile(r"re-?run", re.IGNORECASE)


def rerun_claims(audit: Audit) -> None:
    """No result that a rerun replaced is printed again, fenced or not."""
    for path, fragments in REPLACED_CLAIMS:
        flat = collapse(audit.text(path))
        for fragment in fragments:
            found = flat.count(collapse(fragment))
            audit.record(found == 0, path, "a replaced result is not printed", fragment, f"{found} found")


def rerun_check(audit: Audit, pre_submission: bool = False) -> None:
    """No source carries a rerun tag; every tag names an item that waits on a rerun, and none does."""
    for path in RERUN_FILES:
        for number, item in rerun_tags(audit.text(path)):
            audit.record(item in RERUN_ITEMS, f"{path}:{number}", "the tag names an item that waits on a rerun",
                         " ".join(RERUN_ITEMS) or "none", item)
            audit.record(False, f"{path}:{number}", f"[RERUN: item {item}] remains ({RERUN_ITEMS.get(item, 'unknown')})",
                         "no tag", "tag present")


FORBIDDEN_PENDING = re.compile(r"\bpending\b|to be assigned|\btbd\b", re.IGNORECASE)
PENDING_SOURCES = (TEX, SI, COVER, "paper/jctc/macros.tex")


def citation_metadata(audit: Audit) -> None:
    """The author affiliations in the citation records are the ones the manuscript lists."""
    citation_rohit = audit.text("CITATION.cff").split('given-names: "Rohit"', 1)[1].split("- family-names:", 1)[0]
    audit.record('affiliation: "SURF B.V.; TurtleTech ehf"' in citation_rohit,
                 "CITATION.cff", "Rohit's citation affiliations are SURF first and TurtleTech second",
                 "SURF B.V.; TurtleTech ehf",
                 citation_rohit.strip())
    zenodo_creators = json.loads(audit.text(".zenodo.json"))["creators"]
    zenodo_rohit = next(creator for creator in zenodo_creators
                        if creator["name"] == "Goswami, Rohit")
    audit.record(zenodo_rohit.get("affiliation") == "SURF B.V.; TurtleTech ehf",
                 ".zenodo.json",
                 "Rohit's Zenodo affiliations are SURF first and TurtleTech second",
                 "SURF B.V.; TurtleTech ehf",
                 zenodo_rohit.get("affiliation", "missing"))


def pending_text(audit: Audit) -> None:
    """No TeX source says a record is pending, to be assigned, or TBD."""
    for path in PENDING_SOURCES:
        hits = sorted({m.group(0) for m in FORBIDDEN_PENDING.finditer(audit.text(path))})
        audit.record(not hits, path, "no pending, to be assigned, or TBD remains in TeX",
                     "none", " | ".join(hits) or "none")


def todo_check(audit: Audit, pre_submission: bool = False) -> None:
    """No source carries a todo marker or a hedge that defers a claim to a run not made."""
    for path in (TEX, SI, COVER):
        text = audit.text(path)
        live = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))
        for m in TODO_TAG.finditer(live):
            number = live[:m.start()].count("\n") + 1
            audit.record(False, f"{path}:{number}", f"[TODO: {m.group(1)}] remains "
                         f"({TODO_ITEMS.get(m.group(1), 'unknown')})", "no marker", "marker present")
        audit.cell(path, "todo markers", "0", float(len(TODO_TAG.findall(live))))
        audit.cell(path, "rerun tags", "0", float(len(rerun_tags(text))))
        flat = collapse(live)
        for hedge in HEDGES:
            found = flat.lower().count(hedge.lower())
            audit.record(found == 0, path, f"no hedge: {hedge}", "absent", f"{found} found")
        modal = sorted({m.group(1).lower() for m in MODAL.finditer(flat)})
        audit.record(not modal, path, "no modal hedge", "absent", " ".join(modal) or "absent")
        reruns = len(RERUN_WORD.findall(flat))
        audit.record(not reruns, path, "no rendered rerun", "absent", f"{reruns} found" if reruns else "absent")


REVISION_MAP_COLUMNS = ("revision", "why", "row_file", "system", "arm", "revision_column", "rows", "seed_min", "seed_max")


def revision_values(value: str) -> list[str]:
    """Hexadecimal revision identifiers in one cell, including joined cells."""
    return re.findall(r"(?<![0-9a-f])[0-9a-f]{7,64}(?![0-9a-f])", value or "")


# A non-empty revision cell is one lower-case hex string of 7 to 64 digits,
# the same span revision_values extracts, and nothing else. A prefix, a
# word, or a second token is malformed.
_REVISION_CELL = re.compile(r"^[0-9a-f]{7,64}$")

# harness_revision is the harness commit. Rows that predate the column
# leave it blank, which provenance already names. Every other revision
# column names the commit the row claims, so a blank cell fails. The
# sentinel "unknown" is a blank: it is not a SHA.
EMPTY_REVISION_ALLOWED = frozenset({"harness_revision"})

# Cells already in the tree that are not a hex SHA, or are blank in a
# required column. Counts, not a waiver inside the extractor: the scan
# must equal this map. A new bad cell fails, and a recorded cell that
# disappears fails.
RECORDED_REVISION_CELL_GAPS = {
    ("bench/abcluster-template-free/rerun-20261119/rows.csv", "code_revision", "abcluster-3.4@80d5ceed91bf"): 12,
    ("bench/ase-gofee-h2h/results.csv", "head", "unknown"): 3,
    ("bench/gold-gupta/seeds.csv", "code_revision", ""): 120,
    ("bench/occat-h2h/seeds.csv", "code_revision", ""): 138,
    ("bench/perf-quench9/rows.csv", "code_revision", "lemma"): 72,
    ("bench/template-free2-r7/rng-order-rows.csv", "code_revision",
     "5445c4f7d6333792dc792482f9a2a082970b9685+strict-tie"): 24,
    ("bench/template-free2-r7/rng-order-rows.csv", "code_revision",
     "pre-commit-draw-order"): 6,
    ("bench/water-gmin/rows.csv", "code_revision",
     "gmin-aac1548a8aec13b53ad12a8d6904625081685d2e"): 60,
    ("data/sc96_78_hits.csv", "code_revision", ""): 64,
    ("results/size-tiered-selection-provenance.csv", "code_revision", "unknown"): 54,
}


def revision_row_tables() -> list[tuple[str, list[dict[str, str]]]]:
    """Every CSV or TSV below the paper, benchmark, data, results and record trees."""
    tables = []
    excluded = {ALIASES, REVISION_MAP, RECORD_REFS}
    for root in ("paper/jctc/data", "bench", "data", "results", "record"):
        directory = ROOT / root
        if not directory.is_dir():
            continue
        paths = sorted((*directory.rglob("*.csv"), *directory.rglob("*.tsv")))
        for absolute in paths:
            path = absolute.relative_to(ROOT).as_posix()
            if path in excluded:
                continue
            delimiter = "\t" if absolute.suffix == ".tsv" else ","
            with absolute.open(newline="") as handle:
                lines = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
            if not lines:
                continue
            rows = list(csv.DictReader(lines, delimiter=delimiter))
            if rows and set(rows[0]).intersection(ALL_REVISION_COLUMNS):
                tables.append((path, rows))
    return tables


def revision_cell_gaps(tables: list[tuple[str, list[dict[str, str]]]] | None = None) -> dict[tuple[str, str, str], int]:
    """Counts of malformed cells and of blank cells in a required column.

    A blank harness_revision is allowed. "unknown" counts as blank on
    every column, including harness_revision.
    """
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for path, rows in revision_row_tables() if tables is None else tables:
        if not rows:
            continue
        columns = [column for column in ALL_REVISION_COLUMNS if column in rows[0]]
        for row in rows:
            for column in columns:
                value = (row.get(column) or "").strip()
                if value == "" or value == "unknown":
                    if column in EMPTY_REVISION_ALLOWED and value == "":
                        continue
                    token = "" if value == "" else "unknown"
                    counts[(path, column, token)] += 1
                    continue
                if _REVISION_CELL.fullmatch(value) is None:
                    counts[(path, column, value)] += 1
    return dict(counts)


def all_revision_provenance(audit: Audit) -> None:
    """Every revision cell is a hex SHA and reachable, or a recorded gap.

    A non-empty cell must be one lower-case hex string of 7 to 64 digits.
    Anything else is malformed and fails closed. An empty cell is allowed
    only in harness_revision; a blank required cell fails. Cells that are
    already malformed or blank in a required column must match
    RECORDED_REVISION_CELL_GAPS exactly. A well-formed cell is on origin,
    aliased, external, or mapped.
    """
    citations: dict[str, set[str]] = defaultdict(set)
    tables = revision_row_tables()
    gaps = revision_cell_gaps(tables)
    audit.record(gaps == RECORDED_REVISION_CELL_GAPS, "revision cells",
                 "malformed and required-empty revision cells match the record",
                 f"{sum(RECORDED_REVISION_CELL_GAPS.values())} recorded",
                 f"{sum(gaps.values())} found")
    for path, rows in tables:
        columns = [column for column in ALL_REVISION_COLUMNS if column in rows[0]]
        for row in rows:
            for column in columns:
                value = (row.get(column) or "").strip()
                if _REVISION_CELL.fullmatch(value) and not external_pin(value):
                    citations[value].add(path)
    names = sorted(set(citations) - set(RECORD_TREES))
    table = alias_table()
    index = alias_index(names, table)
    aliases = verified_aliases(audit, table, index)
    for revision in sorted(citations):
        where = ", ".join(sorted(citations[revision]))
        if revision in RECORD_TREES:
            held = git("rev-parse", f"{RECORD_TREES[revision]}^^{{tree}}").stdout.strip()
            audit.record(held == revision, where, f"row tree {revision[:8]} is verified",
                         revision[:8], held[:8] or "absent")
            continue
        remote = index.on_remote(revision)
        aliased = any(local.startswith(revision) for local in aliases)
        mapped = aliased or revision in UNPUSHED
        state = "origin" if remote else "pushed equivalent" if aliased else UNPUSHED.get(revision, "unmapped")
        audit.record(remote or mapped, where, f"row revision {revision[:8]} is reachable or mapped",
                     "origin or map", state)
    for revision in sorted(set(UNPUSHED) - set(citations)):
        audit.record(False, REVISION_MAP, f"mapped revision {revision[:8]} is cited",
                     UNPUSHED[revision], "no row cites it")


def revision_map_rows(audit: Audit) -> list[dict[str, str]]:
    """Each unreachable revision with every row group that cites it."""
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path, rows in revision_row_tables():
        for row in rows:
            for column in ALL_REVISION_COLUMNS:
                for value in revision_values(row.get(column, "")):
                    if value in UNPUSHED:
                        system = row.get("system", row.get("cluster", row.get("board", "")))
                        arm = row.get("arm", row.get("code", row.get("method", "")))
                        key = (value, path, system, arm, column)
                        groups[key].append(row)
    out = []
    for (revision, path, system, arm, column), rows in sorted(groups.items()):
        seeds = [int(r["seed"]) for r in rows if r.get("seed", "").strip().lstrip("-").isdigit()]
        out.append({"revision": revision, "why": UNPUSHED[revision], "row_file": path, "system": system, "arm": arm,
                    "revision_column": column, "rows": str(len(rows)),
                    "seed_min": str(min(seeds)) if seeds else "", "seed_max": str(max(seeds)) if seeds else ""})
    return out


def revision_map_text(rows: list[dict[str, str]]) -> str:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, REVISION_MAP_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


# The cap-turn summary's pooled McNemar p, withdrawn with the column.
WITHDRAWN_POOLED_MCNEMAR = 3.2e-5
CAP_TURN_README = "bench/template-free2-r7/README.md"
CAP_TURN_ROWS = "bench/template-free2-r7/rows.csv"
CAP_TURN_FLAG_OFF = "bench/template-free2-r7/flag-off-rows.csv"
CAP_TURN_SKIP = "bench/template-free2-r7/concurrent.csv"
# A plain number, scientific notation, or a mantissa times a power of ten.
NUMBER_TOKEN = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)\s*(?:\\times|×)\s*10\^\{?[-+]?\d+\}?"
    r"|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def cap_turn_section(text: str) -> str:
    """The cap-turn campaign write-up, after the fresh-seed McNemar table."""
    marker = "## 2. The cap turn"
    start = text.find(marker)
    if start < 0:
        return ""
    rest = text[start + len(marker):]
    nxt = rest.find("\n## ")
    return text[start:] if nxt < 0 else text[start:start + len(marker) + nxt]


def cap_turn_table_headers(section: str) -> list[str]:
    """Header cells of every pipe table in the cap-turn section."""
    headers: list[str] = []
    pending = True
    for line in section.splitlines():
        if not line.startswith("|"):
            pending = True
            continue
        if not pending:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and not all(set(cell) <= set("-: ") and cell for cell in cells):
            headers.extend(cells)
        pending = False
    return headers


def mcnemar_headers(headers) -> list[str]:
    """Column names that are a McNemar column, ignoring case and separators."""
    return [header for header in headers if "mcnemar" in str(header).casefold()]


def _parsed_number(token: str) -> bool:
    return parse(token.replace("×", r"\times")) is not None


def numeric_tokens(text: str) -> list[str]:
    """Numeric spellings in ``text``, including scientific and power-of-ten forms."""
    return [match.group() for match in NUMBER_TOKEN.finditer(text) if _parsed_number(match.group())]


def mcnemar_column_values(section: str) -> list[str]:
    """Parsed numbers sitting under a McNemar header, in whatever format they use."""
    values: list[str] = []
    lines = section.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("|"):
            index += 1
            continue
        header = [cell.strip() for cell in line.strip().strip("|").split("|")]
        columns = [i for i, name in enumerate(header) if "mcnemar" in name.casefold()]
        if not columns:
            index += 1
            continue
        index += 1
        if index < len(lines) and re.fullmatch(r"\|[\s:\-|]+\|", lines[index].strip()):
            index += 1
        while index < len(lines) and lines[index].startswith("|"):
            cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
            for column in columns:
                if column < len(cells):
                    values.extend(numeric_tokens(cells[column]))
            index += 1
    return values


def mcnemar_p_values(section: str, rows: list[dict[str, str]]) -> list[str]:
    """McNemar p values in the summary or the rows, in any numeric format."""
    found = list(mcnemar_column_values(section))
    for line in section.splitlines():
        if "mcnemar" in line.casefold():
            found.extend(numeric_tokens(line))
    for row in rows:
        for key, value in row.items():
            if "mcnemar" in str(key).casefold():
                found.extend(numeric_tokens(value))
            elif "mcnemar" in value.casefold():
                found.extend(numeric_tokens(value))
    return found


def withdrawn_pooled_spelling(text: str) -> str | None:
    """A numeric token in ``text`` that denotes the withdrawn pooled p, or None."""
    for token in numeric_tokens(text):
        if agrees(token.replace("×", r"\times"), WITHDRAWN_POOLED_MCNEMAR):
            return token
    return None


def cap_turn_skip(audit: Audit) -> set[tuple[str, int]]:
    """Seeds left out of the timing, read as text so a seed list is not a result table."""
    skip = set()
    for line in audit.text(CAP_TURN_SKIP).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("cell"):
            continue
        cell, seed = line.split(",", 2)[:2]
        skip.add((cell, int(seed)))
    return skip


def cap_turn_outside_ratios(rows: list[dict[str, str]], skip: set[tuple[str, int]]) -> dict[str, float]:
    """CT/base ratio of type-7 medians of (wall - pot) / charged, skip applied."""
    by: dict[str, dict[str, dict[int, dict[str, str]]]] = defaultdict(dict)
    for row in rows:
        by[row["cell"]].setdefault(row["arm"], {})[int(row["seed"])] = row
    ratios = {}
    for cell, arms in by.items():
        if "base" not in arms or "CT" not in arms:
            continue
        timed = [seed for seed in sorted(set(arms["base"]) & set(arms["CT"])) if (cell, seed) not in skip]
        if not timed:
            continue

        def per_call(arm: str) -> list[float]:
            return [(float(arms[arm][seed]["wall_s"]) - float(arms[arm][seed]["pot_s"]))
                    / float(arms[arm][seed]["charged"]) for seed in timed]

        ratios[cell] = median(per_call("CT")) / median(per_call("base"))
    return ratios


def cap_turn_ratio_text(ratio: float) -> str:
    """Three decimals, rounded once from full precision, half away from zero.

    ``summarise.py`` prints the ratio to three places with the same rule,
    so 0.992484 is written 0.992 rather than double-rounded via 0.9925.
    """
    return format(Decimal(str(ratio)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP), "f")


def _flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def cap_turn_flag_off_mismatches(base_rows: list[dict[str, str]], off_rows: list[dict[str, str]]) -> list[str]:
    """Off-arm rows that disagree with the base arm on hit, charged calls, or identity."""
    base = {(row["cell"], int(row["seed"])): row for row in base_rows if row.get("arm", "base") == "base"}
    mismatches = []
    for off in off_rows:
        key = (off["cell"], int(off["seed"]))
        row = base.get(key)
        if row is None:
            mismatches.append(f"{off['cell']} seed {off['seed']} has no base row")
            continue
        where = f"{off['cell']} seed {off['seed']}"
        if _flag(off["hit"]) != _flag(row["hit"]):
            mismatches.append(f"{where} hit")
        if int(float(off["charged"])) != int(float(row["charged"])):
            mismatches.append(f"{where} charged")
        off_hash = off.get("best_bits", "").strip()
        base_hash = row.get("best_bits", "").strip()
        energy = "best" in off and "best" in row and off["best"].strip() and row["best"].strip()
        hashed = bool(off_hash) and bool(base_hash)
        if energy and not agrees(off["best"], float(row["best"])):
            mismatches.append(f"{where} energy")
        if hashed and off_hash != base_hash:
            mismatches.append(f"{where} hash")
        if not energy and not hashed:
            mismatches.append(f"{where} identity")
    return mismatches


def cap_turn_campaign(audit: Audit) -> None:
    """The r7 cap-turn table follows its rows and quotes no McNemar p for the unrecorded build."""
    readme = CAP_TURN_README
    rows = audit.rows(CAP_TURN_ROWS)
    off_rows = audit.rows(CAP_TURN_FLAG_OFF)
    hits: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for r in rows:
        hits[(r["cell"], r["arm"])][int(r["seed"])] = int(r["hit"])
    audit.said(readme, "| cell | base | CT | Fisher p | base only | CT only | outside/call ratio |")
    totals = [0, 0, 0, 0, 0]
    for cell in ("lj38", "m38r14", "m55r14", "lj75", "lj98", "lj102", "lj103", "lj104"):
        base, arm = hits[(cell, "base")], hits[(cell, "CT")]
        seeds = sorted(set(base) & set(arm))
        n, hb, ha = len(seeds), sum(base[s] for s in seeds), sum(arm[s] for s in seeds)
        ob = sum(1 for s in seeds if base[s] and not arm[s])
        oa = sum(1 for s in seeds if arm[s] and not base[s])
        totals = [t + v for t, v in zip(totals, (n, hb, ha, ob, oa))]
        audit.said(readme, f"| {cell} | {hb}/{n} | {ha}/{n} | {fisher(hb, n, ha, n):.2g} | {ob} | {oa} |")
    n, hb, ha, ob, oa = totals
    audit.said(readme, f"| all | {hb}/{n} | {ha}/{n} | | {ob} | {oa} | |")
    audit.said(readme, "No McNemar p is given for the cap turn")

    section = cap_turn_section(audit.text(readme))
    row_headers = mcnemar_headers(rows[0].keys() if rows else [])
    table_headers = mcnemar_headers(cap_turn_table_headers(section))
    audit.record(not row_headers, CAP_TURN_ROWS, "the cap-turn rows have no McNemar column",
                 "absent", " ".join(row_headers) or "absent")
    audit.record(not table_headers, readme, "the cap-turn summary has no McNemar column",
                 "absent", " ".join(table_headers) or "absent")
    p_values = mcnemar_p_values(section, rows)
    audit.record(not p_values, readme, "the cap-turn record states no McNemar p",
                 "absent", p_values[0] if p_values else "absent")
    pooled = withdrawn_pooled_spelling(section)
    audit.record(pooled is None, readme, "the withdrawn pooled McNemar p is absent",
                 "absent", pooled or "absent")

    skip = cap_turn_skip(audit)
    computed = cap_turn_outside_ratios(rows, skip)
    recorded = {row["cell"]: row["outside/call ratio"].strip()
                for _, row in md_table(audit, readme, "cell") if row.get("outside/call ratio", "").strip()}
    for cell, shown in recorded.items():
        audit.record(cell in computed and shown == cap_turn_ratio_text(computed[cell]), readme,
                     f"{cell} outside/call ratio matches the rows", shown,
                     cap_turn_ratio_text(computed[cell]) if cell in computed else "missing")
    audit.record(set(recorded) == set(computed), readme, "every timed cell has an outside/call ratio",
                 " ".join(sorted(recorded)), " ".join(sorted(computed)))

    mismatches = cap_turn_flag_off_mismatches(rows, off_rows)
    audit.record(not mismatches, CAP_TURN_FLAG_OFF, "flag-off rows match the base rows",
                 "match", mismatches[0] if mismatches else "match")


def revision_map(audit: Audit) -> None:
    """The committed map from each unreachable revision to its panel rows equals the rows."""
    rows = revision_map_rows(audit)
    audit.record((ROOT / REVISION_MAP).read_text() == revision_map_text(rows), REVISION_MAP, "map equals the cited rows",
                 "committed file", f"{len(rows)} row groups")
    audit.record({r["revision"] for r in rows} == set(UNPUSHED), REVISION_MAP, "every unreachable revision is mapped",
                 f"{len(UNPUSHED)} revisions", f"{len({r['revision'] for r in rows})} mapped")


BRITISH = re.compile(r"\b\w*(?:optimis|neighbour|initialis|symmetris|normalis|minimis|colour|centred|licence|"
         r"favour|behaviour|analyse|organis|recognis|characteris|parametris|summaris|catalogue|"
         r"labell|travell|modell|cancell|fuell)\w*\b",
         re.IGNORECASE)


def crest_equal(audit: Audit) -> None:
    """The committed CREST rows, and the absence of that panel from the paper."""
    h2h = load_script("h2h_tables_crest", "paper/jctc/scripts/h2h_tables.py")
    where = "bench/crest-equal/rows.csv"
    rows = audit.rows(where)
    revision = rows[-1]["code_revision"]
    rows = [r for r in rows if r["code_revision"] == revision]
    arms = {arm: [r for r in rows if r["arm"] == arm] for arm in h2h.CREST_ARMS}
    summary = h2h.crest_equal_summary()
    for arm, runs in arms.items():
        seeds = sorted(int(r["seed"]) for r in runs)
        audit.record(seeds == list(range(2000, 2012)), where, f"crest-equal {arm}: seeds 2000-2011, once each",
                     "2000-2011", " ".join(map(str, seeds)))
        audit.record(all(r["crest_exit"] in ("", "0") for r in runs), where, f"crest-equal {arm}: every run exits 0",
                     "0", " ".join(sorted({r["crest_exit"] or "0" for r in runs})))

    def calls(arm: str) -> list[float]:
        return [float(r["charged"]) for r in arms[arm]]

    def hits(arm: str) -> int:
        return sum(r["hit"] == "true" for r in arms[arm])

    target = calls("anneal")
    audit.record(len(set(target)) == 1, where, "this setting charges the same GFN2 calls on every seed", "one count",
                 " ".join(sorted({f"{v:.0f}" for v in target})))
    center = target[0]
    tolerance = h2h.CREST_CALL_TOLERANCE
    off = abs(median(calls("crest")) - center) / center
    audit.record(off <= tolerance, where,
                 f"CREST at 0.90 ps is at the calls of this setting: median within {tolerance:.0%}",
                 f"<= {tolerance:.0%}", f"{off:.2%}")
    within = sum(abs(v - center) <= tolerance * center for v in calls("crest"))
    audit.cell(where, "CREST runs within the call tolerance", summary["crest"]["within_5pct"], float(within))
    within_wall = sum(abs(v - center) <= tolerance * center for v in calls("crest-wall"))
    audit.cell(where, "CREST runs at 0.11 ps within the call tolerance", summary["crest-wall"]["within_5pct"],
               float(within_wall))
    readme = audit.text(CREST_EQUAL_README)
    for fragment in (
            "The CREST rows are charged the calls the driver served.",
            "CREST's own\n  `Total number of energy+grad calls` line was not read and no CREST log was\n  kept",
            "The driver was started with a ceiling of 50000 calls, so the 4000\n  in the `budget` column of the CREST rows is a label, not a cap.",
    ):
        audit.record(fragment in readme, CREST_EQUAL_README, "the recorded CREST protocol", fragment,
                     "present" if fragment in readme else "absent")
    stale = ("refuses calls past the 4000-call ledger", "charged the larger of the driver's count",
             "for CREST the larger of the")
    for fragment in stale:
        audit.record(fragment not in audit.text(PAPER), PAPER, "no next-run CREST protocol in the recorded panel",
                     "absent", "present" if fragment in audit.text(PAPER) else "absent")
    audit.record(float(summary["crest"]["crest_len_ps"]) == 0.90 and float(summary["crest-wall"]["crest_len_ps"]) == 0.11,
                 where, "the CREST lengths are 0.90 and 0.11 ps", "0.90 0.11",
                 f"{summary['crest']['crest_len_ps']} {summary['crest-wall']['crest_len_ps']}")
    for arm in ("crest", "crest-wall"):
        a, b = arms["anneal"], arms[arm]
        audit.cell(where, f"{arm} Fisher p", summary[arm]["fisher_p"], fisher(hits("anneal"), len(a), hits(arm), len(b)))
        for key, column in (("charged", "mannwhitney_calls_p"), ("pot_s", "mannwhitney_inside_p"),
                            ("wall_s", "mannwhitney_wall_p"), ("outside_s", "mannwhitney_outside_p")):
            p = mann_whitney([float(r[key]) for r in a], [float(r[key]) for r in b], continuity=True)
            audit.cell(where, f"{arm} Mann-Whitney p on {key}", summary[arm][column], p)


def manuscript_followups(audit: Audit) -> None:
    """The cluster-preset hits, the wall-cap reconciliation, the GMIN settings, and the M55 disclosures."""
    audit.said(SI, "The Lennard-Jones rows run the Wales-group LJ\\(_{38}\\) introduction file at every Lennard-Jones size,")
    audit.said(SI, "The Morse rows add GMIN's native \\texttt{MORSE} keyword and \\(\\rho\\) to the same numerical settings.")
    audit.said(SI, "# GMIN: LJ/Morse use TEMPERATURE 1.0 and NEWRESTART 40")
    audit.said(SI, "with \\(T=1.0\\), \\texttt{NEWRESTART 40}, and an energy difference of 0.003")
    data = "examples/h2h_gmin_lj.data"
    for fragment in ("TEMPERATURE 1.0", "NEWRESTART 40", "EDIFF 0.003"):
        audit.said(data, fragment)
    audit.record("SYMMETRISE" not in audit.text(data), data, "the table input has no symmetrization keyword", "absent",
                 "absent" if "SYMMETRISE" not in audit.text(data) else "present")
    for gone in ("Au55 Gupta uses TEMPERATURE", "Darby gold parameters"):
        for path in (TEX, SI):
            audit.record(gone not in audit.text(path), path, f"{gone} is not printed", "absent",
                         "absent" if gone not in audit.text(path) else "present")
    audit.record("Mackay cap" not in audit.text(TEX), TEX, "Mackay-cap disclosure prose is removed",
                 "absent", "absent" if "Mackay cap" not in audit.text(TEX) else "present")
    audit.record(not (ROOT / "src/methods/cluster_hopping/mackay_cap_tests.rs").exists(),
                 "src/methods/cluster_hopping/mackay_cap_tests.rs", "Mackay cap test module is removed",
                 "absent", "present" if (ROOT / "src/methods/cluster_hopping/mackay_cap_tests.rs").exists() else "absent")
    audit.said("paper/jctc/data/ase_panel/morse/PROVENANCE", "source bench/ase-nonlj/rerun-20261119/rows.csv")
    audit.record("two-shell icosahedron" not in audit.text(TEX), TEX, "cap-disclosure prose is removed",
                 "absent", "present" if "two-shell icosahedron" in audit.text(TEX) else "absent")
    audit.record("largest cap" not in audit.text(TEX), TEX, "cap-size prose is removed",
                 "absent", "present" if "largest cap" in audit.text(TEX) else "absent")


APPEND_ONLY_BASES = (
    "HEAD",
    "origin/develop",
    "origin/main",
    "origin/docs/paper-ms4",
    "origin/bench/crest-equal",
)
APPEND_ONLY_RECORDS = {
    "bench/h2h.json",
    "bench/h2h-additions.jsonl",
    "bench/tip4p-h2h/rows-h2h.csv",
    "bench/tip4p-h2h/rows-h2h-r2.csv",
    "bench/tip4p-h2h/hosts-r2.csv",
    "bench/tip4p-h2h/sessions.jsonl",
    "bench/tip4p-h2h/sessions-r2.jsonl",
    "bench/tip4p-h2h/wall-r2.csv",
    "bench/tip4p-h2h/check-r2.md",
    "paper/jctc/FIGURE-SOURCE.tsv",
    "paper/jctc/FIGURE-SOURCE-updates.tsv",
    "results/size-tiered-selection.csv",
    "results/size-tiered-selection-provenance.csv",
    "bench/crest-equal/call-counts.csv",
    "bench/crest-equal/rgpot-revisions.csv",
    "bench/covalent-h2h/gfn2/rgpot-revisions.csv",
    "bench/generic-alloy/g3-relabel.csv",
    "data/hf-files.csv",
    "data/hf-files-updates.csv",
    "data/hf-tree.csv",
}


def touched_record_paths(base: str) -> set[str]:
    """Every csv and jsonl path named from base to HEAD, including one absent at HEAD."""
    log = git("log", "--name-only", "--pretty=format:", "--no-renames", f"{base}..HEAD")
    return {line for line in log.stdout.splitlines() if line.endswith((".csv", ".jsonl"))}


def _upstream_removed_lines(ref: str, base_ref: str | None, path: str) -> set[str]:
    """The lines that ``base_ref`` deleted from ``path`` after it forked from ``ref``.

    Such a line is the base branch's own rewrite. This branch is not held to
    keep it, but it is held to every other line of ``ref``, in order, and to
    every line that ``ref`` added after the fork.
    """
    if base_ref is None or ref == base_ref:
        return set()
    fork = git("merge-base", ref, base_ref).stdout.strip()
    if not fork:
        return set()
    forked = git("show", f"{fork}:{path}")
    upstream = git("show", f"{base_ref}:{path}")
    if forked.returncode != 0 or upstream.returncode != 0:
        return set()
    return set(forked.stdout.splitlines(keepends=True)) - set(upstream.stdout.splitlines(keepends=True))


def _first_rewritten_line(old_lines: list[str], cur_lines: list[str], allowed_missing: set[str]) -> int | None:
    """The 1-based line of ``old_lines`` that ``cur_lines`` rewrites, or None when it extends them."""
    cursor = 0
    for index, old_line in enumerate(old_lines, start=1):
        if cursor < len(cur_lines) and cur_lines[cursor] == old_line:
            cursor += 1
        elif old_line not in allowed_missing:
            return index
    return None


def append_only_rows(audit: Audit) -> None:
    """Every record preserves baseline lines and the series never removes one."""
    paths = set(git("ls-files", "**/rows.csv", "rows.csv").stdout.splitlines())
    paths.update(path for path in APPEND_ONLY_RECORDS if (ROOT / path).is_file())
    existing_refs = [
        ref for ref in APPEND_ONLY_BASES
        if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode == 0
    ]
    base_ref = next(
        (ref for ref in ("origin/develop", "develop")
         if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode == 0),
        None,
    )
    if base_ref is not None:
        paths.update(touched_record_paths(base_ref))
    # The map is regenerated in sorted order from the cited rows and checked against them by revision_map.
    paths.discard(REVISION_MAP)
    for path in sorted(paths):
        if not (ROOT / path).is_file():
            continue
        cur_lines = audit.text(path).splitlines(keepends=True)
        rewritten = []
        refs_for_path = (
            existing_refs
            if path == "rows.csv" or path.endswith("/rows.csv")
            else [ref for ref in existing_refs if ref in ("HEAD", "origin/develop")]
        )
        for ref in refs_for_path:
            shown = git("show", f"{ref}:{path}")
            if shown.returncode != 0:
                continue
            old_lines = shown.stdout.splitlines(keepends=True)
            allowed = _upstream_removed_lines(ref, base_ref, path)
            if len(cur_lines) < len(old_lines) - len(allowed.intersection(old_lines)):
                rewritten.append(f"{ref}: truncated ({len(cur_lines)} < {len(old_lines)} lines)")
                continue
            idx = _first_rewritten_line(old_lines, cur_lines, allowed)
            if idx is not None:
                rewritten.append(f"{ref}: line {idx} rewritten")
        audit.record(
            not rewritten,
            path,
            "committed rows.csv lines are append-only",
            "append-only",
            "; ".join(rewritten) or "append-only",
        )
    base = base_ref
    if base is not None:
        # A file the base never committed is this branch's own record. A
        # correction of that file, or dropping it, is not a removed line of
        # a base record. A file the base committed stays in the log, including
        # when HEAD no longer has it. A name with no commit in the range
        # still goes to the log.
        history_paths = []
        for path in sorted(paths):
            if git("cat-file", "-e", f"{base}:{path}").returncode == 0:
                history_paths.append(path)
            elif not (ROOT / path).is_file() and not git(
                "rev-list", "-1", f"{base}..HEAD", "--", path
            ).stdout.strip():
                history_paths.append(path)
        history = git(
            "log", "-p", "--no-renames", "--format=", f"{base}..HEAD", "--",
            *history_paths,
        ).stdout.splitlines() if history_paths else []
        removed = [line for line in history if line.startswith("-") and not line.startswith("---")]
        audit.record(
            not removed,
            "record-file history",
            f"{base}..HEAD removes no committed record lines",
            "no removed lines",
            f"{len(removed)} removed lines" if removed else "no removed lines",
        )


def tip4p_r2_rows(audit: Audit) -> None:
    """The ABBA TIP4P rows, their session records and the per-size wall summary."""
    tip_rows_path = 'bench/tip4p-h2h/rows-h2h-r2.csv'
    tip_rows = audit.rows(tip_rows_path)
    sizes = ('8', '15', '20', '21', '25')
    groups = {(size, arm): pick(tip_rows, molecules=size, arm=arm) for size in sizes for arm in ('anneal', 'gmin')}
    audit.cell(PAPER, 'TIP4P rows per size and arm', '12', float(min((len(group) for group in groups.values()))))
    seed_values = [int(row['seed']) for row in tip_rows]
    audit.cell(PAPER, 'TIP4P seed range', '2000 2011', f'{min(seed_values)} {max(seed_values)}')
    same_starts = all(({row['seed']: row['start_id'] for row in groups[size, 'anneal']} == {row['seed']: row['start_id'] for row in groups[size, 'gmin']} for size in sizes))
    audit.record(same_starts, tip_rows_path, 'both TIP4P arms use the same start identifier at each seed', 'same', 'same' if same_starts else 'different')
    machine_ids = {row['machine_id'] for row in tip_rows}
    cores = {row['core'] for row in tip_rows}
    audit.record(len(machine_ids) == 1 and cores == {'0'}, tip_rows_path, 'TIP4P r2 uses one machine pinned to core 0', 'one machine core 0', f"{len(machine_ids)} machines cores {' '.join(sorted(cores))}")
    abba = all(([next((row['arm_order'] for row in groups[size, 'anneal'] if row['seed'] == seed)) for seed in sorted((row['seed'] for row in groups[size, 'anneal']))] == ['anneal,gmin' if index % 2 == 0 else 'gmin,anneal' for index in range(12)] for size in sizes))
    audit.record(abba, tip_rows_path, 'TIP4P r2 uses ABBA arm order', 'ABBA', 'ABBA' if abba else 'other')
    hit_counts = {key: sum((int(row['hit']) for row in group)) for key, group in groups.items()}
    audit.record(all((hit_counts[size, 'anneal'] > hit_counts[size, 'gmin'] for size in ('8', '15', '20'))) and all((hit_counts[size, arm] == 0 for size in ('21',) for arm in ('anneal', 'gmin'))), tip_rows_path, 'the TIP4P discussion matches the per-seed hit rows', 'anneal leads at 8, 15, 20; both miss at 21; 25 is unscored', 'matches')
    wall_path = 'bench/tip4p-h2h/wall-r2.csv'
    with (ROOT / wall_path).open(newline='') as handle:
        wall_rows = {row['molecules']: row for row in csv.DictReader(handle)}
    r2_runner = load_script('tip4p_h2h_r2_audit', 'bench/tip4p-h2h/run_h2h_r2.py')
    for size in sizes:
        summary = wall_rows[size]
        anneal_walls = [float(row['wall_s']) for row in groups[size, 'anneal']]
        gmin_walls = [float(row['wall_s']) for row in groups[size, 'gmin']]
        exact_p, method = r2_runner.mann_whitney_u_two_sided_exact(anneal_walls, gmin_walls)
        audit.cell(wall_path, f'TIP4P {size} r2 anneal n', summary['anneal_n'], len(anneal_walls))
        audit.cell(wall_path, f'TIP4P {size} r2 GMIN n', summary['gmin_n'], len(gmin_walls))
        expected_hits = 'unscored' if size == '25' else None
        audit.cell(wall_path, f'TIP4P {size} r2 anneal hits', summary['anneal_hits_n'], expected_hits or f"{hit_counts[size, 'anneal']}/12")
        audit.cell(wall_path, f'TIP4P {size} r2 GMIN hits', summary['gmin_hits_n'], expected_hits or f"{hit_counts[size, 'gmin']}/12")
        for arm, walls in (('anneal', anneal_walls), ('gmin', gmin_walls)):
            calls = [float(row['calls']) for row in groups[size, arm]]
            energies = [float(row['best_energy']) for row in groups[size, arm]]
            audit.cell(wall_path, f'TIP4P {size} r2 {arm} calls median', summary[f'{arm}_calls_median'], median(calls))
            audit.cell(wall_path, f'TIP4P {size} r2 {arm} energy best', summary[f'{arm}_energy_best_kj_mol'], min(energies))
            audit.cell(wall_path, f'TIP4P {size} r2 {arm} energy median', summary[f'{arm}_energy_median_kj_mol'], median(energies))
            audit.cell(wall_path, f'TIP4P {size} r2 {arm} wall median', summary[f'{arm}_wall_median_s'], median(walls))
            audit.cell(wall_path, f'TIP4P {size} r2 {arm} wall IQR', summary[f'{arm}_wall_iqr_s'], type7(walls, 0.75) - type7(walls, 0.25))
            audit.cell(wall_path, f'TIP4P {size} r2 {arm} wall maximum', summary[f'{arm}_wall_max_s'], max(walls))
        audit.record(summary['mwu_method'] == method, wall_path, f'TIP4P {size} exact wall test method', method, summary['mwu_method'])
        audit.cell(wall_path, f'TIP4P {size} exact wall p', summary['wall_mwu_p'], exact_p)
    audit.cell(tip_rows_path, 'TIP4P ABBA comparison runs', '120', float(len(tip_rows)))
    h2h_runner = audit.text('scripts/tip4p_h2h/run_h2h.py')
    shared_hit_tol = 'from target_tol import DEFAULT_HIT_TOL, energy_hits' in h2h_runner and 'HIT_TOL = DEFAULT_HIT_TOL\n' in h2h_runner
    audit.record(shared_hit_tol, 'scripts/tip4p_h2h/run_h2h.py', 'the TIP4P-vs-GMIN hit tolerance', 'shared 0.001 kJ/mol', 'shared 0.001 kJ/mol' if shared_hit_tol else 'other')
    sessions_path = 'bench/tip4p-h2h/sessions-r2.jsonl'
    sessions = [json.loads(line) for line in audit.text(sessions_path).splitlines()]
    session_tols = {(row.get('hit_tol'), row.get('hit_tol_unit')) for row in sessions}
    audit.record(session_tols == {(0.001, 'kJ/mol')}, sessions_path, 'every ABBA session records its hit tolerance, 0.001 kJ/mol', '0.001 kJ/mol', sorted(map(str, session_tols)))


def tip4p_unscored(audit: Audit) -> None:
    """Audit the independent TIP4P tolerances and the unscored size 25."""
    wall_path = "bench/tip4p-h2h/wall-r2.csv"
    wall = next(
        row for row in csv.DictReader(audit.text(wall_path).splitlines())
        if row["molecules"] == "25"
    )
    audit.record(
        wall["anneal_hits_n"] == wall["gmin_hits_n"] == "unscored",
        wall_path,
        "TIP4P size 25 is unscored for both arms",
        "unscored unscored",
        f"{wall['anneal_hits_n']} {wall['gmin_hits_n']}",
    )
    check_row = re.search(
        r"^\|\s*25\s*\|.*$", audit.text("bench/tip4p-h2h/check-r2.md"),
        re.MULTILINE,
    )
    audit.record(
        check_row is not None
        and check_row.group(0).count("unscored") == 2
        and re.search(r"\d+/\d+", check_row.group(0)) is None,
        "bench/tip4p-h2h/check-r2.md",
        "the size-25 record is marked unscored rather than counted",
        "two unscored cells",
        check_row.group(0) if check_row else "missing",
    )
    rates_tolerance = 0.001 * KJ_PER_KCAL
    shown_kj = format(rates_tolerance, '.6f')
    for path in ('bench/tip4p-water-rates/rates-a24ba760.csv', 'bench/tip4p-water-rates/rates.csv'):
        rows = list(csv.DictReader(audit.text(path).splitlines()))
        mismatches = [row for row in rows if int(row['hit']) != int(float(row['best_kj']) <= float(row['reference_kj']) + rates_tolerance)]
        audit.record(not mismatches, path, f'every water-rate hit flag uses the printed {shown_kj} kJ/mol tolerance', 'all flags reproduce', f'{len(mismatches)} mismatches' if mismatches else 'all flags reproduce')
    reproduce = audit.text('paper/jctc/REPRODUCE.md')
    audit.record('--rows-out /work/tip4p-h2h-rerun/rows-r2.csv' in reproduce and '--rows /work/tip4p-h2h-rerun/rows-r2.csv' in reproduce and ('--reference-rows bench/tip4p-h2h/rows-h2h-r2.csv' in reproduce), 'paper/jctc/REPRODUCE.md', 'the TIP4P rerun writes fresh records and compares them with committed rows', 'fresh output and committed comparison', 'present' if '/work/tip4p-h2h-rerun/rows-r2.csv' in reproduce else 'missing')


_ANCESTOR_OF_HEAD: dict[str, bool] = {}
_EXTERNAL_FORM = re.compile(
    r"https://[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+\.git@" + revision_reach.commit_id_pattern() + r"(?![0-9A-Za-z])"
)
_FORM_ATTEMPT = re.compile(r"https://[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+\.git@[0-9a-f]+")
_REVISION_HEADER = re.compile(r"harness|revision|commit|\bsha\b", re.I)
_RESULT_HEADER = re.compile(
    r"hit|call|wall|fisher|mann|median|inside|outside|ratio|gap|\bpot\b|eval|ahead|energy|best",
    re.I,
)
_RUN_HEADER = re.compile(r"hit|call|wall", re.I)
_NUMBER_CELL = re.compile(
    r"^[~≈]?[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-−+]?\d+)?(?:/\d+)?"
    r"\s*(?:%|×|x|s|ms|µs|μs|us|h|min|eV|meV|ε|Å|kcal/mol|kJ/mol)?$",
    re.I,
)
_SHA_CELL = re.compile(f"^{revision_reach.commit_id_pattern(7)}$")
_SHA_TOKEN = re.compile(f"(?<![0-9A-Za-z])({revision_reach.commit_id_pattern(7)})(?![0-9A-Za-z])")
_REVISION_FIELDS = ("code_revision", "harness_revision", "harness_commit", "head", "anneal_head", "anneal_revision",
                    "release_revision")
EXTERNAL_FORM_SOURCES = (
    "scripts/pimc_lj38/run_campaign.py",
    "scripts/pimc_lj38/lj_potential.py",
    "scripts/pimc_lj38/rpqa_driver.py",
    "scripts/pimc_lj38/rpqa_ipi.py",
)


# Install scripts that pin another code, each with a URL= and a PIN= line.
EXTERNAL_PIN_SOURCES = ("scripts/pimc_lj38/install_rpqa.sh",)


@functools.cache
def external_pins() -> frozenset[tuple[str, str]]:
    """The (repository URL, full SHA) pairs the install scripts check out."""
    pins = set()
    for path in EXTERNAL_PIN_SOURCES:
        text = (ROOT / path).read_text()
        urls = re.findall(r"^URL=(\S+)$", text, re.M)
        shas = [sha for sha in re.findall(r"^PIN=(\S+)$", text, re.M) if revision_reach.is_full_sha(sha)]
        if len(urls) == 1 and len(shas) == 1:
            pins.add((urls[0], shas[0]))
    return frozenset(pins)


def qualified_external_form(text: str) -> bool | None:
    """A repo URL plus a full SHA that an install script pins.

    Any other URL@hex, a short or long SHA, or a pair no script pins, is
    rejected, not skipped. None when the text is not a URL@hex at all.
    """
    token = (text or "").strip().strip("`")
    if not _FORM_ATTEMPT.fullmatch(token):
        return None
    url, sha = token.rsplit("@", 1)
    return revision_reach.is_full_sha(sha) and (url, sha) in external_pins()


external_form = revision_reach.external_form


external_pin = revision_reach.external_pin
TUNED_SYSTEMS = revision_reach.TUNED_SYSTEMS
TUNED_CODES = revision_reach.TUNED_CODES


def claimed_commit_reachable(rev: str) -> bool:
    """A commit a paper script names in prose: an ancestor of HEAD, or a pin, whether or not abbreviated."""
    return external_pin(rev, abbreviated=True) or commit_held(rev)


def tuned_gmin_harness_verified(row: dict[str, str], repo: Path = ROOT) -> bool:
    """Verify a tuned-GMIN row. The check is shared with the ingest and the table generators."""
    return revision_reach.tuned_gmin_harness_verified(row, repo, commit_held if repo == ROOT else None)


def panel_rows_held(rows: list[dict[str, str]], *, anneal: bool) -> bool:
    """Anneal rows need their code and harness commits. GMIN rows need the harness commit."""
    if not rows:
        return False
    for row in rows:
        if not commit_held(row["harness_revision"]):
            return False
        if anneal and not commit_held(row["code_revision"]):
            return False
    return True


def quoted_systems(panel: object, groups: dict) -> list[str]:
    """Panel systems whose scored rows have commits this repository holds."""
    return [
        system for system in panel.PAPER_BUDGETS
        if (
            panel_rows_held(groups[(system, "anneal")], anneal=True)
            and panel_rows_held(groups[(system, "gmin")], anneal=False)
        )
    ]


@functools.cache
def commit_of(rev: str) -> str | None:
    """The full commit a hex revision names in this repository; None when it names none or several."""
    rev = (rev or "").strip()
    if not re.fullmatch(revision_reach.commit_id_pattern(7), rev):
        return None
    return git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").stdout.strip() or None


def ancestor_of_head(rev: str) -> bool:
    """The revision is a commit this repository holds, and it is an ancestor of HEAD."""
    rev = (rev or "").strip()
    if rev not in _ANCESTOR_OF_HEAD:
        full = commit_of(rev)
        _ANCESTOR_OF_HEAD[rev] = bool(full) and git("merge-base", "--is-ancestor", full, "HEAD").returncode == 0
    return _ANCESTOR_OF_HEAD[rev]


@functools.cache
def _verified_aliases() -> tuple[tuple[str, str], ...]:
    table = alias_table()
    index = alias_index([r["local_revision"] for r in table], table)
    return tuple(verified_aliases(Audit(), table, index).items())


# Network reads of record refs are off unless main() is passed --fetch-records.
# A record/<name> branch is fetched into one fixed ref. A refs/audit/record/<name>
# ref is fetched by that name.
_FETCH_RECORDS = False
_AUDIT_FETCH_REF = "refs/audit/record/fetched"
_RECORD_REF_NAME = re.compile(r"record/[A-Za-z0-9][A-Za-z0-9._-]*")
_AUDIT_RECORD_REF_NAME = re.compile(r"refs/audit/record/[A-Za-z0-9][A-Za-z0-9._-]*")


def record_ref_name(ref: str) -> bool:
    """True when ref is a single record/<name> git will accept as a branch.

    A colon, a dash option, .., a trailing dot, or a .lock suffix is
    rejected before it can reach a fetch refspec.
    """
    if _RECORD_REF_NAME.fullmatch(ref) is None:
        return False
    if ".." in ref or ref.endswith(".") or ref.endswith(".lock"):
        return False
    return git("check-ref-format", "--branch", ref).returncode == 0


def audit_record_ref_name(ref: str) -> bool:
    """True when ref is the full name refs/audit/record/<name>.

    The same rejects as a branch row: a colon, .., a trailing dot, or a
    .lock suffix never reaches a fetch argument.
    """
    if _AUDIT_RECORD_REF_NAME.fullmatch(ref) is None:
        return False
    if ".." in ref or ref.endswith(".") or ref.endswith(".lock"):
        return False
    return git("check-ref-format", ref).returncode == 0


@functools.cache
def origin_record_heads() -> dict[str, str]:
    """The record/* branches on origin, by name. Empty when origin cannot be reached, so every registry row fails."""
    out = git("ls-remote", "origin", "refs/heads/record/*")
    if out.returncode != 0:
        return {}
    return {ref.removeprefix("refs/heads/"): sha for sha, ref in (line.split("\t") for line in out.stdout.splitlines())}


@functools.cache
def origin_audit_record_refs() -> dict[str, str]:
    """The refs/audit/record/* refs on origin, by full name. Empty when origin cannot be reached."""
    out = git("ls-remote", "origin", "refs/audit/record/*")
    if out.returncode != 0:
        return {}
    return {ref: sha for sha, ref in (line.split("\t") for line in out.stdout.splitlines())}


def record_ref_status(path: str) -> tuple[tuple[str, str, tuple[tuple[str, bool, str, str], ...]], ...]:
    """Each registered record ref with its checks: it is a record ref named by a full SHA, origin holds it at
    that SHA, git fetch origin brings it here, and its tree is the registered one."""
    return _record_ref_status(ROOT, path, _FETCH_RECORDS)


@functools.cache
def _record_ref_status(root: Path, path: str, fetch_records: bool) -> tuple[tuple[str, str, tuple[tuple[str, bool, str, str], ...]], ...]:
    if not (root / path).exists():
        return ()
    with (root / path).open(newline="") as handle:
        table = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    if not table:
        return ()
    heads = {}
    if fetch_records:
        heads.update(origin_record_heads())
        heads.update(origin_audit_record_refs())
    out = []
    for row in table:
        ref, sha, tree = row["ref"].strip(), row["sha"].strip(), row["tree"].strip()
        as_branch = record_ref_name(ref)
        as_ref = audit_record_ref_name(ref)
        named = (as_branch or as_ref) and revision_reach.is_full_sha(sha)
        fetched = False
        if named and fetch_records and as_ref:
            fetch = git("fetch", "--quiet", "--no-tags", "origin", ref)
            tip = git("rev-parse", "--verify", "--quiet", "FETCH_HEAD").stdout.strip()
            fetched = fetch.returncode == 0 and tip == sha
        elif named and fetch_records:
            spec = f"+refs/heads/{ref}:{_AUDIT_FETCH_REF}"
            fetch = git("fetch", "--quiet", "--no-tags", "--no-write-fetch-head", "origin", "--", spec)
            tip = git("rev-parse", "--verify", "--quiet", _AUDIT_FETCH_REF).stdout.strip()
            fetched = fetch.returncode == 0 and tip == sha
        present = git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0 if named else False
        actual = git("rev-parse", "--verify", "--quiet", f"{sha}^{{tree}}").stdout.strip() if present else ""
        out.append((ref, sha, (
            (f"{ref} is a record ref with a full SHA", named, f"{ref} {sha}",
             "record/<name> or refs/audit/record/<name>, and 40 hex digits"),
            (f"{ref} is on origin at {sha[:8]}", heads.get(ref) == sha, heads.get(ref, "absent"), sha),
            (f"{ref} fetches", fetched, "fetched" if fetched else "fails", "fetched"),
            (f"{ref} has tree {tree[:8] or 'unrecorded'}", bool(tree) and actual == tree, actual or "absent",
             tree or "unrecorded"),
        )))
    commit_of.cache_clear()
    _ANCESTOR_OF_HEAD.clear()
    return tuple(out)


def _verified_record_refs() -> tuple[str, ...]:
    return tuple(sha for _ref, sha, checks in record_ref_status(RECORD_REFS) if all(ok for _, ok, _, _ in checks))


def record_ref_checks(audit: Audit) -> None:
    for _ref, _sha, checks in record_ref_status(RECORD_REFS):
        for label, ok, shown, value in checks:
            audit.record(ok, RECORD_REFS, label, shown, value)


@functools.cache
def _on_record_ref(commit: str, refs: tuple[str, ...]) -> bool:
    return any(git("merge-base", "--is-ancestor", commit, sha).returncode == 0 for sha in refs)


def harness_reachable(rev: str) -> bool:
    """A printed revision is an ancestor of HEAD or of a verified record ref, a verified alias of an ancestor of
    HEAD, a recorded external pin, or a qualified form."""
    if qualified_external_form(rev) is True or external_pin(rev, abbreviated=True):
        return True
    if revision_reach.is_full_sha((rev or "").strip()) and commit_held(rev.strip()):
        return True
    refs = _verified_record_refs()
    if ancestor_of_head(rev):
        return True
    full = commit_of(rev)
    if full and _on_record_ref(full, refs):
        return True
    rev = (rev or "").strip()
    return len(rev) >= 7 and any(local.startswith(rev) and ancestor_of_head(pushed) for local, pushed in _verified_aliases())


def is_revision_token(token: str) -> bool:
    """Seven to forty hex digits with a letter. A run of letters alone must name a commit, since acceded is a word."""
    if not _SHA_CELL.fullmatch(token) or not re.search(r"[a-f]", token):
        return False
    return bool(re.search(r"\d", token)) or commit_of(token) is not None


def _revision_tokens(cell: str) -> list[str]:
    """Git revisions and URL@hex forms anywhere in a cell. A pure number is a budget, not a commit."""
    text = cell or ""
    found = [match.group(0) for match in _FORM_ATTEMPT.finditer(text)]
    found += [t for t in _SHA_TOKEN.findall(_FORM_ATTEMPT.sub(" ", text)) if is_revision_token(t)]
    return list(dict.fromkeys(found))


def _number_cell(cell: str) -> str | None:
    """The cell as printed when it is a number. Thousands commas, a unit and emphasis are allowed."""
    shown = cell.strip()
    return shown if _NUMBER_CELL.fullmatch(shown.strip("*_`").strip()) else None


def _markdown_tables(text: str):
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|\s*[-:| ]+\s*$", lines[index + 1]):
            header = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows = []
            cursor = index + 2
            while cursor < len(lines) and lines[cursor].startswith("|"):
                cells = [cell.strip() for cell in lines[cursor].strip().strip("|").split("|")]
                rows.append((cursor + 1, cells))
                cursor += 1
            yield header, rows
            index = cursor
            continue
        index += 1


def _table_revision_flags(rel: str, header: list[str], rows: list[tuple[int, list[str]]],
                          forms: list[tuple[str, str]]):
    """Numbers on a table row that names a revision. None when the table names none."""
    rev_cols = {idx for idx, name in enumerate(header) if _REVISION_HEADER.search(name)}
    for _line, cells in rows:
        for idx, cell in enumerate(cells):
            if _revision_tokens(cell):
                rev_cols.add(idx)
    if not rev_cols:
        return None
    flagged = []
    for line_no, cells in rows:
        where = f"{rel}:{line_no}"
        bad = []
        for idx in sorted(rev_cols):
            if idx >= len(cells):
                continue
            for token in _revision_tokens(cells[idx]):
                if qualified_external_form(token) is not None:
                    forms.append((where, token))
                if not harness_reachable(token) and token not in bad:
                    bad.append(token)
        if not bad:
            continue
        identity = " ".join(
            cells[idx].strip()
            for idx in range(len(cells))
            if idx not in rev_cols and cells[idx].strip()
            and (idx >= len(header) or not _RESULT_HEADER.search(header[idx]))
        )
        for idx, cell in enumerate(cells):
            if idx in rev_cols or idx >= len(header) or not _RESULT_HEADER.search(header[idx]):
                continue
            shown = _number_cell(cell)
            if shown is None:
                continue
            number = f"{identity} {header[idx].strip()}={shown}".strip()
            flagged.extend((where, sha, number) for sha in bad)
    return flagged


def _key(text: str) -> str:
    """A name or value compared without case, spaces or punctuation: LJ38 is lj38."""
    return re.sub(r"[^0-9a-z]", "", (text or "").lower())


def _named_csvs(rel: str, text: str) -> dict[Path, bool]:
    """The CSVs a text names, from the README's directory or the root, each with whether it sits beside the README."""
    directory = (ROOT / rel).parent
    found: dict[Path, bool] = {}
    for token in re.findall(r"(?<![\w./-])([\w./-]*[\w-]\.csv)(?![\w.-])", text):
        for base in (directory, ROOT):
            path = (base / token).resolve()
            if path.is_file() and path.is_relative_to(ROOT.resolve()):
                found.setdefault(path, path.parent == directory.resolve())
                break
    return found


def _backing_csvs(rel: str, text: str, section: str) -> list[tuple[Path, bool]]:
    """The CSVs that back a table: those its section names.

    A README that names no CSV anywhere is backed by every CSV beside it.
    A section that names none, in a README that names some, has no CSV.
    """
    found = _named_csvs(rel, section)
    if not found and not _named_csvs(rel, text):
        found = {path.resolve(): True for path in (ROOT / rel).parent.glob("*.csv")}
    return sorted(found.items())


def _csv_backed_flags(rel: str, text: str, header: list[str], rows: list[tuple[int, list[str]]], section: str):
    """A result table backed by a CSV, joined on every column the two share.

    A system that the CSV lacks is not its row. Another key value that the
    CSV lacks, such as an arm named in prose, falls back to the whole file.
    A table that shares no key column is joined only to the CSVs beside it.
    """
    if not any(_RUN_HEADER.search(name) for name in header):
        return []
    flagged = []
    for csv_path, beside in _backing_csvs(rel, text, section):
        with csv_path.open(newline="") as handle:
            data = list(csv.DictReader(line for line in handle if line.strip() and not line.startswith("#")))
        if not data:
            continue
        fields = [name for name in _REVISION_FIELDS if name in data[0]]
        if not fields:
            continue
        columns = {_key(c): c for c in data[0] if c not in _REVISION_FIELDS}
        keys = [(idx, columns[_key(name)]) for idx, name in enumerate(header)
                if _key(name) in columns and not _RESULT_HEADER.search(name)]
        if not keys and not beside:
            continue
        arm_key = "arm" if "arm" in data[0] else "code" if "code" in data[0] else ""
        known = {row[arm_key] for row in data if arm_key and row.get(arm_key)}
        arm_names = {arm.lower() for arm in known}
        last: dict[int, str] = {}
        for line_no, cells in rows:
            group = data
            for idx, column in keys:
                value = cells[idx].strip() if idx < len(cells) else ""
                value = value or last.get(idx, "")
                last[idx] = value
                if not value:
                    continue
                matched = [row for row in group if _key(row.get(column)) == _key(value)]
                if matched:
                    group = matched
                elif _key(column) == "system":
                    group = []
                    break
            if not group:
                continue
            identity = " ".join(cells[idx].strip() for idx, _ in keys if idx < len(cells) and cells[idx].strip())
            for idx, name in enumerate(header):
                if idx >= len(cells):
                    continue
                shown = _number_cell(cells[idx])
                if shown is None:
                    continue
                if name.lower() in arm_names:
                    arms = {arm for arm in known if arm.lower() == name.lower()}
                elif any(arm.lower() in name.lower() for arm in known):
                    arms = {arm for arm in known if arm.lower() in name.lower()}
                elif _RESULT_HEADER.search(name):
                    arms = set(known)
                else:
                    continue
                bad = []
                for row in group:
                    if arm_key and row.get(arm_key) not in arms:
                        continue
                    for field in fields:
                        for token in _revision_tokens(row.get(field) or ""):
                            if not harness_reachable(token) and token not in bad:
                                bad.append(token)
                for sha in bad:
                    flagged.append((f"{rel}:{line_no}", sha, f"{identity} {name.strip()}={shown}".strip()))
    return flagged


def _prose_measures(line: str) -> list[str]:
    """Measured figures stated with their revision. A method sentence is not a result."""
    numbers: list[str] = []
    if re.search(r"median calls|at a median of|median of", line):
        numbers += re.findall(r"(?:from|to) (\d+)\b(?!\.)", line)
        numbers += re.findall(r"median of (-?\d+(?:\.\d+)?)", line)
        numbers += re.findall(r"\d+ of \d+", line)
    if re.search(r"Mann-Whitney p \d", line):
        numbers += re.findall(r"Mann-Whitney p (\d+(?:\.\d+)?)", line)
    if re.search(r"hits \d+ of \d+", line):
        numbers += re.findall(r"hits (\d+ of \d+)", line)
    if re.search(r"ends at|end between", line):
        for match in re.finditer(r"-?\d+\.\d+", line):
            prefix = line[max(0, match.start() - 16):match.start()]
            if re.search(r"against\s+$", prefix):
                continue
            numbers.append(match.group(0))
    if re.search(r"\bwall is\b", line):
        numbers += re.findall(r"\bwall is (-?\d+(?:\.\d+)?)", line)
    found: list[str] = []
    for number in numbers:
        if number in {"0.05", "0.050"} or number in found:
            continue
        found.append(number)
    return found


def _printed_readme_numbers(audit: Audit, forms: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Each printed README number whose revision is not an ancestor and not a qualified form."""
    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(where: str, revision: str, number: str) -> None:
        item = (where, revision, number)
        if item not in seen:
            seen.add(item)
            found.append(item)

    for path in sorted((ROOT / "bench").rglob("README.md")):
        rel = path.relative_to(ROOT).as_posix()
        text = audit.text(rel)
        table_lines: set[int] = set()
        for header, rows in _markdown_tables(text):
            table_lines.update(line_no for line_no, _cells in rows)
            named = _table_revision_flags(rel, header, rows, forms)
            if named is not None:
                for item in named:
                    add(*item)
                continue
            for item in _csv_backed_flags(rel, text, header, rows, _section(text, rows[0][0] if rows else 1)):
                add(*item)
        for paragraph in _prose_paragraphs(text, table_lines):
            for line_no, line in paragraph:
                for match in _FORM_ATTEMPT.finditer(line):
                    forms.append((f"{rel}:{line_no}", match.group(0)))
            carried: list[str] = []
            for line_no, sentence in _paragraph_sentences(paragraph):
                named = [t for t in _SHA_TOKEN.findall(sentence) if is_revision_token(t)]
                if named:
                    carried = list(dict.fromkeys(t for t in named if not harness_reachable(t)))
                if not carried:
                    continue
                for number in _prose_measures(sentence):
                    if any(number in sha for sha in carried):
                        continue
                    for sha in carried:
                        add(f"{rel}:{line_no}", sha, number)
    return found


def _section(text: str, line_no: int) -> str:
    """The lines from the heading above line_no to the next heading."""
    lines = text.splitlines()
    heads = [i for i, line in enumerate(lines) if line.startswith("#")]
    start = max((i for i in heads if i < line_no - 1), default=0)
    end = min((i for i in heads if i >= line_no - 1), default=len(lines))
    return "\n".join(lines[start:end])


def _prose_paragraphs(text: str, skip: set[int]):
    """Runs of prose lines as (line number, line). A blank line, heading, fence, table row or list item starts a new run."""
    paragraph: list[tuple[int, str]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        breaks = not stripped or line_no in skip or line.startswith("|") or stripped.startswith(("#", "```"))
        if breaks or re.match(r"\s*(?:[-*+]|\d+[.)])\s", line):
            if paragraph:
                yield paragraph
            paragraph = []
            if breaks:
                continue
        paragraph.append((line_no, line))
    if paragraph:
        yield paragraph


def _paragraph_sentences(paragraph: list[tuple[int, str]]):
    """The sentences of a paragraph, each with the line it starts on. Qualified forms are blanked first."""
    joined, starts = "", []
    for line_no, line in paragraph:
        starts.append((len(joined), line_no))
        joined += _FORM_ATTEMPT.sub(" ", line.strip()) + " "
    at = 0
    for piece in re.split(r"(?<=\.)\s+", joined):
        if not piece.strip():
            continue
        start = joined.index(piece, at)
        at = start + len(piece)
        yield max(line_no for offset, line_no in starts if offset <= start), piece


SENTENCE_END = re.compile(r"[.?!](?![\d])")


def _sentences(text: str) -> list[str]:
    prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", text, flags=re.S)
    flat, found, start = collapse(prose), [], 0
    for end in [match.end() for match in SENTENCE_END.finditer(flat)] + [len(flat)]:
        sentence, start = flat[start:end].strip(), end
        if sentence:
            found.append(sentence)
    return found


def _check_external_source(audit: Audit, path: str) -> None:
    """The i-PI citation is a repo URL and a full SHA. A bare or short SHA fails."""
    text = audit.text(path)
    attempts = _FORM_ATTEMPT.findall(text)
    if not attempts:
        audit.record(False, path, "qualified external revision", "missing", "repo URL @ full SHA")
        return
    for token in attempts:
        audit.record(qualified_external_form(token) is True, path, "qualified external revision", token, "repo URL @ full SHA")
    stripped = _FORM_ATTEMPT.sub(" ", text)
    for _url, sha in sorted(external_pins()):
        for bare in re.findall(rf"(?<![0-9a-f])({sha[:7]}[0-9a-f]*)(?![0-9a-f])", stripped):
            if sha.startswith(bare):
                audit.record(False, path, "qualified external revision", bare, "repo URL @ full SHA")


def printed_harnesses(audit: Audit) -> None:
    """Fail when a printed number rests on a revision that is not an ancestor of HEAD or of a verified record ref.

    A qualified external revision is a repository URL plus a full SHA. Each one is
    checked. A short SHA in that position fails. A README row listed in
    ``UNVERIFIABLE`` on the flagged revision passes with its reason shown.
    """
    record_ref_checks(audit)
    forms: list[tuple[str, str]] = []
    flagged = _printed_readme_numbers(audit, forms)
    listed = {(path, row, revision): why for path, row, revisions, why in UNVERIFIABLE for revision in revisions}
    matched: set[tuple[str, str, str]] = set()
    for where, revision, number in flagged:
        path, _, line = where.rpartition(":")
        key = (path, audit.text(path).splitlines()[int(line) - 1].strip(), revision[:12])
        if key in listed:
            matched.add(key)
        audit.record(key in listed, where, "a printed number comes from a reachable revision",
                     f"unverifiable: {listed[key]}" if key in listed else "ancestor or qualified external revision",
                     f"{revision[:12]} {number}")
    for path, row, revision in sorted(set(listed) - matched):
        audit.record(False, path, "an unverifiable row is still printed and flagged", row, revision)
    for where, token in forms:
        audit.record(qualified_external_form(token) is True, where, "qualified external revision",
                     token, "repo URL @ full SHA")
    for path in EXTERNAL_FORM_SOURCES:
        _check_external_source(audit, path)
    if not flagged:
        audit.record(True, "bench", "a printed number comes from a reachable revision",
                     "ancestor or qualified external revision", "ancestor or qualified external revision")


def figure_source_scripts(audit: Audit, root: Path = ROOT) -> None:
    """Every effective receipt names the sha256 of its current source."""
    receipt = "paper/jctc/FIGURE-SOURCE.tsv"
    with (root / receipt).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    updates_path = root / "paper/jctc/FIGURE-SOURCE-updates.tsv"
    if updates_path.is_file():
        by_key = {(row["path"], row["kind"]): row for row in rows}
        with updates_path.open(newline="") as handle:
            for update in csv.DictReader(handle, delimiter="\t"):
                key = (update["path"], update["kind"])
                if update["status"] == "superseded":
                    by_key.pop(key, None)
                elif update["status"] == "active":
                    by_key[key] = {
                        field: update[field]
                        for field in ("path", "kind", "source_sha256")
                    }
                else:
                    audit.record(False, str(updates_path.relative_to(root)),
                                 f"{key} update status", "active or superseded",
                                 update["status"])
        rows = list(by_key.values())
    for row in rows:
        path = row["path"]
        source = root / path
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else "missing"
        audit.record(digest == row["source_sha256"], receipt,
                     f"{path} current source sha256",
                     row["source_sha256"], digest)
    if root == ROOT:
        audit.cell(receipt, "effective figure data rows", "276",
                   float(sum(row["kind"] == "data" for row in rows)))


def bibliography_keys(audit: Audit) -> None:
    """Require every citation in the JCTC TeX sources to resolve in a bibliography the sources load."""
    cited: set[str] = set()
    loaded: set[str] = set()
    for source in sorted((ROOT / "paper/jctc").glob("*.tex")):
        path = source.relative_to(ROOT).as_posix()
        text = re.sub(r"(?m)(?<!\\)%.*$", "", audit.text(path))
        for names in re.findall(r"\\bibliography\{([^}]*)\}", text):
            loaded.update(f"paper/jctc/{name.strip()}.bib" for name in names.split(","))
        for group in re.findall(
            r"\\(?:[Cc]ite(?:alp|alt|author|num|p|t|yearpar|year)?|nocite)\*?\s*(?:\[[^\]]*\]\s*)*\{([^{}]+)\}",
            text,
            re.DOTALL,
        ):
            cited.update(
                key.strip()
                for key in group.split(",")
                if key.strip() and key.strip() != "*"
            )

    available = {
        key
        for name in sorted(loaded | {"paper/jctc/refs.bib"})
        for key in re.findall(r"(?mi)^@\w+\s*\{\s*([^,\s]+)\s*,", audit.text(name))
    }
    missing = sorted(cited - available)
    audit.record(
        not missing,
        "paper/jctc/refs.bib",
        "every \\cite key resolves in the paper bibliography",
        f"{len(cited)} cited keys resolve",
        ", ".join(missing) if missing else f"{len(cited)} cited keys resolve",
    )


CONVERSION_LITERAL = re.compile(
    r"(?<![\d.])(?:27\.211_?[0-9]|0\.529_?1[0-9_]*|1\.889_?7[0-9_]*|4\.184(?![\d]))"
)


def conversion_literals(audit: Audit) -> None:
    """Hartree, Bohr, and kilocalorie conversions come from a library, never a typed literal."""
    listed = git("ls-files", "src", "examples", "tests", "scripts").stdout.splitlines()
    for path in sorted(name for name in listed if name.endswith((".rs", ".py", ".md"))):
        found = sorted({match.group(0) for match in CONVERSION_LITERAL.finditer(audit.text(path))})
        audit.record(not found, path, "no typed Hartree, Bohr, or kilocalorie conversion literal",
                     "none", " ".join(found) or "none")


def own_references(audit: Audit) -> None:
    """Every \\ref of a document names a label of that document."""
    labels = load_script("cross_labels", "paper/jctc/scripts/cross_labels.py")
    for path, document in ((TEX, "manuscript"), (SI, "si")):
        own = labels.labels(document)
        loose = sorted({m for m in re.findall(r"\\ref\{([^}]*)\}", labels.uncomment(audit.text(path)))
                        if m not in own})
        audit.record(not loose, path, "every \\ref names a label of its own document", "all defined",
                     " ".join(loose) or "all defined")


def build_receipts(audit: Audit) -> None:
    """The cross-document label files and the pdf-length receipt match the sources they were written from."""
    labels = load_script("cross_labels", "paper/jctc/scripts/cross_labels.py")
    stale = set(labels.stale())
    for document in labels.DOCUMENTS:
        name = f"labels-{document}.tex"
        audit.record(name not in stale, f"paper/jctc/{name}", f"cross_labels.py output for {document}.tex",
                     "current", "stale" if name in stale else "current")
    length = load_script("pdf_length", "paper/jctc/scripts/pdf_length.py")
    receipt = length.RECEIPT.relative_to(length.ROOT).as_posix()
    if not length.RECEIPT.is_file():
        audit.record(False, receipt, "pdf_length.py receipt", "present", "missing")
        return
    counts, digests = length.read_receipt(length.RECEIPT)
    current = length.source_digests(length.ROOT)
    for source in sorted(set(current) | set(digests)):
        audit.record(digests.get(source) == current.get(source), receipt, f"sha256 of {source}",
                     digests.get(source, "absent"), current.get(source, "not a source"))
    missing = sorted({"abstract_words", "abstract_numerals", "body_words"} - set(counts))
    if missing:
        audit.record(False, receipt, "pdf_length.py counts", "present", "no " + ", ".join(missing))
        return
    # Over-length text is an editorial decision, not a stale receipt; it is shown without failing.
    found = length.problems(counts)
    if found:
        audit.expect(receipt, "rendered length within the journal limits", "within", "; ".join(found))
    else:
        audit.record(True, receipt, "rendered length within the journal limits", "within", "within")


def selection_tables(audit: Audit) -> None:
    """The arm-selection tables beside the two result files of tab:ablation.

    The depth-posterior README takes the median over hits; the size-tiered
    table counts a miss as the budget."""
    budgets = {38: 400000, 55: 1000000, 75: 4000000, 98: 4000000, 104: 4000000}
    first = audit.rows("bench/depth-posterior-ablation/results.csv")
    second = audit.rows("results/size-tiered-selection.csv")
    for path, head, rows, size_key, calls_key, hits_only in (
            ("bench/depth-posterior-ablation/README.md", "Size | Arm", first, "Size", "force_calls_to_hit", True),
            ("results/size-tiered-selection.md", "Cluster | Arm", second, "Cluster", "calls_to_hit", False)):
        for line, row in md_table(audit, path, head):
            size = int(row[size_key].removeprefix("LJ"))
            if rows is first:
                g = pick(rows, system=f"lj{size}", arm=row["Arm"])
            else:
                g = pick(rows, n=str(size), arm=row["Arm"])
            calls = [float(r[calls_key]) if r["hit"] == "1" else float(budgets[size]) for r in g
                     if r["hit"] == "1" or not hits_only]
            where = f"{path}:{line} lj{size} {row['Arm']}"
            audit.cell(where, "hits", row["Hits"], frac(g))
            audit.cell(where, "median calls", row["Median calls"], median(calls))
            audit.cell(where, "IQR", row["IQR"], iqr(calls))
            audit.cell(where, "wall sum", row["Wall sum (s)"], sum(float(r["wall_s"]) for r in g))

def gmin_panel_drivers(audit: Audit) -> None:
    """The driver settings of the GMIN panel, read from the driver sources."""
    metal = audit.text("examples/h2h_metal.rs")
    morse = audit.text("examples/h2h_morse.rs")
    stride = re.search(r"cfg\.basin_stride = .*?\.unwrap_or\((\d+)\)", metal, re.S)
    audit.cell("examples/h2h_metal.rs", "basin stride", "8", float(stride.group(1)) if stride else None)
    copper = re.search(r'"au55" => \(\s*55,\s*[-\d.]+,\s*[\d_]+,\s*Kind::CopperGold \{ n_copper: (\d+) \}', metal)
    audit.cell("examples/h2h_metal.rs", "Au55 copper atoms", "0", float(copper.group(1)) if copper else None)
    audit.record("second_neighbour_energy() <" not in morse, "examples/h2h_morse.rs",
                 "long-ranged threshold gate is removed", "absent",
                 "absent" if "second_neighbour_energy() <" not in morse else "present")

TEMPLATE_SWITCHES = ("prior_reseed", "prior_growth", "prior_core", "prior_motif")


def reads_switch(revision: str, name: str) -> bool:
    """Some file under src at the revision reads ``cfg.<name>`` as a whole word.

    git grep -E leaves the word boundary to the platform regex library, and
    the BSD one has no \\b. The pattern is fixed text and the match is made here."""
    hits = git_bytes("grep", "-a", "-h", "-F", f"cfg.{name}", revision, "--", "src").decode(errors="replace")
    word = re.compile(rf"cfg\.{name}\b")
    return any(word.search(line) for line in hits.splitlines())

def prior_switches(audit: Audit) -> None:
    """The current method exposes only prior_soap and rejects retired controls."""
    pattern = "|".join(rf"cfg\.{name}\b" for name in TEMPLATE_SWITCHES)
    current = [str(path.relative_to(ROOT)) for path in (ROOT / "src").rglob("*.rs")
               if re.search(pattern, path.read_text(encoding="utf-8"))]
    audit.record(not current, "src", "no current source reads the four template switches", "none",
                 " ".join(current) or "none")
    config = audit.text("src/methods/cluster_hopping/config.rs")
    for name in TEMPLATE_SWITCHES:
        audit.record(f'"{name}"' in config, "src/methods/cluster_hopping/config.rs",
                     f"Config rejects retired {name}", name,
                     name if f'"{name}"' in config else "absent")
    removed = git("merge-base", "--is-ancestor", "1c1b1238", "HEAD").returncode == 0
    audit.record(removed, PAPER, "the template removal is in this history", "1c1b1238", "1c1b1238" if removed else "absent")
    for revision in ("c2de652d", "e3ed7dad", "9a30f319", "beb89163"):
        found = sorted({name for name in TEMPLATE_SWITCHES if reads_switch(revision, name)})
        audit.record(len(found) == len(TEMPLATE_SWITCHES), PAPER, f"revision {revision} reads the template switches",
                     " ".join(sorted(TEMPLATE_SWITCHES)), " ".join(found) or "none")
    has = git("grep", "-q", "prior_core", "a24ba760", "--", "src").returncode == 0
    audit.record(not has, PAPER, "revision a24ba760 predates the switches", "none", "prior_core" if has else "none")

def method_definitions(audit: Audit) -> None:
    """The constants of the quench, the climb, and the descriptor moves, as the source sets them."""
    sources = {
        "src/methods/warm_lbfgs.rs": ("max_pairs: 8,", "gtol: 1e-6,", "armijo: 1e-4,", "curvature: 0.9,",
                                      "max_line_evals: 20,", "self.inner.norm = GradNorm::Infinity;"),
        "src/methods/cluster_hopping/config.rs": ("screen_steps: 25,", "cfg.relax_steps = 1000;",
                                                 "escape_lanczos_steps: 16,", "escape_overshoot: 1.5,",
                                                 "escape_max_climb: 24,", "escape_stall_patience: 5_000,"),
        "src/methods/activation.rs": ("perp_steps: 3,", "perp_rate: 0.02,", "refresh: 3,"),
        "src/methods/cluster_hopping.rs": ("const FINISH_ROUNDS: usize = 8;", "const STEEP_WELL_PER_ATOM: f64 = 64.0;",
                                           "factor.sqrt().clamp(1.0, 16.0)", "1e-10 * (1.0 + out.0.abs())"),
        "src/soap.rs": ("const FIVEFOLD_AXIS: f64 = 1.40;", "const FIVEFOLD_GATE: f64 = 1.15;",
                        "const FIVEFOLD_MIN_AXES: usize = 2;", "const PENTAGON_CAP: f64 = 0.75;",
                        "const RECON_LO: f64 = 0.89;", "const RECON_HI: f64 = 1.00;", "const NU3_DEFECT: f64 = 0.05;",
                        "concentrate_residual(&mut dr, 5);", "tikhonov_jtj(&j, dp.view(), 1e-3)",
                        "0.5 * (1.0 + libm::cos(PI * r / rcut))", "libm::exp(-0.5 * (r / (rcut / 3.0)).powi(2))"),
        "src/methods/cluster_hopping/moves.rs": ("n_max: 3,\n                    l_max: 6,",
                                                "crate::soap::step_away_fivefold(x, *rmsd, rng)",
                                                "crate::soap::step_away_fivefold_measured(x, *rmsd)"),
    }
    for path, fragments in sources.items():
        text = audit.text(path)
        for fragment in fragments:
            found = fragment.replace("\\n", "\n") in text
            audit.record(found, path, "a constant of the method definitions", fragment, fragment if found else "absent")
    soap = audit.text("src/soap.rs")
    far = re.search(r"for &\(_, v\) in far\.iter\(\)\.take\((\d+)\.min\(n\)\)", soap)
    audit.cell("src/soap.rs", "farthest atoms that seed fivefold axes", "12", float(far.group(1)) if far else float("nan"))

def au55_superseded(audit: Audit) -> None:
    """Each row of rows.csv that rows-for-cluster.csv replaces is listed, and no other row is."""
    old = audit.rows("bench/ase-au55/rows.csv")
    new = {
        (r["arm"], r["seed"]): r
        for r in audit.rows("bench/ase-au55/rows-for-cluster.csv")
        if r["code_revision"].startswith("fc1af066")
    }
    listed = audit.rows("bench/ase-au55/superseded.csv")
    want = {(r["system"], r["arm"], r["seed"], r["code_revision"], "bench/ase-au55/rows-for-cluster.csv",
             new[(r["arm"], r["seed"])]["code_revision"]) for r in old
            if r["arm"] != "recommended" and not r["code_revision"].startswith("1ce3ab64")}
    got = {(r["system"], r["arm"], r["seed"], r["code_revision"], r["superseded_by_file"], r["superseded_by_revision"])
           for r in listed if r["host"] == "bench"}
    audit.record(got == want, "bench/ase-au55/superseded.csv", "the superseded rows are the replaced ones",
                 f"{len(want)} rows", f"{len(got)} rows")


def run(report: bool = False, write_map: bool = False, pre_submission: bool = False) -> int:
    problems = unauditable()
    if problems:
        for problem in problems:
            print(f"paper_audit: cannot check revisions: {problem}", file=sys.stderr)
        return 2
    audit = Audit()
    for check in (
        paired_statistics, manuscript_gmin, withdrawn_benchmarks,
        supporting_information, release, paper_0120,
        gad_published, openconf, conf_miss4, head_to_head, sc96_78_howto, stale_claims,
        tip4p_panel, covalent_panels, covalent_gfn2, printed_harnesses, ase_gofee,
        lj_large, bench_prose, manuscript_ase, manuscript_knockouts,
        manuscript_priors, panel_references, generated_tables,
        supporting_numbers, manuscript_new, manuscript_followups, cover_letter,
        science_claims, rigid_stall, presets_and_templates, no_anneal_restart,
        hexamer_provenance, template_era, outside_session, morse_g1, gmin_sources,
        g3_session, no_move_choice_claim,
        reproduction_records, sutton_chen, crest_equal, crest_equal_bench,
        append_only_rows, tip4p_unscored, tip4p_r2_rows, pending_text, citation_metadata,
                  figure_source_scripts, bibliography_keys, conversion_literals,
                  own_references, build_receipts, selection_tables, gmin_panel_drivers, prior_switches,
                  method_definitions, au55_superseded, cap_turn_campaign):
        check(audit)
    if write_map:
        (ROOT / REVISION_MAP).write_text(revision_map_text(revision_map_rows(audit)))
    revision_map(audit)
    rerun_claims(audit)
    rerun_check(audit, pre_submission)
    todo_check(audit, pre_submission)
    for path, fragment, _why in UNTRACED:
        audit.said(path, fragment)
    gaps = provenance(audit)
    all_revision_provenance(audit)
    for path, why in sorted(KNOWN_GAPS.items()):
        audit.record(any(word in why for word in ("unrecorded", "revision recorded", "is not in this repository", "is not on a pushed branch")), path,
                     "a known gap names the revision or says it is unrecorded", "revision or unrecorded", why)
    for path in sorted(set(gaps) | set(KNOWN_GAPS)):
        listed = path in KNOWN_GAPS
        audit.record(listed and path in gaps and KNOWN_GAPS[path] == gaps[path], path, "provenance",
                     KNOWN_GAPS.get(path, "not listed"), gaps.get(path, "no gap"))
    if report:
        for status, where, label, shown, value in audit.results:
            print(f"{status:5}  {where}  {label}  shown={shown}  computed={value}")
        for path, fragment, why in UNTRACED:
            print(f"UNTRACED  {path}  {fragment!r}  {why}")
        for path, row, revisions, why in UNVERIFIABLE:
            print(f"UNVERIFIABLE  {path}  {row!r}  {' '.join(revisions)}  {why}")
    drift = audit.drifts()
    expected = audit.expected()
    for status, where, label, shown, value in drift:
        print(f"{status}  {where}  {label}  shown={shown}  computed={value}", file=sys.stderr)
    for status, where, label, shown, value in expected:
        print(f"{status}  {where}  {label}  shown={shown}  computed={value}", file=sys.stderr)
    print(f"{len(audit.results)} checks, {len(drift)} drifted, {len(expected)} expected, {len(audit.read)} row files")
    return 1 if drift else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="print every check, not only the drifted ones")
    parser.add_argument("--write-revision-map", action="store_true", help=f"rewrite {REVISION_MAP} from the rows")
    parser.add_argument("--pre-submission", action="store_true",
                        help="fail while any [RERUN: item N] tag or [TODO: Gn] marker remains")
    parser.add_argument("--fetch-records", action="store_true",
                        help="fetch record branches into refs/audit/record/fetched, and refs/audit/record/* by name")
    args = parser.parse_args(argv)
    global _FETCH_RECORDS
    _FETCH_RECORDS = args.fetch_records
    _record_ref_status.cache_clear()
    origin_record_heads.cache_clear()
    return run(args.report, args.write_revision_map, args.pre_submission)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
