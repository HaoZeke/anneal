# Copyright (c) anneal developers
"""A rescore comparison lists the hit cells that moved and nothing else."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "openconf_h2h"))

import pytest  # noqa: E402
import rescore_diff as rd  # noqa: E402

COLUMNS = ["molecule", "arm", "seed", "hit", "hit_mirror", "calls_to_hit", "delta_min_kcal", "candidates", "invalid", "code_revision"]


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"calls_to_hit": "", "delta_min_kcal": "0.1", "candidates": "3", "invalid": "0",
                             "code_revision": "x", **row})


def test_a_moved_hit_is_listed_and_a_moved_revision_is_not(tmp_path):
    base = [
        {"molecule": "m", "arm": "anneal", "seed": "1", "hit": "1", "hit_mirror": "1"},
        {"molecule": "m", "arm": "anneal", "seed": "2", "hit": "0", "hit_mirror": "1"},
        {"molecule": "m", "arm": "develop", "seed": "1", "hit": "0", "hit_mirror": "0"},
    ]
    moved = [dict(r) for r in base]
    moved[1]["hit_mirror"] = "0"
    for r in moved:
        r["code_revision"] = "y"
    write(tmp_path / "a.csv", base)
    write(tmp_path / "b.csv", moved)
    before, after = rd.read(tmp_path / "a.csv"), rd.read(tmp_path / "b.csv")
    assert rd.differing(before, after) == [("m", "anneal", "2", "hit_mirror", "1", "0")]
    out = rd.table(before, after)
    assert [(r["arm"], r["hit_mirror_before"], r["hit_mirror_after"], r["cells_differing"]) for r in out] == [
        ("anneal", 2, 1, 1),
        ("develop", 0, 0, 0),
    ]
    assert rd.differing(before, before) == []


def test_runs_in_one_file_only_are_an_error(tmp_path):
    row = {"molecule": "m", "arm": "anneal", "seed": "1", "hit": "1", "hit_mirror": "1"}
    write(tmp_path / "a.csv", [row])
    write(tmp_path / "b.csv", [row, {**row, "seed": "2"}])
    with pytest.raises(SystemExit):
        rd.differing(rd.read(tmp_path / "a.csv"), rd.read(tmp_path / "b.csv"))
