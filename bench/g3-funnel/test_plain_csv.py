# Copyright (c) anneal developers
"""Reconcile the plain G3 tables with the committed panel tables."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FUNNELS = ("icosahedral", "decahedral", "fcc", "other")


def read(name):
    with (HERE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_run_and_miss_totals():
    rows = read("rows.csv")
    misses = read("funnel_misses.csv")
    assert len(rows) == 120
    assert Counter(row["status"] for row in rows) == {"ok": 120}
    hits = Counter(
        (row["method"], row["system"])
        for row in rows
        if row["hit"] == "1"
    )
    assert hits == Counter(
        {
            ("anneal", "lj38"): 12,
            ("anneal", "lj75"): 5,
            ("anneal", "lj98"): 7,
            ("anneal", "lj150"): 8,
            ("anneal", "m80r6"): 12,
        }
    )
    assert sum(hits.values()) == 44
    expected_misses = {
        (row["method"], row["system"], row["seed"])
        for row in rows
        if row["hit"] != "1"
    }
    actual_misses = {
        (row["method"], row["system"], row["seed"]) for row in misses
    }
    assert len(misses) == 76
    assert expected_misses == actual_misses
    assert Counter(row["method"] for row in misses) == {
        "anneal": 16,
        "ase_mh": 60,
    }
    classified = {
        (row["method"], row["system"], row["seed"]): row
        for row in read("misses.csv")
    }
    for miss in misses:
        key = miss["method"], miss["system"], miss["seed"]
        source = classified[key]
        assert miss["final_energy"] == source["best"]
        assert miss["point_group"] == source["point_group"]
        assert miss["cna_class"] == source["cna_class"]
        assert miss["miss_class"] == source["miss_class"]
        assert miss["structure"] == source["structure"]
        assert miss["con_path"] == f"misses/{source['con']}"


def test_run_occupancy_reconciles_with_rows():
    rows = {
        (row["method"], row["system"], row["seed"]): row
        for row in read("rows.csv")
    }
    occupancy = read("funnel_occupancy_runs.csv")
    assert len(occupancy) == len(rows)
    for item in occupancy:
        key = item["method"], item["system"], item["seed"]
        assert int(item["accepted_minima_total"]) == int(rows[key]["minima"])
        assert int(item["force_calls_total"]) == int(rows[key]["charged"])
        assert int(item["accepted_minima_total"]) == sum(
            int(item[f"accepted_minima_{funnel}"]) for funnel in FUNNELS
        )
        assert int(item["force_calls_total"]) == int(
            item["force_calls_before_first"]
        ) + sum(int(item[f"force_calls_{funnel}"]) for funnel in FUNNELS)
        assert int(item["observed_hop_steps_total"]) == int(
            item["observed_hop_steps_before_first"]
        ) + sum(
            int(item[f"observed_hop_steps_{funnel}"])
            for funnel in FUNNELS
        )


def test_occupancy_summary_reconciles_with_runs_and_prior_table():
    runs = read("funnel_occupancy_runs.csv")
    summary = read("funnel_occupancy_summary.csv")
    numeric = [field for field in summary[0] if field not in {"method", "system"}]
    totals = defaultdict(Counter)
    for row in runs:
        key = row["method"], row["system"]
        for field in numeric:
            totals[key][field] += int(row[field])
    for row in summary:
        key = row["method"], row["system"]
        assert {field: int(row[field]) for field in numeric} == totals[key]
        assert int(row["runs"]) == 12
    prior = {
        (row["method"], row["system"], row["funnel"]): row
        for row in read("occupancy.csv")
    }
    for row in summary:
        key = row["method"], row["system"]
        for funnel in FUNNELS:
            old = prior[key + (funnel,)]
            assert int(row[f"accepted_minima_{funnel}"]) == int(old["minima"])
            assert int(row[f"force_calls_{funnel}"]) == int(old["calls"])
        before = prior[key + ("before_first",)]
        assert int(row["force_calls_before_first"]) == int(before["calls"])


def test_transition_summaries_reconcile_with_runs_and_readme_totals():
    runs = read("funnel_transitions_runs.csv")
    summary = read("funnel_transitions_summary.csv")
    row_keys = {
        (row["method"], row["system"], row["seed"]) for row in read("rows.csv")
    }
    assert {
        (row["method"], row["system"], row["seed"]) for row in runs
    } == row_keys
    assert {
        (row["method"], row["system"]) for row in summary
    } == {(method, system) for method, system, _seed in row_keys}
    totals = Counter()
    for row in runs:
        key = (
            row["method"],
            row["system"],
            row["from_funnel"],
            row["to_funnel"],
            row["move"],
        )
        totals[key] += int(row["count"])
    assert totals == Counter(
        {
            (
                row["method"],
                row["system"],
                row["from_funnel"],
                row["to_funnel"],
                row["move"],
            ): int(row["count"])
            for row in summary
        }
    )
    prior = Counter(
        {
            (
                row["method"],
                row["system"],
                row["from"],
                row["to"],
                row["move"],
            ): int(row["count"])
            for row in read("transitions.csv")
        }
    )
    assert totals == prior
    by_panel = Counter()
    for row in summary:
        by_panel[row["method"], row["system"]] += int(row["count"])
    assert by_panel == Counter(
        {
            ("anneal", "lj38"): 1084,
            ("anneal", "lj75"): 399,
            ("anneal", "lj98"): 1942,
            ("anneal", "lj150"): 1601,
            ("anneal", "m80r6"): 627,
            ("ase_mh", "lj38"): 11,
            ("ase_mh", "lj75"): 7,
            ("ase_mh", "lj98"): 0,
            ("ase_mh", "lj150"): 2,
            ("ase_mh", "m80r6"): 4,
        }
    )
