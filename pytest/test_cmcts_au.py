# Copyright (c) anneal developers
"""Sutton–Chen c-MCTS head-to-head: parameters, budget anchors, and tests."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "scripts" / "cmcts_au" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMPAIGN = load("run_campaign")
SUMMARY = load("summarize")


def test_budgets_are_the_basin_hopping_anchor_in_calls():
    for system, per in CAMPAIGN.CALLS_PER_MINIMISATION.items():
        assert CAMPAIGN.PAPER_BUDGETS[system] == int(CAMPAIGN.BH_MINIMISATIONS * per)
        assert CAMPAIGN.anchor_fields(system)["paper_anchor_calls"] == str(
            int(CAMPAIGN.PAPER_EVALUATIONS * per)
        )


def test_references_are_the_cambridge_10_8_table():
    assert CAMPAIGN.REFERENCES == {
        "au35": -9276.2927,
        "au38": -10117.2454,
        "au55": -14814.5225,
    }
    assert CAMPAIGN.SEEDS == list(range(2000, 2012))
    assert (CAMPAIGN.VENDOR / "UPSTREAM").read_text().strip() == (
        "d6eb071a7404550487654a28ec6181a60af44f05"
    )


def test_fisher_two_sided_matches_a_small_table():
    # 2/2 against 0/2. The two-sided probability of this table or any
    # table at least as uneven, with the margins fixed, is 1/3.
    assert SUMMARY.fisher_two_sided(2, 2, 0, 2) == pytest.approx(1 / 3)


def test_mann_whitney_corrects_ties():
    # One block of three tied 2s. U is 8 against a mean of 4.5, and the
    # tie correction is what scipy's asymptotic test uses.
    assert SUMMARY.mann_whitney([1.0, 2.0, 2.0], [2.0, 3.0, 4.0]) == pytest.approx(
        0.16415972847851523
    )
    assert SUMMARY.mann_whitney([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 1.0


def test_append_keeps_existing_rows():
    path = ROOT / "bench" / "cmcts-au" / "append-test.csv"
    path.unlink(missing_ok=True)
    row = {field: "1" for field in CAMPAIGN.ROW_FIELDS}
    row.update(system="au35", arm="cmcts", seed="2000", code_revision="abc", host="bench")
    CAMPAIGN.append_rows(path, [row])
    CAMPAIGN.append_rows(path, [row])
    stored = CAMPAIGN.read_rows(path)
    path.unlink()
    assert stored == [row]


def test_summary_uses_the_latest_revision():
    rows = []
    for revision, hit in (("old", "0"), ("new", "1")):
        for seed in ("2000", "2001"):
            rows.append(
                {
                    "system": "au35",
                    "arm": "recommended",
                    "seed": seed,
                    "hit": hit,
                    "evals_to_hit": "10",
                    "wall_s": "1",
                    "pot_s": "0.4",
                    "outside_s": "0.6",
                    "code_revision": revision,
                    "calls_per_minimisation": "1225.5",
                    "paper_anchor_calls": "14706000",
                    "bh_anchor_calls": "30637500",
                }
            )
            rows.append(
                {
                    "system": "au35",
                    "arm": "cmcts",
                    "seed": seed,
                    "hit": "0",
                    "evals_to_hit": "30",
                    "wall_s": "2",
                    "pot_s": "1",
                    "outside_s": "1",
                    "code_revision": "tree",
                    "calls_per_minimisation": "1225.5",
                    "paper_anchor_calls": "14706000",
                    "bh_anchor_calls": "30637500",
                }
            )
    lines = SUMMARY.summarise(rows)
    assert lines[0]["anneal_hits"] == "2"
    assert lines[0]["anneal_n"] == "2"
    assert lines[0]["cmcts_hits"] == "0"


def test_dimer_matches_the_reduced_gold_row_to_1e_8():
    binary = ROOT.parent / "tmp" / "anneal-target" / "release" / "examples" / "cmcts_au"
    if not binary.is_file():
        pytest.skip("cmcts_au is not built")
    proc = subprocess.run(
        [str(binary), "energy", "2"],
        input="0 0 0 1 0 0\n",
        text=True,
        capture_output=True,
        check=True,
    )
    energy = float(proc.stdout.strip())
    assert energy == pytest.approx(1.0 - 2.0 * 34.408, abs=1e-8)


def anneal_line(budget: str) -> str:
    return f"recommended,au35,2000,{budget},true,10,10,0.1,-9276.29272778,0.05,0.05\n"


def fake_binary_run(monkeypatch, budget: str) -> dict:
    seen: dict = {}

    def run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout=anneal_line(budget), stderr="")

    monkeypatch.setattr(CAMPAIGN.subprocess, "run", run)
    return seen


def test_a_leaked_budget_override_does_not_reach_the_anneal_binary(monkeypatch):
    monkeypatch.setenv("H2H_BUDGET", "1000")
    seen = fake_binary_run(monkeypatch, str(CAMPAIGN.PAPER_BUDGETS["au35"]))
    row = CAMPAIGN.run_anneal(Path("cmcts_au"), "recommended", "au35", 2000)
    assert "H2H_BUDGET" not in seen["env"]
    assert row["budget"] == str(CAMPAIGN.PAPER_BUDGETS["au35"])


def test_an_anneal_row_with_another_budget_than_the_campaign_is_refused(monkeypatch):
    fake_binary_run(monkeypatch, "1000")
    with pytest.raises(SystemExit):
        CAMPAIGN.run_anneal(Path("cmcts_au"), "recommended", "au35", 2000)


def test_every_arm_names_the_anneal_build_that_ran_it(tmp_path, monkeypatch):
    head = "a" * 40
    monkeypatch.setattr(CAMPAIGN, "anneal_revision", lambda: head)

    def run_one(binary, system, seed, arm):
        return {field: "1" for field in CAMPAIGN.ROW_FIELDS} | {
            "system": system,
            "arm": arm,
            "seed": str(seed),
        }

    monkeypatch.setattr(CAMPAIGN, "run_one", run_one)
    seeds = tmp_path / "seeds.csv"
    for arm in ("recommended", "cmcts"):
        CAMPAIGN.campaign(Path("cmcts_au"), seeds, ("au35", 2000, arm), {arm})
    stored = {row["arm"]: row for row in CAMPAIGN.read_rows(seeds)}
    assert stored["recommended"]["code_revision"] == head
    assert stored["cmcts"]["code_revision"] == CAMPAIGN.PYPHYSTREE_COMMIT
    for row in stored.values():
        assert row["harness_revision"] == head
        assert row["host"]


def test_append_gives_an_older_file_the_new_column_and_keeps_every_row(tmp_path):
    older = [field for field in CAMPAIGN.ROW_FIELDS if field != "harness_revision"]
    path = tmp_path / "seeds.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=older, lineterminator="\n")
        writer.writeheader()
        for seed in ("2000", "2001"):
            writer.writerow({field: "old" for field in older} | {"seed": seed})
    fresh = {field: "new" for field in CAMPAIGN.ROW_FIELDS} | {"seed": "2002"}
    CAMPAIGN.append_rows(path, [fresh])
    stored = CAMPAIGN.read_rows(path)
    assert [row["seed"] for row in stored] == ["2000", "2001", "2002"]
    assert [row["harness_revision"] for row in stored] == ["", "", "new"]
    assert all(None not in row for row in stored)
    assert stored[0]["code_revision"] == "old"
