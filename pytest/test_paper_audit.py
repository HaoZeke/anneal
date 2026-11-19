# Copyright (c) anneal developers
"""Every paper and benchmark-table number checked by scripts/paper_audit.py
matches its committed rows, and the audit notices a drifted cell."""

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("paper_audit", ROOT / "scripts" / "paper_audit.py")
audit_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_mod)
PAPER_CONSTANTS = audit_mod.load_script(
    "paper_constants_test",
    "paper/jctc/scripts/paper_constants.py",
).values()

CHECKS = (
    audit_mod.manuscript_ablation,
    audit_mod.selection_tables,
    audit_mod.manuscript_retry,
    audit_mod.manuscript_gmin,
    audit_mod.manuscript_tip4p,
    audit_mod.manuscript_hexamer,
    audit_mod.manuscript_cross_field,
    audit_mod.supporting_information,
    audit_mod.release,
    audit_mod.paper_0120,
    audit_mod.gad_published,
    audit_mod.openconf,
    audit_mod.head_to_head,
    audit_mod.sc96_78_howto,
    audit_mod.stale_claims,
    audit_mod.tip4p_panel,
    audit_mod.reproduction_records,
    audit_mod.covalent_panels,
    audit_mod.covalent_gfn2,
    audit_mod.ase_gofee,
    audit_mod.lj_large,
    audit_mod.bench_prose,
    audit_mod.manuscript_ase,
    audit_mod.panel_references,
    audit_mod.generated_tables,
    audit_mod.sutton_chen,
    audit_mod.supporting_numbers,
    audit_mod.method_definitions,
    audit_mod.prior_switches,
    audit_mod.jctc_requirements,
    audit_mod.crest_equal,
    audit_mod.crest_equal_bench,
    audit_mod.au55_superseded,
    audit_mod.manuscript_new,
    audit_mod.manuscript_followups,
    audit_mod.manuscript_au55,
    audit_mod.cover_letter,
    audit_mod.rerun_check,
    audit_mod.append_only_rows,
    audit_mod.bibliography_keys,
    audit_mod.revision_map,
)


@pytest.fixture(scope="module")
def audited():
    audit = audit_mod.Audit()
    for check in CHECKS:
        check(audit)
    for path, fragment, _why in audit_mod.UNTRACED:
        audit.said(path, fragment)
    return audit


def test_every_table_cell_matches_its_rows(audited):
    drift = audited.drifts()
    assert not drift, "\n".join(f"{w}  {label}  shown={s}  computed={v}" for _, w, label, s, v in drift)
    assert len(audited.results) > 2500


def test_provenance_gaps_are_exactly_the_listed_ones(audited):
    assert audit_mod.provenance(audited) == audit_mod.KNOWN_GAPS


def test_untraced_numbers_are_still_quoted(audited):
    for path, fragment, _why in audit_mod.UNTRACED:
        assert audit_mod.collapse(fragment) in audit_mod.collapse((ROOT / path).read_text()), (path, fragment)


def test_paper_audit_has_no_drift(capsys):
    assert audit_mod.UNTRACED == []
    assert audit_mod.run() == 0
    drifted = [line for line in capsys.readouterr().err.splitlines() if line.startswith("DRIFT")]
    assert drifted == []


def test_panel_reproduce_commands_and_session_metadata(tmp_path):
    reproduce = (ROOT / "paper/jctc/REPRODUCE.md").read_text()
    commands = (
        ("sutton-chen-n12", "sc_h2h_n12", 1, ""),
        ("tip4p-gmin", "tip4p_h2h", 1, ""),
        ("agox-gofee", "agox_gofee_h2h", 1, ""),
        ("cuh2-fcc-rec", "cuh2_fcc_rec", 1, ""),
    )
    for _campaign, rule, cores, suffix in commands:
        command = f"snakemake -s workflow/Snakefile {rule} --cores {cores} --force{suffix}"
        assert reproduce.count(command) == 1
        assert command in (ROOT / "workflow/Snakefile").read_text()
    assert "crest_equal" not in reproduce
    assert "water_crest_fair" not in reproduce

    out = tmp_path / "sessions.jsonl"
    command = ["snakemake", "-s", "workflow/Snakefile", "tip4p_h2h", "--cores", "1", "--force"]
    proc = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/session_metadata.py"),
            "--output",
            str(out),
            "--campaign",
            "tip4p-gmin",
            "--root",
            str(ROOT),
            "--",
            *command,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    row = json.loads(out.read_text())
    assert row["schema"] == "anneal-session-v2"
    assert set(row["compilers"]) == {"rustc", "cargo", "gfortran", "cc"}
    assert row["python"]["version"].count(".") == 2
    assert set(row["openmp"]) == {"libraries", "omp_num_threads", "settings"}
    assert row["anneal_repro"] is None
    assert row["campaign"] == "tip4p-gmin"
    assert row["command"] == command
    assert row["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    assert row["cpu_affinity"] == sorted(__import__("os").sched_getaffinity(0))
    assert re.fullmatch(r"[0-9a-f]{40}", row["code_revision"])


def test_an_untraced_number_fails_the_audit():
    audit = audit_mod.Audit()
    audit_mod.untraced_check(audit)
    assert [row[1] for row in audit.drifts()] == [path for path, _fragment, _why in audit_mod.UNTRACED]
    assert not any(row[0] == "DRIFT" for row in audit.results if "no committed row backs" not in row[2])


def test_a_drifted_table_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex.replace("& 12/12 & 10615.5 & 3591.25", "& 12/12 & 10616.5 & 3591.25")
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["median calls"]


def test_a_drifted_ase_table_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.SI)
    old = "& 246.5 & 10/12 & 0/12 & 0/12 & 0 & 12 &"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.SI] = tex.replace(old, "& 246.5 & 11/12 & 0/12 & 0/12 & 0 & 12 &")
    audit_mod.manuscript_ase(audit)
    audit_mod.generated_tables(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["recommended hits", "tab:ase is the summariser's block"]


def test_a_drifted_morse_ase_table_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.SI)
    old = "& 54.135 & 0.037 & 0.037 \\\\"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.SI] = tex.replace(old, "& 54.135 & 0.037 & 0.37 \\\\")
    audit_mod.manuscript_ase(audit)
    audit_mod.generated_tables(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["Fisher p against ase_mh", "tab:ase-morse is the summariser's block"]


def test_a_rank_test_in_the_ase_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.SI)
    old = "Every basin-hopping and minima-hopping run of"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.SI] = tex.replace(old, "A Mann--Whitney test is not run. " + old)
    audit_mod.manuscript_ase(audit)
    assert [r[2] for r in audit.drifts()] == ["sec:ase reports Fisher only"]


def test_a_drifted_gmin_panel_summary_cell_is_reported():
    audit = audit_mod.Audit()
    summary = audit.rows("bench/gmin-panel/summary.csv")
    au = next(r for r in summary if r["system"] == "au55")
    au["gmin_hits"] = "0"
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["gmin_hits"]


def test_a_censored_gmin_panel_wall_p_claim_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex.replace("& 0.40 & 0.16 & -- \\\\", "& 0.40 & 0.16 & 0.0073 \\\\")
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["MW p wall"]


def test_an_uncharged_gmin_panel_miss_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows("bench/gmin-panel/rows.csv")
    miss = next(r for r in reversed(rows) if r["system"] == "au55" and r["code"] == "gmin" and r["hit"] == "0")
    miss["charged"] = "50000"
    audit_mod.manuscript_gmin(audit)
    assert "row invariants" in {r[2] for r in audit.drifts()}


def test_a_drifted_sutton_chen_howto_cell_is_reported():
    audit = audit_mod.Audit()
    path = audit_mod.SC96_78_HOWTO
    audit.text_cache[path] = audit.text(path).replace("gmin 20/32 1272371 2187979 -", "gmin 21/32 1272371 2187979 -")
    audit_mod.sc96_78_howto(audit)
    assert [r[2] for r in audit.drifts()] == ["hits"]


SC_N12 = "bench/metal-cluster-h2h/sc-n12/README.md"


def test_an_old_gmin_number_in_a_sentence_is_reported():
    audit = audit_mod.Audit()
    path = audit_mod.SC96_78_HOWTO
    old = "=gmin= hits 20 of 32 seeds, median 1272371 calls to a hit. It misses 12."
    audit.text_cache[path] = audit.text(path).replace(old, "=gmin= hits 0 of 32 seeds. It misses 32.")
    audit_mod.stale_claims(audit, [path])
    assert [(r[2], r[3]) for r in audit.drifts()] == [("old GMIN number", "=gmin= hits 0 of 32 seeds.")]


def test_an_old_gmin_p_value_in_a_new_document_is_reported():
    audit = audit_mod.Audit()
    audit.text_cache["paper/note.md"] = "On 9-6 at 78 atoms iso beats GMIN, p<1e-4.\n"
    audit_mod.stale_claims(audit, ["paper/note.md"])
    assert [r[2] for r in audit.drifts()] == ["old GMIN number"]


def test_a_retired_rerun_sentence_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(SC_N12)
    audit.text_cache[SC_N12] = text + "\nOnly 9-6 at N=38 has been rerun, under harness 9149d1b0.\n"
    audit_mod.stale_claims(audit, [SC_N12])
    assert [r[2] for r in audit.drifts()] == ["retired claim"]


def test_an_old_sutton_chen_gmin_cell_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(SC_N12)
    audit.text_cache[SC_N12] = text.replace("| sc126-38 | gmin | f9d1ac05 | 12/12 |", "| sc126-38 | gmin | f9d1ac05 | 1/12 |")
    audit_mod.head_to_head(audit)
    assert [(r[1].split()[-3:], r[2]) for r in audit.drifts()] == [(["sc126-38", "gmin", "f9d1ac05"], "hits")]


def test_a_dropped_sutton_chen_rerun_row_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(SC_N12)
    audit.text_cache[SC_N12] = text.replace("| sc108-38 | gmin | f9d1ac05 | 0/12 | 400000 | 4.305 |\n", "")
    audit_mod.head_to_head(audit)
    assert [r[2] for r in audit.drifts()] == ["table row sc108-38 gmin f9d1ac05"]


def test_a_sutton_chen_p_value_against_the_old_gmin_rows_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows("bench/metal-cluster-h2h/sc-n12/pvalues.csv")
    row = next(r for r in rows if r["system"] == "sc126-38")
    row["fisher_p"] = "9.61483e-06"
    audit_mod.sutton_chen_pvalues(audit)
    assert [r[2] for r in audit.drifts()] == ["sc126-38 Fisher p"]


def test_a_sutton_chen_p_value_row_without_its_superseded_row_is_reported():
    audit = audit_mod.Audit()
    results = audit.rows("bench/metal-cluster-h2h/sc-n12/results.csv")
    results.remove(next(r for r in results if r["system"] == "sc96-80" and r["arm"] == "anneal"))
    audit_mod.sutton_chen_pvalues(audit)
    assert [r[2] for r in audit.drifts()] == ["sc96-80 names the results.csv row it supersedes, which stays"]


def test_a_rerun_missing_from_the_sutton_chen_readme_is_reported():
    audit = audit_mod.Audit()
    summary = audit.rows("bench/metal-cluster-h2h/sc-n12/results.csv")
    row = dict([r for r in summary if r["system"] == "sc96-55" and r["arm"] == "gmin"][-1])
    summary.append({**row, "harness_revision": "0" * 40})
    audit_mod.head_to_head(audit)
    labels = {r[2] for r in audit.drifts() if r[1] == SC_N12}
    assert {"table row sc96-55 gmin 00000000", "test row sc96-55", "prose"} <= labels


def test_a_rerun_without_its_tests_is_reported():
    audit = audit_mod.Audit()
    summary = audit.rows("bench/metal-cluster-h2h/sc-n12/results.csv")
    row = [r for r in summary if r["system"] == "sc96-55" and r["arm"] == "gmin"][-1]
    row["fisher_p_vs_gmin"] = ""
    audit_mod.head_to_head(audit)
    assert ("Fisher p", "1") in [(r[2], r[3]) for r in audit.drifts()]


def test_a_gmin_row_left_on_the_old_file_is_reported():
    audit = audit_mod.Audit()
    summary = audit.rows("bench/metal-cluster-h2h/sc-n12/results.csv")
    old = next(r["harness_revision"] for r in summary if r["system"] == "sc108-80" and r["arm"] == "anneal")
    summary[:] = [r for r in summary if not (r["system"] == "sc108-80" and r["arm"] == "gmin"
                                             and r["harness_revision"] != old)]
    audit_mod.head_to_head(audit)
    assert "sc108-80 GMIN row from the corrected file" in {r[2] for r in audit.drifts()}


def test_a_gmin_lead_must_be_stated_as_the_rows_give_it():
    audit = audit_mod.Audit()
    text = audit.text(SC_N12)
    audit.text_cache[SC_N12] = text.replace("| 0.113 | tie | GMIN, n.s. | GMIN |", "| 0.113 | tie | GMIN, n.s. | anneal |")
    audit_mod.head_to_head(audit)
    assert [r[2] for r in audit.drifts()] == ["Wall ahead"]


def test_a_drifted_readme_p_value_is_reported():
    audit = audit_mod.Audit()
    path = "bench/openconf-h2h/paired/README.md"
    audit.text_cache[path] = audit.text(path).replace("| 9617 | 4339 | 0.6171 |", "| 9617 | 4339 | 0.6429 |")
    audit_mod.openconf(audit)
    assert [(r[2], r[3]) for r in audit.drifts()] == [("Mann-Whitney p", "0.6429")]


GFN2_README = "bench/covalent-h2h/README.md"


def test_a_drifted_gfn2_paper_cell_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(GFN2_README)
    audit.text_cache[GFN2_README] = text.replace("| 11/12 | 943.5 | 1,006 |", "| 11/12 | 944.5 | 1,006 |")
    audit_mod.covalent_gfn2(audit)
    assert [r[2] for r in audit.drifts()] == ["median calls"]


def test_a_drifted_gfn2_revision_table_cell_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(GFN2_README)
    old = "| Si13 | anneal G, `c1c02d16` | 6/6 | 1,974.5 |"
    assert old in text
    audit.text_cache[GFN2_README] = text.replace(old, "| Si13 | anneal G, `c1c02d16` | 6/6 | 1,975.5 |")
    audit_mod.covalent_gfn2(audit)
    assert [r[2] for r in audit.drifts()] == ["median calls"]


def test_a_gfn2_revision_table_the_audit_cannot_find_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(GFN2_README)
    start = text.rindex("| system | code | hits |", 0, text.index("| C20 | anneal G, `5351f272`"))
    end = text.index("\n", start)
    audit.text_cache[GFN2_README] = text[:start] + text[start:end].replace("best / eV", "best (eV)") + text[end:]
    audit_mod.covalent_gfn2(audit)
    assert sorted((r[1].split()[-2], r[2]) for r in audit.drifts()) == [("C20", "GFN2 table row"),
                                                                        ("Si13", "GFN2 table row")]


def test_a_gfn2_sentence_count_follows_the_rows():
    # The latest CREST hit moves, and no table cell or p value with it.
    audit = audit_mod.Audit()
    rows = audit.rows("bench/covalent-h2h/gfn2/rows.csv")
    row = next(r for r in rows if r["system"] == "si13" and r["code"] == "crest" and r["evals_to_hit"] == "143630")
    row["evals_to_hit"] = "150000"
    audit_mod.covalent_gfn2(audit)
    assert [(r[2], "143,630" in r[3]) for r in audit.drifts()] == [("prose", False)]


def test_an_unnamed_gfn2_false_hit_is_reported():
    audit = audit_mod.Audit()
    audit.rows("bench/covalent-h2h/gfn2/false-hits.csv").clear()
    audit_mod.covalent_gfn2(audit)
    drifted = {r[2] for r in audit.drifts()}
    assert {"hits", "median calls", "best", "GFN2 paper Si10 Mann-Whitney p"} <= drifted


SC_N12_ROWS = ("bench/metal-cluster-h2h/sc-n12/results.csv", "bench/metal-cluster-h2h/sc-n12/seeds.csv")
CONF_FIX4 = "bench/openconf-h2h/conf-fix4/seeds.csv"
ASE_GOFEE = "bench/ase-gofee-h2h/results.csv"


def aliased_audit(tmp_path, monkeypatch, edit, paths=SC_N12_ROWS):
    table = (ROOT / audit_mod.ALIASES).read_text()
    edited = tmp_path / "revision-aliases.csv"
    edited.write_text(edit(table))
    monkeypatch.setattr(audit_mod, "ALIASES", str(edited))
    audit = audit_mod.Audit()
    for path in paths:
        audit.rows(path)
    return audit, audit_mod.provenance(audit)


def test_every_alias_is_verified_and_clears_the_sc_n12_rows(tmp_path, monkeypatch):
    audit, gaps = aliased_audit(tmp_path, monkeypatch, lambda table: table)
    assert not audit.drifts()
    table = [line for line in (ROOT / audit_mod.ALIASES).read_text().splitlines() if line[:1] not in ("#", "l")]
    assert {r[1] for r in audit.results} == {f"{audit_mod.ALIASES} {line[:8]}" for line in table}
    assert gaps == {}


def test_an_alias_off_the_rows_commit_parent_is_held_to_its_paths(tmp_path, monkeypatch):
    audit, gaps = aliased_audit(tmp_path, monkeypatch, lambda t: t.replace(",workflow/Snakefile\n", ",README.md\n"),
                                (CONF_FIX4,))
    assert [r[2] for r in audit.drifts()] == ["fc1ddc5b differs from the rows commit's parent only in README.md",
                                              "unpushed revision 2a5b4553"]
    assert gaps == {CONF_FIX4: "revision 2a5b4553 is not in this repository"}


def test_an_alias_off_the_rows_commit_parent_must_be_its_ancestor(tmp_path, monkeypatch):
    audit, _ = aliased_audit(tmp_path, monkeypatch, lambda t: t.replace("fc1ddc5b6aedcb1bac5cf428ee4ff8d8c894d9cd",
                                                                        "d99354073ae917995172e9a49f7e2694a0848441"),
                             (CONF_FIX4,))
    assert {"rows commit 93c14173 sits on a descendant of d9935407",
            "d9935407 differs from the rows commit's parent only in workflow/Snakefile"} <= {r[2] for r in audit.drifts()}


def listed_audit(tmp_path, monkeypatch, edit, paths):
    table = (ROOT / audit_mod.UNPUSHED_FILE).read_text()
    edited = tmp_path / "unpushed-revisions.csv"
    edited.write_text(edit(table))
    monkeypatch.setattr(audit_mod, "UNPUSHED_FILE", str(edited))
    audit = audit_mod.Audit()
    for path in paths:
        audit.rows(path)
    return audit, audit_mod.provenance(audit)


def test_an_unpushed_revision_needs_a_reason(tmp_path, monkeypatch):
    def edit(table):
        lines = [line for line in table.splitlines() if not line.startswith("bf09f7a2b7a6bd6c")]
        lines = [line for line in lines if not line.startswith("bf09f7a2b7a6bd6359898db675eeafefcd20548b")]
        return "\n".join(lines + ["e09e0487c52b7f4789bc556ee7fe07ead4999824,listed although its alias holds"]) + "\n"

    audit, _ = listed_audit(tmp_path, monkeypatch, edit, (ASE_GOFEE,))
    assert [(r[2], r[3], r[4]) for r in audit.drifts()] == [
        ("unpushed revision bf09f7a2", "not listed", "no pushed equivalent"),
        ("unpushed revision e09e0487", "listed although its alias holds", "resolved as alias of 57735de1"),
    ]


def remote_tip():
    return git("for-each-ref", "--count=1", "--format=%(objectname)", "refs/remotes").stdout.strip()


def synthetic_audit(*rows, revisions=("code_revision", "code_tree")):
    audit = audit_mod.Audit()
    audit.read["synthetic.csv"] = [dict(zip(revisions, row), host="bench") for row in rows]
    return audit


LOST = "0123456789abcdef0123456789abcdef01234567"


def test_a_cited_revision_on_a_remote_ref_is_not_unresolved(audited):
    found = audit_mod.resolve_revisions(audited)
    assert found.unresolved == set(audit_mod.unpushed_reasons())
    assert found.unresolved.isdisjoint(found.pushed | set(found.aliased) | set(found.same_tree))
    assert found.pushed and all(found.index.on_remote(name) for name in found.pushed)


def test_a_listed_revision_that_a_remote_ref_contains_is_reported(tmp_path, monkeypatch):
    tip = remote_tip()
    monkeypatch.setattr(audit_mod, "unpushed_reasons", lambda: {tip: "listed although a remote ref holds it"})
    audit = synthetic_audit((tip, ""))
    assert audit_mod.provenance(audit) == {}
    assert [(r[2], r[3]) for r in audit.drifts()] == [(f"unpushed revision {tip[:8]}", "listed although a remote ref holds it")]
    assert audit.drifts()[0][4].startswith("resolved on ")


def test_a_commit_no_remote_ref_contains_is_unpushed_although_it_exists(monkeypatch):
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "t")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "t@example.org")
        monkeypatch.setenv(f"GIT_{who}_DATE", "@86400 +0000")
    tree = git("rev-parse", f"{remote_tip()}^{{tree}}").stdout.strip()
    local = git("commit-tree", tree, "-m", "a commit no ref points at").stdout.strip()
    assert local and git("for-each-ref", "--contains", local, "refs/remotes").stdout == ""
    audit = synthetic_audit((local, ""))
    assert audit_mod.provenance(audit) == {"synthetic.csv": f"revision {local[:8]} is not on a pushed branch"}
    assert [(r[2], r[3], r[4]) for r in audit.drifts()] == [(f"unpushed revision {local[:8]}", "not listed", "no pushed equivalent")]


def test_the_recorded_tree_of_a_pushed_commit_resolves_a_lost_revision():
    tree = git("rev-parse", f"{remote_tip()}^{{tree}}").stdout.strip()
    audit = synthetic_audit((LOST, tree))
    assert audit_mod.provenance(audit) == {}
    assert not audit.drifts()
    assert any(r[2] == f"revision {LOST[:8]} has the tree of a pushed commit" for r in audit.results)
    assert audit_mod.resolve_revisions(audit).same_tree[LOST]


def test_a_recorded_tree_no_pushed_commit_has_resolves_nothing():
    audit = synthetic_audit((LOST, "f" * 40))
    assert audit_mod.provenance(audit) == {"synthetic.csv": f"revision {LOST[:8]} is not in this repository"}
    assert [r[2] for r in audit.drifts()] == [f"unpushed revision {LOST[:8]}"]


def test_one_revision_with_two_recorded_trees_is_reported():
    tree = git("rev-parse", f"{remote_tip()}^{{tree}}").stdout.strip()
    audit = synthetic_audit((LOST, tree), (LOST, "f" * 40))
    audit_mod.provenance(audit)
    assert f"rows cite revision {LOST[:8]} with one code_tree" in {r[2] for r in audit.drifts()}


def test_the_recorded_tree_of_a_pushed_revision_must_be_its_tree():
    tip = remote_tip()
    tree = git("rev-parse", f"{tip}^{{tree}}").stdout.strip()
    audit = synthetic_audit((tip, tree))
    audit_mod.provenance(audit)
    assert not audit.drifts()
    audit = synthetic_audit((tip, "f" * 40))
    audit_mod.provenance(audit)
    assert [r[2] for r in audit.drifts()] == [f"code_tree of revision {tip[:8]} is its tree"]


def test_an_alias_with_another_patch_id_is_reported(tmp_path, monkeypatch):
    audit, gaps = aliased_audit(tmp_path, monkeypatch, lambda t: t.replace("d4c75b4441714e8f", "d4c75b4441714e8e"))
    assert {r[2] for r in audit.drifts()} == {"pushed_revision bc3e0cb1 has patch-id d4c75b44",
                                               "branch_revision 6e425a39 has patch-id d4c75b44",
                                               "unpushed revision f9d1ac05"}
    assert gaps == {path: "revision f9d1ac05 is not in this repository" for path in SC_N12_ROWS}


def test_an_alias_to_an_unknown_commit_is_reported(tmp_path, monkeypatch):
    audit, gaps = aliased_audit(tmp_path, monkeypatch, lambda t: t.replace("79ec1b7f16366d0c", "79ec1b7f16366d0d"))
    assert {r[2] for r in audit.drifts()} == {"pushed_revision 79ec1b7f is in this repository",
                                               "pushed_revision 79ec1b7f is on the remote",
                                               "pushed_revision 79ec1b7f has patch-id debc2a84",
                                               "rows commit ba7c3b45 sits on 79ec1b7f",
                                               "unpushed revision a50f4b02"}
    assert gaps == {path: "revision a50f4b02 is not in this repository" for path in SC_N12_ROWS}


def test_an_alias_whose_rows_commit_is_elsewhere_is_reported(tmp_path, monkeypatch):
    audit, gaps = aliased_audit(tmp_path, monkeypatch, lambda t: t.replace("dd9ef6f76ff8ec3adbc4495d401893d03d3bf916",
                                                                             "ba7c3b45c0ddc9394156b1441519561555731e88", 1))
    assert {r[2] for r in audit.drifts()} == {"rows commit ba7c3b45 sits on 016b6911",
                                               "rows commit ba7c3b45 first cites 9149d1b0",
                                               "unpushed revision 9149d1b0"}
    assert gaps == {path: "revision 9149d1b0 is not in this repository" for path in SC_N12_ROWS}


def git(*args, stdin=None):
    return subprocess.run(["git", *args], cwd=ROOT, input=stdin, capture_output=True, text=True, check=False)


def test_the_revision_index_answers_as_one_git_call_per_question(audited):
    names = sorted({r[c].strip() for rows in audited.read.values() for r in rows for c in audit_mod.REVISION_COLUMNS
                    if c in r and re.fullmatch(r"[0-9a-f]{7,40}", r[c].strip())} - set(audit_mod.EXTERNAL_REVISIONS))
    table = audit_mod.alias_table()
    index = audit_mod.alias_index(names, table)
    revisions = sorted({*names, *(r[c] for r in table for c in ("local_revision", "pushed_revision", "branch_revision") if r[c])})
    batched, single = {}, {}
    for revision in revisions:
        batched[revision, "exists"] = index.exists(revision)
        single[revision, "exists"] = git("cat-file", "-e", f"{revision}^{{commit}}").returncode == 0
    for row in table:
        for column in ("pushed_revision", "branch_revision"):
            revision = row[column]
            if not revision:
                continue
            batched[revision, "remote"] = index.on_remote(revision)
            single[revision, "remote"] = bool(git("for-each-ref", "--count=1", "--contains", revision, "refs/remotes").stdout.strip())
            batched[revision, "patch-id"] = index.patch_id(revision)
            single[revision, "patch-id"] = git("patch-id", "--stable", stdin=git("show", revision).stdout).stdout.split()[0]
        rows_commit, local, pushed = row["rows_commit"], row["local_revision"], row["pushed_revision"]
        parents = git("rev-parse", f"{rows_commit}^@").stdout.split()
        batched[rows_commit, "parents"] = index.parents_of(rows_commit)
        single[rows_commit, "parents"] = parents
        batched[pushed, rows_commit, "ancestor"] = index.is_ancestor(pushed, parents[0])
        single[pushed, rows_commit, "ancestor"] = git("merge-base", "--is-ancestor", pushed, parents[0]).returncode == 0
        citing = git("log", "--reverse", "--format=%H", f"-S{local}", rows_commit).stdout.split()
        batched[local, rows_commit, "citing"] = index.cites[local] & index.closure([rows_commit])
        single[local, rows_commit, "citing"] = set(citing)
        batched[local, rows_commit, "first"] = index.first_cites(local, rows_commit)
        single[local, rows_commit, "first"] = citing[:1] == [rows_commit]
    assert batched == single
    assert sum(single[k] for k in single if k[-1] == "first") == len(table)


def test_revision_reachability_comes_from_remote_refs(tmp_path, monkeypatch):
    def repo(*args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "core.hooksPath=/dev/null", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    repo("init", "-q")
    (tmp_path / "row.csv").write_text("revision\none\n")
    repo("add", "row.csv")
    repo("commit", "-qm", "pushed")
    pushed = repo("rev-parse", "HEAD")
    repo("update-ref", "refs/remotes/origin/main", pushed)
    (tmp_path / "row.csv").write_text("revision\ntwo\n")
    repo("commit", "-qam", "local only")
    local = repo("rev-parse", "HEAD")
    monkeypatch.setattr(audit_mod, "ROOT", tmp_path)
    index = audit_mod.RevisionIndex((pushed, local), (), (), ())
    assert index.exists(local)
    assert index.on_remote(pushed)
    assert not index.on_remote(local)


@pytest.mark.parametrize(("driver", "attribute"), [(True, True), (True, False), (False, True)])
def test_a_textconv_driver_sends_the_index_back_to_git_log(tmp_path, monkeypatch, driver, attribute):
    text = "0123456789abcdef0123456789abcdef01234567"

    def repo(*args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "core.hooksPath=/dev/null", *args],
                              cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()

    repo("init", "-q")
    if driver:
        repo("config", "diff.swap.textconv", f"sed s/TOKEN/{text}/")
    if attribute:
        (tmp_path / ".gitattributes").write_text("*.txt diff=swap\n")
    (tmp_path / "a.txt").write_text("nothing\n")
    repo("add", "-A")
    repo("commit", "-qm", "one")
    (tmp_path / "a.txt").write_text("TOKEN\n")
    repo("commit", "-qam", "two")
    head = repo("rev-parse", "HEAD")
    assert text not in repo("cat-file", "-p", "HEAD:a.txt")
    monkeypatch.setattr(audit_mod, "ROOT", tmp_path)
    index = audit_mod.RevisionIndex((), (text,), (head,), ())
    assert index.textconv == ({"a.txt": "swap"} if driver and attribute else {})
    assert index.cites[text] == set(repo("log", "--format=%H", f"-S{text}", head).split())
    assert index.cites[text] == ({head} if driver and attribute else set())
    assert index.first_cites(text, head) is (driver and attribute)


@pytest.mark.parametrize(
    ("shown", "value", "ok"),
    [
        ("10,616", 10615.5, True),
        ("1", 1.0, True),
        ("1", 0.6, False),
        ("0.0044", 0.00436, True),
        ("0.0011", 0.00436, False),
        ("3.2e-05", 3.24e-05, True),
        ("$-1.3509\\times 10^{-7}$", -1.35093e-07, True),
        ("−628.372", -628.3714, False),
        ("−628.371", -628.3714, True),
        ("2_000_000", 2e6, True),
        ("", None, True),
        ("-", None, True),
        ("0.5", None, False),
    ],
)
def test_agrees_reads_half_a_unit_in_the_last_digit(shown, value, ok):
    assert audit_mod.agrees(shown, value) is ok


def test_statistics_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    for table in ((5, 6, 0, 6), (11, 12, 12, 12), (9, 12, 5, 12), (2, 12, 6, 12), (12, 12, 0, 12)):
        ha, na, hb, nb = table
        expected = stats.fisher_exact([[ha, na - ha], [hb, nb - hb]], alternative="two-sided")[1]
        assert math.isclose(audit_mod.fisher(*table), expected, rel_tol=1e-9)
    samples = (
        ([1.0, 2.0, 2.0, 5.0, 9.0], [2.0, 3.0, 4.0, 4.0, 10.0, 12.0]),
        ([4e6, 4e6, 1.2e6, 3e5], [4e6, 4e6, 4e6, 2.5e6]),
        ([8266.0, 9000.0, 7000.0], [1779.0, 2000.0, 2500.0, 1900.0]),
    )
    for a, b in samples:
        for continuity in (True, False):
            expected = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic",
                                          use_continuity=continuity).pvalue
            assert math.isclose(audit_mod.mann_whitney(a, b, continuity), expected, rel_tol=1e-9)
    assert audit_mod.mann_whitney([3.0, 3.0], [3.0, 3.0], True) == 1.0
    assert audit_mod.mann_whitney([3.0], [3.0], True, degenerate=None) is None


def test_type7_quantiles():
    np = pytest.importorskip("numpy")
    values = [4e6, 1801909.5, 12.0, 3.0, 99.5, 7.25]
    for p in (0.25, 0.5, 0.75):
        assert math.isclose(audit_mod.type7(values, p), float(np.quantile(values, p)), rel_tol=1e-12)


@pytest.mark.parametrize("token", [
    "\\rerun{5}",
    "\\rerun{8}",
    "\\rerun",
    "[RERUN: item 5]",
    "[RERUN",
    "[rerun: item 1]",
])
@pytest.mark.parametrize("target", [
    audit_mod.TEX,
    audit_mod.SI,
    audit_mod.COVER,
    "paper/jctc/macros.tex",
])
def test_injected_rerun_or_bracket_rerun_fails_the_audit(target, token):
    audit = audit_mod.Audit()
    audit.text_cache[target] = audit.text(target) + f"\n{token}\n"
    audit_mod.rerun_check(audit)
    assert [(row[1], row[2]) for row in audit.drifts()] == [(target, "no \\rerun or [RERUN remains")]


@pytest.mark.parametrize("phrase", [
    "The Zenodo record is pending.",
    "DOI: [to be assigned by Zenodo on acceptance].",
    "DOI: TBD.",
    "% pending Zenodo upload",
])
@pytest.mark.parametrize("target", [
    audit_mod.TEX,
    audit_mod.SI,
    audit_mod.COVER,
    "paper/jctc/macros.tex",
])
def test_injected_pending_or_tbd_text_fails_the_audit(target, phrase):
    audit = audit_mod.Audit()
    audit.text_cache[target] = audit.text(target) + f"\n{phrase}\n"
    audit_mod.rerun_check(audit)
    assert [(row[1], row[2]) for row in audit.drifts()] == [
        (target, "no pending, to be assigned, or TBD remains in TeX")
    ]


def test_doi_macro_must_be_defined_in_one_place():
    audit = audit_mod.Audit()
    audit.text_cache[audit_mod.TEX] = audit.text(audit_mod.TEX) + "\n\\newcommand{\\DOI}{dup}\n"
    audit_mod.rerun_check(audit)
    assert any(row[2] == "\\DOI macro is defined in one place" for row in audit.drifts())


def test_copper_tolerance_and_jcc_line_link_the_sources():
    audit = audit_mod.Audit()
    audit_mod.supporting_numbers(audit)
    tolerance = [row for row in audit.results if row[2] == "TARGET_TOL matches tally tolerance"]
    jcc = [row for row in audit.results if row[2] == "the jcc script's target line"]
    assert tolerance == [("OK", audit_mod.SI, "TARGET_TOL matches tally tolerance", "0.001", "0.001")]
    assert jcc and jcc[0][0] == "OK"
    claimed, actual = audit_mod.jcc_target_lines(
        (ROOT / "paper/jctc/scripts/hq_sci_tally.py").read_text(),
        (ROOT / "scripts/elja_jcc_molslab_ensemble.sh").read_text(),
    )
    assert claimed == actual
    assert jcc[0][3] == str(claimed)
    assert float(jcc[0][4]) == actual


def test_a_tally_tolerance_literal_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/scripts/hq_sci_tally.py"
    audit.text_cache[path] = audit.text(path).replace(
        "REFERENCE_TOL_EV = DEFAULT_HIT_TOL",
        "REFERENCE_TOL_EV = 1e-4",
    )
    audit_mod.supporting_numbers(audit)
    drift = [row for row in audit.drifts() if row[2] == "TARGET_TOL matches tally tolerance"]
    assert len(drift) == 1
    assert drift[0][3] == "0.0001"


def test_a_shell_target_tol_literal_is_reported():
    audit = audit_mod.Audit()
    path = "scripts/elja_hq_sci_molslab_bank.sh"
    text = audit.text(path)
    assert text.count("export TARGET_TOL=${DEFAULT_HIT_TOL}") == 2
    audit.text_cache[path] = text.replace(
        "export TARGET_TOL=${DEFAULT_HIT_TOL}",
        "export TARGET_TOL=1e-4",
    )
    audit_mod.supporting_numbers(audit)
    labels = {row[2] for row in audit.drifts()}
    assert "TARGET_TOL value" in labels
    assert "TARGET_TOL matches tally tolerance" in labels


def test_a_shifted_jcc_claim_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/scripts/hq_sci_tally.py"
    text = audit.text(path)
    match = re.search(r"elja_jcc_molslab_ensemble\.sh line (\d+) sets that target\.", text)
    assert match
    claimed = int(match.group(1))
    audit.text_cache[path] = text.replace(
        f"line {claimed} sets that target.",
        f"line {claimed + 1} sets that target.",
        1,
    )
    audit_mod.supporting_numbers(audit)
    drift = [row for row in audit.drifts() if row[2] == "the jcc script's target line"]
    assert len(drift) == 1
    assert drift[0][3] == str(claimed + 1)
    assert float(drift[0][4]) == claimed


def test_resolved_hit_tolerances_follow_the_expression():
    names = {"DEFAULT_HIT_TOL": 1.0e-3}
    assert audit_mod.resolve_named_hit_tol("${DEFAULT_HIT_TOL}", names) == 1.0e-3
    assert audit_mod.resolve_named_hit_tol("DEFAULT_HIT_TOL", names) == 1.0e-3
    assert audit_mod.resolve_named_hit_tol("1e-4", names) == 1.0e-4
    assert audit_mod.resolve_named_hit_tol("not-a-tolerance", names) is None


def test_doi_macro_must_match_citation_metadata():
    audit = audit_mod.Audit()
    citation = "CITATION.cff"
    audit.text_cache[citation] = audit.text(citation).replace(
        "reproducibility archive concept DOI 10.5281/zenodo.20672620",
        "reproducibility archive concept DOI 10.5281/zenodo.0",
    )
    audit_mod.rerun_check(audit)
    assert any(
        row[2] == "\\DOI matches the reproducibility archive concept DOI in CITATION.cff"
        for row in audit.drifts()
    )


def test_the_british_spelling_of_labeled_is_caught():
    doubled_label = ["label" + "l" + suffix for suffix in ("ed", "ing")]
    for word in (*doubled_label, "modelled", "travelled"):
        assert audit_mod.BRITISH.search(word)
    assert not audit_mod.BRITISH.search("labeled")


def test_water_header_keeps_the_committed_record_spelling():
    source = (ROOT / "paper/jctc/scripts/water_h2h.py").read_text()
    record = (ROOT / "paper/jctc/logs/water/NUMBERS.tsv").read_text()
    assert "rmsd_A=labelled Kabsch" in source
    assert "rmsd_A=labelled Kabsch" in record


@pytest.mark.parametrize(("cluster", "revision"), [
    ("au38", "77dece29d43ac3b33066c2887d602766ddbe61d2"),
    ("au55", "77dece29d43ac3b33066c2887d602766ddbe61d2"),
    ("cu19au19", "e1ae0658e730c8aa53f75b48ac7c009487422553"),
    ("cu27au28", "e1ae0658e730c8aa53f75b48ac7c009487422553"),
    ("cu6au32", "e1ae0658e730c8aa53f75b48ac7c009487422553"),
])
def test_each_darby_log_receipt_names_its_cluster_and_revision(cluster, revision):
    text = (ROOT / f"bench/metal-cluster-h2h/darby-logs/{cluster}/README.md").read_text()
    label = {
        "au38": "Au38",
        "au55": "Au55",
        "cu19au19": "Cu19Au19",
        "cu27au28": "Cu27Au28",
        "cu6au32": "Cu6Au32",
    }[cluster]
    assert f"This directory contains the {label} anneal" in text
    assert revision in text
    assert "the parent of" not in text


HISTORY_MUTATIONS = [
    ("previously", "The value was previously larger."),
    ("no longer", "The branch no longer stops early."),
    ("used to", "The walk used to ignore the flag."),
    ("used to stop", "Seed 2007 USED TO STOP at -263.04."),
    ("earlier draft", "Softened sentences from the Earlier Draft."),
    ("earlier version", "An earlier version reported a constant."),
    ("v1.N release tag", "The sealed v1.28 tip."),
    ("stay open for", "Two hits stay open for Papers."),
    ("stays open", "The comparison stays open."),
    ("withheld", "The hit count is withheld."),
    ("removed from the article", "The hit count was removed from the article."),
    ("unrecorded", "The revision is unrecorded."),
    ("unrecoverable", "The revision is unrecoverable."),
    ("cannot be recovered", "The revision cannot be recovered."),
    ("not captured", "The host was not captured."),
    ("Recorded limitations", "## Recorded limitations"),
    ("Left for Papers", "## Left for Papers"),
    ("the parent of", "Built from the parent of the table commit."),
]


@pytest.mark.parametrize(("label", "sentence"), HISTORY_MUTATIONS)
@pytest.mark.parametrize("path", [
    "bench/metal-cluster-h2h/logs/README.md",
    "bench/tip4p-h2h/README-h2h.md",
    "paper/jctc/PROSE-RECEIPT.md",
    "paper/jctc/REPRODUCE.md",
    "paper/jctc/si.tex",
    "paper/related-work.md",
])
def test_history_narration_is_rejected(path, label, sentence):
    audit = audit_mod.Audit()
    audit_mod.history_narration(audit)
    assert not [row for row in audit.drifts() if row[1] == path]
    audit = audit_mod.Audit()
    audit.text_cache[path] = audit.text(path) + f"\n{sentence}\n"
    audit_mod.history_narration(audit)
    flagged = [row for row in audit.drifts() if row[1] == path]
    assert flagged
    assert label in flagged[0][4].split(", ")


@pytest.mark.parametrize("sentence", [
    "The seed is used toward the next hop.",
    "Version v2.1 of the preset.",
    "The first v1 row has no decimal tag.",
])
def test_history_narration_ignores_look_alikes(sentence):
    path = "bench/tip4p-h2h/README-h2h.md"
    audit = audit_mod.Audit()
    audit.text_cache[path] = audit.text(path) + f"\n{sentence}\n"
    audit_mod.history_narration(audit)
    assert not [row for row in audit.drifts() if row[1] == path]


@pytest.mark.parametrize("citation", [
    r"\cite{missingCitationKey}",
    r"\citep{missingCitationKey}",
    r"\citet{missingCitationKey}",
    r"\nocite{missingCitationKey}",
    r"\cite*{missingCitationKey}",
    r"\citep*{missingCitationKey}",
    r"\citet*{missingCitationKey}",
    r"\citeauthor{missingCitationKey}",
    r"\citeauthor*{missingCitationKey}",
    r"\citealp{missingCitationKey}",
    r"\citealp*{missingCitationKey}",
    r"\citealt{missingCitationKey}",
    r"\citealt*{missingCitationKey}",
    r"\citeyear{missingCitationKey}",
    r"\citeyear*{missingCitationKey}",
    r"\citeyearpar{missingCitationKey}",
    r"\citeyearpar*{missingCitationKey}",
    r"\citenum{missingCitationKey}",
    r"\citenum*{missingCitationKey}",
    r"\Cite{missingCitationKey}",
    r"\Cite*{missingCitationKey}",
    r"\Citet{missingCitationKey}",
    r"\Citet*{missingCitationKey}",
    r"\Citep{missingCitationKey}",
    r"\Citep*{missingCitationKey}",
    r"\Citeauthor{missingCitationKey}",
    r"\Citeauthor*{missingCitationKey}",
    r"\citep[see][12]{missingCitationKey}",
])
def test_an_unresolved_cite_key_is_rejected(citation):
    path = "paper/jctc/manuscript.tex"
    audit = audit_mod.Audit()
    audit.text_cache[path] = audit.text(path) + f"\n{citation}\n"
    audit_mod.bibliography_keys(audit)
    assert any(
        row[1] == "paper/jctc/refs.bib"
        and row[2] == "every \\cite key resolves in the paper bibliography"
        and "missingCitationKey" in row[4]
        for row in audit.drifts()
    )


@pytest.mark.parametrize("literal", [
    "27.2113862", "27.211_386_245_988", "0.529177210903", "0.529_177_210_903", "1.8897261",
    "4" + ".184",
])
@pytest.mark.parametrize("path", [
    "src/methods/tric_opt.rs",
    "tests/catalog_signature.rs",
    "examples/gfn2_water_crest_driver.rs",
    "examples/ase_objective.py",
    "scripts/tric_quench/quench_compare.py",
    "bench/water-h2o16-quench/README.md",
    "bench/covalent-h2h/gfn2/scc-false-minimum/README.md",
])
def test_typed_hartree_and_bohr_literals_are_rejected(path, literal):
    audit = audit_mod.Audit()
    audit_mod.conversion_literals(audit)
    assert not [row for row in audit.drifts() if row[1] == path]
    audit = audit_mod.Audit()
    audit.text_cache[path] = audit.text(path) + f"\nconst HARTREE: f64 = {literal};\n"
    audit_mod.conversion_literals(audit)
    assert [row for row in audit.drifts() if row[1] == path]


def test_a_stale_prose_receipt_line_is_rejected():
    path = "paper/jctc/PROSE-RECEIPT.md"
    audit = audit_mod.Audit()
    audit_mod.prose_receipt_lines(audit)
    assert not audit.drifts()
    audit = audit_mod.Audit()
    audit.text_cache[path] = audit.text(path).replace("paper/jctc/si.tex:13\n", "paper/jctc/si.tex:14\n", 1)
    audit_mod.prose_receipt_lines(audit)
    assert [row[2] for row in audit.drifts()] == ["paper/jctc/si.tex:14 holds its recorded string"]


def test_a_stale_figure_hash_is_rejected():
    path = "paper/jctc/FIGURES.md"
    audit = audit_mod.Audit()
    audit_mod.figure_row_hashes(audit)
    assert not audit.drifts()
    audit = audit_mod.Audit()
    text = audit.text(path)
    digest = re.search(r"^Row 1 is \S+\.\n\n([0-9a-f]{64})$", text, re.MULTILINE).group(1)
    audit.text_cache[path] = text.replace(digest, "0" * 64, 1)
    audit_mod.figure_row_hashes(audit)
    assert [row[2] for row in audit.drifts()] == ["row 1 algorithm_stack.pdf sha256"]


@pytest.mark.parametrize("path", ["bench/tip4p-h2h/wall-r2.csv", "bench/tip4p-h2h/check-r2.md"])
def test_generated_tip4p_summaries_are_append_only_records(path):
    assert path in audit_mod.APPEND_ONLY_RECORDS


@pytest.mark.parametrize(("path", "old", "new"), [
    ("bench/tip4p-h2h/wall-r2.csv", "unscored,unscored", "0/12,0/12"),
    ("bench/tip4p-h2h/check-r2.md", "| unscored | unscored |", "| 0/12 | 0/12 |"),
    (
        "bench/tip4p-h2h/README-h2h.md",
        "Size 25 has no independent reference and is\nunscored.",
        "Size 25 has no independent reference and has 0/12 hits.",
    ),
    (
        "paper/jctc/manuscript.tex",
        "Size 25 has no independent reference and is unscored.",
        "Size 25 has no independent reference and has 0/12 hits.",
    ),
    (
        "paper/jctc/si.tex",
        "Size 25 has no independent reference and is unscored.",
        "Size 25 has no independent reference and has 0/12 hits.",
    ),
])
def test_a_tip4p_size_25_hit_count_is_rejected(path, old, new):
    audit = audit_mod.Audit()
    text = audit.text(path)
    assert old in text
    audit.text_cache[path] = text.replace(old, new, 1)
    audit_mod.tip4p_unscored(audit)
    assert [row for row in audit.drifts() if row[1] == path]


def test_a_tip4p_size_25_dash_table_cell_is_rejected():
    path = "paper/jctc/si.tex"
    audit = audit_mod.Audit()
    marker = "% tip4p-gmin table: end"
    row = "(H\\(_2\\)O)\\(_{25}\\) & -- & -- & -- & -- & -- & -- & -- \\\\\\\\\n"
    audit.text_cache[path] = audit.text(path).replace(marker, row + marker)
    audit_mod.tip4p_unscored(audit)
    assert [row for row in audit.drifts() if row[1] == audit_mod.PAPER]


@pytest.mark.parametrize(("path", "sentence"), [
    (
        "paper/jctc/manuscript.tex",
        "At 25 molecules the median costs are 17.865 and 53.555 microseconds.",
    ),
    (
        "paper/jctc/si.tex",
        "The wall-time p is 7.4e-7 at every size.",
    ),
])
def test_a_tip4p_size_25_cost_or_p_claim_is_rejected(path, sentence):
    audit = audit_mod.Audit()
    marker = (
        "\\section*{Author contributions}"
        if path == "paper/jctc/manuscript.tex"
        else "% tip4p-gmin prose: end"
    )
    audit.text_cache[path] = audit.text(path).replace(
        marker, sentence + "\n" + marker, 1
    )
    audit_mod.tip4p_unscored(audit)
    assert [row for row in audit.drifts() if row[1] == path]


@pytest.mark.parametrize("path", [
    "bench/tip4p-water-rates/rates-a24ba760.csv",
    "bench/tip4p-water-rates/rates.csv",
])
def test_a_water_rate_hit_flag_outside_its_printed_tolerance_is_rejected(path):
    audit = audit_mod.Audit()
    rows = list(csv.DictReader(audit.text(path).splitlines()))
    rows[0]["hit"] = str(1 - int(rows[0]["hit"]))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    audit.text_cache[path] = buffer.getvalue()
    audit_mod.tip4p_unscored(audit)
    assert [row for row in audit.drifts() if row[1] == path]


@pytest.mark.parametrize("output", [
    "bench/tip4p-h2h/wall-r2.csv",
    "paper/jctc/logs/cuh2-hardware-receipt.txt",
])
def test_a_reproduce_command_that_writes_a_tracked_record_is_rejected(output):
    path = "paper/jctc/REPRODUCE.md"
    audit = audit_mod.Audit()
    audit.text_cache[path] = (
        audit.text(path)
        + f"\n    python3 runner.py --rows-out {output}\n"
    )
    audit_mod.reproduction_records(audit)
    assert [
        row for row in audit.drifts()
        if row[1] == path and "never write tracked records" in row[2]
    ]


def test_reproduce_rules_never_declare_tracked_record_outputs():
    reproduce = (ROOT / "paper/jctc/REPRODUCE.md").read_text()
    rules = set(re.findall(
        r"snakemake -s workflow/Snakefile (\w+) ",
        reproduce,
    ))
    snakefile = (ROOT / "workflow/Snakefile").read_text()
    tracked = set(subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines())
    records = {
        path for path in tracked
        if path.endswith((".csv", ".tsv", ".jsonl"))
        or path.startswith("paper/jctc/logs/")
    }
    for rule in rules:
        body = re.search(
            rf"(?ms)^rule {re.escape(rule)}:\n(.*?)(?=^rule \w+:|\Z)",
            snakefile,
        )
        assert body is not None
        output = re.search(
            r"(?ms)^    output:\s*\n(.*?)(?=^    \w+(?:\.\w+)?:|\Z)",
            body.group(1),
        )
        declared = set(re.findall(
            r"[\"']([^\"']+)[\"']",
            output.group(1) if output else "",
        ))
        assert not declared & records, (
            f"{rule} declares tracked records as outputs: "
            f"{sorted(declared & records)}"
        )


def _stub_snakefile(text: str) -> str:
    shell = (
        '    shell:\n'
        '        """\n'
        "        set -euo pipefail\n"
        "        mkdir -p /work/anneal-snakemake /work/gmin-panel/output\n"
        '        if [ -n "{output}" ]; then\n'
        "          for target in {output}; do\n"
        '            mkdir -p "$(dirname "$target")"\n'
        '            touch "$target"\n'
        "          done\n"
        "        fi\n"
        '        """\n'
    )
    return re.sub(r'(?ms)^    shell:\n        """\n.*?"""\n', shell, text)


def _concrete_rules(text: str) -> list[str]:
    rules = []
    for name, body in re.findall(r"(?ms)^rule (\w+):\n(.*?)(?=^rule \w+:|\Z)", text):
        if name == "all":
            continue
        output = re.search(r"(?ms)^    output:\n(.*?)(?=^    \w|\Z)", body)
        if output and "{" in output.group(1):
            continue
        rules.append(name)
    return rules


def _tracked_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )


def test_stubbed_snakemake_force_leaves_tracked_files_unchanged(tmp_path):
    pytest.importorskip("snakemake")
    source = (ROOT / "workflow/Snakefile").read_text()
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    for name, body in re.findall(r"(?ms)^rule (\w+):\n(.*?)(?=^rule \w+:|\Z)", source):
        output = re.search(r"(?ms)^    output:\n(.*?)(?=^    \w|\Z)", body)
        declared = re.findall(r"""["']([^"']+)["']""", output.group(1) if output else "")
        assert not set(declared) & tracked, f"{name} declares tracked outputs"
    stub = tmp_path / "Snakefile"
    stub.write_text(_stub_snakefile(source))
    before = _tracked_status()
    completed = subprocess.run(
        [
            sys.executable, "-m", "snakemake", "-s", str(stub),
            *_concrete_rules(source), "--cores", "1", "--force", "-k", "--nocolor",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert _tracked_status() == before


def test_a_tracked_snakemake_output_is_deleted_on_force(tmp_path):
    pytest.importorskip("snakemake")
    target = ROOT / "bench/gmin-panel/summary.csv"
    original = target.read_bytes()
    source = (ROOT / "workflow/Snakefile").read_text().replace(
        'touch("/work/anneal-snakemake/gmin_panel.stamp")',
        '"bench/gmin-panel/summary.csv"',
    )
    stub = tmp_path / "Snakefile"
    stub.write_text(_stub_snakefile(source))
    try:
        subprocess.run(
            [
                sys.executable, "-m", "snakemake", "-s", str(stub), "gmin_panel",
                "--cores", "1", "--force", "--nocolor",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert target.read_bytes() != original
    finally:
        target.write_bytes(original)


def test_a_tip4p_unscored_reference_is_rejected():
    path = "paper/jctc/manuscript.tex"
    audit = audit_mod.Audit()
    text = audit.text(path)
    audit.text_cache[path] = text.replace(
        "(\\mathrm{H_2O})_{21} &: -916.7074302940",
        "(\\mathrm{H_2O})_{25} &: -1113.0346568571",
    )
    audit_mod.tip4p_unscored(audit)
    assert [
        row for row in audit.drifts()
        if row[1] == path and "reference list is exactly" in row[2]
    ]


def test_every_figure_script_hash_is_checked(tmp_path):
    script = tmp_path / "paper/jctc/scripts/example.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('figure')\n")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    receipt = tmp_path / "paper/jctc/FIGURE-SOURCE.tsv"
    receipt.write_text(
        "path\tsource_sha256\tkind\n"
        f"paper/jctc/scripts/example.py\t{digest}\tscript\n"
    )

    audit = audit_mod.Audit()
    audit_mod.figure_source_scripts(audit, tmp_path)
    assert not audit.drifts()

    script.write_text("print('changed')\n")
    audit = audit_mod.Audit()
    audit_mod.figure_source_scripts(audit, tmp_path)
    assert len(audit.drifts()) == 1


def test_every_committed_figure_source_hash_matches_the_current_file():
    audit = audit_mod.Audit()
    audit_mod.figure_source_scripts(audit)
    assert not audit.drifts()
    [count] = [row for row in audit.results if row[2] == "effective figure data rows"]
    assert count[4] == "276.0"


@pytest.mark.parametrize("sentence", [
    "On Au\\(_{55}\\) this setting hits 0 of 12 where GMIN hits 1 of 12.",
    "Au\\(_{55}\\) is significant by Fisher's test.",
    "Au\\(_{55}\\) gives \\(p=0.014\\).",
    "6 of the 9 wins are significant.",
    "Six of our nine wins are significant.",
])
def test_a_stale_au55_claim_is_reported(sentence):
    assert audit_mod.stale_sentences(f"Intro sentence. {sentence} Closing sentence.") == [sentence]


@pytest.mark.parametrize("sentence", [
    "On Au\\(_{55}\\) the ASE panel hits 0 of 12 where GMIN hits 3 of 12.",
    "Au\\(_{55}\\) is not significant, with \\(p=0.069\\).",
    "On Au\\(_{55}\\) this setting hits 6 of 12 where GMIN hits 1 of 12.",
])
def test_a_current_au55_claim_is_accepted(sentence):
    assert audit_mod.stale_sentences(f"Intro sentence. {sentence} Closing sentence.") == []


def test_a_table_row_is_not_read_as_a_claim():
    table = "\\begin{tabular}{ll}\nAu\\(_{55}\\) & GMIN 0 of 12 significant\\\\\n\\end{tabular}\n"
    assert audit_mod.stale_sentences(table) == []


def test_the_au55_claims_hold_in_the_committed_text():
    audit = audit_mod.Audit()
    audit_mod.manuscript_au55(audit)
    assert not audit.drifts()
    assert any("no Au55 count" in row[2] for row in audit.results)


def test_the_landfold_figure_records_its_redrawn_panels():
    audit = audit_mod.Audit()
    audit_mod.manuscript_new(audit)
    rows = [row for row in audit.results if "landfold" in row[2]]
    assert rows
    assert all(row[0] == "OK" for row in rows)


@pytest.mark.parametrize("stale, sentence", [
    ("on 8 of the 8 clusters", "The median wall is lower for this setting on 99 of the 8 clusters."),
    ("on 1 of the 8", "GMIN hits more seeds on 9 of the 8."),
])
def test_a_stale_second_copy_of_a_gmin_number_is_reported(stale, sentence):
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.TEX)
    assert stale in " ".join(text.split())
    audit.text_cache[audit_mod.TEX] = text + f"\n\\section{{Copy}}\n{sentence}\n"
    panel, groups = audit_mod.gmin_panel_groups(audit)
    audit_mod.gmin_body_copies(audit, panel, groups)
    assert [row for row in audit.drifts() if "every copy" in row[2]]


def pdf_length():
    spec = importlib.util.spec_from_file_location("pdf_length", ROOT / "paper/jctc/scripts/pdf_length.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_rendered_abstract_is_counted_without_page_numbers():
    length = pdf_length()
    abstract = " ".join(["word"] * 150) + " on 9 of 13 at 0.317 and GFN2. " + "end " * 40
    text = (f"Title\nAbstract\n{abstract}\n\n1\n\f2\n\nIntroduction\n" + "body " * 6000
            + "12 of 12\n\n3\n\fAuthor contributions\nnot counted\n")
    counts = length.measure(text)
    assert counts == {"abstract_words": 150 + 8 + 40, "abstract_numerals": 4, "body_words": 6003}
    assert not length.problems(counts)
    assert length.problems(dict(counts, abstract_words=201, abstract_numerals=7, body_words=7001)) == [
        "abstract has 201 words, more than 200", "abstract has 7 numerals, more than 6",
        "body has 7001 words, outside 5000-7000"]


def test_a_long_rendered_abstract_or_a_stale_receipt_is_reported(tmp_path):
    length = pdf_length()
    counts = {"abstract_words": 201, "abstract_numerals": 7, "body_words": 6000}
    receipt = tmp_path / "pdf-length.txt"
    receipt.write_text(length.receipt_text(counts, 30, {path: "0" * 64 for path in length.SOURCES}))
    audit = audit_mod.Audit()
    audit_mod.rendered_length(audit, str(receipt))
    labels = {row[2] for row in audit.drifts()}
    assert labels >= {"the PDF counts were measured from the committed sources",
                      "rendered abstract words, at most 200 for JCTC", "rendered abstract numerals, at most 6 for JCTC"}


def test_a_british_spelling_that_reaches_a_pdf_is_reported(tmp_path):
    scripts = tmp_path / "paper/jctc/scripts"
    scripts.mkdir(parents=True)
    (scripts / "table.py").write_text(
        'CAPTION = r"core-orbit symmetrisation~\\cite{oakleySymmetrisationSchemesGlobal2013}"\n'
        'HEADER = f"# nearest-neighbour O...O under {3.5} A"\n'
        'LABEL = f"the nearest-neighbour distance, {1.2} A"\n')
    (tmp_path / "paper/jctc/manuscript.tex").write_text("The colour of each basin.\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    audit = audit_mod.Audit()
    audit_mod.british_spellings(audit, tmp_path)
    [row] = audit.drifts()
    assert row[-1] == ("paper/jctc/manuscript.tex:colour paper/jctc/scripts/table.py:symmetrisation "
                       "paper/jctc/scripts/table.py:neighbour")


def test_a_supporting_information_entry_without_its_format_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.TEX)
    assert text.count("copper slab (PDF).") == 1
    audit.text_cache[audit_mod.TEX] = text.replace("copper slab (PDF).", "copper slab.")
    audit_mod.jctc_requirements(audit)
    labels = {row[2] for row in audit.drifts()}
    assert "the Supporting Information entry names its format" in labels


def cross_labels():
    spec = importlib.util.spec_from_file_location("cross_labels", ROOT / "paper/jctc/scripts/cross_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_labels_are_numbered_as_achemso_numbers_them():
    text = "\n".join((
        "\\section{One}\\label{sec:one}",
        "\\begin{equation}x\\label{eq:a}\\end{equation}",
        "\\subsection{Two}\\label{sec:two}",
        "% \\label{sec:commented}",
        "\\begin{figure*}\\caption{x}\\label{fig:a}\\end{figure*}",
        "\\section*{Notes}",
        "\\begin{table}\\label{tab:a}\\end{table}\\begin{figure}\\label{fig:b}\\end{figure}",
        "\\section{Three}\\begin{equation}y\\label{eq:b}\\end{equation}\\label{sec:three}",
    ))
    numbers = cross_labels().numbers(text, "S")
    assert numbers == {"sec:one": "S1", "eq:a": "S1", "sec:two": "S1.1", "fig:a": "S1", "tab:a": "S1",
                       "fig:b": "S2", "eq:b": "S2", "sec:three": "S2"}


def test_an_undefined_cross_reference_or_a_stale_label_file_is_reported(monkeypatch):
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.TEX)
    assert text.count("Table~\\siref{tab:gofee}") >= 1
    audit.text_cache[audit_mod.TEX] = text.replace("Table~\\siref{tab:gofee}", "Table~\\siref{tab:gofee-moved}", 1)
    labels = cross_labels()
    monkeypatch.setattr(audit_mod, "load_script", lambda name, path: labels)
    monkeypatch.setattr(labels, "stale", lambda: ["labels-si.tex"])
    audit_mod.cross_references(audit)
    labels_seen = {row[2] for row in audit.drifts()}
    assert labels_seen == {"the cross-document labels match the sources", "every \\siref names a label of si.tex"}


def test_a_rendered_body_outside_the_band_is_reported(tmp_path):
    length = pdf_length()
    receipt = tmp_path / "pdf-length.txt"
    receipt.write_text(length.receipt_text({"abstract_words": 190, "abstract_numerals": 5, "body_words": 16380}, 60,
                                           length.source_digests()))
    audit = audit_mod.Audit()
    audit_mod.rendered_length(audit, str(receipt))
    assert [row[2] for row in audit.drifts()] == ["rendered body words, 5000-7000 for JCTC"]


def test_the_crest_equal_panel_holds_in_the_committed_text():
    audit = audit_mod.Audit()
    audit_mod.crest_equal(audit)
    assert not audit.drifts()


def test_a_crest_run_outside_the_call_tolerance_is_reported():
    audit = audit_mod.Audit()
    rows = [dict(r) for r in audit.rows("bench/crest-equal/rows.csv")]
    for row in rows:
        if row["arm"] == "crest":
            row["charged"] = str(int(row["charged"]) + 400)
    audit.read["bench/crest-equal/rows.csv"] = rows
    audit_mod.crest_equal(audit)
    labels = {row[2] for row in audit.drifts()}
    assert "CREST at 0.90 ps is at the calls of this setting: median within 5%" in labels


def test_the_equal_call_panel_is_absent_from_the_paper():
    audit = audit_mod.Audit()
    audit_mod.crest_equal(audit)
    assert not audit.drifts()
    text = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = text + "\nAt equal GFN2 calls the panel returns.\n"
    audit_mod.crest_equal(audit)
    assert any(row[2] == "the equal-call panel and the withdrawn hexamer counts are absent" for row in audit.drifts())


def test_session_metadata_records_the_anneal_repro_checkout(tmp_path):
    repro = tmp_path / "anneal_repro"
    repro.mkdir()
    subprocess.run(["git", "init", "-q", str(repro)], check=True)
    (repro / "a.txt").write_text("a\n")
    git = ["git", "-C", str(repro), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "a"], check=True)
    out = tmp_path / "sessions.jsonl"
    env = dict(__import__("os").environ, OMP_NUM_THREADS="3")
    proc = subprocess.run(
        ["python3", str(ROOT / "scripts/session_metadata.py"), "--output", str(out), "--campaign", "water-crest-fair-n192",
         "--root", str(ROOT), "--anneal-repro", str(repro), "--", "snakemake", "water_crest_fair"],
        cwd=ROOT, check=False, capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    row = json.loads(out.read_text())
    head = subprocess.run(["git", "-C", str(repro), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert row["anneal_repro"]["revision"] == head
    assert re.fullmatch(r"[0-9a-f]{40}", row["anneal_repro"]["tree"])
    assert row["openmp"]["omp_num_threads"] == "3"
    assert row["openmp"]["settings"]["OMP_NUM_THREADS"] == "3"


def test_a_tip4p_separate_cores_claim_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.SI)
    marker = "% tip4p-gmin prose: end"
    audit.text_cache[audit_mod.SI] = text.replace(
        marker, "The TIP4P r2 arms ran on separate cores.\n" + marker
    )
    audit_mod.manuscript_new(audit)
    assert [row for row in audit.drifts() if "does not claim separate cores" in row[2]]


def crest_equal_runner():
    spec = importlib.util.spec_from_file_location("crest_equal_run_panel", ROOT / "scripts/crest_equal/run_panel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_crest_equal_runner_records_its_session_before_any_run(tmp_path):
    runner = crest_equal_runner()
    core = str(min(__import__("os").sched_getaffinity(0)))
    out = tmp_path / "sessions.jsonl"
    runner.record_session(out, core, ["scripts/crest_equal/run_panel.py", "--seed", "2000"])
    row = json.loads(out.read_text())
    assert row["schema"] == "anneal-session-v2"
    assert row["campaign"] == "crest-equal-h2o6"
    assert row["command"] == ["scripts/crest_equal/run_panel.py", "--seed", "2000"]
    assert row["cpu_affinity"] == [int(core)]
    assert set(row["compilers"]) == {"rustc", "cargo", "gfortran", "cc"}


def test_the_crest_equal_runner_refuses_a_campaign_without_a_session_file(tmp_path, monkeypatch):
    runner = crest_equal_runner()
    binaries = [arg for name in ("anneal", "driver", "crest", "xtb", "engine") for arg in (f"--{name}", str(tmp_path / name))]
    monkeypatch.setattr("sys.argv", ["run_panel.py", *binaries, "--rows", str(tmp_path / "rows.csv")])
    with pytest.raises(SystemExit, match="--sessions is required"):
        runner.main()
    assert not (tmp_path / "rows.csv").exists()


def test_the_crest_equal_entry_passes_the_session_file():
    text = (ROOT / "scripts/crest_equal/enter_bench.sh").read_text()
    assert '--sessions "$ROOT/bench/crest-equal/sessions.jsonl"' in text


def ablation_campaign():
    spec = importlib.util.spec_from_file_location("depth_ablation_campaign",
                                                  ROOT / "scripts/depth_posterior_ablation/campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_held_out_rule_writes_rows_in_the_committed_schema(tmp_path):
    campaign = ablation_campaign()
    committed = (ROOT / "results/size-tiered-selection.csv").read_text().splitlines()
    provenance = (ROOT / "results/size-tiered-selection-provenance.csv").read_text().splitlines()
    assert committed[0] == campaign.HELD_OUT_HEADER
    assert provenance[0] == campaign.PROVENANCE_HEADER
    assert len(committed) == len(provenance)
    for line, provenance_line in zip(committed[1:], provenance[1:]):
        arm, n, seed, hit, calls, best, gap, charged, wall = line.split(",")
        p_arm, p_n, p_seed, revision, host = provenance_line.split(",")
        assert (arm, n, seed) == (p_arm, p_n, p_seed)
        if arm == "tiered":
            continue
        row = {"arm": arm, "seed": seed, "hit": hit, "force_calls_to_hit": calls if hit == "1" else "",
               "budget": str(campaign.PAPER_BUDGETS[int(n)]), "best": best, "charged": charged, "wall_s": wall}
        assert campaign.held_out_line(row, int(n)) == line
        assert campaign.provenance_line(row, int(n), revision, host) == provenance_line
    jobs = campaign.held_out_schedule()
    assert sorted(jobs) == sorted((size, seed, arm) for size in (75, 98, 104) for seed in range(7000, 7006)
                                  for arm in ("posterior", "uniform"))
    provenance_file = tmp_path / "provenance.csv"
    provenance_file.write_text("\n".join(provenance) + "\nposterior,75,7000,abc,bench\n")
    assert campaign.held_out_done(provenance_file, "abc") == {(75, 7000, "posterior")}
    assert campaign.held_out_done(provenance_file, "unknown") == {(int(f[1]), int(f[2]), f[0])
                                                                 for f in (line.split(",") for line in provenance[1:])}


def test_held_out_renames_the_results_before_the_sidecar(tmp_path, monkeypatch):
    campaign = ablation_campaign()
    results = tmp_path / "size-tiered-selection.csv"
    provenance = tmp_path / "size-tiered-selection-provenance.csv"
    results.write_text(campaign.HELD_OUT_HEADER + "\n")
    provenance.write_text(campaign.PROVENANCE_HEADER + "\n")
    monkeypatch.setattr(campaign, "held_out_schedule", lambda: [(75, 7000, "posterior")])
    monkeypatch.setattr(campaign.socket, "gethostname", lambda: "bench")

    def fake_run_job(binary, log_dir, order, size, seed, arm):
        return {
            "arm": arm,
            "seed": str(seed),
            "hit": "1",
            "force_calls_to_hit": "10",
            "budget": str(campaign.PAPER_BUDGETS[size]),
            "best": "-397.49233098",
            "charged": "10",
            "wall_s": "1.0",
        }

    monkeypatch.setattr(campaign, "run_job", fake_run_job)
    seen = []
    real_replace = os.replace

    def spy_replace(src, dst):
        real_replace(src, dst)
        seen.append(Path(dst).name)

    monkeypatch.setattr(campaign.os, "replace", spy_replace)
    campaign.run_held_out(tmp_path / "depth_ablation", tmp_path / "logs", results, provenance, "abc")
    assert seen == ["size-tiered-selection.csv", "size-tiered-selection-provenance.csv"]
    assert results.read_text().splitlines() == [
        campaign.HELD_OUT_HEADER,
        "posterior,75,7000,1,10,-397.49233098,0.00000002,10,1.0",
    ]
    assert provenance.read_text().splitlines() == [
        campaign.PROVENANCE_HEADER,
        "posterior,75,7000,abc,bench",
    ]
    names = sorted(path.name for path in tmp_path.iterdir() if path.is_file())
    assert names == sorted([results.name, provenance.name])


def test_held_out_leaves_both_files_when_the_results_rename_fails(tmp_path, monkeypatch):
    campaign = ablation_campaign()
    results = tmp_path / "size-tiered-selection.csv"
    provenance = tmp_path / "size-tiered-selection-provenance.csv"
    results.write_text(campaign.HELD_OUT_HEADER + "\n")
    provenance.write_text(campaign.PROVENANCE_HEADER + "\n")
    before_results = results.read_bytes()
    before_provenance = provenance.read_bytes()
    monkeypatch.setattr(campaign, "held_out_schedule", lambda: [(75, 7000, "posterior")])
    monkeypatch.setattr(campaign.socket, "gethostname", lambda: "bench")
    monkeypatch.setattr(
        campaign,
        "run_job",
        lambda binary, log_dir, order, size, seed, arm: {
            "arm": arm,
            "seed": str(seed),
            "hit": "0",
            "force_calls_to_hit": "",
            "budget": str(campaign.PAPER_BUDGETS[size]),
            "best": "-1.0",
            "charged": str(campaign.PAPER_BUDGETS[size]),
            "wall_s": "1.0",
        },
    )

    replace_calls = 0
    attempted = []
    real_replace = os.replace

    def failing_replace(src, dst):
        nonlocal replace_calls
        replace_calls += 1
        attempted.append(Path(dst))
        if replace_calls == 1:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(campaign.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        campaign.run_held_out(tmp_path / "depth_ablation", tmp_path / "logs", results, provenance, "abc")
    assert attempted == [results]
    assert results.read_bytes() == before_results
    assert provenance.read_bytes() == before_provenance
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([results.name, provenance.name])


def test_held_out_redoes_run_when_the_second_rename_fails(tmp_path, monkeypatch):
    campaign = ablation_campaign()
    results = tmp_path / "size-tiered-selection.csv"
    provenance = tmp_path / "size-tiered-selection-provenance.csv"
    results.write_text(campaign.HELD_OUT_HEADER + "\n")
    provenance.write_text(campaign.PROVENANCE_HEADER + "\n")
    monkeypatch.setattr(campaign, "held_out_schedule", lambda: [(75, 7000, "posterior")])
    monkeypatch.setattr(campaign.socket, "gethostname", lambda: "bench")
    calls = []

    def fake_run_job(binary, log_dir, order, size, seed, arm):
        calls.append((size, seed, arm))
        return {
            "arm": arm,
            "seed": str(seed),
            "hit": "1",
            "force_calls_to_hit": "10",
            "budget": str(campaign.PAPER_BUDGETS[size]),
            "best": "-397.49233098",
            "charged": "10",
            "wall_s": "1.0",
        }

    monkeypatch.setattr(campaign, "run_job", fake_run_job)
    real_replace = os.replace
    replace_calls = 0

    def failing_second_replace(src, dst):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(campaign.os, "replace", failing_second_replace)
    with pytest.raises(OSError, match="disk full"):
        campaign.run_held_out(tmp_path / "depth_ablation", tmp_path / "logs", results, provenance, "abc")
    assert campaign.held_out_done(provenance, "abc") == set()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([results.name, provenance.name])

    monkeypatch.setattr(campaign.os, "replace", real_replace)
    campaign.run_held_out(tmp_path / "depth_ablation", tmp_path / "logs", results, provenance, "abc")
    assert len(calls) == 2
    assert campaign.held_out_done(provenance, "abc") == {(75, 7000, "posterior")}
    assert results.read_text().splitlines() == [
        campaign.HELD_OUT_HEADER,
        "posterior,75,7000,1,10,-397.49233098,0.00000002,10,1.0",
        "posterior,75,7000,1,10,-397.49233098,0.00000002,10,1.0",
    ]
    assert provenance.read_text().splitlines() == [
        campaign.PROVENANCE_HEADER,
        "posterior,75,7000,abc,bench",
    ]


def test_an_unrecorded_revision_without_a_reason_is_reported(monkeypatch):
    reasons = audit_mod.unrecorded_reasons()
    path = "results/size-tiered-selection-provenance.csv"
    assert (path, "code_revision") in reasons
    monkeypatch.setattr(audit_mod, "unrecorded_reasons",
                        lambda: {k: v for k, v in reasons.items() if k[0] != path})
    audit = audit_mod.Audit()
    audit_mod.manuscript_ablation(audit)
    audit_mod.revision_map(audit)
    assert [row for row in audit.drifts() if f"{path} code_revision" in row[2]]


def test_disagreeing_hf_fetcher_revisions_fail_the_audit():
    audit = audit_mod.Audit()
    fetcher = "paper/jctc/scripts/fetch_bench_data.py"
    audit.text_cache[fetcher] = audit.text(fetcher).replace(
        audit_mod.pinned_hf_revision(audit, fetcher),
        "0" * 40,
        1,
    )
    audit_mod.reproduction_records(audit)
    assert any(row[1] == fetcher and "same dataset revision" in row[2]
               for row in audit.drifts())


@pytest.mark.parametrize("target", [
    "bench/covalent-h2h/gfn2/rows.csv",
    "bench/crest-equal/rows.csv",
    "bench/gmin-panel/rows.csv",
    "bench/unreachable-revisions.csv",
    "bench/unreachable-revisions-updates.csv",
    "paper/jctc/FIGURE-SOURCE.tsv",
    "paper/jctc/FIGURE-SOURCE-updates.tsv",
    "results/size-tiered-selection.csv",
    "bench/generic-alloy/g3-relabel.csv",
    "data/hf-files.csv",
    "data/hf-files-updates.csv",
])
def test_rewriting_or_truncating_a_record_line_fails_the_audit(target):
    base_text = (ROOT / target).read_text()
    lines = base_text.splitlines(keepends=True)

    appended = audit_mod.Audit()
    appended.text_cache[target] = base_text + lines[-1]
    audit_mod.append_only_rows(appended)
    assert not [r for r in appended.drifts() if r[1] == target]

    header_rewritten = audit_mod.Audit()
    header_rewritten.text_cache[target] = lines[0].rstrip("\n") + ",extra_col\n" + "".join(lines[1:])
    audit_mod.append_only_rows(header_rewritten)
    assert any(r[1] == target and "line 1 rewritten" in r[4] for r in header_rewritten.drifts())

    row_rewritten = audit_mod.Audit()
    row_rewritten.text_cache[target] = lines[0] + lines[1].rstrip("\n") + ",x\n" + "".join(lines[2:])
    audit_mod.append_only_rows(row_rewritten)
    assert any(r[1] == target and "line 2 rewritten" in r[4] for r in row_rewritten.drifts())

    truncated = audit_mod.Audit()
    truncated.text_cache[target] = lines[0]
    audit_mod.append_only_rows(truncated)
    assert any(r[1] == target and "truncated" in r[4] for r in truncated.drifts())


def test_the_noun_analyses_is_american_and_the_verb_analyse_is_not():
    assert audit_mod.british_words("all code, analyses and text") == []
    assert audit_mod.british_words("they analyse the cluster") == ["analyse"]


def test_a_changed_declaration_is_rejected():
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.TEX)
    interest = "The authors declare no competing financial interest."
    notes = text.split("\\section*{Notes}", 1)[1].split("\\begin{suppinfo}", 1)[0]
    assert interest in notes
    declaration = notes.split(interest, 1)[1].strip()
    assert declaration
    mutated_decl = declaration.replace("publication.", "manuscript.", 1)
    assert mutated_decl != declaration
    audit.text_cache[audit_mod.TEX] = text.replace(declaration, mutated_decl, 1)
    audit_mod.manuscript_new(audit)
    assert any(
        row[2] == "the Notes declaration matches the sealed text"
        and row[4] == "changed"
        for row in audit.drifts()
    )


def test_a_declaration_away_from_the_competing_interest_sentence_is_rejected():
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = text.replace(
        "The authors declare no competing financial interest.\n",
        "The authors declare no competing financial interest.\nThe archive is public.\n",
        1,
    )
    audit_mod.manuscript_new(audit)
    assert any(
        row[2] == "the declaration follows the competing-interest sentence"
        and row[4] == "elsewhere"
        for row in audit.drifts()
    )


def test_a_removed_row_of_a_record_absent_at_head_fails_the_audit(monkeypatch):
    real_git = audit_mod.git

    def history_names_a_deleted_record(*args, **kwargs):
        result = real_git(*args, **kwargs)
        if args[:2] == ("log", "--name-only"):
            return subprocess.CompletedProcess(
                result.args, 0, result.stdout + "bench/deleted-campaign.csv\n", ""
            )
        if args[:2] == ("log", "-p") and "bench/deleted-campaign.csv" in args:
            return subprocess.CompletedProcess(
                result.args, 0, result.stdout + "-old record\n", ""
            )
        return result

    monkeypatch.setattr(audit_mod, "git", history_names_a_deleted_record)
    audit = audit_mod.Audit()
    audit_mod.append_only_rows(audit)
    assert any(row[1] == "record-file history" and row[4] == "1 removed lines"
               for row in audit.drifts())


def test_removing_a_record_line_in_an_intermediate_commit_fails_the_audit(monkeypatch):
    real_git = audit_mod.git

    def history_with_removal(*args, **kwargs):
        result = real_git(*args, **kwargs)
        if args[:2] == ("log", "-p"):
            return subprocess.CompletedProcess(result.args, 0, "--- a/bench/h2h.json\n-old record\n", "")
        return result

    monkeypatch.setattr(audit_mod, "git", history_with_removal)
    audit = audit_mod.Audit()
    audit_mod.append_only_rows(audit)
    assert any(row[1] == "record-file history" and "1 removed lines" in row[4]
               for row in audit.drifts())


def test_gfn2_rgpot_sidecar_drift_is_reported():
    audit = audit_mod.Audit()
    sidecar = audit_mod.GFN2_RGPOT_SIDECAR
    audit.text_cache[sidecar] = audit.text(sidecar).replace(
        PAPER_CONSTANTS["RgpotRevision"],
        "0" * 40,
        1,
    )
    audit_mod.covalent_gfn2(audit)
    assert any(row[1] == sidecar for row in audit.drifts())


def test_tblite_receipt_reads_package_metadata(tmp_path):
    metadata = tmp_path / "fpm.toml"
    metadata.write_text('name = "tblite"\nversion = "0.7.0"\n')
    output = tmp_path / "toolchain.json"
    subprocess.run(
        [
            "python3",
            str(ROOT / "paper/jctc/scripts/tblite_receipt.py"),
            "--metadata",
            str(metadata),
            "--output",
            str(output),
        ],
        check=True,
    )
    receipt = json.loads(output.read_text())
    assert receipt["package"] == "tblite"
    assert receipt["version"] == "0.7.0"
    assert receipt["metadata"] == "fpm.toml"
    assert receipt["metadata_sha256"] == hashlib.sha256(
        metadata.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("path", ["CITATION.cff", ".zenodo.json"])
def test_a_swapped_citation_affiliation_order_is_rejected(path):
    audit = audit_mod.Audit()
    audit.text_cache[path] = audit.text(path).replace("SURF B.V.; TurtleTech ehf", "TurtleTech ehf; SURF B.V.")
    audit_mod.manuscript_new(audit)
    assert [row for row in audit.drifts() if row[1] == path]


def _unreachable_number(audit):
    return [
        row for row in audit.drifts()
        if row[2] == "a printed number traces to a commit this repository holds"
    ]


def test_a_table_quote_of_an_unreachable_panel_number_fails():
    audit = audit_mod.Audit()
    audit.text_cache[audit_mod.TEX] = audit.text(audit_mod.TEX).replace(
        r"LJ\(_{150}\) & \(4\times 10^{6}\) & this work & 5/12",
        "LJ\\(_{150}\\) & \\(4\\times 10^{6}\\) & this work & 5/12 \\\\\n"
        "M\\(_{80}\\), \\(\\rho=6\\) & \\(4\\times 10^{6}\\) & this work & 2/12",
        1,
    )
    audit_mod.manuscript_gmin(audit)
    assert any(row[1].endswith("tab:gmin m80r6") for row in _unreachable_number(audit))


def test_a_prose_quote_of_an_unreachable_panel_number_fails():
    audit = audit_mod.Audit()
    audit.text_cache[audit_mod.TEX] = audit.text(audit_mod.TEX).replace(
        r"\label{sec:gmin}",
        "\\label{sec:gmin}\nOn M\\(_{80}\\) at \\(\\rho=6\\) this setting hits 2 of 12.\n",
        1,
    )
    audit_mod.manuscript_gmin(audit)
    assert any(
        row[1] == "paper/jctc/manuscript.tex m80r6" and row[4] == "2 of 12"
        for row in _unreachable_number(audit)
    )


def test_a_prose_quote_of_an_unreachable_panel_wall_fails():
    audit = audit_mod.Audit()
    audit.text_cache[audit_mod.TEX] = audit.text(audit_mod.TEX).replace(
        r"\label{sec:gmin}",
        "\\label{sec:gmin}\nOn M\\(_{80}\\) at \\(\\rho=6\\) the median wall is 161.4~s.\n",
        1,
    )
    audit_mod.manuscript_gmin(audit)
    assert any(row[1] == "paper/jctc/manuscript.tex m80r6" and row[4] == "161.4"
               for row in _unreachable_number(audit))


def test_a_mark_does_not_excuse_an_unreachable_panel_number():
    audit = audit_mod.Audit()
    audit.text_cache[audit_mod.TEX] = audit.text(audit_mod.TEX).replace(
        r"\label{sec:gmin}",
        "\\label{sec:gmin}\nOn M\\(_{80}\\) at \\(\\rho=6\\) this setting hits \\StalePanel{m80r6}2 of 12.\n",
        1,
    )
    audit_mod.manuscript_gmin(audit)
    assert any(row[4] == "2 of 12" for row in _unreachable_number(audit))


def test_reachable_panel_numbers_are_not_flagged_as_unheld():
    audit = audit_mod.Audit()
    audit_mod.manuscript_gmin(audit)
    assert _unreachable_number(audit) == []
    assert any(row[2] == "a printed number traces to a commit this repository holds" for row in audit.results)


def test_an_unmarked_table_quote_of_an_unreachable_panel_row_fails():
    audit = audit_mod.Audit()
    audit.text_cache[audit_mod.TEX] = audit.text(audit_mod.TEX).replace(
        r"LJ\(_{150}\) & \(4\times 10^{6}\) & this work & 5/12",
        "LJ\\(_{150}\\) & \\(4\\times 10^{6}\\) & this work & 5/12 \\\\\n"
        "M\\(_{80}\\), \\(\\rho=6\\) & \\(4\\times 10^{6}\\) & this work & 2/12",
        1,
    )
    audit_mod.manuscript_gmin(audit)
    assert any(row[1].endswith("tab:gmin m80r6") for row in _unreachable_number(audit))


def test_an_unmarked_prose_quote_of_an_unreachable_panel_row_fails():
    audit = audit_mod.Audit()
    audit.text_cache[audit_mod.TEX] = audit.text(audit_mod.TEX).replace(
        r"\label{sec:gmin}",
        "\\label{sec:gmin}\nM\\(_{80}\\) at \\(\\rho=6\\) hits 2 of 12.\n",
        1,
    )
    audit_mod.manuscript_gmin(audit)
    assert any(
        row[1] == "paper/jctc/manuscript.tex m80r6" and row[4] == "2 of 12"
        for row in _unreachable_number(audit)
    )


def test_a_stale_mark_on_a_reachable_panel_row_fails():
    """A mark on a held row does not turn that row into an unheld number."""
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.TEX)
    old = "& 12/12 & 10615.5 &"
    assert text.count(old) == 1
    audit.text_cache[audit_mod.TEX] = text.replace(old, r"& \StalePanel{lj38}12/12 & 10615.5 &", 1)
    audit_mod.manuscript_gmin(audit)
    assert not any(row[1].endswith("lj38") for row in _unreachable_number(audit))
