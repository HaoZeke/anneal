# Copyright (c) anneal developers
"""Every paper and benchmark-table number checked by scripts/paper_audit.py
matches its committed rows, and the audit notices a drifted cell."""

import importlib.util
import math
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("paper_audit", ROOT / "scripts" / "paper_audit.py")
audit_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_mod)

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
    audit_mod.covalent_panels,
    audit_mod.covalent_gfn2,
    audit_mod.ase_gofee,
    audit_mod.lj_large,
    audit_mod.bench_prose,
    audit_mod.manuscript_ase,
    audit_mod.panel_references,
    audit_mod.generated_tables,
    audit_mod.supporting_gmin,
    audit_mod.supporting_numbers,
    audit_mod.au55_superseded,
    audit_mod.manuscript_new,
    audit_mod.manuscript_followups,
    audit_mod.cover_letter,
    audit_mod.rerun_claims,
    audit_mod.rerun_check,
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


def test_script_exits_zero():
    assert audit_mod.run() == 0


def workspace_version():
    """The crate version, read the same way ``unauditable`` reads it."""
    return re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)


def test_missing_release_tag_and_referenced_remote_stop_the_audit(monkeypatch):
    real_git = audit_mod.git
    version = workspace_version()
    missing = {
        f"refs/tags/v{version}",
        "refs/remotes/origin/develop",
    }

    def without_required_refs(*args, stdin=None):
        if args[:3] == ("show-ref", "--verify", "--quiet") and args[3] in missing:
            return subprocess.CompletedProcess(["git", *args], 1, "", "")
        return real_git(*args, stdin=stdin)

    monkeypatch.setattr(audit_mod, "git", without_required_refs)
    problems = audit_mod.unauditable()
    assert f"the release tag v{version} is not present; fetch tags from origin" in problems
    assert "the remote ref origin/develop is not present; fetch develop from origin" in problems


def test_a_merged_branch_that_no_alias_names_is_not_required(monkeypatch, tmp_path):
    """The remote closure is the named branches and the optional tips, no more.

    A clone that never had, or no longer has, an optional tip is still
    auditable. A tip that is present joins the closure. A remote branch that
    neither the alias table nor the optional list names does not.
    """
    repo = tmp_path / "clone"
    (repo / "bench").mkdir(parents=True)
    (repo / "Cargo.toml").write_text('[package]\nversion = "0.0.1"\n')
    (repo / "bench" / "revision-aliases.csv").write_text(
        "local_revision,pushed_revision,branch_revision,branch,rows_commit,patch_id\n")

    def run_git(*args):
        done = subprocess.run(
            ["git", "-c", "user.name=audit", "-c", "user.email=audit@example.invalid",
             "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
            cwd=repo, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    def commit(message):
        run_git("commit", "--allow-empty", "-m", message)
        return run_git("rev-parse", "HEAD")

    run_git("init", "--quiet")
    root = commit("root")
    run_git("tag", "v0.0.1")
    for branch in ("develop", "bench/crest-gaps", "bench/gmin-sc3", "fix/gmin-sc", "fix/gmin-sc2"):
        run_git("update-ref", f"refs/remotes/origin/{branch}", root)
    run_git("checkout", "--quiet", "--detach")
    optional = commit("tip kept by an optional remote")
    run_git("checkout", "--quiet", "--detach", root)
    unnamed = commit("tip on a branch nothing names")
    run_git("checkout", "--quiet", "--detach", root)
    run_git("update-ref", "refs/remotes/origin/topic/unnamed", unnamed)

    monkeypatch.setattr(audit_mod, "ROOT", repo)
    optional_ref = "refs/remotes/origin/record/repro-pin"

    def index():
        return audit_mod.RevisionIndex((root, optional, unnamed), (), (), ())

    assert audit_mod.unauditable() == []
    without_tip = index()
    assert without_tip.on_remote(root)
    assert not without_tip.on_remote(optional)
    assert not without_tip.on_remote(unnamed), "an unnamed remote branch joined the closure"

    run_git("update-ref", optional_ref, optional)
    assert audit_mod.unauditable() == []
    with_tip = index()
    assert with_tip.on_remote(optional), "a present optional tip did not join the closure"
    assert not with_tip.on_remote(unnamed), "an unnamed remote branch joined the closure"


def test_a_verified_record_ref_joins_the_audited_closure(tmp_path, monkeypatch):
    """A commit only a verified record ref holds is in the audited and remote closures; unverified, it is not."""
    repo = tmp_path / "clone"
    repo.mkdir()

    def run_git(*args):
        done = subprocess.run(
            ["git", "-c", "user.name=audit", "-c", "user.email=audit@example.invalid",
             "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
            cwd=repo, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    run_git("init", "--quiet")
    run_git("commit", "--allow-empty", "-m", "root")
    root = run_git("rev-parse", "HEAD")
    run_git("update-ref", "refs/remotes/origin/develop", root)
    run_git("checkout", "--quiet", "--detach")
    run_git("commit", "--allow-empty", "-m", "record")
    record = run_git("rev-parse", "HEAD")
    run_git("checkout", "--quiet", "--detach", root)
    monkeypatch.setattr(audit_mod, "ROOT", repo)

    monkeypatch.setattr(audit_mod, "_verified_record_refs", lambda: ())
    unverified = audit_mod.RevisionIndex((root, record), (), (), ())
    assert not unverified.audited_exists(record)
    assert not unverified.on_remote(record)

    monkeypatch.setattr(audit_mod, "_verified_record_refs", lambda: (record,))
    verified = audit_mod.RevisionIndex((root, record), (), (), ())
    assert verified.audited_exists(record)
    assert verified.on_remote(record)
    assert verified.on_remote(root)


def test_the_sc_h2h_harness_is_pushed_through_its_record_ref(tmp_path, monkeypatch):
    need_origin()
    harness = "d59b0b6cf02fc300ff618f79d65da365128eed5a"
    audit_mod.revision_index.cache_clear()
    index = audit_mod.alias_index([harness], [])
    assert index.audited_exists(harness) and index.on_remote(harness)
    registry(tmp_path, monkeypatch, [])
    audit_mod.revision_index.cache_clear()
    index = audit_mod.alias_index([harness], [])
    assert not index.audited_exists(harness) and not index.on_remote(harness)


def test_a_drifted_table_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex.replace("& 12/12 & 10615.5 & 3591.25", "& 12/12 & 10616.5 & 3591.25")
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["median calls"]


def test_a_drifted_ase_table_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "& 246.5 & 10/12 & 0/12 & 0/12 & 0 & 12 &"
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "& 246.5 & 11/12 & 0/12 & 0/12 & 0 & 12 &")
    audit_mod.manuscript_ase(audit)
    audit_mod.generated_tables(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["recommended hits", "tab:ase is the summariser's block"]


def test_a_drifted_morse_ase_table_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "& 54.135 & 0.037 & 0.037 \\\\"
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "& 54.135 & 0.037 & 0.37 \\\\")
    audit_mod.manuscript_ase(audit)
    audit_mod.generated_tables(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["Fisher p against ase_mh", "tab:ase-morse is the summariser's block"]


def test_a_rank_test_in_the_ase_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Every basin-hopping and minima-hopping run of"
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "A Mann--Whitney test is not run. " + old)
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
    audit.text_cache[SC_N12] = text.replace("| sc126-38 | gmin | 61fa67cf | 12/12 |", "| sc126-38 | gmin | 61fa67cf | 1/12 |")
    audit_mod.head_to_head(audit)
    assert [(r[1].split()[-2:], r[2]) for r in audit.drifts()] == [(["sc126-38", "gmin"], "hits")]


def test_a_dropped_sutton_chen_rerun_row_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(SC_N12)
    old = "| sc108-38 | gmin | 61fa67cf | 3/12 | 400000 |  |  |\n"
    assert old in text
    audit.text_cache[SC_N12] = text.replace(old, "")
    audit_mod.head_to_head(audit)
    assert [r[2] for r in audit.drifts()] == ["table row sc108-38 gmin"]


def test_a_rerun_missing_from_the_sutton_chen_readme_is_reported():
    audit = audit_mod.Audit()
    summary = audit.rows(audit_mod.SC_RERUN_SUMMARY)
    summary.append({**summary[0], "system": "sc96-99", "arm": "gmin", "hits_over_n": "0/12"})
    audit_mod.head_to_head(audit)
    assert "table row sc96-99 gmin" in {r[2] for r in audit.drifts()}


def test_a_rerun_without_its_tests_is_reported():
    audit = audit_mod.Audit()
    summary = audit.rows(audit_mod.SC_RERUN_SUMMARY)
    row = next(r for r in summary if r["system"] == "sc126-38" and r["arm"] == "anneal")
    row["fisher_p_vs_gmin"] = ""
    audit_mod.head_to_head(audit)
    assert ("Fisher p", "1") in [(r[2], r[3]) for r in audit.drifts()]


SC_RERUN_ROWS = "bench/paper-rerun/sc-h2h/rows.csv"
SIDE_COMMIT = "e3ed7dadfce4f0c0aa3b9616f6f07dd9b3171e0a"


def test_a_sutton_chen_summary_off_its_rows_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows(SC_RERUN_ROWS)
    for row in rows:
        if row["system"] == "sc126-38" and row["arm"] == "anneal" and row["hit"] == "1":
            row["evals_to_hit"] = str(int(row["evals_to_hit"]) + 400)
    audit_mod.head_to_head(audit)
    drifted = {(r[1], r[2]) for r in audit.drifts()}
    assert (f"{audit_mod.SC_RERUN_SUMMARY} sc126-38 anneal", "calls_median") in drifted


def test_a_sutton_chen_summary_cell_off_its_rows_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows(SC_RERUN_ROWS)
    row = next(r for r in rows if r["system"] == "sc108-38" and r["arm"] == "gmin" and r["hit"] == "0")
    row["hit"] = "1"
    audit_mod.head_to_head(audit)
    drifted = {(r[1], r[2]) for r in audit.drifts()}
    assert {(f"{audit_mod.SC_RERUN_SUMMARY} sc108-38 gmin", "hits"),
            (f"{audit_mod.SC_RERUN_SUMMARY} sc108-38 anneal", "fisher_p_vs_gmin")} <= drifted


def test_a_missing_sutton_chen_rerun_row_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows(SC_RERUN_ROWS)
    assert len(rows) == 216
    dropped = rows.pop(5)
    audit_mod.head_to_head(audit)
    drifted = {(r[2], r[3]) for r in audit.drifts()}
    assert ("rows", "215") in drifted
    assert (f"manifest cell {dropped['system']} {dropped['arm']} {dropped['seed']}", "0") in drifted


def test_the_sutton_chen_code_revision_comes_from_the_rows():
    audit = audit_mod.Audit()
    for row in audit.rows(SC_RERUN_ROWS):
        row["code_revision"] = SIDE_COMMIT
    audit_mod.head_to_head(audit)
    labels = [r[2] for r in audit.drifts()]
    assert "the rerun code revision is an ancestor of HEAD" in labels
    assert labels.count("code revision") == 36


@pytest.mark.parametrize("shown", ["61fa67cfNOT_A_COMMIT", "61fa67cf0", "61fa", "develop"])
def test_a_sutton_chen_code_revision_must_name_the_commit(shown):
    audit = audit_mod.Audit()
    path = "bench/paper-rerun/sc-h2h/README.md"
    audit.text_cache[path] = audit.text(path).replace("| sc126-38 | anneal | 61fa67cf |", f"| sc126-38 | anneal | {shown} |")
    audit_mod.head_to_head(audit)
    assert [(r[1], r[2], r[3]) for r in audit.drifts()] == [(f"{path}:21 sc126-38 anneal", "code revision", shown)]


def test_the_sutton_chen_rerun_rows_are_provenance_checked():
    audit = audit_mod.Audit()
    audit_mod.head_to_head(audit)
    assert SC_RERUN_ROWS in audit.read
    audit.read[SC_RERUN_ROWS][7]["harness_revision"] = "0" * 40
    gaps = audit_mod.provenance(audit)
    assert "revision 00000000 is not in this repository" in gaps[SC_RERUN_ROWS]
    assert ("unpushed revision 00000000", "not listed") in {(r[2], r[3]) for r in audit.drifts()}


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
    audit.text_cache[SC_N12] = text.replace("| sc126-55 | anneal | 61fa67cf | 12/12 | 25529 |",
                                            "| sc126-55 | anneal | 61fa67cf | 12/12 | 25530 |")
    audit_mod.head_to_head(audit)
    assert [r[2] for r in audit.drifts()] == ["calls median"]


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


def test_an_unpushed_revision_needs_a_reason(tmp_path, monkeypatch):
    unpushed = dict(audit_mod.UNPUSHED)
    del unpushed["bf09f7a2b7a6bd6359898db675eeafefcd20548b"]
    unpushed["e09e0487c52b7f4789bc556ee7fe07ead4999824"] = "listed although its alias holds"
    monkeypatch.setattr(audit_mod, "UNPUSHED", unpushed)
    audit, _ = aliased_audit(tmp_path, monkeypatch, lambda table: table, (ASE_GOFEE,))
    assert [(r[2], r[3], r[4]) for r in audit.drifts()] == [
        ("unpushed revision bf09f7a2", "not listed", "no pushed equivalent"),
        ("unpushed revision e09e0487", "listed although its alias holds", "resolved"),
    ]


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


def _audit_with_tags(tex):
    audit = audit_mod.Audit()
    for path in audit_mod.RERUN_FILES:
        audit.text_cache[path] = tex if path == audit_mod.TEX else ""
    audit.text_cache["paper/jctc/macros.tex"] = "\\newcommand{\\rerun}[1]{\\textbf{[RERUN: item #1]}}\n"
    return audit


def test_a_rerun_tag_passes_the_normal_audit_and_fails_pre_submission():
    tex = "GMIN hits 9 of 12. \\rerun{5}\n% \\rerun{6} in a comment\n"
    assert audit_mod.rerun_tags(tex) == [(1, "5")]
    normal = _audit_with_tags(tex)
    audit_mod.rerun_check(normal)
    assert not normal.drifts()
    strict = _audit_with_tags(tex)
    audit_mod.rerun_check(strict, pre_submission=True)
    assert [row[2] for row in strict.drifts()] == ["[RERUN: item 5] remains (GMIN and ASE caps and tuning)"]


def test_a_rerun_tag_must_name_a_known_item():
    audit = _audit_with_tags("a number \\rerun{9}\n")
    audit_mod.rerun_check(audit)
    assert len(audit.drifts()) == 1


def test_no_tag_passes_pre_submission():
    audit = _audit_with_tags("clean text\n")
    audit_mod.rerun_check(audit, pre_submission=True)
    assert not audit.drifts()


def test_a_replaced_claim_without_its_tag_is_reported():
    claims = ("GMIN hits 9 of 12",)
    fenced = "GMIN hits 9 of 12, with p of 0.21.~\\rerun{5}\nNext sentence.\n"
    assert audit_mod.untagged_claims(fenced, claims[0], "5") == (1, 1)
    bare = "GMIN hits 9 of 12, with p of 0.21.\nNext sentence.~\\rerun{5}\n"
    assert audit_mod.untagged_claims(bare, claims[0], "5") == (1, 0)
    assert audit_mod.untagged_claims(bare, "a missing result", "5") == (0, 0)
    assert audit_mod.untagged_claims(fenced, claims[0], "6") == (1, 0)


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


def test_the_british_spelling_of_labeled_is_caught():
    for word in ("labelled", "labelling", "modelled", "travelled"):
        assert audit_mod.BRITISH.search(word)
    assert not audit_mod.BRITISH.search("labeled")


IPI_FORM = "https://github.com/afiorent/i-pi.git@4e86f5fdc1583ef21f105fc2bed96a279b2430c1"


def test_a_side_branch_revision_is_not_an_ancestor():
    assert audit_mod.git("cat-file", "-t", "e3ed7dad").stdout.strip() == "commit"
    assert not audit_mod.ancestor_of_head("e3ed7dad")
    assert not audit_mod.harness_reachable("e3ed7dad")


def test_the_develop_revision_is_an_ancestor():
    assert audit_mod.ancestor_of_head("61fa67cf")
    assert audit_mod.harness_reachable("61fa67cf")


def test_a_qualified_external_revision_is_a_repo_url_and_a_full_sha():
    assert audit_mod.external_form(IPI_FORM) is True
    assert audit_mod.external_form("https://github.com/afiorent/i-pi.git@4e86f5fd") is False
    assert audit_mod.external_form("4e86f5fd") is None
    assert not audit_mod.harness_reachable("4e86f5fd")
    assert audit_mod.harness_reachable(IPI_FORM)
    assert audit_mod.external_pin("aac1548a")
    assert audit_mod.harness_reachable("aac1548a")


IPI_PIN = IPI_FORM.split("@")[1]


@pytest.mark.parametrize("form", [
    "https://example.com/fake/code.git@" + "0" * 40,
    "https://github.com/afiorent/i-pi.git@" + "0" * 40,
    "https://github.com/someone/i-pi.git@" + IPI_PIN,
])
def test_an_unpinned_external_revision_is_rejected(form):
    assert audit_mod.external_form(form) is False
    assert not audit_mod.harness_reachable(form)
    audit = audit_mod.Audit()
    path = "bench/pimc-lj38/README.md"
    audit.text_cache[path] = audit.text(path).replace(IPI_FORM, form)
    audit_mod.printed_harnesses(audit)
    assert ("qualified external revision", form) in {(r[2], r[3]) for r in audit.drifts()}


def test_a_41_digit_external_revision_is_rejected():
    long = IPI_FORM + "0"
    assert audit_mod.external_form(long) is False
    audit = audit_mod.Audit()
    path = "bench/pimc-lj38/README.md"
    audit.text_cache[path] = audit.text(path).replace(IPI_FORM, long)
    audit_mod.printed_harnesses(audit)
    assert ("qualified external revision", long) in {(r[2], r[3]) for r in audit.drifts()}
    assert not [r for r in audit.results if r[1].startswith(path) and r[2] == "qualified external revision" and r[0] == "OK"]


def test_an_inline_external_revision_is_stripped_from_prose():
    line = f"Rows at `{IPI_FORM}` hit at a median of 12 calls."
    assert IPI_PIN not in audit_mod._EXTERNAL_FORM.sub(" ", line)
    audit = audit_mod.Audit()
    path = "bench/pimc-lj38/README.md"
    audit.text_cache[path] = audit.text(path) + f"\n{line}\n"
    audit_mod.printed_harnesses(audit)
    assert [r for r in audit.drifts() if r[1].startswith(path)] == []


def prose_flags(extra):
    audit = audit_mod.Audit()
    path = "bench/pimc-lj38/README.md"
    text = audit.text(path)
    audit.text_cache[path] = text + "\n" + extra + "\n"
    first = text.count("\n") + 2
    audit_mod.printed_harnesses(audit)
    return sorted((int(r[1].rsplit(":", 1)[1]) - first, r[4]) for r in audit.drifts() if r[1].startswith(f"{path}:"))


@pytest.mark.parametrize("word", ["acceded", "defaced", "effaced", "deedeed", "Bacceded"])
def test_a_hex_letter_word_is_not_a_revision(word):
    assert prose_flags(f"The request {word} at a median of 12 calls.") == []


def test_a_revision_wrapped_onto_the_next_line_reaches_its_number():
    assert prose_flags("Rows at e3ed7dad\nreach a median of 400 calls.") == [(0, "e3ed7dad 400")]


def test_a_revision_carries_through_its_paragraph():
    flags = prose_flags("Rows at e3ed7dad hit 3 of 12 seeds. The anneal arm\nhits 5 of 12 at a median of 400 calls.")
    assert flags == [(0, "e3ed7dad 400"), (0, "e3ed7dad 5 of 12")]


def test_a_revision_does_not_carry_past_its_paragraph():
    assert prose_flags("Rows at e3ed7dad.\n\nAnneal reaches a median of 400 calls.") == []


def test_a_reachable_revision_ends_the_carry():
    assert prose_flags("Rows at e3ed7dad. Rows at 61fa67cf reach a median of 9 calls.") == []


def bench_flags(tmp_path, monkeypatch, files):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    monkeypatch.setattr(audit_mod, "ROOT", tmp_path)
    monkeypatch.setattr(audit_mod, "_verified_aliases", lambda: (), raising=False)
    monkeypatch.setattr(audit_mod, "_verified_record_refs", lambda: (), raising=False)
    return sorted((w.split(":")[0], rev, number) for w, rev, number in
                  audit_mod._printed_readme_numbers(audit_mod.Audit(), []))


UNREACHABLE_ROW = "| lj38 | anneal at `0badc0de51` (rerun) | 3/12 | 812 |"


def harness_results(tmp_path, monkeypatch, files, listed):
    bench_flags(tmp_path, monkeypatch, files)
    monkeypatch.setattr(audit_mod, "record_ref_checks", lambda audit: None)
    monkeypatch.setattr(audit_mod, "EXTERNAL_FORM_SOURCES", ())
    monkeypatch.setattr(audit_mod, "UNVERIFIABLE", listed, raising=False)
    audit = audit_mod.Audit()
    audit_mod.printed_harnesses(audit)
    return audit


def unverifiable_readme():
    return {"bench/x/README.md": f"| System | Code | Hits | Calls median |\n|---|---|---|---|\n{UNREACHABLE_ROW}\n"}


def test_a_listed_unverifiable_row_is_still_checked_and_passes(tmp_path, monkeypatch):
    listed = [("bench/x/README.md", UNREACHABLE_ROW, ("0badc0de51",), "no ref holds it; no paper number reads it")]
    audit = harness_results(tmp_path, monkeypatch, unverifiable_readme(), listed)
    assert audit.drifts() == []
    marked = [r for r in audit.results if r[2] == "a printed number comes from a reachable revision"]
    assert [(r[0], r[1], r[3], r[4]) for r in marked] == [
        ("OK", "bench/x/README.md:3", "unverifiable: no ref holds it; no paper number reads it", "0badc0de51 lj38 Hits=3/12"),
        ("OK", "bench/x/README.md:3", "unverifiable: no ref holds it; no paper number reads it",
         "0badc0de51 lj38 Calls median=812"),
    ]


def test_an_unlisted_unverifiable_row_still_drifts(tmp_path, monkeypatch):
    audit = harness_results(tmp_path, monkeypatch, unverifiable_readme(), [])
    assert sorted(r[4] for r in audit.drifts()) == ["0badc0de51 lj38 Calls median=812", "0badc0de51 lj38 Hits=3/12"]


def test_an_unverifiable_entry_on_another_revision_clears_nothing(tmp_path, monkeypatch):
    listed = [("bench/x/README.md", UNREACHABLE_ROW, ("0badc0de99",), "another revision")]
    audit = harness_results(tmp_path, monkeypatch, unverifiable_readme(), listed)
    assert sorted(r[2] for r in audit.drifts()) == ["a printed number comes from a reachable revision"] * 2 + [
        "an unverifiable row is still printed and flagged"]


def test_a_stale_unverifiable_entry_is_flagged(tmp_path, monkeypatch):
    listed = [("bench/x/README.md", "| lj38 | anneal at `0badc0de51` (rerun) | 4/12 | 812 |", ("0badc0de51",), "edited")]
    audit = harness_results(tmp_path, monkeypatch, unverifiable_readme(), listed)
    assert ("bench/x/README.md", "an unverifiable row is still printed and flagged") in {(r[1], r[2]) for r in audit.drifts()}
    assert len(audit.drifts()) == 3


# README rows whose numbers the manuscript prints: tab:ase (manuscript.tex 1233-1238) reads the recommended, basin
# hopping and minima hopping rows and the comparisons of recommended with each ASE arm; tab:gmin (1149-1150) and the
# Au55 prose read the Au55 rows of the GMIN panel, which bench/occat-h2h repeats.
def paper_printed_row(path, row):
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    if re.fullmatch(r"bench/ase-(lj38|lj55|lj98|lj104|lj150|au55)/README\.md", path):
        return cells[0] in ("recommended", "basin hopping", "minima hopping") or cells[0].startswith("recommended against")
    if path in ("bench/gmin-panel/README.md", "bench/occat-h2h/README.md"):
        return "au55" in row.lower() or (path == "bench/gmin-panel/README.md" and cells[:2] == ["", ""])
    return False


def test_no_row_the_paper_prints_is_listed_unverifiable():
    listed = [(path, row) for path, row, _revs, _why in audit_mod.UNVERIFIABLE]
    assert listed
    assert [item for item in listed if paper_printed_row(*item)] == []


def test_every_unverifiable_entry_names_a_revision_and_a_reason():
    for path, row, revisions, why in audit_mod.UNVERIFIABLE:
        assert path.startswith("bench/") and path.endswith("README.md")
        assert row in [line.strip() for line in (ROOT / path).read_text().splitlines()]
        assert revisions and all(re.fullmatch(r"[0-9a-f]{7,40}", r) for r in revisions)
        assert len(why) > 20


def test_a_revision_inside_a_table_cell_is_read(tmp_path, monkeypatch):
    readme = "| System | Code | Hits | Calls median |\n|---|---|---|---|\n| lj38 | anneal at `0badc0de51` (rerun) | 3/12 | 812 |\n"
    assert bench_flags(tmp_path, monkeypatch, {"bench/x/README.md": readme}) == [
        ("bench/x/README.md", "0badc0de51", "lj38 Calls median=812"),
        ("bench/x/README.md", "0badc0de51", "lj38 Hits=3/12"),
    ]


@pytest.mark.parametrize("shown", ["1,974.5", "2.5 s", "18.4 µs", "−13.79658 eV", "**12/12**", "41%"])
def test_a_number_with_commas_or_units_is_read(tmp_path, monkeypatch, shown):
    readme = f"| System | Harness | Calls median |\n|---|---|---|\n| lj38 | 0badc0de52 | {shown} |\n"
    assert bench_flags(tmp_path, monkeypatch, {"bench/x/README.md": readme}) == [
        ("bench/x/README.md", "0badc0de52", f"lj38 Calls median={shown}")]


CSV_README = "Rows are in `rows.csv`.\n\n| System | Arm | Hits |\n|---|---|---|\n| LJ38 | anneal | 3/12 |\n|  | GMIN | 1/12 |\n"


@pytest.mark.parametrize("field", ["code_revision", "anneal_head", "anneal_revision", "release_revision"])
def test_a_csv_revision_reaches_its_readme_table(tmp_path, monkeypatch, field):
    rows = f"system,arm,hits,{field}\nlj38,anneal,3,0badc0de53\nlj38,gmin,1,0badc0de54\nlj55,anneal,1,0badc0de55\n"
    assert bench_flags(tmp_path, monkeypatch, {"bench/x/README.md": CSV_README, "bench/x/rows.csv": rows}) == [
        ("bench/x/README.md", "0badc0de53", "LJ38 anneal Hits=3/12"),
        ("bench/x/README.md", "0badc0de54", "GMIN Hits=1/12"),
    ]


def test_a_table_without_a_system_column_reaches_its_csv(tmp_path, monkeypatch):
    readme = "| arm | hits | median wall s |\n| --- | --- | --- |\n| anneal | 10/12 | 0.98 |\n| basin hopping | 0/12 | 4.65 |\n"
    rows = "system,arm,hit,code_revision\nlj38,anneal,1,0badc0de56\nlj38,ase_bh,0,0badc0de57\n"
    assert bench_flags(tmp_path, monkeypatch, {"bench/x/README.md": readme, "bench/x/rows.csv": rows}) == [
        ("bench/x/README.md", "0badc0de56", "anneal hits=10/12"),
        ("bench/x/README.md", "0badc0de56", "anneal median wall s=0.98"),
        ("bench/x/README.md", "0badc0de56", "basin hopping hits=0/12"),
        ("bench/x/README.md", "0badc0de56", "basin hopping median wall s=4.65"),
        ("bench/x/README.md", "0badc0de57", "basin hopping hits=0/12"),
        ("bench/x/README.md", "0badc0de57", "basin hopping median wall s=4.65"),
    ]


def test_a_csv_in_a_subdirectory_reaches_its_readme_table(tmp_path, monkeypatch):
    readme = "Rows are in `gfn2/rows.csv`.\n\n| system | code | hits |\n|---|---|---|\n| Si10 | anneal G | 6/6 |\n"
    rows = "system,code,hit,code_revision\nsi10,anneal G,1,0badc0de58\nsi13,anneal G,1,0badc0de59\n"
    assert bench_flags(tmp_path, monkeypatch, {"bench/x/README.md": readme, "bench/x/gfn2/rows.csv": rows}) == [
        ("bench/x/README.md", "0badc0de58", "Si10 anneal G hits=6/6")]


def test_a_csv_backs_only_the_tables_of_its_section(tmp_path, monkeypatch):
    readme = ("## Before\n\n| system | code | hits |\n|---|---|---|\n| Si10 | anneal G | 6/6 |\n\n"
              "## Gate\n\nRows are in `gate.csv`.\n\n| system | code | hits |\n|---|---|---|\n| Si10 | anneal G | 5/6 |\n")
    rows = "system,code,hit,code_revision\nsi10,anneal G,1,0badc0de5a\n"
    assert bench_flags(tmp_path, monkeypatch, {"bench/x/README.md": readme, "bench/x/gate.csv": rows}) == [
        ("bench/x/README.md", "0badc0de5a", "Si10 anneal G hits=5/6")]


def readme_revisions():
    found = audit_mod._printed_readme_numbers(audit_mod.Audit(), [])
    return {(where.split(":")[0], revision[:8]) for where, revision, _number in found}


def test_unreachable_revisions_in_the_bench_readmes_are_found():
    found = readme_revisions()
    assert {
        ("bench/gmin-panel/README.md", "1ce3ab64"),
        ("bench/ase-au55/README.md", "1ce3ab64"),
        ("bench/ase-au55/README.md", "fc1af066"),
        ("bench/ase-lj38/README.md", "5a7f629d"),
        ("bench/ase-lj38/README.md", "b49341e3"),
        ("bench/ase-lj55/README.md", "36548663"),
        ("bench/ase-nonlj/README.md", "90b88235"),
        ("bench/covalent-h2h/README.md", "ce1e7a46"),
        ("bench/covalent-h2h/README.md", "eb7f8999"),
        ("bench/covalent-h2h/README.md", "a52726af"),
    } <= found


def test_openconf_local_revisions_are_read_through_their_aliases(monkeypatch):
    local = {("bench/openconf-h2h/conf-fix3/README.md", "cf660e19"),
             ("bench/openconf-h2h/conf-fix4/README.md", "2a5b4553"),
             ("bench/openconf-h2h/paired/README.md", "d178bd15")}
    assert not local & readme_revisions()
    monkeypatch.setattr(audit_mod, "_verified_aliases", lambda: (), raising=False)
    assert local <= readme_revisions()


REPRO_PIN = "9a30f3190b3bb7040c675a904f8388eb68224151"


def need_origin():
    """The record tests read origin. Off CI an unreachable origin skips them; on CI it fails them."""
    out = git("ls-remote", "origin", "refs/heads/record/repro-pin")
    if out.returncode != 0 or not out.stdout.strip():
        if os.environ.get("CI"):
            pytest.fail(f"origin is unreachable on CI: {out.stderr.strip()}")
        pytest.skip("origin is unreachable")


def registry(tmp_path, monkeypatch, rows):
    path = tmp_path / "record-refs.csv"
    path.write_text("ref,sha,tree\n" + "".join(f"{ref},{sha},{tree}\n" for ref, sha, tree in rows))
    monkeypatch.setattr(audit_mod, "RECORD_REFS", str(path), raising=False)


def record_drifts():
    audit = audit_mod.Audit()
    audit_mod.printed_harnesses(audit)
    return {r[2] for r in audit.drifts() if r[1].endswith("record-refs.csv")}


def test_a_revision_on_a_registered_record_ref_is_reachable(tmp_path, monkeypatch):
    need_origin()
    assert git("merge-base", "--is-ancestor", REPRO_PIN, "HEAD").returncode == 1
    assert audit_mod.harness_reachable("9a30f319")
    registry(tmp_path, monkeypatch, [])
    assert not audit_mod.harness_reachable("9a30f319")


def test_a_revision_on_the_cov_floor2_record_ref_is_reachable(tmp_path, monkeypatch):
    need_origin()
    assert git("merge-base", "--is-ancestor", "c1c02d1646a7", "HEAD").returncode == 1
    assert audit_mod.harness_reachable("c1c02d1646a7")
    registry(tmp_path, monkeypatch, [])
    assert not audit_mod.harness_reachable("c1c02d1646a7")


@pytest.mark.parametrize("revision", [SIDE_COMMIT, "b7e90ce985ff"])
def test_a_revision_on_no_registered_record_ref_is_flagged(revision):
    # e3ed7dad is on origin/record/bench-sc-h2h-n12, which is not registered;
    # b7e90ce9 is only on origin/bench/cov-floor2, which is not a record ref.
    need_origin()
    assert audit_mod.commit_of(revision)
    assert not audit_mod.harness_reachable(revision)


def test_a_registry_row_off_origin_is_flagged(tmp_path, monkeypatch):
    need_origin()
    parent = git("rev-parse", f"{REPRO_PIN}^").stdout.strip()
    tree = git("rev-parse", f"{parent}^{{tree}}").stdout.strip()
    registry(tmp_path, monkeypatch, [("record/repro-pin", parent, tree)])
    assert record_drifts() == {f"record/repro-pin is on origin at {parent[:8]}"}
    assert not audit_mod.harness_reachable("9a30f319")


def test_a_registry_row_with_another_tree_is_flagged(tmp_path, monkeypatch):
    need_origin()
    registry(tmp_path, monkeypatch, [("record/repro-pin", REPRO_PIN, "0" * 40)])
    assert record_drifts() == {"record/repro-pin has tree 00000000"}
    assert not audit_mod.harness_reachable("9a30f319")


def test_a_registry_row_origin_does_not_hold_is_flagged(tmp_path, monkeypatch):
    need_origin()
    registry(tmp_path, monkeypatch, [("record/no-such-ref-r184", "1" * 40, "2" * 40)])
    assert record_drifts() == {"record/no-such-ref-r184 is on origin at 11111111", "record/no-such-ref-r184 fetches",
                               "record/no-such-ref-r184 has tree 22222222"}


def test_a_registry_row_that_is_not_a_record_ref_is_flagged(tmp_path, monkeypatch):
    need_origin()
    head = git("rev-parse", "origin/develop").stdout.strip()
    registry(tmp_path, monkeypatch, [("develop", head, git("rev-parse", f"{head}^{{tree}}").stdout.strip())])
    assert "develop is a record ref with a full SHA" in record_drifts()


def test_printed_revisions_are_ancestors_or_qualified():
    audit = audit_mod.Audit()
    audit_mod.printed_harnesses(audit)
    assert audit.drifts() == []
    assert any(row[2] == "qualified external revision" and row[3] == IPI_FORM for row in audit.results)
    assert any(row[1] == "scripts/pimc_lj38/run_campaign.py" for row in audit.results)


def test_a_short_external_revision_is_reported():
    audit = audit_mod.Audit()
    path = "bench/pimc-lj38/README.md"
    text = audit.text(path)
    short = "https://github.com/afiorent/i-pi.git@4e86f5fd"
    audit.text_cache[path] = text.replace(IPI_FORM, short)
    audit_mod.printed_harnesses(audit)
    assert any(row[2] == "qualified external revision" and row[3] == short for row in audit.drifts())


def test_an_unreachable_printed_revision_is_reported():
    audit = audit_mod.Audit()
    path = "bench/metal-cluster-h2h/sc-n12/README.md"
    text = audit.text(path)
    audit.text_cache[path] = text.replace("| sc126-38 | anneal | 61fa67cf |", "| sc126-38 | anneal | e3ed7dad |")
    audit_mod.printed_harnesses(audit)
    assert any(row[4].startswith("e3ed7dad") for row in audit.drifts())
