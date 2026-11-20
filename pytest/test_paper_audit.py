# Copyright (c) anneal developers
"""Every paper and benchmark-table number checked by scripts/paper_audit.py
matches its committed rows, and the audit notices a drifted cell."""

import importlib.util
import math
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("paper_audit", ROOT / "scripts" / "paper_audit.py")
audit_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_mod)

CHECKS = (
    audit_mod.paired_statistics,
    audit_mod.manuscript_gmin,
    audit_mod.withdrawn_benchmarks,
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
    audit_mod.manuscript_knockouts,
    audit_mod.manuscript_priors,
    audit_mod.panel_references,
    audit_mod.generated_tables,
    audit_mod.supporting_numbers,
    audit_mod.manuscript_new,
    audit_mod.manuscript_followups,
    audit_mod.cover_letter,
    audit_mod.science_claims,
    audit_mod.rerun_claims,
    audit_mod.rerun_check,
    audit_mod.todo_check,
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


def test_a_drifted_table_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "this work & 12/12 & 0 & 12 & 0 & 5722 & 5372.5"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("5722", "5723"))
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["median calls"]


def test_ase_rows_of_a_non_ancestor_are_reported(monkeypatch):
    audit = audit_mod.Audit()
    path = f"{audit_mod.ASE_DATA}/lj98/rows.csv"
    rows = [dict(r) for r in audit.rows(path)]
    rows[0]["code_revision"] = "9a30f3190b3bb7040c675a904f8388eb68224151"
    audit.read[path] = rows
    audit_mod.manuscript_ase(audit)
    assert "every row is the PROVENANCE ancestor" in {r[2] for r in audit.drifts()}


def test_a_drifted_ase_table_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "& 289.0 & 7/12 & 0/12 & 0/12 & 0 & 12 &"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "& 289.0 & 8/12 & 0/12 & 0/12 & 0 & 12 &")
    audit_mod.manuscript_ase(audit)
    audit_mod.generated_tables(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["recommended hits", "tab:ase is the summariser's block"]


def test_a_drifted_morse_ase_table_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "& 47.556 & \\(1.3\\times 10^{-3}\\) & \\(1.3\\times 10^{-3}\\) \\\\"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "& 47.556 & \\(1.3\\times 10^{-3}\\) & 0.37 \\\\")
    audit_mod.manuscript_ase(audit)
    audit_mod.generated_tables(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["Fisher p against ase_mh", "tab:ase-morse is the summariser's block"]


def test_a_rank_test_in_the_ase_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "ASE minima hopping is wall-capped"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "A Mann--Whitney test is not run. " + old)
    audit_mod.manuscript_ase(audit)
    assert [r[2] for r in audit.drifts()] == ["sec:ase reports Fisher only"]


def test_a_returning_endpoint_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "\\section{Conclusions}"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "\\subsection{Endpoints}\n\\label{sec:endpoints}\n" + old)
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["the endpoint-energy section is not printed"]


def test_a_trap_in_the_results_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed."
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old + " The misses end in a trap.")
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["the results and conclusions do not say trap"]


def test_a_drifted_template_count_is_reported():
    audit = audit_mod.Audit()
    tex = audit_mod.collapse(audit.text(audit_mod.TEX))
    old = "at \\(\\rho=14\\) from 12 of 12 to 4,"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("to 4,", "to 3,"))
    audit_mod.manuscript_knockouts(audit)
    assert {r[2] for r in audit.drifts()} == {"prose"}


def test_a_withdrawn_knockout_in_the_text_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex + "\n% Section~\\ref{sec:k-priors}\n"
    audit_mod.manuscript_knockouts(audit)
    assert [r[2] for r in audit.drifts()] == ["withdrawn knockout text stays out: sec:k-priors"]


def test_the_retired_rigid_restart_sentence_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex + "\nIt quenches a new rigid start and stands there.\n"
    audit_mod.rigid_stall(audit)
    assert [r[2] for r in audit.drifts()] == ["retired rigid-stall wording stays out: quenches a new rigid start"]


def test_a_rigid_stall_constant_that_drifts_is_reported():
    audit = audit_mod.Audit()
    path = "src/methods/stall_archive.rs"
    audit.text_cache[path] = audit.text(path).replace("pub const MEMORY: usize = 4;", "pub const MEMORY: usize = 5;")
    audit_mod.rigid_stall(audit)
    assert [r[2] for r in audit.drifts()] == ["barred stall destinations"]


def test_a_template_era_row_in_the_text_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex + "\n% revision \\texttt{c2de652d}\n"
    audit_mod.presets_and_templates(audit)
    assert [r[2] for r in audit.drifts()] == ["no table is presented as c2de652d"]


def test_a_table_without_its_preset_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "This code is \\texttt{Config::recommended\\_tip4p}\\((N)\\) of revision \\texttt{e1a69a57}"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "This code is of revision \\texttt{e1a69a57}")
    audit_mod.presets_and_templates(audit)
    assert [r[2] for r in audit.drifts()] == ["tab:tip4p-gmin names its preset"]


def test_every_todo_marker_fails_both_audits():
    clean = audit_mod.Audit()
    for path in (audit_mod.TEX, audit_mod.SI, audit_mod.COVER):
        assert not audit_mod.TODO_TAG.findall(clean.text(path))
    audit_mod.todo_check(clean, pre_submission=True)
    assert not clean.drifts()
    for strict in (False, True):
        audit = audit_mod.Audit()
        tex = audit.text(audit_mod.TEX)
        audit.text_cache[audit_mod.TEX] = tex.replace("\\section{Conclusions}", "\\todo{G1}\n\\section{Conclusions}")
        audit_mod.todo_check(audit, pre_submission=strict)
        assert sorted(r[2] for r in audit.drifts()) == ["[TODO: G1] remains (unknown)", "todo markers"]


def test_a_hedge_that_defers_a_claim_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed."
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old + " A longer run will test that.")
    audit_mod.todo_check(audit)
    assert [r[2] for r in audit.drifts()] == ["no hedge: will test"]


def test_a_rerun_tag_in_the_manuscript_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed.\n"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old[:-1] + "~\\rerun{5}\n", 1)
    audit_mod.todo_check(audit)
    audit_mod.rerun_check(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["[RERUN: item 5] remains (unknown)", "rerun tags",
                                                   "the tag names an item that waits on a rerun"]


def test_a_cover_sentence_off_the_abstract_is_reported():
    audit = audit_mod.Audit()
    cover = audit.text(audit_mod.COVER)
    old = "The title and the content have changed."
    assert old in cover
    audit.text_cache[audit_mod.COVER] = cover.replace(old, old + "\nGMIN hits LJ\\(_{150}\\) on 9 of 12.")
    audit_mod.cover_letter(audit)
    assert [r[2] for r in audit.drifts()] == [
        "a cover-letter sentence with a number is a sentence of the abstract",
        "the only cover-letter sentences with a number are the preprint and graphic lines"]


def test_a_drifted_gmin_panel_summary_cell_is_reported():
    audit = audit_mod.Audit()
    summary = audit.rows("bench/gmin-panel/summary.csv")
    au = next(r for r in summary if r["system"] == "au55")
    au["gmin_hits"] = "0"
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["gmin_hits"]


def test_a_drifted_paired_wall_p_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "& 0.0625 & 0.09 to 0.68 & 0.021 & \\(4.9\\times 10^{-4}\\) \\\\"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("10^{-4}\\) \\\\", "10^{-3}\\) \\\\"))
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["Wilcoxon p wall"]


def test_an_unpaired_fisher_p_in_the_gmin_table_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "& 0.0625 & 0.09 to 0.68 & 0.021 &"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "& 0.037 & 0.09 to 0.68 & 0.021 &")
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["exact McNemar p"]


def test_a_paired_count_that_drifts_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "this work & 12/12 & 5 & 7 & 0 &"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "this work & 12/12 & 4 & 8 & 0 &")
    audit_mod.manuscript_gmin(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["both hit", "seeds this arm alone hits"]


def test_a_drifted_develop_morse_count_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Morse M\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=14\\) on 9 and 6 of 12 against none"
    assert old in audit_mod.collapse(tex)
    audit.text_cache[audit_mod.TEX] = audit_mod.collapse(tex).replace(old, old.replace("on 9 and 6", "on 7 and 9"))
    audit_mod.manuscript_gmin(audit)
    assert [(r[2], r[3]) for r in audit.drifts()] == [
        ("prose", "Morse M\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=14\\) on 9 and 6 of 12 against none "
                  "(\\(p=3.9\\times 10^{-3}\\) and 0.031), and rigid")]


def test_a_develop_gmin_cell_that_drifts_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "this work & 9/12 & 9 & 0 & 3 &"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "this work & 7/12 & 9 & 0 & 3 &")
    audit_mod.manuscript_gmin(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [
        (f"{audit_mod.TEX} tab:gmin develop m55r14 this work", "hits")]


def test_a_develop_row_off_the_submitted_revision_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows(f"{audit_mod.GMIN_DEVELOP}/rows.csv")
    row = next(r for r in rows if r["system"] == "m80r14" and r["code"] == "anneal")
    row["code_revision"] = "1ce3ab64493196afd25e8690648e21157d91b882"
    audit_mod.manuscript_gmin(audit)
    assert "row invariants, pinned revisions, host and session" in {r[2] for r in audit.drifts()}


def test_paired_statistics_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    for only_a, only_b in ((5, 0), (1, 4), (12, 0), (5, 1), (2, 10), (4, 6)):
        expected = stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue
        assert math.isclose(audit_mod.mcnemar(only_a, only_b), expected, rel_tol=1e-9)
    for diffs in ([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, -1.0, 4.0, -1.5, 5.0, 9.0, -2.6, 5.3],
                  [12.0, -7.0, 3.0, 8.0, 10.0, -1.0, 6.0, 4.0, 2.0, 11.0, 9.0, 5.0]):
        expected = stats.wilcoxon(diffs, alternative="two-sided", method="exact").pvalue
        assert math.isclose(audit_mod.wilcoxon_signed_rank(diffs)[2], expected, rel_tol=1e-9)


def test_the_shared_statistics_match_scipy():
    shared = audit_mod.paired_stats_module()
    shared.self_test()
    stats = pytest.importorskip("scipy.stats")
    for only_a, only_b in ((5, 0), (1, 4), (12, 0), (9, 0), (4, 1), (2, 10)):
        expected = stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue
        assert math.isclose(shared.mcnemar_exact(only_a, only_b), expected, rel_tol=1e-9)
    for diffs in ([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, -1.0, 4.0, -1.5, 5.0, 9.0, -2.6, 5.3]):
        expected = stats.wilcoxon(diffs, alternative="two-sided", method="exact").pvalue
        assert math.isclose(shared.wilcoxon_exact(diffs)[2], expected, rel_tol=1e-9)
    for a, b in ((12, 0), (7, 0), (4, 1), (121, 94)):
        n = 192 if a > 12 else 12
        expected = stats.fisher_exact([[a, n - a], [b, n - b]]).pvalue
        assert math.isclose(shared.fisher_exact(a, n, b, n), expected, rel_tol=1e-9)


def test_a_literal_normal_quantile_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/scripts/water_rerun.py"
    audit.text_cache[path] = audit.text(path) + "\nHALF = 1.96 * 0.1\n"
    audit_mod.paired_statistics(audit)
    assert [(r[1], r[2], r[4]) for r in audit.drifts()] == [(path, "literal normal quantile", "1.96")]


def test_the_paired_newcombe_interval_matches_the_published_example():
    low, high = audit_mod.newcombe_paired_interval(36, 12, 2, 0)
    assert (round(low, 4), round(high, 4)) == (0.0569, 0.3404)
    assert audit_mod.mcnemar(5, 0) == 0.0625 and audit_mod.mcnemar(6, 0) == 0.03125


def test_a_title_across_funnels_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "of a basin-hopping search: a"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "of a basin-hopping search across funnels: a")
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["no overclaim: across funnels"]
    titled = audit_mod.Audit()
    titled.text_cache[audit_mod.TEX] = audit.text_cache[audit_mod.TEX]
    audit_mod.manuscript_new(titled)
    assert "prose" in {r[2] for r in titled.drifts()}


def test_a_fisher_test_in_the_gmin_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed."
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old + " Fisher's test agrees.", 1)
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["the GMIN comparison does not print Fisher"]


def test_a_tip4p_p_off_its_rows_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "On (H\\(_2\\)O)\\(_8\\) this code hits 12 of 12 and GMIN 8, with \\(p=0.125\\),"
    assert old in audit_mod.collapse(tex)
    audit.text_cache[audit_mod.TEX] = audit_mod.collapse(tex).replace(old, old.replace("0.125", "0.037"))
    audit_mod.manuscript_knockouts(audit)
    assert {r[2] for r in audit.drifts()} == {"prose"}


def test_a_limits_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex.replace("\\section{Conclusions}", "\\section{Limits}\nText.\n\n\\section{Conclusions}")
    audit_mod.todo_check(audit)
    assert [r[2] for r in audit.drifts()] == ["no hedge: \\section{Limits}"]


def test_a_caption_missing_the_serial_patch_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "(\\texttt{JP=1})"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "")
    audit_mod.gmin_patches(audit)
    assert [r[2] for r in audit.drifts()] == ["prose"]


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


def test_a_rerun_tag_fails_the_normal_audit_and_pre_submission():
    tex = "GMIN hits 9 of 12. \\rerun{5}\n% \\rerun{6} in a comment\n"
    assert audit_mod.rerun_tags(tex) == [(1, "5")]
    for strict in (False, True):
        audit = _audit_with_tags(tex)
        audit_mod.rerun_check(audit, pre_submission=strict)
        assert sorted(row[2] for row in audit.drifts()) == ["[RERUN: item 5] remains (unknown)",
                                                           "the tag names an item that waits on a rerun"]


def test_no_rerun_item_is_pending():
    assert audit_mod.RERUN_ITEMS == {} and audit_mod.TODO_ITEMS == {}
    audit = _audit_with_tags("a number \\rerun{9}\n")
    audit_mod.rerun_check(audit)
    assert len(audit.drifts()) == 2


def test_no_tag_passes_pre_submission():
    audit = _audit_with_tags("clean text\n")
    audit_mod.rerun_check(audit, pre_submission=True)
    assert not audit.drifts()


def test_a_replaced_claim_is_reported_fenced_or_not():
    for tail in ("", "~\\rerun{5}"):
        audit = audit_mod.Audit()
        tex = audit.text(audit_mod.TEX)
        audit.text_cache[audit_mod.TEX] = tex.replace(
            "\\section{Conclusions}", "No row was run with the basin bias or the tabu list switched off." + tail
            + "\n\n\\section{Conclusions}")
        audit_mod.rerun_claims(audit)
        assert [(r[2], r[3]) for r in audit.drifts()] == [
            ("a replaced result is not printed", "No row was run with the basin bias or the tabu list switched off")]


def test_copper_tolerance_and_jcc_line_link_the_sources():
    audit = audit_mod.Audit()
    audit_mod.supporting_numbers(audit)
    tolerance = [row for row in audit.results if row[2] == "TARGET_TOL matches tally tolerance"]
    jcc = [row for row in audit.results if row[2] == "the jcc script's target line"]
    assert tolerance == [("OK", "paper/jctc/scripts/hq_sci_tally.py", "TARGET_TOL matches tally tolerance", "0.001",
                          "0.001")]
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


def test_a_todo_marker_in_the_si_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "Wall times are compared only within a cell."
    assert old in si
    audit.text_cache[audit_mod.SI] = si.replace(old, old[:-1] + ".~\\todo{G2}")
    audit_mod.todo_check(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["[TODO: G2] remains (unknown)", "todo markers"]


def test_a_process_note_in_the_cover_letter_is_reported():
    audit = audit_mod.Audit()
    cover = audit.text(audit_mod.COVER)
    old = "The title and the content have changed."
    audit.text_cache[audit_mod.COVER] = cover.replace(old, old + "\nA fair GMIN arm is being run.")
    audit_mod.cover_letter(audit)
    assert [r[2] for r in audit.drifts()] == ["the cover letter carries no process note (being run)"]


def test_a_drifted_ase_call_range_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "stopped at the cap after 3933 to 20950 force calls"
    assert old in audit_mod.collapse(tex)
    audit.text_cache[audit_mod.TEX] = audit_mod.collapse(tex).replace(old, old.replace("3933", "3934"))
    audit_mod.manuscript_ase(audit)
    assert [r[2] for r in audit.drifts()] == ["prose"]


def test_a_drifted_newcombe_interval_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "& 12/12 & 4/12 & 8 & 0 & \\(7.8\\times 10^{-3}\\) & 0.30 to 0.86 \\\\"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("0.86", "0.87"))
    audit_mod.manuscript_knockouts(audit)
    assert [r[2] for r in audit.drifts()] == ["Newcombe upper"]


def test_the_newcombe_interval_matches_the_published_example():
    low, high = audit_mod.newcombe_interval(56, 70, 48, 80)
    assert (round(low, 4), round(high, 4)) == (0.0524, 0.3339)
    assert audit_mod.interval_text(-0.204, 0.123) == "\\(-\\)0.20 to 0.12"


def test_a_gmin_line_off_the_rerun_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "LJ\\(_{150}\\) & \\(4\\times 10^{6}\\) & this work"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "M\\(_{55}\\), \\(\\rho=14\\) & \\(1\\times 10^{6}\\) & this work")
    audit_mod.manuscript_gmin(audit)
    assert {"table line", "tab:gmin lines"} <= {r[2] for r in audit.drifts()}


def test_a_relabeled_develop_block_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "\\emph{The submitted revision \\texttt{61fa67cf}}"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "\\emph{The submitted revision \\texttt{1ce3ab64}}")
    audit_mod.manuscript_gmin(audit)
    assert {"tab:gmin block headings", "table line", "tab:gmin lines"} <= {r[2] for r in audit.drifts()}


def test_an_overclaim_that_returns_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed."
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old + " The misses sit in the same trap.")
    audit_mod.science_claims(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["no overclaim: same trap", "the results and conclusions do not say trap"]


def test_a_hypothesis_in_the_results_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed."
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old + " This supports the funnel hypothesis.")
    audit_mod.science_claims(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["the manuscript states no hypothesis",
                                                   "the results and conclusions do not say funnel",
                                                   "the results and conclusions do not say hypothes"]


def test_an_unresolved_citation_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "\\cite{fisherLogicInductiveInference1935}"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "\\cite{fisherLogicInductiveInference1936}", 1)
    audit_mod.science_claims(audit)
    assert "cited keys without an entry" in {r[2] for r in audit.drifts()}


def test_a_stray_gmin_patch_line_is_reported():
    audit = audit_mod.Audit()
    path = "scripts/morse_h2h/morse-timer.patch"
    patch = audit.text(path)
    old = "+      CALL H2H_POT_LEAVE(H2H_TICK)\n"
    assert old in patch
    audit.text_cache[path] = patch.replace(old, old + "+      TEMP = 2.0D0\n", 1)
    audit_mod.gmin_patches(audit)
    assert [r[2] for r in audit.drifts()] == ["added GMIN lines outside the cap and timers"]


def test_a_second_gate_claim_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "For atomic clusters the preset has one atom-count gate:"
    assert old in audit_mod.collapse(tex)
    audit.text_cache[audit_mod.TEX] = audit_mod.collapse(tex).replace(
        old, "For atomic clusters the preset has two atom-count gates, both at 98 atoms:")
    audit_mod.count_gates(audit, "c2de652dca0bc75e2b0cb4afffac91fd656404e4")
    assert sorted(r[2] for r in audit.drifts()) == ["prose", "retired gate wording stays out: two atom-count gates"]


def test_a_withdrawn_benchmark_in_the_text_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex.replace("\\end{abstract}", "It hits 121 of 192 seeds.\n\\end{abstract}", 1)
    audit_mod.withdrawn_benchmarks(audit)
    assert [r[2] for r in audit.drifts()] == ["the water hexamer against CREST is not printed: 121 of 192"]


def test_crest_outside_the_related_searches_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    audit.text_cache[audit_mod.SI] = si.replace("\\section{Statistical methods}", "CREST.\n\\section{Statistical methods}", 1)
    audit_mod.withdrawn_benchmarks(audit)
    assert [r[2] for r in audit.drifts()] == ["CREST is named only among the searches that report no comparable rate"]


def test_a_banned_word_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "landscape to propose structures"
    assert old in si
    audit.text_cache[audit_mod.SI] = si.replace(old, "landscape to propose candidates")
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["the text does not say candidate"]


def test_record_only_revisions_resolve_and_unpushed_ones_do_not():
    """f009acc9 and 0cc3814c resolve from the listed record branches.

    An id that is not in the clone, and a commit that is in the clone and
    on no listed remote, stay unpushed.
    """
    import os

    harness = "f009acc96bb95023eeb6e394e863630c752a5d05"
    traced = "0cc3814c053c2f5a709f0f375ebae07e5373c171"
    absent = "deadbeefdead"
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=audit_mod.ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    env = dict(os.environ)
    env.update(GIT_AUTHOR_NAME="audit", GIT_AUTHOR_EMAIL="audit@example.invalid",
               GIT_COMMITTER_NAME="audit", GIT_COMMITTER_EMAIL="audit@example.invalid")
    local = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "commit-tree", tree, "-m", "local only"],
        cwd=audit_mod.ROOT, capture_output=True, text=True, check=True, env=env).stdout.strip()
    index = audit_mod.RevisionIndex((harness, traced, absent, local), (), (), ())
    assert index.audited_exists(harness) and index.on_remote(harness)
    assert index.audited_exists(traced) and index.on_remote(traced)
    assert not index.audited_exists(absent)
    assert not index.audited_exists(local) and not index.on_remote(local)

    audit = audit_mod.Audit()
    path = "paper/jctc/data/gmin_develop/rows.csv"
    rows = [dict(row) for row in audit.rows(path)]
    rows[0]["harness_revision"] = traced
    rows[1]["harness_revision"] = absent
    rows[2]["harness_revision"] = local
    audit.read[path] = rows
    gap = audit_mod.provenance(audit).get(path, "")
    assert f"revision {harness[:8]}" not in gap
    assert f"revision {traced[:8]}" not in gap
    assert f"revision {absent[:8]} is not in this repository" in gap
    assert f"revision {local[:8]} is not in this repository" in gap
