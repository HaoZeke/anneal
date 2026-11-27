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
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import machine_provenance  # noqa: E402

SPEC = importlib.util.spec_from_file_location("paper_audit", ROOT / "scripts" / "paper_audit.py")
audit_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_mod)
PAPER_CONSTANTS = audit_mod.load_script(
    "paper_constants_test",
    "paper/jctc/scripts/paper_constants.py",
).values()
# The suite checks origin record refs. main() leaves the fetch off unless
# --fetch-records is passed; tests that call the functions set it here.
audit_mod._FETCH_RECORDS = True

CHECKS = (
    audit_mod.paired_statistics,
    audit_mod.manuscript_gmin,
    audit_mod.withdrawn_benchmarks,
    audit_mod.supporting_information,
    audit_mod.release,
    audit_mod.paper_0120,
    audit_mod.gad_published,
    audit_mod.openconf,
    audit_mod.conf_miss4,
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
    audit_mod.manuscript_knockouts,
    audit_mod.manuscript_priors,
    audit_mod.panel_references,
    audit_mod.generated_tables,
    audit_mod.sutton_chen,
    audit_mod.supporting_numbers,
    audit_mod.crest_equal,
    audit_mod.crest_equal_bench,
    audit_mod.crest_table,
    audit_mod.gmin_tuned,
    audit_mod.morse_m70_pair,
    audit_mod.rerun_summaries,
    audit_mod.outside_session,
    audit_mod.g3_session,
    audit_mod.manuscript_new,
    audit_mod.manuscript_followups,
    audit_mod.cover_letter,
    audit_mod.science_claims,
    audit_mod.cap_turn_campaign,
    audit_mod.rerun_check,
    audit_mod.append_only_rows,
    audit_mod.pending_text,
    audit_mod.citation_metadata,
    audit_mod.figure_source_scripts,
    audit_mod.bibliography_keys,
    audit_mod.conversion_literals,
    audit_mod.own_references,
    audit_mod.build_receipts,
    audit_mod.cross_document_equations,
    audit_mod.selection_tables,
    audit_mod.gmin_panel_drivers,
    audit_mod.prior_switches,
    audit_mod.method_definitions,
    audit_mod.au55_superseded,
    audit_mod.tip4p_unscored,
    audit_mod.tip4p_r2_rows,
    audit_mod.rerun_claims,
    audit_mod.todo_check,
    audit_mod.revision_map,
    audit_mod.all_revision_provenance,
)
PANEL_COMMIT = "9837358a"
ASE_COMMIT = "61fa67cf"


def withhold(monkeypatch, *prefixes):
    """Make a checkout that does not hold the commits that start with one of ``prefixes``."""
    held = audit_mod.commit_held
    monkeypatch.setattr(
        audit_mod, "commit_held",
        lambda rev: False if any((rev or "").startswith(p) for p in prefixes) else held(rev),
    )


def panel_tokens(system):
    """The hits and wall of this work's rows of one panel system, as the audit prints them."""
    audit = audit_mod.Audit()
    panel, groups = audit_mod.gmin_panel_groups(audit)
    rows = groups[(system, "anneal")]
    hits = f"{audit_mod.hits_of(rows)} of {len(rows)}"
    wall = panel.show_s(audit_mod.median([float(r["wall_s"]) for r in rows]))
    return hits, wall


@pytest.fixture(scope="module")
def audited():
    audit = audit_mod.Audit()
    for check in CHECKS:
        check(audit)
    for path, fragment, _why in audit_mod.UNTRACED:
        audit.said(path, fragment)
    return audit


def test_tex_without_comments_strips_inline_comment():
    assert audit_mod.tex_without_comments("visible % hidden") == "visible "


def test_tex_without_comments_preserves_escaped_percent():
    assert audit_mod.tex_without_comments(r"visible \% value") == r"visible \% value"


def test_tex_without_comments_strips_whole_line_comment():
    assert audit_mod.tex_without_comments("before\n% hidden\nafter") == "before\nafter"


def test_tex_without_comments_joins_a_commented_line_end():
    assert audit_mod.tex_without_comments("foo%\nbar") == "foobar"


def test_tex_without_comments_preserves_literal_environment():
    text = "\\begin{verbatim}\nvisible % literal\n\\end{verbatim}\n"
    assert audit_mod.tex_without_comments(text) == text


def replace_collapsed(text, old, new):
    """Replace one whitespace-collapsed fragment without flattening the file."""
    pattern = re.escape(old).replace(r"\ ", r"\s+")
    replaced, count = re.subn(pattern, lambda _match: new, text, count=1)
    assert count == 1
    return replaced


def test_every_table_cell_matches_its_rows(audited):
    drift = audited.drifts()
    assert not drift, "\n".join(f"{w}  {label}  shown={s}  computed={v}" for _, w, label, s, v in drift)
    assert len(audited.results) > 2500


def test_provenance_gaps_are_exactly_the_listed_ones(audited):
    assert audit_mod.provenance(audited) == audit_mod.KNOWN_GAPS


def test_new_machine_provenance_requires_a_digest_and_cpu_model():
    audit = audit_mod.Audit()
    audit.read["new.csv"] = [
        {
            "code_revision": "",
            "host": "bench",
            "machine_id": "not-a-digest",
            "cpu_model": "",
        }
    ]

    gap = audit_mod.provenance(audit)["new.csv"]

    assert "cpu_model empty on some rows" in gap
    assert "machine_id is not a salted sha256 on some rows" in gap


def test_machine_provenance_accepts_aggregated_digests():
    audit = audit_mod.Audit()
    audit.read["summary.csv"] = [
        {
            "code_revision": "",
            "host": "bench",
            "machine_id": f"{'a' * 64};{'b' * 64}",
            "cpu_model": "CPU A;CPU B",
        }
    ]

    gap = audit_mod.provenance(audit)["summary.csv"]

    assert "machine_id" not in gap


def test_machine_schema_requires_both_stable_columns():
    audit = audit_mod.Audit()
    audit.read["missing-cpu.csv"] = [
        {
            "code_revision": next(iter(audit_mod.EXTERNAL_REVISIONS)),
            "host": "bench",
            "machine_id": "a" * 64,
        }
    ]
    audit.read["missing-id.csv"] = [
        {
            "code_revision": next(iter(audit_mod.EXTERNAL_REVISIONS)),
            "host": "bench",
            "cpu_model": "Example CPU",
        }
    ]

    gaps = audit_mod.provenance(audit)

    assert "machine_id recorded; no cpu_model column" in gaps["missing-cpu.csv"]
    assert "cpu_model recorded; no machine_id column" in gaps["missing-id.csv"]


def test_legacy_schema_without_machine_columns_remains_exempt():
    audit = audit_mod.Audit()
    audit.read["old.csv"] = [
        {
            "code_revision": next(iter(audit_mod.EXTERNAL_REVISIONS)),
            "host": "bench",
        }
    ]

    assert "old.csv" not in audit_mod.provenance(audit)


def test_untraced_numbers_are_still_quoted(audited):
    for path, fragment, _why in audit_mod.UNTRACED:
        assert audit_mod.collapse(fragment) in audit_mod.collapse((ROOT / path).read_text()), (path, fragment)


def test_script_exits_zero():
    assert audit_mod.run() == 0


def test_paper_audit_has_no_drift(capsys):
    assert [(path, why) for path, _fragment, why in audit_mod.UNTRACED] == [
        ("bench/water-h2o20-goat/README.md", "the seed 16001 rows were not committed")]
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
            sys.executable,
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
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    raw = boot_id.read_text().strip() if boot_id.is_file() else None
    assert row["boot_id"] == (machine_provenance.boot_label(raw) if raw else None)
    assert raw is None or raw not in out.read_text()
    assert row["cpu_affinity"] == available_cpus()
    assert re.fullmatch(r"[0-9a-f]{40}", row["code_revision"])


def available_cpus():
    """The CPUs the session records, by the rule scripts/session_metadata.py applies."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def session_metadata_module():
    spec = importlib.util.spec_from_file_location("session_metadata_test", ROOT / "scripts" / "session_metadata.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_affinity_is_the_scheduler_affinity_where_the_os_has_one(monkeypatch):
    module = session_metadata_module()
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {5, 2, 3}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert module.cpu_affinity() == [2, 3, 5]


def test_session_affinity_falls_back_to_the_cpu_count_without_a_scheduler_call(monkeypatch):
    module = session_metadata_module()
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert module.cpu_affinity() == [0, 1, 2, 3]
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert module.cpu_affinity() == [0]


def test_the_session_record_takes_its_affinity_from_that_rule(monkeypatch):
    module = session_metadata_module()
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    row = module.record("campaign", ["cmd"], ROOT)
    assert row["cpu_affinity"] == [0, 1, 2] and row["cpu_affinity_text"] == "0,1,2"


def test_the_crest_equal_session_is_pinned_only_where_taskset_exists(monkeypatch):
    runner = crest_equal_runner()
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/taskset")
    assert runner.pin_prefix("3") == ["taskset", "-c", "3"]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert runner.pin_prefix("3") == []


def test_tex_prose_checks_ignore_comments():
    fragment = "Required visible sentence."
    audit = audit_mod.Audit()
    audit.text_cache["sample.tex"] = f"% {fragment}\n   % {fragment}\nVisible text.\n"

    audit.said("sample.tex", fragment)

    assert [result[2] for result in audit.drifts()] == ["prose"]


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
        if args[:4] == ("ls-remote", "--exit-code", "--tags", "origin"):
            return subprocess.CompletedProcess(["git", *args], 2, "", "")
        return real_git(*args, stdin=stdin)

    monkeypatch.setattr(audit_mod, "git", without_required_refs)
    problems = audit_mod.unauditable()
    assert f"the release tag v{version} is neither local nor advertised by origin" in problems
    assert "the remote ref origin/develop is not present; fetch develop from origin" in problems


def test_an_advertised_release_tag_need_not_be_fetched(monkeypatch):
    real_git = audit_mod.git
    tag = f"refs/tags/v{workspace_version()}"

    def without_local_tag(*args, stdin=None):
        if args == ("show-ref", "--verify", "--quiet", tag):
            return subprocess.CompletedProcess(["git", *args], 1, "", "")
        if args == ("ls-remote", "--exit-code", "--tags", "origin", tag):
            return subprocess.CompletedProcess(["git", *args], 0, "tag\\n", "")
        return real_git(*args, stdin=stdin)

    monkeypatch.setattr(audit_mod, "git", without_local_tag)
    assert audit_mod.release_tag_available(f"v{workspace_version()}")
    assert not [p for p in audit_mod.unauditable() if "release tag" in p]


@pytest.mark.parametrize(
    ("branch", "revision"),
    [
        ("fix/morse-gmin", "0ef6ba4e39e7a8f2812b75a8e02fcd75a12e19e2"),
        ("fix/sc-portfolio", "17210f1b11c25e1c2031cdb33c4f1cccfb7cc807"),
        ("bench/ptcu-gupta", "55fde657c867f1220a2b6a9792338c67421ebd68"),
    ],
)
def test_optional_remote_branches_join_the_remote_closure(branch, revision):
    assert branch in audit_mod.AUDIT_OPTIONAL_REMOTES
    ref = f"refs/remotes/origin/{branch}"
    if git("show-ref", "--verify", "--quiet", ref).returncode != 0:
        pytest.skip(f"optional remote {ref} is not fetched in this clone")
    index = audit_mod.RevisionIndex((revision,), (), (), ())
    assert index.audited_exists(revision)
    assert index.on_remote(revision)


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


def test_ase_rows_match_the_recorded_runner():
    audit = audit_mod.Audit()
    path = f"{audit_mod.ASE_DATA}/lj38/rows.csv"
    rows = [dict(r) for r in audit.rows(path)]
    rows[0]["harness_blob"] = "0" * 40
    audit.read[path] = rows
    audit_mod.ase_groups(audit)
    assert "harness_blob" in {r[2] for r in audit.drifts()}


def test_ase_rows_use_one_recorded_host():
    audit = audit_mod.Audit()
    path = f"{audit_mod.ASE_DATA}/lj38/rows.csv"
    rows = [dict(r) for r in audit.rows(path)]
    rows[0]["host"] = "f" * 64
    audit.read[path] = rows
    audit_mod.ase_groups(audit)
    assert "single machine-id sha256 host" in {r[2] for r in audit.drifts()}


def test_a_drifted_ase_table_cell_is_reported(monkeypatch):
    """An ASE table row, or its caption revision, fails when the commit is not in this repository."""
    withhold(monkeypatch, ASE_COMMIT)
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
    old = "& 11.333 & 47.553 & 47.556 \\\\"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "& 11.333 & 47.553 & 47.557 \\\\")
    audit_mod.manuscript_ase(audit)
    audit_mod.generated_tables(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["ase_mh median wall", "tab:ase-morse is the summariser's block"]


def test_a_cross_arm_p_value_in_the_ase_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "ASE minima hopping is wall-capped"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "\\(p_\\mathrm{BH}=0.01\\). " + old)
    audit_mod.manuscript_ase(audit)
    assert [r[2] for r in audit.drifts()] == ["sec:ase reports no cross-arm p-values"]


def test_a_rank_test_in_the_ase_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "\\section{Conclusions}"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(
        old, old + "\nA Mann--Whitney test is not run.", 1
    )
    audit_mod.manuscript_ase(audit)
    assert [r[2] for r in audit.drifts()] == [
        "the conclusions report no rank test"
    ]


def test_unequal_ase_limits_are_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "same numeric call ceiling and wall cap"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "different call and wall limits", 1)
    audit_mod.manuscript_ase(audit)
    assert sorted(r[2] for r in audit.drifts()) == [
        "ASE limits are not described as unequal",
        "prose",
    ]


def test_a_returning_endpoint_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "\\section{Conclusions}"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "\\subsection{Endpoints}\n\\label{sec:endpoints}\n" + old)
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["the endpoint-energy section is not printed"]


def test_the_trap_claim_is_required_in_the_results():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    assert "trap" in tex.lower()
    audit.text_cache[audit_mod.TEX] = re.sub("trap", "site", tex, flags=re.IGNORECASE)
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["the results and conclusions state the trap claim"]


def test_a_drifted_template_count_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "at \\(\\rho=14\\) fall from 12 to 4 of 12"
    assert old in audit_mod.collapse(tex)
    audit.text_cache[audit_mod.TEX] = replace_collapsed(tex, old, old.replace("to 4", "to 3"))
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
    old = "recommended template-free rigid-water preset"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "rigid-water preset")
    audit_mod.presets_and_templates(audit)
    assert [r[2] for r in audit.drifts()] == ["tab:tip4p-gmin names its preset"]


def test_a_restart_of_this_code_in_the_text_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex + "\nA stalled chain restarts from a random cluster.\n"
    audit_mod.no_anneal_restart(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [(audit_mod.TEX, "every restart the text names is GMIN's")]


def test_a_hexamer_whose_energy_leaves_its_record_is_reported():
    audit = audit_mod.Audit()
    audit_mod.hexamer_provenance(audit)
    assert not audit.drifts()
    path = f"{audit_mod.VISUALS}/hexamer_gfn2/RECORD"
    audit = audit_mod.Audit()
    audit.text_cache[path] = audit.text(path).replace("relaxed_energy_Eh -30.49650686764", "relaxed_energy_Eh -30.5")
    audit_mod.hexamer_provenance(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [
        (f"{audit_mod.VISUALS}/hexamer_gfn2/prism.xyz", "energy is the one RECORD gives")]


def test_a_record_that_names_another_relaxation_script_is_reported():
    path = f"{audit_mod.VISUALS}/hexamer_gfn2/RECORD"
    audit = audit_mod.Audit()
    record = audit.text(path)
    blob = record.split("hexamer_gfn2_relax.py blob ")[1].split()[0]
    audit.text_cache[path] = record.replace(blob, "0" * 40)
    audit_mod.hexamer_provenance(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [(path, "blob of the relaxation script")]


def test_landfold_provenance_does_not_need_pull_refs(monkeypatch):
    real_git = audit_mod.git

    def heads_only(*args, stdin=None):
        assert "refs/remotes/origin/pull" not in args
        return real_git(*args, stdin=stdin)

    monkeypatch.setattr(audit_mod, "git", heads_only)
    audit = audit_mod.Audit()
    audit_mod.hexamer_provenance(audit)
    assert not audit.drifts()
    labels = [row[2] for row in audit.results]
    assert labels.count("committed landfold outputs have sha256 digests") == 1
    assert len([label for label in labels
                if label.startswith("digest of system_visuals/landfold/")]) == 8


def test_a_placeholder_in_the_text_is_reported():
    for word in ("rerun pending", "TBD", "TODO"):
        audit = audit_mod.Audit()
        tex = audit.text(audit_mod.TEX)
        audit.text_cache[audit_mod.TEX] = tex.replace("\\section{Conclusions}", f"LJ\\(_{{75}}\\): {word}.\n\n"
                                                      "\\section{Conclusions}")
        audit_mod.science_claims(audit)
        said = word.split()[-1]
        assert [r[2] for r in audit.drifts()] == [f"the text does not say {said}"]


def test_a_claim_that_the_template_era_rows_hold_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "The comparison spans every change between the revisions, not the templates\nalone."
    assert si.count(old) == 1
    audit.text_cache[audit_mod.SI] = si.replace(old, old + "\nThe later code reproduces the template-assisted rows.")
    audit_mod.template_era(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [
        (audit_mod.SI, "no sentence says the later code matches the template-assisted rows")]


def test_a_drifted_template_era_cell_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "& \\texttt{c2de652d} & 2/12 & 0 & 2 & 0 & 4000000 & 0 &"
    assert si.count(old) == 2
    audit.text_cache[audit_mod.SI] = si.replace(old, "& \\texttt{c2de652d} & 12/12 & 0 & 2 & 0 & 4000000 & 0 &", 1)
    audit_mod.generated_tables(audit)
    assert [r[2] for r in audit.drifts()] == ["tab:si-template-era is the summariser's block"]


def test_c2de652d_in_the_manuscript_fails():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "\\section{Conclusions}"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "Revision \\texttt{c2de652d} hits M\\(_{80}\\) on 2 of 12.\n\n" + old)
    audit_mod.template_era(audit)
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["no row is labelled c2de652d"]


def test_c2de652d_in_the_si_outside_its_section_fails():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "\\section{Earlier template-assisted revision}"
    assert si.count(old) == 1
    audit.text_cache[audit_mod.SI] = si.replace(old, "Revision \\texttt{c2de652d}.\n\n" + old)
    audit_mod.science_claims(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [(audit_mod.SI, "no row is labelled c2de652d")]


def test_an_outside_provenance_with_another_tree_is_reported():
    audit = audit_mod.Audit()
    audit_mod.outside_session(audit)
    assert not audit.drifts()
    path = f"{audit_mod.OUTSIDE_DATA}/PROVENANCE"
    audit = audit_mod.Audit()
    audit.text_cache[path] = audit.text(path).replace("tree c6096c4a", "tree 06096c4a")
    audit_mod.outside_session(audit)
    assert [r[2] for r in audit.drifts()] == ["tree of the record commit"]


def test_an_unreachable_outside_arm_is_reported():
    audit = audit_mod.Audit()
    path = f"{audit_mod.OUTSIDE_DATA}/rows.csv"
    rows = audit.rows(path)
    old = "61fa67cf380e46876b569da45e7cc091647a775b"
    assert rows[0]["code_revision"] == old
    rows[0]["code_revision"] = "1e38f2dacd9981fa9602381cebf1bca108b97230"
    audit_mod.outside_session(audit)
    descriptions = {r[2] for r in audit.drifts()}
    assert "code revisions" in descriptions


def test_a_drifted_outside_cell_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "& 38.987 & 37.806 &"
    assert si.count(old) == 1
    audit.text_cache[audit_mod.SI] = si.replace(old, "& 38.987 & 30.806 &")
    audit_mod.outside_session(audit)
    assert [r[2] for r in audit.drifts()] == ["tab:si-outside is the summariser's block"]


def test_a_move_choice_success_claim_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "\\section{Conclusions}"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, "The Thompson draw drives the hits on LJ\\(_{98}\\).\n\n" + old)
    audit_mod.no_move_choice_claim(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [(audit_mod.TEX, "no sentence credits the move choice with success")]


def test_a_modal_hedge_fails_both_audits():
    for strict in (False, True):
        audit = audit_mod.Audit()
        si = audit.text(audit_mod.SI)
        old = "and both algorithms return an assignment of the optimal cost."
        assert si.count(old) == 1
        audit.text_cache[audit_mod.SI] = si.replace(old, old + "\nA tie may break differently.")
        audit_mod.todo_check(audit, pre_submission=strict)
        assert [(r[1], r[2], r[4]) for r in audit.drifts()] == [(audit_mod.SI, "no modal hedge", "may")]


def test_a_rendered_rerun_fails_both_audits():
    for strict in (False, True):
        audit = audit_mod.Audit()
        si = audit.text(audit_mod.SI)
        old = "\\path{paper/jctc/data/gmin_eeec8e2c}"
        assert si.count(old) == 1
        audit.text_cache[audit_mod.SI] = si.replace(old, "\\path{paper/jctc/data/gmin_rerun}")
        audit_mod.todo_check(audit, pre_submission=strict)
        assert [(r[1], r[2]) for r in audit.drifts()] == [(audit_mod.SI, "no rendered rerun")]


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
    assert sorted(r[2] for r in audit.drifts()) == ["[RERUN: item 5] remains (unknown)", "no rendered rerun",
                                                   "rerun tags", "the tag names an item that waits on a rerun"]


def test_a_cover_sentence_off_the_abstract_is_reported():
    audit = audit_mod.Audit()
    cover = audit.text(audit_mod.COVER)
    old = "The title and the content have changed."
    assert old in cover
    audit.text_cache[audit_mod.COVER] = cover.replace(old, old + "\nGMIN hits LJ\\(_{150}\\) on 9 of 12.")
    audit_mod.cover_letter(audit)
    assert [r[2] for r in audit.drifts()] == [
        "a cover-letter sentence with a number is a sentence of the abstract",
        "the numbered cover-letter claims are the numbered abstract claims"]


def test_a_drifted_gmin_panel_summary_cell_is_reported():
    audit = audit_mod.Audit()
    summary = audit.rows("bench/gmin-panel/summary.csv")
    au = next(r for r in summary if r["system"] == "au55")
    au["gmin_hits"] = "0"
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["gmin_hits"]


def test_a_cross_arm_work_p_value_in_the_gmin_table_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "no cross-arm hit, work, or wall test is reported."
    assert tex.count(old) == 2
    start, rest = tex.split("% gmin-panel table: start", 1)
    table, end = rest.split("% gmin-panel table: end", 1)
    audit.text_cache[audit_mod.TEX] = (
        start + "% gmin-panel table: start"
        + table.replace(old, "\\(p_\\mathrm{calls}=0.021\\). " + old, 1)
        + "% gmin-panel table: end" + end)
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["tab:gmin reports no cross-arm tests"]


def test_a_cross_arm_hit_p_value_in_the_gmin_prose_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = ("At those arm-specific ceilings, this code hits 7 of 12 seeds on\n"
           "LJ\\(_{98}\\) and 5 of 12 on LJ\\(_{75}\\), while GMIN hits neither; on\n"
           "LJ\\(_{38}\\), the counts are 12 and 7.")
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old + "\nThe paired result is \\(p=0.016\\).")
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["the GMIN results report no cross-arm hit test"]


def test_the_develop_gmin_rows_keep_the_harness_that_ran_them():
    table = audit_mod.load_script("gmin_table", "paper/jctc/scripts/gmin_table.py")
    rows = table.develop_rows()
    assert {r["harness_revision"] for r in rows} == {table.DEVELOP_HARNESS}
    lines, paired = table.develop_summary()
    assert lines and paired
    assert {line["harness_revision"] for line in lines} == {table.DEVELOP_HARNESS}
    assert {line["anneal_revision"] for line in lines} == {table.SUBMITTED}


def test_a_drifted_length_unit_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "M\\(_{80}\\), \\(\\rho=14\\) & \\(4\\times 10^{6}\\) & 12 & 7/12 & 4/12 &"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("7/12 & 4/12", "9/12 & 4/12"))
    audit_mod.manuscript_knockouts(audit)
    assert [r[2] for r in audit.drifts()] == ["with"]


def test_a_drifted_pooled_length_unit_sentence_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "it hits 40 of 48 against 35\n(pooled McNemar \\(p=0.30\\)"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("against 35", "against 30"))
    audit_mod.manuscript_knockouts(audit)
    assert [r[2] for r in audit.drifts()] == ["prose", "prose"]


def test_a_length_unit_row_from_another_driver_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/data/knockouts/pairscale/rows.csv"
    rows = [dict(r) for r in audit.rows(path)]
    rows[0]["driver_blob"] = "0" * 40
    audit.read[path] = rows
    knock = audit_mod.load_script("knockout_tables_audit", "paper/jctc/scripts/knockout_tables.py")
    audit_mod.knockout_pairscale(audit, audit_mod.knockout_cells(audit, knock, knock.KNOCKOUTS[1]))
    assert "every row ran the Morse driver of the tree the rows name" in [r[2] for r in audit.drifts()]


def test_a_drifted_g1_cell_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "M\\(_{38}\\), \\(\\rho=14\\) & \\(4\\times 10^{5}\\) & 4/12 &"
    assert si.count(old) == 1
    audit.text_cache[audit_mod.SI] = si.replace(old, old.replace("4/12", "5/12"))
    audit_mod.morse_g1(audit)
    assert [r[2] for r in audit.drifts()] == ["tab:si-morse-g1 is the summariser's block"]


def test_a_drifted_g1_sentence_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "and 4 of 12 on M\\(_{38}\\)"
    assert si.count(old) == 1
    audit.text_cache[audit_mod.SI] = si.replace(old, "and 5 of 12 on M\\(_{38}\\)")
    audit_mod.morse_g1(audit)
    assert [r[2] for r in audit.drifts()] == ["prose"]


def test_a_stored_g1_gmin_count_in_si_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "the article prints this code alone here."
    assert si.count(old) == 1
    audit.text_cache[audit_mod.SI] = si.replace(old, old + " They take other calls on all 24 hits.")
    audit_mod.morse_g1(audit)
    assert [r[2] for r in audit.drifts()] == ["the SI prints no count from the stored GMIN rows of G1"]


def test_a_g1_row_of_another_revision_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/data/gmin_g1/rows.csv"
    rows = [dict(r) for r in audit.rows(path)]
    rows[5]["code_revision"] = "dd6ff9273be30ab317bfd5508596f4beb19cb3a4"
    audit.read[path] = rows
    audit_mod.morse_g1(audit)
    labels = [r[2] for r in audit.drifts()]
    assert "rows off the stock preset, the budget or the reference of the run script" in labels
    assert "the paired rows of the record carry these anneal rows" in labels


def test_a_gmin_row_without_a_machine_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/data/gmin_develop/rows.csv"
    rows = [dict(r) for r in audit.rows(path)]
    next(r for r in rows if r["code"] == "gmin")["host"] = ""
    audit.read[path] = rows
    audit_mod.gmin_sources(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [(path, "every GMIN row names its machine")]


def test_a_tip4p_gmin_row_of_another_source_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/data/tip4p_gmin/rows.csv"
    rows = [dict(r) for r in audit.rows(path)]
    next(r for r in rows if r["arm"] == "gmin")["gmin_revision"] = "0" * 40
    audit.read[path] = rows
    audit_mod.gmin_sources(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [
        (path, "historical identifier of every GMIN row"),
        (path, "every TIP4P row carries the historical GMIN identifier"),
    ]


def test_the_rendered_gmin_sources_match_the_rows():
    si = (ROOT / audit_mod.SI).read_text()
    recorded = (
        audit_mod.TIP4P_GMIN,
        audit_mod.GMIN_TARBALL_SHA256,
        audit_mod.GMIN_TARBALL_SVNREV,
    )
    assert [si.count(value) for value in recorded] == [1, 1, 1]
    for value in recorded:
        audit = audit_mod.Audit()
        audit.text_cache[audit_mod.SI] = si.replace(value, "0" * len(value))
        audit_mod.gmin_sources(audit)
        assert "prose" in [r[2] for r in audit.drifts()]


def test_a_g3_row_with_other_calls_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/data/g3_funnel/rows.csv"
    rows = [dict(r) for r in audit.rows(path)]
    row = next(r for r in rows if r["method"] == "anneal" and r["hit"] == "1")
    row["evals_to_hit"] = str(int(row["evals_to_hit"]) + 1)
    audit.read[path] = rows
    audit_mod.g3_session(audit)
    labels = [r[2] for r in audit.drifts()]
    assert any(label.startswith("seeds with the calls and lowest energy of the GMIN table") for label in labels)


def test_a_gmin_claim_in_the_template_era_section_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "Both sessions start each seed from the geometry the driver writes for it.\n"
    assert si.count(old) == 2
    audit.text_cache[audit_mod.SI] = si.replace(
        old, old + "GMIN takes the same calls to the same minimum in both.\n", 1)
    audit_mod.template_era(audit)
    assert "the template-era section prints no GMIN claim: GMIN takes the same calls" in [
        r[2] for r in audit.drifts()]


def test_a_drifted_move_choice_cell_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "LJ\\(_{75}\\) & \\(4\\times 10^{6}\\) & 12 & 5/12 & 6/12 &"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("5/12 & 6/12", "7/12 & 6/12"))
    audit_mod.manuscript_knockouts(audit)
    assert "with" in [r[2] for r in audit.drifts()]


def test_a_move_choice_row_from_another_driver_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/data/knockouts/move-a/rows.csv"
    rows = [dict(r) for r in audit.rows(path)]
    next(r for r in rows if r["system"] == "au55")["driver_blob"] = "0" * 40
    audit.read[path] = rows
    knock = audit_mod.load_script("knockout_tables_audit", "paper/jctc/scripts/knockout_tables.py")
    audit_mod.knockout_move_a(audit, audit_mod.knockout_cells(audit, knock, knock.KNOCKOUTS[2]))
    assert "every row ran the driver of the tree the rows name for its cluster" in [r[2] for r in audit.drifts()]


def test_a_drifted_pooled_move_choice_sentence_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "preset hits 73 of 96 seeds and the uniform\ndraw 75 (pooled McNemar \\(p=0.85\\)"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("draw 75", "draw 70"))
    audit_mod.manuscript_knockouts(audit)
    assert [r[2] for r in audit.drifts()] == ["prose"]


def test_lj75_is_present_in_the_gmin_comparison():
    table = audit_mod.load_script("gmin_table_lj75", "paper/jctc/scripts/gmin_table.py")
    raw = [row for row in table.rows() if row["system"] == "lj75"]
    assert len(raw) == 24
    assert "LJ\\(_{75}\\)" in table.table()
    assert "GMIN & 0/12" in table.table()

    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    gmin = tex.split("% gmin-panel table: start", 1)[1].split("% gmin-panel table: end", 1)[0]
    assert "LJ\\(_{75}\\)" in gmin
    assert "LJ\\(_{75}\\) & \\(4\\times 10^{6}\\) & this work & 5/12" in gmin
    assert "LJ\\(_{75}\\) & \\(4\\times 10^{6}\\) & 12 & 5/12 & 6/12" in tex

    si = audit.text(audit_mod.SI)
    assert "They hit LJ\\(_{75}\\) on 5 of 12 against 12" in si
    assert "The published GOFEE curves count local optimizations on LJ\\(_{75}\\)" in si
    assert "A fingerprint bias during local optimization is reported on LJ\\(_{38}\\)" in si
    assert "Published global minima of LJ\\(_{38}\\), LJ\\(_{75}\\), LJ\\(_{98}\\)" in tex
    assert '"lj75_marks"' in audit.text("paper/jctc/scripts/system_visuals.py")


def test_a_gmin_interval_in_the_table_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "different operations. The numeric ceiling is applied separately in each arm's unit"
    assert tex.count(old) == 1
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old + " a Newcombe interval is 0.1 to 0.7;")
    audit_mod.manuscript_gmin(audit)
    assert [r[2] for r in audit.drifts()] == ["tab:gmin reports no cross-arm tests"]


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
    old = ("At the submitted revision, the hit counts for this code and GMIN are 9 and "
           "0 on M\\(_{55}\\) at \\(\\rho=14\\), 6 and 0 on M\\(_{80}\\) at \\(\\rho=14\\), "
           "and 4 and 1 on Au\\(_{55}\\).")
    assert old in audit_mod.collapse(tex)
    audit.text_cache[audit_mod.TEX] = replace_collapsed(tex, old, old.replace("are 9 and", "are 7 and"))
    audit_mod.manuscript_gmin(audit)
    assert [(r[2], r[3]) for r in audit.drifts()] == [("prose", old)]


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
    old = "Template-free basin hopping with adaptive moves for funneled cluster landscapes"
    assert old in audit_mod.collapse(tex)
    audit.text_cache[audit_mod.TEX] = replace_collapsed(
        tex,
        old, "Template-free basin hopping across funnels with adaptive moves")
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["no overclaim: across funnels", "prose"]
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


def test_a_tip4p_cross_arm_interval_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "no cross-arm hit, work, or wall test is reported."
    assert tex.count(old) == 2
    start, rest = tex.split("% tip4p-gmin table: start", 1)
    audit.text_cache[audit_mod.TEX] = start + "% tip4p-gmin table: start" + rest.replace(
        old, old + " A Newcombe interval is 0.1 to 0.7.", 1)
    audit_mod.manuscript_knockouts(audit)
    assert {r[2] for r in audit.drifts()} == {"tab:tip4p-gmin reports no cross-arm test"}


def test_a_limits_section_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex.replace("\\section{Conclusions}", "\\section{Limits}\nText.\n\n\\section{Conclusions}")
    audit_mod.todo_check(audit)
    assert [r[2] for r in audit.drifts()] == ["no hedge: \\section{Limits}"]


def test_a_caption_missing_the_serial_patch_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.SI)
    old = "(\\texttt{JP=1})"
    assert old in tex
    audit.text_cache[audit_mod.SI] = tex.replace(old, "")
    audit_mod.gmin_patches(audit)
    assert [r[2] for r in audit.drifts()] == ["the GMIN provenance names the serial run index"]


def test_an_uncharged_gmin_panel_miss_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows("bench/gmin-panel/rows.csv")
    miss = next(r for r in reversed(rows) if r["system"] == "au55" and r["code"] == "gmin" and r["hit"] == "0")
    miss["charged"] = "50000"
    audit_mod.manuscript_gmin(audit)
    assert "row invariants" in {r[2] for r in audit.drifts()}


def test_a_stale_gmin_panel_source_patch_digest_is_reported():
    audit = audit_mod.Audit()
    path = "bench/gmin-panel/gmin-source.txt"
    audit.text_cache[path] = audit.text(path).replace(
        "patch=bench/lj104-lattice/v2/h2h-gmin.patch 3bf5d49a0a1cb0512a228212482e76e98195946096ceae3b54e88ca4a43b7189",
        "patch=bench/lj104-lattice/v2/h2h-gmin.patch a31221a67883d20032a029e34e3d81bd62c5bcb786c97fff02a9c234adc34553",
    )
    panel, groups = audit_mod.gmin_panel_groups(audit)
    audit_mod.gmin_panel_readme(audit, panel, groups)
    assert [r[2] for r in audit.drifts()] == [
        "in-tree patch sha256 bench/lj104-lattice/v2/h2h-gmin.patch"
    ]


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


def sc_code_revisions(audit, path, system, arm, label="code revision"):
    """Values shown on the passing revision checks of one printed Sutton-Chen row."""
    return [r[3] for r in audit.results
            if r[0] == "OK" and r[2] == label and r[1].startswith(f"{path}:") and r[1].endswith(f" {system} {arm}")]


def test_an_old_sutton_chen_gmin_cell_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(SC_N12)
    audit.text_cache[SC_N12] = text.replace("| sc126-38 | gmin | 9cbec1a4 | 12/12 |", "| sc126-38 | gmin | 9cbec1a4 | 1/12 |")
    audit_mod.head_to_head(audit)
    assert [(r[1].split()[-2:], r[2], r[3], r[4]) for r in audit.drifts()] == [(["sc126-38", "gmin"], "hits", "1/12", "12/12")]
    assert sc_code_revisions(audit, SC_N12, "sc126-38", "gmin", "gmin pin") == ["9cbec1a4"]


def test_a_dropped_sutton_chen_rerun_row_is_reported():
    kept = audit_mod.Audit()
    audit_mod.head_to_head(kept)
    assert sc_code_revisions(kept, SC_N12, "sc108-38", "gmin", "gmin pin") == ["9cbec1a4"]
    audit = audit_mod.Audit()
    text = audit.text(SC_N12)
    old = "| sc108-38 | gmin | 9cbec1a4 | 3/12 | 400000 |  |  |\n"
    assert old in text
    audit.text_cache[SC_N12] = text.replace(old, "")
    audit_mod.head_to_head(audit)
    assert [(r[1], r[2], r[3], r[4]) for r in audit.drifts()] == [(SC_N12, "table row sc108-38 gmin", "missing", "shown")]
    assert sc_code_revisions(audit, SC_N12, "sc108-38", "gmin", "gmin pin") == []


def test_a_sutton_chen_harness_blob_off_its_revision_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows(SC_RERUN_ROWS)
    rows[0]["harness_blob"] = "0" * 40
    audit_mod.sc_h2h_harness_blob(audit)
    drifted = [(r[2], r[3], r[4]) for r in audit.drifts()]
    assert len(drifted) == 1
    assert drifted[0][0] == "harness_blob"
    assert "0" * 40 in drifted[0][1]
    assert drifted[0][2] == "ac6b3c85b4eb435c4787a8a7425ab7e6858f9b34"


def test_a_different_sutton_chen_relabel_is_rejected(monkeypatch):
    parent = "3f759aa8409fb49ad09e2b45d143dab2d21ae53c"
    monkeypatch.setattr(audit_mod, "SC_H2H_RELABEL", parent)
    audit = audit_mod.Audit()
    audit_mod.sc_h2h_code_relabel(audit)
    drifts = {r[2] for r in audit.drifts()}
    assert "relabel commit" in drifts
    assert "gmin code_revision relabel" in drifts
    shown = [r for r in audit.drifts() if r[2] == "gmin code_revision relabel"]
    assert shown[0][4] == audit_mod.SC_H2H_GMIN_PIN


def test_a_new_sutton_chen_rows_blob_is_rejected(monkeypatch):
    real = audit_mod.git

    def wrapped(*args, **kwargs):
        if args and args[0] == "hash-object":
            done = real(*args, **kwargs)
            done.stdout = "f" * 40 + "\n"
            return done
        return real(*args, **kwargs)

    monkeypatch.setattr(audit_mod, "git", wrapped)
    audit = audit_mod.Audit()
    audit_mod.sc_h2h_code_relabel(audit)
    drifts = {r[2]: r for r in audit.drifts()}
    assert drifts["rows blob"][3] == "f" * 40
    assert drifts["gmin code_revision relabel"][4] == audit_mod.SC_H2H_GMIN_PIN


def test_an_edited_sutton_chen_gmin_code_revision_is_rejected():
    audit = audit_mod.Audit()
    row = next(r for r in audit.rows(SC_RERUN_ROWS) if r["arm"] == "gmin")
    row["code_revision"] = "0" * 40
    audit_mod.sc_h2h_code_relabel(audit)
    drifts = [r for r in audit.drifts() if r[2] == "gmin code_revision relabel"]
    assert len(drifts) == 1
    assert "0" * 40 in drifts[0][3]
    assert not any(r[2] == "rows blob" for r in audit.drifts())
    assert not any(r[2] == "relabel commit" for r in audit.drifts())


def test_the_sutton_chen_harness_blob_matches_the_cited_harness():
    audit = audit_mod.Audit()
    audit_mod.sc_h2h_harness_blob(audit)
    assert audit.drifts() == []
    assert any(r[2] == "harness_blob" and r[3] == "ac6b3c85b4eb435c4787a8a7425ab7e6858f9b34" for r in audit.results)


def test_the_printed_sutton_chen_gmin_rows_have_one_harness():
    # The printed GMIN rows no longer name their harness; the stored rows
    # give it, and with it the GMIN data file of scripts/metal_h2h/run_sc.py.
    rows = audit_mod.Audit().rows("bench/paper-rerun/sc-h2h/rows.csv")
    assert {(r["harness_revision"], r["harness_blob"], r["gmin_pin"]) for r in rows if r["arm"] == "gmin"} == {
        ("d59b0b6cf02fc300ff618f79d65da365128eed5a", "ac6b3c85b4eb435c4787a8a7425ab7e6858f9b34",
         "9cbec1a446352bc9ef064bd2b1a5767194655919")}


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
    # The rerun summary holds one row per system and arm, and sc96-55 gmin is
    # already printed; sc96-99 is a system the rerun never ran.
    audit = audit_mod.Audit()
    summary = audit.rows(audit_mod.SC_RERUN_SUMMARY)
    summary.append({**summary[0], "system": "sc96-99", "arm": "gmin", "hits_over_n": "0/12"})
    audit_mod.head_to_head(audit)
    assert sorted((r[1], r[2], r[3], r[4]) for r in audit.drifts()) == [
        (SC_N12, "table row sc96-99 gmin", "missing", "shown"),
        ("bench/paper-rerun/sc-h2h/README.md", "table row sc96-99 gmin", "missing", "shown"),
        (f"{audit_mod.SC_RERUN_SUMMARY} sc96-99 gmin", "summary row", "shown", "no rows"),
    ]


def test_an_sc_n12_gmin_rerun_without_its_seed_rows_is_reported():
    # The sc-n12 README no longer prints harness revisions, a test table or a
    # rerun sentence, so the stored rows must still back every summary line.
    audit = audit_mod.Audit()
    summary = audit.rows("bench/metal-cluster-h2h/sc-n12/results.csv")
    row = dict([r for r in summary if r["system"] == "sc96-55" and r["arm"] == "gmin"][-1])
    summary.append({**row, "harness_revision": "0" * 40})
    audit_mod.head_to_head(audit)
    readme = "bench/metal-cluster-h2h/sc-n12/README.md"
    assert sorted((r[1], r[2], r[3], r[4]) for r in audit.drifts()) == sorted([
        ("bench/metal-cluster-h2h/sc-n12/results.csv sc96-55 gmin 00000000", "per-seed rows", "12", "0"),
            ("bench/metal-cluster-h2h/sc-n12/pvalues.csv sc96-55", "per-seed rows", "12", "12 0"),
        (f"{readme}:131 sc96-55 a50f4b02", "latest GMIN harness", "a50f4b02", "00000000"),
        (f"{readme}:131 sc96-55 a50f4b02", "Calls ahead", "anneal, n.s.", "tie"),
        (f"{readme}:131 sc96-55 a50f4b02", "Wall ahead", "GMIN", "tie"),
        (readme, "table row sc96-55 gmin 00000000", "missing", "shown"),
        (readme, "test row sc96-55", "a50f4b02", "00000000"),
        (readme, "prose", "GMIN beats anneal on wall time on sc126-55 and sc108-80, at Mann-Whitney p < 0.05.", "present"),
    ])


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
    # Anneal rows name code_revision. GMIN rows name gmin_pin, so a rewritten
    # code_revision drifts the nine anneal rows in each of the two READMEs.
    assert labels.count("code revision") == 18
    assert "gmin pin" not in labels


def test_a_wrong_printed_gmin_pin_is_reported():
    audit = audit_mod.Audit()
    for path in (SC_N12, "bench/paper-rerun/sc-h2h/README.md"):
        audit.text_cache[path] = audit.text(path).replace("| gmin | 9cbec1a4 |", "| gmin | 61fa67cf |")
    audit_mod.head_to_head(audit)
    pins = [r for r in audit.drifts() if r[2] == "gmin pin"]
    assert len(pins) == 18
    assert {r[3] for r in pins} == {"61fa67cf"}
    assert {r[4] for r in pins} == {"9cbec1a4"}


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


def test_a_gmin_lead_in_the_earlier_harness_rows_must_be_stated_as_the_rows_give_it():
    audit = audit_mod.Audit()
    text = audit.text(SC_N12)
    audit.text_cache[SC_N12] = text.replace(
        "| 0.349 | tie | anneal, n.s. | GMIN |",
        "| 0.349 | tie | anneal, n.s. | anneal |",
    )
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


def test_a_drifted_gfn2_c20_revision_table_cell_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(GFN2_README)
    old = "| C20 | anneal G, `5351f272` | 0/12 | 100,000 |"
    assert old in text
    audit.text_cache[GFN2_README] = text.replace(old, "| C20 | anneal G, `5351f272` | 0/12 | 99,999 |")
    audit_mod.covalent_gfn2(audit)
    assert [r[2] for r in audit.drifts()] == ["median calls"]


def test_a_gfn2_revision_table_the_audit_cannot_find_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(GFN2_README)
    start = text.rindex("| system | code | hits |", 0, text.index("| C20 | anneal G, `5351f272`"))
    end = text.index("\n", start)
    audit.text_cache[GFN2_README] = text[:start] + text[start:end].replace("best / eV", "best (eV)") + text[end:]
    audit_mod.covalent_gfn2(audit)
    assert [(r[1].split()[-2], r[2]) for r in audit.drifts()] == [("C20", "GFN2 table row")]


def test_voided_si13_rows_are_not_audited_as_valid_results():
    audit = audit_mod.Audit()
    rows = audit.rows("bench/covalent-h2h/gfn2/rows.csv")
    row = next(r for r in rows if r["system"] == "si13" and r["code"] == "crest" and r["evals_to_hit"] == "143630")
    row["evals_to_hit"] = "150000"
    audit_mod.covalent_gfn2(audit)
    assert "Si13" not in audit_mod.GFN2_SYSTEMS
    assert all(test[0] != "Si13" for test in audit_mod.GFN2_REVISION_TESTS)
    assert not audit.drifts()


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
    listed_rows = {", ".join(SC_N12_ROWS)}
    assert {r[1] for r in audit.results} == {f"{audit_mod.ALIASES} {line[:8]}" for line in table} | listed_rows
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


def remote_tip(cwd=ROOT):
    """The tip of a branch in the audit's own remote set, develop first and then main where the set has it.

    A clone holds every branch of origin, and the audit counts a revision as
    pushed only when a ref of audit_remote_refs() contains it; the first ref
    in refs/remotes order, or the newest, is therefore not a tip the audit
    knows."""
    refs = audit_mod.audit_remote_refs()
    preferred = [ref for ref in ("refs/remotes/origin/develop", "refs/remotes/origin/main") if ref in refs]
    for ref in dict.fromkeys((*preferred, *refs)):
        found = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                               cwd=cwd, capture_output=True, text=True, check=False)
        if found.returncode == 0:
            return found.stdout.strip()
    pytest.fail("no audited remote branch is present: fetch develop from origin "
                "(git fetch origin '+refs/heads/*:refs/remotes/origin/*')")


def remote_layout(path, *branches):
    """A repository whose refs/remotes hold one empty-tree commit for each named origin branch.

    A branch named later has the later commit date, so the last one is the newest ref."""
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}

    def run(*args, stdin=None):
        return subprocess.run(["git", *args], cwd=path, input=stdin, capture_output=True, text=True, check=True,
                              env=env).stdout.strip()

    path.mkdir()
    run("init", "-q")
    tree = run("hash-object", "-t", "tree", "-w", "--stdin", stdin="")
    tips = {}
    for number, branch in enumerate(branches):
        env["GIT_COMMITTER_DATE"] = env["GIT_AUTHOR_DATE"] = f"@{1_000_000 + 100 * number} +0000"
        tips[branch] = run("commit-tree", tree, "-m", f"tip of {branch} {number}")
        run("update-ref", f"refs/remotes/origin/{branch}", tips[branch])
    return tips


def test_the_remote_tip_is_an_audited_branch_whatever_sorts_first(tmp_path):
    tips = remote_layout(tmp_path / "ci", "anneal-gmin-pin-followup3", "bench/crest-gaps", "main", "develop")
    assert remote_tip(tmp_path / "ci") == tips["develop"]
    assert remote_tip(tmp_path / "ci") != tips["anneal-gmin-pin-followup3"]
    tips = remote_layout(tmp_path / "no-develop", "anneal-gmin-pin-followup3", "main")
    assert remote_tip(tmp_path / "no-develop") == tips["main"]
    tips = remote_layout(tmp_path / "audited-only", "anneal-gmin-pin-followup3", "fix/gmin-sc2", "bench/gmin-sc3")
    assert remote_tip(tmp_path / "audited-only") == tips["bench/gmin-sc3"]


def newest_remote(path):
    return subprocess.run(["git", "for-each-ref", "--sort=-committerdate", "--count=1", "--format=%(refname)",
                           "refs/remotes"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()


def test_the_remote_tip_is_never_chosen_for_being_the_newest_ref(tmp_path):
    tips = remote_layout(tmp_path / "newest-main", "develop", "bench/crest-gaps", "zzz-unaudited", "main")
    assert newest_remote(tmp_path / "newest-main") == "refs/remotes/origin/main"
    assert remote_tip(tmp_path / "newest-main") == tips["develop"]
    tips = remote_layout(tmp_path / "newest-foreign", "develop", "main", "zzz-unaudited")
    assert newest_remote(tmp_path / "newest-foreign") == "refs/remotes/origin/zzz-unaudited"
    assert remote_tip(tmp_path / "newest-foreign") == tips["develop"]


def test_the_remote_tip_takes_main_only_while_the_audit_counts_it(tmp_path, monkeypatch):
    tips = remote_layout(tmp_path / "set", "bench/crest-gaps", "main")
    assert newest_remote(tmp_path / "set") == "refs/remotes/origin/main"
    assert "refs/remotes/origin/main" in audit_mod.audit_remote_refs()
    assert remote_tip(tmp_path / "set") == tips["main"]
    monkeypatch.setattr(audit_mod, "AUDIT_OPTIONAL_REMOTES",
                        tuple(b for b in audit_mod.AUDIT_OPTIONAL_REMOTES if b != "main"))
    assert "refs/remotes/origin/main" not in audit_mod.audit_remote_refs()
    assert remote_tip(tmp_path / "set") == tips["bench/crest-gaps"]
    remote_layout(tmp_path / "only-main", "main")
    with pytest.raises(pytest.fail.Exception, match="no audited remote branch is present"):
        remote_tip(tmp_path / "only-main")


def test_the_remote_tip_fails_clearly_when_no_audited_branch_is_present(tmp_path):
    remote_layout(tmp_path / "foreign", "anneal-gmin-pin-followup3", "zzz")
    with pytest.raises(pytest.fail.Exception, match="no audited remote branch is present"):
        remote_tip(tmp_path / "foreign")
    remote_layout(tmp_path / "bare")
    with pytest.raises(pytest.fail.Exception, match="no audited remote branch is present"):
        remote_tip(tmp_path / "bare")


def synthetic_audit(*rows, revisions=("code_revision", "code_tree")):
    audit = audit_mod.Audit()
    audit.read["synthetic.csv"] = [dict(zip(revisions, row), host="bench") for row in rows]
    return audit


LOST = "0123456789abcdef0123456789abcdef01234567"


def test_a_recorded_tree_no_pushed_commit_has_resolves_nothing():
    audit = synthetic_audit((LOST, "f" * 40))
    assert audit_mod.provenance(audit) == {"synthetic.csv": f"revision {LOST[:8]} is not in this repository"}
    assert [r[2] for r in audit.drifts()] == [f"unpushed revision {LOST[:8]}"]


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


def test_an_unmapped_revision_in_any_benchmark_table_fails(monkeypatch):
    tables = audit_mod.revision_row_tables()
    path = "bench/ase-nonlj/summary.csv"
    altered = []
    for table_path, rows in tables:
        if table_path == path:
            rows = [dict(row) for row in rows]
            rows[0]["code_revision"] = "0" * 40
        altered.append((table_path, rows))
    monkeypatch.setattr(audit_mod, "revision_row_tables", lambda: altered)
    audit = audit_mod.Audit()
    audit_mod.all_revision_provenance(audit)
    assert [row[2] for row in audit.drifts() if row[1] == path] == [
        "row revision 00000000 is reachable or mapped"
    ]


def test_recorded_revision_cell_gaps_match_the_tree():
    assert audit_mod.revision_cell_gaps() == audit_mod.RECORDED_REVISION_CELL_GAPS


def test_a_malformed_revision_cell_is_rejected():
    tables = [("bench/example/rows.csv", [{"code_revision": "not-a-sha"}])]
    assert audit_mod.revision_cell_gaps(tables) == {
        ("bench/example/rows.csv", "code_revision", "not-a-sha"): 1
    }
    short = [("bench/example/rows.csv", [{"code_revision": "abc123"}])]
    assert audit_mod.revision_cell_gaps(short) == {
        ("bench/example/rows.csv", "code_revision", "abc123"): 1
    }
    upper = [("bench/example/rows.csv", [{"code_revision": "A" * 40}])]
    assert ("bench/example/rows.csv", "code_revision", "A" * 40) in audit_mod.revision_cell_gaps(upper)


def test_an_empty_required_revision_cell_is_rejected():
    tables = [("bench/example/rows.csv", [{"code_revision": "", "harness_revision": "", "head": "unknown"}])]
    assert audit_mod.revision_cell_gaps(tables) == {
        ("bench/example/rows.csv", "code_revision", ""): 1,
        ("bench/example/rows.csv", "head", "unknown"): 1,
    }


def test_an_empty_harness_revision_is_allowed():
    tables = [("bench/example/rows.csv", [{"harness_revision": "  "}])]
    assert audit_mod.revision_cell_gaps(tables) == {}


def test_a_new_malformed_revision_cell_fails_the_audit(monkeypatch):
    tables = audit_mod.revision_row_tables()
    altered = []
    for path, rows in tables:
        if path == "bench/ase-nonlj/summary.csv":
            rows = [dict(row) for row in rows]
            rows[0]["code_revision"] = "not-a-sha"
        altered.append((path, rows))
    monkeypatch.setattr(audit_mod, "revision_row_tables", lambda: altered)
    audit = audit_mod.Audit()
    audit_mod.all_revision_provenance(audit)
    assert [row[2] for row in audit.drifts() if row[1] == "revision cells"] == [
        "malformed and required-empty revision cells match the record"
    ]


def test_an_uncited_unpushed_revision_is_reported(monkeypatch):
    unpushed = dict(audit_mod.UNPUSHED)
    unpushed["f" * 40] = "not cited anywhere"
    monkeypatch.setattr(audit_mod, "UNPUSHED", unpushed)
    audit = audit_mod.Audit()
    audit_mod.all_revision_provenance(audit)
    assert [row[2] for row in audit.drifts() if row[1] == audit_mod.REVISION_MAP] == [
        "mapped revision ffffffff is cited"
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
                    if c in r and not audit_mod.external_gmin_identifier(r, c)
                    and re.fullmatch(r"[0-9a-f]{7,40}", r[c].strip())})
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
    audit_mod.pending_text(audit)
    assert [(row[1], row[2]) for row in audit.drifts()] == [
        (target, "no pending, to be assigned, or TBD remains in TeX")
    ]


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
    doubled_label = ["label" + "l" + suffix for suffix in ("ed", "ing")]
    for word in (*doubled_label, "modelled", "travelled"):
        assert audit_mod.BRITISH.search(word)
    assert not audit_mod.BRITISH.search("labeled")


CONF_MISS4 = audit_mod.CONF_MISS4


def test_conf_miss4_recomputes_the_exact_tests_without_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    import random

    rng = random.Random(4)
    for n in (1, 5, 12, 24):
        diffs = [rng.uniform(-1, 1) + 0.2 for _ in range(n)]
        assert audit_mod.signed_rank_exact(diffs) == pytest.approx(
            float(scipy_stats.wilcoxon(diffs, method="exact").pvalue), rel=1e-9
        )
    assert audit_mod.signed_rank_exact([0.0, 0.0]) == 1.0
    assert audit_mod.signed_rank_exact([1.0, 2.0, 3.0, 4.0, 5.0, 0.0]) == pytest.approx(2 / 32)
    for k, n in ((0, 6), (2, 7), (3, 6), (6, 6)):
        assert audit_mod.binomial_half(k, n) == pytest.approx(float(scipy_stats.binomtest(k, n, 0.5).pvalue))
    assert audit_mod.binomial_half(0, 0) == 1.0


def test_a_drifted_conf_miss4_panel_cell_is_reported():
    audit = audit_mod.Audit()
    path = f"{CONF_MISS4}/panel.tsv"
    audit.text_cache[path] = audit.text(path).replace("\t15/24\t15/24\t16888", "\t14/24\t15/24\t16888", 1)
    audit_mod.conf_miss4(audit)
    assert [r[2] for r in audit.drifts()] == ["hits"]


def test_a_drifted_conf_miss4_wilcoxon_cell_is_reported():
    audit = audit_mod.Audit()
    path = f"{CONF_MISS4}/paired.tsv"
    audit.text_cache[path] = audit.text(path).replace("\t0.0179\t", "\t0.0189\t", 1)
    audit_mod.conf_miss4(audit)
    assert sorted(r[2] for r in audit.drifts()) == ["p", "wilcoxon_p"]


def test_a_conf_miss4_row_citing_a_missing_commit_is_reported():
    audit = audit_mod.Audit()
    for r in audit.rows(f"{CONF_MISS4}/guard.csv"):
        if r["code_revision"] != "61fa67cf380e46876b569da45e7cc091647a775b":
            r["code_revision"] = "0123456789abcdef0123456789abcdef01234567"
    audit_mod.conf_miss4(audit)
    assert sorted(r[2] for r in audit.drifts()) == [
        "cited revision 01234567 is a commit object",
        "cited revision 01234567 is on its provenance merge parent",
        "cited revision 01234567 is reachable from origin",
        "every revision the rows cite is listed",
    ]


def test_conf_miss4_requires_the_documented_two_parent_merge(monkeypatch):
    monkeypatch.setattr(
        audit_mod,
        "CONF_MISS4_PROVENANCE_MERGE",
        "5c40c711f6f49ce0950d6025349adcde7f17cdce",
    )
    audit = audit_mod.Audit()
    audit_mod.conf_miss4(audit)
    assert "the documented provenance merge has two parents" in {
        r[2] for r in audit.drifts()
    }


def test_conf_miss4_checks_revisions_against_their_merge_parent():
    audit = audit_mod.Audit()
    path = f"{CONF_MISS4}/provenance.tsv"
    audit.text_cache[path] = audit.text(path).replace(
        "anneal and openconf rows; anneal wheel",
        "develop base",
        1,
    )
    audit_mod.conf_miss4(audit)
    assert "cited revision 259befae is on its provenance merge parent" in {
        r[2] for r in audit.drifts()
    }


def test_conf_miss4_requires_one_revision_for_both_paired_arms():
    audit = audit_mod.Audit()
    row = next(r for r in audit.rows(f"{CONF_MISS4}/rows.csv") if r["arm"] == "openconf")
    row["code_revision"] = "5c40c711f6f49ce0950d6025349adcde7f17cdce"
    audit_mod.conf_miss4(audit)
    assert [r[2] for r in audit.drifts()] == [
        "anneal and openconf carry the paired runner's one revision"
    ]


def test_a_conf_miss4_guard_mismatch_and_a_moved_mirror_hit_are_reported():
    audit = audit_mod.Audit()
    rows = audit.rows(f"{CONF_MISS4}/guard.csv")
    series = next(r for r in rows if r["code_revision"] != "61fa67cf380e46876b569da45e7cc091647a775b")
    series["charged"] = str(int(series["charged"]) + 1)
    audit_mod.conf_miss4(audit)
    assert [r[2] for r in audit.drifts()] == ["156 jobs, none differs between develop and the series"]
    audit = audit_mod.Audit()
    mirror = next(r for r in audit.rows(f"{CONF_MISS4}/rows.csv") if r["hit_mirror"] == "1" and r["hit"] == "0")
    mirror["hit_mirror"] = "0"
    audit_mod.conf_miss4(audit)
    labels = {r[2] for r in audit.drifts()}
    assert {"hits_mirror", "hit_mirror_after", "the runs that count only as a mirror hit"} <= labels


def test_a_conf_miss4_provenance_tree_that_is_not_the_commits_is_reported():
    audit = audit_mod.Audit()
    path = f"{CONF_MISS4}/provenance.tsv"
    first = audit.text(path).splitlines()[1].split("\t")[2]
    audit.text_cache[path] = audit.text(path).replace(first, "0" * 40)
    audit_mod.conf_miss4(audit)
    assert sorted(r[2] for r in audit.drifts() if "tree of" in r[2]) == [
        "tree of 259befae (anneal and openconf rows; anneal wheel)"
    ]


def test_a_conf_miss4_run_order_claim_is_checked_against_the_schedule():
    audit = audit_mod.Audit()
    path = f"{CONF_MISS4}/README.md"
    audit.text_cache[path] = audit.text(path).replace(
        "Develop ran first on 98 of the 192 pairs and anneal on 94.",
        "Develop ran first on 97 of the 192 pairs and anneal on 95.",
        1,
    )
    audit_mod.conf_miss4(audit)
    assert [r[2] for r in audit.drifts()] == ["prose"]

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


@pytest.mark.parametrize("path", ["bench/tip4p-h2h/wall-r2.csv", "bench/tip4p-h2h/check-r2.md"])
def test_generated_tip4p_summaries_are_append_only_records(path):
    assert path in audit_mod.APPEND_ONLY_RECORDS


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


def _stub_snakefile(text: str, stamp_root: Path) -> str:
    text = text.replace('"/work/', f'"{stamp_root.as_posix()}/')
    shell = (
        '    shell:\n'
        '        """\n'
        "        set -euo pipefail\n"
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


def _stamp_env(stamp_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ANNEAL_STAMP_ROOT"] = str(stamp_root)
    return env


def test_stubbed_snakemake_force_leaves_tracked_files_unchanged(tmp_path):
    pytest.importorskip("snakemake")
    stamp_root = tmp_path / "stamps"
    stamp_root.mkdir()
    source = (ROOT / "workflow/Snakefile").read_text()
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    for name, body in re.findall(r"(?ms)^rule (\w+):\n(.*?)(?=^rule \w+:|\Z)", source):
        output = re.search(r"(?ms)^    output:\n(.*?)(?=^    \w|\Z)", body)
        declared = re.findall(r"""["']([^"']+)["']""", output.group(1) if output else "")
        assert not set(declared) & tracked, f"{name} declares tracked outputs"
    stub = tmp_path / "Snakefile"
    stub.write_text(_stub_snakefile(source, stamp_root))
    before = _tracked_status()
    completed = subprocess.run(
        [
            sys.executable, "-m", "snakemake", "-s", str(stub),
            *_concrete_rules(source), "--cores", "1", "--force", "-k", "--nocolor",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_stamp_env(stamp_root),
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert _tracked_status() == before


def test_a_tracked_snakemake_output_is_deleted_on_force(tmp_path):
    pytest.importorskip("snakemake")
    stamp_root = tmp_path / "stamps"
    stamp_root.mkdir()
    target = ROOT / "bench/gmin-panel/summary.csv"
    original = target.read_bytes()
    source = (ROOT / "workflow/Snakefile").read_text().replace(
        'touch(STAMP_ROOT + "/gmin_panel.stamp")',
        '"bench/gmin-panel/summary.csv"',
    )
    stub = tmp_path / "Snakefile"
    stub.write_text(_stub_snakefile(source, stamp_root))
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
            env=_stamp_env(stamp_root),
        )
        assert target.read_bytes() != original
    finally:
        target.write_bytes(original)


def test_the_landfold_figure_records_its_redrawn_panels():
    audit = audit_mod.Audit()
    audit_mod.manuscript_new(audit)
    rows = [row for row in audit.results if "landfold" in row[2]]
    assert rows
    assert all(row[0] == "OK" for row in rows)


def pdf_length():
    spec = importlib.util.spec_from_file_location("pdf_length", ROOT / "paper/jctc/scripts/pdf_length.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_rendered_abstract_counts_pdftotext_tokens():
    length = pdf_length()
    abstract = " ".join(["word"] * 150) + " on 9 of 13 at 0.317 and GFN2. " + "end " * 40
    text = (f"Title\nAbstract\n{abstract}\n\n1\n\f2\n\nIntroduction\n" + "body " * 6000
            + "12 of 12\n\n2\n\nThe search\n" + "3\n\fAuthor contributions\nnot counted\n")
    counts = length.measure(text)
    assert counts == {"abstract_words": 150 + 8 + 40 + 2, "abstract_numerals": 4, "body_words": 6006}
    assert not length.problems(counts)
    assert length.problems(dict(counts, abstract_words=201, abstract_numerals=7, body_words=7001)) == [
        "abstract has 201 words, more than 200", "abstract has 7 numerals, more than 6",
        "body has 7001 words, outside 5000-7000"]


def test_a_mixed_stopping_pair_is_reported():
    """Putting a fixed budget back beside a full-run median is a drift."""
    audit = audit_mod.Audit()
    path = audit_mod.TEX
    audit.text_cache[path] = audit.text(path) + "\n4000 & 3055\n"
    audit_mod.stopping_pairs(audit)
    assert [row for row in audit.drifts() if row[2] == "a mixed stopping pair stays out" and row[3] == "4000 & 3055"] == [
        ("DRIFT", path, "a mixed stopping pair stays out", "4000 & 3055", "present")]


def test_a_missed_printed_definition_is_a_drift():
    """A definition pattern that no longer matches the source is a drift, not a skip."""
    audit = audit_mod.Audit()
    path = "src/methods/cluster_hopping/config.rs"
    text = audit.text(path)
    old = "tabu_capacity: 8,"
    assert text.count(old) == 1
    audit.text_cache[path] = text.replace(old, "tabu_size: 8,", 1)
    audit_mod.printed_definitions(audit)
    assert [row for row in audit.drifts() if row[2] == "tabu_capacity is printed"] == [
        ("DRIFT", path, "tabu_capacity is printed", "matched", "absent")]


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


def test_the_label_files_and_the_length_receipt_match_the_sources():
    audit = audit_mod.Audit()
    audit_mod.build_receipts(audit)
    assert len(audit.results) == 6
    assert not audit.drifts(), audit.drifts()


def receipt_tree(tmp_path, monkeypatch):
    """A copy of the paper sources, label files, receipt and their scripts that the audit reads as ROOT."""
    for name in ("manuscript.tex", "si.tex", "macros.tex", "labels-manuscript.tex", "labels-si.tex",
                 "logs/pdf-length.txt", "scripts/cross_labels.py", "scripts/pdf_length.py"):
        dest = tmp_path / "paper/jctc" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "paper/jctc" / name, dest)
    for name in ("cross_labels", "pdf_length"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(audit_mod, "ROOT", tmp_path)
    return tmp_path / "paper/jctc"


def receipt_drifts():
    audit = audit_mod.Audit()
    audit_mod.build_receipts(audit)
    assert len(audit.results) == 6
    return [(where, label) for _, where, label, _, _ in audit.drifts()]


def test_a_label_file_left_behind_its_source_fails_the_audit(tmp_path, monkeypatch):
    jctc = receipt_tree(tmp_path, monkeypatch)
    assert receipt_drifts() == []
    si = jctc / "si.tex"
    si.write_text(si.read_text().replace("\\begin{document}", "\\begin{document}\n\\section{Added}\\label{sec:added}", 1))
    assert receipt_drifts() == [("paper/jctc/labels-si.tex", "cross_labels.py output for si.tex")]


def test_a_length_receipt_left_behind_its_source_fails_the_audit(tmp_path, monkeypatch):
    jctc = receipt_tree(tmp_path, monkeypatch)
    macros = jctc / "macros.tex"
    macros.write_text(macros.read_text() + "% edited\n")
    assert receipt_drifts() == [("paper/jctc/logs/pdf-length.txt", "sha256 of paper/jctc/macros.tex")]


def test_a_length_over_the_limit_is_shown_without_failing_the_audit(tmp_path, monkeypatch):
    jctc = receipt_tree(tmp_path, monkeypatch)
    receipt = jctc / "logs/pdf-length.txt"
    text = re.sub(r"(?m)^abstract_words=\d+$", "abstract_words=120", receipt.read_text())
    text = re.sub(r"(?m)^abstract_numerals=\d+$", "abstract_numerals=4", text)
    receipt.write_text(re.sub(r"(?m)^body_words=\d+$", "body_words=6000", text))
    audit = audit_mod.Audit()
    audit_mod.build_receipts(audit)
    assert not audit.drifts() and not audit.expected()
    receipt.write_text(re.sub(r"(?m)^body_words=\d+$", "body_words=7001", receipt.read_text()))
    audit = audit_mod.Audit()
    audit_mod.build_receipts(audit)
    assert not audit.drifts()
    assert [row[1:] for row in audit.expected()] == [
        ("paper/jctc/logs/pdf-length.txt", "rendered length within the journal limits", "within",
         "body has 7001 words, outside 5000-7000")]


def test_a_missing_length_receipt_or_count_fails_the_audit(tmp_path, monkeypatch):
    jctc = receipt_tree(tmp_path, monkeypatch)
    receipt = jctc / "logs/pdf-length.txt"
    receipt.write_text(re.sub(r"(?m)^body_words=\d+\n", "", receipt.read_text()))
    assert receipt_drifts() == [("paper/jctc/logs/pdf-length.txt", "pdf_length.py counts")]
    receipt.unlink()
    audit = audit_mod.Audit()
    audit_mod.build_receipts(audit)
    assert [row[1:] for row in audit.drifts()] == [
        ("paper/jctc/logs/pdf-length.txt", "pdf_length.py receipt", "present", "missing")]


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
        [sys.executable, str(ROOT / "scripts/session_metadata.py"), "--output", str(out), "--campaign", "water-crest-fair-n192",
         "--root", str(ROOT), "--anneal-repro", str(repro), "--", "snakemake", "water_crest_fair"],
        cwd=ROOT, check=False, capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    row = json.loads(out.read_text())
    head = subprocess.run(["git", "-C", str(repro), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert row["anneal_repro"]["revision"] == head
    assert re.fullmatch(r"[0-9a-f]{40}", row["anneal_repro"]["tree"])
    assert row["anneal_repro"]["dirty"] is False
    assert row["openmp"]["omp_num_threads"] == "3"
    assert row["openmp"]["settings"]["OMP_NUM_THREADS"] == "3"


def checkout_for_dirty_tests(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text("a\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.hooksPath=/dev/null",
           "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "a"], check=True)
    return repo


def test_session_metadata_records_whether_the_checkout_is_dirty(tmp_path):
    module = session_metadata_module()
    repo = checkout_for_dirty_tests(tmp_path)
    assert module.git_dirty(repo) is False
    assert module.record("c", ["true"], repo)["dirty"] is False
    (repo / "a.txt").write_text("changed\n")
    assert module.git_dirty(repo) is True
    assert module.record("c", ["true"], repo)["dirty"] is True
    (repo / "a.txt").write_text("a\n")
    assert module.git_dirty(repo) is False
    (repo / "new.txt").write_text("untracked\n")
    assert module.git_dirty(repo) is True
    assert module.repro_checkout(repo)["dirty"] is True


def test_session_metadata_does_not_guess_when_git_cannot_say(tmp_path, monkeypatch):
    module = session_metadata_module()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    bare = tmp_path / "not-a-checkout"
    bare.mkdir()
    assert module.git_dirty(bare) is None
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    assert module.git_dirty(tmp_path) is None


def test_the_session_record_of_this_checkout_matches_git_status():
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
    module = session_metadata_module()
    assert module.record("c", ["true"], ROOT)["dirty"] == bool(status.stdout.strip())


MAPS = """\
7f0000000000-7f0000020000 r--p 00000000 08:01 11 /usr/lib/x86_64-linux-gnu/libc.so.6
7f0000100000-7f0000120000 r-xp 00000000 08:01 12 /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
7f0000200000-7f0000220000 r--p 00000000 08:01 12 /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
7f0000300000-7f0000320000 r-xp 00000000 08:01 13 /opt/lib/libomp.so.5 (deleted)
7f0000400000-7f0000420000 rw-p 00000000 00:00 0
"""


def test_the_openmp_runtime_is_read_from_the_mapped_libraries():
    module = session_metadata_module()
    found = module.loaded_openmp(MAPS)
    assert sorted(found) == ["libgomp", "libomp"]
    assert found["libgomp"] == "/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0"
    assert module.loaded_openmp("7f00-7f01 r-xp 0 08:01 1 /usr/lib/libc.so.6\n") == {}
    assert module.openmp_stem("libiomp5.so") == "libiomp5"
    assert module.openmp_stem("/usr/local/lib/libomp.dylib") == "libomp"
    assert module.openmp_stem("libgompx.so.1") is None
    assert module.openmp_stem("libm.so.6") is None


def test_the_named_openmp_runtime_is_the_one_in_the_evidence(tmp_path, monkeypatch):
    module = session_metadata_module()
    maps = tmp_path / "maps"
    maps.write_text(MAPS.replace("/opt/lib/libomp.so.5 (deleted)", "/usr/lib/libc.so.6"))
    loaded = module.identify_openmp(["true"], maps)
    assert loaded["library"] == "libgomp"
    assert loaded["path"] == "/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0"
    assert loaded["source"] == "loaded in the recorder process"

    maps.write_text("7f00-7f01 r-xp 0 08:01 1 /usr/lib/libc.so.6\n")
    monkeypatch.setattr(module, "needed_libraries", lambda executable: ["libm.so.6", "libiomp5.so", "libc.so.6"])
    named = module.identify_openmp(["./campaign"], maps)
    assert named["library"] == "libiomp5"
    assert named["path"] is None
    assert named["source"] == "dynamic section of ./campaign"

    monkeypatch.setattr(module, "needed_libraries", lambda executable: ["libgomp.so.1", "libomp.so.5"])
    both = module.identify_openmp(["./campaign"], maps)
    assert both["library"] == "unknown"
    assert both["candidates"] == ["libgomp", "libomp"]


def test_an_unidentifiable_openmp_runtime_is_unknown_even_where_libraries_are_installed(tmp_path, monkeypatch):
    module = session_metadata_module()
    maps = tmp_path / "maps"
    maps.write_text("7f00-7f01 r-xp 0 08:01 1 /usr/lib/libc.so.6\n")
    monkeypatch.setattr(module, "needed_libraries", lambda executable: None)
    shown = module.identify_openmp(["snakemake", "x"], maps)
    assert shown["library"] == "unknown" and shown["path"] is None
    assert shown["candidates"] == []
    absent = module.identify_openmp(None, tmp_path / "no-maps")
    assert absent["library"] == "unknown"
    monkeypatch.setattr(module, "needed_libraries", lambda executable: ["libm.so.6"])
    assert module.identify_openmp(["./campaign"], maps)["library"] == "unknown"
    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/" + name if name == "ldconfig" else None)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout="\tlibgomp.so.1 (libc6,x86-64) => /usr/lib/libgomp.so.1\n", stderr=""))
    assert module.openmp_runtime()["libraries"] == {"libgomp": str(Path("/usr/lib/libgomp.so.1").resolve())}
    assert module.identify_openmp(["./campaign"], maps)["library"] == "unknown"


def test_needed_libraries_come_from_the_dynamic_section_without_running_the_file(tmp_path, monkeypatch):
    module = session_metadata_module()
    tool = tmp_path / "bin"
    tool.mkdir()
    readelf = tool / "readelf"
    readelf.write_text(
        "#!/bin/sh\n"
        'case "$2" in *campaign) ;; *) exit 1 ;; esac\n'
        'echo " 0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]"\n'
        'echo " 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]"\n'
    )
    readelf.chmod(0o755)
    campaign = tool / "campaign"
    campaign.write_text("#!/bin/sh\ntouch ran\n")
    campaign.chmod(0o755)
    script = tool / "script"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tool) + os.pathsep + "/usr/bin" + os.pathsep + "/bin")
    assert module.needed_libraries(str(campaign)) == ["libgomp.so.1", "libc.so.6"]
    assert module.needed_libraries(str(script)) is None
    assert module.needed_libraries("absent-command") is None
    assert not (tmp_path / "ran").exists() and not Path("ran").exists()


def test_the_session_record_names_its_openmp_runtime_and_checkout_state(tmp_path):
    out = tmp_path / "sessions.jsonl"
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/session_metadata.py"), "--output", str(out), "--campaign", "c",
         "--root", str(ROOT), "--", "true"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    row = json.loads(out.read_text())
    assert isinstance(row["dirty"], bool)
    assert set(row["openmp_in_use"]) >= {"library", "path", "source"}
    assert row["openmp_in_use"]["library"] in {"libgomp", "libomp", "libiomp5", "unknown"}


def test_a_missing_boot_id_is_recorded_as_null(tmp_path):
    module = session_metadata_module()
    assert module.boot_id_label(tmp_path / "boot_id") is None
    (tmp_path / "boot_id").write_text("0123abcd-0000-4000-8000-000000000000\n")
    assert module.boot_id_label(tmp_path / "boot_id") == machine_provenance.boot_label("0123abcd-0000-4000-8000-000000000000")


def crest_equal_runner():
    spec = importlib.util.spec_from_file_location("crest_equal_run_panel", ROOT / "scripts/crest_equal/run_panel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_crest_equal_runner_records_its_session_before_any_run(tmp_path):
    runner = crest_equal_runner()
    core = str(min(available_cpus()))
    out = tmp_path / "sessions.jsonl"
    runner.record_session(out, core, ["scripts/crest_equal/run_panel.py", "--seed", "2000"])
    row = json.loads(out.read_text())
    assert row["schema"] == "anneal-session-v2"
    assert row["campaign"] == "crest-equal-h2o6"
    assert row["command"] == ["scripts/crest_equal/run_panel.py", "--seed", "2000"]
    assert row["cpu_affinity"] == ([int(core)] if shutil.which("taskset") else available_cpus())
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
            sys.executable,
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


def test_tblite_receipt_falls_back_to_tomli_without_tomllib(tmp_path):
    """Python 3.10 has no tomllib: the receipt reads the metadata through tomli there."""
    metadata = tmp_path / "fpm.toml"
    metadata.write_text('name = "tblite"\nversion = "0.7.0"\n')
    code = (
        "import importlib.abc, importlib.util, sys, types\n"
        "try:\n"
        "    import tomllib as parser\n"
        "except ImportError:\n"
        "    import tomli as parser\n"
        "tomli = types.ModuleType('tomli')\n"
        "tomli.loads = parser.loads\n"
        "tomli.loaded_as = 'tomli'\n"
        "sys.modules['tomli'] = tomli\n"
        "sys.modules.pop('tomllib', None)\n"
        "class NoTomllib(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'tomllib':\n"
        "            raise ImportError('tomllib is not part of this Python')\n"
        "sys.meta_path.insert(0, NoTomllib())\n"
        "spec = importlib.util.spec_from_file_location('tblite_receipt', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert module.tomllib.loaded_as == 'tomli'\n"
        "print(module.receipt(open(sys.argv[2], 'rb').read())['version'])\n"
    )
    run = subprocess.run([sys.executable, "-c", code, str(ROOT / "paper/jctc/scripts/tblite_receipt.py"), str(metadata)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "0.7.0"


def test_the_test_dependencies_declare_tomli_for_python_310():
    marker = "tomli>=2; python_version < '3.11'"
    assert marker in (ROOT / "pyproject.toml").read_text()
    assert marker in (ROOT / ".github/workflows/ci.yml").read_text()


@pytest.mark.parametrize("path", ["CITATION.cff", ".zenodo.json"])
def test_a_swapped_citation_affiliation_order_is_rejected(path):
    audit = audit_mod.Audit()
    audit.text_cache[path] = audit.text(path).replace("SURF B.V.; TurtleTech ehf", "TurtleTech ehf; SURF B.V.")
    audit_mod.citation_metadata(audit)
    assert [row for row in audit.drifts() if row[1] == path]


def _unreachable_number(audit):
    return [
        row for row in audit.drifts()
        if row[2] == "a printed number traces to a commit this repository holds"
    ]


def test_withholding_narration_in_manuscript_or_si_fails():
    for doc in (audit_mod.TEX, audit_mod.SI):
        audit = audit_mod.Audit()
        audit.text_cache[doc] = audit.text(doc) + "\nThese hit counts are not quoted.\n"
        audit_mod.sutton_chen(audit)
        assert any(row[1] == doc and "withholding narration" in row[2] for row in audit.drifts())


def test_a_cite_key_missing_from_every_loaded_bibliography_fails():
    audit = audit_mod.Audit()
    audit.text_cache[audit_mod.TEX] = audit.text(audit_mod.TEX) + "\n\\cite{noSuchKey2026}\n"
    audit_mod.bibliography_keys(audit)
    assert [row for row in audit.drifts() if "noSuchKey2026" in row[4]]


def test_a_cite_key_of_the_second_loaded_bibliography_resolves():
    audit = audit_mod.Audit()
    audit_mod.bibliography_keys(audit)
    assert not audit.drifts()


def test_a_changed_figure_source_fails_its_receipt(tmp_path):
    (tmp_path / "paper/jctc").mkdir(parents=True)
    (tmp_path / "paper/jctc/fig.py").write_text("print(1)\n")
    (tmp_path / "paper/jctc/FIGURE-SOURCE.tsv").write_text(
        "path\tsource_sha256\tkind\npaper/jctc/fig.py\t" + "0" * 64 + "\tscript\n")
    audit = audit_mod.Audit()
    audit_mod.figure_source_scripts(audit, tmp_path)
    assert [row for row in audit.drifts() if "paper/jctc/fig.py" in row[2]]


def test_an_update_row_replaces_the_receipt_hash_of_a_figure_source(tmp_path):
    source = "print(1)\n"
    digest = audit_mod.hashlib.sha256(source.encode()).hexdigest()
    (tmp_path / "paper/jctc").mkdir(parents=True)
    (tmp_path / "paper/jctc/fig.py").write_text(source)
    (tmp_path / "paper/jctc/FIGURE-SOURCE.tsv").write_text(
        "path\tsource_sha256\tkind\npaper/jctc/fig.py\t" + "0" * 64 + "\tscript\n")
    (tmp_path / "paper/jctc/FIGURE-SOURCE-updates.tsv").write_text(
        "path\tsource_sha256\tkind\tstatus\tupdate_reason\n"
        f"paper/jctc/fig.py\t{digest}\tscript\tactive\tnew source\n")
    audit = audit_mod.Audit()
    audit_mod.figure_source_scripts(audit, tmp_path)
    assert not audit.drifts()


def test_a_typed_hartree_literal_in_source_fails():
    audit = audit_mod.Audit()
    path = "scripts/paired_stats.py"
    audit.text_cache[path] = audit.text(path) + "\nHARTREE = 27.2114\n"
    audit_mod.conversion_literals(audit)
    assert [row for row in audit.drifts() if row[1] == path]


def test_a_selection_table_cell_that_differs_from_its_rows_fails():
    audit = audit_mod.Audit()
    path = "bench/depth-posterior-ablation/README.md"
    text = audit.text(path)
    row = next(line for line in text.splitlines() if line.startswith("| LJ"))
    cells = row.split("|")
    cells[3] = " 999/999 "
    audit.text_cache[path] = text.replace(row, "|".join(cells))
    audit_mod.selection_tables(audit)
    assert [r for r in audit.drifts() if r[1].startswith(path)]


def test_a_superseded_list_that_loses_a_row_fails():
    audit = audit_mod.Audit()
    rows = audit.rows("bench/ase-au55/superseded.csv")
    audit.read["bench/ase-au55/superseded.csv"] = rows[1:]
    audit_mod.au55_superseded(audit)
    assert audit.drifts()


def test_a_retired_switch_that_config_stops_rejecting_fails():
    audit = audit_mod.Audit()
    path = "src/methods/cluster_hopping/config.rs"
    audit.text_cache[path] = audit.text(path).replace('"prior_core"', '"prior_x"')
    audit_mod.prior_switches(audit)
    assert [row for row in audit.drifts() if "prior_core" in row[2]]


NUMPY_BLOCK = """
import importlib.abc, importlib.util, sys

class NoNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("numpy", "scipy", "matplotlib"):
            raise ImportError(f"{name} is not installed")

sys.meta_path.insert(0, NoNumpy())
spec = importlib.util.spec_from_file_location("paper_audit", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
audit = module.Audit()
module.sutton_chen(audit)
assert not audit.drifts(), audit.drifts()
assert not any(name.split(".")[0] == "numpy" for name in sys.modules)
"""


def test_the_sutton_chen_check_runs_without_numpy():
    run = subprocess.run([sys.executable, "-c", NUMPY_BLOCK, str(ROOT / "scripts" / "paper_audit.py")],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-2000:]


FULL_AUDIT_WITHOUT_NUMPY = """
import importlib.abc, runpy, sys

BLOCKED = ("numpy", "scipy", "matplotlib")
attempts = []

class NoNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            attempts.append(name)
            raise ImportError(f"{name} is not installed")

for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        del sys.modules[name]
sys.meta_path.insert(0, NoNumpy())
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
status = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as stop:
    status = stop.code if isinstance(stop.code, int) else 1
loaded = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
print("blocked imports tried:", attempts, "loaded:", loaded)
sys.exit(status or (3 if attempts or loaded else 0))
"""


def full_audit_without_numpy(script, *args):
    return subprocess.run([sys.executable, "-c", FULL_AUDIT_WITHOUT_NUMPY, str(script), *args],
                          cwd=ROOT, capture_output=True, text=True, check=False)


def test_the_audit_runs_without_numpy():
    run = full_audit_without_numpy(ROOT / "scripts" / "paper_audit.py", "--fetch-records")
    assert run.returncode == 0, (run.stdout[-1500:], run.stderr[-1500:])
    assert "blocked imports tried: [] loaded: []" in run.stdout
    assert " 0 drifted" in run.stdout


def test_the_numpy_free_audit_run_fails_when_a_path_imports_numpy(tmp_path):
    caught = tmp_path / "caught.py"
    caught.write_text("import numpy\n")
    run = full_audit_without_numpy(caught)
    assert run.returncode != 0
    assert "ImportError" in run.stderr or "numpy" in run.stderr

    swallowed = tmp_path / "swallowed.py"
    swallowed.write_text("try:\n    import numpy\nexcept ImportError:\n    pass\n")
    run = full_audit_without_numpy(swallowed)
    assert run.returncode == 3
    assert "blocked imports tried: ['numpy']" in run.stdout

    lazy = tmp_path / "lazy.py"
    lazy.write_text("def helper():\n    import numpy\nhelper()\n")
    assert full_audit_without_numpy(lazy).returncode != 0


def test_the_sutton_chen_helpers_import_without_numpy():
    code = NUMPY_BLOCK.split("spec = ")[0] + (
        "sys.path.insert(0, sys.argv[1])\n"
        "import sc96_78_common, sc96_78_hits\n"
        "assert 'numpy' not in sys.modules\n")
    run = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts")],
                         cwd=ROOT, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-2000:]


IPI_FORM = "https://github.com/afiorent/i-pi.git@4e86f5fdc1583ef21f105fc2bed96a279b2430c1"


def test_a_side_branch_revision_is_not_an_ancestor():
    assert audit_mod.git("cat-file", "-t", "e3ed7dad").stdout.strip() == "commit"
    assert not audit_mod.ancestor_of_head("e3ed7dad")
    assert not audit_mod.harness_reachable("e3ed7dad")


def test_the_develop_revision_is_an_ancestor():
    assert audit_mod.ancestor_of_head("61fa67cf")
    assert audit_mod.harness_reachable("61fa67cf")


def test_a_qualified_external_revision_is_a_repo_url_and_a_full_sha():
    assert audit_mod.qualified_external_form(IPI_FORM) is True
    assert audit_mod.qualified_external_form("https://github.com/afiorent/i-pi.git@4e86f5fd") is False
    assert audit_mod.qualified_external_form("4e86f5fd") is None
    assert not audit_mod.harness_reachable("4e86f5fd")
    assert audit_mod.harness_reachable(IPI_FORM)
    assert audit_mod.external_pin("aac1548a", abbreviated=True)
    assert audit_mod.harness_reachable("aac1548a")


IPI_PIN = IPI_FORM.split("@")[1]


@pytest.mark.parametrize("form", [
    "https://example.com/fake/code.git@" + "0" * 40,
    "https://github.com/afiorent/i-pi.git@" + "0" * 40,
    "https://github.com/someone/i-pi.git@" + IPI_PIN,
])
def test_an_unpinned_external_revision_is_rejected(form):
    assert audit_mod.qualified_external_form(form) is False
    assert not audit_mod.harness_reachable(form)
    audit = audit_mod.Audit()
    path = "bench/pimc-lj38/README.md"
    audit.text_cache[path] = audit.text(path).replace(IPI_FORM, form)
    audit_mod.printed_harnesses(audit)
    assert ("qualified external revision", form) in {(r[2], r[3]) for r in audit.drifts()}


def test_a_41_digit_external_revision_is_rejected():
    long = IPI_FORM + "0"
    assert audit_mod.qualified_external_form(long) is False
    audit = audit_mod.Audit()
    path = "bench/pimc-lj38/README.md"
    audit.text_cache[path] = audit.text(path).replace(IPI_FORM, long)
    audit_mod.printed_harnesses(audit)
    assert ("qualified external revision", long) in {(r[2], r[3]) for r in audit.drifts()}
    assert not [r for r in audit.results if r[1].startswith(path) and r[2] == "qualified external revision" and r[0] == "OK"]


def clear_audit_caches():
    for value in vars(audit_mod).values():
        if callable(getattr(value, "cache_clear", None)):
            value.cache_clear()
    audit_mod._ANCESTOR_OF_HEAD.clear()


def isolated_readme(request, tmp_path, monkeypatch, path, text):
    """A root whose bench holds this one README, an empty record-ref registry and no unverifiable rows, with the
    scripts the revision checks read and git on this repository.

    The audit's caches are cleared on the way in and out, so no result crosses between this root and the real one.
    """
    clear_audit_caches()
    request.addfinalizer(clear_audit_caches)
    for name in (audit_mod.ALIASES, *audit_mod.EXTERNAL_FORM_SOURCES, *audit_mod.EXTERNAL_PIN_SOURCES):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, tmp_path / name)
    (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / path).write_text(text)
    (tmp_path / audit_mod.RECORD_REFS).write_text("ref,sha,tree\n")
    git_dir = subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
    monkeypatch.setenv("GIT_DIR", git_dir)
    monkeypatch.setattr(audit_mod, "ROOT", tmp_path)
    monkeypatch.setattr(audit_mod, "UNVERIFIABLE", [])


def test_an_inline_external_revision_is_stripped_from_prose(request, tmp_path, monkeypatch):
    line = f"Rows at `{IPI_FORM}` hit at a median of 12 calls."
    assert IPI_PIN not in audit_mod._EXTERNAL_FORM.sub(" ", line)
    path = "bench/pimc-lj38/README.md"
    text = (ROOT / path).read_text() + f"\n{line}\n"
    isolated_readme(request, tmp_path, monkeypatch, path, text)
    audit = audit_mod.Audit()
    audit_mod.printed_harnesses(audit)
    assert audit.drifts() == []
    assert ("OK", f"{path}:{len(text.splitlines())}", "qualified external revision", IPI_FORM) in {r[:4] for r in audit.results}


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
        ("bench/agox-lj75/README.md", "ccc22cee"),
        ("bench/ase-lj75/README.md", "162d2307"),
        ("bench/ase-lj75/README.md", "8173c5e2"),
        ("bench/ase-nonlj/README.md", "90b88235"),
        ("bench/covalent-h2h/README.md", "ce1e7a46"),
        ("bench/covalent-h2h/README.md", "eb7f8999"),
        ("bench/covalent-h2h/README.md", "a52726af"),
    } <= found
    reruns = {f"bench/ase-{s}/README.md" for s in ("lj38", "lj55", "lj98", "lj104", "lj150", "au55")}
    assert not (reruns & {path for path, _ in found})


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
    # The fetch lands on refs/audit/record/fetched, which is the branch tip,
    # not this parent. A local copy of the parent is not a successful fetch.
    assert record_drifts() == {
        f"record/repro-pin is on origin at {parent[:8]}",
        "record/repro-pin fetches",
    }
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


def test_record_refs_are_not_fetched_unless_asked(tmp_path, monkeypatch):
    calls = []
    real = audit_mod.git

    def wrapped(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(audit_mod, "git", wrapped)
    monkeypatch.setattr(audit_mod, "_FETCH_RECORDS", False)
    audit_mod._record_ref_status.cache_clear()
    audit_mod.origin_record_heads.cache_clear()
    registry(tmp_path, monkeypatch, [("record/repro-pin", REPRO_PIN, "0" * 40)])
    assert "record/repro-pin fetches" in record_drifts()
    assert "record/repro-pin is on origin at " + REPRO_PIN[:8] in record_drifts()
    assert not any(args and args[0] in {"fetch", "ls-remote"} for args in calls)


def test_a_record_ref_is_fetched_into_the_audit_namespace(tmp_path, monkeypatch):
    need_origin()
    seen = []
    real = audit_mod.git

    def wrapped(*args, **kwargs):
        seen.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(audit_mod, "git", wrapped)
    monkeypatch.setattr(audit_mod, "_FETCH_RECORDS", True)
    audit_mod._record_ref_status.cache_clear()
    tree = git("rev-parse", f"{REPRO_PIN}^{{tree}}").stdout.strip()
    registry(tmp_path, monkeypatch, [("record/repro-pin", REPRO_PIN, tree)])
    assert record_drifts() == set()
    fetches = [args for args in seen if args and args[0] == "fetch"]
    assert len(fetches) == 1
    assert fetches[0][-1] == "+refs/heads/record/repro-pin:refs/audit/record/fetched"
    assert fetches[0][-2] == "--"


def test_the_gmin_record_is_fetched_by_its_ref(tmp_path, monkeypatch):
    need_origin()
    seen = []
    real = audit_mod.git

    def wrapped(*args, **kwargs):
        seen.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(audit_mod, "git", wrapped)
    monkeypatch.setattr(audit_mod, "_FETCH_RECORDS", True)
    audit_mod._record_ref_status.cache_clear()
    audit_mod.origin_record_heads.cache_clear()
    audit_mod.origin_audit_record_refs.cache_clear()
    registry(tmp_path, monkeypatch, [(audit_mod.GMIN_RECORD_REF, audit_mod.GMIN_RECORD_SHA, audit_mod.GMIN_RECORD_TREE)])
    assert record_drifts() == set()
    fetches = [args for args in seen if args and args[0] == "fetch"]
    assert fetches == [("fetch", "--quiet", "--no-tags", "origin", "refs/audit/record/gmin-184")]
    registry_text = (ROOT / "bench/record-refs.csv").read_text()
    assert "\nrecord/gmin-184," not in registry_text


def test_a_record_ref_cannot_choose_its_fetch_destination(tmp_path, monkeypatch):
    calls = []
    real = audit_mod.git

    def wrapped(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(audit_mod, "git", wrapped)
    monkeypatch.setattr(audit_mod, "_FETCH_RECORDS", True)
    audit_mod._record_ref_status.cache_clear()
    for ref in (
        "record/repro-pin:refs/heads/main",
        "record/../main",
        "--upload-pack=touch",
        "record/repro-pin.lock",
    ):
        calls.clear()
        registry(tmp_path, monkeypatch, [(ref, REPRO_PIN, "0" * 40)])
        audit_mod._record_ref_status.cache_clear()
        assert f"{ref} is a record ref with a full SHA" in record_drifts()
        assert not any(args and args[0] == "fetch" for args in calls)


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


def test_the_gmin_panel_session_rows_keep_their_harness():
    rows = audit_mod.Audit().rows("bench/gmin-panel/rows.csv")
    session = [r for r in rows if r["harness_revision"] == audit_mod.GMIN_PANEL_SESSION]
    assert len(session) == 120
    assert sum(r["code"] == "gmin" for r in session) == 60
    registry = (ROOT / audit_mod.RECORD_REFS).read_text()
    assert f"{audit_mod.GMIN_RECORD_REF},{audit_mod.GMIN_RECORD_SHA},{audit_mod.GMIN_RECORD_TREE}" in registry
    assert "\nrecord/gmin-184," not in registry


def test_a_short_external_revision_is_reported():
    audit = audit_mod.Audit()
    path = "bench/pimc-lj38/README.md"
    text = audit.text(path)
    short = "https://github.com/afiorent/i-pi.git@4e86f5fd"
    audit.text_cache[path] = text.replace(IPI_FORM, short)
    audit_mod.printed_harnesses(audit)
    assert any(row[2] == "qualified external revision" and row[3] == short for row in audit.drifts())


def test_a_best_column_is_checked_with_its_revision():
    header = ["System", "Code revision", "Best", "Note"]
    rows = [(4, ["lj38", "e3ed7dad", "-1.50", "kept"])]
    flagged = audit_mod._table_revision_flags("bench/x/README.md", header, rows, [])
    assert flagged is not None
    assert any(item[1] == "e3ed7dad" and "Best=-1.50" in item[2] for item in flagged)
    reached = [(4, ["lj38", "61fa67cf", "-1.50", "kept"])]
    assert audit_mod._table_revision_flags("bench/x/README.md", header, reached, []) == []


def test_an_unreachable_printed_revision_is_reported():
    audit = audit_mod.Audit()
    path = "bench/metal-cluster-h2h/sc-n12/README.md"
    text = audit.text(path)
    audit.text_cache[path] = text.replace("| sc126-38 | anneal | 61fa67cf |", "| sc126-38 | anneal | e3ed7dad |")
    audit_mod.printed_harnesses(audit)
    assert any(row[4].startswith("e3ed7dad") for row in audit.drifts())


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


def test_a_drifted_ase_hit_sentence_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "and neither ASE arm hits any"
    assert old in audit_mod.collapse(tex)
    audit.text_cache[audit_mod.TEX] = replace_collapsed(tex, old, "and neither ASE arm hits all")
    audit_mod.manuscript_ase(audit)
    assert [r[2] for r in audit.drifts()] == ["prose"]


def test_a_wall_cap_span_in_the_following_sentence_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "and neither ASE arm hits any."
    assert old in audit_mod.collapse(tex)
    added = " Each ASE run there stopped at the cap after 3933 to 20950 calls."
    audit.text_cache[audit_mod.TEX] = replace_collapsed(tex, old, old + added)
    audit_mod.manuscript_ase(audit)
    assert [row for row in audit.drifts() if row[2] == "the following sentence names no wall-cap span"] == [
        ("DRIFT", audit_mod.TEX, "the following sentence names no wall-cap span", "absent", "present")]


def test_a_drifted_newcombe_interval_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "& 12/12 & 4/12 & 8 & 0 & \\(7.8\\times 10^{-3}\\) & 0.30 to 0.86 \\\\"
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("0.86", "0.87"))
    audit_mod.manuscript_knockouts(audit)
    assert [r[2] for r in audit.drifts()] == ["Newcombe upper"]


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
    si = audit.text(audit_mod.SI)
    old = audit_mod.SUBMITTED
    assert old in si
    audit.text_cache[audit_mod.SI] = si.replace(old, "1ce3ab64493196afd25e8690648e21157d91b882")
    audit_mod.presets_and_templates(audit)
    assert "tab:gmin revision 61fa67cf is named in the SI" in {r[2] for r in audit.drifts()}


def test_an_overclaim_that_returns_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed."
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old + " The misses sit in the same trap.")
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["no overclaim: same trap"]


def test_a_hypothesis_in_the_results_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed."
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old + " This supports the funnel hypothesis.")
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["the manuscript states no hypothesis"]


def test_reported_run_figures_exclude_unused_components_and_crossings():
    audit = audit_mod.Audit()
    stack = audit.text("paper/jctc/scripts/fig_algorithm_stack.py")
    assert audit_mod.stack_route_clears_temperature(stack)
    assert '("Lennard-Jones", "Morse", "Gupta", "binary LJ")' in stack
    assert '("TIP4P", "Sutton-Chen")' in stack
    assert "GFN2-xTB" not in stack
    arms = audit.text("paper/jctc/scripts/fig_rec_arms.py")
    assert '("single", "surface", "sym", "burst", "soap", "dls-kernel")' in arms
    crossing = stack.replace("(2.25, 0.74)", "(3.15, 0.74)").replace(
        "(2.25, 1.52)", "(3.15, 1.52)")
    assert not audit_mod.stack_route_clears_temperature(crossing)
    assert "bank" not in audit.text("paper/jctc/scripts/toc_graphic.py")


def test_unused_potential_and_wrong_arm_label_are_reported():
    audit = audit_mod.Audit()
    stack_path = "paper/jctc/scripts/fig_algorithm_stack.py"
    arms_path = "paper/jctc/scripts/fig_rec_arms.py"
    audit.text_cache[stack_path] = audit.text(stack_path).replace('"Gupta"', '"EMT"')
    audit.text_cache[arms_path] = audit.text(arms_path).replace('"dls-kernel"', '"lattice"')
    audit_mod.science_claims(audit)
    descriptions = {r[2] for r in audit.drifts()}
    assert "the rgpot figure row is derived from ClusterKernels" in descriptions
    assert "the stack figure excludes unused potential EMT" in descriptions
    assert "the arm figure uses the six ClusterMove names" in descriptions


def test_stack_potentials_follow_the_kernel_dispatch():
    audit = audit_mod.Audit()
    source = audit.text("src/potentials.rs")
    old = "    pub binary_lj: fn("
    assert old in source
    audit.text_cache["src/potentials.rs"] = source.replace(
        old, "    pub omitted_binary_lj: fn(", 1)
    audit_mod.science_claims(audit)
    assert "the rgpot figure row is derived from ClusterKernels" in {
        r[2] for r in audit.drifts()
    }


def test_default_screen_cannot_be_described_as_the_margin_path():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = ("For every other proposal, the preset uses the Bayesian screen of\n"
           "equation~\\sieqref{eq:bayes-screen}.")
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(
        old, "For every other proposal, the preset uses the margin screen.")
    audit_mod.science_claims(audit)
    assert any(r[2] == "prose" and r[3] == old for r in audit.drifts())


def test_icosahedral_stall_skip_cannot_return():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    marker = "The preset enables two responses to a stuck chain."
    assert marker in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(
        marker,
        ("A structure already dominated by an icosahedral signature skips "
         "that projection.\n" + marker),
    )
    audit_mod.science_claims(audit)
    assert any(
        r[2].startswith("stale mechanism claim is absent:")
        for r in audit.drifts()
    )


def test_fivefold_arm_amplitude_is_locked_to_the_code():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Their root-mean-square displacement is \\(0.75\\) in reduced units."
    assert old in tex
    audit.text_cache[audit_mod.TEX] = tex.replace(old, old.replace("0.75", "0.35"))
    audit_mod.science_claims(audit)
    assert any(r[2] == "prose" and r[3] == old for r in audit.drifts())


def test_a_historical_printed_row_with_the_trail_disabled_is_reported(monkeypatch):
    audit = audit_mod.Audit()
    real_git = audit_mod.git
    target = audit_mod.TRAIL_ROW_REVISIONS[0]

    def trail_disabled(*args, stdin=None):
        result = real_git(*args, stdin=stdin)
        if args == ("show", f"{target}:src/methods/cluster_hopping/config.rs"):
            return subprocess.CompletedProcess(
                result.args,
                result.returncode,
                result.stdout.replace("cfg.trail_on_stall = true;", "cfg.trail_on_stall = false;"),
                result.stderr,
            )
        return result

    monkeypatch.setattr(audit_mod, "git", trail_disabled)
    audit_mod.historical_trail_presets(audit)
    assert [(r[1], r[2]) for r in audit.drifts()] == [
        ("src/methods/cluster_hopping/config.rs at eeec8e2c",
         "printed-row preset enables the trail response")
    ]


def test_an_unresolved_citation_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.SI)
    old = "\\cite{fisherLogicInductiveInference1935}"
    assert old in tex
    audit.text_cache[audit_mod.SI] = tex.replace(old, "\\cite{fisherLogicInductiveInference1936}", 1)
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
    si = audit.text(audit_mod.SI)
    old = "For atomic clusters the preset has one atom-count gate:"
    assert old in audit_mod.collapse(si)
    audit.text_cache[audit_mod.SI] = replace_collapsed(
        si,
        old, "For atomic clusters the preset has two atom-count gates, both at 98 atoms:")
    audit_mod.count_gates(audit, "c2de652dca0bc75e2b0cb4afffac91fd656404e4")
    assert sorted(r[2] for r in audit.drifts()) == ["prose", "retired gate wording stays out: two atom-count gates"]


def test_a_withdrawn_benchmark_in_the_text_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex.replace("\\end{abstract}", "It hits 121 of 192 seeds.\n\\end{abstract}", 1)
    audit_mod.withdrawn_benchmarks(audit)
    assert [r[2] for r in audit.drifts()] == ["the water hexamer against CREST is not printed: 121 of 192"]


def test_a_stale_gmin_single_session_claim_is_reported():
    audit = audit_mod.Audit()
    audit.text_cache[audit_mod.COVER] = (
        audit.text(audit_mod.COVER)
        + "\nthirteen Lennard-Jones, Gupta, and Morse clusters in one session.\n"
    )
    audit.text_cache[audit_mod.SI] = (
        audit.text(audit_mod.SI) + "\nThe same-session panel of the main text.\n"
    )
    audit_mod.withdrawn_benchmarks(audit)
    assert {(r[1], r[2]) for r in audit.drifts()} == {
        (audit_mod.COVER, "no whole-table single-session claim"),
        (audit_mod.SI, "no same-session panel claim in SI"),
    }


def test_a_drifted_crest_holm_sentence_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    old = "Only the water octamer shares one stopping rule, and that adjusted value is not below 0.05."
    assert audit_mod.collapse(tex).count(old) == 1
    audit.text_cache[audit_mod.TEX] = replace_collapsed(tex, old, old.replace("not below 0.05", "below 0.05"))
    audit_mod.crest_table(audit)
    assert [r[2] for r in audit.drifts()] == ["prose"]


def test_a_banned_word_is_reported():
    audit = audit_mod.Audit()
    si = audit.text(audit_mod.SI)
    old = "landscape to propose structures"
    assert old in si
    audit.text_cache[audit_mod.SI] = si.replace(old, "landscape to propose candidates")
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["the text does not say candidate"]


def test_a_single_host_claim_is_reported():
    audit = audit_mod.Audit()
    tex = audit.text(audit_mod.TEX)
    audit.text_cache[audit_mod.TEX] = tex + "\nBoth arms ran on one host.\n"
    audit_mod.science_claims(audit)
    assert [r[2] for r in audit.drifts()] == ["no single-host claim: one host"]


def test_a_gmin_match_required_by_the_template_era_script_is_reported():
    audit = audit_mod.Audit()
    path = "paper/jctc/scripts/template_era_table.py"
    audit.text_cache[path] = audit.text(path) + "\n# GMIN differs\n"
    audit_mod.template_era(audit)
    assert "the template-era script does not require a GMIN match: GMIN differs" in [r[2] for r in audit.drifts()]


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


def _cap_turn_copies():
    audit = audit_mod.Audit()
    rows = [dict(row) for row in audit.rows(audit_mod.CAP_TURN_ROWS)]
    off = [dict(row) for row in audit.rows(audit_mod.CAP_TURN_FLAG_OFF)]
    return rows, off


def test_cap_turn_record_matches_its_rows():
    audit = audit_mod.Audit()
    audit_mod.cap_turn_campaign(audit)
    drift = audit.drifts()
    assert not drift, "\n".join(f"{label} shown={shown} computed={value}" for _, _, label, shown, value in drift)
    assert audit_mod.withdrawn_pooled_spelling(audit.text(audit_mod.CAP_TURN_README).split("## 2. The cap turn", 1)[1]) is None


def test_a_mcnemar_column_in_the_cap_turn_rows_is_reported():
    audit = audit_mod.Audit()
    rows, _off = _cap_turn_copies()
    for row in rows:
        row["McNemar_p"] = "2.10e-2"
    audit.read[audit_mod.CAP_TURN_ROWS] = rows
    audit_mod.cap_turn_campaign(audit)
    labels = {row[2] for row in audit.drifts()}
    assert "the cap-turn rows have no McNemar column" in labels
    assert "the cap-turn record states no McNemar p" in labels
    assert "3.2e-5" not in "".join(row["McNemar_p"] for row in rows)


def test_a_mcnemar_column_in_the_cap_turn_summary_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.CAP_TURN_README)
    extra = "\n| cell | MCNEMAR_P |\n| --- | --- |\n| lj38 | 1.000 |\n"
    assert "3.2e-5" not in extra
    audit.text_cache[audit_mod.CAP_TURN_README] = text + extra
    audit_mod.cap_turn_campaign(audit)
    labels = {row[2] for row in audit.drifts()}
    assert "the cap-turn summary has no McNemar column" in labels
    p = [row for row in audit.drifts() if row[2] == "the cap-turn record states no McNemar p"]
    assert p and p[0][4] == "1.000"
    assert not any("outside/call" in row[2] for row in audit.drifts())


def test_a_mcnemar_p_written_beside_its_name_is_reported():
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.CAP_TURN_README)
    audit.text_cache[audit_mod.CAP_TURN_README] = text + "\nThe McNemar p was 2.1e-2.\n"
    audit_mod.cap_turn_campaign(audit)
    labels = {row[2] for row in audit.drifts()}
    assert "the cap-turn record states no McNemar p" in labels
    assert "the cap-turn summary has no McNemar column" not in labels
    assert "the cap-turn rows have no McNemar column" not in labels


@pytest.mark.parametrize("spelling", ["0.000032", "3.20e-05", "3.2 \\times 10^{-5}", "3.2 × 10^{-5}"])
def test_a_reformatted_pooled_mcnemar_p_is_reported(spelling):
    audit = audit_mod.Audit()
    text = audit.text(audit_mod.CAP_TURN_README)
    assert "3.2e-5" not in text
    audit.text_cache[audit_mod.CAP_TURN_README] = text + f"\nWithdrawn pooled value {spelling}.\n"
    audit_mod.cap_turn_campaign(audit)
    drift = [row for row in audit.drifts() if row[2] == "the withdrawn pooled McNemar p is absent"]
    assert len(drift) == 1
    assert drift[0][4] == spelling
    assert "3.2e-5" not in drift[0][4]


def test_a_cap_turn_ratio_that_disagrees_with_the_rows_is_reported():
    audit = audit_mod.Audit()
    rows, _off = _cap_turn_copies()
    for row in rows:
        if row["cell"] == "lj38" and row["arm"] == "base":
            row["wall_s"] = str(float(row["pot_s"]) + 10.0 * float(row["charged"]))
    audit.read[audit_mod.CAP_TURN_ROWS] = rows
    audit_mod.cap_turn_campaign(audit)
    drift = [row for row in audit.drifts() if row[2] == "lj38 outside/call ratio matches the rows"]
    assert len(drift) == 1
    assert drift[0][3] == "0.998"
    assert drift[0][4] != "0.998"


def test_cap_turn_ratio_rounds_once_from_full_precision():
    assert audit_mod.cap_turn_ratio_text(0.992484) == "0.992"
    assert audit_mod.cap_turn_ratio_text(1.002459) == "1.002"
    assert audit_mod.cap_turn_ratio_text(0.993487) == "0.993"
    assert audit_mod.cap_turn_ratio_text(0.9925) == "0.993"

    audit = audit_mod.Audit()
    text = audit.text(audit_mod.CAP_TURN_README)
    audit.text_cache[audit_mod.CAP_TURN_README] = text.replace(
        "| m55r14 | 18/24 | 18/24 | 1 | 5 | 5 | 0.992 |",
        "| m55r14 | 18/24 | 18/24 | 1 | 5 | 5 | 0.993 |",
        1,
    )
    audit_mod.cap_turn_campaign(audit)
    drift = [row for row in audit.drifts() if row[2] == "m55r14 outside/call ratio matches the rows"]
    assert len(drift) == 1
    assert drift[0][3] == "0.993"
    assert drift[0][4] == "0.992"


def _flag_off_drift(mutate):
    audit = audit_mod.Audit()
    rows, off = _cap_turn_copies()
    mutate(rows, off)
    audit.read[audit_mod.CAP_TURN_ROWS] = rows
    audit.read[audit_mod.CAP_TURN_FLAG_OFF] = off
    audit_mod.cap_turn_campaign(audit)
    drift = [row for row in audit.drifts() if row[2] == "flag-off rows match the base rows"]
    assert len(drift) == 1
    return drift[0][4]


def test_a_flag_off_hit_that_differs_from_base_is_reported():
    def mutate(_rows, off):
        off[0]["hit"] = "false" if off[0]["hit"].strip().lower() == "true" else "true"

    assert "hit" in _flag_off_drift(mutate)


def test_a_flag_off_charged_count_that_differs_from_base_is_reported():
    def mutate(_rows, off):
        off[0]["charged"] = str(int(off[0]["charged"]) + 1)

    assert "charged" in _flag_off_drift(mutate)


def test_a_flag_off_energy_that_differs_from_base_is_reported():
    def mutate(_rows, off):
        best = off[0]["best"]
        off[0]["best"] = best[:-1] + ("0" if best[-1] != "0" else "1")

    assert "energy" in _flag_off_drift(mutate)


def test_a_flag_off_hash_that_differs_from_base_is_reported():
    def mutate(rows, off):
        base = next(row for row in rows if row["arm"] == "base" and row["cell"] == off[0]["cell"]
                    and row["seed"] == off[0]["seed"])
        bits = base["best_bits"]
        off[0]["best_bits"] = bits[:-1] + ("0" if bits[-1] != "0" else "1")

    assert "hash" in _flag_off_drift(mutate)


def test_an_append_only_check_still_names_a_rewritten_line_and_spares_only_an_upstream_removal():
    old = ["a\n", "b\n", "c\n"]
    assert audit_mod._first_rewritten_line(old, [*old, "d\n"], set()) is None
    assert audit_mod._first_rewritten_line(old, ["a\n", "B\n", "c\n"], set()) == 2
    assert audit_mod._first_rewritten_line(old, ["a\n", "c\n"], set()) == 2
    assert audit_mod._first_rewritten_line(old, ["a\n", "c\n"], {"b\n"}) is None
    assert audit_mod._first_rewritten_line(old, ["a\n", "c\n"], {"c\n"}) == 2
    assert audit_mod._first_rewritten_line(old, ["x\n", "b\n", "c\n"], {"a\n"}) == 2


def test_only_the_lines_a_base_branch_deleted_after_the_fork_are_spared(monkeypatch):
    import types

    shows = {"fork:rows.csv": "a\nb\nc\n", "base:rows.csv": "a\nc\nz\n"}

    def fake(*args, **kwargs):
        if args[0] == "merge-base":
            return types.SimpleNamespace(returncode=0, stdout="fork\n")
        if args[0] == "show":
            key = args[1]
            return types.SimpleNamespace(returncode=0 if key in shows else 1, stdout=shows.get(key, ""))
        raise AssertionError(args)

    monkeypatch.setattr(audit_mod, "git", fake)
    assert audit_mod._upstream_removed_lines("ref", "base", "rows.csv") == {"b\n"}
    assert audit_mod._upstream_removed_lines("base", "base", "rows.csv") == set()
    assert audit_mod._upstream_removed_lines("ref", None, "rows.csv") == set()


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


def test_cross_document_equation_references_name_their_document():
    audit = audit_mod.Audit()
    audit_mod.cross_document_equations(audit)
    assert not audit.drifts(), "\n".join(
        f"{where}  shown={shown}  computed={value}" for _, where, _, shown, value in audit.drifts())


def test_stripping_the_si_equation_prefix_is_reported():
    audit = audit_mod.Audit()
    macros = audit.text("paper/jctc/macros.tex")
    assert "\\@sieqref@strip" not in macros
    audit.text_cache["paper/jctc/macros.tex"] = macros + "\n\\def\\@sieqref@strip S#1\\@nil{#1}\n"
    audit_mod.cross_document_equations(audit)
    drifts = [where for _, where, label, _, _ in audit.drifts()
              if label == "an SI equation reference keeps its S or names the SI"]
    assert drifts
    assert all(where.startswith("paper/jctc/manuscript.tex:") for where in drifts)


def test_an_si_qualifier_covers_a_stripped_equation_reference():
    audit = audit_mod.Audit()
    macros = audit.text("paper/jctc/macros.tex")
    tex = audit.text("paper/jctc/manuscript.tex")
    audit.text_cache["paper/jctc/macros.tex"] = macros + "\n\\def\\@sieqref@strip S#1\\@nil{#1}\n"
    audit.text_cache["paper/jctc/manuscript.tex"] = tex.replace(
        "equation~\\sieqref{", "SI equation~\\sieqref{").replace(
        "Equation~\\sieqref{", "SI equation~\\sieqref{")
    audit_mod.cross_document_equations(audit)
    assert not [where for _, where, label, _, _ in audit.drifts()
                if label == "an SI equation reference keeps its S or names the SI"]


def test_a_main_text_equation_without_its_name_is_reported():
    audit = audit_mod.Audit()
    si = audit.text("paper/jctc/si.tex")
    assert "of the main text" in si
    audit.text_cache["paper/jctc/si.tex"] = si.replace(" of the main text", "")
    audit_mod.cross_document_equations(audit)
    drifts = [where for _, where, label, _, _ in audit.drifts()
              if label == "a manuscript equation reference names the main text"]
    assert drifts
    assert all(where.startswith("paper/jctc/si.tex:") for where in drifts)
