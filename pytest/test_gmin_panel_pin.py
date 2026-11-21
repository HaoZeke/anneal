# Copyright (c) anneal developers
"""Every GMIN build and row uses one verified source pin."""

from __future__ import annotations

import hashlib
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "gmin_panel" / "build_gmin.sh"
PANEL = ROOT / "scripts" / "gmin_panel" / "run_panel.py"
SOURCE = ROOT / "bench" / "gmin-panel" / "gmin-source.txt"
sys.path.insert(0, str(ROOT / "scripts"))
from gmin_pin import REVISION, SNAPSHOT, TARBALL_SHA256, URL, read_svnrev
from morse_gmin.run_panel import gmin_line


def _assign(text: str, name: str) -> str:
    match = re.search(rf"^{name}=(\S+)$", text, re.M)
    if match is None:
        raise AssertionError(f"{name} is not set")
    return match.group(1)


def test_every_gmin_builder_loads_the_shared_pin() -> None:
    builds = tuple((ROOT / "scripts").glob("**/build_gmin*.sh"))
    assert len(builds) == 6
    for build in builds:
        text = build.read_text()
        assert "gmin_pin.py" in text, build
        assert (
            "verify-tarball" in text
            or "build_gmin.sh" in text
            or "build_gmin_darby.sh" in text
        ), build


def test_fake_built_svnrev_flows_into_code_revision(tmp_path: Path) -> None:
    svnrev = tmp_path / "SVNREV"
    svnrev.write_text(f"{REVISION}\n")
    revision = read_svnrev(svnrev)
    row = gmin_line(
        "h2h_row best=-1 wall_s=1 pot_s=0.5 charged=10 evals_to_hit=10",
        "m30r6",
        2000,
        400_000,
        -1.0,
        revision,
    )
    assert row["code_revision"] == REVISION


def test_the_committed_gmin_source_matches_the_build_pin() -> None:
    script = BUILD.read_text()
    source = SOURCE.read_text()
    assert _assign(source, "revision") == REVISION
    assert _assign(source, "svnrev") == REVISION
    assert _assign(source, "tarball_sha256") == TARBALL_SHA256
    assert _assign(source, "snapshot") == SNAPSHOT
    hf_url = _assign(script, "HF_URL").replace("${HF_REVISION}", _assign(script, "HF_REVISION"))
    assert _assign(source, "url") == hf_url
    assert _assign(source, "fallback_url") == URL
    assert _assign(source, "source_date_epoch") == _assign(script, "SOURCE_DATE_EPOCH")
    assert "www-wales.ch.cam.ac.uk" not in _assign(source, "url")


def _gmin_line(sums: str) -> str:
    hashes = [line.split()[0] for line in sums.splitlines() if line.split()[1:] == ["GMIN"]]
    assert len(hashes) == 1
    return hashes[0]


def test_the_committed_binary_hash_matches_bin_sha256() -> None:
    source = SOURCE.read_text()
    sums = (ROOT / "bench" / "gmin-panel" / "bin-sha256.txt").read_text()
    if re.search(r"^binary_sha256=", source, re.M):
        assert _assign(source, "binary_sha256") == _gmin_line(sums)


def test_the_historical_receipt_pairs_with_bin_sha256() -> None:
    from gmin_pin import HISTORICAL_REVISION

    receipt = (ROOT / "bench" / "gmin-panel" / "gmin-source-aac1548a.txt").read_text()
    sums = (ROOT / "bench" / "gmin-panel" / "bin-sha256.txt").read_text()
    assert _assign(receipt, "svnrev") == HISTORICAL_REVISION
    assert _assign(receipt, "revision") == HISTORICAL_REVISION
    assert _assign(receipt, "binary_sha256") == _gmin_line(sums)


def test_the_build_records_the_compiler_it_runs() -> None:
    script = BUILD.read_text()
    assert "exec /usr/bin/gfortran" in script
    assert 'echo "gfortran=$(/usr/bin/gfortran -dumpfullversion)"' in script


def test_binary_sha256_comes_from_the_bin_sha256_pass(tmp_path: Path) -> None:
    assert "binary_sha256" not in BUILD.read_text()
    enter = (ROOT / "scripts" / "gmin_panel" / "enter_bench.sh").read_text()
    step = re.search(
        r'^GMIN_PANEL_WORK=\$WORK bash "\$HERE/build_gmin.sh"\n(.*?^cp -f "\$WORK/gmin-source.txt" "\$OUT/gmin-source.txt"\n)',
        enter,
        re.M | re.S,
    )
    assert step is not None
    work, out = tmp_path / "work", tmp_path / "out"
    (work / "bin").mkdir(parents=True)
    out.mkdir()
    (work / "bin" / "GMIN").write_bytes(b"gmin build\n")
    (work / "bin" / "h2h_morse").write_bytes(b"anneal build\n")
    (work / "gmin-source.txt").write_text("revision=r\n")
    env = dict(os.environ, WORK=str(work), OUT=str(out))
    subprocess.run(["bash", "-euo", "pipefail", "-c", step.group(1)], env=env, check=True)
    source = (out / "gmin-source.txt").read_text()
    sums = (out / "bin-sha256.txt").read_text()
    assert source.count("binary_sha256=") == 1
    assert _assign(source, "binary_sha256") == _gmin_line(sums)
    assert _gmin_line(sums) == hashlib.sha256(b"gmin build\n").hexdigest()


def test_the_build_normalizes_paths_and_stamps() -> None:
    script = BUILD.read_text()
    assert 'WORK=$(cd "$WORK" && pwd -P)' in script
    assert "SRC=$WORK/gmin-src" in script
    assert "TARBALL=$WORK/svn.tar.bz2" in script
    assert '-ffile-prefix-map=$SRC=/usr/src/gmin' in script
    assert '-fdebug-prefix-map=$SRC=/usr/src/gmin' in script
    assert '-ffile-prefix-map=$WORK=/build/gmin' in script
    assert '-fdebug-prefix-map=$WORK=/build/gmin' in script
    assert 'realpath --relative-to="$PWD"' in script
    assert "SOURCE_DATE_EPOCH=1795057278" in script
    assert "LC_ALL=C exec /usr/bin/date" in script
    assert "export LC_ALL=C" in script
    assert "export USER=builder" in script
    assert "export HOSTNAME=gmin-panel" in script
    assert "-o) echo linux" in script
    assert "-m) echo x86_64" in script


def test_the_build_uses_file_identity_and_cleans_partial_downloads() -> None:
    script = BUILD.read_text()
    assert 'test "$LOCAL" -ef "$TARBALL"' in script
    assert "trap cleanup_partial EXIT" in script


def _mock_fetch(tmp_path: Path, fail_hf: bool = False) -> subprocess.CompletedProcess[str]:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    curl = mock_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$MOCK_CURL_LOG"
if [[ ${MOCK_HF_FAIL:-0} == 1 && $* == *huggingface.co* ]]; then
  exit 22
fi
while [[ $# -gt 0 ]]; do
  if [[ $1 == -o ]]; then
    printf 'wrong archive\n' > "$2"
    exit 0
  fi
  shift
done
exit 2
"""
    )
    curl.chmod(0o755)
    work = tmp_path / "work"
    log = tmp_path / "curl.log"
    env = os.environ.copy()
    env["PATH"] = f"{mock_bin}:{env['PATH']}"
    env["MOCK_CURL_LOG"] = str(log)
    env["MOCK_HF_FAIL"] = "1" if fail_hf else "0"
    env["GMIN_PANEL_WORK"] = str(work)
    proc = subprocess.run(
        ["bash", str(BUILD)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    proc.curl_log = log.read_text()  # type: ignore[attr-defined]
    assert not list(work.glob("svn.tar.bz2.partial.*"))
    return proc


def test_fetch_uses_the_pinned_dataset_and_refuses_its_wrong_hash(tmp_path: Path) -> None:
    proc = _mock_fetch(tmp_path)
    assert proc.returncode == 1
    assert (
        "https://huggingface.co/datasets/HaoZeke/anneal-paper-data/resolve/"
        "2d639566b6e9a7243bfd485aa3cb9a904c915451/gmin/svn-20261119.tar.bz2"
        in proc.curl_log  # type: ignore[attr-defined]
    )
    assert "www-wales.ch.cam.ac.uk" not in proc.curl_log  # type: ignore[attr-defined]
    assert "tarball sha256 mismatch:" in proc.stderr


def test_fetch_falls_back_to_wales_after_dataset_failure(tmp_path: Path) -> None:
    proc = _mock_fetch(tmp_path, fail_hf=True)
    assert proc.returncode == 1
    assert "pinned dataset download failed" in proc.stderr
    assert "www-wales.ch.cam.ac.uk/svn.tar.bz2" in proc.curl_log  # type: ignore[attr-defined]
    assert "tarball sha256 mismatch:" in proc.stderr


def _load_panel(name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, PANEL)
    assert spec is not None and spec.loader is not None
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    return panel


def _documented_summarize_args() -> dict[str, list[str]]:
    readme = (ROOT / "bench" / "gmin-panel" / "README.md").read_text()
    tex = (ROOT / "paper" / "jctc" / "manuscript.tex").read_text()
    found = {
        "README": re.search(r"`bash scripts/gmin_panel/enter_bench\.sh (--summarize[^`]*)`", readme),
        "manuscript": re.search(r"^bash scripts/gmin_panel/enter_bench\.sh (--summarize.*)$", tex, re.M),
    }
    commands = {}
    for where, match in found.items():
        assert match is not None, f"no summarize command in the {where}"
        commands[where] = match.group(1).split()
    return commands


@pytest.mark.parametrize("where", ["README", "manuscript"])
@pytest.mark.parametrize("anneal_revision", ["enter_bench", "latest"])
def test_documented_summarize_commands_select_the_committed_rows(
    tmp_path: Path, where: str, anneal_revision: str
) -> None:
    panel = _load_panel(f"gmin_panel_summarize_{where}_{anneal_revision}")
    rows_path = ROOT / "bench" / "gmin-panel" / "rows.csv"
    rows = panel.read_rows(rows_path)
    assert rows
    for relative in ("bench/gmin-panel/README.md", "paper/jctc/manuscript.tex"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text((ROOT / relative).read_text())
    if anneal_revision == "enter_bench":
        anneal_revision = re.search(
            r"^REV=\$\{GMIN_PANEL_REV:-(\w+)\}$", (ROOT / "scripts/gmin_panel/enter_bench.sh").read_text(), re.M
        ).group(1)
    summary = tmp_path / "summary.csv"
    argv = [str(PANEL), "--root", str(tmp_path), "--rows", str(rows_path), "--revision", anneal_revision,
            "--summary", str(summary), *_documented_summarize_args()[where]]
    assert panel.main(argv) == 0
    written = panel.read_rows(summary)
    assert len(written) > 0
    assert {r["gmin_revision"] for r in written} == {r["code_revision"] for r in rows if r["code"] == "gmin"}


SUMMARIZE_FILES = (
    "scripts/gmin_panel/enter_bench.sh",
    "scripts/gmin_panel/run_panel.py",
    "scripts/target_tol.py",
    "scripts/gmin_pin.py",
    "bench/gmin-panel/rows.csv",
)
SUMMARIZED_FILES = (
    "bench/gmin-panel/summary.csv",
    "bench/gmin-panel/README.md",
    "paper/jctc/manuscript.tex",
)


def _documented_summarize_commands() -> dict[str, str]:
    command = r"bash scripts/gmin_panel/enter_bench\.sh --summarize[^`\n]*"
    sources = {
        "README": (ROOT / "bench/gmin-panel/README.md", rf"`({command})`"),
        "manuscript": (ROOT / "paper/jctc/manuscript.tex", rf"^({command})$"),
        "Snakefile": (ROOT / "workflow/Snakefile", rf"^\s*({command})$"),
    }
    found = {}
    for where, (path, pattern) in sources.items():
        matches = re.findall(pattern, path.read_text(), re.M)
        assert len(matches) == 1, f"{where}: {matches}"
        found[where] = " ".join(matches[0].split())
    return found


@pytest.mark.parametrize("where", ["README", "manuscript", "Snakefile"])
def test_documented_summarize_command_reproduces_the_committed_tables(tmp_path: Path, where: str) -> None:
    panel = _load_panel(f"gmin_panel_documented_{where}")
    for relative in SUMMARIZE_FILES + SUMMARIZED_FILES:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes((ROOT / relative).read_bytes())
    (tmp_path / "bench/gmin-panel/summary.csv").write_bytes(b"")
    for relative, (start, end) in (("bench/gmin-panel/README.md", panel.README_MARKS),
                                   ("paper/jctc/manuscript.tex", panel.TEX_MARKS)):
        text = (tmp_path / relative).read_text()
        head, rest = text.split(start, 1)
        (tmp_path / relative).write_text(f"{head}{start}\n{end}{rest.split(end, 1)[1]}")
        assert (tmp_path / relative).read_bytes() != (ROOT / relative).read_bytes()
    env = {key: value for key, value in os.environ.items() if not key.startswith("GMIN_PANEL_")}
    subprocess.run(["bash", "-c", _documented_summarize_commands()[where]], cwd=tmp_path, env=env, check=True)
    for relative in SUMMARIZED_FILES:
        assert (tmp_path / relative).read_bytes() == (ROOT / relative).read_bytes(), relative


def test_summarize_refuses_to_wipe_the_tables(tmp_path: Path) -> None:
    panel = _load_panel("gmin_panel_summarize_empty")
    rows = [r for r in panel.read_rows(ROOT / "bench" / "gmin-panel" / "rows.csv") if r["code"] == "gmin"]
    assert rows
    with pytest.raises(SystemExit, match="no system has both arms"):
        panel.write_tables(tmp_path, rows, "latest", True, tmp_path / "summary.csv")
    assert not (tmp_path / "summary.csv").exists()


def test_the_gmin_caption_follows_the_row_labels() -> None:
    panel = _load_panel("gmin_panel_caption")
    rows = panel.read_rows(ROOT / "bench" / "gmin-panel" / "rows.csv")
    tex = panel.tex_table(panel.summarise(rows, "latest"), "latest")
    labels = {r["code_revision"] for r in rows if r["code"] == "gmin"}
    shown = " ".join(sorted(label[:8] for label in labels))
    assert f"GMIN \\texttt{{{shown}}}." in tex
    assert tex in (ROOT / "paper" / "jctc" / "manuscript.tex").read_text()


def _load_losses2() -> object:
    spec = importlib.util.spec_from_file_location("gmin_losses2_test", PANEL.parent / "run_losses2.py")
    assert spec is not None and spec.loader is not None
    losses2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses2)
    return losses2


def test_the_known_gmin_revisions_are_the_rows_and_the_pin() -> None:
    from gmin_pin import KNOWN_REVISIONS

    assert KNOWN_REVISIONS == ("aac1548a8aec13b53ad12a8d6904625081685d2e", REVISION)
    assert REVISION == "9cbec1a446352bc9ef064bd2b1a5767194655919"


@pytest.mark.parametrize(
    "svnrev", ["aac1548a8aec13b53ad12a8d6904625081685d2e", "9cbec1a446352bc9ef064bd2b1a5767194655919"]
)
def test_losses2_selects_rows_at_each_known_gmin_revision(svnrev: str) -> None:
    losses2 = _load_losses2()
    rows = [
        {"system": "lj38", "code": "gmin", "seed": str(seed), "code_revision": svnrev, "harness_revision": "head"}
        for seed in losses2.panel.SEEDS
    ]
    assert len(losses2.selected(rows, "lj38", "gmin", "head", "base")) == len(losses2.panel.SEEDS)
    unknown = [dict(row, code_revision="0" * 40) for row in rows]
    assert losses2.selected(unknown, "lj38", "gmin", "head", "base") == []


def test_losses2_runs_the_pinned_build_and_refuses_another(tmp_path: Path) -> None:
    losses2 = _load_losses2()
    svnrev = tmp_path / "SVNREV"
    note = tmp_path / "gmin-source.txt"
    svnrev.write_text(f"{REVISION}\n")
    note.write_text(f"svnrev={REVISION}\ntarball_sha256={TARBALL_SHA256}\n")
    assert losses2.gmin_label(note, svnrev) == REVISION
    historical = "aac1548a8aec13b53ad12a8d6904625081685d2e"
    svnrev.write_text(f"{historical}\n")
    note.write_text(f"svnrev={historical}\ntarball_sha256={TARBALL_SHA256}\n")
    with pytest.raises(ValueError, match="expected"):
        losses2.gmin_label(note, svnrev)


@pytest.mark.parametrize("enter, work", [("enter_bench.sh", "$WORK"), ("enter_losses2.sh", "$OUT")])
def test_the_runners_read_the_svnrev_the_build_extracts(enter: str, work: str) -> None:
    src = _assign(BUILD.read_text(), "SRC").replace("$WORK", work)
    text = (PANEL.parent / enter).read_text()
    assert f"--svnrev {src}/wales/GMIN/source/SVNREV" in text


@pytest.mark.parametrize("how", ["env", "flag"])
def test_a_local_tarball_with_the_wrong_sha256_is_refused(tmp_path: Path, how: str) -> None:
    bad = tmp_path / "svn.tar.bz2"
    bad.write_bytes(b"not the pinned snapshot\n")
    work = tmp_path / "work"
    env = os.environ.copy()
    env["GMIN_PANEL_WORK"] = str(work)
    env.pop("GMIN_TARBALL", None)
    command = ["bash", str(BUILD)]
    if how == "env":
        env["GMIN_TARBALL"] = str(bad)
    else:
        command += ["--tarball", str(bad)]
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    assert "tarball sha256 mismatch:" in proc.stderr
    assert TARBALL_SHA256 in proc.stderr
    assert not work.exists()
