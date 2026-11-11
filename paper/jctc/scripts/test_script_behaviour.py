#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Behaviour tests for the paper scripts that do not need a campaign run."""
from __future__ import annotations

import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import check_cuh2_blobs
import coverage_receipts
import extract_water_logs
import figure_provenance
import hexamer_numbers
import hq_sci_tally
import lockfile_delta
import replay_v127
import toc_graphic
import watch_crest_steps
import write_figure_provenance_v128


def test_toc_graphic_writes_a_pdf(tmp_path: Path) -> None:
    body = toc_graphic.stream()
    assert b"potential" in body and b"hop" in body and b"bank" in body
    raw = toc_graphic.pdf(body)
    assert raw.startswith(b"%PDF-1.4\n")
    assert b"%%EOF\n" in raw
    dest = tmp_path / "slot" / "toc.pdf"
    toc_graphic.main([str(dest)])
    assert dest.read_bytes() == raw


def test_extract_water_logs_round_trip(tmp_path: Path) -> None:
    payload = b"seed line\n"
    archive = tmp_path / "logs.tar"
    with tarfile.open(archive, "w") as packed:
        info = tarfile.TarInfo("dir")
        info.type = tarfile.DIRTYPE
        packed.addfile(info)
        info = tarfile.TarInfo("nested/a.txt")
        info.size = len(payload)
        packed.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(payload).hexdigest()
    table = tmp_path / "map.txt"
    table.write_text(
        "member original_sha256 new_sha256\n"
        f"nested/a.txt abc {digest}\n"
    )
    dest = tmp_path / "out"
    old = sys.argv
    sys.argv = ["extract", str(archive), str(table), str(dest)]
    try:
        extract_water_logs.main()
    finally:
        sys.argv = old
    assert (dest / "nested" / "a.txt").read_bytes() == payload


def test_extract_water_logs_rejects_a_bad_call(tmp_path: Path) -> None:
    old = sys.argv
    sys.argv = ["extract"]
    try:
        with pytest.raises(SystemExit, match="usage"):
            extract_water_logs.main()
    finally:
        sys.argv = old
    missing = tmp_path / "nope.tar"
    sys.argv = ["extract", str(missing), "map", "dest"]
    try:
        with pytest.raises(SystemExit, match="missing archive"):
            extract_water_logs.main()
    finally:
        sys.argv = old
    archive = tmp_path / "one.tar"
    with tarfile.open(archive, "w") as packed:
        info = tarfile.TarInfo("a.txt")
        info.size = 1
        packed.addfile(info, io.BytesIO(b"x"))
    table = tmp_path / "map.txt"
    table.write_text("a.txt old deadbeef\n")
    sys.argv = ["extract", str(archive), str(table), str(tmp_path / "out")]
    try:
        with pytest.raises(SystemExit, match="sha mismatch"):
            extract_water_logs.main()
    finally:
        sys.argv = old


def test_lockfile_blocks_and_a_real_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert lockfile_delta.minor("1.2.3+git") == ("1", "2")
    assert lockfile_delta.minor("9") == ("9", "0")
    pairs, names = lockfile_delta.blocks(
        "[[package]]\nname = \"solo\"\nversion = \"1.0.0\"\n"
    )
    assert pairs == [("solo", "1.0.0")]
    assert "solo" in names
    old = textwrap.dedent(
        """\
        [[package]]
        name = "gone"
        version = "1.0.0"

        [[package]]
        name = "moved"
        version = "1.0.0"

        [[package]]
        name = "partial"
        version = "1.0.0"

        [[package]]
        name = "partial"
        version = "1.2.0"

        [[package]]
        name = "stay"
        version = "1.0.0"
        """
    )
    new = textwrap.dedent(
        """\
        [[package]]
        name = "moved"
        version = "2.0.0"

        [[package]]
        name = "partial"
        version = "1.2.0"

        [[package]]
        name = "stay"
        version = "1.0.0"

        [[package]]
        name = "added"
        version = "3.1.0"
        """
    )
    _git_repo(tmp_path, old, new)
    monkeypatch.chdir(tmp_path)
    old_rev = subprocess.check_output(["git", "rev-parse", "HEAD~1"], text=True).strip()
    new_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    old_argv = sys.argv
    sys.argv = ["lock", old_rev, new_rev]
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        lockfile_delta.main()
    finally:
        sys.argv = old_argv
    text = buf.getvalue()
    assert "packages_leave 1" in text
    assert "leave gone 1.0.0" in text
    assert "minor moved 1.0.0 2.0.0" in text
    assert "drop partial 1.0.0 stays 1.2.0" in text
    assert "diff_check ok" in text
    sys.argv = ["lock"]
    try:
        with pytest.raises(SystemExit, match="usage"):
            lockfile_delta.main()
    finally:
        sys.argv = old_argv


def test_figure_records_mark_absence_and_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    tip = figure_provenance.blob(pdf)
    assert len(tip) == 40

    def fake_git(*args: str) -> subprocess.CompletedProcess[str]:
        if args[0] == "hash-object":
            return subprocess.CompletedProcess(args, 0, tip + "\n", "")
        if args[:2] == ("rev-parse", "--verify"):
            name = args[2].rsplit("/", 1)[-1]
            if name == "one.pdf":
                return subprocess.CompletedProcess(args, 0, tip + "\n", "")
            return subprocess.CompletedProcess(args, 1, "", "missing")
        return subprocess.CompletedProcess(args, 1, "", "no")

    monkeypatch.setattr(figure_provenance, "git", fake_git)
    text = figure_provenance.records(tmp_path, "abcdef")
    assert "one.pdf" in text and "equal True" in text
    other = tmp_path / "two.png"
    other.write_bytes(b"png")
    text = figure_provenance.records(tmp_path, "abcdef")
    assert "two.png" in text and "main absent" in text
    monkeypatch.setattr(
        figure_provenance,
        "git",
        lambda *args: subprocess.CompletedProcess(args, 1, "", "boom"),
    )
    with pytest.raises(SystemExit, match="boom"):
        figure_provenance.blob(pdf)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(figure_provenance, "git", fake_git)
    assert figure_provenance.main([str(tmp_path), "abcdef"]) == 0
    assert "one.pdf" in buf.getvalue()


def test_tally_covers_hits_misses_and_missing(tmp_path: Path) -> None:
    hexamer = tmp_path / "hex"
    copper = tmp_path / "cu"
    hexamer.mkdir()
    copper.mkdir()
    for index in range(8):
        token = "inf" if index == 3 else f"-{1 + index * 0.001:.3f}"
        (hexamer / f"h2o6_rec_{index}.txt").write_text(f"seed {index}: best {token} eV\n")
        (hexamer / f"h2o6_rec_{index}.out").write_text(
            f"seed {index}: best {token} eV\n"
        )
    for index in range(8):
        if index == 7:
            token = "inf"
        elif index == 6:
            token = "-415.000000"
        else:
            token = "-415.971500"
        (copper / f"cuh2_fcc_rec_{index}.out").write_text(
            f"seed {index}: best {token} eV\n"
        )
    text = hq_sci_tally.tally(hexamer, copper)
    assert "hexamer hq_sci_h2o6" in text
    assert "hit=False" in text
    assert "gap_to_reference 0" in text
    assert "set_relative_0.01" in text
    assert hq_sci_tally.main([str(hexamer), str(copper)]) == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    for index in range(8):
        (empty / f"h2o6_rec_{index}.out").write_text(f"seed {index}: best inf eV\n")
    with pytest.raises(SystemExit, match="no finite best"):
        hq_sci_tally.block(empty, "h2o6_rec", "hex")
    (tmp_path / "note.txt").write_text("nothing\n")
    with pytest.raises(SystemExit, match="no best line"):
        hq_sci_tally.best_token(tmp_path / "note.txt")
    with pytest.raises(SystemExit, match="missing"):
        hq_sci_tally.seed_file(empty, "absent", 0)


def test_coverage_receipt_writes_both_toolchains(tmp_path: Path) -> None:
    lines = {
        "data": [
            {
                "files": [
                    {
                        "filename": "src/methods/rigid_quench.rs",
                        "summary": {
                            "totals": {
                                "lines": {"covered": 10, "count": 10},
                                "regions": {"covered": 8, "count": 8},
                                "branches": {"covered": 0, "count": 0},
                            }
                        },
                    }
                ]
            }
        ]
    }
    branches = json.loads(json.dumps(lines))
    branches["data"][0]["files"][0]["summary"]["totals"]["lines"] = {
        "covered": 9,
        "count": 10,
    }
    branches["data"][0]["files"][0]["summary"]["totals"]["branches"] = {
        "covered": 3,
        "count": 4,
    }
    line_path = tmp_path / "lines.json"
    branch_path = tmp_path / "branches.json"
    line_path.write_text(json.dumps(lines))
    branch_path.write_text(json.dumps(branches))
    listed = tmp_path / "files.txt"
    listed.write_text("src/methods/rigid_quench.rs\nmissing.rs\n")
    out = tmp_path / "out"
    assert coverage_receipts.main([str(line_path), str(branch_path), str(out), str(listed)]) == 0
    text = (out / "src__methods__rigid_quench.rs.txt").read_text()
    assert "lines_188 10/10 100.00" in text
    assert "branches 3/4 75.00" in text
    assert "line_gap_reason" in text
    assert (out / "missing.rs.txt").read_text().count("absent") >= 2
    with pytest.raises(SystemExit, match="usage"):
        coverage_receipts.main([])


def test_hexamer_numbers_from_a_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert hexamer_numbers.type7([3.0], 0.5) == 3.0
    assert hexamer_numbers.median_iqr([1.0, 2.0, 3.0, 4.0])[0] == pytest.approx(2.5)
    assert "10^{" in hexamer_numbers._sci(0.0001234)
    logs = tmp_path / "water"
    logs.mkdir()
    header = (
        "seed\tarm\tsigned_dE_polished_Eh\trmsd_label_free_A\twall_s\n"
    )
    rows = [header]
    for seed in range(6):
        for arm, gap in (("crest", "0"), ("paper4000", "1e-6"), ("fair", "2e-6")):
            rows.append(f"{seed}\t{arm}\t{gap}\t0.1\t{10 + seed}\n")
    rows.append("# comment\n")
    rows.append("summary\tnope\n")
    rows.append("\n")
    (logs / "NUMBERS.tsv").write_text("".join(rows))
    body = "\n".join(
        [
            "a",
            "b",
            "c",
            hexamer_numbers.CPU,
            "dynamics_steps=4",
            "opt_steps=5",
            "Total number of energy+grad calls: 9",
        ]
    )
    for seed in range(6):
        (logs / f"crest-hexamer-seed{seed}.log").write_text(body + "\n")
    monkeypatch.setenv("WATER_LOGS", str(logs))
    loaded = hexamer_numbers.load()
    assert loaded[0]["old"] == 9
    assert loaded[0]["engrad"] == 9
    stats = hexamer_numbers.walls(loaded)
    assert "crest" in stats
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    hexamer_numbers.main()
    assert "median_iqr type-7" in buf.getvalue()
    sys.argv = ["hex", "tables"]
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    hexamer_numbers.main()
    assert "counters" in buf.getvalue()
    (logs / "crest-hexamer-seed0.log").write_text(
        "a\nb\nc\n" + hexamer_numbers.CPU + "\nno counters\n"
    )
    with pytest.raises(SystemExit, match="missing step counters"):
        hexamer_numbers.load()
    (logs / "crest-hexamer-seed0.log").write_text(
        body.replace("Total number of energy+grad calls: 9", "none") + "\n"
    )
    with pytest.raises(SystemExit, match="missing energy"):
        hexamer_numbers._engrad(logs / "crest-hexamer-seed0.log")
    (logs / "crest-hexamer-seed0.log").write_text("a\nb\nc\nwrong cpu\n")
    with pytest.raises(SystemExit, match="cpu"):
        hexamer_numbers.load()
    (logs / "crest-hexamer-seed0.log").unlink()
    with pytest.raises(SystemExit, match="missing"):
        hexamer_numbers.load()
    monkeypatch.setenv("WATER_LOGS", str(tmp_path / "absent"))
    with pytest.raises(SystemExit, match="missing water logs"):
        hexamer_numbers._logs()


def test_watch_crest_counts_replacements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "scratch"
    out = tmp_path / "out"
    seed = root / "crest-seed0"
    seed.mkdir(parents=True)
    log = seed / "crestopt.log"
    prefix = "x" * 400
    log.write_text(prefix + "\nEtot=1\n")
    empty = root / "crest-seed1"
    empty.mkdir()
    monkeypatch.setattr(watch_crest_steps, "ROOT", root)
    monkeypatch.setattr(watch_crest_steps, "OUT", out)
    phase = {"n": 0}

    def stop(_seconds: float) -> None:
        phase["n"] += 1
        if phase["n"] == 1:
            log.write_text(prefix + "\nEtot=1\nEtot=2\n")
        elif phase["n"] == 2:
            log.write_text("z" * 400 + "\nEtot=9\n")
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_crest_steps.time, "sleep", stop)
    with pytest.raises(KeyboardInterrupt):
        watch_crest_steps.main()
    text = (out / "crest-seed0-optsteps.txt").read_text()
    assert "opt_steps=3" in text
    assert "last_file=1" in text


def test_replay_control_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_MSGDIR", str(tmp_path))
    (tmp_path / "0001.txt").write_bytes(b"subject\n\nbody\n")
    rows = (
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
        b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
        b"dddddddddddddddddddddddddddddddddddddddd "
        b"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n"
    )

    stamps = iter([b"200", b"100"])

    def fake_git(*args, env=None, data=None):
        if args[:1] == ("log",):
            return rows
        if args[0] == "-c":
            return b"cccccccccccccccccccccccccccccccccccccccc\n"
        if args[:2] == ("cat-file", "commit") and args[2].startswith("d"):
            return b"tree eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n\nsecond line\n"
        if args[:2] == ("cat-file", "commit"):
            stamp = next(stamps)
            return (
                b"author Rohit Goswami <rgoswami@ieee.org> " + stamp + b" -0600\n"
            )
        raise AssertionError(args)

    monkeypatch.setattr(replay_v127, "git", fake_git)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    replay_v127.main()
    text = buf.getvalue()
    assert "earlier [2]" in text
    assert "TIP48" in text


def test_replay_git_reports_a_missing_ref() -> None:
    with pytest.raises(SystemExit):
        replay_v127.git("rev-parse", "this-ref-is-not-in-the-repo-xyz")


def test_replay_rejects_a_bad_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_MSGDIR", str(tmp_path))
    (tmp_path / "0001.txt").write_bytes(b"no trailing newline")

    def fake_git(*args, env=None, data=None):
        if args[:1] == ("log",):
            return (
                b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
                b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
            )
        raise AssertionError(args)

    monkeypatch.setattr(replay_v127, "git", fake_git)
    with pytest.raises(SystemExit, match="newline"):
        replay_v127.main()


def test_archived_figure_record_matches_the_blob() -> None:
    proc = subprocess.run(
        [sys.executable, str(HERE / "write_figure_provenance_v128.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    blob = subprocess.check_output(
        ["git", "cat-file", "blob", write_figure_provenance_v128.BLOB],
        cwd=ROOT,
    )
    assert proc.stdout == blob


def test_check_cuh2_records_and_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = check_cuh2_blobs.records(ROOT / "paper/jctc/data/hq_sci_cuh2/PROVENANCE.txt")
    assert len(rows) == 8
    short = tmp_path / "PROVENANCE.txt"
    short.write_text("Log only.txt\nBlob abc\nHash def\n")
    with pytest.raises(SystemExit, match="PROVENANCE lists"):
        check_cuh2_blobs.records(short)
    root = tmp_path / "logs"
    root.mkdir()
    name, blob, sha = rows[0]
    (root / "PROVENANCE.txt").write_text(
        f"Log {name}.\nBlob {blob}.\nHash {sha}.\n" * 1
        + "".join(
            f"Log extra{i}.txt.\nBlob {blob}.\nHash {sha}.\n" for i in range(7)
        )
    )
    (root / name).write_bytes(b"not the log")
    with pytest.raises(SystemExit, match="sha256"):
        check_cuh2_blobs.main([str(root), str(tmp_path)])

    def fake_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args[:1] == ("rev-parse",):
            return subprocess.CompletedProcess(args, 1, "", "missing commit")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(check_cuh2_blobs, "git", fake_git)
    real = ROOT / "paper/jctc/data/hq_sci_cuh2"
    with pytest.raises(SystemExit, match="not available|at "):
        check_cuh2_blobs.compare(tmp_path, real)


def test_redact_main_rewrites_a_home_path(tmp_path: Path) -> None:
    import redact_paths

    raw = tmp_path / "raw.txt"
    out = tmp_path / "out.txt"
    mapped = tmp_path / "map.txt"
    home = str(Path.home())
    raw.write_text(f"see {home}/notes\n")
    old = sys.argv
    sys.argv = ["redact", str(raw), str(out), str(mapped), home]
    try:
        redact_paths.main()
    finally:
        sys.argv = old
    assert home not in out.read_text()
    assert "raw_sha256" in mapped.read_text()
    sys.argv = ["redact"]
    try:
        with pytest.raises(SystemExit, match="usage"):
            redact_paths.main()
    finally:
        sys.argv = old


def test_house_drawers_write_pdfs(tmp_path: Path) -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    import fig_algorithm_stack
    import fig_rec_arms
    import figstyle

    stack = tmp_path / "stack.pdf"
    arms = tmp_path / "arms.pdf"
    fig_algorithm_stack.draw(stack)
    fig_rec_arms.draw(arms)
    assert stack.stat().st_size > 100
    assert arms.stat().st_size > 100
    figstyle.setup()
    assert figstyle.plain(0.5) == "0.5"
    assert "500" in figstyle.plain(1500)
    runpy.run_path(str(HERE / "tip4p_constants.py"), run_name="__main__")


def test_modules_run_as_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "toc.pdf"
    monkeypatch.setattr(sys, "argv", ["toc_graphic.py", str(dest)])
    runpy.run_path(str(HERE / "toc_graphic.py"), run_name="__main__")
    assert dest.is_file()
    monkeypatch.setattr(
        sys,
        "argv",
        ["write_figure_provenance_v128.py"],
    )
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(HERE / "write_figure_provenance_v128.py"), run_name="__main__")
    assert caught.value.code in (None, 0)


def test_message_newline_script(tmp_path: Path) -> None:
    _git_identity(tmp_path)
    env = os.environ.copy()
    env.pop("GIT_AUTHOR_DATE", None)
    env.pop("GIT_COMMITTER_DATE", None)
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    (tmp_path / "f").write_text("x\n")
    subprocess.check_call(["git", "add", "f"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-m", "one line\n"], cwd=tmp_path, env=env)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    (tmp_path / "f").write_text("y\n")
    subprocess.check_call(["git", "add", "f"], cwd=tmp_path)
    tree = subprocess.check_output(["git", "write-tree"], cwd=tmp_path, text=True).strip()
    parent = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    new = subprocess.check_output(
        ["git", "-c", "commit.gpgsign=false", "commit-tree", tree, "-p", parent],
        cwd=tmp_path,
        input=b"two blank lines\n\n",
        env=env,
    ).decode().strip()
    subprocess.check_call(["git", "reset", "--soft", new], cwd=tmp_path)
    script = HERE / "check_message_newlines.sh"
    bad = subprocess.run(["sh", script, base], cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode == 1
    assert "trailing_newlines" in bad.stdout
    good_repo = tmp_path / "good"
    good_repo.mkdir()
    subprocess.check_call(["git", "init"], cwd=good_repo)
    (good_repo / "f").write_text("x\n")
    subprocess.check_call(["git", "add", "f"], cwd=good_repo)
    subprocess.check_call(["git", "commit", "-m", "fine\n"], cwd=good_repo, env=env)
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=good_repo, text=True
    ).strip()
    (good_repo / "f").write_text("z\n")
    subprocess.check_call(["git", "add", "f"], cwd=good_repo)
    subprocess.check_call(["git", "commit", "-m", "also fine\n"], cwd=good_repo, env=env)
    good = subprocess.run(["sh", script, parent], cwd=good_repo, capture_output=True, text=True)
    assert good.returncode == 0
    assert "messages end in one newline" in good.stdout


def test_pack_script_refuses_a_foreign_tree(tmp_path: Path) -> None:
    _git_identity(tmp_path)
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    (tmp_path / "Cargo.toml").write_text("package = {}\n")
    subprocess.check_call(["git", "add", "Cargo.toml"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-m", "init\n"], cwd=tmp_path)
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=tmp_path, text=True).strip()
    script = HERE / "pack_lj75_source.sh"
    refused = subprocess.run(
        ["sh", script, tree, str(tmp_path / "out.tar.xz")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 1
    assert "not present" in refused.stderr


def test_hardware_capture_reads_the_host(tmp_path: Path) -> None:
    sleep = subprocess.Popen(["sleep", "30"])
    try:
        out = tmp_path / "receipt.txt"
        env = os.environ.copy()
        env["BUILD_HEAD"] = "6146da08dbab883c21a5607bfa1da7d4677a8008"
        env["BUILD_TREE"] = "37535ec1a6e99d47a2afe0752072a33af64dccb3"
        subprocess.check_call(
            [HERE / "lj75_hardware_capture.sh", str(sleep.pid), str(out)],
            env=env,
        )
        text = out.read_text()
        assert "host " in text
        assert "host-a" not in text
        assert "sha256" in text
    finally:
        sleep.kill()


def test_build_script_rejects_a_bad_checkout(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "build_librgpot_cuh2.sh"
    usage = subprocess.run([script], capture_output=True, text=True)
    assert usage.returncode == 2
    _git_identity(tmp_path)
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    (tmp_path / "f").write_text("x\n")
    subprocess.check_call(["git", "add", "f"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-m", "init\n"], cwd=tmp_path)
    mismatch = subprocess.run(
        [script, str(tmp_path), str(tmp_path / "out.so")],
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode == 2
    assert "is not" in mismatch.stderr


def _git_identity(path: Path) -> None:
    subprocess.check_call(["git", "init"], cwd=path)
    subprocess.check_call(["git", "config", "user.email", "rgoswami@ieee.org"], cwd=path)
    subprocess.check_call(["git", "config", "user.name", "Rohit Goswami"], cwd=path)


def _git_repo(path: Path, old: str, new: str) -> None:
    _git_identity(path)
    lock = path / "Cargo.lock"
    lock.write_text(old)
    subprocess.check_call(["git", "add", "Cargo.lock"], cwd=path)
    subprocess.check_call(["git", "commit", "-m", "old\n"], cwd=path)
    lock.write_text(new)
    subprocess.check_call(["git", "add", "Cargo.lock"], cwd=path)
    subprocess.check_call(["git", "commit", "-m", "new\n"], cwd=path)
