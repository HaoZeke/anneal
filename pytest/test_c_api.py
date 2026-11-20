# Copyright (c) anneal developers
"""The C ABI: the generated header, and the smoke program against it.

``tests/ffi_smoke.c`` runs the box ensemble through the header cbindgen
generates, so a setting that the Rust options carry and the header lacks
fails here as a compile error, and a setting that does not reach the
ensemble fails as a check.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FFI_RS = ROOT / "src" / "ffi.rs"
SCRIPT = ROOT / "scripts" / "ffi_smoke.sh"


def _needs_toolchain():
    for tool in ("cargo", "cbindgen", "cc"):
        if shutil.which(tool) is None:
            pytest.skip(f"needs {tool} on PATH")
    if not (ROOT / "Cargo.toml").exists():
        pytest.skip("needs the source checkout")


@pytest.fixture(scope="module")
def header(tmp_path_factory):
    _needs_toolchain()
    out = tmp_path_factory.mktemp("header") / "anneal-core.h"
    subprocess.run(
        [
            "cbindgen",
            "--config",
            str(ROOT / "cbindgen.toml"),
            "--crate",
            "anneal-core",
            "--output",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return out.read_text()


def _exported(name):
    return re.findall(rf'pub (?:unsafe )?extern "C" fn ({name}\w*)\(', FFI_RS.read_text())


def test_header_declares_every_exported_function(header):
    names = _exported("anneal_")
    assert "anneal_box_minimize_ex" in names
    missing = [n for n in names if not re.search(rf"\b{n}\(", header)]
    assert not missing, f"exported but not in the header: {missing}"


def test_header_names_the_enums_the_options_hold_as_integers(header):
    for enum in (
        "anneal_box_history_t",
        "anneal_box_membership_t",
        "anneal_box_visit_policy_t",
        "anneal_box_coverage_t",
    ):
        assert re.search(rf"\}} {enum};", header), enum


def test_c_smoke_program_passes():
    _needs_toolchain()
    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]


def test_header_carries_every_coverage_counter_and_the_decisions(header):
    stats = re.search(
        r"typedef struct anneal_box_stats_t \{(.*?)\} anneal_box_stats_t;", header, re.S
    )
    assert stats, "anneal_box_stats_t is not in the header"
    fields = re.findall(r"\b(\w+);", stats.group(1))
    assert fields[0] == "struct_size"
    for field in (
        "coverage_peer_recrossings",
        "coverage_peer_only_recrossings",
        "coverage_escape_updates",
        "coverage_novelty_updates",
        "coverage_published_samples",
        "coverage_applied_foreign_samples",
        "coverage_sample_peer_checks",
        "coverage_sample_anchor_overlaps",
        "coverage_sample_anchor_only_overlaps",
        "coverage_sample_overlaps",
        "coverage_constrained_repulsions",
        "coverage_regions_largest",
        "coverage_regions_smallest",
    ):
        assert field in fields, field
    assert fields[-1] == "coverage_decisions", "new fields are appended at the end"
    assert re.search(r"anneal_box_coverage_decisions_t coverage_decisions;", header)
    decisions = re.search(
        r"typedef struct anneal_box_coverage_decisions_t \{(.*?)\} anneal_box_coverage_decisions_t;",
        header,
        re.S,
    )
    assert decisions
    names = re.findall(r"\b(\w+);", decisions.group(1))
    assert len(names) == 12


def _struct(header, name):
    found = re.search(
        rf"((?:/\*\*(?:(?!\*/).)*\*/\s*)?)typedef struct {name} \{{(.*?)\}} {name};",
        header,
        re.S,
    )
    assert found, f"{name} is not in the header"
    return found.group(1), found.group(2)


def _fields(body):
    return re.findall(r"\b(\w+);", body)


def _documented_fields(body):
    return re.findall(
        r"/\*\*((?:(?!\*/).)*)\*/\s*[\w\s\*]+?\b(\w+);", body, re.S
    )


@pytest.mark.parametrize("name", ["anneal_box_stats_t", "anneal_box_coverage_decisions_t"])
def test_header_documents_every_field(header, name):
    _, body = _struct(header, name)
    documented = _documented_fields(body)
    assert [field for _, field in documented] == _fields(body)
    empty = [field for doc, field in documented if not re.sub(r"[\s*]", "", doc)]
    assert not empty, f"no description in the header for {empty}"


def test_header_says_where_the_first_layout_of_the_stats_ends(header):
    doc, body = _struct(header, "anneal_box_stats_t")
    fields = _fields(body)
    assert fields[16] == "coverage_repelled_proposals"
    assert fields[17] == "coverage_peer_recrossings"
    assert "17 eight-byte words" in doc
    _, decisions = _struct(header, "anneal_box_coverage_decisions_t")
    words = len(fields) - 1 + len(_fields(decisions))
    assert f"{words} words in all" in doc.replace("\n * ", " ")
    for field in ("struct_size", "coverage_repelled_proposals", "coverage_peer_recrossings",
                  "coverage_constrained_repulsions", "coverage_regions_largest",
                  "coverage_regions_smallest", "coverage_decisions", "n_evals"):
        assert f"`{field}`" in doc.replace("\n * ", " "), field


def test_the_bindings_page_lists_the_stats_layouts_as_the_header_has_them(header):
    _, body = _struct(header, "anneal_box_stats_t")
    fields = _fields(body)
    page = (ROOT / "docs" / "orgmode" / "bindings.org").read_text()
    first = page.split("- first layout", 1)[1].split("- second layout", 1)[0]
    second = page.split("- second layout", 1)[1].split("\n\n", 1)[0]
    assert re.findall(r"=(\w+)=", first) == fields[:17]
    assert re.findall(r"=(\w+)=", second) == fields[17:]


def _ci_valgrind_flags():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    step = ci.split("- name: valgrind ffi_smoke", 1)[1].split("\n\n", 1)[0]
    return re.findall(r"(--[a-z-]+=\S+?)(?:\s|\\|$)", step)


def test_the_smoke_script_runs_under_valgrind_with_the_flags_of_ci(tmp_path):
    _needs_toolchain()
    stub = tmp_path / "bin" / "valgrind"
    stub.parent.mkdir()
    record = tmp_path / "argv"
    stub.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > {record}\n'
        'for last; do :; done\n'
        'exec "$last"\n'
    )
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{stub.parent}:{os.environ['PATH']}", "FFI_SMOKE_VALGRIND": "1"}
    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=ROOT, capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    flags = _ci_valgrind_flags()
    assert "--error-exitcode=1" in flags and "--leak-check=full" in flags, flags
    argv = record.read_text().split()
    assert argv[:-1] == flags
    assert argv[-1].endswith("ffi_smoke")

    plain = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**env, "FFI_SMOKE_VALGRIND": "0"},
    )
    assert plain.returncode == 0
    assert record.read_text().split() == argv, "valgrind ran with FFI_SMOKE_VALGRIND=0"
