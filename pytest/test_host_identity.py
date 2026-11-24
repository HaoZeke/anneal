# Copyright (c) anneal developers
"""No raw host identity reaches a row or a report.

A hostname goes through ``host_label``: the sandbox name ``bench`` as is, any
other name as a salted digest. A line that reads the host any other way fails,
with no exemption for a file. A line that must read it raw is exempt by its
exact text and a reason.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import machine_provenance  # noqa: E402

ROOTS = ("scripts", "bench", "paper/jctc/scripts", "examples", "benches", "src", "python")
SUFFIXES = {".py", ".sh", ".rs"}
HOME = {"scripts/machine_provenance.py"}

READS = (
    re.compile(r"gethostname|gethostbyname|getfqdn|platform\.node\b|platform\.uname|os\.uname|\.nodename"),
    re.compile(r"kernel/hostname|/etc/hostname|\bHOSTNAME\b"),
    re.compile(r"""\(\s*\[\s*["']hostname["']|(?:run|call|Popen|getoutput|check_output)\(\s*["']hostname|["']uname["']\s*,\s*["']-[na]["']|Command::new\(\s*"hostname"\s*\)"""),
    re.compile(r"\$\(\s*hostname\b|`\s*hostname\s*`|\bhostname\s*[>|]|^\s*hostname\s*$|uname\s+-[na]\b"),
)
WRAPPED = (
    re.compile(r"host_label\(\s*(?:socket\.gethostname\(\)|os\.uname\(\)\.nodename|platform\.node\(\))\s*\)"),
    re.compile(
        r'host_label\(Path\("/proc/sys/kernel/hostname"\)\.read_text\(\)\.splitlines\(\)\[0\]\.strip\(\)\)'
    ),
    re.compile(
        r'host_label::host_label\(&(?:first_line|read_trimmed)\("/proc/sys/kernel/hostname"\)\)'
    ),
    re.compile(r"host_label::host_label\(&hostname\(\)\)"),
)
TUNED_REASON = "frozen: the committed tuned rows are verified against this file's exact blob"
EXEMPT = {
    (
        "scripts/gmin_panel/tuned.py",
        'if socket.gethostname() != "bench":',
    ): TUNED_REASON + ", and the name only decides whether the run starts",
    (
        "scripts/gmin_panel/tuned.py",
        'raise SystemExit(f"uts name is {socket.gethostname()}")',
    ): TUNED_REASON + ", and the name is shown on a refused start, never written to a row",
    (
        "examples/lj_lean_relax_budget.rs",
        'Command::new("hostname")',
    ): "the raw name feeds host_label in main and is written nowhere",
}


def guarded_lines():
    for base in ROOTS:
        top = ROOT / base
        if not top.is_dir():
            continue
        for path in sorted(top.rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            if {"__pycache__", "target", "node_modules"} & set(path.parts):
                continue
            yield path.relative_to(ROOT).as_posix(), path.read_text(errors="replace").splitlines()


def is_comment(relative: str, line: str) -> bool:
    return line.startswith(("//", "///", "//!")) if relative.endswith(".rs") else line.startswith("#")


def test_no_raw_host_identity_is_read_outside_host_label():
    found, used = [], set()
    for relative, lines in guarded_lines():
        if relative in HOME:
            continue
        for number, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or is_comment(relative, line):
                continue
            for wrapped in WRAPPED:
                line_left = wrapped.sub("", line)
                if line_left != line:
                    line = line_left
            if not any(read.search(line) for read in READS):
                continue
            key = (relative, raw.strip())
            if key in EXEMPT:
                used.add(key)
            else:
                found.append(f"{relative}:{number}: {raw.strip()}")
    assert not found, "pass the hostname to host_label, or drop the field:\n" + "\n".join(found)
    assert used == set(EXEMPT), sorted(set(EXEMPT) - used)
    assert all(reason.strip() and "\n" not in reason for reason in EXEMPT.values())


def test_the_guard_flags_the_reads_it_exists_for():
    def flagged(line: str) -> bool:
        for wrapped in WRAPPED:
            line = wrapped.sub("", line)
        return any(read.search(line) for read in READS)

    for line in (
        "hostname = socket.gethostname()",
        'host = os.uname().nodename',
        "name = platform.node()",
        'host=$(hostname)',
        'echo "host=$(hostname) job=$SLURM_JOB_ID"',
        "hostname > receipts/uts.txt",
        'printf "%s" "$(uname -n)"',
        "host=$(cat /proc/sys/kernel/hostname)",
        'let host = read_trimmed("/proc/sys/kernel/hostname");',
        'subprocess.check_output(["hostname"])',
        'subprocess.check_output("hostname", shell=True)',
        'os.environ["HOSTNAME"]',
    ):
        assert flagged(line), line
    for line in (
        "host = host_label(socket.gethostname())",
        "if host_label(os.uname().nodename) != 'bench':",
        'host=$(python3 "$ROOT/scripts/machine_provenance.py" host-label)',
        "sudo unshare --uts sh -c 'hostname bench; exec \"$@\"'",
        'assert fields.get("hostname") == "bench"',
        'rows.append({"host": info["hostname"]})',
    ):
        assert not flagged(line), line


def test_the_shell_reads_the_label_through_the_same_function():
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "machine_provenance.py"), "host-label"],
        capture_output=True, text=True, check=True,
    )
    assert run.stdout.strip() == machine_provenance.host_label()
    refused = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "machine_provenance.py"), "hostname"],
        capture_output=True, text=True, check=False,
    )
    assert refused.returncode == 2 and refused.stdout == ""


def test_the_board_examples_use_the_digest_rule_of_the_python_label():
    source = (ROOT / "examples" / "common" / "host_label.rs").read_text()
    salt = machine_provenance.HOST_LABEL_SALT.decode().replace("\0", "\\0")
    assert f'b"{salt}"' in source
    assert f'"{machine_provenance.SANDBOX_HOSTNAME}"' in source
    assert 'host_label("elja-compute-17").as_deref(),\n            Some("host-99fd2f34271ff224")' in source
    for example in ("lj_board_row", "lj55_board_arm", "lj_lean_relax_budget"):
        text = (ROOT / "examples" / f"{example}.rs").read_text()
        assert '#[path = "common/host_label.rs"]' in text, example
