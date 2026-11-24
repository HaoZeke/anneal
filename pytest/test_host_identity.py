# Copyright (c) anneal developers
"""No raw host identity reaches a row, a report or a log.

A hostname goes through ``host_label``: the sandbox name ``bench`` as is, any
other name as a salted digest. The kernel boot id goes through ``boot_label``,
a salted digest, and is read only inside ``scripts/machine_provenance.py``.
Python, shell and Rust are all scanned. A line that reads the host any other
way fails, with no exemption for a file. A line that must read it raw is
exempt by its exact text and a reason.
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
    re.compile(r"\$\(\s*hostname\b|`\s*hostname\s*`|\bhostname\s*[>|]|^\s*hostname\s*$|uname\s+-[a-z]*[na]\b"),
    re.compile(r"hostnamectl|\bhostid\b|SLURMD_NODENAME|SLURM_(?:JOB_)?NODELIST|SLURM_SUBMIT_HOST"),
    re.compile(r"random/boot_id|\bboot_id\s*=\s*\$\(\s*cat\b"),
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
    re.compile(
        r'host_label::boot_label\(&(?:first_line|read_trimmed)\("/proc/sys/kernel/random/boot_id"\)\)'
    ),
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
    (
        "scripts/gmin_panel/tuned.py",
        'for path in (Path("/etc/machine-id"), Path("/proc/sys/kernel/random/boot_id")):',
    ): TUNED_REASON + ", and both ids only feed a digest",
    (
        "examples/lj_board_row.rs",
        'emit("boot_id_path=/proc/sys/kernel/random/boot_id");',
    ): "names the file the boot label stands for and reads nothing",
    (
        "scripts/elja_hq_sci_molslab_bank.sh",
        "HOST=${BANK_HOST:-$(hostname)}",
    ): "the address the compute jobs dial, kept in their environment; every log line prints its host label",
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
        "cat /proc/sys/kernel/random/boot_id > receipts/boot_id.txt",
        'echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"',
        'boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()',
        'let boot_id = read_trimmed("/proc/sys/kernel/random/boot_id");',
        "boot_id=$(cat \"$f\")",
        "hostnamectl --static",
        'echo "node=$SLURMD_NODENAME"',
        "echo $(uname -srvmn)",
    ):
        assert flagged(line), line
    for line in (
        "host = host_label(socket.gethostname())",
        "if host_label(os.uname().nodename) != 'bench':",
        'host=$(python3 "$ROOT/scripts/machine_provenance.py" host-label)',
        "sudo unshare --uts sh -c 'hostname bench; exec \"$@\"'",
        'assert fields.get("hostname") == "bench"',
        'rows.append({"host": info["hostname"]})',
        'echo "boot_id=$(python3 "$here/machine_provenance.py" boot-label)"',
        "boot_id = boot_label()",
        'let boot_id = host_label::boot_label(&first_line(path));',
        "uname -s",
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
    if machine_provenance.BOOT_ID_PATH.is_file():
        boot = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "machine_provenance.py"), "boot-label"],
            capture_output=True, text=True, check=True,
        )
        assert boot.stdout.strip() == machine_provenance.boot_label()
        assert machine_provenance.BOOT_ID_PATH.read_text().strip() not in boot.stdout


def test_the_board_examples_use_the_digest_rule_of_the_python_label():
    source = (ROOT / "examples" / "common" / "host_label.rs").read_text()
    salt = machine_provenance.HOST_LABEL_SALT.decode().replace("\0", "\\0")
    assert f'b"{salt}"' in source
    assert f'"{machine_provenance.SANDBOX_HOSTNAME}"' in source
    assert 'host_label("elja-compute-17").as_deref(),\n            Some("host-99fd2f34271ff224")' in source
    boot_salt = machine_provenance.BOOT_LABEL_SALT.decode().replace("\0", "\\0")
    assert f'b"{boot_salt}"' in source
    assert "Some(\"boot-84eaa8c774326c3e\")" in source
    assert machine_provenance.boot_label("00000000-1111-2222-3333-444444444444") == "boot-84eaa8c774326c3e"
    for example in ("lj_board_row", "lj55_board_arm", "lj_lean_relax_budget"):
        text = (ROOT / "examples" / f"{example}.rs").read_text()
        assert '#[path = "common/host_label.rs"]' in text, example
