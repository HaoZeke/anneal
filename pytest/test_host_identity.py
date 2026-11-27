# Copyright (c) anneal developers
"""No raw host identity reaches a row, a report or a log.

A hostname goes through ``host_label``: the sandbox name ``bench`` as is, any
other name as a salted digest. The kernel boot id goes through ``boot_label``,
a salted digest. The boot id is read in two files only,
``scripts/machine_provenance.py`` for Python and shell and
``examples/common/host_label.rs`` for Rust. A Python script may read the host
name, but only as the argument of ``host_label``; shell goes through
``machine_provenance.py``; Rust reads it only in ``host_label.rs``. A bank or
catalog endpoint is written as ``endpoint_label``. The machine id is read
in ``scripts/machine_provenance.py``. The Sutton-Chen rerun calls
``machine_digest`` and does not read the file itself. Python, shell, batch
scripts (``.sbatch``) and Rust are all scanned, ``experiments/`` too. A line
that reads the host any other way fails, with no exemption for a file. A line
that must read it raw is exempt by its exact text and a reason.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import machine_provenance  # noqa: E402

ROOTS = (
    "scripts", "bench", "paper/jctc/scripts", "examples", "benches", "src", "python", "tests",
    "experiments", "workflow", ".github", "docs/source",
)
SUFFIXES = {".py", ".sh", ".bash", ".sbatch", ".slurm", ".smk", ".rs", ".yml", ".yaml"}
NAMES = {"Snakefile"}
HOME = {"scripts/machine_provenance.py", "examples/common/host_label.rs"}

READS = (
    re.compile(r"gethostname|gethostbyname|getfqdn|platform\.node\b|platform\.uname|os\.uname|\.nodename"),
    re.compile(r"kernel/hostname|/etc/hostname|\bHOSTNAME\b"),
    re.compile(r"""\(\s*\[\s*["']hostname["']|(?:run|call|Popen|getoutput|check_output)\(\s*["']hostname|["']uname["']\s*,\s*["']-[na]["']|Command::new\(\s*"hostname"\s*\)"""),
    re.compile(r"\$\(\s*hostname\b|`\s*hostname\s*`|\bhostname\s*[>|]|^\s*hostname\s*$|uname\s+-[a-z]*[na]\b"),
    re.compile(r"hostnamectl|\bhostid\b|SLURMD_NODENAME|SLURM_(?:JOB_)?NODELIST|SLURM_SUBMIT_HOST"),
    re.compile(r"random/boot_id|\bboot_id\s*=\s*\$\(\s*cat\b"),
    re.compile(r"/etc/machine-id|dbus/machine-id|product_uuid|IOPlatformUUID"),
)
WRAPPED = (
    re.compile(r"host_label\(\s*(?:socket\.gethostname\(\)|os\.uname\(\)\.nodename|platform\.node\(\))\s*\)"),
    re.compile(
        r'host_label\(Path\("/proc/sys/kernel/hostname"\)\.read_text\(\)\.splitlines\(\)\[0\]\.strip\(\)\)'
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
        "scripts/gmin_panel/tuned.py",
        'for path in (Path("/etc/machine-id"), Path("/proc/sys/kernel/random/boot_id")):',
    ): TUNED_REASON + ", and both ids only feed a digest",
    (
        "scripts/elja_hq_sci_bankrpc.sh",
        "HOST=${BANK_HOST:-$(hostname)}",
    ): "the address the compute jobs dial, kept in their environment; the log line prints its host label",
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
            if (path.suffix not in SUFFIXES and path.name not in NAMES) or not path.is_file():
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
        'raw = Path("/etc/machine-id").read_bytes().strip()',
        "cat /etc/machine-id > receipts/id.txt",
        'let id = read_trimmed("/var/lib/dbus/machine-id");',
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


LABEL_SOURCE = re.compile(r"\b(?:host|boot|endpoint)-label\b|\b(?:read_)?(?:host|boot|endpoint)_label\b")
SHELL_ASSIGN = re.compile(r"^\s*(?:export\s+|local\s+|declare\s+(?:-\w+\s+)?|readonly\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
ADDRESS_NAME = re.compile(r"ADDR|ENDPOINT|URL|_RPC|RPC_|BANK_HOST|SERVER|PEER|REMOTE", re.I)
SHELL_SINK = re.compile(
    r"_RPC=|\bssh\b|\bscp\b|\brsync\b|\bsftp\b|\bcurl\b|\bwget\b|\bnc\b|\bncat\b|\bping\b"
    r"|\b[a-z][a-z0-9+.-]*://|--(?:host|address|endpoint|server|url)\b"
)
PY_SINK_CALLS = {
    "connect", "create_connection", "urlopen", "bind", "sendto", "getaddrinfo",
    "gethostbyname", "BankClient", "CatalogClient", "Request",
}
PY_HTTP_CALLS = {"get", "post", "put", "head", "delete", "request"}
PY_HTTP_ROOTS = {"requests", "httpx", "urllib", "session", "http"}
PY_SINK_PROGRAMS = {"ssh", "scp", "rsync", "sftp", "curl", "wget", "nc", "ncat", "ping"}
RS_SINK = re.compile(r"(?:TcpStream::connect|connect_timeout|SocketAddr|\bconnect|set_var)\s*(?:::<[^>]*>)?\(|\bBANK_RPC\b|\bCATALOG_RPC\b")


def logical(lines):
    joined, current = [], ""
    for number, line in enumerate(lines, start=1):
        if line.rstrip().endswith("\\"):
            current += line.rstrip()[:-1] + " "
            continue
        joined.append((number, current + line))
        current = ""
    return joined


def mentions(name, line):
    return re.search(rf"\$\{{?{re.escape(name)}\b", line) is not None


def shell_label_address_uses(lines):
    """Lines where a value derived from a label is used as a network address."""
    rows = [(n, text) for n, text in logical(lines) if not text.lstrip().startswith("#")]
    tainted = set()
    changed = True
    while changed:
        changed = False
        for _, text in rows:
            match = SHELL_ASSIGN.match(text)
            if not match or match.group(1) in tainted:
                continue
            value = match.group(2)
            if LABEL_SOURCE.search(value) or any(mentions(name, value) for name in tainted):
                tainted.add(match.group(1))
                changed = True
    found = []
    for number, text in rows:
        match = SHELL_ASSIGN.match(text)
        for name in tainted:
            if not mentions(name, text):
                continue
            if SHELL_SINK.search(text) or re.search(rf"\$\{{?{re.escape(name)}\b\}}?:[0-9$]", text):
                found.append((number, text.strip()))
                break
            if match and ADDRESS_NAME.search(match.group(1)):
                found.append((number, text.strip()))
                break
        else:
            if match and match.group(1) in tainted and ADDRESS_NAME.search(match.group(1)):
                found.append((number, text.strip()))
    return found


def python_label_address_uses(source):
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    def names(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    def labelled(node):
        return any(
            isinstance(n, ast.Call) and LABEL_SOURCE.fullmatch(getattr(n.func, "id", getattr(n.func, "attr", "")))
            for n in ast.walk(node)
        )

    tainted = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if labelled(node.value) or names(node.value) & tainted:
                    for target in targets:
                        for name in names(target):
                            if name not in tainted:
                                tainted.add(name)
                                changed = True
    found = []
    for node in ast.walk(tree):
        hit = False
        if isinstance(node, ast.Call):
            callee = getattr(node.func, "id", getattr(node.func, "attr", ""))
            args = [*node.args, *(k.value for k in node.keywords)]
            used = any(labelled(a) or names(a) & tainted for a in args)
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if used and callee in PY_SINK_CALLS:
                hit = True
            if used and callee in PY_HTTP_CALLS and getattr(root, "id", "") in PY_HTTP_ROOTS:
                hit = True
            for a in args:
                if isinstance(a, (ast.List, ast.Tuple)) and a.elts and isinstance(a.elts[0], ast.Constant):
                    if str(a.elts[0].value) in PY_SINK_PROGRAMS and any(labelled(e) or names(e) & tainted for e in a.elts[1:]):
                        hit = True
        elif isinstance(node, ast.JoinedStr):
            text = "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
            if re.search(r"[a-z][a-z0-9+.-]*://", text) and (names(node) & tainted or labelled(node)):
                hit = True
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            pass
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                    and re.search(r"RPC|ADDR|ENDPOINT", target.slice.value)
                    and (labelled(node.value) or names(node.value) & tainted)
                ):
                    hit = True
        if hit:
            found.append((node.lineno, ast.get_source_segment(source, node) or ""))
    return found


def rust_label_address_uses(lines):
    tainted = set()
    for line in lines:
        match = re.match(r"\s*let\s+(?:mut\s+)?([a-z_][a-z0-9_]*)\b.*\b(?:read_)?(?:host|boot|endpoint)_label\b", line)
        if match:
            tainted.add(match.group(1))
    found = []
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("//"):
            continue
        sink = RS_SINK.search(line)
        if sink and any(re.search(rf"\b{name}\b", line[sink.start():]) for name in tainted):
            found.append((number, line.strip()))
        elif sink and LABEL_SOURCE.search(line[sink.start():]) and "panic!" not in line:
            found.append((number, line.strip()))
    return found


def label_address_uses(relative, text):
    lines = text.splitlines()
    if relative.endswith(".py"):
        return python_label_address_uses(text)
    if relative.endswith(".rs"):
        return rust_label_address_uses(lines)
    return shell_label_address_uses(lines)


def test_a_host_label_is_never_used_as_a_network_address():
    """A label names no machine, so a job that dials it dials nothing."""
    found = []
    for relative, lines in guarded_lines():
        if relative.startswith("pytest/") or relative in HOME:
            continue
        for number, line in label_address_uses(relative, "\n".join(lines)):
            found.append(f"{relative}:{number}: {line}")
    assert not found, "a host label is used as an address:\n" + "\n".join(found)


def test_the_address_check_covers_batch_scripts_and_the_walked_trees():
    walked = {relative for relative, _ in guarded_lines()}
    assert any(name.endswith(".sbatch") for name in walked)
    assert any(name.startswith("experiments/") for name in walked)
    assert "scripts/elja_leave_ridge.sbatch" in walked
    assert "experiments/morphology_bias_lj38.sbatch" in walked


def test_the_address_check_flags_a_label_used_to_dial():
    direct = 'HOST=${BANK_HOST:-$(python3 "$ROOT/scripts/machine_provenance.py" host-label)}\nexport BANK_RPC=${HOST}:${port}\n'
    indirect = 'H=$(python3 m.py host-label)\nADDR=$H\nssh "$ADDR" true\n'
    chained = 'H=$(python3 m.py host-label)\nA=$H\nB="$A"\ncurl "http://$B/x"\n'
    named = 'H=$(python3 m.py host-label)\nSERVER_ADDR=$H\n'
    batch = 'host=$(python3 m.py boot-label)\nrsync -a . "$host:/x/"\n'
    for text in (direct, indirect, chained, named, batch):
        assert shell_label_address_uses(text.splitlines()), text
    clean = (
        'HOST=${BANK_HOST:-$(hostname)}\nLABEL=$(python3 m.py host-label)\n'
        'echo "host=$LABEL"\nexport BANK_RPC=${HOST}:${port}\n'
        'echo "rpc=$(python3 m.py endpoint-label "${HOST}:${port}")"\n'
    )
    assert shell_label_address_uses(clean.splitlines()) == []
    python_bad = (
        "import socket\nlabel = host_label(socket.gethostname())\naddr = f'{label}:80'\n"
        "socket.create_connection((addr, 80))\n"
    )
    python_url = "host = host_label('x')\nurl = f'http://{host}/rows'\n"
    python_ssh = "h = host_label('x')\nsubprocess.run(['ssh', h, 'true'])\n"
    python_env = "h = host_label('x')\nenv['BANK_RPC'] = h + ':1'\n"
    for text in (python_bad, python_url, python_ssh, python_env):
        assert python_label_address_uses(text), text
    python_ok = "h = host_label('x')\nrow = {'host': h}\nprint(f'host={h}')\n"
    assert python_label_address_uses(python_ok) == []
    rust_bad = 'let h = host_label::read_host_label();\nlet s = TcpStream::connect(format!("{h:?}:80"));\n'
    assert rust_label_address_uses(rust_bad.splitlines())
    assert rust_label_address_uses(['let h = host_label::read_host_label();', 'println!("hostname={h:?}");']) == []


def test_the_examples_print_an_endpoint_only_as_its_label():
    """The job dials the real address; nothing it prints may contain it."""
    printing = re.compile(r"(?:e?println|panic|format|write(?:ln)?)!\([^;]*\{(?:endpoint|addr|address|sock)\b[^}]*\}")
    found = []
    for example in ("lj_cluster_search", "bank_peek"):
        text = (ROOT / "examples" / f"{example}.rs").read_text()
        found += [f"{example}: {m.group(0)}" for m in printing.finditer(text)]
        assert "host_label::endpoint_label" in text, example
    assert not found, "\n".join(found)
    source = (ROOT / "examples" / "common" / "host_label.rs").read_text()
    assert '"host-99fd2f34271ff224:7426"' in source
    assert machine_provenance.endpoint_label("elja-compute-17:7426") == "host-99fd2f34271ff224:7426"


def test_the_board_examples_read_the_host_only_through_host_label_rs():
    source = (ROOT / "examples" / "common" / "host_label.rs").read_text()
    assert 'pub const UNKNOWN: &str = "unknown";' in source
    for example in ("lj_board_row", "lj55_board_arm", "lj_lean_relax_budget"):
        text = (ROOT / "examples" / f"{example}.rs").read_text()
        assert "/proc/sys/kernel" not in text, example
        assert "absent" not in text.split("fn main")[1].split("uptime")[0], example
