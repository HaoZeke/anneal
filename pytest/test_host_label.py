# Copyright (c) anneal developers
"""``host_label`` writes the sandbox name as is and any other hostname as a salted digest; ``boot_label`` digests the boot id."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import machine_provenance  # noqa: E402


def test_a_hostname_other_than_the_sandbox_name_is_only_ever_a_salted_digest():
    assert machine_provenance.host_label("bench") == "bench"
    for name in ("elja-compute-17", "terra", "Bench", "bench ", "login1.example.org"):
        label = machine_provenance.host_label(name)
        assert re.fullmatch(r"host-[0-9a-f]{16}", label), label
        assert label == "host-" + hashlib.sha256(machine_provenance.HOST_LABEL_SALT + name.encode()).hexdigest()[:16]
        assert name not in label
    assert machine_provenance.host_label("a") != machine_provenance.host_label("b")
    assert machine_provenance.host_label("elja-compute-17") == "host-99fd2f34271ff224"
    with pytest.raises(RuntimeError):
        machine_provenance.host_label("")


def test_the_default_host_label_reads_this_machine(monkeypatch):
    monkeypatch.setattr(machine_provenance.socket, "gethostname", lambda: "secret-node")
    assert machine_provenance.host_label() == machine_provenance.host_label("secret-node")
    monkeypatch.setattr(machine_provenance.socket, "gethostname", lambda: "bench")
    assert machine_provenance.host_label() == "bench"


def test_a_boot_id_is_only_ever_a_salted_digest():
    for boot_id in ("00000000-1111-2222-3333-444444444444", "ffffffff-eeee-4ddd-8ccc-bbbbbbbbbbbb"):
        label = machine_provenance.boot_label(boot_id)
        assert re.fullmatch(r"boot-[0-9a-f]{16}", label), label
        assert label == "boot-" + hashlib.sha256(machine_provenance.BOOT_LABEL_SALT + boot_id.encode()).hexdigest()[:16]
        assert boot_id not in label
        assert machine_provenance.boot_label(boot_id + "\n") == label
    assert machine_provenance.boot_label("00000000-1111-2222-3333-444444444444") == "boot-84eaa8c774326c3e"
    assert machine_provenance.boot_label("a") != machine_provenance.boot_label("b")
    assert machine_provenance.BOOT_LABEL_SALT != machine_provenance.HOST_LABEL_SALT
    with pytest.raises(RuntimeError):
        machine_provenance.boot_label(" \n")


def test_the_default_boot_label_reads_this_machines_boot_id(monkeypatch, tmp_path):
    path = tmp_path / "boot_id"
    path.write_text("0123abcd-0000-4000-8000-000000000001\n")
    monkeypatch.setattr(machine_provenance, "BOOT_ID_PATH", path)
    assert machine_provenance.boot_label() == machine_provenance.boot_label("0123abcd-0000-4000-8000-000000000001")
    monkeypatch.setattr(machine_provenance, "BOOT_ID_PATH", tmp_path / "missing")
    with pytest.raises(OSError):
        machine_provenance.boot_label()


def test_the_label_command_names_a_host_it_is_given():
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "machine_provenance.py"), "host-label", "elja-compute-17"],
        capture_output=True, text=True, check=True,
    )
    assert run.stdout.strip() == "host-99fd2f34271ff224"


def test_an_endpoint_keeps_its_port_and_scheme_and_loses_its_host():
    label = machine_provenance.endpoint_label
    assert label("elja-compute-17:7426") == "host-99fd2f34271ff224:7426"
    assert label("tcp://elja-compute-17:9000/x") == "tcp://host-99fd2f34271ff224:9000/x"
    assert label("[fe80::1]:80").endswith(":80") and "fe80" not in label("[fe80::1]:80")
    assert label("bench:5") == "bench:5"
    for endpoint in (":80", "", "ipc:///tmp/bank"):
        with pytest.raises(ValueError):
            label(endpoint)


def test_the_endpoint_command_never_echoes_a_host():
    ok = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "machine_provenance.py"), "endpoint-label", "elja-compute-17:7426"],
        capture_output=True, text=True, check=True,
    )
    assert ok.stdout.strip() == "host-99fd2f34271ff224:7426"
    refused = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "machine_provenance.py"), "endpoint-label", ":80"],
        capture_output=True, text=True, check=False,
    )
    assert refused.returncode == 2 and refused.stdout == ""
