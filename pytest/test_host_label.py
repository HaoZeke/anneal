# Copyright (c) anneal developers
"""``host_label`` writes the sandbox name as is and any other hostname as a salted digest."""

from __future__ import annotations

import hashlib
import re
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
