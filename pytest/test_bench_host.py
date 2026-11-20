# Copyright (c) anneal developers
"""A benchmark row names its host by label, never by the machine's own hostname."""

import os
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "openconf_h2h"
sys.path.insert(0, str(HARNESS))

import bench_host  # noqa: E402


def uname_as(monkeypatch, nodename):
    real = os.uname()
    monkeypatch.setattr(
        bench_host.os,
        "uname",
        lambda: os.uname_result((real.sysname, nodename, real.release, real.version, real.machine)),
    )


def test_a_set_label_is_used_as_given(monkeypatch):
    uname_as(monkeypatch, "workstation-17")
    monkeypatch.setenv("BENCH_HOST", "cloud-vm")
    assert bench_host.host_label() == "cloud-vm"


def test_an_unlabelled_machine_is_not_named(monkeypatch):
    monkeypatch.delenv("BENCH_HOST", raising=False)
    uname_as(monkeypatch, "workstation-17")
    assert bench_host.host_label() == "unlabelled"
    monkeypatch.setenv("BENCH_HOST", "  ")
    assert bench_host.host_label() == "unlabelled"


def test_the_bench_namespace_keeps_its_name(monkeypatch):
    monkeypatch.delenv("BENCH_HOST", raising=False)
    uname_as(monkeypatch, "bench")
    assert bench_host.host_label() == "bench"


def test_no_harness_script_reads_the_machine_name():
    for path in sorted(HARNESS.glob("*.py")):
        if path.name == "bench_host.py":
            continue
        text = path.read_text()
        assert "nodename" not in text and "gethostname" not in text, path.name
        assert '"hostname"' not in text, path.name
