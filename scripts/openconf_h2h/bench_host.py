# Copyright (c) anneal developers
"""The label a benchmark row carries for the machine that made it."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import machine_provenance  # noqa: E402


def host_label() -> str:
    """``BENCH_HOST`` when set, ``bench`` inside the bench namespace, else ``unlabelled``.

    The machine's own hostname is never written into a row. The bench
    namespace sets the hostname ``bench`` by construction, so that name is
    the one the label may echo; any other hostname is a machine's name, not
    a property of the run. The name is passed only to
    ``machine_provenance.host_label``.
    """
    label = os.environ.get("BENCH_HOST", "").strip()
    if label:
        return label
    if machine_provenance.host_label(os.uname().nodename) == "bench":
        return "bench"
    return "unlabelled"


def run_host(info: dict) -> str:
    """The host label a ``job.json`` recorded.

    A job written before the label was recorded has no ``host`` key; its row
    reads ``unlabelled``. The machine name such a file may hold under another
    key is never copied into a row.
    """
    return str(info.get("host") or "").strip() or "unlabelled"
