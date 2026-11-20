# Copyright (c) anneal developers
"""The label a benchmark row carries for the machine that made it."""

import os


def host_label() -> str:
    """``BENCH_HOST`` when set, ``bench`` inside the bench namespace, else ``unlabelled``.

    The machine's own hostname is never written into a row. The bench
    namespace sets the hostname ``bench`` by construction, so that name is
    the one the label may echo; any other hostname is a machine's name, not
    a property of the run.
    """
    label = os.environ.get("BENCH_HOST", "").strip()
    if label:
        return label
    return "bench" if os.uname().nodename == "bench" else "unlabelled"
