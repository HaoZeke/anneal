# Copyright (c) anneal developers
"""CREST's true cost is the energy+grad line, not the driver count."""

import re

ENGRAD_LINE = re.compile(r"Total number of energy\+grad calls:\s*(\d+)")


def engrad_total(text):
    """The integer on CREST's `Total number of energy+grad calls` line."""
    found = ENGRAD_LINE.findall(text)
    if len(found) != 1:
        raise ValueError("crest log does not contain one energy+grad total")
    return int(found[0])


def keyed_int(text, name):
    match = re.search(rf"(?m)^{re.escape(name)}=(\d+)\s*$", text)
    if not match:
        raise ValueError(f"crest log has no {name}")
    return int(match.group(1))


def driver_calls(text):
    return keyed_int(text, "calls")


def dynamics_plus_opt(text):
    return keyed_int(text, "dynamics_steps") + keyed_int(text, "opt_steps")


def require_split(text):
    """Return (driver count, engrad total) after the driver identity check."""
    driver = driver_calls(text)
    parts = dynamics_plus_opt(text)
    total = engrad_total(text)
    if driver != parts:
        raise ValueError(f"driver calls {driver} are not dynamics plus opt {parts}")
    if total <= driver:
        raise ValueError(f"engrad total {total} is not above the driver count {driver}")
    return driver, total
