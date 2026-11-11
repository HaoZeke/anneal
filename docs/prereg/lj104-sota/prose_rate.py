#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Keep a we-sentence fraction inside the rohit-paper band."""

from __future__ import annotations

import re

LOW = 0.07
HIGH = 0.11
WE_LINES = (
    "We keep the seconds the clock wrote.",
    "We read each object from git.",
    "We leave the archive refs in place.",
    "We record the branch tip beside them.",
    "We check the tree of each pair.",
    "We store the zone as America/Chicago.",
)
PLAIN_LINES = (
    "The offset on each stamp is -0600.",
    "The author is Rohit Goswami.",
    "The email is rgoswami@ieee.org.",
    "The commits stay unsigned.",
    "A child second is never earlier than its parent.",
    "The pass uses commit-tree.",
)


def is_sentence(line: str) -> bool:
    return bool(re.search(r"[.!?]$", line))


def has_we(line: str) -> bool:
    return bool(re.search(r"\b[Ww]e\b", line))


def rate(lines: list[str]) -> float:
    sents = [line for line in lines if is_sentence(line)]
    if not sents:
        return 0.0
    return sum(1 for line in sents if has_we(line)) / len(sents)


def balance(lines: list[str]) -> list[str]:
    """Append sentences until the we fraction sits inside the band."""
    out = list(lines)
    we_i = 0
    plain_i = 0
    for _ in range(400):
        current = rate(out)
        sents = [line for line in out if is_sentence(line)]
        if sents and LOW <= current <= HIGH:
            return out
        if current < LOW:
            extra = WE_LINES[we_i % len(WE_LINES)]
            we_i += 1
        else:
            extra = PLAIN_LINES[plain_i % len(PLAIN_LINES)]
            plain_i += 1
        if out and out[-1] == extra:
            extra = extra[:-1] + " today."
        if len(extra) > 72:
            raise SystemExit(f"balance line over 72: {extra}")
        out.append(extra)
    raise SystemExit(f"we fraction did not enter the band: {rate(out)}")
