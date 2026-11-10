#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Sum Etot lines from crestopt.log, including replacements of that file."""

import time
from pathlib import Path

ROOT = Path("paper/jctc/logs/scratch/water")
OUT = Path("paper/jctc/logs/water")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {}
    while True:
        if ROOT.exists():
            for folder in ROOT.glob("crest-seed*"):
                log = folder / "crestopt.log"
                if not log.exists():
                    continue
                text = log.read_text(errors="ignore")
                count = text.count("Etot=")
                head = text[:400]
                slot = state.setdefault(folder.name, {"last": 0, "total": 0, "head": None})
                if slot["head"] is None:
                    slot["head"] = head
                    slot["last"] = count
                    slot["total"] = count
                elif head != slot["head"]:
                    slot["total"] += count
                    slot["last"] = count
                    slot["head"] = head
                elif count > slot["last"]:
                    slot["total"] += count - slot["last"]
                    slot["last"] = count
                (OUT / f"{folder.name}-optsteps.txt").write_text(
                    f"opt_steps={slot['total']}\nlast_file={slot['last']}\n"
                )
        time.sleep(0.2)


if __name__ == "__main__":
    main()
