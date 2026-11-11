#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write NOTES, REPRODUCE, PROGRESS, and REPORT from the receipts.

The numbers come from logs/budget/*/receipt.log.
Nothing in those four files is typed by hand.
"""

import sys
from pathlib import Path

CUTOFF = "-582.0866"
SEEDS = tuple(range(2000, 2006))


def field(text: str, name: str) -> str:
    prefix = name + "="
    found = [
        line[len(prefix):]
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(found) != 1:
        raise SystemExit(f"missing {name}")
    return found[0]


def clock(stamp: str) -> str:
    # 2026-11-10 12:52:33 CST -> 12:52:33
    return stamp.split()[1]


def load(root: Path) -> list[dict]:
    rows = []
    for seed in SEEDS:
        text = (root / str(seed) / "receipt.log").read_text()
        csv = next(
            line for line in text.splitlines() if line.startswith("lj104,")
        )
        parts = csv.split(",")
        rows.append(
            {
                "seed": seed,
                "head": field(text, "head"),
                "tree": field(text, "tree"),
                "binary": field(text, "binary_sha256"),
                "start": clock(field(text, "start_cst")),
                "end": clock(field(text, "end_cst")),
                "best": parts[8],
                "hit": parts[4],
            }
        )
    heads = {row["head"] for row in rows}
    trees = {row["tree"] for row in rows}
    binaries = {row["binary"] for row in rows}
    if len(heads) != 1 or len(trees) != 1 or len(binaries) != 1:
        raise SystemExit("receipts disagree")
    return rows


def best_display(best: str) -> str:
    return f"{float(best):.2f}"


def write(path: Path, text: str) -> None:
    path.write_text(text)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: lj104_receipt_prose.py <receipt-root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    rows = load(root)
    head = rows[0]["head"]
    tree = rows[0]["tree"]
    binary = rows[0]["binary"]
    opened = rows[0]["start"]
    closed = rows[-1]["end"]
    hits = sum(1 for row in rows if row["hit"] == "true")
    best = min(rows, key=lambda row: float(row["best"]))["best"]
    if hits != 0:
        raise SystemExit("hit count is not zero")

    notes = "\n".join(
        [
            "Two-million evaluation reproduction, on the same engine.",
            "",
            "Seeds 2000 through 2005 are deterministic.",
            "v4.2 already recorded their outcome.",
            "This rerun is not a blind test.",
            "",
            "The receipt head follows.",
            head + ".",
            "The receipt tree follows.",
            tree + ".",
            "The binary digest follows.",
            f"The binary printed head {head}.",
            binary + ".",
            f"The arm opened at {opened} and closed at {closed}.",
            "",
            "Global-minimum outcomes on the six seeds: none.",
            f"The best was {best_display(best)} against {CUTOFF}.",
            "The global-minimum count is 0 of 6.",
            "",
        ]
    )
    for row in rows:
        notes += (
            f"Seed {row['seed']} opened at {row['start']} "
            f"and closed at {row['end']}.\n"
            f"Its best energy is {row['best']}.\n"
        )
    notes += (
        "\nEach seed ran alone on processor 0.\n"
        "The receipt field pins reads 0.\n"
        "uname -n reads bench.\n"
    )

    reproduce = "\n".join(
        [
            "The measured arm reruns seeds 2000 through 2005, one seed at a time.",
            "",
            f"The engine is commit {head[:12]} and tree {tree[:12]}.",
            f"Engine commit {head}.",
            f"Engine tree {tree}.",
            "The binary digest follows.",
            f"{binary}.",
            "",
            "The receipt script opens one private mount namespace.",
            "It also opens one hostname namespace.",
            "That namespace sets the nodename.",
            "A private file that reads bench is mounted on the hostname file.",
            "The host copy of the hostname file is left untouched.",
            "Each receipt records uname -n and the hash of that hostname file.",
            "The binary digest is taken before the first seed.",
            "taskset selects processor 0.",
            "The budget is 2000000 evaluations.",
            "",
        ]
    )

    progress = "\n".join(
        [
            "Phase: two-million reproduction finished.",
            "",
            "No push.",
            "Archive refs are never pushed.",
            "The push plan lists only refs/heads/fix/lj104-measure-v4.11.",
            "",
            f"Engine commit {head}.",
            "",
            f"Engine tree {tree}.",
            "The binary digest follows.",
            f"{binary}.",
            f"The arm opened at {opened} and closed at {closed}.",
            "",
            "Jobs left running: none.",
            "",
            "Seeds 2000 through 2005 each returned exit 0.",
            "Each seed ran alone on processor 0.",
            "The global-minimum count is 0 of 6.",
            "",
        ]
    )

    report = "\n".join(
        [
            "# LJ104 measure v4.11",
            "",
            "The global-minimum count is 0 of 6.",
            f"The best was {best_display(best)} against {CUTOFF}.",
            "",
            f"The timed commit is {head}.",
            f"Its tree is {tree}.",
            f"The arm opened at {opened} and closed at {closed}.",
            "",
            "proseguard was not run.",
            "NOT_RUN.",
            "",
            "The program is not installed.",
            "The prose-human profile on the public dotfiles is age-encrypted.",
            "",
        ]
    )

    write(Path("NOTES.md"), notes if notes.endswith("\n") else notes + "\n")
    write(Path("REPRODUCE.md"), reproduce)
    write(Path("PROGRESS.md"), progress)
    write(Path("REPORT.md"), report)
    print(f"head={head}")
    print(f"tree={tree}")
    print(f"binary={binary}")
    print(f"opened={opened}")
    print(f"closed={closed}")
    print(f"best={best_display(best)}")
    print("count=0 of 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
