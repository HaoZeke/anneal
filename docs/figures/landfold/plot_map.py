"""Draw the landfold map of the LJ38 minima book for the how-to.

The script writes the LJ38 book with the ``landfold_lj_book`` example,
folds it with ``anneal landfold --seed 0`` and plots ``x`` against ``y``,
coloured by energy. The PNG goes next to this script.

Usage, from the repository root:

    cargo build --release --bin anneal --example landfold_lj_book
    python docs/figures/landfold/plot_map.py

The binaries are read from ``$CARGO_TARGET_DIR/release``, or from
``target/release`` when that variable is unset. The PNG carries no
timestamp or version text, so a rerun with the same matplotlib writes the
same bytes.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from pathlib import Path

from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# The book of the landfold benchmark: 2000 distinct minima from seed 38,
# with chains started from the fcc and icosahedral funnel bottoms.
ATOMS = 38
MINIMA = 2000
BOOK_SEED = 38
FIXTURES = ("lj38_fcc.xyz", "lj38_ico.xyz")
MAP_SEED = 0


def release_dir() -> Path:
    target = os.environ.get("CARGO_TARGET_DIR")
    return (Path(target) if target else ROOT / "target") / "release"


def fold(work: Path) -> list[dict[str, str]]:
    release = release_dir()
    book = work / "lj38.min"
    table = work / "lj38.csv"
    subprocess.run(
        [
            str(release / "examples" / "landfold_lj_book"),
            str(ATOMS),
            str(MINIMA),
            str(BOOK_SEED),
            str(book),
            *(str(ROOT / "tests" / "fixtures" / name) for name in FIXTURES),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(release / "anneal"),
            "landfold",
            str(book),
            "--seed",
            str(MAP_SEED),
            "--out",
            str(table),
        ],
        check=True,
    )
    with table.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot(rows: list[dict[str, str]], out: Path) -> None:
    # Draw high energies first so the low minima sit on top.
    rows = sorted(rows, key=lambda row: float(row["energy"]), reverse=True)
    x = [float(row["x"]) for row in rows]
    y = [float(row["y"]) for row in rows]
    energy = [float(row["energy"]) for row in rows]
    lowest = rows[-1]

    fig = Figure(figsize=(6.0, 5.0), dpi=150)
    ax = fig.subplots()
    points = ax.scatter(x, y, c=energy, s=6, cmap="viridis", linewidths=0)
    ax.scatter(
        [float(lowest["x"])],
        [float(lowest["y"])],
        marker="*",
        s=160,
        facecolors="none",
        edgecolors="crimson",
        linewidths=1.2,
        label=f"lowest minimum, E = {float(lowest['energy']):.6f}",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"LJ{ATOMS}, {len(rows)} minima, anneal landfold --seed {MAP_SEED}")
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.colorbar(points, ax=ax, label="energy (reduced units)")
    fig.tight_layout()
    fig.savefig(out, metadata={"Software": None})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "lj38-map.png")
    parser.add_argument(
        "--keep",
        type=Path,
        help="write the book and the map CSV here instead of a temporary directory",
    )
    args = parser.parse_args()
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)
        rows = fold(args.keep)
    else:
        with tempfile.TemporaryDirectory() as work:
            rows = fold(Path(work))
    plot(rows, args.out)


if __name__ == "__main__":
    main()
