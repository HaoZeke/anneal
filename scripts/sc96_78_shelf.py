# Copyright (c) anneal developers
"""One Sutton-Chen 9-6 N=78 chain, with and without surface hops.

The start is the file stored with the tutorial. The Cambridge table
supplies the stop energy. ``data/sc96_78_shelf.csv`` records the run,
and the tutorial's printed blocks and count sentences are written from it.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sc96_78_common import binary, parse_run, reference_energy  # noqa: E402

START = ROOT / "docs" / "orgmode" / "tutorials" / "starts" / "sc96-78-seed20003.xyz"
PAGE = ROOT / "docs" / "orgmode" / "tutorials" / "sutton-chen-shelf.org"
OUT = ROOT / "data" / "sc96_78_shelf.csv"
SCRATCH = ROOT / "build" / "sc-shelf"
FIGURE = SCRATCH / "coordination.png"
BUDGET = "2000000"
SEED = "20003"
N = "78"


def run_arm(exe: Path, arm: str, reference: str) -> tuple[dict[str, str], str]:
    dest = SCRATCH / f"{arm}.out"
    if dest.is_file() and "hit: " in dest.read_text():
        text = dest.read_text()
        return parse_run(text), text
    db = SCRATCH / f"db-{arm}"
    env = os.environ.copy()
    env["ANNEAL_MINIMA_DB"] = str(db)
    env["RAYON_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    text = subprocess.check_output(
        [str(exe), arm, "9-6", N, BUDGET, SEED, str(START), reference],
        cwd=ROOT,
        env=env,
        text=True,
    )
    dest.write_text(text)
    return parse_run(text), text


def minima_table(reader: Path, parsed: dict[str, str], xyz: Path) -> str:
    xyz.mkdir(parents=True, exist_ok=True)
    return subprocess.check_output(
        [
            str(reader),
            str(SCRATCH / f"db-{parsed['arm']}"),
            parsed["minima_signature"],
            f"sc9-6-{N}",
            parsed["temperature"],
            parsed["length_scale"],
            "3",
            str(xyz),
        ],
        cwd=ROOT,
        text=True,
    )


def coord(spec: str) -> dict[int, int]:
    counts = {}
    for item in spec.split(","):
        if not item:
            continue
        left, right = item.split(":")
        counts[int(left)] = int(right)
    return counts


def first_minimum(table: str) -> list[str]:
    lines = [line for line in table.splitlines() if line[:1].isdigit() or line.startswith("-")]
    if not lines:
        raise SystemExit("sc_minima printed no minimum")
    return lines[0].split()


def write_csv(rows: list[tuple[str, str, str]]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for path in (OUT, SCRATCH / "shelf.csv"):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["arm", "key", "value"])
            writer.writerows(rows)


def replace_block(page: str, marker: str, body: str) -> str:
    start = page.index(marker)
    begin = page.rindex("#+begin_example", 0, start)
    end = page.index("#+end_example", start)
    return page[:begin] + "#+begin_example\n" + body.rstrip("\n") + "\n#+end_example" + page[end + len("#+end_example") :]


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    reference = f"{reference_energy(SCRATCH):.4f}"
    runner = binary("sc_h2h")
    reader = binary("sc_minima")
    iso, _iso_text = run_arm(runner, "iso", reference)
    hop, _hop_text = run_arm(runner, "iso-hop", reference)
    iso_table = minima_table(reader, iso, SCRATCH / "xyz-iso")
    hop_table = minima_table(reader, hop, SCRATCH / "xyz-hop")
    iso_line = first_minimum(iso_table)
    hop_line = first_minimum(hop_table)
    iso_z = coord(iso_line[-1])
    hop_z = coord(hop_line[-1])
    gap = float(reference) - float(iso["best"])
    rows = []
    for arm, parsed, table in (("iso", iso, iso_table), ("iso-hop", hop, hop_table)):
        for key in (
            "spent",
            "best",
            "hit",
            "hit_calls",
            "surface_hops",
            "surface_descents",
            "minima_signature",
            "temperature",
            "length_scale",
        ):
            rows.append((arm, key, parsed.get(key, "")))
        rows.append((arm, "table", table.rstrip("\n")))
    rows.append(("iso", "gap_above_reference", f"{gap:.6f}"))
    write_csv(rows)

    iso_block = "\n".join(f"{key}: {iso[key]}" for key in ("spent", "best", "hit"))
    hop_head = "\n".join(
        f"{key}: {hop[key]}" for key in ("surface_hops", "surface_descents", "hit", "hit_calls")
    )
    distinct = iso_table.split("distinct: ", 1)[1].splitlines()[0]
    hop_distinct = hop_table.split("distinct: ", 1)[1].splitlines()[0]
    length = hop["length_scale"]
    if iso["hit"] == "true":
        iso_outcome = f"The chain reached the global minimum after {iso['spent']} calls.\n"
    else:
        iso_outcome = (
            f"The chain spent {iso['spent']} calls and stopped {gap:.6f} above the reference.\n"
        )
    page = f"""#+TITLE: One Sutton-Chen chain on the stored 78-atom start
#+OPTIONS: toc:nil todo:nil num:nil author:nil title:nil ^:nil

#+begin_export rst
=================================================
One Sutton-Chen chain on the stored 78-atom start
=================================================
#+end_export

This page runs one 9-6 N=78 chain from the stored start, then the
iso-hop arm on the same start.
=scripts/sc96_78_shelf.py= writes =data/sc96_78_shelf.csv= and the blocks below.

The Cambridge Cluster Database lists the global minimum at {reference}.
The search uses that number only to stop.

* Build the two programs

#+begin_src bash
cargo build --release --example sc_h2h --example sc_minima
#+end_src

The start is the file stored with this page.

#+begin_src bash
START=docs/orgmode/tutorials/starts/sc96-78-seed20003.xyz
SIG={iso['minima_signature']}
#+end_src

=SIG= is the digest the runner prints as =minima_signature= for this row and size.

* Record the minima of one run

Start with =iso=, one chain of arm 3.
=ANNEAL_MINIMA_DB= names the directory that receives every minimum it finds.

#+begin_src bash
ANNEAL_MINIMA_DB=db cargo run -q --release --example sc_h2h -- \\
  iso 9-6 78 {BUDGET} {SEED} $START {reference} > iso.out
grep -E '^(spent|best|hit):' iso.out
#+end_src

#+begin_example
{iso_block}
#+end_example

{iso_outcome}
* List the lowest minima

=sc_minima= groups them by energy.
Pass it the temperature and the length scale the run printed.

#+begin_src bash
cargo run -q --release --example sc_minima -- db $SIG sc9-6-78 {iso['temperature']} {length} 3 xyz
#+end_src

#+begin_example
{iso_table.rstrip()}
#+end_example

The chain found {distinct} distinct minima.
The columns 555, 421 and 422 count bonded pairs by their common-neighbour signature.
The lowest minimum has {iso_line[2]} bonds, {iso_z.get(12, 0)} twelve-coordinated atoms, {iso_z.get(5, 0)} five-coordinated atoms and {iso_z.get(4, 0)} four-coordinated atoms.

* Run the iso-hop arm

=iso-hop= sets =Config::surface_hop_on_best=. The legacy surface-hop
counters stay at zero and the arm matches =iso= on the shelf fixture
(=tests/surface_hop.rs=, =surface_hops_run_from_new_bests_only_when_asked=).

#+begin_src bash
ANNEAL_MINIMA_DB=db-hop cargo run -q --release --example sc_h2h -- \\
  iso-hop 9-6 78 {BUDGET} {SEED} $START {reference} > hop.out
grep -E '^(surface_hops|surface_descents|hit|hit_calls):' hop.out
cargo run -q --release --example sc_minima -- db-hop $SIG sc9-6-78 {hop['temperature']} {length} 3 xyz-hop
#+end_src

#+begin_example
{hop_head}
{hop_table.rstrip()}
#+end_example

The iso-hop chain hit: {hop['hit']}, after {hop['hit_calls']} calls.
=surface_hops= and =surface_descents= stayed at {hop['surface_hops']} and
{hop['surface_descents']}; the lowest minimum matched =iso=.
It found {hop_distinct} distinct minima.
Its lowest minimum has {hop_line[2]} bonds, {hop_z.get(12, 0)} twelve-coordinated atoms, {hop_z.get(5, 0)} five-coordinated atoms and {hop_z.get(4, 0)} four-coordinated atoms.

* Compare the two lowest structures

#+begin_src bash
python3 docs/scripts/plot_sc_coordination.py {length} coordination.png \\
  "iso lowest, {iso_line[0]}=xyz/sc9-6-78-00.xyz" \\
  "iso-hop lowest, {hop_line[0]}=xyz-hop/sc9-6-78-00.xyz"
#+end_src

#+CAPTION: Neighbor counts of the lowest minimum of each arm. The script writes the figure to =build/sc-shelf/coordination.png=.

=tests/surface_hop.rs= checks the separate shelf fixture.
The flag is =Config::surface_hop_on_best=, described in the [[file:../methods/catalog.org][method catalog]].
"""
    PAGE.write_text(page)
    plot = ROOT / "docs" / "scripts" / "plot_sc_coordination.py"
    iso_xyz = sorted((SCRATCH / "xyz-iso").glob("*.xyz"))[0]
    hop_xyz = sorted((SCRATCH / "xyz-hop").glob("*.xyz"))[0]
    subprocess.check_call(
        [
            sys.executable,
            str(plot),
            length,
            str(FIGURE),
            f"iso lowest, {iso_line[0]}={iso_xyz}",
            f"iso-hop lowest, {hop_line[0]}={hop_xyz}",
        ],
        cwd=ROOT,
    )
    print(iso_block)
    print(hop_head)
    print("gap", f"{gap:.6f}", "distinct", distinct)


if __name__ == "__main__":
    main()
