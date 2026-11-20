# Lennard-Jones GMIN data files

One rule covers every Lennard-Jones size. No keyword is chosen for one
size and reused for the others, and no keyword is edited after a run.

The keyword lines of `examples/h2h_gmin_lj.data` are copied in order,
unchanged. The only keyword added is `RADIUS`. The driver appends
`RANSEED` for each seed. That line is not part of the data file.

`RADIUS` is the container GMIN assigns for the default potential when
the keyword is absent. In `source/io1.f90` at GMIN revision
`9cbec1a446352bc9ef064bd2b1a5767194655919`, an unset radius is

    2 + (3 * N / 17.77153175) ** (1/3)

with `N` the number of atoms. The same block then multiplies by
`2 ** (1/6)` on the default-potential branch (`CASE DEFAULT`). A data
file that names no potential takes that branch. Morse is a named case
and is outside this rule. An explicit `RADIUS` skips the block, so the
value written here already includes `2 ** (1/6)`. The literal
`17.77153175` is the one in that source.

The keyword is printed as `RADIUS` and that value in decimal with ten
digits after the point, half-even rounding.

`N` is the atom count of the size: 38, 55, 75, 98, 102, 103, 104, 150.
The generator is `scripts/gmin_panel/lj_datafile.py`. It writes
`bench/gmin-panel/lj-data/<system>.data`.
