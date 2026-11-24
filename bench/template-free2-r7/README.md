# template-free2 r7: m55r14 noise check and the cap turn

Host `machine-9eb296774121518f`, Intel Xeon, serial on core 0 through
`taskset -c 0`, one thread per process. Budgets are the drivers' paper
budgets: 400k for lj38 and m38r14, 1M for m55r14, 4M for lj75 and
lj98 to lj104. A hit is `scripts/target_tol.py::energy_hits` against
the reference in `run_cells.py`.

## 1. The m55r14 loss from 9 to 8 hits is noise

r4 and r5 saw m55r14 fall from 9/12 to 8/12 hits on seeds 2000-2011
with the template-free2 escape on (develop A against S1). Rerun on 24
fresh seeds, 2012-2035, with develop 9837358a (arm `A`) against the
r4/S1 behaviour at 795737db (arm `T`):

    python3 run_cells.py m55r14-fresh-rows.csv m55r14 2012:2035 \
        A=<develop examples>=9837358a... T=<795737db examples>=795737db...
    python3 summarise.py m55r14-fresh-rows.csv A T

| seeds | A hits | T hits | Fisher p | A only | T only | McNemar p |
|---|---|---|---|---|---|---|
| 2000-2011 (r5 record) | 9/12 | 8/12 | 1.0 | 3 | 2 | 1.0 |
| 2012-2035 (fresh) | 18/24 | 22/24 | 0.245 | 2 | 6 | 0.289 |

The direction reverses on fresh seeds. Neither split is significant,
so there is no mechanism to fix. Rows in `m55r14-fresh-rows.csv`, logs
in `raw/m55r14-fresh/`.

795737db3e80a7f00d1b05a47f9532c7ffa38d98 is the tip of the record
branch `record/template-free2-r4-escape`, eight commits on develop
61fa67cf380e46876b569da45e7cc091647a775b, so arm `T` resolves once
that branch is on the remote.

## 2. The cap turn

`src/cap_twist.rs` builds the proposal from the incumbent alone. A
pivot is drawn among atoms with at least median coordination. Its
first shell is tested under rotations of order 3 to 6 about axes
through shell points, bonded pair midpoints and bonded triangle
centroids; the best self-map gives the local axis and its order k.
Everything beyond half a layer from the pivot along the axis turns by
pi/k. About a five-fold axis that is the icosahedral to decahedral
twist; about a three-fold axis the fcc to hcp stacking change. There
are no size constants, stored structures or per-system switches.

`Config::cap_twist` (off by default) makes it a stall arm and, on a
call-plateau leave, lets a learned allocator draw between the
soft-mode climb and the cap turn, scored by the energy the leave gains.
The drivers turn it on with `H2H_CAP_TWIST=1`.

### Campaign

Base is 4d2fa0f4 (develop 9837358a plus r6 commits 1-2). Arm `CT` ran
with `H2H_CAP_TWIST=1` from a build whose revision is unrecorded. Its
rows cite 5445c4f7, but no branch or audit ref holds that commit, the
run logs record the head as `unknown`, and `run_cells.py` writes
whatever revision it is given on the command line. aa10a14c, the commit
that added these rows, sits on 403c3d00, which is the cap turn on
91732899. That base carries source changes that 4d2fa0f4 does not, so
403c3d00 is not shown to be the build either. The rows stay as written,
and `scripts/paper_audit.py` lists 5445c4f7 as an unpushed revision.
Seeds 2000-2023, arm order rotating with the seed.

    python3 run_cells.py rows.csv lj38,m38r14,m55r14,lj75,lj98,lj102,lj103,lj104 \
        2000:2023 base=<4d2fa0f4 examples>=4d2fa0f4... \
        CT=<5445c4f7 examples>=5445c4f7...=H2H_CAP_TWIST:1
    python3 summarise.py rows.csv base CT concurrent.csv

| cell | base | CT | Fisher p | base only | CT only | outside/call ratio |
|---|---|---|---|---|---|---|
| lj38 | 23/24 | 23/24 | 1 | 1 | 1 | 0.998 |
| m38r14 | 11/24 | 19/24 | 0.036 | 1 | 9 | 0.959 |
| m55r14 | 18/24 | 18/24 | 1 | 5 | 5 | 0.993 |
| lj75 | 7/24 | 12/24 | 0.24 | 3 | 8 | 1.003 |
| lj98 | 16/24 | 24/24 | 0.0039 | 0 | 8 | 1.058 |
| lj102 | 11/24 | 18/24 | 0.075 | 3 | 10 | 1.016 |
| lj103 | 13/24 | 18/24 | 0.23 | 5 | 10 | 0.968 |
| lj104 | 11/24 | 16/24 | 0.24 | 4 | 9 | 0.994 |
| all | 110/192 | 148/192 | | 22 | 60 | |

The table describes the unrecorded CT build against 4d2fa0f4 and
nothing later. No McNemar p is given for the cap turn: the branch head
has changed the stall draw since (a turn that does not move the state is
now redrawn, and the leave and stall allocators score the arm that
acts), and no rows have been run at that head, so no p value here
measures it.
The Fisher p values compare the same two builds and carry the same
limit.

The outside ratio is the type-7 median over seeds of
(wall_s - pot_s) / charged, CT over base. No cell loses a hit and the
largest ratio is 1.058 (lj98), so that build met the default-on bar (no
hit loss, outside at most 1.10x). The cap stays opt-in in this series.

No row shows a pause: every row's elapsed time is within 10 % and a
second of its wall time. Two short side jobs ran on cores 2-7 while the
campaign held core 0: the draw-order check below (11:25-11:32 UTC) and
the flag-off identity check (12:36-12:52 UTC). Hits do not depend on
timing, but the 13 seeds whose runs overlapped those windows, listed in
`concurrent.csv`, are left out of the timing for lj75 and lj102. With
them in, the ratios are 0.995 and 1.018.

### Flag off is the base

The CT binaries, cited as 5445c4f7, were run without the variable on
all 24 seeds of lj38, m38r14, m55r14, lj75 and lj98 and on seeds
2000-2005 of lj102 to lj104. All 138 runs match the base rows on
budget, hit, calls to hit, charged calls and best energy
(`flag-off-rows.csv`, `raw/flag-off/`).

### Draw order and the tie rule

A development screen of the same design drew the cap's side and sense
after the axis was found, not before, and had no tie rule. It hit lj75
on 19 of the 23 seeds it finished (`raw/screen-leave/`; one run was
cut off). The committed code draws pivot, side and sense up front and
prefers the higher order on a tie within 1e-9. The committed code with
a strict tie reproduces the campaign's CT row on all 24 lj75 seeds, so
the tie rule never changed a trajectory here; the screen binary
reproduces its own screen rows (`rng-order-rows.csv`). The gap from 19
to 12 lj75 hits is between two random streams of one method.

### What did not work

A first version put the cap turn among the move kernels as a regular
proposal. On lj38 it fell from 23/24 to 20/24 hits and the median calls
to hit rose from 117832 to 186746, and lj75 stayed at 7/24
(`raw/screen-kernel/`, `-on` against `-off`). A second version made it
only a stall arm; under the recommended preset stalls leave through the
call plateau, which always climbed, so that arm never fired and the
trajectories were unchanged. The committed version draws on the call
plateau leave.

### Single-shot probe

Before the in-search runs, `raw/probe/reach7.rs.txt`, an example
harness against the draft module, drew 100 cap
turns from each dumped stall incumbent and quenched them once. None
reached the global minimum (0 of 700 on lj38, 0 of 1900 on lj75, 0 of
100 on m55r6), but 342, 733 and 30 landed below the incumbent. The cap
turn does not jump to the target; it opens a new funnel that the
hopping then descends.
