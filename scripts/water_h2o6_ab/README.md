# Water hexamer code-only A/B

This directory holds the harness for a code-only comparison of two builds of
the annealing search on the water hexamer panel: A, the code at commit
`36086e988c5a71c156ffcbde152bf321a6f5ef25`, and B, the code at commit
`bc16d35cf13e71c2edaf343b80fd8c41368631d4`. The two builds run the four arms of
the panel on the same GFN2 engine, one thread on one pinned core, strictly
serially, seeds 0 to 191, in a single session. For every seed and arm the two
builds run back to back, and the order of the pair alternates with the parity
of the seed.

Nothing in the search code of either build is changed. The only edits are an
input layer on A, described below, and a shim between the builds and the
engine.

## Files

| file | role |
| --- | --- |
| `make_inputs.py` | regenerates the start geometry from the template of the catalog source and reorders the three references so that every water is O, H, H |
| `engine_shim.c` | library loaded in place of the engine; forwards to the real engine, counts calls and the time spent inside the force call, and can flip the sign of the buffer |
| `crest_driver.c` | call server and per-call client that give CREST the same engine and the same call budget |
| `polish.py` | L-BFGS polish of one structure within a fixed number of engine calls |
| `run_ab.py` | the orchestrator; writes `runs.csv`, `crest_frames.csv`, `inputs.csv`, `meta.json` |
| `analyze.py` | tables, tests and sensitivity checks from the two CSV files |
| `controls.py`, `analyze_controls.py` | controls that rerun only the CREST arm from other start geometries (displaced, packed at random, the isomers themselves), on another engine, and against other reference orderings, and the tables of their results |
| `sign_check.py`, `preflight.py` | finite-difference check of the buffer sign and the short runs that show why the shim is needed |
| `a_input_layer.patch` | the input-layer edit of the example driver of A |

## Builds

Both builds are the `water_hexamer` example in release mode with the features
`rgpot-ex`, `featomic`, `bank-rpc` and `ira`. The IRA library directory is
given to the build script and added to the run path of the binary, so the two
binaries load the same IRA library. Build B from its commit as it is. Build A
from its commit after applying `a_input_layer.patch`.

`a_input_layer.patch` touches the input and output layer of the example driver
of A only: it lets the budget and the seed range be set per process, as B
already does, and it writes the best structure of every seed to an xyz file in
the same format as B. The search, the kernels and the scoring of A are
untouched.

## Engine

The engine is the rgpot GFN2-xTB library (method 3, accuracy 0.01, electronic
temperature 300 K, 500 iterations, neutral, closed shell, vacuum box of 60 A)
linked against xtb 6.7.1. The two builds are given the shim in its place.

The buffer the engine returns is the force, minus dE/dx; `sign_check.py` checks
this against central finite differences. A publishes it as a force. B negates
it, because it takes the buffer for dE/dx, and with the real engine it walks
uphill and never leaves a dissociated state. The shim therefore hands A the
buffer unchanged and hands B the negated buffer. `preflight.py` prints the
finite-difference table for the engine and for both shim modes, and one short
seed of each build with each buffer.

## Inputs

All inputs are command-line arguments of `run_ab.py`; there are no defaults.

* `make_inputs.py` writes the start geometry, six waters on the template of the
  catalog source, and the three reference isomers (prism, cage, book) reordered
  O, H, H. The hashes of every input, the binaries, the engine, the shim, the
  driver and CREST are written to `inputs.csv` by `run_ab.py`.
* Both builds read the same start geometry and the same reference files. A has
  no start-geometry option, but its built-in template is the same six waters as
  the generated file; B is given the file.

## Arms

| arm | budget | answer |
| --- | --- | --- |
| `anneal` | 4000 engine calls | best structure of the search |
| `polish_fair` | 3382 calls of search, then at most 618 calls of L-BFGS | polished best structure |
| `crest` | CREST default flow stopped after 4000 engine calls | the metadynamics frames |
| `crest_fair` | CREST stopped after 3382 calls, then at most 618 calls of L-BFGS on its lowest-energy frame | the frames and the polished structure |

The polish is `scipy` L-BFGS-B with history 10 and a gradient tolerance of
5e-4 eV/A, started cold, cut off by exception at 618 evaluations; the answer is
the lowest-energy point it evaluated. It is applied by the harness on top of
whatever native polish a build has.

CREST runs with one thread and its default flow (initial optimization, trial
metadynamics, then the metadynamics runs of the standard flow) with the engine
reached through the generic calculator. The call server counts every request
and refuses requests beyond the budget; CREST then ends its run by itself. The
frames scored are the structures of the metadynamics trajectory, 180 of them
for a budget of 4000 calls. CREST is not seeded, so its runs under A and under
B are independent draws; only the scoring differs between them.

## Hit criterion

A structure is a hit when its monomers are intact and the species-aware IRA
match against the reference isomers names an isomer inside the gate. The gate
is half the smallest IRA RMSD between two references, computed by the example
drivers before the first seed. A frame fails the match when IRA cannot align it
with every reference; such a frame is not assigned. A set of CREST frames is a
hit when any frame is a hit; for `crest_fair` the polished structure counts as
well. Each build scores with its own scoring code. The other build scores the
same structures too, in the columns marked cross, so that an effect of the
scoring code can be told apart from an effect of the search. For every CREST
frame the IRA RMSD against each single reference is also recorded.

## Time

Wall time is measured by the harness with a monotonic clock around the whole
process, or, for CREST, from the start of the process to the answer of the last
engine call within the budget. Time inside the potential is the time spent in
the force call of the engine, measured by the shim. Time outside is the
difference. The orchestrator is pinned with a CPU affinity mask that its
children inherit, thread-count variables are fixed to one, and nothing else
runs in the session.

## Running

```
python run_ab.py --binary-a <build A> --binary-b <build B> --engine <engine library> \
    --shim <shim library> --driver <driver binary> --crest <crest binary> \
    --water-xyz <start geometry> --references-dir <directory with prism.xyz, cage.xyz, book.xyz> \
    --out-dir <results directory> --scratch-dir <working directory> --core <cpu> --seeds 0-191
python analyze.py --runs <results directory>/runs.csv --frames <results directory>/crest_frames.csv \
    --reference-energy <energy of the prism minimum in eV> \
    --old-perseed <per-seed table of the earlier record> --old-receipts <per-seed CREST receipts of the earlier record>
```

The shim and the driver are built with `cc -O2 -shared -fPIC engine_shim.c -ldl`
and `cc -O2 crest_driver.c -ldl -lm`. Python needs numpy and scipy.

## Controls

`controls.py` is run after the session, never inside it, and times nothing. For
every run index it starts CREST from a given start geometry, once through the
call server on the engine of the A/B with the full budget, and once on the
built-in GFN2 of CREST, which is stopped when it has written more than 180
frames. The first 180 frames are scored by both builds against the references
as used in the A/B, against the reference files in their original atom order,
and against references whose molecules and hydrogens are shuffled. The
program records how many references each build loaded, the gate, and the
number of frames inside it. For every frame it also records how many connected
groups the six oxygens form at an O-O contact of 3.5 A, the smallest O-O
distance between two groups and the radius of gyration of the oxygens, so that
the shape of the structure CREST starts from and of the frames it produces can
be read from the same files. `--start-noise` displaces every start coordinate
by a Gaussian of the given width, with a seed that depends on the run index;
CREST stops at its own check of the bonding of the start when the
displacement changes it, and such runs have no frames. `--pack-radius`
replaces the start file by a random packing of waters with the geometry of its
first molecule and their centres inside a sphere of the given radius, again
with a seed that depends on the run index. `analyze_controls.py` prints the
tables; runs without frames are counted as failed.

```
python controls.py --binary-a <build A> --binary-b <build B> --engine <engine library> \
    --shim <shim library> --driver <driver binary> --crest <crest binary> \
    --water-xyz <start geometry> --start-label <name> --references-dir <references of the A/B> \
    --original-references-dir <references in their original order> --out-dir <results directory> \
    --scratch-dir <working directory> --core <cpu> --runs 0-23
python analyze_controls.py <results directory> [<results directory> ...]
```
