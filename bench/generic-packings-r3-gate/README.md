# generic-packings-r3 gate

This is the preregistered replacement for the r2 measurement. The r2 rows
and verdict remain unchanged and are superseded, not relabelled.

## Measurement fixes

1. A thread inside each timed process samples its own
   `/proc/self/status`. The maximum sampled `RssAnon`, including a final
   sample after the search, is the gating memory value. External polling,
   `VmHWM`, and `ru_maxrss` are report-only.
2. The timed process emits the generic-packing offered, hit, and probe-call
   counters. A G cell with fewer than 12 offered rows is invalid.
3. D is built at five layouts. Each contains an uncalled, noinline
   NOP-padding symbol of a different size and has a distinct `.text`
   sha256. D2 is an independent build of the first layout. For each cell,
   the placement band is the largest median-RssAnon ratio to D among the
   five layouts and D2. The memory clause passes when G/D is no greater
   than that band, or when the absolute G-D median difference is at most
   8 kB.

The hit clause and outside-time-per-call clause are unchanged from r2.
Hit loss is allowed only within the D2/D hit-count spread. Outside time per
charged call passes when G/D is at most 1.10 or within the symmetric D2/D
band.

## Protocol

The panel, references, budgets, and seeds 2000 through 2011 are unchanged:
24 cells and 12 rows per arm per cell. All binaries are release builds.
Runs are serial on core 0 with every thread-count variable set to one.
The core-arm order is D, D2, G, P on even seed offsets and the reverse on
odd offsets. There is no quick mode. Four additional D-layout arms follow
the core arms and reverse with them.

At most two replacement runs are allowed per cell. A third replacement
makes that cell and the overall result inconclusive.
