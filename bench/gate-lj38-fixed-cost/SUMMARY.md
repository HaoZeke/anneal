# lj38 fixed-cost gate

One session, one host, core 0, release builds, every thread variable
set to 1. The host CPU is Intel(R) Xeon(R) Processor. Its flag list
is the `cpu_flags` row of `results/sessions.csv` and includes `avx2`.

The main panel is 288 runs. The identity panel is 72 runs, sealed in
its own rows after the main panel. There is no quick mode and no GMIN
run.

## Main panel

| cell | n | verdict | hits A/B/A2 | out/call B/A [lo,hi] | A2/A hi | out total B/A [lo,hi] | RSS peak B/A [lo,hi] | identity |
|---|---|---|---|---|---|---|---|---|
| lj38 | 24 | PASS | 24/24/24 | 0.9115 [0.8892,0.9373] | 1.013 | 0.9115 [0.8892,0.9373] | 0.9807 [0.9745,0.9863] | identical |
| m38r14d | 12 | PASS | 12/12/12 | 0.8557 [0.8218,0.8866] | 1.002 | 0.8557 [0.8218,0.8866] | 0.9883 [0.9819,0.9936] | identical |
| lj75 | 12 | PASS | 5/5/5 | 0.9958 [0.9766,1.016] | 1.037 | 0.9958 [0.9766,1.016] | 1 [0.9998,1] | identical |
| lj98 | 12 | PASS | 7/7/7 | 0.9408 [0.9202,0.9612] | 0.9868 | 0.9408 [0.9202,0.9612] | 0.9989 [0.9971,1] | identical |
| m80r6 | 12 | PASS | 12/12/12 | 0.8763 [0.8427,0.9108] | 1.035 | 0.8763 [0.8427,0.9108] | 1.002 [0.9999,1.003] | identical |
| au55 | 12 | PASS | 4/4/4 | 0.9862 [0.9732,0.9993] | 1.013 | 0.9862 [0.9732,0.9993] | 1 [0.9997,1] | identical |
| hexamer | 12 | PASS | 12/12/12 | 0.9854 [0.9534,1.019] | 1.009 | 0.9854 [0.9534,1.019] | 0.9983 [0.9907,1.005] | identical |

Strict identity holds on every seed: hit, calls, charged calls,
best-energy bits and trajectory hash. A and A2 agree on those fields
too, so the session repeats. No row has an error. The calls ratio is
1 on every cell, so the per-call and total outside ratios agree. The
early-stop note is present because hits stop before the budget; it
does not split the two ratios.

Outside time, per call and total: every upper bound is at most 1.10,
and at most max(1.10, the A2/A upper bound). No lower bound is above
1. Memory: no lower bound exceeds max(1.10, the A2/A upper bound),
and every upper bound is at most 1.10. No hit loss.

The builder's unverified lj38 per-call claim is about 0.92. The
measured per-call ratio is 0.9115, interval [0.8892, 0.9373]. The
upper end is below 1. The value 0.92 lies inside the interval. The
total outside ratio is the same.

## Path counters

The counter copy matches the timed B rows on trajectory, energy bits,
calls and charged calls. Sums:

| counter | total |
|---|---|
| solve_shortest_paths_loop_avx2 | 752 |
| kuhn_munkres_loop_avx2 | 188 |
| augment_tight_avx2 | 87092 |
| csm_recall_op | 1104 |
| candidate_from_basis | 3 |

Every counter is nonzero.

## Identity panel

tip4p6 and sc126-38, twelve seeds each, arms A, B and A2. Strict
identity only. Both cells match on all twelve seeds for hit, calls,
charged calls, best-energy bits and trajectory hash. A and A2 match
as well. tip4p6 also matches the hexamer cell of the main panel on
those fields for every shared seed and arm. No row has an error.

## Verdict

Main panel: PASS on every cell. Identity panel: PASS on strict
identity for both cells. Path counters: PASS. The gate is PASS.
