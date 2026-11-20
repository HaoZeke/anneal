# b10 trajectory-neutral rigid quench

b10 is based on b9 `b58dd00910d10cc6cc75ac1a61427730234cac0d`.
All release runs use the recommended rigid preset. Raw logs and exact-bit
outcomes are retained with the delivery evidence.

## Change

The b8 coordinate chart reduced objective calls per hop by changing the
L-BFGS path. That cannot be retained while preserving b9's objective-call
sequence. b10 instead removes overhead from the established path.

The most recent paid check previously allocated and copied both its point
and gradient at every objective call. b10 reuses those two buffers. It also
moves the final check into its handoff and compares stored points through
views rather than allocating a temporary point. No floating-point operation,
objective call, random draw, proposal, or acceptance decision changes.

## Exact trajectory

The first-divergence tracer recorded every objective input, energy, and
gradient by `to_bits` for TIP4P19 seed 2000 at 20,000 calls. It also recorded
the final outcome and accepted move names. b9 and b10 have identical traces:

```text
objective trace:
2d2b18f55bf23a8b08590999f4b3df9ff1955f9f492e2c7d0c926a3c18b716ec
outcome:
c0d3f46f78b78f9b5364744eb9650b9c2ad4467c71a56feadc7ff754cce76693
first divergence: none through 20,000 calls
```

The full identity panel compares `Outcome` energy and state bits, accepted
transitions in order, and per-arm draws, accepts, and best energies:

- TIP4P11–21, seeds 0–11, 20,000 calls: 132/132 identical.
- TIP4P14 and TIP4P21, seeds 0–11, two million calls with the paper stop:
  24/24 identical.

Thus calls per hop, random draws, proposal order, energies, states, and hits
are unchanged in all 156 rows.

## Outside time

Timing uses TIP4P21 at 20,000 calls for seeds 0–5. Each seed runs an
`A1 A2 A1 A2` b9 control and then b9/b10 in `A B A B` order. The aggregate
outside times are:

| Comparison | First arm, us/call | Second arm, us/call | Ratio |
| --- | ---: | ---: | ---: |
| b9 A/A | 3.655233 | 3.483350 | 0.952976 |
| b9/b10 | 3.489875 | 3.470800 | 0.994534 |

The six A/A replicate ratios range from 0.779415 to 1.006763, with median
0.996257. One b9 control run in replicate four is the high-side outlier.
The six b9/b10 replicate ratios range from 0.977832 to 1.031179, with median
0.988825. Aggregate b10 outside time per call is 0.994534 times b9.
