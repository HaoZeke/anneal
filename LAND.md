# LAND: anneal PR 171 r-next2 gate (r1 + r2)

- Target: HaoZeke/anneal PR #171
- Lease: d87ca43e18f9dbf7ff67c65f15738824a11ed6af
- Payload: d87ca43e (production identical in r-next3 24135c0a,
  tree 393c478178b0c7b47d067e16f34559c050c38430; the r-next3 diff
  d87ca43e..24135c0a adds 85 lines inside `mod tests` only)
- Gate tree: ca2f2f4a923d5a8761aa2f08765c7698e89de781
- Gate base tree: cf764d0f64a48425c7ac00036ce55a64167ce3de
  (develop 364679f0 less the five bench gate files)
- Builder: told to Inky by message
- Measurer: OpenAI

Gate commits only add files under bench/gate-pr171-rnext2/.

| cell | hits D/D2/G/G2 | Fisher p | outside G/D | RSS G/D | verdict |
| --- | --- | ---: | --- | --- | --- |
| m38r14d | 11/11/12/12 | 1 | 0.843 (0.695, 0.981) | 1.000 | PASS |
| lj75 | 5/5/5/5 | 1 | 1.007 (0.992, 1.027) | 1.051 (1.041, 1.057) | PASS |
| lj98 | 7/7/7/7 | 1 | 0.983 (0.971, 0.996) | 1.071 (1.056, 1.081) | PASS |
| tip4p6 | 24/24/24/24 | 1 | 0.992 (0.972, 1.012) | 1.000 | PASS |

Box recompute from rows-r2.csv agrees on PASS in all four cells.
Point estimates are geometric means of seed-paired ratios; the box
exact order-statistic CIs on the median ratio differ slightly in
endpoints but give the same verdicts.
