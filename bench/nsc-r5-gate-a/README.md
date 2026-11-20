# No-size-constants r5 gate A

Verdict: **PASS**

The rule was fixed before execution: FIX if any cell has fewer B hits, if B loses on calls at two-sided p < 0.05, or if the B/A ratio of median outside-potential time per charged call exceeds 1.10. Otherwise PASS.

The campaign used seeds 2000–2011 at the paper budgets in one serial session. Adjacent seeds used AB then BA order, giving six observations for every (slot, arm) pair in every cell. No budget override was set.

## Provenance

| item | value |
|---|---|
| arm A revision | `61fa67cf380e46876b569da45e7cc091647a775b` |
| arm A tree | `cff796c507a8fa1d0bfb28888c3570b61988055e` |
| arm B revision | `09e9df918569dad484f2feb5d0e4eb82ff49f62c` |
| arm B tree | `64b12739c8707669e6ae203d8f8b99a160609b55` |
| runner blob | `76f5c16a41d8706d970e68a7730b58d03cddb826` |
| summarizer blob | `a8d6d197cda9270bdcac4c41b4fd1c0c143258f2` |
| machine identity hash | `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873` |

The recorded runner blob is the pre-strip harness, and the only difference is that one env line.

Binary hashes are recorded on every row in `records.csv`.

## Cell results

| panel | cell | hits A/B | Fisher p | calls A/B | MW p | wall A/B (s) | potential A/B (s) | outside A/B (s) | outside/call B/A | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| au | au38 | 8/8 | 1 | 186585/186585 | 1 | 3.167107/3.158993 | 2.37273/2.3646935 | 0.801164/0.800488 | 0.99610012 | PASS |
| lj | lj103 | 6/6 | 1 | 3905072.5/3905072.5 | 1 | 101.42932/102.67142 | 62.896707/62.457823 | 38.532612/39.160808 | 0.98708332 | PASS |
| lj | lj104 | 5/5 | 1 | 4000000/4000000 | 1 | 103.43384/103.70933 | 64.433237/64.174375 | 38.988263/39.205259 | 1.0036894 | PASS |
| lj | lj38 | 12/12 | 1 | 127839.5/127839.5 | 1 | 0.608464/0.6103185 | 0.2876555/0.2864425 | 0.3208085/0.323876 | 1.0156317 | PASS |
| lj | lj75 | 5/5 | 1 | 4000000/4000000 | 1 | 62.819431/61.418321 | 34.664165/34.78385 | 27.823775/27.237523 | 1.0296023 | PASS |
| lj | lj98 | 7/7 | 1 | 3166811.5/3166811.5 | 1 | 85.262011/82.457169 | 50.719008/48.817362 | 33.697618/32.267226 | 1.0026419 | PASS |
| morse | m38r14 | 4/4 | 1 | 400000/400000 | 1 | 3.400265/3.3825515 | 2.4099805/2.407977 | 0.9822885/0.971381 | 0.99588553 | PASS |
| morse | m38r6 | 12/12 | 1 | 49438.5/49438.5 | 1 | 0.4496295/0.447647 | 0.305767/0.306956 | 0.148044/0.1444345 | 0.99346405 | PASS |
| morse | m55r14 | 9/9 | 1 | 451561.5/451561.5 | 1 | 7.479332/7.4155455 | 5.724033/5.681696 | 1.7552995/1.73385 | 0.99047697 | PASS |
| morse | m55r6 | 12/12 | 1 | 22596/22596 | 1 | 0.406627/0.4062735 | 0.2858025/0.2863545 | 0.120315/0.1201345 | 0.99829301 | PASS |
| morse | m80r14 | 6/6 | 1 | 3901429/3901429 | 1 | 138.04656/137.12311 | 103.24742/102.86779 | 35.084598/34.432896 | 0.99145834 | PASS |
| morse | m80r6 | 12/12 | 1 | 62677/62677 | 1 | 2.2700455/2.265806 | 1.662618/1.661832 | 0.6074275/0.603974 | 0.98442948 | PASS |

Calls use misses at the paper budget. Mann–Whitney is tie-corrected with continuity correction. Times and calls are medians. Outside-per-call uses charged calls.

## Multi-system panel tests

| panel | cells | exact CMH p | van Elteren z | van Elteren p |
|---|---:|---:|---:|---:|
| lj | 5 | 1 | 0 | 1 |
| morse | 6 | 1 | 0 | 1 |

## Hit-status changes

None.
