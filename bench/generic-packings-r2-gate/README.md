# generic-packings-r2 full-panel gate

Verdict: **FAIL**.

## Provenance

- Machine id: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.
- Session: `20261121T015950Z`; 1,152 rows.
- Harness sha256: `f19728320e7b8066157fb8d3bb94b9776ad4373231cf8185963bc8c2a8f87715`.
- D: revision `61fa67cf380e46876b569da45e7cc091647a775b`, tree `cff796c507a8fa1d0bfb28888c3570b61988055e`.
- D2: revision `61fa67cf380e46876b569da45e7cc091647a775b`, tree `cff796c507a8fa1d0bfb28888c3570b61988055e`.
- G: revision `2f8b6df54ba35c2f0b185d18cacd4df4e3ba13f5`, tree `0bb8d0fa9889c3fc6d08906fc721f81f41dbed6d`.
- P: revision `628773d6837ea2d959a27256ff4163b21162fd4d`, tree `dd98579835b57f4f0c86588f7c9cb2a3cd7a9691`.

The panel has 24 cells and seeds 2000 through 2011. Runs are serial
on core 0 with thread counts fixed at one. Arm order is D, D2, G, P
on even seed offsets and the reverse on odd offsets. Hits are
re-derived from final energies with a tolerance of 0.001.
Side-probe accounting is replayed from the same candidate source;
it stops immediately after the deterministic initial probe and is
not used for timing or final-energy rows.

## Gate clauses

- Hit loss beyond A/A spread: PASS.
- Outside time per call: PASS.
- Peak RssAnon: FAIL (lj102, au55)

VmHWM is reported but does not gate. Calls use the charged count to
the hit; misses rank as positive infinity only in Mann-Whitney.
The memory comparison uses per-cell medians of each process's
sampled peak. Its A/A band is max(D2/D, D/D2).

## Per-cell results

| cell | hits D/D2/G/P | calls G med/IQR | outside G/D | MW calls/outside p | McNemar p | offered/hit | charged hit/probe med | RssAnon G/D | prior/species | cell |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| lj38 | 12/12/12/12 | 549/117.25 | 0.6660 | 3.65846e-05/3.65846e-05 | 1 | 12/12 | 549.0/187.0 | 0.0739 | removed/none | pass |
| lj55 | 12/12/12/12 | 1395.5/180.5 | 0.4324 | 3.65846e-05/3.65846e-05 | 1 | 12/12 | 1395.5/862.0 | 0.2125 | removed/none | pass |
| lj75 | 5/5/5/12 | 4e+06/960708 | 0.9887 | 0.897417/0.750832 | 1 | 12/0 | 2723420.0/2068.0 | 0.9994 | removed/none | pass |
| lj98 | 7/7/7/10 | 3.16948e+06/2.72406e+06 | 1.0168 | 0.85733/0.795012 | 1 | 12/0 | 1433356.0/2667.0 | 0.9989 | removed/none | pass |
| lj102 | 6/6/6/9 | 3.25703e+06/2.04188e+06 | 1.0171 | 0.877416/0.930987 | 1 | 12/0 | 1818084.5/1951.0 | 1.0019 | removed/none | FAIL |
| lj103 | 6/6/6/8 | 3.9062e+06/3.15852e+06 | 0.9987 | 0.877416/0.885234 | 1 | 12/0 | 641170.5/2255.0 | 0.9992 | removed/none | pass |
| lj104 | 5/5/5/9 | 4e+06/2.26924e+06 | 1.0050 | 0.897417/0.885234 | 1 | 12/0 | 1303398.0/2490.0 | 0.9996 | removed/none | pass |
| lj150 | 8/8/8/5 | 1.13237e+06/3.83917e+06 | 0.9952 | 0.836895/0.83986 | 1 | 12/0 | 172293.0/2987.0 | 1.0020 | removed/none | pass |
| au55 | 4/4/4/6 | 1e+06/736743 | 0.9995 | 0.917812/0.885234 | 1 | 12/0 | 96179.5/1.0 | 1.0000 | removed/none | FAIL |
| m55r6 | 12/12/12/12 | 1373/128.75 | 0.4089 | 3.65846e-05/0.000592042 | 1 | 12/12 | 1373.0/822.0 | 0.1518 | removed/none | pass |
| m55r14 | 9/9/9/7 | 454146/773280 | 0.9824 | 0.815981/0.885234 | 1 | 12/0 | 336805.0/2583.0 | 0.9988 | removed/none | pass |
| m80r6 | 12/12/12/2 | 64797/575468 | 0.9249 | 0.750832/0.214494 | 1 | 12/0 | 64797.0/2119.0 | 0.9982 | removed/none | pass |
| m80r14 | 6/6/6/9 | 3.90293e+06/1.44378e+06 | 1.0107 | 0.877416/0.707454 | 1 | 12/0 | 2135936.0/2992.0 | 0.9998 | removed/none | pass |
| m38r14 | 4/4/12/12 | 1476.5/46.5 | 0.5338 | 2.59339e-05/3.65846e-05 | 0.0078125 | 12/12 | 1476.5/177.0 | 0.0391 | removed/none | pass |
| m38r6 | 12/12/12/12 | 1623.5/65.25 | 0.3466 | 3.65846e-05/3.65846e-05 | 1 | 12/12 | 1623.5/1223.0 | 0.1402 | removed/none | pass |
| sc126-38 | 12/12/12/12 | 1184.5/133.5 | 0.5633 | 3.65846e-05/0.000592042 | 1 | 12/12 | 1184.5/392.0 | 0.2225 | removed/none | pass |
| sc126-55 | 12/12/12/12 | 2838.5/168.75 | 0.3877 | 3.6442e-05/3.65846e-05 | 1 | 12/12 | 2838.5/1782.0 | 0.2956 | removed/none | pass |
| sc126-80 | 5/5/5/6 | 1e+06/488932 | 0.9859 | 0.897417/1 | 1 | 12/0 | 304959.0/3228.0 | 0.9967 | removed/none | pass |
| sc108-38 | 7/7/12/12 | 1706/32.5 | 0.4962 | 3.32821e-05/3.65846e-05 | 0.0625 | 12/12 | 1706.0/492.0 | 0.0665 | removed/none | pass |
| sc108-55 | 5/5/5/6 | 1e+06/379645 | 0.9770 | 0.897417/0.885234 | 1 | 12/0 | 586309.0/1915.0 | 0.9979 | removed/none | pass |
| sc108-80 | 3/3/3/0 | 1e+06/72004.5 | 1.0010 | 0.939502/0.97697 | 1 | 12/0 | 412887.0/2759.0 | 0.9965 | removed/none | pass |
| sc96-38 | 12/12/12/12 | 1325.5/104 | 0.5777 | 3.65846e-05/3.65846e-05 | 1 | 12/12 | 1325.5/561.0 | 0.3042 | removed/none | pass |
| sc96-55 | 12/12/12/12 | 2740/175.5 | 0.4598 | 3.65846e-05/3.65846e-05 | 1 | 12/12 | 2740.0/1772.0 | 0.2609 | removed/none | pass |
| sc96-80 | 7/7/7/7 | 992722/202522 | 0.9978 | 0.85733/0.707454 | 1 | 12/0 | 884431.0/2874.0 | 0.9992 | removed/none | pass |

LJ38: G hits 12/12; best energy -173.92842659; charged-to-hit median 549.0.

LJ98: G hits 7/12; best energy -543.66536077; charged-to-hit median 1433356.0.
