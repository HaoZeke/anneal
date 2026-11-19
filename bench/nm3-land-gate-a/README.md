# nm3-land no-loss gate A

verdict: PASS — no hit or calls-to-hit loss at p < 0.05

## Provenance

- session: `142b566861a97a3de899`
- machine id: `sha256:3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
- arm A: commit `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`, tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`
- arm B: commit `8f1e0b5c5078062d5f241371554555421526452c`, tree `2ba43832b540c6e9005cfb6be31243b36ab086a7`
- kit: `sha256:cddaf32978ceeaeedb73d7ebff0f00d5006689c19d110ce8df69fb2881204134`
- runner blob: `bbd1a97a02552d07489b17d790fcbddf6f98a6d3`
- plain-mask probe blob: `f48de44876140f76174837144d561ff227e81734`

Every recorded object id resolves with `git cat-file -e`. Arm B is exactly two commits
above arm A and has the required tree. The change is exercised with its default settings.

## Protocol

All jobs ran in one session, serially, with process affinity fixed to CPU 0 and every
declared thread pool fixed to one thread. Both arms used separate locked release builds.
No budget override, pre-warm, smoke, rescue, retry, or retuning run was used. Seeds
2000–2011 used alternating AB/BA pairs, producing ABBA over each adjacent seed pair.
For every paper cell, each `(slot, arm)` count is 6; for the mask probe each count is 8.

## Paper-budget cells

| cell | hits A/B | Fisher p | median calls A/B | MW p | wall s A/B | potential s A/B | outside s A/B | outside/call s A/B | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lj38 | 12/11 | 1 | 127839.5/162769.5 | 0.885233914 | 0.602306/0.7821815 | 0.2846705/0.3621595 | 0.3176355/0.420022 | 2.48298682e-06/2.71506149e-06 | PASS |
| lj75 | 5/7 | 0.684272653 | 4000000/3799817.5 | 0.621598019 | 59.907834/55.806792 | 33.932538/32.8637375 | 25.087072/22.9430545 | 6.36703498e-06/6.2068797e-06 | PASS |
| lj98 | 7/11 | 0.154952599 | 3166811.5/1998509 | 0.268982635 | 78.881394/50.3801425 | 47.4044855/31.4202775 | 31.000028/18.878068 | 9.71861677e-06/9.73636735e-06 | PASS |
| lj103 | 6/5 | 1 | 3905072.5/4000000 | 0.549956573 | 101.071025/104.697749 | 62.7424315/64.004237 | 38.179776/39.455051 | 1.15107464e-05/1.10411349e-05 | PASS |
| lj104 | 5/8 | 0.413649213 | 4000000/1083653 | 0.0950047101 | 105.48423/29.8832935 | 65.4402455/17.8019165 | 39.3573345/12.081377 | 1.07497492e-05/1.180374e-05 | PASS |
| m80r6 | 12/12 | 1 | 62677/97451 | 0.885233914 | 2.5363115/3.6214325 | 1.8604175/2.660817 | 0.6758935/1.0097135 | 1.22401349e-05/1.32313479e-05 | PASS |
| m80r14 | 6/9 | 0.400323058 | 3901429/1034077.5 | 0.0469619626 | 142.350278/38.3681495 | 104.979767/28.5819415 | 34.846981/9.786208 | 9.47595813e-06/9.63358239e-06 | PASS |
| au38 | 8/11 | 0.316770186 | 186585/123073.5 | 0.212494496 | 3.2162555/2.0937005 | 2.4117055/1.5835695 | 0.8143495/0.510131 | 4.4234738e-06/4.37351554e-06 | PASS |

## Plain LJ38 two-phase mask

Arm A is `0x240d` (5/16); arm B is `0x0014` (2/16). Hit outcomes changed on seeds 0, 3, 4, 10, 13; calls-to-hit changed on seeds 0, 2, 3, 4, 10, 13.

| seed | A hit | A calls | B hit | B calls |
|---:|---:|---:|---:|---:|
| 0 | 1 | 10266 | 0 | 30000 |
| 1 | 0 | 30000 | 0 | 30000 |
| 2 | 1 | 16188 | 1 | 25722 |
| 3 | 1 | 23393 | 0 | 30000 |
| 4 | 0 | 30000 | 1 | 28328 |
| 5 | 0 | 30000 | 0 | 30000 |
| 6 | 0 | 30000 | 0 | 30000 |
| 7 | 0 | 30000 | 0 | 30000 |
| 8 | 0 | 30000 | 0 | 30000 |
| 9 | 0 | 30000 | 0 | 30000 |
| 10 | 1 | 28304 | 0 | 30000 |
| 11 | 0 | 30000 | 0 | 30000 |
| 12 | 0 | 30000 | 0 | 30000 |
| 13 | 1 | 13110 | 0 | 30000 |
| 14 | 0 | 30000 | 0 | 30000 |
| 15 | 0 | 30000 | 0 | 30000 |
