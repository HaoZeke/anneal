# Water hexamer A/B: 192 seeds, 1536 runs, gate 0.598496 A

## Hits

| arm | A hits | B hits | A - B | Fisher exact p |
|---|---|---|---|---|
| anneal | 96/192 | 185/192 | -89 | 2.96e-27 |
| polish_fair | 119/192 | 183/192 | -64 | 1.63e-16 |
| crest | 0/192 | 0/192 | +0 | 1.000 |
| crest_fair | 32/192 | 32/192 | +0 | 1.000 |

Paired by seed

| arm | both | only A | only B | neither | McNemar exact p |
|---|---|---|---|---|---|
| anneal | 94 | 2 | 91 | 5 | 8.83e-25 |
| polish_fair | 114 | 5 | 69 | 4 | 1.83e-15 |
| crest | 0 | 0 | 0 | 192 | 1.000 |
| crest_fair | 2 | 30 | 30 | 130 | 1.000 |

## Calls and times (runs with status ok)

| arm | code | runs ok | median calls | median wall s | median in s | median out s | median final E eV | mean wall s | p90 wall s | max wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| anneal | A | 192 | 4000 | 19.46 | 17.54 | 1.95 | -829.8384 | 23.48 | 28.80 | 71.96 |
| anneal | B | 192 | 4000 | 18.15 | 15.43 | 2.67 | -829.8522 | 18.63 | 21.23 | 24.71 |
| polish_fair | A | 192 | 3527 | 17.13 | 15.16 | 1.81 | -829.8501 | 20.71 | 22.41 | 69.78 |
| polish_fair | B | 191 | 3383 | 15.31 | 13.05 | 2.16 | -829.8522 | 15.74 | 18.04 | 21.87 |
| crest | A | 192 | 4000 | 18.52 | 14.54 | 4.00 | -828.9328 | 18.99 | 21.30 | 26.69 |
| crest | B | 192 | 4000 | 18.40 | 14.38 | 4.01 | -828.9425 | 18.89 | 21.27 | 25.50 |
| crest_fair | A | 192 | 3776 | 17.33 | 13.84 | 3.43 | -829.7138 | 17.63 | 20.10 | 24.97 |
| crest_fair | B | 192 | 3763 | 17.23 | 13.82 | 3.43 | -829.7142 | 17.65 | 20.32 | 24.63 |

| arm | MWU p calls | MWU p wall | MWU p in | MWU p out |
|---|---|---|---|---|
| anneal | 0.158 | 5.82e-11 | 1.42e-30 | 1.31e-45 |
| polish_fair | 1.31e-68 | 7.10e-29 | 5.30e-36 | 6.97e-24 |
| crest | 1.000 | 0.343 | 0.330 | 0.569 |
| crest_fair | 0.825 | 0.994 | 0.991 | 0.746 |

## Slow runs

| arm | code | runs ok | wall above 1.5 x pooled median | seeds | median ms in the engine per call, slow runs | same, other runs |
|---|---|---|---|---|---|---|
| anneal | A | 192 | 22 | 16, 19, 22, 23, 38, 39, 48, 59, 62, 63, 68, 102, 109, 118, 122, 135, 144, 146, 157, 160, 187, 189 | 13.84 | 4.20 |
| anneal | B | 192 | 0 | - | - | 3.86 |
| polish_fair | A | 192 | 18 | 16, 22, 38, 39, 48, 59, 62, 63, 68, 109, 118, 122, 135, 144, 146, 157, 160, 187 | 15.20 | 4.14 |
| polish_fair | B | 191 | 0 | - | - | 3.86 |
| crest | A | 192 | 0 | - | - | 3.63 |
| crest | B | 192 | 0 | - | - | 3.59 |
| crest_fair | A | 192 | 0 | - | - | 3.67 |
| crest_fair | B | 192 | 0 | - | - | 3.65 |

## Final energies

| arm | code | runs | median E eV | p10 E eV | p90 E eV | lowest E eV | runs within 1 meV of -829.852228 eV |
|---|---|---|---|---|---|---|---|
| anneal | A | 192 | -829.8384 | -829.8458 | -829.7946 | -829.8498 | 0 |
| anneal | B | 192 | -829.8522 | -829.8522 | -829.8522 | -829.8522 | 182 |
| polish_fair | A | 192 | -829.8501 | -829.8522 | -829.8022 | -829.8522 | 96 |
| polish_fair | B | 191 | -829.8522 | -829.8522 | -829.8522 | -829.8522 | 178 |
| crest | A | 192 | -828.9328 | -829.0777 | -828.7572 | -829.2885 | 0 |
| crest | B | 192 | -828.9425 | -829.0748 | -828.7624 | -829.2511 | 0 |
| crest_fair | A | 192 | -829.7138 | -829.8480 | -829.1345 | -829.8522 | 18 |
| crest_fair | B | 192 | -829.7142 | -829.8480 | -829.1345 | -829.8522 | 17 |

## Anneal arms: runs whose answer is within a gate of the nearest of the three references

| arm | code | runs | RMSD <= 0.300 | RMSD <= 0.500 | RMSD <= 0.598 | RMSD <= 0.700 | RMSD <= 0.784 | RMSD <= 1.000 |
|---|---|---|---|---|---|---|---|---|
| anneal | A | 192 | 50 | 90 | 96 | 103 | 123 | 149 |
| anneal | B | 192 | 185 | 185 | 185 | 185 | 185 | 186 |
| polish_fair | A | 192 | 119 | 119 | 119 | 119 | 144 | 159 |
| polish_fair | B | 191 | 183 | 183 | 183 | 183 | 184 | 185 |

## Run status

| arm | code | status | runs |
|---|---|---|---|
| polish_fair | B | no_geometry_to_polish | 1 |

Runs without a final energy (counted as misses)

| seed | code | arm | status | calls | wall s | out s | reason | hit |
|---|---|---|---|---|---|---|---|---|
| 83 | B | anneal | ok | 1 | 13.996491 | 13.980685 | dissociated monomer | 0 |
| 83 | B | polish_fair | no_geometry_to_polish | 1 | 11.793538 | 11.777845 | dissociated monomer | 0 |

## Order within the pair

| arm | code | runs first | runs second | median wall first s | median wall second s | MWU p |
|---|---|---|---|---|---|---|
| anneal | A | 96 | 96 | 19.43 | 19.54 | 0.379 |
| anneal | B | 96 | 96 | 18.15 | 18.16 | 0.238 |
| polish_fair | A | 96 | 96 | 16.83 | 17.24 | 0.637 |
| polish_fair | B | 95 | 96 | 15.36 | 15.31 | 0.422 |
| crest | A | 96 | 96 | 18.40 | 18.62 | 0.414 |
| crest | B | 96 | 96 | 18.40 | 18.40 | 0.772 |
| crest_fair | A | 96 | 96 | 17.24 | 17.39 | 0.916 |
| crest_fair | B | 96 | 96 | 17.23 | 17.23 | 0.970 |

## Wall time per call over the session

| arm | code | seeds | runs | median wall ms per call |
|---|---|---|---|---|
| anneal | A | 0-23 | 24 | 5.667 |
| anneal | A | 24-47 | 24 | 5.459 |
| anneal | A | 48-71 | 24 | 4.758 |
| anneal | A | 72-95 | 24 | 4.538 |
| anneal | A | 96-119 | 24 | 4.632 |
| anneal | A | 120-143 | 24 | 4.513 |
| anneal | A | 144-167 | 24 | 4.506 |
| anneal | A | 168-191 | 24 | 4.782 |
| anneal | B | 0-23 | 24 | 5.302 |
| anneal | B | 24-47 | 24 | 5.244 |
| anneal | B | 48-71 | 24 | 4.551 |
| anneal | B | 72-95 | 24 | 4.463 |
| anneal | B | 96-119 | 24 | 4.514 |
| anneal | B | 120-143 | 24 | 4.450 |
| anneal | B | 144-167 | 24 | 4.437 |
| anneal | B | 168-191 | 24 | 4.438 |
| polish_fair | A | 0-23 | 24 | 5.598 |
| polish_fair | A | 24-47 | 24 | 5.426 |
| polish_fair | A | 48-71 | 24 | 4.783 |
| polish_fair | A | 72-95 | 24 | 4.528 |
| polish_fair | A | 96-119 | 24 | 4.569 |
| polish_fair | A | 120-143 | 24 | 4.493 |
| polish_fair | A | 144-167 | 24 | 4.513 |
| polish_fair | A | 168-191 | 24 | 4.682 |
| polish_fair | B | 0-23 | 24 | 5.194 |
| polish_fair | B | 24-47 | 24 | 5.120 |
| polish_fair | B | 48-71 | 24 | 4.570 |
| polish_fair | B | 72-95 | 23 | 4.422 |
| polish_fair | B | 96-119 | 24 | 4.488 |
| polish_fair | B | 120-143 | 24 | 4.407 |
| polish_fair | B | 144-167 | 24 | 4.429 |
| polish_fair | B | 168-191 | 24 | 4.429 |

## Earlier record next to this run

| arm | earlier record | A | B |
|---|---|---|---|
| anneal | 121/192 | 96/192 | 185/192 |
| polish_fair | 148/192 | 119/192 | 183/192 |
| crest | 97/192 | 0/192 | 0/192 |
| crest_fair | 114/192 | 32/192 | 32/192 |

Per seed against the earlier rows

| arm | code | both | only earlier | only now | neither | McNemar exact p |
|---|---|---|---|---|---|---|
| anneal | A | 59 | 62 | 37 | 34 | 0.015 |
| anneal | B | 118 | 3 | 67 | 4 | 9.69e-17 |
| polish_fair | A | 89 | 59 | 30 | 14 | 0.003 |
| polish_fair | B | 143 | 5 | 40 | 4 | 7.88e-08 |
| crest | A | 0 | 97 | 0 | 95 | 1.26e-29 |
| crest | B | 0 | 97 | 0 | 95 | 1.26e-29 |
| crest_fair | A | 20 | 94 | 12 | 66 | 6.20e-17 |
| crest_fair | B | 16 | 98 | 16 | 62 | 1.48e-15 |

## crest: frames inside the gate per run

| frames of the CREST arm | runs with a hit | mean frames in gate | 0 | 1 | 2 | 3 | 4 or more |
|---|---|---|---|---|---|---|---|
| earlier record, 180 frames per run, gate 0.784 | 94/192 | 0.84 | 98 | 46 | 36 | 8 | 4 |
| A-run frames, all three references, gate 0.598 | 0/192 | 0.00 | 192 | 0 | 0 | 0 | 0 |
| A-run frames, all three references, gate 0.784 | 0/192 | 0.00 | 192 | 0 | 0 | 0 | 0 |
| A-run frames, prism and cage, gate 0.784 | 0/192 | 0.00 | 192 | 0 | 0 | 0 | 0 |
| A-run frames, prism and cage, gate 0.598 | 0/192 | 0.00 | 192 | 0 | 0 | 0 | 0 |
| B-run frames, all three references, gate 0.598 | 0/192 | 0.00 | 192 | 0 | 0 | 0 | 0 |
| B-run frames, all three references, gate 0.784 | 0/192 | 0.00 | 192 | 0 | 0 | 0 | 0 |
| B-run frames, prism and cage, gate 0.784 | 0/192 | 0.00 | 192 | 0 | 0 | 0 | 0 |
| B-run frames, prism and cage, gate 0.598 | 0/192 | 0.00 | 192 | 0 | 0 | 0 | 0 |

## crest: seeds where A and B disagree on the hit

A hit, B no hit (0): []

B hit, A no hit (0): []

| seed | A hit | A frames in gate | A polish hit | A min RMSD | A nearest | B hit | B frames in gate | B polish hit | B min RMSD | B nearest |
|---|---|---|---|---|---|---|---|---|---|---|

## crest: frames

| run | runs | median frames | median intact | median IRA failed | median min RMSD | p10 min RMSD | p90 min RMSD |
|---|---|---|---|---|---|---|---|
| A | 192 | 180 | 60 | 120 | 1.507 | 1.131 | 2.961 |
| B | 192 | 180 | 51 | 129 | 1.532 | 1.255 | 2.565 |

## crest: distance to the nearest reference along the metadynamics runs

| run | frame within its metadynamics run | frames | frames with an RMSD | median RMSD to nearest reference A | p10 A | lowest A |
|---|---|---|---|---|---|---|
| A | 0-2 | 2304 | 0% | 6.482 | 6.242 | 6.034 |
| A | 3-9 | 5376 | 16% | 5.546 | 3.901 | 1.453 |
| A | 10-24 | 11520 | 40% | 3.715 | 2.156 | 0.957 |
| A | 25-48 | 15360 | 48% | 3.271 | 1.888 | 0.855 |
| B | 0-2 | 2304 | 0% | 6.981 | 6.033 | 5.496 |
| B | 3-9 | 5376 | 14% | 5.473 | 3.638 | 1.534 |
| B | 10-24 | 11520 | 38% | 3.775 | 2.115 | 1.063 |
| B | 25-48 | 15360 | 47% | 3.327 | 1.958 | 0.850 |

## crest: classifier effect

frames scored by both builds: 69120; frames where the two builds differ on the gate decision: 0; runs where the hit differs between the builds: 0

## crest: gate, reference set and hit rule

| reference set | A-run gate 0.598 | A-run gate 0.784 | B-run gate 0.598 | B-run gate 0.784 |
|---|---|---|---|---|
| prism | 0/192 | 0/192 | 0/192 | 0/192 |
| cage | 0/192 | 0/192 | 0/192 | 0/192 |
| book | 0/192 | 0/192 | 0/192 | 0/192 |
| prism+cage | 0/192 | 0/192 | 0/192 | 0/192 |
| prism+book | 0/192 | 0/192 | 0/192 | 0/192 |
| all three | 0/192 | 0/192 | 0/192 | 0/192 |

| gate A | A-run hits | B-run hits |
|---|---|---|
| 0.300 | 0/192 | 0/192 |
| 0.400 | 0/192 | 0/192 |
| 0.500 | 0/192 | 0/192 |
| 0.598 | 0/192 | 0/192 |
| 0.700 | 0/192 | 0/192 |
| 0.784 | 0/192 | 0/192 |
| 0.900 | 1/192 | 1/192 |
| 1.000 | 6/192 | 2/192 |
| 1.200 | 30/192 | 13/192 |
| 1.500 | 87/192 | 85/192 |

| run | any frame inside gate | lowest-energy frame inside gate |
|---|---|---|
| A | 0/192 | 0/192 |
| B | 0/192 | 0/192 |

## crest_fair: seeds where A and B disagree on the hit

A hit, B no hit (30): [7, 14, 15, 29, 30, 35, 37, 48, 53, 72, 75, 81, 89, 96, 103, 107, 110, 114, 118, 133, 134, 140, 144, 153, 165, 169, 173, 180, 183, 184]

B hit, A no hit (30): [1, 23, 31, 44, 45, 46, 54, 55, 58, 61, 65, 70, 71, 86, 87, 100, 101, 102, 113, 115, 131, 138, 139, 150, 151, 158, 167, 168, 181, 186]

| seed | A hit | A frames in gate | A polish hit | A min RMSD | A nearest | B hit | B frames in gate | B polish hit | B min RMSD | B nearest |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 4.703900 | book | 1 | 0 | 1 | 1.606447 | book |
| 7 | 1 | 0 | 1 | 2.182814 | prism | 0 | 0 | 0 | 1.207236 | book |
| 14 | 1 | 0 | 1 | 1.489523 | cage | 0 | 0 | 0 | 1.810261 | book |
| 15 | 1 | 0 | 1 | 1.520305 | book | 0 | 0 | 0 | 1.702889 | cage |
| 23 | 0 | 0 | 0 | 3.943099 | book | 1 | 0 | 1 | 1.351198 | prism |
| 29 | 1 | 0 | 1 | 1.634394 | prism | 0 | 0 | 0 | 1.758840 | book |
| 30 | 1 | 0 | 1 | 2.343934 | book | 0 | 0 | 0 | 1.277343 | cage |
| 31 | 0 | 0 | 0 | 1.831305 | book | 1 | 0 | 1 | 1.446473 | prism |
| 35 | 1 | 0 | 1 | 1.483239 | prism | 0 | 0 | 0 | 1.486697 | book |
| 37 | 1 | 0 | 1 | 1.439998 | book | 0 | 0 | 0 | 1.425030 | cage |
| 44 | 0 | 0 | 0 |  | - | 1 | 0 | 1 | 2.120031 | cage |
| 45 | 0 | 0 | 0 | 1.514596 | prism | 1 | 0 | 1 | 1.858323 | cage |
| 46 | 0 | 0 | 0 | 1.095085 | cage | 1 | 0 | 1 | 1.325575 | prism |
| 48 | 1 | 0 | 1 | 1.321114 | prism | 0 | 0 | 0 | 2.454526 | book |
| 53 | 1 | 0 | 1 | 1.200342 | book | 0 | 0 | 0 | 1.502484 | book |
| 54 | 0 | 0 | 0 |  | - | 1 | 0 | 1 | 1.715544 | book |
| 55 | 0 | 0 | 0 | 0.821448 | book | 1 | 0 | 1 | 1.590161 | cage |
| 58 | 0 | 0 | 0 | 1.130823 | prism | 1 | 0 | 1 | 1.354462 | prism |
| 61 | 0 | 0 | 0 | 1.368508 | cage | 1 | 0 | 1 | 1.921484 | book |
| 65 | 0 | 0 | 0 | 1.963996 | prism | 1 | 0 | 1 | 1.565307 | cage |
| 70 | 0 | 0 | 0 | 1.360853 | book | 1 | 0 | 1 | 1.651101 | cage |
| 71 | 0 | 0 | 0 | 2.013222 | cage | 1 | 0 | 1 | 1.222756 | prism |
| 72 | 1 | 0 | 1 | 1.179298 | prism | 0 | 0 | 0 | 5.254085 | book |
| 75 | 1 | 0 | 1 | 1.194245 | prism | 0 | 0 | 0 |  | - |
| 81 | 1 | 0 | 1 | 1.045734 | book | 0 | 0 | 0 | 2.113490 | cage |
| 86 | 0 | 0 | 0 | 1.812398 | prism | 1 | 0 | 1 | 2.227736 | book |
| 87 | 0 | 0 | 0 | 1.381257 | book | 1 | 0 | 1 | 1.377908 | cage |
| 89 | 1 | 0 | 1 | 1.271633 | book | 0 | 0 | 0 | 5.752508 | prism |
| 96 | 1 | 0 | 1 | 1.692726 | book | 0 | 0 | 0 | 1.631051 | book |
| 100 | 0 | 0 | 0 |  | - | 1 | 0 | 1 | 0.918457 | cage |
| 101 | 0 | 0 | 0 | 1.250992 | prism | 1 | 0 | 1 | 1.303812 | prism |
| 102 | 0 | 0 | 0 | 1.707398 | book | 1 | 0 | 1 | 1.898285 | prism |
| 103 | 1 | 0 | 1 | 1.499962 | cage | 0 | 0 | 0 | 1.586779 | prism |
| 107 | 1 | 0 | 1 | 1.245768 | cage | 0 | 0 | 0 | 1.454860 | prism |
| 110 | 1 | 0 | 1 | 1.261884 | cage | 0 | 0 | 0 | 1.179902 | prism |
| 113 | 0 | 0 | 0 | 1.729363 | cage | 1 | 0 | 1 | 1.208994 | prism |
| 114 | 1 | 0 | 1 | 1.174938 | book | 0 | 0 | 0 | 5.686080 | cage |
| 115 | 0 | 0 | 0 | 1.995375 | book | 1 | 0 | 1 | 1.239589 | book |
| 118 | 1 | 0 | 1 | 1.458072 | prism | 0 | 0 | 0 | 1.651630 | book |
| 131 | 0 | 0 | 0 | 1.867532 | book | 1 | 0 | 1 | 1.199919 | prism |
| 133 | 1 | 0 | 1 | 1.198516 | cage | 0 | 0 | 0 | 4.260464 | cage |
| 134 | 1 | 0 | 1 | 1.108463 | book | 0 | 0 | 0 |  | - |
| 138 | 0 | 0 | 0 | 4.249239 | book | 1 | 0 | 1 | 0.920256 | prism |
| 139 | 0 | 0 | 0 | 3.236934 | cage | 1 | 0 | 1 | 1.412702 | book |
| 140 | 1 | 0 | 1 | 1.181677 | book | 0 | 0 | 0 | 2.116735 | cage |
| 144 | 1 | 0 | 1 | 1.660979 | cage | 0 | 0 | 0 | 1.783422 | prism |
| 150 | 0 | 0 | 0 | 1.447407 | prism | 1 | 0 | 1 | 1.629894 | book |
| 151 | 0 | 0 | 0 | 1.647492 | prism | 1 | 0 | 1 | 1.314652 | cage |
| 153 | 1 | 0 | 1 | 1.628353 | book | 0 | 0 | 0 | 2.490814 | book |
| 158 | 0 | 0 | 0 | 1.434219 | prism | 1 | 0 | 1 | 1.123244 | cage |
| 165 | 1 | 0 | 1 | 1.560803 | book | 0 | 0 | 0 | 1.268337 | book |
| 167 | 0 | 0 | 0 | 6.965134 | book | 1 | 0 | 1 | 1.518211 | cage |
| 168 | 0 | 0 | 0 | 7.093959 | book | 1 | 0 | 1 | 1.431156 | prism |
| 169 | 1 | 0 | 1 | 1.517972 | book | 0 | 0 | 0 | 1.171080 | cage |
| 173 | 1 | 0 | 1 | 1.154948 | prism | 0 | 0 | 0 | 1.921392 | prism |
| 180 | 1 | 0 | 1 | 1.238845 | cage | 0 | 0 | 0 | 2.152499 | prism |
| 181 | 0 | 0 | 0 |  | prism | 1 | 0 | 1 | 2.265898 | cage |
| 183 | 1 | 0 | 1 | 1.409270 | cage | 0 | 0 | 0 | 1.790617 | cage |
| 184 | 1 | 0 | 1 | 2.191708 | prism | 0 | 0 | 0 | 1.450621 | prism |
| 186 | 0 | 0 | 0 | 1.675686 | book | 1 | 0 | 1 | 1.195670 | prism |

## crest_fair: frames

| run | runs | median frames | median intact | median IRA failed | median min RMSD | p10 min RMSD | p90 min RMSD |
|---|---|---|---|---|---|---|---|
| A | 192 | 149 | 46 | 104 | 1.541 | 1.165 | 3.148 |
| B | 192 | 149 | 45 | 104 | 1.610 | 1.232 | 3.049 |

## crest_fair: distance to the nearest reference along the metadynamics runs

| run | frame within its metadynamics run | frames | frames with an RMSD | median RMSD to nearest reference A | p10 A | lowest A |
|---|---|---|---|---|---|---|
| A | 0-2 | 2112 | 0% | 6.898 | 6.737 | 6.732 |
| A | 3-9 | 4032 | 14% | 5.561 | 3.642 | 1.420 |
| A | 10-24 | 8640 | 38% | 4.015 | 2.156 | 1.080 |
| A | 25-48 | 13824 | 49% | 3.227 | 1.874 | 0.821 |
| B | 0-2 | 2112 | 0% | 5.628 | 5.278 | 5.190 |
| B | 3-9 | 4032 | 14% | 5.610 | 3.757 | 1.819 |
| B | 10-24 | 8640 | 41% | 3.902 | 2.129 | 1.180 |
| B | 25-48 | 13824 | 50% | 3.187 | 1.929 | 0.918 |

## crest_fair: classifier effect

frames scored by both builds: 57216; frames where the two builds differ on the gate decision: 0; runs where the hit differs between the builds: 0

## crest_fair: gate, reference set and hit rule

| reference set | A-run gate 0.598 | A-run gate 0.784 | B-run gate 0.598 | B-run gate 0.784 |
|---|---|---|---|---|
| prism | 0/192 | 0/192 | 0/192 | 0/192 |
| cage | 0/192 | 0/192 | 0/192 | 0/192 |
| book | 0/192 | 0/192 | 0/192 | 0/192 |
| prism+cage | 0/192 | 0/192 | 0/192 | 0/192 |
| prism+book | 0/192 | 0/192 | 0/192 | 0/192 |
| all three | 0/192 | 0/192 | 0/192 | 0/192 |

| gate A (frames and polished structure) | A-run hits | B-run hits |
|---|---|---|
| 0.300 | 32/192 | 32/192 |
| 0.400 | 32/192 | 32/192 |
| 0.500 | 32/192 | 32/192 |
| 0.598 | 32/192 | 32/192 |
| 0.700 | 32/192 | 32/192 |
| 0.784 | 34/192 | 39/192 |
| 0.900 | 39/192 | 46/192 |
| 1.000 | 40/192 | 46/192 |
| 1.200 | 68/192 | 57/192 |
| 1.500 | 106/192 | 106/192 |

| run | any frame inside gate | lowest-energy frame inside gate |
|---|---|---|
| A | 0/192 | 0/192 |
| B | 0/192 | 0/192 |
