# Rebased shelf paired gate

Pass means B loses on no cell. A cell is a loss when the exact one-sided Fisher p for fewer hits on B, after Bonferroni, is below 0.05, or when the exact one-sided Mann-Whitney p for higher calls to a hit on B, after the same adjustment, is below 0.05, or when the median outside-potential time per charged call on B divided by the median on A is above 1.1. Bonferroni uses every Fisher test and every Mann-Whitney test that has a hit on both arms. Adjusted p is the raw p times that count, capped at 1. Decisions use the exact rational p. The table shows six decimals. The A/A block is the noise band.

Verdict: **FAIL**.

## Record

- Machine id: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
- Core 0, affinity 0, one run at a time.
- Compiler: rustc 1.88.0 (6b00bc388 2025-06-23)
- Tests in the Bonferroni family: 27.
- Arm A commit `59397594b662be91ba86e07b2e013c6b2c9d8d55` tree `f5589a1b53ed374600b4f25be10963c00e6e9364`.
- Arm B commit `d62a3198b778ba99ef95bb70669de50b4a1f3881` tree `a2471b8f410e30e522ddc0eca49e80a736587d3b`.
- Seeds 2000 through 2011. First arm ABBA. No warm-up.
- Sutton-Chen rows use the in-tree reduced potential. `librgpot.so` was not present, so `h2h_sc` was not the binary.
- Arm A `lj_shelf_gate` 250ccfc2c6f3a0a3a46e07204eba8e43bc0bd51b90c4c8f62c220e2396b62602
- Arm A `sc_shelf_gate` 49c1a0761d4dd8350a10d0438609be2018b94ab3cfb6af33b51692b63cf1d3ed
- Arm A `h2h_morse` f43ae7ca89cb69edc35261833d09ee98f3a3c8c15553f23597d791b0e17d1898
- Arm B `lj_shelf_gate` c6923ee00980162bc4314ded27bbcda86ad211147a837f6c2d37a1c5ed458b34
- Arm B `sc_shelf_gate` 24dfd29891bf0e5b38bcd94c2af9b8f19bb2c47c35b2f6dc82b4e8c247bc1760
- Arm B `h2h_morse` 26967640046c78959bb0e6c34780b953ed5b9eac6e438330df8b7d97e5005d6b

## A/A band

| system | hits A1 | hits A2 | median calls A1 | IQR A1 | median calls A2 | IQR A2 | median difference | outside ratio A2/A1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lj38 | 12/12 | 12/12 | 127839.5000 | 139245.2500 | 127839.5000 | 139245.2500 | 0.0000 | 0.997216 |
| lj75 | 5/12 | 5/12 | 2721351.0000 | 2166318.0000 | 2721351.0000 | 2166318.0000 | 0.0000 | 0.993906 |

A/A band: 0.0000 calls, the larger absolute median-call difference on LJ38 and LJ75.

## Cells

| system | role | budget | hits A | hits B | Fisher p | Fisher adjusted | median calls A | IQR A | median calls B | IQR B | MW p | MW adjusted | outside B/A | loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sc126-38 | sc | 400000 | 11/12 | 12/12 | 1.000000 | 1.000000 | 44447.0000 | 193703.0000 | 53475.0000 | 160056.5000 | 0.459390 | 1.000000 | 0.932339 |  |
| sc126-55 | sc | 1000000 | 12/12 | 12/12 | 1.000000 | 1.000000 | 25529.0000 | 39637.7500 | 25529.0000 | 39637.7500 | 0.522663 | 1.000000 | 0.963903 |  |
| sc126-80 | sc | 1000000 | 4/12 | 6/12 | 0.893122 | 1.000000 | 709780.0000 | 161270.0000 | 696472.0000 | 539227.7500 | 0.457143 | 1.000000 | 0.955252 |  |
| sc108-38 | sc | 400000 | 9/12 | 11/12 | 0.953416 | 1.000000 | 253276.0000 | 144980.0000 | 253276.0000 | 107185.5000 | 0.511396 | 1.000000 | 1.029405 |  |
| sc108-55 | sc | 1000000 | 7/12 | 2/12 | 0.044690 | 1.000000 | 449866.0000 | 291300.0000 | 588670.0000 | 138804.0000 | 0.277778 | 1.000000 | 1.133606 | outside time |
| sc108-80 | sc | 1000000 | 0/12 | 3/12 | 1.000000 | 1.000000 |  |  | 596882.0000 | 55492.5000 |  |  | 0.968355 |  |
| sc96-38 | sc | 400000 | 12/12 | 12/12 | 1.000000 | 1.000000 | 20236.0000 | 100335.2500 | 20236.0000 | 79797.2500 | 0.572791 | 1.000000 | 0.988163 |  |
| sc96-55 | sc | 1000000 | 12/12 | 12/12 | 1.000000 | 1.000000 | 28040.0000 | 31071.5000 | 28788.0000 | 31071.5000 | 0.489176 | 1.000000 | 0.993331 |  |
| sc96-80 | sc | 1000000 | 6/12 | 5/12 | 0.500000 | 1.000000 | 723509.0000 | 229922.0000 | 486583.0000 | 231346.0000 | 0.848485 | 1.000000 | 0.974947 |  |
| lj77 | named | 4000000 | 8/12 | 2/12 | 0.018037 | 0.487010 | 2138360.5000 | 1378537.0000 | 1568604.5000 | 1137994.5000 | 0.733333 | 1.000000 | 0.984795 |  |
| m30r14 | named | 400000 | 8/12 | 12/12 | 1.000000 | 1.000000 | 155782.0000 | 103038.7500 | 155961.0000 | 185626.7500 | 0.389370 | 1.000000 | 0.993607 |  |
| lj38 | no-regression | 400000 | 12/12 | 12/12 | 1.000000 | 1.000000 | 127839.5000 | 139245.2500 | 155428.5000 | 105991.0000 | 0.221417 | 1.000000 | 1.003458 |  |
| lj75 | no-regression | 4000000 | 5/12 | 4/12 | 0.500000 | 1.000000 | 2721351.0000 | 2166318.0000 | 2015754.0000 | 881910.5000 | 0.793651 | 1.000000 | 1.012922 |  |
| lj98 | no-regression | 4000000 | 7/12 | 8/12 | 0.799838 | 1.000000 | 1430688.0000 | 1604559.5000 | 427503.0000 | 389319.5000 | 0.905361 | 1.000000 | 1.018992 |  |

## Losses

- sc108-55

Verdict: **FAIL**.
