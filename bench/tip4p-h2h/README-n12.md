# TIP4P, twelve seeds

Seeds 2000–2011, budget 2_000_000 charged evaluations, serial, one core, hostname `bench`, working directory `/work`. One shuffle, `Random(20261113)`. A hit lies at or below the reference plus 0.001 kJ/mol. Medians in `summary-n12.csv` are Hyndman–Fan type 7. GMIN is the fixed step and is not an anneal commit.

`shell`, `ungated` and `mix` ran on `cfa21b1a033e731937de6f0cabe22a451220ab69`. `weight`, `epoch` and `exchange` ran on `e2865427aaaec8b8c558ea5d4df40b3e723d0239`.

| code | commit | arm | (H₂O)₁₅ | (H₂O)₂₀ | (H₂O)₂₁ | total |
| --- | --- | --- | --- | --- | --- | --- |
| shell | `cfa21b1a033e731937de6f0cabe22a451220ab69` | rigid preset, basin index capped at 2048 | 9/12 | 5/12 | 0/12 | 14/36 |
| ungated | `5c9b9c8264cd9395a0f51995836504a6e5568e96` | hydrogen-bond chain reversal in the move library | 11/12 | 2/12 | 2/12 | 15/36 |
| mix | `7b67ba687a6c7f906113da1db41e5b36a55e7685` | that reversal takes one proposal in ten | 9/12 | 3/12 | 0/12 | 12/36 |
| weight | `a48ae427167c9b3e0c59aa71382e874a406a212d` | chain and shell compete, weighted by improvement | 5/12 | 0/12 | 0/12 | 5/36 |
| epoch | `8fb186e21c3e445d4acd2efcc5767554c5534e7e` | shell and chain alternate at each stall, then the lower best is kept | 5/12 | 1/12 | 0/12 | 6/36 |
| exchange | `63a0d1513d3460983db3f337fb488f63742b66aa` | contact-exchange polish on a stalled basin | 8/12 | 4/12 | 0/12 | 12/36 |
| gmin | — | fixed step | 0/12 | 0/12 | 0/12 | 0/36 |

The shell total is 14/36. The chain library reaches 15/36 and drops (H₂O)₂₀ from 5/12 to 2/12. No other arm beats 14/36 without a larger drop.

Per-seed rows are `rows-n12.csv`.
