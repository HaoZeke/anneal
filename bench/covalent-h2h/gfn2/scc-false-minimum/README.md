Copyright (c) anneal developers

# A false SCC minimum in rgpot's xtb engine

Anneal's Si10 seed 2009 at `f3d6a050` (`gfn2/rows.csv`) records a hit at
call 745 with -492.59511796 eV, 31.9 eV below the reference -460.68373809
eV. That energy is the engine's, not the geometry's.

- rgpot `8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`,
  `CppCore/rgpot/XTBPot`, `libxtb_engine.so` sha256
  `4dac51020e4f1419fa930d95894b581012dabf47681b3a883cba21063f7e8672`.
- xtb 6.7.1 (`edcfbbe`, conda-forge), GFN2, accuracy 0.01, 500
  iterations, electronic temperature 3000 K, neutral, closed shell, warm
  move 1.0 Å (the engine default).

`calls.xyz` holds three of the engine's force calls in that run, 605, 606
and 621 of 702, at full precision. `python reproduce.py
<libxtb_engine.so>` evaluates them in order after `rgpot_xtb_reset`, then
each alone after a reset:

| call | in sequence / eV | reused | alone / eV | xtb `--sp` / eV | min pair / Å |
|---|---|---|---|---|---|
| 605 | -458.2274678747 | 0 | -458.2274678747 | -458.227448 | 2.207 |
| 606 | -175.5869857336 | 1 | -61.9047257744 | -61.904723 | 1.168 |
| 621 | -492.5951179571 | 1 | -449.0503535512 | -449.050334 | 1.784 |

The xtb column is `xtb frame.xyz --gfn 2 --sp --etemp 3000 --acc 0.01
--iterations 500` on each frame: -16.839548846661, -2.274956711392 and
-16.502296115731 Eh converted with the CODATA Hartree energy in eV. The largest atom moves are
0.717 Å from 605 to 606 and 0.586 Å from 606 to 621, both inside the warm
move, so the engine passes 605's converged density to 606 as the guess,
and 606's to 621. Call 606 is a compressed geometry, one Si-Si pair at
1.17 Å, and from 605's density its SCC converges 113.7 eV below the cold
solution. From that density 621 converges 43.5 eV below. After a cold
606 alone, 621 gives -448.7214 eV, a third solution. In the run
calls 606-621 are one quench whose energy falls from -175.59 to -492.60
eV with the wavefunction reused at every step; anneal stops at 621 because
it is below the reference. The last calls of the run, the same geometry
again, give -449.0503535512 eV.

Seed 2009 replays call for call: the anneal binary at `f3d6a050`
(`gfn2_covalent_h2h si10 2009 100000 -460.68373809 best.xyz`, with
`RGPOT_XTB_ETEMP=3000`) through `xtb_engine_log.c` in place of the engine
gives the same row, 745 calls and -492.59511796 eV, and the log these three
frames come from. Starting the replay at call 605 or earlier gives
-492.5951179571 eV at 621. Starting it at call 606 or later does not.
