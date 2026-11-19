# GFN2 (H2O)16 molecular-quench record

The full panel below is the original record on `e2865427`, which was
`fix/rigid-thread-local`, not develop. One serial session used hostname
`bench`, working directory `/work`, and one thread. The comparison arm was
`e2865427` (`/work/bin/anneal-e286`). The quench arm was `169924a6`
(`/work/bin/anneal-quench`): the xtb gradient is stored as an rgpot force, and
a molecular quench takes a capped Lindh step. CREST 3.0.2 is
`crest start.xyz --gfn2 -T 1 --legacy -xnam /tmp/crest-env/bin/xtb -len 0.2`
on the same starts. xtb is conda-forge 6.7.1. Seeds 15000–15005 ran at 4000
and 28000 charged calls, and seed 16001 ran at 28000. Arm order is in the
session script and is shuffled per seed. `target_energy` was off.

The port onto the develop record `0ae5bfe0` changes the molecular path, so
seeds 16001 and 15000 were remeasured at 28000 calls. Those rows are in
`port-results.csv` and summarized first.

A hit is a vtight total energy at or below −81.396292669509 + 1e-6 Eh. The objective energies below are the search objective in eV. The hit column is the separate xtb `--opt vtight` score of the reported geometry. Calls per converged quench are `energy_calls / quenches_converged`.

The engine writes `calls` and monotonic nanoseconds around `rgpot_xtb_force`. Inside milliseconds per call are that nanosecond total divided by those calls. Outside milliseconds per call are `(wall_s − inside_s) / energy_calls` when the time file covers at least 90% of `wall_s`. Two rows are short of that: the 28000-call new arm of seed 15005 (1436 s inside a 4406 s wall) and the 28000-call develop arm of seed 15004 (5888 s inside a 10078 s wall). A restarted worker resets the counter, and those two files keep only the last worker.

## Port remeasure on 0ae5bfe0

The port remeasure used pre-fold tree `f0bac79b`, based on `0ae5bfe0`.
The subsequent fold makes the capped Lindh solver an explicit general option
and enables it in this driver, leaving this water path unchanged. Both seeds
ran serially on hostname `bench`, one thread, with the same starts. The newer
base owns answer-seat polishing, so 1600 of the 28000 calls polish recorded
minima. Neither rerun hits the vtight threshold.

| Seed | Vtight Eh | Objective eV | Wall (s) | Potential (s) | Outside (s) | Converged | Capped | Energy calls / converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16001 | −81.387703963668 | −2214.6722482295 | 853.066 | 808.150 | 44.916 | 353 | 13 | 79.32 |
| 15000 | −81.388961921640 | −2214.7064790098 | 861.531 | 815.176 | 46.354 | 315 | 11 | 88.89 |

The callback sign fix remains necessary on `0ae5bfe0`: that tree publishes
the xtb gradient as an rgpot force while rgpot negates forces to make its
objective gradient.

## What the gradient does

The bench engine stores xtb's ∂E/∂x. On a stretched O–H with the hydrogen at 1.20 Å the engine component is 4.89482241 eV/Å and the finite difference is 4.89513530 eV/Å. rgpot then sets the objective gradient to minus the force buffer. Develop `e2865427` publishes that buffer unchanged, so the objective gradient is minus the true slope and L-BFGS walks uphill. That negation is in develop itself. The earlier double negation lived only in a replay onto the rolled-back `fbd21fed` session, which already stored forces; it is not on this branch and not on `e2865427`.

`3477545e` flips the buffer before rgpot negates it. The two tests in `xtb_gradient` lock the finite-difference slope and the callback. With that sign, the Lindh quench converges. Develop converges none.

## 4000 calls

Median wall is 122.1 s on the quench arm and 8363 s on develop. The quench arm converges 30–50 quenches and caps 1–3. Develop converges 0 and caps 208–229. No 4000-call geometry is a hit. Develop's vtight score stays on the uphill side of the start; the quench arm reaches about −81.38 Eh.

| Seed | Quench Eh | Quench wall (s) | Conv | Cap | E/conv | Develop Eh | Develop wall (s) | Develop cap | CREST Eh | CREST wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15005 | −81.383636 | 120.6 | 50 | 1 | 80.0 | −81.369203 | 7740.5 | 227 | −81.379700 | 340.7 |
| 15004 | −81.384130 | 125.5 | 40 | 1 | 100.0 | −81.318506 | 7528.5 | 213 | −81.344737 | 326.6 |
| 15002 | −81.384484 | 110.3 | 30 | 3 | 133.3 | −81.242129 | 8985.5 | 217 | −81.268935 | 150.9 |
| 15000 | −81.382724 | 122.4 | 44 | 1 | 90.9 | −81.367212 | 13016.8 | 222 | −81.385124 | 337.4 |
| 15001 | −81.382194 | 121.9 | 48 | 2 | 83.3 | −81.356340 | 5493.9 | 208 | −81.384575 | aborted |
| 15003 | −81.379546 | 132.6 | 44 | 1 | 90.9 | −81.308382 | 9663.7 | 229 | −81.357894 | 569.7 |

Seed 15001's CREST exited 1 (`confcross_0.xyz` missing). The Eh above is the vtight score of the `crest_best.xyz` it had already written.

Inside the quench arm a call is 26–31 ms, and about 2 ms of the wall sits outside xtb. Develop's calls are 76–179 ms because the uphill steps fail SCF. Outside xtb is about 5 ms per charged call where the time file covers the wall.

## 28000 calls

Median wall is 4015 s on the quench arm and 7502 s on develop. Develop again converges 0 quenches on every seed (268–298 capped) and never hits. The quench arm's calls per converged quench are 52.0, 58.3, 59.2, 70.0, 72.4, 84.8, and 85.6. The median is 70.0. The old 146.6 figure is 28000/191 from a develop that did converge. This develop does not, so that ratio is not defined here.

| Seed | Quench Eh | Quench wall (s) | Conv | Cap | E/conv | Develop Eh | Develop wall (s) | Develop cap | CREST Eh | CREST wall (s) | Hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15005 | −81.383730 | 4406.4 | 473 | 444 | 59.2 | −81.369203 | 5909.1 | 287 | −81.379700 | 340.7 | neither |
| 15004 | −81.387305 | 4155.4 | 480 | 406 | 58.3 | −81.318506 | 10078.4 | 268 | −81.344737 | 326.6 | neither |
| 15002 | −81.386934 | 4015.4 | 330 | 420 | 84.8 | −81.242129 | 10152.1 | 289 | −81.268935 | 150.9 | neither |
| 15000 | −81.387517 | 868.7 | 400 | 2 | 70.0 | −81.367212 | 6748.8 | 271 | −81.385124 | 337.4 | neither |
| 15001 | −81.388560 | 4142.1 | 538 | 440 | 52.0 | −81.356340 | 7501.5 | 285 | −81.384575 | aborted | neither |
| 15003 | −81.384417 | 913.3 | 327 | 4 | 85.6 | −81.308382 | 13450.8 | 298 | −81.357894 | 569.7 | neither |
| 16001 | −81.396293 | 819.8 | 387 | 4 | 72.4 | −81.366038 | 6945.7 | 293 | −81.383248 | 274.3 | quench |

Seed 16001 is the only hit. The vtight score is −81.396292888382 Eh, 2.2e-7 Eh below the reference. CREST on that start is −81.383247622693 Eh in 274.3 s, and develop is −81.366038420 Eh in 6945.7 s with 0 converged quenches. The quench objective first entered the reference window (−81.396292669509 Eh, the CODATA Hartree energy in eV) at 20316 charged calls, hop 266. Only the final geometry was re-scored with vtight; that score is the hit. Develop has no converged quench and no hit, so there is no develop calls-to-first-hit on this tip.

On the quench rows whose time file covers the wall, a call is 27–43 ms inside xtb and about 2 ms outside it. Develop's covered rows are 62–140 ms inside and about 1 ms outside. The long develop walls are failed SCF cycles on the uphill steps, not time outside xtb.
