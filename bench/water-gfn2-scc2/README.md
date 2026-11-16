# GFN2 water, SCC rejection and CREST

Both measurements ran under `sudo unshare --uts` with hostname `bench`, one thread, after sourcing `/work/h2o20-moves/bench-env.sh`. xtb is 6.7.1 (edcfbbe), sha256 `1e3cea0331e155559f8fca6035c87c1225ac2db3e2fa3dbcd684733645534590`. CREST is 3.0.2, commit 6914a25, sha256 `16c5de4578000dfd8ea5758da9f5c497778be0dd7a6a1c1e8933030479d0e56e`.

The fixed binary is `gfn2_water_h2h` from `8b4a2592`, sha256 `2facc837831df4681a14843efc9aea96253088cbf2c39314472c457538e3d123`. The unfixed binary is the `04ae2ea0` search, sha256 `c7f4e29ed47a174a8dfedec38716e9984ad1b987bdc560ce412af19926ff2e4d`.

## The seeds that stalled

Paper budget 4000. The fixed runs wrote `run.txt`. The unfixed runs were stopped with SIGTERM (exit 143) while the rejection log was still growing, so they have no driver wall and no inside/outside split.

Hexamer seed 5, fixed, finished in 18.542 s: 4000 calls, 0 SCC rejections, 18.499 s inside the potential and 0.043 s outside, −829.852228 eV.

Hexamer seed 5, unfixed, was still rejecting one geometry when it was stopped. The log closed at 4935 SCC rejections, each late line at a closest contact of 0.958130 Å. Process elapsed time was 2 h 16 min 49 s when the counter read 4931. Four further rejections were logged before the signal was recorded.

(H2O)12 seed 5, fixed, finished in 67.168 s: 4000 calls, 0 SCC rejections, 67.100 s inside and 0.067 s outside, −1660.643015 eV.

(H2O)12 seed 5, unfixed, ran 621 s from the shell start line to SIGTERM. It logged 126 SCC rejections. The closest contact stayed in 0.956701–0.957045 Å.

| Molecules | Code | Wall (s) | Calls | SCC rejected | Inside (s) | Outside (s) | Energy (eV) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | fixed | 18.542 | 4000 | 0 | 18.499 | 0.043 | −829.852228 | ran |
| 6 | unfixed |  |  | 4935 |  |  |  | stopped at 0.958130 Å |
| 12 | fixed | 67.168 | 4000 | 0 | 67.100 | 0.067 | −1660.643015 | ran |
| 12 | unfixed |  |  | 126 |  |  |  | stopped, 0.956701–0.957045 Å |

## CREST at the same call cap

Sizes 6, 8 and 12, seeds 0–5, on the fixed binary. CREST is `crest start.xyz --gfn2 -T 1 --legacy -xnam` a counting wrapper, started from the anneal geometry. The cap counts one `* total energy` per xtb invocation, plus the finished meta-dynamics steps. A finished block is one that printed `Meta-MTD N finished`; its steps are `MD time /ps` × 1000 / `dt /fs`. Anneal then spends that many potential calls on the same start. The two arms record the same call count on every seed.

CREST's wall is its own `CREST runtime (total)`. The wrapper's `date` deltas sum past that runtime on some seeds, so the potential column in `crest.csv` is the wrapper sum clamped to the wall, and the outside column is the remainder. Anneal's three times are the driver's counters.

A shared basin is |ΔE| < 1.5 meV. CREST's last `CREGEN> E lowest` is converted at 1 Eh = 27.211386245988 eV. Δ is anneal minus CREST, so a negative Δ is anneal lower. The median of six seeds is the mean of the 3rd and 4th sorted values.

On the fixed search, SCC rejections were 0 except (H2O)12 seed 0 (2) and seed 1 (10). Those two still finished the cap.

### (H2O)6

All six seeds share a basin. CREST is lower by 0.085 meV. The median wall is 1105.739 s for CREST and 896.558 s for anneal. Anneal's median time outside the potential is 2.569 s.

| Seed | Calls | CREST eV | Anneal eV | Δ (meV) | CREST wall (s) | Anneal wall (s) | Basin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 202370 | −829.852313 | −829.852228 | +0.085 | 1032.734 | 840.872 | same |
| 1 | 195596 | −829.852313 | −829.852228 | +0.085 | 988.555 | 814.370 | same |
| 2 | 208796 | −829.852313 | −829.852228 | +0.085 | 1066.916 | 871.797 | same |
| 3 | 221691 | −829.852313 | −829.852228 | +0.085 | 1144.562 | 921.319 | same |
| 4 | 237800 | −829.852313 | −829.852228 | +0.085 | 1202.191 | 1018.115 | same |
| 5 | 241480 | −829.852313 | −829.852228 | +0.085 | 1366.760 | 1071.566 | same |

### (H2O)8

Seeds 0–4 share a basin, CREST lower by 0.108 meV. Seed 5 is 2.341 meV lower for anneal, outside the window. The median wall is 1854.558 s for CREST and 1599.466 s for anneal. The median Δ is +0.108 meV. Anneal's median time outside the potential is 2.791 s.

| Seed | Calls | CREST eV | Anneal eV | Δ (meV) | CREST wall (s) | Anneal wall (s) | Basin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 185555 | −1106.995656 | −1106.995547 | +0.108 | 1845.551 | 1598.157 | same |
| 1 | 207317 | −1106.995656 | −1106.995547 | +0.108 | 1963.067 | 1707.639 | same |
| 2 | 210715 | −1106.995656 | −1106.995547 | +0.108 | 1987.198 | 1734.135 | same |
| 3 | 164471 | −1106.995656 | −1106.995547 | +0.108 | 1559.807 | 1346.328 | same |
| 4 | 195572 | −1106.995656 | −1106.995547 | +0.108 | 1863.565 | 1600.774 | same |
| 5 | 145273 | −1106.993207 | −1106.995547 | −2.341 | 1394.109 | 1193.132 | different |

### (H2O)12

Seed 1 is the one shared basin, anneal lower by 0.039 meV. Anneal is lower on four seeds. Seed 3 is the wide gap: anneal −1660.934286 eV against CREST −1659.771021 eV, Δ −1163.265 meV. Seeds 2 and 5 share the anneal energy −1660.718036 eV and sit about 214 meV above CREST. The median wall is 4514.498 s for CREST and 3945.037 s for anneal. The median Δ is −0.825 meV. Anneal's median time outside the potential is 5.694 s.

| Seed | Calls | CREST eV | Anneal eV | Δ (meV) | CREST wall (s) | Anneal wall (s) | Basin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 187434 | −1660.932675 | −1660.934286 | −1.611 | 3544.515 | 3052.272 | different |
| 1 | 294713 | −1660.842606 | −1660.842645 | −0.039 | 5434.144 | 4591.528 | same |
| 2 | 263614 | −1660.932675 | −1660.718036 | +214.639 | 4865.416 | 4108.111 | different |
| 3 | 244543 | −1659.771021 | −1660.934286 | −1163.265 | 4163.581 | 3781.963 | different |
| 4 | 291306 | −1660.842606 | −1660.934286 | −91.680 | 5347.912 | 4935.282 | different |
| 5 | 151471 | −1660.931043 | −1660.718036 | +213.007 | 3259.672 | 2707.306 | different |

`snakemake -s workflow/Snakefile water_gfn2_scc2` repeats the comparison. It is not part of `rule all`. Setting `UNFIXED_BIN` also reruns the two stall seeds.
