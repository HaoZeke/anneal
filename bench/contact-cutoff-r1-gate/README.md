# contact-cutoff-r1 gate

Verdict: PASS.

- Part 1, bit identity at length one: PASS.
- Part 2, non-unit-length cells: PASS.
- Special casing: none.

## Arms

Arms were checked by tree, never by commit id, before the first run and after the last.

- DEVELOP: develop 61fa67cf, tree `cff796c507a8fa1d0bfb28888c3570b61988055e`.
- CC: develop plus `git am -3` of the contact-cutoff-r1 mbox, tree `c5b961b7e51e83ae54d3f748f166ed74f0d5a432`.
- DEVELOP': a second, separate release build of develop, tree `cff796c507a8fa1d0bfb28888c3570b61988055e`.

Machine id (sha256): `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`. Harness blob: `a3f7e22457e705fa69a63c364ae9b287cf76e85e`.

## Protocol

Release builds. Every run is `taskset -c 0`, with every thread variable at 1, and no `H2H_*` or `ANNEAL_*` variable set. Budgets are the paper budgets. Runs are serial. Within a seed the arms run in order on even seed positions and reversed on odd ones (ABBA). No warm-up and no shortened run. A hit is a finite best at or below the reference plus 1e-3. `calls` is calls to the hit, a miss ranked above every hit. `best` is the driver's printed value (`h2h_seed` prints 10 decimals, the other drivers 8), and `best_bits` is the IEEE-754 bits of that printed value. Outside per call is `outside_s / charged` per row. Medians are Hyndman-Fan type 7. Fisher and Mann-Whitney p are exact and two-sided.

Rules, declared before the runs: a Part 2 cell fails if CC has fewer hits than DEVELOP, if the calls Mann-Whitney p is below 0.05 with CC ranked worse, or if CC/DEVELOP outside per call is above 1.10 and outside the A/A spread (the larger of DEVELOP'/DEVELOP and its reciprocal). Gains are the mirror images.

## Part 1: bit identity at length one

LJ38, LJ55 and LJ75 at 400000, 1000000 and 4000000 calls, seeds 2000-2002, through the panel driver `h2h_gmin_proof` and the seed driver `h2h_seed`. The seed driver stops at the reference plus 1e-3 and writes the trace. `h2h_gmin_proof` writes no trace.

| Cell | Seed | best DEVELOP | best CC | best_bits | calls | charged | trace digest | Identical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38/h2h_gmin_proof | 2000 | -173.92842659 | -173.92842659 | same | 124155 / 124155 | 124155 / 124155 | n/a | yes |
| lj38/h2h_gmin_proof | 2001 | -173.92842659 | -173.92842659 | same | 111509 / 111509 | 111509 / 111509 | n/a | yes |
| lj38/h2h_gmin_proof | 2002 | -173.92842659 | -173.92842659 | same | 293837 / 293837 | 293837 / 293837 | n/a | yes |
| lj55/h2h_gmin_proof | 2000 | -279.24847046 | -279.24847046 | same | 5067 / 5067 | 5067 / 5067 | n/a | yes |
| lj55/h2h_gmin_proof | 2001 | -279.24847046 | -279.24847046 | same | 7542 / 7542 | 7570 / 7570 | n/a | yes |
| lj55/h2h_gmin_proof | 2002 | -279.24847046 | -279.24847046 | same | 2147 / 2147 | 2147 / 2147 | n/a | yes |
| lj75/h2h_gmin_proof | 2000 | -396.28224883 | -396.28224883 | same | 4000000 / 4000000 | 4000000 / 4000000 | n/a | yes |
| lj75/h2h_gmin_proof | 2001 | -397.49233098 | -397.49233098 | same | 3327845 / 3327845 | 3327845 / 3327845 | n/a | yes |
| lj75/h2h_gmin_proof | 2002 | -396.28224883 | -396.28224883 | same | 4000000 / 4000000 | 4000000 / 4000000 | n/a | yes |
| lj38/h2h_seed | 2000 | -173.9276311710 | -173.9276311710 | same | 287 / 287 | 287 / 287 | same | yes |
| lj38/h2h_seed | 2001 | -173.9282656691 | -173.9282656691 | same | 1262 / 1262 | 1265 / 1265 | same | yes |
| lj38/h2h_seed | 2002 | -173.9280379068 | -173.9280379068 | same | 1963 / 1963 | 1963 / 1963 | same | yes |
| lj55/h2h_seed | 2000 | -279.2479072953 | -279.2479072953 | same | 2640 / 2640 | 2643 / 2643 | same | yes |
| lj55/h2h_seed | 2001 | -279.2484353516 | -279.2484353516 | same | 782 / 782 | 789 / 789 | same | yes |
| lj55/h2h_seed | 2002 | -266.2394531766 | -266.2394531766 | same | 2250 / 2250 | 2253 / 2253 | same | yes |
| lj75/h2h_seed | 2000 | -396.2822487878 | -396.2822487878 | same | 143083 / 143083 | 144127 / 144127 | same | yes |
| lj75/h2h_seed | 2001 | -392.4290772008 | -392.4290772008 | same | 2345 / 2345 | 2356 / 2356 | same | yes |
| lj75/h2h_seed | 2002 | -392.4290772239 | -392.4290772239 | same | 18971 / 18971 | 18996 / 18996 | same | yes |

## Part 2: cells whose units are not Lennard-Jones units

Part 2 covers every campaign cell whose search does not run on Lennard-Jones lengths, as far as this machine can run them. The spec length is `SOAP_CUTOFF * cfg.length_scale` (`moves.rs:1171`). The Sutton-Chen panel (`h2h_sc`, nine cells in Cambridge units, nearest neighbour a/sqrt(2)), Au55 on `h2h_metal`, and the four Morse cells of the GMIN panel on `h2h_morse` all run `Config::recommended(n)`, so their `length_scale` is 1 even though their units are not Lennard-Jones units. The ASE Morse panel (`ase_nonlj`, `PairKind::Morse.length_scale()` = 2^(-1/6)) and ASE Au55 (`ase_panel`, Gupta dimer scales) run at a length other than 1, under both presets those runners call.

Not run: `gfn2_covalent_h2h`, which always uses measured scales, because no xtb engine is on this machine. `h2h_covalent`'s `general` arm needs `H2H_ARM`, which the protocol forbids; its default arm is `Config::recommended` at length 1. `sc_h2h`, `sc_arm3` and `morse_table` use scaled configs, but no campaign rule runs them.

Seeds 2000-2011, n=12 per arm. D is DEVELOP, D' is DEVELOP'. Hits are CC against D. The Mann-Whitney test is on calls, CC against D.

| Cell | Length | Hits D | Hits CC | Hits D' | Fisher p | MW p (calls) | CC calls rank | D out/call (s) | CC out/call (s) | CC/D | D'/D (A/A) | A/A spread | Seeds same CC=D | Seeds same D'=D | Fail | Gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sc126-38 | 1 | 12/12 | 12/12 | 12/12 | 1 | 1 | tied | 2.543415e-06 | 2.507942e-06 | 0.9861 | 1.0053 | 1.0053 | 12/12 | 12/12 | no | no |
| sc126-55 | 1 | 12/12 | 12/12 | 12/12 | 1 | 1 | tied | 4.906124e-06 | 4.892050e-06 | 0.9971 | 0.9547 | 1.0474 | 12/12 | 12/12 | no | no |
| sc126-80 | 1 | 5/12 | 5/12 | 5/12 | 1 | 1 | tied | 1.189742e-05 | 1.202096e-05 | 1.0104 | 0.9832 | 1.0171 | 12/12 | 12/12 | no | no |
| sc108-38 | 1 | 7/12 | 7/12 | 7/12 | 1 | 1 | tied | 3.030587e-06 | 3.028125e-06 | 0.9992 | 1.0126 | 1.0126 | 12/12 | 12/12 | no | no |
| sc108-55 | 1 | 5/12 | 5/12 | 5/12 | 1 | 1 | tied | 5.590185e-06 | 5.588091e-06 | 0.9996 | 1.0147 | 1.0147 | 12/12 | 12/12 | no | no |
| sc108-80 | 1 | 3/12 | 3/12 | 3/12 | 1 | 1 | tied | 1.096666e-05 | 1.081538e-05 | 0.9862 | 0.9854 | 1.0148 | 12/12 | 12/12 | no | no |
| sc96-38 | 1 | 12/12 | 12/12 | 12/12 | 1 | 1 | tied | 2.465710e-06 | 2.394637e-06 | 0.9712 | 0.9665 | 1.0346 | 12/12 | 12/12 | no | no |
| sc96-55 | 1 | 12/12 | 12/12 | 12/12 | 1 | 1 | tied | 4.258742e-06 | 4.321629e-06 | 1.0148 | 1.0136 | 1.0136 | 12/12 | 12/12 | no | no |
| sc96-80 | 1 | 7/12 | 7/12 | 7/12 | 1 | 1 | tied | 1.031191e-05 | 1.034887e-05 | 1.0036 | 0.9837 | 1.0166 | 12/12 | 12/12 | no | no |
| au55/h2h_metal | 1 | 4/12 | 4/12 | 4/12 | 1 | 1 | tied | 9.334978e-06 | 9.316095e-06 | 0.9980 | 1.0093 | 1.0093 | 12/12 | 12/12 | no | no |
| m55r6/h2h_morse | 1 | 12/12 | 12/12 | 12/12 | 1 | 1 | tied | 4.972994e-06 | 4.970299e-06 | 0.9995 | 0.9969 | 1.0031 | 12/12 | 12/12 | no | no |
| m55r14/h2h_morse | 1 | 9/12 | 9/12 | 9/12 | 1 | 1 | tied | 4.352720e-06 | 4.348924e-06 | 0.9991 | 1.0018 | 1.0018 | 12/12 | 12/12 | no | no |
| m80r6/h2h_morse | 1 | 12/12 | 12/12 | 12/12 | 1 | 1 | tied | 1.163481e-05 | 1.169448e-05 | 1.0051 | 1.0067 | 1.0067 | 12/12 | 12/12 | no | no |
| m80r14/h2h_morse | 1 | 6/12 | 6/12 | 6/12 | 1 | 1 | tied | 9.263878e-06 | 9.343098e-06 | 1.0086 | 1.0087 | 1.0087 | 12/12 | 12/12 | no | no |
| m38r6/ase_nonlj/recommended | 2^(-1/6) | 12/12 | 12/12 | 12/12 | 1 | 1 | tied | 2.347981e-06 | 2.373747e-06 | 1.0110 | 0.9972 | 1.0028 | 12/12 | 12/12 | no | no |
| m38r14/ase_nonlj/recommended | 2^(-1/6) | 9/12 | 9/12 | 9/12 | 1 | 1 | tied | 1.989660e-06 | 2.012918e-06 | 1.0117 | 0.9994 | 1.0006 | 12/12 | 12/12 | no | no |
| m55r6/ase_nonlj/recommended | 2^(-1/6) | 12/12 | 12/12 | 12/12 | 1 | 1 | tied | 4.971548e-06 | 5.017066e-06 | 1.0092 | 0.9997 | 1.0003 | 12/12 | 12/12 | no | no |
| m55r14/ase_nonlj/recommended | 2^(-1/6) | 8/12 | 8/12 | 8/12 | 1 | 1 | tied | 3.216110e-06 | 3.242303e-06 | 1.0081 | 0.9990 | 1.0010 | 12/12 | 12/12 | no | no |
| m38r6/ase_nonlj/for_cluster | 2^(-1/6) | 5/12 | 5/12 | 5/12 | 1 | 1 | tied | 1.579151e-06 | 1.596385e-06 | 1.0109 | 0.9946 | 1.0055 | 12/12 | 12/12 | no | no |
| m38r14/ase_nonlj/for_cluster | 2^(-1/6) | 0/12 | 0/12 | 0/12 | 1 | 1 | tied | 1.522287e-06 | 1.542211e-06 | 1.0131 | 0.9991 | 1.0009 | 12/12 | 12/12 | no | no |
| m55r6/ase_nonlj/for_cluster | 2^(-1/6) | 12/12 | 12/12 | 12/12 | 1 | 1 | tied | 1.901235e-06 | 1.894714e-06 | 0.9966 | 0.9793 | 1.0211 | 12/12 | 12/12 | no | no |
| m55r14/ase_nonlj/for_cluster | 2^(-1/6) | 0/12 | 0/12 | 0/12 | 1 | 1 | tied | 2.724151e-06 | 2.738875e-06 | 1.0054 | 1.0023 | 1.0023 | 12/12 | 12/12 | no | no |
| au55/ase_panel/recommended | Gupta dimer | 0/12 | 0/12 | 0/12 | 1 | 1 | tied | 7.319887e-06 | 7.301839e-06 | 0.9975 | 0.9916 | 1.0085 | 12/12 | 12/12 | no | no |
| au55/ase_panel/for_cluster | Gupta dimer | 0/12 | 0/12 | 0/12 | 1 | 1 | tied | 5.297839e-06 | 5.341146e-06 | 1.0082 | 1.0032 | 1.0032 | 12/12 | 12/12 | no | no |

Failing cells: none.

Gains: none.

Rows where the driver's hit flag differs from the harness score: none.

### Per seed

| Cell | Seed | best DEVELOP | best CC | best DEVELOP' | best_bits CC=D | charged CC=D | best_bits D'=D | charged D'=D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sc126-38 | 2000 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2001 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2002 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2003 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2004 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2005 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2006 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2007 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2008 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2009 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2010 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-38 | 2011 | -35419.98044844 | -35419.98044844 | -35419.98044844 | yes | yes | yes | yes |
| sc126-55 | 2000 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2001 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2002 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2003 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2004 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2005 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2006 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2007 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2008 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2009 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2010 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-55 | 2011 | -52884.68057761 | -52884.68057761 | -52884.68057761 | yes | yes | yes | yes |
| sc126-80 | 2000 | -78402.02236302 | -78402.02236302 | -78402.02236302 | yes | yes | yes | yes |
| sc126-80 | 2001 | -78414.62711523 | -78414.62711523 | -78414.62711523 | yes | yes | yes | yes |
| sc126-80 | 2002 | -78414.62711781 | -78414.62711781 | -78414.62711781 | yes | yes | yes | yes |
| sc126-80 | 2003 | -78402.02236302 | -78402.02236302 | -78402.02236302 | yes | yes | yes | yes |
| sc126-80 | 2004 | -78395.08746305 | -78395.08746305 | -78395.08746305 | yes | yes | yes | yes |
| sc126-80 | 2005 | -78395.08746305 | -78395.08746305 | -78395.08746305 | yes | yes | yes | yes |
| sc126-80 | 2006 | -78402.02236302 | -78402.02236302 | -78402.02236302 | yes | yes | yes | yes |
| sc126-80 | 2007 | -78414.62711781 | -78414.62711781 | -78414.62711781 | yes | yes | yes | yes |
| sc126-80 | 2008 | -78414.62711563 | -78414.62711563 | -78414.62711563 | yes | yes | yes | yes |
| sc126-80 | 2009 | -78395.08746305 | -78395.08746305 | -78395.08746305 | yes | yes | yes | yes |
| sc126-80 | 2010 | -78398.46825105 | -78398.46825105 | -78398.46825105 | yes | yes | yes | yes |
| sc126-80 | 2011 | -78414.62711781 | -78414.62711781 | -78414.62711781 | yes | yes | yes | yes |
| sc108-38 | 2000 | -10117.24540728 | -10117.24540728 | -10117.24540728 | yes | yes | yes | yes |
| sc108-38 | 2001 | -10108.55984979 | -10108.55984979 | -10108.55984979 | yes | yes | yes | yes |
| sc108-38 | 2002 | -10108.55984979 | -10108.55984979 | -10108.55984979 | yes | yes | yes | yes |
| sc108-38 | 2003 | -10117.24540728 | -10117.24540728 | -10117.24540728 | yes | yes | yes | yes |
| sc108-38 | 2004 | -10117.24540728 | -10117.24540728 | -10117.24540728 | yes | yes | yes | yes |
| sc108-38 | 2005 | -10108.55984979 | -10108.55984979 | -10108.55984979 | yes | yes | yes | yes |
| sc108-38 | 2006 | -10117.24540728 | -10117.24540728 | -10117.24540728 | yes | yes | yes | yes |
| sc108-38 | 2007 | -10117.24540728 | -10117.24540728 | -10117.24540728 | yes | yes | yes | yes |
| sc108-38 | 2008 | -10117.24540728 | -10117.24540728 | -10117.24540728 | yes | yes | yes | yes |
| sc108-38 | 2009 | -10117.24540728 | -10117.24540728 | -10117.24540728 | yes | yes | yes | yes |
| sc108-38 | 2010 | -10108.55984979 | -10108.55984979 | -10108.55984979 | yes | yes | yes | yes |
| sc108-38 | 2011 | -10108.55984979 | -10108.55984979 | -10108.55984979 | yes | yes | yes | yes |
| sc108-55 | 2000 | -14810.32143639 | -14810.32143639 | -14810.32143639 | yes | yes | yes | yes |
| sc108-55 | 2001 | -14814.52246011 | -14814.52246011 | -14814.52246011 | yes | yes | yes | yes |
| sc108-55 | 2002 | -14813.19742369 | -14813.19742369 | -14813.19742369 | yes | yes | yes | yes |
| sc108-55 | 2003 | -14812.12379972 | -14812.12379972 | -14812.12379972 | yes | yes | yes | yes |
| sc108-55 | 2004 | -14813.19742369 | -14813.19742369 | -14813.19742369 | yes | yes | yes | yes |
| sc108-55 | 2005 | -14810.32143639 | -14810.32143639 | -14810.32143639 | yes | yes | yes | yes |
| sc108-55 | 2006 | -14814.52248272 | -14814.52248272 | -14814.52248272 | yes | yes | yes | yes |
| sc108-55 | 2007 | -14814.52244035 | -14814.52244035 | -14814.52244035 | yes | yes | yes | yes |
| sc108-55 | 2008 | -14814.52251585 | -14814.52251585 | -14814.52251585 | yes | yes | yes | yes |
| sc108-55 | 2009 | -14813.41110581 | -14813.41110581 | -14813.41110581 | yes | yes | yes | yes |
| sc108-55 | 2010 | -14814.52247089 | -14814.52247089 | -14814.52247089 | yes | yes | yes | yes |
| sc108-55 | 2011 | -14810.32143639 | -14810.32143639 | -14810.32143639 | yes | yes | yes | yes |
| sc108-80 | 2000 | -21819.99520285 | -21819.99520285 | -21819.99520285 | yes | yes | yes | yes |
| sc108-80 | 2001 | -21819.99520285 | -21819.99520285 | -21819.99520285 | yes | yes | yes | yes |
| sc108-80 | 2002 | -21811.87721924 | -21811.87721924 | -21811.87721924 | yes | yes | yes | yes |
| sc108-80 | 2003 | -21817.99676517 | -21817.99676517 | -21817.99676517 | yes | yes | yes | yes |
| sc108-80 | 2004 | -21819.99520285 | -21819.99520285 | -21819.99520285 | yes | yes | yes | yes |
| sc108-80 | 2005 | -21811.87721924 | -21811.87721924 | -21811.87721924 | yes | yes | yes | yes |
| sc108-80 | 2006 | -21811.87721924 | -21811.87721924 | -21811.87721924 | yes | yes | yes | yes |
| sc108-80 | 2007 | -21811.87721924 | -21811.87721924 | -21811.87721924 | yes | yes | yes | yes |
| sc108-80 | 2008 | -21811.87721924 | -21811.87721924 | -21811.87721924 | yes | yes | yes | yes |
| sc108-80 | 2009 | -21811.87721924 | -21811.87721924 | -21811.87721924 | yes | yes | yes | yes |
| sc108-80 | 2010 | -21817.99676517 | -21817.99676517 | -21817.99676517 | yes | yes | yes | yes |
| sc108-80 | 2011 | -21817.99676517 | -21817.99676517 | -21817.99676517 | yes | yes | yes | yes |
| sc96-38 | 2000 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2001 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2002 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2003 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2004 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2005 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2006 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2007 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2008 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2009 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2010 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-38 | 2011 | -8917.70562657 | -8917.70562657 | -8917.70562657 | yes | yes | yes | yes |
| sc96-55 | 2000 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2001 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2002 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2003 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2004 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2005 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2006 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2007 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2008 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2009 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2010 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-55 | 2011 | -13217.89626990 | -13217.89626990 | -13217.89626990 | yes | yes | yes | yes |
| sc96-80 | 2000 | -19558.46750594 | -19558.46750594 | -19558.46750594 | yes | yes | yes | yes |
| sc96-80 | 2001 | -19560.89655231 | -19560.89655231 | -19560.89655231 | yes | yes | yes | yes |
| sc96-80 | 2002 | -19560.89655131 | -19560.89655131 | -19560.89655131 | yes | yes | yes | yes |
| sc96-80 | 2003 | -19557.50715899 | -19557.50715899 | -19557.50715899 | yes | yes | yes | yes |
| sc96-80 | 2004 | -19560.89655355 | -19560.89655355 | -19560.89655355 | yes | yes | yes | yes |
| sc96-80 | 2005 | -19558.46750594 | -19558.46750594 | -19558.46750594 | yes | yes | yes | yes |
| sc96-80 | 2006 | -19560.89655080 | -19560.89655080 | -19560.89655080 | yes | yes | yes | yes |
| sc96-80 | 2007 | -19560.89655795 | -19560.89655795 | -19560.89655795 | yes | yes | yes | yes |
| sc96-80 | 2008 | -19559.45532466 | -19559.45532466 | -19559.45532466 | yes | yes | yes | yes |
| sc96-80 | 2009 | -19553.64572771 | -19553.64572771 | -19553.64572771 | yes | yes | yes | yes |
| sc96-80 | 2010 | -19560.89655353 | -19560.89655353 | -19560.89655353 | yes | yes | yes | yes |
| sc96-80 | 2011 | -19560.89655169 | -19560.89655169 | -19560.89655169 | yes | yes | yes | yes |
| au55/h2h_metal | 2000 | -190.98001732 | -190.98001732 | -190.98001732 | yes | yes | yes | yes |
| au55/h2h_metal | 2001 | -191.00616498 | -191.00616498 | -191.00616498 | yes | yes | yes | yes |
| au55/h2h_metal | 2002 | -191.00270211 | -191.00270211 | -191.00270211 | yes | yes | yes | yes |
| au55/h2h_metal | 2003 | -191.00616498 | -191.00616498 | -191.00616498 | yes | yes | yes | yes |
| au55/h2h_metal | 2004 | -191.00616498 | -191.00616498 | -191.00616498 | yes | yes | yes | yes |
| au55/h2h_metal | 2005 | -191.03085280 | -191.03085280 | -191.03085280 | yes | yes | yes | yes |
| au55/h2h_metal | 2006 | -191.00616498 | -191.00616498 | -191.00616498 | yes | yes | yes | yes |
| au55/h2h_metal | 2007 | -191.00616498 | -191.00616498 | -191.00616498 | yes | yes | yes | yes |
| au55/h2h_metal | 2008 | -191.03085280 | -191.03085280 | -191.03085280 | yes | yes | yes | yes |
| au55/h2h_metal | 2009 | -190.98001732 | -190.98001732 | -190.98001732 | yes | yes | yes | yes |
| au55/h2h_metal | 2010 | -191.03085280 | -191.03085280 | -191.03085280 | yes | yes | yes | yes |
| au55/h2h_metal | 2011 | -191.03085280 | -191.03085280 | -191.03085280 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2000 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2001 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2002 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2003 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2004 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2005 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2006 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2007 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2008 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2009 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2010 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/h2h_morse | 2011 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2000 | -220.64585570 | -220.64585570 | -220.64585570 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2001 | -220.64585570 | -220.64585570 | -220.64585570 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2002 | -220.51853338 | -220.51853338 | -220.51853338 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2003 | -220.64620756 | -220.64620756 | -220.64620756 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2004 | -220.64620756 | -220.64620756 | -220.64620756 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2005 | -220.49847976 | -220.49847976 | -220.49847976 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2006 | -220.64620756 | -220.64620756 | -220.64620756 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2007 | -220.64620756 | -220.64620756 | -220.64620756 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2008 | -220.64620756 | -220.64620756 | -220.64620756 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2009 | -220.51853338 | -220.51853338 | -220.51853338 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2010 | -220.64620756 | -220.64620756 | -220.64620756 | yes | yes | yes | yes |
| m55r14/h2h_morse | 2011 | -220.64585570 | -220.64585570 | -220.64585570 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2000 | -378.33346188 | -378.33346188 | -378.33346188 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2001 | -378.33346544 | -378.33346544 | -378.33346544 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2002 | -378.33347107 | -378.33347107 | -378.33347107 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2003 | -378.33346256 | -378.33346256 | -378.33346256 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2004 | -378.33346065 | -378.33346065 | -378.33346065 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2005 | -378.33347107 | -378.33347107 | -378.33347107 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2006 | -378.33347107 | -378.33347107 | -378.33347107 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2007 | -378.33347107 | -378.33347107 | -378.33347107 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2008 | -378.33346095 | -378.33346095 | -378.33346095 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2009 | -378.33347107 | -378.33347107 | -378.33347107 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2010 | -378.33347107 | -378.33347107 | -378.33347107 | yes | yes | yes | yes |
| m80r6/h2h_morse | 2011 | -378.33346460 | -378.33346460 | -378.33346460 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2000 | -340.81137066 | -340.81137066 | -340.81137066 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2001 | -340.81137066 | -340.81137066 | -340.81137066 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2002 | -340.80256253 | -340.80256253 | -340.80256253 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2003 | -340.81137066 | -340.81137066 | -340.81137066 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2004 | -340.38733557 | -340.38733557 | -340.38733557 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2005 | -340.38733557 | -340.38733557 | -340.38733557 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2006 | -340.81137066 | -340.81137066 | -340.81137066 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2007 | -340.81137066 | -340.81137066 | -340.81137066 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2008 | -340.35713133 | -340.35713133 | -340.35713133 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2009 | -340.80256253 | -340.80256253 | -340.80256253 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2010 | -340.81137066 | -340.81137066 | -340.81137066 | yes | yes | yes | yes |
| m80r14/h2h_morse | 2011 | -340.38733557 | -340.38733557 | -340.38733557 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2000 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2001 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2002 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2003 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2004 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2005 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2006 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2007 | -157.47710809 | -157.47710809 | -157.47710809 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2008 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2009 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2010 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/recommended | 2011 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2000 | -144.32105392 | -144.32105392 | -144.32105392 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2001 | -142.69789782 | -142.69789782 | -142.69789782 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2002 | -142.71594037 | -142.71594037 | -142.71594037 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2003 | -144.32105392 | -144.32105392 | -144.32105392 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2004 | -144.32105392 | -144.32105392 | -144.32105392 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2005 | -144.32105392 | -144.32105392 | -144.32105392 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2006 | -144.32105392 | -144.32105392 | -144.32105392 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2007 | -144.32105392 | -144.32105392 | -144.32105392 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2008 | -142.71594037 | -142.71594037 | -142.71594037 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2009 | -144.32105392 | -144.32105392 | -144.32105392 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2010 | -144.32105392 | -144.32105392 | -144.32105392 | yes | yes | yes | yes |
| m38r14/ase_nonlj/recommended | 2011 | -144.32105392 | -144.32105392 | -144.32105392 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2000 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2001 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2002 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2003 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2004 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2005 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2006 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2007 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2008 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2009 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2010 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/recommended | 2011 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2000 | -220.64620756 | -220.64620756 | -220.64620756 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2001 | -220.64620756 | -220.64620756 | -220.64620756 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2002 | -220.51853338 | -220.51853338 | -220.51853338 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2003 | -220.49847976 | -220.49847976 | -220.49847976 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2004 | -220.64585570 | -220.64585570 | -220.64585570 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2005 | -220.64585570 | -220.64585570 | -220.64585570 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2006 | -220.49847976 | -220.49847976 | -220.49847976 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2007 | -220.64585570 | -220.64585570 | -220.64585570 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2008 | -219.94332884 | -219.94332884 | -219.94332884 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2009 | -220.64585570 | -220.64585570 | -220.64585570 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2010 | -220.64585570 | -220.64585570 | -220.64585570 | yes | yes | yes | yes |
| m55r14/ase_nonlj/recommended | 2011 | -220.64585570 | -220.64585570 | -220.64585570 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2000 | -157.24229009 | -157.24229009 | -157.24229009 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2001 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2002 | -157.24229009 | -157.24229009 | -157.24229009 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2003 | -156.42460637 | -156.42460637 | -156.42460637 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2004 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2005 | -157.40690196 | -157.40690196 | -157.40690196 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2006 | -157.24229009 | -157.24229009 | -157.24229009 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2007 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2008 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2009 | -157.24229009 | -157.24229009 | -157.24229009 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2010 | -157.40690196 | -157.40690196 | -157.40690196 | yes | yes | yes | yes |
| m38r6/ase_nonlj/for_cluster | 2011 | -157.47710811 | -157.47710811 | -157.47710811 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2000 | -141.76724806 | -141.76724806 | -141.76724806 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2001 | -140.32810680 | -140.32810680 | -140.32810680 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2002 | -139.98937551 | -139.98937551 | -139.98937551 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2003 | -142.69779698 | -142.69779698 | -142.69779698 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2004 | -140.27994954 | -140.27994954 | -140.27994954 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2005 | -139.98937551 | -139.98937551 | -139.98937551 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2006 | -139.42590344 | -139.42590344 | -139.42590344 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2007 | -139.65178678 | -139.65178678 | -139.65178678 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2008 | -140.32745506 | -140.32745506 | -140.32745506 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2009 | -142.65336022 | -142.65336022 | -142.65336022 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2010 | -141.71575568 | -141.71575568 | -141.71575568 | yes | yes | yes | yes |
| m38r14/ase_nonlj/for_cluster | 2011 | -140.29584757 | -140.29584757 | -140.29584757 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2000 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2001 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2002 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2003 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2004 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2005 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2006 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2007 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2008 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2009 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2010 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r6/ase_nonlj/for_cluster | 2011 | -250.28660879 | -250.28660879 | -250.28660879 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2000 | -218.52620516 | -218.52620516 | -218.52620516 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2001 | -218.52347271 | -218.52347271 | -218.52347271 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2002 | -218.53575067 | -218.53575067 | -218.53575067 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2003 | -219.94332884 | -219.94332884 | -219.94332884 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2004 | -218.53168703 | -218.53168703 | -218.53168703 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2005 | -218.80404477 | -218.80404477 | -218.80404477 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2006 | -218.53088095 | -218.53088095 | -218.53088095 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2007 | -220.51853338 | -220.51853338 | -220.51853338 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2008 | -218.53557705 | -218.53557705 | -218.53557705 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2009 | -219.52361935 | -219.52361935 | -219.52361935 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2010 | -219.52611477 | -219.52611477 | -219.52611477 | yes | yes | yes | yes |
| m55r14/ase_nonlj/for_cluster | 2011 | -218.83636804 | -218.83636804 | -218.83636804 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2000 | -190.98001732 | -190.98001732 | -190.98001732 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2001 | -191.00616498 | -191.00616498 | -191.00616498 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2002 | -191.00367602 | -191.00367602 | -191.00367602 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2003 | -190.98001732 | -190.98001732 | -190.98001732 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2004 | -190.98001732 | -190.98001732 | -190.98001732 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2005 | -190.97692561 | -190.97692561 | -190.97692561 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2006 | -190.97692561 | -190.97692561 | -190.97692561 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2007 | -190.98001732 | -190.98001732 | -190.98001732 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2008 | -191.00270211 | -191.00270211 | -191.00270211 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2009 | -191.00270211 | -191.00270211 | -191.00270211 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2010 | -190.98001732 | -190.98001732 | -190.98001732 | yes | yes | yes | yes |
| au55/ase_panel/recommended | 2011 | -190.98331173 | -190.98331173 | -190.98331173 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2000 | -190.76399611 | -190.76399611 | -190.76399611 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2001 | -190.62477842 | -190.62477842 | -190.62477842 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2002 | -190.80812536 | -190.80812536 | -190.80812536 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2003 | -190.95198734 | -190.95198734 | -190.95198734 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2004 | -190.81948301 | -190.81948301 | -190.81948301 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2005 | -190.73621308 | -190.73621308 | -190.73621308 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2006 | -190.74444477 | -190.74444477 | -190.74444477 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2007 | -190.89041761 | -190.89041761 | -190.89041761 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2008 | -190.81353567 | -190.81353567 | -190.81353567 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2009 | -190.97692561 | -190.97692561 | -190.97692561 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2010 | -190.98001732 | -190.98001732 | -190.98001732 | yes | yes | yes | yes |
| au55/ase_panel/for_cluster | 2011 | -190.76095495 | -190.76095495 | -190.76095495 | yes | yes | yes | yes |

## Special casing

The patch changes one line of library code, `src/soap.rs:86` in the CC tree:

```
    CONTACT_CUTOFF * (spec.rcut_nn / SoapSpec::default().rcut_nn)
```

It replaces `let _ = spec; CONTACT_CUTOFF` (develop `src/soap.rs:86-87`). The other two files are a news fragment and `tests/contact_cutoff_length.rs`. No added line names a system, a size, a potential or a cell, and there is no new branch.

## Reach

On develop, `soap::contact_cutoff` is read by `target_421`, `atom_w555`, `class_masses` and `ih_dominated` (`src/soap.rs:1306`, `:1413`, `:1434`, `:1454`). Within the search those are reached two ways. `fivefold_openings` calls `ih_dominated(x, SoapSpec::default())` (`src/soap.rs:2522`), which is length one in both arms. `ClusterMove::Soap` calls `soap::step_away` (`src/methods/cluster_hopping/moves.rs:2638`) only when `class` is set, and `class` is `cfg.soap_class_residual && packing` (`moves.rs:1172`). Every preset sets `soap_class_residual` to false (`src/methods/cluster_hopping/config.rs:1835`, `:1943`, `:2228`, `:2428`, `:2548`), and no campaign driver sets it to true. Under the canonical stack, then, no search on any of these cells passes a non-default spec to `contact_cutoff`.

## Files

- `gate.py` runs the arms and appends `rows.csv`.
- `session.json` holds the trees, binary and library digests, and the hashed machine id.
- `verdict.py` writes this file and `summary.csv` from the rows.
