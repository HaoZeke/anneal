``pt111_heptamer`` quenches the OptBench Pt(111) heptamer with rgmin and
certifies edges with an rgsaddle dimer from each quenched endpoint pair.
The hop and the RMSD use the smallest free layer split at 1.0 angstrom.
Other free layers keep a live gradient. Morse parameters and reference
geometries are read from the OptBench problem file and ``.con`` tree.
``mid.con`` is not treated as a saddle. A match is an energy within
0.05 eV and an RMSD within 0.1 angstrom.
