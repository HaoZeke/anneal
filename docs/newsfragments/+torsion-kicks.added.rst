``Config::with_torsion_kicks`` adds an allocated ``tkick`` arm when the
quenched start has a rotatable bond: a heavy-atom bond outside a ring
whose ends each carry another atom and neither of which is a methyl or
amino top. A kick turns the smaller side of between one and half of those
bonds by a third of a turn to a half turn, and is redrawn up to eight
times if it changes the covalent graph. ``rotatable_bond_count`` counts
the bonds. The container, the hold radius and the other arms are
unchanged.
