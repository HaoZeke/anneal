``gfn2_covalent_h2h`` sets the rgpot engine's warm move to zero, so each
call starts from a fresh SCC. A warm guess had converged to solutions a
fresh SCC and xtb do not reproduce. The other GFN2 examples keep the
engine's setting.
