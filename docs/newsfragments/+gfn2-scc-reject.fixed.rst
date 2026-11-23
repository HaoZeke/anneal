The GFN2 covalent examples no longer record an energy xtb did not
converge. An unconverged self-consistent charge (SCC) cycle (rc -4)
is returned to rgpot as ``RGPOT_SCC_NOT_CONVERGED``, which the
``rgpot-core`` dev-dependency now takes from the engine commit, and
every engine failure, including a rejected answer and a dead worker,
is counted in ``scc_rejected``. The gradient callback gives no
gradient for a refused or NaN call. Three checks now use
``element_call_floor``, the floor ``CallFloor`` opens to: the handle
guard, ``clears_species_sanity`` and ``quench_is_sane``. A water O-H
at 0.957 Å is called, and a geometry the search opened is not
refused. The dimer scan starts outside that floor.
``scripts/covalent_h2h/gfn2_crest.py`` keeps each run's final
geometry under ``finals/`` and rescores it with standalone xtb at the
search's settings. A hit whose standalone SCC cycle does not converge
is recorded as a miss. New rows go to ``rows-v2.csv``; ``rows.csv``
keeps its older header.
