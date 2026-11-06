The water arms now record and quench at ``catalog::molecular::WATER_RECORD_GRADIENT``, the xtb ``vtight`` gradient of 2e-4 Eh/bohr.
At the molecular preset's 5e-2 eV/Å a recorded energy could sit 0.3 eV above its own minimum, so the GFN2 hexamer search reached the CREST minimum and then ranked a higher basin first.
The ``gfn2_water_h2h`` comparison now polishes its answer with ``cluster_search::polish_answer`` toward the catalog's GFN2 stationary-point gate, inside the same evaluation budget, so the scored energy is the minimum's and not the quench's.
