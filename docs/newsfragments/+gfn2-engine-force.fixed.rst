The GFN2 callback no longer negates ``rgpot_xtb_force``. That engine
already returns a force, and a second negation made the objective
gradient point uphill.
