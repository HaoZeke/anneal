<!-- Copyright 2023--present rgpot developers -->

A cached metatomic engine no longer serves results computed under a previous charge or spin after `rgpot_engine_set_charge_spin`. The metatomic potential now fingerprints the settings its model sees, so a change of charge or spin is a cache miss with a new key.
