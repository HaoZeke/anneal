# Chain ensemble record

Au55 and rigid TIP4P (H2O)20, seeds 2000-2011, four chains.
comm shares one minimum history. indep keeps a private history
per replica. single spends the whole budget on one replica.
The Au55 budget is 1000000 calls. The TIP4P budget is 2000000
calls. Au55 single hit 3 of 12 seeds. The other Au55 arms and
every TIP4P arm missed. Rows, the success curve, and the tests
are in this directory.

The openconf ligand (rq3) was not run: rgpot has no MMFF potential, and run_ensemble needs an energy and gradient.
