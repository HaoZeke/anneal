<!-- Copyright 2023--present rgpot developers -->

The lazy ASE 3.29 calculator plugins keep the module ASE binds when `rgpot.ase.__ase_plugins__` is first read, including when several threads read that attribute at once.
