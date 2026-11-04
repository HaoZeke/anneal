A molecular quench can be preconditioned with the covalent bond graph (``Config::precondition_bonds``).
The preconditioner has the exponential form of Packwood and co-workers (2016).
It needs ``Config::species`` and is off by default.
On a 30-atom spring chain it settles a quench in fewer than half the calls of the plain descent.
