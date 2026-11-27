Every search that reports a charge reads ``Ledger::charged`` and sets
``settled_beyond_cap`` through ``Outcome::carry_charge``, on the early
returns as well as the last one. ``gmin_hop::hop`` reported no charge
when its opening quench was refused. The DLS, lattice, bank, archive and
binary outcomes report the overrun, and the record rows, the ledger's
record time and the checkpoint's charged count read the charged total.
A held packing's record row names the hop it was found on.
