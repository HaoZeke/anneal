``methods::lattice_ensemble`` runs chains that split one force budget and pool
the minima their lattice descents reach in a bank held apart by an annealed
radial resemblance. Trials start from random clusters, from members with
surface atoms moved onto vacant hollows, or from cut-and-splice pairs of
members. ``Sharing::Private`` gives each chain its own bank and changes
nothing else, as the ablation of the communication. Chains advance in
synchronous generations with parents drawn and offers admitted in chain order,
so a run replays bit for bit at any thread count.
