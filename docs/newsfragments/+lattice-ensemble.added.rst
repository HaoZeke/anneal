``methods::lattice_ensemble`` runs chains that split one force budget and pool
the minima their lattice descents reach in a bank held apart by a radial
resemblance cutoff. Trials start from random clusters, from members with
surface atoms moved onto vacant hollows, or from cut-and-splice pairs of
members. A member drawn twenty times without improving gives its slot to the
next random start that resembles no member, so the bank keeps taking in new
funnels after its first members stop descending. ``Sharing::Private`` gives
each chain its own bank and changes nothing else, as the ablation of the
communication. Chains advance in synchronous generations with parents drawn
and offers admitted in chain order, so a run replays bit for bit at any
thread count.
