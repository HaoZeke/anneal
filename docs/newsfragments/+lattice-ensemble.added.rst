``methods::lattice_ensemble`` runs chains that split one force budget and pool
the minima their lattice descents reach in a bank held apart by a radial
resemblance cutoff, under the bank rule of conformational space annealing
(Lee, Scheraga and Rackovsky, J. Comput. Chem. 18, 1222 (1997)) with the
D_ave/2 merge distance of Lee, Lee and Lee, Phys. Rev. Lett. 91, 080201
(2003), arXiv cond-mat/0307690. Trials start from random clusters, from members
with surface atoms moved onto vacant hollows, or from cut-and-splice pairs of
members (Deaven and Ho, Phys. Rev. Lett. 75, 288 (1995)). A member drawn twenty
times without improving gives its slot to the next random start that resembles
no member, so the bank keeps taking in new funnels after its first members stop
descending. The retirement rule, the radial resemblance key, the held merge
distance and the counted accounting are this work's. ``Sharing::Private``
gives each chain its own bank. A one-slot private bank never splices and never
draws the splice coin, so its random streams part from the shared arm's at the
first bank draw, if not sooner, and the private ablation removes cross-chain
splicing together with the exchange of members. Chains advance in synchronous
generations with parents drawn and offers admitted in chain order, so a run
replays bit for bit at any thread count.
