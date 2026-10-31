``methods::gmin_hop`` is a basin-hopping chain with the structural moves the literature uses on short-range Morse clusters and rigid water: a surface-vacancy relocation and a Deaven-Ho cut-and-splice for atoms, and for TIP4P a whole-cluster orientation step, single-molecule reorientation, relocation of the worst hydrogen-bonded molecule to the surface and a molecular cut-and-splice.
Every value-and-gradient call is charged.
``ccd_hits`` runs it as the ``gmin`` arm.
On Morse 55 at ``rho = 14`` it does worse than the ``recommended`` preset, 0 of 32 hits against 2 of 32 on seeds 9200 to 9231, so it is not a Morse default.
