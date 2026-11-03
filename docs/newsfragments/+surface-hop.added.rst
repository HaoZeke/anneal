``Config::surface_hop_on_best`` adds surface hops to the cluster hop.
Each time the run finds a new lowest structure, every least-bound atom is tried on every vacant hollow site where it keeps its neighbour count.
Every try is quenched, and the descent follows the lowest product while it goes down.
``packing::surface_hops`` builds the tries.
The flag is off by default, and a configuration without it has the same resolved digest as before.
