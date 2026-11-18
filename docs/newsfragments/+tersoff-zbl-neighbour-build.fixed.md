Tersoff and ZBL build a stored neighbour list on the same OpenMP atom slices as the other classical potentials once a system has 8192 atoms and the stencil holds at least 4096 candidates, more than one per atom.
The index matches one thread.
An open axis whose nominal grid does not fit in memory is binned on the atom span, so a cluster in a huge box keeps a real cell list.
