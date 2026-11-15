A threaded pair-list `build` (`OMP_NUM_THREADS` above 1, at least 8192
atoms) now stores the same pairs as one thread when a pair has more than
one image inside the cutoff, as in a periodic box thinner than two search
radii. Before, it kept only one image of each pair and lost the rest.
