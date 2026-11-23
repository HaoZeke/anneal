``lattice_search::run`` now scales the radius of its random starts when
``min_separation`` is wider than the Lennard-Jones preset pair floor of
0.85. The radius is multiplied by ``min_separation / 0.85``, so starts
spread out for larger separations. At or below 0.85, including the
default 0.5, the radius is unchanged. A non-finite ``min_separation``
also leaves it unchanged.
