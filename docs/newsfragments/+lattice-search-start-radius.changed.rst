``lattice_search::run`` now draws starts with the same liquid-density
radius and 0.85 preset pair floor as cluster configuration. A finite
``min_separation`` above 0.85 scales the radius by
``min_separation / 0.85``; a value at or below 0.85 leaves the radius
unscaled. This changes starts for every ``min_separation`` above 0.7:
values through 0.85 are no longer enlarged, and larger values use the
pair floor rather than the liquid density as their length scale.
