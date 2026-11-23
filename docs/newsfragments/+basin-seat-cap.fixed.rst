A capped ``BasinIndex`` reuses its latest seat once ``with_cap`` or
``set_cap`` is full. ``push``, ``basin_of`` and a ``BasinBias`` deposit
follow that seat, which is what a rigid chain does past 2048 basins.
