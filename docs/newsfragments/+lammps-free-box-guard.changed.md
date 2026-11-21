<!-- Copyright 2023--present rgpot developers -->

`rgpot.lammps.FixExternal` now raises `ValueError` when the LAMMPS box is non-periodic in every direction (`boundary f f f`) and the potential reports `caps()["periodic"]`. Such a potential, for example `rgpot.LJPot`, wrapped pairs across the box extents and returned a wrong energy without a warning; use a non-periodic potential such as `rgpot.LJClusterPot`. The new keyword `allow_periodic_on_free_box=True` restores the previous behavior.
