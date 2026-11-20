A rigid-water search no longer hangs when its objective drives the molecules
on top of one another. ``Config.for_tip4p`` and ``Config.recommended_tip4p``
on a quadratic objective never returned: the hydrogen-bond cycle move turned
two coincident oxygens into NaN, and the rotation fold looped forever on it.
The cycle move now leaves such a state alone, the fold returns for any
angle, and a proposal that is not finite is refused before the objective
sees it. ``pytest/test_rigid_water_degenerate.py`` runs each rigid-water
preset on a quadratic objective in a child process with a time limit.
