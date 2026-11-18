`EMTPot` no longer keeps force evaluation serial above `rgpot::kTeamMinAtoms`.
It now evaluates the Asap half list in two OpenMP phases, one owner-row pass
and one incoming-partner pass, so the force sum order is fixed and the
energy and forces are bit-identical at 1, 2, 4 and 8 threads.
Below `rgpot::kTeamMinAtoms` it still runs on the calling thread and opens
no OpenMP region.
