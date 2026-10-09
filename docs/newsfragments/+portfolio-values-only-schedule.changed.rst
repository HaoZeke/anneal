With ``policy="auto"``, no ``grad_fn`` and no ``noise_sigma``,
``global_optimize`` runs one values-only loop whatever the box width, in place
of the regime-routed Thompson allocation and its DE/GSA front-load. Without
``x0`` it starts from the best of a seeded design of ``dim + 1`` points, so
seeds vary the start. When the budget lets it converge, a finite-difference
descent opens the run and goes on while it lowers the incumbent; from the
minimum it reaches, GSA and then CMA-ES each keep the turn until they have
gone eight slices (or as long as their last gain took) without lowering it.
With less, as for a 30-dimensional problem under 4650 evaluations, CMA-ES and
the descent take one slice each from the start, GSA two, and a descent turn
settles the basin GSA reached before the GSA and CMA-ES phases, which then
leave DE a slice. CMA-ES, the descent, GSA, DE, the additive surrogate and the
QMC restart arm then each take a turn if they have not yet and the budget left
holds a slice beyond the closing reserve; later rounds pick uniformly with
probability ``1/round``, otherwise replay an arm whose last turn lowered the
incumbent, and otherwise draw by discounted Thompson sampling. A turn is one
slice, but the descent keeps the turn while each slice lowers its value by more
than the success threshold, until it converges, so a kicked descent, or a long
one from a poor start, is scored on the basin it reaches. The descent closes
the run with a reserve of the ten gradients' worth of evaluations a descent
needs to converge (at most 40% of the budget), going on with its current
descent unless another arm has lowered the incumbent since it last played, in
which case it restarts there. A slice is four gradients' worth of evaluations
or a 48th of the budget, whichever is larger, but at most the geometric mean of
the budget and four gradients' worth, so past 48 times 48 four-gradient slices
the number of slices keeps growing with the budget. The floor the restart
guarantee needs is asymptotic: the opening runs while it improves by the
success threshold, which cannot last on a bounded objective, and after that
every arm keeps a uniform share and is pulled infinitely often as the budget
grows.
