With ``policy="auto"``, no ``grad_fn`` and no ``noise_sigma``,
``global_optimize`` runs one values-only loop whatever the box width, in place
of the regime-routed Thompson allocation and its DE/GSA front-load. Without
``x0`` it starts from the best of a seeded design of ``dim + 1`` points, so
seeds vary the start. When the budget lets it converge, a finite-difference
descent opens the run and goes on while it lowers the incumbent; from the
minimum it reaches, GSA and then CMA-ES each keep the turn until they have
gone eight slices (or as long as their last gain took) without lowering it.
CMA-ES, the descent, GSA, DE, the additive surrogate and the QMC restart arm
then each take a turn if they have not yet; later rounds pick uniformly with
probability ``1/round``, otherwise replay an arm whose last turn lowered the
incumbent, and otherwise draw by discounted Thompson sampling. A turn is one
slice, but the descent keeps the turn while each slice lowers its value by more
than the success threshold, until it converges, so a kicked descent, or a long
one from a poor start, is scored on the basin it reaches. A closing descent from
the incumbent gets the ten gradients' worth of evaluations it needs to converge.
