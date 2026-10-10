With ``policy="auto"``, no ``grad_fn`` and no ``noise_sigma``,
``global_optimize`` runs one values-only loop whatever the box width, in place
of the regime-routed Thompson allocation and its DE/GSA front-load. Without
``x0`` it starts from the best of a seeded design of ``dim + 1`` points, so
seeds vary the start. When the budget lets it converge, five finite-difference
gradients per coordinate, a descent from the start opens the run and goes on
while each slice lowers the incumbent by more than the success threshold and
by more than ``1e-4`` times the opening's largest gain. With less, as for a
30-dimensional problem under 4650 evaluations, CMA-ES takes one slice from the
start, the descent keeps the turn until a slice gains less per evaluation than
the one before, and GSA then keeps it while each of its slices gains at least
as fast as the descent would have gone on to (its last gain per evaluation
times the ratio to the one before). Until GSA has played, the descent also
hands it the last slice short of the closing reserve; if GSA's first slice
falls short, the descent keeps the turn until the run has used twice the
evaluations it had. From there GSA and then CMA-ES each keep the turn while
they pay: a phase ends once it has gone eight slices (or as long as its last
gain took) without lowering the incumbent by more than the success threshold,
GSA's phase also once its gain per evaluation since it began is no more than
the descent's last, and a phase slice that gains less than an eighth of the
phase's best per evaluation lends the other phase's arm a slice, which ends
the phase if it gains faster. After an opening the phases stop short of the
closing reserve plus a slice for each other arm not yet played, and without
one short of a slice for DE while it has not played. CMA-ES, the descent, GSA,
DE, the additive surrogate and the QMC restart arm then each take a turn if
they have not yet and the budget left holds eight evaluations beyond the
closing reserve; later rounds pick uniformly with probability ``1/round``,
otherwise replay an arm whose last turn lowered the incumbent, and otherwise
draw by discounted Thompson sampling. A turn is one slice, but the descent
keeps the turn while each slice lowers its value by more than the success
threshold, until it converges, so a kicked descent, or a long one from a poor
start, is scored on the basin it reaches. The descent closes the run with a
reserve of the ten gradients' worth of evaluations a descent needs to converge
(at most 40% of the budget, and none when that leaves fewer than three
gradients' worth), going on with its current descent unless another arm has
lowered the incumbent since it last played, in which case it restarts there.
A slice is four gradients' worth of evaluations or a 48th of the budget,
whichever is larger, but at most the geometric mean of the budget and four
gradients' worth, so past 48 times 48 four-gradient slices the number of
slices keeps growing with the budget. The floor the restart guarantee needs is
asymptotic in the budget: the opening, the phases and the descent's turns go
on only while they lower the incumbent, or the descent's value, by more than
the success threshold, whose positive floor changes at most once, so on a
bounded objective they last a number of slices that does not grow with the
budget, and every arm keeps a uniform share and is pulled infinitely often as
the budget grows.
