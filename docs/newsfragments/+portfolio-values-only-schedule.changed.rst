With ``policy="auto"``, no ``grad_fn`` and no ``noise_sigma``,
``global_optimize`` runs one values-only loop whatever the box width, in place
of the regime-routed Thompson allocation and its DE/GSA front-load. Without
``x0`` it starts from the best of a seeded design of ``dim + 1`` points, so
seeds vary the start. When the budget lets it converge, five finite-difference
gradients per coordinate, a descent from the start opens the run and goes on
while each slice lowers the incumbent by more than the success threshold and
by more than ``1e-4`` times the largest of the opening's last three gains,
and while its last four slices have gained more, together, than ``1e-4``
times its largest gain; the first slice from below the start is held to the
gain of the slice from the start. Until that descent has
lowered its own value from below the start's, a slice in which it stalls short
of convergence, as when its first curvature pair spans a steep wall, or stops
at the start's value, as behind a sentinel, gives CMA-ES one slice, once, and
the opening goes on from the incumbent CMA-ES leaves. With less, as for a
30-dimensional problem under 4650 evaluations, CMA-ES takes one slice from the
start, the descent keeps the turn until a slice gains less per evaluation than
the one before, and GSA then keeps it while each of its slices gains at least
as fast as the descent would have gone on to (its last gain per evaluation
times the ratio to the one before). Until GSA has played, the descent also
hands it the turn once fewer than three slices are left beyond the closing
reserve; if GSA's first slice falls short, the descent keeps the turn until
the run has used twice the evaluations it had. From there GSA and then CMA-ES
each keep the turn while they pay: a phase ends once it has gone eight slices
(or as long as its last such gain took) without lowering the incumbent by
more than the success threshold and by more than ``1e-4`` times the phase's
largest gain, GSA's first phase also once its gain per evaluation since it
began is no more than the descent's last, and a CMA-ES phase that follows it
without taking the turn likewise from its eighth slice. From its fifth slice
on, a GSA slice that gains less than an eighth of the phase's best per
evaluation lends CMA-ES a slice, which takes the turn if it gains faster than
both that slice and an eighth of the phase's gain per evaluation so far; GSA
lends no other slice until the run has used twice the evaluations it had, and
none after a lent slice that gained nothing. A phase CMA-ES took lends GSA a
slice when its last two slices gain less per evaluation than its lent slice
had to beat, and hands the turn back if GSA's slice gains faster; the two
alternate for up to eight phases. If the bar on the first slice from below
the start ended the opening before the descent converged and GSA's first
phase ends at its first slice, the descent takes a turn before CMA-ES's
phase. After an opening the phases, and that turn, stop short of the closing
reserve plus a slice for each other arm not yet played, counted again at
every slice, and without one short of a slice for DE while it has not played.
CMA-ES, the descent, GSA,
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
the budget grows. Measured over 40 paired seeds against the rules they
replaced (medians, "A against B" with A's wins/ties/losses, rows named
``problem@budget`` from ``x0`` as in the regime notes), these rules trail on
the following rows, and putting a replaced rule back costs the rows named with
it. The rate hand-over short of the opening budget, against two fixed GSA
slices after the descent's first: ``ellS30@1000`` 1673 against 711 (7/0/33),
``ellS50@5000`` 6.88 against 1.52 (2/0/38), ``ellS100@5000`` 1839 against 1110
(2/0/38), ``ellS100@20000`` 1.17 against 0.0282 (0/0/40), ``ellR100@20000``
2.48 against 2.41 (19/0/21), ``rosen100@20000`` 74.9 against 72.0 (17/0/23),
``stybR30@1000`` -1057 against -1057 (4/18/18), ``levy100@20000`` 0.0416
against 1.0e-11 (5/15/20), ``rast100_shift@20000`` 9.95 against 7.96 (9/3/28)
and ``rastS100@20000`` 10.9 against 7.96 (8/3/29); the two slices back cost
``ackS50@5000`` 4.31 against 2.94 and ``ackS100@5000`` 13.2 against 5.94
(1/0/39 each), ``ellR30@1000`` 12557 against 5589 (3/0/37), ``ellR100@5000``
20110 against 15750 (5/0/35) and ``rosen100@5000`` 252 against 168 (7/0/33).
The opening's bar on its recent gains, against an opening that goes on while
each slice is a success: ``ellS30@5000`` 0.0626 against 0 (0/13/27),
``rosen30@5000`` 22.2 against 16.2 (2/1/37) with its copies and controls (22.0
to 22.5 against 15.9 to 16.7), and the ``ellR30@5000`` controls for +1e2, +1e4
and -1e4 (2.8e-11, 2.3e-4 and 8.7e-4 against 3.2e-12, 1.4e-5 and 1.2e-5);
dropping the bar costs ``ackS30@5000`` 3.8e-6 against 6.2e-7 (1/18/21),
``levyR30@5000`` 3.51 against 0.544 (5/0/35) and ``levy30@5000`` 7.2e-7
against 4.5e-10 (6/17/17). ``ellR30@5000``, 3.4e-8 against 0 (0/19/21), and
its x1e-6 and x2^-20 copies come level only with both the opening without its
bar and the descent's ``1e-12`` tolerance (0, 0/38/2), at the costs of each.
Lending from a GSA phase's fifth slice: ``mich10@5000`` -9.655 against -9.66
(3/28/9, p 0.021), which no tried rule brings level without putting other rows
behind: lending from any slow slice puts ``mich10@5000`` without ``x0``,
``rast10_shift@1000`` and ``styb10_45@1000`` without ``x0`` behind, from the
fourth slice ``rast10_fixed@1000`` and ``schwefel10_cec@1000``, a lent slice
held to a quarter of the phase's pace ``ackley10@1000`` (1.16 against 0.0694),
and ending a phase CMA-ES took once its slices slow ``ackS30@5000``. With
phases that keep the turn while they still find lower basins, the fifth-slice
lend also costs ``ackS30@20000``, 8.6e-7 against 6.3e-7 (12/1/27, p 0.048),
and ``ackley10@5000`` without ``x0``, 4.1e-11 against 3.2e-11 (6/23/11), with
its -1e4 copy and two controls. Rotated ellipsoids short of the opening budget
trail an earlier form of the loop: ``ellR50@1000`` 1.38e5 against 4.79e4,
``ellR50@5000`` 37.7 against 9.20, ``ellR100@20000`` 2.48 against 0.633 and
``ellR30@1000`` 5589 against 2987; letting the descent keep the turn while it
pays brings ``ellR50@5000`` to 17.4 and ``ellR100@20000`` to 0.890 but costs
``ellR30@1000`` (20880), ``ellR100@5000`` (41370), ``ackS50@5000`` (3.26),
``ackS100@5000`` (9.86), ``levy100@5000``, ``rosen50@5000`` and
``stybS100@5000``. Against 0.10.0 the loop trails on rotated Rastrigin and
Levy: ``rastR30@1000`` 281 against 173 (3/0/37), ``rastR50@5000`` 427 against
296 (1/0/39), ``rastR100@5000`` 893 against 650 (9/0/31), ``rastR100@20000``
870 against 470 (1/0/39), ``levyR50@5000`` 98.8 against 69.8 (8/0/32) and
``levyR100@20000`` 203 against 108 (0/0/40); the diffusion population in DE's
place and a rotation-invariant DE leave each of them behind, and CMA-ES
restarting each phase with four times its population brings ``rastR50@5000``
level (313) and ``rastR100@20000`` to 749 but costs ``ackS30@5000`` (2.01
against 6.2e-7).
