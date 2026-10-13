In a box where no coordinate admits a difference step, the values-only
loop's descent indexed past its empty stencil, so ``minimize`` without
``jac`` and ``global_optimize`` on that loop raised ``PanicException``. The
step at the start is the square root of the machine epsilon times the larger
of the coordinate's magnitude and a tenth of its side, at most a quarter of
the side, and a coordinate admits none when adding the step leaves it
unchanged: on a side one unit in the last place wide, on one two units wide
when half a unit rounds back to the coordinate, and near zero, where the step
underflows, as on ``[0, 1e-316]`` from either end. The descent now ends at
its start when no coordinate admits a step, and the loop ends the run there:
it evaluates ``x0`` once, with one replica or more, or without ``x0`` the
box's centre once per replica, up to the budget.
