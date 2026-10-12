In a box with no side wide enough for a difference step to change its
coordinate, a unit or two in the last place, the values-only loop's descent
indexed past its empty stencil, so ``minimize`` without ``jac`` and
``global_optimize`` on that loop raised ``PanicException``. The descent now
ends at its start when no coordinate admits a step, and the loop ends the run
there: it evaluates ``x0`` once, with one replica or more, or without ``x0``
the box's centre once per replica, up to the budget.
