Mirror reflection keeps a finite point finite in a half-infinite Rust box far
from zero, such as ``[1e308, inf)`` or ``(-inf, -1e308]``. There a step of the
boxed ``Boltzmann`` or ``Fast`` preset whose sum overflowed toward the
infinite wall stayed infinite, and ``reflect_coord`` returned ``+-inf`` for a
point whose image across the finite wall lay past ``+-MAX``: ``fast_in_box``
with ``t_init = 1`` and ``gamma = 3e307`` on ``-x``, run for 200 steps from
``1.6e308`` in ``[1e308, inf)``, evaluated the objective at ``+inf`` on all 40
seeds tried, and now does so on none. The infinite wall now stands at
``+-MAX``, the last finite value on its side, and such a sum or image folds
back across it; ``reflect_coord_with_slope`` gives the slope of the fold. A
finite point whose distance from the wall it crossed overflowed, such as
``-8e307`` for ``[1e308, inf)``, stopped on that wall and now folds too, and
so does one far outside a finite Rust box wider than ``MAX / 2``:
``reflect_coord(-0.95 * MAX, 0.1 * MAX, 0.9 * MAX)`` returned ``lo`` and now
returns about ``0.65 * MAX``.
