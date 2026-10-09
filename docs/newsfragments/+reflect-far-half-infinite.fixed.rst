Mirror reflection keeps a finite point finite in a half-infinite Rust box far
from zero, such as ``[1e308, inf)`` or ``(-inf, -1e308]``. The image across
the finite wall could lie past ``+-MAX``, so ``reflect_coord`` returned
``+-inf`` and the boxed ``Boltzmann`` and ``Fast`` presets evaluated the
objective there: ``fast_in_box`` with ``gamma = 3e307`` on ``-x``, started at
``1.6e308`` in ``[1e308, inf)``, did so on 29 of 40 seeds. A point whose
distance from that wall overflowed, such as ``-8e307`` for ``[1e308, inf)``,
landed on the wall. The infinite wall now stands at ``+-MAX``, the last finite
value on its side, and such an image folds back across it, as a step whose
sum overflows already did; ``reflect_coord_with_slope`` gives the slope of
that fold.
