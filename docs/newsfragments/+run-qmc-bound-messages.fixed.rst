``run_qmc`` names the dimension and the reason when it refuses a box, as
``run`` and the other box drivers do:
``low[0] must not exceed high[0] (got 1 > 0)`` for an inverted axis,
``bounds must be finite, with a finite width, at dimension 0`` for
``[-1e308, 1e308]``, and
``the box is too wide: the sum of its widths is not finite``. It used to say
only that each bound must be finite and each upper bound at least its lower
bound. ``low == high`` still pins that coordinate.
