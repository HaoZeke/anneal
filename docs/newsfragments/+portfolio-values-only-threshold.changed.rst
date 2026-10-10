In values-only portfolio runs the success threshold, which decides whether the
opening descent goes on, whether a descent keeps its turn, whether a GSA or
CMA-ES phase has gained and when a CMA-ES run stops, is ``1e-4 max(g, s)`` in
place of ``1e-4 max(|f|, 1)``, which turned absolute below ``|f| = 1`` and
coarsened when a constant was added to the objective. ``g`` is the gain of the
last slice that lowered the incumbent from below the start's value, and ``s``
the run's resolution: ``eps |f|`` of the first value the run finds below the
start's, but never less than the smallest positive normal double, which it is
until such a value is found (when the start's value is not finite, every
finite value is below it). Gains are differences of values and a slice from
the start's value records none, so the start's value sets no threshold, and a
constant added to the objective changes one only through the resolution of the
shifted values: the rounding of each gain, and ``s``. ``s`` changes at most
once and never falls below that positive floor, so a bounded objective still
allows only a bounded number of successes, as the restart guarantee needs.
The values-only descent likewise stops on the resolution of its values, an
iteration that lowers its value by less than ``8 eps`` of it being slow, and
until a curvature pair scales the inverse Hessian each line search of a
finite-difference descent starts with a trial whose largest move is 5% of a
side whatever the gradient's size, where a small gradient used to shorten it.
Multiplying the objective by a power of two leaves a run unchanged until the
surrogate or the restart arm plays, whose temperatures are absolute.
