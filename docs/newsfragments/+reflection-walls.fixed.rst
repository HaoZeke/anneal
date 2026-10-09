Reflection into a box returns a point already inside unchanged, so a point
on the upper wall is no longer folded to ``high`` plus an ulp and evaluated
outside the box. The Tsallis visiting step is formed from logarithms: near
``q_v = 3`` its scale underflowed to zero while ``|y|^{-e}`` overflowed, the
product was NaN, and ``Gsa`` walks stalled with fewer evaluations than the
schedule promised; below ``q_v`` of about 1.007, ``Gamma`` in the scale
overflowed and every step was zero or a tail redraw. ``run`` and ``run_qmc`` take ``max_evals`` to spend an
exact number of objective calls.
