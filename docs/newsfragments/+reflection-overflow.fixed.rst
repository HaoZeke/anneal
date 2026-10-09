Box-reflected moves near the float limit keep their symmetry. A proposal
whose sum ``x + s`` overflowed, or whose offset ``x - low`` did, was put on a
wall, which skewed walks in boxes near the largest float toward it. The
overshoot is now measured from the step, so it stays finite, and folded from
the wall it crossed; an infinite step lands uniformly in the box. Ordinary
proposals are unchanged bit for bit.
