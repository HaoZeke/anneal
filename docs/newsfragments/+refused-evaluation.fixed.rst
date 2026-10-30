``WarmLbfgs`` backtracks from an evaluation answered with a non-finite
value or gradient, and a refused start ends only that relaxation, at
``+inf``. ``rgmin`` 0.5.0's line search doubled the step on a NaN value,
and a NaN gradient at the start became a NaN direction, so one refused SCF
sent the engine geometries far out and then NaN coordinates.
``RelaxStats::refused`` counts the refusals.
