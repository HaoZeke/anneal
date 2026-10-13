``run_qmc`` on a box with a fixed coordinate, one whose two ends are equal,
ran its design and its chains in the whole box, every move on that
coordinate reflected back onto its value; it now runs them on the other
coordinates, as ``run`` does, so its design points, moves and results on
such a box differ from before. With every coordinate fixed the run is one
call, where it made ``n_starts * (1 + n_epochs * steps_per_epoch)`` calls,
or ``max_evals``, at the box's one point.
