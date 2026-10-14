``run_qmc`` on a box with a fixed coordinate, one whose two ends are equal,
ran its design and its chains in the whole box, every move on that
coordinate reflected back onto its value; it now runs them on the other
coordinates, as ``run`` does. Its design points are those of before where
every fixed coordinate comes after the free ones, and differ where a fixed
coordinate comes before a free one. They held a coordinate fixed at
``-0.0`` at ``0.0``, and now hold it at the zero ``x0`` has there, or at
``-0.0`` without ``x0``. Its moves differ, and so can its results. With
every coordinate fixed the run is one call, where it made
``n_starts * (1 + n_epochs * steps_per_epoch)`` calls, or ``max_evals``, at
the box's one point.
