Central differences in the values-only finite-difference descents (the ``qn``
arm and the closing polish) adapt their interval to the curvature each stencil
measures. After the first central stencil at a coordinate, its next interval is
a tenth of the distance ``|g / c|`` to the minimum of the quadratic through the
stencil, never above the ``eps^(1/3)`` rule and never below ``eps^(2/3)`` times
the coordinate's scale or the rounding limit ``sqrt(eps |f| / |c|)``. A failed
central line search re-differences at the same point when an interval has
moved by more than a factor of two, at most twice per iterate. Fits whose
valleys are narrower than the ``eps^(1/3)`` interval, such as
``binary_lj_fit``, now descend to the noise floor instead of stalling where the
stencil straddles the valley.
