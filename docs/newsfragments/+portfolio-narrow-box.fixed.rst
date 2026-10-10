``global_optimize``, and with it the default ChemFit portfolio, no longer
panics on a box narrower than about ``2e-7`` in some coordinate. The gradient
arm bounded its central-difference step with ``clamp(1e-8, 0.05 * width)``,
whose floor exceeds its cap on such a coordinate; the step now stays within a
twentieth of the width.
