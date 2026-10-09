``MoveKernel::displacement`` returns the step ``d`` of a kernel whose proposal
is ``i + d``, drawn with the same calls on the random number generator as
``propose``; the default, ``None``, draws nothing. ``Gaussian`` and ``Cauchy``
implement it, and ``Reflected`` uses it to mirror a sum ``i + d`` that
overflows ``f64`` as the exact sum would be, where the proposal alone reaches
the reflection as ``+-inf``. A symmetric kernel of one's own gets the same
fold by implementing ``displacement``.
