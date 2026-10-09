The Boltzmann, Fast and GSA presets mirror-reflect every proposal into the
objective's box, so ``run`` and ``run_qmc`` only evaluate points inside
``[low, high]``. Equal bounds pin a coordinate, and an empty, non-finite or
inverted box, or one whose width ``high - low`` overflows, raises
``ValueError``. Both drivers accept an optional start point ``x0``, which is
the first point evaluated, and read ``low``, ``high`` and ``x0`` from any
array-like of numbers; a multi-dimensional ``x0`` is flattened in C order. A
NaN objective value counts as worse than every number, so a chain no longer
stalls at a NaN start, and a NaN never beats a number as a chain's best value
or as the best start of ``run_qmc``. In Rust, the ``BoltzmannVariant``,
``FastVariant`` and ``GsaVariant`` aliases now pair the ``BoxConstrained``
neighbourhood with the ``Reflected<Gaussian>``, ``Reflected<Cauchy>`` and
``Reflected<TsallisVisit>`` kernels, in place of ``ContinuousR_n`` with the
bare ``Gaussian``, ``Cauchy`` and ``TsallisVisit`` kernels.
