``Config.replace`` checks the settings that the search asserts on, among
them ``theta``, ``bias_gamma``, ``calibrate_quantile`` and ``length_scale``.
A value outside the range raises ``InvalidArgumentError`` where it is set.
Before, it reached the search and raised ``PanicException``, which is not an
``Exception``. ``shape_keyed=True`` raises ``FeatureUnavailableError`` in a
build without the ``ira`` feature.
