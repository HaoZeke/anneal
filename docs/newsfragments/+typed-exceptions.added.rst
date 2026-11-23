Errors raised for rejected input are now typed. ``anneal.AnnealError``
derives from ``ValueError`` and has the subclasses
``InvalidArgumentError``, ``BoundsError``, ``ConfigError``,
``RemovedConfigKeyError``, ``NoFiniteStartError`` (with ``attempts``) and
``FeatureUnavailableError``. ``except ValueError`` clauses keep working.
