The macOS test harness links libpython and does not pass ``-undefined dynamic_lookup``.
With that flag and no libpython, dyld stops the feature tests before they start because ``_PyBaseObject_Type`` is missing from the flat namespace.
Maturin still builds the extension with that lookup. The interpreter that imports the extension provides the symbols.
