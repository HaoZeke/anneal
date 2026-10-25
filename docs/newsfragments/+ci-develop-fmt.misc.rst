CI runs on pushes to ``develop`` as well as ``main``. ``cargo fmt`` checks
the workspace member only. Vendored ``nng`` is excluded from the
workspace and is not reformatted.
