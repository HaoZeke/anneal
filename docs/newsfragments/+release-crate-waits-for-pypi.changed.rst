The release workflow publishes ``anneal-core`` to crates.io only after
the PyPI publish job has succeeded. Building the sdist and the wheels
is not enough: a failed PyPI upload no longer leaves a crates.io
release that cannot be undone.
