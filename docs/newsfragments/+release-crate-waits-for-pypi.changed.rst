The release workflow publishes ``anneal-core`` to crates.io only after the
sdist and every wheel are built. A tag that cannot reach PyPI no longer
leaves a crates.io release that cannot be undone.
