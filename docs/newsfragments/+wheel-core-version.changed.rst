The wheel crate manifest names the ``anneal-core`` version on that
dependency. The version is the workspace package version. v0.12.0
depended on the path and did not name a version. The manifest still
sets ``description``, ``license = "MIT"`` and ``publish = false``,
as it did in v0.12.0.
