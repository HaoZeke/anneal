``ClusterKernels`` is ``#[non_exhaustive]``. A struct literal or a
``match`` on it outside the crate no longer compiles without
``ClusterKernels::new`` and a wildcard arm. A later kernel is not a
further breaking change.
