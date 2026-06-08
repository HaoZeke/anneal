The ``anneal_core::grad`` module (and its public re-exports of ``Gradient``, ``AnalyticGradient``, and ``FiniteDiffGradient``) has been removed entirely.

Gradient support is now provided exclusively by the ``eindir_core`` primitives crate (on which anneal is built):

- ``eindir_core::Gradient``
- ``eindir_core::DifferentiableObjective``
- ``eindir_core::AnalyticGradient``
- ``eindir_core::FiniteDiffGradient``
- The full ``eindir_core::gradient`` module

All drivers (HMC/NUTS, GLE-Langevin, projected-gradient polish and QMC variants, etc.) continue to work unchanged. Internal code and tests now import directly from ``eindir_core``.

Python users should supply gradients via ``grad_fn`` when constructing ``eindir.PyObjective`` (or pass plain callables to the anneal wrappers, which continue to work).

See the eindir documentation (cross-linked via intersphinx) for the full story, including analytic gradients on the surrogates and best practices for supplying native derivatives from Python frameworks.
