The ChemFit bridges refuse a fitter with neither session pair before
``init``, by a message that credited ``ask`` and ``tell`` to ChemFit 3.1, as
the ChemFit docs, docstrings and examples did. They first ship in 3.1.1:
3.1.0 has neither pair, so the message told a 3.1.0 user that their own
release had one. The message, the docs, the docstrings and the examples now
name 3.1.1 as the first release with ``ask`` and ``tell``, and ChemFit's
source from commit c57d6f5, which no release has yet, for ``evaluate`` and
``step``.
