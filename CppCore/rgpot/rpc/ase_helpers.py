#!/usr/bin/env python3
"""Ergonomic ASE helpers for the rgpot + anneal + readcon stack.

These wrap the lower-level pieces so a user works in plain ASE terms:

* :class:`AnnealRunner` -- a driver that is not an ``ase.optimize.Optimizer``.
  Construct it with an :class:`ase.Atoms` that already has a calculator, call
  :meth:`run`, and the atoms are relaxed in place by anneal's global portfolio
  optimizer using the calculator's energy and forces.
  :class:`AnnealOptimizer` is the deprecated name of the same driver.

* :func:`atoms_from_con` / :func:`atoms_to_con` / :func:`atoms_to_calc` --
  load and write atomic structures as EON ``.con`` files via ``readcon``
  (readcon-core's Python module). Loading keeps the
  ``SinglePointCalculator`` that ``to_ase`` attached unless the caller passes
  ``calc``. Writing uses ``ConFrame.from_ase`` and ``write_con`` with
  precision omitted and canonical left off, and does not open a corpus.

The heavy dependencies (``anneal``, ``readcon``) are imported lazily so importing
this module only needs ASE + numpy.
"""

from __future__ import annotations

from ase import Atoms


def __getattr__(name: str):
    # The rgpot package imports its compiled _core, which the CON helpers
    # do not need.
    if name in ("AnnealOptimizer", "AnnealRunner"):
        from rgpot.ase import anneal

        return getattr(anneal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def atoms_from_con(path, index: int = -1, *, calc=None, cell=None, pbc=None) -> Atoms:
    """Read an EON ``.con`` file into an :class:`ase.Atoms` via readcon-core.

    Uses readcon's native ASE export (``read_con_as_ase``), so by default the
    cell is taken from the ``.con`` cellpar and element symbols are resolved for
    you. ``cell`` / ``pbc`` override those defaults when given -- handy when the
    ``.con`` cell is absent, or when a cluster needs a large non-periodic box.

    A frame that stored forces comes back with the ``SinglePointCalculator``
    ``to_ase`` attached, including the frame energy when that calculator
    carries one. That calculator stays unless ``calc`` is passed.

    Parameters
    ----------
    path:
        Path to the ``.con`` file.
    index:
        Frame index to return (default last frame).
    calc:
        Optional calculator to attach to the returned atoms.
    cell:
        Optional ASE cell override (anything ``Atoms.set_cell`` accepts).
    pbc:
        Optional periodic-boundary override (bool or 3-tuple).
    """
    import readcon  # lazy: readcon-core's Python module

    atoms = readcon.read_con_as_ase(str(path))[index]
    if cell is not None:
        atoms.set_cell(cell)
    if pbc is not None:
        atoms.set_pbc(pbc)
    if calc is not None:
        atoms.calc = calc
    return atoms


def atoms_to_con(path, atoms: Atoms) -> None:
    """Write ``atoms`` as one EON ``.con`` frame.

    ``ConFrame.from_ase`` then ``write_con`` with precision omitted and
    canonical left off. No corpus.
    """
    import readcon  # lazy: readcon-core's Python module

    frame = readcon.ConFrame.from_ase(atoms)
    readcon.write_con(str(path), [frame])


def atoms_to_calc(atoms: Atoms, calc) -> Atoms:
    """Attach ``calc`` to ``atoms`` and return it (fluent helper)."""
    atoms.calc = calc
    return atoms
