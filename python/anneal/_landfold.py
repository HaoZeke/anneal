"""Sketch-map embedding of a minima book.

``landfold(rows)`` folds an ``(n, D)`` array of descriptor rows.
``landfold_book(paths)`` reads ``.min`` or multi-frame ``.xyz`` books,
describes each minimum by its DECAF class histogram, and folds those
rows with the L1 distance.

Both return a dict with ``xy`` ``(n, 2)``, ``landmarks``, ``sigma``,
``sigma_low``, ``high``, ``low``, ``stress``, ``spectrum`` (lambda1 and
lambda2 of the landmark MDS start) and ``map``. ``map.project(rows)``
places held-out rows against the fixed landmarks. ``landfold_book``
adds ``energy``, ``rows``, ``family`` and ``community``.

A seed fixes the map bit for bit at any ``threads``.
"""

import os

import numpy as np

from anneal._core import LandfoldMap
from anneal._core import landfold as _core_landfold
from anneal._core import landfold_book as _core_landfold_book

__all__ = ["LandfoldMap", "landfold", "landfold_book"]


def _options(kwargs):
    for key in ("high", "low"):
        if key in kwargs and kwargs[key] is not None:
            a, b = kwargs[key]
            kwargs[key] = (float(a), float(b))
    if kwargs.get("mix") is not None:
        kwargs["mix"] = [float(m) for m in kwargs["mix"]]
    return kwargs


def landfold(rows, **kwargs):
    """Fold descriptor rows onto the plane.

    Keyword options: ``landmarks`` (400), ``seed`` (0), ``sigma`` (auto),
    ``sigma_scale`` (0.5), ``sigma_low`` (sigma), ``high`` ((2, 4)),
    ``low`` ((2, 4)), ``grid`` (21), ``iterations`` (200), ``mix``
    ((1, 0.5, 0.25, 0.1, 0)), ``weight_power`` (0.5), ``metric``
    ("l1" or "l2") and ``threads`` (all cores).
    """
    arr = np.ascontiguousarray(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"rows must be a 2-D array, not {arr.ndim}-D")
    return _core_landfold(arr, **_options(kwargs))


def landfold_book(paths, **kwargs):
    """Read minima books and fold their DECAF rows.

    ``paths`` is one path or a sequence of paths, read in order. The
    options of :func:`landfold` apply, apart from ``metric``.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return _core_landfold_book([os.fspath(p) for p in paths], **_options(kwargs))
