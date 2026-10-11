"""The Python blocks of the pages that drive a ChemFit Fitter run as written.

Each page runs top to bottom in one namespace, as in ``test_docs.py``; its
ChemFit blocks skip without a ChemFit whose Fitter the bridges can drive.
"""

import pytest

pytest.importorskip("anneal")

from doc_blocks import cases, check  # noqa: E402


@pytest.mark.parametrize("page, index", cases(chemfit=True))
def test_the_block_runs_after_the_ones_before_it(page, index, tmp_path_factory):
    check(page, index, tmp_path_factory)
