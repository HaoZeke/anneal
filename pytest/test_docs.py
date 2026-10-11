"""The Python blocks of the README, the docs and the notebooks run as written.

Each page runs top to bottom in one namespace; ``doc_blocks`` says how a
block that cannot run here is skipped. The pages with a ChemFit block are in
``test_chemfit_docs.py``, so the ChemFit steps of CI run them too.
"""

import pytest

pytest.importorskip("anneal")

from doc_blocks import cases, check  # noqa: E402


@pytest.mark.parametrize("page, index", cases(chemfit=False))
def test_the_block_runs_after_the_ones_before_it(page, index, tmp_path_factory):
    check(page, index, tmp_path_factory)
