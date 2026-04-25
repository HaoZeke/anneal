import numpy as np
import pytest

from anneal.funcs.obj2d import StybTang2d
from anneal.quenchers.boltzmann import BoltzmannQuencher


def test_BQST2d():
    ff = StybTang2d()
    bq = BoltzmannQuencher(StybTang2d(), T_init=5, pos_init=np.ones(2) * -2)
    bq()
    assert bq.best.val == pytest.approx(ff.globmin.val, 1e-2)
    assert bq.best.pos == pytest.approx(ff.globmin.pos, 1e1)
