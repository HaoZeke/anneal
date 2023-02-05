import pytest
import numpy.testing as nptt

from anneal.funcs.obj2d import *
from anneal.quenchers.boltzmann import BoltzmannQuencher


def test_BQST2d():
    ff = StybTang2d()
    bq = BoltzmannQuencher(StybTang2d(), T_init=5, pos_init=np.ones(2) * -2)
    bq()
    assert bq.fCalls < 120000
