import pytest
import numpy.testing as nptt

from anneal.funcs.obj2d import *
from anneal.mcsamplers.chains import MHChain


def test_MHChain():
    ff = StybTang2d()
    mha = MHChain(StybTang2d(), Temperature=10, Nsim=9000)
    bpair = mha()
    assert bpair.val == pytest.approx(ff.globmin.val, 1e-2)
    assert bpair.pos == pytest.approx(ff.globmin.pos, 1e1)
