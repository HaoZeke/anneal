import pytest
import numpy.testing as nptt

from anneal.funcs.obj2d import *
from anneal.mcsamplers.sa_chains import MHChainSA
from anneal.quenchers.boltzmann import BoltzmannCooler


def test_MHChain():
    ff = StybTang2d()
    mhcsa = MHChainSA(ff, BoltzmannCooler(50))
    mhcsa(np.random.default_rng().normal)
    assert mhcsa.best.val == pytest.approx(ff.globmin.val, 1e-2)
    assert mhcsa.best.pos == pytest.approx(ff.globmin.pos, 1e1)
