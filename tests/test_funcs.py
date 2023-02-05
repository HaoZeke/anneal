import pytest
import numpy.testing as nptt

from anneal.funcs.obj2d import *
from anneal.funcs.objNd import *


def test_StybTang2d():
    ff = StybTang2d()
    assert ff(np.array([-5, -5])) == 200
    assert ff(np.array([-4.9, -4.9, -5, -5])) == pytest.approx(
        np.array([167.8201, 200.0])
    )
    assert ff(ff.glob_minma_pos) == pytest.approx(ff.glob_minma_val, abs=1e-3)


def test_StybTangNd_two():
    ffTwo = StybTangNd(dims=2)
    assert ffTwo(np.array([-5, -5])) == 200
    assert ffTwo(np.array([-4.9, -4.9, -5, -5])) == pytest.approx(
        np.array([167.8201, 200.0])
    )
    assert ffTwo(ffTwo.glob_minma_pos) == pytest.approx(
        ffTwo.glob_minma_val, abs=1e-3
    )


def test_StybTangNd():
    ff = StybTangNd(dims=5)
    assert ff(ff.glob_minma_pos) == pytest.approx(ff.glob_minma_val, abs=1e-3)
    assert str(ff) == "5D Styblinski-Tang"
