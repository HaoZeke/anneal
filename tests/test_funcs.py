import numpy as np
import pytest

from anneal.funcs.obj2d import StybTang2d
from anneal.funcs.objNd import StybTangNd


def test_StybTang2d():
    ff = StybTang2d()
    assert ff(np.array([-5, -5])) == 200
    assert ff(np.array([-4.9, -4.9, -5, -5])) == pytest.approx(
        np.array([167.8201, 200.0])
    )
    assert ff(ff.globmin.pos) == pytest.approx(ff.globmin.val, abs=1e-3)
    assert ff(
        np.array(
            [
                [0.61520743, 0.73489428],
                [0.42910286, 0.02296883],
                [0.33404362, 0.29842335],
            ]
        )
    ) == pytest.approx(
        np.array([-3.755682790653861, -0.3301234957151148, -0.01377451063767443])
    )


def test_StybTangNd_two():
    ffTwo = StybTangNd(dims=2)
    assert ffTwo(np.array([-5, -5])) == 200
    assert ffTwo(np.array([-4.9, -4.9, -5, -5])) == pytest.approx(
        np.array([167.8201, 200.0])
    )
    assert ffTwo(ffTwo.globmin.pos) == pytest.approx(ffTwo.globmin.val, abs=1e-3)


def test_StybTangNd():
    ff = StybTangNd(dims=5)
    assert ff(np.array([0.61520743, 0.73489428, 0.2, 0.6, 0.21])) == pytest.approx(
        -4.71691038565
    )
    assert ff(ff.globmin.pos) == pytest.approx(ff.globmin.val, abs=1e-3)
    assert str(ff) == "5D Styblinski-Tang"
