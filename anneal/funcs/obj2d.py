from anneal.core.components import NumLimit, ObjectiveFunction, FPair

import numpy as np


class StybTang2d(ObjectiveFunction):
    def __init__(
        self, limits=NumLimit(dims=2, low=np.ones(2) * -5, high=np.ones(2) * 5)
    ):
        super().__init__(
            limits,
            FPair(val=-39.16599 * 2, pos=np.array([-2.903534, -2.903534])),
        )

    def singlepoint(self, pos):
        self.limits.check(pos)
        return np.sum((pos**4) - (16 * (pos**2)) + (5 * pos)) / 2

    def multipoint(self, pos):
        return np.apply_along_axis(
            self.singlepoint, 1, pos.reshape(-1, self.limits.dims)
        )

    def __repr__(self):
        return "2D Styblinski-Tang"
