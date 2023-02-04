from anneal.core.components import NumLimit, ObjectiveFunction

import numpy as np


class StybTang2d(ObjectiveFunction):
    def __init__(
        self, limits=NumLimit(dims=2, low=np.ones(2) * -5, high=np.ones(2) * 5)
    ):
        self.limits = limits
        self.glob_minma = -39.16599*2
        pass

    def singlepoint(self, pos):
        self.limits.check(pos)
        return np.sum((pos**4) - (16 * (pos**2)) + (5 * pos)) / 2

    def multipoint(self, pos):
        return np.apply_along_axis(
            self.singlepoint, 1, pos.reshape(self.limits.dims, -1)
        )
