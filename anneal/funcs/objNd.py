from anneal.core.components import NumLimit, ObjectiveFunction, FPair

import numpy as np


class StybTangNd(ObjectiveFunction):
    """This is a generalization as the function is evaluated on a hypercube"""

    def __init__(self, dims):
        self.dims = dims
        self.limits = NumLimit(
            dims=self.dims,
            low=np.ones(self.dims) * -5,
            high=np.ones(self.dims) * 5,
        )
        super().__init__(
            self.limits,
            FPair(
                val=-39.16599 * self.dims, pos=np.array([-2.903534] * self.dims)
            ),
        )

    def singlepoint(self, pos):
        self.limits.check(pos)
        return np.sum((pos**4) - (16 * (pos**2)) + (5 * pos)) / 2

    def multipoint(self, pos):
        return np.apply_along_axis(
            self.singlepoint, 1, pos.reshape(-1, self.limits.dims)
        )

    def __repr__(self):
        return f"{self.dims}D Styblinski-Tang"
