import numpy as np
from eindir.core.components import FPair, NumLimit, ObjectiveFunction


class StybTangNd(ObjectiveFunction):
    """
    Defines the n-dimensional Styblinski-Tang objective function.

    #### Notes
    This class represents the n-dimensional Styblinski-Tang function. It is a
    subclass of the `ObjectiveFunction` class.

    The Styblinski-Tang function in n-dimensions is defined as:

    $$f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^{n} \left[ x_i^4 - 16x_i^2 + 5x_i \right]$$

    The function is minimized at $x = [-2.903534, -2.903534, ..., -2.903534]$ (n
    times) with a minimum value of $-39.16599 \times n$. This class provides a
    generalized implementation of the function over an n-dimensional hypercube.
    """

    def __init__(self, dims):
        """
        Initializes an instance of the `StybTangNd` class.

        #### Parameters
        **dims**
        : The number of dimensions for the Styblinski-Tang function. The
          function is then defined over a hypercube in this number of dimensions.
        """
        self.dims = dims
        self.limits = NumLimit(
            dims=self.dims,
            low=np.ones(self.dims) * -5,
            high=np.ones(self.dims) * 5,
        )
        super().__init__(
            self.limits,
            FPair(val=-39.16599 * self.dims, pos=np.array([-2.903534] * self.dims)),
        )

    def singlepoint(self, pos):
        """
        Calculates the value of the Styblinski-Tang function at a single point.

        #### Parameters
        **pos**
        : The point at which to calculate the function value.

        #### Returns
        `float`
        : The value of the function at the given point.
        """
        return np.sum((pos**4) - (16 * (pos**2)) + (5 * pos)) / 2

    def multipoint(self, pos):
        """
        Applies the `singlepoint` method along an axis of a numpy array.

        #### Parameters
        **pos**
        : A numpy array of points at which to calculate the function values.

        #### Returns
        `numpy.ndarray`
        : A numpy array of the function values at the given points.
        """
        return np.apply_along_axis(
            self.singlepoint, 1, pos.reshape(-1, self.limits.dims)
        )

    def __repr__(self):
        return f"{self.dims}D Styblinski-Tang"
