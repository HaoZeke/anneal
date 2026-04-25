import numpy as np
from eindir.core.components import FPair, NumLimit, ObjectiveFunction


class StybTang2d(ObjectiveFunction):
    """
    Defines the 2D Styblinski-Tang objective function.

    #### Notes
    This class represents the 2-dimensional Styblinski-Tang function. It is a
    subclass of the `ObjectiveFunction` class.

    The Styblinski-Tang function in 2D is defined as:

    $$f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^{2} \left[ x_i^4 - 16x_i^2 + 5x_i
    \right]$$

    The function is minimized at $x = [-2.903534, -2.903534]$ with a minimum
    value of $-39.16599 \times 2$.
    """

    def __init__(
        self, limits=NumLimit(dims=2, low=np.ones(2) * -5, high=np.ones(2) * 5)
    ):
        """
        Initializes an instance of the `StybTang2d` class.

        #### Parameters
        **limits**, optional
        : The limits for the function. If not provided, the function is defined
           over the domain $[-5, 5] \times [-5, 5]$.
        """
        super().__init__(
            limits,
            FPair(val=-39.16599 * 2, pos=np.array([-2.903534, -2.903534])),
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
        return "2D Styblinski-Tang"
