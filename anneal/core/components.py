from dataclasses import dataclass
from collections import namedtuple
import abc

import numpy as np
import numpy.typing as npt
import typing

from anneal.core.exceptions import OutOfBounds


@dataclass
class NumLimit:
    """Class for tracking function bounds

    Parameters
    ----------
    low: A list of values which the function must not be below
    high: A list of values which the function must not exceed
    slack: The amount by which bounds may be exceeded without an error
    dims: This should be the same as the length of the list
    """

    low: npt.NDArray
    high: npt.NDArray
    slack: float = 1e-6
    dims: int = 1

    def check(self, pos: npt.NDArray):
        if not (
            np.all(pos > self.low - self.slack)
            and np.all(pos < self.high + self.slack)
        ):
            raise OutOfBounds(
                f"{pos} is not within {self.slack} of {self.low} and {self.high}"
            )
        return


class ObjectiveFunction(metaclass=abc.ABCMeta):
    def __init__(self, limits: NumLimit):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            and hasattr(subclass, "pointwise")
            and callable(subclass.pointwise)
            and hasattr(subclass, "multipoint")
            and callable(subclass.multipoint)
            and hasattr(subclass, "__repr__")
            and callable(subclass.__repr__)
            or NotImplemented
        )

    def __call__(self, pos):
        if pos.ravel().shape[0] != self.limits.dims:
            return self.multipoint(pos)
        else:
            return self.singlepoint(pos)

    @abc.abstractmethod
    def singlepoint(self, pos):
        """Evaluate the function at a single configuration"""
        raise NotImplementedError(
            "Need to be able to call the objective function on a single point"
        )

    @abc.abstractmethod
    def multipoint(self, pos):
        """Evaluate the function at many configurations

        TODO: This allows for a faster implementaiton in C
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """Name the function
        """
        raise NotImplementedError


class AcceptCriteria(metaclass=abc.ABCMeta):
    def __init__(self):
        pass


class ConstructNeighborhood(metaclass=abc.ABCMeta):
    def __init__(self):
        pass


class CoolingSchedule(metaclass=abc.ABCMeta):
    def __init__(self):
        pass


class MoveClass(metaclass=abc.ABCMeta):
    def __init__(self):
        pass
