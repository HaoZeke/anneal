from dataclasses import dataclass
from collections import namedtuple
import abc

import numpy as np
import nptyping as npt
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
    def __init__(self, high, low):
        pass


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
