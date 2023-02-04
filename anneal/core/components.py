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
    dims: This should be the same as the length of the list
    """
    low: npt.NDArray
    high: npt.NDArray
    dims: int = 1

    def check(pos: npt.NDArray)->bool:
        if not (np.all(pos > self.low) and np.all(pos < self.high)):
            raise OutOfBounds(f"{pos} is not within {self.low} and {self.high}")

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
