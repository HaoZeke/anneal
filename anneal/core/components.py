from dataclasses import dataclass
from collections import namedtuple
from enum import Enum
import abc

import numpy as np
import numpy.typing as npt
import typing

from anneal.core.exceptions import OutOfBounds

MAX_LIMITS = namedtuple("MAX_LIMITS", ["EPOCHS", "STEPS_PER_EPOCH"])
EpochLine = namedtuple(
    "EpochLine", ["epoch", "temperature", "step", "pos", "val", "accept"]
)
AcceptStates = Enum("AcceptStates", ["IMPROVED", "MHACCEPT", "REJECT"])


@dataclass
class FPair:
    """Class for handing position value pairs"""

    pos: npt.NDArray
    val: float

    def EvalFunc(self, ObjFunc: callable):
        self.val = ObjFunc(self.pos)


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

    def mkpoint(self) -> npt.NDArray:
        """Generate a random point

        TODO: Handle other constraints (undefined regions)
        """
        return np.random.uniform(self.low, self.high)

    def clip(self, point: npt.NDArray) -> npt.NDArray:
        """Clips values"""
        return np.clip(point, self.low, self.high)


class ObjectiveFunction(metaclass=abc.ABCMeta):
    def __init__(self, limits: NumLimit, global_min: FPair = None):
        self.calls = 0
        self.limits = limits
        self.globmin = global_min

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
        ## TODO: calls in multipoint may be more than once
        if pos.ravel().shape[0] != self.limits.dims:
            self.calls += 1
            return self.multipoint(pos)
        else:
            self.calls += 1
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

        TODO: This allows for a faster implementation in C
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """Name the function"""
        raise NotImplementedError


class AcceptCriteria(metaclass=abc.ABCMeta):
    """The acceptance criteria for selecting an unfavorable move"""

    def __init__(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            and hasattr(subclass, "__repr__")
            and callable(subclass.__repr__)
            or NotImplemented
        )

    @abc.abstractmethod
    def __call__(self):
        """Accept or reject"""
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """Name the function"""
        raise NotImplementedError


class ConstructNeighborhood(metaclass=abc.ABCMeta):
    """Choosing a feasible point / direction

    For most methods, this is a uniform distribution
    """

    def __init__(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            and hasattr(subclass, "__repr__")
            and callable(subclass.__repr__)
            or NotImplemented
        )

    @abc.abstractmethod
    def __call__(self):
        """Yield a point"""
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """Name the function"""
        raise NotImplementedError


class CoolingSchedule(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            and hasattr(subclass, "__repr__")
            and callable(subclass.__repr__)
            or NotImplemented
        )

    @abc.abstractmethod
    def __call__(self):
        """Generate a temperature"""
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """Name the function"""
        raise NotImplementedError


class MoveClass(metaclass=abc.ABCMeta):
    """The probability distribution for the step-size

    Also called the visiting distribution.
    """

    def __init__(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            and hasattr(subclass, "__repr__")
            and callable(subclass.__repr__)
            or NotImplemented
        )

    @abc.abstractmethod
    def __call__(self):
        """Generate a move"""
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """Name the function"""
        raise NotImplementedError


class Quencher(metaclass=abc.ABCMeta):
    """A quenching class"""

    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        MkNeigh: ConstructNeighborhood,
        Accepter: AcceptCriteria,
        Mover: MoveClass,
        Cooler: CoolingSchedule,
    ):
        self.ObjFunc = ObjFunc
        self.MkNeigh = MkNeigh
        self.Accepter = Accepter
        self.Mover = Mover
        self.Cooler = Cooler
        self.cur = FPair(None, None)
        self.candidate = FPair(None, None)
        self.best = FPair(None, None)
        self.acceptances = 0
        self.rejections = 0
        self.samestate_time = 0
        self.fCalls = 0
        self.PlotData = []

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            and hasattr(subclass, "__repr__")
            and callable(subclass.__repr__)
            and hasattr(subclass, "HasConverged")
            and callable(subclass.HasConverged)
            or NotImplemented
        )

    @abc.abstractmethod
    def __call__(self):
        """Generate a move"""
        raise NotImplementedError

    @abc.abstractmethod
    def HasConverged(self):
        """Check for Convergence"""
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """Name the function"""
        raise NotImplementedError


class BaseChainSA(metaclass=abc.ABCMeta):
    """The Base chain for MCMC samplers used in SA"""

    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        Temperature,
        Nsim=5000,
    ):
        """A key point is that the temperature is used to define the target probability distribution"""
        self.ObjFunc = ObjFunc
        self.Nsim = Nsim
        self.Temperature = Temperature
        self.stepNum = 0
        self.states = np.array(
            [self.ObjFunc.limits.mkpoint() for x in range(self.Nsim)]
        )
        self.status = np.empty(self.Nsim, dtype=bool)
        self.fvals = np.zeros(self.Nsim)
        ## This is the probability at each temperature
        self.TargetDistrib = lambda point: np.exp(
            -self.ObjFunc(point) / self.Temperature
        )
