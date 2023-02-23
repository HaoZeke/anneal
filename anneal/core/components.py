from dataclasses import dataclass
from collections import namedtuple
from enum import Enum
import abc

import numpy as np
import numpy.typing as npt
import typing

from eindir.core.components import FPair, NumLimit, ObjectiveFunction

MAX_LIMITS = namedtuple("MAX_LIMITS", ["EPOCHS", "STEPS_PER_EPOCH"])
EpochLine = namedtuple(
    "EpochLine", ["epoch", "temperature", "step", "pos", "val", "accept"]
)
AcceptStates = Enum("AcceptStates", ["IMPROVED", "MHACCEPT", "REJECT"])


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

    # Default implementation
    def addPlotPoint(
        self, temperature: float, step: int, acceptState: AcceptStates
    ):
        self.PlotData.append(
            EpochLine(
                epoch=self.epoch,
                temperature=temperature,
                step=step,
                pos=self.candidate.pos,
                val=self.candidate.val,
                accept=acceptState,
            )
        )


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
