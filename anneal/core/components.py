from dataclasses import dataclass
from collections import namedtuple
from enum import Enum
import abc

import numpy as np
import numpy.typing as npt
import typing
import pytest

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


# class ProbabilityDistribution:
#     def __init__(self):
#         pass

# class BaseChain(metaclass=abc.ABCMeta):
#     """Base class for MCMC sampler chains"""
#     TODO: Add thinning, burn_in, gelman-rubin, make basechainsa inherit this
#     def __init__(
#             self,
#             n_samples: int = 500,
#             burn_in: int = 50,
#             thining: int = 1,
#     ):


class BaseChainSA(metaclass=abc.ABCMeta):
    """The Base chain for MCMC samplers used in SA"""

    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        Chain,  # The type of chain used
        Cooler: CoolingSchedule,
        init_temp: float,
        n_sim: int = 20000,
        maxiter: MAX_LIMITS = MAX_LIMITS(int(1e6), int(1e3)),
    ):
        """A key point is that the temperature is used to define the target
        probability distribution"""
        self.ObjFunc = ObjFunc
        self.n_sim = n_sim
        self.Temperature = init_temp
        self.stepNum = 0
        self.epoch_best = []
        self.Chain = Chain
        self.epoch = 1
        self.maxiter = maxiter
        self.best = None
        self.TargetDistrib = None
        self.Cooler = Cooler

    def mk_target(self, Temperature):
        return lambda point: np.exp(-self.ObjFunc(point) / Temperature)

    def __call__(self, Proposal, init_state=None):
        if isinstance(init_state, type(None)):
            init_state = self.ObjFunc.limits.mkpoint()
        while (
            temperature := self.Cooler(self.epoch)
        ) > 0.01 and self.epoch < self.maxiter.EPOCHS:
            target = self.mk_target(temperature)
            chain = self.Chain(target, Proposal, init_state)
            for _ in range(self.n_sim):
                chain.step()
            self.epoch_best.append(self.getBest(chain.states))
            if self.epoch > 1:
                print(f"{self.epoch} for {temperature} has {self.best}")
            if self.HasConverged():
                return
            else:
                self.epoch += 1

    def getBest(self, statelist: list):
        energies = np.array([self.ObjFunc(x) for x in statelist])
        return FPair(pos=statelist[energies.argmin()], val=np.min(energies))

    def HasConverged(self):
        min_ee = min([x.val for x in self.epoch_best])
        self.best = [x for x in self.epoch_best if x.val == min_ee][0]
        if min_ee == pytest.approx(self.ObjFunc.globmin.val, 1e-3):
            self.fCalls = self.ObjFunc.calls
            self.ObjFunc.calls = 0
            return True
