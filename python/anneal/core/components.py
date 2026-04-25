import abc
from collections import namedtuple
from enum import Enum

import numpy as np
import pytest
from eindir.core.components import FPair, ObjectiveFunction

MAX_LIMITS = namedtuple("MAX_LIMITS", ["EPOCHS", "STEPS_PER_EPOCH"])
"""
Namedtuple to represent the maximum limits for epochs and steps per epoch.

#### Fields
**EPOCHS**
: The maximum number of epochs.

**STEPS_PER_EPOCH**
: The maximum number of steps per epoch.
"""

EpochLine = namedtuple(
    "EpochLine", ["epoch", "temperature", "step", "pos", "val", "accept"]
)
"""
Namedtuple to represent a line of epoch data.

#### Fields
**epoch**
: The current epoch number.

**temperature**
: The current temperature.

**step**
: The current step number.

**pos**
: The current position.

**val**
: The current value.

**accept**
: The acceptance state.
"""

AcceptStates = Enum("AcceptStates", ["IMPROVED", "MHACCEPT", "REJECT"])
"""
Enumeration of possible acceptance states.

#### Members
**IMPROVED**
: The state indicating that the new position is an improvement.

**MHACCEPT**
: The state indicating that the new position is not an improvement, but is
 accepted according to the Metropolis-Hastings criterion.

**REJECT**
: The state indicating that the new position is rejected.
"""


class AcceptCriteria(metaclass=abc.ABCMeta):
    """
    Abstract base class for the acceptance criteria for selecting an
    unfavorable move.

    #### Description
    This class defines the interface for acceptance criteria in a simulated
    annealing algorithm. Subclasses must implement the `__call__` and
    `__repr__` methods.

    #### Notes
    This class uses the `abc.ABCMeta` metaclass and the `abc.abstractmethod`
    decorator to define an abstract base class.
    """

    def __init__(self):
        """
        Initializes an instance of the `AcceptCriteria` class.
        """
        pass

    @abc.abstractmethod
    def __call__(self):
        """
        Abstract method for acceptance or rejection of a move.

        #### Notes
        Subclasses must provide an implementation for this method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """
        Abstract method to return a string representation of the acceptance
        criteria.

        #### Notes
        Subclasses must provide an implementation for this method.
        """
        raise NotImplementedError


class ConstructNeighborhood(metaclass=abc.ABCMeta):
    """
    Abstract base class for constructing a neighborhood of points.

    #### Description
    This class defines a method for generating a feasible point or
    direction. In most methods, this is a uniform distribution.

    #### Notes
    Subclasses must implement the `__call__` and `__repr__` methods.
    """

    def __init__(self):
        """
        Initializes an instance of the `ConstructNeighborhood` class.
        """
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Check if a class provides the required methods to be considered a
        subclass of this abstract base class.

        #### Parameters
        **subclass** (`type`)
        : The class to check.

        #### Returns
        **bool**
        : True if the class provides the required methods, False otherwise.
        """
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            and hasattr(subclass, "__repr__")
            and callable(subclass.__repr__)
            or NotImplemented
        )

    @abc.abstractmethod
    def __call__(self):
        """
        Abstract method to generate a point in the neighborhood.

        #### Notes
        Subclasses must provide an implementation for this method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """
        Abstract method to return a string representation of the instance.

        #### Notes
        Subclasses must provide an implementation for this method.
        """
        raise NotImplementedError


class CoolingSchedule(metaclass=abc.ABCMeta):
    """
    Abstract base class for cooling schedules in simulated annealing.

    #### Description
    This class defines the interface for cooling schedules in a simulated
    annealing algorithm. Subclasses must implement the `__call__` and
    `__repr__` methods.

    #### Parameters
    **T_init** (`float`)
    : The initial temperature for the cooling schedule.

    #### Notes
    This class uses the `abc.ABCMeta` metaclass and the `abc.abstractmethod`
    decorator to define an abstract base class.
    """

    def __init__(self, T_init: float):
        """
        Initializes an instance of the `CoolingSchedule` class.

        #### Parameters
        **T_init** (`float`)
        : The initial temperature for the cooling schedule.
        """
        self.Tinit = T_init

    @abc.abstractmethod
    def __call__(self):
        """
        Abstract method to generate a temperature for the cooling schedule.

        #### Notes
        Subclasses must provide an implementation for this method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """
        Abstract method to return a string representation of the cooling
        schedule.

        #### Notes
        Subclasses must provide an implementation for this method.
        """
        raise NotImplementedError


class MoveClass(metaclass=abc.ABCMeta):
    """
    Abstract base class for the probability distribution for the step-size in
    simulated annealing.

    #### Description
    This class defines the interface for the step-size distribution, also known
    as the visiting distribution, in a simulated annealing algorithm. Subclasses
    must implement the `__call__` and `__repr__` methods.

    #### Notes
    This class uses the `abc.ABCMeta` metaclass and the `abc.abstractmethod`
    decorator to define an abstract base class.
    """

    def __init__(self):
        """
        Initializes an instance of the `MoveClass` class.
        """
        pass

    @abc.abstractmethod
    def __call__(self):
        """
        Abstract method to generate a move in the search space.

        #### Notes
        Subclasses must provide an implementation for this method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """
        Abstract method to return a string representation of the move class.

        #### Notes
        Subclasses must provide an implementation for this method.
        """
        raise NotImplementedError


class Quencher(metaclass=abc.ABCMeta):
    """
    Abstract base class for a quenching class in simulated annealing.

    #### Description
    This class defines the interface for a quenching class in a simulated
    annealing algorithm. Subclasses must implement the `__call__`, `HasConverged`,
    and `__repr__` methods.

    #### Notes
    This class uses the `abc.ABCMeta` metaclass and the `abc.abstractmethod`
    decorator to define an abstract base class.
    """

    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        MkNeigh: ConstructNeighborhood,
        Accepter: AcceptCriteria,
        Mover: MoveClass,
        Cooler: CoolingSchedule,
    ):
        """
        Initializes an instance of the Quencher class.

        #### Parameters
        **ObjFunc** : `ObjectiveFunction`
        : An instance of the objective function.

        **MkNeigh** : `ConstructNeighborhood`
        : An instance of the neighborhood constructor.

        **Accepter** : `AcceptCriteria`
        : An instance of the acceptance criteria.

        **Mover** : `MoveClass`
        : An instance of the move class.

        **Cooler** : `CoolingSchedule`
        : An instance of the cooling schedule.
        """
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
        """
        Custom subclass check.

        #### Description
        This method is used to provide a custom mechanism for subclass check. It
        checks if a class provides the required methods (`__call__`, `__repr__`, and
        `HasConverged`) and returns True if it does, False otherwise.

        #### Parameters
        **subclass**: `class`
        : The class to check.

        #### Returns
        `bool` or `NotImplemented`
        : True if the subclass provides the required methods, False otherwise. If
        the subclass check is continued with the usual mechanism, `NotImplemented`
        is returned.

        #### Notes
        This method is implemented in the base class and does not need to be
        overridden by subclasses. The `hasattr` function is used to check if the
        subclass has the required methods, and the `callable` function is used to
        check if these methods are callable.
        """
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
        """
        Abstract method to generate a move.

        #### Notes
        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def HasConverged(self):
        """
        Abstract method to check for convergence.

        #### Notes
        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        """
        Abstract method to return a string representation of the quencher.

        #### Notes
        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    # Default implementation
    def addPlotPoint(self, temperature: float, step: int, acceptState: AcceptStates):
        """
        Adds a point to the plot data.

        #### Parameters
        **temperature**: `float`
        : The current temperature of the annealing process.

        **step**: `int`
        : The current step of the annealing process.

        **acceptState**: `AcceptStates`
        : The state of acceptance of the current move.

        #### Notes
        This method is implemented in the base class and does not need to be
        overridden by subclasses.
        """
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
    """
    Base class for the Chain Simulated Annealing.

    #### Description
    The `BaseChainSA` class encapsulates the core of a simulated annealing
    algorithm, which is based on the creation and evolution of a chain of states.
    The chain is evolved according to a probability distribution that is gradually
    "cooled" to find the minimum of a given objective function. The specific
    characteristics of the chain and the cooling schedule are defined by the
    arguments passed to the class at instantiation.

    A key point is that the temperature is used to define the target
    probability distribution.
    #### Notes
    This class uses the `abc.ABCMeta` metaclass and the `abc.abstractmethod`
    decorator to define an abstract base class.
    """

    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        Chain,  # The type of chain used
        Cooler: CoolingSchedule,
        n_sim: int = 20000,
        maxiter: MAX_LIMITS = MAX_LIMITS(int(1e6), int(1e3)),
    ):
        """
        Initializes an instance of the BaseChainSA class.

        #### Parameters
        **ObjFunc** : `ObjectiveFunction`
        : The objective function to minimize.

        **Chain**
        : The type of chain to be used in the simulated annealing algorithm.

        **Cooler** : `CoolingSchedule`
        : The cooling schedule to be used in the simulated annealing algorithm.

        **n_sim** : `int`, optional
        : The number of iterations in each epoch of the algorithm. Default is 20000.

        **maxiter** : `MAX_LIMITS`, optional
        : The maximum number of iterations and epochs in the algorithm. Default
        is a `MAX_LIMITS` object with `int(1e6)` for the maximum number of
        iterations and `int(1e3)` for the maximum number of epochs.
        """
        self.ObjFunc = ObjFunc
        self.n_sim = n_sim
        self.stepNum = 0
        self.epoch_best = []
        self.Chain = Chain
        self.epoch = 1
        self.maxiter = maxiter
        self.best = None
        self.TargetDistrib = None
        self.Cooler = Cooler

    def mk_target(self, Temperature):
        """
        Generates the target distribution for a given temperature.

        #### Parameters
        **Temperature** : `float`
        : The current temperature in the simulated annealing algorithm.

        #### Returns
        : A function of a point that gives the probability of that point being
        selected according to the objective function and the current temperature.
        """
        return lambda point: np.exp(-self.ObjFunc(point) / Temperature)

    def __call__(self, Proposal, init_state=None):
        """
        Executes the simulated annealing process.

        #### Parameters
        **Proposal**
        : The proposal distribution function used to generate new states in the
        chain.

        **init_state**, optional
        : The initial state for the simulated annealing algorithm. If not provided,
        a point is generated using the `mkpoint` method of the objective function's
        limits.

        #### Notes
        The simulated annealing algorithm is executed as follows: for each epoch until
        the temperature drops below 0.01 or the maximum number of epochs is reached,
        a target distribution is generated using the current temperature, a chain is
        created using this target distribution and the proposal distribution, the chain
        is stepped `n_sim` times, the best state of this epoch is saved, and if the
        algorithm has converged according to the `HasConverged` method, it ends.
        Otherwise, the epoch number is incremented and the process repeats.
        """
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
        """
        Finds the best state in a list of states.

        #### Parameters
        **statelist** : `list`
        : A list of states to find the best state from.

        #### Returns
        `FPair`
        : An `FPair` object representing the state with the lowest energy in the
        provided list of states.

        #### Notes
        The best state is the one that minimizes the objective function. This method
        computes the energy of each state in the provided list using the objective
        function, and returns the state with the minimum energy.
        """
        energies = np.array([self.ObjFunc(x) for x in statelist])
        return FPair(pos=statelist[energies.argmin()], val=np.min(energies))

    def HasConverged(self):
        """
        Checks if the simulated annealing algorithm has converged.

        #### Returns
        `bool`
        : True if the algorithm has converged, False otherwise.

        #### Notes
        The simulated annealing algorithm is considered to have converged if the
        minimum energy of the best states from all epochs is approximately equal to
        the minimum value of the objective function. If the algorithm has converged,
        the number of function calls is saved and the counter in the objective function
        is reset.
        """
        min_ee = min([x.val for x in self.epoch_best])
        self.best = [x for x in self.epoch_best if x.val == min_ee][0]
        if min_ee == pytest.approx(self.ObjFunc.globmin.val, 1e-3):
            self.fCalls = self.ObjFunc.calls
            self.ObjFunc.calls = 0
            return True
