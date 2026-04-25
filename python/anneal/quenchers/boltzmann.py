import numpy as np
import numpy.typing as npt
import pytest
from eindir.core.components import FPair, NumLimit, ObjectiveFunction

from anneal.core.components import (MAX_LIMITS, AcceptCriteria, AcceptStates,
                                    ConstructNeighborhood, CoolingSchedule,
                                    MoveClass, Quencher)


class BoltzmannQuencher(Quencher):
    """
    Implements the Boltzmann Quencher method for optimization.

    #### Notes
    Boltzmann Quencher is a type of Simulated Annealing algorithm that uses a
    Boltzmann distribution for the acceptance criteria of new points.
    """

    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        pos_init: npt.NDArray = None,
        *,
        T_init: float = None,
        maxiter: MAX_LIMITS = MAX_LIMITS(int(1e6), int(1e3)),
    ):
        """
        Initializes an instance of the `BoltzmannQuencher` class.

        #### Parameters
        **ObjFunc: ObjectiveFunction**
        : An instance of the `ObjectiveFunction` class representing the
          function to be optimized.

        **pos_init: npt.NDArray, optional**
        : The initial position in the function's search space.

        **T_init: float, optional**
        : The initial temperature for the simulated annealing process.

        **maxiter: MAX_LIMITS, optional**
        : An instance of the `MAX_LIMITS` class representing the maximum number
          of iterations for the simulated annealing process. If not provided,
          the default maximum number of steps is 1e6 and the default maximum
          number of epochs is 1e3.
        """
        self.ObjFunc = ObjFunc
        self.Mover = BoltzmannMove(self.ObjFunc.limits)
        self.MkNeigh = BoltzmannNeighbor(self.ObjFunc.limits, self.Mover)
        self.Accepter = BoltzmannAccept()
        ## Maybe Tinit = −∆E / ln x0
        self.T_init = T_init
        self.Cooler = BoltzmannCooler(self.T_init)
        super().__init__(
            self.ObjFunc,
            self.MkNeigh,
            self.Accepter,
            self.Mover,
            self.Cooler,
        )
        self.epoch = 1
        if pos_init is None:
            self.cur = FPair(pt := self.ObjFunc.limits.mkpoint(), self.ObjFunc(pt))
        else:
            self.cur = FPair(pt := pos_init, self.ObjFunc(pt))
        self.maxiter = maxiter
        ## Initially
        self.best = self.cur

    def __call__(self, trackPlot=False):
        """
        Runs the Boltzmann Quencher algorithm.

        #### Parameters
        **trackPlot: bool, optional (default=False)**
        : Whether to track the performance of the algorithm for plotting.
        """
        while (
            temperature := self.Cooler(self.epoch)
        ) > 0.01 and self.epoch < self.maxiter.EPOCHS:
            for step in range(1, self.maxiter.STEPS_PER_EPOCH + 1):
                self.candidate = FPair(
                    pt := self.MkNeigh(self.cur.pos), self.ObjFunc(pt)
                )
                diff = self.DeltaFunc()
                if diff <= 0:
                    ## Better point, accept always
                    self.AcceptMove()
                    self.addPlotPoint(temperature, step, AcceptStates.IMPROVED)
                else:
                    if self.Accepter(diff, temperature):
                        self.AcceptMove()
                        self.addPlotPoint(temperature, step, AcceptStates.MHACCEPT)
                    else:
                        self.RejectMove()
                        self.addPlotPoint(temperature, step, AcceptStates.REJECT)
            print(f"{self.epoch} for {temperature} has {self.best}")
            if self.HasConverged():
                return
            else:
                self.epoch += 1

    def DeltaFunc(self):
        """
        Computes the difference in objective function values between the
        current state and the proposed state.

        #### Returns
        `float`
        : The difference in objective function values.
        """
        return self.candidate.val - self.cur.val

    def AcceptMove(self):
        """
        Accepts the move from the current state to the proposed state.
        """
        self.cur = self.candidate
        if self.cur.val < self.best.val:
            self.best = self.cur
            self.samestate_time = 0
        else:
            self.samestate_time += 1
        self.acceptances += 1

    def RejectMove(self):
        """
        Rejects the move from the current state to the proposed state.
        """
        self.rejections += 1
        self.samestate_time += 1

    # TODO: Generalize for other kinds of convergence
    def HasConverged(self):
        """
        Checks if the optimization process has converged.

        #### Returns
        `bool`
        : `True` if the optimization has converged, `False` otherwise.
        """
        if (
            self.best.val == pytest.approx(self.ObjFunc.globmin.val, 1e-3)
            or self.best.pos == pytest.approx(self.ObjFunc.globmin.pos, 1e-3)
            or self.samestate_time > 1e4
        ):
            if self.samestate_time > 1e4:
                print("Terminating due to too much time in state")
            self.fCalls = self.ObjFunc.calls
            self.ObjFunc.calls = 0
            return True

    def __repr__(self):
        return "An instance of the Boltzmann Quencher"


class BoltzmannCooler(CoolingSchedule):
    """
    Implements the logarithmic cooling schedule proposed by Geman and Geman (1983).

    #### Notes
    This cooling schedule is designed for use with the Boltzmann Quencher. The
    temperature is reduced at each epoch according to a logarithmic function to
    ensure that the Geman & Geman inequality holds.
    """

    def __init__(self, T_init, c_param=1):
        """
        Initializes an instance of the `BoltzmannCooler` class.

        #### Parameters
        **T_init: float**
        : The initial temperature for the cooling schedule.

        **c_param: float, optional (default=1)**
        : The parameter 'c' for the cooling schedule. According to Salazar and
          Torale (1997), this can be set to the initial temperature.
        """
        self.c_param = c_param
        super().__init__(T_init)

    def __call__(self, epoch):
        """
        Calculates the temperature for a given epoch.

        #### Parameters
        **epoch: int**
        : The current epoch (iteration) of the optimization process.

        #### Returns
        `float`
        : The temperature for the given epoch.

        #### Notes
        Can also consider a random perturbation to ensure G&G inequality holds
        np.random.default_rng().uniform(0.01, 0.5) +
        """
        Temp = (self.c_param * self.Tinit) / (1 + np.log(epoch))
        return Temp

    def __repr__(self):
        return "Log schedule cooler for Boltzmann"


class BoltzmannMove(MoveClass):
    """
    Implements a Boltzmann move with a random step size.

    #### Notes
    For Boltzmann simulated annealing, the step size of a move is randomly
    chosen from a standard Gaussian distribution.
    """

    def __init__(self, limits: NumLimit):
        """
        Initializes an instance of the `BoltzmannMove` class.

        #### Parameters
        **limits: NumLimit**
        : The limits for the move.
        """
        self.limits = limits

    def __call__(self):
        """
        Generates a random step size for the move.

        #### Returns
        `float`
        : The step size for the move.
        """
        return np.random.default_rng().normal()

    def __repr__(self):
        return "Random stepsize for the Boltzmann"


## Consider np.random.default_rng().normal(0, np.random.rand(1), ndim)
class BoltzmannNeighbor(ConstructNeighborhood):
    """
    Constructs a neighborhood for Boltzmann simulated annealing.

    #### Notes
    For Boltzmann simulated annealing, a random direction is chosen. This
    behavior is based on the recommendation of Casella and Robert (2010).
    However, Gall (2014) disagrees.
    """

    def __init__(self, limits: NumLimit, mover: BoltzmannMove):
        """
        Initializes an instance of the `BoltzmannNeighbor` class.

        #### Parameters
        **limits: NumLimit**
        : The limits for the move.

        **mover: BoltzmannMove**
        : An instance of the `BoltzmannMove` class to make the move.
        """
        self.limits = limits
        self.stepper = mover

    def __call__(self, c_pos):
        """
        Generates a new candidate position in the neighborhood of the current position.

        #### Parameters
        **c_pos: numpy.ndarray**
        : The current position.

        #### Returns
        `numpy.ndarray`
        : The candidate position.
        """
        ## Project current onto candidate
        candidate = self.limits.mkpoint()
        proj = candidate * np.dot(c_pos, candidate) / np.dot(candidate, candidate)
        dir_vec = proj / np.linalg.norm(proj)
        final_cand = candidate + dir_vec * self.stepper()
        ## Clipping is not part of the original formulation
        # final_cand = self.limits.clip(final_cand)
        return final_cand

    def __repr__(self):
        return "Random candidate for the Boltzmann"


class BoltzmannAccept(AcceptCriteria):
    """
    Implements the acceptance criteria for Boltzmann simulated annealing.

    #### Notes
    The Metropolis-Hastings criterion is used for acceptance in Boltzmann
    simulated annealing.
    """

    def __init__(self, k=1):
        """
        Initializes an instance of the `BoltzmannAccept` class.

        #### Parameters
        **k: int, optional (default=1)**
        : The parameter 'k' for the acceptance criteria.
        """
        self.k = 1

    def __call__(self, diff, Temperature):
        """
        Decides whether to accept a candidate solution.

        #### Parameters
        **diff: float**
        : The difference in objective function values between the
          candidate and current solutions.

        **Temperature: float**
        : The current temperature.

        #### Returns
        `bool`
        : `True` if the candidate solution is accepted, `False` otherwise.
        """
        metropolis = np.min([np.exp(-self.k * diff / Temperature), 1])
        return np.random.default_rng().uniform(0, 1) < metropolis

    def __repr__(self):
        return "Basic Metropolis acceptance for Boltzmann"
