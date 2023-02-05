import numpy as np
import numpy.typing as npt
import pytest

from anneal.core.components import (
    Quencher,
    ObjectiveFunction,
    CoolingSchedule,
    AcceptCriteria,
    ConstructNeighborhood,
    MoveClass,
    NumLimit,
    MAX_LIMITS,
    EpochLine,
    FPair,
    AcceptStates,
)


class BoltzmannQuencher(Quencher):
    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        pos_init: npt.NDArray = None,
        *,
        T_init: float = None,
        maxiter: MAX_LIMITS = MAX_LIMITS(int(1e6), int(1e3)),
    ):
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
            self.cur = FPair(
                pt := self.ObjFunc.limits.mkpoint(), self.ObjFunc(pt)
            )
        else:
            self.cur = FPair(pt := pos_init, self.ObjFunc(pt))
        self.maxiter = maxiter
        ## Initially
        self.best = self.cur

    def __call__(self, trackPlot=False):
        while (
            temperature := self.Cooler(self.epoch)
        ) > 0.1 and self.epoch < self.maxiter.EPOCHS:
            for step in range(1, self.maxiter.STEPS_PER_EPOCH + 1):
                self.candidate = FPair(
                    pt := self.MkNeigh(self.cur.pos), self.ObjFunc(pt)
                )
                diff = self.DeltaFunc()
                if diff <= 0:
                    ## Better point, accept always
                    self.AcceptMove()
                    self.PlotData.append(
                        EpochLine(
                            epoch=self.epoch,
                            temperature=temperature,
                            step=step,
                            pos=self.candidate.pos,
                            val=self.candidate.val,
                            accept=AcceptStates.IMPROVED,
                        )
                    )
                else:
                    if self.Accepter(diff, temperature):
                        self.AcceptMove()
                        self.PlotData.append(
                            EpochLine(
                                epoch=self.epoch,
                                temperature=temperature,
                                step=step,
                                pos=self.candidate.pos,
                                val=self.candidate.val,
                                accept=AcceptStates.MHACCEPT,
                            )
                        )
                    else:
                        self.RejectMove()
                        self.PlotData.append(
                            EpochLine(
                                epoch=self.epoch,
                                temperature=temperature,
                                step=step,
                                pos=self.candidate.pos,
                                val=self.candidate.val,
                                accept=AcceptStates.REJECT,
                            )
                        )
            print(f"{self.epoch} for {temperature} has {self.best}")
            if self.HasConverged():
                return
            else:
                self.epoch += 1

    def DeltaFunc(self):
        return self.candidate.val - self.cur.val

    def AcceptMove(self):
        self.cur = self.candidate
        if self.cur.val < self.best.val:
            self.best = self.cur
        else:
            self.samestate_time += 1
        self.acceptances += 1

    def RejectMove(self):
        self.rejections += 1
        self.samestate_time += 1

    # TODO: Generalize for other kinds of convergence
    def HasConverged(self):
        if self.best.val == pytest.approx(
            self.ObjFunc.globmin.val, 1e-3
        ) or self.best.pos == pytest.approx(self.ObjFunc.globmin.pos, 1e-3):
            self.fCalls = self.ObjFunc.calls
            self.ObjFunc.calls = 0
            return True

    def __repr__(self):
        return "An instance of the Boltzmann Quencher"


class BoltzmannCooler(CoolingSchedule):
    """This is the log schedule of Geman and Geman (1983)"""

    def __init__(self, T_init, c_param=1):
        """Salazar and Torale (1997) set c_param to T_init"""
        self.Tinit = T_init
        self.c_param = c_param

    def __call__(self, epoch):
        """Random perturbation to ensure G&G inequality holds
        np.random.uniform(0.01, 0.5) +
        """
        Temp = (self.c_param * self.Tinit) / (1 + np.log(epoch))
        return Temp

    def __repr__(self):
        return "Log schedule cooler for Boltzmann"


class BoltzmannMove(MoveClass):
    """For the Boltzmann the stepsize is random"""

    def __init__(self, limits: NumLimit):
        self.limits = limits

    def __call__(self):
        return self.limits.mkpoint()

    def __repr__(self):
        return "Random stepsize for the Boltzmann"


## Consider np.random.normal(0, np.random.rand(1), ndim)
class BoltzmannNeighbor(ConstructNeighborhood):
    """For the Boltzmann this is a random direction
    Gall (2014) disagrees though
    """

    def __init__(self, limits: NumLimit, mover: BoltzmannMove):
        self.limits = limits
        self.stepper = mover

    def __call__(self, c_pos):
        ## Project current onto candidate
        candidate = self.limits.mkpoint()
        proj = (
            candidate * np.dot(c_pos, candidate) / np.dot(candidate, candidate)
        )
        dir_vec = proj / np.linalg.norm(proj)
        final_cand = dir_vec * self.stepper()
        ## Clipping is not part of the original formulation
        final_cand = self.limits.clip(final_cand)
        return final_cand

    def __repr__(self):
        return "Random candidate for the Boltzmann"


class BoltzmannAccept(AcceptCriteria):
    def __init__(self, k=1):
        self.k = 1

    def __call__(self, diff, Temperature):
        metropolis = np.exp(-self.k * diff / Temperature)
        return np.random.uniform(0, 1) < metropolis

    def __repr__(self):
        return "Basic Metropolis acceptance for Boltzmann"
