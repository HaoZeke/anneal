from anneal.mcsamplers.chains import *
from anneal.core.components import BaseChainSA


class MHChainSA(BaseChainSA):
    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        Cooler: CoolingSchedule,
        n_sim: int = 5000,
        maxiter: MAX_LIMITS = MAX_LIMITS(int(1e6), int(1e3)),
    ):
        super().__init__(ObjFunc, MHChain, Cooler, n_sim, maxiter)
