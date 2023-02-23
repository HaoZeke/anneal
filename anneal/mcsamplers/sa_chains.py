from anneal.mcsamplers.chains import *
from anneal.core.components import BaseChainSA


class MHChainSA(BaseChainSA):
    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        Cooler: CoolingSchedule,
        init_temp: float,
        n_sim: int = 5000,
        maxiter: MAX_LIMITS = MAX_LIMITS(int(1e6), int(1e3)),
    ):
        super().__init__(ObjFunc, MHChain, Cooler, init_temp, n_sim, maxiter)
