from eindir.core.components import ObjectiveFunction

from anneal.core.components import MAX_LIMITS, BaseChainSA, CoolingSchedule
from anneal.mcsamplers.chains import MHChain


class MHChainSA(BaseChainSA):
    """
    Implements the Metropolis Hastings Simulated Annealing method.

    This class is a specific implementation of the BaseChainSA class where
    the Metropolis Hastings algorithm (MHChain) is used.

    #### Notes
    Simulated Annealing (SA) is a probabilistic technique used for finding an
    approximation to the global optimum of a given function. It uses a cooling
    schedule and temperature to control the search process.
    """

    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        Cooler: CoolingSchedule,
        n_sim: int = 5000,
        maxiter: MAX_LIMITS = MAX_LIMITS(int(1e6), int(1e3)),
    ):
        """
        Initializes an instance of the `MHChainSA` class.

        #### Parameters
        **ObjFunc: ObjectiveFunction**
        : An instance of the `ObjectiveFunction` class representing the
          function to be optimized.

        **Cooler: CoolingSchedule**
        : An instance of the `CoolingSchedule` class representing the cooling
          schedule to be used in the simulated annealing process.

        **n_sim: int, optional (default=5000)**
        : The number of simulations to run for the simulated annealing process.

        **maxiter: MAX_LIMITS, optional**
        : An instance of the `MAX_LIMITS` class representing the maximum number
          of iterations for the simulated annealing process. If not provided,
          the default maximum number of steps is 1e6 and the default maximum
          number of epochs is 1e3.
        """
        super().__init__(ObjFunc, MHChain, Cooler, n_sim, maxiter)
