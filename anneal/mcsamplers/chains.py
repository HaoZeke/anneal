from anneal.core.components import *


class MHChain(BaseChainSA):
    def __init__(self, ObjFunc: ObjectiveFunction, Temperature, Nsim=5000):
        super().__init__(ObjFunc, Temperature, Nsim)
        self.states[0] = np.zeros(ObjFunc.limits.dims)
        self.status[0] = True
        self.cand = None
        self.cur_state = self.states[0]
        self.best = None

    def step(self):
        """Here we pick a candidate which is chain-independent"""
        for step in range(1, self.Nsim):
            cand = self.ObjFunc.limits.mkpoint()
            rho = self.bounds_eval_rho(cand, step - 1)
            if self.proposal() < rho:
                self.states[step] = self.states[step - 1] + (
                    cand - self.states[step]
                )
                self.status[step] = True
            else:
                self.states[step] = self.states[step - 1]
                self.status[step] = False

    def bounds_eval_rho(self, cand, chain_id):
        try:
            val = self.TargetDistrib(self.states[chain_id])
        except OutOfBounds:
            self.states[chain_id] = self.ObjFunc.limits.clip(
                self.states[chain_id]
            )
            val = self.TargetDistrib(self.states[chain_id])
        return self.TargetDistrib(cand) / val

    def proposal(self):
        return np.random.uniform(0, 1, 1)

    def __call__(self):
        self.step()
        # TODO: Why is that last one out of bounds?
        energies = np.array([self.ObjFunc(x) for x in self.states[:-1]])
        min_state = self.states[energies.argmin()]
        self.best = FPair(pos=min_state, val=np.min(energies))
        return self.best
