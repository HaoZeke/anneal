With ``replicas`` above one, every rung of the cluster-hopping ladder hopped
at ``temperature``, so the ladder's temperatures reached only the swap test.
Rung ``k`` of ``R`` now hops at ``ladder_top^(k/(R-1))`` times the temperature
a single chain would hop at. Under ``budget_window`` the law sets the coldest
rung's temperature, and a rise a rung declines enters the barrier estimate
divided by that rung's ratio. Under ``statistical_temperature`` the estimate is
clamped to its band around ``temperature`` before the ratio multiplies it, and
only the rungs at ratio one feed the density of states it is read from. The
swap factor reads each rung's temperature from the state it holds and weighs
the funnel bias, the packing pile and the energy bias every rung shares by the
difference of the two inverse temperatures. Under ``flat_histogram`` it weighs
the biases alone, since the flat-histogram cost is the same on every rung and
cancels from the factor with the energies. Under ``energy_bias`` the bias's
tempering factor and its deposits read the temperature a single chain would
hop at, so which rung fills its first sample no longer sets its factor. A state
keeps its basin and its validation gradient as it moves between rungs, so the
first step a rung takes after a switch is recorded from the state that rung
holds, and every rung's starting quench is validated before it is recorded. A
ladder refuses ``delayed_acceptance``: a hop its surrogate decides is tested on
the bare energy and one it abstains on with the biases, so no swap factor can
balance an exchange between rungs that hop by two weights. A single chain is
unchanged.
