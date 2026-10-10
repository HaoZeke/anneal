With ``replicas`` above one, every rung of the cluster-hopping ladder hopped
at ``temperature``, so the ladder's temperatures reached only the swap test.
Rung ``k`` of ``R`` now hops at ``ladder_top^(k/(R-1))`` times the temperature
a single chain would hop at. Under ``budget_window`` the law sets the coldest
rung's temperature, and a rise a rung declines enters the barrier estimate
divided by that rung's ratio. Under ``statistical_temperature`` the estimate is
clamped to its band around ``temperature`` before the ratio multiplies it, and
only the rungs at ratio one feed the density of states it is read from. The
swap factor reads each rung's temperature from the state it holds and weighs
the funnel and energy biases every rung shares by the difference of the two
inverse temperatures. A single chain is unchanged.
