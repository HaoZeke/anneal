A replica ladder on a rigid-body move library now starts every rung on the rigid chart.
Each rung after the first takes the first rung's quenched start, because a rigid chain moves only to minima it has quenched.
Those rungs used to start as atomic clusters, with three coordinates per molecule where the chart needs six.
``Config::for_tip4p`` with more than one replica therefore panicked in the TIP4P potential before its first hop.
