In values-only portfolio runs the additive surrogate and the QMC restart arm
take values in the run's unit, the power of two at or below the success
threshold, where their temperatures and floors were absolute. The surrogate
fits values, and sets its temperatures and their floors, in that unit, and the
restart arm's annealing chains weigh each rise in it against a temperature
that starts at one, so they all but never take a rise as large as the last
gain. Multiplying the objective by a power of two now leaves a whole
values-only run unchanged, short of overflow and underflow, where a run used
to depart from the unscaled one once the surrogate or the restart arm played.
