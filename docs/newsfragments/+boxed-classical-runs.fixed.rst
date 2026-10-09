Classical ``run`` / ``run_qmc`` chains now stay inside ``[low, high]``:
the Boltzmann, Fast, and GSA presets run on the box-constrained
neighborhood with mirror-reflected proposals, so bounded objectives never
see an out-of-box evaluation. Both drivers accept an optional ``x0``
starting position (clipped into the box), and the portfolio's
parallel-tempering communicating chains and pilot-tuned classical arm use
the same boxed variants.
