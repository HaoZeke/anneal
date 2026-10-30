``Config::with_loose_quench`` stops every search relaxation at four times
``record_gradient`` and continues a full relaxation to ``record_gradient``
only when it stops within a window of the incumbent. The window is open
for the first three continuations and then twice the largest drop any of
them made. Every continuation is charged; ``RelaxStats::refines`` and
``RelaxStats::refine_charged`` count them. ``Config::recommended_molecular``
is unchanged.
