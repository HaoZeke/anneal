A diagnostic quench, such as the generic packing probe's, now descends
on the plain surface under a legacy ``two_phase`` configured without a
surface portfolio. Before, only the portfolio was bypassed, so with
``Config.surfaces`` empty a probe's quenches still ran their first
phase on the compacted surface. The walk's own quenches still use
``two_phase``, and every call is charged as before.
