The finite-difference descent of the values-only loop, which
``global_optimize`` runs with no ``grad_fn`` or ``noise_sigma`` under the
``auto`` policy and ``minimize`` runs by default, took central differences on
a side too narrow for their step, as one a unit in the last place wide, with
steps that round away: each such gradient asked for the iterate itself once
or twice, where forward differences ask for nothing on that side. A central
stencil whose steps round away now leaves the side to the one-sided rule and,
when that one's do too, to the forward rule, which asks for nothing there and
takes the derivative along the side as zero. On ``[-1, 1] x [1, 1 + ulp] x
[-1, 1]``, from ``(0.4, 1, -0.3)`` and from ``(0.4, 1 + ulp, -0.3)``,
``minimize`` on a sphere centred at 0.3 and on its base-10 logarithm asked
for 42 points a second time in the 40 runs of 40 evaluations over 10 seeds,
and now asks for none; on Rosenbrock's function, whose descent took no
central differences on that side, every run is as before. The other runs
part from their earlier paths at the first such gradient. At 300 evaluations
each ends at the value it did. At 40, from the upper end, the sphere's runs
end higher on 7 of the 10 seeds, by at most 3.3e-16, because the stencil that
rounded onto the iterate also asked for the side's lower end, where the
sphere is lower by about that much, and the logarithm's end higher on 3, by
at most 2.8e-16, and lower on 1, by 2.9e-15; from the lower end the sphere's
runs end at the values they did, and the logarithm's too but for one that
ends lower, by 3.1e-15.
