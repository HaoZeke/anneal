The regime notes and the catalog describe the floor 0.11.0 put on the
values-only descent's slow test, which the 0.11.0 changelog left out: an
iteration is slow when it lowers the descent's value by no more than
``8 eps`` of it, or by no more than ``1e-7`` of the success threshold. That
changelog said an iteration was slow when it lowered the value by less than
``8 eps`` of it, gave the resolution test's figures from before the floor,
and said a slow test at a fraction of the threshold cost more. Measured over
40 paired seeds (medians, "A against B" with A's wins/ties/losses, two runs
tying when their values differ by at most ``1e-9 max(1, |b|)`` with ``b``
B's, rows named ``problem@budget`` from ``x0`` as in the regime notes), the
floor brings level each row on which that changelog had the test trail a
fixed ``1e-12`` tolerance (``fit6@1000`` -4.82, 4/30/6) and puts no row
behind the loop without it, ``fit6@5000`` being -12.94 against -12.83
(15/10/15). With the floor the test trails the tolerance on
``griewank10@5000`` without ``x0``, 0.0308 against 0.0221 (3/31/6, p 0.049),
where the loop without the floor is level (3/32/5). The tolerance brings that
row level but puts ``griewank10@5000`` from ``x0`` behind (0.0382 against
0.0295, 2/29/9, p 0.021), and the offset copies of ``fit6@5000`` fall back,
+1e2 to -11.6 against the loop's -12.6 (0/23/17), +1e4 to -9.33 against
-12.6 (0/17/23) and -1e4 to -8.15 against -12.8 (0/24/16), with its x1e-6
copy and +1e2 control behind the loop too, though all five still lead the
loop's form before the rules that changelog describes. The figures for the
``|f|``-scaled threshold were measured before the floor, and its costs on the
offset copies are against the loop, not the rules replaced, +1e4
``rosen10@1000`` being 5.40 against the loop's 1.2e-10 (0/0/40). With the
floor five of the 88 offset and x1e-6 copies trail their rounding-only
control at p < 0.05, where that changelog counted six, and none past a
correction for the 88 tests: +1e4 ``ackley10@5000`` is no longer among them,
and +1e2 ``fit6@1000`` is -4.825 against -4.821 (16/0/24). With exact ties
nine trail and none passes the correction either (least p 0.00089); +1e4
``ackley10@5000``, one of the four this adds, trails at 10/4/26 (p 0.0017),
its runs ending near 1.6e-10 and its control's near 2.3e-11.
