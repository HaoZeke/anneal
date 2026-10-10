In values-only portfolio runs the success threshold, which decides whether the
opening descent goes on, whether a descent keeps its turn, whether a GSA or
CMA-ES phase has gained and when a CMA-ES run stops, is ``1e-4 max(g, s)`` in
place of ``1e-4 max(|f|, 1)``, which turned absolute below ``|f| = 1`` and
coarsened when a constant was added to the objective. ``g`` is the gain of the
last slice that lowered the incumbent from below the start's value, and ``s``
the run's resolution: ``eps |f|`` of the first value the run finds below the
start's, but never less than the smallest positive normal double, which it is
until such a value is found (when the start's value is not finite, every
finite value is below it). Gains are differences of values and a slice from
the start's value records none, so the start's value sets no threshold, and a
constant added to the objective changes one only through the resolution of the
shifted values: the rounding of each gain, and ``s``. ``s`` changes at most
once and never falls below that positive floor, so a bounded objective still
allows only a bounded number of successes, as the restart guarantee needs.
The values-only descent likewise stops on the resolution of its values, an
iteration that lowers its value by no more than ``8 eps`` of it, or by no more
than ``1e-7`` of the success threshold, being slow, and until a curvature pair
scales the inverse Hessian each line search of a finite-difference descent
starts with a trial whose largest move is 5% of a side whatever the
gradient's size, where a small gradient used to shorten it.
Multiplying the objective by a power of two multiplies every threshold
exactly. Measured over 40 paired seeds against the loop's form before these
rules, or against the loop where putting a rule back costs a row (medians, "A
against B" with A's wins/ties/losses, two runs tying when their values differ
by at most ``1e-9 max(1, |b|)`` with ``b`` B's, rows named ``problem@budget``
from ``x0`` as in the regime notes), the descent's resolution test with its
floor trails a fixed ``1e-12`` tolerance on ``griewank10@5000`` without
``x0``, 0.0308 against 0.0221 (3/31/6, p 0.049), where the loop without the
floor is level (3/32/5). The tolerance brings that row level but puts
``griewank10@5000`` from ``x0`` behind (0.0382 against 0.0295, 2/29/9,
p 0.021), and the offset copies of ``fit6@5000`` fall back, +1e2 to -11.6
against the loop's -12.6 (0/23/17), +1e4 to -9.33 against -12.6 (0/17/23)
and -1e4 to -8.15 against -12.8 (0/24/16), with its x1e-6 copy and +1e2
control behind the loop too, though all five still lead the form before these
rules. Without the floor the test trails the tolerance on
``fit6@1000``, -4.78 against -4.89 (3/20/17), and without ``x0``, with its
x1e-6 and x2^-20 copies and its +1e2 and x1e-6 controls, on ``ellR30@20000``
and ``ellR50@20000``, where five and six seeds stop short of 0, and on the
x1e-6 control of ``ellR30@5000``; the floor brings each level (``fit6@1000``
-4.82, 4/30/6) and puts no row behind the loop without it, ``fit6@5000`` being
-12.94 against -12.83 (15/10/15). The threshold on the last gain trails
``1e-4 max(|f|, s)`` on ``rastR100@20000``, 870 against 833 (4/18/18), and on
the +1e2 copy of ``levy30@5000``, 6.4e-11 against 4.3e-12 (5/22/13); the
``|f|``-scaled threshold, measured before the descent's slow test had its
floor, brings both level but costs the offset copies: +1e4 ``rosen10@1000`` is
5.40 against the loop's 1.2e-10 (0/0/40), +1e4 ``fit6@1000`` is -0.847 against
-4.67 (1/0/39) and at 5000 -6.16 against -12.6 (1/1/38), and +1e4
``ellR30@5000`` is 4.4e-3 against 2.9e-4 (0/6/34). Of the 88 offset and x1e-6
copies, five trail their rounding-only control at p < 0.05, about as many as
chance gives and none past a correction for the 88 tests: +1e2 ``fit6@1000``
(-4.825 against -4.821, 16/0/24), +1e2 ``ackS30@5000`` (1.05e-6 against 6.9e-7,
11/8/21), -1e4 ``ellR30@1000`` (5589 against 5589, 12/0/28) and x1e-6
``rosen30`` at 1000 and 5000 (74.1 against 74.1, 12/1/27; 22.1 against 22.0,
14/0/26). With exact ties nine trail and none passes the correction either
(least p 0.00089); +1e4 ``ackley10@5000``, one of the four this adds, trails at
10/4/26 (p 0.0017), its runs ending near 1.6e-10 and its control's near
2.3e-11. Every x2^-20 copy repeats its plain row run for run.
