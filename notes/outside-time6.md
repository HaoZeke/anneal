# Outside-time 6: skip assignments the nearest-image bound already settles

Base `22110d77784e6a9e8282c2f0758ec980f55ea9eb`. Same session, serial,
hostname bench, seeds 2000–2011.

The fivefold hop draws the axis closest to each of the two reconstruction
lengths, and it yields unless two axes sit below the fivefold cut. The
nearest-image length is a lower bound on the assignment residual. Once two
residuals are under the cut, an axis whose nearest image is already farther
from both lengths cannot change the draw, so that assignment is not run.
The chosen pair matches the full table. Pins stay
`a22093e707faa91d683813fa7cc507706c5265e6`.

Sorted pair distances do not settle it: a rotation has the same spectrum as
the cluster. The energy gap is applied after the proposal, so it does not
decide which axis is drawn.

| system | develop µs/call | pruned µs/call | change | hits |
| --- | ---: | ---: | ---: | --- |
| Au98 | 26.407 | 26.080 | −1.24% | 0 / 0 |
| LJ150 | 40.768 | 40.922 | +0.38% | 6 / 6 |

Charged evaluations and the best energy match on every seed. The 10% gate
is not met.
