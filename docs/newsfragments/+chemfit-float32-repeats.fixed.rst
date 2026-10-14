With a float32 or float16 leaf, the ChemFit bridges' portfolio called the
fitter again at parameters it had already given it, as candidates that
differ in their float64 coordinates can round to one point of the leaf's
dtype. The bridges now match a candidate on the parameters the fitter is
given, each coordinate of such a leaf rounded to the leaf's dtype, and
answer a match with the loss of the earlier call. On a fitter with a float32
leaf of two coordinates and a float64 parameter, ``fit_anneal`` with seed 0
and a budget of 200 made 200 calls at 183 distinct parameter sets. With this
and the answers the portfolio now gives to a candidate it has asked before,
it makes 183 calls, and 178 at 178 with ``batch_size=4``. Where the loss is a
function of those parameters, the portfolio is given the loss the call would
have returned, so answering a match changes no candidate it asks and no end
of a fit. An answered candidate still counts against the
budget, but it is not a call of the fitter, so it does not count toward
``tell_every``, and a batch of answered candidates has no ``step``. This
changes no call of a fitter whose leaves are all float64.
