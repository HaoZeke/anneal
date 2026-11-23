A cluster search no longer hands its objective a proposal that is not finite.
A move that cannot define its step on a geometry (two rigid molecules on top of one another, for one) gave a coordinate of NaN, and the next quench evaluated it.
Such a proposal is refused as an overlapping one is: it costs a hop and no energy call, and a run that proposes nothing else stops on the same bound.
