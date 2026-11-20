Copyright (c) anneal developers

# Why pair scale cannot keep its gains

Negative record. Nothing here is proposed for develop.

Pair scale is `Config::recommended_for_pair`. Every preset length
is multiplied by the pair minimum over `2^(1/6)`, and the record
gradient is divided by that factor. For Morse the factor is
0.8908987181403393. The temperature stays 0.8, because the well
depth is 1. Lennard-Jones `length_scale()` is exactly 1, so
`recommended_for_pair` is the same builder call as `recommended`.
LJ38, LJ75 and LJ98 were not searched again. Their outside ratio
against recommended is 1.

Seeds 2000-2011, serial `h2h_morse`, develop 61fa67cf, paper
budgets (400000 at 38, 1000000 at 55, 4000000 at 80). A hit is a
quenched energy at or below the Cambridge reference plus 0.001.
The median of `outside_s/charged` is the pair-scale median
divided by the recommended median.

| system | recommended | pair scale | outside ratio |
| --- | --- | --- | --- |
| m38r6 | 12/12 | 12/12 | 0.785 |
| m38r14 | 4/12 | 9/12 | 0.811 |
| m55r14 | 9/12 | 8/12 | 0.669 |
| m80r6 | 12/12 | 12/12 | 0.696 |
| m80r14 | 6/12 | 8/12 | 0.747 |

The m55 net of one hit is three seeds lost and two gained.
Pair scale loses m55 seeds 2003, 2006 and 2008, and gains 2005
and 2009. It loses m38r14 seeds 2001 and 2008. It loses m80r14
seeds 2007 and 2010 while gaining 2002, 2004, 2008 and 2011.

## The runs diverge at the first quench

The start is the same draw. `min_separation` and the container
scale by the same factor, so the rejection decisions coincide.
On every loss seed the pair-scale coordinates are the
recommended coordinates times 0.8908987181403393, with a max
absolute error under 1e-10. The search generator is reseeded
from the same seed.

Morse is not homogeneous, so the first quench already differs.
The record lists diverge at index 0, with a basin count of 0.

| seed | recommended | pair scale |
| --- | --- | --- |
| m55 2003 | 3875, -174.174828 | 1931, -182.300345 |
| m55 2006 | 2876, -192.034003 | 2154, -181.388619 |
| m55 2008 | 3834, -184.432010 | 1698, -188.410102 |
| m38 2001 | 2293, -122.373615 | 2121, -122.745963 |
| m38 2008 | 1938, -120.173472 | 800, -125.885795 |
| m80 2007 | 2808, -274.045608 | 1641, -285.408048 |
| m80 2010 | 6116, -253.424761 | 2338, -272.244430 |

m55 2003, m55 2008, m38 2001, m38 2008 and m80 2007 share no
later record. m55 2006 meets at -220.49847976, recommended at
calls 286543 and 336869 and pair scale at call 221283.
Recommended leaves it and hits 55C at call 519192. Pair scale
stops there. m80 2010 meets at -340.38733557, recommended at
call 1514022 and pair scale at call 2087416. Recommended hits
80G at call 2973511. Pair scale stops on that shelf.

## Neither half of the rescaling is a fix

`scaled_start` keeps the recommended operators and uses the
pair-scale start. `scaled_ops` keeps the recommended start and
uses the pair-scale operators. A hit is marked, otherwise the
best energy.

| seed | pair scale | scaled_start | scaled_ops |
| --- | --- | --- | --- |
| m55 2003 | -220.49847976 | hit | -220.49016163 |
| m55 2006 | -220.49847976 | hit | -220.51853338 |
| m55 2008 | -219.94332884 | hit | hit |
| m38 2001 | -142.69789782 | -141.76806589 | -142.65482948 |
| m38 2008 | -142.71594037 | -142.69789782 | hit |
| m80 2007 | -340.80256253 | hit | hit |
| m80 2010 | -340.38733557 | -340.38733557 | -340.80256253 |

m55 2003 and 2006 follow the operators: the scaled start still
hits, and the scaled operators miss from the original start.
m55 2006's operator-only miss is 55D, -220.51853338. m38 2008
follows the start: scaled operators on the original start still
hit. m55 2008 and m80 2007 miss only when both are scaled.
m38 2001 and m80 2010 miss when either half is scaled. They hit
only on the unscaled preset.

A rescaling that keeps the m38r14 and m80r14 gains has to scale
the start, the operators, or both. Each of those drops m38 seed
2001 and m80 seed 2010. The halves were not run on the gain
seeds. No other rule was added. Rows are rows.csv.
