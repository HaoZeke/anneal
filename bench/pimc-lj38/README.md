# LJ38 path-integral annealing

32 replicas. 40 local-optimisation rounds. One round every 10000 steps. Last round at step 400000. Input length 400001. Code revision `4e86f5fd`.

The paper gives no LJ55 or LJ75 schedule; it says larger clusters need longer annealing times for the same success probability, so those sizes scale the 40 LJ38 rounds by N/38, which is 58 rounds ending at step 580000 for LJ55 and 79 rounds ending at step 790000 for LJ75, with the same 32 replicas and 10000-step interval.

| Arm | Hits | Median calls | Median wall (s) | Inside us | Outside us | Code revision |
| --- | --- | --- | --- | --- | --- | --- |
| anneal | 12/12 | 13469.5 | 0.224667 | 11.9589 | 4.8309 | f2a628c0 |
| gmin | 7/12 | 389399 | 3.792015 | 7.1525 | 2.5612 | aac1548a |
| rpqa_paper | 0/6 | 12925808 | 7751.411038 | 210.4986 | 389.3438 | 4e86f5fd |

| Seed | Hit | Best energy | Force calls | Wall (s) | Inside us | Outside us | Code revision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2000 | 0 | -173.13450824 | 12904608 | 7741.751243 | 210.7652 | 389.1563 | 4e86f5fd |
| 2001 | 0 | -173.13450824 | 14748672 | 8713.107052 | 212.6269 | 378.1454 | 4e86f5fd |
| 2002 | 0 | -170.99005171 | 12940224 | 7761.070833 | 210.2321 | 389.5312 | 4e86f5fd |
| 2003 | 0 | -170.99005171 | 12926720 | 7723.013968 | 209.1776 | 388.2682 | 4e86f5fd |
| 2004 | 0 | -170.99005171 | 12924896 | 8137.186797 | 222.9327 | 406.6420 | 4e86f5fd |
| 2005 | 0 | -171.35753078 | 12923520 | 7725.678026 | 208.1988 | 389.6010 | 4e86f5fd |
