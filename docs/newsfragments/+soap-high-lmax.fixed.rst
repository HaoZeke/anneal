The SOAP ``nu = 3`` scalars came back infinite or NaN at ``l_max`` 7 and 8.
The factorial table in the Wigner 3j prefactor stopped at 20! and returned zero above it, so every coupling with ``l1 + l2 + l3 >= 20`` divided by zero.
The table is exact through 20! and continues as a running product, and the default ``l_max`` of 6 is unchanged.
