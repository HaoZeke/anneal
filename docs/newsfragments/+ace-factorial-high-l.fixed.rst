The Wigner 3j prefactor divides by ``(l1 + l2 + l3 + 1)!``. The
factorial table stopped at ``20!`` and returned 0 above it, so every
coupling with ``l1 + l2 + l3 >= 20`` was infinite or NaN at ``l_max`` 7
and 8. Larger factorials are the running product of that table.
``l_max`` 6 never reached the hole.
