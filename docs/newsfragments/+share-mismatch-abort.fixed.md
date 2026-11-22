<!-- Copyright 2023--present rgpot developers -->
A rejected `shareFromCalculator` now waits until every rank has written the mismatch before `MPI_Abort`, so a peer killed by the first abort no longer leaves an empty log.
