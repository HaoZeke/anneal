<!-- Copyright 2023--present rgpot developers -->

Python wheel CI tests now ship the EAM table fixtures under `CppCore/tests/data/eam_table`. The C++ coverage build enables `-fprofile-update=atomic` for concurrent OpenMP gcov counters, and the capture script disables synthetic exception-unwind branches under lcov 2 so capture and extract run without error suppressions.
