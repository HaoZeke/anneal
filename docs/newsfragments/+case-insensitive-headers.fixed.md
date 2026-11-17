The headers in `include/rgpot` include each other by file name, so `rgpot/rgpot.hpp` compiles on a case-insensitive file system such as APFS or NTFS.
There, with `CppCore` ahead of `include` on the search path, `rgpot/potential.hpp` also named `CppCore/rgpot/Potential.hpp`, and code using `rgpot.hpp` failed with "no member named 'PotentialHandle'".
