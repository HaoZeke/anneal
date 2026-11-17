Deduplicated three-body inner loops and neighbour-cache synchronization in
`SWSi.cc` and `TersoffSiC.cc` via `ThreeBodyCommon.hpp`, shared cell-team
pair walks, free-box fallbacks, and the 12-6 Lennard-Jones pair kernel across
`LJPot`, `LJClusterPot`, `MorsePot`, `ZBLPot`, `SuttonChenPot`, and
`EAMTablePot` via `PairEval.hpp` and `LJPair.hpp`, and unified orthorhombic
minimum-image and Morse/density evaluation in `EAM.cpp`.
