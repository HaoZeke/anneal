Finite-difference checks now run the CuH2, Sutton-Chen, and covalent
force conversions when those libraries are present, and the PET-MAD
pipe when pet-mad imports. The profile negation is checked when the
build enables the capnpc profile feature. A missing library is still
named and skipped.
