// MIT License
// Copyright 2023--present rgpot developers
//
// A plugin initializer re-enters the external-potential constructor. The
// registry mutex must not be held while the dynamic loader runs it.

#include "rgpot/ExternalPot/external_potentials.h"

#include <cstdio>

namespace {

const char *inner_engine;
bool reentered;
bool inner_constructed;
char inner_error[512];

} // namespace

extern "C" void rgpot_test_reenter(void) {
  reentered = true;
  rgpot_external_potential *pot =
      rgpot_lammps_potential(inner_engine, 1.0, 8.0, 1.0, inner_error,
                             sizeof(inner_error));
  if (pot != nullptr) {
    inner_constructed = true;
    rgpot_external_potential_free(pot);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr,
                 "usage: external_constructor_reentry outer.so inner.so\n");
    return 2;
  }
  inner_engine = argv[2];
  char err[512] = {};
  rgpot_external_potential *outer =
      rgpot_lammps_potential(argv[1], 1.0, 8.0, 1.0, err, sizeof(err));
  if (outer == nullptr) {
    std::fprintf(stderr, "outer engine constructor: %s\n", err);
    return 1;
  }
  rgpot_external_potential_free(outer);
  if (!reentered) {
    std::fputs("engine initializer did not re-enter the constructor\n", stderr);
    return 1;
  }
  if (!inner_constructed) {
    std::fprintf(stderr, "re-entered constructor: %s\n", inner_error);
    return 1;
  }
  std::puts("engine initializer re-entered the constructor");
  return 0;
}
