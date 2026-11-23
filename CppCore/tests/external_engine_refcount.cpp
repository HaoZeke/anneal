// MIT License
// Copyright 2023--present rgpot developers
//
// Exercise loader ownership through portable library lifecycle behaviour.
// Failed opens must unload their fixture, while the accepted engine must be
// initialized once and stay resident across repeated potential lifetimes.

#include "rgpot/ExternalPot/ExternalPot.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>

#include <unistd.h>

namespace {

int fail(const char *sentence) {
  std::fprintf(stderr, "%s\n", sentence);
  return 1;
}

rgpot_external_potential *make(const char *path, char *err,
                               std::size_t err_len) {
  err[0] = '\0';
  return rgpot_lammps_potential(path, 1.0, 8.0, 1.0, err, err_len);
}

int occurrences(const std::string &path, const std::string &event) {
  std::ifstream input(path);
  std::string line;
  int count = 0;
  while (std::getline(input, line)) {
    count += line == event;
  }
  return count;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr,
                 "usage: external_engine_refcount engine.so empty.so\n");
    return 2;
  }
  const char *engine = argv[1];
  const char *empty = argv[2];
  const std::string log =
      "/tmp/rgpot-external-lifecycle-" + std::to_string(getpid()) + ".log";
  std::remove(log.c_str());
  if (setenv("RGPOT_EXTERNAL_LIFECYCLE_LOG", log.c_str(), 1) != 0) {
    return fail("the lifecycle log environment variable could not be set");
  }
  char err[512] = {};

  for (int i = 0; i < 1000; ++i) {
    rgpot_external_potential *pot = make(empty, err, sizeof(err));
    if (pot != nullptr) {
      rgpot_external_potential_free(pot);
      return fail("a library without engine symbols was accepted");
    }
  }
  if (occurrences(log, "empty load") != 1000 ||
      occurrences(log, "empty unload") != 1000) {
    return fail("a failed constructor did not balance library load/unload");
  }

  rgpot_external_potential *warm = make(engine, err, sizeof(err));
  if (warm == nullptr) {
    std::fprintf(stderr, "engine constructor: %s\n", err);
    return 1;
  }
  rgpot_external_potential_free(warm);

  {
    rgpot::ExternalLjConfig config;
    config.u0 = 1.0;
    config.cutoff = 8.0;
    config.psi = 1.0;
    config.engine_path = engine;
    rgpot::LammpsPot original(config);
    rgpot::LammpsPot moved(std::move(original));
  }

  for (int i = 0; i < 1000; ++i) {
    rgpot_external_potential *pot = make(engine, err, sizeof(err));
    if (pot == nullptr) {
      std::fprintf(stderr, "engine constructor %d: %s\n", i, err);
      return 1;
    }
    rgpot_external_potential_free(pot);
  }

  if (occurrences(log, "engine load") != 1) {
    return fail("the resident engine initializer ran more than once");
  }
  if (occurrences(log, "engine unload") != 0) {
    return fail("freeing a potential unloaded its resident engine");
  }
  std::remove(log.c_str());
  std::puts("failed opens unload and the accepted engine stays resident");
  return 0;
}
