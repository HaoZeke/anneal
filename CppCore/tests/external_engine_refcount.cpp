// MIT License
// Copyright 2023--present rgpot developers
//
// Constructing and freeing an external potential must not raise the dynamic
// linker's reference on the engine. On glibc, RTLD_NOLOAD increments
// l_direct_opencount and dlclose of a library that is not RTLD_NODELETE
// decrements it; the word that moves by one is the reference count. A
// NODELETE object ignores dlclose, so the same pair reports that too.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "rgpot/ExternalPot/ExternalPot.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <malloc.h>
#include <optional>
#include <string>
#include <vector>

namespace {

int fail(const char *sentence) {
  std::fprintf(stderr, "%s\n", sentence);
  return 1;
}

std::uint32_t word_at(void *map, std::size_t offset) {
  std::uint32_t value = 0;
  std::memcpy(&value, static_cast<unsigned char *>(map) + offset, sizeof(value));
  return value;
}

// The uint32 in the link map that grows by one for RTLD_NOLOAD and shrinks
// by one for the matching dlclose. That is the loader reference.
std::optional<std::size_t> opencount_offset(const char *path, void *map) {
  // The link map is a glibc allocation. Reading past it is an over-read;
  // malloc_usable_size is the bytes that belong to this object.
  const std::size_t bytes = malloc_usable_size(map);
  if (bytes < sizeof(std::uint32_t)) {
    return std::nullopt;
  }
  std::vector<unsigned char> before(bytes);
  std::memcpy(before.data(), map, bytes);
  void *extra = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  if (extra == nullptr) {
    return std::nullopt;
  }
  std::vector<std::size_t> grew;
  for (std::size_t i = 0; i + sizeof(std::uint32_t) <= bytes; i += 4) {
    const std::uint32_t was = word_at(before.data(), i);
    const std::uint32_t now = word_at(map, i);
    if (now == was + 1) {
      grew.push_back(i);
    }
  }
  const int closed = dlclose(extra);
  std::vector<std::size_t> confirmed;
  for (std::size_t i : grew) {
    const std::uint32_t was = word_at(before.data(), i);
    if (word_at(map, i) == was) {
      confirmed.push_back(i);
    }
  }
  if (confirmed.size() != 1 || closed != 0) {
    std::fprintf(stderr,
                 "loader reference offset: %zu candidates, dlclose %d\n",
                 confirmed.size(), closed);
    return std::nullopt;
  }
  return confirmed[0];
}

std::uint32_t opencount(void *handle, std::size_t offset) {
  struct link_map *map = nullptr;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0 || map == nullptr) {
    return 0;
  }
  return word_at(map, offset);
}

rgpot_external_potential *make(const char *path, char *err, std::size_t err_len) {
  err[0] = '\0';
  return rgpot_lammps_potential(path, 1.0, 8.0, 1.0, err, err_len);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: external_engine_refcount engine.so empty.so\n");
    return 2;
  }
  const char *engine = argv[1];
  const char *empty = argv[2];
  char err[512] = {};

  void *held = dlopen(engine, RTLD_NOW | RTLD_LOCAL);
  if (held == nullptr) {
    std::fprintf(stderr, "dlopen %s: %s\n", engine, dlerror());
    return 1;
  }
  struct link_map *map = nullptr;
  if (dlinfo(held, RTLD_DI_LINKMAP, &map) != 0 || map == nullptr) {
    return fail("dlinfo did not return the engine link map");
  }
  const std::optional<std::size_t> offset = opencount_offset(engine, map);
  if (dlclose(held) != 0) {
    return fail("the calibration open could not be closed");
  }
  if (!offset) {
    return fail("the loader reference count could not be read");
  }
  if (dlopen(engine, RTLD_NOW | RTLD_NOLOAD) != nullptr) {
    return fail("the calibration open is still held");
  }

  for (int i = 0; i < 1000; ++i) {
    rgpot_external_potential *pot = make(empty, err, sizeof(err));
    if (pot != nullptr) {
      rgpot_external_potential_free(pot);
      return fail("a library without engine symbols was accepted");
    }
  }
  void *leftover = dlopen(empty, RTLD_NOW | RTLD_NOLOAD);
  if (leftover != nullptr) {
    dlclose(leftover);
    return fail("a failed constructor left its library loaded");
  }

  rgpot_external_potential *warm = make(engine, err, sizeof(err));
  if (warm == nullptr) {
    std::fprintf(stderr, "engine constructor: %s\n", err);
    return 1;
  }
  rgpot_external_potential_free(warm);

  // One NOLOAD probe, then dlclose, then a second probe. The two reads
  // differ by the probe that dlclose did not drop.
  void *first = dlopen(engine, RTLD_NOW | RTLD_NOLOAD);
  if (first == nullptr) {
    return fail("freeing the potential unloaded its engine");
  }
  const std::uint32_t first_count = opencount(first, *offset);
  if (dlclose(first) != 0) {
    return fail("dlclose of the first probe failed");
  }
  void *second = dlopen(engine, RTLD_NOW | RTLD_NOLOAD);
  if (second == nullptr) {
    return fail("the engine was unloaded by the probe's dlclose");
  }
  const std::uint32_t second_count = opencount(second, *offset);
  if (dlclose(second) != 0) {
    return fail("dlclose of the second probe failed");
  }
  if (second_count != first_count && second_count != first_count + 1) {
    std::fprintf(stderr, "noload/dlclose moved the reference by %u\n",
                 second_count - first_count);
    return 1;
  }
  const bool close_drops = second_count == first_count;

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

  void *after = dlopen(engine, RTLD_NOW | RTLD_NOLOAD);
  if (after == nullptr) {
    return fail("the engine was unloaded by freeing the potential");
  }
  const std::uint32_t after_count = opencount(after, *offset);
  dlclose(after);
  // close_drops: both probes were balanced, so the later probe matches the
  // first. Otherwise each probe sticks and the later one is two past the
  // first (the second baseline probe and this one).
  const std::uint32_t expected = close_drops ? first_count : first_count + 2;
  if (after_count != expected) {
    std::fprintf(stderr,
                 "loader reference grew: baseline %u after %u expected %u\n",
                 first_count, after_count, expected);
    return 1;
  }
  std::printf("loader reference stable at %u across 1000 constructions\n",
              after_count);
  return 0;
}
