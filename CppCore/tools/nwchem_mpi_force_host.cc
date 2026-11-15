// MIT License
// Copyright 2023--present rgpot developers
//
// Multi-rank host for NWChemPot: proves host-owned MPI through the rgpot
// frontend (same public nwchemc_energy_gradient path as nwchemc's timer).
//
// Contract: every MPI rank must call force with the same geometry/params.
// Cap'n Proto carries method/geometry only — not MPI ranks or communicators.
//
// Usage:
//   mpirun -np P nwchem_mpi_force_host [options]
//
// Options:
//   --system h2|water|benzene   built-in geometry (default: h2)
//   --geom PATH                 file: n_atoms then "Z x y z" (Angstrom) lines
//   --basis NAME                default sto-3g
//   --theory NAME               default scf
//   --scf-type NAME             default rhf
//   --engine PATH               libnwchemc.so (else NWCHEMC_LIBRARY / …)
//
// Rank 0 prints one machine-readable line:
//   rgpot_mpi_force system=… ranks=P wall_s=… energy_ev=… maxabs_f=… ok=…
//
// Environment for the engine (same as potserv / eOn):
//   NWCHEMC_LIBRARY, RGPOT_NWCHEMC_ENGINE, NWCHEM_TOP, LD_LIBRARY_PATH, …

#include "MpiHost.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

namespace {

struct Geom {
  std::string label;
  std::vector<int> Z;
  std::vector<double> xyz; // n*3 Angstrom
};

static Geom geom_h2() {
  Geom g;
  g.label = "h2";
  g.Z = {1, 1};
  g.xyz = {0.0, 0.0, -0.3707, 0.0, 0.0, 0.3707};
  return g;
}

static Geom geom_water() {
  Geom g;
  g.label = "water";
  g.Z = {8, 1, 1};
  g.xyz = {0.0, 0.0, 0.1173, 0.0, 0.7572, -0.4692, 0.0, -0.7572, -0.4692};
  return g;
}

static Geom geom_benzene() {
  Geom g;
  g.label = "benzene";
  g.Z.assign(12, 0);
  g.xyz.assign(36, 0.0);
  const double cc = 1.397;
  const double ch = 1.084;
  constexpr double pi = 3.14159265358979323846;
  for (int i = 0; i < 6; ++i) {
    const double a = (pi / 3.0) * static_cast<double>(i);
    g.Z[static_cast<size_t>(i)] = 6;
    g.xyz[static_cast<size_t>(3 * i)] = cc * std::cos(a);
    g.xyz[static_cast<size_t>(3 * i + 1)] = cc * std::sin(a);
    g.xyz[static_cast<size_t>(3 * i + 2)] = 0.0;
    g.Z[static_cast<size_t>(6 + i)] = 1;
    g.xyz[static_cast<size_t>(3 * (6 + i))] = (cc + ch) * std::cos(a);
    g.xyz[static_cast<size_t>(3 * (6 + i) + 1)] = (cc + ch) * std::sin(a);
    g.xyz[static_cast<size_t>(3 * (6 + i) + 2)] = 0.0;
  }
  return g;
}

static bool load_geom_file(const std::string &path, Geom *out) {
  std::ifstream in(path);
  if (!in)
    return false;
  int n = 0;
  if (!(in >> n) || n < 1 || n > 512)
    return false;
  out->label = path;
  out->Z.assign(static_cast<size_t>(n), 0);
  out->xyz.assign(static_cast<size_t>(n) * 3u, 0.0);
  for (int i = 0; i < n; ++i) {
    int z = 0;
    double x = 0, y = 0, zz = 0;
    if (!(in >> z >> x >> y >> zz))
      return false;
    out->Z[static_cast<size_t>(i)] = z;
    out->xyz[static_cast<size_t>(3 * i)] = x;
    out->xyz[static_cast<size_t>(3 * i + 1)] = y;
    out->xyz[static_cast<size_t>(3 * i + 2)] = zz;
  }
  return true;
}

static void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: mpirun -np P %s [--system h2|water|benzene] "
               "[--geom file] [--basis B] [--theory T] [--scf-type S] "
               "[--engine PATH]\n",
               argv0);
}

static const char *env_or(const char *key, const char *fallback) {
  const char *v = std::getenv(key);
  return (v && v[0]) ? v : fallback;
}

} // namespace

int main(int argc, char **argv) {
  rgpot::tools::MpiHost host(argc, argv);
  try {
    std::string system_name = "h2";
    std::string geom_path;
    std::string basis = env_or("RGPOT_NWCHEM_BASIS", "sto-3g");
    std::string theory = env_or("RGPOT_NWCHEM_THEORY", "scf");
    std::string scf_type = env_or("RGPOT_NWCHEM_SCF_TYPE", "rhf");
    std::string engine_path;
    Geom geometry;
    std::vector<std::uint64_t> words;
    int help = 0;
    host.collective([&] {
      if (host.rank() != 0)
        return;
      for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto value = [&]() -> std::string {
          if (++i >= argc)
            throw std::runtime_error("missing value for " + option);
          return argv[i];
        };
        if (option == "--system")
          system_name = value();
        else if (option == "--geom")
          geom_path = value();
        else if (option == "--basis")
          basis = value();
        else if (option == "--theory")
          theory = value();
        else if (option == "--scf-type")
          scf_type = value();
        else if (option == "--engine")
          engine_path = value();
        else if (option == "-h" || option == "--help")
          help = 1;
        else
          throw std::runtime_error("unknown argument: " + option);
      }
      if (help)
        return;
      if (!geom_path.empty()) {
        if (!load_geom_file(geom_path, &geometry))
          throw std::runtime_error("cannot read geometry: " + geom_path);
      } else if (system_name == "h2") {
        geometry = geom_h2();
      } else if (system_name == "water") {
        geometry = geom_water();
      } else if (system_name == "benzene") {
        geometry = geom_benzene();
      } else {
        throw std::runtime_error("unknown system: " + system_name);
      }
      capnp::MallocMessageBuilder message;
      auto parameters = message.initRoot<NWChemParams>();
      parameters.setBasis(basis);
      parameters.setTheory(theory);
      parameters.setScfType(scf_type);
      parameters.setCharge(0);
      parameters.setMultiplicity(1);
      parameters.setTask("gradient");
      parameters.setEnginePath(engine_path);
      auto flat = capnp::messageToFlatArray(message);
      words.resize(flat.size());
      std::memcpy(words.data(), flat.begin(), flat.size() * sizeof(capnp::word));
    });
    host.broadcastCommand(help);
    if (help) {
      if (host.rank() == 0)
        usage(argv[0]);
      return 0;
    }
    host.broadcast(geometry.Z);
    host.broadcast(geometry.xyz);
    host.broadcast(words);
    std::unique_ptr<rgpot::NWChemPot> potential;
    host.collective([&] {
      capnp::FlatArrayMessageReader reader(
          kj::arrayPtr<const capnp::word>(
              reinterpret_cast<const capnp::word *>(words.data()), words.size()));
      const auto parameters = reader.getRoot<NWChemParams>();
      engine_path = parameters.getEnginePath().cStr();
      if (!engine_path.empty()) {
#if defined(_WIN32)
        if (_putenv_s("NWCHEMC_LIBRARY", engine_path.c_str()) != 0)
#else
        if (setenv("NWCHEMC_LIBRARY", engine_path.c_str(), 1) != 0)
#endif
          throw std::runtime_error("cannot set explicit NWChem engine path");
      }
      if (!rgpot::NWChemPot::probeAvailable())
        throw std::runtime_error("NWChem engine is not available on every rank");
    });
    host.collective([&] {
      capnp::FlatArrayMessageReader reader(
          kj::arrayPtr<const capnp::word>(
              reinterpret_cast<const capnp::word *>(words.data()), words.size()));
      potential = std::make_unique<rgpot::NWChemPot>(reader.getRoot<NWChemParams>());
      if (!potential->available())
        throw std::runtime_error("NWChem engine setup failed");
    });
    AtomMatrix positions;
    host.collective([&] {
      positions = AtomMatrix(geometry.Z.size(), 3);
      std::copy(geometry.xyz.begin(), geometry.xyz.end(), positions.data());
    });
    const std::array<std::array<double, 3>, 3> box = {
        {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
    double energy = 0;
    double maximum_force = 0;
    const auto start = std::chrono::steady_clock::now();
    host.collective([&] {
      auto [value, forces, variance] = (*potential)(positions, geometry.Z, box);
      (void)variance;
      energy = value;
      for (std::size_t i = 0; i < geometry.xyz.size(); ++i) {
        if (!std::isfinite(forces.data()[i]))
          throw std::runtime_error("NWChem returned a nonfinite force");
        maximum_force = std::max(maximum_force, std::abs(forces.data()[i]));
      }
      if (!std::isfinite(energy))
        throw std::runtime_error("NWChem returned a nonfinite energy");
    });
    const double elapsed = host.maximum(std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count());
    if (host.rank() == 0) {
      std::printf("rgpot_mpi_force system=%s n_atoms=%zu ranks=%d wall_s=%.6f "
                  "energy_ev=%.17g maxabs_f=%.17g ok=1 basis=%s theory=%s\n",
                  geometry.label.c_str(), geometry.Z.size(), host.size(), elapsed,
                  energy, maximum_force, basis.c_str(), theory.c_str());
    }
    return 0;
  } catch (const std::exception &error) {
    if (host.rank() == 0)
      std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
