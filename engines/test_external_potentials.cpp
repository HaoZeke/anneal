// MIT License
// Copyright 2023--present rgpot developers
//
// The dimer in engines/compare_geom.txt is -0.3203213355 eV through
// every constructor. GROMACS and OpenMM factors are the SI values.

#include "lj_eval.hpp"
#include "rgpot/ExternalPot/external_potentials.h"
#include "rgpot/ExternalPot/md_units.hpp"
#include "rgpot/units.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <filesystem>
#include <unistd.h>
#endif

namespace {

int fail(const char *sentence) {
  std::fprintf(stderr, "%s\n", sentence);
  return 1;
}

bool close(double got, double ref, double tol) {
  return std::abs(got - ref) <= tol;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: test_external_potentials liblj_engine.so "
                         "libnative_lj_engine.so\n");
    return 2;
  }
  const double published_kj = 96.48533212331002;
  if (!close(rgpot::md::kKjPerEv, published_kj, 1e-12)) {
    std::fprintf(stderr, "kJ/mol per eV %.16e published %.16e\n",
                 rgpot::md::kKjPerEv, published_kj);
    return 1;
  }
  if (rgpot::md::kAngstromPerNm != 10.0 || rgpot::md::kNmPerAngstrom != 0.1) {
    return fail("nanometre scale is not 10 Angstrom");
  }
  if (rgpot_md_kj_per_ev() != rgpot::md::kKjPerEv ||
      rgpot_md_nm_per_angstrom() != rgpot::md::kNmPerAngstrom) {
    return fail("C factor getters disagree with the stored constants");
  }
  const double parsed = rgpot::units::unit_conversion_factor("eV", "kJ/mol");
  if (!close(parsed, rgpot::md::kKjPerEv, 1e-12)) {
    std::fprintf(stderr, "unit parser %.16e stored %.16e\n", parsed,
                 rgpot::md::kKjPerEv);
    return 1;
  }
  const double nm = rgpot::units::unit_conversion_factor("nm", "angstrom");
  if (nm != 10.0) {
    return fail("unit parser does not give 10 Angstrom per nm");
  }

  const double xyz[6] = {0, 0, 0, 1.5, 0, 0};
  const int z[2] = {1, 1};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double direct_f[6];
  double direct_e = 0;
  int has_stress = 0;
  double stress[9];
  if (rgpot_lj_eval(xyz, z, 2, box, 1.0, 8.0, 1.0, direct_f, &direct_e,
                    &has_stress, stress) != 0) {
    return fail("rgpot_lj_eval rejected the dimer");
  }
  char printed[64];
  std::snprintf(printed, sizeof(printed), "%.10f", direct_e);
  if (std::strcmp(printed, "-0.3203213355") != 0) {
    std::fprintf(stderr, "direct energy prints as %s\n", printed);
    return 1;
  }

  char err[512] = {};
  rgpot_external_potential *lammps =
      rgpot_lammps_potential(argv[1], 1.0, 8.0, 1.0, err, sizeof(err));
  if (lammps == nullptr) {
    std::fprintf(stderr, "lammps constructor: %s\n", err);
    return 1;
  }
  double energy = 0;
  double forces[6];
  if (rgpot_external_potential_energy_forces(lammps, xyz, z, 2, box, forces,
                                             &energy, err, sizeof(err)) != 0) {
    std::fprintf(stderr, "lammps energy: %s\n", err);
    return 1;
  }
  std::snprintf(printed, sizeof(printed), "%.10f", energy);
  if (std::strcmp(printed, "-0.3203213355") != 0 ||
      !close(energy, direct_e, 1e-12)) {
    std::fprintf(stderr, "lammps %s direct %.16e\n", printed, direct_e);
    return 1;
  }
  for (int i = 0; i < 6; ++i) {
    if (!close(forces[i], direct_f[i], 1e-12)) {
      std::fprintf(stderr, "lammps force %d %g ref %g\n", i, forces[i],
                   direct_f[i]);
      return 1;
    }
  }
  rgpot_external_potential_free(lammps);
  std::printf("lammps energy %s eV\n", printed);
#if !defined(_WIN32)
  if (dlopen(argv[1], RTLD_NOW | RTLD_NOLOAD) == nullptr) {
    return fail("freeing the lammps potential unloaded its engine library");
  }
#endif

  auto check_native = [&](const char *name, auto *constructor) -> int {
    err[0] = '\0';
    rgpot_external_potential *pot =
        constructor(argv[2], 1.0, 8.0, 1.0, err, sizeof(err));
    if (pot == nullptr) {
      std::fprintf(stderr, "%s constructor: %s\n", name, err);
      return 1;
    }
    double got = 0;
    double got_f[6];
    if (rgpot_external_potential_energy_forces(pot, xyz, z, 2, box, got_f, &got,
                                               err, sizeof(err)) != 0) {
      std::fprintf(stderr, "%s energy: %s\n", name, err);
      rgpot_external_potential_free(pot);
      return 1;
    }
    std::snprintf(printed, sizeof(printed), "%.10f", got);
    if (std::strcmp(printed, "-0.3203213355") != 0 ||
        !close(got, direct_e, 1e-9)) {
      std::fprintf(stderr, "%s %s direct %.16e\n", name, printed, direct_e);
      rgpot_external_potential_free(pot);
      return 1;
    }
    for (int i = 0; i < 6; ++i) {
      if (!close(got_f[i], direct_f[i], 1e-8)) {
        std::fprintf(stderr, "%s force %d %g ref %g\n", name, i, got_f[i],
                     direct_f[i]);
        rgpot_external_potential_free(pot);
        return 1;
      }
    }
    rgpot_external_potential_free(pot);
    std::printf("%s energy %s eV\n", name, printed);
    return 0;
  };
  if (check_native("gromacs", rgpot_gromacs_potential) != 0) {
    return 1;
  }
  if (check_native("openmm", rgpot_openmm_potential) != 0) {
    return 1;
  }

  err[0] = '\0';
  rgpot_external_potential *in_process =
      rgpot_gromacs_potential(nullptr, 1.0, 8.0, 1.0, err, sizeof(err));
  if (in_process == nullptr) {
    std::fprintf(stderr, "in-process gromacs: %s\n", err);
    return 1;
  }
  double bare = 0;
  if (rgpot_external_potential_energy_forces(in_process, xyz, z, 2, box, forces,
                                             &bare, err, sizeof(err)) != 0 ||
      !close(bare, direct_e, 1e-12)) {
    std::fprintf(stderr, "in-process gromacs %.16e\n", bare);
    return 1;
  }
  err[0] = '\0';
  if (rgpot_external_potential_energy_forces(in_process, nullptr, nullptr, 0,
                                             nullptr, nullptr, &bare, err,
                                             sizeof(err)) != 1 ||
      std::strstr(err, "needs a potential and a geometry") == nullptr) {
    std::fprintf(stderr, "empty geometry with a null box: %s\n", err);
    return 1;
  }
  rgpot_external_potential_free(in_process);

  const double inf = HUGE_VAL;
  const double bad_lj[3][3] = {{inf, 8.0, 1.0}, {1.0, inf, 1.0}, {1.0, 8.0, inf}};
  for (const auto &cfg : bad_lj) {
    err[0] = '\0';
    rgpot_external_potential *rejected =
        rgpot_gromacs_potential(nullptr, cfg[0], cfg[1], cfg[2], err, sizeof(err));
    if (rejected != nullptr || std::strstr(err, "u0, cutoff and psi") == nullptr) {
      std::fprintf(stderr, "infinite LJ parameter was accepted: %s\n", err);
      rgpot_external_potential_free(rejected);
      return 1;
    }
  }

  const char *openmm = std::getenv("RGPOT_OPENMM_ENGINE");
  if (openmm == nullptr || openmm[0] == '\0') {
    std::printf("SKIP openmm engine: OpenMM was not found at configure time, "
                "so libopenmm_engine.so was not built\n");
  } else {
    err[0] = '\0';
    rgpot_external_potential *real =
        rgpot_openmm_potential(openmm, 1.0, 8.0, 1.0, err, sizeof(err));
    if (real == nullptr) {
      std::fprintf(stderr, "openmm engine: %s\n", err);
      return 1;
    }
    double real_e = 0;
    double real_f[6];
    if (rgpot_external_potential_energy_forces(
            real, xyz, z, 2, box, real_f, &real_e, err, sizeof(err)) != 0) {
      std::fprintf(stderr, "openmm engine energy: %s\n", err);
      return 1;
    }
    // OpenMM's own expression matches rgpot-lj to 1e-6 eV and the force
    // to 1e-5 eV/Angstrom, the same window as engines/openmm/run_openmm.py.
    std::snprintf(printed, sizeof(printed), "%.10f", real_e);
    if (!close(real_e, direct_e, 1e-6) || !close(real_f[3], direct_f[3], 1e-5)) {
      std::fprintf(stderr, "openmm engine %s (%.16e) force %g ref %g\n", printed,
                   real_e, real_f[3], direct_f[3]);
      return 1;
    }
    rgpot_external_potential_free(real);
    std::printf("openmm engine energy %s eV (rgpot-lj -0.3203213355)\n", printed);
  }

  err[0] = '\0';
  rgpot_external_potential *missing = rgpot_lammps_potential(
      "./liblj_engine_missing.so", 1.0, 8.0, 1.0, err, sizeof(err));
  if (missing != nullptr ||
      std::strstr(err, "liblj_engine_missing") == nullptr) {
    std::fprintf(stderr, "missing engine error was: %s\n", err);
    return 1;
  }

#if !defined(_WIN32)
  namespace fs = std::filesystem;
  const fs::path scratch =
      fs::temp_directory_path() /
      ("rgpot_external_resident_" + std::to_string(getpid()));
  fs::create_directories(scratch);
  const fs::path copy = scratch / "liblj_engine_copy.so";
  fs::copy_file(argv[1], copy, fs::copy_options::overwrite_existing);
  err[0] = '\0';
  rgpot_external_potential *first =
      rgpot_lammps_potential(copy.c_str(), 1.0, 8.0, 1.0, err, sizeof(err));
  if (first == nullptr) {
    std::fprintf(stderr, "copied engine: %s\n", err);
    return 1;
  }
  rgpot_external_potential_free(first);
  const fs::path staging = scratch / "liblj_engine_staging.so";
  fs::copy_file(argv[1], staging, fs::copy_options::overwrite_existing);
  fs::rename(staging, copy);
  err[0] = '\0';
  rgpot_external_potential *replaced =
      rgpot_lammps_potential(copy.c_str(), 1.0, 8.0, 1.0, err, sizeof(err));
  if (replaced != nullptr || std::strstr(err, "was replaced") == nullptr) {
    std::fprintf(stderr, "replaced engine error was: %s\n", err);
    rgpot_external_potential_free(replaced);
    return 1;
  }

  const fs::path target_a = scratch / "liblj_engine_target_a.so";
  const fs::path target_b = scratch / "liblj_engine_target_b.so";
  const fs::path alias = scratch / "liblj_engine_alias.so";
  fs::copy_file(argv[1], target_a, fs::copy_options::overwrite_existing);
  fs::copy_file(argv[1], target_b, fs::copy_options::overwrite_existing);
  fs::create_symlink(target_a, alias);
  rgpot_external_potential *through_alias =
      rgpot_lammps_potential(alias.c_str(), 1.0, 8.0, 1.0, err, sizeof(err));
  if (through_alias == nullptr) {
    std::fprintf(stderr, "symlinked engine: %s\n", err);
    return 1;
  }
  rgpot_external_potential_free(through_alias);
  fs::remove(alias);
  fs::create_symlink(target_b, alias);
  err[0] = '\0';
  rgpot_external_potential *retargeted =
      rgpot_lammps_potential(alias.c_str(), 1.0, 8.0, 1.0, err, sizeof(err));
  if (retargeted != nullptr || std::strstr(err, "retargeted") == nullptr) {
    std::fprintf(stderr, "retargeted symlink error was: %s\n", err);
    rgpot_external_potential_free(retargeted);
    return 1;
  }
  fs::remove_all(scratch);
  std::printf("replaced files and retargeted symlinks are refused\n");
#endif
  return 0;
}
