#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/EMT/Asap/Exception.h"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace {

struct Args {
  std::string potential;
  std::size_t atoms = 0;
  int threads = 0;
  int samples = 0;
  double sample_seconds = 0.0;
  std::optional<std::string> dump_force;
};

[[noreturn]] void usage(const char *argv0) {
  std::cerr << "usage: " << argv0
            << " --potential NAME --atoms N --threads N --samples N "
               "--sample-seconds SECONDS [--dump-force FILE]\n";
  std::exit(2);
}

Args parse_args(int argc, char **argv) {
  Args args;
  std::unordered_map<std::string, std::string> values;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) {
      usage(argv[0]);
    }
    const std::string key = argv[i];
    if (key.rfind("--", 0) != 0) {
      usage(argv[0]);
    }
    values[key] = argv[i + 1];
  }

  auto require = [&](const char *key) -> std::string {
    const auto it = values.find(key);
    if (it == values.end() || it->second.empty()) {
      usage(argv[0]);
    }
    return it->second;
  };

  args.potential = require("--potential");
  args.atoms = static_cast<std::size_t>(std::stoull(require("--atoms")));
  args.threads = std::stoi(require("--threads"));
  args.samples = std::stoi(require("--samples"));
  args.sample_seconds = std::stod(require("--sample-seconds"));
  const auto dump_it = values.find("--dump-force");
  if (dump_it != values.end()) {
    args.dump_force = dump_it->second;
  }

  if (args.atoms == 0 || args.threads < 1 || args.samples < 1 ||
      !(args.sample_seconds > 0.0)) {
    usage(argv[0]);
  }
  return args;
}

enum class LatticeKind : std::uint8_t { Fcc, Bcc, DiamondPrimitive };

struct LatticeData {
  std::vector<double> positions;
  std::vector<int> atomic_numbers;
  std::array<double, 9> box{};
};

std::tuple<int, int, int> factor_triplet(std::size_t cells) {
  if (cells == 0) {
    throw std::invalid_argument("factor_triplet requires cells > 0");
  }
  int best_x = 1;
  int best_y = 1;
  int best_z = static_cast<int>(cells);
  double best_score = std::numeric_limits<double>::infinity();
  const int limit = static_cast<int>(std::cbrt(static_cast<double>(cells))) + 2;
  for (int x = 1; x <= limit; ++x) {
    if (cells % static_cast<std::size_t>(x) != 0) {
      continue;
    }
    const std::size_t yz = cells / static_cast<std::size_t>(x);
    const int y_limit = static_cast<int>(std::sqrt(static_cast<double>(yz))) + 2;
    for (int y = x; y <= y_limit; ++y) {
      if (yz % static_cast<std::size_t>(y) != 0) {
        continue;
      }
      const int z = static_cast<int>(yz / static_cast<std::size_t>(y));
      if (z < y) {
        continue;
      }
      const double ratio =
          static_cast<double>(z) / static_cast<double>(x);
      const double score = ratio + static_cast<double>(z - x);
      if (score < best_score) {
        best_score = score;
        best_x = x;
        best_y = y;
        best_z = z;
      }
    }
  }
  return {best_x, best_y, best_z};
}

LatticeData build_fcc(std::size_t atoms, double lattice_a, int z) {
  if (atoms % 4 != 0) {
    throw std::invalid_argument("fcc requires atom count divisible by 4");
  }
  const auto [nx, ny, nz] = factor_triplet(atoms / 4);
  const std::array<std::array<double, 3>, 4> basis{{
      {0.0, 0.0, 0.0},
      {0.0, 0.5, 0.5},
      {0.5, 0.0, 0.5},
      {0.5, 0.5, 0.0},
  }};
  LatticeData out;
  out.positions.reserve(atoms * 3);
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &site : basis) {
          out.positions.push_back((static_cast<double>(ix) + site[0]) * lattice_a);
          out.positions.push_back((static_cast<double>(iy) + site[1]) * lattice_a);
          out.positions.push_back((static_cast<double>(iz) + site[2]) * lattice_a);
        }
      }
    }
  }
  out.atomic_numbers.assign(atoms, z);
  out.box = {
      static_cast<double>(nx) * lattice_a, 0.0, 0.0, 0.0,
      static_cast<double>(ny) * lattice_a, 0.0, 0.0, 0.0,
      static_cast<double>(nz) * lattice_a};
  return out;
}

LatticeData build_bcc(std::size_t atoms, double lattice_a, int z) {
  if (atoms % 2 != 0) {
    throw std::invalid_argument("bcc requires atom count divisible by 2");
  }
  const auto [nx, ny, nz] = factor_triplet(atoms / 2);
  const std::array<std::array<double, 3>, 2> basis{{
      {0.0, 0.0, 0.0},
      {0.5, 0.5, 0.5},
  }};
  LatticeData out;
  out.positions.reserve(atoms * 3);
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &site : basis) {
          out.positions.push_back((static_cast<double>(ix) + site[0]) * lattice_a);
          out.positions.push_back((static_cast<double>(iy) + site[1]) * lattice_a);
          out.positions.push_back((static_cast<double>(iz) + site[2]) * lattice_a);
        }
      }
    }
  }
  out.atomic_numbers.assign(atoms, z);
  out.box = {
      static_cast<double>(nx) * lattice_a, 0.0, 0.0, 0.0,
      static_cast<double>(ny) * lattice_a, 0.0, 0.0, 0.0,
      static_cast<double>(nz) * lattice_a};
  return out;
}

LatticeData build_diamond_primitive(std::size_t atoms, double lattice_a, int z) {
  if (atoms % 2 != 0) {
    throw std::invalid_argument(
        "diamond primitive cell requires atom count divisible by 2");
  }
  const auto [nx, ny, nz] = factor_triplet(atoms / 2);
  const std::array<std::array<double, 3>, 3> a{{
      {0.0, 0.5 * lattice_a, 0.5 * lattice_a},
      {0.5 * lattice_a, 0.0, 0.5 * lattice_a},
      {0.5 * lattice_a, 0.5 * lattice_a, 0.0},
  }};
  const std::array<std::array<double, 3>, 2> basis{{
      {0.0, 0.0, 0.0},
      {0.25, 0.25, 0.25},
  }};

  LatticeData out;
  out.positions.reserve(atoms * 3);
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &site : basis) {
          const double fx = static_cast<double>(ix) + site[0];
          const double fy = static_cast<double>(iy) + site[1];
          const double fz = static_cast<double>(iz) + site[2];
          const double x = fx * a[0][0] + fy * a[1][0] + fz * a[2][0];
          const double y = fx * a[0][1] + fy * a[1][1] + fz * a[2][1];
          const double zc = fx * a[0][2] + fy * a[1][2] + fz * a[2][2];
          out.positions.push_back(x);
          out.positions.push_back(y);
          out.positions.push_back(zc);
        }
      }
    }
  }
  out.atomic_numbers.assign(atoms, z);
  out.box = {
      static_cast<double>(nx) * a[0][0], static_cast<double>(nx) * a[0][1],
      static_cast<double>(nx) * a[0][2], static_cast<double>(ny) * a[1][0],
      static_cast<double>(ny) * a[1][1], static_cast<double>(ny) * a[1][2],
      static_cast<double>(nz) * a[2][0], static_cast<double>(nz) * a[2][1],
      static_cast<double>(nz) * a[2][2]};
  return out;
}

LatticeData build_diamond_conventional(std::size_t atoms, double lattice_a, int z) {
  if (atoms % 8 != 0) {
    throw std::invalid_argument(
        "diamond conventional cell requires atom count divisible by 8");
  }
  const auto [nx, ny, nz] = factor_triplet(atoms / 8);
  const std::array<std::array<double, 3>, 8> basis{{
      {0.0, 0.0, 0.0},
      {0.5, 0.5, 0.0},
      {0.5, 0.0, 0.5},
      {0.0, 0.5, 0.5},
      {0.25, 0.25, 0.25},
      {0.75, 0.75, 0.25},
      {0.75, 0.25, 0.75},
      {0.25, 0.75, 0.75},
  }};
  LatticeData out;
  out.positions.reserve(atoms * 3);
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &site : basis) {
          out.positions.push_back((static_cast<double>(ix) + site[0]) * lattice_a);
          out.positions.push_back((static_cast<double>(iy) + site[1]) * lattice_a);
          out.positions.push_back((static_cast<double>(iz) + site[2]) * lattice_a);
        }
      }
    }
  }
  out.atomic_numbers.assign(atoms, z);
  out.box = {
      static_cast<double>(nx) * lattice_a, 0.0, 0.0, 0.0,
      static_cast<double>(ny) * lattice_a, 0.0, 0.0, 0.0,
      static_cast<double>(nz) * lattice_a};
  return out;
}

void apply_secondary_species(std::vector<int> &z, int primary, int secondary,
                             double secondary_fraction) {
  std::fill(z.begin(), z.end(), primary);
  if (!(secondary_fraction > 0.0) || secondary == primary) {
    return;
  }
  const std::size_t stride = static_cast<std::size_t>(
      std::max(1.0, std::round(1.0 / secondary_fraction)));
  for (std::size_t i = 0; i < z.size(); i += stride) {
    z[i] = secondary;
  }
}

struct BenchSamples {
  std::vector<double> build_ns_per_eval;
  std::vector<double> force_ns_per_eval;
  std::vector<double> total_ns_per_eval;
  std::vector<double> nlist_pairs_per_eval;
  std::vector<double> nlist_pairs_per_atom_per_eval;
  std::vector<std::size_t> evals_per_sample;
  std::vector<int> omp_max_threads_per_sample;
  std::vector<int> omp_parallel_threads_per_sample;
  std::vector<std::string> affinity_per_sample;
  std::vector<std::string> omp_parallel_affinity_per_sample;
  double energy = 0.0;
  std::uint64_t energy_bits = 0;
  std::uint64_t force_hash = 0;
};

std::uint64_t bits_of(double value) {
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

std::uint64_t hash_forces(const std::vector<double> &forces) {
  std::uint64_t h = 14695981039346656037ULL;
  for (double value : forces) {
    const std::uint64_t bits = bits_of(value);
    for (int shift = 0; shift < 64; shift += 8) {
      h ^= (bits >> shift) & 0xffU;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

std::string affinity_string() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
    return "unavailable";
  }
  std::ostringstream out;
  bool first = true;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &mask)) {
      continue;
    }
    if (!first) {
      out << ",";
    }
    out << cpu;
    first = false;
  }
  return out.str();
}

int parallel_region_threads() {
#if defined(_OPENMP)
  int observed = 1;
#pragma omp parallel
  {
#pragma omp master
    { observed = omp_get_num_threads(); }
  }
  return observed;
#else
  return 1;
#endif
}

std::string parallel_region_affinity() {
#if defined(_OPENMP)
  int observed = 1;
  std::vector<std::string> per_thread;
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const std::string affinity = affinity_string();
#pragma omp single
    {
      observed = omp_get_num_threads();
      per_thread.resize(static_cast<std::size_t>(observed));
    }
#pragma omp barrier
    per_thread[static_cast<std::size_t>(tid)] = affinity;
  }
  std::ostringstream out;
  for (std::size_t i = 0; i < per_thread.size(); ++i) {
    if (i != 0) {
      out << ";";
    }
    out << per_thread[i];
  }
  return out.str();
#else
  return affinity_string();
#endif
}

class PotRunner {
public:
  virtual ~PotRunner() = default;
  virtual void set_mode_for_reuse() = 0;
  virtual void evaluate(const rgpot::ForceInput &in, rgpot::ForceOut *out) = 0;
};

template <typename Pot>
concept HasSetNeighbourMode = requires(Pot pot) {
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
};

template <typename Pot>
class PotRunnerT final : public PotRunner {
public:
  template <typename... BuildArgs>
  explicit PotRunnerT(BuildArgs &&...args)
      : pot_(std::forward<BuildArgs>(args)...) {}

  void set_mode_for_reuse() override {
    if constexpr (HasSetNeighbourMode<Pot>) {
      pot_.setNeighbourMode(rgpot::nlist::Mode::Stored);
    }
  }

  void evaluate(const rgpot::ForceInput &in, rgpot::ForceOut *out) override {
    pot_.forceImpl(in, out);
  }

private:
  Pot pot_;
};

struct PotentialDescriptor {
  std::string key;
  std::string display;
  LatticeKind lattice = LatticeKind::Fcc;
  double lattice_a = 1.0;
  int primary_z = 1;
  int secondary_z = 1;
  double secondary_fraction = 0.0;
  double cutoff = 0.0;
  double skin = 0.0;
  bool has_neighbor_timing = false;
  std::function<std::unique_ptr<PotRunner>()> make_runner;
};

std::vector<PotentialDescriptor> potential_descriptors() {
  std::vector<PotentialDescriptor> out;

  rgpot::LJConfig lj_cfg;
  lj_cfg.cutoff = 2.5;
  lj_cfg.psi = 1.0;
  lj_cfg.u0 = 1.0;
  out.push_back(PotentialDescriptor{
      .key = "lj",
      .display = "LJ",
      .lattice = LatticeKind::Fcc,
      .lattice_a = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0),
      .primary_z = 18,
      .secondary_z = 18,
      .secondary_fraction = 0.0,
      .cutoff = lj_cfg.cutoff,
      .skin = 1.0,
      .has_neighbor_timing = true,
      .make_runner = [lj_cfg] {
        return std::make_unique<PotRunnerT<rgpot::LJPot>>(lj_cfg);
      },
  });

  rgpot::MorseConfig morse_cfg;
  morse_cfg.cutoff = 6.0;
  out.push_back(PotentialDescriptor{
      .key = "morse",
      .display = "Morse",
      .lattice = LatticeKind::Fcc,
      .lattice_a = morse_cfg.re * std::sqrt(2.0),
      .primary_z = 78,
      .secondary_z = 78,
      .secondary_fraction = 0.0,
      .cutoff = morse_cfg.cutoff,
      .skin = rgpot::MorsePot::neighbourSkin(morse_cfg.cutoff),
      .has_neighbor_timing = true,
      .make_runner = [morse_cfg] {
        return std::make_unique<PotRunnerT<rgpot::MorsePot>>(morse_cfg);
      },
  });

  out.push_back(PotentialDescriptor{
      .key = "eam",
      .display = "EAM",
      .lattice = LatticeKind::Fcc,
      .lattice_a = 4.05,
      .primary_z = 13,
      .secondary_z = 13,
      .secondary_fraction = 0.0,
      .cutoff = 7.1,
      .skin = 7.1 * 0.2,
      .has_neighbor_timing = true,
      .make_runner = [] {
        return std::make_unique<PotRunnerT<rgpot::EAMPot>>();
      },
  });

  out.push_back(PotentialDescriptor{
      .key = "emt",
      .display = "EMT",
      .lattice = LatticeKind::Fcc,
      .lattice_a = 3.615,
      .primary_z = 29,
      .secondary_z = 29,
      .secondary_fraction = 0.0,
      .cutoff = 7.1,
      .skin = 7.1 * 0.2,
      .has_neighbor_timing = true,
      .make_runner = [] {
        return std::make_unique<PotRunnerT<rgpot::EMTPot>>();
      },
  });

  out.push_back(PotentialDescriptor{
      .key = "sw",
      .display = "StillingerWeberSi",
      .lattice = LatticeKind::DiamondPrimitive,
      .lattice_a = 5.431,
      .primary_z = 14,
      .secondary_z = 14,
      .secondary_fraction = 0.0,
      .cutoff = rgpot::StillingerWeberSi::kCut * rgpot::StillingerWeberSi::kSigma,
      .skin = (rgpot::StillingerWeberSi::kCut * rgpot::StillingerWeberSi::kSigma) /
              5.0,
      .has_neighbor_timing = true,
      .make_runner = [] {
        return std::make_unique<PotRunnerT<rgpot::StillingerWeberSi>>();
      },
  });

  out.push_back(PotentialDescriptor{
      .key = "tersoff",
      .display = "TersoffSiC",
      .lattice = LatticeKind::DiamondPrimitive,
      .lattice_a = 5.431,
      .primary_z = 14,
      .secondary_z = 14,
      .secondary_fraction = 0.0,
      .cutoff = 3.0,
      .skin = 3.0 / 5.0,
      .has_neighbor_timing = true,
      .make_runner = [] {
        return std::make_unique<PotRunnerT<rgpot::TersoffSiC>>();
      },
  });

  rgpot::ZBLConfig zbl_cfg;
  out.push_back(PotentialDescriptor{
      .key = "zbl",
      .display = "ZBL",
      .lattice = LatticeKind::DiamondPrimitive,
      .lattice_a = 5.431,
      .primary_z = 14,
      .secondary_z = 14,
      .secondary_fraction = 0.0,
      .cutoff = zbl_cfg.cut_global,
      .skin = rgpot::ZBLPot::neighbourSkin(zbl_cfg.cut_global),
      .has_neighbor_timing = true,
      .make_runner = [zbl_cfg] {
        return std::make_unique<PotRunnerT<rgpot::ZBLPot>>(zbl_cfg);
      },
  });

  rgpot::SuttonChenConfig sc_cfg = rgpot::sutton_chen_platinum();
  sc_cfg.periodic = true;
  sc_cfg.cutoff = 3.0;
  const rgpot::SuttonChenScales sc_scales = rgpot::sutton_chen_scales(sc_cfg);
  out.push_back(PotentialDescriptor{
      .key = "sutton-chen",
      .display = "SuttonChen",
      .lattice = LatticeKind::Fcc,
      .lattice_a = sc_scales.nearest_neighbor * std::sqrt(2.0),
      .primary_z = 78,
      .secondary_z = 78,
      .secondary_fraction = 0.0,
      .cutoff = sc_cfg.cutoff,
      .skin = sc_cfg.cutoff / 5.0,
      .has_neighbor_timing = true,
      .make_runner = [sc_cfg] {
        return std::make_unique<PotRunnerT<rgpot::SuttonChenPot>>(sc_cfg);
      },
  });

  return out;
}

const PotentialDescriptor &select_descriptor(const std::string &name) {
  static const std::vector<PotentialDescriptor> descriptors =
      potential_descriptors();
  const auto it = std::find_if(descriptors.begin(), descriptors.end(),
                               [&](const PotentialDescriptor &d) {
                                 return d.key == name;
                               });
  if (it == descriptors.end()) {
    std::string known;
    for (const auto &d : descriptors) {
      if (!known.empty()) {
        known += ",";
      }
      known += d.key;
    }
    throw std::invalid_argument("unknown potential '" + name +
                                "'; expected one of: " + known);
  }
  return *it;
}

LatticeData build_lattice(const PotentialDescriptor &d, std::size_t atoms) {
  LatticeData data;
  if (d.lattice == LatticeKind::Fcc) {
    data = build_fcc(atoms, d.lattice_a, d.primary_z);
  } else if (d.lattice == LatticeKind::Bcc) {
    data = build_bcc(atoms, d.lattice_a, d.primary_z);
  } else {
    if (atoms % 8 == 0) {
      data = build_diamond_conventional(atoms, d.lattice_a, d.primary_z);
    } else {
      data = build_diamond_primitive(atoms, d.lattice_a, d.primary_z);
    }
  }
  apply_secondary_species(data.atomic_numbers, d.primary_z, d.secondary_z,
                          d.secondary_fraction);
  return data;
}

rgpot::nlist::Query make_query(const PotentialDescriptor &d) {
  rgpot::nlist::Query q;
  q.cutoff = d.cutoff;
  q.skin = d.skin;
  q.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  q.cut = rgpot::nlist::Cut::Loose;
  q.periodic = {true, true, true};
  q.fields.shifts = false;
  q.fields.vectors = false;
  q.fields.distances = false;
  return q;
}

void configure_threads(int threads) {
#if defined(_OPENMP)
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
#else
  (void)threads;
#endif
}

void evaluate_once(PotRunner &runner, const rgpot::ForceInput &in,
                   std::vector<double> &forces, double &energy) {
  std::fill(forces.begin(), forces.end(), 0.0);
  rgpot::ForceOut out{};
  out.F = forces.data();
  out.energy = 0.0;
  out.variance = 0.0;
  runner.evaluate(in, &out);
  if (!std::isfinite(out.energy)) {
    throw std::runtime_error("non-finite energy");
  }
  energy = out.energy;
}

BenchSamples run_samples(const PotentialDescriptor &desc, const Args &args) {
  configure_threads(args.threads);
  LatticeData lattice = build_lattice(desc, args.atoms);
  auto runner = desc.make_runner();
  runner->set_mode_for_reuse();

  const std::size_t n_atoms = lattice.atomic_numbers.size();
  rgpot::ForceInput in{n_atoms, lattice.positions.data(), lattice.atomic_numbers.data(),
                       lattice.box.data()};

  std::vector<double> scratch_forces(3 * n_atoms, 0.0);
  std::vector<double> measured_forces(3 * n_atoms, 0.0);
  double scratch_energy = 0.0;
  double measured_energy = 0.0;

  rgpot::nlist::Table table;
  const rgpot::nlist::Query query = make_query(desc);

  BenchSamples out;
  out.build_ns_per_eval.reserve(static_cast<std::size_t>(args.samples));
  out.force_ns_per_eval.reserve(static_cast<std::size_t>(args.samples));
  out.total_ns_per_eval.reserve(static_cast<std::size_t>(args.samples));
  out.nlist_pairs_per_eval.reserve(static_cast<std::size_t>(args.samples));
  out.nlist_pairs_per_atom_per_eval.reserve(static_cast<std::size_t>(args.samples));
  out.evals_per_sample.reserve(static_cast<std::size_t>(args.samples));
  out.omp_max_threads_per_sample.reserve(static_cast<std::size_t>(args.samples));
  out.omp_parallel_threads_per_sample.reserve(static_cast<std::size_t>(args.samples));
  out.affinity_per_sample.reserve(static_cast<std::size_t>(args.samples));
  out.omp_parallel_affinity_per_sample.reserve(static_cast<std::size_t>(args.samples));

  for (int sample = 0; sample < args.samples; ++sample) {
    int omp_max_threads = 1;
#if defined(_OPENMP)
    omp_max_threads = omp_get_max_threads();
#endif
    const int team_threads = parallel_region_threads();
    const std::string team_affinity = parallel_region_affinity();
    out.omp_max_threads_per_sample.push_back(omp_max_threads);
    out.omp_parallel_threads_per_sample.push_back(team_threads);
    out.affinity_per_sample.push_back(affinity_string());
    out.omp_parallel_affinity_per_sample.push_back(team_affinity);

    std::size_t evals = 0;
    double build_ns = 0.0;
    double force_ns = 0.0;
    double pair_count_sum = 0.0;
    const auto sample_start = std::chrono::steady_clock::now();
    while (true) {
      if (desc.has_neighbor_timing) {
        const auto b0 = std::chrono::steady_clock::now();
        rgpot::nlist::build(table, lattice.positions.data(), n_atoms,
                            lattice.box.data(), query);
        const auto b1 = std::chrono::steady_clock::now();
        build_ns +=
            std::chrono::duration<double, std::nano>(b1 - b0).count();
        pair_count_sum += static_cast<double>(table.size());
      }

      evaluate_once(*runner, in, scratch_forces, scratch_energy);
      const auto f0 = std::chrono::steady_clock::now();
      evaluate_once(*runner, in, measured_forces, measured_energy);
      const auto f1 = std::chrono::steady_clock::now();
      force_ns += std::chrono::duration<double, std::nano>(f1 - f0).count();
      ++evals;

      const double elapsed_s = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - sample_start)
                                   .count();
      if (elapsed_s >= args.sample_seconds && evals >= 1) {
        break;
      }
    }
    const double build_avg =
        desc.has_neighbor_timing ? build_ns / static_cast<double>(evals) : 0.0;
    const double force_avg = force_ns / static_cast<double>(evals);
    out.build_ns_per_eval.push_back(build_avg);
    out.force_ns_per_eval.push_back(force_avg);
    out.total_ns_per_eval.push_back(build_avg + force_avg);
    const double pair_avg = desc.has_neighbor_timing ? pair_count_sum / static_cast<double>(evals) : 0.0;
    out.nlist_pairs_per_eval.push_back(pair_avg);
    out.nlist_pairs_per_atom_per_eval.push_back(pair_avg / static_cast<double>(n_atoms));
    out.evals_per_sample.push_back(evals);
  }

  out.energy = measured_energy;
  out.energy_bits = bits_of(out.energy);
  out.force_hash = hash_forces(measured_forces);

  if (args.dump_force.has_value()) {
    std::ofstream out_file(*args.dump_force, std::ios::binary);
    if (!out_file) {
      throw std::runtime_error("cannot open dump file");
    }
    const std::uint64_t n = static_cast<std::uint64_t>(n_atoms);
    out_file.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out_file.write(reinterpret_cast<const char *>(&out.energy), sizeof(double));
    out_file.write(reinterpret_cast<const char *>(measured_forces.data()),
                   static_cast<std::streamsize>(measured_forces.size() *
                                                sizeof(double)));
  }
  return out;
}

void print_double_array(const std::vector<double> &values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << std::setprecision(17) << values[i];
  }
  std::cout << "]";
}

void print_size_array(const std::vector<std::size_t> &values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_int_array(const std::vector<int> &values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "]";
}

void print_string_array(const std::vector<std::string> &values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "\"";
    for (char c : values[i]) {
      if (c == '"' || c == '\\') {
        std::cout << "\\";
      }
      std::cout << c;
    }
    std::cout << "\"";
  }
  std::cout << "]";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--identity") {
#if defined(_WIN32)
      _putenv_s("OMP_NUM_THREADS", "2");
#else
      setenv("OMP_NUM_THREADS", "2", 1);
#endif
      std::size_t activation = 0;
      for (std::size_t n = 1; n <= 1048576; ++n) {
        if (rgpot::teamSize(n) > 1) {
          activation = n;
          break;
        }
      }
      std::cout << "{\"configured_team_min_atoms\":"
                << rgpot::kTeamMinAtoms
                << ",\"observed_team_activation_atoms\":" << activation
                << "}\n";
      return 0;
    }
    const Args args = parse_args(argc, argv);
    const PotentialDescriptor &desc = select_descriptor(args.potential);
    const BenchSamples result = run_samples(desc, args);

    std::cout << "{";
    std::cout << "\"potential\":\"" << desc.key << "\",";
    std::cout << "\"display\":\"" << desc.display << "\",";
    std::cout << "\"atoms\":" << args.atoms << ",";
    std::cout << "\"threads\":" << args.threads << ",";
    std::cout << "\"samples\":" << args.samples << ",";
    std::cout << "\"sample_seconds\":" << std::setprecision(17)
              << args.sample_seconds << ",";
    std::cout << "\"build_ns_per_eval\":";
    print_double_array(result.build_ns_per_eval);
    std::cout << ",";
    std::cout << "\"force_ns_per_eval\":";
    print_double_array(result.force_ns_per_eval);
    std::cout << ",";
    std::cout << "\"total_ns_per_eval\":";
    print_double_array(result.total_ns_per_eval);
    std::cout << ",";
    std::cout << "\"nlist_pairs_per_eval\":";
    print_double_array(result.nlist_pairs_per_eval);
    std::cout << ",";
    std::cout << "\"nlist_pairs_per_atom_per_eval\":";
    print_double_array(result.nlist_pairs_per_atom_per_eval);
    std::cout << ",";
    std::cout << "\"evals_per_sample\":";
    print_size_array(result.evals_per_sample);
    std::cout << ",";
    std::cout << "\"omp_max_threads_per_sample\":";
    print_int_array(result.omp_max_threads_per_sample);
    std::cout << ",";
    std::cout << "\"omp_parallel_threads_per_sample\":";
    print_int_array(result.omp_parallel_threads_per_sample);
    std::cout << ",";
    std::cout << "\"affinity_per_sample\":";
    print_string_array(result.affinity_per_sample);
    std::cout << ",";
    std::cout << "\"omp_parallel_affinity_per_sample\":";
    print_string_array(result.omp_parallel_affinity_per_sample);
    std::cout << ",";
    std::cout << "\"energy\":" << std::setprecision(17) << result.energy << ",";
    std::cout << "\"energy_bits\":\"0x" << std::hex << result.energy_bits
              << std::dec << "\",";
    std::cout << "\"force_hash\":\"0x" << std::hex << result.force_hash
              << std::dec << "\"";
    std::cout << "}\n";
    return 0;
  } catch (const Exception &err) {
    const char *raw = err.GetMessage();
    const std::string message = raw == nullptr ? std::string() : std::string(raw);
    if (message.empty()) {
      std::cerr << "error: unknown EMT exception\n";
    } else {
      std::cerr << "error: " << message << "\n";
    }
    return 1;
  } catch (const std::exception &err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "error: unknown exception\n";
    return 1;
  }
}
