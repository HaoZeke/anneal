// MIT License
// Copyright 2023--present rgpot developers
//
// Per-call cost of one evaluation through IPIPot and IPIDriver, over a UNIX
// socket and over TCP on the loopback, against the same potential called
// directly. Server and driver share the process, one thread each, so the
// difference is the protocol: unit and cell conversion, framing, the
// socket round trip and two thread wake-ups.
//
//   rgpot-ipi-bench [--calls K] [--repeats R] [--sizes 64,256,1024,16384]
//                   [--eam FILE]
//
// Each case runs R times. A run makes three untimed calls, then times K
// calls and keeps their median; the table prints the median of the R
// medians, in microseconds. K defaults to about a second of direct work.
// The geometries are fcc blocks with a fixed small displacement on every
// atom: LJ argon at a = 5.26 Angstrom and, with --eam, EAMTable copper at
// a = 3.615 Angstrom (CppCore/tests/data/eam_table/Cu_u3.eam is Foiles
// copper). The 64-atom blocks are thinner than
// twice the cutoff; the minimum-image result is still a fixed amount of
// work, which is all a timing needs.

#include "pot_spec.hpp"
#include "rgpot/IPIPot/IPIDriver.hpp"
#include "rgpot/IPIPot/IPIPot.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct Geometry {
  std::vector<double> pos;
  std::vector<int> species;
  double rows[9]{};
};

// The fcc block with n atoms that is closest to a cube.
Geometry fcc(int natoms, double a, int z) {
  const int cells = natoms / 4;
  int best[3] = {cells, 1, 1};
  int spread = cells;
  for (int i = 1; i <= cells; ++i) {
    for (int j = i; j <= cells / i; ++j) {
      if (cells % (i * j) != 0) {
        continue;
      }
      const int k = cells / (i * j);
      if (k < j) {
        continue;
      }
      if (k - i < spread) {
        spread = k - i;
        best[0] = i;
        best[1] = j;
        best[2] = k;
      }
    }
  }
  Geometry g;
  for (int c = 0; c < 3; ++c) {
    g.rows[4 * c] = a * best[c];
  }
  const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  int k = 0;
  for (int i = 0; i < best[0]; ++i) {
    for (int j = 0; j < best[1]; ++j) {
      for (int l = 0; l < best[2]; ++l) {
        for (const auto &b : basis) {
          const double cell[3] = {i + b[0], j + b[1], l + b[2]};
          for (int c = 0; c < 3; ++c) {
            g.pos.push_back(a * cell[c] + 0.03 * std::sin(1.7 * k + c));
          }
          ++k;
        }
      }
    }
  }
  g.species.assign(static_cast<size_t>(k), z);
  return g;
}

using Clock = std::chrono::steady_clock;
using Call = std::function<void(const rgpot::ForceInput &, rgpot::ForceOut *)>;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Median microseconds per call over `calls` timed calls.
double timeCalls(const Call &call, const Geometry &g, int calls) {
  std::vector<double> forces(g.pos.size());
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(calls));
  for (int k = -3; k < calls; ++k) {
    std::fill(forces.begin(), forces.end(), 0.0);
    rgpot::ForceInput in{g.species.size(), g.pos.data(), g.species.data(),
                         g.rows};
    rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
    const auto t0 = Clock::now();
    call(in, &out);
    const auto t1 = Clock::now();
    if (k >= 0) {
      samples.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  }
  return median(samples);
}

double timeSocket(const Call &eval, const Geometry &g, int calls,
                  const std::string &address) {
  auto server = std::make_unique<rgpot::IPIPot>(
      rgpot::IPIConfig{.address = address, .timeout_s = 60.0});
  rgpot::IPIDriverConfig cfg;
  cfg.address = server->endpoint();
  cfg.species = g.species;
  rgpot::IPIDriver driver(cfg, eval);
  std::string error;
  std::thread thread([&driver, &error] {
    try {
      driver.run();
    } catch (const std::exception &e) {
      error = e.what();
    }
  });
  double us = 0.0;
  try {
    const rgpot::IPIPot &pot = *server;
    us = timeCalls([&pot](const rgpot::ForceInput &in,
                          rgpot::ForceOut *out) { pot.forceImpl(in, out); },
                   g, calls);
  } catch (...) {
    server.reset();
    thread.join();
    throw;
  }
  server.reset();
  thread.join();
  if (!error.empty()) {
    throw std::runtime_error(error);
  }
  return us;
}

std::vector<int> parseSizes(const std::string &text) {
  std::vector<int> out;
  std::istringstream in(text);
  for (std::string item; std::getline(in, item, ',');) {
    const int n = std::atoi(item.c_str());
    if (n < 4 || n % 4 != 0) {
      throw std::invalid_argument("--sizes takes multiples of 4, not '" + item +
                                  "'");
    }
    out.push_back(n);
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  int calls = 0;
  int repeats = 5;
  std::vector<int> sizes = {64, 256, 1024, 16384};
  std::string eam_file;
  try {
    for (int k = 1; k < argc; ++k) {
      const std::string a = argv[k];
      if (k + 1 >= argc) {
        throw std::invalid_argument(a + " needs a value");
      }
      if (a == "--calls") {
        calls = std::atoi(argv[++k]);
      } else if (a == "--repeats") {
        repeats = std::atoi(argv[++k]);
      } else if (a == "--eam") {
        eam_file = argv[++k];
      } else if (a == "--sizes") {
        sizes = parseSizes(argv[++k]);
      } else {
        throw std::invalid_argument("unknown argument " + a);
      }
    }
    if (repeats < 1 || calls < 0) {
      throw std::invalid_argument("--repeats must be >= 1 and --calls >= 0");
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr,
                 "rgpot-ipi-bench: %s\nusage: rgpot-ipi-bench [--calls K] "
                 "[--repeats R] [--sizes 64,256,1024,16384] [--eam FILE]\n",
                 e.what());
    return 2;
  }

  struct Case {
    const char *label;
    const char *name;
    rgpot::spec::Options options;
    double a;
    int z;
  };
  std::vector<Case> cases = {
      {"LJ Ar",
       "LJ",
       {{"u0", "0.0104"}, {"psi", "3.40"}, {"cutoff", "8.5"}},
       5.26,
       18},
  };
  if (!eam_file.empty()) {
    cases.push_back(
        {"EAMTable Cu", "EAMTable", {{"file", eam_file}}, 3.615, 29});
  }
  const auto socket_dir = std::filesystem::temp_directory_path();
  const std::string unix_address =
      "unix:" +
      (socket_dir / ("rgpot-ipi-bench-" + std::to_string(::getpid()))).string();

  std::printf("| potential | atoms | direct (us) | unix (us) | tcp (us) | "
              "unix - direct (us) | tcp - direct (us) |\n");
  try {
    for (const Case &c : cases) {
      rgpot::spec::Built built = rgpot::spec::build(c.name, c.options, 0.0);
      if (!built.pot) {
        throw std::runtime_error(built.error);
      }
      for (const int n : sizes) {
        const Geometry g = fcc(n, c.a, c.z);
        // Enough calls for about a second of direct work, within [20, 2000].
        double once = 0.0;
        try {
          once = timeCalls(built.eval, g, 3);
        } catch (const std::invalid_argument &e) {
          // A tabulated EAM cutoff can exceed half of any 64-atom cell.
          std::fprintf(stderr, "rgpot-ipi-bench: %s at %d atoms skipped: %s\n",
                       c.label, n, e.what());
          continue;
        }
        int k = calls;
        if (k == 0) {
          k = static_cast<int>(
              std::clamp(1.0e6 / std::max(once, 1.0), 20.0, 2000.0));
        }
        std::vector<double> direct;
        std::vector<double> unix_us;
        std::vector<double> tcp_us;
        for (int r = 0; r < repeats; ++r) {
          direct.push_back(timeCalls(built.eval, g, k));
          unix_us.push_back(timeSocket(built.eval, g, k, unix_address));
          tcp_us.push_back(timeSocket(built.eval, g, k, "tcp:127.0.0.1:0"));
        }
        const double d = median(direct);
        const double u = median(unix_us);
        const double t = median(tcp_us);
        std::printf("| %s | %zu | %.1f | %.1f | %.1f | %.1f | %.1f |\n",
                    c.label, g.species.size(), d, u, t, u - d, t - d);
        std::fflush(stdout);
      }
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rgpot-ipi-bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
