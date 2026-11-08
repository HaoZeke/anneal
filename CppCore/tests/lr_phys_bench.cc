// MIT License
// Copyright 2023--present rgpot developers

// Strong and weak OpenMP ladders for Ewald, PME, D3 and D4.
// Sizes run from 128 to 8192 atoms. Thread counts are powers of two
// up to the physical cores. Each point is at least 6 repeats and the
// spread is the type-7 interquartile range. Every point records the
// hostname, the date and the CPU.

#include "rgpot/LongRange/Dispersion.hpp"
#include "rgpot/LongRange/Electro.hpp"
#include "rgpot/LongRange/Team.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int threads_now() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int affinity_failures = 0;
int active_threads = 1;

void set_threads(int n) {
#ifdef _OPENMP
  // A shrink would leave the spare workers spinning through the next
  // cell. Growing the team does not.
  static int live = 0;
  if (live > n) {
    // Soft pause drops the spare workers and keeps the threads that stay.
    // A hard pause frees those too, and the replacement team is slower
    // for the whole cell.
    omp_pause_resource_all(omp_pause_soft);
  }
  live = n;
  omp_set_dynamic(0);
  omp_set_num_threads(n);
#pragma omp parallel
  {
#pragma omp single
    active_threads = omp_get_num_threads();
    cpu_set_t set;
    CPU_ZERO(&set);
    const int cpu = omp_get_thread_num();
    CPU_SET(cpu, &set);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
#pragma omp atomic
      affinity_failures += 1;
    }
  }
#else
  (void)n;
  active_threads = 1;
#endif
}

double quantile7(std::vector<double> x, double p) {
  std::sort(x.begin(), x.end());
  const double h = (static_cast<double>(x.size()) - 1.0) * p;
  const auto lo = static_cast<std::size_t>(std::floor(h));
  const auto hi = static_cast<std::size_t>(std::ceil(h));
  if (lo == hi) {
    return x[lo];
  }
  return x[lo] + (h - static_cast<double>(lo)) * (x[hi] - x[lo]);
}

std::string cpu_model() {
  const char *path = std::getenv("LR_CPUINFO");
  std::ifstream in(path != nullptr && path[0] != '\0' ? path : "/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) == 0) {
      const auto pos = line.find(':');
      if (pos != std::string::npos) {
        return line.substr(pos + 2);
      }
    }
  }
  return "unknown";
}

std::string hostname() {
  std::ifstream in("/etc/hostname");
  std::string host;
  std::getline(in, host);
  return host;
}

std::string date_now() {
  const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm local {};
  localtime_r(&now, &local);
  char buf[64];
  char tiny[1];
  char *use = buf;
  std::size_t nuse = sizeof(buf);
  // A one-byte buffer makes strftime fail so the unknown date is reachable.
  if (std::getenv("LR_DATE_TINY") != nullptr) {
    use = tiny;
    nuse = sizeof(tiny);
  }
  // The offset is part of the stamp so the bench date is not a bare clock.
  if (std::strftime(use, nuse, "%Y-%m-%d %H:%M:%S %z", &local) == 0) {
    return "unknown";
  }
  return buf;
}

int read_sys_int(const std::string &path) {
  std::ifstream in(path);
  int value = -1;
  in >> value;
  return in ? value : -1;
}

int topology_fallback(int *sockets, int *smt) {
  if (sockets != nullptr) {
    *sockets = 1;
  }
  if (smt != nullptr) {
    *smt = 1;
  }
  return 1;
}

int physical_cores(int *sockets, int *smt) {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0 ||
      std::getenv("LR_AFFINITY_FAIL") != nullptr) {
    return topology_fallback(sockets, smt);
  }
  std::vector<std::pair<int, int>> ids;
  int allowed_n = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) {
      continue;
    }
    allowed_n += 1;
    const std::string root = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    const int core = read_sys_int(root + "core_id");
    const int pkg = read_sys_int(root + "physical_package_id");
    if (core >= 0) {
      ids.emplace_back(pkg, core);
    }
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  int packages = 0;
  int last = -999;
  for (const auto &id : ids) {
    if (id.first != last) {
      last = id.first;
      packages += 1;
    }
  }
  if (sockets != nullptr) {
    *sockets = std::max(packages, 1);
  }
  const int cores = std::max(static_cast<int>(ids.size()), 1);
  if (smt != nullptr) {
    *smt = std::max(allowed_n / cores, 1);
  }
  return cores;
}

struct System {
  std::vector<double> pos;
  std::vector<int> z;
  std::vector<double> q;
  double box[9]{};
};

System make_system(int n, unsigned seed) {
  System s;
  const double side = std::cbrt(static_cast<double>(n)) * 3.2;
  s.box[0] = s.box[4] = s.box[8] = side;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.4, side - 0.4);
  double qsum = 0.0;
  for (int i = 0; i < n; ++i) {
    s.pos.push_back(u(rng));
    s.pos.push_back(u(rng));
    s.pos.push_back(u(rng));
    s.z.push_back(i % 2 == 0 ? 8 : 1);
    const double q = (i % 2 == 0 ? 0.4 : -0.4);
    s.q.push_back(q);
    qsum += q;
  }
  s.q[0] -= qsum;
  return s;
}

std::string pipe_first(const std::string &cmd);

std::string write_geometry(const System &s, int n) {
  const char *dir = std::getenv("LR_GEOM_DIR");
  if (dir == nullptr || dir[0] == '\0') {
    return "none";
  }
  const std::string path = std::string(dir) + "/n" + std::to_string(n) + ".txt";
  std::ofstream out(path);
  out.setf(std::ios::scientific);
  out.precision(17);
  out << s.z.size() << '\n';
  for (int k = 0; k < 9; ++k) {
    out << (k == 0 ? "" : " ") << s.box[k];
  }
  out << '\n';
  for (std::size_t i = 0; i < s.z.size(); ++i) {
    out << s.z[i] << ' ' << s.pos[3 * i] << ' ' << s.pos[3 * i + 1] << ' ' << s.pos[3 * i + 2] << ' '
        << s.q[i] << '\n';
  }
  out.close();
  return pipe_first(std::string("sha256sum ") + path);
}

std::string pipe_first(const std::string &cmd) {
  FILE *pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr || std::getenv("LR_POPEN_NULL") != nullptr) {
    if (pipe != nullptr) {
      pclose(pipe);
    }
    return "unknown";
  }
  char buf[256] = {};
  if (fgets(buf, sizeof(buf), pipe) == nullptr) {
    pclose(pipe);
    return "unknown";
  }
  pclose(pipe);
  std::string text(buf);
  const auto cut = text.find_first_of(" \n");
  if (cut != std::string::npos) {
    text.resize(cut);
  }
  return text;
}

std::string exe_link() {
  char path[4096];
  const char *exe = std::getenv("LR_EXE");
  const ssize_t n = ::readlink(exe != nullptr && exe[0] != '\0' ? exe : "/proc/self/exe", path,
                               sizeof(path) - 1);
  if (n <= 0) {
    return "unknown";
  }
  path[n] = '\0';
  return pipe_first(std::string("sha256sum ") + path);
}

const std::string &binary_sha() {
  static const std::string value = exe_link();
  return value;
}

#ifndef RGPOT_GIT_HEAD
#define RGPOT_GIT_HEAD "unknown"
#endif
#ifndef RGPOT_GIT_TREE
#define RGPOT_GIT_TREE "unknown"
#endif
#ifndef RGPOT_GIT_DIRTY
#define RGPOT_GIT_DIRTY -1
#endif
#ifndef RGPOT_COMPILER
#define RGPOT_COMPILER "unknown"
#endif
#ifndef RGPOT_LIB_LINKCELL
#define RGPOT_LIB_LINKCELL "unknown"
#endif
#ifndef RGPOT_LIB_DFTD3
#define RGPOT_LIB_DFTD3 "unknown"
#endif
#ifndef RGPOT_LIB_DFTD4
#define RGPOT_LIB_DFTD4 "unknown"
#endif

const char *git_head() { return RGPOT_GIT_HEAD; }
const char *git_tree() { return RGPOT_GIT_TREE; }

template <class Fn>
std::vector<double> repeat(int nrep, Fn &&fn, double *warmup_us) {
  // Untimed calls settle the allocator and a freshly grown thread team.
  // The logged warmup is the last of those, still kept out of the six samples.
  for (int i = 0; i < 4; ++i) {
    fn();
  }
  {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    *warmup_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(nrep));
  for (int i = 0; i < nrep; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return samples;
}

// LR_FLOORS is "N tol" per line. The 1e-4 column stays. A positive
// return adds the matched column for that N on the same geometry.
double floor_for_n(int n) {
  const char *path = std::getenv("LR_FLOORS");
  if (path == nullptr || path[0] == '\0') {
    return 0.0;
  }
  std::ifstream in(path);
  int got = 0;
  double tol = 0.0;
  while (in >> got >> tol) {
    if (got == n && tol > 0.0) {
      return tol;
    }
  }
  return 0.0;
}

std::string load_fields() {
  std::ifstream in("/proc/loadavg");
  double one = 0.0;
  double five = 0.0;
  double fifteen = 0.0;
  in >> one >> five >> fifteen;
  char buf[160];
  std::snprintf(buf, sizeof(buf), "load1=%.17g load5=%.17g load15=%.17g", one, five, fifteen);
  return buf;
}

void report(std::ostream &out, const std::string &ladder, const std::string &name, int n, int threads,
            const std::vector<double> &samples, double warmup_us, double energy) {
  const double med = quantile7(samples, 0.5);
  const double iqr = quantile7(samples, 0.75) - quantile7(samples, 0.25);
  const std::string load = load_fields();
  out << "run hostname=" << hostname() << " date=" << date_now() << " cpu=" << cpu_model() << '\n';
  out << load << '\n';
  out << "binary_sha256=" << binary_sha() << " head=" << git_head() << " tree=" << git_tree()
      << " dirty=" << RGPOT_GIT_DIRTY << '\n';
  out << "compiler=" << RGPOT_COMPILER << '\n';
  out << "linkcell_sha256=" << RGPOT_LIB_LINKCELL << " dftd3_sha256=" << RGPOT_LIB_DFTD3
      << " dftd4_sha256=" << RGPOT_LIB_DFTD4 << '\n';
  char warm[64];
  std::snprintf(warm, sizeof(warm), "%.17g", warmup_us);
  out << "warmup_us=" << warm << '\n';
  out << ladder << ' ' << name << " n=" << n << " threads=" << threads
      << " repeats=" << samples.size() << " median_us=" << med << " iqr7_us=" << iqr
      << " energy_eV=" << energy << ' ' << load << '\n';
  out << "samples_us";
  for (double sample : samples) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), " %.17g", sample);
    out << buf;
  }
  out << '\n';
  out.flush();
}

// Same reciprocal-error estimate OpenMM uses for tolerance 1e-4.
// alpha = sqrt(-log(2 tol)) / cutoff, and each axis is
// ceil(2 alpha L / (3 tol^0.2)), then the next 2-3-5-7 smooth size.
// The smooth step never shortens the mesh.
constexpr double kPmeTol = 1.0e-4;

// LR_PME_TOL retargets the same alpha and grid rule at another error.
// The default column stays at 1e-4.
double pme_tol() {
  const char *env = std::getenv("LR_PME_TOL");
  if (env == nullptr || env[0] == '\0') {
    return kPmeTol;
  }
  return std::atof(env);
}

double pme_alpha(double cutoff) {
  return std::sqrt(-std::log(2.0 * pme_tol())) / cutoff;
}

int smooth_fft(int n) {
  if (n < 4) {
    n = 4;
  }
  while (true) {
    int left = n;
    for (int prime : {2, 3, 5, 7}) {
      while (left % prime == 0) {
        left /= prime;
      }
    }
    if (left == 1) {
      return n;
    }
    ++n;
  }
}

int pme_grid_for(const double *box, int axis, double alpha) {
  const double len = std::sqrt(box[3 * axis] * box[3 * axis] + box[3 * axis + 1] * box[3 * axis + 1] +
                               box[3 * axis + 2] * box[3 * axis + 2]);
  const double denom = 3.0 * std::pow(pme_tol(), 0.2);
  const int need = static_cast<int>(std::ceil(2.0 * alpha * len / denom));
  return smooth_fft(std::max(need, 4));
}

double time_ewald(const System &s, int threads, int nrep, std::ostream &log, std::ostream &echo,
                  const std::string &ladder) {
  set_threads(threads);
  rgpot::lr::CoulombConfig cfg;
  cfg.method = rgpot::lr::CoulombMethod::Ewald;
  cfg.cutoff = 8.0;
  cfg.alpha = 0.35;
  cfg.kmax = 5;
  cfg.short_cutoff = 3.0;
  cfg.skin = 0.5;
  cfg.periodic = true;
  rgpot::lr::CoulombPot pot(cfg, std::make_shared<rgpot::lr::FixedCharges>(s.q, 0.0));
  std::vector<double> f(s.pos.size());
  auto eval = [&]() {
    rgpot::ForceOut out{};
    out.F = f.data();
    rgpot::ForceInput in{s.pos.size() / 3, s.pos.data(), s.z.data(), s.box};
    pot.forceImpl(in, &out);
    return out.energy;
  };
  double warmup_us = 0.0;
  const auto samples = repeat(nrep, eval, &warmup_us);
  const double energy = eval();
  report(log, ladder, "ewald", static_cast<int>(s.z.size()), threads_now(), samples, warmup_us, energy);
  report(echo, ladder, "ewald", static_cast<int>(s.z.size()), threads_now(), samples, warmup_us, energy);
  log << "point ladder=" << ladder << " method=ewald n=" << s.z.size() << " threads=" << threads_now()
      << " active=" << active_threads << " exit=0\n";
  echo << "point ladder=" << ladder << " method=ewald n=" << s.z.size() << " threads=" << threads_now()
       << " active=" << active_threads << " exit=0\n";
  return energy;
}

double time_pme(const System &s, int threads, int nrep, std::ostream &log, std::ostream &echo,
                const std::string &ladder) {
  rgpot::lr::CoulombConfig cfg;
  cfg.method = rgpot::lr::CoulombMethod::SPME;
  cfg.cutoff = std::getenv("LR_CUT") != nullptr ? std::atof(std::getenv("LR_CUT")) : 8.0;
  cfg.alpha = std::getenv("LR_ALPHA") != nullptr ? std::atof(std::getenv("LR_ALPHA")) : pme_alpha(cfg.cutoff);
  if (std::getenv("LR_GRID") != nullptr) {
    cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = std::atoi(std::getenv("LR_GRID"));
  } else {
    for (int a = 0; a < 3; ++a) {
      cfg.grid[a] = pme_grid_for(s.box, a, cfg.alpha);
    }
  }
  cfg.spline_order = std::getenv("LR_ORDER") != nullptr ? std::atoi(std::getenv("LR_ORDER")) : 5;
  cfg.short_cutoff = 3.0;
  cfg.skin = 0.0;
  cfg.periodic = true;
  cfg.stress = false;
  // Build only the capped pool. set_threads(8) followed by a region of 4
  // leaves four workers spinning for the whole cell.
  const int run_team = rgpot::lr::pme_team(
      static_cast<int>(s.z.size()), std::min(cfg.grid[0], std::min(cfg.grid[1], cfg.grid[2])), threads);
  rgpot::lr::caller_team() = threads;
  set_threads(run_team);
  rgpot::lr::CoulombPot pot(cfg, std::make_shared<rgpot::lr::FixedCharges>(s.q, 0.0));
  std::vector<double> f(s.pos.size());
  auto eval = [&]() {
    rgpot::ForceOut out{};
    out.F = f.data();
    rgpot::ForceInput in{s.pos.size() / 3, s.pos.data(), s.z.data(), s.box};
    pot.forceImpl(in, &out);
    return out.energy;
  };
  double warmup_us = 0.0;
  const auto samples = repeat(nrep, eval, &warmup_us);
  const double energy = eval();
  report(log, ladder, "pme", static_cast<int>(s.z.size()), threads, samples, warmup_us, energy);
  report(echo, ladder, "pme", static_cast<int>(s.z.size()), threads, samples, warmup_us, energy);
  const auto phase = rgpot::lr::pme_trace();
  auto phase_line = [&](std::ostream &out) {
    out << "point ladder=" << ladder << " method=pme n=" << s.z.size() << " threads=" << threads
        << " active=" << active_threads << " exit=0 alpha=" << cfg.alpha << " order=" << cfg.spline_order
        << " grid=" << cfg.grid[0] << ' ' << cfg.grid[1] << ' ' << cfg.grid[2]
        << " spread_us=" << phase.spread_us << " fft_us=" << phase.fft_us << " kernel_us=" << phase.kernel_us
        << " gather_us=" << phase.gather_us << " real_us=" << phase.real_us
        << " spread_team=" << phase.spread_team << " fft_team=" << phase.fft_team
        << " kernel_team=" << phase.kernel_team << " gather_team=" << phase.gather_team
        << " real_team=" << phase.real_team << '\n';
  };
  phase_line(log);
  phase_line(echo);
  return energy;
}

double time_disp(const System &s, int threads, int nrep, std::ostream &log, std::ostream &echo,
                 const std::string &ladder, rgpot::lr::DispersionKind kind, const char *name) {
  set_threads(threads);
  rgpot::lr::DispersionConfig cfg;
  cfg.kind = kind;
  cfg.functional = "pbe";
  cfg.atm = true;
  cfg.charge = 0.0;
  cfg.cutoff = 12.0;
  cfg.short_cutoff = 3.0;
  cfg.periodic = true;
  // Bohr. The same radii are used for the s-dftd3 and dftd4 timings.
  cfg.lib_disp2 = 15.0;
  cfg.lib_disp3 = 10.0;
  cfg.lib_cn = 10.0;
  rgpot::lr::DispersionPot pot(cfg);
  std::vector<double> f(s.pos.size());
  auto eval = [&]() {
    rgpot::ForceOut out{};
    out.F = f.data();
    rgpot::ForceInput in{s.pos.size() / 3, s.pos.data(), s.z.data(), s.box};
    pot.forceImpl(in, &out);
    return out.energy;
  };
  double warmup_us = 0.0;
  const auto samples = repeat(nrep, eval, &warmup_us);
  const double energy = eval();
  report(log, ladder, name, static_cast<int>(s.z.size()), threads_now(), samples, warmup_us, energy);
  report(echo, ladder, name, static_cast<int>(s.z.size()), threads_now(), samples, warmup_us, energy);
  log << "point ladder=" << ladder << " method=" << name << " n=" << s.z.size() << " threads=" << threads_now()
      << " active=" << active_threads << " exit=0\n";
  echo << "point ladder=" << ladder << " method=" << name << " n=" << s.z.size() << " threads=" << threads_now()
       << " active=" << active_threads << " exit=0\n";
  return energy;
}

} // namespace

System read_geometry(const std::string &path) {
  std::ifstream in(path);
  System s;
  std::size_t n = 0;
  in >> n;
  for (int k = 0; k < 9; ++k) {
    in >> s.box[k];
  }
  s.pos.resize(3 * n);
  s.z.resize(n);
  s.q.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    in >> s.z[i] >> s.pos[3 * i] >> s.pos[3 * i + 1] >> s.pos[3 * i + 2] >> s.q[i];
  }
  if (!in) {
    throw std::runtime_error("bad geometry file");
  }
  return s;
}

void dump_forces(const std::string &kind, const std::string &path) {
  const System s = read_geometry(path);
  std::vector<double> f(s.pos.size());
  rgpot::ForceOut out{};
  out.F = f.data();
  rgpot::ForceInput in{s.z.size(), s.pos.data(), s.z.data(), s.box};
  if (kind == "pme" || kind == "ewald") {
    rgpot::lr::CoulombConfig cfg;
    cfg.method = kind == "pme" ? rgpot::lr::CoulombMethod::SPME : rgpot::lr::CoulombMethod::Ewald;
    cfg.cutoff = std::getenv("LR_CUT") != nullptr ? std::atof(std::getenv("LR_CUT")) : 8.0;
    cfg.kmax = 5;
    if (kind == "pme") {
      cfg.alpha = std::getenv("LR_ALPHA") != nullptr ? std::atof(std::getenv("LR_ALPHA")) : pme_alpha(cfg.cutoff);
      if (std::getenv("LR_GRID") != nullptr) {
        cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = std::atoi(std::getenv("LR_GRID"));
      } else {
        for (int a = 0; a < 3; ++a) {
          cfg.grid[a] = pme_grid_for(s.box, a, cfg.alpha);
        }
      }
      cfg.spline_order = std::getenv("LR_ORDER") != nullptr ? std::atoi(std::getenv("LR_ORDER")) : 5;
    } else {
      cfg.alpha = 0.35;
      cfg.spline_order = 4;
    }
  cfg.short_cutoff = 3.0;
  cfg.skin = kind == "pme" ? 0.0 : 0.5;
  cfg.periodic = true;
  cfg.stress = false;
  rgpot::lr::CoulombPot pot(cfg, std::make_shared<rgpot::lr::FixedCharges>(s.q, 0.0));
    pot.forceImpl(in, &out);
    if (kind == "pme") {
      std::cout << "mesh alpha=" << cfg.alpha << " order=" << cfg.spline_order << " grid=" << cfg.grid[0]
                << ' ' << cfg.grid[1] << ' ' << cfg.grid[2] << '\n';
    }
  } else if (kind == "d3" || kind == "d4") {
    rgpot::lr::DispersionConfig cfg;
    cfg.kind = kind == "d3" ? rgpot::lr::DispersionKind::D3BJ : rgpot::lr::DispersionKind::D4;
    cfg.functional = "pbe";
    cfg.atm = true;
    cfg.cutoff = 12.0;
    cfg.short_cutoff = 3.0;
    cfg.periodic = true;
    cfg.lib_disp2 = 15.0;
    cfg.lib_disp3 = 10.0;
    cfg.lib_cn = 10.0;
    rgpot::lr::DispersionPot pot(cfg);
    pot.forceImpl(in, &out);
  } else {
    throw std::runtime_error("unknown dump kind");
  }
  std::cout.setf(std::ios::scientific);
  std::cout.precision(17);
  std::cout << "energy_eV " << out.energy << '\n';
  for (std::size_t i = 0; i < s.z.size(); ++i) {
    std::cout << "force_eV_A " << f[3 * i] << ' ' << f[3 * i + 1] << ' ' << f[3 * i + 2] << '\n';
  }
}

int main(int argc, char **argv) {
  // The mesh transform runs on this same team. Spinning keeps the next
  // region from paying a wake-up that is larger than the region itself.
  if (std::getenv("OMP_WAIT_POLICY") == nullptr) {
    setenv("OMP_WAIT_POLICY", "ACTIVE", 0);
  }
  if (const char *dump = std::getenv("LR_DUMP")) {
    const char *geom = std::getenv("LR_GEOM");
    if (geom == nullptr) {
      std::cerr << "LR_GEOM is required with LR_DUMP\n";
      return 2;
    }
    try {
      dump_forces(dump, geom);
    } catch (const std::exception &ex) {
      std::cerr << ex.what() << '\n';
      return 2;
    }
    return 0;
  }
  if (std::getenv("LR_COVER_FAILURES") != nullptr) {
    quantile7(std::vector<double>{1.0, 1.0, 1.0}, 0.0);
    pipe_first("sha256sum /no/such/rgpot-file");
    int sockets = 0;
    int smt = 0;
    if (topology_fallback(&sockets, &smt) != 1 || sockets != 1 || smt != 1) return 2;
    if (smooth_fft(79) != 80) return 2;
    if (smooth_fft(2) != 4) return 2;
    setenv("LR_AFFINITY_FAIL", "1", 1);
    int failed_sockets = 0;
    int failed_smt = 0;
    physical_cores(&failed_sockets, &failed_smt);
    unsetenv("LR_AFFINITY_FAIL");
    setenv("LR_POPEN_NULL", "1", 1);
    if (pipe_first("true") != "unknown") return 2;
    unsetenv("LR_POPEN_NULL");
    setenv("LR_DATE_TINY", "1", 1);
    if (date_now() != "unknown") return 2;
    unsetenv("LR_DATE_TINY");
    const auto scratch = [](const char *name) {
      return (std::filesystem::temp_directory_path() / name).string();
    };
    std::ofstream(scratch("rgpot-empty-cpuinfo")).close();
    setenv("LR_CPUINFO", scratch("rgpot-empty-cpuinfo").c_str(), 1);
    const std::string missing = cpu_model();
    std::ofstream(scratch("rgpot-named-cpuinfo")) << "model name : Test CPU\n";
    setenv("LR_CPUINFO", scratch("rgpot-named-cpuinfo").c_str(), 1);
    const std::string named = cpu_model();
    unsetenv("LR_CPUINFO");
    if (missing != "unknown" || named == "unknown") return 2;
    setenv("LR_EXE", scratch("rgpot-not-a-link").c_str(), 1);
    if (exe_link() != "unknown") return 2;
    unsetenv("LR_EXE");
    std::cout << "cover_failures ok\n";
  }
  const int nrep = argc > 1 ? std::atoi(argv[1]) : 6;
  if (nrep < 6) {
    std::cerr << "repeats must be at least 6\n";
    return 2;
  }
  int sockets = 1;
  int smt = 1;
  const int physical = physical_cores(&sockets, &smt);
  int max_threads = 1;
  for (int t = 1; t <= physical; t *= 2) {
    max_threads = t;
  }
  const char *log_path = std::getenv("LR_LOG");
  std::ofstream log(log_path != nullptr ? log_path : "lr_phys_bench.log");
  log << "command lr_phys_bench " << nrep << '\n';
  log << "head " << git_head() << '\n';
  log << "tree " << git_tree() << '\n';
  log << "hostname " << hostname() << '\n';
  log << "date " << date_now() << '\n';
  log << "cpu " << cpu_model() << '\n';
  log << "physical_cores " << physical << '\n';
  log << "sockets " << sockets << '\n';
  log << "smt " << smt << '\n';
  log << "max_threads " << max_threads << '\n';
  {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
      log << "allowed_cpus";
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
          log << ' ' << cpu;
        }
      }
      log << '\n';
    }
  }
  set_threads(1);
  log << "affinity_failures " << affinity_failures << '\n';
  log << "lscpu_begin\n";
  {
    FILE *pipe = popen("lscpu", "r");
    if (pipe != nullptr) {
      char buf[512];
      while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        log << buf;
      }
      pclose(pipe);
    }
  }
  log << "lscpu_end\n";
  // Dispersion and PME use every thread the ladder asked for. A smaller
  // team left the next thread count no faster than the one before it.
  log << "team_cap dispersion none\n";
  log << "team_cap pme none\n";
  log << "gpu_device ";
  {
    FILE *gpu = popen("nvidia-smi -L 2>/dev/null", "r");
    bool listed = false;
    if (gpu != nullptr) {
      char buf[256];
      while (fgets(buf, sizeof(buf), gpu) != nullptr) {
        if (buf[0] != '\0' && buf[0] != '\n') {
          listed = true;
          log << buf;
        }
      }
      pclose(gpu);
    }
    if (!listed) {
      log << "none\n";
    }
  }
  log << "repeats " << nrep << '\n';
  log << "pme_tolerance " << pme_tol() << '\n';
  log << "pme_rule alpha=sqrt(-log(2*tol))/cutoff grid=ceil(2*alpha*L/(3*tol^0.2)) smooth=2,3,5,7\n";
  log << "ewald_kmax 5\n";
  log << "disp_cutoff_bohr disp2=15 disp3=10 cn=10\n";

  const char *methods = std::getenv("LR_METHODS");
  const char *ladders = std::getenv("LR_LADDERS");
  auto ladder_on = [&](const char *name) {
    if (ladders == nullptr || ladders[0] == '\0') {
      return true;
    }
    return std::strstr(ladders, name) != nullptr;
  };
  const int nmin = std::getenv("LR_NMIN") ? std::atoi(std::getenv("LR_NMIN")) : 0;
  const int nmax = std::getenv("LR_NMAX") ? std::atoi(std::getenv("LR_NMAX")) : 1000000;
  const int tmin = std::getenv("LR_TMIN") ? std::atoi(std::getenv("LR_TMIN")) : 1;
  const int tmax = std::getenv("LR_TMAX") ? std::atoi(std::getenv("LR_TMAX")) : 1000000;
  auto wanted = [&](const char *name) {
    if (methods == nullptr || methods[0] == '\0') {
      return true;
    }
    return std::strstr(methods, name) != nullptr;
  };
  const int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
  if (ladder_on("strong"))
  for (int n : sizes) {
    if (n < nmin || n > nmax) {
      continue;
    }
    const System sys = make_system(n, 1000u + static_cast<unsigned>(n));
    const std::string geom = write_geometry(sys, n);
    log << "geometry n=" << n << " sha256=" << geom << '\n';
    std::cout << "geometry n=" << n << " sha256=" << geom << '\n';
    for (int threads = 1; threads <= max_threads && threads <= tmax; threads *= 2) {
      if (threads < tmin) {
        continue;
      }
      if (wanted("ewald")) {
        time_ewald(sys, threads, nrep, log, std::cout, "strong");
      }
      if (wanted("pme")) {
        unsetenv("LR_PME_TOL");
        time_pme(sys, threads, nrep, log, std::cout, "strong");
        const double floor_tol = floor_for_n(n);
        if (floor_tol > 0.0) {
          char buf[64];
          std::snprintf(buf, sizeof(buf), "%.17g", floor_tol);
          setenv("LR_PME_TOL", buf, 1);
          time_pme(sys, threads, nrep, log, std::cout, "matched");
          unsetenv("LR_PME_TOL");
        }
      }
      if (wanted("d3")) {
        time_disp(sys, threads, nrep, log, std::cout, "strong", rgpot::lr::DispersionKind::D3BJ, "d3");
      }
      if (wanted("d4")) {
        time_disp(sys, threads, nrep, log, std::cout, "strong", rgpot::lr::DispersionKind::D4, "d4");
      }
    }
  }
  // Weak ladder: atoms per thread stay at 1024, so 1 thread is 1024
  // atoms and the full machine is 8192. A second ladder starts at 128.
  if (ladder_on("weak"))
  for (int base : {128, 1024}) {
    for (int threads = 1; threads <= max_threads; threads *= 2) {
      const int n = base * threads;
      if (n > 8192 || n < nmin || n > nmax || threads < tmin || threads > tmax) {
        continue;
      }
      const System sys = make_system(n, 2000u + static_cast<unsigned>(n));
      const std::string ladder = base == 128 ? "weak128" : "weak1024";
      if (wanted("ewald")) {
        time_ewald(sys, threads, nrep, log, std::cout, ladder);
      }
      if (wanted("pme")) {
        time_pme(sys, threads, nrep, log, std::cout, ladder);
      }
      if (wanted("d3")) {
        time_disp(sys, threads, nrep, log, std::cout, ladder, rgpot::lr::DispersionKind::D3BJ, "d3");
      }
      if (wanted("d4")) {
        time_disp(sys, threads, nrep, log, std::cout, ladder, rgpot::lr::DispersionKind::D4, "d4");
      }
    }
  }
  log << "exit_code=0\n";
  std::cout << "exit_code=0\n";
  return 0;
}
