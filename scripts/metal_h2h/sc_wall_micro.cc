// Copyright (c) anneal developers
// Per-call cost of rgpot's Sutton-Chen entry against the same kernel kept
// across calls and a plain half-pair loop. One CSV row on stdout.
//
//   sc_wall_micro <librgpot.so> <xyz> <n> <m> <c> <reps>
//
// The geometries are the xyz and 63 copies with Gaussian noise of 0.02 a,
// visited in turn, as a quench sees nearby points.
#include <dlfcn.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "rgpot/SuttonChen/SuttonChenPot.hpp"

struct ScC { double epsilon, c, a; int n, m; double cutoff; int periodic; };
using Eval = int (*)(const ScC*, int, const double*, const double*, double*, double*);

static double ipow(double b, int e) { double r = 1; while (e) { if (e & 1) r *= b; b *= b; e >>= 1; } return r; }
static double pow_ar(double ar2, int e) { double p = ipow(ar2, e / 2); if (e % 2) p *= std::sqrt(ar2); return p; }

// Plain half-pair O(N^2): pair powers once, kept for the force pass.
static double reference(const ScC& c, int n, const double* x, double* f,
                        std::vector<double>& rho, std::vector<double>& pp) {
  rho.assign(n, 0.0);
  pp.resize(3 * (size_t)n * (n - 1) / 2);
  double ep = 0; size_t k = 0;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j, k += 3) {
      double dx = x[3*i]-x[3*j], dy = x[3*i+1]-x[3*j+1], dz = x[3*i+2]-x[3*j+2];
      double inv = 1.0 / (dx*dx+dy*dy+dz*dz);
      double ar2 = c.a*c.a*inv, phi = pow_ar(ar2, c.m), rep = pow_ar(ar2, c.n);
      rho[i] += phi; rho[j] += phi; ep += c.epsilon*rep;
      pp[k] = phi; pp[k+1] = rep; pp[k+2] = inv;
    }
  double ee = 0;
  for (int i = 0; i < n; ++i) { double r = std::sqrt(rho[i]); ee -= c.epsilon*c.c*r; rho[i] = c.epsilon*c.c/(2*r); }
  for (int i = 0; i < 3*n; ++i) f[i] = 0;
  k = 0;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j, k += 3) {
      double s = (c.epsilon*c.n*pp[k+1] - c.m*pp[k]*(rho[i]+rho[j]))*pp[k+2];
      for (int d = 0; d < 3; ++d) { double fd = s*(x[3*i+d]-x[3*j+d]); f[3*i+d] += fd; f[3*j+d] -= fd; }
    }
  return ep + ee;
}

static std::vector<double> read_xyz(const char* path) {
  std::ifstream in(path); std::string line; std::vector<double> v;
  while (std::getline(in, line)) { std::istringstream s(line); std::vector<double> t; double a;
    std::string w; while (s >> w) { char* e; a = std::strtod(w.c_str(), &e); if (*e == 0) t.push_back(a); }
    if (t.size() >= 3) { v.push_back(t[t.size()-3]); v.push_back(t[t.size()-2]); v.push_back(t[t.size()-1]); } }
  return v;
}

int main(int argc, char** argv) {
  // bench <lib> <xyz> <n> <m> <c> <reps>
  void* lib = dlopen(argv[1], RTLD_NOW);
  auto entry = (Eval)dlsym(lib, "rgpot_sutton_chen_energy_forces");
  std::vector<double> x0 = read_xyz(argv[2]);
  int nat = x0.size() / 3;
  ScC c{1.0, std::atof(argv[5]), 1.0, std::atoi(argv[3]), std::atoi(argv[4]), 0.0, 0};
  int reps = std::atoi(argv[6]);
  std::mt19937_64 rng(7); std::normal_distribution<double> g(0, 0.02);
  // A set of nearby geometries, as a quench sees.
  std::vector<std::vector<double>> xs(64, x0);
  for (auto& x : xs) for (auto& v : x) v += g(rng);
  std::vector<double> f(3*nat), f2(3*nat), rho, pp; double e = 0, e2 = 0, sink = 0;
  using clk = std::chrono::steady_clock;
  auto time = [&](auto&& body) { auto t = clk::now(); for (int r = 0; r < reps; ++r) body(xs[r % 64]); return std::chrono::duration<double>(clk::now()-t).count()/reps*1e6; };

  double t_entry = time([&](const std::vector<double>& x) { entry(&c, nat, x.data(), nullptr, &e, f.data()); sink += e; });
  double t_wrap = time([&](const std::vector<double>& x) {
    auto fn = (Eval)dlsym(lib, "rgpot_sutton_chen_energy_forces");
    std::vector<double> fv(3*nat, 0.0);
    fn(&c, nat, x.data(), nullptr, &e, fv.data());
    std::vector<double> gv(3*nat); for (int i = 0; i < 3*nat; ++i) gv[i] = -fv[i];
    sink += e + gv[0]; });
  rgpot::SuttonChenConfig sc; sc.epsilon = 1; sc.c = c.c; sc.a = 1; sc.n = c.n; sc.m = c.m; sc.cutoff = 0; sc.periodic = false;
  auto kept = [&](rgpot::nlist::Mode mode) {
    rgpot::SuttonChenPot pot(sc); pot.setNeighbourMode(mode);
    return time([&](const std::vector<double>& x) {
      rgpot::ForceOut out{}; out.F = f2.data();
      const rgpot::ForceInput in{(size_t)nat, x.data(), nullptr, nullptr};
      pot.forceImpl(in, &out); sink += out.energy; });
  };
  double t_kept_auto = kept(rgpot::nlist::Mode::Auto);
  double t_kept_stream = kept(rgpot::nlist::Mode::Stream);
  double t_ref = time([&](const std::vector<double>& x) { e2 = reference(c, nat, x.data(), f2.data(), rho, pp); sink += e2; });

  // Agreement on one geometry.
  entry(&c, nat, xs[3].data(), nullptr, &e, f.data());
  e2 = reference(c, nat, xs[3].data(), f2.data(), rho, pp);
  double df = 0; for (int i = 0; i < 3*nat; ++i) df = std::fmax(df, std::fabs(f[i]-f2[i]));
  std::printf("%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3g,%.3g\n", nat, reps, t_entry, t_wrap, t_kept_auto,
              t_kept_stream, t_ref, std::fabs(e - e2), df);
  if (sink == 0.123) std::fprintf(stderr, "sink\n");
}
