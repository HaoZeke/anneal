// Copyright (c) anneal developers
// Bit comparison of two builds of rgpot's Sutton-Chen C entry on the same
// geometries. One CSV row on stdout.
//
//   sc_omp_bits <reference librgpot.so> <librgpot.so> <xyz> <n> <m> <c>
//
// The geometries are the xyz and 63 copies with Gaussian noise of 0.02 a,
// as sc_wall_micro visits them. Each library is loaded with RTLD_DEEPBIND,
// so each binds its own kernel. A build with OpenMP reads OMP_NUM_THREADS.
// The row counts geometries whose energy bits differ and whose force bits
// differ in any component, then the largest absolute energy and force
// differences and the largest energy difference over the energy.
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct ScC { double epsilon, c, a; int n, m; double cutoff; int periodic; };
using Eval = int (*)(const ScC*, int, const double*, const double*, double*, double*);

static Eval load(const char* path) {
  void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  if (lib == nullptr) { std::fprintf(stderr, "%s\n", dlerror()); std::exit(1); }
  auto entry = (Eval)dlsym(lib, "rgpot_sutton_chen_energy_forces");
  if (entry == nullptr) { std::fprintf(stderr, "%s: no rgpot_sutton_chen_energy_forces\n", path); std::exit(1); }
  return entry;
}

static std::vector<double> read_xyz(const char* path) {
  std::ifstream in(path); std::string line; std::vector<double> v;
  while (std::getline(in, line)) { std::istringstream s(line); std::vector<double> t; double a;
    std::string w; while (s >> w) { char* e; a = std::strtod(w.c_str(), &e); if (*e == 0) t.push_back(a); }
    if (t.size() >= 3) { v.push_back(t[t.size()-3]); v.push_back(t[t.size()-2]); v.push_back(t[t.size()-1]); } }
  return v;
}

int main(int argc, char** argv) {
  if (argc != 7) { std::fprintf(stderr, "sc_omp_bits <reference lib> <lib> <xyz> <n> <m> <c>\n"); return 2; }
  Eval ref = load(argv[1]), got = load(argv[2]);
  std::vector<double> x0 = read_xyz(argv[3]);
  int nat = x0.size() / 3;
  ScC c{1.0, std::atof(argv[6]), 1.0, std::atoi(argv[4]), std::atoi(argv[5]), 0.0, 0};
  std::mt19937_64 rng(7); std::normal_distribution<double> g(0, 0.02);
  std::vector<std::vector<double>> xs(64, x0);
  for (size_t k = 1; k < xs.size(); ++k) for (auto& v : xs[k]) v += g(rng);
  std::vector<double> fr(3*nat), fg(3*nat);
  int e_differ = 0, f_differ = 0; double de = 0, df = 0, rel = 0;
  for (const auto& x : xs) {
    double er = 0, eg = 0;
    if (ref(&c, nat, x.data(), nullptr, &er, fr.data()) != 0 || got(&c, nat, x.data(), nullptr, &eg, fg.data()) != 0) {
      std::fprintf(stderr, "evaluation failed\n"); return 1;
    }
    e_differ += std::memcmp(&er, &eg, sizeof er) != 0;
    f_differ += std::memcmp(fr.data(), fg.data(), fr.size() * sizeof(double)) != 0;
    de = std::fmax(de, std::fabs(er - eg));
    rel = std::fmax(rel, std::fabs(er - eg) / std::fabs(er));
    for (int i = 0; i < 3*nat; ++i) df = std::fmax(df, std::fabs(fr[i] - fg[i]));
  }
  std::printf("%d,%zu,%d,%d,%.3g,%.3g,%.3g\n", nat, xs.size(), e_differ, f_differ, de, df, rel);
}
