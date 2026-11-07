// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/Pairs.hpp"

#include "rgpot/LongRange/Lattice.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rgpot::lr {

namespace {

struct Cand {
  int i = 0;
  int j = 0;
  int sa = 0;
  int sb = 0;
  int sc = 0;
};

int thread_count() {
#ifdef _OPENMP
  const int n = omp_get_max_threads();
  return n > 0 ? n : 1;
#else
  return 1;
#endif
}

int thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

void vacuum_box(const double *pos, std::size_t n, double reach, double *box9) {
  double lo[3] = {pos[0], pos[1], pos[2]};
  double hi[3] = {lo[0], lo[1], lo[2]};
  for (std::size_t i = 1; i < n; ++i) {
    for (int c = 0; c < 3; ++c) {
      lo[c] = std::min(lo[c], pos[3 * i + c]);
      hi[c] = std::max(hi[c], pos[3 * i + c]);
    }
  }
  // L > 2 * extent and L/2 > reach, so the minimum image of every pair
  // inside the cutoff is the Cartesian pair.
  std::fill(box9, box9 + 9, 0.0);
  for (int c = 0; c < 3; ++c) {
    const double extent = hi[c] - lo[c];
    const double L = 2.0 * std::max(extent, reach) + 1.0;
    box9[4 * c] = L;
  }
}

double corner_span(const double *h, const double *href) {
  double best = 0.0;
  for (int bits = 0; bits < 8; ++bits) {
    const double b[3] = {double(bits & 1), double((bits >> 1) & 1), double((bits >> 2) & 1)};
    double d[3] = {};
    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        d[c] += b[a] * (h[3 * a + c] - href[3 * a + c]);
      }
    }
    best = std::max(best, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
  }
  return best;
}

int wrap_cell(int c, int nc, int *shift) {
  int s = 0;
  while (c < 0) {
    c += nc;
    s -= 1;
  }
  while (c >= nc) {
    c -= nc;
    s += 1;
  }
  *shift = s;
  return c;
}

struct Grid {
  int nc[3] = {1, 1, 1};
  int rad[3] = {1, 1, 1};
  std::vector<int> off;
  std::vector<int> atoms;
};

Grid grid_of(const double *pos, std::size_t n, const CellGeom &cell, double reach) {
  Grid g;
  for (int d = 0; d < 3; ++d) {
    const double col = std::sqrt(cell.inv[0][d] * cell.inv[0][d] + cell.inv[1][d] * cell.inv[1][d] +
                                 cell.inv[2][d] * cell.inv[2][d]);
    const double frac = reach * col;
    if (!(frac > 0.0) || frac >= 1.0) {
      g.nc[d] = 1;
      g.rad[d] = std::max(1, static_cast<int>(std::ceil(frac)));
    } else {
      g.nc[d] = std::max(1, static_cast<int>(std::floor(1.0 / frac)));
      g.rad[d] = 1;
    }
  }
  const int nb = g.nc[0] * g.nc[1] * g.nc[2];
  std::vector<int> count(static_cast<std::size_t>(nb), 0);
  std::vector<int> bin(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    double r[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
    double s[3];
    cell.fractional(r, s);
    int id = 0;
    int stride = 1;
    for (int d = 0; d < 3; ++d) {
      double f = s[d] - std::floor(s[d]);
      int c = static_cast<int>(std::floor(f * static_cast<double>(g.nc[d])));
      if (c < 0) {
        c = 0;
      }
      if (c >= g.nc[d]) {
        c = g.nc[d] - 1;
      }
      id += c * stride;
      stride *= g.nc[d];
    }
    bin[i] = id;
    count[static_cast<std::size_t>(id)] += 1;
  }
  g.off.assign(static_cast<std::size_t>(nb + 1), 0);
  for (int b = 0; b < nb; ++b) {
    g.off[static_cast<std::size_t>(b + 1)] = g.off[static_cast<std::size_t>(b)] + count[static_cast<std::size_t>(b)];
  }
  g.atoms.assign(n, 0);
  std::vector<int> cursor(g.off.begin(), g.off.end() - 1);
  for (std::size_t i = 0; i < n; ++i) {
    g.atoms[static_cast<std::size_t>(cursor[static_cast<std::size_t>(bin[i])]++)] = static_cast<int>(i);
  }
  return g;
}

void gather(const double *pos, std::size_t n, const CellGeom &cell, double reach, int team,
            std::vector<Cand> *out) {
  const Grid g = grid_of(pos, n, cell, reach);
  const double reach2 = reach * reach;
  const int nt = team;
  std::vector<std::vector<Cand>> local(static_cast<std::size_t>(nt));
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto &mine = local[static_cast<std::size_t>(thread_id())];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (std::size_t i = 0; i < n; ++i) {
      double ri[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
      double si[3];
      cell.fractional(ri, si);
      int ci[3];
      for (int d = 0; d < 3; ++d) {
        const double f = si[d] - std::floor(si[d]);
        int c = static_cast<int>(std::floor(f * static_cast<double>(g.nc[d])));
        if (c < 0) {
          c = 0;
        }
        if (c >= g.nc[d]) {
          c = g.nc[d] - 1;
        }
        ci[d] = c;
      }
      for (int da = -g.rad[0]; da <= g.rad[0]; ++da) {
        int sa = 0;
        const int wa = wrap_cell(ci[0] + da, g.nc[0], &sa);
        for (int db = -g.rad[1]; db <= g.rad[1]; ++db) {
          int sb = 0;
          const int wb = wrap_cell(ci[1] + db, g.nc[1], &sb);
          for (int dc = -g.rad[2]; dc <= g.rad[2]; ++dc) {
            int sc = 0;
            const int wc = wrap_cell(ci[2] + dc, g.nc[2], &sc);
            const int id = wa + g.nc[0] * (wb + g.nc[1] * wc);
            const int begin = g.off[static_cast<std::size_t>(id)];
            const int end = g.off[static_cast<std::size_t>(id + 1)];
            for (int t = begin; t < end; ++t) {
              const int j = g.atoms[static_cast<std::size_t>(t)];
              if (j < static_cast<int>(i)) {
                continue;
              }
              if (j == static_cast<int>(i) && sa == 0 && sb == 0 && sc == 0) {
                continue;
              }
              const double dx = pos[3 * j] - ri[0] + sa * cell.H[0][0] + sb * cell.H[1][0] + sc * cell.H[2][0];
              const double dy = pos[3 * j + 1] - ri[1] + sa * cell.H[0][1] + sb * cell.H[1][1] + sc * cell.H[2][1];
              const double dz = pos[3 * j + 2] - ri[2] + sa * cell.H[0][2] + sb * cell.H[1][2] + sc * cell.H[2][2];
              const double r2 = dx * dx + dy * dy + dz * dz;
              if (!(r2 > 0.0) || r2 >= reach2) {
                continue;
              }
              mine.push_back(Cand{static_cast<int>(i), j, sa, sb, sc});
            }
          }
        }
      }
    }
  }
  std::size_t total = 0;
  for (const auto &part : local) {
    total += part.size();
  }
  out->clear();
  out->reserve(total);
  for (const auto &part : local) {
    out->insert(out->end(), part.begin(), part.end());
  }
}

void emit_pairs(const std::vector<Cand> &cand, const double *pos, const CellGeom &cell, double cutoff,
                int team, std::vector<Pair> *out) {
  const double cut2 = cutoff * cutoff;
  const int nt = team;
  std::vector<std::vector<Pair>> local(static_cast<std::size_t>(nt));
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto &mine = local[static_cast<std::size_t>(thread_id())];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (std::size_t t = 0; t < cand.size(); ++t) {
      const Cand &c = cand[t];
      const double dx = pos[3 * c.j] - pos[3 * c.i] + c.sa * cell.H[0][0] + c.sb * cell.H[1][0] +
                        c.sc * cell.H[2][0];
      const double dy = pos[3 * c.j + 1] - pos[3 * c.i + 1] + c.sa * cell.H[0][1] + c.sb * cell.H[1][1] +
                        c.sc * cell.H[2][1];
      const double dz = pos[3 * c.j + 2] - pos[3 * c.i + 2] + c.sa * cell.H[0][2] + c.sb * cell.H[1][2] +
                        c.sc * cell.H[2][2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || r2 >= cut2) {
        continue;
      }
      Pair p;
      p.i = c.i;
      p.j = c.j;
      p.x = dx;
      p.y = dy;
      p.z = dz;
      p.r = std::sqrt(r2);
      mine.push_back(p);
    }
  }
  std::size_t total = 0;
  for (const auto &part : local) {
    total += part.size();
  }
  out->clear();
  out->reserve(total);
  for (const auto &part : local) {
    out->insert(out->end(), part.begin(), part.end());
  }
}

} // namespace

struct DualList::Skin {
  std::vector<double> ref;
  double ref_h[9]{};
  std::vector<Cand> cand;
  bool built = false;
};

DualList::DualList(double short_cut, double long_cut, double skin)
    : short_cut_(short_cut), long_cut_(long_cut), skin_(skin) {
  if (!(short_cut > 0.0) || !(long_cut > 0.0) || !(skin >= 0.0)) {
    throw std::invalid_argument("cutoffs must be positive and skin non-negative");
  }
  if (short_cut > long_cut) {
    throw std::invalid_argument("short cutoff exceeds the long cutoff");
  }
}

DualList::~DualList() = default;

void DualList::fill(Skin &skin, double cutoff, const double *pos, std::size_t n, const double *box9,
                    std::vector<Pair> *out, bool *rebuilt) {
  const CellGeom cell = cell_from_box(box9);
  const double half = skin_ * 0.5;
  const double corner = corner_span(box9, skin.ref_h) * 0.5;
  bool rebuild = !skin.built || skin.ref.size() != 3 * n || corner > half;
  if (!rebuild) {
    const double room = half - corner;
    const double room2 = room * room;
    for (std::size_t i = 0; i < n; ++i) {
      const double dx = pos[3 * i] - skin.ref[3 * i];
      const double dy = pos[3 * i + 1] - skin.ref[3 * i + 1];
      const double dz = pos[3 * i + 2] - skin.ref[3 * i + 2];
      if (dx * dx + dy * dy + dz * dz > room2) {
        rebuild = true;
        break;
      }
    }
  }
  *rebuilt = rebuild;
  const int team = thread_count();
  if (rebuild) {
    // One OpenMP team. omp_set_num_threads(1) keeps this a single thread.
    gather(pos, n, cell, cutoff + skin_, team, &skin.cand);
    skin.ref.assign(pos, pos + 3 * n);
    std::copy(box9, box9 + 9, skin.ref_h);
    skin.built = true;
  }
  emit_pairs(skin.cand, pos, cell, cutoff, team, out);
}

void DualList::update(const double *pos, std::size_t n, const double *box, bool periodic) {
  short_.clear();
  long_.clear();
  short_rebuilt_ = false;
  long_rebuilt_ = false;
  if (n < 2) {
    return;
  }
  if (pos == nullptr) {
    throw std::invalid_argument("positions are null");
  }
  const double *use = box;
  if (!periodic || box == nullptr) {
    double need[9];
    vacuum_box(pos, n, long_cut_ + skin_, need);
    if (!have_vacuum_) {
      std::copy(need, need + 9, vacuum_);
      have_vacuum_ = true;
    } else {
      for (int c = 0; c < 3; ++c) {
        vacuum_[4 * c] = std::max(vacuum_[4 * c], need[4 * c]);
      }
    }
    use = vacuum_;
  }
  if (!short_skin_) {
    short_skin_ = std::make_unique<Skin>();
    long_skin_ = std::make_unique<Skin>();
  }
  fill(*short_skin_, short_cut_, pos, n, use, &short_, &short_rebuilt_);
  fill(*long_skin_, long_cut_, pos, n, use, &long_, &long_rebuilt_);
}

} // namespace rgpot::lr
