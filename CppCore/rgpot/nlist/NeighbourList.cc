// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/rgpot_nlist.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace rgpot::nlist {

// Periodic coordinates are folded into the primary cell before both binning
// and image enumeration. `frac` holds those folded fractional coordinates.
// A pair within `search` has |df_k + S_k| at most `reach[k]`, since the
// distance is at least that times the cell height, so the shifts to try are
// those around -df. `span` bounds |S_k| over every pair. `cell_pos` aliases
// the caller's positions when they are already folded and `wrapped` otherwise.
// `wrap` holds the whole cells W_i taken off each atom, r'_i = r_i - W_i H,
// three per atom, and is empty when every atom was already inside.
struct Frame {
  double normal[3][3]{};
  double reach[3]{};
  bool periodic[3]{};
  int span[3]{};
  std::vector<double> frac;
  std::vector<double> wrapped;
  std::vector<std::int32_t> wrap;
  const double *caller = nullptr;
  bool geometry = true;
  const double *cell_pos = nullptr;
};

namespace {

struct Rec {
  std::int32_t first = 0;
  std::int32_t second = 0;
  std::int32_t shift[3]{};
  double vector[3]{};
  double distance = 0;
  std::uint8_t mask = 0xff;
};

void check_geometry(const double *positions, std::size_t n, const double *box9,
                    const Query &query) {
  if (n > 0 && positions == nullptr) {
    throw Error(Code::Null, "positions is null");
  }
  if (box9 == nullptr) {
    throw Error(Code::Null, "box is null");
  }
  if (!std::isfinite(query.cutoff) || query.cutoff < 0.0) {
    throw Error(Code::Cutoff, "cutoff must be finite and >= 0");
  }
  if (!std::isfinite(query.skin) || query.skin < 0.0) {
    throw Error(Code::Cutoff, "skin must be finite and >= 0");
  }
  if (query.skin > 0.0 && query.cut == Cut::Strict) {
    throw Error(Code::Skin,
                "a positive skin needs a loose cut; a strict cut stores no buffer");
  }
  if (const auto *pairs = std::get_if<Pairs>(&query.shape)) {
    if (pairs->shell != Shell::Half && pairs->shell != Shell::Full) {
      throw Error(Code::Shape, "a pair list shell must be half or full");
    }
  }
  if (query.cut != Cut::Strict && query.cut != Cut::Loose) {
    throw Error(Code::Shape, "cut must be strict or loose");
  }
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(box9[k])) {
      throw Error(Code::Box, "box has a non-finite element");
    }
  }
  const bool any_periodic =
      query.periodic[0] || query.periodic[1] || query.periodic[2];
  if (any_periodic) {
    const double a[3] = {box9[0], box9[1], box9[2]};
    const double b[3] = {box9[3], box9[4], box9[5]};
    const double c[3] = {box9[6], box9[7], box9[8]};
    const double bx = b[1] * c[2] - b[2] * c[1];
    const double by = b[2] * c[0] - b[0] * c[2];
    const double bz = b[0] * c[1] - b[1] * c[0];
    const double vol = std::fabs(a[0] * bx + a[1] * by + a[2] * bz);
    if (!(vol > 0.0)) {
      throw Error(Code::Box, "a periodic box has zero volume");
    }
  }
}

double search_radius(const Query &query) {
  if (query.cut == Cut::Loose) {
    return query.cutoff + query.skin;
  }
  return query.cutoff;
}

bool clustered(const Query &query) { return std::holds_alternative<Cluster4>(query.shape); }

bool full_shell(const Query &query) {
  if (const auto *pairs = std::get_if<Pairs>(&query.shape)) {
    return pairs->shell == Shell::Full;
  }
  return true;
}

bool accept_distance(double r2, const Query &query) {
  if (!(r2 > 0.0)) {
    return false;
  }
  const double cut2 = query.cutoff * query.cutoff;
  if (query.cut == Cut::Strict) {
    return r2 < cut2;
  }
  const double reach = search_radius(query);
  if (query.skin > 0.0) {
    return r2 < reach * reach;
  }
  return r2 <= cut2;
}

// vesin half-list rule for a pair and its periodic self image.
bool keep_half(std::int32_t first, std::int32_t second, int sa, int sb, int sc) {
  if (first > second) {
    return false;
  }
  if (first < second) {
    return true;
  }
  const int sum = sa + sb + sc;
  if (sum < 0) {
    return false;
  }
  if (sum == 0 && (sc < 0 || (sc == 0 && sb < 0))) {
    return false;
  }
  return sa != 0 || sb != 0 || sc != 0;
}

void shift_bounds(const double *box, const std::array<bool, 3> &periodic,
                  double search, int nmax[3]) {
  const double a[3] = {box[0], box[1], box[2]};
  const double b[3] = {box[3], box[4], box[5]};
  const double c[3] = {box[6], box[7], box[8]};
  const double bx[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2],
                        b[0] * c[1] - b[1] * c[0]};
  const double cx[3] = {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2],
                        c[0] * a[1] - c[1] * a[0]};
  const double ax[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]};
  const double vol = std::fabs(a[0] * bx[0] + a[1] * bx[1] + a[2] * bx[2]);
  const double nb = std::sqrt(bx[0] * bx[0] + bx[1] * bx[1] + bx[2] * bx[2]);
  const double nc = std::sqrt(cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]);
  const double na = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
  const double height[3] = {nb > 0.0 ? vol / nb : 0.0, nc > 0.0 ? vol / nc : 0.0,
                            na > 0.0 ? vol / na : 0.0};
  for (int k = 0; k < 3; ++k) {
    if (!periodic[static_cast<std::size_t>(k)] || !(height[k] > 0.0) ||
        !(search > 0.0)) {
      nmax[k] = 0;
    } else {
      nmax[k] = static_cast<int>(std::ceil(search / height[k]));
    }
  }
}

// 2^29 cells keeps every shift, and a shift plus a span, inside an int32.
constexpr double kMaxCells = 536870912.0;

void lattice_normals(const double *box, const std::array<bool, 3> &periodic, double search,
                     double normal[3][3], double reach[3], bool on[3]) {
  const double a[3] = {box[0], box[1], box[2]};
  const double b[3] = {box[3], box[4], box[5]};
  const double c[3] = {box[6], box[7], box[8]};
  const double cross[3][3] = {
      {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]},
      {c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0]},
      {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]},
  };
  const double det = a[0] * cross[0][0] + a[1] * cross[0][1] + a[2] * cross[0][2];
  for (int k = 0; k < 3; ++k) {
    on[k] = periodic[static_cast<std::size_t>(k)] && det != 0.0;
    const double norm = std::sqrt(cross[k][0] * cross[k][0] + cross[k][1] * cross[k][1] +
                                  cross[k][2] * cross[k][2]);
    for (int c2 = 0; c2 < 3; ++c2) {
      normal[k][c2] = on[k] ? cross[k][c2] / det : 0.0;
    }
    reach[k] = on[k] && search > 0.0 ? search * norm / std::fabs(det) : 0.0;
  }
}

// Folded fractional extent of the atoms on each periodic axis, and the span
// that bounds |S_k| for any pair inside `search`. `frac` receives the raw
// fractions so make_frame can fold the Cartesian positions. Rejects a
// coordinate whose shift would not fit an int32.
void image_span(const double *pos, std::size_t n, const double *box,
                const std::array<bool, 3> &periodic, double search, int span[3],
                std::vector<double> *frac) {
  double normal[3][3];
  double reach[3];
  bool on[3];
  lattice_normals(box, periodic, search, normal, reach, on);
  double lo[3] = {0.0, 0.0, 0.0};
  double hi[3] = {0.0, 0.0, 0.0};
  if (frac != nullptr) {
    frac->assign(3 * n, 0.0);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      if (!on[k]) {
        continue;
      }
      const double f = pos[3 * i] * normal[k][0] + pos[3 * i + 1] * normal[k][1] +
                       pos[3 * i + 2] * normal[k][2];
      if (!(std::fabs(f) <= kMaxCells)) {
        throw Error(Code::Box, "a periodic coordinate is not finite or lies more than 2^29 "
                               "cells from the origin");
      }
      const double folded = f - std::floor(f);
      if (i == 0 || folded < lo[k]) {
        lo[k] = folded;
      }
      if (i == 0 || folded > hi[k]) {
        hi[k] = folded;
      }
      if (frac != nullptr) {
        (*frac)[3 * i + static_cast<std::size_t>(k)] = f;
      }
    }
  }
  for (int k = 0; k < 3; ++k) {
    if (!on[k] || !(search > 0.0)) {
      span[k] = 0;
      continue;
    }
    const double bound = std::ceil(reach[k] + (hi[k] - lo[k]));
    if (!(bound <= kMaxCells)) {
      throw Error(Code::Cutoff, "the search spans more than 2^29 periodic images");
    }
    span[k] = static_cast<int>(bound);
  }
}

Frame make_frame(const double *pos, std::size_t n, const double *box,
                 const std::array<bool, 3> &periodic, double search, const Fields &fields) {
  Frame frame;
  frame.geometry = fields.vectors || fields.distances;
  lattice_normals(box, periodic, search, frame.normal, frame.reach, frame.periodic);
  image_span(pos, n, box, periodic, search, frame.span, &frame.frac);
  frame.caller = pos;
  frame.cell_pos = pos;
  bool inside = true;
  for (std::size_t i = 0; i < n && inside; ++i) {
    for (int k = 0; k < 3; ++k) {
      const double f = frame.frac[3 * i + static_cast<std::size_t>(k)];
      if (frame.periodic[k] && !(f >= 0.0 && f < 1.0)) {
        inside = false;
        break;
      }
    }
  }
  if (inside) {
    return frame;
  }
  frame.wrapped.assign(pos, pos + 3 * n);
  frame.wrap.assign(3 * n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      if (!frame.periodic[k]) {
        continue;
      }
      const auto u = static_cast<std::size_t>(k);
      const double cells = std::floor(frame.frac[3 * i + u]);
      frame.frac[3 * i + u] -= cells;
      frame.wrap[3 * i + u] = static_cast<std::int32_t>(cells);
      for (int c = 0; c < 3; ++c) {
        frame.wrapped[3 * i + static_cast<std::size_t>(c)] -= cells * box[3 * k + c];
      }
    }
  }
  frame.cell_pos = frame.wrapped.data();
  return frame;
}

// Applies the whole-cell offsets W of `wrap` to a pair shift.
void caller_shift(const std::int32_t *wrap, std::size_t i, std::size_t j, int &sa, int &sb,
                  int &sc) {
  sa += wrap[3 * i] - wrap[3 * j];
  sb += wrap[3 * i + 1] - wrap[3 * j + 1];
  sc += wrap[3 * i + 2] - wrap[3 * j + 2];
}

// Rewrites the shifts of a pair table by the per-atom offsets in `wrap`.
void caller_shifts(Table &table, const std::int32_t *wrap) {
  for (std::size_t p = 0; p < table.size(); ++p) {
    caller_shift(wrap, static_cast<std::size_t>(table.first[p]),
                 static_cast<std::size_t>(table.second[p]), table.shift[3 * p],
                 table.shift[3 * p + 1], table.shift[3 * p + 2]);
  }
}

// Shifts S_k in [lo, hi] for the pair (i, j). A non-periodic axis has 0.
void shift_range(const Frame &frame, std::size_t i, std::size_t j, int lo[3], int hi[3]) {
  // Widens the window by a rounding of the fractional coordinates. The
  // distance test decides each image.
  constexpr double kSlack = 1e-9;
  for (int k = 0; k < 3; ++k) {
    if (!frame.periodic[k] || !(frame.reach[k] > 0.0)) {
      lo[k] = 0;
      hi[k] = 0;
      continue;
    }
    const auto u = static_cast<std::size_t>(k);
    const double df = frame.frac[3 * j + u] - frame.frac[3 * i + u];
    lo[k] = static_cast<int>(std::ceil(-df - frame.reach[k] - kSlack));
    hi[k] = static_cast<int>(std::floor(-df + frame.reach[k] + kSlack));
  }
}

void image_vector(const double *pos, std::size_t i, std::size_t j, int sa, int sb,
                  int sc, const double *box, double &vx, double &vy, double &vz) {
  vx = pos[3 * j] - pos[3 * i];
  vy = pos[3 * j + 1] - pos[3 * i + 1];
  vz = pos[3 * j + 2] - pos[3 * i + 2];
  vx += static_cast<double>(sa) * box[0] + static_cast<double>(sb) * box[3] +
        static_cast<double>(sc) * box[6];
  vy += static_cast<double>(sa) * box[1] + static_cast<double>(sb) * box[4] +
        static_cast<double>(sc) * box[7];
  vz += static_cast<double>(sa) * box[2] + static_cast<double>(sb) * box[5] +
        static_cast<double>(sc) * box[8];
}

// The search runs on the folded coordinates and finds S' with
// r'_j - r'_i + S' H. The caller's positions give the same image with
// S = S' + W_i - W_j, and the vector and distance are taken from those
// positions, so a table holds r_j - r_i + S H for the coordinates as
// given. A query that stores neither vectors nor distances keeps the folded
// ones. Every |W| is at most 2^29 and so is |S'|, so S fits an int32.
void to_caller(const Frame &frame, const double *box, std::size_t i, std::size_t j, int &sa,
               int &sb, int &sc, double &vx, double &vy, double &vz, double &dist) {
  if (frame.wrap.empty()) {
    return;
  }
  caller_shift(frame.wrap.data(), i, j, sa, sb, sc);
  if (!frame.geometry) {
    return;
  }
  image_vector(frame.caller, i, j, sa, sb, sc, box, vx, vy, vz);
  dist = std::sqrt(vx * vx + vy * vy + vz * vz);
}

int axis_shift(double delta, double inv, bool periodic) {
  if (!periodic || !(inv > 0.0) || !std::isfinite(delta)) {
    return 0;
  }
  const double t = delta * inv;
  const double shifted = t + std::copysign(0.5, t);
  if (!std::isfinite(shifted) || shifted >= 9223372036854775808.0 ||
      shifted <= -9223372036854775808.0) {
    return 0;
  }
  return static_cast<int>(static_cast<long long>(shifted));
}

bool one_minimum_image(const double *box, const Query &query, double search) {
  const bool sheared = box[1] != 0.0 || box[2] != 0.0 || box[3] != 0.0 || box[5] != 0.0 ||
                       box[6] != 0.0 || box[7] != 0.0;
  if (sheared) {
    return false;
  }
  const double len[3] = {box[0], box[4], box[8]};
  for (int k = 0; k < 3; ++k) {
    if (!query.periodic[static_cast<std::size_t>(k)]) {
      continue;
    }
    if (!(len[k] >= 2.0 * search)) {
      return false;
    }
  }
  return true;
}

// The cells and image walks see `frame.cell_pos`, which is inside the cell on
// every periodic axis.
void prepare_cells(Neighbours &cells, const Frame &frame, std::size_t n, const double *box,
                   const Query &query) {
  Spec spec;
  spec.cutoff = search_radius(query);
  spec.skin = 0;
  spec.periodic = query.periodic;
  const bool sheared = box[1] != 0.0 || box[2] != 0.0 || box[3] != 0.0 || box[5] != 0.0 ||
                       box[6] != 0.0 || box[7] != 0.0;
  spec.image = sheared ? Image::General : Image::Round;
  spec.mode = Mode::Stream;
  cells.update(frame.cell_pos, n, box, spec);
}

// Cell-list candidates from Neighbours, then the same image test the brute
// force used. A box at least two cutoffs wide has one image, the nearest one,
// and its shift is rounded from the folded coordinates. A narrower or
// sheared cell tries the shifts of `shift_range` on those candidates only.
// Every vector is r_j + S H - r_i from the folded coordinates.
template <typename Fn>
void each_cell(const Neighbours &cells, const double *pos, const double *box,
               const Query &query, const Frame &frame, bool one_image, std::size_t begin,
               std::size_t end, Fn &&fn) {
  auto accept = [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx,
                    double vy, double vz) {
    const double r2 = vx * vx + vy * vy + vz * vz;
    if (!accept_distance(r2, query)) {
      return;
    }
    if (!full_shell(query) && !keep_half(i, j, sa, sb, sc)) {
      return;
    }
    fn(i, j, sa, sb, sc, vx, vy, vz, std::sqrt(r2));
  };
  if (one_image) {
    const double inv[3] = {box[0] > 0.0 ? 1.0 / box[0] : 0.0, box[4] > 0.0 ? 1.0 / box[4] : 0.0,
                           box[8] > 0.0 ? 1.0 / box[8] : 0.0};
    // forHalf is keep_half when a single image exists: j < i is dropped
    // and a self pair is not emitted.
    auto take = [&](std::int32_t i, std::int32_t j, double, double, double, double) {
      const auto ui = static_cast<std::size_t>(i);
      const auto uj = static_cast<std::size_t>(j);
      const int sa = axis_shift(pos[3 * ui] - pos[3 * uj], inv[0], query.periodic[0]);
      const int sb = axis_shift(pos[3 * ui + 1] - pos[3 * uj + 1], inv[1], query.periodic[1]);
      const int sc = axis_shift(pos[3 * ui + 2] - pos[3 * uj + 2], inv[2], query.periodic[2]);
      double vx = 0;
      double vy = 0;
      double vz = 0;
      image_vector(pos, ui, uj, sa, sb, sc, box, vx, vy, vz);
      const double r2 = vx * vx + vy * vy + vz * vz;
      if (!accept_distance(r2, query)) {
        return;
      }
      fn(i, j, sa, sb, sc, vx, vy, vz, std::sqrt(r2));
    };
    if (full_shell(query)) {
      cells.forDirected(static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end), take);
    } else {
      cells.forHalf(static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end), take);
    }
    return;
  }
  cells.forDirected(static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end),
                    [&](std::int32_t i, std::int32_t j, double, double, double, double) {
                      int lo[3];
                      int hi[3];
                      shift_range(frame, static_cast<std::size_t>(i),
                                  static_cast<std::size_t>(j), lo, hi);
                      for (int sa = lo[0]; sa <= hi[0]; ++sa) {
                        for (int sb = lo[1]; sb <= hi[1]; ++sb) {
                          for (int sc = lo[2]; sc <= hi[2]; ++sc) {
                            double vx = 0;
                            double vy = 0;
                            double vz = 0;
                            image_vector(pos, static_cast<std::size_t>(i),
                                         static_cast<std::size_t>(j), sa, sb, sc, box, vx, vy,
                                         vz);
                            accept(i, j, sa, sb, sc, vx, vy, vz);
                          }
                        }
                      }
                    });
  for (std::size_t i = begin; i < end; ++i) {
    int lo[3];
    int hi[3];
    shift_range(frame, i, i, lo, hi);
    for (int sa = lo[0]; sa <= hi[0]; ++sa) {
      for (int sb = lo[1]; sb <= hi[1]; ++sb) {
        for (int sc = lo[2]; sc <= hi[2]; ++sc) {
          if (sa == 0 && sb == 0 && sc == 0) {
            continue;
          }
          double vx = 0;
          double vy = 0;
          double vz = 0;
          image_vector(pos, i, i, sa, sb, sc, box, vx, vy, vz);
          accept(static_cast<std::int32_t>(i), static_cast<std::int32_t>(i), sa, sb, sc, vx,
                 vy, vz);
        }
      }
    }
  }
}

// The shift is kept whether or not the query stores it: merge_clusters
// tells the images of one partner apart by it.
void push_rec(std::vector<Rec> &out, const Query &query, std::int32_t first,
              std::int32_t second, int sa, int sb, int sc, double vx, double vy,
              double vz, double dist, std::uint8_t mask) {
  Rec rec;
  rec.first = first;
  rec.second = second;
  rec.shift[0] = sa;
  rec.shift[1] = sb;
  rec.shift[2] = sc;
  if (query.fields.vectors) {
    rec.vector[0] = vx;
    rec.vector[1] = vy;
    rec.vector[2] = vz;
  }
  if (query.fields.distances) {
    rec.distance = dist;
  }
  rec.mask = mask;
  out.push_back(rec);
}

void merge_clusters(std::vector<Rec> &rows) {
  if (rows.empty()) {
    return;
  }
  std::sort(rows.begin(), rows.end(), [](const Rec &a, const Rec &b) {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    if (a.second != b.second) {
      return a.second < b.second;
    }
    if (a.shift[0] != b.shift[0]) {
      return a.shift[0] < b.shift[0];
    }
    if (a.shift[1] != b.shift[1]) {
      return a.shift[1] < b.shift[1];
    }
    return a.shift[2] < b.shift[2];
  });
  std::vector<Rec> merged;
  merged.reserve(rows.size());
  for (const Rec &rec : rows) {
    if (!merged.empty()) {
      Rec &last = merged.back();
      if (last.first == rec.first && last.second == rec.second &&
          last.shift[0] == rec.shift[0] && last.shift[1] == rec.shift[1] &&
          last.shift[2] == rec.shift[2]) {
        const auto lowest = [](std::uint8_t bits) {
          return static_cast<unsigned>(bits & -static_cast<int>(bits));
        };
        if (lowest(rec.mask) < lowest(last.mask)) {
          last.vector[0] = rec.vector[0];
          last.vector[1] = rec.vector[1];
          last.vector[2] = rec.vector[2];
          last.distance = rec.distance;
        }
        last.mask = static_cast<std::uint8_t>(last.mask | rec.mask);
        continue;
      }
    }
    merged.push_back(rec);
  }
  rows.swap(merged);
}

void collect(const Neighbours &cells, const double *pos, std::size_t n, const double *box,
             const Query &query, const Frame &frame, bool one_image, std::size_t begin,
             std::size_t end, std::vector<Rec> &out) {
  if (clustered(query)) {
    const std::size_t c0 = begin & ~std::size_t{3};
    const std::size_t c1 = std::min(n, (end + 3) & ~std::size_t{3});
    Query full = query;
    full.shape = Pairs{Shell::Full};
    std::vector<Rec> raw;
    each_cell(cells, pos, box, full, frame, one_image, c0, c1,
              [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx, double vy,
                  double vz, double dist) {
                if (static_cast<std::size_t>(i) < begin || static_cast<std::size_t>(i) >= end) {
                  return;
                }
                to_caller(frame, box, static_cast<std::size_t>(i), static_cast<std::size_t>(j),
                          sa, sb, sc, vx, vy, vz, dist);
                const auto lane = static_cast<unsigned>(i & 3);
                push_rec(raw, query, i & ~std::int32_t{3}, j, sa, sb, sc, vx, vy, vz, dist,
                         static_cast<std::uint8_t>(1u << lane));
              });
    merge_clusters(raw);
    out.insert(out.end(), raw.begin(), raw.end());
    return;
  }
  each_cell(cells, pos, box, query, frame, one_image, begin, end,
            [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx, double vy,
                double vz, double dist) {
              to_caller(frame, box, static_cast<std::size_t>(i), static_cast<std::size_t>(j), sa,
                        sb, sc, vx, vy, vz, dist);
              push_rec(out, query, i, j, sa, sb, sc, vx, vy, vz, dist, 0xff);
            });
}

void store_records(Table &table, const Query &query, const std::vector<Rec> &rows) {
  table.first.resize(rows.size());
  table.second.resize(rows.size());
  table.mask.resize(rows.size());
  table.shift.clear();
  table.vector.clear();
  table.distance.clear();
  if (query.fields.shifts) {
    table.shift.resize(rows.size() * 3);
  }
  if (query.fields.vectors) {
    table.vector.resize(rows.size() * 3);
  }
  if (query.fields.distances) {
    table.distance.resize(rows.size());
  }
  for (std::size_t p = 0; p < rows.size(); ++p) {
    const Rec &rec = rows[p];
    table.first[p] = rec.first;
    table.second[p] = rec.second;
    table.mask[p] = rec.mask;
    if (query.fields.shifts) {
      table.shift[3 * p] = rec.shift[0];
      table.shift[3 * p + 1] = rec.shift[1];
      table.shift[3 * p + 2] = rec.shift[2];
    }
    if (query.fields.vectors) {
      table.vector[3 * p] = rec.vector[0];
      table.vector[3 * p + 1] = rec.vector[1];
      table.vector[3 * p + 2] = rec.vector[2];
    }
    if (query.fields.distances) {
      table.distance[p] = rec.distance;
    }
  }
  table.stored = true;
  table.device = 0;
}

struct Columns {
  std::vector<std::int32_t> first;
  std::vector<std::int32_t> second;
  std::vector<std::int32_t> shift;
  std::vector<double> vector;
  std::vector<double> distance;
  std::vector<std::uint8_t> mask;
};

// The slot count is a guess. Value-initialising the spare tail writes
// gigabytes that the next loop overwrites. libstdc++ lets the size
// advance without that write; other libraries fall back to resize.
template <typename T>
void grow_uninitialized(std::vector<T> &values, std::size_t n) {
#if defined(__GLIBCXX__)
  struct Tail : std::vector<T> {
    void extend(std::size_t count) {
      if (count > this->capacity()) {
        this->reserve(count);
      }
      this->_M_impl._M_finish = this->_M_impl._M_start + count;
    }
  };
  static_assert(sizeof(Tail) == sizeof(std::vector<T>));
  reinterpret_cast<Tail &>(values).extend(n);
#else
  values.resize(n);
#endif
}

std::size_t pair_guess(std::size_t natoms, std::size_t nall, const double *box, double search,
                       bool full, int stencil) {
  const double vol =
      std::abs(box[0] * (box[4] * box[8] - box[5] * box[7]) -
               box[1] * (box[3] * box[8] - box[5] * box[6]) +
               box[2] * (box[3] * box[7] - box[4] * box[6]));
  double per = 48;
  if (vol > 0.0 && search > 0.0 && nall > 0) {
    const double sphere = 4.1887902047863905 * search * search * search;
    per = static_cast<double>(nall) / vol * sphere;
    if (!full) {
      per *= 0.5;
    }
    if (stencil > 1) {
      per *= static_cast<double>(stencil);
    }
    per = per * 1.5 + 4.0;
  }
  if (per < 4.0) {
    per = 4.0;
  }
  if (per > 256.0) {
    per = 256.0;
  }
  return static_cast<std::size_t>(per * static_cast<double>(natoms) + 1.0);
}

struct alignas(64) TeamCursor {
  std::size_t count = 0;
};

void compact_i32(std::vector<std::int32_t> &col, int width, const TeamCursor *cursors,
                 const std::size_t *base, int nt) {
  std::size_t dst = 0;
  for (int t = 0; t < nt; ++t) {
    const std::size_t n = cursors[t].count * static_cast<std::size_t>(width);
    const std::size_t src = base[t] * static_cast<std::size_t>(width);
    if (n != 0 && dst != src) {
      std::memmove(col.data() + dst, col.data() + src, n * sizeof(std::int32_t));
    }
    dst += n;
  }
  col.resize(dst);
}

void compact_f64(std::vector<double> &col, int width, const TeamCursor *cursors,
                 const std::size_t *base, int nt) {
  std::size_t dst = 0;
  for (int t = 0; t < nt; ++t) {
    const std::size_t n = cursors[t].count * static_cast<std::size_t>(width);
    const std::size_t src = base[t] * static_cast<std::size_t>(width);
    if (n != 0 && dst != src) {
      std::memmove(col.data() + dst, col.data() + src, n * sizeof(double));
    }
    dst += n;
  }
  col.resize(dst);
}

void compact_u8(std::vector<std::uint8_t> &col, const TeamCursor *cursors, const std::size_t *base,
                int nt) {
  std::size_t dst = 0;
  for (int t = 0; t < nt; ++t) {
    const std::size_t n = cursors[t].count;
    const std::size_t src = base[t];
    if (n != 0 && dst != src) {
      std::memmove(col.data() + dst, col.data() + src, n);
    }
    dst += n;
  }
  col.resize(dst);
}

void move_columns(Table &table, Columns &cols) {
  table.first = std::move(cols.first);
  table.second = std::move(cols.second);
  table.shift = std::move(cols.shift);
  table.vector = std::move(cols.vector);
  table.distance = std::move(cols.distance);
  table.mask = std::move(cols.mask);
  table.stored = true;
  table.device = 0;
}

// Fills `table` from a frame of these positions. The query has passed
// check_geometry, and `table` is cleared only here.
void search_frame(Table &table, const Frame &frame, std::size_t n, const double *box9,
                  const Query &query);

} // namespace

struct BuilderState {
  Table table;
  std::vector<double> ref;
  std::vector<double> ref_frac;
  std::vector<std::int32_t> wrap;
  double box[9]{};
  Query query{};
  int span[3]{};
  std::size_t n = 0;
  std::uint64_t builds = 0;
  std::uint64_t frames = 0;
  bool have = false;
  bool reused = false;
  bool unchanged = false;

  // Moves the stored shifts from the wrap of the table to `now`. A pair row
  // takes the difference of its two atoms. A cluster row shares one shift
  // among four atoms, so a cluster list whose wrap changed is searched
  // again and this returns false.
  bool rewrap(const std::vector<std::int32_t> &now) {
    // An empty wrap is all zeros.
    const auto zero = [](const std::vector<std::int32_t> &w) {
      return std::all_of(w.begin(), w.end(), [](std::int32_t x) { return x == 0; });
    };
    if (now == wrap || (zero(now) && zero(wrap))) {
      wrap = now;
      return true;
    }
    if (clustered(query)) {
      return false;
    }
    if (!table.shift.empty()) {
      const std::size_t atoms = 3 * n;
      std::vector<std::int32_t> delta(atoms, 0);
      for (std::size_t k = 0; k < atoms; ++k) {
        delta[k] = (now.empty() ? 0 : now[k]) - (wrap.empty() ? 0 : wrap[k]);
      }
      caller_shifts(table, delta.data());
    }
    wrap = now;
    return true;
  }

  // Puts each folded atom on the image nearest its folded reference. The
  // whole cells J between the two fractions are taken off a copy in
  // `moved`, and the shifts of that copy have the basis W + J in `basis`.
  // An atom that crosses a face of the cell then moves by its step and not
  // by a lattice vector. Returns the coordinates for the skin test.
  const double *nearest(const Frame &frame, const double *box9,
                        std::vector<std::int32_t> &basis, std::vector<double> &moved) const {
    basis = frame.wrap;
    for (std::size_t i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        if (!frame.periodic[k]) {
          continue;
        }
        const std::size_t at = 3 * i + static_cast<std::size_t>(k);
        const double jump = std::round(frame.frac[at] - ref_frac[at]);
        if (jump == 0.0) {
          continue;
        }
        if (moved.empty()) {
          moved.assign(frame.cell_pos, frame.cell_pos + 3 * n);
        }
        if (basis.empty()) {
          basis.assign(3 * n, 0);
        }
        basis[at] += static_cast<std::int32_t>(jump);
        for (int c = 0; c < 3; ++c) {
          moved[3 * i + static_cast<std::size_t>(c)] -= jump * box9[3 * k + c];
        }
      }
    }
    return moved.empty() ? frame.cell_pos : moved.data();
  }

  void update(const double *positions, std::size_t natoms, const double *box9,
              const Query &q) {
    check_geometry(positions, natoms, box9, q);
    const Frame frame = make_frame(positions, natoms, box9, q.periodic, search_radius(q), q.fields);
    ++frames;
    const double *current = frame.cell_pos;
    reused = false;
    unchanged = false;
    if (have && n == natoms && query.cutoff == q.cutoff && query.skin == q.skin &&
        query.shape == q.shape && query.cut == q.cut && query.periodic == q.periodic &&
        query.fields.shifts == q.fields.shifts && query.fields.vectors == q.fields.vectors &&
        query.fields.distances == q.fields.distances) {
      const bool same_box = std::memcmp(box, box9, sizeof(box)) == 0;
      if (same_box &&
          (natoms == 0 || std::memcmp(ref.data(), current, 3 * natoms * sizeof(double)) == 0)) {
        if (rewrap(frame.wrap)) {
          reused = true;
          unchanged = true;
          return;
        }
      } else {
        bool fits = true;
        if (!same_box) {
          // A cell change that needs a wider folded-image span rebuilds.
          fits = frame.span[0] <= span[0] && frame.span[1] <= span[1] &&
                 frame.span[2] <= span[2];
        }
        const double reach[3] = {
            static_cast<double>(span[0]),
            static_cast<double>(span[1]),
            static_cast<double>(span[2]),
        };
        std::vector<std::int32_t> basis;
        std::vector<double> moved;
        const double *near = fits ? nearest(frame, box9, basis, moved) : current;
        if (fits && withinSkin(ref.data(), near, natoms, box, box9, reach, q.skin) &&
            rewrap(basis)) {
          reused = true;
          return;
        }
      }
    }
    search_frame(table, frame, natoms, box9, q);
    std::copy(frame.span, frame.span + 3, span);
    ref.clear();
    if (natoms != 0) {
      ref.assign(current, current + 3 * natoms);
    }
    ref_frac = frame.frac;
    wrap = frame.wrap;
    std::memcpy(box, box9, sizeof(box));
    query = q;
    n = natoms;
    have = true;
    ++builds;
  }
};

Builder::Builder() : state_(std::make_unique<BuilderState>()) {}

Builder::~Builder() = default;

Builder::Builder(const Builder &other)
    : state_(other.state_ ? std::make_unique<BuilderState>(*other.state_)
                          : std::make_unique<BuilderState>()) {}

Builder &Builder::operator=(const Builder &other) {
  if (this != &other) {
    Builder copy(other);
    state_ = std::move(copy.state_);
  }
  return *this;
}

Builder::Builder(Builder &&other) noexcept = default;

Builder &Builder::operator=(Builder &&other) noexcept = default;

void Builder::update(const double *positions, std::size_t n, const double *box9,
                     const Query &query) {
  if (!state_) {
    state_ = std::make_unique<BuilderState>();
  }
  state_->update(positions, n, box9, query);
}

const Table &Builder::table() const noexcept {
  static const Table empty{};
  return state_ ? state_->table : empty;
}

std::uint64_t Builder::builds() const noexcept { return state_ ? state_->builds : 0; }

std::uint64_t Builder::frames() const noexcept { return state_ ? state_->frames : 0; }

bool Builder::reused() const noexcept { return state_ && state_->reused; }

bool Builder::unchanged() const noexcept { return state_ && state_->unchanged; }

struct WalkState {
  const double *pos = nullptr;
  std::size_t n = 0;
  double box[9]{};
  Query query{};
  Frame frame;
  bool one_image = false;
  Neighbours cells;
  std::size_t atom = 0;
  std::vector<std::int32_t> hold_first;
  std::vector<std::int32_t> hold_second;
  std::vector<std::int32_t> hold_shift;
  std::vector<double> hold_vector;
  std::vector<double> hold_distance;
  std::vector<std::uint8_t> hold_mask;
  std::size_t hold_at = 0;
  bool finished = false;

  std::size_t pull(Buffers dest, bool &done);
};

Walk::Walk(const double *positions, std::size_t n, const double *box9, const Query &query)
    : state_(std::make_unique<WalkState>()) {
  check_geometry(positions, n, box9, query);
  WalkState &w = *state_;
  w.n = n;
  w.query = query;
  std::memcpy(w.box, box9, sizeof(w.box));
  w.frame = make_frame(positions, n, w.box, w.query.periodic, search_radius(w.query),
                       w.query.fields);
  w.pos = w.frame.cell_pos;
  w.one_image = one_minimum_image(w.box, w.query, search_radius(w.query));
  prepare_cells(w.cells, w.frame, n, w.box, w.query);
}

Walk::~Walk() = default;

Walk::Walk(Walk &&other) noexcept = default;

Walk &Walk::operator=(Walk &&other) noexcept = default;

std::size_t Walk::pull(Buffers dest, bool &done) {
  if (!state_) {
    throw Error(Code::Null, "the walk was moved from");
  }
  return state_->pull(dest, done);
}

std::size_t WalkState::pull(Buffers dest, bool &done) {
  const std::size_t cap = dest.first.size();
  if (cap > 0 && dest.second.size() != cap) {
    throw Error(Code::Short, "first and second buffers have different lengths");
  }
  if (cap > 0 && query.fields.shifts && dest.shift.size() < cap * 3) {
    throw Error(Code::Short, "shift buffer needs 3 entries per pair");
  }
  if (cap > 0 && query.fields.vectors && dest.vector.size() < cap * 3) {
    throw Error(Code::Short, "vector buffer needs 3 entries per pair");
  }
  if (cap > 0 && query.fields.distances && dest.distance.size() < cap) {
    throw Error(Code::Short, "distance buffer is shorter than the pair buffer");
  }
  if (cap > 0 && clustered(query) && dest.mask.size() < cap) {
    throw Error(Code::Short, "a cluster list needs one mask byte per pair");
  }
  if (finished || cap == 0) {
    done = finished;
    return 0;
  }
  auto stash = [&](const std::vector<Rec> &rows) {
    const bool shifts = query.fields.shifts;
    for (const Rec &rec : rows) {
      hold_first.push_back(rec.first);
      hold_second.push_back(rec.second);
      hold_shift.push_back(shifts ? rec.shift[0] : 0);
      hold_shift.push_back(shifts ? rec.shift[1] : 0);
      hold_shift.push_back(shifts ? rec.shift[2] : 0);
      hold_vector.push_back(rec.vector[0]);
      hold_vector.push_back(rec.vector[1]);
      hold_vector.push_back(rec.vector[2]);
      hold_distance.push_back(rec.distance);
      hold_mask.push_back(rec.mask);
    }
  };
  std::size_t written = 0;
  while (written < cap) {
    if (hold_at >= hold_first.size()) {
      hold_first.clear();
      hold_second.clear();
      hold_shift.clear();
      hold_vector.clear();
      hold_distance.clear();
      hold_mask.clear();
      hold_at = 0;
      if (atom >= n) {
        finished = true;
        break;
      }
      const std::size_t end = std::min(
          n, atom + (clustered(query) ? std::size_t{4} : std::size_t{1}));
      std::vector<Rec> rows;
      collect(cells, pos, n, box, query, frame, one_image, atom, end, rows);
      atom = end;
      stash(rows);
      if (hold_first.empty()) {
        continue;
      }
    }
    dest.first[written] = hold_first[hold_at];
    dest.second[written] = hold_second[hold_at];
    if (dest.shift.size() >= (written + 1) * 3) {
      dest.shift[3 * written] = hold_shift[3 * hold_at];
      dest.shift[3 * written + 1] = hold_shift[3 * hold_at + 1];
      dest.shift[3 * written + 2] = hold_shift[3 * hold_at + 2];
    }
    if (dest.vector.size() >= (written + 1) * 3) {
      dest.vector[3 * written] = hold_vector[3 * hold_at];
      dest.vector[3 * written + 1] = hold_vector[3 * hold_at + 1];
      dest.vector[3 * written + 2] = hold_vector[3 * hold_at + 2];
    }
    if (written < dest.distance.size()) {
      dest.distance[written] = hold_distance[hold_at];
    }
    if (written < dest.mask.size()) {
      dest.mask[written] = hold_mask[hold_at];
    }
    ++hold_at;
    ++written;
  }
  done = finished && hold_at >= hold_first.size();
  return written;
}

void build(Table &table, const double *positions, std::size_t n, const double *box9,
           const Query &query) {
  check_geometry(positions, n, box9, query);
  if (n == 0 || !(query.cutoff > 0.0)) {
    table = Table{};
    table.stored = true;
    return;
  }
  const Frame frame =
      make_frame(positions, n, box9, query.periodic, search_radius(query), query.fields);
  search_frame(table, frame, n, box9, query);
}

namespace {

void search_frame(Table &table, const Frame &frame, std::size_t n, const double *box9,
                  const Query &query) {
  table = Table{};
  if (n == 0 || !(query.cutoff > 0.0)) {
    table.stored = true;
    return;
  }
  int nmax[3];
  const double search = search_radius(query);
  shift_bounds(box9, query.periodic, search, nmax);
  const bool one_image = one_minimum_image(box9, query, search);
  // A handful of atoms is cheaper as a direct pair loop. The cell list
  // spends its time building bins, and a sheared cell of this size has
  // only a few bins, so the stencil visits every pair anyway.
  if (!clustered(query) && n < 512) {
    Columns cols;
    const bool full = full_shell(query);
    const int stencil =
        one_image ? 1 : (2 * nmax[0] + 1) * (2 * nmax[1] + 1) * (2 * nmax[2] + 1);
    const std::size_t slots = pair_guess(n, n, box9, search, full, stencil);
    cols.first.reserve(slots);
    cols.second.reserve(slots);
    cols.mask.reserve(slots);
    if (query.fields.shifts) {
      cols.shift.reserve(slots * 3);
    }
    if (query.fields.vectors) {
      cols.vector.reserve(slots * 3);
    }
    if (query.fields.distances) {
      cols.distance.reserve(slots);
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = full ? 0 : i; j < n; ++j) {
        int lo[3];
        int hi[3];
        shift_range(frame, i, j, lo, hi);
        for (int sa = lo[0]; sa <= hi[0]; ++sa) {
          for (int sb = lo[1]; sb <= hi[1]; ++sb) {
            for (int sc = lo[2]; sc <= hi[2]; ++sc) {
              const auto ii = static_cast<std::int32_t>(i);
              const auto jj = static_cast<std::int32_t>(j);
              if (!full && !keep_half(ii, jj, sa, sb, sc)) {
                continue;
              }
              double vx = 0;
              double vy = 0;
              double vz = 0;
              image_vector(frame.cell_pos, i, j, sa, sb, sc, box9, vx, vy, vz);
              const double r2 = vx * vx + vy * vy + vz * vz;
              if (!accept_distance(r2, query)) {
                continue;
              }
              double dist = std::sqrt(r2);
              int ta = sa;
              int tb = sb;
              int tc = sc;
              to_caller(frame, box9, i, j, ta, tb, tc, vx, vy, vz, dist);
              cols.first.push_back(ii);
              cols.second.push_back(jj);
              cols.mask.push_back(0xff);
              if (query.fields.shifts) {
                cols.shift.push_back(ta);
                cols.shift.push_back(tb);
                cols.shift.push_back(tc);
              }
              if (query.fields.vectors) {
                cols.vector.push_back(vx);
                cols.vector.push_back(vy);
                cols.vector.push_back(vz);
              }
              if (query.fields.distances) {
                cols.distance.push_back(dist);
              }
            }
          }
        }
      }
    }
    move_columns(table, cols);
    return;
  }
  Neighbours cells;
  prepare_cells(cells, frame, n, box9, query);
  const int nt = teamSize(n);
  if (!clustered(query)) {
    std::vector<Columns> parts(static_cast<std::size_t>(std::max(nt, 1)));
    const bool full = full_shell(query);
    auto fill = [&](Columns &cols, std::size_t begin, std::size_t end) {
      const int stencil = one_image ? 1
                                   : (2 * nmax[0] + 1) * (2 * nmax[1] + 1) * (2 * nmax[2] + 1);
      const std::size_t slots = pair_guess(end - begin, n, box9, search, full, stencil);
      grow_uninitialized(cols.first, slots);
      grow_uninitialized(cols.second, slots);
      grow_uninitialized(cols.mask, slots);
      if (query.fields.shifts) {
        grow_uninitialized(cols.shift, slots * 3);
      }
      if (query.fields.vectors) {
        grow_uninitialized(cols.vector, slots * 3);
      }
      if (query.fields.distances) {
        grow_uninitialized(cols.distance, slots);
      }
      auto *first = cols.first.data();
      auto *second = cols.second.data();
      auto *mask = cols.mask.data();
      auto *shift = cols.shift.data();
      auto *vector = cols.vector.data();
      auto *distance = cols.distance.data();
      std::size_t count = 0;
      Columns extra;
      each_cell(cells, frame.cell_pos, box9, query, frame, one_image, begin, end,
                [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double vx,
                    double vy, double vz, double dist) {
                  to_caller(frame, box9, static_cast<std::size_t>(i),
                            static_cast<std::size_t>(j), sa, sb, sc, vx, vy, vz, dist);
                  if (count == slots) {
                    extra.first.push_back(i);
                    extra.second.push_back(j);
                    extra.mask.push_back(0xff);
                    if (query.fields.shifts) {
                      extra.shift.push_back(sa);
                      extra.shift.push_back(sb);
                      extra.shift.push_back(sc);
                    }
                    if (query.fields.vectors) {
                      extra.vector.push_back(vx);
                      extra.vector.push_back(vy);
                      extra.vector.push_back(vz);
                    }
                    if (query.fields.distances) {
                      extra.distance.push_back(dist);
                    }
                    return;
                  }
                  first[count] = i;
                  second[count] = j;
                  mask[count] = 0xff;
                  if (query.fields.shifts) {
                    shift[3 * count] = sa;
                    shift[3 * count + 1] = sb;
                    shift[3 * count + 2] = sc;
                  }
                  if (query.fields.vectors) {
                    vector[3 * count] = vx;
                    vector[3 * count + 1] = vy;
                    vector[3 * count + 2] = vz;
                  }
                  if (query.fields.distances) {
                    distance[count] = dist;
                  }
                  ++count;
                });
      if (!extra.first.empty()) {
        cols.first.resize(count);
        cols.second.resize(count);
        cols.mask.resize(count);
        if (query.fields.shifts) {
          cols.shift.resize(count * 3);
        }
        if (query.fields.vectors) {
          cols.vector.resize(count * 3);
        }
        if (query.fields.distances) {
          cols.distance.resize(count);
        }
        cols.first.insert(cols.first.end(), extra.first.begin(), extra.first.end());
        cols.second.insert(cols.second.end(), extra.second.begin(), extra.second.end());
        cols.mask.insert(cols.mask.end(), extra.mask.begin(), extra.mask.end());
        if (query.fields.shifts) {
          cols.shift.insert(cols.shift.end(), extra.shift.begin(), extra.shift.end());
        }
        if (query.fields.vectors) {
          cols.vector.insert(cols.vector.end(), extra.vector.begin(), extra.vector.end());
        }
        if (query.fields.distances) {
          cols.distance.insert(cols.distance.end(), extra.distance.begin(), extra.distance.end());
        }
        return;
      }
      cols.first.resize(count);
      cols.second.resize(count);
      cols.mask.resize(count);
      if (query.fields.shifts) {
        cols.shift.resize(count * 3);
      }
      if (query.fields.vectors) {
        cols.vector.resize(count * 3);
      }
      if (query.fields.distances) {
        cols.distance.resize(count);
      }
    };
    auto append = [&](Columns &dst, const Columns &src) {
      dst.first.insert(dst.first.end(), src.first.begin(), src.first.end());
      dst.second.insert(dst.second.end(), src.second.begin(), src.second.end());
      dst.mask.insert(dst.mask.end(), src.mask.begin(), src.mask.end());
      if (query.fields.shifts) {
        dst.shift.insert(dst.shift.end(), src.shift.begin(), src.shift.end());
      }
      if (query.fields.vectors) {
        dst.vector.insert(dst.vector.end(), src.vector.begin(), src.vector.end());
      }
      if (query.fields.distances) {
        dst.distance.insert(dst.distance.end(), src.distance.begin(), src.distance.end());
      }
    };
    if (nt <= 1) {
      fill(parts[0], 0, n);
      move_columns(table, parts[0]);
      return;
    }
    // The cell team walks one folded image per pair. A sheared cell, or one
    // narrower than two search radii, needs the shift stencil, so each
    // slice runs the serial walk on its own atom range instead.
    if (!one_image) {
      forEachSlice(nt, [&](int t) {
        const auto range = partition(static_cast<std::int64_t>(n), t, nt);
        fill(parts[static_cast<std::size_t>(t)], static_cast<std::size_t>(range.begin),
             static_cast<std::size_t>(range.end));
      });
      for (int t = 1; t < nt; ++t) {
        append(parts[0], parts[static_cast<std::size_t>(t)]);
      }
      move_columns(table, parts[0]);
      return;
    }
    const int stencil = 1;
    const std::size_t slice = pair_guess(n, n, box9, search, full, stencil) /
                                  static_cast<std::size_t>(nt) +
                              8;
    const std::size_t slots = slice * static_cast<std::size_t>(nt);
    Columns cols;
    grow_uninitialized(cols.first, slots);
    grow_uninitialized(cols.second, slots);
    grow_uninitialized(cols.mask, slots);
    if (query.fields.shifts) {
      grow_uninitialized(cols.shift, slots * 3);
    }
    if (query.fields.vectors) {
      grow_uninitialized(cols.vector, slots * 3);
    }
    if (query.fields.distances) {
      grow_uninitialized(cols.distance, slots);
    }
    std::vector<std::size_t> base(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
      base[static_cast<std::size_t>(t)] = slice * static_cast<std::size_t>(t);
    }
    std::vector<TeamCursor> cursors(static_cast<std::size_t>(nt));
    std::vector<Columns> extra(static_cast<std::size_t>(nt));
    const double inv0 = box9[0] > 0.0 ? 1.0 / box9[0] : 0.0;
    const double inv1 = box9[4] > 0.0 ? 1.0 / box9[4] : 0.0;
    const double inv2 = box9[8] > 0.0 ? 1.0 / box9[8] : 0.0;
    cells.forCellTeam(
        !full, nt,
        [&](int thread, std::int32_t i, std::int32_t j, double, double, double, double) {
          const auto ui = static_cast<std::size_t>(i);
          const auto uj = static_cast<std::size_t>(j);
          int sa =
              axis_shift(frame.cell_pos[3 * ui] - frame.cell_pos[3 * uj], inv0,
                         query.periodic[0]);
          int sb =
              axis_shift(frame.cell_pos[3 * ui + 1] - frame.cell_pos[3 * uj + 1], inv1,
                         query.periodic[1]);
          int sc =
              axis_shift(frame.cell_pos[3 * ui + 2] - frame.cell_pos[3 * uj + 2], inv2,
                         query.periodic[2]);
          double vx = 0;
          double vy = 0;
          double vz = 0;
          image_vector(frame.cell_pos, ui, uj, sa, sb, sc, box9, vx, vy, vz);
          const double r2 = vx * vx + vy * vy + vz * vz;
          if (!accept_distance(r2, query)) {
            return;
          }
          double dist = std::sqrt(r2);
          to_caller(frame, box9, ui, uj, sa, sb, sc, vx, vy, vz, dist);
          const auto t = static_cast<std::size_t>(thread);
          auto &cursor = cursors[t];
          if (cursor.count == slice) {
            extra[t].first.push_back(i);
            extra[t].second.push_back(j);
            extra[t].mask.push_back(0xff);
            if (query.fields.shifts) {
              extra[t].shift.push_back(sa);
              extra[t].shift.push_back(sb);
              extra[t].shift.push_back(sc);
            }
            if (query.fields.vectors) {
              extra[t].vector.push_back(vx);
              extra[t].vector.push_back(vy);
              extra[t].vector.push_back(vz);
            }
            if (query.fields.distances) {
              extra[t].distance.push_back(dist);
            }
            return;
          }
          const std::size_t p = base[t] + cursor.count;
          cols.first[p] = i;
          cols.second[p] = j;
          cols.mask[p] = 0xff;
          if (query.fields.shifts) {
            cols.shift[3 * p] = sa;
            cols.shift[3 * p + 1] = sb;
            cols.shift[3 * p + 2] = sc;
          }
          if (query.fields.vectors) {
            cols.vector[3 * p] = vx;
            cols.vector[3 * p + 1] = vy;
            cols.vector[3 * p + 2] = vz;
          }
          if (query.fields.distances) {
            cols.distance[p] = dist;
          }
          ++cursor.count;
        });
    compact_i32(cols.first, 1, cursors.data(), base.data(), nt);
    compact_i32(cols.second, 1, cursors.data(), base.data(), nt);
    compact_u8(cols.mask, cursors.data(), base.data(), nt);
    if (query.fields.shifts) {
      compact_i32(cols.shift, 3, cursors.data(), base.data(), nt);
    }
    if (query.fields.vectors) {
      compact_f64(cols.vector, 3, cursors.data(), base.data(), nt);
    }
    if (query.fields.distances) {
      compact_f64(cols.distance, 1, cursors.data(), base.data(), nt);
    }
    for (int t = 0; t < nt; ++t) {
      append(cols, extra[static_cast<std::size_t>(t)]);
    }
    move_columns(table, cols);
    return;
  }
  std::vector<std::vector<Rec>> parts(static_cast<std::size_t>(std::max(nt, 1)));
  if (nt <= 1) {
    collect(cells, frame.cell_pos, n, box9, query, frame, one_image, 0, n, parts[0]);
  } else {
    forEachSlice(nt, [&](int t) {
      const auto range = partition(static_cast<std::int64_t>(n), t, nt);
      collect(cells, frame.cell_pos, n, box9, query, frame, one_image,
              static_cast<std::size_t>(range.begin), static_cast<std::size_t>(range.end),
              parts[static_cast<std::size_t>(t)]);
    });
  }
  std::vector<Rec> rows;
  for (const auto &part : parts) {
    rows.insert(rows.end(), part.begin(), part.end());
  }
  if (nt > 1) {
    merge_clusters(rows);
  }
  store_records(table, query, rows);
}

} // namespace

void buildByAtom(Table &table, const double *positions, std::size_t n, const double *box9,
                 const Query &query) {
  check_geometry(positions, n, box9, query);
  if (n == 0 || !(query.cutoff > 0.0) || n < 512 || clustered(query)) {
    build(table, positions, n, box9, query);
    return;
  }
  const double search = search_radius(query);
  const bool one_image = one_minimum_image(box9, query, search);
  const Frame frame = make_frame(positions, n, box9, query.periodic, search, query.fields);
  table = Table{};
  Neighbours cells;
  prepare_cells(cells, frame, n, box9, query);
  std::vector<Rec> rows;
  rows.reserve(n * 8);
  for (std::size_t i = 0; i < n; ++i) {
    collect(cells, frame.cell_pos, n, box9, query, frame, one_image, i, i + 1, rows);
  }
  store_records(table, query, rows);
}

void build(Batch &batch, const Systems &systems, const Query &query) {
  if (systems.systems > 0 &&
      (systems.positions == nullptr || systems.count == nullptr || systems.box9 == nullptr)) {
    throw Error(Code::Null, "a batch needs positions, a count per system and a box per system");
  }
  Batch out;
  std::size_t origin = 0;
  for (std::size_t s = 0; s < systems.systems; ++s) {
    const auto count = systems.count[s];
    if (count < 0) {
      throw Error(Code::Shape, "a system count must be >= 0");
    }
    const auto n = static_cast<std::size_t>(count);
    Table table;
    build(table, systems.positions + 3 * origin, n, systems.box9 + 9 * s, query);
    const auto base = static_cast<std::int32_t>(origin);
    for (std::size_t p = 0; p < table.size(); ++p) {
      out.pairs.first.push_back(table.first[p] + base);
      out.pairs.second.push_back(table.second[p] + base);
      out.pairs.mask.push_back(table.mask[p]);
      out.system.push_back(static_cast<std::int32_t>(s));
    }
    if (query.fields.shifts) {
      out.pairs.shift.insert(out.pairs.shift.end(), table.shift.begin(), table.shift.end());
    }
    if (query.fields.vectors) {
      out.pairs.vector.insert(out.pairs.vector.end(), table.vector.begin(), table.vector.end());
    }
    if (query.fields.distances) {
      out.pairs.distance.insert(out.pairs.distance.end(), table.distance.begin(),
                                table.distance.end());
    }
    origin += n;
  }
  out.pairs.stored = true;
  out.pairs.device = 0;
  batch = std::move(out);
}

} // namespace rgpot::nlist

namespace {

rgpot::nlist::Query query_from_c(const rgpot_nlist_request *request) {
  if (request == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "request is null");
  }
  const auto minimum =
      offsetof(rgpot_nlist_request, want_distances) + sizeof(std::int32_t);
  if (request->bytes < minimum || request->abi != RGPOT_NLIST_ABI) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Abi,
                              "request abi or bytes does not match this library");
  }
  rgpot_nlist_request taken{};
  std::memcpy(&taken, request,
              std::min(static_cast<std::size_t>(request->bytes), sizeof(taken)));
  if (taken.shape != 0 && taken.shape != 1) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Shape,
                              "shape must be 0 (pairs) or 1 (cluster of four)");
  }
  if (taken.shell != 0 && taken.shell != 1) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Shape, "shell must be 0 (half) or 1 (full)");
  }
  if (taken.shape == 1 && taken.shell != 1) {
    throw rgpot::nlist::Error(
        rgpot::nlist::Code::Shape,
        "a cluster list stores every neighbour; set shell to 1 (full)");
  }
  if (taken.cut != 0 && taken.cut != 1) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Shape,
                              "cut must be 0 (strict) or 1 (loose)");
  }
  rgpot::nlist::Query query;
  query.cutoff = taken.cutoff;
  query.skin = taken.skin;
  if (taken.shape == 1) {
    query.shape = rgpot::nlist::Cluster4{};
  } else {
    query.shape = rgpot::nlist::Pairs{taken.shell == 1 ? rgpot::nlist::Shell::Full
                                                       : rgpot::nlist::Shell::Half};
  }
  query.cut = taken.cut == 0 ? rgpot::nlist::Cut::Strict : rgpot::nlist::Cut::Loose;
  query.periodic = {taken.periodic[0] != 0, taken.periodic[1] != 0, taken.periodic[2] != 0};
  query.fields.shifts = taken.want_shifts != 0;
  query.fields.vectors = taken.want_vectors != 0;
  query.fields.distances = taken.want_distances != 0;
  return query;
}

int32_t code_of(const rgpot::nlist::Error &err, size_t *count) {
  if (count != nullptr) {
    *count = 0;
  }
  return static_cast<int32_t>(err.code());
}

rgpot_nlist_soa load_soa(const rgpot_nlist_soa *out) {
  if (out == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "neighbour buffers are null");
  }
  const auto minimum = offsetof(rgpot_nlist_soa, cap) + sizeof(size_t);
  if (out->bytes < minimum) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Abi,
                              "neighbour buffer bytes do not cover this library's struct");
  }
  rgpot_nlist_soa taken{};
  std::memcpy(&taken, out, std::min(static_cast<std::size_t>(out->bytes), sizeof(taken)));
  return taken;
}

void require_columns(const rgpot::nlist::Query &query, const rgpot_nlist_soa &out) {
  if (out.cap == 0) {
    return;
  }
  if (out.first == nullptr || out.second == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "first and second buffers are null");
  }
  if (query.fields.shifts && out.shift == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "shift buffer is null");
  }
  if (query.fields.vectors && out.vector == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "vector buffer is null");
  }
  if (query.fields.distances && out.distance == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "distance buffer is null");
  }
  if (std::holds_alternative<rgpot::nlist::Cluster4>(query.shape) && out.mask == nullptr) {
    throw rgpot::nlist::Error(rgpot::nlist::Code::Null, "a cluster list needs a mask buffer");
  }
}

rgpot::nlist::Buffers buffers_from(const rgpot_nlist_soa &out) {
  rgpot::nlist::Buffers dest;
  if (out.cap == 0) {
    return dest;
  }
  dest.first = std::span<std::int32_t>(out.first, out.cap);
  dest.second = std::span<std::int32_t>(out.second, out.cap);
  if (out.shift != nullptr) {
    dest.shift = std::span<std::int32_t>(out.shift, out.cap * 3);
  }
  if (out.vector != nullptr) {
    dest.vector = std::span<double>(out.vector, out.cap * 3);
  }
  if (out.distance != nullptr) {
    dest.distance = std::span<double>(out.distance, out.cap);
  }
  if (out.mask != nullptr) {
    dest.mask = std::span<std::uint8_t>(out.mask, out.cap);
  }
  return dest;
}

} // namespace

extern "C" {

int32_t rgpot_nlist_abi_version(void) { return RGPOT_NLIST_ABI; }

const char *rgpot_nlist_strerror(int32_t code) {
  switch (code) {
  case RGPOT_NLIST_OK:
    return "ok";
  case RGPOT_NLIST_ERR_ABI:
    return "neighbour request abi or size does not match this library";
  case RGPOT_NLIST_NULL:
    return "a required neighbour pointer is null";
  case RGPOT_NLIST_CUTOFF:
    return "cutoff and skin must be finite and >= 0";
  case RGPOT_NLIST_BOX:
    return "box must be finite, and a periodic box must have positive volume";
  case RGPOT_NLIST_SHAPE:
    return "neighbour shape, shell or cut is not a supported value";
  case RGPOT_NLIST_SHORT:
    return "neighbour buffer is shorter than the list; needed holds the full count";
  case RGPOT_NLIST_NOMEM:
    return "neighbour list allocation failed";
  case RGPOT_NLIST_SKIN:
    return "a positive skin needs a loose cut; a strict cut stores no buffer";
  default:
    return "unknown neighbour error";
  }
}

void rgpot_nlist_request_init(rgpot_nlist_request *request) {
  if (request == nullptr) {
    return;
  }
  std::memset(request, 0, sizeof(*request));
  request->bytes = static_cast<uint32_t>(sizeof(*request));
  request->abi = RGPOT_NLIST_ABI;
  request->shape = 0;
  request->shell = 0;
  request->cut = 0;
  request->periodic[0] = 1;
  request->periodic[1] = 1;
  request->periodic[2] = 1;
  request->want_shifts = 1;
  request->want_vectors = 1;
  request->want_distances = 1;
}

int32_t rgpot_nlist_build(const double *positions, size_t n, const double *box9,
                          const rgpot_nlist_request *request, rgpot_nlist_soa *out,
                          size_t *count, size_t *needed) {
  try {
    const auto query = query_from_c(request);
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, positions, n, box9, query);
    if (needed != nullptr) {
      *needed = table.size();
    }
    if (out == nullptr) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    const rgpot_nlist_soa taken = load_soa(out);
    if (taken.cap == 0) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    require_columns(query, taken);
    const size_t nwrite = std::min(taken.cap, table.size());
    const auto view = table.view();
    for (size_t p = 0; p < nwrite; ++p) {
      taken.first[p] = view.first[p];
      taken.second[p] = view.second[p];
      if (taken.shift != nullptr && view.shift.size() >= (p + 1) * 3) {
        taken.shift[3 * p] = view.shift[3 * p];
        taken.shift[3 * p + 1] = view.shift[3 * p + 1];
        taken.shift[3 * p + 2] = view.shift[3 * p + 2];
      }
      if (taken.vector != nullptr && view.vector.size() >= (p + 1) * 3) {
        taken.vector[3 * p] = view.vector[3 * p];
        taken.vector[3 * p + 1] = view.vector[3 * p + 1];
        taken.vector[3 * p + 2] = view.vector[3 * p + 2];
      }
      if (taken.distance != nullptr && p < view.distance.size()) {
        taken.distance[p] = view.distance[p];
      }
      if (taken.mask != nullptr && p < view.mask.size()) {
        taken.mask[p] = view.mask[p];
      }
    }
    if (count != nullptr) {
      *count = nwrite;
    }
    if (nwrite < table.size()) {
      return RGPOT_NLIST_SHORT;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return code_of(err, count);
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

struct rgpot_nlist_walk {
  rgpot::nlist::Walk walk;
  rgpot::nlist::Query query;
  rgpot_nlist_walk(const double *positions, size_t n, const double *box9,
                   const rgpot::nlist::Query &q)
      : walk(positions, n, box9, q), query(q) {}
};

int32_t rgpot_nlist_walk_open(const double *positions, size_t n, const double *box9,
                              const rgpot_nlist_request *request,
                              rgpot_nlist_walk **walk) {
  if (walk == nullptr) {
    return RGPOT_NLIST_NULL;
  }
  *walk = nullptr;
  try {
    const auto query = query_from_c(request);
    *walk = new (std::nothrow) rgpot_nlist_walk(positions, n, box9, query);
    if (*walk == nullptr) {
      return RGPOT_NLIST_NOMEM;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return static_cast<int32_t>(err.code());
  } catch (const std::bad_alloc &) {
    return RGPOT_NLIST_NOMEM;
  }
}

int32_t rgpot_nlist_walk_pull(rgpot_nlist_walk *walk, rgpot_nlist_soa *out, size_t *count,
                              int32_t *done) {
  if (walk == nullptr || out == nullptr) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NULL;
  }
  try {
    const rgpot_nlist_soa taken = load_soa(out);
    require_columns(walk->query, taken);
    bool finished = false;
    const std::size_t nwrite = walk->walk.pull(buffers_from(taken), finished);
    if (count != nullptr) {
      *count = nwrite;
    }
    if (done != nullptr) {
      *done = finished ? 1 : 0;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return code_of(err, count);
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

void rgpot_nlist_walk_close(rgpot_nlist_walk *walk) { delete walk; }

int32_t rgpot_nlist_build_systems(const double *positions, const int32_t *counts,
                                  size_t nsystems, const double *box9,
                                  const rgpot_nlist_request *request, rgpot_nlist_soa *out,
                                  int32_t *system, size_t *count, size_t *needed) {
  try {
    const auto query = query_from_c(request);
    rgpot::nlist::Systems systems;
    systems.positions = positions;
    systems.count = counts;
    systems.systems = nsystems;
    systems.box9 = box9;
    rgpot::nlist::Batch batch;
    rgpot::nlist::build(batch, systems, query);
    if (needed != nullptr) {
      *needed = batch.pairs.size();
    }
    if (out == nullptr) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    const rgpot_nlist_soa taken = load_soa(out);
    if (taken.cap == 0) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    require_columns(query, taken);
    const size_t nwrite = std::min(taken.cap, batch.pairs.size());
    const auto view = batch.pairs.view();
    for (size_t p = 0; p < nwrite; ++p) {
      taken.first[p] = view.first[p];
      taken.second[p] = view.second[p];
      if (system != nullptr) {
        system[p] = batch.system[p];
      }
      if (taken.shift != nullptr && view.shift.size() >= (p + 1) * 3) {
        taken.shift[3 * p] = view.shift[3 * p];
        taken.shift[3 * p + 1] = view.shift[3 * p + 1];
        taken.shift[3 * p + 2] = view.shift[3 * p + 2];
      }
      if (taken.vector != nullptr && view.vector.size() >= (p + 1) * 3) {
        taken.vector[3 * p] = view.vector[3 * p];
        taken.vector[3 * p + 1] = view.vector[3 * p + 1];
        taken.vector[3 * p + 2] = view.vector[3 * p + 2];
      }
      if (taken.distance != nullptr && p < view.distance.size()) {
        taken.distance[p] = view.distance[p];
      }
      if (taken.mask != nullptr && p < view.mask.size()) {
        taken.mask[p] = view.mask[p];
      }
    }
    if (count != nullptr) {
      *count = nwrite;
    }
    if (nwrite < batch.pairs.size()) {
      return RGPOT_NLIST_SHORT;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return code_of(err, count);
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

struct rgpot_nlist_builder {
  rgpot::nlist::Builder builder;
};

int32_t rgpot_nlist_builder_open(rgpot_nlist_builder **builder) {
  if (builder == nullptr) {
    return RGPOT_NLIST_NULL;
  }
  *builder = new (std::nothrow) rgpot_nlist_builder;
  if (*builder == nullptr) {
    return RGPOT_NLIST_NOMEM;
  }
  return RGPOT_NLIST_OK;
}

int32_t rgpot_nlist_builder_update(rgpot_nlist_builder *builder, const double *positions, size_t n,
                                   const double *box9, const rgpot_nlist_request *request,
                                   rgpot_nlist_soa *out, size_t *count, size_t *needed,
                                   int32_t *reused) {
  if (builder == nullptr) {
    return RGPOT_NLIST_NULL;
  }
  try {
    const auto query = query_from_c(request);
    builder->builder.update(positions, n, box9, query);
    if (reused != nullptr) {
      *reused = builder->builder.reused() ? 1 : 0;
    }
    const auto &table = builder->builder.table();
    if (needed != nullptr) {
      *needed = table.size();
    }
    if (out == nullptr) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    const rgpot_nlist_soa taken = load_soa(out);
    if (taken.cap == 0) {
      if (count != nullptr) {
        *count = 0;
      }
      return RGPOT_NLIST_OK;
    }
    require_columns(query, taken);
    const size_t nwrite = std::min(taken.cap, table.size());
    const auto view = table.view();
    for (size_t p = 0; p < nwrite; ++p) {
      taken.first[p] = view.first[p];
      taken.second[p] = view.second[p];
      if (taken.shift != nullptr && view.shift.size() >= (p + 1) * 3) {
        taken.shift[3 * p] = view.shift[3 * p];
        taken.shift[3 * p + 1] = view.shift[3 * p + 1];
        taken.shift[3 * p + 2] = view.shift[3 * p + 2];
      }
      if (taken.vector != nullptr && view.vector.size() >= (p + 1) * 3) {
        taken.vector[3 * p] = view.vector[3 * p];
        taken.vector[3 * p + 1] = view.vector[3 * p + 1];
        taken.vector[3 * p + 2] = view.vector[3 * p + 2];
      }
      if (taken.distance != nullptr && p < view.distance.size()) {
        taken.distance[p] = view.distance[p];
      }
      if (taken.mask != nullptr && p < view.mask.size()) {
        taken.mask[p] = view.mask[p];
      }
    }
    if (count != nullptr) {
      *count = nwrite;
    }
    if (nwrite < table.size()) {
      return RGPOT_NLIST_SHORT;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    return code_of(err, count);
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

void rgpot_nlist_builder_close(rgpot_nlist_builder *builder) { delete builder; }

struct rgpot_nlist_fortran_list {
  rgpot::nlist::Builder builder;
};

rgpot_nlist_fortran_list *rgpot_nlist_fortran_open(void) {
  return new (std::nothrow) rgpot_nlist_fortran_list;
}

void rgpot_nlist_fortran_close(rgpot_nlist_fortran_list *list) { delete list; }

int32_t rgpot_nlist_fortran_build(rgpot_nlist_fortran_list *list, const double *positions,
                                  size_t natoms, const double *box9, double cutoff, double skin,
                                  const int32_t periodic[3], int32_t *first, int32_t *second,
                                  double *vector, double *distance, size_t cap, size_t *count,
                                  int32_t *reused) {
  if (list == nullptr || positions == nullptr || box9 == nullptr || periodic == nullptr) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NULL;
  }
  try {
    rgpot::nlist::Query query;
    query.cutoff = cutoff;
    query.skin = skin;
    query.cut = skin > 0.0 ? rgpot::nlist::Cut::Loose : rgpot::nlist::Cut::Strict;
    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
    query.fields = {true, true, true};
    for (int k = 0; k < 3; ++k) {
      query.periodic[static_cast<std::size_t>(k)] = periodic[k] != 0;
    }
    list->builder.update(positions, natoms, box9, query);
    if (reused != nullptr) {
      *reused = list->builder.reused() ? 1 : 0;
    }
    const auto &table = list->builder.table();
    const double cut2 = cutoff * cutoff;
    std::size_t kept = 0;
    for (std::size_t p = 0; p < table.size(); ++p) {
      const auto i = static_cast<std::size_t>(table.first[p]);
      const auto j = static_cast<std::size_t>(table.second[p]);
      const int sa = table.shift[3 * p];
      const int sb = table.shift[3 * p + 1];
      const int sc = table.shift[3 * p + 2];
      const double vx = positions[3 * j] - positions[3 * i] + sa * box9[0] + sb * box9[3] +
                        sc * box9[6];
      const double vy = positions[3 * j + 1] - positions[3 * i + 1] + sa * box9[1] +
                        sb * box9[4] + sc * box9[7];
      const double vz = positions[3 * j + 2] - positions[3 * i + 2] + sa * box9[2] +
                        sb * box9[5] + sc * box9[8];
      const double r2 = vx * vx + vy * vy + vz * vz;
      if (!(r2 > 0.0) || !(r2 < cut2)) {
        continue;
      }
      if (first != nullptr && kept < cap) {
        first[kept] = table.first[p];
        second[kept] = table.second[p];
        vector[3 * kept] = vx;
        vector[3 * kept + 1] = vy;
        vector[3 * kept + 2] = vz;
        distance[kept] = std::sqrt(r2);
      }
      ++kept;
    }
    if (count != nullptr) {
      *count = kept;
    }
    if (first == nullptr) {
      return RGPOT_NLIST_OK;
    }
    if (kept > cap) {
      return RGPOT_NLIST_SHORT;
    }
    return RGPOT_NLIST_OK;
  } catch (const rgpot::nlist::Error &err) {
    if (count != nullptr) {
      *count = 0;
    }
    return static_cast<int32_t>(err.code());
  } catch (const std::bad_alloc &) {
    if (count != nullptr) {
      *count = 0;
    }
    return RGPOT_NLIST_NOMEM;
  }
}

} // extern "C"
