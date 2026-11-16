#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// One neighbour layer for the classical potentials: cell binning, the
// minimum image, an optional Verlet list, a streaming walk that does not
// store pairs, and a deterministic OpenMP split.

#include "rgpot/nlist/MinimumImage.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot::nlist {

/// How the minimum image is formed. Round is the orthorhombic truncation
/// used by the pair kernels. Half is one correction about the half-box,
/// used after coordinates have been wrapped into the cell. General is
/// MinimumImage, for a non-orthogonal cell or a periodic side shorter
/// than the search.
enum class Image : std::uint8_t { Round, Half, General };

/// Stored keeps a Verlet list of candidate pairs. Stream walks the cells
/// and stores nothing beyond the atoms and the bins. Auto stores when the
/// candidate index array fits in the memory the OS reports as available,
/// and streams otherwise.
enum class Mode : std::uint8_t { Auto, Stored, Stream };

struct Spec {
  double cutoff = 0;
  double skin = 0;
  std::array<bool, 3> periodic{{true, true, true}};
  Image image = Image::Round;
  Mode mode = Mode::Auto;
};

struct Range {
  std::int64_t begin = 0;
  std::int64_t end = 0;
};

/// Thread count from OMP_NUM_THREADS. Unset, empty, or a non-positive
/// value is one thread, so a call that does not ask for threads matches
/// the serial pair order.
[[nodiscard]] int threadsRequested();

/// Contiguous slice of `n` items for `thread` in `[0, nthreads)`. The
/// slices cover `n` without overlap, in thread order.
[[nodiscard]] Range partition(std::int64_t n, int thread, int nthreads);

/// Bytes the process may still allocate, from MemAvailable. Zero when
/// that figure is not reported.
[[nodiscard]] std::uint64_t availableBytes();

/// True when every atom has moved by at most half of `skin` from
/// `reference`. Skin 0 is true only when the coordinates match. The same
/// test decides `Builder::update`'s reused flag.
[[nodiscard]] inline bool withinSkin(const double *reference, const double *current,
                                     std::size_t n, double skin) noexcept {
  const double limit2 = 0.25 * skin * skin;
  for (std::size_t a = 0; a < n; ++a) {
    const double dx = current[3 * a] - reference[3 * a];
    const double dy = current[3 * a + 1] - reference[3 * a + 1];
    const double dz = current[3 * a + 2] - reference[3 * a + 2];
    if (dx * dx + dy * dy + dz * dz > limit2) {
      return false;
    }
  }
  return true;
}

/// Candidate pairs around one configuration.
///
/// A stored list is reused while every atom has moved by at most half
/// the skin, and the cell, the count and the spec are unchanged. The
/// list holds partner indices only and every walk takes the image from
/// the current coordinates, so the move is measured to the nearest
/// periodic image of the reference position: an atom wrapped back
/// through a face of the cell has moved by its small step, not by a
/// cell vector. The bins are redone after such a wrap, for the walks
/// that read them. Skin 0 reuses the list only when the coordinates and
/// the cell match.
///
/// An infinite cutoff or skin makes every pair a candidate. Such an
/// update bins nothing and stores nothing, in any mode: each walk takes
/// the pairs straight from the coordinates, i ascending and then j
/// ascending, which is the order of the single cell the bins would hold.
class Neighbours {
public:
  void update(const double *xyz, std::size_t n, const double *box9,
              const Spec &spec);

  [[nodiscard]] Mode mode() const noexcept { return mode_; }
  [[nodiscard]] bool streaming() const noexcept { return streaming_; }
  /// True when the search has no bound and the walks visit every pair.
  /// Such an update also reports streaming(). One atom has no pairs and
  /// keeps the empty stored list.
  [[nodiscard]] bool allPairs() const noexcept { return streaming_ && std::isinf(search_); }
  /// True when the last update kept the previous list. The displacement
  /// test is `withinSkin` on the nearest image of each reference position.
  [[nodiscard]] bool reused() const noexcept { return reused_; }
  /// Stored lists built so far. A reuse does not count.
  [[nodiscard]] std::uint64_t builds() const noexcept { return builds_; }
  /// Stored pair index. A stream leaves this empty.
  [[nodiscard]] std::size_t slots() const noexcept { return nbr_.size(); }
  [[nodiscard]] const std::vector<std::int32_t> &offsets() const noexcept {
    return off_;
  }
  [[nodiscard]] const std::vector<std::int32_t> &neighbours() const noexcept {
    return nbr_;
  }

  /// Directed pairs for atoms in `[begin, end)`. `dx` is `r_i - r_j`
  /// after the selected image. `r2` is the squared length of that vector.
  /// Candidates lie inside the cutoff plus the skin; the caller drops the
  /// rest. In stored mode the partners of each atom are in increasing
  /// index order.
  template <typename Fn>
  void forDirected(std::int64_t begin, std::int64_t end, Fn &&fn) const {
    if (allPairs()) {
      walkAll(begin, end, fn, false);
    } else if (streaming_) {
      walkCells(begin, end, fn, false);
    } else {
      walkStored(begin, end, fn, false);
    }
  }

  /// The pairs of `forDirected` with `j > i` only. Geometry for `j <= i`
  /// is not computed. A serial two-pass potential uses this so each
  /// unordered pair is visited once per pass.
  template <typename Fn>
  void forHalf(std::int64_t begin, std::int64_t end, Fn &&fn) const {
    if (allPairs()) {
      walkAll(begin, end, fn, true);
    } else if (streaming_) {
      walkCells(begin, end, fn, true);
    } else {
      walkStored(begin, end, fn, true);
    }
  }

  /// MSVC rejects `#pragma omp` inside a lambda (C2760, token PRAGMA_OMP).
  /// The region lives in this method; the lambdas only supply the body.
  template <typename Body>
  void withTeam(int nthreads, Body &&body) const {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nthreads)
#endif
    {
#if defined(_OPENMP)
      const int thread = omp_get_thread_num();
#else
      const int thread = 0;
#endif
      body(thread);
    }
  }

  /// One walk of the directed pairs, split by cell across `nthreads`.
  /// The bins are built once and shared. `fn(thread, i, j, dx, dy, dz, r2)`.
  /// A thread owns each cell, so a pair is visited once. Without bins,
  /// as in an unbounded search, a thread owns a contiguous run of atoms.
  template <typename Fn>
  void forCellTeam(bool half, int nthreads, Fn &&fn) const {
    if (nthreads < 1) {
      nthreads = 1;
    }
    if (!fastTiles([&](const auto &tile, const auto &neigh, const auto &nneigh,
                       const std::int32_t *head, std::size_t ncells,
                       bool cell_shift, double search2, double L0, double L1, double L2, double i0,
                       double i1, double i2, bool p0, bool p1, bool p2) {
          withTeam(nthreads, [&](int thread) {
            const auto range = partition(static_cast<std::int64_t>(ncells), thread, nthreads);
            auto accept = [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                              double r2) { fn(thread, i, j, dx, dy, dz, r2); };
            if (head != nullptr) {
              visitTiles<true>(tile, neigh, nneigh, head, static_cast<std::size_t>(range.begin),
                               static_cast<std::size_t>(range.end), 0, n_, half, cell_shift,
                               search2, L0, L1, L2, i0, i1, i2, p0, p1, p2, accept);
            } else {
              visitTiles<false>(tile, neigh, nneigh, head, static_cast<std::size_t>(range.begin),
                                static_cast<std::size_t>(range.end), 0, n_, half, cell_shift,
                                search2, L0, L1, L2, i0, i1, i2, p0, p1, p2, accept);
            }
          });
        })) {
      withTeam(nthreads, [&](int thread) {
        const auto range = partition(static_cast<std::int64_t>(n_), thread, nthreads);
        auto adapt = [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                         double r2) { fn(thread, i, j, dx, dy, dz, r2); };
        if (half) {
          forHalf(range.begin, range.end, adapt);
        } else {
          forDirected(range.begin, range.end, adapt);
        }
      });
    }
  }

  /// Neighbours of one atom from the bins already built. Does not allocate
  /// the tile stencil. A three-body stream uses this so the scratch is one
  /// atom, not every bond.
  template <typename Fn>
  void forAtom(std::int32_t i, Fn fn) const {
    if (allPairs()) {
      walkAll(static_cast<std::int64_t>(i), static_cast<std::int64_t>(i) + 1, fn, false);
      return;
    }
    // One atom, often from several threads. The cached stencil is built
    // once for a whole walk; this call does not touch it.
    walkCellsScattered(static_cast<std::int64_t>(i), static_cast<std::int64_t>(i) + 1, fn, false,
                       search_ * search_);
  }

private:
  struct SpecKey {
    double cutoff = 0;
    double skin = 0;
    Image image = Image::Round;
    Mode mode = Mode::Auto;
    std::array<bool, 3> periodic{{true, true, true}};
    bool operator==(const SpecKey &o) const {
      return cutoff == o.cutoff && skin == o.skin && image == o.image &&
             mode == o.mode && periodic == o.periodic;
    }
  };

  void buildCells(const double *xyz, std::size_t n);
  void buildStored(const double *xyz, std::size_t n);
  [[nodiscard]] std::uint64_t candidateBound() const;
  /// `wrapped` is set when some atom is within the skin only through
  /// another periodic image of its reference position.
  [[nodiscard]] bool reusable(const double *xyz, std::size_t n,
                              const double *box9, const Spec &spec,
                              bool &wrapped) const;
  void fold(double &dx, double &dy, double &dz) const;

  template <typename Fn>
  void walkStored(std::int64_t begin, std::int64_t end, Fn &fn,
                  bool half) const {
    for (std::int64_t i = begin; i < end; ++i) {
      const auto a = static_cast<std::size_t>(i);
      auto lo = off_[a];
      const auto hi = off_[a + 1];
      if (half) {
        // The partners are sorted, so those after i are a tail of the row.
        const auto row = nbr_.begin();
        lo = static_cast<std::int32_t>(
            std::upper_bound(row + lo, row + hi, static_cast<std::int32_t>(i)) -
            row);
      }
      for (std::int32_t p = lo; p < hi; ++p) {
        const auto j = nbr_[static_cast<std::size_t>(p)];
        double dx = 0;
        double dy = 0;
        double dz = 0;
        foldPair(static_cast<std::int32_t>(i), j, dx, dy, dz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        fn(static_cast<std::int32_t>(i), j, dx, dy, dz, r2);
      }
    }
  }

  /// Every pair of an unbounded search. A free axis has no image, so the
  /// separation is the coordinate difference that foldPair returns there.
  template <typename Fn>
  void walkAll(std::int64_t begin, std::int64_t end, Fn &fn, bool half) const {
    const auto n = static_cast<std::int64_t>(n_);
    const double search2 = search_ * search_;
    const double *x = xyz_;
    const bool no_image = !use_general_ && !periodic_[0] && !periodic_[1] && !periodic_[2];
    for (std::int64_t i = begin; i < end; ++i) {
      const auto a = static_cast<std::int32_t>(i);
      const double xi = x[3 * i];
      const double yi = x[3 * i + 1];
      const double zi = x[3 * i + 2];
      auto visit = [&](std::int64_t jj) {
        const auto j = static_cast<std::int32_t>(jj);
        double dx = 0;
        double dy = 0;
        double dz = 0;
        if (no_image) {
          dx = xi - x[3 * jj];
          dy = yi - x[3 * jj + 1];
          dz = zi - x[3 * jj + 2];
        } else {
          foldPair(a, j, dx, dy, dz);
        }
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= search2) {
          fn(a, j, dx, dy, dz, r2);
        }
      };
      if (!half) {
        for (std::int64_t jj = 0; jj < i; ++jj) {
          visit(jj);
        }
      }
      for (std::int64_t jj = i + 1; jj < n; ++jj) {
        visit(jj);
      }
    }
  }

  /// True when one materialised 27-slot stencil per cell is no larger than
  /// the pair index it stands in for. A short cutoff makes one bin per atom
  /// and that stencil no longer fits; the same cache then keeps only the
  /// cells that hold a pair inside the search.
  [[nodiscard]] bool tileStencilFits() const {
    const auto ncells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) *
                        static_cast<std::size_t>(nz_);
    const double per_cell =
        static_cast<double>(n_) / static_cast<double>(ncells > 0 ? ncells : 1);
    const double list_bytes = per_cell * 27.0 * static_cast<double>(n_) * 4.0;
    const double tile_bytes =
        static_cast<double>(ncells) * 27.0 * 32.0 + static_cast<double>(n_) * 32.0;
    return !(tile_bytes > list_bytes);
  }

  template <typename Loop>
  bool fastTiles(Loop &&loop) const {
    // Half is the same orthorhombic minimum image once coordinates sit in
    // the cell. A sheared box stays on the scattered walk. An unbounded
    // search has no bins to tile.
    if (allPairs() || use_general_ || (image_ != Image::Round && image_ != Image::Half) ||
        n_ < 2) {
      return false;
    }
    // One stencil. A 27-slot entry per cell is reused while the coordinates
    // are unchanged. When that array is larger than the pair index, the same
    // cache keeps only the cells that hold a pair inside the search. A stored
    // list does not take the compact form; it walks the index.
    const bool dense = tileStencilFits();
    if (!dense && !streaming_) {
      return false;
    }
    const auto ncells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) *
                        static_cast<std::size_t>(nz_);
    const bool cell_shift = (!periodic_[0] || nx_ >= 5) && (!periodic_[1] || ny_ >= 5) &&
                            (!periodic_[2] || nz_ >= 5);
    struct Nbor {
      int id;
      double sx;
      double sy;
      double sz;
    };
    struct Tile {
      double x;
      double y;
      double z;
      std::int32_t id;
    };
    // The stencil depends only on the coordinates. A warm call with
    // the same positions reuses it.
    struct Cache {
      const Neighbours *owner = nullptr;
      int nx = 0;
      int ny = 0;
      int nz = 0;
      double search = 0;
      std::vector<double> xyz;
      std::vector<Nbor> neigh;
      std::vector<int> nneigh;
      std::vector<Tile> tile;
      bool compact = false;
      std::vector<std::int32_t> head;
    };
    // One stencil per thread. The cache is process-wide otherwise, and two
    // evaluations then rewrite the same vectors.
    thread_local Cache cache;
    // The stored evaluation rebuilds its stencil. A stream keeps one and
    // reuses it when the coordinates are unchanged.
    const bool same_grid = cache.owner == this && cache.nx == nx_ && cache.ny == ny_ &&
                           cache.nz == nz_ && cache.search == search_ && cache.compact == !dense;
    // update() already compared the coordinates when it reused the bins.
    const bool reuse =
        streaming_ && same_grid &&
        (reused_ || (cache.xyz.size() == n_ * 3 &&
                     std::memcmp(cache.xyz.data(), xyz_, n_ * 3 * sizeof(double)) == 0));
    if (!reuse) {
    if (!dense) {
      cache.compact = true;
      cache.neigh.clear();
      cache.nneigh.assign(ncells, 0);
      cache.head.assign(ncells + 1, 0);
      cache.tile.clear();
      const double search2 = search_ * search_;
      const auto cap = ncells * 2;
      bool overflow = ncells < 1 || ncells > static_cast<std::size_t>(0x7fffffff) / 2;
      for (std::size_t c = 0; c < ncells && !overflow; ++c) {
        cache.head[c] = static_cast<std::int32_t>(cache.neigh.size());
        if (cell_start_[c] == cell_start_[c + 1]) {
          continue;
        }
        const int cell = static_cast<int>(c);
        const int iz = cell % nz_;
        const int iy = (cell / nz_) % ny_;
        const int ix = cell / (nz_ * ny_);
        int seen[27];
        int nseen = 0;
        for (int ox = -1; ox <= 1 && !overflow; ++ox) {
          for (int oy = -1; oy <= 1 && !overflow; ++oy) {
            for (int oz = -1; oz <= 1; ++oz) {
              int jx = ix + ox;
              int jy = iy + oy;
              int jz = iz + oz;
              int sa = 0;
              int sb = 0;
              int sc = 0;
              if (periodic_[0]) {
                if (jx < 0) {
                  jx += nx_;
                  sa = -1;
                } else if (jx >= nx_) {
                  jx -= nx_;
                  sa = 1;
                }
              } else if (jx < 0 || jx >= nx_) {
                continue;
              }
              if (periodic_[1]) {
                if (jy < 0) {
                  jy += ny_;
                  sb = -1;
                } else if (jy >= ny_) {
                  jy -= ny_;
                  sb = 1;
                }
              } else if (jy < 0 || jy >= ny_) {
                continue;
              }
              if (periodic_[2]) {
                if (jz < 0) {
                  jz += nz_;
                  sc = -1;
                } else if (jz >= nz_) {
                  jz -= nz_;
                  sc = 1;
                }
              } else if (jz < 0 || jz >= nz_) {
                continue;
              }
              const int id = (jx * ny_ + jy) * nz_ + jz;
              bool dup = false;
              for (int s = 0; s < nseen; ++s) {
                if (seen[s] == id) {
                  dup = true;
                  break;
                }
              }
              if (dup) {
                continue;
              }
              seen[nseen++] = id;
              const auto nb = static_cast<std::size_t>(id);
              if (cell_start_[nb] == cell_start_[nb + 1]) {
                continue;
              }
              bool hit = false;
              for (std::int32_t q = cell_start_[c]; q < cell_start_[c + 1] && !hit; ++q) {
                const auto i = cell_atoms_[static_cast<std::size_t>(q)];
                for (std::int32_t r = cell_start_[nb]; r < cell_start_[nb + 1]; ++r) {
                  const auto j = cell_atoms_[static_cast<std::size_t>(r)];
                  if (j == i) {
                    continue;
                  }
                  double dx = 0;
                  double dy = 0;
                  double dz = 0;
                  foldPair(i, j, dx, dy, dz);
                  if (dx * dx + dy * dy + dz * dz <= search2) {
                    hit = true;
                    break;
                  }
                }
              }
              if (!hit) {
                continue;
              }
              if (cache.neigh.size() >= cap) {
                overflow = true;
                break;
              }
              cache.neigh.push_back(Nbor{id, sa * len_[0], sb * len_[1], sc * len_[2]});
              ++cache.nneigh[c];
            }
          }
        }
      }
      if (!overflow) {
        cache.head[ncells] = static_cast<std::int32_t>(cache.neigh.size());
        const double per_cell = static_cast<double>(n_) / static_cast<double>(ncells);
        const double list_bytes = per_cell * 27.0 * static_cast<double>(n_) * 4.0;
        const double csr_bytes =
            static_cast<double>(cache.neigh.size()) * 32.0 +
            static_cast<double>(ncells + 1) * static_cast<double>(sizeof(std::int32_t));
        if (csr_bytes > list_bytes) {
          overflow = true;
        }
      }
      if (overflow) {
        cache.owner = nullptr;
        cache.neigh.clear();
        cache.head.clear();
        cache.nneigh.clear();
        return false;
      }
      if (!cache.neigh.empty()) {
        cache.tile.resize(n_);
        for (std::size_t q = 0; q < n_; ++q) {
          const auto j = cell_atoms_[q];
          const auto u = static_cast<std::size_t>(j) * 3;
          cache.tile[q] = Tile{xyz_[u], xyz_[u + 1], xyz_[u + 2], j};
        }
      }
    } else {
    cache.compact = false;
    cache.head.clear();
    auto &neigh = cache.neigh;
    auto &nneigh = cache.nneigh;
    neigh.assign(ncells * 27, Nbor{});
    nneigh.assign(ncells, 0);
    for (std::size_t c = 0; c < ncells; ++c) {
      const int cell = static_cast<int>(c);
      int iz = cell % nz_;
      int iy = (cell / nz_) % ny_;
      int ix = cell / (nz_ * ny_);
      int seen[27];
      int nseen = 0;
      for (int ox = -1; ox <= 1; ++ox) {
        for (int oy = -1; oy <= 1; ++oy) {
          for (int oz = -1; oz <= 1; ++oz) {
            int jx = ix + ox;
            int jy = iy + oy;
            int jz = iz + oz;
            int sa = 0;
            int sb = 0;
            int sc = 0;
            if (periodic_[0]) {
              if (jx < 0) {
                jx += nx_;
                sa = -1;
              } else if (jx >= nx_) {
                jx -= nx_;
                sa = 1;
              }
            } else if (jx < 0 || jx >= nx_) {
              continue;
            }
            if (periodic_[1]) {
              if (jy < 0) {
                jy += ny_;
                sb = -1;
              } else if (jy >= ny_) {
                jy -= ny_;
                sb = 1;
              }
            } else if (jy < 0 || jy >= ny_) {
              continue;
            }
            if (periodic_[2]) {
              if (jz < 0) {
                jz += nz_;
                sc = -1;
              } else if (jz >= nz_) {
                jz -= nz_;
                sc = 1;
              }
            } else if (jz < 0 || jz >= nz_) {
              continue;
            }
            const int id = (jx * ny_ + jy) * nz_ + jz;
            bool dup = false;
            for (int s = 0; s < nseen; ++s) {
              if (seen[s] == id) {
                dup = true;
                break;
              }
            }
            if (dup) {
              continue;
            }
            seen[nseen] = id;
            neigh[c * 27 + static_cast<std::size_t>(nseen)] =
                Nbor{id, sa * len_[0], sb * len_[1], sc * len_[2]};
            ++nseen;
          }
        }
      }
      nneigh[c] = nseen;
    }
    auto &tile = cache.tile;
    tile.resize(n_);
    for (std::size_t q = 0; q < n_; ++q) {
      const auto j = cell_atoms_[q];
      const auto u = static_cast<std::size_t>(j) * 3;
      tile[q] = Tile{xyz_[u], xyz_[u + 1], xyz_[u + 2], j};
    }
    }
    if (streaming_) {
      cache.owner = this;
      cache.nx = nx_;
      cache.ny = ny_;
      cache.nz = nz_;
      cache.search = search_;
      cache.xyz.assign(xyz_, xyz_ + n_ * 3);
    }
    }
    if (cache.compact && cache.neigh.empty()) {
      return true;
    }
    loop(cache.tile, cache.neigh, cache.nneigh, cache.compact ? cache.head.data() : nullptr, ncells,
         cell_shift, search_ * search_, len_[0], len_[1], len_[2], inv_[0], inv_[1], inv_[2],
         periodic_[0], periodic_[1], periodic_[2]);
    return true;
  }

  template <bool Compact, typename Tile, typename Nbor, typename Fn>
  void visitTiles(const std::vector<Tile> &tile, const std::vector<Nbor> &neigh,
                  const std::vector<int> &nneigh, const std::int32_t *head, std::size_t c0,
                  std::size_t c1,
                  std::int64_t atom0, std::size_t atom1, bool half, bool cell_shift,
                  double search2, double L0, double L1, double L2, double i0, double i1,
                  double i2, bool p0, bool p1, bool p2, Fn &&fn) const {
    auto round_axis = [](double delta, double inv, double length, bool periodic, int &shift) {
      if (!periodic) {
        shift = 0;
        return delta;
      }
      const double t = delta * inv;
      if (!(t > -1.0e15 && t < 1.0e15)) {
        shift = 0;
        return delta;
      }
      const double shifted = t + std::copysign(0.5, t);
      const auto image = static_cast<long long>(shifted);
      shift = static_cast<int>(image);
      return delta - length * static_cast<double>(image);
    };
    for (std::size_t c = c0; c < c1; ++c) {
      const auto a0 = cell_start_[c];
      const auto a1 = cell_start_[c + 1];
      if (a0 == a1) {
        continue;
      }
      const int nn = nneigh[c];
      if constexpr (Compact) {
        if (nn == 0) {
          continue;
        }
      }
      const Nbor *nb;
      if constexpr (Compact) {
        nb = neigh.data() + static_cast<std::size_t>(head[c]);
      } else {
        nb = neigh.data() + c * 27;
      }
      for (std::int32_t q = a0; q < a1; ++q) {
        const Tile &a = tile[static_cast<std::size_t>(q)];
        if (a.id < atom0 || static_cast<std::size_t>(a.id) >= atom1) {
          continue;
        }
        for (int k = 0; k < nn; ++k) {
          const Nbor &src = nb[k];
          const auto b0 = static_cast<std::size_t>(cell_start_[static_cast<std::size_t>(src.id)]);
          const auto b1 =
              static_cast<std::size_t>(cell_start_[static_cast<std::size_t>(src.id) + 1]);
          // Atom ids increase inside a cell, so a half pair starts at the
          // first neighbour with a larger id and does not test the rest.
          const std::size_t rstart = [&]() {
            if (!half) {
              return b0;
            }
            const auto begin = tile.begin() + static_cast<std::ptrdiff_t>(b0);
            const auto end = tile.begin() + static_cast<std::ptrdiff_t>(b1);
            const auto found = std::lower_bound(
                begin, end, a.id, [](const Tile &atom, std::int32_t id) { return atom.id < id; });
            auto next = found;
            if (next != end && next->id == a.id) {
              ++next;
            }
            return static_cast<std::size_t>(next - tile.begin());
          }();
          for (std::size_t r = rstart; r < b1; ++r) {
            const Tile &b = tile[r];
            if (!half && b.id == a.id) {
              continue;
            }
            double dx;
            double dy;
            double dz;
            if (cell_shift) {
              dx = (a.x - b.x) - src.sx;
              dy = (a.y - b.y) - src.sy;
              dz = (a.z - b.z) - src.sz;
            } else {
              int sa = 0;
              int sb = 0;
              int sc = 0;
              dx = round_axis(a.x - b.x, i0, L0, p0, sa);
              dy = round_axis(a.y - b.y, i1, L1, p1, sb);
              dz = round_axis(a.z - b.z, i2, L2, p2, sc);
            }
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= search2) {
              fn(a.id, b.id, dx, dy, dz, r2);
            }
          }
        }
      }
    }
  }

  template <typename Fn>
  void walkCells(std::int64_t begin, std::int64_t end, Fn &fn, bool half) const {
    const double search2 = search_ * search_;
    if (!fastTiles([&](const auto &tile, const auto &neigh, const auto &nneigh,
                       const std::int32_t *head, std::size_t ncells, bool cell_shift,
                       double search2_fast, double L0, double L1, double L2, double i0, double i1,
                       double i2, bool p0, bool p1, bool p2) {
          if (head != nullptr) {
            visitTiles<true>(tile, neigh, nneigh, head, 0, ncells, begin,
                             static_cast<std::size_t>(end), half, cell_shift, search2_fast, L0, L1,
                             L2, i0, i1, i2, p0, p1, p2, fn);
          } else {
            visitTiles<false>(tile, neigh, nneigh, head, 0, ncells, begin,
                              static_cast<std::size_t>(end), half, cell_shift, search2_fast, L0, L1,
                              L2, i0, i1, i2, p0, p1, p2, fn);
          }
        })) {
      walkCellsScattered(begin, end, fn, half, search2);
    }
  }

  template <typename Fn>
  void walkCellsScattered(std::int64_t begin, std::int64_t end, Fn &fn, bool half,
                          double search2) const {
    for (std::int64_t i = begin; i < end; ++i) {
      const auto a = static_cast<std::size_t>(i);
      const int cell = cell_of_[a];
      int iz = cell % nz_;
      int iy = (cell / nz_) % ny_;
      int ix = cell / (nz_ * ny_);
      int seen[27];
      int nseen = 0;
      for (int ox = -1; ox <= 1; ++ox) {
        for (int oy = -1; oy <= 1; ++oy) {
          for (int oz = -1; oz <= 1; ++oz) {
            const int id = cellId(ix + ox, iy + oy, iz + oz);
            if (id < 0) {
              continue;
            }
            bool dup = false;
            for (int s = 0; s < nseen; ++s) {
              if (seen[s] == id) {
                dup = true;
                break;
              }
            }
            if (dup) {
              continue;
            }
            seen[nseen++] = id;
            const auto c = static_cast<std::size_t>(id);
            const auto a0 = cell_start_[c];
            const auto a1 = cell_start_[c + 1];
            for (std::int32_t q = a0; q < a1; ++q) {
              const auto j = cell_atoms_[static_cast<std::size_t>(q)];
              if (j == static_cast<std::int32_t>(i)) {
                continue;
              }
              if (half && !(j > static_cast<std::int32_t>(i))) {
                continue;
              }
              double dx = 0;
              double dy = 0;
              double dz = 0;
              if (!foldPairWithin(static_cast<std::int32_t>(i), j, search2, dx,
                                  dy, dz)) {
                continue;
              }
              const double r2 = dx * dx + dy * dy + dz * dz;
              if (r2 <= search2) {
                fn(static_cast<std::int32_t>(i), j, dx, dy, dz, r2);
              }
            }
          }
        }
      }
    }
  }

  void foldPair(std::int32_t i, std::int32_t j, double &dx, double &dy,
                double &dz) const;
  /// foldPair for a walk that drops pairs beyond `within2`. False means
  /// the pair has no image that close; true leaves the foldPair bits.
  [[nodiscard]] bool foldPairWithin(std::int32_t i, std::int32_t j,
                                    double within2, double &dx, double &dy,
                                    double &dz) const;
  [[nodiscard]] int cellId(int ix, int iy, int iz) const;

  std::vector<std::int32_t> off_;
  std::vector<std::int32_t> nbr_;
  std::vector<std::int32_t> cell_of_;
  std::vector<std::int32_t> cell_start_;
  std::vector<std::int32_t> cell_atoms_;
  std::vector<double> ref_;
  double box_[9]{};
  double len_[3]{};
  double inv_[3]{};
  bool periodic_[3]{};
  int nx_ = 1;
  int ny_ = 1;
  int nz_ = 1;
  double search_ = 0;
  Image image_ = Image::Round;
  Mode mode_ = Mode::Auto;
  SpecKey key_{};
  MinimumImage general_;
  bool use_general_ = false;
  bool streaming_ = true;
  bool built_ = false;
  bool reused_ = false;
  std::size_t n_ = 0;
  std::uint64_t builds_ = 0;
  const double *xyz_ = nullptr;
};

inline void Neighbours::fold(double &dx, double &dy, double &dz) const {
  if (use_general_) {
    general_.fold(dx, dy, dz);
    return;
  }
  auto round_visit = [](double t) {
    if (!std::isfinite(t)) {
      return 0.0;
    }
    const double shifted = t + std::copysign(0.5, t);
    if (!std::isfinite(shifted) || shifted >= 9223372036854775808.0 ||
        shifted <= -9223372036854775808.0) {
      return 0.0;
    }
    return static_cast<double>(static_cast<long long>(shifted));
  };
  if (image_ == Image::Half) {
    const double h0 = 0.5 * len_[0];
    const double h1 = 0.5 * len_[1];
    const double h2 = 0.5 * len_[2];
    if (periodic_[0]) {
      if (dx > h0) {
        dx -= len_[0];
      } else if (dx < -h0) {
        dx += len_[0];
      }
    }
    if (periodic_[1]) {
      if (dy > h1) {
        dy -= len_[1];
      } else if (dy < -h1) {
        dy += len_[1];
      }
    }
    if (periodic_[2]) {
      if (dz > h2) {
        dz -= len_[2];
      } else if (dz < -h2) {
        dz += len_[2];
      }
    }
    return;
  }
  if (periodic_[0]) {
    dx -= len_[0] * round_visit(dx * inv_[0]);
  }
  if (periodic_[1]) {
    dy -= len_[1] * round_visit(dy * inv_[1]);
  }
  if (periodic_[2]) {
    dz -= len_[2] * round_visit(dz * inv_[2]);
  }
}

inline void Neighbours::foldPair(std::int32_t i, std::int32_t j, double &dx, double &dy,
                                 double &dz) const {
  const auto ui = static_cast<std::size_t>(i) * 3;
  const auto uj = static_cast<std::size_t>(j) * 3;
  dx = xyz_[ui] - xyz_[uj];
  dy = xyz_[ui + 1] - xyz_[uj + 1];
  dz = xyz_[ui + 2] - xyz_[uj + 2];
  fold(dx, dy, dz);
}

inline bool Neighbours::foldPairWithin(std::int32_t i, std::int32_t j,
                                       double within2, double &dx, double &dy,
                                       double &dz) const {
  if (!use_general_) {
    foldPair(i, j, dx, dy, dz);
    return true;
  }
  const auto ui = static_cast<std::size_t>(i) * 3;
  const auto uj = static_cast<std::size_t>(j) * 3;
  dx = xyz_[ui] - xyz_[uj];
  dy = xyz_[ui + 1] - xyz_[uj + 1];
  dz = xyz_[ui + 2] - xyz_[uj + 2];
  return general_.foldWithin(dx, dy, dz, within2);
}

inline int Neighbours::cellId(int ix, int iy, int iz) const {
  if (periodic_[0]) {
    if (ix < 0) {
      ix += nx_;
    } else if (ix >= nx_) {
      ix -= nx_;
    }
  } else if (ix < 0 || ix >= nx_) {
    return -1;
  }
  if (periodic_[1]) {
    if (iy < 0) {
      iy += ny_;
    } else if (iy >= ny_) {
      iy -= ny_;
    }
  } else if (iy < 0 || iy >= ny_) {
    return -1;
  }
  if (periodic_[2]) {
    if (iz < 0) {
      iz += nz_;
    } else if (iz >= nz_) {
      iz -= nz_;
    }
  } else if (iz < 0 || iz >= nz_) {
    return -1;
  }
  return (ix * ny_ + iy) * nz_ + iz;
}

} // namespace rgpot::nlist
