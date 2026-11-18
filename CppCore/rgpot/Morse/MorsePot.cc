// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the Morse potential methods.
 *
 * Ported from eOn (https://github.com/TheochemUI/eOn,
 * client/potentials/Morse), BSD-3-Clause licensed, copyright the eOn
 * Development Team.
 */

// clang-format off
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
// clang-format on

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/Morse/detail/MorseSplitSnapshot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/simd/exp.hpp"
#include "rgpot/stress.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_MORSE_LAMBDA_INLINE __attribute__((always_inline))
#else
#define RGPOT_MORSE_LAMBDA_INLINE
#endif

namespace rgpot {

namespace {

/// Stored upper-pair count below which a call stays on one thread even when
/// in.nAtoms >= kTeamMinAtoms (dilute configurations where barrier latency
/// exceeds the pair work).
constexpr std::size_t kMinParallelHalfPairs = 4096;

/// Interior atom split indices in partitionAtomsByWork are rounded to the
/// nearest multiple of 8 atoms (bounded to [prev, n]). Non-empty interior
/// slices are therefore at least 8 atoms wide, without assuming that the
/// std::vector or caller output buffers are aligned to cache lines.
constexpr std::size_t kSliceAtomQuantum = 8;

/// Stack batch width for vectorised simd::exp_fixed_n evaluation.
constexpr std::size_t kPairBatch = 64;

struct alignas(64) PairBatch {
  double arg[kPairBatch];
  double ex[kPairBatch];
  double r[kPairBatch];
  double dx[kPairBatch];
  double dy[kPairBatch];
  double dz[kPairBatch];
  std::int32_t i[kPairBatch];
  std::int32_t j_or_u[kPairBatch];
  std::size_t count = 0;
};

struct PairForce {
  double fx;
  double fy;
  double fz;
};

struct MorseWorkspace {
  const nlist::Neighbours *owner = nullptr;
  std::uint64_t build_id = 0;
  std::size_t index_atoms = 0;
  std::size_t slots = 0;

  std::vector<std::int32_t> up_off;
  std::vector<std::int32_t> lo_off;
  std::vector<std::int32_t> lo_u;
  std::vector<std::int32_t> lo_count;

  std::vector<std::uint64_t> active_bits;
  std::vector<PairForce> pair_forces;
  std::vector<double> pair_he;
  std::vector<double> pair_hv;
  std::vector<double> row_e;
  std::vector<double> row_s;
  std::vector<std::size_t> p0_splits;
  std::vector<std::size_t> p1_splits;
  const nlist::Neighbours *split_owner = nullptr;
  std::uint64_t split_build_id = 0;
  std::size_t split_atoms = 0;
  int split_slices = 0;
  std::size_t split_total_up = 0;
  std::size_t split_total_inc = 0;

  const nlist::Neighbours *scratch_owner = nullptr;
  std::uint64_t scratch_build_id = 0;
  std::size_t scratch_slots = 0;
  std::uint64_t scratch_avail = 0;
};

struct WorkspaceLease {
  MorseWorkspace *ws = nullptr;
  std::unique_ptr<MorseWorkspace> fallback;
  bool *busy_flag = nullptr;

  WorkspaceLease() {
    thread_local MorseWorkspace tls_ws;
    thread_local bool tls_busy = false;
    if (!tls_busy) {
      tls_busy = true;
      busy_flag = &tls_busy;
      ws = &tls_ws;
    } else {
      fallback = std::make_unique<MorseWorkspace>();
      ws = fallback.get();
    }
  }

  ~WorkspaceLease() {
    if (busy_flag != nullptr) {
      *busy_flag = false;
    }
  }

  WorkspaceLease(const WorkspaceLease &) = delete;
  WorkspaceLease &operator=(const WorkspaceLease &) = delete;
};

void buildStoredHalfIndex(MorseWorkspace &ws, const nlist::Neighbours &cells, std::size_t n) {
  const auto &off = cells.offsets();
  const auto &nbr = cells.neighbours();
  ws.up_off.resize(n + 1);
  ws.lo_count.assign(n, 0);
  const auto row = nbr.begin();
  std::int32_t total_up = 0;
  for (std::size_t i = 0; i < n; ++i) {
    ws.up_off[i] = total_up;
    const auto self = static_cast<std::int32_t>(i);
    const auto lo = off[i];
    const auto hi = off[i + 1];
    const auto mid = static_cast<std::int32_t>(std::upper_bound(row + lo, row + hi, self) - row);
    total_up += (hi - mid);
    for (std::int32_t p = mid; p < hi; ++p) {
      const auto j = static_cast<std::size_t>(nbr[static_cast<std::size_t>(p)]);
      ++ws.lo_count[j];
    }
  }
  ws.up_off[n] = total_up;

  ws.lo_off.resize(n + 1);
  std::int32_t acc = 0;
  for (std::size_t i = 0; i < n; ++i) {
    ws.lo_off[i] = acc;
    acc += ws.lo_count[i];
    ws.lo_count[i] = ws.lo_off[i];
  }
  ws.lo_off[n] = acc;

  ws.lo_u.resize(static_cast<std::size_t>(total_up));
  for (std::size_t i = 0; i < n; ++i) {
    const auto hi = off[i + 1];
    const auto count_i = ws.up_off[i + 1] - ws.up_off[i];
    const auto mid = hi - count_i;
    std::int32_t u = ws.up_off[i];
    for (std::int32_t p = mid; p < hi; ++p) {
      const auto j = static_cast<std::size_t>(nbr[static_cast<std::size_t>(p)]);
      ws.lo_u[static_cast<std::size_t>(ws.lo_count[j]++)] = u++;
    }
  }

  ws.owner = &cells;
  ws.build_id = cells.builds();
  ws.index_atoms = n;
  ws.slots = cells.slots();
  ws.split_owner = nullptr;
  ws.split_build_id = 0;
  ws.split_atoms = 0;
  ws.split_slices = 0;
  ws.split_total_up = 0;
  ws.split_total_inc = 0;
}

template <typename WorkFn>
void partitionAtomsByWork(std::size_t n, int slices, std::int64_t total_work, WorkFn &&work_at,
                          std::vector<std::size_t> &splits) {
  splits.resize(static_cast<std::size_t>(slices) + 1);
  splits[0] = 0;
  for (int s = 1; s < slices; ++s) {
    const std::int64_t target =
        (total_work * static_cast<std::int64_t>(s)) / static_cast<std::int64_t>(slices);
    std::size_t lo = splits[static_cast<std::size_t>(s - 1)];
    std::size_t hi = n;
    while (lo < hi) {
      const std::size_t mid = lo + (hi - lo) / 2;
      if (work_at(mid) < target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    std::size_t rounded =
        ((lo + kSliceAtomQuantum / 2) / kSliceAtomQuantum) * kSliceAtomQuantum;
    const std::size_t prev = splits[static_cast<std::size_t>(s - 1)];
    if (rounded < prev) {
      rounded = prev;
    }
    if (rounded > n) {
      rounded = n;
    }
    splits[static_cast<std::size_t>(s)] = rounded;
  }
  splits[static_cast<std::size_t>(slices)] = n;
}

std::uint64_t cachedAvailableBytes(MorseWorkspace &ws, const nlist::Neighbours &cells) {
  if (ws.scratch_owner != &cells || ws.scratch_build_id != cells.builds() ||
      ws.scratch_slots != cells.slots()) {
    ws.scratch_owner = &cells;
    ws.scratch_build_id = cells.builds();
    ws.scratch_slots = cells.slots();
    ws.scratch_avail = nlist::availableBytes();
  }
  return ws.scratch_avail;
}

bool pairScratchFits(std::size_t num_pairs, const PerAtomOut *atoms, std::uint64_t avail) {
  std::size_t bytes_per_pair = sizeof(PairForce) + sizeof(std::int32_t) + 1;
  if (atoms != nullptr) {
    if (atoms->energy != nullptr) {
      bytes_per_pair += sizeof(double);
    }
    if (atoms->virial != nullptr) {
      bytes_per_pair += 6 * sizeof(double);
    }
  }
  if (avail == 0) {
    return true;
  }
  return static_cast<std::uint64_t>(num_pairs) <= (avail / 2) / bytes_per_pair;
}

} // namespace

/**
 * @class MorsePot
 * @details
 *
 * Pairwise interactions within the cutoff radius. The minimum image
 * comes from the shared neighbour layer, on an orthogonal or sheared
 * box. The pair term runs during the cell walk. A stream does not copy
 * a half-list. A stored list keeps candidates out to the cutoff plus a
 * skin of one fifth of the cutoff, and a later call inside half that
 * skin reuses them. The distance filter uses the positions of this call.
 *
 * With ``MorseConfig::switch_width`` set, each pair term is the unshifted
 * Morse energy times the C^2 quintic switch of PairSwitch.hpp; otherwise
 * the energy is shifted to zero at the cutoff and the force jumps there.
 * The pair term is ``simd::exp_fixed``, so a per-pair accumulation and this
 * row return the same bits. Cauchy stress is the derivative of that energy.
 *
 * @note Orthogonal and sheared cells are supported: a diagonal box wraps each
 * axis with a positive diagonal entry (``box[0]``, ``box[4]``, ``box[8]``) via
 * ``nlist::Image::Round`` and leaves non-positive axes free, while any
 * non-zero off-diagonal entry switches ``nlist::Neighbours`` to
 * ``nlist::MinimumImage`` across the full 3x3 cell (which must be finite and
 * non-singular). Each distinct atom pair contributes at most one minimum image,
 * and cell binning divides Cartesian coordinates by the diagonal lengths.
 */
void MorsePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void MorsePot::forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const {
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

inline void MorsePot::pairTerm(double r, double &energy, double &fscale) const {
  const double twoDeA = 2.0 * De * a;
  const double d = 1.0 - simd::detail::exp_fixed_lane(-a * (r - re));
  const double bare = De * d * d - De;
  if (m_config.switch_width != 0.0) {
    const auto s = m_switch(r);
    energy = bare * s.s;
    fscale = (twoDeA * d * (d - 1.0) * s.s - bare * s.dsdr) / r;
  } else {
    energy = bare - energyCutoff;
    fscale = twoDeA * d * (d - 1.0) / r;
  }
}

void MorsePot::compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const {
  const auto N = static_cast<long>(in.nAtoms);
  double *F = out->F;
  double *U = &out->energy;
  double strain[6] = {};
  const double volume = cellVolume(in.box);
  publishCauchyStress(out, strain, volume);
  *U = 0.0;
  if (N < 2 || cuttOffR <= 0.0) {
    for (long k = 0; k < 3 * N; ++k) {
      F[k] = 0.0;
    }
    return;
  }

  const auto n_atoms = static_cast<std::size_t>(N);
  updateNeighbours(in);
  const double cutoff2 = cuttOffR * cuttOffR;
  const double depth = De;
  const double range = a;
  const double req = re;
  const double shift = energyCutoff;
  const double twoDeA = 2.0 * depth * range;
  const bool switched = m_config.switch_width != 0.0;
  const QuinticSwitch sw_fn = m_switch;

  auto flushBatch = [=](PairBatch &batch, auto &&emit) RGPOT_MORSE_LAMBDA_INLINE {
    if (batch.count == 0) {
      return;
    }
    simd::exp_fixed_n(batch.arg, batch.ex, batch.count);
    for (std::size_t k = 0; k < batch.count; ++k) {
      const double r = batch.r[k];
      const double d = 1.0 - batch.ex[k];
      const double bare = depth * d * d - depth;
      double pair_e = 0.0;
      double fscale = 0.0;
      if (switched) {
        const auto s = sw_fn(r);
        pair_e = bare * s.s;
        fscale = (twoDeA * d * (d - 1.0) * s.s - bare * s.dsdr) / r;
      } else {
        pair_e = bare - shift;
        fscale = twoDeA * d * (d - 1.0) / r;
      }
      emit(k, pair_e, fscale);
    }
    batch.count = 0;
  };

  if (n_atoms < kTeamMinAtoms) {
    std::fill_n(F, 3 * n_atoms, 0.0);
    double energy = 0.0;
    PairBatch batch;
    auto emit = [&](std::size_t k, double pair_e, double fscale) RGPOT_MORSE_LAMBDA_INLINE {
      const auto ii = batch.i[k];
      const auto jj = batch.j_or_u[k];
      const double dx = batch.dx[k];
      const double dy = batch.dy[k];
      const double dz = batch.dz[k];
      energy += pair_e;
      const double fx = fscale * dx;
      const double fy = fscale * dy;
      const double fz = fscale * dz;
      F[3 * ii] += fx;
      F[3 * ii + 1] += fy;
      F[3 * ii + 2] += fz;
      F[3 * jj] -= fx;
      F[3 * jj + 1] -= fy;
      F[3 * jj + 2] -= fz;
      accumulatePairStrain(strain, fscale, dx, dy, dz);
      if (atoms != nullptr) {
        tallyPair(*atoms, static_cast<std::size_t>(ii), static_cast<std::size_t>(jj), pair_e,
                  fscale, dx, dy, dz);
      }
    };
    neighbours_.forHalf(0, N,
                        [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                            double r2) RGPOT_MORSE_LAMBDA_INLINE {
                          if (!(i < j) || !(r2 > 0.0) || r2 > cutoff2) {
                            return;
                          }
                          const std::size_t c = batch.count++;
                          const double r = std::sqrt(r2);
                          batch.arg[c] = -range * (r - req);
                          batch.r[c] = r;
                          batch.dx[c] = dx;
                          batch.dy[c] = dy;
                          batch.dz[c] = dz;
                          batch.i[c] = i;
                          batch.j_or_u[c] = j;
                          if (batch.count == kPairBatch) {
                            flushBatch(batch, emit);
                          }
                        });
    flushBatch(batch, emit);
    *U = energy;
    publishCauchyStress(out, strain, volume);
    return;
  }

  WorkspaceLease lease;
  MorseWorkspace &ws = *lease.ws;
  const int slices = teamSize(n_atoms);

  if (!neighbours_.streaming()) {
    const std::size_t est_half_pairs = neighbours_.slots() / 2;
    const std::uint64_t scratch_avail = cachedAvailableBytes(ws, neighbours_);
    if (slices > 1 && est_half_pairs >= kMinParallelHalfPairs &&
        pairScratchFits(est_half_pairs, atoms, scratch_avail)) {
      const bool index_valid = neighbours_.reused() && ws.owner == &neighbours_ &&
                               ws.build_id == neighbours_.builds() && ws.index_atoms == n_atoms &&
                               ws.slots == neighbours_.slots();
      if (!index_valid) {
        buildStoredHalfIndex(ws, neighbours_, n_atoms);
      }
      const auto total_up = static_cast<std::size_t>(ws.up_off[n_atoms]);
      if (total_up >= kMinParallelHalfPairs) {
        const std::size_t nwords = (total_up + 63) / 64;
        ws.active_bits.assign(nwords, 0ULL);
        ws.pair_forces.resize(total_up);
        const bool want_he = atoms != nullptr && atoms->energy != nullptr;
        const bool want_hv = atoms != nullptr && atoms->virial != nullptr;
        if (want_he) {
          ws.pair_he.resize(total_up);
        }
        if (want_hv) {
          ws.pair_hv.resize(6 * total_up);
        }
        ws.row_e.resize(n_atoms);
        ws.row_s.resize(6 * n_atoms);

        const std::int64_t total_inc =
            static_cast<std::int64_t>(ws.lo_off[n_atoms]) + static_cast<std::int64_t>(total_up);
        if (ws.split_owner != &neighbours_ || ws.split_build_id != ws.build_id ||
            ws.split_atoms != n_atoms || ws.split_slices != slices ||
            ws.split_total_up != total_up ||
            ws.split_total_inc != static_cast<std::size_t>(total_inc)) {
          partitionAtomsByWork(n_atoms, slices, static_cast<std::int64_t>(total_up),
                               [&](std::size_t idx) {
                                 return static_cast<std::int64_t>(ws.up_off[idx]);
                               },
                               ws.p0_splits);
          partitionAtomsByWork(
              n_atoms, slices, total_inc,
              [&](std::size_t idx) {
                return static_cast<std::int64_t>(ws.lo_off[idx]) +
                       static_cast<std::int64_t>(ws.up_off[idx]);
              },
              ws.p1_splits);
          ws.split_owner = &neighbours_;
          ws.split_build_id = ws.build_id;
          ws.split_atoms = n_atoms;
          ws.split_slices = slices;
          ws.split_total_up = total_up;
          ws.split_total_inc = static_cast<std::size_t>(total_inc);
        }
        const std::size_t *p0_splits = ws.p0_splits.data();
        const std::size_t *p1_splits = ws.p1_splits.data();

        std::uint64_t *active_bits = ws.active_bits.data();
        PairForce *pair_forces = ws.pair_forces.data();
        double *pair_he = want_he ? ws.pair_he.data() : nullptr;
        double *pair_hv = want_hv ? ws.pair_hv.data() : nullptr;
        double *row_e = ws.row_e.data();
        double *row_s = ws.row_s.data();
        const std::int32_t *up_off = ws.up_off.data();
        const std::int32_t *lo_off = ws.lo_off.data();
        const std::int32_t *lo_u = ws.lo_u.data();

        forEachSlicePhase(2, slices, [&](int phase, int slice) {
          const auto s_idx = static_cast<std::size_t>(slice);
          if (phase == 0) {
            const std::size_t i0 = p0_splits[s_idx];
            const std::size_t i1 = p0_splits[s_idx + 1];
            if (i0 == i1) {
              return;
            }
            std::fill(row_e + i0, row_e + i1, 0.0);
            std::fill(row_s + 6 * i0, row_s + 6 * i1, 0.0);
            const std::int32_t u_begin = up_off[i0];
            const std::int32_t u_end = up_off[i1];
            if (u_begin == u_end) {
              return;
            }
            const std::size_t first_w = static_cast<std::size_t>(u_begin) >> 6;
            const std::size_t last_w = static_cast<std::size_t>(u_end - 1) >> 6;
            std::uint64_t first_mask = 0ULL;
            std::uint64_t last_mask = 0ULL;
            PairBatch batch;
            auto emit = [&](std::size_t k, double pair_e, double fscale) RGPOT_MORSE_LAMBDA_INLINE {
              const auto ai = static_cast<std::size_t>(batch.i[k]);
              const auto u = static_cast<std::size_t>(batch.j_or_u[k]);
              const double dx = batch.dx[k];
              const double dy = batch.dy[k];
              const double dz = batch.dz[k];
              row_e[ai] += pair_e;
              pair_forces[u] = PairForce{fscale * dx, fscale * dy, fscale * dz};
              accumulatePairStrain(row_s + 6 * ai, fscale, dx, dy, dz);
              if (want_he) {
                pair_he[u] = 0.5 * pair_e;
              }
              if (want_hv) {
                const double c = 0.5 * fscale;
                double *hv = pair_hv + 6 * u;
                hv[0] = c * dx * dx;
                hv[1] = c * dy * dy;
                hv[2] = c * dz * dz;
                hv[3] = c * dy * dz;
                hv[4] = c * dx * dz;
                hv[5] = c * dx * dy;
              }
            };
            std::int32_t next_u = u_begin;
            neighbours_.forHalf(
                static_cast<std::int64_t>(i0), static_cast<std::int64_t>(i1),
                [&](std::int32_t i, std::int32_t, double dx, double dy, double dz,
                    double r2) RGPOT_MORSE_LAMBDA_INLINE {
                  const std::int32_t u = next_u++;
                  if (!(r2 > 0.0) || r2 > cutoff2) {
                    return;
                  }
                  const std::size_t uw = static_cast<std::size_t>(u) >> 6;
                  const std::uint64_t bit =
                      std::uint64_t{1} << (static_cast<unsigned>(u) & 63u);
                  if (uw == first_w) {
                    first_mask |= bit;
                  } else if (uw == last_w) {
                    last_mask |= bit;
                  } else {
                    active_bits[uw] |= bit;
                  }
                  const std::size_t c = batch.count++;
                  const double r = std::sqrt(r2);
                  batch.arg[c] = -range * (r - req);
                  batch.r[c] = r;
                  batch.dx[c] = dx;
                  batch.dy[c] = dy;
                  batch.dz[c] = dz;
                  batch.i[c] = i;
                  batch.j_or_u[c] = u;
                  if (batch.count == kPairBatch) {
                    flushBatch(batch, emit);
                  }
                });
            flushBatch(batch, emit);
            if (first_mask != 0ULL) {
              std::atomic_ref<std::uint64_t>(active_bits[first_w])
                  .fetch_or(first_mask, std::memory_order_relaxed);
            }
            if (last_mask != 0ULL) {
              std::atomic_ref<std::uint64_t>(active_bits[last_w])
                  .fetch_or(last_mask, std::memory_order_relaxed);
            }
            return;
          }

          const std::size_t a0 = p1_splits[s_idx];
          const std::size_t a1 = p1_splits[s_idx + 1];
          for (std::size_t a_idx = a0; a_idx < a1; ++a_idx) {
            double fx = 0.0;
            double fy = 0.0;
            double fz = 0.0;
            double ae = 0.0;
            double av[6] = {};
            const std::int32_t l0 = lo_off[a_idx];
            const std::int32_t l1 = lo_off[a_idx + 1];
            for (std::int32_t k = l0; k < l1; ++k) {
              const auto u = static_cast<std::size_t>(lo_u[static_cast<std::size_t>(k)]);
              if (((active_bits[u >> 6] >> (u & 63u)) & 1ULL) == 0ULL) {
                continue;
              }
              const PairForce &pf = pair_forces[u];
              fx -= pf.fx;
              fy -= pf.fy;
              fz -= pf.fz;
              if (want_he) {
                ae += pair_he[u];
              }
              if (want_hv) {
                const double *hv = pair_hv + 6 * u;
                for (int c = 0; c < 6; ++c) {
                  av[c] += hv[c];
                }
              }
            }
            const std::int32_t u0 = up_off[a_idx];
            const std::int32_t u1 = up_off[a_idx + 1];
            for (std::int32_t up = u0; up < u1; ++up) {
              const auto u = static_cast<std::size_t>(up);
              if (((active_bits[u >> 6] >> (u & 63u)) & 1ULL) == 0ULL) {
                continue;
              }
              const PairForce &pf = pair_forces[u];
              fx += pf.fx;
              fy += pf.fy;
              fz += pf.fz;
              if (want_he) {
                ae += pair_he[u];
              }
              if (want_hv) {
                const double *hv = pair_hv + 6 * u;
                for (int c = 0; c < 6; ++c) {
                  av[c] += hv[c];
                }
              }
            }
            F[3 * a_idx] = fx;
            F[3 * a_idx + 1] = fy;
            F[3 * a_idx + 2] = fz;
            if (want_he) {
              atoms->energy[a_idx] += ae;
            }
            if (want_hv) {
              double *dst = atoms->virial + 6 * a_idx;
              for (int c = 0; c < 6; ++c) {
                dst[c] += av[c];
              }
            }
          }
        });

        double energy = 0.0;
        for (std::size_t a_idx = 0; a_idx < n_atoms; ++a_idx) {
          energy += row_e[a_idx];
          for (std::size_t c = 0; c < 6; ++c) {
            strain[c] += row_s[6 * a_idx + c];
          }
        }
        *U = energy;
        publishCauchyStress(out, strain, volume);
        return;
      }
    }
  }

  std::fill_n(F, 3 * n_atoms, 0.0);
  ws.row_e.assign(n_atoms, 0.0);
  ws.row_s.assign(6 * n_atoms, 0.0);
  double *row_e = ws.row_e.data();
  double *row_s = ws.row_s.data();

  if (!neighbours_.streaming() || (slices == 1 && neighbours_.allPairs())) {
    PairBatch batch;
    auto emit = [&](std::size_t k, double pair_e, double fscale) RGPOT_MORSE_LAMBDA_INLINE {
      const auto ii = batch.i[k];
      const auto jj = batch.j_or_u[k];
      const auto ai = static_cast<std::size_t>(ii);
      const double dx = batch.dx[k];
      const double dy = batch.dy[k];
      const double dz = batch.dz[k];
      row_e[ai] += pair_e;
      const double fx = fscale * dx;
      const double fy = fscale * dy;
      const double fz = fscale * dz;
      F[3 * ii] += fx;
      F[3 * ii + 1] += fy;
      F[3 * ii + 2] += fz;
      F[3 * jj] -= fx;
      F[3 * jj + 1] -= fy;
      F[3 * jj + 2] -= fz;
      accumulatePairStrain(row_s + 6 * ai, fscale, dx, dy, dz);
      if (atoms != nullptr) {
        tallyPair(*atoms, ai, static_cast<std::size_t>(jj), pair_e, fscale, dx, dy, dz);
      }
    };
    neighbours_.forHalf(0, N,
                        [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                            double r2) RGPOT_MORSE_LAMBDA_INLINE {
                          if (!(i < j) || !(r2 > 0.0) || r2 > cutoff2) {
                            return;
                          }
                          const std::size_t c = batch.count++;
                          const double r = std::sqrt(r2);
                          batch.arg[c] = -range * (r - req);
                          batch.r[c] = r;
                          batch.dx[c] = dx;
                          batch.dy[c] = dy;
                          batch.dz[c] = dz;
                          batch.i[c] = i;
                          batch.j_or_u[c] = j;
                          if (batch.count == kPairBatch) {
                            flushBatch(batch, emit);
                          }
                        });
    flushBatch(batch, emit);
  } else {
    std::vector<PairBatch> batches(static_cast<std::size_t>(slices));
    auto emitSlice = [&](PairBatch &batch) RGPOT_MORSE_LAMBDA_INLINE {
      flushBatch(batch, [&](std::size_t k, double pair_e, double fscale) RGPOT_MORSE_LAMBDA_INLINE {
        const auto ii = batch.i[k];
        const auto jj = batch.j_or_u[k];
        const auto ai = static_cast<std::size_t>(ii);
        const double dx = batch.dx[k];
        const double dy = batch.dy[k];
        const double dz = batch.dz[k];
        const double fx = fscale * dx;
        const double fy = fscale * dy;
        const double fz = fscale * dz;
        F[3 * ai] += fx;
        F[3 * ai + 1] += fy;
        F[3 * ai + 2] += fz;
        if (ii < jj) {
          row_e[ai] += pair_e;
          accumulatePairStrain(row_s + 6 * ai, fscale, dx, dy, dz);
        }
        if (atoms != nullptr) {
          if (atoms->energy != nullptr) {
            atoms->energy[ai] += 0.5 * pair_e;
          }
          tallyAtomVirial(*atoms, ai, 0.5, fscale, dx, dy, dz);
        }
      });
    };
    neighbours_.forRowTeam(
        slices, [&](int slice, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                    double r2) RGPOT_MORSE_LAMBDA_INLINE {
          if (!(r2 > 0.0) || r2 > cutoff2) {
            return;
          }
          PairBatch &batch = batches[static_cast<std::size_t>(slice)];
          const std::size_t c = batch.count++;
          const double r = std::sqrt(r2);
          batch.arg[c] = -range * (r - req);
          batch.r[c] = r;
          batch.dx[c] = dx;
          batch.dy[c] = dy;
          batch.dz[c] = dz;
          batch.i[c] = i;
          batch.j_or_u[c] = j;
          if (batch.count == kPairBatch) {
            emitSlice(batch);
          }
        });
    for (PairBatch &batch : batches) {
      emitSlice(batch);
    }
  }

  double energy = 0.0;
  for (std::size_t a_idx = 0; a_idx < n_atoms; ++a_idx) {
    energy += row_e[a_idx];
    for (std::size_t c = 0; c < 6; ++c) {
      strain[c] += row_s[6 * a_idx + c];
    }
  }
  *U = energy;
  publishCauchyStress(out, strain, volume);
}

void MorsePot::forceStage(const HalfList &list, const double *, double *forces,
                          StageTally *tally) const {
  if (cuttOffR <= 0.0) {
    return;
  }
  const double cutoff2 = cuttOffR * cuttOffR;
  staged::withTally(tally, [&](auto tallied) {
    staged::forEachPair(
        list, [&](int i, int j, double dx, double dy, double dz, double r2, bool both) {
          if (!(r2 > 0.0) || r2 > cutoff2) {
            return;
          }
          double pair_e = 0.0;
          double fscale = 0.0;
          pairTerm(std::sqrt(r2), pair_e, fscale);
          staged::addPair(tallied, forces, tally, i, j, both, pair_e, fscale, dx, dy, dz);
        });
  });
}

void MorsePot::updateNeighbours(const ForceInput &in) const {
  // The cell walk is slower than the stored list from about 20k atoms
  // upward (measured at 108k and 1M). Auto stores there.
  nlist::Mode mode = mode_;
  if (mode == nlist::Mode::Auto && in.nAtoms >= 20000) {
    mode = nlist::Mode::Stored;
  }
  const double skin = mode == nlist::Mode::Stream ? 0.0 : neighbourSkin(cuttOffR);
  const std::array<bool, 3> periodic{{true, true, true}};
  nlist::updateCells(neighbours_, in.pos, in.nAtoms, in.box, cuttOffR, skin,
                     nlist::Image::Round, mode, periodic);
}

/**
 * @details
 *
 * With @f$e = \exp(-a(r - r_e))@f$ the bare pair term has
 * @f$\phi' = 2 D_e a e (1 - e)@f$ and @f$\phi'' = 2 D_e a^2 e (2e - 1)@f$.
 * The shift is a constant and drops out. The switch multiplies the bare
 * term by @f$S@f$, so @f$\phi' = V' S + V S'@f$ and
 * @f$\phi'' = V'' S + 2 V' S' + V S''@f$.
 */
void MorsePot::hessianVectorImpl(const ForceInput &in, const double *v,
                                 double *hv) const {
  const auto N = static_cast<long>(in.nAtoms);
  std::fill_n(hv, 3 * in.nAtoms, 0.0);
  if (N < 2 || cuttOffR <= 0.0) {
    return;
  }
  updateNeighbours(in);
  const double cutoff2 = cuttOffR * cuttOffR;
  const double twoDeA = 2.0 * De * a;
  const bool switched = m_config.switch_width != 0.0;
  nlist::visitHalf(neighbours_, 0, N,
                   [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                       double r2) {
    if (!(i < j) || !(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const double r = std::sqrt(r2);
    const double e = simd::exp_fixed(-a * (r - re));
    double d1 = twoDeA * e * (1.0 - e);
    double d2 = twoDeA * a * e * (2.0 * e - 1.0);
    if (switched) {
      const double bare = De * (1.0 - e) * (1.0 - e) - De;
      const auto s = m_switch(r);
      const double s2 = m_switch.d2sdr2(r);
      d2 = d2 * s.s + 2.0 * d1 * s.dsdr + bare * s2;
      d1 = d1 * s.s + bare * s.dsdr;
    }
    const double inv_r = 1.0 / r;
    const double ux = dx * inv_r;
    const double uy = dy * inv_r;
    const double uz = dz * inv_r;
    const double *vi = v + 3 * static_cast<std::size_t>(i);
    const double *vj = v + 3 * static_cast<std::size_t>(j);
    const double wx = vi[0] - vj[0];
    const double wy = vi[1] - vj[1];
    const double wz = vi[2] - vj[2];
    const double radial = d1 * inv_r;
    const double along = (d2 - radial) * (ux * wx + uy * wy + uz * wz);
    const double kx = along * ux + radial * wx;
    const double ky = along * uy + radial * wy;
    const double kz = along * uz + radial * wz;
    double *hi = hv + 3 * static_cast<std::size_t>(i);
    double *hj = hv + 3 * static_cast<std::size_t>(j);
    hi[0] += kx;
    hi[1] += ky;
    hi[2] += kz;
    hj[0] -= kx;
    hj[1] -= ky;
    hj[2] -= kz;
  });
}

namespace detail {

MorseSplitSnapshot morseStoredSplitSnapshot() {
  WorkspaceLease lease;
  const MorseWorkspace &ws = *lease.ws;
  return MorseSplitSnapshot{
      .owner = ws.split_owner,
      .build_id = ws.split_build_id,
      .atoms = ws.split_atoms,
      .slices = ws.split_slices,
      .total_up = ws.split_total_up,
      .total_inc = ws.split_total_inc,
      .p0_splits = ws.p0_splits,
      .p1_splits = ws.p1_splits,
  };
}

} // namespace detail

} // namespace rgpot

#undef RGPOT_MORSE_LAMBDA_INLINE
