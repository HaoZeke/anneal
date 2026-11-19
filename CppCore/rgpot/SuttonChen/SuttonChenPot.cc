// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Sutton-Chen energy and analytic forces.
 *
 * Each unordered pair contributes epsilon * (a/r)^n once, which is the
 * 1/2 sum_i sum_{j != i} term. The density sum has no 1/2. The force
 * on atom i along d = r_i - r_j is fscale * d, with
 * fscale = -(dE/dr) / r.
 *
 * The energy depends on positions only through pair distances, so with
 * SuttonChenConfig::stress the force pass also sums -fscale * d (x) d,
 * which is dE/dstrain for the embedding and pair terms together.
 */

// clang-format off
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
// clang-format on

#include "rgpot/KeptPotential.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

// The pair terms run once per pair and are called from the serial and the
// team walks. Out of line, the call costs as much as the arithmetic.
#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_SC_PAIR_INLINE __attribute__((always_inline))
#else
#define RGPOT_SC_PAIR_INLINE
#endif

namespace {

double ipow(double base, int exp) {
  double result = 1.0;
  while (exp > 0) {
    if ((exp & 1) != 0) {
      result *= base;
    }
    base *= base;
    exp >>= 1;
  }
  return result;
}

/// (a/r)^exp from ar2 = (a/r)^2.
double pow_ar(double ar2, int exp) {
  double p = ipow(ar2, exp / 2);
  if ((exp % 2) != 0) {
    p *= std::sqrt(ar2);
  }
  return p;
}

/// cutoff <= 0 keeps every pair. The unbounded search walks them all
/// without bins or a list.
double searchCutoff(const rgpot::SuttonChenConfig &cfg) {
  return cfg.cutoff > 0.0 ? cfg.cutoff : std::numeric_limits<double>::infinity();
}

rgpot::SuttonChenConfig config_from_c(const rgpot_sc_config &raw) {
  rgpot::SuttonChenConfig c;
  c.epsilon = raw.epsilon;
  c.c = raw.c;
  c.a = raw.a;
  c.n = raw.n;
  c.m = raw.m;
  c.cutoff = raw.cutoff;
  c.periodic = raw.periodic != 0;
  return c;
}

constexpr std::size_t kMinParallelHalfPairs = 4096;
constexpr std::size_t kSliceAtomQuantum = 8;

struct LoRef {
  std::int32_t u;
  std::int32_t i;
};

struct PairForceData {
  double dx;
  double dy;
  double dz;
  double inv_r2;
  double rep_term;
  double phi_term;
};

struct SuttonChenWorkspace {
  const rgpot::nlist::Neighbours *owner = nullptr;
  std::uint64_t build_id = 0;
  std::size_t index_atoms = 0;
  std::size_t slots = 0;

  std::vector<std::int32_t> up_off;
  std::vector<std::int32_t> lo_off;
  std::vector<std::int32_t> lo_count;
  std::vector<std::int32_t> up_j;
  std::vector<LoRef> lo_refs;

  std::vector<std::uint8_t> active;
  std::vector<double> pair_phi;
  std::vector<PairForceData> pair_force;
  std::vector<double> pair_he;
  std::vector<double> rho;
  std::vector<double> weight;
  std::vector<double> row_e;
  std::vector<double> row_emb;
  std::vector<double> row_s;
  std::vector<std::size_t> p0_splits;
  std::vector<std::size_t> p1_splits;

  const rgpot::nlist::Neighbours *split_owner = nullptr;
  std::uint64_t split_build_id = 0;
  std::size_t split_atoms = 0;
  int split_slices = 0;
  std::size_t split_total_up = 0;
  std::size_t split_total_inc = 0;

  const rgpot::nlist::Neighbours *scratch_owner = nullptr;
  std::uint64_t scratch_build_id = 0;
  std::size_t scratch_slots = 0;
  std::uint64_t scratch_avail = 0;
};

struct WorkspaceLease {
  SuttonChenWorkspace *ws = nullptr;
  std::unique_ptr<SuttonChenWorkspace> fallback;
  bool *busy_flag = nullptr;

  WorkspaceLease() {
    thread_local SuttonChenWorkspace tls_ws;
    thread_local bool tls_busy = false;
    if (!tls_busy) {
      tls_busy = true;
      busy_flag = &tls_busy;
      ws = &tls_ws;
    } else {
      fallback = std::make_unique<SuttonChenWorkspace>();
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

void buildStoredHalfIndex(SuttonChenWorkspace &ws, const rgpot::nlist::Neighbours &cells,
                          std::size_t n) {
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

  ws.up_j.resize(static_cast<std::size_t>(total_up));
  ws.lo_refs.resize(static_cast<std::size_t>(total_up));
  for (std::size_t i = 0; i < n; ++i) {
    const auto self = static_cast<std::int32_t>(i);
    const auto hi = off[i + 1];
    const auto count_i = ws.up_off[i + 1] - ws.up_off[i];
    const auto mid = hi - count_i;
    std::int32_t u = ws.up_off[i];
    for (std::int32_t p = mid; p < hi; ++p) {
      const auto j = nbr[static_cast<std::size_t>(p)];
      ws.up_j[static_cast<std::size_t>(u)] = j;
      ws.lo_refs[static_cast<std::size_t>(ws.lo_count[static_cast<std::size_t>(j)]++)] =
          LoRef{u, self};
      ++u;
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

std::uint64_t cachedAvailableBytes(SuttonChenWorkspace &ws,
                                   const rgpot::nlist::Neighbours &cells) {
  if (ws.scratch_owner != &cells || ws.scratch_build_id != cells.builds() ||
      ws.scratch_slots != cells.slots()) {
    ws.scratch_owner = &cells;
    ws.scratch_build_id = cells.builds();
    ws.scratch_slots = cells.slots();
    ws.scratch_avail = rgpot::nlist::availableBytes();
  }
  return ws.scratch_avail;
}

bool pairScratchFits(std::size_t num_pairs, const rgpot::PerAtomOut *atoms,
                     std::uint64_t avail) {
  std::size_t bytes_per_pair = sizeof(PairForceData) + sizeof(double) + sizeof(std::uint8_t) +
                               sizeof(std::int32_t) + sizeof(LoRef);
  if (atoms != nullptr && atoms->energy != nullptr) {
    bytes_per_pair += sizeof(double);
  }
  if (avail == 0) {
    return true;
  }
  return static_cast<std::uint64_t>(num_pairs) <= (avail / 2) / bytes_per_pair;
}

} // namespace

namespace rgpot {

SuttonChenPot::SuttonChenPot(const SuttonChenConfig &c)
    : Potential(PotType::SuttonChen), m_config{c} {
  if (c.n < 1 || c.m < 1) {
    throw std::invalid_argument("Sutton-Chen exponents n and m must be >= 1");
  }
  if (!(c.a > 0.0) || !std::isfinite(c.epsilon) || !std::isfinite(c.c) ||
      !std::isfinite(c.cutoff)) {
    throw std::invalid_argument("Sutton-Chen parameters are not finite");
  }
  if (c.periodic && !(c.cutoff > 0.0)) {
    throw std::invalid_argument(
        "periodic Sutton-Chen needs a positive cutoff");
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.f64(c.epsilon);
  fp.f64(c.c);
  fp.f64(c.a);
  fp.u64(static_cast<uint64_t>(c.n));
  fp.u64(static_cast<uint64_t>(c.m));
  fp.f64(c.cutoff);
  fp.u64(c.periodic ? 1 : 0);
  if (c.stress) {
    // Mixed in only when set, so the default key keeps its old value.
    fp.u64(1);
  }
  m_paramsKey = fp.h;
}

void SuttonChenPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void SuttonChenPot::forcePerAtom(const ForceInput &in, ForceOut *out,
                                 const PerAtomOut &atoms) const {
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

void SuttonChenPot::compute(const ForceInput &in, ForceOut *out,
                            const PerAtomOut *atoms) const {
  const auto n_atoms = static_cast<long>(in.nAtoms);
  double *F = out->F;
  out->energy = 0.0;
  out->variance = 0.0;
  const SuttonChenConfig cfg = m_config;
  double strain[6] = {};
  if (n_atoms < 2) {
    for (long i = 0; i < n_atoms; ++i) {
      F[3 * i] = 0.0;
      F[3 * i + 1] = 0.0;
      F[3 * i + 2] = 0.0;
    }
    if (cfg.stress) {
      publishCauchyStress(out, strain, cellVolume(in.box));
    }
    return;
  }

  const double *R = in.pos;
  const double cutoff = searchCutoff(cfg);
  const double cutoff2 = cutoff * cutoff;
  const std::array<bool, 3> periodic =
      cfg.periodic ? std::array<bool, 3>{true, true, true}
                   : std::array<bool, 3>{false, false, false};

  static constexpr double free_box[9] = {1.0e6, 0.0, 0.0, 0.0, 1.0e6,
                                         0.0,   0.0, 0.0, 1.0e6};
  const bool box_ok = in.box != nullptr && in.box[0] > 0.0 && in.box[4] > 0.0 &&
                      in.box[8] > 0.0;
  if (cfg.periodic && !box_ok) {
    throw std::invalid_argument("periodic Sutton-Chen needs a real cell");
  }
  const double *box_use = box_ok ? in.box : free_box;

  const double epsilon = cfg.epsilon;
  const double c_coef = cfg.c;
  const double lattice = cfg.a;
  const int n_exp = cfg.n;
  const int m_exp = cfg.m;
  const auto n_size = static_cast<std::size_t>(n_atoms);
  double *atom_e = atoms != nullptr ? atoms->energy : nullptr;

  const double skin = mode_ == nlist::Mode::Stream ? 0.0 : cutoff / 5.0;
  nlist::updateCells(neighbours_, R, n_size, box_use, cutoff, skin, nlist::Image::Round,
                     mode_, periodic);
  const bool stream = neighbours_.streaming();
  const bool rows = n_size >= kTeamMinAtoms;
  const int nt = nlist::teamSize(n_size);
  const double a2 = lattice * lattice;
  const bool unit_a = a2 == 1.0;
  const double en = epsilon * static_cast<double>(n_exp);
  const double em = static_cast<double>(m_exp);

  bool sparse_stored = false;
  if (rows && !stream && nt > 1 && term_limit_ > 0) {
    WorkspaceLease lease;
    SuttonChenWorkspace &ws = *lease.ws;
    const std::size_t est_half_pairs = neighbours_.slots() / 2;
    const std::uint64_t scratch_avail = cachedAvailableBytes(ws, neighbours_);
    if (est_half_pairs < kMinParallelHalfPairs) {
      sparse_stored = true;
    } else if (pairScratchFits(est_half_pairs, atoms, scratch_avail)) {
      const bool index_valid = neighbours_.reused() && ws.owner == &neighbours_ &&
                               ws.build_id == neighbours_.builds() &&
                               ws.index_atoms == n_size &&
                               ws.slots == neighbours_.slots();
      if (!index_valid) {
        buildStoredHalfIndex(ws, neighbours_, n_size);
      }
      const auto total_up = static_cast<std::size_t>(ws.up_off[n_size]);
      if (total_up < kMinParallelHalfPairs) {
        sparse_stored = true;
      } else if (total_up > term_limit_) {
        // The three-phase path keeps every candidate upper pair in scratch.
        // If that count exceeds the configured limit, keep the row-team
        // two-pass path so threaded rows still run.
      } else if (pairScratchFits(total_up, atoms, scratch_avail)) {
        ws.active.resize(total_up);
        ws.pair_phi.resize(total_up);
        ws.pair_force.resize(total_up);
        const bool want_he = atom_e != nullptr;
        const bool want_hv = atoms != nullptr && atoms->virial != nullptr;
        if (want_he) {
          ws.pair_he.resize(total_up);
        }
        ws.rho.resize(n_size);
        ws.weight.resize(n_size);
        ws.row_e.resize(n_size);
        ws.row_emb.resize(n_size);
        if (cfg.stress) {
          ws.row_s.resize(6 * n_size);
        }

        const std::int64_t total_inc =
            static_cast<std::int64_t>(ws.lo_off[n_size]) + static_cast<std::int64_t>(total_up);
        if (ws.split_owner != &neighbours_ || ws.split_build_id != ws.build_id ||
            ws.split_atoms != n_size || ws.split_slices != nt ||
            ws.split_total_up != total_up ||
            ws.split_total_inc != static_cast<std::size_t>(total_inc)) {
          partitionAtomsByWork(
              n_size, nt, static_cast<std::int64_t>(total_up),
              [&](std::size_t idx) { return static_cast<std::int64_t>(ws.up_off[idx]); },
              ws.p0_splits);
          partitionAtomsByWork(
              n_size, nt, total_inc,
              [&](std::size_t idx) {
                return static_cast<std::int64_t>(ws.lo_off[idx]) +
                       static_cast<std::int64_t>(ws.up_off[idx]);
              },
              ws.p1_splits);
          ws.split_owner = &neighbours_;
          ws.split_build_id = ws.build_id;
          ws.split_atoms = n_size;
          ws.split_slices = nt;
          ws.split_total_up = total_up;
          ws.split_total_inc = static_cast<std::size_t>(total_inc);
        }

        const std::size_t *p0_splits = ws.p0_splits.data();
        const std::size_t *p1_splits = ws.p1_splits.data();
        const std::int32_t *up_off = ws.up_off.data();
        const std::int32_t *lo_off = ws.lo_off.data();
        const std::int32_t *up_j = ws.up_j.data();
        const LoRef *lo_refs = ws.lo_refs.data();
        std::uint8_t *active = ws.active.data();
        double *pair_phi = ws.pair_phi.data();
        PairForceData *pair_force = ws.pair_force.data();
        double *pair_he = want_he ? ws.pair_he.data() : nullptr;
        double *rho = ws.rho.data();
        double *weight = ws.weight.data();
        double *row_e = ws.row_e.data();
        double *row_emb = ws.row_emb.data();
        double *row_s = cfg.stress ? ws.row_s.data() : nullptr;
        double *atom_v = want_hv ? atoms->virial : nullptr;
        const bool stress_on = cfg.stress;

        forEachSlicePhase(3, nt, [&](int phase, int slice) {
          const auto s_idx = static_cast<std::size_t>(slice);
          if (phase == 0) {
            const std::size_t i0 = p0_splits[s_idx];
            const std::size_t i1 = p0_splits[s_idx + 1];
            if (i0 == i1) {
              return;
            }
            std::fill(row_e + i0, row_e + i1, 0.0);
            std::int32_t next_u = up_off[i0];
            neighbours_.forHalf(
                static_cast<std::int64_t>(i0), static_cast<std::int64_t>(i1),
                [&](int32_t i, int32_t, double dx, double dy, double dz,
                    double r2) RGPOT_SC_PAIR_INLINE {
                  const auto u = static_cast<std::size_t>(next_u++);
                  if (!(r2 > 0.0) || r2 > cutoff2) {
                    active[u] = 0;
                    return;
                  }
                  active[u] = 1;
                  double phi = 0.0;
                  double rep = 0.0;
                  double rep_term = 0.0;
                  double phi_term = 0.0;
                  double inv_r2 = 0.0;
                  if (unit_a) {
                    inv_r2 = 1.0 / r2;
                    phi = pow_ar(inv_r2, m_exp);
                    rep = pow_ar(inv_r2, n_exp);
                    rep_term = en * rep;
                    phi_term = em * phi;
                  } else {
                    const double ar2 = a2 / r2;
                    phi = pow_ar(ar2, m_exp);
                    rep = pow_ar(ar2, n_exp);
                    inv_r2 = 1.0 / r2;
                    const double ar2_force = a2 * inv_r2;
                    const bool same = ar2_force == ar2;
                    rep_term = en * (same ? rep : pow_ar(ar2_force, n_exp));
                    phi_term = em * (same ? phi : pow_ar(ar2_force, m_exp));
                  }
                  row_e[static_cast<std::size_t>(i)] += epsilon * rep;
                  pair_phi[u] = phi;
                  pair_force[u] = PairForceData{dx, dy, dz, inv_r2, rep_term, phi_term};
                  if (want_he) {
                    pair_he[u] = 0.5 * epsilon * rep;
                  }
                });
            return;
          }

          const std::size_t a0 = p1_splits[s_idx];
          const std::size_t a1 = p1_splits[s_idx + 1];
          if (a0 == a1) {
            return;
          }
          if (phase == 1) {
            const double ec = epsilon * c_coef;
            for (std::size_t a = a0; a < a1; ++a) {
              double rho_a = 0.0;
              double ae = 0.0;
              const std::int32_t l0 = lo_off[a];
              const std::int32_t l1 = lo_off[a + 1];
              for (std::int32_t k = l0; k < l1; ++k) {
                const auto u = static_cast<std::size_t>(lo_refs[k].u);
                if (active[u] == 0) {
                  continue;
                }
                rho_a += pair_phi[u];
                if (want_he) {
                  ae += pair_he[u];
                }
              }
              const std::int32_t u0 = up_off[a];
              const std::int32_t u1 = up_off[a + 1];
              for (std::int32_t up = u0; up < u1; ++up) {
                const auto u = static_cast<std::size_t>(up);
                if (active[u] == 0) {
                  continue;
                }
                rho_a += pair_phi[u];
                if (want_he) {
                  ae += pair_he[u];
                }
              }
              rho[a] = rho_a;
              if (rho_a > 0.0) {
                const double root = std::sqrt(rho_a);
                const double emb = ec * root;
                row_emb[a] = emb;
                weight[a] = ec / (2.0 * root);
                if (want_he) {
                  ae -= emb;
                }
              } else {
                row_emb[a] = 0.0;
                weight[a] = 0.0;
              }
              if (want_he) {
                atom_e[a] = ae;
              }
            }
            return;
          }

          for (std::size_t a = a0; a < a1; ++a) {
            const double wa = weight[a];
            double fx = 0.0;
            double fy = 0.0;
            double fz = 0.0;
            double av[6] = {};
            const std::int32_t l0 = lo_off[a];
            const std::int32_t l1 = lo_off[a + 1];
            for (std::int32_t k = l0; k < l1; ++k) {
              const LoRef ref = lo_refs[k];
              const auto u = static_cast<std::size_t>(ref.u);
              if (active[u] == 0) {
                continue;
              }
              const PairForceData &pd = pair_force[u];
              const double wi = weight[static_cast<std::size_t>(ref.i)];
              const double fscale = (pd.rep_term - pd.phi_term * (wi + wa)) * pd.inv_r2;
              fx -= fscale * pd.dx;
              fy -= fscale * pd.dy;
              fz -= fscale * pd.dz;
              if (want_hv) {
                const double c = 0.5 * fscale;
                av[0] += c * pd.dx * pd.dx;
                av[1] += c * pd.dy * pd.dy;
                av[2] += c * pd.dz * pd.dz;
                av[3] += c * pd.dy * pd.dz;
                av[4] += c * pd.dx * pd.dz;
                av[5] += c * pd.dx * pd.dy;
              }
            }
            double rs[6] = {};
            const std::int32_t u0 = up_off[a];
            const std::int32_t u1 = up_off[a + 1];
            for (std::int32_t up = u0; up < u1; ++up) {
              const auto u = static_cast<std::size_t>(up);
              if (active[u] == 0) {
                continue;
              }
              const PairForceData &pd = pair_force[u];
              const double wj = weight[static_cast<std::size_t>(up_j[u])];
              const double fscale = (pd.rep_term - pd.phi_term * (wa + wj)) * pd.inv_r2;
              fx += fscale * pd.dx;
              fy += fscale * pd.dy;
              fz += fscale * pd.dz;
              if (stress_on) {
                accumulatePairStrain(rs, fscale, pd.dx, pd.dy, pd.dz);
              }
              if (want_hv) {
                const double c = 0.5 * fscale;
                av[0] += c * pd.dx * pd.dx;
                av[1] += c * pd.dy * pd.dy;
                av[2] += c * pd.dz * pd.dz;
                av[3] += c * pd.dy * pd.dz;
                av[4] += c * pd.dx * pd.dz;
                av[5] += c * pd.dx * pd.dy;
              }
            }
            F[3 * a] = fx;
            F[3 * a + 1] = fy;
            F[3 * a + 2] = fz;
            if (stress_on) {
              double *dst = row_s + 6 * a;
              for (int c = 0; c < 6; ++c) {
                dst[c] = rs[c];
              }
            }
            if (want_hv) {
              double *dst = atom_v + 6 * a;
              for (int c = 0; c < 6; ++c) {
                dst[c] = av[c];
              }
            }
          }
        });

        double epair = 0.0;
        double eemb = 0.0;
        for (std::size_t i = 0; i < n_size; ++i) {
          epair += row_e[i];
          if (rho[i] > 0.0) {
            eemb -= row_emb[i];
          }
        }
        if (cfg.stress) {
          for (std::size_t i = 0; i < n_size; ++i) {
            for (std::size_t k = 0; k < 6; ++k) {
              strain[k] += row_s[6 * i + k];
            }
          }
        }
        out->energy = epair + eemb;
        if (cfg.stress) {
          publishCauchyStress(out, strain, cellVolume(in.box));
        }
        return;
      }
    }
  }

  for (long i = 0; i < n_atoms; ++i) {
    F[3 * i] = 0.0;
    F[3 * i + 1] = 0.0;
    F[3 * i + 2] = 0.0;
  }
  rho_.assign(n_size, 0.0);
  double *rho = rho_.data();
  double epair = 0.0;

  // From kTeamMinAtoms up, each atom keeps the pair energy and strain of
  // its partners above it and the totals add them in atom order, as in
  // nlist/PairSum.hpp. A team, or a stream, then walks the rows of the
  // atoms; one thread on a stored list or an unbounded search keeps the
  // half walks, whose densities and forces the rows repeat bit for bit.
  const bool by_row = !sparse_stored && rows && (nt > 1 || (stream && !neighbours_.allPairs()));
  std::vector<double> row_e(rows ? n_size : 0, 0.0);
  std::vector<double> row_s(rows && cfg.stress ? 6 * n_size : 0, 0.0);
  double *e_sum = rows ? row_e.data() : &epair;
  double *s_sum = !cfg.stress ? nullptr : (rows ? row_s.data() : strain);
  // The half walks take the stride of the sums as a type, so below
  // kTeamMinAtoms every pair adds to the one total with no index.
  auto by_stride = [rows](auto walk) {
    if (rows) {
      walk(std::integral_constant<std::size_t, 1>{});
    } else {
      walk(std::integral_constant<std::size_t, 0>{});
    }
  };

  // Each walk visits the same pairs in the same order, one lane per
  // slice, so the force pass reads back what the density pass kept.
  const auto lanes = static_cast<std::size_t>(by_row ? nt : 1);
  if (terms_.size() < lanes) {
    terms_.resize(lanes);
  }
  std::size_t expected = 0;
  if (!by_row) {
    expected = neighbours_.allPairs() ? n_size * (n_size - 1) / 2
               : stream               ? 0
                                      : neighbours_.slots() / 2;
  }
  for (std::size_t t = 0; t < lanes; ++t) {
    terms_[t].setLimit(term_limit_ / lanes);
    terms_[t].record(t == 0 ? expected : 0);
  }
  // The force takes the powers of a^2 * (1/r^2), the density those of
  // a^2 / r^2. With a^2 = 1 they are one number; otherwise the last bit
  // can differ, and the kept powers are the force's.
  // The pair lambdas copy their constants and arrays: a walk that copies
  // its callback then keeps them in registers. Only the sums and the
  // lanes are shared by reference.
  auto pair_term = [=](double r2, double &phi, double &rep, Terms &terms) RGPOT_SC_PAIR_INLINE {
    double *kept = terms.push();
    if (unit_a) {
      const double inv_r2 = 1.0 / r2;
      phi = pow_ar(inv_r2, m_exp);
      rep = pow_ar(inv_r2, n_exp);
      if (kept != nullptr) {
        kept[0] = en * rep;
        kept[1] = em * phi;
        kept[2] = inv_r2;
      }
      return;
    }
    const double ar2 = a2 / r2;
    phi = pow_ar(ar2, m_exp);
    rep = pow_ar(ar2, n_exp);
    if (kept != nullptr) {
      const double inv_r2 = 1.0 / r2;
      const double ar2_force = a2 * inv_r2;
      const bool same = ar2_force == ar2;
      kept[0] = en * (same ? rep : pow_ar(ar2_force, n_exp));
      kept[1] = em * (same ? phi : pow_ar(ar2_force, m_exp));
      kept[2] = inv_r2;
    }
  };
  if (!by_row) {
    Terms &lane = terms_[0];
    by_stride([&](auto stride) {
      nlist::visitHalf(neighbours_,
          0, n_atoms,
          [=, &lane](int32_t i, int32_t j, double, double, double,
                     double r2) RGPOT_SC_PAIR_INLINE {
            if (!(r2 > 0.0) || r2 > cutoff2) {
              return;
            }
            double phi = 0.0;
            double rep = 0.0;
            pair_term(r2, phi, rep, lane);
            rho[static_cast<std::size_t>(i)] += phi;
            rho[static_cast<std::size_t>(j)] += phi;
            e_sum[static_cast<std::size_t>(i) * decltype(stride)::value] += epsilon * rep;
            if (atom_e != nullptr) {
              atom_e[static_cast<std::size_t>(i)] += 0.5 * epsilon * rep;
              atom_e[static_cast<std::size_t>(j)] += 0.5 * epsilon * rep;
            }
          });
    });
  } else {
    Terms *lane_terms = terms_.data();
    // Each row belongs to one slice, so only that thread writes its atom.
    neighbours_.forRowTeam(
        nt,
        [=](int t, int32_t i, int32_t j, double, double, double, double r2) RGPOT_SC_PAIR_INLINE {
          if (!(r2 > 0.0) || r2 > cutoff2) {
            return;
          }
          double phi = 0.0;
          double rep = 0.0;
          pair_term(r2, phi, rep, lane_terms[t]);
          rho[static_cast<std::size_t>(i)] += phi;
          if (j > i) {
            e_sum[static_cast<std::size_t>(i)] += epsilon * rep;
          }
          if (atom_e != nullptr) {
            atom_e[static_cast<std::size_t>(i)] += 0.5 * epsilon * rep;
          }
        });
  }

  weight_.assign(n_size, 0.0);
  double *weight = weight_.data();
  double eemb = 0.0;
  for (std::size_t i = 0; i < n_size; ++i) {
    if (!(rho[i] > 0.0)) {
      continue;
    }
    const double root = std::sqrt(rho[i]);
    eemb -= epsilon * c_coef * root;
    weight[i] = epsilon * c_coef / (2.0 * root);
    if (atom_e != nullptr) {
      atom_e[i] -= epsilon * c_coef * root;
    }
  }

  for (std::size_t t = 0; t < lanes; ++t) {
    terms_[t].replay();
  }
  auto apply = [=](auto stride, int32_t i, int32_t j, double dx, double dy, double dz, double r2,
                   double *force, double *virial, bool both, Terms &terms) RGPOT_SC_PAIR_INLINE {
    if (!(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    double rep_term = 0.0;
    double phi_term = 0.0;
    double inv_r2 = 0.0;
    if (const double *kept = terms.next()) {
      rep_term = kept[0];
      phi_term = kept[1];
      inv_r2 = kept[2];
    } else {
      inv_r2 = 1.0 / r2;
      const double ar2 = a2 * inv_r2;
      rep_term = en * pow_ar(ar2, n_exp);
      phi_term = em * pow_ar(ar2, m_exp);
    }
    const double wi = weight[static_cast<std::size_t>(i)];
    const double wj = weight[static_cast<std::size_t>(j)];
    const double fscale = (rep_term - phi_term * (wi + wj)) * inv_r2;
    const double fx = fscale * dx;
    const double fy = fscale * dy;
    const double fz = fscale * dz;
    force[3 * i] += fx;
    force[3 * i + 1] += fy;
    force[3 * i + 2] += fz;
    if (both) {
      force[3 * j] -= fx;
      force[3 * j + 1] -= fy;
      force[3 * j + 2] -= fz;
    }
    if (virial != nullptr && j > i) {
      accumulatePairStrain(virial + 6 * static_cast<std::size_t>(i) * decltype(stride)::value,
                           fscale, dx, dy, dz);
    }
    if (atoms != nullptr) {
      tallyAtomVirial(*atoms, static_cast<std::size_t>(i), 0.5, fscale, dx, dy, dz);
      if (both) {
        tallyAtomVirial(*atoms, static_cast<std::size_t>(j), 0.5, fscale, dx, dy, dz);
      }
    }
  };
  if (!by_row) {
    Terms &lane = terms_[0];
    double *virial = s_sum;
    by_stride([&](auto stride) {
      nlist::visitHalf(neighbours_,
          0, n_atoms,
          [=, &lane](int32_t i, int32_t j, double dx, double dy, double dz,
                     double r2) RGPOT_SC_PAIR_INLINE {
            apply(stride, i, j, dx, dy, dz, r2, F, virial, true, lane);
          });
    });
  } else {
    double *virial = s_sum;
    Terms *lane_terms = terms_.data();
    // Each row belongs to one slice, so its atom's force is written once.
    neighbours_.forRowTeam(
        nt,
        [=](int t, int32_t i, int32_t j, double dx, double dy, double dz,
            double r2) RGPOT_SC_PAIR_INLINE {
          apply(std::integral_constant<std::size_t, 1>{}, i, j, dx, dy, dz, r2, F, virial, false,
                lane_terms[t]);
        });
  }
  if (rows) {
    for (std::size_t i = 0; i < n_size; ++i) {
      epair += row_e[i];
    }
    if (cfg.stress) {
      for (std::size_t i = 0; i < n_size; ++i) {
        for (std::size_t k = 0; k < 6; ++k) {
          strain[k] += row_s[6 * i + k];
        }
      }
    }
  }

  out->energy = epair + eemb;
  if (cfg.stress) {
    publishCauchyStress(out, strain, cellVolume(in.box));
  }
}

void SuttonChenPot::densityStage(const HalfList &list, double *densities) const {
  const SuttonChenConfig &cfg = m_config;
  const double cutoff = searchCutoff(cfg);
  const double cutoff2 = cutoff * cutoff;
  const double a2 = cfg.a * cfg.a;
  staged::forEachPair(list, [&](int i, int j, double, double, double, double r2, bool both) {
    if (!(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const double phi = pow_ar(a2 / r2, cfg.m);
    densities[i] += phi;
    if (both) {
      densities[j] += phi;
    }
  });
}

void SuttonChenPot::forceStage(const HalfList &list, const double *densities, double *forces,
                               StageTally *tally) const {
  const SuttonChenConfig &cfg = m_config;
  const double cutoff = searchCutoff(cfg);
  const double cutoff2 = cutoff * cutoff;
  const double a2 = cfg.a * cfg.a;
  const double ec = cfg.epsilon * cfg.c;
  std::vector<double> weight(list.nAtoms, 0.0);
  for (std::size_t i = 0; i < list.nAtoms; ++i) {
    if (densities[i] > 0.0) {
      weight[i] = ec / (2.0 * std::sqrt(densities[i]));
    }
  }
  if (tally != nullptr) {
    for (std::size_t k = 0; k < list.nRows; ++k) {
      const int i = list.rows[k];
      if (densities[i] > 0.0) {
        staged::addSite(*tally, i, -(ec * std::sqrt(densities[i])));
      }
    }
  }
  const double n_exp = static_cast<double>(cfg.n);
  const double m_exp = static_cast<double>(cfg.m);
  staged::withTally(tally, [&](auto tallied) {
    staged::forEachPair(
        list, [&](int i, int j, double dx, double dy, double dz, double r2, bool both) {
          if (!(r2 > 0.0) || r2 > cutoff2) {
            return;
          }
          const double inv_r2 = 1.0 / r2;
          const double ar2 = a2 * inv_r2;
          const double phi = pow_ar(ar2, cfg.m);
          const double rep = pow_ar(ar2, cfg.n);
          const double wi = weight[static_cast<std::size_t>(i)];
          const double wj = weight[static_cast<std::size_t>(j)];
          const double fscale = (cfg.epsilon * n_exp * rep - m_exp * phi * (wi + wj)) * inv_r2;
          staged::addPair(tallied, forces, tally, i, j, both, cfg.epsilon * rep, fscale, dx,
                          dy, dz);
        });
  });
}

template <>
RGPOT_SC_API KeptCounts Kept<SuttonChenPot>::counts() noexcept {
  return retired() ? KeptCounts{} : local().counts;
}

template <>
RGPOT_SC_API void Kept<SuttonChenPot>::release() noexcept {
  if (retired()) {
    return;
  }
  for (Slot &slot : local().slots) {
    slot.clear();
  }
}

} // namespace rgpot

namespace {

int export_force(const rgpot_sc_config *cfg, int32_t natoms,
                 const double *positions, const double *box, double *energy,
                 double *forces) {
  if (cfg == nullptr || natoms < 0 || energy == nullptr ||
      (natoms > 0 && (positions == nullptr || forces == nullptr))) {
    return 1;
  }
  const auto n = static_cast<std::size_t>(natoms);
  // The potential reads no species. Its neighbour update and pair terms
  // carry over to the next call with the same parameters and atom count.
  rgpot::KeptKey key;
  key.f64(cfg->epsilon)
      .f64(cfg->c)
      .f64(cfg->a)
      .i64(cfg->n)
      .i64(cfg->m)
      .f64(cfg->cutoff)
      .i64(cfg->periodic != 0 ? 1 : 0)
      .atoms(n, nullptr);
  try {
    *energy = rgpot::Kept<rgpot::SuttonChenPot>::with(
        key, [&] { return std::make_unique<rgpot::SuttonChenPot>(config_from_c(*cfg)); },
        [&](const rgpot::SuttonChenPot &pot) {
          rgpot::ForceOut out{};
          out.F = forces;
          out.energy = 0.0;
          out.variance = 0.0;
          out.has_stress = 0;
          const rgpot::ForceInput in{n, positions, nullptr, box};
          pot.forceImpl(in, &out);
          return out.energy;
        });
  } catch (const std::exception &) {
    return 1;
  }
  return 0;
}

} // namespace

extern "C" RGPOT_SC_API int
rgpot_sutton_chen_energy_forces(const rgpot_sc_config *cfg, int32_t natoms,
                        const double *positions, const double *box,
                        double *energy, double *forces) {
  return export_force(cfg, natoms, positions, box, energy, forces);
}
