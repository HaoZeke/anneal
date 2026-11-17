// MIT License
// Copyright 2023--present rgpot developers
//
// mpirun -n 4 as two calculators of two ranks: a group-collective
// potential behind the result cache. Each rank owns its cache directory.
// Only calculator 0 evaluates (an uneven batch: calculator 1 has nothing),
// so the joint decision must stay inside the calculator. Within it, rank 0
// asks again for a geometry it has cached while rank 1 asks for a new one;
// a per-rank hit decision would leave rank 1 alone in the potential's
// collective until the launcher's timeout. The joint decision makes both
// ranks compute, and a geometry both ranks hold is served from the cache.

#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <mpi.h>
#include <unistd.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>

#include "rgpot/CalculatorGroup.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/PotentialCache.hpp"

namespace {

int g_computed = 0;

/// Energy is the calculator's rank count, summed by an MPI_Allreduce on
/// its communicator inside the call: a rank that skips the call leaves its
/// peers waiting there.
class CollectivePot : public rgpot::Potential<CollectivePot> {
public:
  CollectivePot() : Potential(rgpot::PotType::LJ) {}
  [[nodiscard]] rgpot::PotCaps caps() const noexcept override {
    return {.groupCollective = true};
  }
  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return 0x636f6c6cULL;
  }
  void forceImpl(const rgpot::ForceInput &in,
                 rgpot::ForceOut *out) const override {
    MPI_Comm comm = MPI_COMM_NULL;
    rgpot::calculatorComm(&comm, sizeof(comm));
    double one = 1.0;
    MPI_Allreduce(MPI_IN_PLACE, &one, 1, MPI_DOUBLE, MPI_SUM, comm);
    out->energy = one;
    for (std::size_t k = 0; k < 3 * in.nAtoms; ++k) {
      out->F[k] = 0.0;
    }
    ++g_computed;
  }
};

int fail(int rank, const char *what) {
  std::fprintf(stderr, "collective-cache rank %d: %s\n", rank, what);
  std::fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
  return 1;
}

} // namespace

int main() {
  const rgpot::CalculatorGroup g = rgpot::bindCalculators(2);
  rgpot::finalizeMpiAtExit();
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (g.world_size != 4 || g.ranks != 2 || g.index < 0) {
    return fail(world_rank, "expected two calculators of two ranks");
  }
  if (g.index == 1) {
    // Calculator 1 gets no system in this batch and never enters the call.
    MPI_Barrier(MPI_COMM_WORLD);
    std::printf("collective-cache rank %d idle ok\n", world_rank);
    return 0;
  }
  const int rank = g.rank_in_group;

  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("rgpot_collective_cache_" + std::to_string(::getpid()) + "_" +
        std::to_string(world_rank)))
          .string();
  rocksdb::DestroyDB(path, rocksdb::Options());
  {
    rgpot::cache::PotentialCache cache(path);
    CollectivePot pot;
    pot.setCache(&cache);

    const std::vector<int> types{1, 1};
    const std::array<std::array<double, 3>, 3> box{
        {{10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}, {0.0, 0.0, 10.0}}};
    const rgpot::types::AtomMatrix a{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const rgpot::types::AtomMatrix b{{0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}};

    // Both ranks miss and compute; both caches now hold a.
    auto [e1, f1, v1] = pot(a, types, box);
    if (g_computed != 1 || e1 != 2.0) {
      return fail(rank, "first call did not compute on both ranks");
    }
    // Rank 0 holds a; rank 1 asks for b, which nobody holds.
    auto [e2, f2, v2] = pot(rank == 0 ? a : b, types, box);
    if (g_computed != 2 || e2 != 2.0) {
      return fail(rank, "split hit/miss did not compute on every rank");
    }
    // Both ranks hold a: a joint hit, served without the collective.
    auto [e3, f3, v3] = pot(a, types, box);
    if (g_computed != 2 || e3 != 2.0) {
      return fail(rank, "joint hit reached the potential");
    }

    // Batched entry point: per system, computed when any rank misses.
    std::vector<double> F(12, 0.0);
    const double flat[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
    const double *pa = a.data();
    const rgpot::types::AtomMatrix c{{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    const double *pc = (rank == 0 ? c : a).data();
    std::array<rgpot::ForceInput, 2> in{
        rgpot::ForceInput{
            .nAtoms = 2, .pos = pa, .atmnrs = types.data(), .box = flat},
        rgpot::ForceInput{
            .nAtoms = 2, .pos = pc, .atmnrs = types.data(), .box = flat}};
    std::array<rgpot::ForceOut, 2> out{
        rgpot::ForceOut{.F = F.data(), .energy = 0.0, .variance = 0.0,
                        .stress = {}, .has_stress = 0},
        rgpot::ForceOut{.F = F.data() + 6, .energy = 0.0, .variance = 0.0,
                        .stress = {}, .has_stress = 0}};
    pot.forceBatch(rgpot::ForceBatch{
        .nSystems = 2, .in = in.data(), .out = out.data()});
    if (g_computed != 3 || out[0].energy != 2.0 || out[1].energy != 2.0) {
      return fail(rank, "batched split hit/miss did not compute jointly");
    }
  }
  rocksdb::DestroyDB(path, rocksdb::Options());
  MPI_Barrier(MPI_COMM_WORLD);
  std::printf("collective-cache rank %d ok\n", world_rank);
  return 0;
}
