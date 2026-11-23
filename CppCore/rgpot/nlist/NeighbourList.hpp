#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Public neighbour query. Potentials and this query share updateCells and
// visitDirected. Neighbours is the cell list those calls walk. The stable
// list is a structure of arrays: a pair shell or a cluster of four,
// triclinic and partial periodic boundaries, a skin rebuild, and a chunked
// walk that does not store the pairs.
//
// Distinct Table, Builder and Walk objects may be used from different
// threads. One object is not shared. build() may start an OpenMP team
// when OMP_NUM_THREADS is set. Inside another OpenMP region that team may
// get fewer threads, and the list does not depend on how many. There is
// no mutable global. A Walk borrows positions and the box until it is
// destroyed. A View is invalidated by the next update of the table that
// owns it.

#include "rgpot/nlist/Neighbours.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace rgpot::nlist {

/// Status values shared with the C ABI. Ok is never thrown.
enum class Code : std::int32_t {
  Ok = 0,
  Abi = 1,
  Null = 2,
  Cutoff = 3,
  Box = 4,
  Shape = 5,
  Short = 6,
  Nomem = 7,
  Skin = 8
};

/// A rejected query. code() is the C ABI value. what() is a sentence
/// the caller can show. Builders throw this and leave a previous table
/// in place.
class Error : public std::runtime_error {
public:
  Error(Code code, const std::string &message)
      : std::runtime_error(message), code_(code) {}

  [[nodiscard]] Code code() const noexcept { return code_; }

private:
  Code code_;
};

/// Half keeps one direction of each pair, with the same image rule vesin
/// uses for a self image. Full keeps both directions.
enum class Shell : std::uint8_t { Half, Full };

/// One pair per interaction. `shell` chooses the direction.
struct Pairs {
  Shell shell = Shell::Half;
  friend bool operator==(Pairs a, Pairs b) noexcept { return a.shell == b.shell; }
};

/// Four consecutive central atoms, one partner, and a 4-bit mask.
/// The interactions are the full set. There is no half-list cluster:
/// the type has no shell.
struct Cluster4 {
  friend bool operator==(Cluster4, Cluster4) noexcept { return true; }
};

/// Pairs or a cluster of four. A value cannot be both.
using Shape = std::variant<Pairs, Cluster4>;

/// Strict keeps pairs with distance strictly below the cutoff. Loose keeps
/// pairs out to cutoff + skin, which is what a Verlet rebuild needs.
enum class Cut : std::uint8_t { Strict, Loose };

/// What a stored table or a chunk actually carries. Unset fields are left
/// empty rather than filled with zeros.
struct Fields {
  bool shifts = true;
  bool vectors = true;
  bool distances = true;
};

/// One neighbour query. `build` and `Builder` store the pairs. `Walk`
/// streams them. There is no mode flag that can store nothing.
struct Query {
  double cutoff = 0;
  double skin = 0;
  Shape shape = Pairs{};
  Cut cut = Cut::Strict;
  std::array<bool, 3> periodic{{true, true, true}};
  Fields fields{};
};

/// Non-owning view of a structure-of-arrays neighbour table.
/// `shift` is empty or 3 entries per pair, `vector` is empty or 3,
/// `mask` is one byte per pair (0xff for a pair list; bits 0..3 for
/// a cluster of four). The spans are invalidated by the next rebuild
/// of the table that owns them.
struct View {
  std::span<const std::int32_t> first;
  std::span<const std::int32_t> second;
  std::span<const std::int32_t> shift;
  std::span<const double> vector;
  std::span<const double> distance;
  std::span<const std::uint8_t> mask;
};

/// Caller-owned buffers for one chunk. `first` and `second` have the
/// same length, which is the maximum number of pairs this pull writes.
/// `shift` and `vector` are empty or three times that length.
struct Buffers {
  std::span<std::int32_t> first;
  std::span<std::int32_t> second;
  std::span<std::int32_t> shift;
  std::span<double> vector;
  std::span<double> distance;
  std::span<std::uint8_t> mask;
};

/// Owning table. The arrays are the storage a caller can wrap without
/// copying. Pair `p` has `vector = r_j - r_i + S H`, with `S` the three
/// entries of `shift` and `r` the positions passed to the build, also when
/// they lie whole cells outside the box. That relation holds for the
/// positions of the latest search. After a reused update only `shift`
/// follows the new positions; `vector` and `distance` still describe the
/// geometry of the last search and are stale until a full rebuild.
/// `device` is 0 for host memory. A
/// GPU fill would use the same layout and a non-zero device id; this build
/// fills host memory.
struct Table {
  std::vector<std::int32_t> first;
  std::vector<std::int32_t> second;
  std::vector<std::int32_t> shift;
  std::vector<double> vector;
  std::vector<double> distance;
  std::vector<std::uint8_t> mask;
  int device = 0;
  bool stored = false;

  [[nodiscard]] std::size_t size() const noexcept { return first.size(); }

  /// `vector` as a Fortran `(3, npairs)` array. Pair `p` is
  /// `x, y, z` at elements `3p`, `3p+1`, `3p+2`. Same memory as `vector`.
  [[nodiscard]] std::span<const double> column_major_vectors() const noexcept {
    return vector;
  }
  [[nodiscard]] View view() const noexcept {
    return View{first, second,
                shift.empty() ? std::span<const std::int32_t>{} : std::span<const std::int32_t>{shift},
                vector.empty() ? std::span<const double>{} : std::span<const double>{vector},
                distance.empty() ? std::span<const double>{} : std::span<const double>{distance},
                mask.empty() ? std::span<const std::uint8_t>{} : std::span<const std::uint8_t>{mask}};
  }
};

struct BuilderState;
struct WalkState;

/// Stored list with a skin rebuild. A later update keeps the table while
/// atom moves and the image shifts, including both signs of the stored
/// stencil, stay within the skin and the query is unchanged. Skin 0
/// rebuilds unless the folded coordinates and the cell match. Positions may
/// lie any whole number of cells outside the box. Periodic coordinates are
/// folded before the search, and the skin test measures each atom from the
/// image of its reference nearest it, so whole-cell drift and a step across
/// a face of the cell count as the step alone. The stored shifts follow the
/// positions of the latest update: a pair list rewrites them in place. That
/// is all a reused update rewrites. `vector` and `distance` are not
/// refreshed, so they describe the geometry of the last search, and a caller
/// that needs them for the new positions forms them from `shift` or forces a
/// rebuild. A
/// cluster list, whose four lanes share one shift, is searched again when
/// an atom's coordinates as given move by whole cells, as when the caller
/// wraps it. A negative or non-finite skin is rejected before that test. Throws Error
/// and keeps the previous table when the geometry is rejected. The same
/// builder is used by one thread. Two builders may update at the same time.
/// The state is behind a pointer, so the size of a Builder does not depend
/// on it. A moved-from builder reports an empty table until its next update.
class Builder {
public:
  Builder();
  ~Builder();
  Builder(const Builder &other);
  Builder &operator=(const Builder &other);
  Builder(Builder &&other) noexcept;
  Builder &operator=(Builder &&other) noexcept;

  void update(const double *positions, std::size_t n, const double *box9,
              const Query &query);

  [[nodiscard]] const Table &table() const noexcept;
  [[nodiscard]] std::uint64_t builds() const noexcept;
  /// Times an update folded the positions into the cell. Each accepted
  /// update folds once, whether it keeps the table or searches again, and
  /// a search runs on that fold.
  [[nodiscard]] std::uint64_t frames() const noexcept;
  [[nodiscard]] bool reused() const noexcept;
  /// True when the latest update kept the table and the folded coordinates
  /// and the cell were bit-identical to the previous update.
  [[nodiscard]] bool unchanged() const noexcept;

private:
  std::unique_ptr<BuilderState> state_;
};

/// Walk pairs for one central atom, or one cluster of four, at a time
/// into caller storage. The walk does not keep a list of the whole
/// system. It borrows `positions` and `box9` until it is destroyed.
/// `pull` writes at most `dest.first.size()` pairs and returns how
/// many it wrote. `done` is set when the geometry has been exhausted.
/// A requested field whose span is too short throws Error and does not
/// advance. One walk is used by one thread. The state is behind a pointer.
/// A moved-from walk throws Error with Code::Null from `pull`.
class Walk {
public:
  Walk(const double *positions, std::size_t n, const double *box9,
       const Query &query);
  ~Walk();
  Walk(const Walk &) = delete;
  Walk &operator=(const Walk &) = delete;
  Walk(Walk &&other) noexcept;
  Walk &operator=(Walk &&other) noexcept;

  std::size_t pull(Buffers dest, bool &done);

private:
  std::unique_ptr<WalkState> state_;
};

/// Fill `table` from `query`. The table holds every pair. A caller that
/// must not store the list uses Walk. Throws Error before clearing
/// `table` when the geometry is rejected. Two builds on different tables
/// may run at the same time. They share no mutable global. A build of
/// at least 8192 atoms may start an OpenMP team. A smaller build stays
/// on the calling thread.
void build(Table &table, const double *positions, std::size_t n,
           const double *box9, const Query &query);

/// The same pairs as `build`, gathered one atom at a time. A CUDA, HIP
/// or Kokkos launch gives one thread one atom and runs this walk. The
/// host `build` remains the source of truth.
void buildByAtom(Table &table, const double *positions, std::size_t n,
                 const double *box9, const Query &query);

/// Several systems that do not interact. `positions` is the concatenation
/// of every system, `count` atoms each. `box9` holds nine lattice values
/// per system, in the same order. `count[s] == 0` is an empty system.
struct Systems {
  const double *positions = nullptr;
  const std::int32_t *count = nullptr;
  std::size_t systems = 0;
  const double *box9 = nullptr;
};

/// Pairs of a `Systems` build. `first` and `second` index the concatenated
/// positions. `system[p]` is the system that owns pair `p`.
struct Batch {
  Table pairs;
  std::vector<std::int32_t> system;
};

/// One query for every system. The systems do not see each other's atoms.
/// A rejected system throws before `batch` is cleared.
void build(Batch &batch, const Systems &systems, const Query &query);

/// Bind `cells` to this geometry. `cutoff` is the pair cutoff and `skin`
/// is the Verlet buffer the kernels drop inside the cutoff. `image` is
/// Round for an orthorhombic cell, Half after coordinates have been
/// wrapped, and General for a sheared cell. `mode` stores the index when
/// it fits and streams otherwise, unless the caller asks.
inline void updateCells(Neighbours &cells, const double *positions, std::size_t n,
                        const double *box9, double cutoff, double skin, Image image, Mode mode,
                        const std::array<bool, 3> &periodic) {
  Spec spec;
  spec.cutoff = cutoff;
  spec.skin = skin;
  spec.image = image;
  spec.mode = mode;
  spec.periodic = periodic;
  cells.update(positions, n, box9, spec);
}

/// Pairs for atoms in `[begin, end)`. `dx` is `r_i - r_j` after the image,
/// the negation of the vector a stored table holds for the minimum image.
template <typename Fn>
inline void visitDirected(const Neighbours &cells, std::int64_t begin, std::int64_t end,
                          Fn &&fn) {
  cells.forDirected(begin, end, std::forward<Fn>(fn));
}

/// The pairs of `visitDirected` with `j > i` only.
template <typename Fn>
inline void visitHalf(const Neighbours &cells, std::int64_t begin, std::int64_t end, Fn &&fn) {
  cells.forHalf(begin, end, std::forward<Fn>(fn));
}

} // namespace rgpot::nlist
