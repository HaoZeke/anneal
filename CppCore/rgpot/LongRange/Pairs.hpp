#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Dual cutoff lists. The short list is the local model. The long list
// is Coulomb real space and dispersion. Each list is a half list with a
// Verlet skin: nlist::Builder keeps candidates at cutoff + skin until
// atom moves or an image shift of either sign leaves the skin, and
// DualList filters them to cutoff on the current OpenMP team (skipping
// the filter when positions and the cell are unchanged). Positions may lie
// any whole number of cells outside the box. Periodic coordinates are folded
// before the search, shifts are reported for the positions as given, and the
// skin test measures each atom from the nearest image of its reference, so
// whole-cell drift or a step across a face of the cell does not rebuild the
// lists. The skin must be finite and non-negative. A list holds one of +S
// and -S for an atom's own image S; a consumer that expands it to a full list
// adds the other.

#include <cstddef>
#include <memory>
#include <vector>

namespace rgpot::lr {

struct Pair {
  int i = 0;
  int j = 0;
  double x = 0.0; // r_j + S H - r_i
  double y = 0.0;
  double z = 0.0;
  double r = 0.0;
};

class DualList {
public:
  DualList(double short_cut, double long_cut, double skin);
  ~DualList();

  DualList(const DualList &) = delete;
  DualList &operator=(const DualList &) = delete;

  // box is 9 row-major lattice vectors. A null box or periodic == false
  // builds a vacuum cell large enough that no periodic image lies inside
  // the long cutoff.
  void update(const double *pos, std::size_t n, const double *box, bool periodic);

  [[nodiscard]] const std::vector<Pair> &short_pairs() const noexcept;
  [[nodiscard]] const std::vector<Pair> &long_pairs() const noexcept;
  [[nodiscard]] bool short_rebuilt() const noexcept { return short_rebuilt_; }
  [[nodiscard]] bool long_rebuilt() const noexcept { return long_rebuilt_; }

private:
  struct Skin;
  double short_cut_;
  double long_cut_;
  double skin_;
  std::unique_ptr<Skin> short_skin_;
  std::unique_ptr<Skin> long_skin_;
  std::vector<Pair> short_;
  std::vector<Pair> long_;
  bool short_rebuilt_ = false;
  bool long_rebuilt_ = false;
  bool have_vacuum_ = false;
  double vacuum_[9]{};

  void fill(Skin &skin, double cutoff, const double *pos, std::size_t n,
            const double *box9, std::vector<Pair> *out, bool *rebuilt);
};

} // namespace rgpot::lr
