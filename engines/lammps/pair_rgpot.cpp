// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright 2023--present rgpot developers
//
// pair_style rgpot. This file includes the LAMMPS headers, so it is
// distributed under the LAMMPS license. The rest of rgpot stays MIT.
//
//   pair_style rgpot cutoff [virial auto|fd] [mode auto|gather|domain]
//   pair_coeff * * u0 psi
//   pair_coeff * * engine /path/to/libengine.so [El ...]
//   pair_coeff * * Name [key value ...] El ...
//
// The first form is the shifted 12-6 Lennard-Jones potential LJ with well
// depth u0 and zero crossing psi, truncated at the pair_style cutoff.
// The second dlopens an rgpot_engine_* plugin. The third builds the rgpot
// potential Name (a PotType name such as LJ, Morse, SuttonChen, EAMTable;
// see engines/pot_spec.cpp) with its options. In the last two forms the
// trailing words are one element symbol per LAMMPS atom type, or NULL for
// a type the style leaves out. Without them the engine form receives the
// LAMMPS type numbers as atomic numbers.
//
// Units must be metal: Angstrom, eV, eV/Angstrom. There are two ways to
// evaluate a step.
//
// Gather: the owned atoms of all ranks are gathered on rank 0 in tag
// order, the backend evaluates the whole periodic cell once, and the
// forces go back to the owners. Energy and virial are tallied on rank 0
// only. That costs one O(N) gather and scatter per step and the
// evaluation is serial, so extra ranks only distribute the integration.
//
// Domain: for a potential with a finite range (rgpot/Locality.hpp) and
// the staged evaluation of rgpot/Staged.hpp, each rank evaluates its
// owned atoms on the LAMMPS half neighbour list, the scheme of pair_style
// eam. init_one returns the potential's cutoff, so the ghosts reach one
// cutoff plus the skin past the subdomain and LAMMPS rebuilds the list
// only when atoms have moved half the skin. The density stage sums each
// atom's densities (the electron density of EAM and Sutton-Chen; a
// pairwise potential has none) over the list. Under newton_pair on, the
// entries it wrote to ghosts go back to their owners by reverse
// communication. Forward communication then gives every ghost its
// owner's completed densities, and the force stage adds the pair forces
// into the LAMMPS force array: to ghosts as well under newton_pair on,
// for LAMMPS to return to their owners as it does for any pair style.
// Every rank tallies its share of the energy and virial and LAMMPS sums
// them over ranks.
//
// mode auto takes the domain path on more than one rank when the
// potential allows it and the virial is not the strain difference; mode
// gather and mode domain force one (domain stops at init when the
// potential cannot use it).
//
// The global virial is -V sigma from the backend's Cauchy stress. When
// the backend returns no stress (an engine plugin, or a potential whose
// caps lack stress), or with virial fd, it is the central difference
// W_ij = -dE/de_ij over the six lower-triangular strains e of the cell
// and the atoms, step kStrain: twelve more evaluations on each step that
// LAMMPS asks for the virial (the setup of every run, thermo steps that
// print a pressure, and every step of a barostat).
//
// Per-atom energy and virial (compute pe/atom, stress/atom and
// centroid/stress/atom) come from potentials with forcePerAtom
// (rgpot/PerAtom.hpp): the pair potentials split each pair term half to
// each atom and EAM gives each atom its embedding term. Rank 0 fills them
// on the gathered box and they are scattered by tag like the forces. A
// potential without them, or an engine plugin, stops the run when they are
// requested. On the domain path the force stage tallies the per-atom
// arrays with the same split, and the global virial is the sum of its
// pair terms.

#include "pair.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "lammpsplugin.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"
#include "version.h"

#include "lj_eval.hpp"
#include "pot_spec.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

namespace {

enum class Backend { Engine, Potential };
enum class Mode { Auto, Gather, Domain };

constexpr int kWhyLen = 512;
constexpr double kStrain = 1.0e-5;

} // namespace

class PairRgpot : public Pair {
 public:
  PairRgpot(LAMMPS *lmp);
  ~PairRgpot() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  double cut_global = 0.0;
  Backend backend = Backend::Potential;
  std::string engine_path;
  rgpot::spec::Built built;
  // What the error names when per-atom output is requested but missing.
  std::string pot_name;
  // Every rank keeps the kernel: rank 0 evaluates the gathered box, and on
  // the domain path every rank evaluates its own atoms.
  bool per_atom = false;
  bool virial_fd = false;
  Mode mode = Mode::Auto;
  // Set by init_style.
  bool use_domain = false;
  // Atomic number per LAMMPS type, index 1..ntypes. 0 leaves the type out.
  std::vector<int> type_z;
  // Domain path, over owned and ghost atoms: atomic numbers, the
  // Locality::densities values of each atom that the comm functions move,
  // and per-atom virials in rgpot order.
  std::vector<int> atom_z;
  std::vector<double> densities;
  std::vector<double> atom_virial;

  void allocate();
  void map_elements(int narg, char **arg, int first);
  void use_potential(const char *name, const rgpot::spec::Options &options);
  int evaluate(std::size_t n, const double *xyz, const int *z, const double *box, double *forces,
               double *energy, double *stress, int *has_stress, char *why,
               const rgpot::PerAtomOut *atoms = nullptr);
  int strain_virial(std::size_t n, const double *xyz, const int *z, const double *box, double *w,
                    char *why);
  // Empty when the domain path can be used, else why not.
  std::string domain_refusal() const;
  void compute_gather();
  void compute_domain();
};

PairRgpot::PairRgpot(LAMMPS *lmp) : Pair(lmp) {
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
}

PairRgpot::~PairRgpot() {
  if (copymode) {
    return;
  }
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairRgpot::allocate() {
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  for (int i = 1; i <= n; ++i) {
    for (int j = i; j <= n; ++j) {
      setflag[i][j] = 0;
    }
  }
}

void PairRgpot::settings(int narg, char **arg) {
  if (narg < 1 || narg % 2 != 1) {
    error->all(FLERR, "pair_style rgpot takes the cutoff, then optionally virial auto|fd and "
                      "mode auto|gather|domain");
  }
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (!(cut_global > 0.0)) {
    error->all(FLERR, "pair_style rgpot cutoff must be positive");
  }
  virial_fd = false;
  mode = Mode::Auto;
  for (int k = 1; k < narg; k += 2) {
    const char *key = arg[k];
    const char *value = arg[k + 1];
    if (strcmp(key, "virial") == 0) {
      if (strcmp(value, "fd") == 0) {
        virial_fd = true;
      } else if (strcmp(value, "auto") == 0) {
        virial_fd = false;
      } else {
        error->all(FLERR, "pair_style rgpot virial is auto or fd, not {}", value);
      }
    } else if (strcmp(key, "mode") == 0) {
      if (strcmp(value, "auto") == 0) {
        mode = Mode::Auto;
      } else if (strcmp(value, "gather") == 0) {
        mode = Mode::Gather;
      } else if (strcmp(value, "domain") == 0) {
        mode = Mode::Domain;
      } else {
        error->all(FLERR, "pair_style rgpot mode is auto, gather or domain, not {}", value);
      }
    } else {
      error->all(FLERR, "pair_style rgpot has no keyword {}", key);
    }
  }
}

void PairRgpot::map_elements(int narg, char **arg, int first) {
  const int ntypes = atom->ntypes;
  type_z.assign(static_cast<std::size_t>(ntypes) + 1, 0);
  if (first < 0) {
    for (int t = 1; t <= ntypes; ++t) {
      type_z[static_cast<std::size_t>(t)] = t;
    }
  } else {
    if (narg - first != ntypes) {
      error->all(FLERR, "pair_coeff rgpot needs one element or NULL per atom type ({} types)",
                 ntypes);
    }
    for (int t = 1; t <= ntypes; ++t) {
      const char *word = arg[first + t - 1];
      if (strcmp(word, "NULL") == 0) {
        continue;
      }
      const int z = rgpot::spec::atomicNumber(word);
      if (z == 0) {
        error->all(FLERR, "pair_coeff rgpot: {} is not an element symbol", word);
      }
      type_z[static_cast<std::size_t>(t)] = z;
    }
  }
  for (int i = 1; i <= ntypes; ++i) {
    for (int j = i; j <= ntypes; ++j) {
      setflag[i][j] =
          (type_z[static_cast<std::size_t>(i)] != 0 && type_z[static_cast<std::size_t>(j)] != 0)
              ? 1
              : 0;
    }
  }
}

void PairRgpot::coeff(int narg, char **arg) {
  if (narg < 3 || strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
    error->all(FLERR, "pair_style rgpot only supports pair_coeff * * ...");
  }
  if (!allocated) {
    allocate();
  }
  const int ntypes = atom->ntypes;
  built = rgpot::spec::Built{};
  per_atom = false;
  centroidstressflag = CENTROID_NOTAVAIL;

  if (strcmp(arg[2], "engine") == 0) {
    if (narg != 4 && narg != 4 + ntypes) {
      error->all(FLERR, "pair_coeff * * engine path takes no words or one element per type");
    }
    backend = Backend::Engine;
    engine_path = arg[3];
    pot_name = "the engine " + engine_path;
    map_elements(narg, arg, narg == 4 ? -1 : 4);
    return;
  }

  if (narg == 4 && utils::is_double(arg[2]) && utils::is_double(arg[3])) {
    const double psi = utils::numeric(FLERR, arg[3], false, lmp);
    if (!(psi > 0.0)) {
      error->all(FLERR, "pair_style rgpot psi must be positive");
    }
    use_potential("LJ", {{"u0", arg[2]}, {"psi", arg[3]}});
    map_elements(narg, arg, -1);
    return;
  }

  const int first_element = narg - ntypes;
  if (first_element < 3 || (first_element - 3) % 2 != 0) {
    error->all(FLERR,
               "pair_coeff * * Name [key value ...] El ... needs key value pairs and one "
               "element or NULL for each of the {} types",
               ntypes);
  }
  rgpot::spec::Options options;
  for (int k = 3; k < first_element; k += 2) {
    options.emplace_back(arg[k], arg[k + 1]);
  }
  use_potential(arg[2], options);
  map_elements(narg, arg, first_element);
}

void PairRgpot::use_potential(const char *name, const rgpot::spec::Options &options) {
  built = rgpot::spec::build(name, options, cut_global);
  if (!built.pot) {
    error->all(FLERR, "pair_coeff rgpot: {}", built.error);
  }
  const rgpot::PotCaps caps = built.pot->caps();
  if (!caps.periodic) {
    error->all(FLERR,
               "rgpot potential {} has free boundaries; pair_style rgpot needs a "
               "periodic one",
               name);
  }
  backend = Backend::Potential;
  pot_name = std::string("the rgpot potential ") + name;
  per_atom = static_cast<bool>(built.perAtom);
  // Every force is a central pair force, so the centroid stress is vatom.
  centroidstressflag = per_atom ? CENTROID_SAME : CENTROID_NOTAVAIL;
}

std::string PairRgpot::domain_refusal() const {
  if (backend == Backend::Engine) {
    return pot_name + " is an engine, which reports no range or staged evaluation";
  }
  if (!built.locality.finite()) {
    return pot_name + " declares no finite range";
  }
  if (!built.forceStage) {
    return pot_name + " has no staged evaluation";
  }
  if (virial_fd) {
    return "virial fd differentiates the whole box";
  }
  return {};
}

void PairRgpot::init_style() {
  if (strcmp(update->unit_style, "metal") != 0) {
    error->all(FLERR, "pair_style rgpot needs units metal, not units {}", update->unit_style);
  }
  if (atom->tag_enable == 0) {
    error->all(FLERR, "pair_style rgpot needs atom IDs");
  }
  if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic) {
    error->all(FLERR, "pair_style rgpot needs boundary p p p");
  }
  const std::string refusal = domain_refusal();
  if (mode == Mode::Domain && !refusal.empty()) {
    error->all(FLERR, "pair_style rgpot mode domain: {}", refusal);
  }
  // One rank has nothing to split.
  use_domain = mode == Mode::Domain || (mode == Mode::Auto && refusal.empty() && comm->nprocs > 1);
  if (!use_domain && atom->natoms > MAXSMALLINT) {
    error->all(FLERR, "pair_style rgpot: too many atoms");
  }
  comm_forward = comm_reverse = use_domain ? built.locality.densities : 0;
  if (use_domain) {
    neighbor->add_request(this);
  }
  if (comm->me == 0) {
    if (use_domain) {
      utils::logmesg(lmp, "pair_style rgpot: domain decomposition, cutoff {:.8g}, densities {}\n",
                     built.locality.cutoff, built.locality.densities);
    } else {
      const std::string why =
          !refusal.empty() ? refusal : (mode == Mode::Auto ? "one rank" : "");
      utils::logmesg(lmp, "pair_style rgpot: gather on rank 0{}{}\n", why.empty() ? "" : "; ",
                     why);
    }
  }
}

double PairRgpot::init_one(int, int) {
  // The neighbour list of the domain path, and its ghosts, reach the
  // cutoff plus the skin; the density exchange stands in for the second
  // cutoff that a many-body force needs.
  return use_domain ? built.locality.cutoff : cut_global;
}

int PairRgpot::evaluate(std::size_t n, const double *xyz, const int *z, const double *box,
                        double *forces, double *energy, double *stress, int *has_stress,
                        char *why, const rgpot::PerAtomOut *atoms) {
  *has_stress = 0;
  switch (backend) {
  case Backend::Engine:
    return rgpot_engine_eval(engine_path.c_str(), nullptr, 0, xyz, z, n, box, forces, energy, why,
                             kWhyLen);
  case Backend::Potential: {
    std::fill(forces, forces + 3 * n, 0.0);
    rgpot::ForceInput in{n, xyz, z, box};
    rgpot::ForceOut out{forces, 0.0, 0.0, {}, 0};
    try {
      if (atoms != nullptr) {
        built.perAtom(in, &out, *atoms);
      } else {
        built.eval(in, &out);
      }
    } catch (const std::exception &e) {
      std::snprintf(why, kWhyLen, "%s", e.what());
      return 1;
    } catch (...) {
      std::snprintf(why, kWhyLen, "the rgpot potential failed with a non-standard exception");
      return 1;
    }
    *energy = out.energy;
    *has_stress = out.has_stress;
    std::copy(out.stress, out.stress + 9, stress);
    return 0;
  }
  }
  return 1;
}

int PairRgpot::strain_virial(std::size_t n, const double *xyz, const int *z, const double *box,
                             double *w, char *why) {
  // Rows and positions go to r (I + e). The entry e_ij, i >= j, keeps the
  // cell lower triangular, and -dE/de_ij is the symmetric W_ij.
  const int rows[6] = {0, 1, 2, 1, 2, 2};
  const int cols[6] = {0, 1, 2, 0, 0, 1};
  std::vector<double> sx(3 * n);
  std::vector<double> sf(3 * n);
  for (int k = 0; k < 6; ++k) {
    const int i = rows[k];
    const int j = cols[k];
    double e_pm[2] = {};
    for (int side = 0; side < 2; ++side) {
      const double e = side == 0 ? kStrain : -kStrain;
      double sbox[9];
      std::copy(box, box + 9, sbox);
      for (int r = 0; r < 3; ++r) {
        sbox[3 * r + j] += box[3 * r + i] * e;
      }
      std::copy(xyz, xyz + 3 * n, sx.begin());
      for (std::size_t a = 0; a < n; ++a) {
        sx[3 * a + j] += xyz[3 * a + i] * e;
      }
      double stress[9];
      int has_stress = 0;
      if (evaluate(n, sx.data(), z, sbox, sf.data(), &e_pm[side], stress, &has_stress, why) != 0) {
        return 1;
      }
    }
    w[k] = -(e_pm[0] - e_pm[1]) / (2.0 * kStrain);
  }
  return 0;
}

void PairRgpot::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  const bool want_atoms = eflag_atom || vflag_atom;
  if (want_atoms && !per_atom) {
    error->all(FLERR,
               "pair_style rgpot: {} has no per-atom energy or virial, so compute pe/atom "
               "and stress/atom cannot be used with it",
               pot_name);
  }
  if (use_domain) {
    compute_domain();
  } else {
    compute_gather();
  }
}

void PairRgpot::compute_domain() {
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const auto n = static_cast<std::size_t>(nall);
  const int *type = atom->type;
  atom_z.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    atom_z[i] = type_z[static_cast<std::size_t>(type[i])];
  }
  rgpot::HalfList half;
  half.nAtoms = n;
  half.nOwned = static_cast<std::size_t>(nlocal);
  half.pos = nall > 0 ? atom->x[0] : nullptr;
  half.atmnrs = atom_z.data();
  half.nRows = static_cast<std::size_t>(list->inum);
  half.rows = list->ilist;
  half.count = list->numneigh;
  half.neighbours = list->firstneigh;
  half.indexMask = NEIGHMASK;
  half.newton = force->newton_pair != 0;

  const bool want_e = eflag_global || eflag_atom;
  const bool want_v = vflag_global || vflag_atom;
  rgpot::StageTally tally;
  if (eflag_atom) {
    tally.atoms.energy = eatom;
  }
  if (vflag_atom) {
    atom_virial.assign(6 * n, 0.0);
    tally.atoms.virial = atom_virial.data();
  }
  // The exchange between the stages is collective, so every rank takes
  // part even when it holds no atoms.
  densities.assign(static_cast<std::size_t>(comm_forward) * n, 0.0);
  try {
    if (comm_forward > 0) {
      built.densityStage(half, densities.data());
      if (half.newton) {
        comm->reverse_comm(this);
      }
      comm->forward_comm(this);
    }
    built.forceStage(half, densities.data(), nall > 0 ? atom->f[0] : nullptr,
                     (want_e || want_v) ? &tally : nullptr);
  } catch (const std::exception &e) {
    error->one(FLERR, "pair_style rgpot: {}", e.what());
  }

  // rgpot Voigt xx yy zz yz xz xy to LAMMPS xx yy zz xy xz yz.
  const int to_lammps[6] = {0, 1, 2, 5, 4, 3};
  if (eflag_global) {
    eng_vdwl += tally.energy;
  }
  if (vflag_global) {
    for (int c = 0; c < 6; ++c) {
      virial[c] += tally.virial[to_lammps[c]];
    }
  }
  if (vflag_atom) {
    // Under newton_pair on, LAMMPS returns the ghosts' entries to their
    // owners as it does their forces.
    const int rows = half.newton ? nall : nlocal;
    for (int i = 0; i < rows; ++i) {
      for (int c = 0; c < 6; ++c) {
        vatom[i][c] += atom_virial[6 * static_cast<std::size_t>(i) +
                                   static_cast<std::size_t>(to_lammps[c])];
      }
    }
  }
}

int PairRgpot::pack_forward_comm(int n, int *atoms, double *buf, int, int *) {
  const auto width = static_cast<std::size_t>(comm_forward);
  double *out = buf;
  for (int k = 0; k < n; ++k) {
    const double *row = densities.data() + width * static_cast<std::size_t>(atoms[k]);
    out = std::copy(row, row + width, out);
  }
  return static_cast<int>(out - buf);
}

void PairRgpot::unpack_forward_comm(int n, int first, double *buf) {
  const auto width = static_cast<std::size_t>(comm_forward);
  std::copy(buf, buf + width * static_cast<std::size_t>(n),
            densities.data() + width * static_cast<std::size_t>(first));
}

int PairRgpot::pack_reverse_comm(int n, int first, double *buf) {
  const auto width = static_cast<std::size_t>(comm_reverse);
  const double *row = densities.data() + width * static_cast<std::size_t>(first);
  std::copy(row, row + width * static_cast<std::size_t>(n), buf);
  return static_cast<int>(width) * n;
}

void PairRgpot::unpack_reverse_comm(int n, int *atoms, double *buf) {
  const auto width = static_cast<std::size_t>(comm_reverse);
  for (int k = 0; k < n; ++k) {
    double *row = densities.data() + width * static_cast<std::size_t>(atoms[k]);
    for (std::size_t c = 0; c < width; ++c) {
      row[c] += buf[width * static_cast<std::size_t>(k) + c];
    }
  }
}

void PairRgpot::compute_gather() {
  const bool want_atoms = eflag_atom || vflag_atom;
  const int me = comm->me;
  const int nprocs = comm->nprocs;
  const int nlocal = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  const int *type = atom->type;
  const tagint *tag = atom->tag;

  std::vector<int> mine;
  mine.reserve(static_cast<std::size_t>(nlocal));
  for (int i = 0; i < nlocal; ++i) {
    if (type_z[static_cast<std::size_t>(type[i])] != 0) {
      mine.push_back(i);
    }
  }
  // Not const: the serial LAMMPS STUBS declare MPI_Gather's send buffer void *.
  int nmine = static_cast<int>(mine.size());

  std::vector<tagint> my_tag(static_cast<std::size_t>(nmine));
  std::vector<int> my_z(static_cast<std::size_t>(nmine));
  std::vector<double> my_x(3 * static_cast<std::size_t>(nmine));
  for (int k = 0; k < nmine; ++k) {
    const int i = mine[static_cast<std::size_t>(k)];
    my_tag[static_cast<std::size_t>(k)] = tag[i];
    my_z[static_cast<std::size_t>(k)] = type_z[static_cast<std::size_t>(type[i])];
    double p[3] = {x[i][0], x[i][1], x[i][2]};
    domain->remap(p);
    for (int d = 0; d < 3; ++d) {
      my_x[3 * static_cast<std::size_t>(k) + d] = p[d] - domain->boxlo[d];
    }
  }

  std::vector<int> counts(static_cast<std::size_t>(nprocs));
  MPI_Gather(&nmine, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, world);
  std::vector<int> displs(static_cast<std::size_t>(nprocs), 0);
  std::vector<int> counts3(static_cast<std::size_t>(nprocs));
  std::vector<int> displs3(static_cast<std::size_t>(nprocs));
  std::vector<int> counts6(static_cast<std::size_t>(nprocs));
  std::vector<int> displs6(static_cast<std::size_t>(nprocs));
  int total = 0;
  if (me == 0) {
    for (int r = 0; r < nprocs; ++r) {
      displs[static_cast<std::size_t>(r)] = total;
      total += counts[static_cast<std::size_t>(r)];
    }
    for (int r = 0; r < nprocs; ++r) {
      counts3[static_cast<std::size_t>(r)] = 3 * counts[static_cast<std::size_t>(r)];
      displs3[static_cast<std::size_t>(r)] = 3 * displs[static_cast<std::size_t>(r)];
      counts6[static_cast<std::size_t>(r)] = 6 * counts[static_cast<std::size_t>(r)];
      displs6[static_cast<std::size_t>(r)] = 6 * displs[static_cast<std::size_t>(r)];
    }
  }
  const auto n = static_cast<std::size_t>(total);
  std::vector<tagint> all_tag(std::max<std::size_t>(n, 1));
  std::vector<int> all_z(std::max<std::size_t>(n, 1));
  std::vector<double> all_x(std::max<std::size_t>(3 * n, 1));
  std::vector<double> all_f(std::max<std::size_t>(3 * n, 1), 0.0);
  std::vector<double> all_e(eflag_atom ? std::max<std::size_t>(n, 1) : 0, 0.0);
  std::vector<double> all_v(vflag_atom ? std::max<std::size_t>(6 * n, 1) : 0, 0.0);
  MPI_Gatherv(my_tag.data(), nmine, MPI_LMP_TAGINT, all_tag.data(), counts.data(), displs.data(),
              MPI_LMP_TAGINT, 0, world);
  MPI_Gatherv(my_z.data(), nmine, MPI_INT, all_z.data(), counts.data(), displs.data(), MPI_INT, 0,
              world);
  MPI_Gatherv(my_x.data(), 3 * nmine, MPI_DOUBLE, all_x.data(), counts3.data(), displs3.data(),
              MPI_DOUBLE, 0, world);

  // h rows: a = (xprd, 0, 0), b = (xy, yprd, 0), c = (xz, yz, zprd).
  const double box[9] = {domain->xprd, 0.0,        0.0,        domain->xy,  domain->yprd,
                         0.0,          domain->xz, domain->yz, domain->zprd};
  double energy = 0.0;
  double w[6] = {};
  int status = 0;
  char why[kWhyLen] = {};
  if (me == 0) {
    // Tag order makes the backend input independent of the decomposition.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return all_tag[a] < all_tag[b]; });
    std::vector<double> xyz(3 * n);
    std::vector<int> z(n);
    for (std::size_t k = 0; k < n; ++k) {
      z[k] = all_z[order[k]];
      for (int d = 0; d < 3; ++d) {
        xyz[3 * k + d] = all_x[3 * order[k] + d];
      }
    }
    std::vector<double> forces(3 * n, 0.0);
    std::vector<double> atom_e(eflag_atom ? n : 0);
    std::vector<double> atom_v(vflag_atom ? 6 * n : 0);
    rgpot::PerAtomOut atoms{eflag_atom ? atom_e.data() : nullptr,
                            vflag_atom ? atom_v.data() : nullptr};
    double stress[9] = {};
    int has_stress = 0;
    status = evaluate(n, xyz.data(), z.data(), box, forces.data(), &energy, stress, &has_stress,
                      why, want_atoms ? &atoms : nullptr);
    if (status == 0 && vflag_global) {
      if (has_stress != 0 && !virial_fd) {
        // W = -dE/d(strain) = -V sigma, LAMMPS order xx yy zz xy xz yz.
        const double volume = domain->xprd * domain->yprd * domain->zprd;
        const int voigt[6] = {0, 4, 8, 1, 2, 5};
        for (int k = 0; k < 6; ++k) {
          w[k] = -volume * stress[voigt[k]];
        }
      } else {
        status = strain_virial(n, xyz.data(), z.data(), box, w, why);
      }
    }
    // rgpot Voigt xx yy zz yz xz xy to LAMMPS xx yy zz xy xz yz.
    const int to_lammps[6] = {0, 1, 2, 5, 4, 3};
    for (std::size_t k = 0; k < n && status == 0; ++k) {
      for (int d = 0; d < 3; ++d) {
        all_f[3 * order[k] + d] = forces[3 * k + d];
      }
      if (eflag_atom) {
        all_e[order[k]] = atom_e[k];
      }
      if (vflag_atom) {
        for (int c = 0; c < 6; ++c) {
          all_v[6 * order[k] + c] = atom_v[6 * k + to_lammps[c]];
        }
      }
    }
  }
  MPI_Bcast(&status, 1, MPI_INT, 0, world);
  if (status != 0) {
    MPI_Bcast(why, kWhyLen, MPI_CHAR, 0, world);
    why[kWhyLen - 1] = '\0';
    error->all(FLERR, "pair_style rgpot: {}", why[0] != '\0' ? why : "evaluation failed");
  }

  std::vector<double> my_f(std::max<std::size_t>(3 * static_cast<std::size_t>(nmine), 1));
  MPI_Scatterv(all_f.data(), counts3.data(), displs3.data(), MPI_DOUBLE, my_f.data(), 3 * nmine,
               MPI_DOUBLE, 0, world);
  for (int k = 0; k < nmine; ++k) {
    const int i = mine[static_cast<std::size_t>(k)];
    for (int d = 0; d < 3; ++d) {
      f[i][d] += my_f[3 * static_cast<std::size_t>(k) + d];
    }
  }
  if (eflag_atom) {
    std::vector<double> my_e(std::max(nmine, 1));
    MPI_Scatterv(all_e.data(), counts.data(), displs.data(), MPI_DOUBLE, my_e.data(), nmine,
                 MPI_DOUBLE, 0, world);
    for (int k = 0; k < nmine; ++k) {
      eatom[mine[static_cast<std::size_t>(k)]] += my_e[static_cast<std::size_t>(k)];
    }
  }
  if (vflag_atom) {
    std::vector<double> my_v(std::max<std::size_t>(6 * static_cast<std::size_t>(nmine), 1));
    MPI_Scatterv(all_v.data(), counts6.data(), displs6.data(), MPI_DOUBLE, my_v.data(),
                 6 * nmine, MPI_DOUBLE, 0, world);
    for (int k = 0; k < nmine; ++k) {
      const int i = mine[static_cast<std::size_t>(k)];
      for (int c = 0; c < 6; ++c) {
        vatom[i][c] += my_v[6 * static_cast<std::size_t>(k) + static_cast<std::size_t>(c)];
      }
    }
  }

  if (me == 0) {
    if (eflag_global) {
      eng_vdwl += energy;
    }
    if (vflag_global) {
      for (int k = 0; k < 6; ++k) {
        virial[k] += w[k];
      }
    }
  }
}

static Pair *rgpot_creator(LAMMPS *lmp) {
  return new PairRgpot(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  auto *register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  lammpsplugin_t plugin;
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "rgpot";
  plugin.info = "rgpot potentials, Lennard-Jones, or an rgpot_engine plugin";
  plugin.author = "Rohit Goswami (rgoswami@ieee.org)";
  plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(&rgpot_creator);
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
