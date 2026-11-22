#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file MEAMModel.hpp
 * @brief Parameters and tabulated pair term of the modified embedded-atom
 *        method.
 *
 * M. I. Baskes, Phys. Rev. B 46, 2727 (1992), doi:10.1103/PhysRevB.46.2727,
 * is the first-neighbour form. B.-J. Lee and M. I. Baskes, Phys. Rev. B 62,
 * 8564 (2000), doi:10.1103/PhysRevB.62.8564, adds the second neighbours.
 *
 * The energy is
 *
 *   E = sum_i F_a(rhobar_i) + 1/2 sum_{i != j} phi_ab(r_ij) S_ij,
 *   F(rhobar) = A E_c rhobar ln(rhobar),
 *   rhobar = rho0 G(Gamma) / rho_ref,
 *   Gamma = (t1 rho1 + t2 rho2 + t3 rho3) / rho0^2.
 *
 * S_ij is the product of the radial cutoff of r_ij and the many-body
 * screening of every third atom k (Cmin and Cmax). The partial densities
 * rho0 to rho3 are those of the Baskes paper and rho_ref is the density of
 * the element's reference structure. phi_ab is not a parameter: it is
 * whatever makes the energy of the reference structure equal the Rose
 * universal equation of state, so the model builds it from the reference
 * structure at construction. With nn2 set for a pair the second neighbours
 * of the reference structure are screened by S and enter both rho0 and the
 * pair term through the recursion of equation (21) of the 2000 paper,
 *
 *   phi(r) = phi_ref(r) - sum_{n >= 1} (-Z2 S / Z1)^n phi(a^n r),
 *
 * summed to n = 10 and stopped at the first term below 1e-20, as the
 * LAMMPS MEAM package does. The tabulated pair term is that package's:
 * 1000 points to 1.1 rc and a cubic through the five point central
 * difference slope.
 *
 * The parameter names, defaults and keyword set are those of the LAMMPS
 * library and parameter files (pair_style meam), so the same files drive
 * both programs. Energies are in eV and lengths in Angstrom.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rgpot {

/// Reference structures, in the order of the LAMMPS MEAM package.
enum class MEAMLattice : int {
  FCC,
  BCC,
  HCP,
  DIM,
  DIA,
  DIA3,
  B1,
  C11,
  L12,
  B2,
  CH4,
  LIN,
  ZIG,
  TRI,
  SC
};

/// Lattice from its library or parameter file name ("fcc", "b2", ...).
/// With `single` only the structures an element can have are accepted.
/// Returns false for a name it does not know.
bool meam_lattice_from_name(const std::string &name, bool single,
                            MEAMLattice *out);

/// First neighbours of a reference structure.
int meam_coordination(MEAMLattice lattice);

/// One library entry. Lengths in Angstrom, energy in eV.
struct MEAMElement {
  std::string name;
  MEAMLattice lattice = MEAMLattice::FCC;
  int z = 0;           //!< Atomic number, which selects the element of an atom.
  double mass = 0.0;   //!< Atomic weight, g/mol.
  double alpha = 0.0;  //!< sqrt(9 Omega B / Ec).
  double beta[4] = {}; //!< Decay lengths of rho0 to rho3.
  double alat = 0.0;   //!< Lattice constant of the reference structure.
  double esub = 0.0;   //!< Cohesive energy Ec.
  double asub = 0.0;   //!< Embedding constant A.
  double t[4] = {};    //!< t0 (always 1) to t3.
  double rho0 = 0.0;   //!< Density scale.
  int ibar = 0;        //!< Selects G(Gamma), see MEAMModel::G.
};

/**
 * @brief Everything a MEAM library and parameter file set.
 *
 * The constructor is meam_setup_global of the LAMMPS package: it takes the
 * element table and sets every default. set() is one line of a parameter
 * file. Indices are zero based. MEAMModel finishes the job (the off
 * diagonal mixing rules, the reference densities and the pair tables).
 */
class MEAMParameters {
public:
  /// Parameter file keywords, in the order of the LAMMPS package.
  enum class Key : int {
    Ec,
    alpha,
    rho0,
    delta,
    lattce,
    attrac,
    repuls,
    nn2,
    Cmin,
    Cmax,
    rc,
    delr,
    augt1,
    gsmooth_factor,
    re,
    ialloy,
    mixture_ref_t,
    erose_form,
    zbl,
    emb_lin_neg,
    bkgd_dyn,
    theta
  };

  /// Keyword spelling of a Key, e.g. "Cmin".
  static const char *keyword(Key key);
  /// Key of a spelling. Returns false for an unknown keyword.
  static bool key_from_keyword(const std::string &word, Key *out);

  /// Throws std::invalid_argument for an empty table, a lattice that no
  /// element can have, t0 != 1, or rho0 <= 0.
  explicit MEAMParameters(std::vector<MEAMElement> elements);

  [[nodiscard]] std::size_t size() const noexcept { return m_elements.size(); }
  [[nodiscard]] const std::vector<MEAMElement> &elements() const noexcept {
    return m_elements;
  }

  /**
   * @brief Apply one parameter file line.
   * @param value The value. For Key::lattce, the MEAMLattice as a number.
   * @param index Zero based element indices: 2 for the pair keywords, 3
   *        for Cmin and Cmax, 1 for rho0, none for the global ones.
   * @throws std::invalid_argument when too few indices are given, an index
   *         is outside the element table, or a value is not allowed (rho0
   *         must be positive).
   */
  void set(Key key, double value, const std::vector<int> &index);

  /// Convenience for the lattice of a pair.
  void set_lattice(int i, int j, MEAMLattice lattice) {
    set(Key::lattce, static_cast<double>(lattice), {i, j});
  }

private:
  friend class MEAMModel;

  std::size_t pair(int i, int j) const {
    return static_cast<std::size_t>(i) * m_elements.size() +
           static_cast<std::size_t>(j);
  }
  std::size_t triple(int i, int j, int k) const {
    return pair(i, j) * m_elements.size() + static_cast<std::size_t>(k);
  }

  std::vector<MEAMElement> m_elements;
  std::vector<double> m_ec, m_re, m_alpha, m_delta, m_attrac, m_repuls;
  std::vector<double> m_stheta, m_ctheta;
  std::vector<int> m_lattice, m_nn2, m_zbl;
  std::vector<double> m_cmin, m_cmax;
  double m_rc = 4.0;
  double m_delr = 0.1;
  double m_gsmooth = 99.0;
  int m_augt1 = 1;
  int m_ialloy = 0;
  int m_mix_ref_t = 0;
  int m_erose_form = 0;
  int m_emb_lin_neg = 0;
  int m_bkgd_dyn = 0;
};

/**
 * @brief The completed model: mixed parameters, reference densities and the
 *        tabulated pair terms.
 *
 * Immutable after construction, so one model may serve many potentials.
 */
class MEAMModel {
public:
  /// One interval of a pair table. c[0] to c[3] are the value and the
  /// coefficients of the cubic in the local coordinate p on [0, 1], and
  /// c[4] to c[6] those of its derivative in r (the LAMMPS phirar to
  /// phirar6 arrays).
  struct Knot {
    double c[7];
  };

  /// Builds the tables. Throws std::invalid_argument when a pair table
  /// cannot be built: an unsupported ibar, nn2 set for a c11 pair, or a
  /// non positive rc or delr.
  explicit MEAMModel(MEAMParameters parameters);

  [[nodiscard]] std::size_t size() const noexcept { return m_n; }
  [[nodiscard]] const std::vector<MEAMElement> &elements() const noexcept {
    return m_p.m_elements;
  }

  [[nodiscard]] double rc() const noexcept { return m_p.m_rc; }
  [[nodiscard]] double delr() const noexcept { return m_p.m_delr; }
  /// Distance within which a neighbour can still screen a pair: rc times
  /// the root of the largest ellipse bound.
  [[nodiscard]] double screening_radius() const noexcept { return m_radius; }

  [[nodiscard]] int ialloy() const noexcept { return m_p.m_ialloy; }
  [[nodiscard]] int mixture_ref_t() const noexcept { return m_p.m_mix_ref_t; }
  [[nodiscard]] int bkgd_dyn() const noexcept { return m_p.m_bkgd_dyn; }
  [[nodiscard]] double cmin(int i, int j, int k) const {
    return m_p.m_cmin[m_p.triple(i, j, k)];
  }
  [[nodiscard]] double cmax(int i, int j, int k) const {
    return m_p.m_cmax[m_p.triple(i, j, k)];
  }
  /// Squared-distance bound of the screening ellipse for the pair i, j.
  [[nodiscard]] double ebound(int i, int j) const { return m_ebound[m_p.pair(i, j)]; }

  [[nodiscard]] double re(int i) const { return m_p.m_re[m_p.pair(i, i)]; }
  [[nodiscard]] double ec(int i) const { return m_p.m_ec[m_p.pair(i, i)]; }
  [[nodiscard]] const MEAMElement &element(int i) const {
    return m_p.m_elements[static_cast<std::size_t>(i)];
  }
  /// Background density of the element's reference structure.
  [[nodiscard]] double rho_ref(int i) const {
    return m_rho_ref[static_cast<std::size_t>(i)];
  }
  /// Beta0 to beta3 and t1 to t3 after the augt1 correction.
  [[nodiscard]] double beta(int i, int l) const {
    return m_p.m_elements[static_cast<std::size_t>(i)].beta[l];
  }
  [[nodiscard]] double t(int i, int l) const {
    return m_t[static_cast<std::size_t>(i)][l];
  }
  [[nodiscard]] double rho0(int i) const {
    return m_p.m_elements[static_cast<std::size_t>(i)].rho0;
  }
  [[nodiscard]] int ibar(int i) const {
    return m_p.m_elements[static_cast<std::size_t>(i)].ibar;
  }
  [[nodiscard]] MEAMLattice lattice(int i, int j) const {
    return static_cast<MEAMLattice>(m_p.m_lattice[m_p.pair(i, j)]);
  }
  [[nodiscard]] double stheta(int i, int j) const {
    return m_p.m_stheta[m_p.pair(i, j)];
  }
  [[nodiscard]] double ctheta(int i, int j) const {
    return m_p.m_ctheta[m_p.pair(i, j)];
  }

  /// Index of the pair i, j in the table, with i <= j order irrelevant.
  [[nodiscard]] std::size_t pair_index(int i, int j) const {
    return m_eltind[m_p.pair(i, j)];
  }

  /// phi_ab(r) and its derivative from the cubic table. r must be below
  /// 1.1 rc. The cubic is the one LAMMPS evaluates, so the derivative is
  /// the exact slope of the value.
  void phi(std::size_t pair, double r, double *value, double *slope) const {
    double q = r * m_rdr;
    std::size_t k = static_cast<std::size_t>(q);
    if (k > m_nr - 2) {
      k = m_nr - 2;
    }
    double p = q - static_cast<double>(k);
    if (p > 1.0) {
      p = 1.0;
    }
    const Knot &n = m_table[pair * m_nr + k];
    *value = ((n.c[3] * p + n.c[2]) * p + n.c[1]) * p + n.c[0];
    *slope = (n.c[6] * p + n.c[5]) * p + n.c[4];
  }

  /// Pair function straight from the reference structure, with no table.
  /// Used by the tests that compare it with the Rose energy.
  [[nodiscard]] double phi_reference(double r, int a, int b) const {
    return phi_meam(r, a, b);
  }

  /// G(Gamma), the angular function of the embedding argument, for the
  /// selector ibar. Returns false for a selector it does not define.
  [[nodiscard]] bool G(double gamma, int ibar, double *value, double *slope) const;

  /// F(rhobar) and F'(rhobar).
  [[nodiscard]] double embedding(int element, double rhobar, double *slope) const;

  /// Shape factors s1, s2, s3 of a reference structure.
  static void shape_factors(MEAMLattice lattice, double stheta, double ctheta,
                            double s[3]);

  /// Rose energy of the pair i, j at separation r.
  [[nodiscard]] double rose(double r, int i, int j) const;

  /// Hash of every number that decides the model, for paramsKey().
  [[nodiscard]] std::uint64_t fingerprint() const noexcept { return m_fingerprint; }

private:
  void mix_pairs();
  void reference_density();
  void build_pair_tables();
  double phi_meam(double r, int a, int b) const;
  double phi_meam_series(double scrn, int z1, int z2, int a, int b, double r,
                         double arat) const;
  void reference_densities(double r, int a, int b, double *rho01, double *rho11,
                           double *rho21, double *rho31, double *rho02,
                           double *rho12, double *rho22, double *rho32) const;
  void reference_t(double *t11, double *t21, double *t31, double *t12,
                   double *t22, double *t32, double ta1, double ta2, double ta3,
                   double tb1, double tb2, double tb3, double r, int a, int b,
                   MEAMLattice latt) const;
  /// What the second neighbour recursion adds to phi_meam at r for a pair
  /// with nn2 set (equation 21 of the 2000 paper).
  double second_neighbour_series(double r, int a, int b) const;
  double screening(int i, int j, int k, double c) const;
  int second_neighbours(MEAMLattice latt, double cmin, double cmax,
                        double stheta, double *a, double *s) const;
  int second_neighbours_b2nn(MEAMLattice latt, double cmin, double cmax,
                             double *s) const;

  MEAMParameters m_p;
  std::size_t m_n = 0;
  std::vector<double> m_ebound;
  std::vector<double> m_rho_ref;
  std::vector<std::array<double, 4>> m_t;
  std::vector<std::size_t> m_eltind;
  std::size_t m_nr = 1000;
  double m_dr = 0.0;
  double m_rdr = 0.0;
  double m_radius = 0.0;
  std::vector<Knot> m_table;
  std::uint64_t m_fingerprint = 0;
};

} // namespace rgpot
