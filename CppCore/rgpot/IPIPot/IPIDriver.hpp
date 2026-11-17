#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file IPIDriver.hpp
 * @brief i-PI driver client: answers an i-PI-protocol server with any
 * rgpot evaluation.
 *
 * i-PI socket protocol: Ceriotti, More and Manolopoulos, Comput. Phys.
 * Commun. 185, 1019 (2014), doi:10.1016/j.cpc.2013.10.027; i-PI 3.0,
 * J. Chem. Phys. 161, 062504 (2024), doi:10.1063/5.0215869.
 *
 * IPIDriver takes the client role, the opposite of IPIPot. It connects to
 * a server (i-PI, ASE SocketIOCalculator, or IPIPot) and runs the driver
 * state machine of the reference driver.py:
 * - STATUS is answered with NEEDINIT before the first INIT, HAVEDATA while
 *   a result waits for GETFORCE, and READY otherwise
 * - INIT carries a bead index and an init string, both kept
 * - POSDATA carries the cell, its inverse and the positions; the atom
 *   count must equal the species list on every message
 * - GETFORCE is answered with FORCEREADY, the energy, the atom count, the
 *   forces, the virial and an extra string
 * - EXIT ends run()
 *
 * Wire units are bohr, hartree and hartree/bohr. The cell has lattice
 * vectors in its columns; rgpot keeps them in rows, so the driver
 * transposes. The virial is the extensive tensor W = -sigma V in hartree,
 * row-major, as IPIPot reads it. Atomic numbers are not on the wire; the
 * driver is given one per atom when it is built.
 */

#include "rgpot/ForceStructs.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rgpot {

/**
 * @brief Where the driver connects, the species it serves, and how the
 * geometry is handed to the evaluation.
 *
 * address is `unix:/absolute/path`, `unix:<name>` (connects to
 * `<unix_prefix><name>`, by default `/tmp/ipi_<name>`), `tcp:<host>:<port>`
 * or `tcp:<port>` (host 127.0.0.1).
 */
struct IPIDriverConfig {
  std::string address;
  /// Atomic number of each atom, in the server's atom order.
  std::vector<int> species;
  std::string unix_prefix = "/tmp/ipi_";
  /// Seconds connect() keeps retrying a refused or missing socket.
  double connect_timeout_s = 30.0;
  /// Seconds one message may take to arrive or send. 0 waits forever.
  double timeout_s = 0.0;
  /// Interaction range in Angstrom. A periodic cell narrower than twice
  /// this is tiled, and an open box (singular cell) is padded by it.
  double cutoff = 0.0;
  /// Strain step of the central-difference virial used when the
  /// evaluation reports no stress. 0 sends a zero virial instead.
  double stress_step = 0.0;
  /// Send a small JSON object as the FORCEREADY extra string. False
  /// sends an empty string.
  bool extras_json = true;
};

/// One rgpot evaluation. Positions and cell in Angstrom, energy in eV,
/// forces in eV/Angstrom; stress, when written, is the Cauchy tensor in
/// eV/Angstrom^3 with has_stress set. ForceOut arrives zeroed.
using IPIEvaluator = std::function<void(const ForceInput &, ForceOut *)>;

/**
 * @brief Client side of the i-PI socket protocol.
 * @ingroup rgpot
 */
class IPIDriver {
public:
  /// Validates the config and the species; does not connect.
  IPIDriver(const IPIDriverConfig &config, IPIEvaluator evaluate);
  ~IPIDriver();

  IPIDriver(const IPIDriver &) = delete;
  IPIDriver &operator=(const IPIDriver &) = delete;

  /// Connects, retrying a refused or missing socket for connect_timeout_s.
  void connect();

  /// Handles one server message. Returns false after EXIT.
  bool step();

  /// Connects if needed, then handles messages until EXIT.
  void run();

  /// Closes the connection without sending anything.
  void close() noexcept;

  [[nodiscard]] const IPIDriverConfig &config() const noexcept {
    return m_config;
  }

  /// `unix:<path>` or `tcp:<host>:<port>` the driver connects to.
  [[nodiscard]] const std::string &endpoint() const noexcept {
    return m_endpoint;
  }

  [[nodiscard]] bool connected() const noexcept { return m_fd >= 0; }

  /// Bead index of the last INIT, -1 before one.
  [[nodiscard]] int32_t bead() const noexcept { return m_bead; }

  /// Init string of the last INIT.
  [[nodiscard]] const std::string &initString() const noexcept {
    return m_init;
  }

  /// Number of POSDATA messages evaluated.
  [[nodiscard]] uint64_t evaluations() const noexcept { return m_evals; }

private:
  void compute(const double h[9], const double *pos_bohr);

  IPIDriverConfig m_config;
  IPIEvaluator m_evaluate;
  std::string m_endpoint;
  std::string m_unix_path;
  bool m_tcp = false;
  std::string m_host;
  int m_port = 0;

  int m_fd = -1;
  bool m_initialised = false;
  bool m_have_data = false;
  int32_t m_bead = -1;
  std::string m_init;
  uint64_t m_evals = 0;

  double m_energy = 0.0;
  std::vector<double> m_forces;
  double m_virial[9]{};
  std::string m_extra;
};

} // namespace rgpot
