// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the standalone Cap'n Proto potential server.
 *
 * This file implements a basic RPC server which exposes toy potentials over a
 * network interface. It utilizes the @c EzRpcServer for handling requests.
 */

#include <algorithm>
#include <sstream>
#include <vector>

#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <kj/debug.h>

#ifdef RGPOT_POTSERV_MPI
#include "tools/MpiPefSession.hpp"
#endif

namespace rgpot::tools {
class MpiPefSession;
}

#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif // RGPOT_HAS_FORTRAN_POTS

#ifdef RGPOT_HAS_XTB
#include "rgpot/XTBPot/XTBPot.hpp"
#endif // RGPOT_HAS_XTB

#ifdef RGPOT_HAS_TBLITE
#include "rgpot/TBLitePot/TBLitePot.hpp"
#endif // RGPOT_HAS_TBLITE

#ifdef RGPOT_HAS_DFTD3
#include "rgpot/D3Pot/D3Pot.hpp"
#endif // RGPOT_HAS_DFTD3

#ifdef RGPOT_HAS_DFTD4
#include "rgpot/D4Pot/D4Pot.hpp"
#endif // RGPOT_HAS_DFTD4

#ifdef RGPOT_HAS_METATOMIC
#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#endif // RGPOT_HAS_METATOMIC

#ifdef RGPOT_HAS_UMA
#include "rgpot/UmaPot/UmaPot.hpp"
#endif
#ifdef RGPOT_HAS_SKALA
#include "rgpot/SkalaPot/SkalaPot.hpp"
#endif // RGPOT_HAS_UMA

#ifdef RGPOT_HAS_CP2K
#include "rgpot/CP2KPot/CP2KPot.hpp"
#endif
#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/abi/Handshake.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#ifdef RGPOT_HAS_IPI
#include "rgpot/IPIPot/IPIPot.hpp"
#endif
#include "rgpot/rpc/MorseConfigure.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/types/adapters/capnp/capnp_adapter.hpp"
#include "rgpot/units.hpp"

#ifdef RGPOT_HAS_CP2K
constexpr const char *kOwnArmBackends = "NWChem, CPMD, CP2K and Morse";
#else
constexpr const char *kOwnArmBackends = "NWChem, CPMD and Morse";
#endif
#ifdef RGPOT_POTSERV_MPI
#ifdef RGPOT_HAS_CP2K
constexpr const char *kCollectiveBackends = "NWChem, CPMD or CP2K";
#else
constexpr const char *kCollectiveBackends = "NWChem or CPMD";
#endif
#endif

/**
 * @class GenericPotImpl
 * @brief Server implementation for the Potential RPC interface.
 *
 * This class wraps a polymorphic @c PotentialBase instance and dispatches
 * RPC calculate requests to the underlying physics engine.
 */
class GenericPotImpl final : public Potential::Server {
private:
  std::unique_ptr<rgpot::PotentialBase>
      m_potential; //!< The polymorphic potential engine.
  /// Optional typed handle when backend is NWChem (for configure()).
  rgpot::NWChemPot *m_nwchem = nullptr;
  /// Optional typed handle when backend is CPMD (for configure()).
  rgpot::CPMDPot *m_cpmd = nullptr;
  /// Set when the backend is Morse; configure() replaces the pot it names.
  rgpot::MorsePot *m_morse = nullptr;
#ifdef RGPOT_HAS_IPI
  /// Set when the backend is the i-PI socket server. calculate copies the
  /// Cauchy stress stashed by the last force call.
  rgpot::IPIPot *m_ipi = nullptr;
#endif
#ifdef RGPOT_HAS_CP2K
  /// Optional typed handle when backend is CP2K (for configure()).
  rgpot::CP2KPot *m_cp2k = nullptr;
#endif
  rgpot::tools::MpiPefSession *m_pef = nullptr;

public:
  /**
   * @brief Constructor for GenericPotImpl.
   * @param pot Ownership of a PotentialBase derived object.
   */
  GenericPotImpl(std::unique_ptr<rgpot::PotentialBase> pot)
      : m_potential(std::move(pot)),
        m_morse(dynamic_cast<rgpot::MorsePot *>(m_potential.get()))
#ifdef RGPOT_HAS_IPI
        ,
        m_ipi(dynamic_cast<rgpot::IPIPot *>(m_potential.get()))
#endif
  {
  }

  /**
   * @brief Constructor retaining an NWChemPot pointer for configure().
   * Members initialize in declaration order (m_potential then m_nwchem).
   */
  GenericPotImpl(std::unique_ptr<rgpot::NWChemPot> pot,
                 rgpot::tools::MpiPefSession *pef = nullptr)
      : m_potential(std::move(pot)),
        m_nwchem(static_cast<rgpot::NWChemPot *>(m_potential.get())), m_pef(pef) {}

  GenericPotImpl(std::unique_ptr<rgpot::CPMDPot> pot,
                 rgpot::tools::MpiPefSession *pef = nullptr)
      : m_potential(std::move(pot)),
        m_cpmd(static_cast<rgpot::CPMDPot *>(m_potential.get())), m_pef(pef) {}

#ifdef RGPOT_HAS_CP2K
  GenericPotImpl(std::unique_ptr<rgpot::CP2KPot> pot,
                 rgpot::tools::MpiPefSession *pef = nullptr)
      : m_potential(std::move(pot)),
        m_cp2k(static_cast<rgpot::CP2KPot *>(m_potential.get())), m_pef(pef) {}
#endif

  /**
   * @brief Reports what this server speaks, so a client can refuse an
   * incompatible peer before the first calculate call.
   */
  kj::Promise<void> getCapabilities(GetCapabilitiesContext context) override {
    auto caps = context.getResults().initCapabilities();
    caps.setBackendName("rgpot");
    caps.setBackendVersion(RGPOT_BUILD_VERSION);
    caps.setAvailable(true);
    auto ops = caps.initOperations(2);
    ops.set(0, ::Capabilities::Operation::ENERGY);
    ops.set(1, ::Capabilities::Operation::FORCES);
    rgpot::abi::fill_compatibility(caps);
    return kj::READY_NOW;
  }

  /**
   * @details
   * This method performs the following translation steps:
   * 1. Extracts the @c ForceInput (fip) from the RPC context.
   * 2. Validates the size of the atomic number list.
   * 3. Converts Cap'n Proto lists into native @c AtomMatrix and @c std::vector
   * types.
   * 4. Executes the calculation via the @c PotentialBase virtual operator.
   * 5. Populates the @c PotentialResult with energy and force data.
   *
   * @param context The Cap'n Proto RPC call context.
   * @return An asynchronous promise for completion.
   */
  kj::Promise<void> calculate(CalculateContext context) override {
    auto fip = context.getParams().getFip();
    const size_t numAtoms = fip.getPos().size() / 3;

    KJ_REQUIRE(fip.getAtmnrs().size() == numAtoms, "AtomNumbers size mismatch");

    // Unit conversion factors (caller units -> internal angstrom/eV)
    std::string lengthUnit = fip.getLengthUnit();
    std::string energyUnit = fip.getEnergyUnit();
    double len_to_angstrom = rgpot::units::unit_conversion_factor(lengthUnit, "angstrom");
    double ev_to_caller = rgpot::units::unit_conversion_factor("eV", energyUnit);
    // Force unit: energy/length in caller's system
    double force_to_caller = ev_to_caller / len_to_angstrom;

    rgpot::types::AtomMatrix nativePositions =
        rgpot::types::adapt::capnp::convertPositionsFromCapnp(fip.getPos(),
                                                              numAtoms);
    // Convert positions from caller's length unit to angstrom
    if (len_to_angstrom != 1.0) {
      for (size_t i = 0; i < numAtoms * 3; ++i) {
        nativePositions.data()[i] *= len_to_angstrom;
      }
    }

    std::vector<int> nativeAtomTypes =
        rgpot::types::adapt::capnp::convertAtomNumbersFromCapnp(
            fip.getAtmnrs());
    std::array<std::array<double, 3>, 3> nativeBoxMatrix =
        rgpot::types::adapt::capnp::convertBoxMatrixFromCapnp(fip.getBox());
    // Convert box from caller's length unit to angstrom
    if (len_to_angstrom != 1.0) {
      for (auto &row : nativeBoxMatrix)
        for (auto &val : row)
          val *= len_to_angstrom;
    }

    // Potential always computes in eV/angstrom
    auto evaluate = [&]() {
#ifdef RGPOT_POTSERV_MPI
      if (m_pef) {
        try {
          return m_pef->calculate(nativePositions, nativeAtomTypes, nativeBoxMatrix);
        } catch (const std::exception &error) {
          throw KJ_EXCEPTION(FAILED, error.what());
        }
      }
#endif
      return (*m_potential)(nativePositions, nativeAtomTypes, nativeBoxMatrix);
    };
    auto [energy, forces, variance] = evaluate();
    (void)variance;

    // Convert results to caller's units
    auto result = context.getResults();
    auto pres = result.initResult();
    pres.setEnergy(energy * ev_to_caller);

    auto forcesList = pres.initForces(numAtoms * 3);
    if (force_to_caller != 1.0) {
      for (size_t i = 0; i < numAtoms * 3; ++i) {
        forces.data()[i] *= force_to_caller;
      }
    }
    rgpot::types::adapt::capnp::populateForcesToCapnp(forcesList, forces);

#ifdef RGPOT_HAS_IPI
    if (m_ipi != nullptr) {
      double stress[9];
      if (m_ipi->copyStress(stress)) {
        // sigma_caller = sigma_eV/A^3 * (eV -> caller) * (A -> caller length)^3
        const double stress_to_caller = ev_to_caller * len_to_angstrom *
                                        len_to_angstrom * len_to_angstrom;
        auto stressList = pres.initStress(9);
        for (unsigned int i = 0; i < 9; ++i) {
          stressList.set(i, stress[i] * stress_to_caller);
        }
      }
    }
#endif

    return kj::READY_NOW;
  }

  /// NWChem, CPMD and CP2K take every arm through setPotentialConfig.
  [[nodiscard]] bool appliesOwnArm() const {
#ifdef RGPOT_HAS_CP2K
    if (m_cp2k)
      return true;
#endif
    return m_nwchem || m_cpmd;
  }

  bool applyOwnArm(const ::PotentialConfig::Reader &cfg, std::string *msg) {
#ifdef RGPOT_HAS_CP2K
    if (m_cp2k)
      return m_cp2k->setPotentialConfig(cfg, msg);
#endif
    return m_nwchem ? m_nwchem->setPotentialConfig(cfg, msg)
                    : m_cpmd->setPotentialConfig(cfg, msg);
  }

  kj::Promise<void> configure(ConfigureContext context) override {
    auto cfg = context.getParams().getConfig();
    auto results = context.getResults();
    if (m_morse) {
      auto applied = rgpot::rpc::configureMorse(cfg, *m_morse);
      if (applied.replacement) {
        m_morse = applied.replacement.get();
        m_potential = std::move(applied.replacement);
      }
      results.setOk(applied.ok);
      results.setMessage(applied.message);
      return kj::READY_NOW;
    }
    if (!appliesOwnArm()) {
      if (cfg.which() == ::PotentialConfig::NONE) {
        results.setOk(true);
        results.setMessage("no-op (rgpot params: none)");
        return kj::READY_NOW;
      }
      results.setOk(false);
      results.setMessage("configure() with the " +
                         rgpot::rpc::potentialConfigArm(cfg) +
                         " arm is not supported by this backend; only " +
                         kOwnArmBackends + " apply their own arm");
      return kj::READY_NOW;
    }
    std::string msg;
    bool ok = false;
#ifdef RGPOT_POTSERV_MPI
    if (m_pef) {
      try {
        msg = m_pef->configure(cfg);
        ok = true;
      } catch (const std::exception &error) {
        msg = error.what();
      }
    } else
#endif
    {
      ok = applyOwnArm(cfg, &msg);
    }
    results.setOk(ok);
    results.setMessage(msg);
    return kj::READY_NOW;
  }
};

/**
 * @details
 * The main entry point handles command-line arguments to specify the
 * network port and the potential type. It instantiates the requested
 * physics engine and blocks until the server is terminated.
 *
 * # Usage
 * @code
 * ./potserv <port> <PotentialType>
 * @endcode
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, 1 on initialization failure.
 */
int runServer(int argc, char *argv[]
#ifdef RGPOT_POTSERV_MPI
              , const rgpot::tools::MpiHost &host
#endif
) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <port> <PotentialType>" << std::endl;
    std::cerr << "  Available PotentialTypes: CuH2, LJ, LJCluster, Morse, ZBL"
#ifdef RGPOT_HAS_IPI
              << ", IPI:unix:<path>, IPI:tcp:<host>:<port>"
#endif
#ifdef RGPOT_HAS_XTB
              << ", XTB, GFNFF, GFN0xTB, GFN1xTB"
#endif
#ifdef RGPOT_HAS_TBLITE
              << ", TBLite, TBLiteGFN1, TBLiteIPEA1"
#endif
#ifdef RGPOT_HAS_DFTD3
              << ", D3"
#endif
#ifdef RGPOT_HAS_DFTD4
              << ", D4"
#endif
#ifdef RGPOT_HAS_METATOMIC
              << ", Metatomic:<model_path>"
#endif
#ifdef RGPOT_HAS_UMA
              << ", Uma:<model_path>, Uma:<model_path>:<task>"
#endif
#ifdef RGPOT_HAS_SKALA
              << ", Skala, Skala:<basis>"
#endif
              << ", NWChem[:basis[:theory[:xc]]]"
              << ", CPMD"
#ifdef RGPOT_HAS_CP2K
              << ", CP2K"
#endif
              << std::endl;
    return 1;
  }

  int port = 12345;
  try {
    port = std::stoi(argv[1]);
  } catch (const std::exception &e) {
    std::cerr << "Invalid port argument '" << argv[1]
              << "'. Using default 12345." << std::endl;
  }

  std::string pot_type = argv[2];
#ifdef RGPOT_POTSERV_MPI
  host.collective([&] {
    bool collective_backend = pot_type == "NWChem" ||
        pot_type.rfind("NWChem:", 0) == 0 || pot_type == "CPMD";
#ifdef RGPOT_HAS_CP2K
    collective_backend = collective_backend || pot_type == "CP2K";
#endif
    if (host.size() > 1 && !collective_backend)
      throw std::runtime_error(std::string("multi-rank RPC requires ") +
                               kCollectiveBackends);
  });
#endif
  std::unique_ptr<rgpot::PotentialBase> potential_to_use;

#ifdef RGPOT_HAS_FORTRAN_POTS
  if (pot_type == "CuH2") {
    std::cout << "Loading CuH2 potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::fortranpots::CuH2Pot>();
  } else
#endif // RGPOT_HAS_FORTRAN_POTS
      if (pot_type == "LJ") {
    std::cout << "Loading LJ potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::LJPot>();
  } else if (pot_type == "LJCluster") {
    std::cout << "Loading LJ cluster potential (free boundaries)..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::LJClusterPot>();
  } else if (pot_type == "Morse") {
    std::cout << "Loading Morse potential (Pt parameters)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::MorsePot>();
  } else if (pot_type == "ZBL") {
    std::cout << "Loading ZBL potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::ZBLPot>();
#ifdef RGPOT_HAS_IPI
  } else if (pot_type == "IPI" || pot_type.rfind("IPI:", 0) == 0) {
    if (pot_type.size() < 5) {
      std::cerr << "Error: IPI requires an address, for example "
                   "IPI:unix:/tmp/ipi_rgpot or IPI:tcp:127.0.0.1:31415"
                << std::endl;
      return 1;
    }
    rgpot::IPIConfig cfg;
    cfg.address = pot_type.substr(4);
    std::cout << "Loading IPI socket server at " << cfg.address << "..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::IPIPot>(cfg);
#endif
#ifdef RGPOT_HAS_XTB
  } else if (pot_type == "XTB") {
    std::cout << "Loading XTB potential (GFN2-xTB)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>();
  } else if (pot_type == "GFNFF") {
    rgpot::XTBConfig cfg;
    cfg.method = rgpot::GFNMethod::GFNFF;
    std::cout << "Loading XTB potential (GFNFF)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>(cfg);
  } else if (pot_type == "GFN1xTB") {
    rgpot::XTBConfig cfg;
    cfg.method = rgpot::GFNMethod::GFN1xTB;
    std::cout << "Loading XTB potential (GFN1-xTB)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>(cfg);
  } else if (pot_type == "GFN0xTB") {
    rgpot::XTBConfig cfg;
    cfg.method = rgpot::GFNMethod::GFN0xTB;
    std::cout << "Loading XTB potential (GFN0-xTB)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>(cfg);
#endif // RGPOT_HAS_XTB
#ifdef RGPOT_HAS_TBLITE
  } else if (pot_type == "TBLite" || pot_type == "TBLiteGFN2") {
    std::cout << "Loading TBLite potential (GFN2)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::TBLitePot>();
  } else if (pot_type == "TBLiteGFN1") {
    rgpot::TBLiteConfig cfg;
    cfg.method = rgpot::TBLiteMethod::GFN1;
    std::cout << "Loading TBLite potential (GFN1)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::TBLitePot>(cfg);
  } else if (pot_type == "TBLiteIPEA1") {
    rgpot::TBLiteConfig cfg;
    cfg.method = rgpot::TBLiteMethod::IPEA1;
    std::cout << "Loading TBLite potential (IPEA1)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::TBLitePot>(cfg);
#endif // RGPOT_HAS_TBLITE
#ifdef RGPOT_HAS_DFTD3
  } else if (pot_type == "D3") {
    std::cout << "Loading D3 potential (BJ PBE ATM)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::D3Pot>();
#endif // RGPOT_HAS_DFTD3
#ifdef RGPOT_HAS_DFTD4
  } else if (pot_type == "D4") {
    std::cout << "Loading D4 potential (PBE ATM charge 0)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::D4Pot>();
#endif // RGPOT_HAS_DFTD4
#ifdef RGPOT_HAS_METATOMIC
  } else if (pot_type.rfind("Metatomic:", 0) == 0) {
    auto spec = pot_type.substr(10);
    rgpot::MetatomicConfig cfg;
    // Metatomic:<model.pt>[:<device>] -- the device is part of the request,
    // cuda on an accelerated node, cpu elsewhere.
    auto colon = spec.rfind(':');
    if (colon != std::string::npos &&
        spec.substr(colon + 1).find('/') == std::string::npos) {
      cfg.device = spec.substr(colon + 1);
      spec = spec.substr(0, colon);
    }
    cfg.model_path = spec;
    auto model_path = spec;
    std::cout << "Loading Metatomic potential from '" << model_path << "'..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::MetatomicPot>(cfg);
#endif // RGPOT_HAS_METATOMIC
#ifdef RGPOT_HAS_UMA
  } else if (pot_type.rfind("Uma:", 0) == 0) {
    rgpot::UmaConfig cfg;
    auto rest = pot_type.substr(4);
    const auto colon = rest.rfind(':');
    if (colon != std::string::npos) {
      const auto tail = rest.substr(colon + 1);
      if (tail == "omol" || tail == "omat" || tail == "oc20" ||
          tail == "oc22" || tail == "oc25" || tail == "odac" ||
          tail == "omc") {
        cfg.task_name = tail;
        rest = rest.substr(0, colon);
      }
    }
    cfg.model_path = rest;
    std::cout << "Loading UMA AOTI package '" << cfg.model_path
              << "' task='" << cfg.task_name << "'..." << std::endl;
    potential_to_use = std::make_unique<rgpot::UmaPot>(cfg);
#endif // RGPOT_HAS_UMA
#ifdef RGPOT_HAS_SKALA
  } else if (pot_type == "Skala" || pot_type.rfind("Skala:", 0) == 0) {
    rgpot::SkalaConfig cfg;
    if (pot_type.rfind("Skala:", 0) == 0)
      cfg.basis = pot_type.substr(6);
    std::cout << "Loading Skala XC via NWChem DFT (xc=" << cfg.xc
              << " basis=" << cfg.basis << ")..." << std::endl;
    potential_to_use = std::make_unique<rgpot::SkalaPot>(cfg);
#endif // RGPOT_HAS_SKALA
  } else if (pot_type.rfind("NWChem", 0) == 0) {
    std::cout << "Loading NWChem potential (dlopen libnwchemc)..."
              << std::endl;
    // NWChem[:<basis>[:<theory>[:<xc>]]] -- unset fields keep the schema
    // defaults; commas in the xc field stand in for spaces so a request
    // like NWChem:6-31g*:dft:xpbe96,cpbe96 stays one shell token.
    std::string nw_basis, nw_theory, nw_xc;
    {
      std::vector<std::string> parts;
      std::stringstream ss(pot_type);
      std::string tok;
      while (std::getline(ss, tok, ':'))
        parts.push_back(tok);
      if (parts.size() > 1)
        nw_basis = parts[1];
      if (parts.size() > 2)
        nw_theory = parts[2];
      if (parts.size() > 3)
        nw_xc = parts[3];
      std::replace(nw_xc.begin(), nw_xc.end(), ',', ' ');
    }
    std::unique_ptr<rgpot::NWChemPot> nw;
    auto construct = [&] {
    if (nw_basis.empty() && nw_theory.empty() && nw_xc.empty()) {
      nw = std::make_unique<rgpot::NWChemPot>();
    } else {
      capnp::MallocMessageBuilder nw_msg;
      auto nw_params = nw_msg.initRoot<NWChemParams>();
      if (!nw_basis.empty())
        nw_params.setBasis(nw_basis);
      if (!nw_theory.empty())
        nw_params.setTheory(nw_theory);
      if (!nw_xc.empty()) {
        // The xc directive rides the raw input-block escape hatch; the
        // scfType field is the HF reference choice, not the functional.
        // The iteration cap bounds a pathological geometry's SCF: the
        // engine returns a convergence failure the driver refuses and
        // moves past, instead of one stuck call eating the RPC budget.
        auto nw_blocks = nw_params.initInputBlocks(1);
        nw_blocks.set(0, std::string("dft\n  xc " + nw_xc +
                                     "\n  iterations 60\nend"));
      }
      std::cout << "  basis='" << nw_basis << "' theory='" << nw_theory
                << "' xc='" << nw_xc << "'" << std::endl;
      nw = std::make_unique<rgpot::NWChemPot>(nw_params.asReader());
    }
    };
#ifdef RGPOT_POTSERV_MPI
    host.collective([&] {
      if (!rgpot::NWChemPot::probeAvailable())
        throw std::runtime_error("NWChem engine is not available on every rank");
    });
    host.collective(construct);
#else
    construct();
#endif
    if (!nw->available()) {
      std::cerr << "Warning: libnwchemc not loaded; calculate() will fail "
                   "until engine is available (configure() still accepted)."
                << std::endl;
    }
    potential_to_use = nullptr; // use dedicated path below
    rgpot::tools::MpiPefSession *channel = nullptr;
#ifdef RGPOT_POTSERV_MPI
    rgpot::tools::MpiPefSession session(host, *nw);
    if (host.rank() != 0) {
      session.work();
      return 0;
    }
    channel = &session;
#endif
    capnp::EzRpcServer server(kj::heap<GenericPotImpl>(std::move(nw), channel),
                              "localhost", port);
    auto &waitScope = server.getWaitScope();
    std::cout << "Server running on port " << port << " with " << pot_type
              << " potential." << std::endl;
    kj::NEVER_DONE.wait(waitScope);
    return 0;
  } else if (pot_type == "CP2K") {
#ifdef RGPOT_HAS_CP2K
    std::cout << "Loading CP2K potential (dlopen libcp2kc_rgpot)..."
              << std::endl;
    std::unique_ptr<rgpot::CP2KPot> cp;
#ifdef RGPOT_POTSERV_MPI
    host.collective([&] {
      if (!rgpot::CP2KPot::probeAvailable())
        throw std::runtime_error("CP2K shim is not available on every rank");
    });
    host.collective([&] { cp = std::make_unique<rgpot::CP2KPot>(); });
#else
    cp = std::make_unique<rgpot::CP2KPot>();
#endif
    if (!cp->available()) {
      std::cerr << "Warning: libcp2kc_rgpot not loaded; calculate() will fail "
                   "until the shim is available (configure() still accepted)."
                << std::endl;
    }
    rgpot::tools::MpiPefSession *channel = nullptr;
#ifdef RGPOT_POTSERV_MPI
    rgpot::tools::MpiPefSession session(host, *cp);
    if (host.rank() != 0) {
      session.work();
      return 0;
    }
    channel = &session;
#endif
    capnp::EzRpcServer server(kj::heap<GenericPotImpl>(std::move(cp), channel),
                              "localhost", port);
    auto &waitScope = server.getWaitScope();
    std::cout << "Server running on port " << port << " with " << pot_type
              << " potential." << std::endl;
    kj::NEVER_DONE.wait(waitScope);
    return 0;
#else
    std::cerr << "Error: this potserv was built without CP2K; rebuild with "
                 "-Dwith_cp2k=enabled (CMake: -DRGPOT_WITH_CP2K=ON) against "
                 "libcp2k."
              << std::endl;
    return 1;
#endif
  } else if (pot_type == "CPMD") {
    std::cout << "Loading CPMD potential (dlopen libcpmdc)..." << std::endl;
    std::unique_ptr<rgpot::CPMDPot> cp;
#ifdef RGPOT_POTSERV_MPI
    host.collective([&] {
      if (!rgpot::CPMDPot::probeAvailable())
        throw std::runtime_error("CPMD engine is not available on every rank");
    });
    host.collective([&] { cp = std::make_unique<rgpot::CPMDPot>(); });
#else
    cp = std::make_unique<rgpot::CPMDPot>();
#endif
    if (!cp->available()) {
      std::cerr << "Warning: libcpmdc not loaded; calculate() will fail "
                   "until engine is available (configure() still accepted)."
                << std::endl;
    }
    potential_to_use = nullptr;
    rgpot::tools::MpiPefSession *channel = nullptr;
#ifdef RGPOT_POTSERV_MPI
    rgpot::tools::MpiPefSession session(host, *cp);
    if (host.rank() != 0) {
      session.work();
      return 0;
    }
    channel = &session;
#endif
    capnp::EzRpcServer server(kj::heap<GenericPotImpl>(std::move(cp), channel),
                              "localhost", port);
    auto &waitScope = server.getWaitScope();
    std::cout << "Server running on port " << port << " with " << pot_type
              << " potential." << std::endl;
    kj::NEVER_DONE.wait(waitScope);
    return 0;
  } else {
    std::cerr << "Error: Unknown potential type '" << pot_type << "'"
              << std::endl;
    return 1;
  }

  if (!potential_to_use) {
    std::cerr << "Error: potential not initialized" << std::endl;
    return 1;
  }

  capnp::EzRpcServer server(
      kj::heap<GenericPotImpl>(std::move(potential_to_use)), "localhost", port);

  auto &waitScope = server.getWaitScope();
  std::cout << "Server running on port " << port << " with " << pot_type
            << " potential." << std::endl;
  kj::NEVER_DONE.wait(waitScope);

  return 0;
}

int main(int argc, char **argv) {
  try {
#ifdef RGPOT_POTSERV_MPI
    rgpot::tools::MpiHost host(argc, argv);
    return runServer(argc, argv, host);
#else
    return runServer(argc, argv);
#endif
  } catch (const kj::Exception &error) {
    std::cerr << error.getDescription().cStr() << std::endl;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
  }
  return 1;
}
