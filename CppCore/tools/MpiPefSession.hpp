// MIT License
// Copyright 2023--present rgpot developers
#pragma once

#include "MpiHost.hpp"
#ifdef RGPOT_HAS_CP2K
#include "rgpot/CP2KPot/CP2KPot.hpp"
#endif
#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace rgpot::tools {

// Rank zero owns the RPC socket. Each request reaches every engine rank in
// the same order; worker errors are returned by the root's collective call.
class MpiPefSession {
public:
  using Result = std::tuple<double, types::AtomMatrix, double>;
  using Box = std::array<std::array<double, 3>, 3>;
  static_assert(sizeof(std::uint64_t) == sizeof(capnp::word));
  static_assert(alignof(std::uint64_t) >= alignof(capnp::word));

  MpiPefSession(const MpiHost &host, NWChemPot &potential)
      : host_(host), potential_(potential), nwchem_(&potential) {}
  MpiPefSession(const MpiHost &host, CPMDPot &potential)
      : host_(host), potential_(potential), cpmd_(&potential) {}
#ifdef RGPOT_HAS_CP2K
  MpiPefSession(const MpiHost &host, CP2KPot &potential)
      : host_(host), potential_(potential), cp2k_(&potential) {}
#endif

  ~MpiPefSession() {
    if (host_.rank() == 0 && !stopped_) {
      int command = Stop;
      host_.broadcastCommand(command);
    }
  }

  MpiPefSession(const MpiPefSession &) = delete;
  MpiPefSession &operator=(const MpiPefSession &) = delete;

  Result calculate(const types::AtomMatrix &positions,
                   const std::vector<int> &numbers, const Box &box,
                   std::optional<std::int64_t> orbital_key = std::nullopt) {
    // Root-only preparation precedes the command, so a local exception cannot
    // leave workers waiting for a payload that will never be sent.
    if (positions.cols() != 3 || positions.rows() != numbers.size())
      throw std::invalid_argument("collective force geometry has incompatible dimensions");
    std::vector<double> coordinates(positions.data(),
                                    positions.data() + positions.rows() * 3);
    std::vector<int> atom_numbers(numbers);
    std::vector<double> cell(9);
    std::vector<std::int64_t> orbital_keys;
    if (orbital_key)
      orbital_keys.push_back(*orbital_key);
    static_assert(sizeof(Box) == 9 * sizeof(double));
    std::memcpy(cell.data(), &box, sizeof(box));
    int command = Calculate;
    host_.broadcastCommand(command);
    return calculateReceived(coordinates, atom_numbers, cell, orbital_keys);
  }

  std::string configure(const ::PotentialConfig::Reader &parameters) {
    capnp::MallocMessageBuilder message;
    message.setRoot(parameters);
    auto flat = capnp::messageToFlatArray(message);
    std::vector<std::uint64_t> words(flat.size());
    std::memcpy(words.data(), flat.begin(), flat.size() * sizeof(capnp::word));
    int command = Configure;
    host_.broadcastCommand(command);
    return configureReceived(words);
  }

  void work() {
    for (;;) {
      int command = Stop;
      host_.broadcastCommand(command);
      if (command == Stop) {
        stopped_ = true;
        return;
      }
      try {
        if (command == Calculate) {
          std::vector<double> coordinates;
          std::vector<int> numbers;
          std::vector<double> cell;
          std::vector<std::int64_t> orbital_keys;
          (void)calculateReceived(coordinates, numbers, cell, orbital_keys);
        } else if (command == Configure) {
          std::vector<std::uint64_t> words;
          (void)configureReceived(words);
        } else {
          MPI_Abort(host_.communicator(), 1);
        }
      } catch (const std::exception &) {
        // Every rank receives the same collective error. The root reports it
        // to the client; workers remain available for its following request.
      }
    }
  }

private:
  enum Command { Stop = 0, Calculate = 1, Configure = 2 };

  Result calculateReceived(std::vector<double> &coordinates,
                           std::vector<int> &numbers,
                           std::vector<double> &cell,
                           std::vector<std::int64_t> &orbital_keys) {
    host_.broadcast(coordinates);
    host_.broadcast(numbers);
    host_.broadcast(cell);
    host_.broadcast(orbital_keys);
    std::optional<types::AtomMatrix> positions;
    Box box{};
    host_.collective([&] {
      if (numbers.empty() || numbers.size() > INT_MAX / 3 ||
          coordinates.size() != numbers.size() * 3 || cell.size() != 9)
        throw std::runtime_error("invalid collective force geometry");
      if (orbital_keys.size() > 1)
        throw std::runtime_error("invalid collective orbital key");
      if (!engineAvailable())
        throw std::runtime_error("collective engine is unavailable");
      positions.emplace(numbers.size(), 3);
      std::memcpy(positions->data(), coordinates.data(),
                  coordinates.size() * sizeof(double));
      std::memcpy(&box, cell.data(), sizeof(box));
    });
    std::optional<Result> result;
    host_.collective([&] {
      if (!orbital_keys.empty())
        selectOrbitals(orbital_keys.front());
      result.emplace(potential_(*positions, numbers, box));
    });
    return std::move(*result);
  }

  std::string configureReceived(std::vector<std::uint64_t> &words) {
    host_.broadcast(words);
    std::string response;
    host_.collective([&] {
      capnp::FlatArrayMessageReader reader(
          kj::arrayPtr<const capnp::word>(
              reinterpret_cast<const capnp::word *>(words.data()), words.size()));
      auto parameters = reader.getRoot<::PotentialConfig>();
      const bool accepted = applyConfig(parameters, &response);
      if (!accepted)
        throw std::runtime_error(response.empty()
                                     ? "engine rejected collective configuration"
                                     : response);
    });
    return response;
  }

  bool engineAvailable() const {
#ifdef RGPOT_HAS_CP2K
    if (cp2k_)
      return cp2k_->available();
#endif
    return nwchem_ ? nwchem_->available() : cpmd_->available();
  }

  void selectOrbitals(std::int64_t key) {
#ifdef RGPOT_HAS_CP2K
    if (cp2k_) {
      cp2k_->selectOrbitals(key);
      return;
    }
#endif
    if (cpmd_) {
      cpmd_->selectOrbitals(key);
      return;
    }
    throw std::runtime_error(
        "orbital key is supported only by CPMD and CP2K backends");
  }

  bool applyConfig(const ::PotentialConfig::Reader &parameters,
                   std::string *response) {
#ifdef RGPOT_HAS_CP2K
    if (cp2k_)
      return cp2k_->setPotentialConfig(parameters, response);
#endif
    return nwchem_ ? nwchem_->setPotentialConfig(parameters, response)
                   : cpmd_->setPotentialConfig(parameters, response);
  }

  const MpiHost &host_;
  PotentialBase &potential_;
  NWChemPot *nwchem_ = nullptr;
  CPMDPot *cpmd_ = nullptr;
#ifdef RGPOT_HAS_CP2K
  CP2KPot *cp2k_ = nullptr;
#endif
  bool stopped_ = false;
};

} // namespace rgpot::tools
