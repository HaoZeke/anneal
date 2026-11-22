#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Shared Cap'n Proto word serialization and ForceInput marshalling for
 * dlopen QM frontends (CP2KPot, CPMDPot, NWChemPot).
 */

#include "rgpot/CalculatorGroup.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <capnp/message.h>
#include <capnp/serialize.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rgpot {
namespace abi {

template <typename Reader>
inline std::vector<::capnp::word> serialize_capnp(const Reader &root) {
  ::capnp::MallocMessageBuilder msg;
  msg.setRoot(root);
  auto words = ::capnp::messageToFlatArray(msg);
  std::vector<::capnp::word> out(words.size());
  if (!out.empty()) {
    std::memcpy(out.data(), words.begin(),
                words.size() * sizeof(::capnp::word));
  }
  return out;
}

template <typename SchemaRoot>
inline std::vector<::capnp::word> default_capnp_words() {
  ::capnp::MallocMessageBuilder msg;
  auto root = msg.initRoot<SchemaRoot>();
  return serialize_capnp(root.asReader());
}

struct BytesView {
  const void *data = nullptr;
  size_t size = 0;
};

inline BytesView words_view(const std::vector<::capnp::word> &words) noexcept {
  BytesView view;
  if (!words.empty()) {
    view.data = words.data();
    view.size = words.size() * sizeof(::capnp::word);
  }
  return view;
}

inline std::vector<::capnp::word>
serialize_force_input_ev(const ForceInput &in) {
  ::capnp::MallocMessageBuilder msg;
  auto force_input = msg.initRoot<::ForceInput>();
  const unsigned int coord_count = static_cast<unsigned int>(in.nAtoms * 3u);
  auto pos = force_input.initPos(coord_count);
  auto atmnrs = force_input.initAtmnrs(static_cast<unsigned int>(in.nAtoms));
  auto box = force_input.initBox(9);
  for (unsigned int i = 0; i < coord_count; ++i)
    pos.set(i, in.pos[i]);
  for (unsigned int i = 0; i < in.nAtoms; ++i)
    atmnrs.set(i, in.atmnrs[i]);
  for (unsigned int i = 0; i < 9; ++i)
    box.set(i, in.box[i]);
  force_input.setLengthUnit("angstrom");
  force_input.setEnergyUnit("eV");
  auto words = ::capnp::messageToFlatArray(msg);
  std::vector<::capnp::word> out(words.size());
  if (!out.empty()) {
    std::memcpy(out.data(), words.begin(), words.asBytes().size());
  }
  return out;
}

using AdoptCommFn = int (*)(const void *, std::size_t, int);

/// Retain one handle per registered communicator-adopter callback until
/// process exit so instance teardown cannot unmap a registered callback.
inline bool register_calculator_adopter(const std::string &loaded_path,
                                        const char *symbol_name,
                                        AdoptCommFn callback,
                                        const char *changed_error) {
  if (!callback) {
    return true;
  }
  struct PinnedEngine {
    DynLib library;
    AdoptCommFn callback;
  };
  static auto *engines = new std::vector<PinnedEngine>;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto &engine : *engines) {
    if (engine.callback == callback) {
      return addCalculatorCommAdopter(engine.callback);
    }
  }
  DynLib library(loaded_path);
  if (library.sym_optional<AdoptCommFn>(symbol_name) != callback) {
    throw std::runtime_error(changed_error);
  }
  engines->push_back({std::move(library), callback});
  return addCalculatorCommAdopter(callback);
}

} // namespace abi
} // namespace rgpot
