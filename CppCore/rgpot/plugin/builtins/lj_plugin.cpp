// MIT License
// Copyright 2023--present rgpot developers

#include <rgpot/plugin.h>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif

#include <charconv>
#include <exception>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace rgpot::plugin::builtin {

struct Value {
  enum class Kind { Number, Bool, Text };
  Kind kind;
  double number = 0.0;
  bool flag = false;
  std::string text;
};

struct Field {
  std::string key;
  Value value;
};

enum class Shape { Empty, Object, Bare, Invalid };

bool isSpace(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

const char *skip(const char *p, const char *end) {
  while (p < end && isSpace(*p)) {
    ++p;
  }
  return p;
}

bool parseString(const char *&p, const char *end, std::string &out,
                 std::string &error) {
  if (p >= end || *p != '"') {
    error = "invalid parameter: config value is invalid";
    return false;
  }
  ++p;
  out.clear();
  while (p < end && *p != '"') {
    if (*p == '\\') {
      ++p;
      if (p >= end) {
        error = "invalid parameter: config value is invalid";
        return false;
      }
      if (*p == '"' || *p == '\\' || *p == '/') {
        out.push_back(*p);
      } else {
        error = "invalid parameter: config value is invalid";
        return false;
      }
    } else {
      out.push_back(*p);
    }
    ++p;
  }
  if (p >= end || *p != '"') {
    error = "invalid parameter: config value is invalid";
    return false;
  }
  ++p;
  return true;
}

bool parseNumber(const char *&p, const char *end, double &out,
                 std::string &error) {
  const char *start = p;
  auto parsed = std::from_chars(start, end, out);
  if (parsed.ec != std::errc{} || parsed.ptr == start) {
    error = "invalid parameter: config value is invalid";
    return false;
  }
  p = parsed.ptr;
  return true;
}

bool parseValue(const char *&p, const char *end, Value &out,
                std::string &error) {
  p = skip(p, end);
  if (p >= end) {
    error = "invalid parameter: config value is invalid";
    return false;
  }
  if (*p == '"') {
    out.kind = Value::Kind::Text;
    return parseString(p, end, out.text, error);
  }
  if (end - p >= 4 && p[0] == 't' && p[1] == 'r' && p[2] == 'u' &&
      p[3] == 'e') {
    out.kind = Value::Kind::Bool;
    out.flag = true;
    p += 4;
    return true;
  }
  if (end - p >= 5 && p[0] == 'f' && p[1] == 'a' && p[2] == 'l' &&
      p[3] == 's' && p[4] == 'e') {
    out.kind = Value::Kind::Bool;
    out.flag = false;
    p += 5;
    return true;
  }
  out.kind = Value::Kind::Number;
  return parseNumber(p, end, out.number, error);
}

Shape parseConfig(const char *text, std::string &bare,
                  std::vector<Field> &fields, std::string &error) {
  fields.clear();
  bare.clear();
  if (text == nullptr) {
    return Shape::Empty;
  }
  const char *begin = text;
  const char *end = text;
  while (*end != '\0') {
    ++end;
  }
  const char *p = skip(begin, end);
  if (p == end) {
    return Shape::Empty;
  }
  if (*p != '{') {
    bare.assign(p, end);
    while (!bare.empty() && isSpace(bare.back())) {
      bare.pop_back();
    }
    return bare.empty() ? Shape::Empty : Shape::Bare;
  }
  ++p;
  p = skip(p, end);
  if (p < end && *p == '}') {
    p = skip(p + 1, end);
    if (p != end) {
      error = "invalid parameter: plugin config is not an object";
      return Shape::Invalid;
    }
    return Shape::Empty;
  }
  while (p < end) {
    p = skip(p, end);
    std::string key;
    if (!parseString(p, end, key, error)) {
      return Shape::Invalid;
    }
    for (const auto &field : fields) {
      if (field.key == key) {
        error = "invalid parameter: duplicate config key";
        return Shape::Invalid;
      }
    }
    p = skip(p, end);
    if (p >= end || *p != ':') {
      error = "invalid parameter: config value is invalid";
      return Shape::Invalid;
    }
    ++p;
    Value value;
    if (!parseValue(p, end, value, error)) {
      return Shape::Invalid;
    }
    fields.push_back(Field{std::move(key), std::move(value)});
    p = skip(p, end);
    if (p < end && *p == ',') {
      ++p;
      continue;
    }
    if (p < end && *p == '}') {
      p = skip(p + 1, end);
      if (p != end) {
        error = "invalid parameter: plugin config is not an object";
        return Shape::Invalid;
      }
      return Shape::Object;
    }
    error = "invalid parameter: plugin config is not an object";
    return Shape::Invalid;
  }
  error = "invalid parameter: plugin config is not an object";
  return Shape::Invalid;
}

const Field *findField(const std::vector<Field> &fields, const char *key) {
  for (const auto &field : fields) {
    if (field.key == key) {
      return &field;
    }
  }
  return nullptr;
}

bool takeNumber(const Field &field, double &dest, std::string &error) {
  if (field.value.kind != Value::Kind::Number) {
    error = "invalid parameter: config value is invalid";
    return false;
  }
  dest = field.value.number;
  return true;
}

bool takeInt(const Field &field, int &dest, std::string &error) {
  if (field.value.kind != Value::Kind::Number) {
    error = "invalid parameter: config value is invalid";
    return false;
  }
  const double number = field.value.number;
  const int asInt = static_cast<int>(number);
  if (static_cast<double>(asInt) != number) {
    error = "invalid parameter: config value is invalid";
    return false;
  }
  dest = asInt;
  return true;
}

bool takeBool(const Field &field, bool &dest, std::string &error) {
  if (field.value.kind != Value::Kind::Bool) {
    error = "invalid parameter: config value is invalid";
    return false;
  }
  dest = field.value.flag;
  return true;
}

bool takeText(const Field &field, std::string &dest, std::string &error) {
  if (field.value.kind != Value::Kind::Text) {
    error = "invalid parameter: config value is invalid";
    return false;
  }
  dest = field.value.text;
  return true;
}

bool known(const std::vector<Field> &fields, const char *const *keys,
           std::string &error) {
  for (const auto &field : fields) {
    bool match = false;
    for (const char *const *key = keys; *key != nullptr; ++key) {
      if (field.key == *key) {
        match = true;
        break;
      }
    }
    if (!match) {
      error = "invalid parameter: unknown config key";
      return false;
    }
  }
  return true;
}

rgpot_plugin_status_t failCreate(rgpot_plugin_instance **output,
                                 const std::string &error) {
  if (output != nullptr) {
    *output = nullptr;
  }
  setForceLastError(error.c_str());
  return RGPOT_PLUGIN_INVALID_PARAM;
}

template <class Pot>
rgpot_plugin_status_t calculatePot(rgpot_plugin_instance *instance,
                                   const rgpot_plugin_input_t *input,
                                   rgpot_plugin_output_t *output) {
  if (instance == nullptr || input == nullptr || output == nullptr) {
    setForceLastError("invalid parameter: input is null");
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  auto *pot = reinterpret_cast<Pot *>(instance);
  ForceOut result{};
  result.F = output->forces;
  const ForceInput request{input->n_atoms, input->positions,
                           input->atomic_numbers, input->cell};
  if (rejectForceCall(request, &result, pot->caps().periodic) != 0) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  try {
    pot->forceImpl(request, &result);
    output->energy = result.energy;
    output->variance = result.variance;
    return RGPOT_PLUGIN_OK;
  } catch (const std::exception &ex) {
    setForceLastError(ex.what());
    return RGPOT_PLUGIN_ERROR;
  } catch (...) {
    setForceLastError("unknown exception in the potential");
    return RGPOT_PLUGIN_ERROR;
  }
}

template <class Pot> void destroyPot(rgpot_plugin_instance *instance) {
  delete reinterpret_cast<Pot *>(instance);
}

template <class Pot>
uint32_t capabilitiesOf(const rgpot_plugin_instance *instance) {
  uint32_t bits = RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES | RGPOT_CAP_INSTANCE_SAFE;
  const bool periodic =
      instance == nullptr ||
      reinterpret_cast<const Pot *>(instance)->caps().periodic;
  if (periodic) {
    bits |= RGPOT_CAP_PERIODIC;
  }
  return bits;
}

rgpot_plugin_status_t createLennardJones(const char *,
                                         rgpot_plugin_instance **output) {
  if (output == nullptr) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  *output = nullptr;
  try {
    *output = reinterpret_cast<rgpot_plugin_instance *>(new LJPot());
    return RGPOT_PLUGIN_OK;
  } catch (...) {
    setForceLastError("plugin create failed");
    return RGPOT_PLUGIN_ERROR;
  }
}

rgpot_plugin_status_t createMorse(const char *config,
                                  rgpot_plugin_instance **output) {
  if (output == nullptr) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  *output = nullptr;
  std::string bare;
  std::string error;
  std::vector<Field> fields;
  const Shape shape = parseConfig(config, bare, fields, error);
  if (shape == Shape::Invalid) {
    return failCreate(output, error);
  }
  if (shape == Shape::Bare) {
    return failCreate(output,
                      "invalid parameter: plugin config is not an object");
  }
  static constexpr const char *keys[] = {"De",           "a",    "re", "cutoff",
                                         "switch_width", nullptr};
  if (!known(fields, keys, error)) {
    return failCreate(output, error);
  }
  MorseConfig parsed;
  if (const Field *field = findField(fields, "De")) {
    if (!takeNumber(*field, parsed.De, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "a")) {
    if (!takeNumber(*field, parsed.a, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "re")) {
    if (!takeNumber(*field, parsed.re, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "cutoff")) {
    if (!takeNumber(*field, parsed.cutoff, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "switch_width")) {
    if (!takeNumber(*field, parsed.switch_width, error)) {
      return failCreate(output, error);
    }
  }
  try {
    *output = reinterpret_cast<rgpot_plugin_instance *>(new MorsePot(parsed));
    return RGPOT_PLUGIN_OK;
  } catch (const std::exception &ex) {
    return failCreate(output, ex.what());
  } catch (...) {
    return failCreate(output, "plugin create failed");
  }
}

rgpot_plugin_status_t createSuttonChen(const char *config,
                                       rgpot_plugin_instance **output) {
  if (output == nullptr) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  *output = nullptr;
  std::string bare;
  std::string error;
  std::vector<Field> fields;
  const Shape shape = parseConfig(config, bare, fields, error);
  if (shape == Shape::Invalid) {
    return failCreate(output, error);
  }
  if (shape == Shape::Bare) {
    return failCreate(output,
                      "invalid parameter: plugin config is not an object");
  }
  static constexpr const char *keys[] = {
      "epsilon", "c", "a", "n", "m", "cutoff", "periodic", "stress", nullptr};
  if (!known(fields, keys, error)) {
    return failCreate(output, error);
  }
  // The plugin advertises a periodic cell unless the config turns it off.
  // The struct default leaves the cell unused.
  SuttonChenConfig parsed;
  parsed.periodic = true;
  if (const Field *field = findField(fields, "epsilon")) {
    if (!takeNumber(*field, parsed.epsilon, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "c")) {
    if (!takeNumber(*field, parsed.c, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "a")) {
    if (!takeNumber(*field, parsed.a, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "n")) {
    if (!takeInt(*field, parsed.n, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "m")) {
    if (!takeInt(*field, parsed.m, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "cutoff")) {
    if (!takeNumber(*field, parsed.cutoff, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "periodic")) {
    if (!takeBool(*field, parsed.periodic, error)) {
      return failCreate(output, error);
    }
  }
  if (const Field *field = findField(fields, "stress")) {
    if (!takeBool(*field, parsed.stress, error)) {
      return failCreate(output, error);
    }
  }
  try {
    *output =
        reinterpret_cast<rgpot_plugin_instance *>(new SuttonChenPot(parsed));
    return RGPOT_PLUGIN_OK;
  } catch (const std::exception &ex) {
    return failCreate(output, ex.what());
  } catch (...) {
    return failCreate(output, "plugin create failed");
  }
}

#ifdef RGPOT_HAS_EAM_TABLE
rgpot_plugin_status_t createEam(const char *config,
                                rgpot_plugin_instance **output,
                                EAMTableFormat format) {
  if (output == nullptr) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  *output = nullptr;
  std::string bare;
  std::string error;
  std::vector<Field> fields;
  const Shape shape = parseConfig(config, bare, fields, error);
  if (shape == Shape::Invalid) {
    return failCreate(output, error);
  }
  EAMTableConfig parsed;
  parsed.format = format;
  if (shape == Shape::Bare) {
    parsed.path = bare;
  } else if (shape == Shape::Empty) {
    return failCreate(output, "invalid parameter: eam table path is missing");
  } else {
    static constexpr const char *keys[] = {"path",   "format", "periodic",
                                           "stress", "skin",   nullptr};
    if (!known(fields, keys, error)) {
      return failCreate(output, error);
    }
    if (const Field *field = findField(fields, "path")) {
      if (!takeText(*field, parsed.path, error)) {
        return failCreate(output, error);
      }
    }
    if (parsed.path.empty()) {
      return failCreate(output, "invalid parameter: eam table path is missing");
    }
    if (const Field *field = findField(fields, "format")) {
      std::string name;
      if (!takeText(*field, name, error)) {
        return failCreate(output, error);
      }
      if (name == "funcfl") {
        parsed.format = EAMTableFormat::Funcfl;
      } else if (name == "setfl") {
        parsed.format = EAMTableFormat::Setfl;
      } else if (name == "fs") {
        parsed.format = EAMTableFormat::FinnisSinclair;
      } else {
        return failCreate(output, "invalid parameter: config value is invalid");
      }
    }
    if (const Field *field = findField(fields, "periodic")) {
      if (!takeBool(*field, parsed.periodic, error)) {
        return failCreate(output, error);
      }
    }
    if (const Field *field = findField(fields, "stress")) {
      if (!takeBool(*field, parsed.stress, error)) {
        return failCreate(output, error);
      }
    }
    if (const Field *field = findField(fields, "skin")) {
      if (!takeNumber(*field, parsed.skin, error)) {
        return failCreate(output, error);
      }
    }
  }
  try {
    *output =
        reinterpret_cast<rgpot_plugin_instance *>(new EAMTablePot(parsed));
    return RGPOT_PLUGIN_OK;
  } catch (const std::exception &ex) {
    return failCreate(output, ex.what());
  } catch (...) {
    return failCreate(output, "plugin create failed");
  }
}
#endif

} // namespace rgpot::plugin::builtin

extern "C" {

static rgpot_plugin_status_t
rgpot_builtin_lj_create(const char *config, rgpot_plugin_instance **output) {
  return rgpot::plugin::builtin::createLennardJones(config, output);
}
static void rgpot_builtin_lj_destroy(rgpot_plugin_instance *instance) {
  rgpot::plugin::builtin::destroyPot<rgpot::LJPot>(instance);
}
static rgpot_plugin_status_t
rgpot_builtin_lj_calculate(rgpot_plugin_instance *instance,
                           const rgpot_plugin_input_t *input,
                           rgpot_plugin_output_t *output) {
  return rgpot::plugin::builtin::calculatePot<rgpot::LJPot>(instance, input,
                                                            output);
}
static uint32_t
rgpot_builtin_lj_capabilities(const rgpot_plugin_instance *instance) {
  return rgpot::plugin::builtin::capabilitiesOf<rgpot::LJPot>(instance);
}
static const char *rgpot_builtin_lj_last_error(const rgpot_plugin_instance *) {
  return rgpot::forceLastError();
}

static rgpot_plugin_status_t
rgpot_builtin_morse_create(const char *config, rgpot_plugin_instance **output) {
  return rgpot::plugin::builtin::createMorse(config, output);
}
static void rgpot_builtin_morse_destroy(rgpot_plugin_instance *instance) {
  rgpot::plugin::builtin::destroyPot<rgpot::MorsePot>(instance);
}
static rgpot_plugin_status_t
rgpot_builtin_morse_calculate(rgpot_plugin_instance *instance,
                              const rgpot_plugin_input_t *input,
                              rgpot_plugin_output_t *output) {
  return rgpot::plugin::builtin::calculatePot<rgpot::MorsePot>(instance, input,
                                                               output);
}
static uint32_t
rgpot_builtin_morse_capabilities(const rgpot_plugin_instance *instance) {
  return rgpot::plugin::builtin::capabilitiesOf<rgpot::MorsePot>(instance);
}

static rgpot_plugin_status_t
rgpot_builtin_sc_create(const char *config, rgpot_plugin_instance **output) {
  return rgpot::plugin::builtin::createSuttonChen(config, output);
}
static void rgpot_builtin_sc_destroy(rgpot_plugin_instance *instance) {
  rgpot::plugin::builtin::destroyPot<rgpot::SuttonChenPot>(instance);
}
static rgpot_plugin_status_t
rgpot_builtin_sc_calculate(rgpot_plugin_instance *instance,
                           const rgpot_plugin_input_t *input,
                           rgpot_plugin_output_t *output) {
  return rgpot::plugin::builtin::calculatePot<rgpot::SuttonChenPot>(
      instance, input, output);
}
static uint32_t
rgpot_builtin_sc_capabilities(const rgpot_plugin_instance *instance) {
  return rgpot::plugin::builtin::capabilitiesOf<rgpot::SuttonChenPot>(instance);
}

#ifdef RGPOT_HAS_EAM_TABLE
static rgpot_plugin_status_t
rgpot_builtin_eam_funcfl_create(const char *config,
                                rgpot_plugin_instance **output) {
  return rgpot::plugin::builtin::createEam(config, output,
                                           rgpot::EAMTableFormat::Funcfl);
}
static rgpot_plugin_status_t
rgpot_builtin_eam_setfl_create(const char *config,
                               rgpot_plugin_instance **output) {
  return rgpot::plugin::builtin::createEam(config, output,
                                           rgpot::EAMTableFormat::Setfl);
}
static void rgpot_builtin_eam_destroy(rgpot_plugin_instance *instance) {
  rgpot::plugin::builtin::destroyPot<rgpot::EAMTablePot>(instance);
}
static rgpot_plugin_status_t
rgpot_builtin_eam_calculate(rgpot_plugin_instance *instance,
                            const rgpot_plugin_input_t *input,
                            rgpot_plugin_output_t *output) {
  return rgpot::plugin::builtin::calculatePot<rgpot::EAMTablePot>(
      instance, input, output);
}
static uint32_t
rgpot_builtin_eam_capabilities(const rgpot_plugin_instance *instance) {
  return rgpot::plugin::builtin::capabilitiesOf<rgpot::EAMTablePot>(instance);
}
#endif

} // extern "C"

namespace rgpot::plugin {
namespace {

constexpr rgpot_plugin_descriptor_t kLennardJones{
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "lennard-jones",
    "1.0.0",
    nullptr,
    nullptr,
    rgpot_builtin_lj_create,
    rgpot_builtin_lj_destroy,
    rgpot_builtin_lj_calculate,
    rgpot_builtin_lj_capabilities,
    rgpot_builtin_lj_last_error,
};

constexpr rgpot_plugin_descriptor_t kClassicalMorse{
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "classical-morse",
    "1.0.0",
    nullptr,
    nullptr,
    rgpot_builtin_morse_create,
    rgpot_builtin_morse_destroy,
    rgpot_builtin_morse_calculate,
    rgpot_builtin_morse_capabilities,
    rgpot_builtin_lj_last_error,
};

constexpr rgpot_plugin_descriptor_t kSuttonChen{
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "sutton-chen",
    "1.0.0",
    nullptr,
    nullptr,
    rgpot_builtin_sc_create,
    rgpot_builtin_sc_destroy,
    rgpot_builtin_sc_calculate,
    rgpot_builtin_sc_capabilities,
    rgpot_builtin_lj_last_error,
};

#ifdef RGPOT_HAS_EAM_TABLE
constexpr rgpot_plugin_descriptor_t kEamFuncfl{
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "eam-funcfl",
    "1.0.0",
    nullptr,
    nullptr,
    rgpot_builtin_eam_funcfl_create,
    rgpot_builtin_eam_destroy,
    rgpot_builtin_eam_calculate,
    rgpot_builtin_eam_capabilities,
    rgpot_builtin_lj_last_error,
};

constexpr rgpot_plugin_descriptor_t kEamSetfl{
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "eam-setfl",
    "1.0.0",
    nullptr,
    nullptr,
    rgpot_builtin_eam_setfl_create,
    rgpot_builtin_eam_destroy,
    rgpot_builtin_eam_calculate,
    rgpot_builtin_eam_capabilities,
    rgpot_builtin_lj_last_error,
};
#endif

static_assert(kLennardJones.api_version == RGPOT_PLUGIN_API_VERSION);
static_assert(kClassicalMorse.name[0] == 'c');
static_assert(kSuttonChen.name[0] == 's');

} // namespace

const rgpot_plugin_descriptor_t *builtinLennardJones() {
  return &kLennardJones;
}
const rgpot_plugin_descriptor_t *builtinClassicalMorse() {
  return &kClassicalMorse;
}
const rgpot_plugin_descriptor_t *builtinSuttonChen() { return &kSuttonChen; }
#ifdef RGPOT_HAS_EAM_TABLE
const rgpot_plugin_descriptor_t *builtinEamFuncfl() { return &kEamFuncfl; }
const rgpot_plugin_descriptor_t *builtinEamSetfl() { return &kEamSetfl; }
#endif

} // namespace rgpot::plugin
