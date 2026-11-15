#pragma once
// MIT License — the system contract a UMA AOTI package embeds

#include "rgpot/UmaPot/UmaConfig.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rgpot {

/**
 * @brief A UMA package and the calculator disagree on the system.
 *
 * ``merge_mole`` folds one composition, charge and spin into the
 * compiled graph. Evaluating it for anything else either aborts inside
 * the graph (composition) or silently returns a different potential
 * energy surface (charge, spin, task), so UmaPot refuses instead.
 */
/**
 * @brief The positions or the edge list are not the shape this package
 * was compiled for.
 *
 * The message names both shapes. Callers can match ``shape is``.
 */
class UmaShapeError : public std::runtime_error {
public:
  UmaShapeError(std::string field, std::string actual, std::string package)
      : std::runtime_error("UmaPot: " + field + " shape is " + actual +
                           ", the package was exported for " + package),
        m_field(std::move(field)), m_actual(std::move(actual)),
        m_package(std::move(package)) {}

  [[nodiscard]] const std::string &field() const noexcept { return m_field; }
  [[nodiscard]] const std::string &actual() const noexcept { return m_actual; }
  [[nodiscard]] const std::string &package() const noexcept {
    return m_package;
  }

private:
  std::string m_field;
  std::string m_actual;
  std::string m_package;
};

class UmaContractError : public std::runtime_error {
public:
  UmaContractError(std::string field, std::string package, std::string actual,
                   const std::string &where)
      : std::runtime_error("UmaPot: " + where + " " + field + " is " + actual +
                           ", the package was exported for " + package),
        m_field(std::move(field)), m_package(std::move(package)),
        m_actual(std::move(actual)) {}

  /// Metadata key that disagrees (``charge``, ``spin``, ``task_name``,
  /// ``z_set``, ``counts`` or ``natoms``).
  [[nodiscard]] const std::string &field() const noexcept { return m_field; }
  /// Value embedded in the package.
  [[nodiscard]] const std::string &package() const noexcept {
    return m_package;
  }
  /// Value the config or the input carries.
  [[nodiscard]] const std::string &actual() const noexcept { return m_actual; }

private:
  std::string m_field;
  std::string m_package;
  std::string m_actual;
};

/**
 * @brief Charge, spin, task and composition embedded in a ``.pt2``.
 *
 * Built from the package's embedded metadata JSON. Every field is
 * optional: a key the exporter did not write is not checked.
 * ``scripts/export_uma_aoti.py`` writes ``task_name``, ``charge``,
 * ``spin``, ``z_set`` (a list such as ``[1, 6, 7]``), ``natoms`` and
 * ``counts`` (a JSON object of atomic number to atom count, such as
 * ``{"1": 1, "6": 1, "7": 1}``), plus ``nedges_min`` and ``nedges_max``,
 * the edge-count range the exported graph accepts (equal for a static
 * package, whose graph is traced at one edge count); packages from older
 * exporters carry the first four only.
 *
 * A dynamic package declares ``pos_shape`` ``natoms,3`` with
 * ``natoms_min`` and ``natoms_max``, and ``edge_index_shape``
 * ``2,nedges``. When it keeps the MoLE experts it records no ``z_set``,
 * ``natoms`` or ``counts``; ``z_max`` then bounds the atomic numbers the
 * element embeddings can index.
 */
struct UmaContract {
  std::optional<std::string> task_name;
  std::optional<int64_t> charge;
  std::optional<int64_t> spin;
  std::optional<std::vector<int64_t>> z_set;
  std::optional<std::map<int64_t, int64_t>> counts;
  std::optional<int64_t> natoms;
  std::optional<int64_t> natoms_min;
  std::optional<int64_t> natoms_max;
  std::optional<int64_t> nedges_min;
  std::optional<int64_t> nedges_max;
  std::optional<int64_t> z_max;

  /// One axis of a declared tensor shape. A symbolic axis is a name
  /// such as ``nedges`` with an inclusive range; a fixed axis is one
  /// integer.
  struct Axis {
    bool symbolic{false};
    int64_t fixed{0};
    int64_t lo{0};
    int64_t hi{0};
    std::string name;
  };

  struct ShapeDecl {
    bool known{false};
    std::vector<Axis> axes;
    std::string text;
  };

  ShapeDecl pos_decl;
  ShapeDecl edge_decl;

  /// Second extent of ``edge_index`` inside the exporter's ``shapes``
  /// dict (``{'edge_index': [2, 1332], ...}``). Packages written before
  /// ``nedges_min`` and ``nedges_max`` record a fixed edge list there
  /// and nowhere else. A missing or non-fixed entry is not a count.
  static std::optional<int64_t>
  fixedEdgesFromShapes(const std::string *shapes) {
    if (shapes == nullptr)
      return std::nullopt;
    const std::string &s = *shapes;
    const std::string key = "edge_index";
    std::size_t at = 0;
    while (at < s.size()) {
      const auto found = s.find(key, at);
      if (found == std::string::npos)
        return std::nullopt;
      const bool boundary = found == 0 || s[found - 1] == '\'' ||
                            s[found - 1] == '"' || s[found - 1] == '{' ||
                            s[found - 1] == ' ' || s[found - 1] == ',';
      std::size_t i = found + key.size();
      if (!boundary || i > s.size()) {
        at = found + 1;
        continue;
      }
      if (i < s.size() && (s[i] == '\'' || s[i] == '"'))
        ++i;
      // ``edge_index_shape`` and other longer names are not this key.
      if (i < s.size() &&
          ((s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= 'a' && s[i] <= 'z') ||
           s[i] == '_')) {
        at = i;
        continue;
      }
      while (i < s.size() && (s[i] == ' ' || s[i] == ':' || s[i] == '='))
        ++i;
      if (i >= s.size() || s[i] != '[')
        return std::nullopt;
      const auto close = s.find(']', i);
      if (close == std::string::npos)
        return std::nullopt;
      const auto dims = integers(s.substr(i, close - i + 1));
      if (dims.size() == 2 && dims[0] == 2 && dims[1] > 0)
        return dims[1];
      return std::nullopt;
    }
    return std::nullopt;
  }

  /// Signed decimal integers in order of appearance; anything else
  /// separates them.
  static std::vector<int64_t> integers(const std::string &s) {
    std::vector<int64_t> out;
    size_t i = 0;
    while (i < s.size()) {
      const bool neg =
          s[i] == '-' && i + 1 < s.size() && s[i + 1] >= '0' && s[i + 1] <= '9';
      if (!neg && !(s[i] >= '0' && s[i] <= '9')) {
        ++i;
        continue;
      }
      size_t j = neg ? i + 1 : i;
      int64_t v = 0;
      while (j < s.size() && s[j] >= '0' && s[j] <= '9')
        v = v * 10 + (s[j++] - '0');
      out.push_back(neg ? -v : v);
      i = j;
    }
    return out;
  }

  static UmaContract
  fromMetadata(const std::unordered_map<std::string, std::string> &meta) {
    auto get = [&](const char *key) -> const std::string * {
      const auto it = meta.find(key);
      return it == meta.end() || it->second.empty() ? nullptr : &it->second;
    };
    auto one_int = [&](const char *key) -> std::optional<int64_t> {
      const std::string *raw = get(key);
      if (raw == nullptr)
        return std::nullopt;
      const auto v = integers(*raw);
      if (v.size() != 1)
        throw std::runtime_error(std::string("UmaPot: package metadata ") +
                                 key + " is not one integer: " + *raw);
      return v.front();
    };
    UmaContract c;
    if (const std::string *raw = get("task_name"))
      c.task_name = *raw;
    c.charge = one_int("charge");
    c.spin = one_int("spin");
    c.natoms = one_int("natoms");
    c.nedges_min = one_int("nedges_min");
    c.nedges_max = one_int("nedges_max");
    // Published packages record a fixed edge list in ``shapes`` and
    // carry neither range key. That count is the traced size: the
    // neighbour list is built or padded to it, and a graph that cannot
    // fit is named before the compiled graph runs. One key without the
    // other is still not a range.
    if (!c.nedges_min && !c.nedges_max) {
      if (const auto fixed = fixedEdgesFromShapes(get("shapes"))) {
        c.nedges_min = fixed;
        c.nedges_max = fixed;
      } else {
        throw UmaContractError("nedges", "nedges_min and nedges_max", "absent",
                               "package");
      }
    } else if (!c.nedges_min || !c.nedges_max) {
      const char *missing = c.nedges_min ? "nedges_max" : "nedges_min";
      throw UmaContractError(missing, "nedges_min and nedges_max", "absent",
                             "package");
    }
    c.natoms_min = one_int("natoms_min");
    c.natoms_max = one_int("natoms_max");
    c.z_max = one_int("z_max");
    c.pos_decl = c.parseShape(get("pos_shape"));
    c.edge_decl = c.parseShape(get("edge_index_shape"));
    if (!c.pos_decl.known && c.natoms) {
      c.pos_decl.known = true;
      c.pos_decl.axes.push_back(Axis{false, *c.natoms, 0, 0, {}});
      c.pos_decl.axes.push_back(Axis{false, 3, 0, 0, {}});
      c.pos_decl.text = "[" + std::to_string(*c.natoms) + ", 3]";
    }
    if (!c.edge_decl.known && (c.nedges_min || c.nedges_max)) {
      const int64_t lo = c.nedges_min ? *c.nedges_min : 0;
      const int64_t hi = c.nedges_max ? *c.nedges_max : lo;
      c.edge_decl.known = true;
      c.edge_decl.axes.push_back(Axis{false, 2, 0, 0, {}});
      if (c.nedges_min && c.nedges_max && lo == hi) {
        c.edge_decl.axes.push_back(Axis{false, lo, 0, 0, {}});
        c.edge_decl.text = "[2, " + std::to_string(lo) + "]";
      } else {
        c.edge_decl.axes.push_back(Axis{true, 0, lo, hi, "nedges"});
        c.edge_decl.text = "[2, nedges] with nedges in [" +
                           std::to_string(lo) + ", " + std::to_string(hi) +
                           "]";
      }
    }
    if (const std::string *raw = get("z_set")) {
      auto zs = integers(*raw);
      std::sort(zs.begin(), zs.end());
      c.z_set = std::move(zs);
    }
    if (const std::string *raw = get("counts")) {
      const auto v = integers(*raw);
      if (v.size() % 2 != 0)
        throw std::runtime_error(
            "UmaPot: package metadata counts is not {Z: n} pairs: " + *raw);
      std::map<int64_t, int64_t> m;
      for (size_t k = 0; k < v.size(); k += 2)
        m[v[k]] = v[k + 1];
      c.counts = std::move(m);
    }
    return c;
  }

  static std::string str(const std::vector<int64_t> &v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i)
      s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "]";
  }

  static std::string str(const std::map<int64_t, int64_t> &m) {
    std::string s = "{";
    bool first = true;
    for (const auto &[z, n] : m) {
      s += (first ? "" : ", ") + std::to_string(z) + ": " + std::to_string(n);
      first = false;
    }
    return s + "}";
  }

  /// Task, charge and spin of the calculator against the package.
  void checkConfig(const UmaConfig &cfg) const {
    if (task_name && *task_name != cfg.task_name)
      throw UmaContractError("task_name", *task_name, cfg.task_name, "config");
    if (charge && *charge != cfg.charge)
      throw UmaContractError("charge", std::to_string(*charge),
                             std::to_string(cfg.charge), "config");
    if (spin && *spin != cfg.spin)
      throw UmaContractError("spin", std::to_string(*spin),
                             std::to_string(cfg.spin), "config");
  }

  /// ``spec`` is ``2,6`` or ``2,nedges`` (brackets and spaces allowed).
  ShapeDecl parseShape(const std::string *spec) const {
    ShapeDecl decl;
    if (spec == nullptr)
      return decl;
    std::string token;
    std::vector<std::string> tokens;
    auto flush = [&]() {
      if (!token.empty()) {
        tokens.push_back(token);
        token.clear();
      }
    };
    for (char ch : *spec) {
      if (ch == ',' || ch == '[' || ch == ']' || ch == ' ')
        flush();
      else
        token.push_back(ch);
    }
    flush();
    if (tokens.empty())
      throw std::runtime_error("UmaPot: package shape is empty: " + *spec);
    decl.known = true;
    decl.text = "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
      const std::string &tok = tokens[i];
      const bool number =
          !tok.empty() &&
          std::all_of(tok.begin(), tok.end(),
                      [](char ch) { return ch >= '0' && ch <= '9'; });
      Axis axis;
      if (number) {
        axis.fixed = std::stoll(tok);
      } else {
        axis.symbolic = true;
        axis.name = tok;
        if (tok == "nedges") {
          axis.lo = nedges_min ? *nedges_min : 0;
          axis.hi = nedges_max ? *nedges_max : axis.lo;
        } else if (tok == "natoms") {
          axis.lo = natoms_min ? *natoms_min : (natoms ? *natoms : 1);
          axis.hi = natoms_max ? *natoms_max : (natoms ? *natoms : axis.lo);
        } else {
          throw std::runtime_error(
              "UmaPot: package shape has an unknown dimension " + tok);
        }
      }
      decl.axes.push_back(axis);
      decl.text += (i ? ", " : "") + tok;
    }
    decl.text += "]";
    for (const Axis &axis : decl.axes) {
      if (!axis.symbolic)
        continue;
      decl.text += " with " + axis.name + " in [" + std::to_string(axis.lo) +
                   ", " + std::to_string(axis.hi) + "]";
    }
    return decl;
  }

  static std::string shapeText(const std::vector<int64_t> &dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i)
      s += (i ? ", " : "") + std::to_string(dims[i]);
    return s + "]";
  }

  void matchShape(const std::string &field, const ShapeDecl &decl,
                  const std::vector<int64_t> &actual) const {
    if (!decl.known)
      return;
    bool ok = actual.size() == decl.axes.size();
    for (size_t i = 0; ok && i < actual.size(); ++i) {
      const Axis &axis = decl.axes[i];
      if (axis.symbolic)
        ok = actual[i] >= axis.lo && actual[i] <= axis.hi;
      else
        ok = actual[i] == axis.fixed;
    }
    if (!ok)
      throw UmaShapeError(field, shapeText(actual), decl.text);
  }

  /// Positions are ``[natoms, 3]`` and the edge list is ``[2, nedges]``.
  /// A static package names the traced sizes. A dynamic package names
  /// the symbolic axis and its range. Either way the message carries
  /// both shapes.
  void checkCallShapes(int64_t nAtoms, int64_t nedges) const {
    matchShape("pos", pos_decl, {nAtoms, 3});
    matchShape("edge_index", edge_decl, {2, nedges});
  }

  /// Every system, on the single path and on the band path.
  ///
  /// An exact package stores the traced total, so each system must
  /// hold an equal share. A range package checks each system's own
  /// count against ``[nedges_min, nedges_max]`` instead of skipping
  /// the band. A molecular package is traced on every ordered pair,
  /// so each system must also have ``natoms * (natoms - 1)`` edges.
  void checkEachSystem(int64_t systems, const int64_t *counts,
                       int64_t natoms = 0, bool molecular = false) const {
    if (systems <= 0 || counts == nullptr)
      return;
    if (!nedges_min || !nedges_max)
      return;
    const bool exact = *nedges_min == *nedges_max;
    int64_t each_exact = 0;
    if (exact) {
      if (*nedges_min % systems != 0) {
        throw UmaContractError("nedges", std::to_string(*nedges_min),
                               "not divisible by the band", "package");
      }
      each_exact = *nedges_min / systems;
    }
    const int64_t complete = natoms > 1 ? natoms * (natoms - 1) : 0;
    for (int64_t s = 0; s < systems; ++s) {
      const int64_t count = counts[s];
      if (molecular && count != complete) {
        throw UmaShapeError("edge_index",
                            "[2, " + std::to_string(count) + "]",
                            "[2, " + std::to_string(complete) + "]");
      }
      if (exact) {
        if (count == each_exact)
          continue;
        throw UmaShapeError("edge_index",
                            "[2, " + std::to_string(count) + "]",
                            "[2, " + std::to_string(each_exact) + "]");
      }
      if (count < *nedges_min || count > *nedges_max) {
        throw UmaShapeError(
            "edge_index", "[2, " + std::to_string(count) + "]",
            "[2, nedges] with nedges in [" + std::to_string(*nedges_min) +
                ", " + std::to_string(*nedges_max) + "]");
      }
    }
  }

  /// Edge count of one call (the whole band for a batched package)
  /// against the range the compiled graph was exported for.
  void checkEdges(int64_t nedges) const {
    const bool below = nedges_min && nedges < *nedges_min;
    const bool above = nedges_max && nedges > *nedges_max;
    if (!below && !above)
      return;
    const int64_t lo = nedges_min ? *nedges_min : 0;
    const std::string want =
        nedges_max && lo == *nedges_max
            ? std::to_string(lo)
            : "[" + std::to_string(lo) + ", " +
                  (nedges_max ? std::to_string(*nedges_max) : "inf") + "]";
    throw UmaContractError("nedges", want, std::to_string(nedges), "input");
  }

  /// Atom count, element set and per-element counts of one system.
  void checkSystem(size_t nAtoms, const int *atmnrs) const {
    if (natoms && *natoms != static_cast<int64_t>(nAtoms))
      throw UmaContractError("natoms", std::to_string(*natoms),
                             std::to_string(nAtoms), "input");
    if (z_max) {
      for (size_t i = 0; i < nAtoms; ++i) {
        if (atmnrs[i] >= 1 && atmnrs[i] <= *z_max)
          continue;
        throw UmaContractError("z_max", "1 to " + std::to_string(*z_max),
                               std::to_string(atmnrs[i]), "input atomic number");
      }
    }
    if (!z_set && !counts)
      return;
    std::map<int64_t, int64_t> have;
    for (size_t i = 0; i < nAtoms; ++i)
      ++have[atmnrs[i]];
    if (counts && *counts != have)
      throw UmaContractError("counts", str(*counts), str(have), "input");
    if (z_set) {
      std::vector<int64_t> zs;
      zs.reserve(have.size());
      for (const auto &kv : have)
        zs.push_back(kv.first);
      if (zs != *z_set)
        throw UmaContractError("z_set", str(*z_set), str(zs), "input");
    }
  }
};

} // namespace rgpot
