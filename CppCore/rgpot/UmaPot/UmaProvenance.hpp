#pragma once
// MIT License — the checkpoint a UMA AOTI package was exported from

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace rgpot {

/**
 * @brief Where the weights of a ``.pt2`` came from, and their digest.
 *
 * Built from the package metadata, as UmaContract is. Every field is
 * optional: packages exported before these keys report all of them
 * absent and load as before. ``scripts/export_uma_aoti.py`` writes
 *
 * - ``checkpoint_source``: ``cli`` (``--checkpoint``), ``env``
 *   (``RGPOT_UMA_CHECKPOINT``) or ``hf`` (downloaded for ``--model``);
 * - ``checkpoint_path``: the local file, for ``cli`` and ``env``;
 * - ``checkpoint_hf_repo``, ``checkpoint_hf_filename`` and
 *   ``checkpoint_hf_revision`` (the 40-digit commit the file was served
 *   from), for ``hf``;
 * - ``checkpoint_sha256``: SHA-256 of the checkpoint file the exporter
 *   loaded, 64 lowercase hexadecimal digits.
 *
 * The digest names the weights, not the ``.pt2``; the engine ABI's
 * ``rgpot_engine_model_digest`` hashes the package file itself.
 */
struct UmaCheckpointProvenance {
  std::optional<std::string> source;
  std::optional<std::string> path;
  std::optional<std::string> hf_repo;
  std::optional<std::string> hf_filename;
  std::optional<std::string> hf_revision;
  std::optional<std::string> sha256;

  /// True when the package records any of the keys.
  [[nodiscard]] bool present() const noexcept {
    return source || path || hf_repo || hf_filename || hf_revision || sha256;
  }

  static UmaCheckpointProvenance
  fromMetadata(const std::unordered_map<std::string, std::string> &meta) {
    auto get = [&](const char *key) -> std::optional<std::string> {
      const auto it = meta.find(key);
      if (it == meta.end() || it->second.empty())
        return std::nullopt;
      return it->second;
    };
    auto lower_hex = [](const std::string &s, std::size_t n) {
      return s.size() == n && std::all_of(s.begin(), s.end(), [](char ch) {
               return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
             });
    };
    UmaCheckpointProvenance p;
    p.source = get("checkpoint_source");
    p.path = get("checkpoint_path");
    p.hf_repo = get("checkpoint_hf_repo");
    p.hf_filename = get("checkpoint_hf_filename");
    p.hf_revision = get("checkpoint_hf_revision");
    p.sha256 = get("checkpoint_sha256");
    if (p.source && *p.source != "cli" && *p.source != "env" &&
        *p.source != "hf")
      throw std::runtime_error(
          "UmaPot: package metadata checkpoint_source is not cli, env or hf: " +
          *p.source);
    if (p.sha256 && !lower_hex(*p.sha256, 64))
      throw std::runtime_error("UmaPot: package metadata checkpoint_sha256 is "
                               "not 64 lowercase hexadecimal digits: " +
                               *p.sha256);
    if (p.hf_revision && !lower_hex(*p.hf_revision, 40))
      throw std::runtime_error(
          "UmaPot: package metadata checkpoint_hf_revision is not a commit "
          "hash: " +
          *p.hf_revision);
    return p;
  }
};

} // namespace rgpot
