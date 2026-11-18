#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <deque>
#include <memory>
#include <mutex>
#include <rgpot/plugin.h>
#include <string>
#include <vector>

namespace rgpot::plugin {

struct LoadedPlugin {
  void *handle;
  const rgpot_plugin_descriptor_t *desc;
  std::string path;
  std::shared_ptr<void> library;
};

class PluginRegistry {
public:
  static PluginRegistry &instance();
  /// Load descriptors from the directories named by RGPOT_PLUGIN_PATH.
  void discover();
  /// Return false for an invalid library or a name that is already
  /// registered. A collision keeps the existing plugin, including a
  /// built-in, and prints the reason on standard error.
  bool load_plugin(const std::string &path);
  /// The descriptor and callbacks must have static lifetime.
  void register_builtin(const rgpot_plugin_descriptor_t *desc);
  std::vector<std::string> list_plugins() const;
  /// Returned entries remain valid across further registrations.
  const LoadedPlugin *find(const std::string &name) const;
  /// Keep a registered descriptor's library mapped for a live instance.
  std::shared_ptr<void>
  retain_library(const rgpot_plugin_descriptor_t *desc) const;

  PluginRegistry(const PluginRegistry &) = delete;
  PluginRegistry &operator=(const PluginRegistry &) = delete;

private:
  PluginRegistry();
  ~PluginRegistry() = default;
  void scan_directory(const std::string &directory);
  mutable std::mutex m_mutex;
  std::deque<LoadedPlugin> m_plugins;
};

} // namespace rgpot::plugin
