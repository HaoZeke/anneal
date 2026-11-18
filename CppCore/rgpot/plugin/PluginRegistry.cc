// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/plugin/PluginRegistry.hpp"
#include "rgpot/plugin/Descriptor.hpp"
#include "rgpot/plugin/DynLib.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace rgpot::plugin {
const rgpot_plugin_descriptor_t *builtinLennardJones();
const rgpot_plugin_descriptor_t *builtinClassicalMorse();
const rgpot_plugin_descriptor_t *builtinSuttonChen();
#ifdef RGPOT_HAS_EAM_TABLE
const rgpot_plugin_descriptor_t *builtinEamFuncfl();
const rgpot_plugin_descriptor_t *builtinEamSetfl();
#endif

PluginRegistry &PluginRegistry::instance() {
  static PluginRegistry registry;
  return registry;
}

PluginRegistry::PluginRegistry() {
  register_builtin(builtinLennardJones());
  register_builtin(builtinClassicalMorse());
  register_builtin(builtinSuttonChen());
#ifdef RGPOT_HAS_EAM_TABLE
  register_builtin(builtinEamFuncfl());
  register_builtin(builtinEamSetfl());
#endif
}

void PluginRegistry::discover() {
  const char *path = std::getenv("RGPOT_PLUGIN_PATH");
  if (!path)
    return;
#ifdef _WIN32
  constexpr char delimiter = ';';
#else
  constexpr char delimiter = ':';
#endif
  std::istringstream stream(path);
  std::string directory;
  while (std::getline(stream, directory, delimiter)) {
    if (!directory.empty())
      scan_directory(directory);
  }
}

void PluginRegistry::scan_directory(const std::string &directory) {
  std::error_code error;
  if (!std::filesystem::is_directory(directory, error))
    return;
  std::vector<std::filesystem::path> paths;
  for (std::filesystem::directory_iterator it(directory, error), end;
       !error && it != end; it.increment(error)) {
    std::error_code entry_error;
    if (it->is_regular_file(entry_error) &&
        it->path().extension() == dynlib::so_extension())
      paths.push_back(it->path());
  }
  std::sort(paths.begin(), paths.end());
  for (const auto &path : paths)
    load_plugin(path.string());
}

bool PluginRegistry::load_plugin(const std::string &path) {
  const auto handle = dynlib::open(path.c_str());
  if (!handle)
    return false;
  const std::shared_ptr<void> library(
      static_cast<void *>(handle),
      [](void *value) { dynlib::close(static_cast<dynlib::Handle>(value)); });
  const auto init =
      dynlib::loadSym<rgpot_plugin_init_fn>(handle, "rgpot_plugin_init");
  if (!init)
    return false;
  const rgpot_plugin_descriptor_t *descriptor = nullptr;
  try {
    descriptor = init();
    detail::checkedDescriptor(descriptor);
  } catch (const std::exception &error) {
    std::cerr << "[rgpot] invalid plugin " << path << ": " << error.what()
              << '\n';
    return false;
  }
  std::lock_guard lock(m_mutex);
  for (const auto &plugin : m_plugins) {
    if (std::string(plugin.desc->name) == descriptor->name) {
      std::cerr << "[rgpot] plugin name collision " << path << ": "
                << descriptor->name << " is already registered\n";
      return false;
    }
  }
  m_plugins.push_back({static_cast<void *>(handle), descriptor, path, library});
  return true;
}

void PluginRegistry::register_builtin(
    const rgpot_plugin_descriptor_t *descriptor) {
  detail::checkedDescriptor(descriptor);
  std::lock_guard lock(m_mutex);
  for (const auto &plugin : m_plugins) {
    if (std::string(plugin.desc->name) == descriptor->name)
      throw std::invalid_argument("plugin name is already registered");
  }
  m_plugins.push_back({nullptr, descriptor, {}, {}});
}

std::vector<std::string> PluginRegistry::list_plugins() const {
  std::lock_guard lock(m_mutex);
  std::vector<std::string> names;
  names.reserve(m_plugins.size());
  for (const auto &plugin : m_plugins)
    names.emplace_back(plugin.desc->name);
  return names;
}

const LoadedPlugin *PluginRegistry::find(const std::string &name) const {
  std::lock_guard lock(m_mutex);
  for (const auto &plugin : m_plugins) {
    if (name == plugin.desc->name)
      return &plugin;
  }
  return nullptr;
}

std::shared_ptr<void> PluginRegistry::retain_library(
    const rgpot_plugin_descriptor_t *descriptor) const {
  std::lock_guard lock(m_mutex);
  for (const auto &plugin : m_plugins) {
    if (plugin.desc == descriptor)
      return plugin.library;
  }
  return {};
}

} // namespace rgpot::plugin
