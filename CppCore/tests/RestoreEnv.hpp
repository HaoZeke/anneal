#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cstdlib>
#include <string>

namespace rgpot::test_support {

class RestoreEnv {
public:
  explicit RestoreEnv(const char *name) : name_(name) {
    if (const char *value = std::getenv(name)) {
      had_ = true;
      value_ = value;
    }
  }

  void set(const char *value) const {
#ifdef _WIN32
    _putenv_s(name_, value);
#else
    setenv(name_, value, 1);
#endif
  }

  void clear() const {
#ifdef _WIN32
    _putenv_s(name_, "");
#else
    unsetenv(name_);
#endif
  }

  ~RestoreEnv() {
    if (had_)
      set(value_.c_str());
    else
      clear();
  }

  RestoreEnv(const RestoreEnv &) = delete;
  RestoreEnv &operator=(const RestoreEnv &) = delete;

private:
  const char *name_;
  bool had_ = false;
  std::string value_;
};

} // namespace rgpot::test_support
