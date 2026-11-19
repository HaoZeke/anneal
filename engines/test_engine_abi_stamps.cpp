// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/engine_c_abi.h"

#include <cstddef>
#include <cstdio>
#include <dlfcn.h>

static_assert(RGPOT_ENGINE_ABI_VERSION == 1);
static_assert(RGPOT_ENGINE_ABI_MINOR == 4);
static_assert(RGPOT_ENGINE_ABI_LAYOUT_REVISION == 1);
static_assert(sizeof(RgpotEngineAbiStamp) == 3 * sizeof(unsigned short));
static_assert(offsetof(RgpotEngineAbiStamp, abi_major) == 0);
static_assert(offsetof(RgpotEngineAbiStamp, abi_minor) ==
              sizeof(unsigned short));
static_assert(offsetof(RgpotEngineAbiStamp, layout_revision) ==
              2 * sizeof(unsigned short));

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "no engine libraries supplied\n");
    return 2;
  }

  bool failed = false;
  for (int i = 1; i < argc; ++i) {
    void *handle = dlopen(argv[i], RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      std::fprintf(stderr, "%s: dlopen: %s\n", argv[i], dlerror());
      failed = true;
      continue;
    }

    dlerror();
    void *symbol = dlsym(handle, "rgpot_engine_abi_stamp");
    const char *error = dlerror();
    if (error != nullptr || symbol == nullptr) {
      std::fprintf(stderr, "%s: missing rgpot_engine_abi_stamp\n", argv[i]);
      failed = true;
    } else {
      using StampFn = RgpotEngineAbiStamp (*)();
      const auto stamp = reinterpret_cast<StampFn>(symbol)();
      if (stamp.abi_major != RGPOT_ENGINE_ABI_VERSION ||
          stamp.abi_minor != RGPOT_ENGINE_ABI_MINOR ||
          stamp.layout_revision != RGPOT_ENGINE_ABI_LAYOUT_REVISION) {
        std::fprintf(stderr,
                     "%s: stamp is {%hu, %hu, %hu}, expected {%d, %d, %d}\n",
                     argv[i], stamp.abi_major, stamp.abi_minor,
                     stamp.layout_revision, RGPOT_ENGINE_ABI_VERSION,
                     RGPOT_ENGINE_ABI_MINOR, RGPOT_ENGINE_ABI_LAYOUT_REVISION);
        failed = true;
      }
    }
    dlclose(handle);
  }
  return failed ? 1 : 0;
}
