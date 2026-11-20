// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/engine_c_abi.h"

extern "C" RGPOT_ENGINE_API RgpotEngineAbiStamp rgpot_engine_abi_stamp(void) {
  return {
      RGPOT_ENGINE_ABI_VERSION,
      RGPOT_ENGINE_ABI_MINOR,
      RGPOT_ENGINE_ABI_LAYOUT_REVISION,
  };
}
