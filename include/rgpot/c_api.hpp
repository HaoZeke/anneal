#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file c_api.hpp
 * @brief Include @c rgpot.h as a C ABI from C++.
 *
 * cbindgen emits unmangled C symbols without an @c extern "C" guard.
 */

#include "eindir_abi.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include "rgpot.h"
#ifdef __cplusplus
}
#endif
