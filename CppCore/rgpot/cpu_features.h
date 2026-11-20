// Copyright 2023--present rgpot developers

#ifndef RGPOT_CPU_FEATURES_H
#define RGPOT_CPU_FEATURES_H

#if defined(__GNUC__)
#define RGPOT_HIDDEN __attribute__((visibility("hidden")))
#else
#define RGPOT_HIDDEN
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#include <stdint.h>

#define RGPOT_AVX2_CLONE 1
#define RGPOT_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define RGPOT_NOINLINE __attribute__((noinline))

static uint64_t rgpot_xgetbv0(void) {
  uint32_t eax;
  uint32_t edx;
  __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
  return ((uint64_t)edx << 32) | eax;
}

static RGPOT_NOINLINE int rgpot_detect_avx2_fma(void) {
  unsigned int eax;
  unsigned int ebx;
  unsigned int ecx;
  unsigned int edx;

  if (__get_cpuid_max(0, 0) < 7 || !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return 0;
  }
  if ((ecx & (1U << 12)) == 0 || (ecx & (1U << 27)) == 0 ||
      (ecx & (1U << 28)) == 0 || (rgpot_xgetbv0() & 0x6U) != 0x6U) {
    return 0;
  }
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return 0;
  }
  return (ebx & (1U << 5)) != 0;
}

static int rgpot_have_avx2_fma(void) {
  static int cached;
  int value = __atomic_load_n(&cached, __ATOMIC_ACQUIRE);
  if (value == 0) {
    int expected = 0;
    const int detected = rgpot_detect_avx2_fma() ? 2 : 1;
    if (__atomic_compare_exchange_n(&cached, &expected, detected, 0,
                                    __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
      value = detected;
    } else {
      value = expected;
    }
  }
  return value == 2;
}
#else
static int rgpot_have_avx2_fma(void) { return 0; }
#endif

#endif
