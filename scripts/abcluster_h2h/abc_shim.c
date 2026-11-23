// Copyright (c) anneal developers
//
// LD_PRELOAD shim for the ABCluster 3.4 Linux `atom` executable.
//
// ABCluster reports neither its energy-and-gradient calls nor the time it
// spends in the potential, and it takes no random seed. The shim adds all
// three without changing what the program computes:
//
//   * `time` returns 1000000000 + $ABC_SEED and `srand` seeds with $ABC_SEED.
//     The program seeds its `rand` stream, a Boost minstd generator in a
//     static initialiser and a Mersenne twister, each from `time`, so every
//     stream it draws from is a function of the seed. Its own run-time
//     report reads 0 seconds and means nothing; the driver times the run.
//   * The entry of the force-field kernel (`ForceFieldLennardJones::EnergyGrad`
//     or `ForceFieldGupta::EnergyGrad`) and of `PotentialEnergySurface::
//     LocalOptimization` (one quench) are patched in memory to run a counter
//     and a clock around the original function. The kernel is the call that
//     returns an energy and writes a gradient, so one kernel call is one
//     energy-and-gradient call. The clock is `clock_gettime(CLOCK_MONOTONIC)`
//     read once on entry and once on return, the way the anneal arm reads
//     `Instant::now()` around its potential.
//   * The run stops at the first quench whose energy is at or below
//     $ABC_TARGET (a hit), or at the kernel call after the $ABC_BUDGET-th
//     (a miss), writes one `key=value` line to $ABC_OUT and exits.
//
// The search clock starts at the first kernel call, which is the energy of
// the initial guess, after the input is read and before any bee moves.
//
// The patch is checked against the bytes of the v3.4 Linux release. Any
// other build is refused. A hooked function's first instructions are copied
// to a trampoline and replaced by an absolute jump to a stub; the stub
// saves the argument registers, calls `abc_enter`, and returns into the
// original through the trampoline. The return address is replaced by
// `abc_exit_stub`, which calls `abc_leave` with the saved return registers.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define HIDDEN __attribute__((visibility("hidden"), used))
#define PATCH_BYTES 14
#define MAX_COPY 16
#define MAX_DEPTH 64

struct hook {
    const char *name;
    uint64_t offset;
    size_t copy;
    uint8_t expect[MAX_COPY];
};

static const struct hook KERNEL_LJ = {
    "ForceFieldLennardJones::EnergyGrad", 0x2ccc6, 14,
    {0xf3, 0x0f, 0x1e, 0xfa, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x55, 0x53}};
static const struct hook KERNEL_GUPTA = {
    "ForceFieldGupta::EnergyGrad", 0x38cb4, 15,
    {0xf3, 0x0f, 0x1e, 0xfa, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x49, 0x89, 0xd4}};
static const struct hook QUENCH = {
    "PotentialEnergySurface::LocalOptimization", 0x2d676, 14,
    {0xf3, 0x0f, 0x1e, 0xfa, 0x41, 0x56, 0x41, 0x55, 0x49, 0x89, 0xd5, 0x41, 0x54, 0x55}};

struct frame {
    void *ret;
    int id;
    uint64_t t0;
    uint64_t rdi, rsi, rdx;
};

static __thread struct frame shadow[MAX_DEPTH];
static __thread int depth;

HIDDEN void *abc_tramp0;
HIDDEN void *abc_tramp1;
HIDDEN uint64_t abc_calls;
HIDDEN uint64_t abc_budget;
static uint64_t t_start, inside_ns, quenches;
static double target, best_quench = 1e300, best_kernel = 1e300;
static uint64_t trace_left;
static double trace[8];
static unsigned trace_n;
static const char *out_path;
static unsigned long seed_value;
static int seed_given;
static int started;

extern void abc_stub0(void);
extern void abc_stub1(void);
extern void abc_exit_stub(void);

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void finish(int hit, uint64_t calls_hit, const char *stop) {
    uint64_t t = now_ns();
    uint64_t wall = started ? t - t_start : 0;
    FILE *out = fopen(out_path, "w");
    if (out) {
        fprintf(out,
                "hit=%d calls=%llu calls_at_hit=%llu wall_ns=%llu inside_ns=%llu "
                "quenches=%llu best_quench=%.10f best_kernel=%.10f stop=%s seed=%lu",
                hit, (unsigned long long)abc_calls, (unsigned long long)calls_hit,
                (unsigned long long)wall, (unsigned long long)inside_ns,
                (unsigned long long)quenches, best_quench, best_kernel, stop, seed_value);
        for (unsigned i = 0; i < trace_n; i++) fprintf(out, " e%u=%.12f", i, trace[i]);
        fprintf(out, "\n");
        fclose(out);
    }
    fflush(NULL);
    _exit(0);
}

HIDDEN void abc_enter(int id, void **slot, uint64_t *saved) {
    if (id == 0) {
        if (abc_calls >= abc_budget) finish(0, abc_budget, "budget");
        abc_calls++;
        if (!started) {
            started = 1;
            t_start = now_ns();
        }
    }
    if (depth >= MAX_DEPTH) abort();
    struct frame *f = &shadow[depth++];
    f->ret = *slot;
    f->id = id;
    f->rdi = saved[5];
    f->rsi = saved[4];
    f->rdx = saved[3];
    *slot = (void *)abc_exit_stub;
    f->t0 = now_ns();
}

HIDDEN void *abc_leave(uint64_t *saved) {
    uint64_t t = now_ns();
    struct frame *f = &shadow[--depth];
    if (f->id == 0) {
        double energy;
        memcpy(&energy, &saved[0], sizeof energy);
        inside_ns += t - f->t0;
        if (energy < best_kernel) best_kernel = energy;
        if (trace_n < trace_left) trace[trace_n++] = energy;
    } else {
        double energy = *(double *)f->rdx;
        quenches++;
        if (energy < best_quench) best_quench = energy;
        if (energy <= target) finish(1, abc_calls, "hit");
    }
    return f->ret;
}

__asm__(
    ".text\n"
    ".globl abc_stub0\n.hidden abc_stub0\n.type abc_stub0,@function\n"
    "abc_stub0:\n"
    "  pushq %rdi\n  pushq %rsi\n  pushq %rdx\n  pushq %rcx\n  pushq %r8\n  pushq %r9\n"
    "  subq $8, %rsp\n"
    "  movl $0, %edi\n"
    "  leaq 56(%rsp), %rsi\n"
    "  leaq 8(%rsp), %rdx\n"
    "  call abc_enter\n"
    "  addq $8, %rsp\n"
    "  popq %r9\n  popq %r8\n  popq %rcx\n  popq %rdx\n  popq %rsi\n  popq %rdi\n"
    "  jmp *abc_tramp0(%rip)\n"
    ".size abc_stub0,.-abc_stub0\n"
    ".globl abc_stub1\n.hidden abc_stub1\n.type abc_stub1,@function\n"
    "abc_stub1:\n"
    "  pushq %rdi\n  pushq %rsi\n  pushq %rdx\n  pushq %rcx\n  pushq %r8\n  pushq %r9\n"
    "  subq $8, %rsp\n"
    "  movl $1, %edi\n"
    "  leaq 56(%rsp), %rsi\n"
    "  leaq 8(%rsp), %rdx\n"
    "  call abc_enter\n"
    "  addq $8, %rsp\n"
    "  popq %r9\n  popq %r8\n  popq %rcx\n  popq %rdx\n  popq %rsi\n  popq %rdi\n"
    "  jmp *abc_tramp1(%rip)\n"
    ".size abc_stub1,.-abc_stub1\n"
    ".globl abc_exit_stub\n.hidden abc_exit_stub\n.type abc_exit_stub,@function\n"
    "abc_exit_stub:\n"
    "  subq $64, %rsp\n"
    "  movaps %xmm0, 0(%rsp)\n"
    "  movaps %xmm1, 16(%rsp)\n"
    "  movq %rax, 32(%rsp)\n"
    "  movq %rdx, 40(%rsp)\n"
    "  movq %rsp, %rdi\n"
    "  call abc_leave\n"
    "  movq %rax, %r11\n"
    "  movaps 0(%rsp), %xmm0\n"
    "  movaps 16(%rsp), %xmm1\n"
    "  movq 32(%rsp), %rax\n"
    "  movq 40(%rsp), %rdx\n"
    "  addq $64, %rsp\n"
    "  jmp *%r11\n"
    ".size abc_exit_stub,.-abc_exit_stub\n");

static int load_bias_cb(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    *(uintptr_t *)data = (uintptr_t)info->dlpi_addr;
    return 1;
}

static void die(const char *what) {
    fprintf(stderr, "abc_shim: %s\n", what);
    _exit(97);
}

static void install(const struct hook *hook, uintptr_t bias, void *stub, void **tramp_slot) {
    uint8_t *target_fn = (uint8_t *)(bias + hook->offset);
    if (memcmp(target_fn, hook->expect, hook->copy) != 0) {
        fprintf(stderr, "abc_shim: %s at %p:", hook->name, (void *)target_fn);
        for (size_t i = 0; i < hook->copy; i++) fprintf(stderr, " %02x", target_fn[i]);
        fprintf(stderr, "\n");
        die("function bytes differ from the ABCluster 3.4 Linux release");
    }
    uint8_t *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) die("trampoline page");
    memcpy(page, target_fn, hook->copy);
    uint8_t *jump = page + hook->copy;
    uint64_t back = (uint64_t)(uintptr_t)(target_fn + hook->copy);
    jump[0] = 0xff;
    jump[1] = 0x25;
    memset(jump + 2, 0, 4);
    memcpy(jump + 6, &back, 8);
    *tramp_slot = page;
    uintptr_t base = (uintptr_t)target_fn & ~(uintptr_t)4095;
    size_t span = ((uintptr_t)target_fn + hook->copy) - base;
    if (mprotect((void *)base, span, PROT_READ | PROT_WRITE | PROT_EXEC)) die("mprotect rwx");
    uint64_t addr = (uint64_t)(uintptr_t)stub;
    target_fn[0] = 0xff;
    target_fn[1] = 0x25;
    memset(target_fn + 2, 0, 4);
    memcpy(target_fn + 6, &addr, 8);
    for (size_t i = PATCH_BYTES; i < hook->copy; i++) target_fn[i] = 0x90;
    if (mprotect((void *)base, span, PROT_READ | PROT_EXEC)) die("mprotect rx");
}

// The program ended before the budget or a hit: its own generation limit.
__attribute__((destructor)) static void abc_exit(void) {
    if (out_path && abc_tramp0) finish(0, abc_calls, "gmax");
}

__attribute__((constructor)) static void abc_init(void) {
    const char *kind = getenv("ABC_KERNEL");
    const char *text;
    if (strcmp(program_invocation_short_name, "atom") != 0) return;
    if ((text = getenv("ABC_SEED"))) {
        seed_value = strtoul(text, NULL, 10);
        seed_given = 1;
    }
    if (!kind) return;
    out_path = getenv("ABC_OUT");
    if (!out_path) die("ABC_OUT is not set");
    abc_budget = (text = getenv("ABC_BUDGET")) ? strtoull(text, NULL, 10) : UINT64_MAX;
    target = (text = getenv("ABC_TARGET")) ? strtod(text, NULL) : -1e300;
    trace_left = (text = getenv("ABC_TRACE")) ? strtoull(text, NULL, 10) : 0;
    if (trace_left > 8) trace_left = 8;
    uintptr_t bias = 0;
    dl_iterate_phdr(load_bias_cb, &bias);
    const struct hook *kernel;
    if (!strcmp(kind, "lj")) kernel = &KERNEL_LJ;
    else if (!strcmp(kind, "gupta")) kernel = &KERNEL_GUPTA;
    else die("ABC_KERNEL is lj or gupta");
    install(kernel, bias, (void *)abc_stub0, &abc_tramp0);
    install(&QUENCH, bias, (void *)abc_stub1, &abc_tramp1);
}

// ABCluster seeds from the wall clock; seed with the driver's value instead.
__attribute__((visibility("default"))) void srand(unsigned int requested) {
    void (*real)(unsigned int) = (void (*)(unsigned int))dlsym(RTLD_NEXT, "srand");
    real(seed_given ? (unsigned int)seed_value : requested);
}

__attribute__((visibility("default"))) time_t time(time_t *out) {
    static time_t (*real)(time_t *);
    if (seed_given) {
        time_t fixed = 1000000000 + (time_t)seed_value;
        if (out) *out = fixed;
        return fixed;
    }
    if (!real) real = (time_t (*)(time_t *))dlsym(RTLD_NEXT, "time");
    return real(out);
}
