// Copyright (c) anneal developers
// In-process probe for the 153 r-next2 gate, loaded with LD_PRELOAD.
//
// It is inert unless GATE_PROBE_OUT is set and /proc/self/exe is the path
// in GATE_PROBE_EXE, so taskset and any git child stay untouched. In the
// search process a CPU-time timer (ITIMER_PROF) raises SIGPROF every
// GATE_PROBE_PERIOD_US microseconds. Each tick compares the interrupted
// instruction pointer with the address ranges in GATE_PROBE_RANGES, one
// "start size" hex pair per line in file addresses. Every
// GATE_PROBE_RSS_EVERY ticks, and once at start and at exit, the process
// reads RssAnon from its own /proc/self/status and keeps the maximum.
// The exit line goes to GATE_PROBE_OUT.

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>

#define MAX_RANGES 256

static uintptr_t range_lo[MAX_RANGES];
static uintptr_t range_hi[MAX_RANGES];
static int range_count;
static int active;
static int out_fd = -1;
static long rss_every = 10;
static long period_us = 1000;
static volatile long anon_peak_kb = -1;
static volatile unsigned long ticks;
static volatile unsigned long vector_ticks;
static volatile unsigned long rss_reads;
static volatile int status_busy;
static char status_buf[8192];

static long field_kb(const char *text, const char *key) {
    const char *at = strstr(text, key);
    if (at == NULL) {
        return -1;
    }
    at += strlen(key);
    while (*at == ' ' || *at == '\t') {
        at++;
    }
    long value = 0;
    int digits = 0;
    while (*at >= '0' && *at <= '9') {
        value = value * 10 + (*at - '0');
        at++;
        digits++;
    }
    return digits ? value : -1;
}

// Only open, read, and close, so it is safe inside the signal handler.
static int read_status(void) {
    int fd = open("/proc/self/status", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return 0;
    }
    size_t used = 0;
    for (;;) {
        ssize_t got = read(fd, status_buf + used, sizeof(status_buf) - 1 - used);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            break;
        }
        used += (size_t)got;
        if (used >= sizeof(status_buf) - 1) {
            break;
        }
    }
    close(fd);
    status_buf[used] = '\0';
    return used > 0;
}

static void sample_rss(void) {
    if (__atomic_exchange_n(&status_busy, 1, __ATOMIC_ACQUIRE)) {
        return;
    }
    if (read_status()) {
        long anon = field_kb(status_buf, "RssAnon:");
        if (anon > anon_peak_kb) {
            anon_peak_kb = anon;
        }
        __atomic_add_fetch(&rss_reads, 1, __ATOMIC_RELAXED);
    }
    __atomic_store_n(&status_busy, 0, __ATOMIC_RELEASE);
}

static void on_tick(int sig, siginfo_t *info, void *context) {
    (void)sig;
    (void)info;
    int saved = errno;
    const ucontext_t *uc = (const ucontext_t *)context;
    uintptr_t pc = (uintptr_t)uc->uc_mcontext.gregs[REG_RIP];
    unsigned long tick = __atomic_add_fetch(&ticks, 1, __ATOMIC_RELAXED);
    for (int i = 0; i < range_count; i++) {
        if (pc >= range_lo[i] && pc < range_hi[i]) {
            __atomic_add_fetch(&vector_ticks, 1, __ATOMIC_RELAXED);
            break;
        }
    }
    if (rss_every > 0 && tick % (unsigned long)rss_every == 0) {
        sample_rss();
    }
    errno = saved;
}

static int main_bias(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    *(uintptr_t *)data = (uintptr_t)info->dlpi_addr;
    return 1;
}

static int load_ranges(const char *path) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        return -1;
    }
    uintptr_t bias = 0;
    dl_iterate_phdr(main_bias, &bias);
    unsigned long start;
    unsigned long size;
    while (range_count < MAX_RANGES && fscanf(file, "%lx %lx", &start, &size) == 2) {
        range_lo[range_count] = bias + start;
        range_hi[range_count] = bias + start + size;
        range_count++;
    }
    int overflow = !feof(file) && range_count == MAX_RANGES;
    fclose(file);
    return overflow ? -1 : 0;
}

__attribute__((constructor)) static void probe_start(void) {
    const char *out = getenv("GATE_PROBE_OUT");
    const char *want = getenv("GATE_PROBE_EXE");
    if (out == NULL || want == NULL) {
        return;
    }
    char exe[4096];
    ssize_t len = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (len <= 0) {
        return;
    }
    exe[len] = '\0';
    if (strcmp(exe, want) != 0) {
        return;
    }
    const char *ranges = getenv("GATE_PROBE_RANGES");
    if (ranges == NULL || load_ranges(ranges) != 0) {
        _exit(97);
    }
    const char *every = getenv("GATE_PROBE_RSS_EVERY");
    if (every != NULL) {
        rss_every = strtol(every, NULL, 10);
    }
    const char *period = getenv("GATE_PROBE_PERIOD_US");
    if (period != NULL) {
        period_us = strtol(period, NULL, 10);
    }
    out_fd = open(out, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (out_fd < 0 || period_us <= 0) {
        _exit(98);
    }
    sample_rss();
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_sigaction = on_tick;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGPROF, &action, NULL) != 0) {
        _exit(99);
    }
    struct itimerval timer;
    timer.it_interval.tv_sec = period_us / 1000000;
    timer.it_interval.tv_usec = period_us % 1000000;
    timer.it_value = timer.it_interval;
    setitimer(ITIMER_PROF, &timer, NULL);
    active = 1;
}

__attribute__((destructor)) static void probe_stop(void) {
    if (!active) {
        return;
    }
    struct itimerval off;
    memset(&off, 0, sizeof(off));
    setitimer(ITIMER_PROF, &off, NULL);
    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGPROF);
    sigprocmask(SIG_BLOCK, &block, NULL);
    sample_rss();
    long anon_exit = field_kb(status_buf, "RssAnon:");
    long hwm = field_kb(status_buf, "VmHWM:");
    dprintf(
        out_fd,
        "probe rss_anon_peak_kb=%ld rss_anon_exit_kb=%ld vm_hwm_kb=%ld rss_reads=%lu "
        "ticks=%lu vector_ticks=%lu vector_ranges=%d period_us=%ld\n",
        anon_peak_kb, anon_exit, hwm, rss_reads, ticks, vector_ticks, range_count, period_us);
    close(out_fd);
    active = 0;
}
