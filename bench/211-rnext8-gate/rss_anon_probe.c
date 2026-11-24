#define _GNU_SOURCE

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static atomic_int running = 0;
static atomic_long peak_kb = 0;
static pthread_t sampler;
static int sampler_started = 0;
static int report_enabled = 1;

static long read_rss_anon(void) {
    FILE *status = fopen("/proc/self/status", "r");
    char line[512];
    long value = -1;

    if (status == NULL) {
        return value;
    }
    while (fgets(line, sizeof(line), status) != NULL) {
        if (sscanf(line, "RssAnon: %ld kB", &value) == 1) {
            break;
        }
    }
    fclose(status);
    return value;
}

static void note_peak(long value) {
    long seen = atomic_load(&peak_kb);

    while (value > seen &&
           !atomic_compare_exchange_weak(&peak_kb, &seen, value)) {
    }
}

static void *sample(void *unused) {
    const struct timespec delay = {.tv_sec = 0, .tv_nsec = 1000000};

    (void)unused;
    while (atomic_load(&running)) {
        note_peak(read_rss_anon());
        nanosleep(&delay, NULL);
    }
    note_peak(read_rss_anon());
    return NULL;
}

static void after_fork_child(void) {
    atomic_store(&running, 0);
    sampler_started = 0;
    report_enabled = 0;
}

__attribute__((constructor)) static void begin_sampling(void) {
    pthread_atfork(NULL, NULL, after_fork_child);
    atomic_store(&running, 1);
    note_peak(read_rss_anon());
    sampler_started = pthread_create(&sampler, NULL, sample, NULL) == 0;
}

__attribute__((destructor)) static void report_rss_anon(void) {
    long end_kb;

    if (!report_enabled) {
        return;
    }
    atomic_store(&running, 0);
    if (sampler_started) {
        pthread_join(sampler, NULL);
    }
    end_kb = read_rss_anon();
    note_peak(end_kb);
    fprintf(stderr, "rss_anon_self peak_kb=%ld end_kb=%ld\n",
            atomic_load(&peak_kb), end_kb);
}
