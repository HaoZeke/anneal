#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>

static void trim_newline(char *text) {
    text[strcspn(text, "\r\n")] = '\0';
}

__attribute__((destructor))
static void report_rss_anon(void) {
    FILE *status = fopen("/proc/self/status", "r");
    FILE *comm_file = fopen("/proc/self/comm", "r");
    char line[512];
    char comm[128] = "unknown";
    char affinity[128] = "unknown";
    long rss_anon = -1;

    if (comm_file != NULL) {
        if (fgets(comm, sizeof(comm), comm_file) != NULL) {
            trim_newline(comm);
        }
        fclose(comm_file);
    }
    if (status != NULL) {
        while (fgets(line, sizeof(line), status) != NULL) {
            if (sscanf(line, "RssAnon: %ld kB", &rss_anon) == 1) {
                continue;
            }
            if (strncmp(line, "Cpus_allowed_list:", 18) == 0) {
                char *value = line + 18;
                while (*value == ' ' || *value == '\t') {
                    value++;
                }
                snprintf(affinity, sizeof(affinity), "%s", value);
                trim_newline(affinity);
            }
        }
        fclose(status);
    }
    fprintf(
        stderr,
        "rss_anon_self_kb=%ld exec_image=%s affinity=%s\n",
        rss_anon,
        comm,
        affinity
    );
}
