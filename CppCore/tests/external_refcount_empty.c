/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * A shared library with none of the engine symbols. A constructor that
 * opens this file must close it again.
 */

#include <stdio.h>
#include <stdlib.h>

static void record_lifecycle(const char *event) {
  const char *path = getenv("RGPOT_EXTERNAL_LIFECYCLE_LOG");
  if (path == NULL || path[0] == '\0') {
    return;
  }
  FILE *log = fopen(path, "a");
  if (log != NULL) {
    fprintf(log, "%s\n", event);
    fclose(log);
  }
}

__attribute__((constructor)) static void empty_load(void) {
  record_lifecycle("empty load");
}

__attribute__((destructor)) static void empty_unload(void) {
  record_lifecycle("empty unload");
}

void rgpot_external_refcount_empty_anchor(void) {}
