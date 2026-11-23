/* Smoke test of the C ABI: the box ensemble through DLPack callbacks.
 *
 * Built against the installed header and library by the ffi job in
 * .github/workflows/ci.yml, and by scripts/ffi_smoke.sh. Every check
 * prints what failed and the program exits non-zero.
 */
#include <anneal-core.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, \
                    #condition);                                             \
            failures++;                                                      \
        }                                                                    \
    } while (0)

typedef struct {
    size_t evals;
    size_t stop_after;
} counter_t;

static enum anneal_status_t wells(void *user, const DLManagedTensorVersioned *x,
                                  double *value_out) {
    counter_t *counter = (counter_t *)user;
    const double *v = (const double *)x->dl_tensor.data;
    int64_t i;
    double total = 0.0;
    for (i = 0; i < x->dl_tensor.shape[0]; i++) {
        total += cos(3.0 * v[i]) + 0.05 * v[i] * v[i];
    }
    *value_out = total;
    counter->evals++;
    if (counter->stop_after != 0 && counter->evals >= counter->stop_after) {
        return ANNEAL_STOPPED;
    }
    return ANNEAL_SUCCESS;
}

static enum anneal_status_t wells_grad(void *user,
                                       const DLManagedTensorVersioned *x,
                                       DLManagedTensorVersioned *grad_out) {
    const double *v = (const double *)x->dl_tensor.data;
    double *g = (double *)grad_out->dl_tensor.data;
    int64_t i;
    (void)user;
    for (i = 0; i < x->dl_tensor.shape[0]; i++) {
        g[i] = -3.0 * sin(3.0 * v[i]) + 0.1 * v[i];
    }
    return ANNEAL_SUCCESS;
}

static void point_at(DLManagedTensorVersioned *t, double *data, int64_t *shape,
                     int64_t *strides, int64_t n) {
    shape[0] = n;
    strides[0] = 1;
    t->version.major = 1;
    t->version.minor = 0;
    t->manager_ctx = NULL;
    t->deleter = NULL;
    t->flags = 0;
    t->dl_tensor.data = data;
    t->dl_tensor.device.device_type = kDLCPU;
    t->dl_tensor.device.device_id = 0;
    t->dl_tensor.ndim = 1;
    t->dl_tensor.dtype.code = kDLFloat;
    t->dl_tensor.dtype.bits = 64;
    t->dl_tensor.dtype.lanes = 1;
    t->dl_tensor.shape = shape;
    t->dl_tensor.strides = strides;
    t->dl_tensor.byte_offset = 0;
}

typedef struct {
    enum anneal_status_t status;
    double x[2];
    anneal_box_stats_t stats;
    size_t evals;
} run_t;

static run_t run(const anneal_box_options_t *options, size_t stop_after) {
    run_t out;
    counter_t counter;
    int64_t shape[1], strides[1];
    DLManagedTensorVersioned tensor;
    const double low[2] = {-5.0, -5.0};
    const double high[2] = {5.0, 5.0};

    memset(&out, 0, sizeof out);
    counter.evals = 0;
    counter.stop_after = stop_after;
    out.x[0] = 3.0;
    out.x[1] = -2.0;
    out.stats.struct_size = sizeof out.stats;
    point_at(&tensor, out.x, shape, strides, 2);
    out.status = anneal_box_minimize_ex(wells, wells_grad, &counter, &tensor,
                                        low, high, options, &out.stats);
    out.evals = counter.evals;
    return out;
}

static anneal_box_options_t defaults(void) {
    anneal_box_options_t options;
    memset(&options, 0, sizeof options);
    options.struct_size = sizeof options;
    CHECK(anneal_box_options_default(&options) == ANNEAL_SUCCESS);
    CHECK(options.struct_size == sizeof options);
    options.seed = 4;
    options.budget = 3000;
    return options;
}

static void test_version(void) {
    const char *version = anneal_core_version();
    printf("%s\n", version);
    CHECK(version != NULL && version[0] != '\0');
}

static void test_legacy_call(void) {
    anneal_box_config_t config;
    anneal_box_report_t report;
    counter_t counter = {0, 0};
    double x[2] = {3.0, -2.0};
    int64_t shape[1], strides[1];
    DLManagedTensorVersioned tensor;
    const double low[2] = {-5.0, -5.0};
    const double high[2] = {5.0, 5.0};

    point_at(&tensor, x, shape, strides, 2);
    CHECK(anneal_box_config_default(&config) == ANNEAL_SUCCESS);
    config.seed = 4;
    config.budget = 3000;
    CHECK(anneal_box_minimize(wells, wells_grad, &counter, &tensor, low, high,
                              &config, &report) == ANNEAL_SUCCESS);
    CHECK(report.n_evals > 0 && report.n_grads > 0 && report.hops > 0);
    CHECK(report.n_evals + report.n_grads <= 3000);
    CHECK(counter.evals >= report.n_evals);
}

static void test_options_reproduce_the_legacy_call(void) {
    anneal_box_config_t config;
    anneal_box_report_t report;
    counter_t counter = {0, 0};
    double x[2] = {3.0, -2.0};
    int64_t shape[1], strides[1];
    DLManagedTensorVersioned tensor;
    const double low[2] = {-5.0, -5.0};
    const double high[2] = {5.0, 5.0};
    anneal_box_options_t options = defaults();
    run_t modern;

    point_at(&tensor, x, shape, strides, 2);
    CHECK(anneal_box_config_default(&config) == ANNEAL_SUCCESS);
    config.seed = 4;
    config.budget = 3000;
    CHECK(anneal_box_minimize(wells, wells_grad, &counter, &tensor, low, high,
                              &config, &report) == ANNEAL_SUCCESS);
    modern = run(&options, 0);
    CHECK(modern.status == ANNEAL_SUCCESS);
    CHECK(modern.x[0] == x[0] && modern.x[1] == x[1]);
    CHECK(modern.stats.value == report.value);
    CHECK(modern.stats.n_evals == report.n_evals);
    CHECK(modern.stats.n_grads == report.n_grads);
    CHECK(modern.stats.hops == report.hops);
}

static void test_each_setting_reaches_the_ensemble(void) {
    anneal_box_options_t options = defaults();
    run_t base = run(&options, 0);
    run_t merged, split, off, private_coverage, flat, none, langevin;

    CHECK(base.status == ANNEAL_SUCCESS);
    CHECK(base.stats.n_evals + base.stats.n_grads <= options.budget);
    CHECK(base.stats.history_minima > 1);
    CHECK(base.stats.shared_deposits > 0);
    CHECK(base.stats.coverage_applied_foreign > 0);
    CHECK(base.stats.coverage_repelled_proposals > 0);

    options = defaults();
    options.identity_tol = 1e6;
    merged = run(&options, 0);
    CHECK(merged.stats.history_minima == 1);

    options = defaults();
    options.identity_tol = 1e-12;
    split = run(&options, 0);
    CHECK(split.stats.history_minima > base.stats.history_minima);

    options = defaults();
    options.shared_deposits = 0;
    off = run(&options, 0);
    CHECK(off.stats.shared_deposits == 0);
    CHECK(off.stats.coverage_applied_foreign == 0);

    options = defaults();
    options.coverage_shared = ANNEAL_BOX_COVERAGE_PRIVATE;
    private_coverage = run(&options, 0);
    CHECK(private_coverage.stats.coverage_applied_foreign == 0);

    options = defaults();
    options.coverage_height = 0.0;
    flat = run(&options, 0);
    CHECK(flat.stats.coverage_repelled_proposals == 0);

    options = defaults();
    options.history = ANNEAL_BOX_HISTORY_NONE;
    none = run(&options, 0);
    CHECK(none.stats.history_minima == 0);

    options = defaults();
    options.escape = ANNEAL_BOX_ESCAPE_LANGEVIN;
    options.langevin_friction = 2.0;
    langevin = run(&options, 0);
    CHECK(langevin.status == ANNEAL_SUCCESS);
    CHECK(langevin.stats.n_evals + langevin.stats.n_grads <= options.budget);
    CHECK(langevin.stats.hops != base.stats.hops);
}

static void test_the_same_seed_gives_the_same_run(void) {
    anneal_box_options_t options = defaults();
    run_t first = run(&options, 0);
    run_t second = run(&options, 0);
    CHECK(first.stats.value == second.stats.value);
    CHECK(first.stats.n_evals == second.stats.n_evals);
    CHECK(first.x[0] == second.x[0] && first.x[1] == second.x[1]);
}

static void test_a_bad_option_is_refused_before_a_callback(void) {
    anneal_box_options_t options = defaults();
    run_t refused;
    const char *message;

    options.identity_tol = 0.0;
    refused = run(&options, 0);
    CHECK(refused.status == ANNEAL_INVALID_PARAMETER);
    CHECK(refused.evals == 0);
    message = anneal_last_error();
    CHECK(message != NULL && strstr(message, "identity_tol") != NULL);

    options = defaults();
    options.history = 9;
    refused = run(&options, 0);
    CHECK(refused.status == ANNEAL_INVALID_PARAMETER);
    CHECK(refused.evals == 0);

    options = defaults();
    options.escape = ANNEAL_BOX_ESCAPE_LANGEVIN;
    options.langevin_steps = 0;
    refused = run(&options, 0);
    CHECK(refused.status == ANNEAL_INVALID_PARAMETER);
    CHECK(refused.evals == 0);
}

static void test_struct_size_ignores_what_lies_past_it(void) {
    anneal_box_options_t options = defaults();
    run_t full = run(&options, 0);
    run_t older;

    options.struct_size = offsetof(anneal_box_options_t, history);
    options.identity_tol = -1.0;
    options.coverage_radius = NAN;
    older = run(&options, 0);
    CHECK(older.status == ANNEAL_SUCCESS);
    CHECK(older.stats.n_evals == full.stats.n_evals);

    options = defaults();
    options.struct_size = 8;
    CHECK(run(&options, 0).status == ANNEAL_INVALID_PARAMETER);
}

static void test_the_defaults_fill_only_the_callers_struct_size(void) {
    union {
        anneal_box_options_t options;
        unsigned char bytes[sizeof(anneal_box_options_t) + 16];
    } storage;
    size_t least = offsetof(anneal_box_options_t, history);
    size_t i;
    int untouched = 1;

    memset(&storage, 0xA5, sizeof storage);
    storage.options.struct_size = least;
    CHECK(anneal_box_options_default(&storage.options) == ANNEAL_SUCCESS);
    CHECK(storage.options.struct_size == least);
    CHECK(storage.options.budget == 400);
    for (i = least; i < sizeof storage; i++) {
        untouched = untouched && storage.bytes[i] == 0xA5;
    }
    CHECK(untouched);

    memset(&storage, 0xA5, sizeof storage);
    storage.options.struct_size = 8;
    CHECK(anneal_box_options_default(&storage.options) ==
          ANNEAL_INVALID_PARAMETER);
    CHECK(strstr(anneal_last_error(), "struct_size") != NULL);
    untouched = 1;
    for (i = sizeof(size_t); i < sizeof storage; i++) {
        untouched = untouched && storage.bytes[i] == 0xA5;
    }
    CHECK(untouched);
}

static void test_mid_field_struct_sizes_are_refused(void) {
    anneal_box_options_t options = defaults();
    anneal_box_stats_t stats;
    counter_t counter = {0, 0};
    double x[2] = {3.0, -2.0};
    int64_t shape[1], strides[1];
    DLManagedTensorVersioned tensor;
    const double low[2] = {-5.0, -5.0};
    const double high[2] = {5.0, 5.0};

    options.struct_size = 33;
    CHECK(anneal_box_options_default(&options) == ANNEAL_INVALID_PARAMETER);
    CHECK(run(&options, 0).status == ANNEAL_INVALID_PARAMETER);

    options = defaults();
    memset(&stats, 0, sizeof stats);
    stats.struct_size = 20;
    point_at(&tensor, x, shape, strides, 2);
    CHECK(anneal_box_minimize_ex(wells, wells_grad, &counter, &tensor, low,
                                 high, &options, &stats) ==
          ANNEAL_INVALID_PARAMETER);
    CHECK(counter.evals == 0);
}

static void test_a_box_or_start_that_is_not_finite_is_refused(void) {
    anneal_box_options_t options = defaults();
    double x[2] = {0.0, 0.0};
    int64_t shape[1], strides[1];
    DLManagedTensorVersioned tensor;
    const double wide_low[2] = {-1e308, -1.0};
    const double wide_high[2] = {1e308, 1.0};
    const double low[2] = {-5.0, -5.0};
    const double high[2] = {5.0, 5.0};
    counter_t counter = {0, 0};

    point_at(&tensor, x, shape, strides, 2);
    CHECK(anneal_box_minimize_ex(wells, wells_grad, &counter, &tensor,
                                 wide_low, wide_high, &options, NULL) ==
          ANNEAL_INVALID_PARAMETER);
    CHECK(strstr(anneal_last_error(), "finite width") != NULL);

    x[1] = NAN;
    CHECK(anneal_box_minimize_ex(wells, wells_grad, &counter, &tensor, low,
                                 high, &options, NULL) ==
          ANNEAL_INVALID_PARAMETER);
    CHECK(strstr(anneal_last_error(), "not finite") != NULL);

    x[1] = 0.0;
    shape[0] = -1;
    CHECK(anneal_box_minimize_ex(wells, wells_grad, &counter, &tensor, low,
                                 high, &options, NULL) ==
          ANNEAL_INVALID_PARAMETER);
    CHECK(strstr(anneal_last_error(), "shape") != NULL);
    CHECK(counter.evals == 0);
}

static void test_a_callback_can_stop_the_search(void) {
    anneal_box_options_t options = defaults();
    run_t stopped;

    options.budget = 20000;
    stopped = run(&options, 24);
    CHECK(stopped.status == ANNEAL_STOPPED);
    CHECK(stopped.evals >= 24 && stopped.evals <= 28);
    CHECK(isfinite(stopped.stats.value));
}

static void test_null_arguments_are_refused(void) {
    anneal_box_options_t options = defaults();
    double x[2] = {0.0, 0.0};
    int64_t shape[1], strides[1];
    DLManagedTensorVersioned tensor;
    const double bounds[2] = {-1.0, 1.0};
    counter_t counter = {0, 0};

    point_at(&tensor, x, shape, strides, 2);
    CHECK(anneal_box_options_default(NULL) == ANNEAL_INVALID_PARAMETER);
    CHECK(anneal_box_minimize_ex(wells, wells_grad, &counter, &tensor, NULL,
                                 bounds, &options, NULL) ==
          ANNEAL_INVALID_PARAMETER);
    CHECK(anneal_box_minimize_ex(wells, wells_grad, &counter, &tensor, bounds,
                                 bounds, NULL, NULL) ==
          ANNEAL_INVALID_PARAMETER);
    CHECK(counter.evals == 0);
}

int main(void) {
    test_version();
    test_legacy_call();
    test_options_reproduce_the_legacy_call();
    test_each_setting_reaches_the_ensemble();
    test_the_same_seed_gives_the_same_run();
    test_a_bad_option_is_refused_before_a_callback();
    test_struct_size_ignores_what_lies_past_it();
    test_the_defaults_fill_only_the_callers_struct_size();
    test_mid_field_struct_sizes_are_refused();
    test_a_box_or_start_that_is_not_finite_is_refused();
    test_a_callback_can_stop_the_search();
    test_null_arguments_are_refused();
    if (failures != 0) {
        fprintf(stderr, "%d checks failed\n", failures);
        return 1;
    }
    return 0;
}
