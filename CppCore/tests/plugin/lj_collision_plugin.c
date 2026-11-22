// MIT License
// Copyright 2023--present rgpot developers

/*
 * External plugin that claims the built-in name lennard-jones.
 * The registry must refuse it and keep the built-in.
 */

#include <rgpot/plugin.h>

static rgpot_plugin_status_t clash_create(const char *config,
                                          rgpot_plugin_instance **out) {
  (void)config;
  *out = (rgpot_plugin_instance *)(void *)1;
  return RGPOT_PLUGIN_OK;
}

static void clash_destroy(rgpot_plugin_instance *instance) { (void)instance; }

static rgpot_plugin_status_t clash_calculate(rgpot_plugin_instance *instance,
                                             const rgpot_plugin_input_t *input,
                                             rgpot_plugin_output_t *output) {
  (void)instance;
  (void)input;
  (void)output;
  return RGPOT_PLUGIN_ERROR;
}

static const rgpot_plugin_descriptor_t clash_desc = {
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "lennard-jones",
    "9.9.9",
    NULL,
    NULL,
    clash_create,
    clash_destroy,
    clash_calculate,
    NULL,
    NULL,
};

RGPOT_PLUGIN_EXPORT const rgpot_plugin_descriptor_t *rgpot_plugin_init(void) {
  return &clash_desc;
}
