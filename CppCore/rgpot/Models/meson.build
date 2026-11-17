model_pots = static_library(
    'model_pots',
    ['ModelPots.cc', 'ModelCapi.cc'],
    dependencies: _deps,
    cpp_args: _args + ['-DRGPOT_MODEL_BUILD'],
    include_directories: ['../../', '.'],
    pic: true,
    install: false,
)
