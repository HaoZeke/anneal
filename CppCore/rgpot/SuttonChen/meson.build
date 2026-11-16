# Static convenience lib; ships inside the librgpot umbrella (link_whole).
sutton_chen = static_library(
    'sutton_chen',
    'SuttonChenPot.cc',
    'SuttonChenScales.cc',
    dependencies: _deps,
    # The force pass reads back terms the density pass rounded, and computes
    # the rest again; fusing either into the force sum would split the bits.
    cpp_args: _args + _fp_contract_off,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
