# Static convenience lib; ships inside the librgpot umbrella (link_whole).
# Contraction off. sumPairs inlines the one-thread half walk and the row
# walk into this file, and a compiler that contracts across statements
# fuses the two copies differently. The stored list then depends on the
# thread count on a target with a fused multiply-add, which arm64 is.
zbl = static_library(
    'zbl',
    'ZBLPot.cc',
    dependencies: _deps,
    cpp_args: _args + _fp_contract_off,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
