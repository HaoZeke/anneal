# Copyright 2023--present rgpot developers
# Static convenience lib; ships inside the librgpot umbrella (link_whole).
meam = static_library(
    'meam',
    'MEAMModel.cc',
    'MEAMReader.cc',
    'MEAMPot.cc',
    dependencies: _deps,
    cpp_args: _args + _fp_contract_off,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
