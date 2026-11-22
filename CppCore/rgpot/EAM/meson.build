eam_cell = static_library(
    'eam_cell',
    ['EAM.cpp', 'EAMPot.cc'],
    dependencies: _deps,
    cpp_args: _args + _fp_contract_off + ['-DEAM_STANDALONE'],
    include_directories: ['../../', '.'],
    pic: true,
    install: false,
)
