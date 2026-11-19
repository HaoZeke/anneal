# Static convenience lib; ships inside the librgpot umbrella (link_whole).
eam_table = static_library(
    'eam_table',
    'EAMTablePot.cc',
    'EAMTableImages.cc',
    dependencies: _deps,
    cpp_args: _args,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
