# POSIX sockets only. No third-party dependency.
ipi = static_library(
    'ipi',
    'IPIWire.cc',
    'IPIPot.cc',
    dependencies: _deps,
    cpp_args: _args,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
