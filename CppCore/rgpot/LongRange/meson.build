# Long-range summands. linkcell is the neighbour-list dependency.
lr_phys = static_library(
    'lr_phys',
    files(
        'Pairs.cc',
        'Electro.cc',
        'Dispersion.cc',
        'Nlh.cc',
        'SumPot.cc',
    ),
    dependencies: _deps + [linkcell_dep],
    cpp_args: _args,
    include_directories: _incdirs,
    pic: true,
    install: false,
)
