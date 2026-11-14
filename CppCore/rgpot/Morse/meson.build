# Static convenience lib; ships inside the librgpot umbrella (link_whole).
# Contraction off: the row kernel's clones and the scalar switched kernel
# must round every product and sum on their own. _vector_loops lets the row
# loop vectorise (see CppCore/meson.build).
morse = static_library(
    'morse',
    'MorsePot.cc',
    'cambridge_morse.c',
    dependencies: _deps,
    cpp_args: _args + _fp_contract_off + _vector_loops,
    c_args: _fp_contract_off,
    include_directories: ['../../', '.'],
    pic: true,
    install: false,
)
