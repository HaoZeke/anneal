# Public LAMMPS / GROMACS / OpenMM constructors. No engine library is
# linked here: LAMMPS dlopens its plugin, and the other two dlopen a
# native-unit library only when the caller passes a path.

external_pot_dl = dependency('dl', required: false)
if not external_pot_dl.found()
    external_pot_dl = meson.get_compiler('cpp').find_library(
        'dl',
        required: false,
    )
endif

external_pot = static_library(
    'external_pot',
    'ExternalPot.cc',
    dependencies: _deps_sans_core + [external_pot_dl],
    cpp_args: _args,
    include_directories: _incdirs,
    pic: true,
    install: false,
)

external_pot_dep = declare_dependency(
    link_with: external_pot,
    dependencies: [external_pot_dl],
    include_directories: _incdirs,
)
