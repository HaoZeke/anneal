# librgpot_gromacs: the libcp2k C API that GROMACS built with -DGMX_CP2K=ON
# links, served by any rgpot potential. GROMACS is built after this library,
# against build/engines/gromacs/pkgconfig/libcp2k.pc, so the library needs
# no GROMACS.
gmx_dropin = shared_library(
    'rgpot_gromacs',
    files('rgpot_gromacs.cpp'),
    dependencies: [pot_spec_dep, dl_dep],
    link_with: rgpot,
    include_directories: [engines_inc, rgpot_inc],
    cpp_args: _args + [
        '-DRGPOT_GROMACS_VERSION="@0@"'.format(meson.project_version()),
    ],
    gnu_symbol_visibility: 'hidden',
    install: not meson.is_subproject(),
    install_rpath: get_option('prefix') / get_option('libdir'),
)
if not meson.is_subproject()
    install_headers(files('libcp2k.h'), subdir: 'rgpot' / 'gromacs')
endif
gmx_dropin_build_dir = meson.current_build_dir()
gmx_dropin_source_dir = meson.current_source_dir()
subdir('pkgconfig')
