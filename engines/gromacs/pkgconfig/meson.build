# GROMACS' cmake/gmxManageCP2K.cmake reads CP2K_DIR/pkgconfig/libcp2k.pc.
# The build-tree file points at this build; the installed one at the prefix.
gmx_pc_where = {
    'libcp2k.pc': [
        gmx_dropin_build_dir,
        gmx_dropin_source_dir,
    ],
    'libcp2k-installed.pc': [
        get_option('prefix') / get_option('libdir'),
        get_option('prefix') / get_option('includedir') / 'rgpot' / 'gromacs',
    ],
}
foreach name, where : gmx_pc_where
    gmx_pc_conf = configuration_data()
    gmx_pc_conf.set('LIBDIR', where[0])
    gmx_pc_conf.set('INCLUDEDIR', where[1])
    gmx_pc_conf.set('VERSION', meson.project_version())
    gmx_pc = configure_file(
        input: 'libcp2k.pc.in',
        output: name,
        configuration: gmx_pc_conf,
    )
    if name == 'libcp2k-installed.pc' and not meson.is_subproject()
        install_data(
            gmx_pc,
            rename: 'libcp2k.pc',
            install_dir: get_option('libdir') / 'rgpot' / 'gromacs' / 'pkgconfig',
        )
    endif
endforeach
