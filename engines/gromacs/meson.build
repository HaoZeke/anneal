# librgpot_gromacs: the libcp2k C API that GROMACS built with -DGMX_CP2K=ON
# links, served by any rgpot potential. GROMACS is built after this library,
# against build/engines/gromacs/pkgconfig/libcp2k.pc, so the library needs
# no GROMACS; the mdrun tests run once gromacs_dropin_gmx names that build.
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
gmx_dropin_bench = executable(
    'rgpot-gromacs-bench',
    files('bench_dropin.cpp'),
    dependencies: [pot_spec_dep],
    link_with: [rgpot, gmx_dropin],
    include_directories: [engines_inc, rgpot_inc],
    cpp_args: _args,
    install: false,
)
gmx_dropin_build_dir = meson.current_build_dir()
gmx_dropin_source_dir = meson.current_source_dir()
subdir('pkgconfig')

gmx_dropin_here = meson.current_source_dir()
gmx_dropin_pytest = python3_engines.found() ? run_command(
    python3_engines,
    '-c',
    'import pytest, numpy',
    check: false,
).returncode() == 0 : false
if gmx_dropin_pytest
    gmx_dropin_env = environment()
    gmx_dropin_env.set('RGPOT_GROMACS_DROPIN', gmx_dropin.full_path())
    gmx_dropin_env.set('RGPOT_EVAL', rgpot_eval.full_path())
    gmx_dropin_depends = [gmx_dropin, rgpot_eval]
    if is_variable('classical_engine')
        gmx_dropin_env.set(
            'RGPOT_CLASSICAL_ENGINE',
            classical_engine.full_path(),
        )
        gmx_dropin_env.set(
            'RGPOT_CAPNP_SCHEMA',
            project_root / 'CppCore' / 'rgpot' / 'rpc' / 'Potentials.capnp',
        )
        gmx_dropin_depends += [classical_engine]
    endif
    gmx_dropin_pytest_args = [
        '-m',
        'pytest',
        '-q',
        '-p',
        'no:cacheprovider',
        '--rootdir',
        gmx_dropin_here,
    ]
    test(
        'gromacs_dropin_api',
        python3_engines,
        args: gmx_dropin_pytest_args + [files('test_dropin_api.py')],
        env: gmx_dropin_env,
        depends: gmx_dropin_depends,
        workdir: meson.current_build_dir(),
        timeout: 300,
    )

    gmx_dropin_gmx = get_option('gromacs_dropin_gmx')
    if gmx_dropin_gmx == ''
        gmx_dropin_gmx_d = find_program('gmx_d', required: false)
        gmx_dropin_gmx = gmx_dropin_gmx_d.found() ? gmx_dropin_gmx_d.full_path() : ''
    endif
    if gmx_dropin_gmx != ''
        gmx_dropin_env.set('RGPOT_GMX', gmx_dropin_gmx)
        if has_eam_table
            gmx_dropin_env.set(
                'RGPOT_EAM_ALLOY',
                project_root / 'CppCore/tests/data/eam_table/CuNi.eam.alloy',
            )
        endif
        test(
            'gromacs_dropin_mdrun',
            python3_engines,
            args: gmx_dropin_pytest_args + ['-s', files('test_dropin_mdrun.py')],
            env: gmx_dropin_env,
            depends: gmx_dropin_depends,
            workdir: meson.current_build_dir(),
            timeout: 1800,
        )
    else
        message(
            'GROMACS drop-in mdrun tests skipped: set gromacs_dropin_gmx to a gmx_d built by engines/gromacs/build_gromacs.sh',
        )
    endif
else
    message('GROMACS drop-in tests skipped: needs python3 with pytest and numpy')
endif
