# Engine drivers. Each one is built when its library is found.
# with_* = enabled and a missing library is an error.

fs = import('fs')
engines_inc = include_directories('.')
rgpot_inc = include_directories('../CppCore', '../include')
# include_directories() rejects an absolute path that sits inside the
# source tree. A pixi prefix and a LAMMPS checkout under the workspace
# both do. A path relative to this file is accepted; a path outside the
# tree can stay absolute.
project_root = meson.project_source_root()
engines_here = meson.current_source_dir()
dl_dep = cppc.find_library('dl', required: false)
thread_dep = dependency('threads')
lj_lib = static_library(
    'rgpot_lj_eval',
    files('lj_eval.cpp'),
    include_directories: [engines_inc, rgpot_inc],
    dependencies: dl_dep,
    cpp_args: _args,
    pic: true,
)
lj_dep = declare_dependency(
    link_with: lj_lib,
    include_directories: [engines_inc, rgpot_inc],
)

pot_spec_lib = static_library(
    'rgpot_pot_spec',
    files('pot_spec.cpp'),
    include_directories: [engines_inc, rgpot_inc],
    cpp_args: _args,
    pic: true,
)
pot_spec_dep = declare_dependency(
    link_with: pot_spec_lib,
    include_directories: [engines_inc, rgpot_inc],
)
rgpot_eval = executable(
    'rgpot-eval',
    files('rgpot_eval.cpp'),
    dependencies: pot_spec_dep,
    link_with: rgpot,
    cpp_args: _args,
)

rgpot_lj = executable(
    'rgpot-lj',
    files('rgpot_lj.cpp'),
    dependencies: lj_dep,
    link_with: rgpot,
    include_directories: rgpot_inc,
    cpp_args: _args,
    install: not meson.is_subproject(),
)
test(
    'lj_eval',
    executable(
        'test_lj_eval',
        files('test_lj_eval.cpp'),
        dependencies: lj_dep,
        link_with: rgpot,
        cpp_args: _args,
    ),
    timeout: 60,
)

lj_engine = shared_library(
    'lj_engine',
    files('lj_engine.cpp'),
    dependencies: lj_dep,
    link_with: rgpot,
    include_directories: rgpot_inc,
    cpp_args: _args + ['-DRGPOT_ENGINE_BUILD'],
    install: not meson.is_subproject(),
)
test(
    'lj_engine',
    executable(
        'test_engine',
        files('test_engine.cpp'),
        dependencies: lj_dep,
        link_with: rgpot,
        cpp_args: _args,
    ),
    args: [lj_engine.full_path()],
    timeout: 60,
)

capi_threads = executable(
    'capi_threads',
    files('capi_threads.c'),
    link_with: rgpot,
    include_directories: rgpot_inc,
    dependencies: thread_dep,
    link_language: 'cpp',
)
test('capi_threads', capi_threads, timeout: 60)
# A second binary built only with the thread sanitizer. b_sanitize is
# already on every compile line, and -fsanitize=thread cannot be combined
# with address or undefined. It is a test binary, so it follows
# with_tests: gcc-toolset-14 in the manylinux images that build the wheel
# accepts -fsanitize=thread when compiling but has no libtsan to link.
if (
    get_option('with_tests')
    and cppc.has_argument('-fsanitize=thread')
    and get_option('b_sanitize') == 'none'
)
    capi_tsan = executable(
        'capi_threads_tsan',
        files('capi_threads.c'),
        link_with: rgpot,
        include_directories: rgpot_inc,
        c_args: ['-fsanitize=thread'],
        link_args: ['-fsanitize=thread'],
        dependencies: thread_dep,
        link_language: 'cpp',
    )
    # GCC's runtime (14 through 16) aborts with "unexpected memory
    # mapping" when the kernel randomises mmap with more than 28 bits, as
    # Ubuntu 24.04 and later do. LLVM's runtime re-executes itself with
    # ASLR off; setarch -R does that for GCC's. A sandbox may refuse the
    # personality call, so the wrapper is used only when it works here.
    tsan_cmd = capi_tsan
    tsan_args = []
    setarch = find_program('setarch', required: false)
    if (
        not meson.is_cross_build()
        and setarch.found()
        and run_command(setarch, '-R', 'true', check: false).returncode() == 0
    )
        tsan_cmd = setarch
        tsan_args = ['-R', capi_tsan]
    endif
    test('capi_threads_tsan', tsan_cmd, args: tsan_args, timeout: 120)
endif

python3_engines = find_program('python3', required: false)

lammps_opt = get_option('with_lammps')
if lammps_opt.allowed() and python3_engines.found()
    # Drop '' and '.'. Meson runs this from engines/, and engines/lammps
    # would otherwise import as the LAMMPS module.
    lammps_py = run_command(
        python3_engines,
        '-c',
        'import sys; sys.path = [p for p in sys.path if p not in ("", ".")]; import lammps',
        check: false,
    )
    if lammps_py.returncode() != 0 and lammps_opt.enabled()
        error(
            'with_lammps is enabled but the lammps Python module was not found',
        )
    endif
    if lammps_py.returncode() == 0
        lammps_args = [rgpot_lj.full_path()]
        lammps_src = get_option('lammps_src')
        if lammps_src != ''
            # -DLAMMPS_SMALLBIG matches the default LAMMPS integer layout.
            # The plugin is loaded by an lmp built from the same headers.
            # A conda liblammps is a different build and is not that loader.
            lammps_dep = cppc.find_library('lammps', required: false)
            lammps_exe = get_option('lammps_exe')
            lammps_inc = lammps_src.startswith(project_root + '/') ? fs.relative_to(
                lammps_src,
                engines_here,
            ) : lammps_src
            # A serial lmp is built against src/STUBS/mpi.h and its library
            # exports those MPI functions. The plugin's gather and scatter
            # must use the same header, or they bind to the system MPI.
            lammps_stubs = false
            if lammps_exe != ''
                lammps_probe_env = environment()
                lammps_lib = fs.parent(lammps_exe) / 'liblammps.so'
                if fs.exists(lammps_lib)
                    lammps_probe_env.set('LD_PRELOAD', lammps_lib)
                endif
                lammps_help = run_command(
                    lammps_exe,
                    '-h',
                    check: false,
                    env: lammps_probe_env,
                )
                lammps_stubs = lammps_help.stdout().contains('LAMMPS MPI STUBS')
            endif
            pair_plugin = shared_library(
                'pair_rgpot',
                files('lammps/pair_rgpot.cpp'),
                dependencies: [lj_dep, pot_spec_dep] + (
                    lammps_dep.found() ? [lammps_dep] : []
                ),
                link_with: rgpot,
                include_directories: include_directories(
                    (lammps_stubs ? [lammps_inc / 'STUBS'] : []) + [lammps_inc],
                ),
                # The MPI C++ bindings of Open MPI 4 would leave
                # MPI::Win::Free undefined; LAMMPS skips them too.
                cpp_args: _args + [
                    '-DLAMMPS_SMALLBIG',
                    '-DOMPI_SKIP_MPICXX',
                    '-DMPICH_SKIP_MPICXX',
                ],
                override_options: ['b_lundef=false'],
            )
            if lammps_exe != ''
                lammps_args += [pair_plugin.full_path(), lammps_exe]
                lammps_pytest = run_command(
                    python3_engines,
                    '-c',
                    'import pytest',
                    check: false,
                )
                if lammps_pytest.returncode() == 0
                    pair_env = environment()
                    pair_env.set('RGPOT_PAIR_PLUGIN', pair_plugin.full_path())
                    pair_env.set('RGPOT_LMP', lammps_exe)
                    pair_env.set('RGPOT_EVAL', rgpot_eval.full_path())
                    pair_env.set('RGPOT_LJ_ENGINE', lj_engine.full_path())
                    if has_eam_table
                        pair_env.set(
                            'RGPOT_EAM_ALLOY',
                            project_root / 'CppCore/tests/data/eam_table/CuNi.eam.alloy',
                        )
                    endif
                    lammps_mpirun = find_program(
                        'mpirun',
                        'mpiexec',
                        required: false,
                    )
                    # Each rank of a serial lmp is a world of one, so the
                    # MPI cases skip instead of comparing rank counts.
                    if lammps_mpirun.found() and not lammps_stubs
                        pair_env.set('RGPOT_MPIRUN', lammps_mpirun.full_path())
                    endif
                    test(
                        'lammps_pair_rgpot',
                        python3_engines,
                        args: [
                            '-m',
                            'pytest',
                            '-q',
                            '-p',
                            'no:cacheprovider',
                            '--rootdir',
                            engines_here / 'lammps',
                            files('lammps/test_pair_rgpot.py'),
                        ],
                        env: pair_env,
                        workdir: meson.current_build_dir(),
                        timeout: 900,
                    )
                endif
            endif
        elif lammps_opt.enabled()
            error(
                'with_lammps is enabled but lammps_src is empty, so pair_style rgpot cannot be built',
            )
        endif
        test(
            'lammps_lj',
            python3_engines,
            args: [files('lammps/run_lammps.py')] + lammps_args,
            timeout: 180,
        )
    endif
endif

gromacs_opt = get_option('with_gromacs')
plumed_opt = get_option('with_plumed')
have_plumed = false
if plumed_opt.allowed()
    plumed_dep = dependency('plumed', required: false)
    if not plumed_dep.found() and plumed_opt.enabled()
        error('with_plumed is enabled but pkg-config plumed was not found')
    endif
    if plumed_dep.found()
        plumed_kernel = cppc.find_library('plumedKernel', required: false)
        plumed_link = [plumed_dep]
        if plumed_kernel.found()
            plumed_link += [plumed_kernel]
        endif
        have_plumed = true
        plumed_lib = shared_library(
            'RgpotPlumed',
            files('plumed/RgpotColvar.cpp'),
            dependencies: plumed_link + [lj_dep],
            link_with: rgpot,
            include_directories: [engines_inc, rgpot_inc],
            cpp_args: _args,
            name_prefix: 'lib',
            install: not meson.is_subproject(),
        )
        if python3_engines.found() and find_program('plumed', required: false).found()
            test(
                'plumed_driver',
                python3_engines,
                args: [
                    files('plumed/run_plumed.py'),
                    rgpot_lj.full_path(),
                    plumed_lib.full_path(),
                ],
                timeout: 180,
            )
        endif
    endif
endif

if gromacs_opt.allowed()
    gmx = find_program('gmx', required: gromacs_opt.enabled())
    if gmx.found() and have_plumed and python3_engines.found()
        test(
            'gromacs_mdrun',
            python3_engines,
            args: [
                files('gromacs/run_gromacs.py'),
                rgpot_lj.full_path(),
                plumed_lib.full_path(),
                meson.current_build_dir() / 'gromacs-work',
            ],
            timeout: 180,
        )
    elif gromacs_opt.enabled()
        error('with_gromacs is enabled but gmx or the PLUMED module is missing')
    endif
endif

if python3_engines.found()
    test(
        'c_symbols',
        python3_engines,
        args: [files('../docs/check_c_symbols.py')],
        timeout: 30,
    )
endif

# Configures a parent project around this tree, so it needs meson on PATH
# and only runs from the top-level build.
meson_prog = find_program('meson', required: false)
if (
    get_option('with_tests')
    and not meson.is_subproject()
    and python3_engines.found()
    and meson_prog.found()
)
    test(
        'optional_engine_install',
        python3_engines,
        args: [
            files('test_optional_engine_install.py'),
            meson.project_source_root(),
            meson_prog.full_path(),
        ],
        timeout: 300,
    )
endif

abi_script = find_program('../scripts/abi_symbols.sh')
abi_baselines = files('../scripts/abi_baseline.txt')
if has_fast_nlist
    abi_baselines += files('../scripts/abi_baseline_fast.txt')
endif
test(
    'abi_symbols',
    abi_script,
    args: [rgpot.full_path(), abi_baselines],
    timeout: 30,
)
# abi_symbols above reads only the kind of librgpot this build makes. A
# small fixture built as a shared object, a lib*.a archive and a *.lib
# archive runs the script's other nm modes in every build. The two
# should_fail tests keep the check strict: a shared object hides hidden
# names, and an archive still reports a missing one.
if get_option('with_tests')
    abi_fixture_src = files('abi_fixture.c')
    abi_fixture_baseline = files('abi_fixture_baseline.txt')
    abi_fixture_so = shared_library('rgpot_abi_fixture', abi_fixture_src)
    abi_fixture_a = static_library('rgpot_abi_fixture_a', abi_fixture_src)
    abi_fixture_lib = static_library(
        'rgpot_abi_fixture_lib',
        abi_fixture_src,
        name_prefix: '',
        name_suffix: 'lib',
    )
    foreach _kind : [
        ['shared', abi_fixture_so],
        ['archive_a', abi_fixture_a],
        ['archive_lib', abi_fixture_lib],
    ]
        test(
            'abi_symbols_fixture_' + _kind[0],
            abi_script,
            args: [_kind[1].full_path(), abi_fixture_baseline],
            depends: _kind[1],
            timeout: 30,
        )
    endforeach
    test(
        'abi_symbols_fixture_shared_hidden',
        abi_script,
        args: [abi_fixture_so.full_path(), files('abi_fixture_hidden.txt')],
        depends: abi_fixture_so,
        should_fail: true,
        timeout: 30,
    )
    test(
        'abi_symbols_fixture_archive_missing',
        abi_script,
        args: [abi_fixture_a.full_path(), abi_baselines],
        depends: abi_fixture_a,
        should_fail: true,
        timeout: 30,
    )
endif

openmm_opt = get_option('with_openmm')
if openmm_opt.allowed() and python3_engines.found()
    # Drop '' and '.'. Meson runs this from engines/, and engines/openmm
    # would otherwise import as the OpenMM module.
    openmm_py = run_command(
        python3_engines,
        '-c',
        'import sys; sys.path = [p for p in sys.path if p not in ("", ".")]; import openmm',
        check: false,
    )
    if openmm_py.returncode() != 0 and openmm_opt.enabled()
        error(
            'with_openmm is enabled but the openmm Python module was not found',
        )
    endif
    if openmm_py.returncode() == 0
        test(
            'openmm_lj',
            python3_engines,
            args: [files('openmm/run_openmm.py'), rgpot_lj.full_path()],
            timeout: 120,
        )
    endif
endif

native_lj_engine = shared_library(
    'native_lj_engine',
    files('native_lj_engine.cpp'),
    dependencies: lj_dep,
    link_with: rgpot,
    include_directories: rgpot_inc,
    cpp_args: _args + ['-DRGPOT_ENGINE_BUILD'],
    install: not meson.is_subproject(),
)
external_pot_test = executable(
    'test_external_potentials',
    files('test_external_potentials.cpp'),
    dependencies: lj_dep,
    link_with: rgpot,
    include_directories: rgpot_inc,
    cpp_args: _args,
)
external_env = environment()
if openmm_opt.allowed() and python3_engines.found()
    _openmm_where = run_command(
        python3_engines,
        '-c',
        'import openmm.version; print(openmm.version.openmm_library_path)',
        check: false,
    )
    if _openmm_where.returncode() == 0
        openmm_libdir = _openmm_where.stdout().strip()
        openmm_incdir = fs.parent(openmm_libdir) / 'include'
        openmm_inc = include_directories(
            openmm_incdir.startswith(project_root + '/') ? fs.relative_to(
                openmm_incdir,
                engines_here,
            ) : openmm_incdir,
        )
        openmm_lib = cppc.find_library(
            'OpenMM',
            dirs: [openmm_libdir],
            required: openmm_opt.enabled(),
        )
        if openmm_lib.found() and cppc.has_header(
            'openmm/Platform.h',
            include_directories: openmm_inc,
        )
            openmm_engine = shared_library(
                'openmm_engine',
                files('openmm/openmm_engine.cpp'),
                dependencies: [openmm_lib],
                include_directories: [rgpot_inc, openmm_inc],
                cpp_args: _args + ['-DRGPOT_ENGINE_BUILD'],
                build_rpath: openmm_libdir,
                install_rpath: openmm_libdir,
                install: not meson.is_subproject(),
            )
            external_env.set('RGPOT_OPENMM_ENGINE', openmm_engine.full_path())
            external_env.set('OPENMM_PLUGIN_DIR', openmm_libdir / 'plugins')
        endif
    endif
endif
test(
    'external_potentials',
    external_pot_test,
    args: [lj_engine.full_path(), native_lj_engine.full_path()],
    env: external_env,
    timeout: 120,
)

gromacs_inc = ''
if python3_engines.found()
    _gmx_inc = run_command(
        python3_engines,
        '-c',
        'import pathlib, sys; print(pathlib.Path(sys.prefix) / "include")',
        check: false,
    )
    if _gmx_inc.returncode() == 0
        gromacs_inc = _gmx_inc.stdout().strip()
    endif
endif
if gromacs_inc != ''
    gromacs_headers = include_directories(
        gromacs_inc.startswith(project_root + '/') ? fs.relative_to(
            gromacs_inc,
            engines_here,
        ) : gromacs_inc,
    )
    if cppc.has_header(
        'gromacs/math/units.h',
        include_directories: gromacs_headers,
    )
        test(
            'gromacs_unit_reference',
            executable(
                'test_gromacs_units',
                files('test_gromacs_units.cpp'),
                include_directories: [engines_inc, rgpot_inc, gromacs_headers],
                cpp_args: _args,
            ),
            timeout: 30,
        )
    endif
endif
