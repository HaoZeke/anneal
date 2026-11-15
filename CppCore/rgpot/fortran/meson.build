# Fortran-backed potentials, ported to Fortran 2018.
#
# Sources are listed in dependency order for compilers without automatic
# module ordering; meson's Fortran scanner reorders as needed.
#
# Only the bind(c) entry points need to leave the enclosing librgpot, and
# C++ callers reach them from inside the same shared object, so the whole
# static library builds with hidden visibility: no bare Fortran symbol
# (or legacy COMMON block name) ever reaches the dynamic table.

_fpot_args = fc.get_supported_arguments(
    [
        '-std=f2018',
        '-fimplicit-none',
        '-Wall',
        '-Wno-unused-dummy-argument',
        '-fvisibility=hidden',
        # F2: gfortran keeps some procedure locals in static storage unless
        # this is set, and those locals are then shared by every thread.
        # The _ws entries are one workspace per caller and must be reentrant.
        '-frecursive',
    ],
)
# rgpot_water_h.f90 selects the minimage wrap with the preprocessor.
if fc.get_id() == 'intel' or fc.get_id() == 'intel-cl'
    _fpot_args += ['-fpp']
else
    _fpot_args += ['-cpp']
endif
if has_fast_nlist
    _fpot_args += ['-DRGPOT_HAS_FAST_NLIST']
endif

fortran_pots_sources = files(
    'rgpot_cuh2.f90',
    'rgpot_cuh2_capi.f90',
    'rgpot_eam_al.f90',
    'rgpot_eam_al_capi.f90',
    'rgpot_edip.f90',
    'rgpot_edip_capi.f90',
    'rgpot_fehe.f90',
    'rgpot_fehe_capi.f90',
    'rgpot_ferror.f90',
    'rgpot_kinds.f90',
    'rgpot_lenosky.f90',
    'rgpot_lenosky_capi.f90',
    'rgpot_neighbors.f90',
    'rgpot_tersoff.f90',
    'rgpot_tersoff_capi.f90',
    'rgpot_water_h.f90',
    'rgpot_water_h_capi.f90',
    'rgpot_workspace.f90',
)

# meson passes gfortran no depfile flags, and its scanner reads `include`
# lines only to find modules, so an edit of sw_centred_body.inc would leave
# the rgpot_sw object stale. The kernel compiles from a copy that depends
# on the include. gfortran then finds the include through the -I of this
# source directory.
_rgpot_sw_f90 = custom_target(
    'rgpot_sw_f90',
    input: 'rgpot_sw.f90',
    output: 'rgpot_sw.f90',
    command: [
        find_program('python3', 'python'),
        '-c',
        'import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])',
        '@INPUT@',
        '@OUTPUT@',
    ],
    depend_files: files('sw_centred_body.inc'),
)

# C++ faces over the bind(c) entries.
fortran_pots_cpp_sources = files('FortranPots.cc')

# Kernels only. gfortran ignores -fvisibility for module symbols, so the
# archive is kept separate and hidden at link time (see the umbrella
# library's --exclude-libs).
fortran_kernels = static_library(
    'rgpot_fortran_kernels',
    fortran_pots_sources,
    'water_h_image.cc',
    dependencies: [],
    fortran_args: _fpot_args,
    # Same uncontracted products as the cluster kernel. The scan is the
    # only copy; fusing a product here would move the H-water image.
    cpp_args: cppc.get_supported_arguments(
        ['-ffp-contract=off', '/fp:precise', '-fvisibility=hidden'],
    ),
    pic: true,
    install: false,
)

# The copied rgpot_sw.f90 is generated, so meson cannot scan its `use`
# lines until the copy exists and a parallel build compiles it before
# rgpot_kinds.mod and rgpot_neighbors.mod. It is its own library, ordered
# after the kernels. The C API uses that module, so it is a third library.
# Both archives are hidden in the umbrella library with the kernels.
rgpot_sw_lib = static_library(
    'rgpot_sw_lib',
    _rgpot_sw_f90,
    fortran_args: _fpot_args,
    include_directories: fortran_kernels.private_dir_include(),
    link_with: fortran_kernels,
    pic: true,
    install: false,
)
rgpot_sw_capi_lib = static_library(
    'rgpot_sw_capi_lib',
    files('rgpot_sw_capi.f90'),
    fortran_args: _fpot_args,
    include_directories: [
        rgpot_sw_lib.private_dir_include(),
        fortran_kernels.private_dir_include(),
    ],
    link_with: [rgpot_sw_lib, fortran_kernels],
    pic: true,
    install: false,
)
# The C++ faces stay visible: consumers construct these classes.
fortran_pots = static_library(
    'rgpot_fortran_pots',
    fortran_pots_cpp_sources,
    dependencies: _deps,
    cpp_args: _args,
    include_directories: _incdirs,
    link_with: [
        rgpot_sw_capi_lib,
        rgpot_sw_lib,
        fortran_kernels,
        tersoff_sic,
        rgpot_halfshell,
    ],
    pic: true,
    install: false,
)

fortran_pots_dep = declare_dependency(
    link_with: [fortran_pots, rgpot_sw_capi_lib, rgpot_sw_lib, fortran_kernels],
    dependencies: [],
)

# Fortran-only test programs link the kernels and the neighbour query.
# The OpenMP dependency is the C++ one. Its compile flag (/openmp, -fopenmp)
# must not be handed to flang, which treats /openmp as a source file.
_fortran_nlist_omp = dependency('openmp', language: 'cpp', required: false)
_fortran_nlist_omp_link = []
if _fortran_nlist_omp.found()
    _fortran_nlist_omp_link = [
        _fortran_nlist_omp.partial_dependency(
            compile_args: false,
            includes: false,
        ),
    ]
endif
fortran_kernels_dep = declare_dependency(
    link_with: [
        rgpot_sw_capi_lib,
        rgpot_sw_lib,
        fortran_kernels,
        rgpot_halfshell,
        rgpot_minimage,
        rgpot_exact_fma,
    ],
    dependencies: [fastnlist_dep] + _fortran_nlist_omp_link,
)

# Plugin anneal dlopens as RGPOT_CUH2_LIBRARY. The umbrella librgpot hides
# Fortran symbols; this .so exports only the bind(c) entry.
_cuh2_plugin_args = fc.get_supported_arguments(
    [
        '-std=f2018',
        '-fimplicit-none',
        '-Wall',
        '-Wno-unused-dummy-argument',
        '-frecursive',
    ],
)
rgpot_cuh2 = library(
    'rgpot_cuh2',
    [
        'rgpot_kinds.f90',
        'rgpot_ferror.f90',
        'rgpot_neighbors.f90',
        'rgpot_workspace.f90',
        'rgpot_cuh2.f90',
        'rgpot_cuh2_capi.f90',
    ],
    link_with: [rgpot_halfshell, rgpot_minimage, rgpot_exact_fma],
    # C++ is the link language so the half-shell archives resolve, and meson
    # then does not add the Fortran runtime. fortran_rt_deps is the same
    # search the umbrella uses for _gfortran_* (and the flang equivalents).
    link_language: 'cpp',
    dependencies: _fortran_nlist_omp_link + fortran_rt_deps,
    fortran_args: _cuh2_plugin_args,
    install: not meson.is_subproject(),
    gnu_symbol_visibility: 'default',
)

# Physics checks that run without the C++ stack: a failure here points at
# the kernel, not the bindings.
if get_option('with_tests')
    test_sw_gather = executable(
        'test_sw_gather',
        'test_sw_gather.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranSWGather', test_sw_gather, timeout: 120)

    test_cuh2_gather = executable(
        'test_cuh2_gather',
        'test_cuh2.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranCuH2Gather', test_cuh2_gather, timeout: 120)

    test_eam_al = executable(
        'test_eam_al',
        'test_eam_al.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranEAMAlGather', test_eam_al, timeout: 120)

    test_water_h = executable(
        'test_water_h',
        'test_water_h.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranWaterHGather', test_water_h, timeout: 120)

    test_fehe = executable(
        'test_fehe',
        'test_fehe.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranFeHeGather', test_fehe, timeout: 120)

    test_tersoff = executable(
        'test_tersoff',
        'test_tersoff.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranTersoffGather', test_tersoff, timeout: 180)

    test_edip = executable(
        'test_edip',
        'test_edip.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranEDIPGather', test_edip, timeout: 180)

    test_lenosky = executable(
        'test_lenosky',
        'test_lenosky.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranLenoskyGather', test_lenosky, timeout: 180)
endif
