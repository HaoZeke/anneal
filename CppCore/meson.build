_fs = import('fs')
_rev_env = run_command(
    'sh',
    '-c',
    'printf %s "$RGPOT_SOURCE_REVISION"',
    check: false,
).stdout().strip()
_rev_git = run_command(
    'git',
    '-C',
    meson.project_source_root(),
    'rev-parse',
    '--short=12',
    'HEAD',
    check: false,
)
_build_revision = _rev_env
if _build_revision == '' and _rev_git.returncode() == 0
    _build_revision = _rev_git.stdout().strip()
endif
_build_identity_args = [
    '-DRGPOT_BUILD_VERSION="@0@"'.format(meson.project_version()),
    '-DRGPOT_BUILD_REVISION="@0@"'.format(_build_revision),
]
if _fs.read('rgpot/rpc/Potentials.capnp').contains('buildVersion')
    _build_identity_args += ['-DRGPOT_SCHEMA_HAS_BUILD_IDENTITY']
endif
_args += _build_identity_args

# tests/registry.txt lists every test with the features that enable it;
# cmake/RgpotTestRegistry.cmake reads the same file. The unit, fortran,
# nlist, py and rerun entries this build enables land in _rt_tests and are
# registered in loops below. _rt_kw holds each entry's test() options. The
# custom entries' gates are checked by test_list_parity, which also takes
# the features only known later (see the end of meson.build).
_rt_atoms = {
    'tests': get_option('with_tests'),
    'examples': get_option('with_examples'),
    'core': not get_option('with_rpc_client_only'),
    'client': get_option('with_rpc_client_only'),
    'rpc': rpc_enabled,
    'fortran': has_fortran,
    'cache': get_option('with_cache'),
    'ipi': has_ipi and not get_option('with_rpc_client_only'),
    'eam_table': has_eam_table,
    'eigen': has_eigen,
    'xtensor': has_xtensor,
    'python': find_program('python3', required: false).found(),
    'water': has_water,
    'tblite': has_tblite,
    'dftd3': has_dftd3,
    'dftd4': has_dftd4,
    'long_range': has_long_range,
    'metatomic': has_metatomic,
    'rust_core': get_option('with_rust_core'),
}
_rt_tests = []
_rt_kw = {}
_tsan_supp = meson.project_source_root() / 'CppCore' / 'tests' / 'tsan.supp'
_sanitize = get_option('b_sanitize')
_timeout_scale = 1
if 'thread' in _sanitize
    _timeout_scale = 8
elif _sanitize != 'none'
    _timeout_scale = 3
endif
_tsan_options = run_command(
    'sh',
    '-c',
    'printf %s "$TSAN_OPTIONS"',
    check: false,
).stdout()
if _tsan_options != ''
    _tsan_options += ':'
endif
_tsan_options += 'suppressions=' + _tsan_supp
foreach _line : _fs.read('tests/registry.txt').split('\n')
    _line = _line.strip()
    if _line == '' or _line.startswith('#') or _line.startswith('@')
        continue
    endif
    _fields = _line.split('|')
    _name = _fields[0].strip()
    _words = _fields[2].strip().split()
    _kind = _words[0]
    _positional = []
    _unfused = false
    _workdir = _kind == 'unit' ? '.' : ''
    _env = {}
    _kw = {}
    _has_timeout = false
    _first = true
    foreach _word : _words
        if _first
            _first = false
        elif _word.startswith('timeout=')
            _has_timeout = true
            _kw += {
                'timeout': _word.substring(8).to_int() * _timeout_scale,
            }
        elif _word.startswith('suite=')
            _kw += {'suite': _word.substring(6)}
        elif _word.startswith('workdir=')
            _workdir = _word.substring(8)
        elif _word.startswith('env=')
            _pair = _word.substring(4).split('=')
            _env += {_pair[0]: _pair[1]}
        elif _word == 'serial'
            _kw += {'is_parallel': false}
        elif _word == 'should_fail'
            _kw += {'should_fail': true}
        elif _word == 'fp_contract=off'
            _unfused = true
        else
            _positional += [_word]
        endif
    endforeach
    if _workdir == '.'
        _kw += {'workdir': meson.project_source_root()}
    elif _workdir != ''
        _kw += {'workdir': meson.project_source_root() / _workdir}
    endif
    if _has_timeout and _timeout_scale != 1
        _env += {'RGPOT_TEST_TIMEOUT_SCALE': _timeout_scale.to_string()}
    endif
    if 'thread' in _sanitize and _kind != 'custom'
        _env += {'TSAN_OPTIONS': _tsan_options}
    endif
    if _env.keys().length() > 0
        _kw += {'env': _env}
    endif
    _rt_kw += {_name: _kw}
    if _kind != 'custom'
        _on = true
        foreach _clause : _fields[1].strip().split()
            _any = false
            foreach _atom : _clause.split(',')
                _any = _any or _rt_atoms[_atom]
            endforeach
            _on = _on and _any
        endforeach
        if _on
            _rt_tests += [[_name, _kind, _positional, _unfused]]
        endif
    endif
endforeach

if get_option('with_cache') and not get_option('with_rpc_client_only')
    _deps += dependency('rocksdb')
    _deps += subproject('xxhash', default_options: ['cli=false']).get_variable(
        'xxhash_dep',
    )
    _args += ['-DRGPOT_HAS_CACHE=TRUE']
    _rgpot_srcs += files('rgpot/PotentialCache.cc')
endif

# ELF soversion of librgpot. The single assignment is kCxxSoversion in
# rgpot/abi/CxxLayout.hpp, next to the C++ layout stamp. A layout edit
# that bumps the constant moves every build system's SONAME with it.
_cxx_layout_hpp = _fs.read('rgpot/abi/CxxLayout.hpp')
_cxx_so_parts = _cxx_layout_hpp.split('kCxxSoversion = ')
if _cxx_so_parts.length() != 2
    error(
        'CxxLayout.hpp must contain exactly one "kCxxSoversion = " assignment',
    )
endif
rgpot_cxx_soversion = _cxx_so_parts[1].split(';')[0].strip()

# Calculator groups split MPI_COMM_WORLD and broadcast results between
# groups. librgpot itself never links MPI: the MPI side is librgpot_mpi,
# which CalculatorGroup.cc loads with dlopen when a host asks for groups
# (calculator_mpi_abi.h). Without it a process is one group of one rank.
_mpi_dep = dependency('mpi', language: 'cpp', required: get_option('with_mpi'))
_dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
_deps += _dl_dep
_rgpot_srcs += files('rgpot/CalculatorGroup.cc')
rgpot_mpi_lib = disabler()
rgpot_mpi_dep = disabler()
if _mpi_dep.found()
    rgpot_mpi_lib = shared_library(
        'rgpot_mpi',
        files('rgpot/CalculatorGroupMpi.cc'),
        dependencies: [_mpi_dep],
        include_directories: include_directories('.'),
        gnu_symbol_visibility: 'hidden',
        version: meson.project_version(),
        # Own SONAME. RGPOT_MPI_ABI_VERSION is still 1, and 3.4.0 did not
        # ship this library, so librgpot_mpi.so.3 does not name two layouts.
        # The loader opens RGPOT_MPI_LIBRARY_NAME, which is that filename.
        soversion: '3',
        install: not meson.is_subproject(),
    )
    # Hosts never need to link this: librgpot finds it next to itself.
    # Tests and hosts that want it loaded up front link it.
    rgpot_mpi_dep = declare_dependency(
        link_with: rgpot_mpi_lib,
        dependencies: [_mpi_dep],
        compile_args: ['-DRGPOT_HAS_MPI'],
    )
endif
_rgpot_srcs += files('rgpot/PotHelpers.cc')
_rgpot_srcs += files('rgpot/units.cc')
_rgpot_srcs += files('rgpot/abi/CxxLayout.cc')
_rgpot_srcs += files(
    'rgpot/simd/dispatch.cc',
    'rgpot/simd/scalar_kernels.cc',
)
if have_highway
    _rgpot_srcs += files('rgpot/simd/hwy_kernels.cc')
endif
# Product-sums in MinimumImage go through rgpot::simd::mul_add. Contraction
# stays off so a multiply that is not written as mul_add is not fused, and
# the folded bits match on scalar, SSE, AVX2, AVX-512 and NEON.
# exact_fma_hw is a separate object: the ISA variants below recompile
# MinimumImage.cpp with -mno-fma and still need this hardware definition.
# exp.cc is compiled once, here: the fixed-sequence exp bits must not depend
# on contraction, and ExpFixedEq links this archive alone.
_fp_contract_off = cppc.get_supported_arguments(
    ['-ffp-contract=off', '/fp:precise'],
)
# Lets GCC if-convert the exp clamps and use sqrtpd: neither flag changes a
# computed value (no errno from sqrt, no FP traps assumed), and the vector
# loops then run the scalar body's operations lane by lane. GCC's -O2 cost
# model (very-cheap) refuses loops that need an epilogue, so the dynamic
# model is asked for here too.
_vector_loops = cppc.get_supported_arguments(
    [
        '-fno-trapping-math',
        '-fno-math-errno',
        '-ftree-vectorize',
        '-fvect-cost-model=dynamic',
    ],
)
# target("fma"), target("avx2") and target_clones emit ifunc resolvers.
# ThreadSanitizer instruments the resolver, which then calls
# __tsan_func_entry through the PLT while RTLD_NOW is still binding it,
# and the extension segfaults on load. These units keep the resolvers
# (the bits match the baseline), so the instrumentation stays off them.
# clones.hpp drops target_clones only when the unit itself is instrumented.
_no_tsan_ifunc = []
if 'thread' in get_option('b_sanitize')
    _no_tsan_ifunc = ['-fno-sanitize=thread']
endif
rgpot_exact_fma = static_library(
    'rgpot_exact_fma',
    files('rgpot/simd/exact_fma_hw.cc', 'rgpot/simd/exp.cc'),
    include_directories: _incdirs,
    cpp_args: _args + _fp_contract_off + _vector_loops + _no_tsan_ifunc,
    pic: true,
    install: false,
)
rgpot_minimage = static_library(
    'rgpot_minimage',
    files('rgpot/nlist/MinimumImage.cpp'),
    include_directories: _incdirs,
    cpp_args: _args + _fp_contract_off + _no_tsan_ifunc,
    pic: true,
    install: false,
)
# threadsRequested lives in Slices.cc. The regions are templates in
# Slices.hpp, but the thread count is read here, so this archive has to be
# built with OpenMP or every potential stays on one thread.
_halfshell_omp = dependency('openmp', language: 'cpp', required: false)
rgpot_halfshell = static_library(
    'rgpot_halfshell',
    files(
        'rgpot/Slices.cc',
        'rgpot/nlist/HalfShell.cc',
        'rgpot/nlist/HalfShellWalk.cc',
        'rgpot/nlist/NeighbourList.cc',
        'rgpot/nlist/Neighbours.cc',
    ),
    include_directories: _incdirs,
    cpp_args: _args,
    dependencies: _halfshell_omp.found() ? [_halfshell_omp] : [],
    pic: true,
    install: false,
)

# Cap'n Proto schema + NWChem/CPMD frontends need libcapnp (ptlrpc). Only pull
# them in when RPC is on (server or client-only); lean -Dwith_rpc=false builds
# (xtb/tblite/metatomic CI) must not require pkg-config 'capnp-rpc'.
_rpc_stack = rpc_enabled or get_option('with_rpc_client_only')
# Inputs of the effective schema (the vendored copy plus the rgpot-local
# series in rgpot/rpc/local/series); every target that applies it depends
# on them.
rgpot_schema_inputs = files('rgpot/rpc/local/series')
foreach _line : _fs.read('rgpot/rpc/local/series').split('\n')
    _line = _line.strip()
    if _line != '' and not _line.startswith('#')
        rgpot_schema_inputs += files('rgpot/rpc/local' / _line)
    endif
endforeach
_metatomic_build_rpath = ''
if _rpc_stack
    subdir('rgpot/rpc')
    # Pure dlopen consumers of split nwchemc / cpmdc; need generated schema only.
    subdir('rgpot/NWChemPot')  # 'nwchempot' frontend + no-op ABI stub
    subdir('rgpot/CPMDPot')  # 'cpmdpot' frontend + no-op ABI stub
    if has_cp2k
        # 'cp2kpot' frontend + libcp2k shim
        subdir('rgpot/CP2KPot')
    endif
endif

# ------------------------ Main library
if not get_option('with_rpc_client_only')
    # RGPOT_WITH_HIGHWAY stays on this archive. The installed dependency
    # does not pass it on, so a consumer's own Highway headers are untouched.
    # Product-sums in the simd sources are explicit. Contraction stays off
    # in this archive so a multiply that is not written as mul_add is not fused.
    _deps += dependency('threads')
    _nlist_openmp = dependency('openmp', language: 'cpp', required: false)
    if _nlist_openmp.found()
        _deps += _nlist_openmp
    endif
    _rgpot_srcs += files(
        'rgpot/plugin/PluginBridge.cc',
        'rgpot/plugin/PluginRegistry.cc',
        'rgpot/plugin/builtins/lj_plugin.cpp',
    )
    _core_args = _args + _fp_contract_off
    if have_highway
        _core_args += ['-DRGPOT_WITH_HIGHWAY']
    endif
    _core_deps = _deps
    if have_highway
        _core_deps += highway_dep
    endif
    rgpot_core = static_library(
        'rgpot_core',
        sources: _rgpot_srcs,
        dependencies: _core_deps,
        cpp_args: _core_args,
        include_directories: _incdirs,
        link_whole: [rgpot_minimage, rgpot_exact_fma],
        pic: true,
        install: false,
    )

    # Snapshot before the core link-dep: the umbrella library link_wholes
    # rgpot_core explicitly and must not also pull it through link_with.
    _deps_sans_core = _deps
    _deps += declare_dependency(link_with: rgpot_core)

    # Implemented rgpotentials
    subdir('rgpot/LennardJones')  # 'lennard_jones': LJPot + LJClusterPot
    rgpotentials = [lennard_jones, rgpot_halfshell]

    subdir('rgpot/Morse')  # 'morse'
    rgpotentials += [morse]

    subdir('rgpot/Models')
    rgpotentials += [model_pots]

    if has_cbe
        subdir('rgpot/CBEPot')
        rgpotentials += [cbe_pot]
    endif

    subdir('rgpot/Gupta')  # 'gupta'
    rgpotentials += [gupta]

    subdir('rgpot/StillingerWeber')  # 'sw_si': 1985 Stillinger-Weber silicon
    rgpotentials += [sw_si]

    subdir('rgpot/Tersoff')  # 'tersoff_sic': 1989 Tersoff Si-C
    rgpotentials += [tersoff_sic]

    subdir('rgpot/EMT')  # 'asap_emt'
    rgpotentials += [asap_emt]

    subdir('rgpot/EAM')  # 'eam_cell'
    rgpotentials += [eam_cell]

    if has_eam_table
        subdir('rgpot/EAMTable')  # 'eam_table'
        rgpotentials += [eam_table]
    endif

    if has_water
        subdir('rgpot/Water')  # 'water_pots'
        rgpotentials += [water_pots]
    endif

    subdir('rgpot/ZBL')  # 'zbl'
    rgpotentials += [zbl]

    if has_ipi
        subdir('rgpot/IPIPot')  # 'ipi'
        rgpotentials += [ipi]
    endif

    # Sutton-Chen. Own directory so a Stillinger-Weber or Tersoff
    # series can land beside it.
    subdir('rgpot/SuttonChen')  # 'sutton_chen'
    rgpotentials += [sutton_chen]

    # Unshifted cluster pairs. Separate from LJCluster, which is shifted.
    subdir('rgpot/ReducedPair')  # 'reduced_pair'
    rgpotentials += [reduced_pair]

    # Pure dlopen consumer of split mopacc (no Cap'n Proto, no OpenMOPAC embed).
    subdir('rgpot/MOPACPot')  # 'mopacpot'
    rgpotentials += [mopacpot]

    subdir('rgpot/ExternalPot')  # lammps / gromacs / openmm constructors
    rgpotentials += [external_pot]

    subdir('rgpot/CuH2')  # 'cuh2': slab-geometry helpers
    rgpotentials += [cuh2]

    if has_fortran
        subdir('rgpot/fortran')  # Fortran 2018 pot kernels + C entry points
        rgpotentials += [fortran_pots, fortran_kernels]
        # The kernel archive's symbols (module symbols and the bind(c)
        # entries) stay local to librgpot: only the C++ faces above are
        # part of the library's interface.
        # vesin's Fortran interface is an implementation detail of the
        # kernels and is hidden with them.
        _fortran_hide_args = cppc.get_supported_link_arguments(
            [
                '-Wl,--exclude-libs,librgpot_fortran_kernels.a',
                '-Wl,--exclude-libs,librgpot_sw_lib.a',
                '-Wl,--exclude-libs,librgpot_sw_capi_lib.a',
                '-Wl,--exclude-libs,libvesin_fortran.a',
            ],
        )
    endif

    # XTBDlopen frontend always (no libxtb). Linked pot + engine when has_xtb.
    subdir('rgpot/XTBPot')
    _linkto += [xtb_dlopen]
    if has_xtb
        rgpotentials += [xtbpot]
    endif

    if has_xckernel
        subdir('rgpot/XcKernel')  # in-process XcKernel, not a Potential
        rgpotentials += [xckernel]
    endif

    if has_tblite
        subdir('rgpot/TBLitePot')  # 'tblitepot'
        rgpotentials += [tblitepot]
    endif

    if has_dftd3
        subdir('rgpot/D3Pot')  # 'd3pot'
        rgpotentials += [d3pot]
    endif

    if has_dftd4
        subdir('rgpot/D4Pot')  # 'd4pot'
        rgpotentials += [d4pot]
    endif

    if has_long_range
        subdir('rgpot/LongRange')  # 'lr_phys'
        rgpotentials += [lr_phys]
    endif

    if has_expr
        subdir('rgpot/ExprPot')  # 'exprpot'
        rgpotentials += [exprpot]
    endif

    # MetatomicDlopen frontend always (no torch). Engine/fat pot when has_metatomic.
    subdir('rgpot/MetatomicPot')
    if has_metatomic
        _metatomic_build_rpath = _engine_build_rpath
    endif
    _linkto += [metatomic_dlopen]
    # Fat MetatomicPot NEEDs torch. Folding it into the umbrella makes
    # `import rgpot` (LJ / dlopen) require libtorch — the pip contract is
    # numpy-only until `rgpot[metatomic]`. The wheel still *builds* the
    # engine plugin (has_metatomic); it is dlopen'd, not linked.
    if has_metatomic and not get_option('with_python')
        rgpotentials += [metatomicpot]
        subdir('rgpot/UmaPot')
        rgpotentials += [uma]
    endif

    if _rpc_stack
        rgpotentials += [nwchempot]
        rgpotentials += [cpmdpot]
        if has_cp2k
            rgpotentials += [cp2kpot]
        endif
        if has_skala
            subdir('rgpot/SkalaPot')
            rgpotentials += [skala]
        endif
    endif

    _linkto += rgpotentials

    # Create the umbrella library: every pot and the core ship inside one
    # versioned SONAME. link_whole keeps all symbols even where nothing in
    # librgpot itself references them (consumers do).
    # vesin objects and the Fortran runtime enter here, once, rather than
    # through the convenience archives (see the vesin_hdr_dep note).
    rgpot_deps = _deps_sans_core + [vesin_dep, fastnlist_dep] + fortran_rt_deps
    if eam_openmp.found()
        rgpot_deps += eam_openmp
    endif
    _rpc_whole = []
    if _rpc_stack
        # link_whole, not link_with: the RPC entry points are called by
        # consumers rather than by librgpot itself.
        rgpot_deps += [ptlrpc_hdr_dep]
        _rpc_whole += [ptlrpc]
    endif
    _rgpot_link_args = []
    if has_fortran
        _rgpot_link_args += _fortran_hide_args
    endif
    if has_fast_nlist and not _fast_as_object and host_system != 'windows'
        if host_system == 'darwin'
            _rgpot_link_args += [
                '-Wl,-force_load,' + _fastnlist_tgt.full_path(),
            ]
        else
            _rgpot_link_args += [
                '-Wl,--whole-archive',
                _fastnlist_tgt.full_path(),
                '-Wl,--no-whole-archive',
            ]
        endif
    endif
    _rgpot_srcs = []
    _rgpot_cpp_args = _args
    if host_system == 'windows'
        # MSVC omits the import library when a DLL exports no symbols.
        _rgpot_srcs += files('rgpot/win_export.cc')
        _rgpot_cpp_args += [
            '-DMESON_PROJECT_VERSION="@0@"'.format(meson.project_version()),
        ]
    endif
    rgpot = library(
        'rgpot',
        _rgpot_srcs,
        dependencies: rgpot_deps,
        cpp_args: _rgpot_cpp_args,
        link_args: _rgpot_link_args,
        build_rpath: _metatomic_build_rpath,
        include_directories: _incdirs,
        link_whole: [rgpot_core] + _linkto + _rpc_whole,
        version: meson.project_version(),
        soversion: rgpot_cxx_soversion,
        install: not meson.is_subproject(),
    )

    # Generic rgpot_engine_* plugins (engine_c_abi.h) for the backends that
    # have no tensor runtime: one shared table over PotentialBase
    # (rgpot/engine/EngineTable.cc) and a small backend file each. The UMA,
    # Metatomic and XTB engines add the same table to their own libraries.
    if _rpc_stack and host_system != 'windows'
        _engine_common = files('rgpot/engine/EngineTable.cc')
        _engine_args = _args + [
            '-DRGPOT_ENGINE_BUILD',
            '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(
                meson.project_version(),
            ),
        ]
        # Fortran water-H calls rgpot_fast_wrap when the fast list is on.
        # The umbrella links that archive; this plugin has to as well.
        _engine_deps = _deps_sans_core + [
            vesin_dep,
            fastnlist_dep,
            capnp_rpc_dep,
            ptlrpc_dep,
        ] + fortran_rt_deps
        _classical_whole = [
            lennard_jones,
            morse,
            zbl,
            model_pots,
            rgpot_halfshell,
        ]
        if has_cbe
            _classical_whole += [cbe_pot]
        endif
        if has_water
            _classical_whole += [water_pots]
        endif
        if has_fortran
            _classical_whole += [cuh2, fortran_pots, fortran_kernels]
        endif
        classical_engine = shared_library(
            'classical_engine',
            [_engine_common, files('rgpot/engine/ClassicalEngine.cc')],
            dependencies: _engine_deps,
            link_with: [rgpot_core, rgpot_halfshell],
            link_whole: _classical_whole,
            cpp_args: _engine_args,
            include_directories: _incdirs,
            gnu_symbol_visibility: 'default',
            install: not meson.is_subproject(),
        )
        mopac_engine = shared_library(
            'mopac_engine',
            [_engine_common, files('rgpot/engine/MopacEngine.cc')],
            dependencies: _engine_deps + [mopac_dl_dep],
            link_with: [rgpot_core],
            link_whole: [mopacpot],
            cpp_args: _engine_args,
            include_directories: _incdirs,
            gnu_symbol_visibility: 'default',
            install: not meson.is_subproject(),
        )
        nwchem_engine = shared_library(
            'nwchem_engine',
            [_engine_common, files('rgpot/engine/NWChemEngine.cc')],
            dependencies: _engine_deps + [nwchem_dl_dep],
            link_with: [rgpot_core],
            link_whole: [nwchempot],
            cpp_args: _engine_args,
            include_directories: _incdirs,
            gnu_symbol_visibility: 'default',
            install: not meson.is_subproject(),
        )
        if has_skala
            skala_engine = shared_library(
                'skala_engine',
                [_engine_common, files('rgpot/engine/SkalaEngine.cc')],
                dependencies: _engine_deps + [nwchem_dl_dep],
                link_with: [rgpot_core],
                link_whole: [nwchempot, skala],
                cpp_args: _engine_args,
                include_directories: _incdirs,
                gnu_symbol_visibility: 'default',
                install: not meson.is_subproject(),
            )
        endif
    endif

    # Subproject consumers (eOn's wrap fallback) take the umbrella through
    # this variable; pkg-config consumers get the same surface from
    # rgpot.pc.
    # The static umbrella already contains the relocatable fast-list
    # object. Repeating that object through rgpot_dep defines it twice.
    # Shared builds keep the archive on the consumer link line.
    _rgpot_consumer_deps = _deps_sans_core + [
        vesin_dep,
        fastnlist_native_dep,
    ] + fortran_rt_deps
    if _rpc_stack
        _rgpot_consumer_deps += [ptlrpc_hdr_dep]
    endif
    rgpot_dep = declare_dependency(
        link_with: rgpot,
        dependencies: _rgpot_consumer_deps,
        include_directories: _incdirs,
        compile_args: _args,
    )

    # The Fortran kernels must stay invisible outside librgpot: an
    # exported legacy name is how two potentials used to bind to the same
    # symbol.
    if has_fortran and host_system == 'linux' and get_option('with_tests')
        test(
            'FortranSymbolAudit',
            find_program('../scripts/audit_fortran_symbols.sh'),
            args: [rgpot.full_path()],
            depends: [rgpot],
        )
    endif
    # librgpot must not pull libmpi into a host: the MPI side is dlopen'd.
    if get_option('with_tests') and host_system == 'linux'
        test(
            'CoreLinksNoMpi',
            find_program('../scripts/check_no_mpi_needed.sh'),
            args: [rgpot.full_path()],
            depends: [rgpot],
        )
    endif
    # nm/readelf ELF dump; same host gate as FortranSymbolAudit.
    if get_option('with_tests') and host_system != 'windows'
        test(
            'ExprLeptonSymbols',
            find_program('../scripts/check_lepton_symbols.sh'),
            args: [rgpot.full_path(), has_expr ? 'present' : 'absent'],
            depends: [rgpot],
            kwargs: _rt_kw['ExprLeptonSymbols'],
        )
    endif
endif

# Client-only (eOn with_rgpot): thin dlopen frontends without fat link deps.
# XTB always; metatomic only when torch/metatomic are available to build the
# engine, whose optional libmetatomic_engine.so holds torch.
if get_option('with_rpc_client_only')
    subdir('rgpot/XTBPot')
    _linkto += [xtb_dlopen]
    if has_metatomic
        subdir('rgpot/MetatomicPot')
        _metatomic_build_rpath = _engine_build_rpath
    endif
endif

# ------------------------ Server
if rpc_enabled and not get_option('with_rpc_client_only')
    server = executable(
        'potserv',
        ['rgpot/rpc/server.cpp'],
        link_with: [ptlrpc, _linkto],
        dependencies: _deps + [
            capnp_rpc_dep,
            ptlrpc_hdr_dep,
            vesin_dep,
            fastnlist_dep,
        ] + _mta_engine_deps + fortran_rt_deps,
        cpp_args: _args,
        build_rpath: _metatomic_build_rpath,
        include_directories: _incdirs,
        link_language: 'cpp',
        install: not meson.is_subproject(),
    )
    if _mpi_dep.found()
        mpi_server = executable(
            'potserv_mpi',
            ['rgpot/rpc/server.cpp'],
            link_with: [ptlrpc, _linkto],
            dependencies: _deps + [
                capnp_rpc_dep,
                ptlrpc_hdr_dep,
                vesin_dep,
                fastnlist_dep,
                _mpi_dep,
            ] + _mta_engine_deps + fortran_rt_deps,
            cpp_args: _args + ['-DRGPOT_POTSERV_MPI'],
            build_rpath: _metatomic_build_rpath,
            include_directories: _incdirs,
            link_language: 'cpp',
            install: not meson.is_subproject(),
        )
        nwchem_mpi_force_host = executable(
            'nwchem_mpi_force_host',
            ['tools/nwchem_mpi_force_host.cc'],
            link_with: [ptlrpc, _linkto],
            dependencies: _deps + [ptlrpc_hdr_dep, vesin_dep, _mpi_dep] + fortran_rt_deps,
            cpp_args: _args,
            build_rpath: _metatomic_build_rpath,
            include_directories: _incdirs,
            install: not meson.is_subproject(),
        )
    endif
endif

# ------------------------ Examples
# Each example only builds when its required dependencies are present.

if get_option('with_examples') and not get_option('with_rpc_client_only')
    example_array = []
    if has_fortran
        example_array += [
            ['CallCuH2Pot', 'cuh2_call', 'call_cuh2.cc', ''],
        ]
    endif
    if has_eigen
        example_array += [
            ['EigenCallLJPot', 'ljpot_call_eig', 'eigen_call_ljpot.cc', ''],
        ]
        if has_fortran
            example_array += [
                ['EigenCallCuH2Pot', 'cuh2_call_eig', 'eigen_call_cuh2.cc', ''],
            ]
        endif
    endif
    if has_xtensor and has_fortran
        example_array += [
            ['XtCallCuH2Pot', 'cuh2_call_xt', 'xt_call_cuh2.cc', ''],
        ]
    endif
    if has_expr
        example_array += [
            ['CallExprPot', 'exprpot_call', 'call_exprpot.cc', ''],
        ]
    endif
    # Not a meson test: scripts/time_lj_rgpot_vs_pyeonclient.py drives this.
    executable(
        'time_lj_force',
        sources: ['examples/time_lj_force.cc'],
        dependencies: _deps + [vesin_dep, fastnlist_dep] + fortran_rt_deps,
        link_with: _linkto,
        cpp_args: _args,
    )
    # Not a meson test: scripts/bench_pair_scaling.sh drives this.
    executable(
        'time_pair_scaling',
        sources: ['examples/time_pair_scaling.cc'],
        dependencies: _deps + [vesin_dep, fastnlist_dep] + fortran_rt_deps,
        link_with: _linkto,
        cpp_args: _args,
    )
    executable(
        'time_sutton_chen',
        sources: ['examples/time_sutton_chen.cc'],
        dependencies: _deps + [vesin_dep] + fortran_rt_deps,
        link_with: _linkto,
        cpp_args: _args,
    )
    executable(
        'time_reduced_lj',
        sources: ['examples/time_reduced_lj.cc'],
        dependencies: _deps + [vesin_dep] + fortran_rt_deps,
        link_with: _linkto,
        cpp_args: _args + _fp_contract_off,
    )
    executable(
        'relax_sutton_chen',
        sources: ['examples/relax_sutton_chen.cc'],
        dependencies: _deps + [vesin_dep] + fortran_rt_deps,
        link_with: _linkto,
        cpp_args: _args,
    )
    if rpc_enabled
        # Not a meson test: needs a running potserv (see the file header).
        executable(
            'time_rpc_call',
            sources: ['examples/time_rpc_call.cc'],
            link_with: [pot_bridge],
            include_directories: _incdirs,
            cpp_args: _args,
        )
    endif
    if _mpi_dep.found()
        # Not a meson test: run under mpirun (see the file header).
        executable(
            'time_calculator_share',
            sources: ['examples/time_calculator_share.cc'],
            dependencies: _deps + [vesin_dep, rgpot_mpi_dep] + fortran_rt_deps,
            link_with: _linkto,
            cpp_args: _args,
        )
    endif
    foreach example : example_array
        test(
            example.get(0),
            executable(
                example.get(1),
                sources: ['examples/' + example.get(2)],
                dependencies: _deps + [vesin_dep, fastnlist_dep] + fortran_rt_deps,
                link_with: _linkto,
                cpp_args: _args,
                build_rpath: _metatomic_build_rpath,
            ),
            kwargs: _rt_kw[example.get(0)],
        )
    endforeach
endif

# ------------------------ Tests

if get_option('with_tests')
    test(
        'AotiPublicationTest',
        executable(
            'aoti_publication_test',
            'tests/AotiPublicationTest.cc',
            include_directories: ['.'],
            dependencies: dependency('threads'),
            override_options: ['cpp_std=c++20'],
        ),
    )
    test(
        'AotiPublicationStressTest',
        executable(
            'aoti_publication_stress_test',
            'tests/AotiPublicationStressTest.cc',
            include_directories: ['.'],
            dependencies: dependency('threads'),
            override_options: ['cpp_std=c++20'],
        ),
        kwargs: _rt_kw['AotiPublicationStressTest'],
    )
    test(
        'MolecularFrameTest',
        executable(
            'molecular_frame_test',
            'tests/MolecularFrameTest.cc',
            include_directories: ['.'],
            override_options: ['cpp_std=c++20'],
        ),
    )
    test_deps = _deps + [vesin_dep, fastnlist_dep] + fortran_rt_deps
    if has_metatomic
        test_deps += _mta_engine_deps
    endif
    catch2_dep = dependency(
        'Catch2',
        method: 'cmake',
        modules: ['Catch2::Catch2WithMain'],
    )
    test_deps += catch2_dep
    test(
        'AotiUnpackTest',
        executable(
            'aoti_unpack_test',
            'tests/AotiUnpackTest.cc',
            include_directories: ['.'],
            dependencies: [catch2_dep, dependency('threads')],
            override_options: ['cpp_std=c++20'],
        ),
        workdir: meson.current_build_dir(),
    )
    test_args = _args
    test_args += ['-DRGPOTTEST']
    if _mpi_dep.found()
        # Tests expect librgpot_mpi next to them (it builds in this
        # directory), whether or not they link MPI themselves.
        test_args += ['-DRGPOT_HAS_MPI']
    endif
    if not get_option('with_rpc_client_only')
        subdir('tests/plugin')
    endif
    # Client-only builds skip rgpot_core; UnitsTest still needs units.cc.
    if get_option('with_rpc_client_only')
        units_for_tests = static_library(
            'units_for_tests',
            sources: ['rgpot/units.cc'],
            include_directories: _incdirs,
            cpp_args: _args,
            install: false,
        )
        test_deps += declare_dependency(link_with: units_for_tests)
    endif
    if has_expr and not get_option('with_rpc_client_only')
        test(
            'ExprPotTest',
            executable(
                'expr_pot_test',
                sources: ['tests/ExprPotTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            workdir: meson.source_root(),
            suite: 'expr',
        )
        test(
            'ExprGoldens',
            executable(
                'expr_golden_test',
                sources: ['tests/ExprGoldenTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            workdir: meson.source_root(),
            suite: 'expr',
        )
        executable(
            'dump_expr_goldens',
            sources: ['tests/dump_expr_goldens.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
        if get_option('with_rust_core')
            test(
                'ExprEindirTest',
                executable(
                    'expr_eindir_test',
                    sources: ['tests/ExprEindirTest.cc'],
                    dependencies: test_deps,
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                ),
                workdir: meson.source_root(),
                suite: 'expr',
            )
        endif
    endif
    if has_metatomic
        executable(
            'uma_ase_parity',
            sources: ['tests/uma_ase_parity.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
        )
        if _rpc_stack
            uma_engine_env = environment()
            uma_engine_env.set('RGPOT_UMA_ENGINE', uma_engine.full_path())
            uma_engine_test = executable(
                'uma_engine_test',
                sources: ['tests/UmaEngineTest.cc'],
                dependencies: test_deps + [capnp_rpc_dep, ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            )
        endif
        if _rpc_stack
            executable(
                'uma_engine_bench',
                sources: ['tests/uma_engine_bench.cc'],
                dependencies: test_deps + [capnp_rpc_dep, ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            )
        endif
        _uma_py = find_program('python3', required: true)
        _uma_export = meson.project_source_root() / 'scripts' / 'export_tiny_uma_contract.py'
        _uma_fix_pkgs = []
        _uma_fix_env = environment()
        _uma_cache_dir = meson.current_build_dir() / 'uma-cache'
        _uma_fix_env.set('RGPOT_CACHE_DIR', _uma_cache_dir)
        foreach _spec : [
            ['float64', 'both', 'UMA_FLOAT64_PT2', 'line-float64.pt2'],
            ['float32', 'both', 'UMA_FLOAT32_PT2', 'line-float32.pt2'],
            ['float64', 'model', 'UMA_FLOAT64_MODEL_ONLY_PT2', 'line-model.pt2'],
            ['float32', 'none', 'UMA_FLOAT32_NOMETA_PT2', 'line-nometa.pt2'],
            [
                'float64',
                'mismatch',
                'UMA_DTYPE_MISMATCH_PT2',
                'line-mismatch.pt2',
            ],
        ]
            _pkg = custom_target(
                'uma-' + _spec[2],
                output: _spec[3],
                command: [
                    _uma_py,
                    _uma_export,
                    '--kind',
                    'precision',
                    '--dtype',
                    _spec[0],
                    '--meta',
                    _spec[1],
                    '--out',
                    '@OUTPUT@',
                ],
                build_by_default: false,
            )
            _uma_fix_pkgs += [_pkg]
            _uma_fix_env.set(_spec[2], _pkg.full_path())
            if _spec[2] == 'UMA_FLOAT64_PT2'
                _uma_float64 = _pkg
            endif
        endforeach
        _uma_batch = custom_target(
            'uma-line-batch',
            output: 'line-float64-batch.pt2',
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'batch',
                '--dtype',
                'float64',
                '--out',
                '@OUTPUT@',
            ],
            build_by_default: false,
        )
        _uma_fix_pkgs += [_uma_batch]
        _uma_fix_env.set('UMA_FLOAT64_BATCH_PT2', _uma_batch.full_path())
        _uma_packed = custom_target(
            'uma-line-packed',
            output: 'line-float64-packed.pt2',
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'packed',
                '--dtype',
                'float64',
                '--out',
                '@OUTPUT@',
            ],
            build_by_default: false,
        )
        _uma_fix_pkgs += [_uma_packed]
        _uma_fix_env.set('UMA_PACKED_PT2', _uma_packed.full_path())
        _uma_gather_static = custom_target(
            'uma-gather-static',
            output: 'gather-static.pt2',
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'gather',
                '--dtype',
                'float64',
                '--out',
                '@OUTPUT@',
            ],
            build_by_default: false,
        )
        _uma_gather_dynamic = custom_target(
            'uma-gather-dynamic',
            output: 'gather-dynamic.pt2',
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'gather',
                '--dtype',
                'float64',
                '--dynamic',
                '--out',
                '@OUTPUT@',
            ],
            build_by_default: false,
        )
        _uma_fix_pkgs += [_uma_gather_static, _uma_gather_dynamic]
        _uma_fix_env.set('UMA_GATHER_STATIC_PT2', _uma_gather_static.full_path())
        _uma_fix_env.set(
            'UMA_GATHER_DYNAMIC_PT2',
            _uma_gather_dynamic.full_path(),
        )
        _uma_no_edges = custom_target(
            'uma-no-edges',
            output: 'line-no-edges.pt2',
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'no-edges',
                '--dtype',
                'float64',
                '--out',
                '@OUTPUT@',
            ],
            build_by_default: false,
        )
        _uma_box = custom_target(
            'uma-molecular-box',
            output: 'line-box.pt2',
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'box',
                '--dtype',
                'float64',
                '--out',
                '@OUTPUT@',
            ],
            build_by_default: false,
        )
        _uma_batch_static = custom_target(
            'uma-batch-static',
            output: 'line-batch-static.pt2',
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'batch',
                '--dtype',
                'float64',
                '--static',
                '--out',
                '@OUTPUT@',
            ],
            build_by_default: false,
        )
        _uma_fix_pkgs += [_uma_no_edges, _uma_box, _uma_batch_static]
        _uma_fix_env.set('UMA_NO_EDGES_PT2', _uma_no_edges.full_path())
        _uma_fix_env.set('UMA_MOLECULAR_BOX_PT2', _uma_box.full_path())
        _uma_fix_env.set('UMA_BATCH_STATIC_PT2', _uma_batch_static.full_path())
        _uma_tetrazine_keyless = custom_target(
            'uma-tetrazine-keyless',
            output: 'tetrazine-keyless.pt2',
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'tetrazine-keyless',
                '--dtype',
                'float64',
                '--out',
                '@OUTPUT@',
            ],
            build_by_default: false,
        )
        _uma_fix_pkgs += [_uma_tetrazine_keyless]
        _uma_fix_env.set(
            'UMA_TETRAZINE_KEYLESS_PT2',
            _uma_tetrazine_keyless.full_path(),
        )
        _uma_batch_molecular = custom_target(
            'uma-batch-molecular',
            output: 'line-batch-molecular.pt2',
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'batch-molecular',
                '--dtype',
                'float64',
                '--out',
                '@OUTPUT@',
            ],
            build_by_default: false,
        )
        _uma_fix_pkgs += [_uma_batch_molecular]
        _uma_fix_env.set(
            'UMA_BATCH_MOLECULAR_PT2',
            _uma_batch_molecular.full_path(),
        )
        test(
            'UmaGraphSizeTest',
            executable(
                'uma_graph_size_test',
                sources: ['tests/UmaGraphSizeTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: _uma_fix_env,
            depends: _uma_fix_pkgs,
            workdir: meson.source_root(),
        )
        test(
            'UmaExportBounds',
            _uma_py,
            args: [
                meson.project_source_root() / 'tests' / 'test_uma_export_bounds.py',
            ],
            workdir: meson.source_root(),
        )
        if _rpc_stack
            uma_engine_env.set('RGPOT_CACHE_DIR', _uma_cache_dir)
            uma_engine_env.set('UMA_FLOAT64_PT2', _uma_float64.full_path())
            test(
                'UmaEngineTest',
                uma_engine_test,
                env: uma_engine_env,
                depends: [uma_engine, _uma_float64],
                workdir: meson.source_root(),
            )
            _uma_harness_env = environment()
            _uma_harness_env.prepend('LD_LIBRARY_PATH', _torch_libdir)
            _uma_harness_env.set('RGPOT_CACHE_DIR', _uma_cache_dir)
            test(
                'UmaEngineHarness',
                _uma_py,
                args: [
                    meson.project_source_root() / 'scripts' / 'uma_engine.py',
                    uma_engine.full_path(),
                    _uma_gather_static.full_path(),
                    meson.project_source_root() / 'CppCore' / 'rgpot' / 'rpc' / 'Potentials.capnp',
                ],
                depends: [_uma_gather_static, uma_engine],
                env: _uma_harness_env,
                workdir: meson.source_root(),
            )
            test(
                'UmaOffReference',
                _uma_py,
                args: [
                    meson.project_source_root() / 'scripts' / 'uma_off_reference.py',
                    '--tap',
                    uma_engine.full_path(),
                    'uma-off-reference-package-absent.pt2',
                    meson.project_source_root() / 'CppCore' / 'rgpot' / 'rpc' / 'Potentials.capnp',
                ],
                protocol: 'tap',
                depends: [uma_engine],
                env: _uma_harness_env,
                workdir: meson.source_root(),
            )
        endif
        # Not parallel: it asserts the system temporary directory gains no entry.
        test(
            'UmaCacheDirTest',
            executable(
                'uma_cache_dir_test',
                sources: ['tests/UmaCacheDirTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: _uma_fix_env,
            depends: _uma_fix_pkgs,
            workdir: meson.current_build_dir(),
            is_parallel: false,
        )
        test(
            'UmaFloat64Test',
            executable(
                'uma_float64_test',
                sources: ['tests/UmaFloat64Test.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: _uma_fix_env,
            depends: _uma_fix_pkgs,
            workdir: meson.source_root(),
        )
        _uma_provenance = custom_target(
            'uma-provenance',
            output: ['line-provenance.pt2', 'line-provenance-weights.pt'],
            command: [
                _uma_py,
                _uma_export,
                '--kind',
                'provenance',
                '--dtype',
                'float64',
                '--out',
                '@OUTPUT0@',
                '--checkpoint-out',
                '@OUTPUT1@',
            ],
            build_by_default: false,
        )
        _uma_fix_env.set('UMA_PROVENANCE_PT2', _uma_provenance[0].full_path())
        _uma_fix_env.set(
            'UMA_PROVENANCE_WEIGHTS',
            _uma_provenance[1].full_path(),
        )
        test(
            'UmaProvenanceTest',
            executable(
                'uma_provenance_test',
                sources: ['tests/UmaProvenanceTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: _uma_fix_env,
            depends: _uma_fix_pkgs + [_uma_provenance],
            workdir: meson.source_root(),
        )
    endif
    # NWChem/CPMD frontend tests (need ptlrpc + frontends from _rpc_stack)
    if _rpc_stack and not get_option('with_rpc_client_only')
        nwchemc_fake_engine = shared_library(
            'nwchemc_fake_engine',
            sources: ['tests/nwchemc_fake_engine.c'],
            include_directories: include_directories('rgpot/NWChemPot'),
            install: false,
        )
        nwchemc_lifetime_engine = shared_library(
            'nwchemc_lifetime_engine',
            sources: ['tests/nwchemc_lifetime_engine.cc'],
            include_directories: _incdirs,
            install: false,
        )
        nwchem_lifetime_test = executable(
            'nwchem_lifetime_test',
            sources: ['tests/NWChemPotLifetimeTest.cc'],
            dependencies: test_deps + [ptlrpc_dep],
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
        )
        foreach lifetime_mode : ['ordinary', 'probe']
            test(
                'NWChemPotLifetime-' + lifetime_mode,
                find_program('python3'),
                args: [
                    meson.current_source_dir() / 'tests/test_nwchem_lifetime.py',
                    nwchem_lifetime_test.full_path(),
                    nwchemc_lifetime_engine.full_path(),
                    lifetime_mode,
                ],
                depends: [nwchem_lifetime_test, nwchemc_lifetime_engine],
                kwargs: _rt_kw['NWChemPotLifetime-' + lifetime_mode],
            )
        endforeach
        nwchemc_fake_env = environment()
        nwchemc_fake_env.set(
            'RGPOT_NWCHEM_ENGINE',
            nwchemc_fake_engine.full_path(),
        )
        test(
            'NWChemPotTest',
            executable(
                'nwchem_pot_test',
                sources: ['tests/NWChemPotTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: nwchemc_fake_env,
            depends: nwchemc_fake_engine,
            kwargs: _rt_kw['NWChemPotTest'],
        )
        if has_skala
            test(
                'SkalaPotTest',
                executable(
                    'skala_pot_test',
                    sources: ['tests/SkalaPotTest.cc'],
                    dependencies: test_deps + [ptlrpc_dep],
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                ),
                env: nwchemc_fake_env,
                depends: nwchemc_fake_engine,
                workdir: meson.source_root(),
                suite: 'skala',
            )
        endif
        test(
            'NWChemPotMessageAbiTest',
            executable(
                'nwchem_pot_message_abi_test',
                sources: ['tests/NWChemPotMessageAbiTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: nwchemc_fake_env,
            depends: nwchemc_fake_engine,
            kwargs: _rt_kw['NWChemPotMessageAbiTest'],
        )
        test(
            'test_nwchem_abi',
            executable(
                'test_nwchem_abi',
                sources: ['tests/test_nwchem_abi_main.c'],
                dependencies: [nwchem_abi_stub_dep],
                include_directories: include_directories('rgpot/NWChemPot'),
            ),
            kwargs: _rt_kw['test_nwchem_abi'],
        )
        cpmdc_fake_engine = shared_library(
            'cpmdc_fake_engine',
            sources: ['tests/cpmdc_fake_engine.cc'],
            cpp_args: _build_identity_args,
            dependencies: [ptlrpc_dep],
            include_directories: _incdirs + ['.'] + include_directories(
                'rgpot/CPMDPot',
            ),
            install: false,
        )
        cpmdc_fake_env = environment()
        cpmdc_fake_env.set(
            'RGPOT_CPMD_ENGINE',
            cpmdc_fake_engine.full_path(),
        )
        cpmdc_gradient_engine = shared_library(
            'cpmdc_gradient_engine',
            sources: ['tests/cpmdc_gradient_engine.cc'],
            include_directories: _incdirs + ['.'] + include_directories(
                'rgpot/CPMDPot',
            ),
            install: false,
        )
        cpmdc_gradient_env = environment()
        cpmdc_gradient_env.set(
            'RGPOT_CPMD_ENGINE',
            cpmdc_gradient_engine.full_path(),
        )
        test(
            'CPMDPotGradientStressTest',
            executable(
                'cpmd_pot_gradient_stress_test',
                sources: ['tests/CPMDPotGradientStressTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: cpmdc_gradient_env,
            depends: cpmdc_gradient_engine,
            kwargs: _rt_kw['CPMDPotGradientStressTest'],
        )
        test(
            'CPMDPotMessageAbiTest',
            executable(
                'cpmd_pot_message_abi_test',
                sources: ['tests/CPMDPotMessageAbiTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: cpmdc_fake_env,
            depends: cpmdc_fake_engine,
            kwargs: _rt_kw['CPMDPotMessageAbiTest'],
        )
        if has_cp2k
            cp2kc_fake_engine = shared_library(
                'cp2kc_fake_engine',
                sources: ['tests/cp2kc_fake_engine.cc'],
                include_directories: _incdirs + ['.'],
                install: false,
            )
            cp2kc_fake_env = environment()
            cp2kc_fake_env.set(
                'RGPOT_CP2K_ENGINE',
                cp2kc_fake_engine.full_path(),
            )
            cp2kc_fake_env.set('CP2KC_LIBRARY', cp2kc_fake_engine.full_path())
            test(
                'CP2KPotFakeEngineTest',
                executable(
                    'cp2k_pot_fake_engine_test',
                    sources: ['tests/CP2KPotFakeEngineTest.cc'],
                    dependencies: test_deps + [ptlrpc_dep],
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                    build_rpath: _metatomic_build_rpath,
                ),
                env: cp2kc_fake_env,
                depends: cp2kc_fake_engine,
                kwargs: _rt_kw['CP2KPotFakeEngineTest'],
            )
            test(
                'CP2KDeckTest',
                executable(
                    'cp2k_deck_test',
                    sources: ['tests/CP2KDeckTest.cc'],
                    dependencies: test_deps + [ptlrpc_dep],
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                    build_rpath: _metatomic_build_rpath,
                ),
                kwargs: _rt_kw['CP2KDeckTest'],
            )
        endif
        mpiexec = find_program('mpirun', 'mpiexec', required: false)
        if has_cp2k
            cp2k_smoke_env = environment()
            cp2k_smoke_env.set('RGPOT_CP2K_ENGINE', cp2kc_shim.full_path())
            cp2k_smoke_env.set('OMP_NUM_THREADS', '2')
            _cp2k_prefix = cp2k_lib_dep.get_variable(
                pkgconfig: 'prefix',
                default_value: '',
            )
            if _cp2k_prefix != ''
                cp2k_smoke_env.set(
                    'CP2K_DATA_DIR',
                    _cp2k_prefix / 'share' / 'cp2k' / 'data',
                )
            endif
            cp2k_smoke = executable(
                'cp2k_pot_smoke_test',
                sources: ['tests/CP2KPotSmokeTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            )
            test(
                'CP2KPotSmokeTest',
                cp2k_smoke,
                env: cp2k_smoke_env,
                depends: cp2kc_shim,
                kwargs: _rt_kw['CP2KPotSmokeTest'],
            )
            if mpiexec.found()
                _cp2k_mpi_args = ['-n', '2']
                _mpiexec_version = run_command(
                    mpiexec,
                    '--version',
                    check: false,
                ).stdout()
                if _mpiexec_version.contains('Open MPI') or _mpiexec_version.contains(
                    'OpenRTE',
                )
                    _cp2k_mpi_args = ['--oversubscribe'] + _cp2k_mpi_args
                endif
                test(
                    'CP2KPotSmokeMpi',
                    mpiexec,
                    # Every rank runs the test cases in declaration order:
                    # their CP2K calls are collectives.
                    args: _cp2k_mpi_args + [
                        cp2k_smoke.full_path(),
                        '--order',
                        'decl',
                    ],
                    env: cp2k_smoke_env,
                    depends: [cp2kc_shim, cp2k_smoke],
                    kwargs: _rt_kw['CP2KPotSmokeMpi'],
                )
            endif
        endif
        if _mpi_dep.found() and mpiexec.found()
            mpi_host_contract = executable(
                'mpi_host_contract',
                'tests/MpiHostContractTest.cc',
                dependencies: [_mpi_dep],
                include_directories: _incdirs,
            )
            collective_engine = shared_library(
                'collective_engine',
                'tests/nwchemc_collective_engine.cc',
                dependencies: [ptlrpc_dep, _mpi_dep],
                include_directories: _incdirs,
                install: false,
            )
            mpi_rpc_client = executable(
                'mpi_rpc_contract_client',
                'tests/MpiRpcContractClient.cc',
                dependencies: _deps + [ptlrpc_dep],
                include_directories: _incdirs,
                link_with: _linkto,
                cpp_args: test_args,
                build_rpath: _metatomic_build_rpath,
            )
            mpi_host_script = meson.current_source_dir() / 'tests/test_mpi_hosts.py'
            foreach mode : ['owned', 'borrowed']
                test(
                    'MpiHost-' + mode,
                    find_program('python3'),
                    args: [
                        mpi_host_script,
                        mode,
                        mpiexec.full_path(),
                        mpi_host_contract.full_path(),
                    ],
                    depends: mpi_host_contract,
                    kwargs: _rt_kw['MpiHost-' + mode],
                )
            endforeach
            test(
                'NWChemMpiForceHost',
                find_program('python3'),
                args: [
                    mpi_host_script,
                    'force',
                    mpiexec.full_path(),
                    nwchem_mpi_force_host.full_path(),
                    collective_engine.full_path(),
                ],
                depends: [nwchem_mpi_force_host, collective_engine],
                kwargs: _rt_kw['NWChemMpiForceHost'],
            )
            foreach backend : ['NWChem', 'CPMD'] + (has_cp2k ? ['CP2K'] : [])
                foreach layout : ['serial', 'mpi']
                    rpc_server = layout == 'mpi' ? mpi_server : server
                    test(
                        'CollectiveRPC-' + backend + '-' + layout,
                        find_program('python3'),
                        args: [
                            mpi_host_script,
                            'rpc',
                            mpiexec.full_path(),
                            rpc_server.full_path(),
                            mpi_rpc_client.full_path(),
                            collective_engine.full_path(),
                            backend,
                            layout,
                        ],
                        depends: [rpc_server, mpi_rpc_client, collective_engine],
                        kwargs: _rt_kw['CollectiveRPC-' + backend + '-' + layout],
                    )
                endforeach
            endforeach
            rank_exception = executable(
                'rank_exception_abort',
                sources: ['tests/RankExceptionAbort.cc'],
                dependencies: test_deps + [ptlrpc_dep, rgpot_mpi_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            )
            rank_exception_sh = find_program(
                meson.current_source_dir() / 'tests/rank_exception_abort.sh',
            )
            test(
                'RankExceptionAbort',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'abort'],
                env: cpmdc_fake_env,
                depends: [cpmdc_fake_engine, rank_exception],
                kwargs: _rt_kw['RankExceptionAbort'],
            )
            test(
                'CalculatorUnevenBatch',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'uneven'],
                env: cpmdc_fake_env,
                depends: [cpmdc_fake_engine, rank_exception],
                kwargs: _rt_kw['CalculatorUnevenBatch'],
            )
            test(
                'RankExceptionFinalizeOwner',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'finalize-owner'],
                env: cpmdc_fake_env,
                depends: [rank_exception],
                kwargs: _rt_kw['RankExceptionFinalizeOwner'],
            )
            test(
                'RankExceptionFinalizeGuest',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'finalize-guest'],
                env: cpmdc_fake_env,
                depends: [rank_exception],
                kwargs: _rt_kw['RankExceptionFinalizeGuest'],
            )
            test(
                'ShareBroadcastMismatch',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'share-bad'],
                env: cpmdc_fake_env,
                depends: [rank_exception],
                kwargs: _rt_kw['ShareBroadcastMismatch'],
            )
            test(
                'ShareBroadcastMatch',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'share-ok'],
                env: cpmdc_fake_env,
                depends: [rank_exception],
                kwargs: _rt_kw['ShareBroadcastMatch'],
            )
            refused_split = executable(
                'refused_split',
                sources: ['tests/RefusedSplit.cc'],
                dependencies: test_deps + [ptlrpc_dep, rgpot_mpi_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            )
            refused_split_sh = find_program(
                meson.current_source_dir() / 'tests/refused_split.sh',
            )
            test(
                'RefusedSplit',
                refused_split_sh,
                args: [refused_split.full_path()],
                depends: [refused_split],
                kwargs: _rt_kw['RefusedSplit'],
            )
            cpmdc_adopt_engine = shared_library(
                'cpmdc_adopt_engine',
                sources: ['tests/cpmdc_adopt_engine.cc'],
                dependencies: [_mpi_dep],
                include_directories: _incdirs + ['.'] + include_directories(
                    'rgpot/CPMDPot',
                ),
                install: false,
            )
            mpi_split_profile = shared_library(
                'mpi_split_profile',
                sources: ['tests/mpi_split_profile.c'],
                dependencies: [dependency('mpi', language: 'c')],
                install: false,
            )
            adopt_comm = executable(
                'adopt_comm',
                sources: ['tests/AdoptComm.cc'],
                dependencies: test_deps + [ptlrpc_dep, rgpot_mpi_dep, _dl_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            )
            adopt_comm_sh = find_program(
                meson.current_source_dir() / 'tests/adopt_comm.sh',
            )
            adopt_cases = [
                ['AdoptCommMissing', 'missing', cpmdc_gradient_engine],
                ['AdoptCommSplit', 'split', cpmdc_adopt_engine],
                ['AdoptCommRefusedHook', 'refused-hook', cpmdc_adopt_engine],
                ['AdoptCommProbe', 'probe', cpmdc_adopt_engine],
                ['AdoptCommInstanceFirst', 'instance-first', cpmdc_adopt_engine],
                ['AdoptCommBoundStatic', 'bound-static', cpmdc_adopt_engine],
                ['AdoptCommBoundInstance', 'bound-instance', cpmdc_adopt_engine],
                ['AdoptCommRefusedForce', 'refused-force', cpmdc_adopt_engine],
            ]
            if has_cp2k
                cp2kc_adopt_engine = shared_library(
                    'cp2kc_adopt_engine',
                    sources: ['tests/cp2kc_adopt_engine.cc'],
                    dependencies: [_mpi_dep],
                    include_directories: _incdirs + ['.'],
                    install: false,
                )
                adopt_cases += [
                    ['AdoptCommCP2K', 'cp2k-split', cp2kc_adopt_engine],
                ]
            endif
            foreach entry : adopt_cases
                test(
                    entry[0],
                    adopt_comm_sh,
                    args: [
                        entry[1],
                        adopt_comm.full_path(),
                        entry[2].full_path(),
                        mpi_split_profile.full_path(),
                    ],
                    depends: [adopt_comm, entry[2], mpi_split_profile],
                    kwargs: _rt_kw[entry[0]],
                )
            endforeach
            if get_option('with_cache')
                collective_cache = executable(
                    'collective_cache_test',
                    sources: ['tests/CollectiveCacheTest.cc'],
                    dependencies: test_deps + [rgpot_mpi_dep],
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                    build_rpath: _metatomic_build_rpath,
                )
                test(
                    'CollectiveCache',
                    find_program(
                        meson.current_source_dir() / 'tests/collective_cache.sh',
                    ),
                    args: [collective_cache.full_path()],
                    depends: [collective_cache],
                    kwargs: _rt_kw['CollectiveCache'],
                )
            endif
        endif
        test(
            'ProfileLoaderTest',
            executable(
                'profile_loader_test',
                sources: ['tests/ProfileLoaderTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: cpmdc_fake_env,
            depends: cpmdc_fake_engine,
            kwargs: _rt_kw['ProfileLoaderTest'],
        )
        test(
            'test_cpmd_abi',
            executable(
                'test_cpmd_abi',
                sources: ['tests/test_cpmd_abi_main.c'],
                dependencies: [cpmd_abi_stub_dep],
                include_directories: include_directories('rgpot/CPMDPot'),
            ),
            kwargs: _rt_kw['test_cpmd_abi'],
        )
        test(
            'RPCSchemaSync',
            find_program('python3'),
            args: [
                meson.current_source_dir() / 'tests/test_rpc_schema_sync.py',
                meson.source_root(),
                capnp.full_path(),
                meson.project_build_root(),
            ],
            depends: gen_rpc,
            kwargs: _rt_kw['RPCSchemaSync'],
        )
    endif
    if rpc_enabled
        test_deps += dependency('capnp-rpc')
        test_deps += ptlrpc_dep
        if get_option('with_rpc_client_only')
            test_deps += pot_bridge_dep
        endif
    endif
    if not get_option('with_rpc_client_only')
        mopacc_fake_engine = shared_library(
            'mopacc_fake_engine',
            sources: ['tests/mopacc_fake_engine.c'],
            include_directories: include_directories('rgpot/MOPACPot'),
            install: false,
        )
        mopacc_fake_env = environment()
        mopacc_fake_env.set(
            'RGPOT_MOPACC_ENGINE',
            mopacc_fake_engine.full_path(),
        )
        # The case cannot empty the variable itself: the MSVC runtime has no
        # setenv, and _putenv_s removes a variable given an empty value.
        mopacc_empty_env = environment(
            {
                'RGPOT_MOPACC_ENGINE': mopacc_fake_engine.full_path(),
                'MOPACC_LIBRARY': '',
            },
        )
        mopac_pot_test = executable(
            'mopac_pot_test',
            sources: ['tests/MOPACPotTest.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
        )
        test(
            'MOPACPotTest',
            mopac_pot_test,
            env: mopacc_fake_env,
            depends: mopacc_fake_engine,
            kwargs: _rt_kw['MOPACPotTest'],
        )
        test(
            'MOPACPotEmptyVariable',
            mopac_pot_test,
            args: ['[empty-variable]'],
            env: mopacc_empty_env,
            depends: mopacc_fake_engine,
            kwargs: _rt_kw['MOPACPotEmptyVariable'],
        )
        test(
            'test_mopac_abi',
            executable(
                'test_mopac_abi',
                sources: ['tests/test_mopac_abi_main.c'],
                dependencies: [mopac_abi_stub_dep],
                include_directories: include_directories('rgpot/MOPACPot'),
            ),
            kwargs: _rt_kw['test_mopac_abi'],
        )
    endif

    # asv loads this shared library and times one force call per method.
    # Built with the tests because it links the same pot archives.
    # Client-only builds never define rgpot_core or the pot archives.
    if has_fortran and not get_option('with_rpc_client_only')
        asv_fastpath = shared_library(
            'asv_fastpath',
            sources: [files('../benchmarks/asv_fastpath.cc')],
            dependencies: rgpot_deps,
            include_directories: _incdirs,
            link_with: [
                rgpot_core,
                lennard_jones,
                morse,
                fortran_kernels,
                rgpot_halfshell,
                rgpot_minimage,
                rgpot_exact_fma,
            ],
            cpp_args: [
                '-DRGPOT_ASV_DATA="@0@"'.format(
                    meson.project_source_root() / 'benchmarks' / 'data',
                ),
            ],
            install: false,
        )
    endif

    calculator_group_test_exe = disabler()
    _rt_exes = {}
    foreach _rt : _rt_tests
        if _rt[1] != 'unit'
            continue
        endif
        # The fused-scan oracles are plain multiplies and adds, the same
        # form as the cluster kernel and the Fortran 26-image wrap. Apple
        # Clang fuses those products unless contraction is off, and the
        # sums then miss by an ulp. ReducedPair.cc is built unfused, and
        # its test compares exactly against the same expressions.
        case_args = test_args
        if _rt[3]
            case_args += cppc.get_supported_arguments(
                ['-ffp-contract=off', '/fp:precise'],
            )
        endif
        # The calculator-group symbol check opens librgpot.so when meson
        # builds one, and otherwise looks the symbols up in the test
        # binary, which then has to export them.
        if _rt[0] == 'CalculatorGroupTest'
            if get_option('default_library') == 'static'
                _export_dynamic = true
            else
                _export_dynamic = false
                case_args += [
                    '-DRGPOT_TEST_SHARED_LIBRARY',
                    '-DRGPOT_TEST_SHARED_LIBRARY_PATH="@0@"'.format(
                        rgpot.full_path(),
                    ),
                ]
            endif
        else
            _export_dynamic = false
        endif
        if _rt[0] == 'ModelPotsTest'
            # This test links the model C API archive directly. On Windows its
            # declarations must not name DLL import thunks.
            case_args += ['-DRGPOT_MODEL_BUILD']
        endif
        _test_exe = executable(
            _rt[2][0],
            sources: ['tests/' + _rt[2][1]],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: case_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            export_dynamic: _export_dynamic,
        )
        if _rt[0] == 'CalculatorGroupTest'
            test(
                _rt[0],
                _test_exe,
                depends: [rgpot],
                kwargs: _rt_kw[_rt[0]],
            )
        else
            test(_rt[0], _test_exe, kwargs: _rt_kw[_rt[0]])
        endif
        _rt_exes += {_rt[0]: _test_exe}
        if _rt[0] == 'CalculatorGroupTest'
            calculator_group_test_exe = _test_exe
        endif
    endforeach
    foreach _rt : _rt_tests
        if _rt[1] == 'rerun'
            test(_rt[0], _rt_exes[_rt[2][0]], kwargs: _rt_kw[_rt[0]])
        endif
    endforeach
    _nlist_link = [rgpot_halfshell, rgpot_minimage, rgpot_exact_fma]
    foreach _rt : _rt_tests
        if _rt[1] == 'nlist'
            test(
                _rt[0],
                executable(
                    _rt[2][0],
                    sources: ['examples/' + _rt[2][1]],
                    include_directories: _incdirs + ['.'],
                    c_args: _args,
                    cpp_args: _args,
                    link_with: _nlist_link,
                    link_language: 'cpp',
                ),
                kwargs: _rt_kw[_rt[0]],
            )
        endif
    endforeach
    _capnp_dep = dependency('capnp', required: false)
    _capnpc_dep = dependency('capnpc', required: false)
    # socketpair, fork and waitpid. Windows has no POSIX process pair.
    if _capnp_dep.found() and _capnpc_dep.found() and host_machine.system() != 'windows'
        # The neighbour structs are rgpot-local: they exist only in the
        # effective schema, not in the vendored copy.
        if is_variable('rgpot_effective_schema')
            _nlist_schema = rgpot_effective_schema
        else
            _nlist_schema = custom_target(
                'nlist_effective_schema',
                input: 'rgpot/rpc/Potentials.capnp',
                output: 'Potentials.capnp',
                command: [
                    find_program('python3', 'python'),
                    files('rgpot/rpc/schema_ledger.py'),
                    'apply',
                    '--root',
                    meson.project_source_root(),
                    '--schema',
                    '@INPUT@',
                    '--out',
                    '@OUTPUT@',
                ],
                depend_files: rgpot_schema_inputs,
            )
        endif
        test(
            'nlist_stream',
            executable(
                'nlist_stream',
                sources: ['examples/nlist_stream.cc'],
                include_directories: _incdirs + ['.'],
                cpp_args: _args,
                dependencies: [_capnp_dep, _capnpc_dep],
                link_with: _nlist_link,
            ),
            args: [_nlist_schema.full_path()],
            depends: _nlist_schema,
        )
        _capnp_tool = find_program('capnp', required: false)
        if _capnp_tool.found()
            test(
                'nlist_capnp_roundtrip',
                find_program('python3', 'python'),
                args: [
                    meson.current_source_dir() / 'tests' / 'test_nlist_capnp.py',
                    meson.project_source_root(),
                    _capnp_tool.full_path(),
                    _nlist_schema.full_path(),
                ],
                depends: _nlist_schema,
            )
        endif
    endif
    _rustc = find_program('rustc', required: false)
    if _rustc.found()
        _nlist_rust_args = [
            meson.current_source_dir() / 'examples' / 'nlist_rust_test.sh',
            meson.project_source_root(),
            meson.current_build_dir(),
        ]
        if _cxx_rt_flag != ''
            _nlist_rust_args += [_cxx_rt_flag]
        endif
        if _sanitize != 'none'
            _nlist_rust_args += ['-fsanitize=' + _sanitize]
        endif
        _nlist_rust_cc = cc.cmd_array()
        _nlist_rust_cxx = cppc.cmd_array()
        _nlist_rust_env = {
            'CC': _nlist_rust_cc[_nlist_rust_cc.length() - 1],
            'CXX': _nlist_rust_cxx[_nlist_rust_cxx.length() - 1],
            'RUSTC': _rustc.full_path(),
        }
        if 'thread' in _sanitize
            _nlist_rust_env += {'TSAN_OPTIONS': _tsan_options}
        endif
        test(
            'nlist_rust',
            find_program('sh'),
            args: _nlist_rust_args,
            env: _nlist_rust_env,
            depends: _nlist_link,
        )
    endif
    if not get_option('with_rpc_client_only')
        # Same fold, compiled once per instruction set. The driver checks
        # the CPU and prints skip when the binary cannot run here.
        _isa_specs = [
            [
                'scalar',
                [
                    '-ffp-contract=off',
                    '-fno-tree-vectorize',
                    '-mno-avx',
                    '-mno-fma',
                ],
            ],
            ['sse2', ['-ffp-contract=off', '-msse2', '-mno-avx', '-mno-fma']],
            ['avx2', ['-ffp-contract=off', '-mavx2', '-mfma']],
            ['avx512', ['-ffp-contract=off', '-mavx512f']],
            ['neon', ['-ffp-contract=off']],
        ]
        _isa_exes = []
        foreach spec : _isa_specs
            _isa_args = cppc.get_supported_arguments(spec[1])
            if spec[0] == 'neon' and host_machine.cpu_family() == 'x86_64'
                continue
            endif
            if spec[0] != 'neon' and host_machine.cpu_family() != 'x86_64'
                continue
            endif
            if spec[0] != 'neon' and _isa_args.length() < spec[1].length()
                continue
            endif
            _isa_lib = static_library(
                'minimage_' + spec[0],
                files('rgpot/nlist/MinimumImage.cpp'),
                include_directories: _incdirs,
                cpp_args: _args + _isa_args + _no_tsan_ifunc,
                pic: true,
                install: false,
            )
            _isa_exes += executable(
                'minimage_hash_' + spec[0],
                sources: ['tests/minimage_isa_hash_main.cc'],
                include_directories: _incdirs + ['.'],
                cpp_args: [
                    '-ffp-contract=off',
                    '-DRGPOT_MINIMAGE_ISA_' + spec[0].to_upper(),
                ],
                link_whole: [_isa_lib, rgpot_exact_fma],
                install: false,
            )
        endforeach
        executable(
            'minimage_bench',
            sources: ['tests/MinImageBench.cc'],
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            dependencies: test_deps,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
        if _isa_exes.length() > 0
            test(
                'MinImageIsaVariants',
                find_program(
                    meson.project_source_root() / 'ci/gha/check_minimage_isa.sh',
                ),
                args: [
                    'd6f5cbaa04ac49e6f02ee64c10eb83d44d951756556924dcc17966453dcd8b11',
                ] + _isa_exes,
            )
        endif
        test(
            'ExactFmaEq',
            executable(
                'exact_fma_eq',
                sources: ['tests/exact_fma_eq.cc'],
                include_directories: _incdirs,
                cpp_args: _args + _fp_contract_off,
                # exp.cc's pair loops call MinimumImage::fold.
                link_whole: [rgpot_exact_fma, rgpot_minimage],
                install: false,
            ),
        )
        test(
            'ExpFixedEq',
            executable(
                'exp_fixed_eq',
                sources: ['tests/exp_fixed_eq.cc'],
                include_directories: _incdirs,
                cpp_args: _args + _fp_contract_off,
                link_whole: [rgpot_exact_fma, rgpot_minimage],
                install: false,
            ),
        )
        test(
            'FmaPathEq',
            executable(
                'fma_path_eq',
                sources: ['tests/fma_path_eq.cc'],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args + _fp_contract_off,
                dependencies: test_deps,
                link_with: _linkto,
                link_language: 'cpp',
                install: false,
            ),
        )
        layout_fixture = shared_library(
            'rgpot_layout_fixture',
            sources: ['tests/layout_fixture.cc'],
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            dependencies: test_deps,
            install: false,
        )
        test(
            'CxxLayoutTest',
            executable(
                'cxx_layout_test',
                sources: ['tests/CxxLayoutTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            env: {'RGPOT_LAYOUT_FIXTURE': layout_fixture.full_path()},
            depends: [layout_fixture],
            kwargs: _rt_kw['CxxLayoutTest'],
        )
        simd_env = executable(
            'simd_env',
            sources: ['tests/simd_env.cc'],
            dependencies: _deps + [vesin_dep] + fortran_rt_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: _args,
            link_with: _linkto,
            link_language: 'cpp',
        )
        test(
            'simd-env-scalar',
            simd_env,
            args: ['known', 'scalar'],
            kwargs: _rt_kw['simd-env-scalar'],
        )
        test(
            'simd-env-unknown',
            simd_env,
            args: ['unknown'],
            kwargs: _rt_kw['simd-env-unknown'],
        )
        test(
            'simd-env-rvv',
            simd_env,
            args: ['known', 'rvv'],
            kwargs: _rt_kw['simd-env-rvv'],
        )
        test(
            'simd-env-avx2',
            simd_env,
            args: ['known', 'avx2'],
            kwargs: _rt_kw['simd-env-avx2'],
        )
        if host_system != 'windows'
            # The generic engine table over a stand-in backend: no rpc stack,
            # no torch.
            engine_stub = shared_library(
                'engine_stub',
                ['tests/engine_stub.cc', 'rgpot/engine/EngineTable.cc'],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args + [
                    '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(
                        meson.project_version(),
                    ),
                ],
                dependencies: test_deps,
                install: false,
            )
            engine_stub_env = environment()
            engine_stub_env.set('RGPOT_ENGINE_STUB', engine_stub.full_path())
            engine_stub_env.set(
                'RGPOT_ENGINE_STUB_MODEL',
                meson.current_build_dir() / 'engine_stub_model.bin',
            )
            test(
                'EngineAdapterTest',
                executable(
                    'engine_adapter_test',
                    sources: ['tests/EngineAdapterTest.cc'],
                    dependencies: test_deps,
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_language: 'cpp',
                ),
                env: engine_stub_env,
                depends: [engine_stub],
                kwargs: _rt_kw['EngineAdapterTest'],
            )
        endif
        if _rpc_stack and host_system != 'windows'
            engine_table_env = environment()
            engine_table_expect = ['CLASSICAL', 'MOPAC', 'NWCHEM']
            engine_table_depends = [
                classical_engine,
                mopac_engine,
                nwchem_engine,
            ]
            engine_table_env.set(
                'RGPOT_ENGINE_CLASSICAL',
                classical_engine.full_path(),
            )
            engine_table_env.set('RGPOT_ENGINE_MOPAC', mopac_engine.full_path())
            engine_table_env.set(
                'RGPOT_ENGINE_NWCHEM',
                nwchem_engine.full_path(),
            )
            if has_skala
                engine_table_expect += ['SKALA']
                engine_table_depends += [skala_engine]
                engine_table_env.set(
                    'RGPOT_ENGINE_SKALA',
                    skala_engine.full_path(),
                )
            endif
            if has_xtb
                engine_table_expect += ['XTB']
                engine_table_depends += [xtb_engine]
                engine_table_env.set('RGPOT_ENGINE_XTB', xtb_engine.full_path())
            endif
            if has_metatomic
                engine_table_expect += ['METATOMIC']
                engine_table_depends += [metatomic_engine]
                engine_table_env.set(
                    'RGPOT_ENGINE_METATOMIC',
                    metatomic_engine.full_path(),
                )
                # Models with no weights that read the charge, the spin,
                # both or neither; the last reads both without declaring
                # them.
                _mta_export = meson.project_source_root() / 'scripts' / 'export_tiny_metatomic_model.py'
                foreach _spec : [
                    [
                        ['--inputs=charge,spin_multiplicity'],
                        'MTA_CHARGE_SPIN_PT',
                        'mta-charge-spin.pt',
                    ],
                    [['--inputs=charge'], 'MTA_CHARGE_PT', 'mta-charge.pt'],
                    [['--inputs=spin'], 'MTA_SPIN_PT', 'mta-spin.pt'],
                    [['--inputs='], 'MTA_PLAIN_PT', 'mta-plain.pt'],
                    [
                        ['--inputs=charge,spin_multiplicity', '--undeclared'],
                        'MTA_UNDECLARED_PT',
                        'mta-undeclared.pt',
                    ],
                ]
                    _model = custom_target(
                        'mta-' + _spec[1],
                        output: _spec[2],
                        command: [_uma_py, _mta_export] + _spec[0] + [
                            '--out',
                            '@OUTPUT@',
                        ],
                        build_by_default: false,
                    )
                    engine_table_depends += [_model]
                    engine_table_env.set(_spec[1], _model.full_path())
                endforeach
            endif
            engine_table_env.set(
                'RGPOT_ENGINE_EXPECT',
                ','.join(engine_table_expect),
            )
            test(
                'EngineTableTest',
                executable(
                    'engine_table_test',
                    sources: ['tests/EngineTableTest.cc'],
                    dependencies: test_deps + [capnp_rpc_dep, ptlrpc_dep],
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                    build_rpath: _metatomic_build_rpath,
                ),
                env: engine_table_env,
                depends: engine_table_depends,
                workdir: meson.source_root(),
            )
        endif
        python3_prog = find_program('python3', required: false)
        if python3_prog.found()
            cxx_layout_stamp = executable(
                'print_cxx_layout_stamp',
                sources: ['tests/print_cxx_layout_stamp.cc'],
                include_directories: _incdirs + ['.'],
                cpp_args: _args,
                link_language: 'cpp',
            )
            test(
                'CxxSoversion',
                python3_prog,
                args: [
                    files('tests/check_cxx_soversion.py'),
                    rgpot_cxx_soversion,
                    rgpot.full_path(),
                    cxx_layout_stamp.full_path(),
                ],
                depends: [rgpot, cxx_layout_stamp],
            )
        endif
    endif
    if _mpi_dep.found() and not is_disabler(calculator_group_test_exe)
        _calc_mpiexec = find_program('mpirun', 'mpiexec', required: false)
        if _calc_mpiexec.found()
            calculator_group_mpi_sh = find_program(
                meson.current_source_dir() / 'tests/calculator_group_mpi.sh',
            )
            test(
                'CalculatorGroupShare',
                calculator_group_mpi_sh,
                args: [calculator_group_test_exe.full_path(), 'share'],
                depends: [calculator_group_test_exe],
                kwargs: _rt_kw['CalculatorGroupShare'],
            )
            test(
                'CalculatorGroupBadOwner',
                calculator_group_mpi_sh,
                args: [calculator_group_test_exe.full_path(), 'bad-owner'],
                depends: [calculator_group_test_exe],
                kwargs: _rt_kw['CalculatorGroupBadOwner'],
            )
        endif
    endif

    if has_dftd3 or has_dftd4
        test(
            'DftdGoldens',
            executable(
                'dftd_golden_test',
                sources: ['tests/DftdGoldenTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            workdir: meson.source_root(),
            suite: 'dftd',
        )
        executable(
            'dump_dftd_goldens',
            sources: ['tests/dump_dftd_goldens.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
    endif

    if has_xckernel
        test(
            'XcKernelGoldens',
            executable(
                'xckernel_test',
                sources: ['tests/XcKernelTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            workdir: meson.source_root(),
            suite: 'xckernel',
        )
        executable(
            'xckernel_bench',
            sources: ['tests/xckernel_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
        executable(
            'dump_xckernel_goldens',
            sources: ['tests/dump_xckernel_goldens.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
    endif

    if has_xtb
        xtb_test_env = environment()
        xtb_test_env.set('RGPOT_XTB_ENGINE', xtb_engine.full_path())
        # Tells XTBDlopenTest to require that the bare soname resolves, so a
        # broken loader path cannot skip the module-load regression.
        xtb_test_env.set('RGPOT_XTB_EXPECT_BARE_SONAME', '1')
        # OpenMP threads spin while waiting. Beside other tests they slow a
        # hot cluster SCC by more than a hundredfold.
        xtb_test_env.set('OMP_NUM_THREADS', '1')
        # Glibc and dyld snapshot the loader path at exec. setenv inside a
        # Catch process does not make the bare soname visible, so the test
        # environment (and the binary rpath below) publish xtb_engine_dir
        # before XTBDlopenTest starts.
        if host_system == 'windows'
            xtb_test_env.prepend('PATH', xtb_engine_dir, separator: ';')
        elif host_system == 'darwin'
            xtb_test_env.prepend('DYLD_LIBRARY_PATH', xtb_engine_dir)
            xtb_test_env.prepend('DYLD_FALLBACK_LIBRARY_PATH', xtb_engine_dir)
        else
            xtb_test_env.prepend('LD_LIBRARY_PATH', xtb_engine_dir)
        endif
        _xtb_test_rpath = xtb_engine_dir
        if _metatomic_build_rpath != ''
            _xtb_test_rpath = ':'.join([_metatomic_build_rpath, xtb_engine_dir])
        endif
        test(
            'XTBtest',
            executable(
                'xtb_test',
                sources: ['tests/XTBPotTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _xtb_test_rpath,
            ),
            env: xtb_test_env,
            depends: [xtb_engine],
            workdir: meson.source_root(),
            suite: 'xtb',
            # Hot Si13 frames whose SCC runs out its iterations, twice.
            timeout: 120,
        )
        if get_option('with_rust_core')
            test(
                'FusedXtbTest',
                executable(
                    'fused_xtb_test',
                    sources: ['tests/FusedXtbTest.cc'],
                    dependencies: test_deps,
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                    build_rpath: _xtb_test_rpath,
                ),
                env: xtb_test_env,
                depends: [xtb_engine],
                workdir: meson.source_root(),
                suite: 'xtb',
                timeout: 120,
            )
        endif
        test(
            'XTBDlopenTest',
            executable(
                'xtb_dlopen_test',
                sources: ['tests/XTBDlopenTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _xtb_test_rpath,
            ),
            env: xtb_test_env,
            depends: [xtb_engine],
            kwargs: _rt_kw['XTBDlopenTest'],
        )
        # Timing microbench: linked vs dlopen (same process OK for xTB).
        xtb_backend_bench = executable(
            'xtb_backend_bench',
            sources: ['tests/xtb_backend_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        # Warm versus cold GFN2 single points on (H2O)16 and (H2O)20.
        # Not a default test: the timed run is serial, under hostname bench.
        xtb_warm_bench = executable(
            'xtb_warm_bench',
            sources: ['tests/xtb_warm_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
    else
        if not get_option('with_rpc_client_only')
            # The dlopen frontend is in librgpot whether or not with_xtb
            # built an engine. Without an engine directory the bare-soname
            # checks skip themselves.
            test(
                'XTBDlopenTest',
                executable(
                    'xtb_dlopen_test',
                    sources: ['tests/XTBDlopenTest.cc'],
                    dependencies: test_deps,
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                    build_rpath: _metatomic_build_rpath,
                ),
                kwargs: _rt_kw['XTBDlopenTest'],
            )
        endif
    endif

    # Pot microbenchmarks. Own binary, outside the default suite: the cases
    # carry the hidden Catch2 tag pair "[.][benchmark]" and only run when the
    # tag is requested, which `meson test --benchmark` does through args.
    # LJPot lives in rgpot_core, absent from client-only builds.
    if not get_option('with_rpc_client_only')
        pot_bench_args = test_args
        if has_fortran
            pot_bench_args += ['-DRGPOT_BENCH_CUH2']
        endif
        pot_bench = executable(
            'pot_bench',
            sources: ['tests/PotBench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: pot_bench_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        benchmark(
            'pot_bench',
            pot_bench,
            args: ['[benchmark]'],
            workdir: meson.source_root(),
            timeout: 300,
        )
        sw_si_bench = executable(
            'sw_si_bench',
            sources: ['tests/sw_si_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        benchmark(
            'sw_si_bench',
            sw_si_bench,
            workdir: meson.source_root(),
            timeout: 120,
        )
        tersoff_sic_bench = executable(
            'tersoff_sic_bench',
            sources: ['tests/tersoff_sic_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        benchmark(
            'tersoff_sic_bench',
            tersoff_sic_bench,
            workdir: meson.source_root(),
            timeout: 120,
        )
        eam_cell_bench = executable(
            'eam_cell_bench',
            sources: ['tests/eam_cell_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        benchmark(
            'eam_cell_bench',
            eam_cell_bench,
            workdir: meson.source_root(),
            timeout: 300,
        )
        classical_scaling_bench = executable(
            'classical_scaling_bench',
            sources: ['tests/classical_scaling_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        benchmark(
            'classical_scaling_bench',
            classical_scaling_bench,
            workdir: meson.source_root(),
            timeout: 0,
        )
        sc_call_bench = executable(
            'sc_call_bench',
            sources: ['tests/sc_call_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        benchmark(
            'sc_call_bench',
            sc_call_bench,
            workdir: meson.source_root(),
            timeout: 0,
        )
        thread_count_bench = executable(
            'thread_count_bench',
            sources: ['tests/thread_count_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        benchmark(
            'thread_count_bench',
            thread_count_bench,
            workdir: meson.source_root(),
            timeout: 0,
        )
        morse_omp_bench = executable(
            'morse_omp_bench',
            sources: ['tests/morse_omp_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        benchmark(
            'morse_omp_bench',
            morse_omp_bench,
            workdir: meson.project_source_root(),
            timeout: 0,
        )
        # tools/lr_physics drives these. The bench pins workers with
        # sched_setaffinity, which only Linux has.
        if has_long_range and host_machine.system() == 'linux'
            foreach _lr : ['lr_phys_bench', 'lr_sets', 'lr_d4_periodic']
                set_variable(
                    _lr,
                    executable(
                        _lr,
                        sources: ['tests/' + _lr + '.cc'],
                        dependencies: test_deps,
                        include_directories: _incdirs + ['.'],
                        cpp_args: test_args,
                        link_with: _linkto,
                        link_language: 'cpp',
                        build_rpath: _metatomic_build_rpath,
                        install: false,
                    ),
                )
            endforeach
            benchmark(
                'lr_phys_bench',
                lr_phys_bench,
                args: ['6'],
                env: {'LR_LOG': meson.current_build_dir() / 'lr_phys_bench.log'},
                timeout: 0,
            )
        endif
        # One child process per row, via fork and waitpid, and the peak
        # resident set is read from /proc. Windows has neither.
        if host_machine.system() != 'windows'
            stream_scale_bench = executable(
                'stream_scale_bench',
                sources: ['tests/stream_scale_bench.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
                install: false,
            )
            benchmark(
                'stream_scale_bench',
                stream_scale_bench,
                workdir: meson.source_root(),
                timeout: 0,
            )
        endif
        tersoff_held_bench = executable(
            'tersoff_held_bench',
            sources: ['tests/tersoff_held_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
            install: false,
        )
        benchmark(
            'tersoff_held_bench',
            tersoff_held_bench,
            args: ['--runs', '3', '--calls', '50'],
            workdir: meson.source_root(),
            timeout: 300,
        )
    endif
endif

# ------------------------ Install / pkg-config for eOn and other consumers
# eOn prefers dependency('rgpot') over subproject wrap. Packaging profile
# matching eOn wrap defaults (no torch/xtb at link time; engines dlopen):
#   -Dwith_rpc_client_only=true -Dpure_lib=true
#   -Dwith_tests=false -Dwith_examples=false
#   -Dwith_xtb=false -Dwith_metatomic=false
# XTBDlopen frontend still builds; set RGPOT_XTB_ENGINE for libxtb_engine.so.
# See docs/eon_pkgconfig.md.
if not meson.is_subproject()
    install_headers('../include/rgpot/plugin.h', subdir: 'rgpot')
    # Headers: #include "rgpot/NWChemPot/..." with -I${prefix}/include
    # install_subdir matches exclude_files against each file's path
    # relative to rgpot/. A glob is not such a path, so '*.cc' drops
    # nothing and PotHelpers.cc, EMT/EMTPot.cc and the other
    # implementation files are installed. Pass every implementation path
    # literally. rpc/Potentials.capnp stays out because rpc/meson.build
    # installs the effective schema in its place.
    _install_py = find_program('python3', required: false)
    if not _install_py.found()
        error('installing headers needs python3 to list implementation sources')
    endif
    _install_impl = run_command(
        _install_py,
        meson.current_source_dir() / 'tests' / 'check_installed_sources.py',
        '--list',
        meson.current_source_dir() / 'rgpot',
        check: true,
    ).stdout().strip().split('\n')
    _install_exclude = ['rpc/Potentials.capnp']
    foreach _rel : _install_impl
        if _rel != ''
            _install_exclude += [_rel]
        endif
    endforeach
    install_subdir(
        'rgpot',
        install_dir: get_option('includedir'),
        exclude_directories: ['__pycache__', 'rpc/local', 'Morse/detail'],
        exclude_files: _install_exclude,
    )
    if get_option('with_tests')
        _install_meson = find_program('meson', required: false)
        if _install_meson.found()
            test(
                'installed_sources',
                _install_py,
                args: [
                    meson.current_source_dir() / 'tests' / 'check_installed_sources.py',
                    meson.project_build_root(),
                    _install_meson.full_path(),
                ],
                kwargs: _rt_kw['installed_sources'],
            )
        endif
    endif

    # The .pc always ships: consumers discover whatever this build enabled
    # through the RGPOT_HAS_* cflags rather than by the file's absence.
    _pc_libs = []
    _pc_requires = []
    _pc_cflags = []
    if not get_option('with_rpc_client_only')
        _pc_libs += [rgpot]
        _pc_cflags += ['-DRGPOT_HAS_POTLIB=TRUE']
        if has_fortran
            _pc_cflags += ['-DRGPOT_HAS_FORTRAN_POTS=TRUE']
        endif
        if has_ipi
            _pc_cflags += ['-DRGPOT_HAS_IPI=TRUE']
        endif
        if has_water
            _pc_cflags += ['-DRGPOT_HAS_WATER=TRUE']
        endif
        if has_eam_table
            _pc_cflags += ['-DRGPOT_HAS_EAM_TABLE=TRUE']
        endif
        if has_metatomic
            _pc_cflags += ['-DRGPOT_HAS_METATOMIC_POT=TRUE']
            _pc_cflags += ['-DRGPOT_HAS_UMA=TRUE']
        endif
        if has_xtb
            _pc_cflags += ['-DRGPOT_HAS_XTB_POT=TRUE']
        endif
        if has_tblite
            _pc_cflags += ['-DRGPOT_HAS_TBLITE_POT=TRUE']
        endif
        if has_dftd3
            _pc_cflags += ['-DRGPOT_HAS_DFTD3_POT=TRUE']
        endif
        if has_dftd4
            _pc_cflags += ['-DRGPOT_HAS_DFTD4_POT=TRUE']
        endif
        if has_cp2k
            _pc_cflags += ['-DRGPOT_HAS_CP2K=TRUE']
        endif
        if has_expr
            _pc_cflags += ['-DRGPOT_HAS_EXPR=TRUE']
        endif
        if has_skala
            _pc_cflags += ['-DRGPOT_HAS_SKALA=TRUE']
        endif
        if get_option('with_cache')
            # PotentialCache.hpp includes rocksdb/db.h.
            _pc_cflags += ['-DRGPOT_HAS_CACHE=TRUE']
            _pc_requires += ['rocksdb']
        endif
    endif
    if _rpc_stack
        # Umbrella builds keep these symbols inside librgpot. The archives
        # are not installed, so they cannot appear on the link line.
        # Client-only builds have no umbrella; the archives are the product.
        if get_option('with_rpc_client_only')
            _pc_libs += [nwchempot, cpmdpot]
        endif
        _pc_requires += ['capnp-rpc']
        _pc_cflags += [
            '-DRGPOT_HAS_NWCHEM_FRONTEND=TRUE',
            '-DRGPOT_HAS_CPMD_FRONTEND=TRUE',
        ]
    endif

    pkg = import('pkgconfig')
    pkg.generate(
        name: 'rgpot',
        description: 'rgpot potentials (librgpot umbrella) + QM frontends',
        url: 'https://github.com/OmniPotentRPC/rgpot',
        version: meson.project_version(),
        libraries: _pc_libs,
        requires: _pc_requires,
        subdirs: ['.'],
        extra_cflags: _pc_cflags,
        filebase: 'rgpot',
    )
    if _mpi_dep.found()
        # The MPI side of the calculator groups. librgpot loads it with
        # dlopen; linking it is for hosts that want MPI mapped up front.
        pkg.generate(
            rgpot_mpi_lib,
            name: 'rgpot-mpi',
            description: 'rgpot calculator groups over MPI (dlopen side of librgpot)',
            url: 'https://github.com/OmniPotentRPC/rgpot',
            version: meson.project_version(),
            subdirs: ['.'],
            extra_cflags: ['-DRGPOT_HAS_MPI'],
            filebase: 'rgpot-mpi',
        )
    endif
endif

if _mpi_dep.found()
    # Subproject consumers: dependency('rgpot-mpi') resolves to this build.
    meson.override_dependency('rgpot-mpi', rgpot_mpi_dep)
endif
