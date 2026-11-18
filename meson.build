project(
    'rgpot',
    'cpp',
    # [[[cog import rgpot_version as v; cog.outl(f"    version: '{v.VERSION}',")]]]
    version: '3.7.0',
    # [[[end]]]
    default_options: ['warning_level=2', 'cpp_std=c++20'],
)
# IMPORTANT!! warning_level=3 passes -fimplicit-none
# Many of the older Fortran codes need implicit typing

host_system = host_machine.system()

# Add C++ compiler options
_cppc_early = meson.get_compiler('cpp')
if _cppc_early.get_id() == 'msvc' or _cppc_early.get_id() == 'clang-cl'
    # WIN32_LEAN_AND_MEAN / NOMINMAX: keep windows.h from defining COM
    # symbols (e.g. IPrintDialogServices) that break Cap'n Proto templates
    # under MSVC when any TU mixes windows.h with capnp headers.
    cpp_args = [
        '/W3',
        '/DWIN32_LEAN_AND_MEAN',
        '/DNOMINMAX',
    ]
else
    cpp_args = ['-Wall', '-Wextra', '-Wpedantic']
endif
_args = [cpp_args]  # Extra arguments
_deps = []  # Dependencies
_linkto = []  # All the sub-libraries (potentials)
_incdirs = []  # All the includes
_rgpot_srcs = []
_args += ['-DRGPOT_TEAM_MIN_ATOMS=' + get_option('team_min_atoms').to_string()]

# Add conditionals
if host_system == 'darwin'
    _args += ['-DOSX=TRUE']
    cpp_args += ['-faligned-allocation']
endif

# Languages
add_languages('c', required: true, native: false)
# Fortran pot kernels on all hosts where a compiler is available. On
# Darwin/CI, pair with system clang++ (not conda arm64-apple-darwin20 clang)
# so C++ targets link the host SDK.
#
# with_fortran_pots is independent of the RPC options: client-only builds
# skip the main library (and with it every pot), but the decision belongs
# to the pot option, not to the RPC role.
_fortran_pots_opt = get_option('with_fortran_pots')
if _fortran_pots_opt.disabled()
    has_fortran = false
else
    has_fortran = add_languages(
        'fortran',
        required: _fortran_pots_opt.enabled(),
        native: false,
    )
endif
fc = disabler()
if has_fortran
    fc = meson.get_compiler('fortran')
    # Matches the RGPOT_HAS_FORTRAN_POTS cflag rgpot.pc exports, so in-tree
    # sources and installed consumers gate the Fortran pots on one name.
    _args += ['-DRGPOT_HAS_FORTRAN_POTS=TRUE']
endif

# i-PI socket server. POSIX sockets only; auto skips Windows.
_ipi_opt = get_option('with_ipi')
has_ipi = false
if host_system == 'windows'
    if _ipi_opt.enabled()
        error(
            'with_ipi requires POSIX sockets and cannot be enabled on Windows',
        )
    endif
elif not _ipi_opt.disabled()
    has_ipi = true
endif
if has_ipi
    _args += ['-DRGPOT_HAS_IPI=TRUE']
endif

# C++ water family. Off unless the consumer asks, same default as eOn.
has_water = get_option('with_water')
if has_water
    _args += ['-DRGPOT_HAS_WATER=TRUE']
endif

# Tabulated EAM / Finnis-Sinclair files. Off unless asked for.
has_eam_table = get_option('with_eam_table')
if has_eam_table
    _args += ['-DRGPOT_HAS_EAM_TABLE=TRUE']
endif

rpc_enabled = get_option('with_rpc') or get_option('with_rpc_client_only')

if get_option('with_rpc_client_only')
    _args += ['-DRGPOT_RPC_CLIENT_ONLY=TRUE']
endif

cc = meson.get_compiler('c')
cppc = meson.get_compiler('cpp')

# Dependencies
# libm for Unix systems
m_dep = cppc.find_library('m', required: false)
_deps += m_dep
# Match the active C++ standard library (libc++ or libstdc++) on Unix hosts.
_cxx_rt_flag = ''
if host_system != 'windows'
    if cppc.get_define('_LIBCPP_VERSION', prefix: '#include <version>') != ''
        _cxx_rt_dep = cppc.find_library('c++', required: false)
        if _cxx_rt_dep.found()
            _cxx_rt_flag = '-lc++'
        endif
    else
        _cxx_rt_dep = cppc.find_library('stdc++', required: false)
        if _cxx_rt_dep.found()
            _cxx_rt_flag = '-lstdc++'
        endif
    endif
    if _cxx_rt_dep.found()
        _deps += [_cxx_rt_dep]
    endif
endif

# ------------------------ vesin (neighbour lists)
# One vesin for the whole tree: the nlist headers, MetatomicPot, and the
# Fortran pots all consume this dependency. Prefer an installed vesin
# >= 0.6; fall back to the vendored single-TU build, which carries the
# local patches maintained as vesin upstream candidates.
vesin_threads_dep = dependency('threads')
vesin_dep = dependency('vesin', version: '>=0.6', required: false)
# vesin_hdr_dep carries compile information only. Meson copies the objects
# of a static library into every static library that links it, so the
# convenience archives that link_whole into librgpot must see headers
# alone; the objects enter the umbrella once through vesin_dep.
vesin_hdr_dep = vesin_dep
if not vesin_dep.found()
    vesin_int = static_library(
        'vesin_internal',
        ['third_party/vesin/vesin-single-build.cpp'],
        dependencies: [vesin_threads_dep],
        # Consumers link statically; hidden symbols stay out of the
        # enclosing shared objects' dynamic tables.
        cpp_args: cppc.get_supported_arguments(['-fvisibility=hidden']),
        pic: true,
    )
    vesin_hdr_dep = declare_dependency(
        dependencies: [vesin_threads_dep],
        include_directories: include_directories('third_party/vesin'),
    )
    vesin_dep = declare_dependency(
        link_with: vesin_int,
        dependencies: [vesin_hdr_dep],
    )
    message('vesin: vendored third_party/vesin')
else
    message('vesin: system / pkg-config')
endif
_deps += vesin_hdr_dep

# vesin Fortran interface, vendored at the same version. Fortran pots take
# `use vesin` through vesin_f_dep (.mod directory propagated).
vesin_f_dep = disabler()
if has_fortran
    vesin_fortran_lib = static_library(
        'vesin_fortran',
        [
            'third_party/vesin/fortran/cdef.f90',
            'third_party/vesin/fortran/vesin.f90',
        ],
        dependencies: vesin_hdr_dep,
        pic: true,
    )
    # vesin_dep, not vesin_hdr_dep: the Fortran wrapper calls vesin_free
    # and vesin_neighbors_byref, so anything linking the wrapper needs the
    # vesin objects too. Compile information alone is enough for the
    # umbrella, which links vesin_dep separately, but not for an executable
    # that pulls wrapper objects of its own -- a static archive linked into
    # a shared library does not re-export what it resolved. Meson
    # deduplicates the static link, so naming it here does not duplicate
    # objects in the umbrella.
    vesin_f_dep = declare_dependency(
        link_with: vesin_fortran_lib,
        dependencies: vesin_dep,
        include_directories: vesin_fortran_lib.private_dir_include(),
    )
endif

# Fortran runtime for the umbrella. librgpot is a C++ target that takes the
# kernels through link_whole, which leaves meson no Fortran source of its
# own to infer the runtime from.
#
# Windows + MSVC + Flang: flang-new embeds /DEFAULTLIB:FortranRuntime.dynamic.lib
# (and FortranDecimal.dynamic.lib) in COFF objects. Those libs live next to the
# compiler under Library/lib on conda-forge / pixi. Search that dir so link.exe
# gets full paths rather than LNK1104 on a bare DEFAULTLIB name.
fortran_rt_deps = []
if has_fortran
    _fc = meson.get_compiler('fortran')
    _fc_id = _fc.get_id()
    _fc_rt_names = []
    _fc_rt_dirs = []
    if _fc_id == 'gcc'
        _fc_rt_names = ['gfortran']
    elif _fc_id == 'llvm-flang' or _fc_id == 'flang'
        if host_system == 'windows'
            _fc_rt_names = [
                'FortranRuntime.dynamic',
                'FortranDecimal.dynamic',
                'FortranRuntime.static',
                'FortranDecimal.static',
                'FortranRuntime',
                'FortranDecimal',
                'flang_rt.runtime',
                'flang',
            ]
            _fs = import('fs')
            _fc_bin = _fc.cmd_array()[0]
            if _fs.is_absolute(_fc_bin) or _fs.exists(_fc_bin)
                _fc_bindir = _fs.parent(_fc_bin)
                # .../Library/bin -> .../Library/lib
                _fc_rt_dirs += [_fs.parent(_fc_bindir) / 'lib']
                # .../bin -> .../lib (some layouts)
                _fc_rt_dirs += [_fc_bindir / '..' / 'lib']
            endif
        else
            _fc_rt_names = ['flang_rt.runtime', 'FortranRuntime', 'flang']
        endif
    elif _fc_id == 'intel-llvm' or _fc_id == 'intel'
        _fc_rt_names = ['ifcoremt']
    endif
    # librgpot_cuh2 links as C++. find_library('gfortran') often comes back
    # as a bare -lgfortran, which gfortran's driver searches and clang/c++
    # does not (Homebrew keeps the dylib off the default link path). Ask
    # the compiler for the file and pass that path.
    if _fc_id == 'gcc'
        _fs_rt = import('fs')
        foreach _cand : [
            'libgfortran.dylib',
            'libgfortran.5.dylib',
            'libgfortran.so',
            'libgfortran.so.5',
            'libgfortran.a',
        ]
            _pf = run_command(
                _fc.cmd_array() + ['-print-file-name=' + _cand],
                check: false,
            )
            if _pf.returncode() == 0
                _rt_path = _pf.stdout().strip()
                if _rt_path != _cand and _fs_rt.is_absolute(_rt_path) and _fs_rt.exists(
                    _rt_path,
                )
                    fortran_rt_deps += declare_dependency(link_args: [_rt_path])
                    message('fortran runtime (compiler file): ' + _rt_path)
                    break
                endif
            endif
        endforeach
        if host_system == 'darwin'
            foreach _cand : [
                'libquadmath.dylib',
                'libquadmath.0.dylib',
                'libquadmath.so',
                'libquadmath.a',
            ]
                _pf = run_command(
                    _fc.cmd_array() + ['-print-file-name=' + _cand],
                    check: false,
                )
                if _pf.returncode() == 0
                    _rt_path = _pf.stdout().strip()
                    if _rt_path != _cand and _fs_rt.is_absolute(_rt_path) and _fs_rt.exists(
                        _rt_path,
                    )
                        fortran_rt_deps += declare_dependency(
                            link_args: [_rt_path],
                        )
                        message('fortran runtime (compiler file): ' + _rt_path)
                        break
                    endif
                endif
            endforeach
        endif
    endif
    if fortran_rt_deps.length() == 0
        foreach _n : _fc_rt_names
            if _fc_rt_dirs.length() > 0
                _rt = _fc.find_library(_n, dirs: _fc_rt_dirs, required: false)
            else
                _rt = _fc.find_library(_n, required: false)
            endif
            if _rt.found()
                fortran_rt_deps += _rt
                message('fortran runtime: ' + _n)
                if host_system != 'windows'
                    break
                endif
            endif
        endforeach
    endif
endif

# --------------------- Optional dependencies
# Each dependency is controlled by a single flag; examples select
# themselves based on which deps are available (see CppCore/meson.build).

has_eigen = false
if get_option('with_eigen') or (
    get_option('with_examples')
    and not get_option(
    'with_rpc_client_only',
    )
)
    eigen_dep = dependency('eigen3', required: false)
    if not eigen_dep.found()
        eigen_dep = dependency(
            'eigen3',
            fallback: ['eigen', 'eigen_dep'],
            required: get_option('with_eigen'),
        )
    endif
    if eigen_dep.found()
        _deps += eigen_dep
        has_eigen = true
    endif
endif

has_xtensor = false
if get_option('with_xtensor')
    _deps += dependency('xtensor')
    _deps += dependency('xtensor-blas')
    _args += ['-DWITH_XTENSOR=TRUE']
    has_xtensor = true
endif

has_fmt = false
if not get_option('pure_lib') or (
    get_option('with_examples')
    and not get_option(
    'with_rpc_client_only',
    )
)
    _args += ['-DNOT_PURE_LIB=TRUE']
    fmt_dep = dependency('fmt', required: true)
    _deps += fmt_dep
    has_fmt = true
endif

has_xtb = false
if get_option('with_xtb')
    xtb_dep = dependency('xtb', required: true)
    _deps += xtb_dep
    _args += ['-DRGPOT_HAS_XTB=TRUE']
    has_xtb = true
endif

has_xckernel = false
if get_option('with_xckernel')
    _args += ['-DRGPOT_HAS_XCKERNEL=TRUE']
    has_xckernel = true
endif

# Static wrap fallbacks are Fortran archives; they need the language
# and libgfortran even when the in-tree Fortran pots are off.
if get_option('with_dftd3') or get_option('with_dftd4')
    if not add_languages('fortran', required: true, native: false)
        error('with_dftd3/with_dftd4 need a Fortran compiler')
    endif
    _dftd_fc = meson.get_compiler('fortran')
    _dftd_gfrt = _dftd_fc.find_library('gfortran', required: false)
    if _dftd_gfrt.found()
        _deps += _dftd_gfrt
    endif
endif

has_dftd3 = false
if get_option('with_dftd3')
    dftd3_dep = dependency(
        's-dftd3',
        required: true,
        fallback: ['s-dftd3', 'sdftd3_dep'],
        default_options: [
            'default_library=static',
            'python=false',
            'openmp=false',
        ],
    )
    _deps += dftd3_dep
    _args += ['-DRGPOT_HAS_DFTD3=TRUE']
    has_dftd3 = true
endif

has_long_range = get_option('with_long_range')

has_dftd4 = false
if get_option('with_dftd4')
    dftd4_dep = dependency(
        'dftd4',
        required: true,
        fallback: ['dftd4', 'dftd4_dep'],
        default_options: [
            'default_library=static',
            'python=false',
            'openmp=false',
            # auto looks for liblapack; manylinux/brew/apt ship OpenBLAS
            # as libopenblas, not a separate lapack soname.
            'lapack=openblas',
        ],
    )
    _deps += dftd4_dep
    _args += ['-DRGPOT_HAS_DFTD4=TRUE']
    has_dftd4 = true
endif

# CP2KPot, potserv's CP2K backend and the libcp2k shim build together or
# not at all, so a build without libcp2k compiles no CP2K code.
has_cp2k = false
if get_option('with_rpc')
    cp2k_lib_dep = dependency('libcp2k', required: get_option('with_cp2k'))
    has_cp2k = cp2k_lib_dep.found()
elif get_option('with_cp2k').enabled()
    error(
        'with_cp2k needs with_rpc: CP2KPot reads CP2KParams from the generated schema',
    )
endif
if has_cp2k
    _args += ['-DRGPOT_HAS_CP2K=TRUE']
endif

has_expr = false
if get_option('with_expr')
    subdir('third_party/lepton')
    _deps += lepton_dep
    _args += ['-DRGPOT_HAS_EXPR=TRUE']
    has_expr = true
endif

# With RPC on, the NWChemPot frontend builds and dlopens the split nwchemc engine
# (libnwchemc.so) at runtime. rgpot carries no in-process NWChem embed.
_args += ['-DRGPOT_HAS_NWCHEM_FRONTEND=TRUE']


has_skala = get_option('with_skala')
if has_skala
    if not get_option('with_rpc') and not get_option('with_rpc_client_only')
        error('with_skala=true needs with_rpc=true (NWChemPot / Cap\'n Proto)')
    endif
    _args += ['-DRGPOT_HAS_SKALA=TRUE']
endif

has_tblite = false
if get_option('with_tblite')
    # Installed pkg-config or cmake, or a wrap/subproject that exports
    # tblite_dep. declare_dependency() in tblite defaults its version to
    # the project version, so the same comparison covers both.
    tblite_dep = dependency(
        'tblite',
        required: true,
        fallback: ['tblite', 'tblite_dep'],
    )
    _deps += tblite_dep
    _args += ['-DRGPOT_HAS_TBLITE=TRUE']
    _tblite_ver = tblite_dep.version()
    if (
        tblite_dep.type_name() == 'internal'
        and (_tblite_ver == 'unknown'
        or _tblite_ver == 'undefined')
    )
        _tblite_sub = subproject('tblite', required: false)
        if _tblite_sub.found()
            _tblite_sub_ver = _tblite_sub.get_variable('tblite_dep').version()
            if _tblite_sub_ver != 'unknown' and _tblite_sub_ver != 'undefined'
                _tblite_ver = _tblite_sub_ver
            endif
        endif
    endif
    if _tblite_ver.version_compare('>=0.6.0')
        _args += ['-DRGPOT_TBLITE_XTB_CONFIG=1']
    endif
    has_tblite = true
endif

has_metatomic = false
_mta_engine_deps = []
if get_option('with_metatomic')
    # Discover torch without importing it (import can load NumPy C extensions
    # during meson configure). Supports multi-ABI builds via:
    #   RGPOT_TORCH_ROOT=/path/to/site-packages/torch  (isolated major)
    #   or first torch/version.py on sys.path (PYTHONPATH prefixes).
    # Prefer manual -isystem/-L over cmake Torch: old/CUDA wheels break cmake
    # with missing nvcc; libtorch_cpu.so is enough for the engine plugin.
    # Metatomic multi-ABI headers always come from the host purelib.
    _py_probe = run_command(
        find_program('python3'),
        '-c',
        '''
import os
import re
import sys
import sysconfig
from pathlib import Path

purelib = Path(sysconfig.get_paths()["purelib"])

def torch_root() -> Path:
    env = os.environ.get("RGPOT_TORCH_ROOT", "").strip()
    if env:
        p = Path(env)
        if (p / "version.py").is_file():
            return p
        if (p / "torch" / "version.py").is_file():
            return p / "torch"
        raise SystemExit(f"RGPOT_TORCH_ROOT has no version.py: {env}")
    for entry in sys.path:
        cand = Path(entry) / "torch" / "version.py"
        if cand.is_file():
            return cand.parent
    host = purelib / "torch" / "version.py"
    if host.is_file():
        return host.parent
    raise SystemExit("torch not found (set RGPOT_TORCH_ROOT or install torch)")

troot = torch_root()
version_file = troot / "version.py"
match = re.search(
    r"^__version__\s*=\s*['\"]([^'\"]+)",
    version_file.read_text(),
    re.MULTILINE,
)
if match is None:
    raise SystemExit(f"torch version not found in {version_file}")
version = ".".join(match.group(1).split("+")[0].split(".")[:2])
libdir = troot / "lib"
inc = troot / "include"
api_inc = troot / "include" / "torch" / "csrc" / "api" / "include"
if not (libdir / "libtorch_cpu.so").is_file() and not (libdir / "libtorch_cpu.dylib").is_file():
    raise SystemExit(f"libtorch_cpu not under {libdir}")
print(purelib)
print(version)
print(libdir)
print(inc)
print(api_inc)
''',
        check: true,
    ).stdout().strip().split('\n')
    _py_purelib = _py_probe[0]
    _torch_major = _py_probe[1]
    _torch_libdir = _py_probe[2]
    _torch_inc = _py_probe[3]
    _torch_api_inc = _py_probe[4]
    # Declaring the runtime library lets Meson populate the build RPATH.
    # The ordered link arguments retain the split-library Torch ABI.
    #
    # -rpath-link and --as-needed are GNU ld only; Apple's ld rejects them
    # outright ("unknown options"), and it resolves the dylibs from -L plus
    # their install_name anyway. Older libtorch splits symbols across
    # libtorch + libtorch_cpu + libc10, which is what --no-as-needed guards
    # on the GNU side.
    _torch_link_args = ['-L' + _torch_libdir]
    if host_system == 'darwin'
        _torch_link_args += ['-ltorch', '-ltorch_cpu', '-lc10']
    else
        _torch_link_args += [
            '-Wl,-rpath-link,' + _torch_libdir,
            '-Wl,--no-as-needed',
            '-ltorch',
            '-ltorch_cpu',
            '-lc10',
            '-Wl,--as-needed',
        ]
    endif
    torch_dep = declare_dependency(
        dependencies: cppc.find_library(
            'torch_cpu',
            dirs: [_torch_libdir],
            static: false,
        ),
        compile_args: [
            '-isystem',
            _torch_inc,
            '-isystem',
            _torch_api_inc,
        ],
        link_args: _torch_link_args,
    )
    # Torch stays off the umbrella _deps list. Putting it there makes
    # librgpot NEEDED libtorch_cpu and `import rgpot` requires torch.

    # metatensor-torch / metatomic-torch headers live in the
    # python interpreter's site-packages under
    # <purelib>/metatensor/torch/torch-X.Y/include and
    # <purelib>/metatomic/torch/torch-X.Y/include. The cmake /
    # pkg-config dependency lookups don't see those, and
    # find_library() doesn't add include paths. Ask sysconfig for
    # the right purelib and propagate -isystem includes manually.
    # Layout (current PyPI wheels):
    #   metatensor/lib/libmetatensor.so
    #   metatensor_torch/torch-X.Y/{lib,include}
    #   metatomic/torch/torch-X.Y/{lib,include}
    #   vesin/lib/libvesin.so
    _mts_torch_root = _py_purelib / 'metatensor_torch' / (
        'torch-' + _torch_major
    )
    _mta_torch_root = _py_purelib / 'metatomic' / 'torch' / (
        'torch-' + _torch_major
    )
    # Legacy layout fallback
    _mts_torch_legacy = _py_purelib / 'metatensor' / 'torch' / (
        'torch-' + _torch_major
    )

    _mts_inc_dirs = [
        _py_purelib / 'metatensor' / 'include',
        _mts_torch_root / 'include',
        _mts_torch_legacy / 'include',
        _mta_torch_root / 'include',
        _py_purelib / 'vesin' / 'include',
    ]
    _mts_inc_args = []
    foreach d : _mts_inc_dirs
        _mts_inc_args += ['-isystem', d]
    endforeach

    # Always use purelib find_library + -isystem (no pkg-config/cmake): multi-ABI
    # builds must not pull host Torch cmake includes (dtype symbol conflicts).
    _mts_lib_dirs = [
        _mts_torch_root / 'lib',
        _mts_torch_legacy / 'lib',
        _py_purelib / 'metatensor' / 'lib',
    ]
    mts_dep = declare_dependency(
        dependencies: [
            cppc.find_library(
                'metatensor',
                dirs: [_py_purelib / 'metatensor' / 'lib'],
            ),
            cppc.find_library(
                'metatensor_torch',
                dirs: _mts_lib_dirs,
            ),
        ],
        compile_args: _mts_inc_args,
    )

    mta_dep = cppc.find_library(
        'metatomic_torch',
        dirs: [
            _mta_torch_root / 'lib',
        ],
    )
    # Engine plugin only. Do not add these to _deps: the umbrella
    # librgpot is the pip import path and must stay torch-free.
    _mta_engine_deps = [
        torch_dep,
        mts_dep,
        mta_dep,
        declare_dependency(
            link_args: [
                '-L' + (_mts_torch_root / 'lib'),
                '-L' + (_mta_torch_root / 'lib'),
                '-L' + (_py_purelib / 'metatensor' / 'lib'),
            ],
        ),
    ]

    # vesin comes from the top-level dependency (vendored static fallback):
    # compile-time and runtime layouts always match, so the purelib lookup
    # is unnecessary here.

    _args += ['-DRGPOT_HAS_METATOMIC=TRUE']
    _args += ['-DRGPOT_HAS_UMA=TRUE']
    has_metatomic = true
endif

# --------------------- Rust Core (rgpot-core)
#
# Build the Rust crate via Cargo and link the resulting static library.
# The crate also generates include/rgpot.h via cbindgen.

if get_option('with_rust_core')
    cargo = find_program('cargo', required: true)

    # Determine Cargo build profile and library name
    if get_option('buildtype') == 'release'
        _cargo_profile_dir = 'release'
        _cargo_args = ['build', '--release']
    else
        _cargo_profile_dir = 'debug'
        _cargo_args = ['build']
    endif

    # Add feature flags for the Rust crate
    _cargo_features = []
    if rpc_enabled
        _cargo_features += ['rpc']
        # The rpc feature turns on schema, which guards
        # rgpot_potential_set_peer_capabilities in rgpot.h.
        _args += ['-DRGPOT_HAS_RPC=TRUE', '-DRGPOT_HAS_SCHEMA=TRUE']
    endif
    if get_option('with_cache')
        _cargo_features += ['cache']
    endif

    if _cargo_features.length() > 0
        _cargo_args += ['--features', ','.join(_cargo_features)]
    endif

    _rgpot_core_dir = meson.current_source_dir() / 'rgpot-core'

    # Build the Rust crate; produces librgpot_core.a and include/rgpot.h
    rgpot_rust_target = custom_target(
        'rgpot_rust_core',
        command: [
            cargo,
            _cargo_args,
            '--manifest-path',
            _rgpot_core_dir / 'Cargo.toml',
            '--target-dir',
            meson.current_build_dir() / 'cargo-target',
        ],
        output: ['librgpot_core.a'],
        build_by_default: true,
        console: true,
    )

    # Link the Rust static library
    _rust_target_dir = meson.current_build_dir() / 'cargo-target' / _cargo_profile_dir
    rgpot_rust_lib = cppc.find_library(
        'rgpot_core',
        dirs: [_rust_target_dir],
        required: true,
        static: true,
    )

    # Thread and dl libraries needed by the Rust runtime
    thread_dep = dependency('threads')
    dl_dep = cppc.find_library('dl', required: false)

    _deps += [rgpot_rust_lib, thread_dep, dl_dep]

    # librgpot links the archive and keeps only the members it references,
    # so the C ABI of rgpot-core is checked on the cdylib cargo builds
    # next to it.
    if host_machine.system() == 'linux'
        test(
            'abi_core_symbols',
            find_program('scripts/abi_symbols.sh'),
            args: [
                '-p',
                '^rgpot_',
                _rust_target_dir / 'librgpot_core.so',
                files('scripts/abi_core_baseline.txt'),
            ],
            depends: rgpot_rust_target,
            timeout: 30,
        )
    endif

    # Include paths: generated C header + C++ RAII wrappers
    _incdirs += [
        include_directories('rgpot-core' / 'include'),  # rgpot.h (cbindgen)
        include_directories('include'),  # rgpot/*.hpp (C++ RAII)
    ]
endif

# --------------------- linkcell / minimage pair list
# Optional. Auto builds it when cargo and rustc 1.89 (or rustup's 1.89.0)
# are available. Otherwise the classical pots stay on the C++ pair scan.
# MSVC cargo writes rgpot_fastnlist.lib; Unix and MinGW write
# librgpot_fastnlist.a. Link arguments are the system libraries rustc
# printed for this toolchain, not a fixed -pthread -ldl -lm list.
_fast_opt = get_option('with_fast_nlist')
_fast_cargo = find_program('cargo', required: _fast_opt.enabled())
_fast_rustc = find_program('rustc', required: false)
_fast_rustup = find_program('rustup', required: false)

_fast_minor = 0
if _fast_rustc.found()
    _vv = run_command(_fast_rustc, '-vV', check: false)
    if _vv.returncode() == 0
        foreach _line : _vv.stdout().split('\n')
            if _line.startswith('release:')
                _ver = _line.split()[1].split('-')[0]
                _parts = _ver.split('.')
                if _parts.length() >= 2 and _parts[0] == '1'
                    _fast_minor = _parts[1].to_int()
                endif
            endif
        endforeach
    endif
endif
if _fast_minor < 89 and _fast_rustup.found()
    _vv = run_command(
        _fast_rustup,
        'run',
        '1.89.0',
        'rustc',
        '-vV',
        check: false,
    )
    if _vv.returncode() == 0
        foreach _line : _vv.stdout().split('\n')
            if _line.startswith('release:')
                _ver = _line.split()[1].split('-')[0]
                _parts = _ver.split('.')
                if _parts.length() >= 2 and _parts[0] == '1'
                    _fast_minor = _parts[1].to_int()
                endif
            endif
        endforeach
    endif
endif

if _fast_opt.disabled()
    has_fast_nlist = false
elif _fast_minor >= 89 and _fast_cargo.found()
    has_fast_nlist = true
elif _fast_opt.enabled()
    error('with_fast_nlist=enabled needs cargo and rustc 1.89 or newer')
else
    has_fast_nlist = false
endif

if has_fast_nlist
    # A static umbrella cannot take the rust archive as a generated
    # source. Meson appends a custom-target library to the static link
    # arguments, and Apple ar (also GNU ar) reads the first file name as
    # the archive it writes. One relocatable object is a normal member.
    _fast_as_object = (
        get_option('default_library') == 'static'
        and host_system != 'windows'
    )
    _fast_lib_name = 'librgpot_fastnlist.a'
    if _fast_as_object
        _fast_lib_name = 'rgpot_fastnlist.o'
    elif host_system == 'windows' and (
        cppc.get_id() == 'msvc'
        or cppc.get_id() == 'clang-cl'
    )
        _fast_lib_name = 'rgpot_fastnlist.lib'
    endif
    _fastnlist_src = meson.current_source_dir() / 'CppCore/rgpot/nlist/fastnlist'
    _fast_target_dir = meson.current_build_dir() / 'fastnlist-cargo'
    _fast_py = find_program('python3')
    _fast_probe = run_command(
        _fast_py,
        files('scripts/build_fastnlist.py'),
        '--native-libs',
        _fast_cargo.full_path(),
        _fastnlist_src / 'Cargo.toml',
        _fast_target_dir,
        check: false,
    )
    if _fast_probe.returncode() != 0
        message(_fast_probe.stderr())
        error('fast pair list: rustc did not print native-static-libs')
    endif
    _fast_native = []
    foreach _line : _fast_probe.stdout().split('\n')
        _lib = _line.strip()
        if _lib != ''
            _fast_native += [_lib]
        endif
    endforeach
    if _fast_native.length() == 0
        error('fast pair list: rustc printed no native-static-libs')
    endif
    _fastnlist_tgt = custom_target(
        'rgpot_fastnlist',
        command: [
            _fast_py,
            files('scripts/build_fastnlist.py'),
            '--copy',
            _fast_cargo.full_path(),
            _fastnlist_src / 'Cargo.toml',
            _fast_target_dir,
            '@OUTPUT@',
        ],
        input: files(
            'CppCore/rgpot/nlist/fastnlist/Cargo.lock',
            'CppCore/rgpot/nlist/fastnlist/Cargo.toml',
            'CppCore/rgpot/nlist/fastnlist/src/lib.rs',
        ),
        output: _fast_lib_name,
        build_by_default: true,
        console: true,
    )
    _args += ['-DRGPOT_HAS_FAST_NLIST']
    # Archive first, then the system libraries rustc named. The generated
    # source is what ninja builds; repeating its path here keeps that
    # order on a single-pass link. A response file is not used: gcc leaves
    # a nested @file for ld, which then searches for a file of that name.
    # The static case passes the relocatable object as a source and keeps
    # only the native libraries in link_args, so ar is not handed a second
    # archive name. Consumers of the umbrella then take those libraries
    # without linking the object a second time.
    if _fast_as_object
        fastnlist_dep = declare_dependency(
            sources: [_fastnlist_tgt],
            link_args: _fast_native,
        )
        fastnlist_native_dep = declare_dependency(
            link_args: _fast_native,
        )
    else
        fastnlist_dep = declare_dependency(
            sources: [_fastnlist_tgt],
            link_args: [_fastnlist_tgt.full_path()] + _fast_native,
        )
        fastnlist_native_dep = fastnlist_dep
    endif
    message(
        'fast pair list: enabled (rustc minor ' + _fast_minor.to_string() + ')',
    )
else
    fastnlist_dep = declare_dependency()
    fastnlist_native_dep = fastnlist_dep
    message('fast pair list: disabled, classical pots use the C++ scan')
endif

# ------------------------ Highway (runtime ISA dispatch)
# System libhwy >= 1.2 when it is installed. Otherwise the wrap, and only
# when that tree is already on disk or the feature is enabled. auto must
# not download: a nodownload rehearsal can archive the wrap file without
# the extracted sources. Kernel TUs include hwy_disabled.h so a system
# libhwy and the wrap agree on the mask the dispatch table uses.
fs_hwy = import('fs')
highway_opt = get_option('with_highway')
have_highway = false
if not highway_opt.disabled()
    highway_dep = dependency('libhwy', version: '>=1.2', required: false)
    if not highway_dep.found()
        _hwy_tree = meson.project_source_root() / 'subprojects' / 'highway-1.2.0'
        _hwy_ready = fs_hwy.exists(_hwy_tree / 'hwy' / 'highway.h')
        if _hwy_ready or highway_opt.enabled()
            highway_dep = dependency(
                'libhwy',
                version: '>=1.2',
                fallback: ['highway', 'libhwy_dep'],
                required: highway_opt,
            )
        endif
    endif
    if highway_dep.found()
        # Not added to _deps. An installed static library copies objects from
        # a link_with dependency, and several pot archives install. Highway
        # stays on rgpot_core, which does not install.
        have_highway = true
        message('highway: enabled')
    else
        message('highway: scalar kernels only')
    endif
else
    message('highway: scalar kernels only')
endif

# --------------------- Projects
_incdirs += [include_directories('CppCore'), include_directories('include')]
subdir('CppCore')
rgpot_has_rpc = _rpc_stack
if get_option('with_rpc_client_only')
    rgpot_dep = declare_dependency(
        include_directories: _incdirs,
        link_with: _linkto,
        compile_args: _args,
        dependencies: _deps + [ptlrpc_dep, fastnlist_dep],
    )
endif

# Engine drivers link the umbrella. A client-only build has no LJPot.
# On Windows rgpot.dll exports only rgpot_library_version, so the drivers
# have nothing to link against, and they load plugins with dlopen and run
# pthreads; the CppCore engine plugins are left out there for the same
# reason.
if host_system == 'windows'
    message(
        'engines/ is not built on Windows: rgpot.dll does not export the C and C++ API the engine drivers link against',
    )
elif not get_option('with_rpc_client_only')
    subdir('engines')
endif

# --------------------- Python package (pip install rgpot)
if get_option('with_python')
    if get_option('with_rpc_client_only')
        error(
            'with_python requires the full pot library (not with_rpc_client_only)',
        )
    endif
    subdir('python')
endif

# Wheel builds set with_python. meson-python maps {includedir} onto the
# wheel headers scheme, so this header stays out of those wheels.
if not get_option('with_python')
    install_headers('bind/include/rgbind/rgbind.hpp', subdir: 'rgbind')
endif

# The documentation's run and compile blocks against this build;
# doccheck_static checks the rest against the source.
_doc_python = find_program('python3', required: false)
if get_option('with_tests') and _doc_python.found()
    _doccheck = files('docs/doccheck.py')
    # The C and C++ blocks are built with a cc driver (-fsyntax-only, -l,
    # -Wl,-rpath), and most run blocks link librgpot. Windows has cl.exe,
    # and rgpot.dll exports only rgpot_library_version, as for engines/.
    if host_system != 'windows'
        test(
            'doccheck_examples',
            _doc_python,
            args: [
                _doccheck,
                'blocks',
                '--root',
                meson.project_source_root(),
                '--build-dir',
                meson.project_build_root(),
            ],
            suite: 'docs',
            timeout: 1200,
            priority: -10,
        )
    endif
endif

# The py entries of CppCore/tests/registry.txt. @root@ is the source root
# and @capnp@ the capnp compiler.
_rt_python = find_program('python3', 'python')
foreach _rt : _rt_tests
    if _rt[1] != 'py'
        continue
    endif
    _rt_args = []
    foreach _arg : _rt[2]
        if _rt_args.length() == 0
            _rt_args += meson.project_source_root() / _arg
        elif _arg == '@root@'
            _rt_args += meson.project_source_root()
        elif _arg == '@capnp@'
            _rt_args += capnp.full_path()
        else
            _rt_args += _arg
        endif
    endforeach
    test(_rt[0], _rt_python, args: _rt_args, kwargs: _rt_kw[_rt[0]])
endforeach

# Every test registered above against registry.txt, given the features
# this configuration found.
if get_option('with_tests') and _rt_atoms['python']
    _rt_isa_variants = false
    if is_variable('_isa_exes')
        _rt_isa_variants = _isa_exes.length() > 0
    endif
    _rt_features = _rt_atoms + {
        'cp2k': has_cp2k,
        'mpi': _mpi_dep.found(),
        'mpiexec': find_program('mpirun', 'mpiexec', required: false).found(),
        'linux': host_system == 'linux',
        'posix': host_system != 'windows',
        'capnp': (
            dependency('capnp', required: false).found()
            and dependency('capnpc', required: false).found()
        ),
        'capnp_tool': find_program('capnp', required: false).found(),
        'rustc': find_program('rustc', required: false).found(),
        'tsan': (
            cppc.has_argument('-fsanitize=thread')
            and get_option('b_sanitize') == 'none'
        ),
        'isa_variants': _rt_isa_variants,
        'toplevel': not meson.is_subproject(),
        'xtb': has_xtb,
        'xckernel': has_xckernel,
        'expr': has_expr,
        'skala': has_skala,
    }
    _rt_args = []
    foreach _atom, _on : _rt_features
        _rt_args += _atom + '=' + (_on ? '1' : '0')
    endforeach
    test(
        'test_list_parity',
        _rt_python,
        args: [
            files('CppCore/tests/check_test_registry.py'),
            'meson',
            meson.global_build_root(),
        ] + _rt_args,
        kwargs: _rt_kw['test_list_parity'],
    )
endif
