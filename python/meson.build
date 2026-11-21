# Python package: rgpot — nanobind-only bindings (no pybind11).
#
# Policy (match pyeonclient):
#   * Nanobind only for the public extension.
#   * Stable ABI (abi3 / Py_LIMITED_API 3.12) on GIL builds with Python >= 3.12.
#   * Free-threaded (NB_FREE_THREADED) when Py_GIL_DISABLED; no abi3 then.
#
# Layout:
#   rgpot/__init__.py
#   rgpot/_core*.so | _core*.abi3.so
#   rgpot/lib/torch-X.Y/libmetatomic_engine.so  (multi-ABI engines, post-pack)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

_nb_probe = run_command(
    py,
    '-c',
    '''
import nanobind, pathlib, sys, sysconfig
p = pathlib.Path(nanobind.__file__).resolve().parent
cands = [p / "src" / "nb_combined.cpp", p / "nb_combined.cpp"]
comb = next(c for c in cands if c.is_file())
robin = p / "ext" / "robin_map" / "include"
if not (robin / "tsl" / "robin_map.h").is_file():
    for alt in (p / "include", pathlib.Path(nanobind.include_dir())):
        if (alt / "tsl" / "robin_map.h").is_file():
            robin = alt
            break
ft = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
ver = sys.version_info
print(nanobind.include_dir())
print(robin)
print(comb)
print("1" if ft else "0")
print(f"{ver.major}.{ver.minor}")
print(nanobind.__version__)
''',
    check: false,
)
if _nb_probe.returncode() != 0
    error(
        'with_python requires nanobind>=2.2 in the build Python ' + '(pip/pixi install nanobind)',
    )
endif
_nb_lines = _nb_probe.stdout().strip().split('\n')
_nb_include = _nb_lines[0]
_nb_robin = _nb_lines[1]
_nb_combined = _nb_lines[2]
_nb_free_threaded = _nb_lines[3] == '1'
_nb_version = _nb_lines[5]

message('rgpot python: nanobind @0@'.format(_nb_version))
message(
    'rgpot python: free-threaded interpreter = @0@'.format(_nb_free_threaded),
)

# -isystem so abs paths under .pixi/site-packages are allowed
nanobind_dep = declare_dependency(
    compile_args: [
        '-isystem' + _nb_include,
        '-isystem' + _nb_robin,
    ],
)

if not is_variable('rgpot')
    error(
        'with_python requires the full rgpot library (disable with_rpc_client_only)',
    )
endif
if not is_variable('metatomic_dlopen')
    error('with_python requires MetatomicDlopen (rgpot/MetatomicPot)')
endif

_py_incs = [_incdirs, include_directories('../bind/include')]
_py_rpath = '$ORIGIN:$ORIGIN/../.rgpot.mesonpy.libs:$ORIGIN/lib'
# The umbrella librgpot link_wholes every pot and the dlopen frontends;
# linking the static convenience libs again would just duplicate symbols
# into the extension.
_py_link = [rgpot]
# Header/compile flags only. Do not iterate _deps: that list includes
# torch when with_metatomic=true, and even a partial_dependency walk
# can leak link_args onto _core (NEEDED libtorch_cpu).
_py_hdr_deps = []
if has_dftd3
    _py_hdr_deps += dftd3_dep.partial_dependency(
        includes: true,
        compile_args: true,
    )
endif
if has_dftd4
    _py_hdr_deps += dftd4_dep.partial_dependency(
        includes: true,
        compile_args: true,
    )
endif
# Do not link cuh2 into the pip extension: the umbrella already carries the
# slab helpers, and LJ + metatomic dlopen are the product.

dl_dep = dependency('dl', required: false)
if not dl_dep.found()
    dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

_nb_cpp_args = _args + [
    '-DNB_DOMAIN=rgpot',
    '-Wno-unused-parameter',
    '-DRGPOT_PY_VERSION="@0@"'.format(meson.project_version()),
]
# F9 and the binding contract: no fused multiply-adds in this translation
# unit. The kernel libraries keep their own flags.
_nb_cpp_args += meson.get_compiler('cpp').get_supported_arguments(
    ['-ffp-contract=off'],
)

# F9: the extension exports PyInit__core and nothing else.
_core_link_args = []
if host_machine.system() == 'linux'
    _core_link_args = [
        '-Wl,--exclude-libs,ALL',
        '-Wl,--version-script,' + meson.current_source_dir() / 'core_exports.map',
    ]
elif host_machine.system() == 'darwin'
    _core_link_args = [
        '-Wl,-exported_symbols_list,' + meson.current_source_dir() / 'core_exports.txt',
    ]
endif

_core_sources = files('bind/bind_module.cpp') + [_nb_combined]

if _nb_free_threaded
    _nb_cpp_args += ['-DNB_FREE_THREADED']
    message('rgpot python: FREE_THREADED build; stable ABI skipped')
    _py_core = py.extension_module(
        '_core',
        sources: _core_sources,
        dependencies: [py_dep, nanobind_dep, dl_dep] + _py_hdr_deps,
        include_directories: _py_incs,
        link_with: _py_link,
        cpp_args: _nb_cpp_args,
        link_args: _core_link_args,
        install: true,
        install_rpath: _py_rpath,
        build_rpath: _py_rpath,
        subdir: 'rgpot',
        gnu_symbol_visibility: 'hidden',
    )
else
    _use_abi3 = py.language_version().version_compare('>=3.12')
    if _use_abi3
        message('rgpot python: STABLE_ABI build (abi3, Py_LIMITED_API=3.12)')
        _py_core = py.extension_module(
            '_core',
            sources: _core_sources,
            dependencies: [py_dep, nanobind_dep, dl_dep] + _py_hdr_deps,
            include_directories: _py_incs,
            link_with: _py_link,
            cpp_args: _nb_cpp_args,
            link_args: _core_link_args,
            install: true,
            install_rpath: _py_rpath,
            build_rpath: _py_rpath,
            subdir: 'rgpot',
            gnu_symbol_visibility: 'hidden',
            limited_api: '3.12',
        )
    else
        warning(
            'rgpot python: Python < 3.12 — version-specific extension ' + '(stable ABI requires 3.12+)',
        )
        _py_core = py.extension_module(
            '_core',
            sources: _core_sources,
            dependencies: [py_dep, nanobind_dep, dl_dep] + _py_hdr_deps,
            include_directories: _py_incs,
            link_with: _py_link,
            cpp_args: _nb_cpp_args,
            link_args: _core_link_args,
            install: true,
            install_rpath: _py_rpath,
            build_rpath: _py_rpath,
            subdir: 'rgpot',
            gnu_symbol_visibility: 'hidden',
        )
    endif
endif

py.install_sources(
    files(
        'rgpot/__init__.py',
        'rgpot/_core.pyi',
        'rgpot/lammps.py',
        'rgpot/py.typed',
    ),
    pure: false,
    subdir: 'rgpot',
)
# rgpot.ase speaks the schema the server serves: the vendored copy with the
# rgpot-local series applied.
_py_ase_schema = custom_target(
    'ase_effective_schema',
    input: '../CppCore/rgpot/rpc/Potentials.capnp',
    output: 'Potentials.capnp',
    command: [
        py,
        files('../CppCore/rgpot/rpc/schema_ledger.py'),
        'apply',
        '--root',
        meson.project_source_root(),
        '--schema',
        '@INPUT@',
        '--out',
        '@OUTPUT@',
    ],
    depend_files: rgpot_schema_inputs,
    install: true,
    install_dir: py.get_install_dir() / 'rgpot' / 'ase',
)
py.install_sources(
    files(
        'rgpot/ase/__init__.py',
        'rgpot/ase/anneal.py',
        'rgpot/ase/calculator.py',
        'rgpot/ase/claim.py',
        'rgpot/ase/rpc.py',
    ),
    pure: false,
    subdir: 'rgpot/ase',
)

if is_variable('metatomic_engine')
    message(
        'metatomic_engine installs into the wheel; multi-ABI pack via scripts/rgpot_pack_multi_abi_engines.sh',
    )
endif

# The extension NEEDs the librgpot SONAME. Meson installs the versioned
# library and a SONAME symlink. meson-python records the versioned file,
# and a wheel has no symlink, so a direct wheel or an sdist install cannot
# open the SONAME. Copy the library this build linked to a regular file in
# the build directory and install that file on the Python platlib. The
# directory is the one meson-python already uses for vendored libraries.
# The name is kCxxSoversion, checked against the built library. The source
# tree is not written.
if host_machine.system() == 'linux'
    _readelf = find_program('readelf', required: true)
    _librgpot_soname = 'librgpot.so.' + rgpot_cxx_soversion
    librgpot_soname = custom_target(
        'librgpot_soname',
        input: rgpot,
        output: _librgpot_soname,
        command: [
            py,
            '-c',
            '''
import pathlib, shutil, subprocess, sys
readelf, src_arg, dst_arg = sys.argv[1:4]
src = pathlib.Path(src_arg).resolve()
dst = pathlib.Path(dst_arg)
text = subprocess.check_output([readelf, "-d", str(src)], text=True)
soname = ""
for line in text.splitlines():
    if "(SONAME)" in line and "[" in line:
        soname = line.split("[", 1)[1].split("]", 1)[0]
        break
if soname != dst.name:
    raise SystemExit("SONAME is %r, expected %r" % (soname, dst.name))
shutil.copyfile(src, dst)
''',
            _readelf.full_path(),
            '@INPUT@',
            '@OUTPUT@',
        ],
        install: true,
        install_dir: py.get_install_dir(
            pure: false,
            subdir: '.rgpot.mesonpy.libs',
        ),
        build_by_default: true,
    )
endif
