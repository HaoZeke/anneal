# XTB: thin dlopen frontend (always) + linked pot + engine plugin (with_xtb)
#
# Match Metatomic: hosts link XTBDlopen without pkg-config xtb / NEEDED libxtb.
# Linked XTBPot and libxtb_engine.so still require -Dwith_xtb=true.

xtb_inc = include_directories('.')

dl_dep = dependency('dl', required: false)
if not dl_dep.found()
    dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

# Thin frontend, no libxtb; safe for with_xtb=false hosts (eOn, gprd paper).
xtb_dlopen = static_library(
    'xtb_dlopen',
    'XTBDlopen.cc',
    dependencies: _deps + [dl_dep],
    cpp_args: _args,
    include_directories: _incdirs + [xtb_inc],
    install: false,
)

xtb_dlopen_dep = declare_dependency(
    link_with: [xtb_dlopen],
    dependencies: [dl_dep],
    include_directories: _incdirs + [xtb_inc],
)

# Linked pot + portable engine plugin only when libxtb is available.
if has_xtb
    # Static convenience lib; ships inside the librgpot umbrella (link_whole).
    xtbpot = static_library(
        'xtbpot',
        'XTBPot.cc',
        dependencies: _deps,
        cpp_args: _args,
        include_directories: _incdirs + [xtb_inc],
        pic: true,
        install: false,
    )

    # Portable plugin: links libxtb + XTBPot implementation via XTBEngineAbi.
    _xtb_engine_srcs = ['XTBPot.cc', 'XTBEngineAbi.cc']
    _xtb_engine_deps = []
    _xtb_engine_args = []
    if _rpc_stack
        # The generic rgpot_engine_* table needs the Cap'n Proto params schema.
        _xtb_engine_srcs += [
            '../engine/XtbEngine.cc',
            '../engine/EngineTable.cc',
            engine_abi_stamp_source,
        ]
        _xtb_engine_deps = [capnp_rpc_dep, ptlrpc_dep]
        _xtb_engine_args = [
            '-DRGPOT_ENGINE_BUILD',
            '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(
                meson.project_version(),
            ),
        ]
    endif
    xtb_engine = library(
        'xtb_engine',
        _xtb_engine_srcs,
        dependencies: _deps + [vesin_dep] + _xtb_engine_deps,
        cpp_args: _args + _xtb_engine_args + ['-DRGPOT_XTB_ENGINE_BUILD'],
        include_directories: _incdirs + [xtb_inc],
        install: not meson.is_subproject(),
        gnu_symbol_visibility: 'default',
        # The OpenMP and Fortran runtimes keep threads and process state; a
        # dlclose of the engine must not unmap them.
        link_args: meson.get_compiler('cpp').get_supported_link_arguments(
            ['-Wl,-z,nodelete'],
        ),
    )

    xtb_engine_dep = declare_dependency(
        link_with: [xtb_engine],
        include_directories: _incdirs + [xtb_inc],
    )

    # Shared library directory. Tests put this on the process-start loader path
    # so a bad explicit engine_path cannot be rescued by the bare soname.
    xtb_engine_dir = meson.current_build_dir()
endif
