# Metatomic: thin dlopen frontend (always) + linked pot + engine plugin (with_metatomic)

mta_inc = include_directories('.')

dl_dep = dependency('dl', required: false)
if not dl_dep.found()
    dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

# Thin frontend — no torch; safe for pip host without metatomic at link time
metatomic_dlopen = static_library(
    'metatomic_dlopen',
    'MetatomicDlopen.cc',
    dependencies: _deps + [dl_dep],
    cpp_args: _args,
    include_directories: _incdirs + [mta_inc],
    install: false,
)

metatomic_dlopen_dep = declare_dependency(
    link_with: [metatomic_dlopen],
    dependencies: [dl_dep],
    include_directories: _incdirs + [mta_inc],
)

# Fat linked pot + portable engine plugin only when torch/metatomic are available
if has_metatomic
    # Static convenience lib; ships inside the librgpot umbrella
    # (link_whole), which then carries the torch link when metatomic is on.
    metatomicpot = static_library(
        'metatomicpot',
        'MetatomicPot.cc',
        dependencies: _deps + _mta_engine_deps,
        cpp_args: _args + ['-DRGPOT_METATOMIC_LINKED=1'],
        include_directories: _incdirs + [mta_inc],
        pic: true,
        install: false,
    )

    metatomicpot_dep = declare_dependency(
        link_with: [metatomicpot],
        include_directories: _incdirs + [mta_inc],
        compile_args: ['-DRGPOT_METATOMIC_LINKED=1'],
    )

    # Portable plugin RUNPATH: $ORIGIN peers under site-packages.
    # Single torch major only — multi-ABI packages install every torch-X.Y
    # directory; listing many makes ld.so load the first hit (wrong ABI).
    _engine_rpath_parts = [
        '$ORIGIN',
        '$ORIGIN/../torch/lib',
        '$ORIGIN/../metatensor/lib',
        '$ORIGIN/../vesin/lib',
        '$ORIGIN/../metatensor_torch/torch-' + _torch_major + '/lib',
        '$ORIGIN/../metatomic/torch/torch-' + _torch_major + '/lib',
        '$ORIGIN/../metatensor/torch/torch-' + _torch_major + '/lib',
    ]
    _engine_install_rpath = ':'.join(_engine_rpath_parts)
    # build_rpath: absolute purelib so ninja link/tests find libs during build only
    _engine_build_rpath = ''
    if is_variable('_torch_libdir')
        _engine_build_rpath = _torch_libdir
        if is_variable('_py_purelib')
            _engine_build_rpath = ':'.join(
                [
                    _torch_libdir,
                    _py_purelib / 'metatensor' / 'lib',
                    _py_purelib / 'metatensor_torch' / ('torch-' + _torch_major) / 'lib',
                    _py_purelib / 'metatomic' / 'torch' / (
                        'torch-' + _torch_major
                    ) / 'lib',
                    _py_purelib / 'vesin' / 'lib',
                ],
            )
        endif
    endif

    # The generic rgpot_engine_* table needs the Cap'n Proto params schema,
    # so it joins the engine only when the rpc stack is built.
    _metatomic_engine_srcs = ['MetatomicPot.cc', 'MetatomicEngineAbi.cc']
    _engine_table_deps = []
    _engine_table_args = []
    if _rpc_stack
        _metatomic_engine_srcs += [
            'MetatomicEngineTable.cc',
            '../engine/EngineTable.cc',
            engine_abi_stamp_source,
        ]
        _engine_table_deps = [capnp_rpc_dep, ptlrpc_dep]
        _engine_table_args = [
            '-DRGPOT_ENGINE_BUILD',
            '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(
                meson.project_version(),
            ),
        ]
    endif

    metatomic_engine = library(
        'metatomic_engine',
        _metatomic_engine_srcs,
        dependencies: _deps + _mta_engine_deps + [dl_dep, vesin_dep] + _engine_table_deps,
        cpp_args: _args + _engine_table_args + ['-DRGPOT_MTA_ENGINE_BUILD'],
        # NeighbourList.cc lives in rgpot_halfshell, not in rgpot_core.
        # The archive is absorbed; the engine still must not NEEDED host pot libs.
        link_with: [rgpot_halfshell],
        include_directories: _incdirs + [mta_inc],
        install: not meson.is_subproject(),
        build_rpath: _engine_build_rpath,
        install_rpath: _engine_install_rpath,
        gnu_symbol_visibility: 'default',
        # torch and its thread pools keep process-wide state; a dlclose of the
        # engine must not unmap them.
        link_args: meson.get_compiler('cpp').get_supported_link_arguments(
            ['-Wl,-z,nodelete'],
        ),
    )

    metatomic_engine_dep = declare_dependency(
        link_with: [metatomic_engine],
        include_directories: _incdirs + [mta_inc],
    )
endif
