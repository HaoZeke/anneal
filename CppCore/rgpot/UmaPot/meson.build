# UMA / OMol: vesin neighbor list + AOTInductor .pt2 via the Torch dependency.
uma = static_library(
    'uma',
    'UmaPot.cc',
    dependencies: _deps + [torch_dep, vesin_dep],
    cpp_args: _args,
    include_directories: ['../../'],
    pic: true,
    install: false,
)

# Portable engine plugin behind the generic rgpot engine C ABI
# (engine_c_abi.h): dlopen'd by hosts (eonclient), never linked, so the
# host carries no torch NEEDED entries. Config travels as a capnp
# UmaParams message, so the plugin folds in the generated schema TU
# (ptlrpc) and needs the rpc stack at build time. RUNPATH mirrors the
# metatomic engine; both reuse variables MetatomicPot/meson.build minted.
if _rpc_stack
    # Without the metatomic engine nothing has minted the plugin's run paths.
    # A build against torch alone finds libtorch where it was configured: a
    # libtorch tree does not travel with the plugin the way site-packages
    # peers do.
    if not is_variable('_engine_build_rpath')
        _engine_build_rpath = _torch_libdir
        _engine_install_rpath = _torch_libdir
    endif
    uma_engine = library(
        'uma_engine',
        [
            'UmaPot.cc',
            'UmaEngineAbi.cc',
            '../engine/EngineTable.cc',
            engine_abi_stamp_source,
        ],
        # ptlrpc_dep carries the generated Potentials.capnp.h include
        # view plus the schema TU objects (final-binary rule: a plugin
        # is a leaf, no double-compile risk).
        dependencies: _deps + [
            torch_dep,
            dl_dep,
            vesin_dep,
            capnp_rpc_dep,
            ptlrpc_dep,
        ],
        cpp_args: _args + [
            '-DRGPOT_ENGINE_BUILD',
            '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(
                meson.project_version(),
            ),
        ],
        # NeighbourList.cc is in rgpot_halfshell. The static archive is
        # absorbed into the plugin.
        link_with: [rgpot_halfshell],
        include_directories: ['../../'],
        install: not meson.is_subproject(),
        build_rpath: _engine_build_rpath,
        install_rpath: _engine_install_rpath,
        gnu_symbol_visibility: 'default',
        # torch and the AOTI runtime keep process-wide state and thread pools
        # alive; a dlclose of the engine must not unmap them.
        link_args: meson.get_compiler('cpp').get_supported_link_arguments(
            ['-Wl,-z,nodelete'],
        ),
    )

    uma_engine_dep = declare_dependency(
        link_with: [uma_engine],
        include_directories: ['../../'],
    )
else
    message(
        'uma_engine skipped: capnp config needs with_rpc or with_rpc_client_only',
    )
endif
