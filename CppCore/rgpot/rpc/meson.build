# Find dependencies and tools
capnp_rpc_dep = dependency('capnp-rpc')
capnp = find_program('capnp')
python = find_program('python3', 'python')

# Canonical contract pinned via subprojects/potentials-schema.wrap; the two
# vendored Potentials.capnp copies must stay byte-identical to it, which is
# what the sync tests check here and in a parent such as gpridge that
# resolves the subproject from its own wrap. An offline build
# (--wrap-mode=nodownload) skips them.
# rgpot-local fields are the add-only diffs in local/series. The build
# compiles, installs and serves the effective schema: the vendored copy
# with the series applied. schema-ordinal-ledger holds the copies, the
# series and the effective schema to Potentials.ordinals.
schema_ledger = files('schema_ledger.py')
rgpot_effective_schema = custom_target(
    'effective_schema',
    input: 'Potentials.capnp',
    output: 'Potentials.capnp',
    command: [
        python,
        schema_ledger,
        'apply',
        '--root',
        meson.project_source_root(),
        '--schema',
        '@INPUT@',
        '--series',
        meson.current_source_dir() / 'local',
        '--out',
        '@OUTPUT@',
    ],
    depend_files: rgpot_schema_inputs,
    install: not meson.is_subproject(),
    install_dir: get_option('includedir') / 'rgpot' / 'rpc',
)
test(
    'schema-ordinal-ledger-vendored',
    python,
    args: [schema_ledger, 'check', '--root', meson.project_source_root()],
    suite: 'schema',
)
test(
    'schema-ordinal-ledger-selftest',
    python,
    args: [
        meson.project_source_root() / 'CppCore/tests/test_schema_ledger.py',
    ],
    suite: 'schema',
)
potentials_schema_proj = subproject('potentials-schema', required: false)
if potentials_schema_proj.found()
    potentials_schema_canonical = potentials_schema_proj.get_variable(
        'potentials_capnp_file',
    )
    schema_sync_check = files('schema_sync_check.py')
    test(
        'schema-sync-canonical-rpc',
        python,
        args: [
            schema_sync_check,
            meson.current_source_dir() / 'Potentials.capnp',
            potentials_schema_canonical,
        ],
        suite: 'schema',
    )
    test(
        'schema-sync-canonical-core',
        python,
        args: [
            schema_sync_check,
            meson.project_source_root() / 'rgpot-core/schema/Potentials.capnp',
            potentials_schema_canonical,
        ],
        suite: 'schema',
    )
    test(
        'schema-ordinal-ledger',
        python,
        args: [
            schema_ledger,
            'check',
            '--root',
            meson.project_source_root(),
            '--subproject-schema',
            potentials_schema_canonical,
        ],
        suite: 'schema',
    )
endif

gen_rpc = custom_target(
    'gen_rpc',
    input: rgpot_effective_schema,
    output: ['Potentials.capnp.cpp', 'Potentials.capnp.h'],
    command: [
        python,
        meson.current_source_dir() / 'capnp_compile.py',
        capnp,
        '@OUTDIR@',
        meson.current_build_dir(),
        '@INPUT@',
    ],
    # Install generated header next to vendored schema for pkg-config consumers.
    install: not meson.is_subproject(),
    install_dir: [
        false,
        get_option('includedir') / 'rgpot' / 'rpc',
    ],
)
# Use the target object itself for includes to ensure proper dependency tracking
_incdirs += include_directories('.')

# Convenience archive, not a shipped shared object: the generated capnp
# translation unit belongs inside the one librgpot SONAME, and a separate
# libptlrpc.so would become a runtime dependency of every consumer that
# links a QM frontend.
ptlrpc = static_library(
    'ptlrpc',
    gen_rpc,
    dependencies: _deps + capnp_rpc_dep + [vesin_dep],
    cpp_args: _args,
    include_directories: _incdirs,
    pic: true,
    install: false,
)

_ptlrpc_extra_deps = [capnp_rpc_dep]
if host_machine.system() == 'windows'
    _ptlrpc_extra_deps += cppc.find_library('ws2_32')
endif

# Header only: listing the generated .cpp here would compile a second copy
# of the capnp translation unit into every consumer, and the convenience
# archives all link_whole into one librgpot. The archives take this view;
# only final binaries take the objects through ptlrpc_dep below.
ptlrpc_hdr_dep = declare_dependency(
    sources: gen_rpc[1],
    include_directories: _incdirs,
    dependencies: _ptlrpc_extra_deps,
)

ptlrpc_dep = declare_dependency(
    link_with: ptlrpc,
    dependencies: [ptlrpc_hdr_dep],
)

pot_bridge = library(
    'pot_client_bridge',
    ['pot_bridge.cc'],
    link_with: ptlrpc,
    dependencies: [capnp_rpc_dep, ptlrpc_hdr_dep],
    cpp_args: _args + ['-DPOT_BRIDGE_BUILD'],
    # gen_rpc outputs live in the build dir; include it explicitly so
    # Potentials.capnp.h is found even when only pot_bridge.cc is compiled first.
    include_directories: _incdirs + [include_directories('.')],
    install: not meson.is_subproject(),
)

pot_bridge_dep = declare_dependency(
    link_with: pot_bridge,
)
