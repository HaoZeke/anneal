plugin_test_inc = include_directories('../../../include')
plugin_null = shared_library(
    'rgpot_null',
    '../../examples/plugins/null_plugin.c',
    include_directories: plugin_test_inc,
    name_prefix: 'lib',
    install: false,
)
plugin_lj_collision = shared_library(
    'rgpot_lj_collision',
    'lj_collision_plugin.c',
    include_directories: plugin_test_inc,
    name_prefix: 'lib',
    install: false,
)
plugin_morse = shared_library(
    'rgpot_morse',
    '../../examples/plugins/morse_plugin.c',
    include_directories: plugin_test_inc,
    dependencies: cppc.find_library('m', required: false),
    name_prefix: 'lib',
    install: false,
)
plugin_fixture = shared_library(
    'rgpot_plugin_lifetime',
    '../plugin_lifetime_fixture.c',
    include_directories: plugin_test_inc,
    install: false,
)
plugin_env = environment()
plugin_env.set('RGPOT_TEST_PLUGIN_DIR', meson.current_build_dir())
foreach item : [
    ['PluginTest', 'plugin_test', '../PluginTest.cc'],
    ['PluginContractTest', 'plugin_contract_test', '../PluginContractTest.cc'],
    ['ExternalPluginTest', 'external_plugin_test', '../ExternalPluginTest.cc'],
]
    plugin_test = executable(
        item[1],
        item[2],
        dependencies: test_deps,
        include_directories: _incdirs,
        cpp_args: test_args,
        link_with: _linkto,
        link_language: 'cpp',
        build_rpath: _metatomic_build_rpath,
    )
    test(
        item[0],
        plugin_test,
        env: plugin_env,
        depends: [plugin_null, plugin_morse, plugin_lj_collision],
        kwargs: _rt_kw[item[0]],
    )
endforeach
if has_fortran
    plugin_coulomb = shared_library(
        'rgpot_coulomb',
        [
            '../../examples/plugins/coulomb_plugin.f90',
            '../../examples/plugins/coulomb_plugin_wrapper.c',
        ],
        include_directories: plugin_test_inc,
        name_prefix: 'lib',
        install: false,
    )
    plugin_fortran_test = executable(
        'external_fortran_plugin_test',
        '../ExternalFortranPluginTest.cc',
        dependencies: test_deps,
        include_directories: _incdirs,
        cpp_args: test_args,
        link_with: _linkto,
        link_language: 'cpp',
        build_rpath: _metatomic_build_rpath,
    )
    test(
        'ExternalFortranPluginTest',
        plugin_fortran_test,
        env: plugin_env,
        depends: plugin_coulomb,
        kwargs: _rt_kw['ExternalFortranPluginTest'],
    )
endif
plugin_lifetime = executable(
    'plugin_lifetime_test',
    '../PluginLifetimeTest.cc',
    dependencies: test_deps,
    include_directories: _incdirs,
    cpp_args: test_args,
    link_with: _linkto,
    link_language: 'cpp',
    build_rpath: _metatomic_build_rpath,
)
test(
    'PluginLifetimeTest',
    find_program('python3', 'python'),
    args: [
        files('../test_plugin_lifetime.py')[0],
        plugin_lifetime.full_path(),
        plugin_fixture.full_path(),
    ],
    depends: [plugin_lifetime, plugin_fixture],
    kwargs: _rt_kw['PluginLifetimeTest'],
)
