# CP2KPot: dlopen consumer of the libcp2k shim (libcp2kc_rgpot). Only
# reached with has_cp2k, which needs with_rpc and libcp2k.

cp2k_inc = include_directories('.')

cp2k_dl_dep = dependency('dl', required: false)
if not cp2k_dl_dep.found()
    cp2k_dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

cp2kpot = static_library(
    'cp2kpot',
    'CP2KPot.cc',
    'CP2KDeck.cc',
    dependencies: _deps + [cp2k_dl_dep, ptlrpc_hdr_dep],
    cpp_args: _args,
    include_directories: _incdirs + [cp2k_inc],
    pic: true,
    # Convenience archive that librgpot link_wholes, as cpmdpot. Installing
    # it would copy the core objects into it and break the umbrella link.
    install: false,
)

cp2kpot_dep = declare_dependency(
    link_with: [cp2kpot],
    dependencies: [cp2k_dl_dep, ptlrpc_dep],
    include_directories: _incdirs + [cp2k_inc],
)

# The shim is the only target that links libcp2k and MPI; CP2KPot dlopens
# it, so librgpot links neither.
cp2k_mpi_c_dep = dependency('mpi', language: 'c', required: true)
_cp2k_libdir = cp2k_lib_dep.get_variable(pkgconfig: 'libdir', default_value: '')
cp2kc_shim = shared_library(
    'cp2kc_rgpot',
    'cp2kc_shim.c',
    dependencies: [cp2k_lib_dep, cp2k_mpi_c_dep],
    include_directories: [cp2k_inc],
    build_rpath: _cp2k_libdir,
    install_rpath: _cp2k_libdir,
    install: not meson.is_subproject(),
)
