# Long-range summands. linkcell is the neighbour-list dependency.
# The CUDA translation unit is optional. The host fallback in
# CudaPath.cc is always part of the library.
lr_src = files(
    'Pairs.cc',
    'Electro.cc',
    'Dispersion.cc',
    'FastD3.cc',
    'FastD4.cc',
    'Nlh.cc',
    'SumPot.cc',
    'CudaPath.cc',
)
lr_deps = _deps + [linkcell_dep]
if get_option('with_cuda')
    nvcc = find_program('nvcc', required: true)
    lr_src += custom_target(
        'lr_cuda',
        input: 'CudaPath.cu',
        output: 'CudaPath.cu.o',
        command: [
            nvcc,
            '-arch=sm_70',
            '-std=c++20',
            '-c',
            '@INPUT@',
            '-o',
            '@OUTPUT@',
            '-I' + meson.project_source_root() / 'CppCore',
            '-I' + meson.project_source_root() / 'include',
        ],
    )
    lr_deps += find_library('cudart', required: true)
endif
lr_phys = static_library(
    'lr_phys',
    lr_src,
    dependencies: lr_deps,
    cpp_args: _args,
    include_directories: _incdirs,
    pic: true,
    install: false,
)
