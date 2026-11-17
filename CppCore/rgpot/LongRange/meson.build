# Long-range summands. Pairs.cc builds the neighbour lists.
# The CUDA translation unit is optional. The host fallback in
# CudaPath.cc is always part of the library.
lr_src = files(
    'CudaPath.cc',
    'Dispersion.cc',
    'Electro.cc',
    'FastD3.cc',
    'FastD4.cc',
    'Nlh.cc',
    'Pairs.cc',
    'SumPot.cc',
)
# The thread teams are OpenMP. Without it every summand runs serially.
lr_deps = _deps + [dependency('openmp', required: false)]
lr_args = _args
if get_option('with_cuda')
    nvcc = find_program('nvcc', required: true)
    # The object lands in the shared umbrella library, so it is PIC.
    lr_src += custom_target(
        'lr_cuda',
        input: 'CudaPath.cu',
        output: 'CudaPath.cu.o',
        command: [
            nvcc,
            '-arch=sm_70',
            '-std=c++20',
            '-Xcompiler',
            '-fPIC',
            '-c',
            '@INPUT@',
            '-o',
            '@OUTPUT@',
            '-I' + meson.project_source_root() / 'CppCore',
            '-I' + meson.project_source_root() / 'include',
        ],
    )
    lr_deps += cppc.find_library('cudart', required: true)
    lr_args += ['-DRGPOT_LR_HAS_CUDA']
endif
lr_phys = static_library(
    'lr_phys',
    lr_src,
    dependencies: lr_deps,
    cpp_args: lr_args,
    include_directories: _incdirs,
    pic: true,
    install: false,
)
