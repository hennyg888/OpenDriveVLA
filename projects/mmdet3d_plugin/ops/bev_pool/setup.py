from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='bev_pool_ext',
    ext_modules=[
        CUDAExtension(
            name='bev_pool_ext',
            sources=[
                'src/bev_pool_cpu.cpp',
                'src/bev_pool_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)