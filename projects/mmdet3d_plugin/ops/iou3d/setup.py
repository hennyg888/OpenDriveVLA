from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

COMPILE_ARGS = {
    'cxx': ['-O3', '-std=c++17'], 
    'nvcc': ['-O3', '--use_fast_math', '-std=c++17'] 
}

setup(
    name='iou3d_ext',
    
    ext_modules=[
        CUDAExtension(
            name='iou3d_cuda',
            sources=[
                "src/iou3d.cpp",
                "src/iou3d_kernel.cu",
            ],
            extra_compile_args=COMPILE_ARGS
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)