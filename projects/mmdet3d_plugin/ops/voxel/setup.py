from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

COMPILE_ARGS = {
    'cxx': ['-O3', '-std=c++17'], 
    'nvcc': ['-O3', '--use_fast_math', '-std=c++17'] 
}
setup(
    name='voxel_ext',
    
    ext_modules=[
        CUDAExtension(
            name='voxel_layer',
            sources=[
                "src/voxelization.cpp",
                "src/scatter_points_cpu.cpp",
                "src/scatter_points_cuda.cu",
                "src/voxelization_cpu.cpp",
                "src/voxelization_cuda.cu",
            ],
            extra_compile_args=COMPILE_ARGS
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)