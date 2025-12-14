from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def src(path):
    return os.path.join(THIS_DIR, path)

COMPILE_ARGS = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
}

setup(
    name="voxel_ext",
    ext_modules=[
        CUDAExtension(
            name="projects.mmdet3d_plugin.ops.voxel.voxel_layer",
            sources=[
                src("src/voxelization.cpp"),
                src("src/scatter_points_cpu.cpp"),
                src("src/scatter_points_cuda.cu"),
                src("src/voxelization_cpu.cpp"),
                src("src/voxelization_cuda.cu"),
            ],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args=COMPILE_ARGS,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
