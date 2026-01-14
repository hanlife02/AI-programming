from __future__ import annotations

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_ext_modules() -> list[CUDAExtension]:
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
    }
    return [
        CUDAExtension(
            name="task3_ops",
            sources=["csrc/bindings.cpp", "csrc/ops.cu"],
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    name="task3_ops",
    version="0.1.0",
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
)

