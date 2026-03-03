import os
import shutil
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")


def configure_gcc_toolset(min_major=9):
    gxx = shutil.which("g++")
    if gxx is not None:
        try:
            major = int(
                subprocess.check_output([gxx, "-dumpfullversion", "-dumpversion"], text=True)
                .strip()
                .split(".")[0]
            )
            if major >= min_major:
                return None
        except Exception:
            pass

    for version in ("14", "13", "12", "11", "10", "9"):
        root = f"/opt/rh/gcc-toolset-{version}/root/usr"
        gcc = os.path.join(root, "bin", "gcc")
        gxx = os.path.join(root, "bin", "g++")
        lib64 = os.path.join(root, "lib64")
        if os.path.exists(gcc) and os.path.exists(gxx):
            os.environ["CC"] = gcc
            os.environ["CXX"] = gxx
            os.environ["CUDAHOSTCXX"] = gxx
            os.environ["PATH"] = os.path.join(root, "bin") + os.pathsep + os.environ.get("PATH", "")
            os.environ["LD_LIBRARY_PATH"] = lib64 + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            print(f"Using gcc-toolset-{version} from {root}")
            return lib64
    return None


toolset_lib64 = configure_gcc_toolset()
extra_link_args = [f"-Wl,-rpath,{toolset_lib64}"] if toolset_lib64 else []


setup(
    name="cuda-attention-project",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="attention_ext._C",
            sources=[
                "attention_ext/attention.cpp",
                "attention_ext/attention_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
