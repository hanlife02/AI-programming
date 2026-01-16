from __future__ import annotations

import os
import re
import subprocess

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def _arch_str_from_sm(sm: str) -> str | None:
    m = re.fullmatch(r"sm_(\d+)", sm.strip())
    if not m:
        return None
    num = int(m.group(1))
    return f"{num // 10}.{num % 10}"


def _nvcc_supported_arches() -> set[str]:
    if not CUDA_HOME:
        return set()
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    if not os.path.exists(nvcc):
        return set()
    try:
        out = subprocess.check_output([nvcc, "--list-gpu-arch"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return set()
    arches: set[str] = set()
    for tok in out.split():
        arch = _arch_str_from_sm(tok)
        if arch:
            arches.add(arch)
    return arches


def _parse_arch_list(value: str) -> list[str]:
    items: list[str] = []
    for raw in re.split(r"[;, ]+", value.strip()):
        if not raw:
            continue
        items.append(raw)
    return items


def _base_arch(arch: str) -> str:
    return arch.replace("+PTX", "").strip()


def _arch_key(arch: str) -> tuple[int, int]:
    major, minor = arch.split(".", 1)
    return int(major), int(minor)


def _ensure_highest_has_ptx(arch_list: list[str]) -> list[str]:
    if not arch_list:
        return arch_list
    bases = [_base_arch(a) for a in arch_list]
    highest = max(bases, key=_arch_key)
    out: list[str] = []
    for a in arch_list:
        if _base_arch(a) == highest:
            out.append(f"{highest}+PTX")
        else:
            out.append(_base_arch(a))
    return out


def _configure_cuda_arch_list() -> None:
    supported = _nvcc_supported_arches()
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    override_arch = os.environ.get("TASK3_CUDA_ARCH_LIST", "").strip()

    if override_arch:
        chosen = _parse_arch_list(override_arch)
    elif env_arch:
        chosen = _parse_arch_list(env_arch)
    elif torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        chosen = [f"{major}.{minor}+PTX"]
    else:
        chosen = ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]

    if supported:
        filtered: list[str] = []
        for a in chosen:
            if _base_arch(a) in supported:
                filtered.append(a)
        if not filtered:
            filtered = [max(supported, key=_arch_key) + "+PTX"]
        chosen = filtered

    chosen = _ensure_highest_has_ptx(chosen)

    if env_arch and chosen != _parse_arch_list(env_arch):
        print(
            f"[task3_ops] Overriding TORCH_CUDA_ARCH_LIST={env_arch!r} -> {(';'.join(chosen))!r} "
            f"(nvcc supported: {sorted(supported) if supported else 'unknown'})"
        )

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(chosen)


def get_ext_modules() -> list[CUDAExtension]:
    _configure_cuda_arch_list()
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
