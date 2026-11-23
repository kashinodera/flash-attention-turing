#!/usr/bin/env python3
"""fa2-turing: FlashAttention-2 for NVIDIA Turing (sm_75).

Build profiles:
  FA_PROFILE=full     (default) Forward + Backward, all head dims (full training/eval support)
  FA_PROFILE=fwd_all  Forward-only, all head dims (32, 64, 96, 128, 256)
  FA_PROFILE=lean     Forward-only, head_dim=128 only (fastest compile, ~30s, ~2MB)
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT = Path(__file__).resolve().parent

# Locate csrc (local or shared flash-attention-turing directory)
if (ROOT / "csrc").is_dir():
    CSRC = ROOT / "csrc"
elif (ROOT.parent / "flash-attention-turing" / "csrc").is_dir():
    CSRC = ROOT.parent / "flash-attention-turing" / "csrc"
else:
    CSRC = ROOT / "csrc"

# Determine build profile
PROFILE = os.environ.get("FA_PROFILE", "full").lower()
if os.environ.get("FA_INFERENCE_ONLY") == "1" and os.environ.get("FA_HDIMS") == "128":
    PROFILE = "lean"
elif os.environ.get("FA_INFERENCE_ONLY") == "1":
    PROFILE = "fwd_all"

nvcc_flags = [
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-arch=sm_75",
    "-O3",
    "--use_fast_math",
    "--ptxas-options=-v",
    "-lineinfo",
    "-t", "2",
]

cxx_flags = ["-std=c++17", "-O3"]
define_macros = []

# Match installed PyTorch C++11 ABI unless explicitly overridden via TORCH_CXX11_ABI
default_abi = "1" if getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True) else "0"
cxx11_abi = os.environ.get("TORCH_CXX11_ABI", default_abi)
if "TORCH_CXX11_ABI" in os.environ:
    import torch.utils.cpp_extension
    torch.utils.cpp_extension._get_glibcxx_abi_build_flags = lambda: [f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}"]
    nvcc_flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}")
    cxx_flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}")
    define_macros.append(("_GLIBCXX_USE_CXX11_ABI", cxx11_abi))

if PROFILE == "lean":
    define_macros.extend([("FA_INFERENCE_ONLY", None), ("FA_ONLY_HDIM128", None)])
    nvcc_flags.extend(["-DFA_INFERENCE_ONLY", "-DFA_ONLY_HDIM128"])
    cxx_flags.extend(["-DFA_INFERENCE_ONLY", "-DFA_ONLY_HDIM128"])
    source_names = [
        "flash_attn/flash_api.cpp",
        "flash_attn/src/flash_fwd_hdim128_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim128_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim128_fp16_sm75.cu",
    ]
elif PROFILE == "fwd_all":
    define_macros.append(("FA_INFERENCE_ONLY", None))
    nvcc_flags.append("-DFA_INFERENCE_ONLY")
    cxx_flags.append("-DFA_INFERENCE_ONLY")
    source_names = [
        "flash_attn/flash_api.cpp",
        "flash_attn/src/flash_fwd_hdim32_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim32_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_hdim64_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim64_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_hdim96_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim96_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_hdim128_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim128_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_hdim256_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim256_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim32_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim64_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim96_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim128_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim256_fp16_sm75.cu",
    ]
else:  # full
    source_names = [
        "flash_attn/flash_api.cpp",
        "flash_attn/src/flash_fwd_hdim32_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim32_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_hdim64_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim64_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_hdim96_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim96_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_hdim128_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim128_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_hdim256_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_hdim256_fp16_causal_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim32_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim64_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim96_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim128_fp16_sm75.cu",
        "flash_attn/src/flash_fwd_kvcache_hdim256_fp16_sm75.cu",
        "flash_attn/src/flash_bwd_hdim32_fp16_sm75.cu",
        "flash_attn/src/flash_bwd_hdim32_fp16_causal_sm75.cu",
        "flash_attn/src/flash_bwd_hdim64_fp16_sm75.cu",
        "flash_attn/src/flash_bwd_hdim64_fp16_causal_sm75.cu",
        "flash_attn/src/flash_bwd_hdim96_fp16_sm75.cu",
        "flash_attn/src/flash_bwd_hdim96_fp16_causal_sm75.cu",
        "flash_attn/src/flash_bwd_hdim128_fp16_sm75.cu",
        "flash_attn/src/flash_bwd_hdim128_fp16_causal_sm75.cu",
        "flash_attn/src/flash_bwd_hdim256_fp16_sm75.cu",
        "flash_attn/src/flash_bwd_hdim256_fp16_causal_sm75.cu",
    ]

sources = [str(CSRC / name) for name in source_names]

# Ensure cutlass headers exist
cutlass_include = CSRC / "cutlass" / "include"
if not cutlass_include.is_dir():
    raise FileNotFoundError(
        f"CUTLASS headers not found at {cutlass_include}. "
        "Expected in-tree headers under csrc/cutlass/include."
    )

include_dirs = [
    CSRC / "flash_attn",
    CSRC / "flash_attn" / "src",
    cutlass_include,
]

BASE_VERSION = "0.5.0"


def get_package_version():
    """Derive PEP 440 package version with optional CUDA/PyTorch local tag."""
    local_ver = os.environ.get("FA_LOCAL_VERSION")
    if local_ver is not None:
        if local_ver == "" or local_ver.lower() == "none":
            return BASE_VERSION
        return f"{BASE_VERSION}+{local_ver}"

    if os.environ.get("FA_TAG_LOCAL_VERSION", "1") == "1":
        parts = []
        cuda_ver = getattr(torch.version, "cuda", None)
        if cuda_ver:
            parts.append("cu" + cuda_ver.replace(".", ""))
        torch_ver = torch.__version__.split("+")[0]
        torch_parts = torch_ver.split(".")
        if len(torch_parts) >= 2:
            parts.append(f"torch{torch_parts[0]}.{torch_parts[1]}")
        else:
            parts.append(f"torch{torch_ver}")
        if parts:
            return f"{BASE_VERSION}+{''.join(parts)}"
    return BASE_VERSION


package_version = get_package_version()
print(f"[*] Building fa2-turing v{package_version} [Profile: {PROFILE}] with {len(sources)} source files.", flush=True)

setup(
    name="flash_attn_turing",
    version=package_version,
    description="FlashAttention-2 for NVIDIA Turing (sm_75)",
    py_modules=["flash_attention_interface"],
    ext_modules=[
        CUDAExtension(
            name="flash_attn_turing",
            sources=sources,
            include_dirs=[str(p) for p in include_dirs],
            define_macros=define_macros,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension},
)
