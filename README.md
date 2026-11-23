# flash_attn_turing

Fork of [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) implementing [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) kernels for NVIDIA Turing (`sm_75`), compiled and configured for the NVIDIA Tesla T4 (40 SMs, 64 KB shared memory per SM, `float16`).

## Benchmarks

Benchmark scripts (`benchmark.sh`, `profile.sh`, `utils/plot_kernels.py`) and plots (`utils/*.png`) in this repository are inherited unmodified from upstream ([ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing)). This project does not maintain or update benchmark figures and makes no performance claims.

## Implementation Scope

### Target & Types
- **Architecture:** `-arch=sm_75` (Turing Tensor Cores via `SM75_16x8x8_F32F16F16F32_TN` / `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`)
- **Input / Output dtype:** `torch.float16` (`cutlass::half_t`)
- **Accumulator / LSE dtype:** `torch.float32` (`float`)
- **Supported `head_dim`:** `32`, `64`, `96`, `128`, `256` (or `128` only when built with `FA_PROFILE=lean`)
- **Attention layouts:** MHA (`nheads_q == nheads_k`), GQA (`nheads_q % nheads_k == 0`), MQA (`nheads_k == 1`)

### Supported Operations
- **Forward + Backward (autograd):**
  - `flash_attn_func(q, k, v, ...)`
  - `flash_attn_qkvpacked_func(qkv, ...)`
  - `flash_attn_kvpacked_func(q, kv, ...)`
  - `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)`
  - `flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, ...)`
  - `flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)`
- **KV-Cache Forward (inference only, no autograd):**
  - `flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, ..., cache_seqlens=..., ...)`

### Supported Modifiers
- `causal: bool = False` (bottom-right aligned causal mask: key index `<= query_index + seqlen_k - seqlen_q`; when `causal=True`, `window_size_right` is forced to `0`)
- `window_size: Tuple[int, int] = (-1, -1)` (sliding-window local attention `(left, right)`; `-1` means infinite)
- `softcap: float = 0.0` (logit softcapping: `softcap * tanh(score * softmax_scale / softcap)` when `softcap > 0.0`)
- `alibi_slopes: Optional[torch.Tensor] = None` (CUDA `torch.float32`, contiguous last dim, shape `(nheads,)` or `(batch_size, nheads)`)
- `softmax_scale: Optional[float] = None` (defaults to `head_dim ** -0.5`)

### Unsupported Features (raises `ValueError` or `RuntimeError`)
- `bfloat16`, `float32`, or `float8` Q/K/V inputs
- `head_dim` values other than `{32, 64, 96, 128, 256}`
- `dropout_p != 0.0`
- `return_attn_probs=True`
- `block_table` (paged KV cache)
- `rotary_cos`, `rotary_sin` (fused RoPE in `flash_attn_with_kvcache`)
- `cache_batch_idx`, `cache_leftpad` (in `flash_attn_with_kvcache`)
- Backward pass when compiled with `FA_PROFILE=fwd_all` or `FA_PROFILE=lean` (`FA_INFERENCE_ONLY`)

## Python & C++ API Reference

Importing:
```python
# Top-level Python module (autograd wrappers + validation):
import flash_attention_interface
from flash_attention_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

# Low-level pybind11 C++ extension module:
import flash_attn_turing  # exposes: fwd, bwd, varlen_fwd, varlen_bwd, fwd_kvcache
```
> Note: The package installs `flash_attention_interface` (Python module) and `flash_attn_turing` (compiled extension), not the `flash_attn` namespace.

### Function Signatures & Tensor Contracts

#### 1. `flash_attn_func`
```python
def flash_attn_func(
    q: torch.Tensor,                              # (batch_size, seqlen_q, nheads, head_dim), float16, CUDA
    k: torch.Tensor,                              # (batch_size, seqlen_k, nheads_k, head_dim), float16, CUDA
    v: torch.Tensor,                              # (batch_size, seqlen_k, nheads_k, head_dim), float16, CUDA
    dropout_p: float = 0.0,                       # must be 0.0
    softmax_scale: Optional[float] = None,        # default: head_dim ** -0.5
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,  # (nheads,) or (batch_size, nheads), float32, CUDA
    deterministic: bool = False,                  # accepted as no-op (backward kernels do not use atomics)
    return_attn_probs: bool = False,              # must be False
) -> torch.Tensor:                                # returns out: (batch_size, seqlen_q, nheads, head_dim), float16
```

#### 2. `flash_attn_qkvpacked_func`
```python
def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,                            # (batch_size, seqlen, 3, nheads, head_dim), float16, CUDA
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:                                # returns out: (batch_size, seqlen, nheads, head_dim), float16
```

#### 3. `flash_attn_kvpacked_func`
```python
def flash_attn_kvpacked_func(
    q: torch.Tensor,                              # (batch_size, seqlen_q, nheads, head_dim), float16, CUDA
    kv: torch.Tensor,                             # (batch_size, seqlen_k, 2, nheads_k, head_dim), float16, CUDA
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:                                # returns out: (batch_size, seqlen_q, nheads, head_dim), float16
```

#### 4. `flash_attn_varlen_func`
```python
def flash_attn_varlen_func(
    q: torch.Tensor,                              # (total_q, nheads, head_dim), float16, CUDA
    k: torch.Tensor,                              # (total_k, nheads_k, head_dim), float16, CUDA
    v: torch.Tensor,                              # (total_k, nheads_k, head_dim), float16, CUDA
    cu_seqlens_q: torch.Tensor,                   # (batch_size + 1,), int32, contiguous, CUDA
    cu_seqlens_k: torch.Tensor,                   # (batch_size + 1,), int32, contiguous, CUDA
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,   # must be None
) -> torch.Tensor:                                # returns out: (total_q, nheads, head_dim), float16
```

#### 5. `flash_attn_varlen_qkvpacked_func`
```python
def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,                            # (total_tokens, 3, nheads, head_dim), float16, CUDA
    cu_seqlens: torch.Tensor,                     # (batch_size + 1,), int32, contiguous, CUDA
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:                                # returns out: (total_tokens, nheads, head_dim), float16
```

#### 6. `flash_attn_varlen_kvpacked_func`
```python
def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,                              # (total_q, nheads, head_dim), float16, CUDA
    kv: torch.Tensor,                             # (total_k, 2, nheads_k, head_dim), float16, CUDA
    cu_seqlens_q: torch.Tensor,                   # (batch_size + 1,), int32, contiguous, CUDA
    cu_seqlens_k: torch.Tensor,                   # (batch_size + 1,), int32, contiguous, CUDA
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:                                # returns out: (total_q, nheads, head_dim), float16
```

#### 7. `flash_attn_with_kvcache`
```python
def flash_attn_with_kvcache(
    q: torch.Tensor,                              # (batch_size, seqlen_q, nheads, head_dim), float16, CUDA, seqlen_q >= 1
    k_cache: torch.Tensor,                        # (batch_size, cache_cap, nheads_k, head_dim), float16, CUDA
    v_cache: torch.Tensor,                        # (batch_size, cache_cap, nheads_k, head_dim), float16, CUDA
    k: Optional[torch.Tensor] = None,             # optional new keys: (batch_size, seqlen_q, nheads_k, head_dim), float16, CUDA
    v: Optional[torch.Tensor] = None,             # optional new values: (batch_size, seqlen_q, nheads_k, head_dim), float16, CUDA
    rotary_cos: Optional[torch.Tensor] = None,    # must be None
    rotary_sin: Optional[torch.Tensor] = None,    # must be None
    cache_seqlens: Union[int, torch.Tensor] = None, # required: int or (batch_size,) int32 CUDA tensor
    cache_batch_idx: Optional[torch.Tensor] = None, # must be None
    cache_leftpad: Optional[torch.Tensor] = None,   # must be None
    block_table: Optional[torch.Tensor] = None,     # must be None
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,              # ignored (rotary not supported)
    alibi_slopes: Optional[torch.Tensor] = None,
    num_splits: int = 0,                          # 0 = automatic heuristic; >= 1 = explicit split-KV count
    return_softmax_lse: bool = False,
    out: Optional[torch.Tensor] = None,           # optional preallocated output buffer matching q shape/dtype
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # returns out, or (out, softmax_lse) when return_softmax_lse=True where softmax_lse is (batch_size, nheads, seqlen_q) float32
```

### Behavioral Specifications & Constraints
1. **Strides & Contiguity:**
   - All forward and backward kernels require `stride(-1) == 1` (contiguous `head_dim`). The Python wrapper calls `.contiguous()` automatically if `x.stride(-1) != 1`.
   - Non-contiguous outer dimensions (`batch_stride`, `row_stride`, `head_stride`) are supported directly by the kernels (including `unbind` views from packed QKV/KV tensors).
   - `flash_attn_with_kvcache` additionally requires `q`, `k_cache`, `v_cache` (and `k`, `v` if provided) to have 16-byte aligned base data pointers (`data_ptr() % 16 == 0`) and `stride(0)`, `stride(1)`, `stride(2)` each divisible by 8 elements (for 128-bit `float4` vectorized loads/stores).
2. **In-Place KV-Cache Mutation (`flash_attn_with_kvcache`):**
   - When `k` and `v` are provided (`k is not None and v is not None`), the kernel writes `k` and `v` in-place into `k_cache[b, cache_len : cache_len + seqlen_q]` and `v_cache[b, cache_len : cache_len + seqlen_q]` and attends over `0 : cache_len + seqlen_q`.
   - Note: `cache_seqlens` itself is **not** incremented in-place by the kernel; the caller owns updating `cache_seqlens`.
   - When `k is None and v is None` (attend-only mode), `k_cache` and `v_cache` are read-only over `0 : cache_len`.
   - If `cache_seqlens[b] + new_tokens > cache_cap`, `cache_len` is clamped on-device to `max(0, cache_cap - new_tokens)`.
3. **CUDA Stream & CUDA Graphs:**
   - Kernels execute on `at::cuda::getCurrentCUDAStream()`.
   - When `cache_seqlens` is passed as a device `torch.int32` tensor and `num_splits >= 1` is fixed (or `cache_seqlens` is a Python `int`), `flash_attn_with_kvcache` performs no host-device synchronization and is compatible with `torch.cuda.CUDAGraph` capture.
4. **Fully Masked Rows:**
   - If all keys for a query row are masked out (or `seqlen_k == 0`), the output row is written as `0.0` and `softmax_lse` is `-inf`.
5. **Determinism:**
   - `deterministic: bool` is accepted for API compatibility. Backward `dQ` (`compute_dq_1rowblock`) and `dK/dV` (`compute_dk_dv_1colblock` with in-kernel GQA reduction across `h_h_k_ratio`) do not use global atomic adds and are deterministic by construction.

## Build Configuration

```bash
pip install torch setuptools ninja wheel
pip install -v --no-build-isolation .
```

### Environment Variables (`setup.py`)
- `FA_PROFILE`:
  - `full` (default): Compiles forward, KV-cache forward, and backward kernels for `head_dim` in `{32, 64, 96, 128, 256}` (26 translation units).
  - `fwd_all`: Compiles forward and KV-cache forward kernels only (`-DFA_INFERENCE_ONLY`) for `head_dim` in `{32, 64, 96, 128, 256}` (16 translation units).
  - `lean`: Compiles forward and KV-cache forward kernels only (`-DFA_INFERENCE_ONLY -DFA_ONLY_HDIM128`) for `head_dim=128` (4 translation units).
- `FA_INFERENCE_ONLY=1`: Alias for `FA_PROFILE=fwd_all` (or `FA_PROFILE=lean` if `FA_HDIMS=128` is also set).
- `FA_LOCAL_VERSION`: Overrides the PEP 440 local version tag (`0.5.0+<FA_LOCAL_VERSION>`). Set to `""` or `"none"` to omit the local version tag.
- `FA_TAG_LOCAL_VERSION`: Defaults to `"1"`, tagging the wheel version as `0.5.0+cu<cuda>torch<major>.<minor>`.
- `TORCH_CXX11_ABI`: Optional override (`"0"` or `"1"`) for `_GLIBCXX_USE_CXX11_ABI`; defaults to matching the installed `torch._C._GLIBCXX_USE_CXX11_ABI`.

## Repository Layout

- `setup.py`: Build script and profile selector (`full`, `fwd_all`, `lean`).
- `flash_attention_interface.py`: Python autograd functions and input validation.
- `csrc/flash_attn/flash_api.cpp`: PyBind11 bindings (`fwd`, `bwd`, `varlen_fwd`, `varlen_bwd`, `fwd_kvcache`), tensor checks, and split-KV heuristic.
- `csrc/flash_attn/src/flash.h`: `Qkv_params`, `Flash_fwd_params`, `Flash_bwd_params` structs and stride normalization.
- `csrc/flash_attn/src/kernel_traits.h`: CuTe `SM75` MMA and `LDSM` swizzled shared-memory layout traits.
- `csrc/flash_attn/src/flash_fwd_kernel.h`, `flash_fwd_launch_template.h`: Forward kernel and launch configuration (`kShareQK` 64 KB SMEM aliasing at `head_dim=256`).
- `csrc/flash_attn/src/flash_bwd_kernel.h`, `flash_bwd_preprocess_kernel.h`, `flash_bwd_launch_template.h`: Backward preprocess (`dO * O`), `dQ`, and `dK/dV` kernels (`kShareBwdSmem` two-slot 64 KB SMEM layout at `head_dim >= 128`).
- `csrc/flash_attn/src/flash_fwd_kvcache_kernel.h`, `flash_fwd_kvcache_launch.h`: Head-grouped (`kMaxGroupM=8`) KV-cache decode/append kernel and split-KV combine kernel.
- `csrc/flash_attn/src/mask.h`: Forward and backward causal, local sliding-window, and ALiBi mask application.
- `csrc/cutlass/include/`: In-tree vendored CUTLASS / CuTe headers (no git submodules required).
- `test_flash_attn.py`: Pytest regression suite against FP32 / PyTorch SDPA references.

## Tests

```bash
pip install pytest numpy pandas
pytest -q test_flash_attn.py
```
