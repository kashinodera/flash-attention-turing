"""Python interface for FlashAttention Turing extension."""

from typing import Optional, Tuple

import torch
import flash_attn_turing as flash_attn_gpu


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_fa2_common_args(
    dropout_p: float,
    window_size: Tuple[int, int],
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_attn_probs: bool,
) -> None:
    if dropout_p != 0.0:
        raise ValueError("dropout_p > 0.0 is not supported by fa2-turing")
    if len(window_size) != 2:
        raise ValueError("window_size must be a 2-tuple (left, right)")
    if softcap < 0.0:
        raise ValueError("softcap must be >= 0.0")
    if alibi_slopes is not None:
        if not alibi_slopes.is_cuda or alibi_slopes.dtype != torch.float32:
            raise ValueError("alibi_slopes must be a CUDA float32 tensor")
    if return_attn_probs:
        raise ValueError("return_attn_probs=True is not supported by fa2-turing")


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    alibi_slopes = maybe_contiguous(alibi_slopes)
    out, lse = flash_attn_gpu.fwd(
        q,
        k,
        v,
        softmax_scale,
        causal,
        int(window_size[0]),
        int(window_size[1]),
        float(softcap),
        alibi_slopes,
    )
    return out, lse


def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    alibi_slopes = maybe_contiguous(alibi_slopes)
    lse = lse.contiguous() if not lse.is_contiguous() else lse
    dq, dk, dv = flash_attn_gpu.bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        causal,
        dq,
        dk,
        dv,
        int(window_size[0]),
        int(window_size[1]),
        float(softcap),
        alibi_slopes,
    )
    return dq, dk, dv


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    alibi_slopes = maybe_contiguous(alibi_slopes)
    out, lse = flash_attn_gpu.varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        int(window_size[0]),
        int(window_size[1]),
        float(softcap),
        alibi_slopes,
    )
    return out, lse


def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    alibi_slopes = maybe_contiguous(alibi_slopes)
    lse = lse.contiguous() if not lse.is_contiguous() else lse
    dq, dk, dv = flash_attn_gpu.varlen_bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        dq,
        dk,
        dv,
        int(window_size[0]),
        int(window_size[1]),
        float(softcap),
        alibi_slopes,
    )
    return dq, dk, dv


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal, window_size, softcap, alibi_slopes)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, alibi_slopes)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, alibi_slopes = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(
            dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal,
            window_size=ctx.window_size, softcap=ctx.softcap, alibi_slopes=alibi_slopes,
        )
        return dq, dk, dv, None, None, None, None, None, None


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        is_grad_enabled: bool,
    ):
        qkv = maybe_contiguous(qkv)
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal, window_size, softcap, alibi_slopes)
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(qkv, out, lse, alibi_slopes)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        qkv, out, lse, alibi_slopes = ctx.saved_tensors
        q, k, v = qkv.unbind(dim=2)
        dqkv = torch.empty_like(qkv)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            dq=dqkv[:, :, 0],
            dk=dqkv[:, :, 1],
            dv=dqkv[:, :, 2],
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=alibi_slopes,
        )
        return dqkv, None, None, None, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        is_grad_enabled: bool,
    ):
        q = maybe_contiguous(q)
        kv = maybe_contiguous(kv)
        k, v = kv.unbind(dim=2)
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal, window_size, softcap, alibi_slopes)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, kv, out, lse, alibi_slopes)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, kv, out, lse, alibi_slopes = ctx.saved_tensors
        k, v = kv.unbind(dim=2)
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            dq=dq,
            dk=dkv[:, :, 0],
            dv=dkv[:, :, 1],
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=alibi_slopes,
        )
        return dq, dkv, None, None, None, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, alibi_slopes)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, alibi_slopes = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=alibi_slopes,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        is_grad_enabled: bool,
    ):
        qkv = maybe_contiguous(qkv)
        q, k, v = qkv.unbind(dim=1)
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
        )
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(qkv, out, lse, cu_seqlens, alibi_slopes)
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        qkv, out, lse, cu_seqlens, alibi_slopes = ctx.saved_tensors
        q, k, v = qkv.unbind(dim=1)
        dqkv = torch.empty_like(qkv)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.softmax_scale,
            ctx.causal,
            dq=dqkv[:, 0],
            dk=dqkv[:, 1],
            dv=dqkv[:, 2],
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=alibi_slopes,
        )
        return dqkv, None, None, None, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        alibi_slopes: Optional[torch.Tensor],
        is_grad_enabled: bool,
    ):
        q = maybe_contiguous(q)
        kv = maybe_contiguous(kv)
        k, v = kv.unbind(dim=1)
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, kv, out, lse, cu_seqlens_q, cu_seqlens_k, alibi_slopes)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, kv, out, lse, cu_seqlens_q, cu_seqlens_k, alibi_slopes = ctx.saved_tensors
        k, v = kv.unbind(dim=1)
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
            dq=dq,
            dk=dkv[:, 0],
            dv=dkv[:, 1],
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=alibi_slopes,
        )
        return dq, dkv, None, None, None, None, None, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    _validate_fa2_common_args(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs)
    _ = deterministic
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        tuple(window_size),
        float(softcap),
        alibi_slopes,
        torch.is_grad_enabled(),
    )


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    _validate_fa2_common_args(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs)
    _ = deterministic
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        tuple(window_size),
        float(softcap),
        alibi_slopes,
        torch.is_grad_enabled(),
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    _validate_fa2_common_args(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs)
    _ = deterministic
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        softmax_scale,
        causal,
        tuple(window_size),
        float(softcap),
        alibi_slopes,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
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
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if block_table is not None:
        raise ValueError("paged KV cache (block_table) is not supported by fa2-turing")
    _validate_fa2_common_args(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs)
    _ = deterministic
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        tuple(window_size),
        float(softcap),
        alibi_slopes,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    _validate_fa2_common_args(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs)
    _ = deterministic
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        softmax_scale,
        causal,
        tuple(window_size),
        float(softcap),
        alibi_slopes,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
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
) -> torch.Tensor:
    _validate_fa2_common_args(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs)
    _ = deterministic
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        tuple(window_size),
        float(softcap),
        alibi_slopes,
        torch.is_grad_enabled(),
    )


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens=None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
    out: Optional[torch.Tensor] = None,
):
    if rotary_cos is not None or rotary_sin is not None:
        raise ValueError("rotary embedding is not supported by this port")
    if cache_batch_idx is not None or cache_leftpad is not None:
        raise ValueError("cache_batch_idx / cache_leftpad are not supported by this port")
    if block_table is not None:
        raise ValueError("paged KV cache is not supported by this port")
    if len(window_size) != 2:
        raise ValueError("window_size must be a 2-tuple (left, right)")
    if alibi_slopes is not None:
        if not alibi_slopes.is_cuda or alibi_slopes.dtype != torch.float32:
            raise ValueError("alibi_slopes must be a CUDA float32 tensor")
        alibi_slopes = maybe_contiguous(alibi_slopes)
    if (k is None) ^ (v is None):
        raise ValueError("k and v must both be None or both be passed")
    has_new = k is not None

    if cache_seqlens is None:
        raise ValueError("cache_seqlens must be passed")

    if q.dim() != 4 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("q/k_cache/v_cache must be rank-4")
    if has_new and (k.dim() != 4 or v.dim() != 4):
        raise ValueError("k/v must be rank-4")
    if q.shape[1] < 1:
        raise ValueError("flash_attn_with_kvcache requires seqlen_q >= 1")
    batch, seqlen_q, nheads, head_dim = q.shape
    if head_dim not in (32, 64, 96, 128, 256):
        raise ValueError(f"unsupported head_dim {head_dim}")

    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q = maybe_contiguous(q)
    if has_new:
        k = maybe_contiguous(k)
        v = maybe_contiguous(v)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if num_splits < 0:
        raise ValueError("num_splits must be >= 0")
    active_seqlen_k = 0
    if isinstance(cache_seqlens, int):
        active_seqlen_k = max(1, cache_seqlens + (seqlen_q if has_new else 0))
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
    cache_seqlens = maybe_contiguous(cache_seqlens)

    if not k_cache.is_cuda or not v_cache.is_cuda:
        raise ValueError("k_cache/v_cache must be CUDA tensors")
    if cache_seqlens.dtype != torch.int32 or not cache_seqlens.is_cuda:
        raise ValueError("cache_seqlens must be a CUDA int32 tensor")
    if cache_seqlens.numel() != batch:
        raise ValueError("cache_seqlens must have one entry per batch row")
    if k_cache.shape[0] != batch or v_cache.shape[0] != batch:
        raise ValueError("cache batch size must match q")
    if k_cache.shape[1] != v_cache.shape[1]:
        raise ValueError("k_cache/v_cache capacity must match")
    if k_cache.shape[3] != head_dim or v_cache.shape[3] != head_dim:
        raise ValueError("cache head_dim must match q")
    if k_cache.shape[2] != v_cache.shape[2]:
        raise ValueError("k_cache and v_cache head count must match")
    nheads_k = k_cache.shape[2]
    if nheads % nheads_k != 0:
        raise ValueError("nheads must be divisible by nheads_k (GQA)")

    if has_new:
        if k.shape[0] != batch or v.shape[0] != batch or k.shape[1] != seqlen_q or v.shape[1] != seqlen_q:
            raise ValueError("k/v must be (batch, seqlen_q, nheads_k, headdim)")
        if k.shape[2] != v.shape[2] or k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        if k.shape[2] != nheads_k:
            raise ValueError("k/v head count must match k_cache")
        if k.dtype != torch.float16 or v.dtype != torch.float16:
            raise ValueError("k/v must be float16")

    if q.dtype != torch.float16:
        raise ValueError("q must be float16")
    if k_cache.dtype != torch.float16 or v_cache.dtype != torch.float16:
        raise ValueError("k_cache/v_cache must be float16")
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("k_cache/v_cache must have contiguous last dimension")

    out, softmax_lse = flash_attn_gpu.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        int(window_size[0]),
        int(window_size[1]),
        float(softcap),
        bool(rotary_interleaved),
        int(num_splits),
        int(active_seqlen_k),
    )
    return (out, softmax_lse) if return_softmax_lse else out
