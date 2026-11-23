#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    half_t *__restrict__ q_ptr;
    half_t *__restrict__ k_ptr;
    half_t *__restrict__ v_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_k;
    int h_h_k_ratio;

    // Optional varlen cumulative sequence lengths (device pointers, int32).
    int *__restrict__ cu_seqlens_q;
    int *__restrict__ cu_seqlens_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    half_t * __restrict__ o_ptr;

    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the softmax sum.
    float * __restrict__ l_ptr;
    float softmax_scale;
    float scale_softmax_log2;
    int b, seqlen_q, seqlen_k, d;
    bool is_causal;

    // Dense KV-cache decode / append.
    half_t * __restrict__ k_cache_ptr = nullptr;
    half_t * __restrict__ v_cache_ptr = nullptr;
    half_t * __restrict__ k_new_ptr = nullptr;
    half_t * __restrict__ v_new_ptr = nullptr;
    int * __restrict__ cache_seqlens = nullptr;

    index_t k_cache_batch_stride = 0;
    index_t k_cache_seq_stride = 0;
    index_t k_cache_head_stride = 0;
    index_t v_cache_batch_stride = 0;
    index_t v_cache_seq_stride = 0;
    index_t v_cache_head_stride = 0;

    index_t k_new_batch_stride = 0;
    index_t k_new_seq_stride = 0;
    index_t k_new_head_stride = 0;
    index_t v_new_batch_stride = 0;
    index_t v_new_seq_stride = 0;
    index_t v_new_head_stride = 0;

    int num_splits = 1;
    // Partial results for split-KV combine: [b][h][split][d] / [b][h][split]
    float * __restrict__ o_partials = nullptr;
    float * __restrict__ lse_partials = nullptr;
    float softcap = 0.0f;
    int window_size_left = -1;
    int window_size_right = -1;
    const float * __restrict__ alibi_slopes_ptr = nullptr;
    index_t alibi_slopes_batch_stride = 0;
    cudaStream_t stream = nullptr;
};


struct Flash_bwd_params : public Flash_fwd_params {

    half_t *__restrict__ do_ptr;
    half_t *__restrict__ dq_ptr;
    half_t *__restrict__ dk_ptr;
    half_t *__restrict__ dv_ptr;
    float *__restrict__ do_o_ptr;

    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;
};

inline void normalize_fwd_params(Flash_fwd_params &params) {
    if (params.h_h_k_ratio == 0 && params.h_k > 0) {
        params.h_h_k_ratio = params.h / params.h_k;
    }
    if (params.scale_softmax_log2 == 0.0f && params.softmax_scale != 0.0f) {
        params.scale_softmax_log2 = params.softmax_scale * float(M_LOG2E);
    }
    if (params.q_head_stride == 0) params.q_head_stride = params.d;
    if (params.k_head_stride == 0) params.k_head_stride = params.d;
    if (params.v_head_stride == 0) params.v_head_stride = params.d;
    if (params.o_head_stride == 0) params.o_head_stride = params.d;

    if (params.q_row_stride == 0) params.q_row_stride = params.h * params.d;
    if (params.k_row_stride == 0) params.k_row_stride = params.h_k * params.d;
    if (params.v_row_stride == 0) params.v_row_stride = params.h_k * params.d;
    if (params.o_row_stride == 0) params.o_row_stride = params.h * params.d;

    if (params.cu_seqlens_q == nullptr) {
        if (params.q_batch_stride == 0) params.q_batch_stride = params.seqlen_q * params.q_row_stride;
        if (params.o_batch_stride == 0) params.o_batch_stride = params.seqlen_q * params.o_row_stride;
    }
    if (params.cu_seqlens_k == nullptr) {
        if (params.k_batch_stride == 0) params.k_batch_stride = params.seqlen_k * params.k_row_stride;
        if (params.v_batch_stride == 0) params.v_batch_stride = params.seqlen_k * params.v_row_stride;
    }
}

inline void normalize_bwd_params(Flash_bwd_params &params) {
    normalize_fwd_params(params);
    if (params.do_head_stride == 0) params.do_head_stride = params.d;
    if (params.dq_head_stride == 0) params.dq_head_stride = params.d;
    if (params.dk_head_stride == 0) params.dk_head_stride = params.d;
    if (params.dv_head_stride == 0) params.dv_head_stride = params.d;

    if (params.do_row_stride == 0) params.do_row_stride = params.h * params.d;
    if (params.dq_row_stride == 0) params.dq_row_stride = params.h * params.d;
    if (params.dk_row_stride == 0) params.dk_row_stride = params.h_k * params.d;
    if (params.dv_row_stride == 0) params.dv_row_stride = params.h_k * params.d;

    if (params.cu_seqlens_q == nullptr) {
        if (params.do_batch_stride == 0) params.do_batch_stride = params.seqlen_q * params.do_row_stride;
        if (params.dq_batch_stride == 0) params.dq_batch_stride = params.seqlen_q * params.dq_row_stride;
    }
    if (params.cu_seqlens_k == nullptr) {
        if (params.dk_batch_stride == 0) params.dk_batch_stride = params.seqlen_k * params.dk_row_stride;
        if (params.dv_batch_stride == 0) params.dv_batch_stride = params.seqlen_k * params.dv_row_stride;
    }
}

template<int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params);

template<int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params);

template<int Headdim> void run_mha_fwd_kvcache_(Flash_fwd_params &params);
