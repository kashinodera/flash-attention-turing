#pragma once

#include <cuda_runtime.h>

#include "flash.h"
#include "flash_fwd_kvcache_kernel.h"

template <int kHeadDim>
void run_mha_fwd_kvcache_hdim(Flash_fwd_params& params) {
    constexpr int kBlockN = 32;
    constexpr int kNWarps = 8;
    constexpr int kThreads = 256;
    constexpr int kMaxGroupM = 8;

    if (params.h_h_k_ratio == 0 && params.h_k > 0) {
        params.h_h_k_ratio = params.h / params.h_k;
    }
    if (params.scale_softmax_log2 == 0.0f && params.softmax_scale != 0.0f) {
        params.scale_softmax_log2 = params.softmax_scale * float(M_LOG2E);
    }

    const int batch = params.b;
    const int heads = params.h;
    const int heads_k = params.h_k;
    const int seqlen_q = params.seqlen_q;
    const int splits = params.num_splits;

    const int total_m_per_kv = seqlen_q * params.h_h_k_ratio;
    const int groups_per_kv = (total_m_per_kv + kMaxGroupM - 1) / kMaxGroupM;
    const int total_groups = heads_k * groups_per_kv;

    cudaStream_t stream = params.stream;

    const int smem_q = kMaxGroupM * kHeadDim * static_cast<int>(sizeof(half_t));
    const int smem_kv = 2 * kBlockN * kHeadDim * static_cast<int>(sizeof(half_t));
    const int smem_scores = kMaxGroupM * kBlockN * static_cast<int>(sizeof(float));
    const int smem_red = kMaxGroupM * 4 * static_cast<int>(sizeof(float));
    const int smem_total = smem_q + smem_kv + smem_scores + smem_red;

    auto kernel = flash_fwd_kvcache_kernel<kHeadDim, kBlockN, kNWarps, kMaxGroupM>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_total);

    dim3 grid(splits, batch, total_groups);
    kernel<<<grid, kThreads, smem_total, stream>>>(params);

    if (splits > 1 && params.o_partials != nullptr) {
        auto combine = flash_fwd_kvcache_combine_kernel<kHeadDim, kBlockN>;
        dim3 cgrid(batch, heads, seqlen_q);
        combine<<<cgrid, kThreads, 0, stream>>>(params);
    }
}
