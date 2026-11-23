#pragma once
#include "flash.h"
#include "flash_fwd_kernel.h"
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include "static_switch.h"
using half_t = cutlass::half_t;

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
__global__ __launch_bounds__(256)
void flash_fwd_kernel(const Flash_fwd_params params)
{
    compute_attn<Kernel_traits, Is_causal, Is_even_MN>(params);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params) {
    normalize_fwd_params(params);

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;

    const bool is_even_MN  = params.cu_seqlens_q == nullptr &&
                             params.cu_seqlens_k == nullptr &&
                             params.seqlen_q % kBlockM == 0 &&
                             params.seqlen_k % kBlockN == 0;

    dim3 dimGrid(num_m_block, params.b, params.h);
    dim3 dimBlock(256);
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    int maxbytes = (kHeadDim == 256)
        ? (kBlockM * kHeadDim * (int)sizeof(half_t))
        : ((kBlockM + kBlockN) * kHeadDim * (int)sizeof(half_t));
    cudaStream_t stream = params.stream;

    BOOL_SWITCH(is_even_MN, Is_even_MN, [&] {
        auto kernel = flash_fwd_kernel<Kernel_traits, Is_causal, Is_even_MN>;
        if (maxbytes > 49152) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        }
        kernel<<<dimGrid, dimBlock, maxbytes, stream>>>(params);
    });
}

template<bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<bool Is_causal>
void run_mha_fwd_hdim96(Flash_fwd_params &params) {
    constexpr static int Headdim = 96;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params) {
    constexpr static int Headdim = 32;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<bool Is_causal>
void run_mha_fwd_hdim256(Flash_fwd_params &params) {
    constexpr static int Headdim = 256;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}
