#pragma once
#include "flash.h"
#include "flash_bwd_kernel.h"
#include "flash_bwd_preprocess_kernel.h"
#include "static_switch.h"
#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;

template<typename Kernel_traits, bool Is_causal>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    compute_dot_do_o<Kernel_traits, Is_causal>(params);
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN>
__global__ __launch_bounds__(256, 1)
void flash_bwd_dq_kernel(const Flash_bwd_params params) {
    compute_dq<Kernel_traits, Is_causal, Is_even_MN>(params);
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN>
__global__ __launch_bounds__(256, 1)
void flash_bwd_dk_dv_kernel(const Flash_bwd_params params) {
    compute_dk_dv<Kernel_traits, Is_causal, Is_even_MN>(params);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd(Flash_bwd_params &params) {
    normalize_bwd_params(params);

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const bool is_even_MN  = params.cu_seqlens_q == nullptr &&
                             params.cu_seqlens_k == nullptr &&
                             params.seqlen_q % kBlockM == 0 &&
                             params.seqlen_k % kBlockN == 0;
    cudaStream_t stream = params.stream;

    // 1. Preprocess: D = rowsum(dO \circ O)
    dim3 dimGrid_dot_do_o(params.b, params.h, (params.seqlen_q + 32 - 1) / 32);
    dim3 dimBlock_dot_do_o(1024);
    flash_bwd_dot_do_o_kernel<Kernel_traits, Is_causal><<<dimGrid_dot_do_o, dimBlock_dot_do_o, 0, stream>>>(params);

    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    const int slot0_elems = (kHeadDim >= 128)
        ? (kBlockM * kHeadDim)
        : ((kBlockM + kBlockN) * kHeadDim);
    const int slot1_pds = 2 * kBlockM * kBlockN;
    const int slot1_elems = (kHeadDim >= 128)
        ? ((kBlockM * kHeadDim > slot1_pds) ? (kBlockM * kHeadDim) : slot1_pds)
        : (kBlockM * kHeadDim + ((kBlockN * kHeadDim > slot1_pds) ? (kBlockN * kHeadDim) : slot1_pds));
    const int maxbytes = (slot0_elems + slot1_elems) * (int)sizeof(half_t);

    // 2. Compute dQ across query heads (params.h)
    const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;
    dim3 dimGrid_dq(num_m_block, params.b, params.h);
    dim3 dimBlock_dq(256);

    BOOL_SWITCH(is_even_MN, Is_even_MN, [&] {
        auto dq_kernel = flash_bwd_dq_kernel<Kernel_traits, Is_causal, Is_even_MN>;
        if (maxbytes > 49152) {
            cudaFuncSetAttribute(dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        }
        dq_kernel<<<dimGrid_dq, dimBlock_dq, maxbytes, stream>>>(params);
    });

    // 3. Compute dK, dV across KV heads (params.h_k) with in-kernel GQA reduction over h_h_k_ratio
    const int num_n_block = (params.seqlen_k + kBlockN - 1) / kBlockN;
    dim3 dimGrid_dk_dv(num_n_block, params.b, params.h_k);
    dim3 dimBlock_dk_dv(256);

    BOOL_SWITCH(is_even_MN, Is_even_MN, [&] {
        auto dk_dv_kernel = flash_bwd_dk_dv_kernel<Kernel_traits, Is_causal, Is_even_MN>;
        if (maxbytes > 49152) {
            cudaFuncSetAttribute(dk_dv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        }
        dk_dv_kernel<<<dimGrid_dk_dv, dimBlock_dk_dv, maxbytes, stream>>>(params);
    });
}

template<bool Is_causal>
void run_mha_bwd_hdim64(Flash_bwd_params &params) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;

    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<bool Is_causal>
void run_mha_bwd_hdim96(Flash_bwd_params &params) {
    constexpr static int Headdim = 96;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;

    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<bool Is_causal>
void run_mha_bwd_hdim128(Flash_bwd_params &params) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;

    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<bool Is_causal>
void run_mha_bwd_hdim32(Flash_bwd_params &params) {
    constexpr static int Headdim = 32;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;

    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<bool Is_causal>
void run_mha_bwd_hdim256(Flash_bwd_params &params) {
    constexpr static int Headdim = 256;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;

    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}
