#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <float.h>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "kernel_traits.h"
#include "utils.h"
#include "block_info.h"
#include "mask.h"

using namespace cute;





template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_dq_1rowblock(
    const Flash_bwd_params &params,
    const int bidb,
    const int bidh,
    const int m_block
)
{
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const half_t * __restrict__ q_ptr = params.q_ptr;
    const half_t * __restrict__ k_ptr = params.k_ptr;
    const half_t * __restrict__ v_ptr = params.v_ptr;
    float * __restrict__ l_ptr = params.l_ptr;
    float * __restrict__ d_ptr = params.do_o_ptr;
    const half_t * __restrict__ do_ptr = params.do_ptr;
    half_t * __restrict__ dq_ptr = params.dq_ptr;
    const int batch_size = params.b;
    const int max_seqlen_q = params.seqlen_q;
    const int max_seqlen_k = params.seqlen_k;
    const int num_heads = params.h;
    const int num_heads_k = params.h_k;
    const int h_h_k_ratio = params.h_h_k_ratio;
    const int head_dim = params.d;
    const float softmax_scale = params.softmax_scale;
    const float scale_softmax_log2 = params.scale_softmax_log2;

    const BlockInfo binfo(params, bidb);
    const int seqlen_q = binfo.actual_seqlen_q;
    const int seqlen_k = binfo.actual_seqlen_k;

    if (m_block * kBlockM >= seqlen_q) {
        return;
    }

    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;

    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(params.q_row_stride, params.q_head_stride, Int<1>{}));

    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    Tensor mK = make_tensor(make_gmem_ptr(k_ptr + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(params.k_row_stride, params.k_head_stride, Int<1>{}));

    Tensor gK = local_tile(mK(_, bidh / h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    Tensor mV = make_tensor(make_gmem_ptr(v_ptr + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(params.v_row_stride, params.v_head_stride, Int<1>{}));

    Tensor gV = local_tile(mV(_, bidh / h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads, max_seqlen_q, Int<1>{}));

    Tensor gL = local_tile(mL(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(m_block));

    Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads, max_seqlen_q, Int<1>{}));

    Tensor gD = local_tile(mD(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(m_block));

    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr + binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)),
                             make_shape(seqlen_q, num_heads, head_dim),
                             make_stride(params.do_row_stride, params.do_head_stride, Int<1>{}));

    Tensor gdO = local_tile(mdO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    Tensor mdQ = make_tensor(make_gmem_ptr(dq_ptr + binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(params.dq_row_stride, params.dq_head_stride, Int<1>{}));

    Tensor gdQ = local_tile(mdQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));
    extern __shared__ char smem_[];


    constexpr bool kShareBwdSmem = (kHeadDim >= 128);
    // Slot 0 (32KB at D=256, 16KB at D=128): sQ -> sK/sKt -> sdQ
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});
    Tensor sQt = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQTransposed{});

    Tensor sK = make_tensor(sQ.data() + (kShareBwdSmem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sQ.data() + (kShareBwdSmem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKVTransposed{});

    // Slot 1 (32KB at D=256): sdO (aliases sQ+size(sQ) when kShareBwdSmem) -> sV -> sP + sdS
    Tensor sdO = make_tensor(sQ.data() + (kShareBwdSmem ? size(sQ) : (size(sQ) + size(sK))), typename Kernel_traits::SmemLayoutQ{});
    Tensor sdOt = make_tensor(sQ.data() + (kShareBwdSmem ? size(sQ) : (size(sQ) + size(sK))), typename Kernel_traits::SmemLayoutQTransposed{});

    Tensor sV = make_tensor(sdO.data() + (kShareBwdSmem ? 0 : size(sdO)), typename Kernel_traits::SmemLayoutKV{});

    Tensor sP = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutPdSTransposed{});

    Tensor sdS = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdSTransposed{});

    Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});


    //int thread_id = threadIdx.x;
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    //int thread_row = warp_id * 16 + lane_id / 4;
    const int warp_offset = (warp_id % 2) * 16;
    const int thread_offset = lane_id / 4;

    const int global_row_offset = m_block * kBlockM;


    float rL[2][2] = {0};
    float rD[2][2] = {0};


    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_QKV;

    ThrCopy thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);

    Tensor tQgQ = thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = thr_copy_QKV.partition_D(sQ);

    Tensor tKgK = thr_copy_QKV.partition_S(gK);
    Tensor tKsK = thr_copy_QKV.partition_D(sK);
    Tensor tKrK = make_fragment_like(tKsK);

    Tensor tVgV = thr_copy_QKV.partition_S(gV);
    Tensor tVsV = thr_copy_QKV.partition_D(sV);
    Tensor tVrV = make_fragment_like(tVsV);

    Tensor tdOgdO = thr_copy_QKV.partition_S(gdO);
    Tensor tdOsdO = thr_copy_QKV.partition_D(sdO);

    Tensor tdQsdQ_copy = thr_copy_QKV.partition_S(sdQ);
    Tensor tdQgdQ_copy = thr_copy_QKV.partition_D(gdQ);

    // Identity tensors for bound-checking the masked copies
    Tensor cQ_identity = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));
    Tensor cK_identity = make_identity_tensor(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
    auto tCqQ = thr_copy_QKV.partition_S(cQ_identity);
    auto tCqK = thr_copy_QKV.partition_S(cK_identity);


    // S = QK^T
    typename Kernel_traits::TiledMma_SdP tiled_mma_S;
    ThrMMA thr_mma_S = tiled_mma_S.get_slice(threadIdx.x);
    Tensor tSgQ = thr_mma_S.partition_A(gQ);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);

    Tensor tSgK = thr_mma_S.partition_B(gK);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);

    Tensor tSrS_float = partition_fragment_C(tiled_mma_S, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tSsP = thr_mma_S.partition_C(sP);


    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_S);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(threadIdx.x);
    auto tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_S);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(threadIdx.x);
    auto tSsK_copy_view = smem_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);



    // dP = dOV^T
    typename Kernel_traits::TiledMma_SdP tiled_mma_dP;
    ThrMMA thr_mma_dP = tiled_mma_dP.get_slice(threadIdx.x);

    Tensor tdPgdO = thr_mma_dP.partition_A(gdO);
    Tensor tdPsdO = thr_mma_dP.partition_A(sdO);
    Tensor tdPrdO = thr_mma_dP.make_fragment_A(tdPsdO);

    Tensor tdPgV = thr_mma_dP.partition_B(gV);
    Tensor tdPsV = thr_mma_dP.partition_B(sV);
    Tensor tdPrV = thr_mma_dP.make_fragment_B(tdPsV);

    Tensor tdPrdP_float = partition_fragment_C(tiled_mma_dP, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdPsdS = thr_mma_dP.partition_C(sdS);

    auto smem_tiled_copy_dO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_dP);
    auto smem_thr_copy_dO = smem_tiled_copy_dO.get_slice(threadIdx.x);
    auto tdPsdO_copy_view = smem_thr_copy_dO.partition_S(sdO);
    auto tdPrdO_copy_view = smem_thr_copy_dO.retile_D(tdPrdO);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_dP);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_slice(threadIdx.x);
    auto tdPsV_copy_view = smem_thr_copy_V.partition_S(sV);
    auto tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);


    // dQ = dSK
    typename Kernel_traits::TiledMma_dQ tiled_mma_dQ;
    ThrMMA thr_mma_dQ = tiled_mma_dQ.get_slice(threadIdx.x);
    Tensor tdQsdS = thr_mma_dQ.partition_A(sdS);
    Tensor tdQrdS = thr_mma_dQ.make_fragment_A(tdQsdS);
    Tensor tdQsKt = thr_mma_dQ.partition_B(sKt);
    Tensor tdQrKt = thr_mma_dQ.make_fragment_B(tdQsKt);

    Tensor tdQrdQ_float = partition_fragment_C(tiled_mma_dQ, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tdQsdQ = thr_mma_dQ.partition_C(sdQ);
    Tensor tdQgdQ = thr_mma_dQ.partition_C(gdQ);

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_dQ);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_slice(threadIdx.x);
    auto tdQsdS_copy_view = smem_thr_copy_dS.partition_S(sdS);
    auto tdQrdS_copy_view = smem_thr_copy_dS.retile_D(tdQrdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKt{}, tiled_mma_dQ);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_slice(threadIdx.x);
    auto tdQsKt_copy_view = smem_thr_copy_Kt.partition_S(sKt);
    auto tdQrKt_copy_view = smem_thr_copy_Kt.retile_D(tdQrKt);


    const int window_size_left = params.window_size_left;
    const int window_size_right = Is_causal ? 0 : params.window_size_right;
    const bool has_local = (window_size_left >= 0) || (window_size_right >= 0);
    const float softcap = params.softcap;
    const float alibi_slope = (params.alibi_slopes_ptr == nullptr)
        ? 0.0f
        : params.alibi_slopes_ptr[bidb * params.alibi_slopes_batch_stride + bidh];
    const bool has_alibi = (alibi_slope != 0.0f);

    int n_block_min = 0;
    const int m_block_max = ceil_div(seqlen_q, kBlockM);
    int n_block_max = ceil_div(seqlen_k, kBlockN);

    if (Is_causal || window_size_right >= 0) {
        const int max_col = (m_block + 1) * kBlockM + seqlen_k - seqlen_q + window_size_right;
        n_block_max = min(n_block_max, (max_col <= 0) ? 0 : ceil_div(max_col, kBlockN));
    }
    if (window_size_left >= 0) {
        const int min_col = m_block * kBlockM + seqlen_k - seqlen_q - window_size_left;
        n_block_min = max(0, min_col / kBlockN);
    }

    int n_masking_steps = (has_local || has_alibi)
        ? max(0, n_block_max - n_block_min)
        : ((!Is_causal)
            ? 1
            : ((Is_even_MN && Is_causal) ? ceil_div(kBlockM, kBlockN) : ceil_div(kBlockM, kBlockN) + 1));
    n_masking_steps = min(n_masking_steps, max(0, n_block_max - n_block_min));

    // if this m_block has zero active key blocks, zero dQ and exit early
    if (n_block_max <= n_block_min) {
        clear(tdQrdQ_float);
        Tensor tdQrdQ = convert_type<half_t>(tdQrdQ_float);
        copy(tdQrdQ, tdQsdQ);
        __syncthreads();
        masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tdQsdQ_copy, tdQgdQ_copy, tCqQ,
            seqlen_q - m_block * kBlockM,
            /*clear_D=*/false);
        return;
    }



    //auto KV_TILE_MAX = size<3>(tSgK);
    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto dSKt_BLOCK_MAX = size<2>(tdQsdS);


    // load K, V, dK, dV tiles

    int n_block = n_block_max - 1;


//    Mask<Is_causal> accum_s_mask(seqlen_q, seqlen_k);

    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tCqQ, 
        seqlen_q - m_block * kBlockM, 
        /*clear_D=*/true);
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tdOgdO, tdOsdO, tCqQ, 
        seqlen_q - m_block * kBlockM, 
        /*clear_D=*/true);
    if constexpr (kShareBwdSmem) {
        __syncthreads();
        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
            copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
        }
        __syncthreads();
    }

    Mask<Is_causal> accum_SdP_mask(seqlen_q, seqlen_k, window_size_left, window_size_right, alibi_slope);


    clear(tdQrdQ_float);

            // load rL, rD from gmem to rmem


    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int global_row = global_row_offset + warp_offset + thread_offset + 8 * j + 32 * i;
            if (global_row < seqlen_q) {
                const float l_val = gL(warp_offset + thread_offset + 8 * j + 32 * i);
                rL[i][j] = (l_val == -INFINITY) ? -INFINITY : (l_val * float(M_LOG2E));
                rD[i][j] = gD(warp_offset + thread_offset + 8 * j + 32 * i);
            } else {
                rL[i][j] = -INFINITY;
                rD[i][j] = 0.0f;
            }
        }
    }

    CUTE_NO_UNROLL
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        clear(tSrS_float);
        clear(tdPrdP_float);

        masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tKgK(_,_,_,n_block), tKrK, tCqK, 
            seqlen_k - n_block * kBlockN, 
            /*clear_D=*/true);
        masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tVgV(_,_,_,n_block), tVrV, tCqK, 
            seqlen_k - n_block * kBlockN, 
            /*clear_D=*/true);
        copy(gmem_tiled_copy_QKV, tKrK, tKsK);
        copy(gmem_tiled_copy_QKV, tVrV, tVsV);

        __syncthreads();


//        if (n_block > n_block_min) {
//            masked_copy<Is_even_MN>(gmem_tiled_copy_QKV, tKgK(_,_,_,n_block - 1), tKrK, warp_id, lane_id, seqlen_k - n_block * kBlockN);
//            masked_copy<Is_even_MN>(gmem_tiled_copy_QKV, tVgV(_,_,_,n_block - 1), tVrV, warp_id, lane_id, seqlen_k - n_block * kBlockN);
//        }



        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            if constexpr (!kShareBwdSmem) {
                copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
                copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
            }
            copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));
            copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));

            gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
            gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
        }


        __syncthreads();


        Tensor tSrS_dtanh = make_fragment_like(tSrS_float);
        if (softcap > 0.0f) {
            const float inv_softcap = softmax_scale / softcap;
            const float softcap_log2 = softcap * float(M_LOG2E);
            for (int i = 0; i < tSrS_float.size(); i++) {
                const float u = tanhf(tSrS_float[i] * inv_softcap);
                tSrS_dtanh[i] = 1.0f - u * u;
                tSrS_float[i] = u * softcap_log2;
            }
        } else {
            for (int i = 0; i < tSrS_float.size(); i++) {
                tSrS_float[i] *= scale_softmax_log2;
            }
        }


        accum_SdP_mask.template apply_mask_bwd_dq<Is_causal, Is_even_MN>(
            tSrS_float, tdPrdP_float, warp_id, lane_id, m_block, n_block, seqlen_q, seqlen_k, kBlockM, kBlockN, head_dim
        );



        Tensor tSrP_float = tSrS_float;
        Tensor tdPrdS_float = tdPrdP_float;



        // compute P = exp(S-l)
        for (int i=0; i<2; i++) {
            for (int j=0;j<2;j++) {
                for (int k=0; k< tSrS_float(make_coord(_,j),i,_).size(); k++) {
                    tSrP_float(make_coord(_,j),i,_)[k] = (rL[i][j] == -INFINITY || tSrS_float(make_coord(_,j),i,_)[k] == -FLT_MAX) ? 0.0f : exp2f(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
                }
            }
        }



        // compute dS = P \circ (dP - D)
        // tS has the same mma layout as tdP
        for (int i=0; i<2; i++) {
            for (int j=0;j<2;j++) {
                for (int k=0; k< tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
                    float ds_val = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
                    if (softcap > 0.0f) { ds_val *= tSrS_dtanh(make_coord(_,j),i,_)[k]; }
                    tdPrdS_float(make_coord(_,j),i,_)[k] = ds_val;
                }
            }
        }


        Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

        // copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);


        __syncthreads();

        // dQ += dSK


        CUTE_UNROLL
        for (int dskt_block = 0; dskt_block < dSKt_BLOCK_MAX; dskt_block++) {
            copy(smem_tiled_copy_dS, tdQsdS_copy_view(_,_,dskt_block), tdQrdS_copy_view(_,_,dskt_block));
            copy(smem_tiled_copy_Kt, tdQsKt_copy_view(_,_,dskt_block), tdQrKt_copy_view(_,_,dskt_block));


            gemm(tiled_mma_dQ, tdQrdS(_,_,dskt_block), tdQrKt(_,_,dskt_block), tdQrdQ_float);


        }


        __syncthreads();

    }



    CUTE_NO_UNROLL
    for (; n_block >= n_block_min; --n_block) {
        clear(tSrS_float);
        clear(tdPrdP_float);

        copy(gmem_tiled_copy_QKV, tKgK(_,_,_,n_block), tKrK);
        copy(gmem_tiled_copy_QKV, tVgV(_,_,_,n_block), tVrV);
        copy(gmem_tiled_copy_QKV, tKrK, tKsK);
        copy(gmem_tiled_copy_QKV, tVrV, tVsV);


        __syncthreads();


//        if (n_block > n_block_min) {
//            copy(gmem_tiled_copy_QKV, tKgK(_,_,_,n_block - 1), tKrK);
//            copy(gmem_tiled_copy_QKV, tVgV(_,_,_,n_block - 1), tVrV);
//        }



        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            if constexpr (!kShareBwdSmem) {
                copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
                copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
            }
            copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));
            copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));

            gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
            gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
        }




        __syncthreads();



        Tensor tSrS_dtanh = make_fragment_like(tSrS_float);
        if (softcap > 0.0f) {
            const float inv_softcap = softmax_scale / softcap;
            const float softcap_log2 = softcap * float(M_LOG2E);
            for (int i = 0; i < tSrS_float.size(); i++) {
                const float u = tanhf(tSrS_float[i] * inv_softcap);
                tSrS_dtanh[i] = 1.0f - u * u;
                tSrS_float[i] = u * softcap_log2;
            }
        } else {
            for (int i = 0; i < tSrS_float.size(); i++) {
                tSrS_float[i] *= scale_softmax_log2;
            }
        }



        Tensor tSrP_float = tSrS_float;
        Tensor tdPrdS_float = tdPrdP_float;


        // compute P = exp(S-l)
        for (int i=0; i<2; i++) {
            for (int j=0;j<2;j++) {
                for (int k=0; k< tSrS_float(make_coord(_,j),i,_).size(); k++) {
                    tSrP_float(make_coord(_,j),i,_)[k] = (rL[i][j] == -INFINITY || tSrS_float(make_coord(_,j),i,_)[k] == -FLT_MAX) ? 0.0f : exp2f(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
                }
            }
        }




        // compute dS = P \circ (dP - D)
        // tS has the same mma layout as tdP
        for (int i=0; i<2; i++) {
            for (int j=0;j<2;j++) {
                for (int k=0; k< tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
                    float ds_val = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
                    if (softcap > 0.0f) { ds_val *= tSrS_dtanh(make_coord(_,j),i,_)[k]; }
                    tdPrdS_float(make_coord(_,j),i,_)[k] = ds_val;
                }
            }
        }



        Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

        //copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);

        __syncthreads();



        // dQ += dSK
 



        CUTE_UNROLL
        for (int dskt_block = 0; dskt_block < dSKt_BLOCK_MAX; dskt_block++) {
            copy(smem_tiled_copy_dS, tdQsdS_copy_view(_,_,dskt_block), tdQrdS_copy_view(_,_,dskt_block));
            copy(smem_tiled_copy_Kt, tdQsKt_copy_view(_,_,dskt_block), tdQrKt_copy_view(_,_,dskt_block));


            gemm(tiled_mma_dQ, tdQrdS(_,_,dskt_block), tdQrKt(_,_,dskt_block), tdQrdQ_float);

        }
 
        __syncthreads();

    }
    

    // rescale by head dim
    for (int i=0;i< tdQrdQ_float.size();i ++ ) {
        tdQrdQ_float[i] *= softmax_scale;
    }



    Tensor tdQrdQ = convert_type<half_t>(tdQrdQ_float);



    copy(tdQrdQ, tdQsdQ);


    __syncthreads();



    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tdQsdQ_copy, tdQgdQ_copy, tCqQ, 
        seqlen_q - m_block * kBlockM, 
        /*clear_D=*/false);


}



template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_dk_dv_1colblock(
    const Flash_bwd_params &params,
    const int bidb,
    const int bidh_kv,
    const int n_block
)
{
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const half_t * __restrict__ q_ptr = params.q_ptr;
    const half_t * __restrict__ k_ptr = params.k_ptr;
    const half_t * __restrict__ v_ptr = params.v_ptr;
    float * __restrict__ l_ptr = params.l_ptr;
    float * __restrict__ d_ptr = params.do_o_ptr;
    const half_t * __restrict__ do_ptr = params.do_ptr;
    half_t * __restrict__ dk_ptr = params.dk_ptr;
    half_t * __restrict__ dv_ptr = params.dv_ptr;
    const int batch_size = params.b;
    const int max_seqlen_q = params.seqlen_q;
    const int max_seqlen_k = params.seqlen_k;
    const int num_heads = params.h;
    const int num_heads_k = params.h_k;
    const int h_h_k_ratio = params.h_h_k_ratio;
    const int head_dim = params.d;
    const float softmax_scale = params.softmax_scale;
    const float scale_softmax_log2 = params.scale_softmax_log2;

    const BlockInfo binfo(params, bidb);
    const int seqlen_q = binfo.actual_seqlen_q;
    const int seqlen_k = binfo.actual_seqlen_k;

    if (n_block * kBlockN >= seqlen_k) {
        return;
    }

    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;

    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(params.q_row_stride, params.q_head_stride, Int<1>{}));

    Tensor mK = make_tensor(make_gmem_ptr(k_ptr + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(params.k_row_stride, params.k_head_stride, Int<1>{}));

    Tensor gK = local_tile(mK(_, bidh_kv, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));

    Tensor mV = make_tensor(make_gmem_ptr(v_ptr + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(params.v_row_stride, params.v_head_stride, Int<1>{}));

    Tensor gV = local_tile(mV(_, bidh_kv, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));

    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads, max_seqlen_q, Int<1>{}));

    Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads, max_seqlen_q, Int<1>{}));

    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr + binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)),
                             make_shape(seqlen_q, num_heads, head_dim),
                             make_stride(params.do_row_stride, params.do_head_stride, Int<1>{}));

    Tensor mdV = make_tensor(make_gmem_ptr(dv_ptr + binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(params.dv_row_stride, params.dv_head_stride, Int<1>{}));

    Tensor gdV = local_tile(mdV(_, bidh_kv, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));

    Tensor mdK = make_tensor(make_gmem_ptr(dk_ptr + binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(params.dk_row_stride, params.dk_head_stride, Int<1>{}));

    Tensor gdK = local_tile(mdK(_, bidh_kv, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));

    extern __shared__ char smem_[];

    constexpr bool kShareBwdSmem = (kHeadDim >= 128);
    // Slot 0 (32KB at D=256): sQ/sQt (and sK during prologue, sdK at epilogue)
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});
    Tensor sQt = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQTransposed{});

    Tensor sK = make_tensor(sQ.data() + (kShareBwdSmem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sQ.data() + (kShareBwdSmem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKVTransposed{});

    // Slot 1 (32KB at D=256): sdO/sdOt -> sV -> sP + sdS -> sdV
    Tensor sdO = make_tensor(sQ.data() + (kShareBwdSmem ? size(sQ) : (size(sQ) + size(sK))), typename Kernel_traits::SmemLayoutQ{});
    Tensor sdOt = make_tensor(sQ.data() + (kShareBwdSmem ? size(sQ) : (size(sQ) + size(sK))), typename Kernel_traits::SmemLayoutQTransposed{});

    Tensor sV = make_tensor(sdO.data() + (kShareBwdSmem ? 0 : size(sdO)), typename Kernel_traits::SmemLayoutKV{});

    Tensor sP = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutPdSTransposed{});

    Tensor sdS = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdSTransposed{});

    Tensor sdK = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutKV{});
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutKV{});

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int warp_offset = (warp_id % 2) * 16;
    const int thread_offset = lane_id / 4;

    float rL[2][2] = {0};
    float rD[2][2] = {0};

    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_QKV;
    ThrCopy thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);

    Tensor tQsQ = thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = thr_copy_QKV.partition_S(gK);
    Tensor tKsK = thr_copy_QKV.partition_D(sK);

    Tensor tVgV = thr_copy_QKV.partition_S(gV);
    Tensor tVsV = thr_copy_QKV.partition_D(sV);
    Tensor tVrV = make_fragment_like(tVsV);

    Tensor tdOsdO = thr_copy_QKV.partition_D(sdO);

    Tensor tdKsdK_copy = thr_copy_QKV.partition_S(sdK);
    Tensor tdKgdK_copy = thr_copy_QKV.partition_D(gdK);

    Tensor tdVsdV_copy = thr_copy_QKV.partition_S(sdV);
    Tensor tdVgdV_copy = thr_copy_QKV.partition_D(gdV);

    Tensor cQ_identity = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));
    Tensor cK_identity = make_identity_tensor(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
    auto tCqQ = thr_copy_QKV.partition_S(cQ_identity);
    auto tCqK = thr_copy_QKV.partition_S(cK_identity);

    // S = QK^T
    typename Kernel_traits::TiledMma_SdP tiled_mma_S;
    ThrMMA thr_mma_S = tiled_mma_S.get_slice(threadIdx.x);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);

    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);

    Tensor tSrS_float = partition_fragment_C(tiled_mma_S, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tSsP = thr_mma_S.partition_C(sP);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_S);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(threadIdx.x);
    auto tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_S);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(threadIdx.x);
    auto tSsK_copy_view = smem_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

    // dP = dOV^T
    typename Kernel_traits::TiledMma_SdP tiled_mma_dP;
    ThrMMA thr_mma_dP = tiled_mma_dP.get_slice(threadIdx.x);

    Tensor tdPsdO = thr_mma_dP.partition_A(sdO);
    Tensor tdPrdO = thr_mma_dP.make_fragment_A(tdPsdO);

    Tensor tdPsV = thr_mma_dP.partition_B(sV);
    Tensor tdPrV = thr_mma_dP.make_fragment_B(tdPsV);

    Tensor tdPrdP_float = partition_fragment_C(tiled_mma_dP, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdPsdS = thr_mma_dP.partition_C(sdS);

    auto smem_tiled_copy_dO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_dP);
    auto smem_thr_copy_dO = smem_tiled_copy_dO.get_slice(threadIdx.x);
    auto tdPsdO_copy_view = smem_thr_copy_dO.partition_S(sdO);
    auto tdPrdO_copy_view = smem_thr_copy_dO.retile_D(tdPrdO);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_dP);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_slice(threadIdx.x);
    auto tdPsV_copy_view = smem_thr_copy_V.partition_S(sV);
    auto tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);

    // dV += P^TdO
    typename Kernel_traits::TiledMma_dKdV tiled_mma_dV;
    ThrMMA thr_mma_dV = tiled_mma_dV.get_slice(threadIdx.x);

    Tensor tdVsPt = thr_mma_dV.partition_A(sPt);
    Tensor tdVrPt = thr_mma_dV.make_fragment_A(tdVsPt);

    Tensor tdVsdOt = thr_mma_dV.partition_B(sdOt);
    Tensor tdVrdOt = thr_mma_dV.make_fragment_B(tdVsdOt);

    Tensor tdVrdV_float = partition_fragment_C(tiled_mma_dV, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tdVsdV = thr_mma_dV.partition_C(sdV);

    auto smem_tiled_copy_Pt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomPtdSt{}, tiled_mma_dV);
    auto smem_thr_copy_Pt = smem_tiled_copy_Pt.get_slice(threadIdx.x);
    auto tdVsPt_copy_view = smem_thr_copy_Pt.partition_S(sPt);
    auto tdVrPt_copy_view = smem_thr_copy_Pt.retile_D(tdVrPt);

    auto smem_tiled_copy_dOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomdOt{}, tiled_mma_dV);
    auto smem_thr_copy_dOt = smem_tiled_copy_dOt.get_slice(threadIdx.x);
    auto tdVsdOt_copy_view = smem_thr_copy_dOt.partition_S(sdOt);
    auto tdVrdOt_copy_view = smem_thr_copy_dOt.retile_D(tdVrdOt);

    // dK += dS^TQ
    typename Kernel_traits::TiledMma_dKdV tiled_mma_dK;
    ThrMMA thr_mma_dK = tiled_mma_dK.get_slice(threadIdx.x);
    Tensor tdKsdSt = thr_mma_dK.partition_A(sdSt);
    Tensor tdKrdSt = thr_mma_dK.make_fragment_A(tdKsdSt);

    Tensor tdKsQt = thr_mma_dK.partition_B(sQt);
    Tensor tdKrQt = thr_mma_dK.make_fragment_B(tdKsQt);

    Tensor tdKrdK_float = partition_fragment_C(tiled_mma_dK, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tdKsdK = thr_mma_dK.partition_C(sdK);

    auto smem_tiled_copy_dSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomPtdSt{}, tiled_mma_dK);
    auto smem_thr_copy_dSt = smem_tiled_copy_dSt.get_slice(threadIdx.x);
    auto tdKsdSt_copy_view = smem_thr_copy_dSt.partition_S(sdSt);
    auto tdKrdSt_copy_view = smem_thr_copy_dSt.retile_D(tdKrdSt);

    auto smem_tiled_copy_Qt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomQt{}, tiled_mma_dK);
    auto smem_thr_copy_Qt = smem_tiled_copy_Qt.get_slice(threadIdx.x);
    auto tdKsQt_copy_view = smem_thr_copy_Qt.partition_S(sQt);
    auto tdKrQt_copy_view = smem_thr_copy_Qt.retile_D(tdKrQt);

    const int window_size_left = params.window_size_left;
    const int window_size_right = Is_causal ? 0 : params.window_size_right;
    const bool has_local = (window_size_left >= 0) || (window_size_right >= 0);
    const float softcap = params.softcap;

    int m_block_min = 0;
    int m_block_max = ceil_div(seqlen_q, kBlockM);
    const int n_block_max = ceil_div(seqlen_k, kBlockN);

    if (Is_causal || window_size_right >= 0) {
        const int min_row = n_block * kBlockN + seqlen_q - seqlen_k - window_size_right;
        m_block_min = max(0, (min_row <= 0) ? 0 : (min_row / kBlockM));
    }
    if (window_size_left >= 0) {
        const int max_row = (n_block + 1) * kBlockN + seqlen_q - seqlen_k + window_size_left;
        m_block_max = min(m_block_max, (max_row <= 0) ? 0 : ceil_div(max_row, kBlockM));
    }

    int n_masking_steps_base = (has_local || params.alibi_slopes_ptr != nullptr)
        ? max(0, m_block_max - m_block_min)
        : ((!Is_causal)
            ? 0
            : ((Is_even_MN && Is_causal) ? ceil_div(kBlockN, kBlockM) : ceil_div(kBlockN, kBlockM) + 1));
    n_masking_steps_base = min(n_masking_steps_base, max(0, m_block_max - m_block_min));

    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto PtdOt_BLOCK_MAX = size<2>(tdVsPt);

    // Load KV tile once for the entire GQA query-head group
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tCqK,
        seqlen_k - n_block * kBlockN,
        /*clear_D=*/true);
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tVgV, tVrV, tCqK,
        seqlen_k - n_block * kBlockN,
        /*clear_D=*/true);
    if constexpr (kShareBwdSmem) {
        copy(gmem_tiled_copy_QKV, tVrV, tVsV);
        __syncthreads();
        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));
            copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));
        }
        __syncthreads();
    }

    clear(tdVrdV_float);
    clear(tdKrdK_float);

    // In-kernel GQA reduction loop: accumulate dK and dV in FP32 registers across all query heads sharing bidh_kv
    for (int h_offset = 0; h_offset < h_h_k_ratio; ++h_offset) {
        const int bidh = bidh_kv * h_h_k_ratio + h_offset;
        const float alibi_slope = (params.alibi_slopes_ptr == nullptr)
            ? 0.0f
            : params.alibi_slopes_ptr[bidb * params.alibi_slopes_batch_stride + bidh];
        Mask<Is_causal> accum_SdP_mask(seqlen_q, seqlen_k, window_size_left, window_size_right, alibi_slope);

        Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                               make_coord(_, 0));
        Tensor gdO = local_tile(mdO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_coord(_, 0));
        Tensor gL = local_tile(mL(bidb, bidh, _), Shape<Int<kBlockM>>{},
                               make_coord(_));
        Tensor gD = local_tile(mD(bidb, bidh, _), Shape<Int<kBlockM>>{},
                               make_coord(_));

        Tensor tQgQ = thr_copy_QKV.partition_S(gQ);
        Tensor tdOgdO = thr_copy_QKV.partition_S(gdO);

        int m_block = m_block_min;
        const int n_masking_steps = n_masking_steps_base;

        CUTE_NO_UNROLL
        for (int masking_step = 0; masking_step < n_masking_steps && m_block < m_block_max; ++masking_step, ++m_block) {
            if constexpr (!kShareBwdSmem) {
                copy(gmem_tiled_copy_QKV, tVrV, tVsV);
            }

            clear(tSrS_float);
            clear(tdPrdP_float);

            masked_copy<Is_even_MN>(
                gmem_tiled_copy_QKV, tQgQ(_,_,_,m_block), tQsQ, tCqQ,
                seqlen_q - m_block * kBlockM,
                /*clear_D=*/true);
            masked_copy<Is_even_MN>(
                gmem_tiled_copy_QKV, tdOgdO(_,_,_,m_block), tdOsdO, tCqQ,
                seqlen_q - m_block * kBlockM,
                /*clear_D=*/true);

            __syncthreads();

            CUTE_UNROLL
            for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
                copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
                copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
                if constexpr (!kShareBwdSmem) {
                    copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));
                    copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));
                }

                gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
                gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
            }

            __syncthreads();

            const int global_row_offset = m_block * kBlockM;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    const int block_row = warp_offset + thread_offset + 8 * j + 32 * i;
                    const int global_row = global_row_offset + block_row;
                    if (global_row < seqlen_q) {
                        const float l_val = gL(block_row, m_block);
                        rL[i][j] = (l_val == -INFINITY) ? -INFINITY : (l_val * float(M_LOG2E));
                        rD[i][j] = gD(block_row, m_block);
                    } else {
                        rL[i][j] = -INFINITY;
                        rD[i][j] = 0.0f;
                    }
                }
            }

            Tensor tSrS_dtanh = make_fragment_like(tSrS_float);
            if (softcap > 0.0f) {
                const float inv_softcap = softmax_scale / softcap;
                const float softcap_log2 = softcap * float(M_LOG2E);
                for (int i = 0; i < tSrS_float.size(); i++) {
                    const float u = tanhf(tSrS_float[i] * inv_softcap);
                    tSrS_dtanh[i] = 1.0f - u * u;
                    tSrS_float[i] = u * softcap_log2;
                }
            } else {
                for (int i = 0; i < tSrS_float.size(); i++) {
                    tSrS_float[i] *= scale_softmax_log2;
                }
            }

            if (!Is_even_MN && (n_block == n_block_max - 1 || m_block == m_block_max - 1)) {
                accum_SdP_mask.template apply_mask_bwd_dk_dv</*Is_causal=*/Is_causal, /*Is_even_MN=*/false>(
                    tSrS_float, tdPrdP_float, warp_id, lane_id, m_block, n_block, seqlen_q, seqlen_k, kBlockM, kBlockN, head_dim
                );
            } else {
                accum_SdP_mask.template apply_mask_bwd_dk_dv</*Is_causal=*/Is_causal, /*Is_even_MN=*/true>(
                    tSrS_float, tdPrdP_float, warp_id, lane_id, m_block, n_block, seqlen_q, seqlen_k, kBlockM, kBlockN, head_dim
                );
            }

            Tensor tSrP_float = tSrS_float;
            Tensor tdPrdS_float = tdPrdP_float;

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < tSrS_float(make_coord(_,j),i,_).size(); k++) {
                        tSrP_float(make_coord(_,j),i,_)[k] = (rL[i][j] == -INFINITY || tSrS_float(make_coord(_,j),i,_)[k] == -FLT_MAX)
                            ? 0.0f
                            : exp2f(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
                    }
                }
            }

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
                        float ds_val = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
                    if (softcap > 0.0f) { ds_val *= tSrS_dtanh(make_coord(_,j),i,_)[k]; }
                    tdPrdS_float(make_coord(_,j),i,_)[k] = ds_val;
                    }
                }
            }

            Tensor tSrP = convert_type<half_t>(tSrP_float);
            Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

            if constexpr (kShareBwdSmem) {
                CUTE_UNROLL
                for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
                    copy(smem_tiled_copy_dOt, tdVsdOt_copy_view(_,_,ptdot_block), tdVrdOt_copy_view(_,_,ptdot_block));
                }
            }
            __syncthreads();

            copy(tSrP, tSsP);
            copy(tdPrdS, tdPsdS);

            __syncthreads();

            CUTE_UNROLL
            for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
                copy(smem_tiled_copy_Pt, tdVsPt_copy_view(_,_,ptdot_block), tdVrPt_copy_view(_,_,ptdot_block));
                if constexpr (!kShareBwdSmem) {
                    copy(smem_tiled_copy_dOt, tdVsdOt_copy_view(_,_,ptdot_block), tdVrdOt_copy_view(_,_,ptdot_block));
                }
                gemm(tiled_mma_dV, tdVrPt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block), tdVrdV_float);
            }

            CUTE_UNROLL
            for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
                copy(smem_tiled_copy_dSt, tdKsdSt_copy_view(_,_,ptdot_block), tdKrdSt_copy_view(_,_,ptdot_block));
                copy(smem_tiled_copy_Qt, tdKsQt_copy_view(_,_,ptdot_block), tdKrQt_copy_view(_,_,ptdot_block));
                gemm(tiled_mma_dK, tdKrdSt(_,_,ptdot_block), tdKrQt(_,_,ptdot_block), tdKrdK_float);
            }

            __syncthreads();
        }

        CUTE_NO_UNROLL
        for (; m_block < m_block_max; ++m_block) {
            if constexpr (!kShareBwdSmem) {
                copy(gmem_tiled_copy_QKV, tVrV, tVsV);
            }

            clear(tSrS_float);
            clear(tdPrdP_float);

            masked_copy<Is_even_MN>(
                gmem_tiled_copy_QKV, tQgQ(_,_,_,m_block), tQsQ, tCqQ,
                seqlen_q - m_block * kBlockM,
                /*clear_D=*/true);
            masked_copy<Is_even_MN>(
                gmem_tiled_copy_QKV, tdOgdO(_,_,_,m_block), tdOsdO, tCqQ,
                seqlen_q - m_block * kBlockM,
                /*clear_D=*/true);

            __syncthreads();

            CUTE_UNROLL
            for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
                copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
                copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
                if constexpr (!kShareBwdSmem) {
                    copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));
                    copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));
                }

                gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
                gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
            }

            __syncthreads();

            const int global_row_offset = m_block * kBlockM;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    const int block_row = warp_offset + thread_offset + 8 * j + 32 * i;
                    const int global_row = global_row_offset + block_row;
                    if (global_row < seqlen_q) {
                        const float l_val = gL(block_row, m_block);
                        rL[i][j] = (l_val == -INFINITY) ? -INFINITY : (l_val * float(M_LOG2E));
                        rD[i][j] = gD(block_row, m_block);
                    } else {
                        rL[i][j] = -INFINITY;
                        rD[i][j] = 0.0f;
                    }
                }
            }

            Tensor tSrS_dtanh = make_fragment_like(tSrS_float);
            if (softcap > 0.0f) {
                const float inv_softcap = softmax_scale / softcap;
                const float softcap_log2 = softcap * float(M_LOG2E);
                for (int i = 0; i < tSrS_float.size(); i++) {
                    const float u = tanhf(tSrS_float[i] * inv_softcap);
                    tSrS_dtanh[i] = 1.0f - u * u;
                    tSrS_float[i] = u * softcap_log2;
                }
            } else {
                for (int i = 0; i < tSrS_float.size(); i++) {
                    tSrS_float[i] *= scale_softmax_log2;
                }
            }

            if (!Is_even_MN && (n_block == n_block_max - 1 || m_block == m_block_max - 1)) {
                accum_SdP_mask.template apply_mask_bwd_dk_dv</*Is_causal=*/false, /*Is_even_MN=*/Is_even_MN>(
                    tSrS_float, tdPrdP_float, warp_id, lane_id, m_block, n_block, seqlen_q, seqlen_k, kBlockM, kBlockN, head_dim
                );
            }

            Tensor tSrP_float = tSrS_float;
            Tensor tdPrdS_float = tdPrdP_float;

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < tSrS_float(make_coord(_,j),i,_).size(); k++) {
                        tSrP_float(make_coord(_,j),i,_)[k] = (rL[i][j] == -INFINITY || tSrS_float(make_coord(_,j),i,_)[k] == -FLT_MAX)
                            ? 0.0f
                            : exp2f(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
                    }
                }
            }

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
                        float ds_val = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
                    if (softcap > 0.0f) { ds_val *= tSrS_dtanh(make_coord(_,j),i,_)[k]; }
                    tdPrdS_float(make_coord(_,j),i,_)[k] = ds_val;
                    }
                }
            }

            Tensor tSrP = convert_type<half_t>(tSrP_float);
            Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

            if constexpr (kShareBwdSmem) {
                CUTE_UNROLL
                for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
                    copy(smem_tiled_copy_dOt, tdVsdOt_copy_view(_,_,ptdot_block), tdVrdOt_copy_view(_,_,ptdot_block));
                }
            }
            __syncthreads();

            copy(tSrP, tSsP);
            copy(tdPrdS, tdPsdS);

            __syncthreads();

            CUTE_UNROLL
            for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
                copy(smem_tiled_copy_Pt, tdVsPt_copy_view(_,_,ptdot_block), tdVrPt_copy_view(_,_,ptdot_block));
                if constexpr (!kShareBwdSmem) {
                    copy(smem_tiled_copy_dOt, tdVsdOt_copy_view(_,_,ptdot_block), tdVrdOt_copy_view(_,_,ptdot_block));
                }
                gemm(tiled_mma_dV, tdVrPt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block), tdVrdV_float);
            }

            CUTE_UNROLL
            for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
                copy(smem_tiled_copy_dSt, tdKsdSt_copy_view(_,_,ptdot_block), tdKrdSt_copy_view(_,_,ptdot_block));
                copy(smem_tiled_copy_Qt, tdKsQt_copy_view(_,_,ptdot_block), tdKrQt_copy_view(_,_,ptdot_block));
                gemm(tiled_mma_dK, tdKrdSt(_,_,ptdot_block), tdKrQt(_,_,ptdot_block), tdKrdK_float);
            }

            __syncthreads();
        }
    }

    for (int i = 0; i < tdKrdK_float.size(); i++) {
        tdKrdK_float[i] *= softmax_scale;
    }

    Tensor tdKrdK = convert_type<half_t>(tdKrdK_float);
    Tensor tdVrdV = convert_type<half_t>(tdVrdV_float);

    copy(tdVrdV, tdVsdV);
    copy(tdKrdK, tdKsdK);

    __syncthreads();

    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tdKsdK_copy, tdKgdK_copy, tCqK,
        seqlen_k - n_block * kBlockN,
        /*clear_D=*/false);
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tdVsdV_copy, tdVgdV_copy, tCqK,
        seqlen_k - n_block * kBlockN,
        /*clear_D=*/false);
}

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_dq(const Flash_bwd_params &params) {
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;

    compute_dq_1rowblock<Kernel_traits, Is_causal, Is_even_MN>(params, bidb, bidh, m_block);
}

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_dk_dv(const Flash_bwd_params &params) {
    const int n_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh_kv = blockIdx.z;

    compute_dk_dv_1colblock<Kernel_traits, Is_causal, Is_even_MN>(params, bidb, bidh_kv, n_block);
}
