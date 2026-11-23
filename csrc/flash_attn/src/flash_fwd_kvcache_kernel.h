#pragma once

#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "flash.h"

using half_t = cutlass::half_t;
using namespace cute;

__device__ __forceinline__ float h2f(half_t h) { return static_cast<float>(h); }
__device__ __forceinline__ half_t f2h(float f) { return half_t(f); }

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return v;
}

// Compute active splits for a given seqlen_k and num_splits
__device__ __host__ __forceinline__ int compute_active_splits(int seqlen_k, int kBlockN, int num_splits) {
    if (seqlen_k <= 0 || num_splits <= 1) {
        return 1;
    }
    const int num_tiles = (seqlen_k + kBlockN - 1) / kBlockN;
    const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
    return (num_tiles + tiles_per_split - 1) / tiles_per_split;
}

// Turing SM75 16x8x8 FP32<=FP16*FP16 Tensor Core MMA helper for 8 grouped query rows x 32 KV rows
__device__ __forceinline__ void sm75_mma_m16n8k8_tn(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t b0,
    float c0, float c1, float c2, float c3)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

// Head-grouped & multi-token KV-cache forward kernel.
// Grid: (num_splits, batch, num_kv_heads * num_m_groups)
// Each CTA processes up to kMaxGroupM = 8 query items (q_row, h_offset) that share a single KV head `h_kv`,
// reading K_cache and V_cache from DRAM only ONCE per KV head group.
template <int kHeadDim, int kBlockN, int kNWarps, int kMaxGroupM = 8>
__global__ __launch_bounds__(256) void flash_fwd_kvcache_kernel(const Flash_fwd_params params) {
    constexpr int kThreads = kNWarps * 32;
    static_assert(kThreads == 256, "kThreads must be 256");
    static_assert(kBlockN == 32, "kBlockN must be 32 for optimal Turing occupancy");
    static_assert(kHeadDim % 8 == 0, "kHeadDim must be a multiple of 8 for 128-bit loads");
    static_assert(kMaxGroupM <= 8, "sm75_mma_m16n8k8_tn path assumes kMaxGroupM <= 8");

    const int split = blockIdx.x;
    const int bidb = blockIdx.y;
    const int group_z = blockIdx.z;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;

    const int seqlen_q = params.seqlen_q;
    const int h_h_k_ratio = params.h_h_k_ratio;
    const int total_m_per_kv = seqlen_q * h_h_k_ratio;
    const int groups_per_kv = (total_m_per_kv + kMaxGroupM - 1) / kMaxGroupM;

    const int h_kv = group_z / groups_per_kv;
    const int m_group_idx = group_z % groups_per_kv;
    const int m_start = m_group_idx * kMaxGroupM;
    const int group_m = min(kMaxGroupM, total_m_per_kv - m_start);

    const int* cache_seqlens = params.cache_seqlens;
    const int cache_cap = params.seqlen_k;
    const bool has_new = (params.k_new_ptr != nullptr && params.v_new_ptr != nullptr);
    const int new_tokens = has_new ? seqlen_q : 0;
    const int max_valid_cache_len = max(0, cache_cap - new_tokens);
    const int raw_cache_len = (cache_seqlens != nullptr) ? cache_seqlens[bidb] : max_valid_cache_len;
    const int cache_len = max(0, min(raw_cache_len, max_valid_cache_len));
    const int seqlen_k = cache_len + new_tokens;
    const int num_splits = params.num_splits;
    const float softmax_scale = params.softmax_scale;

    const int num_tiles = (seqlen_k + kBlockN - 1) / kBlockN;
    const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
    int tile_start = split * tiles_per_split;
    int tile_end = min(tile_start + tiles_per_split, num_tiles);
    const int active_splits = compute_active_splits(seqlen_k, kBlockN, num_splits);

    // Device-side active-split early exit: inactive splits (> 0) exit immediately without touching SMEM or GMEM.
    if (num_splits > 1 && split >= active_splits && split > 0) {
        return;
    }

    const bool is_empty = tile_start >= tile_end;

    // Only the first m_group (m_group_idx == 0) of each KV head appends new tokens in-place to k_cache / v_cache.
    const bool owns_append = has_new && (!is_empty) && (m_group_idx == 0);

    half_t* k_cache_row =
        params.k_cache_ptr + bidb * params.k_cache_batch_stride + h_kv * params.k_cache_head_stride;
    half_t* v_cache_row =
        params.v_cache_ptr + bidb * params.v_cache_batch_stride + h_kv * params.v_cache_head_stride;
    const half_t* k_new_base =
        has_new ? (params.k_new_ptr + bidb * params.k_new_batch_stride + h_kv * params.k_new_head_stride) : nullptr;
    const half_t* v_new_base =
        has_new ? (params.v_new_ptr + bidb * params.v_new_batch_stride + h_kv * params.v_new_head_stride) : nullptr;

    if (owns_append && k_new_base != nullptr && v_new_base != nullptr) {
        constexpr int num_vec = kHeadDim / 8;
        const int tok_start = max(0, tile_start * kBlockN - cache_len);
        const int tok_end = min(new_tokens, tile_end * kBlockN - cache_len);
        for (int t_new = tok_start + (tid / num_vec); t_new < tok_end; t_new += (kThreads / num_vec)) {
            const int v_col = tid % num_vec;
            const int pos = cache_len + t_new;
            half_t* k_dst = k_cache_row + int64_t(pos) * params.k_cache_seq_stride;
            half_t* v_dst = v_cache_row + int64_t(pos) * params.v_cache_seq_stride;
            const half_t* k_src = k_new_base + int64_t(t_new) * params.k_new_seq_stride;
            const half_t* v_src = v_new_base + int64_t(t_new) * params.v_new_seq_stride;
            reinterpret_cast<float4*>(k_dst)[v_col] = reinterpret_cast<const float4*>(k_src)[v_col];
            reinterpret_cast<float4*>(v_dst)[v_col] = reinterpret_cast<const float4*>(v_src)[v_col];
        }
    }

    if (is_empty) {
        // Empty prefix (seqlen_k == 0 on split 0): write 0 to O and -INFINITY to LSE
        if (split == 0) {
            for (int m_it = 0; m_it < group_m; ++m_it) {
                const int m_linear = m_start + m_it;
                const int q_row = m_linear / h_h_k_ratio;
                const int h_off = m_linear % h_h_k_ratio;
                const int bidh = h_kv * h_h_k_ratio + h_off;
                if (num_splits > 1 && params.o_partials != nullptr) {
                    const int64_t lse_idx =
                        (((int64_t(bidb) * params.h + bidh) * seqlen_q + q_row) * num_splits) + split;
                    if (params.lse_partials != nullptr && tid == 0) {
                        params.lse_partials[lse_idx] = -INFINITY;
                    }
                } else {
                    half_t* o_ptr = params.o_ptr
                                  + int64_t(bidb) * params.o_batch_stride
                                  + int64_t(q_row) * params.o_row_stride
                                  + int64_t(bidh) * params.o_head_stride;
                    if (tid < kHeadDim / 2) {
                        reinterpret_cast<half2*>(o_ptr)[tid] = __float22half2_rn(make_float2(0.f, 0.f));
                    }
                    if (params.l_ptr != nullptr && tid == 0) {
                        params.l_ptr[(int64_t(bidb) * params.h + bidh) * seqlen_q + q_row] = -INFINITY;
                    }
                }
            }
        }
        return;
    }

    extern __shared__ char smem_bytes[];
    // Layout: sQ[kMaxGroupM * kHeadDim] | sK[kBlockN * kHeadDim] | sV[kBlockN * kHeadDim] | sScores[kMaxGroupM * kBlockN] | sRed[kMaxGroupM * 4]
    half_t* sQ = reinterpret_cast<half_t*>(smem_bytes);
    half_t* sK = sQ + kMaxGroupM * kHeadDim;
    half_t* sV = sK + kBlockN * kHeadDim;
    float* sScores = reinterpret_cast<float*>(sV + kBlockN * kHeadDim);
    float* sRed = sScores + kMaxGroupM * kBlockN;

    // Load all `group_m` query vectors into sQ[m_it * kHeadDim]
    constexpr int q_vecs = kHeadDim / 8;
    for (int idx = tid; idx < group_m * q_vecs; idx += kThreads) {
        const int m_it = idx / q_vecs;
        const int v_col = idx % q_vecs;
        const int m_linear = m_start + m_it;
        const int q_row = m_linear / h_h_k_ratio;
        const int h_off = m_linear % h_h_k_ratio;
        const int bidh = h_kv * h_h_k_ratio + h_off;
        const half_t* q_ptr = params.q_ptr
                            + int64_t(bidb) * params.q_batch_stride
                            + int64_t(q_row) * params.q_row_stride
                            + int64_t(bidh) * params.q_head_stride;
        reinterpret_cast<float4*>(sQ + m_it * kHeadDim)[v_col] = reinterpret_cast<const float4*>(q_ptr)[v_col];
    }
    __syncthreads();

    constexpr int kHalf2PerRow = kHeadDim / 2;
    constexpr int kDVecs = (kHalf2PerRow + 31) / 32;

    float row_max[kMaxGroupM];
    float row_sum[kMaxGroupM];
    float2 o_acc[kMaxGroupM][kDVecs];

    #pragma unroll
    for (int m_it = 0; m_it < kMaxGroupM; ++m_it) {
        row_max[m_it] = -FLT_MAX;
        row_sum[m_it] = 0.0f;
        #pragma unroll
        for (int d_it = 0; d_it < kDVecs; ++d_it) {
            o_acc[m_it][d_it] = make_float2(0.0f, 0.0f);
        }
    }

    constexpr int total_vecs = (kBlockN * kHeadDim) / 8;
    constexpr int vecs_per_row = kHeadDim / 8;
    constexpr int rows_per_warp = kBlockN / kNWarps;
    const int row_base = warp * rows_per_warp;
    const int causal_base = seqlen_k - seqlen_q;
    const int window_size_left = params.window_size_left;
    const int window_size_right = params.is_causal ? 0 : params.window_size_right;

    // Tighten tile bounds for causal and sliding-window attention across the CTA's query group
    const int min_q_row = m_start / h_h_k_ratio;
    const int max_q_row = (m_start + group_m - 1) / h_h_k_ratio;
    if (window_size_left >= 0) {
        const int min_valid_col = max(0, causal_base + min_q_row - window_size_left);
        tile_start = max(tile_start, min_valid_col / kBlockN);
    }
    if (params.is_causal || window_size_right >= 0) {
        const int max_valid_col = min(seqlen_k - 1, causal_base + max_q_row + window_size_right);
        const int max_tile = (max_valid_col < 0) ? 0 : ((max_valid_col + kBlockN) / kBlockN);
        tile_end = min(tile_end, max_tile);
    }

    // Prologue: load first valid K/V tile into shared memory before entering the main loop
    if (tile_start < tile_end) {
        const int kv0 = tile_start * kBlockN;
        const int valid = min(kBlockN, seqlen_k - kv0);
        #pragma unroll
        for (int vidx = tid; vidx < total_vecs; vidx += kThreads) {
            const int r = vidx / vecs_per_row;
            const int c = (vidx % vecs_per_row) * 8;
            if (r < valid) {
                const int global_row = kv0 + r;
                const half_t* k_src = (global_row >= cache_len && k_new_base != nullptr)
                    ? (k_new_base + int64_t(global_row - cache_len) * params.k_new_seq_stride + c)
                    : (k_cache_row + int64_t(global_row) * params.k_cache_seq_stride + c);
                const half_t* v_src = (global_row >= cache_len && v_new_base != nullptr)
                    ? (v_new_base + int64_t(global_row - cache_len) * params.v_new_seq_stride + c)
                    : (v_cache_row + int64_t(global_row) * params.v_cache_seq_stride + c);
                reinterpret_cast<float4*>(sK)[vidx] = *reinterpret_cast<const float4*>(k_src);
                reinterpret_cast<float4*>(sV)[vidx] = *reinterpret_cast<const float4*>(v_src);
            } else {
                float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
                reinterpret_cast<float4*>(sK)[vidx] = zero4;
                reinterpret_cast<float4*>(sV)[vidx] = zero4;
            }
        }
        __syncthreads();
    }

    for (int tile = tile_start; tile < tile_end; ++tile) {
        const int kv0 = tile * kBlockN;
        const int valid = min(kBlockN, seqlen_k - kv0);

        // 2. Warp-Parallel Q * K^T Dot Product across all `group_m` grouped queries
        if (group_m >= 4 && warp < 4) {
            // Tensor-Core SM75_16x8x8 MMA path: each of Warps 0..3 computes an 8x8 sub-tile of S = sQ[0..group_m-1] * sK[warp*8 .. warp*8+7]^T
            const int n_col_base = warp * 8;
            const int quad_row = lane / 4;       // 0..7
            const int quad_col = (lane % 4) * 2; // 0, 2, 4, 6
            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

            #pragma unroll
            for (int k_step = 0; k_step < kHeadDim; k_step += 8) {
                const uint32_t a0 = (quad_row < group_m)
                    ? *reinterpret_cast<const uint32_t*>(sQ + quad_row * kHeadDim + k_step + (lane % 4) * 2)
                    : 0u;
                const uint32_t a1 = a0;
                const int b_row = n_col_base + quad_row;
                const uint32_t b0 = (b_row < valid)
                    ? *reinterpret_cast<const uint32_t*>(sK + b_row * kHeadDim + k_step + (lane % 4) * 2)
                    : 0u;
                sm75_mma_m16n8k8_tn(c0, c1, c2, c3, a0, a1, b0, c0, c1, c2, c3);
            }

            // Store the valid rows of the 8x8 tile for (m_it = quad_row < group_m, j = n_col_base + quad_col + {0,1})
            const int m_it = quad_row;
            if (m_it < group_m) {
                const int m_linear = m_start + m_it;
                const int q_row = m_linear / h_h_k_ratio;
                const int bidh = h_kv * h_h_k_ratio + (m_linear % h_h_k_ratio);
                const float alibi_slope = (params.alibi_slopes_ptr == nullptr)
                    ? 0.0f
                    : params.alibi_slopes_ptr[bidb * params.alibi_slopes_batch_stride + bidh];
                const int max_col = (params.is_causal || window_size_right >= 0)
                    ? min(seqlen_k - 1, causal_base + q_row + window_size_right)
                    : (seqlen_k - 1);
                const int min_col = (window_size_left >= 0) ? max(0, causal_base + q_row - window_size_left) : 0;

                #pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    const int j = n_col_base + quad_col + ii;
                    const float raw = (ii == 0) ? c0 : c1;
                    const int global_col = kv0 + j;
                    if (j < valid && global_col >= min_col && global_col <= max_col) {
                        float s = raw * softmax_scale;
                        if (params.softcap > 0.0f) {
                            s = params.softcap * tanhf(s / params.softcap);
                        }
                        if (alibi_slope != 0.0f) {
                            s += alibi_slope * float(global_col - (causal_base + q_row));
                        }
                        sScores[m_it * kBlockN + j] = s;
                    } else {
                        sScores[m_it * kBlockN + j] = -FLT_MAX;
                    }
                }
            }
        } else if (group_m < 4) {
            for (int m_it = 0; m_it < group_m; ++m_it) {
                const int m_linear = m_start + m_it;
                const int q_row = m_linear / h_h_k_ratio;
                const int bidh = h_kv * h_h_k_ratio + (m_linear % h_h_k_ratio);
                const float alibi_slope = (params.alibi_slopes_ptr == nullptr)
                    ? 0.0f
                    : params.alibi_slopes_ptr[bidb * params.alibi_slopes_batch_stride + bidh];
                const int max_col = (params.is_causal || window_size_right >= 0)
                    ? min(seqlen_k - 1, causal_base + q_row + window_size_right)
                    : (seqlen_k - 1);
                const int min_col = (window_size_left >= 0) ? max(0, causal_base + q_row - window_size_left) : 0;
                const half2* q_h2 = reinterpret_cast<const half2*>(sQ + m_it * kHeadDim);

                #pragma unroll
                for (int r = 0; r < rows_per_warp; ++r) {
                    const int j = row_base + r;
                    const int global_col = kv0 + j;
                    float qk_acc = 0.0f;
                    if (j < valid && global_col >= min_col && global_col <= max_col) {
                        const half2* k_h2 = reinterpret_cast<const half2*>(sK + j * kHeadDim);
                        #pragma unroll
                        for (int e = lane; e < kHalf2PerRow; e += 32) {
                            float2 q_f2 = __half22float2(q_h2[e]);
                            float2 k_f2 = __half22float2(k_h2[e]);
                            qk_acc += q_f2.x * k_f2.x + q_f2.y * k_f2.y;
                        }
                    }
                    qk_acc = warp_reduce_sum(qk_acc);

                    if (lane == 0) {
                        if (j < valid && global_col >= min_col && global_col <= max_col) {
                            float s = qk_acc * softmax_scale;
                            if (params.softcap > 0.0f) {
                                s = params.softcap * tanhf(s / params.softcap);
                            }
                            if (alibi_slope != 0.0f) {
                                s += alibi_slope * float(global_col - (causal_base + q_row));
                            }
                            sScores[m_it * kBlockN + j] = s;
                        } else {
                            sScores[m_it * kBlockN + j] = -FLT_MAX;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // Prefetch next tile's K/V into registers while Warps 0..group_m-1 compute online softmax!
        constexpr int kVecsPerThread = (total_vecs + kThreads - 1) / kThreads;
        float4 pf_k[kVecsPerThread];
        float4 pf_v[kVecsPerThread];
        const bool has_next_tile = (tile + 1 < tile_end);
        if (has_next_tile) {
            const int next_kv0 = (tile + 1) * kBlockN;
            const int next_valid = min(kBlockN, seqlen_k - next_kv0);
            #pragma unroll
            for (int vi = 0; vi < kVecsPerThread; ++vi) {
                const int vidx = tid + vi * kThreads;
                if (vidx < total_vecs) {
                    const int r = vidx / vecs_per_row;
                    const int c = (vidx % vecs_per_row) * 8;
                    if (r < next_valid) {
                        const int global_row = next_kv0 + r;
                        const half_t* k_src = (global_row >= cache_len && k_new_base != nullptr)
                            ? (k_new_base + int64_t(global_row - cache_len) * params.k_new_seq_stride + c)
                            : (k_cache_row + int64_t(global_row) * params.k_cache_seq_stride + c);
                        const half_t* v_src = (global_row >= cache_len && v_new_base != nullptr)
                            ? (v_new_base + int64_t(global_row - cache_len) * params.v_new_seq_stride + c)
                            : (v_cache_row + int64_t(global_row) * params.v_cache_seq_stride + c);
                        pf_k[vi] = *reinterpret_cast<const float4*>(k_src);
                        pf_v[vi] = *reinterpret_cast<const float4*>(v_src);
                    } else {
                        pf_k[vi] = make_float4(0.f, 0.f, 0.f, 0.f);
                        pf_v[vi] = make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                }
            }
        }

        // 3. Online Softmax across the tile (Warp `m_it` computes softmax for query `m_it` in parallel!)
        if (warp < group_m) {
            const int m_it = warp;
            const float s = (lane < kBlockN) ? sScores[m_it * kBlockN + lane] : -FLT_MAX;
            float tile_m = warp_reduce_max(s);
            tile_m = __shfl_sync(0xffffffffu, tile_m, 0);

            const float prev_max = row_max[m_it];
            const float m_new = (tile_m == -FLT_MAX) ? prev_max : fmaxf(prev_max, tile_m);
            const float alpha = (prev_max == -FLT_MAX) ? 0.0f : ((m_new == prev_max) ? 1.0f : exp2f((prev_max - m_new) * float(M_LOG2E)));

            const float p = (s == -FLT_MAX || m_new == -FLT_MAX) ? 0.0f : exp2f((s - m_new) * float(M_LOG2E));
            if (lane < kBlockN) {
                sScores[m_it * kBlockN + lane] = p;
            }

            const float tile_sum = warp_reduce_sum(p);
            if (lane == 0) {
                sRed[m_it * 4 + 0] = m_new;
                sRed[m_it * 4 + 1] = alpha;
                sRed[m_it * 4 + 2] = row_sum[m_it] * alpha + tile_sum;
            }
        }
        __syncthreads();

        // 4. 8-Warp Parallel P * V Accumulation across all `group_m` queries
        for (int m_it = 0; m_it < group_m; ++m_it) {
            const float m_new = sRed[m_it * 4 + 0];
            const float alpha = sRed[m_it * 4 + 1];
            row_sum[m_it] = sRed[m_it * 4 + 2];
            row_max[m_it] = m_new;

            #pragma unroll
            for (int d_it = 0; d_it < kDVecs; ++d_it) {
                const int d_h2 = lane + d_it * 32;
                o_acc[m_it][d_it].x *= alpha;
                o_acc[m_it][d_it].y *= alpha;
                if (d_h2 < kHalf2PerRow) {
                    float2 acc = make_float2(0.f, 0.f);
                    #pragma unroll
                    for (int r = 0; r < rows_per_warp; ++r) {
                        const int j = row_base + r;
                        if (j < valid) {
                            const float p = sScores[m_it * kBlockN + j];
                            const half2 v_h2 = reinterpret_cast<const half2*>(sV + j * kHeadDim)[d_h2];
                            const float2 v_f2 = __half22float2(v_h2);
                            acc.x += p * v_f2.x;
                            acc.y += p * v_f2.y;
                        }
                    }
                    o_acc[m_it][d_it].x += acc.x;
                    o_acc[m_it][d_it].y += acc.y;
                }
            }
        }
        __syncthreads();

        if (has_next_tile) {
            #pragma unroll
            for (int vi = 0; vi < kVecsPerThread; ++vi) {
                const int vidx = tid + vi * kThreads;
                if (vidx < total_vecs) {
                    reinterpret_cast<float4*>(sK)[vidx] = pf_k[vi];
                    reinterpret_cast<float4*>(sV)[vidx] = pf_v[vi];
                }
            }
            __syncthreads();
        }
    }

    // Cross-warp reduction of o_acc across the 8 warps using sK/sV shared memory as float2 scratch (32 * kHeadDim * 2 bytes = 8 KB)
    float2* sWarpO = reinterpret_cast<float2*>(sK);
    const bool write_direct = (num_splits == 1) || (params.o_partials == nullptr);

    for (int m_it = 0; m_it < group_m; ++m_it) {
        #pragma unroll
        for (int d_it = 0; d_it < kDVecs; ++d_it) {
            const int d_h2 = lane + d_it * 32;
            if (d_h2 < kHalf2PerRow) {
                sWarpO[warp * kHalf2PerRow + d_h2] = o_acc[m_it][d_it];
            }
        }
        __syncthreads();

        float2 final_o = make_float2(0.0f, 0.0f);
        if (tid < kHalf2PerRow) {
            #pragma unroll
            for (int w = 0; w < kNWarps; ++w) {
                float2 w_val = sWarpO[w * kHalf2PerRow + tid];
                final_o.x += w_val.x;
                final_o.y += w_val.y;
            }
        }

        const float r_sum = row_sum[m_it];
        const float r_max = row_max[m_it];
        const float inv_l = (r_sum > 0.0f) ? (1.0f / r_sum) : 0.0f;
        const float lse = (r_sum > 0.0f && r_max > -FLT_MAX) ? (r_max + logf(r_sum)) : -INFINITY;

        const int m_linear = m_start + m_it;
        const int q_row = m_linear / h_h_k_ratio;
        const int h_off = m_linear % h_h_k_ratio;
        const int bidh = h_kv * h_h_k_ratio + h_off;

        if (write_direct) {
            half_t* o_ptr = params.o_ptr
                          + int64_t(bidb) * params.o_batch_stride
                          + int64_t(q_row) * params.o_row_stride
                          + int64_t(bidh) * params.o_head_stride;
            if (tid < kHalf2PerRow) {
                float2 out_f2 = make_float2(final_o.x * inv_l, final_o.y * inv_l);
                reinterpret_cast<half2*>(o_ptr)[tid] = __float22half2_rn(out_f2);
            }
            if (params.l_ptr != nullptr && tid == 0) {
                params.l_ptr[(int64_t(bidb) * params.h + bidh) * seqlen_q + q_row] = lse;
            }
        } else {
            const int64_t base_idx =
                (((int64_t(bidb) * params.h + bidh) * seqlen_q + q_row) * num_splits) + split;
            float* o_part = params.o_partials + base_idx * kHeadDim;
            if (tid < kHalf2PerRow) {
                o_part[2 * tid] = final_o.x * inv_l;
                o_part[2 * tid + 1] = final_o.y * inv_l;
            }
            if (tid == 0 && params.lse_partials != nullptr) {
                params.lse_partials[base_idx] = lse;
            }
        }
        __syncthreads();
    }
}

// Combine split-KV partials into final output, iterating ONLY over active_splits for the batch row.
// grid: (batch, heads, seqlen_q)
template <int kHeadDim, int kBlockN = 32>
__global__ __launch_bounds__(256) void flash_fwd_kvcache_combine_kernel(const Flash_fwd_params params) {
    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;
    const int q_row = blockIdx.z;
    const int tid = static_cast<int>(threadIdx.x);
    const int num_splits = params.num_splits;
    const int seqlen_q = params.seqlen_q;

    const int cache_cap = params.seqlen_k;
    const bool has_new = (params.k_new_ptr != nullptr && params.v_new_ptr != nullptr);
    const int new_tokens = has_new ? seqlen_q : 0;
    const int max_valid_cache_len = max(0, cache_cap - new_tokens);
    const int raw_cache_len = (params.cache_seqlens != nullptr) ? params.cache_seqlens[bidb] : max_valid_cache_len;
    const int cache_len = max(0, min(raw_cache_len, max_valid_cache_len));
    const int seqlen_k = cache_len + new_tokens;
    const int active_splits = (seqlen_k <= 0) ? 0 : compute_active_splits(seqlen_k, kBlockN, num_splits);

    const int64_t row_split_base = ((int64_t(bidb) * params.h + bidh) * seqlen_q + q_row) * num_splits;
    const float* lse_base = params.lse_partials + row_split_base;
    const float* o_base = params.o_partials + row_split_base * kHeadDim;

    float m = -FLT_MAX;
    for (int s = 0; s < active_splits; ++s) {
        const float lse = lse_base[s];
        if (lse > -INFINITY) {
            m = fmaxf(m, lse);
        }
    }

    half_t* o_ptr = params.o_ptr
                  + int64_t(bidb) * params.o_batch_stride
                  + int64_t(q_row) * params.o_row_stride
                  + int64_t(bidh) * params.o_head_stride;

    if (m == -FLT_MAX) {
        if (tid < kHeadDim / 2) {
            reinterpret_cast<half2*>(o_ptr)[tid] = __float22half2_rn(make_float2(0.f, 0.f));
        }
        if (params.l_ptr != nullptr && tid == 0) {
            params.l_ptr[(int64_t(bidb) * params.h + bidh) * seqlen_q + q_row] = -INFINITY;
        }
        return;
    }

    float denom = 0.0f;
    float acc2_x = 0.0f;
    float acc2_y = 0.0f;

    for (int s = 0; s < active_splits; ++s) {
        const float lse = lse_base[s];
        if (lse == -INFINITY) {
            continue;
        }
        const float w = exp2f((lse - m) * float(M_LOG2E));
        denom += w;
        const float* o_s = o_base + int64_t(s) * kHeadDim;
        if (tid < kHeadDim / 2) {
            acc2_x += w * o_s[2 * tid];
            acc2_y += w * o_s[2 * tid + 1];
        }
    }

    const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    if (params.l_ptr != nullptr && tid == 0) {
        const float lse_final = (denom > 0.0f) ? (m + logf(denom)) : -INFINITY;
        params.l_ptr[(int64_t(bidb) * params.h + bidh) * seqlen_q + q_row] = lse_final;
    }
    if (tid < kHeadDim / 2) {
        half2 res = __float22half2_rn(make_float2(acc2_x * inv, acc2_y * inv));
        reinterpret_cast<half2*>(o_ptr)[tid] = res;
    }
}
