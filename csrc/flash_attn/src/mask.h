#pragma once

#include <cfloat>
#include <cmath>
#include <cute/tensor.hpp>

using namespace cute;

template <bool Is_causal>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope_log2;

    __forceinline__ __device__ Mask(const int max_seqlen_q,
                                    const int max_seqlen_k,
                                    const int window_size_left = -1,
                                    const int window_size_right = -1,
                                    const float alibi_slope = 0.0f)
        : max_seqlen_q(max_seqlen_q)
        , max_seqlen_k(max_seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(Is_causal ? 0 : window_size_right)
        , alibi_slope_log2(alibi_slope * float(M_LOG2E)) {
    }

    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask_fwd(Tensor<Engine, Layout> &tensor,
                                                    const int warp_id,
                                                    const int lane_id,
                                                    const int kBlockN,
                                                    int &causal_offset,
                                                    int &is_even_mn_offset) {
        const bool has_local = (window_size_left >= 0) || (window_size_right >= 0);
        const bool has_alibi = (alibi_slope_log2 != 0.0f);
        if (Causal_mask || !Is_even_MN || has_local || has_alibi) {
            const int row_offset = (warp_id * 16) + (lane_id / 4);
            const int col_offset = (lane_id % 4) * 2;
            const int right_lim = Causal_mask ? 0 : window_size_right;

            CUTE_UNROLL
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < size<1>(tensor); k++) {
                        for (int l = 0; l < size<2>(tensor); l++) {
                            const int row = row_offset + 8 * j;
                            const int col = col_offset + i + 8 * l;
                            const int rel_pos = col - row - causal_offset;
                            if (has_alibi) {
                                tensor(make_coord(i, j), k, l) += alibi_slope_log2 * float(rel_pos);
                            }
                            if (Causal_mask || right_lim >= 0) {
                                if (rel_pos > right_lim) {
                                    tensor(make_coord(i, j), k, l) = -FLT_MAX;
                                }
                            }
                            if (window_size_left >= 0) {
                                if (rel_pos < -window_size_left) {
                                    tensor(make_coord(i, j), k, l) = -FLT_MAX;
                                }
                            }
                            if constexpr (!Is_even_MN) {
                                if (col >= is_even_mn_offset) {
                                    tensor(make_coord(i, j), k, l) = -FLT_MAX;
                                }
                            }
                        }
                    }
                }
            }
            causal_offset += kBlockN;
            is_even_mn_offset += kBlockN;
        }
    }

    template <bool Causal_mask=false, bool Is_even_MN=true,
              typename Engine_S, typename Layout_S,
              typename Engine_dP, typename Layout_dP>
    __forceinline__ __device__ void apply_mask_bwd_dq(Tensor<Engine_S, Layout_S> &tensor_S,
                                                       Tensor<Engine_dP, Layout_dP> &tensor_dP,
                                                       int warp_id,
                                                       int lane_id,
                                                       int m_block,
                                                       int n_block,
                                                       const int seqlen_q,
                                                       const int seqlen_k,
                                                       int kBlockM,
                                                       int kBlockN,
                                                       const int head_dim) {
        const bool has_local = (window_size_left >= 0) || (window_size_right >= 0);
        const bool has_alibi = (alibi_slope_log2 != 0.0f);
        if (Causal_mask || !Is_even_MN || has_local || has_alibi) {
            const int row_offset = (warp_id % 2) * 16 + (lane_id / 4);
            const int col_offset = (warp_id / 2) * 8 + (lane_id % 4) * 2;
            const int global_row_offset = m_block * kBlockM;
            const int global_col_offset = n_block * kBlockN;
            const int diag_shift = seqlen_k - seqlen_q;
            const int right_lim = Causal_mask ? 0 : window_size_right;

            CUTE_UNROLL
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < size<1>(tensor_S); k++) {
                        for (int l = 0; l < size<2>(tensor_S); l++) {
                            const int row = row_offset + 8 * j + 32 * k;
                            const int col = col_offset + i + 32 * l;
                            const int global_row = global_row_offset + row;
                            const int global_col = global_col_offset + col;
                            const int rel_pos = global_col - global_row - diag_shift;
                            if (has_alibi) {
                                tensor_S(make_coord(i, j), k, l) += alibi_slope_log2 * float(rel_pos);
                            }
                            if (Causal_mask || right_lim >= 0) {
                                if (rel_pos > right_lim) {
                                    tensor_S(make_coord(i, j), k, l) = -FLT_MAX;
                                    tensor_dP(make_coord(i, j), k, l) = 0;
                                }
                            }
                            if (window_size_left >= 0) {
                                if (rel_pos < -window_size_left) {
                                    tensor_S(make_coord(i, j), k, l) = -FLT_MAX;
                                    tensor_dP(make_coord(i, j), k, l) = 0;
                                }
                            }
                            if constexpr (!Is_even_MN) {
                                if (global_col >= seqlen_k || global_row >= seqlen_q) {
                                    tensor_S(make_coord(i, j), k, l) = -FLT_MAX;
                                    tensor_dP(make_coord(i, j), k, l) = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <bool Causal_mask=false, bool Is_even_MN=true,
              typename Engine_S, typename Layout_S,
              typename Engine_dP, typename Layout_dP>
    __forceinline__ __device__ void apply_mask_bwd_dk_dv(Tensor<Engine_S, Layout_S> &tensor_S,
                                                          Tensor<Engine_dP, Layout_dP> &tensor_dP,
                                                          int warp_id,
                                                          int lane_id,
                                                          int m_block,
                                                          int n_block,
                                                          const int seqlen_q,
                                                          const int seqlen_k,
                                                          int kBlockM,
                                                          int kBlockN,
                                                          const int head_dim) {
        const bool has_local = (window_size_left >= 0) || (window_size_right >= 0);
        const bool has_alibi = (alibi_slope_log2 != 0.0f);
        if (Causal_mask || !Is_even_MN || has_local || has_alibi) {
            const int row_offset = (warp_id % 2) * 16 + (lane_id / 4);
            const int col_offset = (warp_id / 2) * 8 + (lane_id % 4) * 2;
            const int global_row_offset = m_block * kBlockM;
            const int global_col_offset = n_block * kBlockN;
            const int diag_shift = seqlen_k - seqlen_q;
            const int right_lim = Causal_mask ? 0 : window_size_right;

            CUTE_UNROLL
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < size<1>(tensor_S); k++) {
                        for (int l = 0; l < size<2>(tensor_S); l++) {
                            const int row = row_offset + 8 * j + 32 * k;
                            const int col = col_offset + i + 32 * l;
                            const int global_row = global_row_offset + row;
                            const int global_col = global_col_offset + col;
                            const int rel_pos = global_col - global_row - diag_shift;
                            if (has_alibi) {
                                tensor_S(make_coord(i, j), k, l) += alibi_slope_log2 * float(rel_pos);
                            }
                            if (Causal_mask || right_lim >= 0) {
                                if (rel_pos > right_lim) {
                                    tensor_S(make_coord(i, j), k, l) = -FLT_MAX;
                                    tensor_dP(make_coord(i, j), k, l) = 0;
                                }
                            }
                            if (window_size_left >= 0) {
                                if (rel_pos < -window_size_left) {
                                    tensor_S(make_coord(i, j), k, l) = -FLT_MAX;
                                    tensor_dP(make_coord(i, j), k, l) = 0;
                                }
                            }
                            if constexpr (!Is_even_MN) {
                                if (global_row >= seqlen_q || global_col >= seqlen_k) {
                                    tensor_S(make_coord(i, j), k, l) = -FLT_MAX;
                                    tensor_dP(make_coord(i, j), k, l) = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};
