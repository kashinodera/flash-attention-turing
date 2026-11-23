#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <float.h>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "kernel_traits.h"
#include "utils.h"
#include "block_info.h"

using namespace cute;

template <typename Kernel_traits, bool Is_causal>
inline __device__ void compute_dot_do_o(const Flash_bwd_params &params)
{
    half_t rdO[8];
    half_t rO[8];
    float sum = 0.0f;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;
    const int row_block = blockIdx.z;
    const int max_seqlen_q = params.seqlen_q;
    const int num_heads = params.h;

    const BlockInfo binfo(params, bidb);
    const int actual_seqlen_q = binfo.actual_seqlen_q;

    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int elements_per_thread = kHeadDim / 32;

    const int thread_row = warp_id;
    const int thread_col = lane_id * elements_per_thread;
    const int global_row = row_block * 32 + thread_row;

    const int64_t d_offset = (int64_t(bidb) * num_heads + bidh) * max_seqlen_q + row_block * 32;

    if (global_row < actual_seqlen_q) {
        const int64_t o_offset = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
                               + int64_t(global_row) * params.o_row_stride
                               + int64_t(bidh) * params.o_head_stride;
        const int64_t do_offset = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
                                + int64_t(global_row) * params.do_row_stride
                                + int64_t(bidh) * params.do_head_stride;

        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            rO[i] = params.o_ptr[o_offset + thread_col + i];
            rdO[i] = params.do_ptr[do_offset + thread_col + i];
        }

        #pragma unroll
        for (int i = 0; i < elements_per_thread; i++) {
            sum += static_cast<float>(rO[i]) * static_cast<float>(rdO[i]);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            params.do_o_ptr[d_offset + thread_row] = sum;
        }
    } else if (global_row < max_seqlen_q && lane_id == 0) {
        params.do_o_ptr[d_offset + thread_row] = 0.0f;
    }
}
