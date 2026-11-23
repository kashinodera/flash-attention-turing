#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/stl.h>
#include "flash.h"
#include "static_switch.h"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

static void check_float4_aligned(const at::Tensor &t, const char *name) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                name, " data pointer must be 16-byte aligned for float4 loads");
    TORCH_CHECK(t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0 && t.stride(2) % 8 == 0,
                name, " batch/seq/head strides must be multiples of 8 elements (16 bytes)");
}


static void set_params_alibi_window_softcap(
    Flash_fwd_params &params,
    const std::optional<at::Tensor> &alibi_slopes_,
    int window_size_left,
    int window_size_right,
    float softcap,
    int batch_size,
    int num_heads)
{
    if (params.is_causal) { window_size_right = 0; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.softcap = softcap;

    if (alibi_slopes_.has_value()) {
        const at::Tensor &alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.is_cuda(), "alibi_slopes must be on CUDA");
        TORCH_CHECK(alibi_slopes.scalar_type() == torch::kFloat32, "alibi_slopes must be float32");
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "alibi_slopes must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.dim() == 1 || alibi_slopes.dim() == 2, "alibi_slopes must be rank 1 or 2");
        if (alibi_slopes.dim() == 1) {
            TORCH_CHECK(alibi_slopes.size(0) == num_heads, "alibi_slopes 1D size must match num_heads");
            params.alibi_slopes_batch_stride = 0;
        } else {
            TORCH_CHECK(alibi_slopes.size(0) == batch_size && alibi_slopes.size(1) == num_heads,
                        "alibi_slopes 2D shape must be (batch_size, num_heads)");
            params.alibi_slopes_batch_stride = alibi_slopes.stride(0);
        }
        params.alibi_slopes_ptr = reinterpret_cast<const float*>(alibi_slopes.data_ptr());
    } else {
        params.alibi_slopes_ptr = nullptr;
        params.alibi_slopes_batch_stride = 0;
    }
}

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor &q,
                      const at::Tensor &k,
                      const at::Tensor &v,
                      const at::Tensor &out,
                      const at::Tensor &l,
                      //void *softmax_lse_d,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      float softmax_scale,
                      bool is_causal) {

    // Reset the parameters
    params = {};

    // Set the pointers and strides.
    params.q_ptr = q.defined() ? reinterpret_cast<half_t*>(q.data_ptr()) : nullptr;
    params.k_ptr = k.defined() ? reinterpret_cast<half_t*>(k.data_ptr()) : nullptr;
    params.v_ptr = v.defined() ? reinterpret_cast<half_t*>(v.data_ptr()) : nullptr;
    params.o_ptr = out.defined() ? reinterpret_cast<half_t*>(out.data_ptr()) : nullptr;


    // Softmax sum
    params.l_ptr = l.defined() ? reinterpret_cast<float*>(l.data_ptr()) : nullptr;


    // All strides are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);

    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
    }

    // Set the dimensions.
    params.b = b;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.d = d;
    params.softmax_scale = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
    params.is_causal = is_causal;
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.stream = at::cuda::getCurrentCUDAStream().stream();
}


void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor &q,
                      const at::Tensor &k,
                      const at::Tensor &v,
                      const at::Tensor &out,
                      const at::Tensor &l,
                      const at::Tensor &dout,
                      at::Tensor &dq,
                      at::Tensor &dk,
                      at::Tensor &dv,
                      at::Tensor &do_o,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      float softmax_scale,
                      bool is_causal) {

    set_params_fprop(params,
                     b,
                     seqlen_q,
                     seqlen_k,
                     h,
                     h_k,
                     d,
                     q, k, v, out, l,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     softmax_scale,
                     is_causal
                     );

    params.do_o_ptr = reinterpret_cast<float*>(do_o.data_ptr());
    params.do_ptr = reinterpret_cast<half_t*>(dout.data_ptr());

    params.dq_ptr = reinterpret_cast<half_t*>(dq.data_ptr());
    params.dk_ptr = reinterpret_cast<half_t*>(dk.data_ptr());
    params.dv_ptr = reinterpret_cast<half_t*>(dv.data_ptr());

    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);

    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }
}


void run_mha_fwd(Flash_fwd_params &params){
    HEADDIM_SWITCH(params.d, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_<kHeadDim, Is_causal>(params);
        });
    });
}

void run_mha_bwd(Flash_bwd_params &params){
#ifndef FA_INFERENCE_ONLY
    HEADDIM_SWITCH(params.d, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_bwd_<kHeadDim, Is_causal>(params);
        });
    });
#else
    TORCH_CHECK(false, "run_mha_bwd called but fa2-turing was compiled with FA_INFERENCE_ONLY");
#endif
}

void run_mha_fwd_kvcache(Flash_fwd_params &params){
    HEADDIM_SWITCH(params.d, [&] {
        run_mha_fwd_kvcache_<kHeadDim>(params);
    });
}

// Backported verbatim from official flash-attention flash_api.cpp (num_splits_heuristic).
static int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / std::ceil(n_waves);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            return num_splits;
        }
    }
    return 1;
}

static int get_cached_num_sms() {
    static int cached_sms[16] = {0};
    int device = 0;
    cudaGetDevice(&device);
    if (device >= 0 && device < 16 && cached_sms[device] > 0) {
        return cached_sms[device];
    }
    int sms = 40;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (device >= 0 && device < 16) {
        cached_sms[device] = sms;
    }
    return sms;
}

static int choose_kv_num_splits(int batch, int num_groups, int active_seqlen_k) {
    const int sms = get_cached_num_sms();

    constexpr int kBlockN = 32;
    const int num_n_blocks = std::max(1, (active_seqlen_k + kBlockN - 1) / kBlockN);
    return num_splits_heuristic(batch * num_groups, sms * 2, num_n_blocks, 128);
}


std::vector<at::Tensor>
mha_fwd(const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const float softmax_scale,
        bool is_causal,
        int window_size_left = -1,
        int window_size_right = -1,
        const float softcap = 0.0f,
        std::optional<at::Tensor> alibi_slopes_ = std::nullopt)
{
    auto device = q.device();

    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16, "q, k, v must be float16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "q, k, v must have contiguous last dimension (stride(-1) == 1)");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    at::Tensor o = torch::empty(q.sizes(), q.options().dtype(torch::kFloat16));

    std::vector<int64_t> size = {batch_size, num_heads, seqlen_q};
    at::Tensor l = torch::empty(size, q.options().dtype(torch::kFloat32).device(device));

    TORCH_CHECK(o.is_cuda(), "Tensor o is not on CUDA");

//    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
//    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
//    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
//    half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());
//
//    float* l_ptr = reinterpret_cast<float*>(l.data_ptr());

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_heads_k,
                     head_size,
                     q, k, v, o, l,
                     nullptr,
                     nullptr,
                     softmax_scale,
                     is_causal
                     );

    // std::cout << "Q ptr: " << q.data_ptr() << "\n";
    // std::cout << "K ptr: " << k.data_ptr() << "\n";
    // std::cout << "V ptr: " << v.data_ptr() << "\n";
    // std::cout << "O ptr: " << o.data_ptr() << "\n";

    set_params_alibi_window_softcap(params, alibi_slopes_, window_size_left, window_size_right, softcap, batch_size, num_heads);
    run_mha_fwd(params);

    return {o, l};

}




#ifdef FA_INFERENCE_ONLY
std::vector<at::Tensor>
mha_bwd(const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &out,
        const at::Tensor &l,
        const at::Tensor &dout,
        const float softmax_scale,
        bool is_causal,
        std::optional<at::Tensor> dq_ = std::nullopt,
        std::optional<at::Tensor> dk_ = std::nullopt,
        std::optional<at::Tensor> dv_ = std::nullopt,
        int window_size_left = -1,
        int window_size_right = -1,
        const float softcap = 0.0f,
        std::optional<at::Tensor> alibi_slopes_ = std::nullopt)
{
    TORCH_CHECK(false, "mha_bwd called but fa2-turing was compiled with FA_INFERENCE_ONLY");
}
#else
std::vector<at::Tensor>
mha_bwd(const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &out,
        const at::Tensor &l,
        const at::Tensor &dout,
        const float softmax_scale,
        bool is_causal,
        std::optional<at::Tensor> dq_ = std::nullopt,
        std::optional<at::Tensor> dk_ = std::nullopt,
        std::optional<at::Tensor> dv_ = std::nullopt,
        int window_size_left = -1,
        int window_size_right = -1,
        const float softcap = 0.0f,
        std::optional<at::Tensor> alibi_slopes_ = std::nullopt)
{
    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda() && dout.is_cuda() && l.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16 && dout.scalar_type() == torch::kFloat16, "q, k, v, out, dout must be float16");
    TORCH_CHECK(l.scalar_type() == torch::kFloat32, "l must be float32");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && out.stride(-1) == 1 && dout.stride(-1) == 1, "q, k, v, out, dout must have contiguous last dimension");
    TORCH_CHECK(l.is_contiguous(), "l must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(out.dim() == 4 && dout.dim() == 4, "out and dout must be rank-4 tensors");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(out.sizes() == q.sizes() && dout.sizes() == q.sizes(), "out and dout must match q shape");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    at::Tensor dq = dq_.has_value() ? dq_.value() : torch::empty(q.sizes(), q.options().dtype(torch::kFloat16));
    at::Tensor dk = dk_.has_value() ? dk_.value() : torch::empty(k.sizes(), k.options().dtype(torch::kFloat16));
    at::Tensor dv = dv_.has_value() ? dv_.value() : torch::empty(v.sizes(), v.options().dtype(torch::kFloat16));
    TORCH_CHECK(dq.sizes() == q.sizes() && dq.stride(-1) == 1, "dq must match q shape and have stride(-1) == 1");
    TORCH_CHECK(dk.sizes() == k.sizes() && dk.stride(-1) == 1, "dk must match k shape and have stride(-1) == 1");
    TORCH_CHECK(dv.sizes() == v.sizes() && dv.stride(-1) == 1, "dv must match v shape and have stride(-1) == 1");

    at::Tensor do_o = torch::empty(l.sizes(), l.options());

    Flash_bwd_params params;
    set_params_dgrad(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_heads_k,
                     head_size,
                     q,
                     k,
                     v,
                     out,
                     l,
                     dout,
                     dq,
                     dk,
                     dv,
                     do_o,
                     nullptr,
                     nullptr,
                     softmax_scale,
                     is_causal);

    set_params_alibi_window_softcap(params, alibi_slopes_, window_size_left, window_size_right, softcap, batch_size, num_heads);
    run_mha_bwd(params);

    return {dq, dk, dv};
}
#endif

std::vector<at::Tensor>
mha_varlen_fwd(const at::Tensor &q,
               const at::Tensor &k,
               const at::Tensor &v,
               const at::Tensor &cu_seqlens_q,
               const at::Tensor &cu_seqlens_k,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal,
               int window_size_left = -1,
               int window_size_right = -1,
               const float softcap = 0.0f,
               std::optional<at::Tensor> alibi_slopes_ = std::nullopt)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16, "q, k, v must be float16");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "q, k, v must have contiguous last dimension (stride(-1) == 1)");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size + 1] with cumulative offsets");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    const int num_heads = q.size(1);
    const int num_heads_k = k.size(1);
    const int head_size = q.size(2);

    at::Tensor out = torch::empty(q.sizes(), q.options().dtype(torch::kFloat16));
    at::Tensor l = torch::empty({batch_size, num_heads, max_seqlen_q}, q.options().dtype(torch::kFloat32));

    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        head_size,
        q, k, v, out, l,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        softmax_scale,
        is_causal
    );

    set_params_alibi_window_softcap(params, alibi_slopes_, window_size_left, window_size_right, softcap, batch_size, num_heads);
    run_mha_fwd(params);

    return {out, l};
}

#ifdef FA_INFERENCE_ONLY
std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &q,
               const at::Tensor &k,
               const at::Tensor &v,
               const at::Tensor &out,
               const at::Tensor &l,
               const at::Tensor &dout,
               const at::Tensor &cu_seqlens_q,
               const at::Tensor &cu_seqlens_k,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal,
               std::optional<at::Tensor> dq_ = std::nullopt,
               std::optional<at::Tensor> dk_ = std::nullopt,
               std::optional<at::Tensor> dv_ = std::nullopt,
               int window_size_left = -1,
               int window_size_right = -1,
               const float softcap = 0.0f,
               std::optional<at::Tensor> alibi_slopes_ = std::nullopt)
{
    TORCH_CHECK(false, "mha_varlen_bwd called but fa2-turing was compiled with FA_INFERENCE_ONLY");
}
#else
std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &q,
               const at::Tensor &k,
               const at::Tensor &v,
               const at::Tensor &out,
               const at::Tensor &l,
               const at::Tensor &dout,
               const at::Tensor &cu_seqlens_q,
               const at::Tensor &cu_seqlens_k,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal,
               std::optional<at::Tensor> dq_ = std::nullopt,
               std::optional<at::Tensor> dk_ = std::nullopt,
               std::optional<at::Tensor> dv_ = std::nullopt,
               int window_size_left = -1,
               int window_size_right = -1,
               const float softcap = 0.0f,
               std::optional<at::Tensor> alibi_slopes_ = std::nullopt)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda() && dout.is_cuda() && l.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16 && dout.scalar_type() == torch::kFloat16, "q, k, v, out, dout must be float16");
    TORCH_CHECK(l.scalar_type() == torch::kFloat32 && l.is_contiguous(), "l must be contiguous float32");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && out.stride(-1) == 1 && dout.stride(-1) == 1, "q, k, v, out, dout must have contiguous last dimension");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int64_t batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size + 1] with cumulative offsets");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must match q shape");
    TORCH_CHECK(l.dim() == 3, "l must be rank-3 for varlen_bwd");

    const int64_t num_heads = q.size(1);
    const int64_t num_heads_k = k.size(1);
    const int64_t head_size = q.size(2);
    TORCH_CHECK(l.size(0) == batch_size && l.size(1) == num_heads && l.size(2) == max_seqlen_q,
                "l must have shape [batch_size, nheads_q, max_seqlen_q]");

    at::Tensor dq = dq_.has_value() ? dq_.value() : torch::empty(q.sizes(), q.options().dtype(torch::kFloat16));
    at::Tensor dk = dk_.has_value() ? dk_.value() : torch::empty(k.sizes(), k.options().dtype(torch::kFloat16));
    at::Tensor dv = dv_.has_value() ? dv_.value() : torch::empty(v.sizes(), v.options().dtype(torch::kFloat16));
    TORCH_CHECK(dq.sizes() == q.sizes() && dq.stride(-1) == 1, "dq must match q shape and have stride(-1) == 1");
    TORCH_CHECK(dk.sizes() == k.sizes() && dk.stride(-1) == 1, "dk must match k shape and have stride(-1) == 1");
    TORCH_CHECK(dv.sizes() == v.sizes() && dv.stride(-1) == 1, "dv must match v shape and have stride(-1) == 1");

    at::Tensor do_o = torch::empty(l.sizes(), l.options());

    Flash_bwd_params params;
    set_params_dgrad(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        head_size,
        q,
        k,
        v,
        out,
        l,
        dout,
        dq,
        dk,
        dv,
        do_o,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        softmax_scale,
        is_causal
    );

    set_params_alibi_window_softcap(params, alibi_slopes_, window_size_left, window_size_right, softcap, batch_size, num_heads);
    run_mha_bwd(params);

    return {dq, dk, dv};
}
#endif


std::vector<at::Tensor>
mha_fwd_kvcache(const at::Tensor &q,
                const at::Tensor &k_cache,
                const at::Tensor &v_cache,
                const std::optional<at::Tensor> &k_,
                const std::optional<at::Tensor> &v_,
                const std::optional<at::Tensor> &cache_seqlens_,
                const std::optional<at::Tensor> &rotary_cos_,
                const std::optional<at::Tensor> &rotary_sin_,
                const std::optional<at::Tensor> &cache_batch_idx_,
                const std::optional<at::Tensor> &cache_leftpad_,
                const std::optional<at::Tensor> &block_table_,
                const std::optional<at::Tensor> &alibi_slopes_,
                std::optional<at::Tensor> &out_,
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,
                int num_splits,
                int active_seqlen_k = 0)
{
    TORCH_CHECK(!rotary_cos_.has_value() && !rotary_sin_.has_value(),
                "rotary embedding is not supported by this port");
    TORCH_CHECK(!cache_batch_idx_.has_value() && !cache_leftpad_.has_value(),
                "cache_batch_idx / cache_leftpad are not supported by this port");
    TORCH_CHECK(!block_table_.has_value(), "paged KV cache is not supported by this port");

    TORCH_CHECK(cache_seqlens_.has_value(), "cache_seqlens must be passed");
    TORCH_CHECK(k_.has_value() == v_.has_value(),
                "k and v must either both be provided or both be None");
    const bool has_new = k_.has_value() && v_.has_value();

    const at::Tensor &cache_seqlens = cache_seqlens_.value();
    // Decode with seqlen_q == 1: causal and non-causal are identical (official behavior).
    (void)is_causal;
    (void)is_rotary_interleaved;
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "q, k_cache, and v_cache must be CUDA tensors");
    if (has_new) {
        TORCH_CHECK(k_.value().is_cuda() && v_.value().is_cuda(),
                    "k and v must be CUDA tensors");
    }
    TORCH_CHECK(cache_seqlens.is_cuda(), "cache_seqlens must be a CUDA tensor");
    TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32, "cache_seqlens must be int32");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous");

    TORCH_CHECK(q.dim() == 4, "q must be rank-4");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "k_cache/v_cache must be rank-4");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be float16");
    TORCH_CHECK(k_cache.scalar_type() == torch::kFloat16 && v_cache.scalar_type() == torch::kFloat16,
                "k_cache/v_cache must be float16");

    const int batch = q.size(0);
    const int seqlen_q = q.size(1);
    TORCH_CHECK(seqlen_q >= 1, "q seqlen_q must be >= 1 for kvcache forward");
    const int num_heads = q.size(2);
    const int head_size = q.size(3);
    TORCH_CHECK(head_size == 32 || head_size == 64 || head_size == 96 || head_size == 128 || head_size == 256,
                "head_dim must be 32, 64, 96, 128, or 256");

    const int num_heads_k = k_cache.size(2);
    TORCH_CHECK(v_cache.size(2) == num_heads_k, "cache head count must match between k_cache and v_cache");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k");

    TORCH_CHECK(k_cache.size(0) == batch && v_cache.size(0) == batch, "cache batch must match q");
    TORCH_CHECK(k_cache.size(3) == head_size && v_cache.size(3) == head_size,
                "cache head_dim must match q");
    TORCH_CHECK(k_cache.size(1) == v_cache.size(1), "k_cache/v_cache capacity must match");
    const int cache_cap = static_cast<int>(k_cache.size(1));
    TORCH_CHECK(cache_seqlens.numel() == batch, "cache_seqlens must have batch elements");

    TORCH_CHECK(k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
                "k_cache/v_cache must have contiguous last dimension");
    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    check_float4_aligned(q, "q");
    check_float4_aligned(k_cache, "k_cache");
    check_float4_aligned(v_cache, "v_cache");

    if (has_new) {
        const at::Tensor &k = k_.value();
        const at::Tensor &v = v_.value();
        TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be rank-4");
        TORCH_CHECK(k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16,
                    "k/v must be float16");
        TORCH_CHECK(k.size(0) == batch && v.size(0) == batch, "k/v batch must match q");
        TORCH_CHECK(k.size(1) == seqlen_q && v.size(1) == seqlen_q, "k/v seqlen must match q seqlen_q");
        TORCH_CHECK(k.size(2) == num_heads_k && v.size(2) == num_heads_k, "k/v head count must match k_cache");
        TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "head_dim must match");
        TORCH_CHECK(k.stride(-1) == 1 && v.stride(-1) == 1, "k/v must have contiguous last dimension");
        check_float4_aligned(k, "k");
        check_float4_aligned(v, "v");
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == q.scalar_type(), "out must have the same dtype as q");
        TORCH_CHECK(out.is_cuda(), "out must be CUDA");
        TORCH_CHECK(out.stride(-1) == 1, "out must have contiguous last dimension");
        TORCH_CHECK(out.dim() == 4 && out.size(0) == batch && out.size(1) == seqlen_q
                        && out.size(2) == num_heads && out.size(3) == head_size,
                    "out must have shape (batch, seqlen_q, num_heads, head_size)");
    } else {
        out = torch::empty(q.sizes(), q.options().dtype(torch::kFloat16));
    }
    at::Tensor softmax_lse = torch::empty(
        {batch, num_heads, seqlen_q}, q.options().dtype(torch::kFloat));

    // Populate KV-cache decode parameters explicitly.
    Flash_fwd_params params = {};
    TORCH_CHECK(q.defined() && q.numel() > 0, "q missing storage");
    TORCH_CHECK(out.defined() && out.numel() > 0, "out missing storage");
    TORCH_CHECK(k_cache.defined() && k_cache.numel() > 0, "k_cache missing storage");
    TORCH_CHECK(v_cache.defined() && v_cache.numel() > 0, "v_cache missing storage");
    TORCH_CHECK(cache_seqlens.defined() && cache_seqlens.numel() > 0, "cache_seqlens missing storage");

    params.q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    params.o_ptr = reinterpret_cast<half_t*>(out.data_ptr());
    params.l_ptr = reinterpret_cast<float*>(softmax_lse.data_ptr());

    params.b = batch;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = cache_cap;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.d = head_size;
    params.softmax_scale = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
    params.is_causal = is_causal;
    set_params_alibi_window_softcap(params, alibi_slopes_, window_size_left, window_size_right, softcap, batch, num_heads);
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.stream = at::cuda::getCurrentCUDAStream().stream();

    params.q_batch_stride = q.stride(0);
    params.q_row_stride = q.stride(1);
    params.q_head_stride = q.stride(2);
    params.o_batch_stride = out.stride(0);
    params.o_row_stride = out.stride(1);
    params.o_head_stride = out.stride(2);

    params.k_cache_ptr = reinterpret_cast<half_t*>(k_cache.data_ptr());
    params.v_cache_ptr = reinterpret_cast<half_t*>(v_cache.data_ptr());
    params.cache_seqlens = reinterpret_cast<int*>(cache_seqlens.data_ptr());

    params.k_cache_batch_stride = k_cache.stride(0);
    params.k_cache_seq_stride = k_cache.stride(1);
    params.k_cache_head_stride = k_cache.stride(2);
    params.v_cache_batch_stride = v_cache.stride(0);
    params.v_cache_seq_stride = v_cache.stride(1);
    params.v_cache_head_stride = v_cache.stride(2);

    if (has_new) {
        const at::Tensor &k = k_.value();
        const at::Tensor &v = v_.value();
        TORCH_CHECK(k.defined() && k.numel() > 0, "k missing storage");
        TORCH_CHECK(v.defined() && v.numel() > 0, "v missing storage");
        params.k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
        params.v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
        params.k_new_ptr = params.k_ptr;
        params.v_new_ptr = params.v_ptr;
        params.k_new_batch_stride = k.stride(0);
        params.k_new_seq_stride = k.stride(1);
        params.k_new_head_stride = k.stride(2);
        params.v_new_batch_stride = v.stride(0);
        params.v_new_seq_stride = v.stride(1);
        params.v_new_head_stride = v.stride(2);
    } else {
        params.k_ptr = nullptr;
        params.v_ptr = nullptr;
        params.k_new_ptr = nullptr;
        params.v_new_ptr = nullptr;
        params.k_new_batch_stride = 0;
        params.k_new_seq_stride = 0;
        params.k_new_head_stride = 0;
        params.v_new_batch_stride = 0;
        params.v_new_seq_stride = 0;
        params.v_new_head_stride = 0;
    }

    constexpr int kMaxGroupM = 8;
    const int total_m_per_kv = seqlen_q * params.h_h_k_ratio;
    const int groups_per_kv = (total_m_per_kv + kMaxGroupM - 1) / kMaxGroupM;
    const int total_groups = num_heads_k * groups_per_kv;

    TORCH_CHECK(num_splits >= 0, "num_splits must be >= 0");
    TORCH_CHECK(active_seqlen_k >= 0, "active_seqlen_k must be >= 0");
    int splits;
    if (num_splits > 0) {
        splits = num_splits;
    } else {
        const int effective_len = (active_seqlen_k > 0) ? active_seqlen_k : cache_cap;
        splits = (effective_len <= 32) ? 1 : choose_kv_num_splits(batch, total_groups, effective_len);
    }
    TORCH_CHECK(splits >= 1, "num_splits must be >= 1");
    params.num_splits = splits;

    at::Tensor o_partials;
    at::Tensor lse_partials;
    if (splits > 1) {
        o_partials = torch::empty(
            {batch, num_heads, seqlen_q, splits, head_size},
            q.options().dtype(torch::kFloat32));
        lse_partials = torch::empty(
            {batch, num_heads, seqlen_q, splits},
            q.options().dtype(torch::kFloat32));
        params.o_partials = reinterpret_cast<float*>(o_partials.data_ptr());
        params.lse_partials = reinterpret_cast<float*>(lse_partials.data_ptr());
    }

    run_mha_fwd_kvcache(params);

    // Keep partial tensors alive until end of scope (stream-ordered reuse is safe).
    if (splits > 1) {
        (void)o_partials;
        (void)lse_partials;
    }

    return {out, softmax_lse};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &mha_fwd, "Forward pass",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("softmax_scale"), py::arg("is_causal"),
          py::arg("window_size_left") = -1, py::arg("window_size_right") = -1,
          py::arg("softcap") = 0.0f, py::arg("alibi_slopes_") = py::none());
    m.def("bwd", &mha_bwd, "Backward pass",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"), py::arg("l"), py::arg("dout"),
          py::arg("softmax_scale"), py::arg("is_causal"),
          py::arg("dq_") = py::none(), py::arg("dk_") = py::none(), py::arg("dv_") = py::none(),
          py::arg("window_size_left") = -1, py::arg("window_size_right") = -1,
          py::arg("softcap") = 0.0f, py::arg("alibi_slopes_") = py::none());
    m.def("varlen_fwd", &mha_varlen_fwd, "Varlen forward pass",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"),
          py::arg("max_seqlen_q"), py::arg("max_seqlen_k"),
          py::arg("softmax_scale"), py::arg("is_causal"),
          py::arg("window_size_left") = -1, py::arg("window_size_right") = -1,
          py::arg("softcap") = 0.0f, py::arg("alibi_slopes_") = py::none());
    m.def("varlen_bwd", &mha_varlen_bwd, "Varlen backward pass",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"), py::arg("l"), py::arg("dout"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"), py::arg("max_seqlen_q"), py::arg("max_seqlen_k"),
          py::arg("softmax_scale"), py::arg("is_causal"),
          py::arg("dq_") = py::none(), py::arg("dk_") = py::none(), py::arg("dv_") = py::none(),
          py::arg("window_size_left") = -1, py::arg("window_size_right") = -1,
          py::arg("softcap") = 0.0f, py::arg("alibi_slopes_") = py::none());
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"),
          py::arg("k_"), py::arg("v_"),
          py::arg("cache_seqlens_"),
          py::arg("rotary_cos_"), py::arg("rotary_sin_"),
          py::arg("cache_batch_idx_"), py::arg("cache_leftpad_"),
          py::arg("block_table_"), py::arg("alibi_slopes_"),
          py::arg("out_"),
          py::arg("softmax_scale"), py::arg("is_causal"),
          py::arg("window_size_left") = -1, py::arg("window_size_right") = -1,
          py::arg("softcap") = 0.0f, py::arg("is_rotary_interleaved") = true,
          py::arg("num_splits") = 0, py::arg("active_seqlen_k") = 0);
}
