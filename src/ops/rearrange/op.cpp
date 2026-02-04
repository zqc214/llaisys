#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {

// Y_i = (W_i * X_i) / sqrt(mean(X^2) + eps)
// in: [M, D], weight: [D], out: [M, D]
template <typename T>
void rms_norm_cpu(T *out_ptr, const T *in_ptr, const T *weight_ptr,
                  size_t M, size_t D, float eps) {
    for (size_t m = 0; m < M; ++m) {
        // 计算 mean(X^2)
        float sum_sq = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            float val;
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                val = utils::cast<float>(in_ptr[m * D + d]);
            } else {
                val = static_cast<float>(in_ptr[m * D + d]);
            }
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / D + eps);
        
        // 计算输出
        for (size_t d = 0; d < D; ++d) {
            float x_val, w_val;
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                x_val = utils::cast<float>(in_ptr[m * D + d]);
                w_val = utils::cast<float>(weight_ptr[d]);
            } else {
                x_val = static_cast<float>(in_ptr[m * D + d]);
                w_val = static_cast<float>(weight_ptr[d]);
            }
            float result = (w_val * x_val) / rms;
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                out_ptr[m * D + d] = utils::cast<T>(result);
            } else {
                out_ptr[m * D + d] = static_cast<T>(result);
            }
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    
    // in: [M, D], out: [M, D], weight: [D]
    size_t M = in->shape()[0];
    size_t D = in->shape()[1];
    
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_cpu(reinterpret_cast<float *>(out->data()),
                            reinterpret_cast<const float *>(in->data()),
                            reinterpret_cast<const float *>(weight->data()),
                            M, D, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_cpu(reinterpret_cast<fp16_t *>(out->data()),
                            reinterpret_cast<const fp16_t *>(in->data()),
                            reinterpret_cast<const fp16_t *>(weight->data()),
                            M, D, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_cpu(reinterpret_cast<bf16_t *>(out->data()),
                            reinterpret_cast<const bf16_t *>(in->data()),
                            reinterpret_cast<const bf16_t *>(weight->data()),
                            M, D, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops
