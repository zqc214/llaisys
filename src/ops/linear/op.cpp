#include "op.hpp"

#include "../../utils.hpp"

namespace llaisys::ops {

// Y = X * W^T + b
// X: [M, K], W: [N, K], Y: [M, N], b: [N]
template <typename T>
void linear_cpu(T *out_ptr, const T *in_ptr, const T *weight_ptr, const T *bias_ptr,
                size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float x_val, w_val;
                if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                    x_val = utils::cast<float>(in_ptr[m * K + k]);
                    w_val = utils::cast<float>(weight_ptr[n * K + k]);
                } else {
                    x_val = static_cast<float>(in_ptr[m * K + k]);
                    w_val = static_cast<float>(weight_ptr[n * K + k]);
                }
                sum += x_val * w_val;
            }
            // 加上 bias
            if (bias_ptr != nullptr) {
                float b_val;
                if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                    b_val = utils::cast<float>(bias_ptr[n]);
                } else {
                    b_val = static_cast<float>(bias_ptr[n]);
                }
                sum += b_val;
            }
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                out_ptr[m * N + n] = utils::cast<T>(sum);
            } else {
                out_ptr[m * N + n] = static_cast<T>(sum);
            }
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    
    // in: [M, K], weight: [N, K], out: [M, N]
    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];
    
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *bias_ptr = bias ? reinterpret_cast<const float *>(bias->data()) : nullptr;
        return linear_cpu(reinterpret_cast<float *>(out->data()),
                          reinterpret_cast<const float *>(in->data()),
                          reinterpret_cast<const float *>(weight->data()),
                          bias_ptr, M, N, K);
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *bias_ptr = bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr;
        return linear_cpu(reinterpret_cast<fp16_t *>(out->data()),
                          reinterpret_cast<const fp16_t *>(in->data()),
                          reinterpret_cast<const fp16_t *>(weight->data()),
                          bias_ptr, M, N, K);
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *bias_ptr = bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr;
        return linear_cpu(reinterpret_cast<bf16_t *>(out->data()),
                          reinterpret_cast<const bf16_t *>(in->data()),
                          reinterpret_cast<const bf16_t *>(weight->data()),
                          bias_ptr, M, N, K);
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops
