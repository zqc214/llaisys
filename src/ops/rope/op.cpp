#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {

// RoPE: 旋转位置编码
// in/out: [seqlen, nhead, d]
// pos_ids: [seqlen]
// phi_{i,j} = p_i / theta^(2j/d)
// a'_{i,j} = a_{i,j} * cos(phi) - b_{i,j} * sin(phi)
// b'_{i,j} = b_{i,j} * cos(phi) + a_{i,j} * sin(phi)
template <typename T>
void rope_cpu(T *out_ptr, const T *in_ptr, const int64_t *pos_ids_ptr,
              size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t half_d = d / 2;
    
    for (size_t seq = 0; seq < seqlen; ++seq) {
        int64_t pos = pos_ids_ptr[seq];
        for (size_t h = 0; h < nhead; ++h) {
            for (size_t j = 0; j < half_d; ++j) {
                // 计算角度 phi = pos / theta^(2j/d)
                float freq = 1.0f / std::pow(theta, (2.0f * j) / d);
                float phi = pos * freq;
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                
                // 获取 a 和 b (前半部分和后半部分)
                size_t idx_a = seq * nhead * d + h * d + j;
                size_t idx_b = seq * nhead * d + h * d + j + half_d;
                
                float a_val, b_val;
                if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                    a_val = utils::cast<float>(in_ptr[idx_a]);
                    b_val = utils::cast<float>(in_ptr[idx_b]);
                } else {
                    a_val = static_cast<float>(in_ptr[idx_a]);
                    b_val = static_cast<float>(in_ptr[idx_b]);
                }
                
                // 计算旋转后的值
                float a_prime = a_val * cos_phi - b_val * sin_phi;
                float b_prime = b_val * cos_phi + a_val * sin_phi;
                
                if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                    out_ptr[idx_a] = utils::cast<T>(a_prime);
                    out_ptr[idx_b] = utils::cast<T>(b_prime);
                } else {
                    out_ptr[idx_a] = static_cast<T>(a_prime);
                    out_ptr[idx_b] = static_cast<T>(b_prime);
                }
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "pos_ids must be int64");
    
    // in/out: [seqlen, nhead, d]
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];
    
    const int64_t *pos_ids_ptr = reinterpret_cast<const int64_t *>(pos_ids->data());
    
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_cpu(reinterpret_cast<float *>(out->data()),
                        reinterpret_cast<const float *>(in->data()),
                        pos_ids_ptr, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_cpu(reinterpret_cast<fp16_t *>(out->data()),
                        reinterpret_cast<const fp16_t *>(in->data()),
                        pos_ids_ptr, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_cpu(reinterpret_cast<bf16_t *>(out->data()),
                        reinterpret_cast<const bf16_t *>(in->data()),
                        pos_ids_ptr, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops
