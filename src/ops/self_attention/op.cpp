#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>
#include <vector>
#include <limits>

namespace llaisys::ops {

// Self-Attention with causal mask
// A = Q * K^T * scale
// Y = causal_softmax(A) * V
// q: [seqlen, nhead, d], k: [total_len, nkvhead, d], v: [total_len, nkvhead, dv]
// attn_val: [seqlen, nhead, dv]
template <typename T>
void self_attention_cpu(T *attn_val_ptr, const T *q_ptr, const T *k_ptr, const T *v_ptr,
                        size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                        size_t d, size_t dv, float scale) {
    // nhead / nkvhead 应该是整数（GQA）
    size_t head_ratio = nhead / nkvhead;
    
    for (size_t seq_q = 0; seq_q < seqlen; ++seq_q) {
        for (size_t h = 0; h < nhead; ++h) {
            size_t kv_head = h / head_ratio;
            
            // 计算 attention scores: A[seq_q, h, :] = Q[seq_q, h, :] * K[:, kv_head, :]^T * scale
            std::vector<float> scores(total_len);
            float max_score = -std::numeric_limits<float>::infinity();
            
            // causal mask: 只能看到 total_len - seqlen + seq_q 之前的位置（包含）
            size_t causal_limit = total_len - seqlen + seq_q + 1;
            
            for (size_t seq_k = 0; seq_k < total_len; ++seq_k) {
                if (seq_k >= causal_limit) {
                    scores[seq_k] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                
                float dot = 0.0f;
                for (size_t i = 0; i < d; ++i) {
                    float q_val, k_val;
                    size_t q_idx = seq_q * nhead * d + h * d + i;
                    size_t k_idx = seq_k * nkvhead * d + kv_head * d + i;
                    
                    if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                        q_val = utils::cast<float>(q_ptr[q_idx]);
                        k_val = utils::cast<float>(k_ptr[k_idx]);
                    } else {
                        q_val = static_cast<float>(q_ptr[q_idx]);
                        k_val = static_cast<float>(k_ptr[k_idx]);
                    }
                    dot += q_val * k_val;
                }
                scores[seq_k] = dot * scale;
                if (scores[seq_k] > max_score) {
                    max_score = scores[seq_k];
                }
            }
            
            // Softmax
            float sum_exp = 0.0f;
            for (size_t seq_k = 0; seq_k < total_len; ++seq_k) {
                if (scores[seq_k] > -std::numeric_limits<float>::infinity() / 2) {
                    scores[seq_k] = std::exp(scores[seq_k] - max_score);
                    sum_exp += scores[seq_k];
                } else {
                    scores[seq_k] = 0.0f;
                }
            }
            for (size_t seq_k = 0; seq_k < total_len; ++seq_k) {
                scores[seq_k] /= sum_exp;
            }
            
            // 计算输出: attn_val[seq_q, h, :] = scores * V[:, kv_head, :]
            for (size_t i = 0; i < dv; ++i) {
                float val = 0.0f;
                for (size_t seq_k = 0; seq_k < total_len; ++seq_k) {
                    size_t v_idx = seq_k * nkvhead * dv + kv_head * dv + i;
                    float v_val;
                    if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                        v_val = utils::cast<float>(v_ptr[v_idx]);
                    } else {
                        v_val = static_cast<float>(v_ptr[v_idx]);
                    }
                    val += scores[seq_k] * v_val;
                }
                
                size_t out_idx = seq_q * nhead * dv + h * dv + i;
                if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                    attn_val_ptr[out_idx] = utils::cast<T>(val);
                } else {
                    attn_val_ptr[out_idx] = static_cast<T>(val);
                }
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    
    // q: [seqlen, nhead, d]
    // k: [total_len, nkvhead, d]
    // v: [total_len, nkvhead, dv]
    // attn_val: [seqlen, nhead, dv]
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];
    
    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_cpu(reinterpret_cast<float *>(attn_val->data()),
                                  reinterpret_cast<const float *>(q->data()),
                                  reinterpret_cast<const float *>(k->data()),
                                  reinterpret_cast<const float *>(v->data()),
                                  seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_cpu(reinterpret_cast<fp16_t *>(attn_val->data()),
                                  reinterpret_cast<const fp16_t *>(q->data()),
                                  reinterpret_cast<const fp16_t *>(k->data()),
                                  reinterpret_cast<const fp16_t *>(v->data()),
                                  seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_cpu(reinterpret_cast<bf16_t *>(attn_val->data()),
                                  reinterpret_cast<const bf16_t *>(q->data()),
                                  reinterpret_cast<const bf16_t *>(k->data()),
                                  reinterpret_cast<const bf16_t *>(v->data()),
                                  seqlen, total_len, nhead, nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}
} // namespace llaisys::ops
