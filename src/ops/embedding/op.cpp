#include "op.hpp"

#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {

template <typename T>
void embedding_cpu(T *out_ptr, const int64_t *index_ptr, const T *weight_ptr,
                   size_t seq_len, size_t embed_dim) {
    for (size_t i = 0; i < seq_len; ++i) {
        int64_t idx = index_ptr[i];
        const T *src = weight_ptr + idx * embed_dim;
        T *dst = out_ptr + i * embed_dim;
        std::memcpy(dst, src, embed_dim * sizeof(T));
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // index 必须是 int64 类型
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "index must be int64");
    // out 和 weight 应该是相同类型
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DEVICE(out, index, weight);
    
    // index: [seq_len], weight: [vocab_size, embed_dim], out: [seq_len, embed_dim]
    size_t seq_len = index->shape()[0];
    size_t embed_dim = weight->shape()[1];
    
    const int64_t *index_ptr = reinterpret_cast<const int64_t *>(index->data());
    
    switch (weight->dtype()) {
    case LLAISYS_DTYPE_F32:
        return embedding_cpu(reinterpret_cast<float *>(out->data()),
                             index_ptr,
                             reinterpret_cast<const float *>(weight->data()),
                             seq_len, embed_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_cpu(reinterpret_cast<fp16_t *>(out->data()),
                             index_ptr,
                             reinterpret_cast<const fp16_t *>(weight->data()),
                             seq_len, embed_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_cpu(reinterpret_cast<bf16_t *>(out->data()),
                             index_ptr,
                             reinterpret_cast<const bf16_t *>(weight->data()),
                             seq_len, embed_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(weight->dtype());
    }
}
} // namespace llaisys::ops
