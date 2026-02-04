#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>
#include <limits>

namespace llaisys::ops {

template <typename T>
void argmax_cpu(int64_t *max_idx_ptr, T *max_val_ptr, const T *vals_ptr, size_t numel) {
    T max_value = vals_ptr[0];
    int64_t max_index = 0;
    
    for (size_t i = 1; i < numel; ++i) {
        float curr_val, max_val_f;
        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
            curr_val = utils::cast<float>(vals_ptr[i]);
            max_val_f = utils::cast<float>(max_value);
        } else {
            curr_val = static_cast<float>(vals_ptr[i]);
            max_val_f = static_cast<float>(max_value);
        }
        
        if (curr_val > max_val_f) {
            max_value = vals_ptr[i];
            max_index = static_cast<int64_t>(i);
        }
    }
    
    *max_idx_ptr = max_index;
    *max_val_ptr = max_value;
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // max_idx 应该是 int64 类型
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "max_idx must be int64");
    // max_val 和 vals 应该是相同类型
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    
    int64_t *max_idx_ptr = reinterpret_cast<int64_t *>(max_idx->data());
    size_t numel = vals->numel();
    
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_cpu(max_idx_ptr,
                          reinterpret_cast<float *>(max_val->data()),
                          reinterpret_cast<const float *>(vals->data()),
                          numel);
    case LLAISYS_DTYPE_F16:
        return argmax_cpu(max_idx_ptr,
                          reinterpret_cast<fp16_t *>(max_val->data()),
                          reinterpret_cast<const fp16_t *>(vals->data()),
                          numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_cpu(max_idx_ptr,
                          reinterpret_cast<bf16_t *>(max_val->data()),
                          reinterpret_cast<const bf16_t *>(vals->data()),
                          numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops
