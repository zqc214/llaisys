#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {

// SwiGLU: out = up * (gate / (1 + exp(-gate)))
// out, gate, up: [seqlen, intermediate_size]
template <typename T>
void swiglu_cpu(T *out_ptr, const T *gate_ptr, const T *up_ptr, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float gate_val, up_val;
        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
            gate_val = utils::cast<float>(gate_ptr[i]);
            up_val = utils::cast<float>(up_ptr[i]);
        } else {
            gate_val = static_cast<float>(gate_ptr[i]);
            up_val = static_cast<float>(up_ptr[i]);
        }
        
        // silu(gate) = gate / (1 + exp(-gate)) = gate * sigmoid(gate)
        float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
        float silu_gate = gate_val * sigmoid_gate;
        float result = up_val * silu_gate;
        
        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
            out_ptr[i] = utils::cast<T>(result);
        } else {
            out_ptr[i] = static_cast<T>(result);
        }
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    
    size_t numel = out->numel();
    
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_cpu(reinterpret_cast<float *>(out->data()),
                          reinterpret_cast<const float *>(gate->data()),
                          reinterpret_cast<const float *>(up->data()),
                          numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_cpu(reinterpret_cast<fp16_t *>(out->data()),
                          reinterpret_cast<const fp16_t *>(gate->data()),
                          reinterpret_cast<const fp16_t *>(up->data()),
                          numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_cpu(reinterpret_cast<bf16_t *>(out->data()),
                          reinterpret_cast<const bf16_t *>(gate->data()),
                          reinterpret_cast<const bf16_t *>(up->data()),
                          numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
