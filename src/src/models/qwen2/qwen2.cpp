#include "qwen2.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <vector>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Meta& meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id), _cache_len(0) {
    
    // 初始化权重向量
    _weights.attn_norm_w.resize(meta.nlayer);
    _weights.attn_q_w.resize(meta.nlayer);
    _weights.attn_q_b.resize(meta.nlayer);
    _weights.attn_k_w.resize(meta.nlayer);
    _weights.attn_k_b.resize(meta.nlayer);
    _weights.attn_v_w.resize(meta.nlayer);
    _weights.attn_v_b.resize(meta.nlayer);
    _weights.attn_o_w.resize(meta.nlayer);
    _weights.mlp_norm_w.resize(meta.nlayer);
    _weights.mlp_gate_w.resize(meta.nlayer);
    _weights.mlp_up_w.resize(meta.nlayer);
    _weights.mlp_down_w.resize(meta.nlayer);
    
    // 初始化 KV-Cache
    _k_cache.resize(meta.nlayer);
    _v_cache.resize(meta.nlayer);
    
    for (size_t i = 0; i < meta.nlayer; ++i) {
        _k_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
        _v_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
    }
    
    // 预分配中间缓冲区
    allocateBuffers(meta.maxseq);
}

void Qwen2Model::allocateBuffers(size_t max_seq_len) {
    auto dtype = _meta.dtype;
    auto dev = _device_type;
    auto dev_id = _device_id;
    
    // 这些缓冲区会在推理时根据实际 seq_len 重新分配
    _hidden = Tensor::create({1, _meta.hs}, dtype, dev, dev_id);
    _residual = Tensor::create({1, _meta.hs}, dtype, dev, dev_id);
    _norm_out = Tensor::create({1, _meta.hs}, dtype, dev, dev_id);
    
    _q = Tensor::create({1, _meta.nh, _meta.dh}, dtype, dev, dev_id);
    _k = Tensor::create({1, _meta.nkvh, _meta.dh}, dtype, dev, dev_id);
    _v = Tensor::create({1, _meta.nkvh, _meta.dh}, dtype, dev, dev_id);
    _q_rope = Tensor::create({1, _meta.nh, _meta.dh}, dtype, dev, dev_id);
    _k_rope = Tensor::create({1, _meta.nkvh, _meta.dh}, dtype, dev, dev_id);
    _attn_out = Tensor::create({1, _meta.nh, _meta.dh}, dtype, dev, dev_id);
    _attn_proj = Tensor::create({1, _meta.hs}, dtype, dev, dev_id);
    
    _gate = Tensor::create({1, _meta.di}, dtype, dev, dev_id);
    _up = Tensor::create({1, _meta.di}, dtype, dev, dev_id);
    _down = Tensor::create({1, _meta.hs}, dtype, dev, dev_id);
    
    _logits = Tensor::create({1, _meta.voc}, dtype, dev, dev_id);
    _max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
    _max_val = Tensor::create({1}, dtype, dev, dev_id);
    _pos_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
}

void Qwen2Model::resetCache() {
    _cache_len = 0;
}

int64_t Qwen2Model::infer(const int64_t* token_ids, size_t ntoken) {
    core::context().setDevice(_device_type, _device_id);
    
    auto dtype = _meta.dtype;
    auto dev = _device_type;
    auto dev_id = _device_id;
    
    // 重新分配缓冲区如果 seq_len 变化
    if (ntoken != _hidden->shape()[0]) {
        _hidden = Tensor::create({ntoken, _meta.hs}, dtype, dev, dev_id);
        _residual = Tensor::create({ntoken, _meta.hs}, dtype, dev, dev_id);
        _norm_out = Tensor::create({ntoken, _meta.hs}, dtype, dev, dev_id);
        
        _q = Tensor::create({ntoken, _meta.nh, _meta.dh}, dtype, dev, dev_id);
        _k = Tensor::create({ntoken, _meta.nkvh, _meta.dh}, dtype, dev, dev_id);
        _v = Tensor::create({ntoken, _meta.nkvh, _meta.dh}, dtype, dev, dev_id);
        _q_rope = Tensor::create({ntoken, _meta.nh, _meta.dh}, dtype, dev, dev_id);
        _k_rope = Tensor::create({ntoken, _meta.nkvh, _meta.dh}, dtype, dev, dev_id);
        _attn_out = Tensor::create({ntoken, _meta.nh, _meta.dh}, dtype, dev, dev_id);
        _attn_proj = Tensor::create({ntoken, _meta.hs}, dtype, dev, dev_id);
        
        _gate = Tensor::create({ntoken, _meta.di}, dtype, dev, dev_id);
        _up = Tensor::create({ntoken, _meta.di}, dtype, dev, dev_id);
        _down = Tensor::create({ntoken, _meta.hs}, dtype, dev, dev_id);
        
        _logits = Tensor::create({ntoken, _meta.voc}, dtype, dev, dev_id);
        _pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, dev, dev_id);
    }
    
    // 创建输入 token 张量
    auto input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, dev, dev_id);
    input_ids->load(token_ids);
    
    // 设置位置 ID
    std::vector<int64_t> pos_ids_data(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        pos_ids_data[i] = static_cast<int64_t>(_cache_len + i);
    }
    _pos_ids->load(pos_ids_data.data());
    
    // 1. Embedding
    ops::embedding(_hidden, input_ids, _weights.in_embed);
    
    // 2. Transformer layers
    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        // 保存 residual (复制 hidden 到 residual)
        size_t hidden_bytes = _hidden->numel() * _hidden->elementSize();
        llaisysMemcpyKind_t copy_kind = (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
        core::context().runtime().api()->memcpy_sync(
            _residual->data(), _hidden->data(), hidden_bytes, copy_kind);
        
        // 2.1 Input LayerNorm
        ops::rms_norm(_norm_out, _hidden, _weights.attn_norm_w[layer], _meta.epsilon);
        
        // 2.2 Attention
        // Q, K, V 投影
        auto norm_out_2d = _norm_out->view({ntoken, _meta.hs});
        auto q_2d = _q->view({ntoken, _meta.nh * _meta.dh});
        auto k_2d = _k->view({ntoken, _meta.nkvh * _meta.dh});
        auto v_2d = _v->view({ntoken, _meta.nkvh * _meta.dh});
        
        ops::linear(q_2d, norm_out_2d, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        ops::linear(k_2d, norm_out_2d, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        ops::linear(v_2d, norm_out_2d, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);
        
        // RoPE
        ops::rope(_q_rope, _q, _pos_ids, _meta.theta);
        ops::rope(_k_rope, _k, _pos_ids, _meta.theta);
        
        // 更新 KV-Cache
        // 将新的 K, V 复制到 cache 的对应位置
        size_t k_bytes = ntoken * _meta.nkvh * _meta.dh * utils::dsize(dtype);
        size_t offset_bytes = _cache_len * _meta.nkvh * _meta.dh * utils::dsize(dtype);
        std::byte* k_cache_ptr = _k_cache[layer]->data() + offset_bytes;
        std::byte* v_cache_ptr = _v_cache[layer]->data() + offset_bytes;
        
        core::context().runtime().api()->memcpy_sync(
            k_cache_ptr, _k_rope->data(), k_bytes, copy_kind);
        core::context().runtime().api()->memcpy_sync(
            v_cache_ptr, _v->data(), k_bytes, copy_kind);
        
        // Self-Attention (使用完整的 KV-Cache)
        size_t total_len = _cache_len + ntoken;
        auto k_cache_view = _k_cache[layer]->slice(0, 0, total_len);
        auto v_cache_view = _v_cache[layer]->slice(0, 0, total_len);
        
        float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
        ops::self_attention(_attn_out, _q_rope, k_cache_view, v_cache_view, scale);
        
        // 输出投影
        auto attn_out_2d = _attn_out->view({ntoken, _meta.hs});
        ops::linear(_attn_proj, attn_out_2d, _weights.attn_o_w[layer], nullptr);
        
        // 残差连接
        ops::add(_hidden, _residual, _attn_proj);
        
        // 2.3 MLP
        // 保存 residual
        core::context().runtime().api()->memcpy_sync(
            _residual->data(), _hidden->data(), hidden_bytes, copy_kind);
        
        // Post-attention LayerNorm
        ops::rms_norm(_norm_out, _hidden, _weights.mlp_norm_w[layer], _meta.epsilon);
        
        // MLP: gate, up, swiglu, down
        ops::linear(_gate, norm_out_2d, _weights.mlp_gate_w[layer], nullptr);
        ops::linear(_up, norm_out_2d, _weights.mlp_up_w[layer], nullptr);
        
        // SwiGLU: 输出和 gate/up 相同形状 [ntoken, di]
        auto swiglu_out = Tensor::create({ntoken, _meta.di}, dtype, dev, dev_id);
        ops::swiglu(swiglu_out, _gate, _up);
        
        // Down projection
        ops::linear(_down, swiglu_out, _weights.mlp_down_w[layer], nullptr);
        
        // 残差连接
        ops::add(_hidden, _residual, _down);
    }
    
    // 3. Final LayerNorm
    ops::rms_norm(_norm_out, _hidden, _weights.out_norm_w, _meta.epsilon);
    
    // 4. LM Head
    auto norm_out_2d = _norm_out->view({ntoken, _meta.hs});
    ops::linear(_logits, norm_out_2d, _weights.out_embed, nullptr);
    
    // 5. 取最后一个 token 的 logits 做 argmax
    auto last_logits = _logits->slice(0, ntoken - 1, ntoken)->view({_meta.voc});
    auto max_idx_scalar = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
    auto max_val_scalar = Tensor::create({1}, dtype, dev, dev_id);
    ops::argmax(max_idx_scalar, max_val_scalar, last_logits);
    
    // 读取结果
    int64_t next_token;
    llaisysMemcpyKind_t read_kind = (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2H;
    core::context().runtime().api()->memcpy_sync(
        &next_token, max_idx_scalar->data(), sizeof(int64_t), read_kind);
    
    // 更新 cache 长度
    _cache_len += ntoken;
    
    return next_token;
}

} // namespace llaisys::models
