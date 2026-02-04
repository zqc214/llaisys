#pragma once

#include "../../tensor/tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <vector>

namespace llaisys::models {

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer;      // 层数
    size_t hs;          // hidden_size
    size_t nh;          // num_attention_heads
    size_t nkvh;        // num_key_value_heads
    size_t dh;          // head_dim = hs / nh
    size_t di;          // intermediate_size
    size_t maxseq;      // max_position_embeddings
    size_t voc;         // vocab_size
    float epsilon;      // rms_norm_eps
    float theta;        // rope_theta
    int64_t end_token;  // eos_token_id
};

struct Qwen2Weights {
    tensor_t in_embed;      // model.embed_tokens.weight [vocab_size, hidden_size]
    tensor_t out_embed;     // lm_head.weight [vocab_size, hidden_size]
    tensor_t out_norm_w;    // model.norm.weight [hidden_size]
    
    // 每层的权重
    std::vector<tensor_t> attn_norm_w;  // input_layernorm.weight
    std::vector<tensor_t> attn_q_w;     // self_attn.q_proj.weight
    std::vector<tensor_t> attn_q_b;     // self_attn.q_proj.bias
    std::vector<tensor_t> attn_k_w;     // self_attn.k_proj.weight
    std::vector<tensor_t> attn_k_b;     // self_attn.k_proj.bias
    std::vector<tensor_t> attn_v_w;     // self_attn.v_proj.weight
    std::vector<tensor_t> attn_v_b;     // self_attn.v_proj.bias
    std::vector<tensor_t> attn_o_w;     // self_attn.o_proj.weight
    std::vector<tensor_t> mlp_norm_w;   // post_attention_layernorm.weight
    std::vector<tensor_t> mlp_gate_w;   // mlp.gate_proj.weight
    std::vector<tensor_t> mlp_up_w;     // mlp.up_proj.weight
    std::vector<tensor_t> mlp_down_w;   // mlp.down_proj.weight
};

class Qwen2Model {
private:
    Qwen2Meta _meta;
    Qwen2Weights _weights;
    llaisysDeviceType_t _device_type;
    int _device_id;
    
    // KV-Cache: [nlayer][2] for K and V
    // 每个是 [max_seq_len, nkvh, dh]
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    size_t _cache_len;  // 当前缓存的长度
    
    // 中间张量（预分配避免重复分配）
    tensor_t _hidden;       // [seq_len, hidden_size]
    tensor_t _residual;     // [seq_len, hidden_size]
    tensor_t _norm_out;     // [seq_len, hidden_size]
    tensor_t _q, _k, _v;    // attention 中间结果
    tensor_t _q_rope, _k_rope;  // RoPE 后的 Q, K
    tensor_t _attn_out;     // attention 输出
    tensor_t _attn_proj;    // attention 投影后
    tensor_t _gate, _up, _down;  // MLP 中间结果
    tensor_t _logits;       // [seq_len, vocab_size]
    tensor_t _max_idx, _max_val;  // argmax 结果
    tensor_t _pos_ids;      // 位置 ID
    
    void allocateBuffers(size_t max_seq_len);
    void resetCache();
    
public:
    Qwen2Model(const Qwen2Meta& meta, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model() = default;
    
    Qwen2Weights& weights() { return _weights; }
    const Qwen2Meta& meta() const { return _meta; }
    
    // 单步推理，返回下一个 token
    int64_t infer(const int64_t* token_ids, size_t ntoken);
    
    // 重置 KV-Cache（新对话时调用）
    void reset() { resetCache(); }
};

} // namespace llaisys::models

