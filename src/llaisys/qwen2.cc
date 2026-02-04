#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2.hpp"
#include "llaisys/models/qwen2.h"

#include <cstring>

struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model* model;
    LlaisysQwen2Weights weights;
    
    // 用于存储层权重的指针数组
    std::vector<llaisysTensor_t> attn_norm_w_ptrs;
    std::vector<llaisysTensor_t> attn_q_w_ptrs;
    std::vector<llaisysTensor_t> attn_q_b_ptrs;
    std::vector<llaisysTensor_t> attn_k_w_ptrs;
    std::vector<llaisysTensor_t> attn_k_b_ptrs;
    std::vector<llaisysTensor_t> attn_v_w_ptrs;
    std::vector<llaisysTensor_t> attn_v_b_ptrs;
    std::vector<llaisysTensor_t> attn_o_w_ptrs;
    std::vector<llaisysTensor_t> mlp_norm_w_ptrs;
    std::vector<llaisysTensor_t> mlp_gate_w_ptrs;
    std::vector<llaisysTensor_t> mlp_up_w_ptrs;
    std::vector<llaisysTensor_t> mlp_down_w_ptrs;
};

__C {

struct LlaisysQwen2Model* llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta* meta,
    llaisysDeviceType_t device,
    int* device_ids,
    int ndevice) {
    
    llaisys::models::Qwen2Meta cpp_meta;
    cpp_meta.dtype = meta->dtype;
    cpp_meta.nlayer = meta->nlayer;
    cpp_meta.hs = meta->hs;
    cpp_meta.nh = meta->nh;
    cpp_meta.nkvh = meta->nkvh;
    cpp_meta.dh = meta->dh;
    cpp_meta.di = meta->di;
    cpp_meta.maxseq = meta->maxseq;
    cpp_meta.voc = meta->voc;
    cpp_meta.epsilon = meta->epsilon;
    cpp_meta.theta = meta->theta;
    cpp_meta.end_token = meta->end_token;
    
    int device_id = (ndevice > 0) ? device_ids[0] : 0;
    
    auto* wrapper = new LlaisysQwen2Model();
    wrapper->model = new llaisys::models::Qwen2Model(cpp_meta, device, device_id);
    
    // 初始化权重指针数组
    size_t nlayer = meta->nlayer;
    wrapper->attn_norm_w_ptrs.resize(nlayer);
    wrapper->attn_q_w_ptrs.resize(nlayer);
    wrapper->attn_q_b_ptrs.resize(nlayer);
    wrapper->attn_k_w_ptrs.resize(nlayer);
    wrapper->attn_k_b_ptrs.resize(nlayer);
    wrapper->attn_v_w_ptrs.resize(nlayer);
    wrapper->attn_v_b_ptrs.resize(nlayer);
    wrapper->attn_o_w_ptrs.resize(nlayer);
    wrapper->mlp_norm_w_ptrs.resize(nlayer);
    wrapper->mlp_gate_w_ptrs.resize(nlayer);
    wrapper->mlp_up_w_ptrs.resize(nlayer);
    wrapper->mlp_down_w_ptrs.resize(nlayer);
    
    // 初始化 weights 结构体
    wrapper->weights.in_embed = nullptr;
    wrapper->weights.out_embed = nullptr;
    wrapper->weights.out_norm_w = nullptr;
    wrapper->weights.attn_norm_w = wrapper->attn_norm_w_ptrs.data();
    wrapper->weights.attn_q_w = wrapper->attn_q_w_ptrs.data();
    wrapper->weights.attn_q_b = wrapper->attn_q_b_ptrs.data();
    wrapper->weights.attn_k_w = wrapper->attn_k_w_ptrs.data();
    wrapper->weights.attn_k_b = wrapper->attn_k_b_ptrs.data();
    wrapper->weights.attn_v_w = wrapper->attn_v_w_ptrs.data();
    wrapper->weights.attn_v_b = wrapper->attn_v_b_ptrs.data();
    wrapper->weights.attn_o_w = wrapper->attn_o_w_ptrs.data();
    wrapper->weights.mlp_norm_w = wrapper->mlp_norm_w_ptrs.data();
    wrapper->weights.mlp_gate_w = wrapper->mlp_gate_w_ptrs.data();
    wrapper->weights.mlp_up_w = wrapper->mlp_up_w_ptrs.data();
    wrapper->weights.mlp_down_w = wrapper->mlp_down_w_ptrs.data();
    
    return wrapper;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model* model) {
    if (model) {
        delete model->model;
        
        // 清理权重张量
        if (model->weights.in_embed) delete model->weights.in_embed;
        if (model->weights.out_embed) delete model->weights.out_embed;
        if (model->weights.out_norm_w) delete model->weights.out_norm_w;
        
        for (auto& ptr : model->attn_norm_w_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->attn_q_w_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->attn_q_b_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->attn_k_w_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->attn_k_b_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->attn_v_w_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->attn_v_b_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->attn_o_w_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->mlp_norm_w_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->mlp_gate_w_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->mlp_up_w_ptrs) if (ptr) delete ptr;
        for (auto& ptr : model->mlp_down_w_ptrs) if (ptr) delete ptr;
        
        delete model;
    }
}

struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(struct LlaisysQwen2Model* model) {
    return &model->weights;
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model* model, int64_t* token_ids, size_t ntoken) {
    // 同步 C API 权重到 C++ 模型
    auto& cpp_weights = model->model->weights();
    
    if (model->weights.in_embed) {
        cpp_weights.in_embed = model->weights.in_embed->tensor;
    }
    if (model->weights.out_embed) {
        cpp_weights.out_embed = model->weights.out_embed->tensor;
    }
    if (model->weights.out_norm_w) {
        cpp_weights.out_norm_w = model->weights.out_norm_w->tensor;
    }
    
    size_t nlayer = model->model->meta().nlayer;
    for (size_t i = 0; i < nlayer; ++i) {
        if (model->attn_norm_w_ptrs[i]) {
            cpp_weights.attn_norm_w[i] = model->attn_norm_w_ptrs[i]->tensor;
        }
        if (model->attn_q_w_ptrs[i]) {
            cpp_weights.attn_q_w[i] = model->attn_q_w_ptrs[i]->tensor;
        }
        if (model->attn_q_b_ptrs[i]) {
            cpp_weights.attn_q_b[i] = model->attn_q_b_ptrs[i]->tensor;
        }
        if (model->attn_k_w_ptrs[i]) {
            cpp_weights.attn_k_w[i] = model->attn_k_w_ptrs[i]->tensor;
        }
        if (model->attn_k_b_ptrs[i]) {
            cpp_weights.attn_k_b[i] = model->attn_k_b_ptrs[i]->tensor;
        }
        if (model->attn_v_w_ptrs[i]) {
            cpp_weights.attn_v_w[i] = model->attn_v_w_ptrs[i]->tensor;
        }
        if (model->attn_v_b_ptrs[i]) {
            cpp_weights.attn_v_b[i] = model->attn_v_b_ptrs[i]->tensor;
        }
        if (model->attn_o_w_ptrs[i]) {
            cpp_weights.attn_o_w[i] = model->attn_o_w_ptrs[i]->tensor;
        }
        if (model->mlp_norm_w_ptrs[i]) {
            cpp_weights.mlp_norm_w[i] = model->mlp_norm_w_ptrs[i]->tensor;
        }
        if (model->mlp_gate_w_ptrs[i]) {
            cpp_weights.mlp_gate_w[i] = model->mlp_gate_w_ptrs[i]->tensor;
        }
        if (model->mlp_up_w_ptrs[i]) {
            cpp_weights.mlp_up_w[i] = model->mlp_up_w_ptrs[i]->tensor;
        }
        if (model->mlp_down_w_ptrs[i]) {
            cpp_weights.mlp_down_w[i] = model->mlp_down_w_ptrs[i]->tensor;
        }
    }
    
    return model->model->infer(token_ids, ntoken);
}

} // extern "C"

